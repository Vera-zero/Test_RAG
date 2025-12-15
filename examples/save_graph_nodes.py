from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, json, time, asyncio, logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import datetime

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
from openai import AsyncOpenAI
import aiohttp.client_exceptions
import torch, gc

from graphrag import GraphRAG, QueryParam
from graphrag.base import BaseKVStorage
from graphrag._utils import compute_args_hash, logger, wrap_embedding_func_with_attrs

from graphrag import GraphRAG, QueryParam
import json
from tqdm import tqdm
from pathlib import Path

# --- Local BGE embedding + remote LLM wrapper support ---
class LocalBGEEmbedding:
    def __init__(self, model_path: str, max_token_size: int = 8192):
        self.model_path = model_path
        self.max_token_size = max_token_size
        # lazy init of SentenceTransformer model
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        gpu_count = 0
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            device = "cuda" if gpu_count > 0 else "cpu"
            model_kwargs = {}
            if gpu_count > 1:
                model_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
        except Exception:
            device = "cpu"
            model_kwargs = {}

        # 首先尝试按常规方式加载（当模型目录为 SentenceTransformers 格式时）
        try:
            self._model = SentenceTransformer(
                self.model_path,
                device=device,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
            )
            return
        except Exception:
            # 如果直接加载失败（例如本地模型不是严格的 SentenceTransformer 包装），
            # 那么尝试按模块组合的方式构造：Transformer + Pooling
            try:
                transformer = models.Transformer(self.model_path, max_seq_length=self.max_token_size)
                # 获取 transformer 的输出维度以初始化 Pooling
                word_emb_dim = transformer.get_word_embedding_dimension()
                pooling = models.Pooling(word_emb_dim,
                                         pooling_mode_mean_tokens=True,
                                         pooling_mode_cls_token=False,
                                         pooling_mode_max_tokens=False)
                self._model = SentenceTransformer(modules=[transformer, pooling])
                # 把模型移动到目标设备（SentenceTransformer 内部会处理大部分，但显式设置可提高兼容）
                try:
                    import torch
                    if device.startswith("cuda") and torch.cuda.is_available():
                        self._model.to(device)
                except Exception:
                    pass
                return
            except Exception as e:
                # 最后回退到抛出原始异常，方便排查
                raise RuntimeError(f"Failed to load embedding model from {self.model_path}: {e}")

    @property
    def embedding_dim(self):
        self._load()
        return self._model.get_sentence_embedding_dimension()

    async def __call__(self, texts):
        # ensure model loaded
        self._load()
        if isinstance(texts, str):
            texts = [texts]
        loop = asyncio.get_event_loop()
        fn = lambda: self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if loop.is_running():
            return await loop.run_in_executor(None, fn)
        else:
            return fn()


# Check for OPENAI_API_KEY environment variable
def check_deepseek_api_key():
    """
    Check if OPENAI_API_KEY is set and not empty.
    If not set, allow user to input manually.
    Raises SystemExit if not properly configured.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if api_key is None or not api_key.strip():
        print("❌ DEEPSEEK_API_KEY environment variable is not set or empty.")
        print("\nOptions:")
        print("1. Set environment variable: export DEEPSEEK_API_KEY='your-api-key-here'")
        print("2. Enter API key manually now (will be set for this session)")
        
        choice = input("\nWould you like to enter your API key manually? (y/N): ")
        if choice.lower() == 'y':
            manual_key = input("Please enter your DeepSeek API key: ").strip()
            if not manual_key:
                print("❌ Error: No API key provided.")
                sys.exit(1)
            # Set the environment variable for this session (set both for compatibility)
            os.environ["DEEPSEEK_API_KEY"] = manual_key
            os.environ["OPENAI_API_KEY"] = manual_key
            api_key = manual_key
            print("✅ DEEPSEEK_API_KEY has been set for this session.")
        else:
            print("❌ Cannot proceed without API key.")
            sys.exit(1)
    
    # Basic format validation (OpenAI keys typically start with 'sk-')
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: DEEPSEEK_API_KEY doesn't appear to be in the expected format (should start with 'sk-')")
        print(f"   Current key starts with: {api_key[:10]}...")
        response = input("Do you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("✅ DEEPSEEK_API_KEY is properly configured.")
    return True

def read_json_file(fp: Path):
    """读取 JSON 文件"""
    with fp.open(encoding="utf-8") as f:
        return json.load(f)

async def save_graph_nodes(graph_func: GraphRAG, output_dir: Path):
    """
    保存图中所有节点到 JSON 文件
    
    Args:
        graph_func: GraphRAG 实例
        output_dir: 输出目录
    """
    try:
        # 确保输出目录存在
        output_dir.mkdir(exist_ok=True)
        
        # 获取事件图存储实例
        event_graph = graph_func.event_dynamic_graph
        
        # 获取所有节点
        all_nodes = await event_graph.get_all_nodes()
        
        if not all_nodes:
            logger.warning("图中没有找到任何节点")
            return
        
        # 准备节点数据
        nodes_data = {
            "metadata": {
                "export_time": datetime.datetime.now().isoformat(),
                "total_nodes": len(all_nodes),
                "graph_type": "dynamic_event_graph"
            },
            "nodes": {}
        }
        
        # 处理每个节点
        for node_id, node_data in all_nodes.items():
            if node_data:
                # 清理节点数据，确保可序列化
                clean_node_data = {}
                for key, value in node_data.items():
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        clean_node_data[key] = value
                
                nodes_data["nodes"][node_id] = clean_node_data
        
        # 保存到文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"graph_nodes_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 成功保存 {len(nodes_data['nodes'])} 个节点到: {output_file}")
        
        # 打印节点统计信息
        print(f"\n📊 节点统计信息:")
        print(f"   - 总节点数: {len(nodes_data['nodes'])}")
        
        # 按节点类型统计
        node_types = {}
        for node_id, node_data in nodes_data['nodes'].items():
            node_type = node_data.get('entity_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print(f"   - 节点类型分布:")
        for node_type, count in node_types.items():
            print(f"     * {node_type}: {count} 个")
        
        return nodes_data
        
    except Exception as e:
        logger.error(f"保存节点时出错: {e}")
        raise

async def main():
    """主函数"""
    # 检查 API 密钥
    check_deepseek_api_key()
    
    # 设置工作目录
    WORK_DIR = Path("work_dir")
    WORK_DIR.mkdir(exist_ok=True)
    
    # 输出目录
    OUTPUT_DIR = Path("saved_nodes")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    CORPUS_FILE = Path("../demo/Corpus.json")
    
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("DyG-RAG").setLevel(logging.INFO)
    
    # 使用本地 BGE 嵌入
    local_bge_path = os.getenv("LOCAL_BGE_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "bge_m3"))
    embedding_func = LocalBGEEmbedding(local_bge_path)
    
    # 确保模型加载并包装嵌入函数
    try:
        embedding_func._load()
        emb_dim = embedding_func.embedding_dim
    except Exception:
        emb_dim = getattr(embedding_func, 'embedding_dim', 1536)
    
    embedding_func = wrap_embedding_func_with_attrs(embedding_dim=emb_dim, max_token_size=embedding_func.max_token_size)(embedding_func)
    
    # 创建 GraphRAG 实例
    graph_func = GraphRAG(
        working_dir=str(WORK_DIR),
        embedding_func=embedding_func,
        best_model_max_token_size=16384,
        cheap_model_max_token_size=16384
    )
    
    # 读取语料文件
    if not CORPUS_FILE.exists():
        logger.error(f"语料文件不存在: {CORPUS_FILE}")
        return
    
    corpus_data = read_json_file(CORPUS_FILE)
    total_docs = len(corpus_data)
    logger.info(f"开始处理，共有 {total_docs} 个文档")
    
    # 准备文档
    all_docs = []
    for idx, obj in enumerate(tqdm(corpus_data, desc="加载文档", total=total_docs)):
        enriched_content = f"Title: {obj['title']}\nDocument ID: {obj['doc_id']}\n\n{obj['context']}"
        all_docs.append(enriched_content)
    
    # 插入文档并构建图
    logger.info("开始构建知识图谱...")
    # 修复：使用异步插入方法
    await graph_func.ainsert(all_docs)
    logger.info("知识图谱构建完成")
    
    # 保存节点
    logger.info("开始保存节点数据...")
    nodes_data = await save_graph_nodes(graph_func, OUTPUT_DIR)
    
    # 可选：执行一个查询来验证图的功能
    logger.info("执行测试查询...")
    try:
        # 修复：使用异步查询方法
        result = await graph_func.aquery("Which position did Pat Duncan hold in Feb 1996?", param=QueryParam(mode="dynamic"))
        print(f"\n🔍 测试查询结果:")
        print(result)
    except Exception as e:
        logger.warning(f"测试查询失败: {e}")
    
    logger.info("脚本执行完成")

if __name__ == "__main__":
    asyncio.run(main())