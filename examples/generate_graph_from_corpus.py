from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, json, time, asyncio, logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
from openai import AsyncOpenAI
import aiohttp.client_exceptions
import torch, gc
import networkx as nx

from graphrag import GraphRAG, QueryParam
from graphrag.base import BaseKVStorage
from graphrag._utils import compute_args_hash, logger, wrap_embedding_func_with_attrs
from graphrag._storage.gdb_networkx import NetworkXStorage

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

# Check API key before proceeding
check_deepseek_api_key()

# 创建Test文件夹
TEST_DIR = Path("Test")
TEST_DIR.mkdir(exist_ok=True)

WORK_DIR = Path("work_dir")
WORK_DIR.mkdir(exist_ok=True)
CORPUS_FILE = Path("../demo/Corpus.json")

logging.basicConfig(level=logging.INFO)
logging.getLogger("DyG-RAG").setLevel(logging.INFO)


def read_json_file(fp: Path):
    with fp.open(encoding="utf-8") as f:
        return json.load(f)

def extract_nodes_and_graph(graph_func: GraphRAG) -> Dict[str, Any]:
    """
    从GraphRAG实例中提取节点和图数据
    """
    # 获取图存储实例
    graph_storage = None
    for storage in [graph_func._event_graph, graph_func._entity_graph]:
        if storage is not None and hasattr(storage, '_graph'):
            graph_storage = storage
            break
    
    if graph_storage is None:
        logger.error("无法找到图存储实例")
        return {"nodes": {}, "graph": {"nodes": [], "edges": []}}
    
    # 获取所有节点
    nodes_data = {}
    if hasattr(graph_storage, 'get_all_nodes'):
        nodes_data = asyncio.run(graph_storage.get_all_nodes())
    else:
        # 如果get_all_nodes方法不可用，尝试通过其他方式获取节点
        if hasattr(graph_storage, '_graph') and hasattr(graph_storage._graph, 'nodes'):
            for node_id in graph_storage._graph.nodes():
                node_data = graph_storage._graph.nodes[node_id]
                nodes_data[node_id] = node_data
    
    # 获取所有边
    edges_data = []
    if hasattr(graph_storage, '_graph') and hasattr(graph_storage._graph, 'edges'):
        for u, v, edge_data in graph_storage._graph.edges(data=True):
            edge_info = {
                "source": u,
                "target": v,
                "data": edge_data
            }
            edges_data.append(edge_info)
    
    # 构建图结构
    graph_structure = {
        "nodes": list(nodes_data.keys()),
        "edges": edges_data
    }
    
    return {
        "nodes": nodes_data,
        "graph": graph_structure
    }

def save_graph_data(nodes_data: Dict, graph_data: Dict, output_dir: Path):
    """
    保存节点和图数据到JSON文件
    """
    # 保存节点数据
    nodes_file = output_dir / "dygrag_node.json"
    with open(nodes_file, 'w', encoding='utf-8') as f:
        json.dump(nodes_data, f, indent=2, ensure_ascii=False)
    logger.info(f"节点数据已保存到: {nodes_file}")
    
    # 保存图数据
    graph_file = output_dir / "dygrag_graph.json"
    with open(graph_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    logger.info(f"图数据已保存到: {graph_file}")

def analyze_graph_structure(graph_data: Dict):
    """
    分析图结构并输出统计信息
    """
    nodes_count = len(graph_data["nodes"])
    edges_count = len(graph_data["graph"]["edges"])
    
    print("\n" + "="*50)
    print("图结构分析结果")
    print("="*50)
    print(f"节点数量: {nodes_count}")
    print(f"边数量: {edges_count}")
    
    if nodes_count > 0 and edges_count > 0:
        # 计算图的密度
        max_possible_edges = nodes_count * (nodes_count - 1) / 2
        density = edges_count / max_possible_edges if max_possible_edges > 0 else 0
        print(f"图密度: {density:.4f}")
        
        # 分析节点度分布
        degree_distribution = {}
        for edge in graph_data["graph"]["edges"]:
            source = edge["source"]
            target = edge["target"]
            degree_distribution[source] = degree_distribution.get(source, 0) + 1
            degree_distribution[target] = degree_distribution.get(target, 0) + 1
        
        if degree_distribution:
            avg_degree = sum(degree_distribution.values()) / len(degree_distribution)
            max_degree = max(degree_distribution.values())
            min_degree = min(degree_distribution.values())
            
            print(f"平均节点度: {avg_degree:.2f}")
            print(f"最大节点度: {max_degree}")
            print(f"最小节点度: {min_degree}")
    
    print("="*50)

async def main():
    """
    主函数：处理corpus.json并生成图数据
    """
    logger.info("开始处理corpus.json并生成图数据...")
    
    # Use local BGE for embeddings and remote LLM (vLLM/DeepSeek-compatible) for completions
    local_bge_path = os.getenv("LOCAL_BGE_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "bge_m3"))
    embedding_func = LocalBGEEmbedding(local_bge_path)
    
    # Ensure model is loaded once and expose embedding_dim via EmbeddingFunc wrapper
    try:
        embedding_func._load()
        emb_dim = embedding_func.embedding_dim
    except Exception:
        emb_dim = getattr(embedding_func, 'embedding_dim', 1536)

    # Wrap embedding func with attributes so GraphRAG's storage can read embedding_dim
    embedding_func = wrap_embedding_func_with_attrs(embedding_dim=emb_dim, max_token_size=embedding_func.max_token_size)(embedding_func)

    # 创建GraphRAG实例
    graph_func = GraphRAG(
        working_dir=str(WORK_DIR),
        embedding_func=embedding_func,
        best_model_max_token_size=16384,
        cheap_model_max_token_size=16384
    )

    # Read the JSON file
    corpus_data = read_json_file(CORPUS_FILE)
    total_docs = len(corpus_data)
    logger.info(f"开始处理，共有 {total_docs} 个文档需要处理。")

    # 准备文档数据
    all_docs = []
    for idx, obj in enumerate(tqdm(corpus_data, desc="加载文档", total=total_docs)):
        enriched_content = f"Title: {obj['title']}\nDocument ID: {obj['doc_id']}\n\n{obj['context']}"
        all_docs.append(enriched_content)
    
    # 插入文档到GraphRAG
    logger.info("正在将文档插入GraphRAG...")
    graph_func.insert(all_docs)
    
    # 等待图构建完成
    logger.info("等待图构建完成...")
    await asyncio.sleep(2)  # 给图构建一些时间
    
    # 提取节点和图数据
    logger.info("正在提取节点和图数据...")
    graph_data = extract_nodes_and_graph(graph_func)
    
    # 保存数据到Test文件夹
    logger.info("正在保存数据到Test文件夹...")
    save_graph_data(graph_data["nodes"], graph_data["graph"], "./GraphRAG/Test")
    
    # 分析图结构
    analyze_graph_structure(graph_data)
    
    logger.info("处理完成！")

if __name__ == "__main__":
    asyncio.run(main())