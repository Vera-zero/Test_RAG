import json
import os
import hashlib
from tokenizers import Tokenizer

def extract_title(idx):
    """从idx中提取title"""
    # 示例: "/wiki/Ian_Gibson_(politician)#P39#0"
    if "#" in idx:
        return idx.split("/wiki/")[-1].split("#")[0]
    else:
        return idx.split("/wiki/")[-1]

def count_tokens(text, tokenizer=None):
    """使用tokenizers库计算文本中的token数量"""
    if tokenizer:
        return len(tokenizer.encode(text).ids)
    else:
        # 如果没有提供tokenizer，则回退到空格分词
        return len(text.split())

def process_dev_json(input_file, output_file, tokenizer_path=None):
    """处理dev.json文件"""
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 如果提供了tokenizer路径，则加载tokenizer
    tokenizer = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            print(f"成功加载tokenizer: {tokenizer_path}")
        except Exception as e:
            print(f"警告：无法加载tokenizer: {e}，将使用默认方法计算token数")
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 使用字典来存储去重后的数据
    context_dict = {}
    
    # 处理每条记录
    for item in data:
        idx = item.get("idx", "")
        # 提取title
        title = extract_title(idx)
        lenth = len(title)
        question = item.get("question", "")
        context_original = item.get("context", "")
        context = context_original[lenth:]
        level = item.get("level", "")
        
        # 计算context的hash值作为id
        context_hash = hashlib.md5(context.encode('utf-8')).hexdigest()
        
        # 如果context已存在，则添加question到列表中
        if context_hash in context_dict:
            context_dict[context_hash]["question_list"].append({
                "question": question,
                "level": level
            })
        else:
            # 否则创建新的记录
            context_dict[context_hash] = {
                "id": f"context_{context_hash}",
                "title": title,
                "question_list": [
                    {
                        "question": question,
                        "level": level
                    }
                ],
                "context": context
            }
    
    # 转换为列表格式并计算token数
    result = list(context_dict.values())
    max_tokens = 0
    total_tokens = 0
    
    for item in result:
        # 计算context的token数
        token_count = count_tokens(item["context"], tokenizer)
        item["token_count"] = token_count
        
        # 更新总token数和最大token数
        total_tokens += token_count
        if token_count > max_tokens:
            max_tokens = token_count
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 显示处理后的统计信息
    context_count = len(result)
    avg_tokens = total_tokens / context_count if context_count > 0 else 0
    
    print(f"处理完成，结果已保存到 {output_file}")
    print(f"处理后的文件包含 {context_count} 个唯一的context")
    print(f"最大token数: {max_tokens}")
    print(f"平均token数: {avg_tokens:.2f}")
    print(f"总token数: {total_tokens}")
    
    return context_count, max_tokens, avg_tokens, total_tokens

if __name__ == "__main__":
    # 定义输入和输出路径
    input_file = "/mnt/nvm_data/guest24/luoyixingfei/datasets--hugosousa--TimeQA/snapshots/1db7060e4ba5934c25025f107cbf3017e13ef1f6/test.json"
    output_file = "/mnt/nvm_data/guest24/luoyixingfei/datasets_over/test_trans.json"
    
    # 指定tokenizer文件路径
    tokenizer_path = "/mnt/nvm_data/guest24/luoyixingfei/DyG-RAG/models/bge_m3/tokenizer.json"
    
    # 处理文件并显示统计信息
    context_count, max_tokens, avg_tokens, total_tokens = process_dev_json(input_file, output_file, tokenizer_path)