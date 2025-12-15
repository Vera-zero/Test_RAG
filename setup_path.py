import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.absolute()
dyg_root = project_root.parent.absolute()

# 添加必要的路径
paths_to_add = [
    str(project_root),           # Graph-RAG 根目录
    str(dyg_root),              # DyG-RAG 根目录
    str(project_root / "Core"), # Core 目录
    str(dyg_root / "examples"), # DyG-RAG examples
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added to path: {path}")

print(f"Python path updated. Total paths: {len(sys.path)}")
