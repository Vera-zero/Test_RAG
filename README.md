## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (for optimal performance)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Vera-zero/Test_RAG.git
cd Test_RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
```bash
cd models
python download.py
```

4. Process Datasets:  
   download Time QA datasets 
   change the filedirs in datasets_trans.py
```bash
cd datasets
python datasets_trans.py
```

## 🚀 Quick Start
```bash
python examples/openai_all.py
```
