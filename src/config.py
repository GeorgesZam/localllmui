import os
import multiprocessing

APP_NAME = "Local Chat"
WINDOW_SIZE = "1100x700"

MODEL_FILE = "models/model.gguf"
EMBEDDING_MODEL_FOLDER = "embedding_model"

_CPU_COUNT = multiprocessing.cpu_count()
CONTEXT_SIZE = 2048
MAX_TOKENS = 384
THREADS = max(4, _CPU_COUNT - 2)
GPU_LAYERS = -1

SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based ONLY on the provided context documents.
If the answer is not found in the context, say "I don't have this information in the provided documents."
Be concise and specific. Quote relevant parts when possible.
Answer in the same language as the user."""

STOP_TOKENS = ["<|im_end|>", "<end_of_turn>", ""]

RAG_ENABLED = True
RAG_FOLDER = "data"
RAG_CHUNK_SIZE = 384
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 3
RAG_MIN_SCORE = 0.3
RAG_SHOW_SOURCES = True

TEMPERATURE = 0.1
TOP_P = 0.85
REPEAT_PENALTY = 1.15

BATCH_SIZE = 512
LAZY_LOAD_EMBEDDING = True
INDEX_CACHE_ENABLED = True

# === CODE EXECUTION ===
CODE_EXECUTION_ENABLED = True
CODE_EXECUTION_TIMEOUT = 30  # seconds
CODE_EXECUTION_MAX_MEMORY_MB = 512
CODE_EXECUTION_AUTO_DETECT = True
CODE_EXECUTION_PROMPT_SAVE = True  # Ask user where to save files

CODE_EXECUTION_SYSTEM_PROMPT = """You are a helpful assistant with code execution capabilities.

You can WRITE and EXECUTE Python code to solve problems.

When you need to create documents, analyze data, or generate files:
1. Write clear, well-commented Python code
2. Wrap code in triple backticks with 'python' tag
3. Use available libraries (see below)
4. Print progress messages so user knows what's happening

Available libraries:
- python-docx: Create Word documents (.docx)
- python-pptx: Create PowerPoint presentations (.pptx)
- openpyxl: Create Excel spreadsheets (.xlsx)
- reportlab: Create PDF documents (.pdf)
- pandas: Data analysis, CSV/Excel processing
- matplotlib: Charts, graphs, visualizations
- json, csv: Standard file formats

Keep code simple, safe, and focused on the task.
Handle errors gracefully.

Answer in the same language as the user."""
