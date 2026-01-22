"""
Configuration - Optimized for performance.
"""

import os
import multiprocessing

# === APP ===
APP_NAME = "Local Chat"
WINDOW_SIZE = "1100x700"

# === MODEL ===
MODEL_FILE = "models/model.gguf"
EMBEDDING_MODEL_FOLDER = "embedding_model"

# Auto-detect optimal settings
_CPU_COUNT = multiprocessing.cpu_count()
CONTEXT_SIZE = 2048  # Reduced from 4096 - faster, sufficient for most docs
MAX_TOKENS = 384     # Reduced from 512 - faster responses
THREADS = max(4, _CPU_COUNT - 2)  # Leave 2 cores for UI/system
GPU_LAYERS = -1      # Use all available GPU layers

# === PROMPT ===
SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based ONLY on the provided context documents.
If the answer is not found in the context, say "I don't have this information in the provided documents."
Be concise and specific. Quote relevant parts when possible.
Answer in the same language as the user."""

STOP_TOKENS = ["<|im_end|>", "<end_of_turn>", "<|endoftext|>"]

# === RAG ===
RAG_ENABLED = True
RAG_FOLDER = "data"
RAG_CHUNK_SIZE = 384      # Reduced from 512 - better granularity
RAG_CHUNK_OVERLAP = 50    # Reduced from 100 - less redundancy
RAG_TOP_K = 3
RAG_MIN_SCORE = 0.3       # Increased from 0.25 - better relevance
RAG_SHOW_SOURCES = True

# === SAMPLING ===
TEMPERATURE = 0.1    # Reduced from 0.2 - more focused answers
TOP_P = 0.85         # Reduced from 0.9
REPEAT_PENALTY = 1.15

# === PERFORMANCE ===
BATCH_SIZE = 512           # For embedding encoding
LAZY_LOAD_EMBEDDING = True # Load embedding model only when needed
INDEX_CACHE_ENABLED = True # Cache document index
