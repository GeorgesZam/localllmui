"""
Configuration - Edit this file to customize the app.
"""

import os

# === APP ===
APP_NAME = "Local Chat"
WINDOW_SIZE = "1100x700"

# === MODEL ===
MODEL_FILE = "models/model.gguf"
EMBEDDING_MODEL_FOLDER = "embedding_model"
CONTEXT_SIZE = 4096
MAX_TOKENS = 512
THREADS = 8

# === PROMPT ===
SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based ONLY on the provided context documents.
If the answer is not found in the context, say "I don't have this information in the provided documents."
Be specific and quote relevant parts when possible.
Answer in the same language as the user."""

STOP_TOKENS = ["<|im_end|>", "<end_of_turn>"]

# === RAG ===
RAG_ENABLED = True
RAG_FOLDER = "data"
RAG_CHUNK_SIZE = 512
RAG_CHUNK_OVERLAP = 100
RAG_TOP_K = 3
RAG_MIN_SCORE = 0.25
RAG_SHOW_SOURCES = True

# === SAMPLING ===
TEMPERATURE = 0.2
TOP_P = 0.9
REPEAT_PENALTY = 1.1
