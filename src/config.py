"""
Configuration - Edit this file to customize the app.
"""

import os

# === APP ===
APP_NAME = "Local Chat"
WINDOW_SIZE = "900x700"

# === MODEL ===
MODEL_FILE = "models/model.gguf"
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
RAG_CHUNK_SIZE = 512          # CHANGED: était 300, maintenant 512
RAG_CHUNK_OVERLAP = 100       # NEW: overlap entre chunks
RAG_TOP_K = 3                 # CHANGED: était 5, maintenant 3 (moins de bruit)
RAG_MIN_SCORE = 0.25          # NEW: seuil minimum de similarité

# === EMBEDDING MODEL ===
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # NEW: meilleur modèle pour retrieval

# === UI COLORS ===
COLORS = {
    "bg": "#1a1a2e",
    "bg_chat": "#16213e",
    "fg": "#eaeaea",
    "user": "#4a9eff",
    "bot": "#50fa7b",
    "system": "#888888",
    "error": "#ff5555",
    "accent": "#4a9eff",
}

# === UI FONTS ===
FONTS = {
    "title": ("Arial", 20, "bold"),
    "chat": ("Consolas", 11),
    "button": ("Arial", 11, "bold"),
}
