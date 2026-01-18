"""
Configuration - Edit this file to customize the app.
"""

# === APP ===
APP_NAME = "Local Chat"
WINDOW_SIZE = "900x700"

# === MODEL ===
MODEL_FILE = "models/model.gguf"
CONTEXT_SIZE = 4096  # Increased for RAG context
MAX_TOKENS = 512
import os THREADS = os.cpu_count() or 4

# === PROMPT ===
SYSTEM_PROMPT = """You are a helpful assistant specialized in administrative documents. 
Answer questions based on the provided context. If the answer is not in the context, say so.
Answer in the same language as the user."""

STOP_TOKENS = ["<|im_end|>"]

# === RAG ===
RAG_ENABLED = True
RAG_FOLDER = "data"
RAG_CHUNK_SIZE = 300  # Smaller chunks for better precision
RAG_TOP_K = 5  # More results for admin docs

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
