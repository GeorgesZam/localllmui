"""
Configuration - Edit this file to customize the app.
"""

# === APP ===
APP_NAME = "Local Chat"
WINDOW_SIZE = "900x700"

# === MODEL ===
MODEL_FILE = "models/model.gguf"
CONTEXT_SIZE = 2048
MAX_TOKENS = 512
THREADS = 4

# === PROMPT TEMPLATE (Qwen2 format) ===
SYSTEM_PROMPT = "You are a helpful assistant. Answer in the same language as the user."
PROMPT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
"""
STOP_TOKENS = ["<|im_end|}"]

# === RAG ===
RAG_ENABLED = True
RAG_FOLDER = "data"
RAG_CHUNK_SIZE = 500
RAG_TOP_K = 3

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