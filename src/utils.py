"""
Utility functions - Optimized with caching.
"""

import os
import sys
from functools import lru_cache


@lru_cache(maxsize=32)
def get_resource_path(relative_path: str) -> str:
    """Gets path compatible with PyInstaller (read-only bundled files). Cached."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return relative_path


_APP_DATA_DIR = None

def get_writable_path(filename: str) -> str:
    """Gets writable path for user data - works with PyInstaller."""
    global _APP_DATA_DIR
    
    if _APP_DATA_DIR is None:
        if hasattr(sys, '_MEIPASS'):
            _APP_DATA_DIR = os.path.join(os.path.expanduser("~"), ".localchat")
        else:
            _APP_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", ".localchat")
        os.makedirs(_APP_DATA_DIR, exist_ok=True)
    
    return os.path.join(_APP_DATA_DIR, filename)


def log_message(prefix: str, message: str, callback=None):
    """Logs a message and optionally calls a callback."""
    print(f"[{prefix}] {message}")
    if callback:
        callback(message)


def get_file_hash(filepath: str) -> str:
    """Get a quick hash of file for cache invalidation."""
    import hashlib
    stat = os.stat(filepath)
    content = f"{filepath}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()[:16]
