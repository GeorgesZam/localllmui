"""
Utility functions shared across modules.
"""

import os
import sys


def get_resource_path(relative_path: str) -> str:
    """Gets path compatible with PyInstaller (read-only bundled files)."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return relative_path


def get_writable_path(filename: str) -> str:
    """Gets writable path for user data - works with PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        app_data = os.path.join(os.path.expanduser("~"), ".localchat")
    else:
        app_data = os.path.join(os.path.dirname(__file__), "..", ".localchat")
    
    os.makedirs(app_data, exist_ok=True)
    return os.path.join(app_data, filename)


def log_message(prefix: str, message: str, callback=None):
    """Logs a message and optionally calls a callback."""
    print(f"[{prefix}] {message}")
    if callback:
        callback(message)
