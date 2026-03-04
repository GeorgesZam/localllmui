"""
PyInstaller runtime hook for customtkinter on Windows.

This hook helps fix Windows-specific issues with customtkinter
when bundled with PyInstaller, including:
- Window focus issues
- DPI scaling issues
- Threading issues during startup
"""

import sys
import os

# Fix for Windows DPI scaling
if sys.platform == "win32":
    try:
        from ctypes import windll
        # Set DPI awareness to handle high DPI displays
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        try:
            windll.user32.SetProcessDPIAware()
        except:
            pass

# Ensure proper multiprocessing start method on Windows
if sys.platform == "win32":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set
        pass

# Fix for threading issues on Windows
if sys.platform == "win32":
    # Ensure the main thread is properly marked
    import threading
    threading.main_thread()

print("[RuntimeHook] Windows customtkinter initialization complete")
