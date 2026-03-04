"""
Windows-specific helper functions for PyInstaller builds.

This module provides utilities to handle Windows-specific issues
when running applications built with PyInstaller.
"""

import sys
import platform


def force_window_focus(window):
    """
    Force window to grab focus on Windows.

    On Windows, PyInstaller-built apps with console=False
    often have popup windows that don't automatically receive focus.
    This function uses platform-specific methods to force the window
    to the foreground.

    Args:
        window: A tkinter or customtkinter window/toplevel instance
    """
    if platform.system() == "Windows":
        try:
            # Force window to top
            window.attributes('-topmost', True)
            window.update()
            window.attributes('-topmost', False)

            # Lift window to top of stacking order
            window.lift()
            window.focus_force()

            # Make window visible and ensure it's rendered
            window.wm_state('normal')
            window.update_idletasks()

        except Exception as e:
            print(f"[WindowsHelper] Warning: Could not force window focus: {e}")


def is_frozen():
    """Check if running as PyInstaller frozen executable."""
    return hasattr(sys, '_MEIPASS')


def get_executable_dir():
    """Get the directory containing the executable (frozen or script)."""
    if is_frozen():
        return sys._MEIPASS
    else:
        import os
        return os.path.dirname(os.path.abspath(__file__))
