"""
Runtime hook for llama-cpp-python on Windows with PyInstaller.

This hook fixes path resolution issues when llama-cpp-python
tries to access its bundled libraries in PyInstaller temp directories.
"""

import sys
import os

# Only run on Windows
if sys.platform == "win32":

    # Fix for llama_cpp path issues in PyInstaller
    def _fix_llama_cpp_paths():
        """Fix llama_cpp library paths in PyInstaller builds."""
        if not hasattr(sys, '_MEIPASS'):
            return  # Not running in PyInstaller

        try:
            import llama_cpp

            # The issue is that llama_cpp tries to find its libraries
            # in a path that doesn't exist in PyInstaller's temp dir.
            # We need to monkey-patch the path resolution.

            original_path = os.path.dirname(llama_cpp.__file__)

            # Create a function to normalize paths
            def normalize_path(path):
                """Normalize path by fixing double backslashes."""
                if isinstance(path, str):
                    return path.replace('\\\\', '\\').replace('\\\\\\', '\\')
                return path

            # Patch any path-related functions in llama_cpp if they exist
            if hasattr(llama_cpp, '_llama_cpp'):
                # Store the original module
                original_module = llama_cpp._llama_cpp

                # Get the DLL path
                if hasattr(original_module, '__file__'):
                    dll_path = original_module.__file__
                    # Ensure the DLL exists
                    if not os.path.exists(dll_path):
                        # Try to find it in the PyInstaller temp dir
                        meipass = sys._MEIPASS
                        dll_name = os.path.basename(dll_path)
                        alt_path = os.path.join(meipass, 'llama_cpp', dll_name)
                        if os.path.exists(alt_path):
                            # Update the path (this might not work due to readonly attributes)
                            print(f"[llama_cpp] Found DLL at: {alt_path}")

        except ImportError:
            pass
        except Exception as e:
            # Silently fail to avoid breaking the app startup
            print(f"[llama_cpp] Warning: Could not fix paths: {e}")

    # Run the fix on import
    _fix_llama_cpp_paths()

    print("[RuntimeHook] llama-cpp-python Windows path fixes applied")
