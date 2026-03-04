"""
PyInstaller hook for llama-cpp-python on Windows.

This hook ensures that llama-cpp-python and its DLL dependencies
are properly bundled and loaded in PyInstaller builds.
"""

import os
import sys
from PyInstaller.utils.hooks import get_package_paths, collect_data_files, collect_submodules, is_module_satisfies

# Collect all llama_cpp modules
datas = collect_data_files('llama_cpp', include_py_files=False)
binaries = []
hiddenimports = collect_submodules('llama_cpp')

# Add specific hidden imports that are often missed
hiddenimports += [
    'llama_cpp.llama_cpp',
    'llama_cpp._utils',
]

# On Windows, we need to explicitly collect DLLs
if sys.platform == 'win32':
    # Try to find and collect llama_cpp DLLs
    try:
        import llama_cpp
        llama_path = os.path.dirname(llama_cpp.__file__)

        # Look for DLL files in the llama_cpp package
        for root, dirs, files in os.walk(llama_path):
            for file in files:
                if file.endswith('.dll') or file.endswith('.pyd'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, os.path.dirname(llama_path))
                    binaries.append((full_path, 'llama_cpp'))
    except ImportError:
        pass
