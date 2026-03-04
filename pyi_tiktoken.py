"""
PyInstaller hook for tiktoken.

This hook ensures that tiktoken's dynamically loaded extensions
are properly included in the PyInstaller bundle.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all tiktoken data files (includes the registry)
datas = collect_data_files('tiktoken')

# Collect all submodules to ensure extensions are included
hiddenimports = collect_submodules('tiktoken')
