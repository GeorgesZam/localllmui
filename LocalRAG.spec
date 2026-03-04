# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Local RAG Application
Works on Windows, macOS, and Linux
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Collect llama_cpp DLLs on Windows
def collect_llama_cpp_dlls():
    """Collect llama_cpp DLL files for Windows PyInstaller builds."""
    dlls = []
    if sys.platform == "win32":
        try:
            import llama_cpp
            llama_path = os.path.dirname(llama_cpp.__file__)

            # Collect all DLL and PYD files from llama_cpp
            for root, dirs, files in os.walk(llama_path):
                for file in files:
                    if file.endswith(('.dll', '.pyd')):
                        full_path = os.path.join(root, file)
                        # Calculate relative path from llama_cpp directory
                        rel_path = os.path.relpath(full_path, os.path.dirname(llama_path))
                        dlls.append((full_path, os.path.dirname(rel_path)))

            print(f"[Spec] Collected {len(dlls)} llama_cpp DLL files")
        except ImportError:
            print("[Spec] Warning: llama_cpp not found during spec collection")
        except Exception as e:
            print(f"[Spec] Warning: Error collecting llama_cpp DLLs: {e}")

    return dlls

# Collect DLLs before defining Analysis
llama_dlls = collect_llama_cpp_dlls()

# Collect all source files
src_dir = Path('src')
datas = [
    (str(src_dir / 'config.py'), 'src'),
    (str(src_dir / 'llm.py'), 'src'),
    (str(src_dir / 'rag.py'), 'src'),
    (str(src_dir / 'ui.py'), 'src'),
    (str(src_dir / 'conversations.py'), 'src'),
    (str(src_dir / 'ocr.py'), 'src'),
    (str(src_dir / 'utils.py'), 'src'),
    (str(src_dir / 'patterns.py'), 'src'),
    (str(src_dir / 'windows_helper.py'), 'src'),
]

# Add models directory if it exists
models_dir = Path('models')
if models_dir.exists():
    # Collect embedding model
    embedding_model = models_dir / 'embedding_model'
    if embedding_model.exists():
        for item in embedding_model.rglob('*'):
            if item.is_file():
                rel_path = item.relative_to(models_dir)
                datas.append((str(item), str(Path('models') / rel_path.parent)))

    # Add model file
    model_file = models_dir / 'model.gguf'
    if model_file.exists():
        datas.append((str(model_file), 'models'))
    elif (models_dir / 'qwen2.5-0.5b-instruct-q4_k_m.gguf').exists():
        datas.append((str(models_dir / 'qwen2.5-0.5b-instruct-q4_k_m.gguf'), 'models'))

# Add data directory
data_dir = Path('data')
if data_dir.exists():
    for item in data_dir.rglob('*'):
        if item.is_file():
            rel_path = item.relative_to(data_dir)
            datas.append((str(item), str(Path('data') / rel_path.parent)))

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=llama_dlls,  # Add collected llama_cpp DLLs
    datas=datas,
    hiddenimports=[
        'customtkinter',
        'llama_cpp',
        'sentence_transformers',
        'sentence_transformers.models',
        'sentence_transformers.SentenceTransformer',
        'transformers',
        'tokenizers',
        'numpy',
        'torch',
        'PIL',
        'PIL._tkinter_finder',
        'pkg_resources.py2_warn',
        'openpyxl',
        'openpyxl.cell._writer',
        'PyPDF2',
        'python_docx',
        'docx',
        'pptx',
        'tiktoken',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        'patterns',
        'config',
        'llm',
        'rag',
        'ui',
        'conversations',
        'ocr',
        'utils',
        'windows_helper',  # Add Windows helper module
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_customtkinter.py', 'pyi_rth_llama_cpp.py'],  # Add Windows runtime hooks
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'sklearn',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='LocalRAG',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to True for debugging Windows issues
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Windows-specific
    icon=None,
    manifest=None,
    uac_admin=False,
    uac_uiaccess=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LocalRAG',
)
