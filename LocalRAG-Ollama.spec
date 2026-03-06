# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Local RAG Application - OLLAMA VERSION

This creates a single .exe file that uses Ollama instead of llama_cpp.
Benefits:
- Much smaller size (~50-100 MB instead of 500-800 MB)
- Better GPU optimization
- No need to bundle models
- Easier model switching
- AUTO-INSTALLS Ollama on first run!

The app will automatically download and install Ollama if not present.
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Collect ALL source files from src directory
src_dir = Path('src')
datas = []

# Add all .py files from src directory
for py_file in src_dir.glob('*.py'):
    datas.append((str(py_file), 'src'))

# Also collect subdirectories
for item in src_dir.rglob('*.py'):
    if item.parent != src_dir:
        rel_path = item.relative_to(src_dir)
        datas.append((str(item), str(rel_path.parent)))

# Add skills directory
skills_dir = Path('skills')
if skills_dir.exists():
    for item in skills_dir.rglob('*.md'):
        datas.append((str(item), 'skills'))
    if (skills_dir / 'skills_config.json').exists():
        datas.append((str(skills_dir / 'skills_config.json'), 'skills'))

# Add Ollama installer script
if Path('install_ollama.py').exists():
    datas.append(('install_ollama.py', '.'))

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.ttk',
        'customtkinter',
        'customtkinter.windows',
        'customtkinter.windows.widgets',
        # Ollama engine
        'ollama_engine',
        'requests',
        'urllib3',
        # RAG components
        'sentence_transformers',
        'sentence_transformers.models',
        'sentence_transformers.SentenceTransformer',
        'transformers',
        'tokenizers',
        'numpy',
        'torch',
        'PIL',
        'PIL._tkinter_finder',
        # Document processing
        'openpyxl',
        'openpyxl.cell._writer',
        'PyPDF2',
        'python_docx',
        'docx',
        'docx.opc.constants',
        'docx.oxml.xmlchar',
        'docx.opc.packaging',
        'docx.opc.shared',
        'pptx',
        'pptx.enum.shapes',
        'tiktoken',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        # Internal modules
        'patterns',
        'patterns.singleton',
        'patterns.observer',
        'config',
        'ui',
        'conversations',
        'ocr',
        'utils',
        'response_handler',
        'skills_manager',
        'code_executor',
        'model_manager',
        'model_catalog',
        'controllers',
        'observers',
        'parsers',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_customtkinter.py', 'pyi_rth_tiktoken.py'],
    excludes=[
        # Exclude llama_cpp - we use Ollama instead
        'llama_cpp',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# STANDALONE ONE-FILE EXE
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='LocalRAG-Ollama',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
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

# NO COLLECT SECTION - This is a one-file build
