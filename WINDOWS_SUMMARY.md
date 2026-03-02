# Windows Compatibility - What's Ready

## ✅ Files Created for Windows

### Core Files
1. **`LocalRAG.spec`** - PyInstaller configuration for building Windows executable
2. **`src/patterns.py`** - Design patterns (Singleton, Observable) needed by config
3. **`setup.bat`** - Automated setup script for Windows
4. **`run.bat`** - Quick launcher script for Windows
5. **`WINDOWS_SETUP.md`** - Comprehensive Windows setup guide

### What These Files Do

#### `LocalRAG.spec`
- Configures PyInstaller to build a standalone Windows executable
- Bundles all dependencies (Python, libraries, models)
- Creates a single `.exe` file that can be distributed
- No Python installation required for end users

#### `src/patterns.py`
- Provides `SingletonMeta` class used by `ConfigManager`
- Provides `Observable` class for event notifications
- Required by the new config architecture

#### `setup.bat`
- Automated installation script
- Checks Python installation
- Creates virtual environment
- Installs all dependencies
- Downloads models automatically
- Creates necessary directories and symlinks

#### `run.bat`
- Quick launcher to start the app
- Activates virtual environment
- Checks for missing files
- Provides helpful error messages

## 🎯 How to Use on Windows

### Method 1: Automated Setup (Recommended)
```powershell
# 1. Open Command Prompt in project folder
# 2. Run setup script
setup.bat

# 3. Run the app
run.bat
```

### Method 2: Manual Setup
```powershell
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models
mkdir models
# Download qwen2.5-0.5b-instruct-q4_k_m.gguf to models/
mklink /H models\model.gguf models\qwen2.5-0.5b-instruct-q4_k_m.gguf

# 4. Setup embedding model
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-small-en-v1.5'); model.save('models\embedding_model')"

# 5. Run app
python src\main.py
```

### Method 3: Build Executable
```powershell
# 1. Complete Method 2 first
# 2. Build executable
pyinstaller LocalRAG.spec --clean

# 3. Run executable
dist\LocalRAG\LocalRAG.exe
```

## 📦 What Gets Included in the Executable

When you build with PyInstaller:
- Python interpreter (embedded)
- All dependencies
- Source code
- Models (if in models/ folder)
- Data files
- UI components

**Result**: A single folder that can be copied to any Windows machine

## 🔧 Windows-Specific Fixes Applied

### 1. Path Handling
- Config uses forward slashes (works cross-platform)
- Symlink creation handled in setup scripts
- Raw strings for Windows paths where needed

### 2. GPU Support
- CPU-only by default (works on all machines)
- CUDA support available via llama-cpp-python CUDA wheels
- Instructions provided in WINDOWS_SETUP.md

### 3. Dependency Management
- All dependencies listed in requirements.txt
- Alternative installation methods for problematic packages
- Pre-built wheels for llama-cpp-python

### 4. Missing Dependencies
Created `src/patterns.py` with:
- `SingletonMeta` class
- `Observable` class
- `StateEvent` class

## 🚀 Quick Test on Windows

To test if everything works:

```powershell
# Test 1: Check Python
python --version
# Should show: Python 3.11.x

# Test 2: Check imports
python -c "import llama_cpp; print('llama-cpp OK')"
python -c "import customtkinter; print('GUI OK')"
python -c "from sentence_transformers import SentenceTransformer; print('Embeddings OK')"

# Test 3: Check models
dir models
# Should show: model.gguf and embedding_model folder

# Test 4: Run app
python src\main.py
```

## 📋 Current Status

### ✅ Completed
- [x] PyInstaller spec file created
- [x] Missing `patterns.py` created
- [x] Setup script (`setup.bat`)
- [x] Run script (`run.bat`)
- [x] Comprehensive Windows documentation
- [x] Path handling for Windows
- [x] Virtual environment support

### ⚠️ Needs Testing on Actual Windows
- [ ] llama-cpp-python installation (may fail on some systems)
- [ ] GPU acceleration (NVIDIA CUDA)
- [ ] PyInstaller build process
- [ ] Antivirus compatibility
- [ ] Windows 11 compatibility
- [ ] Different Python versions

## 🐛 Known Issues & Solutions

### Issue: llama-cpp-python build fails
**Solution**: Use pre-built wheels
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### Issue: Symlink creation fails
**Solution**: Copy file instead
```powershell
copy models\qwen2.5-0.5b-instruct-q4_k_m.gguf models\model.gguf
```

### Issue: GUI doesn't open
**Solution**: Install tkinter
```powershell
# Usually included with Python
# If missing, reinstall Python with "tcl/tk" option
```

### Issue: "Model not found"
**Solution**: Check paths in config
```python
# In src/config.py
MODEL_FILE = "models/model.gguf"  # Forward slashes work on Windows
```

## 📁 Project Structure (Windows Compatible)

```
localllmui/
├── src/
│   ├── main.py          # Entry point
│   ├── config.py        # Configuration (with Singleton)
│   ├── llm.py           # LLM engine
│   ├── rag.py           # RAG implementation
│   ├── ui.py            # UI components
│   ├── patterns.py      # Design patterns (NEW)
│   └── ...
├── models/
│   ├── model.gguf       # LLM model (symlink)
│   └── embedding_model/ # Embedding model
├── data/                # Document storage
├── venv/                # Virtual environment
├── LocalRAG.spec       # PyInstaller config (NEW)
├── setup.bat           # Setup script (NEW)
├── run.bat             # Run script (NEW)
├── requirements.txt    # Dependencies
└── WINDOWS_SETUP.md    # Windows guide (NEW)
```

## 🎨 UI on Windows

The UI should look identical on Windows:
- Same dark theme
- Same functionality
- Same keyboard shortcuts
- CustomTkinter handles platform differences

## 💡 Tips for Windows Users

1. **Use PowerShell** instead of CMD for better experience
2. **Run as Administrator** if you have permission issues
3. **Add exclusions** in Windows Defender for faster performance
4. **Use SSD** for better model loading performance
5. **Close other apps** when running large models

## 🔮 Future Enhancements

- [ ] Native Windows installer (NSIS)
- [ ] Auto-update mechanism
- [ ] Windows service integration
- [ ] System tray icon
- [ ] Windows notifications
- [ ] Drag-and-drop file support
- [ ] File association (.pdf, .docx -> open with app)

## 📞 Support

For Windows-specific issues:
1. Check `WINDOWS_SETUP.md`
2. Review error logs in console
3. Verify Python version (3.11 recommended)
4. Try manual installation if automated setup fails

---

**Status**: ✅ Windows support ready for testing
**Version**: 1.0
**Platform**: Windows 10/11 (64-bit)
**Tested on**: macOS (needs Windows testing)
