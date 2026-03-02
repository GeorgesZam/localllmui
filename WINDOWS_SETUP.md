# Windows Setup Guide for Local RAG

## Prerequisites

### 1. Install Python 3.11
```powershell
# Download from: https://www.python.org/downloads/
# During installation, check "Add Python to PATH"
```

### 2. Install Visual C++ Build Tools
```powershell
# Download and install Visual Studio Build Tools
# https://visualstudio.microsoft.com/downloads/
# Select "Desktop development with C++"
```

### 3. Install Git (optional)
```powershell
# Download from: https://git-scm.com/download/win
```

## Installation Steps

### 1. Clone or Download the Repository
```powershell
git clone https://github.com/your-username/localllmui.git
cd localllmui
```

### 2. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install PyInstaller for building
pip install pyinstaller
```

### 4. Download Models

#### Option A: Automatic Download
```powershell
# Create models directory
mkdir models

# Download LLM model (Qwen2.5:0.5b)
Invoke-WebRequest -Uri "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" -OutFile "models\qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Create symlink for model
mklink /H "models\model.gguf" "models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
```

#### Option B: Manual Download
1. Download from: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
2. Save `qwen2.5-0.5b-instruct-q4_k_m.gguf` to `models/` folder
3. Create a copy named `model.gguf`

### 5. Setup Embedding Model
```powershell
# Save embedding model in correct format
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-small-en-v1.5'); model.save('models\embedding_model')"
```

### 6. Create Data Directory
```powershell
mkdir data
```

## Running the Application

### Option 1: Run Directly (Development)
```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Run the app
python src\main.py
```

### Option 2: Build Executable with PyInstaller
```powershell
# Build the executable
pyinstaller LocalRAG.spec --clean

# Run the executable
.\dist\LocalRAG\LocalRAG.exe
```

## Troubleshooting

### Issue: "Python not found"
**Solution:**
- Make sure Python 3.11 is installed
- Add Python to system PATH
- Restart command prompt

### Issue: "llama-cpp-python installation fails"
**Solution:**
```powershell
# Install pre-built wheel
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cpu-cuda>

# For CPU only
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### Issue: "Model not found"
**Solution:**
- Check that `models/model.gguf` exists
- Verify file path in `src/config.py`
- Use forward slashes or raw strings in paths

### Issue: "CUDA/GPU not working"
**Solution:**
```powershell
# For NVIDIA GPU
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# For CPU only (no GPU)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### Issue: "Tesseract OCR not working"
**Solution:**
```powershell
# Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to system PATH
```

## Windows-Specific Configurations

### File Paths
Windows uses backslashes, but Python can handle forward slashes:

```python
# Good (works on all platforms)
MODEL_FILE = "models/model.gguf"

# Bad (Windows-only with escaping)
MODEL_FILE = "models\\model.gguf"

# Good (raw string)
MODEL_FILE = r"models\model.gguf"
```

### Performance Optimization

For better performance on Windows:

```python
# In src/config.py
THREADS = 4  # Adjust based on your CPU
GPU_LAYERS = -1  # -1 for all GPU layers (Metal/macOS only, use 0 on Windows)
CONTEXT_SIZE = 2048  # Lower for faster response
MAX_TOKENS = 384  # Lower for faster response
```

### GPU Support on Windows

For NVIDIA GPUs:

```powershell
# Install CUDA-enabled llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Then in config.py
GPU_LAYERS = -1  # Use all GPU layers
```

For AMD GPUs:
```powershell
# Use CPU version or install ROCm support
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

## Building Standalone Executable

### Using PyInstaller
```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Build executable
pyinstaller LocalRAG.spec --clean --noconfirm

# Find the executable
dir dist\LocalRAG\LocalRAG.exe
```

### Building One-File Executable
Modify `LocalRAG.spec`:
```python
exe = EXE(
    ...
    onefile=True,  # Create single .exe file
    ...
)
```

## Running as Background Service

### Using Task Scheduler
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (At startup)
4. Action: Start a program
   - Program: `C:\path\to\venv\Scripts\python.exe`
   - Arguments: `C:\path\to\src\main.py`
   - Start in: `C:\path\to\localllmui`

### Using NSSM (Non-Sucking Service Manager)
```powershell
# Download NSSM from https://nssm.cc/download
nssm install LocalRAG "C:\path\to\venv\Scripts\python.exe" "C:\path\to\src\main.py"
nssm start LocalRAG
```

## Firewall Configuration

If running on a network, allow Python through Windows Firewall:

```powershell
# Allow Python
New-NetFirewallRule -DisplayName "Local RAG" -Direction Inbound -Program "python.exe" -Action Allow
```

## Antivirus Exclusions

Add to Windows Defender exclusions:
- Project directory
- Virtual environment directory
- Models directory
- Python executable

```powershell
# Add exclusions (requires admin)
Add-MpPreference -ExclusionPath "C:\path\to\localllmui"
Add-MpPreference -ExclusionProcess "python.exe"
```

## Portable Distribution

To create a portable version:
1. Build with PyInstaller
2. Copy `dist/LocalRAG/` folder
3. Include models in `models/` folder
4. Distribute as ZIP file

## Testing

### Test Installation
```powershell
# Test Python
python --version

# Test imports
python -c "import llama_cpp; print('llama-cpp-python OK')"
python -c "import customtkinter; print('customtkinter OK')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"
```

### Test Model
```powershell
python -c "from llama_cpp import Llama; model = Llama(model_path='models/model.gguf'); print('Model loaded OK')"
```

## Performance Tips

1. **Use SSD** for model storage
2. **Disable antivirus real-time scanning** for project folder
3. **Close unnecessary applications** while running
4. **Use GPU acceleration** if available
5. **Adjust context size** based on available RAM

## Common Windows Errors

### "DLL load failed"
```powershell
# Install Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### "Access denied"
```powershell
# Run as Administrator
# Right-click Command Prompt -> Run as Administrator
```

### "Module not found"
```powershell
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Getting Help

### Check Logs
```powershell
# Run with verbose output
python src\main.py > output.log 2>&1
type output.log
```

### Debug Mode
```python
# In src/config.py, set:
DEBUG = True
VERBOSE = True
```

## Auto-Startup

Add to Windows startup folder:
```
C:\Users\<YourUsername>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
```

Create a batch file `start_rag.bat`:
```batch
@echo off
cd "C:\path\to\localllmui"
call venv\Scripts\activate
python src\main.py
```

## Uninstallation

```powershell
# Deactivate virtual environment
deactivate

# Remove virtual environment
rmdir /s venv

# Remove models (optional)
rmdir /s models

# Remove app directory
cd ..
rmdir /s localllmui
```

## Minimum System Requirements

- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 2 GB free space
- **CPU**: 4 cores recommended
- **GPU**: Optional (NVIDIA with CUDA for acceleration)

## Recommended System Specifications

- **OS**: Windows 11 (64-bit)
- **RAM**: 16 GB
- **Storage**: SSD with 10 GB free
- **CPU**: 6+ cores
- **GPU**: NVIDIA RTX 3060 or better

---

**Status**: ✅ Ready for Windows
**Version**: 1.0
**Platform**: Windows 10/11 (64-bit)
