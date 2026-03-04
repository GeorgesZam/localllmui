# Windows .exe Build Debugging Guide

This guide helps you diagnose and fix common issues when building and running the application on Windows.

## Common Issues and Solutions

### Issue 1: Popup Windows Go to Background

**Symptoms:**
- When you open the Model Catalog or RAG Config windows, they appear behind the main window
- Windows don't get focus when opened

**Solution:**
The `windows_helper.py` module now includes the `force_window_focus()` function which is called in both popup windows. This uses platform-specific methods to force windows to the foreground.

**If it still happens:**
1. Set `console=True` in `LocalRAG.spec` to see error messages
2. Check if Windows is blocking the focus (some security software does this)
3. Try running as administrator

### Issue 2: Model Loading Hangs Forever

**Symptoms:**
- Application starts but shows "Loading model..." indefinitely
- Embedding model never loads
- Application becomes unresponsive

**Solutions:**

#### A. Check Model Files
Ensure the model files exist in the correct location:
```bash
dist/LocalRAG/models/
├── embedding_model/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── qwen2.5-0.5b-instruct-q4_k_m.gguf
```

#### B. Increase Timeout
The embedding model loading now has a 30-second timeout. If your model is larger, edit `src/rag.py`:
```python
# In EmbeddingModel.load() method
load_thread.join(timeout=60)  # Increase from 30 to 60 seconds
```

#### C. Disable Embedding Model Temporarily
To test if the embedding model is the issue, you can temporarily disable it:
1. Open `src/config.py`
2. Set `lazy_load_embedding = True`
3. Set `rag_enabled = False`

#### D. Check for Threading Issues
The runtime hook `pyi_rth_customtkinter.py` handles multiprocessing issues. Make sure it's in the spec file:
```python
runtime_hooks=['pyi_rth_customtkinter.py'],
```

### Issue 3: Very Slow Startup

**Symptoms:**
- Application takes 30+ seconds to start
- High CPU usage during startup

**Solutions:**

#### A. Disable UPX Compression
In `LocalRAG.spec`, set:
```python
upx=False,  # Instead of upx=True
```

#### B. Exclude Unnecessary Modules
The spec already excludes some heavy modules. You can exclude more:
```python
excludes=[
    'matplotlib',
    'scipy',
    'pandas',
    'sklearn',
    'IPython',  # Add this
    'pytest',   # Add this
],
```

#### C. Use One-File Mode (Alternative)
Try one-file mode instead of one-dir (note: first startup will be slower):
```python
exe = EXE(
    # ... same settings ...
    name='LocalRAG',
    # Remove the COLLECT section entirely
)
```

### Issue 4: "Module Not Found" Errors

**Symptoms:**
- Application crashes with ImportError
- Console shows missing modules

**Solution:**
Add the missing module to `hiddenimports` in `LocalRAG.spec`:
```python
hiddenimports=[
    # ... existing imports ...
    'missing_module_name',  # Add the missing module
],
```

### Issue 5: llama_cpp Path Error (WinError 3)

**Symptoms:**
```
Failed to switch model: [WinError 3] The system cannot find the path specified:
'C:\Users\MICHEL~1\AppData\Local\\Temp\\_MEI40922\\llama_cpp\lib'
```
- Error occurs when switching models
- Path contains double backslashes (`\\`)
- Points to PyInstaller temp directory that no longer exists

**Root Cause:**
PyInstaller extracts bundled files to a temporary directory (`_MEI*`). When llama-cpp-python tries to find its DLL dependencies during model switching, it uses incorrect path resolution that doesn't account for PyInstaller's temporary extraction.

**Solutions:**

#### A. Verify llama_cpp DLL Collection
The spec file now includes a function to collect llama_cpp DLLs:

```python
# In LocalRAG.spec
def collect_llama_cpp_dlls():
    """Collect llama_cpp DLL files for Windows PyInstaller builds."""
    dlls = []
    if sys.platform == "win32":
        try:
            import llama_cpp
            llama_path = os.path.dirname(llama_cpp.__file__)
            for root, dirs, files in os.walk(llama_path):
                for file in files:
                    if file.endswith(('.dll', '.pyd')):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, os.path.dirname(llama_path))
                        dlls.append((full_path, os.path.dirname(rel_path)))
        except Exception as e:
            print(f"[Spec] Warning: Error collecting llama_cpp DLLs: {e}")
    return dlls
```

#### B. Test llama_cpp Installation
Run the diagnostic script to check llama_cpp:

```bash
# Test llama_cpp before building
python test_llama_cpp_windows.py
```

This will show:
- If llama_cpp is installed correctly
- Where the DLL files are located
- If all dependencies are present

#### C. Reinstall llama_cpp with Correct Build
Sometimes the pre-built wheels don't work well with PyInstaller:

```bash
# Uninstall existing
pip uninstall llama-cpp-python -y

# Install from source (slower but more compatible)
pip install llama-cpp-python --no-binary llama-cpp-python

# Or use pre-built wheel with specific version
pip install llama-cpp-python==0.2.28
```

#### D. Use Static Binary (Alternative)
If DLL issues persist, you can use a pre-built llama.cpp binary instead:

1. Download llama.cpp binary for Windows from: https://github.com/ggerganov/llama.cpp/releases
2. Place the `.exe` file in your project directory
3. Modify the code to use the binary instead of the Python package

#### E. Check for VC++ Redistributable
Some llama_cpp DLLs require Microsoft Visual C++ Redistributable:

1. Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Install on the target Windows system
3. Rebuild your application

#### F. Verify Runtime Hook
Ensure the runtime hook is included in the spec:

```python
runtime_hooks=['pyi_rth_customtkinter.py', 'pyi_rth_llama_cpp.py'],
```

This hook runs before your application starts and fixes path resolution issues.

#### G. Build with --debug
For more detailed error information:

```bash
pyinstaller --clean --noconfirm --debug=all LocalRAG.spec
```

This will show exactly where the path resolution fails.

## Debugging Steps

### Step 1: Enable Console
Set `console=True` in `LocalRAG.spec`:
```python
exe = EXE(
    # ...
    console=True,  # Changed from False
    # ...
)
```

### Step 2: Add Debug Prints
Add print statements in critical sections:
```python
# In src/main.py App.__init__
print("[DEBUG] App initialized")
print(f"[DEBUG] Model path: {config.MODEL_FILE}")
```

### Step 3: Check for Resource Issues
Monitor Windows Task Manager while the app starts:
- **CPU**: Should spike then settle
- **Memory**: Should increase steadily then plateau
- **Disk**: Should show activity during model loading

If any metric stays at 100% for more than a minute, there's a hang.

### Step 4: Test in Isolation
Test components separately:
```python
# test_model_load.py
from src.rag import EmbeddingModel

model = EmbeddingModel()
if model.load():
    print("✓ Embedding model loaded")
else:
    print("✗ Embedding model failed to load")
```

## Building for Windows

### From macOS/Linux (Cross-Compilation)
PyInstaller doesn't support cross-compilation well. Use one of these methods:

1. **Use Wine** (Linux):
   ```bash
   wine python -m PyInstaller LocalRAG.spec
   ```

2. **Use GitHub Actions**:
   Create a workflow that builds on Windows runners.

3. **Use a Windows VM**:
   Build in a virtual machine.

### From Windows
```bash
# Install dependencies
pip install -r requirements.txt

# Build using the script
python build_windows.py

# Or manually
pyinstaller --clean --noconfirm LocalRAG.spec
```

## Performance Optimizations

### Reduce Model Size
Use a smaller model for testing:
- Replace `qwen2.5-0.5b-instruct-q4_k_m.gguf` (300MB)
- With an even smaller model if available

### Lazy Loading
Components are already lazy-loaded where possible. The RAG system loads the embedding model only when needed.

### Optimize Context Size
In `src/config.py`, reduce context size:
```python
self.context_size = 8192  # Instead of 32768
```

## Getting Help

If issues persist:
1. Collect console output (with console=True)
2. Note your Windows version and specs
3. Check the PyInstaller documentation: https://pyinstaller.org/
4. Open an issue with the collected information

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Windows in background | Set console=True, check windows_helper.py |
| Model hangs | Increase timeout to 60s in rag.py |
| Slow startup | Set upx=False in spec |
| Import errors | Add to hiddenimports in spec |
| llama_cpp WinError 3 | Run test_llama_cpp_windows.py, check DLL collection in spec |
| General issues | Enable console and check output |
