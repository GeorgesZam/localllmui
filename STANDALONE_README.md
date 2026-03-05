# LocalRAG - Standalone Windows Application

## 🚀 STANDALONE ONE-FILE VERSION

The standalone version is a **single .exe file** that contains everything needed to run the application. No installation, no dependencies, no external files required!

## 📦 How to Use

### Step 1: Download
1. Go to the [Actions tab](https://github.com/GeorgesZam/localllmui/actions)
2. Click on the latest successful build
3. Download `LocalRAG-Standalone.exe.zip`

### Step 2: Extract
1. Extract the ZIP file
2. You'll get `LocalRAG.exe` (~500-800 MB)

### Step 3: Run
1. Double-click `LocalRAG.exe`
2. A console window will appear (showing loading progress)
3. The main application window will open

That's it! No installation needed.

## ✨ Features Included

The standalone .exe includes:
- ✅ All Python dependencies
- ✅ LLM models (Qwen2.5 0.5B)
- ✅ Embedding model for RAG
- ✅ All document processing capabilities
- ✅ OCR support
- ✅ Code execution features

## 🖥️ System Requirements

- Windows 10 or Windows 11
- 4 GB RAM minimum (8 GB recommended)
- 1 GB free disk space
- No admin rights required

## ⚠️ Notes

### First Launch
- First launch will be slower (~30-60 seconds) as Windows extracts the bundled files
- Subsequent launches will be faster

### Console Window
- A black console window appears when running
- This shows loading progress and error messages
- **DO NOT CLOSE** the console window while using the app

### Antivirus Warnings
- Some antivirus may flag the .exe (false positive)
- This is because PyInstaller executables can look suspicious
- If flagged, add to antivirus exceptions

## 🔧 Troubleshooting

### Application won't start
1. Check Windows Event Viewer for errors
2. Try right-click → "Run as administrator"
3. Check if Windows Defender blocked it

### Models not loading
1. Look at the console window for error messages
2. Check available disk space (need ~2 GB free)
3. Try running from a different location

### Performance issues
1. Close other applications
2. Check CPU usage in Task Manager
3. First run is always slower due to extraction

## 📊 File Sizes

| Component | Size |
|-----------|------|
| Base application | ~100 MB |
| Models bundled | ~400-700 MB |
| Total .exe | ~500-800 MB |

## 🆚 One-File vs One-Dir

**One-File (Current):**
- ✅ Single .exe file
- ✅ Easy to distribute
- ✅ No file dependencies
- ⚠️ Slower first launch
- ⚠️ Larger file size

**One-Dir (Previous):**
- ✅ Faster startup
- ✅ Smaller exe file
- ❌ Multiple files to distribute
- ❌ Folder structure to maintain

## 🎯 Use Cases

Perfect for:
- **Enterprise environments** (like Sodexo PCs)
- **Locked-down systems** where you can't install dependencies
- **Quick deployment** - just copy one file
- **Users without technical knowledge**

## 📝 Building from Source

To build the standalone version yourself:

```bash
# On Windows with Python 3.11
pip install -r requirements.txt
pip install pyinstaller

# Download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5').save('models/embedding_model')"

# Build
pyinstaller --clean --noconfirm LocalRAG.spec

# Output: dist/LocalRAG.exe
```

## 🐛 Known Issues

1. **Windows Defender SmartScreen** - May show warning on first run
   - Solution: Click "More info" → "Run anyway"

2. **Slow extraction** on first run
   - Normal behavior for one-file PyInstaller builds
   - Only happens once per location

3. **Large file size**
   - Due to bundled models and dependencies
   - Necessary for standalone functionality

## 💡 Tips

- Save the .exe to a location with write permissions
- Don't run from network drives (slow)
- Keep the original .exe as backup
- Can be renamed if desired (e.g., "LocalChat.exe")

---

**Need help?** Open an issue on GitHub
