# LocalRAG - Ollama Version

A lightweight version of LocalRAG that uses [Ollama](https://ollama.com) instead of llama_cpp.

## Why Ollama?

| Feature | llama_cpp Version | Ollama Version |
|---------|------------------|----------------|
| Download Size | ~500-800 MB | ~50-100 MB |
| Models Bundled | Yes (limited) | No (unlimited) |
| GPU Support | Basic | Excellent |
| Performance | Good | Better |
| Installation | One file | One file (auto-installs Ollama) |

## Installation

### Just Run It!

1. Download `LocalRAG-Ollama.exe`
2. Double-click to run
3. **First launch only:** The app will automatically download and install Ollama
4. Start chatting!

That's it - no manual installation required!

### What Happens on First Launch?

1. App checks if Ollama is installed
2. If not, it downloads the installer (~100 MB)
3. Runs the installer automatically
4. Pulls the default model (qwen2.5:0.5b, ~400 MB)
5. Starts the app!

## Model Switching

You can switch models from within the app, or via command line:

```bash
# Pull a new model
ollama pull llama3.2:3b

# Restart LocalRAG and select the new model
```

## Troubleshooting

### "Ollama not found"
- Install Ollama from https://ollama.com/download
- Restart the application

### "Ollama server not running"
- The app will try to start it automatically
- Or run: `ollama serve`

### Model not available
- Run: `ollama pull <model-name>`
- Check available models: `ollama list`

## Development

To build from source:

```bash
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --clean --noconfirm LocalRAG-Ollama.spec
```

Output: `dist/LocalRAG-Ollama.exe`
