# LocalRAG - Ollama Version

A lightweight version of LocalRAG that uses [Ollama](https://ollama.com) instead of llama_cpp.

## Why Ollama?

| Feature | llama_cpp Version | Ollama Version |
|---------|------------------|----------------|
| Download Size | ~500-800 MB | ~50-100 MB |
| Models Bundled | Yes (limited) | No (unlimited) |
| GPU Support | Basic | Excellent |
| Performance | Good | Better |
| Installation | One file | One file + Ollama |

## Installation

### Step 1: Install Ollama

1. Download from: https://ollama.com/download
2. Run the installer
3. Verify installation: `ollama --version`

### Step 2: Pull a Model

```bash
ollama pull qwen2.5:0.5b
```

Available models:
- `qwen2.5:0.5b` - Smallest, fastest (~400MB)
- `qwen2.5:1.5b` - Better quality (~900MB)
- `phi3:mini` - Great performance (~2GB)
- `llama3.2:1b` - Meta's model (~1GB)

### Step 3: Run LocalRAG

1. Download `LocalRAG-Ollama.exe`
2. Double-click to run
3. Start chatting!

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
