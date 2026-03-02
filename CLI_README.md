# Local LLM UI - CLI Mode

A command-line interface for the Local LLM Assistant that works with or without downloaded AI models.

## Quick Start

### Method 1: Using the launcher script
```bash
./run_cli.sh
```

### Method 2: Direct Python
```bash
source venv/bin/activate
python src/cli.py
```

## Features

### ✅ Currently Working (No Models Required)
- Interactive chat interface
- Conversation history
- Command system (`/help`, `/clear`, `/quit`, `/stats`, `/files`)
- Mock responses for testing
- Fast startup (< 1 second)

### 🔄 Available with Models
- Full LLM responses (Qwen2.5-0.5b-Instruct)
- RAG document search
- Embedding-based semantic search
- Code execution capabilities

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help information |
| `/clear` | Clear conversation history |
| `/quit` | Exit the application |
| `/files` | Show loaded documents |
| `/stats` | Show conversation statistics |

## Example Session

```
🚀 Starting Local LLM CLI...
✓ Using mock LLM (fast startup)

============================================================
🤖 Local LLM Assistant - Ready!
============================================================

You: What is Python?

Assistant: I understand you're asking about: "What is Python?..."

As a language model, I can help you with various tasks including:
• Answering questions
• Writing and editing text
• Coding assistance
• Analysis and explanations

Note: This is a mock response running in simulation mode.

You: /stats

============================================================
CONVERSATION STATISTICS
============================================================
  Messages exchanged: 2
  User messages: 1
  Assistant responses: 1

You: /quit

👋 Goodbye!
```

## Setup

### 1. Virtual Environment (Already Created)
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run CLI
```bash
./run_cli.sh
```

## Adding Documents (RAG)

To enable document search with RAG:

1. Create a `data` folder in `src/`:
```bash
mkdir -p src/data
```

2. Add your documents:
```bash
cp your_document.txt src/data/
cp your_pdf.pdf src/data/
```

3. Supported formats:
- Text files (`.txt`, `.md`)
- PDF (`.pdf`) - requires `PyPDF2`
- Word documents (`.docx`) - requires `python-docx`
- Excel (`.xlsx`) - requires `openpyxl`
- PowerPoint (`.pptx`) - requires `python-pptx`

## Enabling Full AI (Optional)

To use real AI responses instead of mock mode:

### Download Models

1. **Create models directory:**
```bash
mkdir -p src/models
```

2. **Download LLM model (Qwen2.5-0.5b-Instruct):**
```bash
cd src/models
# Download from HuggingFace or other source
# Example: wget https://huggingface.co/.../qwen2.5-0.5b-instruct-q4_k_m.gguf
```

3. **Download embedding model (BGE-small-en-v1.5):**
```bash
cd src/models
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5').save('embedding_model')"
```

### Required Dependencies

```bash
pip install llama-cpp-python sentence-transformers
```

For GPU support (Apple Silicon):
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir
```

For GPU support (NVIDIA CUDA):
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

## Troubleshooting

### Issue: CLI not starting
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install dependencies
pip install pytest pytest-cov

# Run CLI
python src/cli.py
```

### Issue: Models not loading
The CLI automatically falls back to mock mode if models aren't found. This is normal and allows you to test the interface.

### Issue: RAG not working
```bash
# Check data folder exists
ls -la src/data/

# Add test document
echo "Test content" > src/data/test.txt

# Restart CLI
./run_cli.sh
```

## Development

### Run Tests
```bash
pytest tests/ -v
```

### Test CLI with Input
```bash
echo -e "Hello!\n/quit\n" | python src/cli.py
```

## Architecture

```
src/
├── cli.py          # CLI entry point
├── llm.py          # LLM engine
├── rag.py          # RAG functionality
├── config.py       # Configuration
├── utils.py        # Utilities
├── data/           # Documents for RAG
└── models/         # AI models (optional)
```

## License

This is a local AI assistant that runs entirely on your machine. No data is sent to external servers (except when downloading models).
