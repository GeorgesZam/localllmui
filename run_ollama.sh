#!/bin/bash
# Run the app with Ollama engine

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed!"
    echo "Please install from: https://ollama.com/download"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3
fi

# Pull default model if not present
if ! ollama list | grep -q "qwen2.5:0.5b"; then
    echo "Pulling qwen2.5:0.5b model..."
    ollama pull qwen2.5:0.5b
fi

# Run the app
echo "Starting LocalRAG with Ollama..."
USE_OLLAMA=1 python src/main.py
