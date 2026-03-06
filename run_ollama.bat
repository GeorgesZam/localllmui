@echo off
REM Run the app with Ollama engine on Windows

echo Checking Ollama installation...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo Ollama is not installed!
    echo Please install from: https://ollama.com/download
    pause
    exit /b 1
)

echo Ollama is installed!

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Starting Ollama server...
    start /B ollama serve
    timeout /t 3 /nobreak >nul
)

REM Pull default model if not present
ollama list | findstr "qwen2.5:0.5b" >nul 2>&1
if errorlevel 1 (
    echo Pulling qwen2.5:0.5b model...
    ollama pull qwen2.5:0.5b
)

echo Starting LocalRAG with Ollama...
python src\main.py

pause
