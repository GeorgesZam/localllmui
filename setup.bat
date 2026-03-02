@echo off
REM Local RAG Application Setup Script for Windows
REM This script automates the installation process

echo ======================================
echo   Local RAG Setup Wizard
echo ======================================
echo.

REM Check Python version
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.11 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version
echo [OK] Python is installed
echo.

REM Check if in project directory
if not exist "src\main.py" (
    echo [ERROR] Please run this script from the project root directory!
    echo The directory should contain src\main.py
    pause
    exit /b 1
)

REM Create virtual environment
echo [2/6] Creating virtual environment...
if exist "venv\" (
    echo [SKIP] Virtual environment already exists
) else (
    python -m venv venv
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [4/6] Upgrading pip and installing build tools...
python -m pip install --upgrade pip
pip install wheel setuptools
echo [OK] pip upgraded
echo.

REM Install dependencies
echo [5/6] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies!
    echo Trying alternative installation method...
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    pip install -r requirements.txt
)
echo [OK] Dependencies installed
echo.

REM Create necessary directories
echo [6/6] Creating directories...
if not exist "models\" mkdir models
if not exist "data\" mkdir data
echo [OK] Directories created
echo.

REM Download models
echo ======================================
echo   Model Download
echo ======================================
echo.

REM Check if model already exists
if exist "models\model.gguf" (
    echo [SKIP] Model already exists: models\model.gguf
) else if exist "models\qwen2.5-0.5b-instruct-q4_k_m.gguf" (
    echo [INFO] Model file exists, creating symlink...
    mklink /H "models\model.gguf" "models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
    echo [OK] Symlink created
) else (
    echo [INFO] Downloading Qwen2.5:0.5B model (469 MB)...
    echo This may take a few minutes depending on your connection...
    echo.
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf' -OutFile 'models\qwen2.5-0.5b-instruct-q4_k_m.gguf'}"

    if exist "models\qwen2.5-0.5b-instruct-q4_k_m.gguf" (
        echo [OK] Model downloaded

        REM Create symlink
        echo [INFO] Creating symlink...
        mklink /H "models\model.gguf" "models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
        echo [OK] Symlink created
    ) else (
        echo [ERROR] Failed to download model!
        echo Please download manually from:
        echo https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
    )
)
echo.

REM Setup embedding model
if exist "models\embedding_model\config.json" (
    echo [SKIP] Embedding model already exists
) else (
    echo [INFO] Setting up embedding model (BGE-small-en)...
    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-small-en-v1.5'); model.save('models\embedding_model'); print('[OK] Embedding model saved')"
)
echo.

REM Create config symlink for Windows
if not exist "models\embedding_model" (
    if exist "models\models--BAAI--bge-small-en-v1.5" (
        echo [INFO] Creating embedding model directory reference...
        mklink /D "models\embedding_model" "models\models--BAAI--bge-small-en-v1.5"
        echo [OK] Symlink created
    )
)
echo.

REM Final check
echo ======================================
echo   Setup Complete!
echo ======================================
echo.

echo Checking installation...
python -c "import llama_cpp; import customtkinter; import sentence_transformers" 2>nul
if errorlevel 1 (
    echo [WARNING] Some dependencies may not be installed correctly
    echo Try running: pip install -r requirements.txt --force-reinstall
) else (
    echo [OK] All dependencies installed correctly
)
echo.

echo ======================================
echo   Quick Start
echo ======================================
echo.
echo To run the application:
echo   1. Double-click run.bat
echo   2. Or run: python src\main.py
echo   3. Or build executable: pyinstaller LocalRAG.spec
echo.
echo For more information, see:
echo   - README.md
echo   - WINDOWS_SETUP.md
echo.
echo Press any key to exit...
pause >nul
