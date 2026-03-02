@echo off
REM Local RAG Application Launcher for Windows
REM This script activates the virtual environment and runs the app

echo ======================================
echo   Local RAG Application Launcher
echo ======================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if models exist
if not exist "models\model.gguf" (
    echo [WARNING] Model file not found: models\model.gguf
    echo Please download the model from:
    echo https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
    echo.
    echo Save it as: models\qwen2.5-0.5b-instruct-q4_k_m.gguf
    echo Then create a symlink:
    echo   mklink /H models\model.gguf models\qwen2.5-0.5b-instruct-q4_k_m.gguf
    echo.
)

REM Check if embedding model exists
if not exist "models\embedding_model" (
    echo [WARNING] Embedding model not found
    echo Downloading embedding model...
    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-small-en-v1.5'); model.save('models\embedding_model'); print('Done!')"
)

REM Create data directory if not exists
if not exist "data\" (
    mkdir data
    echo [INFO] Created data directory
)

REM Run the application
echo.
echo [INFO] Starting Local RAG Application...
echo.
python src\main.py

REM If app crashes, pause to show error
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code: %errorlevel%
    echo.
    pause
)
