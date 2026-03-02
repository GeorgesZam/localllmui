#!/bin/bash
# Launcher script for Local LLM CLI

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Run the CLI
cd "$SCRIPT_DIR/src"
python cli.py
