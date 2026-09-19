#!/bin/bash

# Find Python 3.11 executable
if command -v python3.11 &> /dev/null; then
    PYTHON=python3.11
elif command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: Python not found. Please install Python 3.11."
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Create virtual environment
$PYTHON -m venv venv

# Check if venv was created successfully
if [ ! -d "venv" ]; then
    echo "Error: Failed to create virtual environment."
    echo "Try installing venv package with: $PYTHON -m pip install virtualenv"
    exit 1
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    # For Windows
    source venv/Scripts/activate
else
    echo "Error: Virtual environment activation script not found."
    exit 1
fi

# Install dependencies
pip install -r requirements.txt || python -m pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models

echo "Setup complete! You can now run the chatbot with:"
echo "python run_chatbot.py --use_sample_data"
