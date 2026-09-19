"""
Setup script for the multilingual code-mix chatbot.
"""
import os
import sys
import subprocess
import platform

def main():
    """Set up the environment for the multilingual code-mix chatbot."""
    print("Setting up the environment for the multilingual code-mix chatbot...")
    
    # Print Python version
    print(f"Using Python: {sys.version}")
    
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Install dependencies
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("Error: Failed to install dependencies.")
        print("Please try installing them manually with:")
        print(f"{sys.executable} -m pip install -r requirements.txt")
    
    print("\nSetup complete! You can now run the chatbot with:")
    print(f"{sys.executable} run_chatbot.py --use_sample_data")

if __name__ == "__main__":
    main()
