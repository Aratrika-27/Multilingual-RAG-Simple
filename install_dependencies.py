"""
Script to install the required dependencies for the multilingual code-mix chatbot.
"""
import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Install the required dependencies."""
    print("Installing dependencies for the multilingual code-mix chatbot...")
    
    # Basic dependencies
    basic_deps = [
        "transformers",
        "torch",
        "langchain",
        "langchain-community",
        "langchain-core",
        "langchain-huggingface",
        "sentence-transformers",
        "faiss-cpu",
        "tqdm"
    ]
    
    # Optional dependencies
    optional_deps = [
        "datasets",
        "sentencepiece",
        "tiktoken",
        "flwr",  # For federated learning
        "pandas",
        "numpy",
        "scikit-learn",
        "python-dotenv",
        "jupyter",
        "matplotlib"
    ]
    
    # Install basic dependencies
    print("\nInstalling basic dependencies...")
    for package in basic_deps:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully.")
        else:
            print(f"❌ Failed to install {package}.")
    
    # Ask if the user wants to install optional dependencies
    install_optional = input("\nDo you want to install optional dependencies? (y/n): ").lower() == 'y'
    
    if install_optional:
        print("\nInstalling optional dependencies...")
        for package in optional_deps:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully.")
            else:
                print(f"❌ Failed to install {package}.")
    
    print("\nDependency installation complete!")
    print("You can now run the chatbot with: python run_chatbot.py --use_sample_data")

if __name__ == "__main__":
    main()
