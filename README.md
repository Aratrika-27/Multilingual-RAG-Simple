# Multilingual Code-Mix Chatbot

A multilingual code-mix chatbot that recognizes Bengali, Hindi, and English, with support for federated learning.

## Features

- **Multilingual Understanding**: Processes queries in English, Hindi, Bengali, or a mix of these languages
- **Retrieval-Augmented Generation (RAG)**: Uses a RAG pipeline to provide context-aware responses
- **Code-Mixing Support**: Handles and generates code-mixed responses
- **Federated Learning**: Supports distributed training across multiple clients

## Project Structure

```
BTP-NLP/
├── data/                  # Data directory
├── models/                # Model directory
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── federated/         # Federated learning modules
│   ├── generator/         # Text generation modules
│   ├── loader/            # Data loading modules
│   ├── retriever/         # Retrieval modules
│   ├── config.py          # Configuration settings
│   └── rag_pipeline.py    # RAG pipeline implementation
├── tests/                 # Test scripts
├── requirements.txt       # Dependencies
├── run_chatbot.py         # Script to run the chatbot
├── run_fl_server.py       # Script to run the federated learning server
└── run_fl_client.py       # Script to run a federated learning client
```

## Installation

### Option 1: Using the installation script

Run the installation script to install the required dependencies:

```bash
python install_dependencies.py
```

This script will install the basic dependencies required for the chatbot and ask if you want to install optional dependencies.

### Option 2: Manual installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Simple Mode

If you encounter issues with dependencies, you can run the chatbot in simple mode:

```bash
python run_chatbot.py --simple
```

This will use a basic implementation that doesn't rely on external libraries.

## Usage

### Running the Chatbot

```bash
python run_chatbot.py --use_sample_data
```

Options:
- `--data_path`: Path to the data directory
- `--vector_store`: Path to the vector store
- `--embedding_model`: Name of the embedding model (default: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
- `--generator_model`: Name of the generator model (default: "google/mt5-small")
- `--use_sample_data`: Use sample data instead of loading from files
- `--simple`: Use simple chatbot mode (no external dependencies)

### Troubleshooting

If you encounter errors when running the chatbot, try the following:

1. Make sure you have installed all the required dependencies:
   ```bash
   python install_dependencies.py
   ```

2. Try running the chatbot in simple mode:
   ```bash
   python run_chatbot.py --simple
   ```

3. If you're still having issues, check the error messages for specific package requirements and install them manually:
   ```bash
   pip install <package_name>
   ```

### Running Federated Learning

1. Start the server:
   ```bash
   python run_fl_server.py
   ```

2. Start clients (run in separate terminals):
   ```bash
   python run_fl_client.py --client_id client1 --train_data data/client1_train.csv
   python run_fl_client.py --client_id client2 --train_data data/client2_train.csv
   ```

## Models

The project uses the following models:
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Generator Model**: `google/mt5-small`

## Data Format

The data should be in CSV format with the following columns:
- `text`: Input text
- `target`: Target text for training
- Additional metadata columns as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.
