"""
Main script to run the multilingual code-mix chatbot.
"""
import argparse
import os
import sys
from pathlib import Path

# Import the simple chatbot first to ensure it's always available
try:
    from src.simple_chatbot import run_simple_chatbot
except ImportError as e:
    # Define a fallback function if even the simple chatbot can't be imported
    def run_simple_chatbot():
        print("Simple chatbot module could not be imported.")
        print("Please make sure the file src/simple_chatbot.py exists.")
        return

# Try to import the required modules for the full mode
try:
    from src.loader.data_loader import load_dataset, load_sample_data
    from src.rag_pipeline import initialize_rag
    from src.config import DATA_DIR, MODELS_DIR
    FULL_MODE = True
except ImportError as e:
    print(f"Warning: Could not import required modules: {str(e)}")
    print("Falling back to simple chatbot mode.")
    FULL_MODE = False

def main():
    """Run the multilingual code-mix chatbot."""
    # If we couldn't import the required modules, run the simple chatbot
    if not FULL_MODE:
        print("Running in simple mode due to missing dependencies.")
        run_simple_chatbot()
        return

    parser = argparse.ArgumentParser(description="Multilingual Code-Mix Chatbot")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data directory")
    parser.add_argument("--vector_store", type=str, default=None, help="Path to the vector store")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                        help="Name of the embedding model")
    parser.add_argument("--generator_model", type=str, default="google/mt5-small",
                        help="Name of the generator model")
    parser.add_argument("--use_sample_data", action="store_true", help="Use sample data instead of loading from files")
    parser.add_argument("--simple", action="store_true", help="Use simple chatbot mode (no external dependencies)")

    args = parser.parse_args()

    # If simple mode is requested, run the simple chatbot
    if args.simple:
        print("Running in simple mode as requested.")
        run_simple_chatbot()
        return

    # Determine vector store path
    vector_store_path = args.vector_store
    if vector_store_path is None:
        vector_store_path = os.path.join(MODELS_DIR, "vector_store")

    # Load documents
    if args.use_sample_data:
        print("Using sample data...")
        documents = load_sample_data()
    elif args.data_path:
        print(f"Loading data from {args.data_path}...")
        documents, _ = load_dataset(
            args.data_path,
            locales=["en", "hi", "bn"],
            splits=["train", "eval", "test"]
        )
    else:
        documents = None

    # Make sure we have valid model names
    if args.embedding_model is None or args.embedding_model.lower() == "none":
        args.embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        print(f"Using default embedding model: {args.embedding_model}")

    if args.generator_model is None or args.generator_model.lower() == "none":
        args.generator_model = "google/mt5-small"
        print(f"Using default generator model: {args.generator_model}")

    try:
        # Initialize RAG pipeline
        rag = initialize_rag(
            documents=documents,
            vector_store_path=vector_store_path,
            embedding_model_name=args.embedding_model,
            generator_model_name=args.generator_model
        )
    except Exception as e:
        print(f"Error initializing RAG pipeline: {str(e)}")
        print("\nPlease make sure you have the required packages installed:")
        print("pip install transformers torch langchain langchain-community sentence-transformers faiss-cpu")
        print("\nFalling back to simple chatbot mode...")
        run_simple_chatbot()
        return

    # Interactive chat loop
    print("\n===== Multilingual Code-Mix Chatbot =====")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'help' for assistance.")
    print("=========================================\n")

    while True:
        # Get user input
        user_input = input("\nYou: ")

        # Check for exit command
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Check for help command
        if user_input.lower() == "help":
            print("\nHelp:")
            print("- Type your question in English, Hindi, Bengali, or a mix of these languages.")
            print("- The chatbot will retrieve relevant information and respond.")
            print("- Type 'exit' or 'quit' to end the conversation.")
            continue

        # Process the query
        try:
            result = rag.query(user_input)
            print(f"\nChatbot: {result['response']}")
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    main()
