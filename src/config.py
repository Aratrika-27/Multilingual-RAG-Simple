"""
Configuration settings for the multilingual code-mix chatbot.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Model settings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # Good for Bengali, Hindi, English
GENERATOR_MODEL = "google/mt5-small"  # Start with small model, can upgrade to base/large later

# Language settings
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali"
}

# RAG settings
RETRIEVER_TOP_K = 5
MAX_CONTEXT_LENGTH = 1024
MAX_RESPONSE_LENGTH = 512

# Federated learning settings
FL_SERVER_ADDRESS = "localhost:8080"
FL_NUM_ROUNDS = 3
FL_MIN_CLIENTS = 2

# Prompt templates
PROMPT_TEMPLATE = """You are a multilingual assistant that understands and responds in English, Hindi, and Bengali.
You can mix these languages in your responses when appropriate.

Use the context below to answer the question.
If the answer is not found in the context, say 'I don't have enough information to answer that question.'

Context:
{context}

Question: {query}
Answer:"""
