"""
Vector store implementation for the multilingual code-mix chatbot.
"""
try:
    # Try importing from langchain_community (newer versions)
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
except ImportError:
    # Fall back to langchain (older versions)
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document

from typing import List, Optional
try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

import os
import pickle

from ..config import EMBEDDING_MODEL, MODELS_DIR

def build_vector_store(
    documents: List[Document],
    embedding_model_name: Optional[str] = EMBEDDING_MODEL,
    save_path: Optional[str] = None
) -> FAISS:
    """
    Build a FAISS vector store from documents.

    Args:
        documents: List of documents to index
        embedding_model_name: Name of the embedding model to use
        save_path: Path to save the vector store (optional)

    Returns:
        FAISS vector store
    """
    # Use the default embedding model from config if None is provided
    if embedding_model_name is None:
        embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print(f"Building vector store from {len(documents)} documents...")
    vectorstore = FAISS.from_documents(tqdm(documents), embedding_model)

    if save_path:
        print(f"Saving vector store to {save_path}")
        vectorstore.save_local(save_path)

    return vectorstore

def load_vector_store(
    load_path: str,
    embedding_model_name: Optional[str] = EMBEDDING_MODEL
) -> FAISS:
    """
    Load a FAISS vector store from disk.

    Args:
        load_path: Path to load the vector store from
        embedding_model_name: Name of the embedding model to use

    Returns:
        FAISS vector store
    """
    # Use the default embedding model from config if None is provided
    if embedding_model_name is None:
        embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print(f"Loading vector store from {load_path}")
    vectorstore = FAISS.load_local(load_path, embedding_model)

    return vectorstore

def save_documents(documents: List[Document], save_path: str):
    """Save documents to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(documents, f)
    print(f"Saved {len(documents)} documents to {save_path}")

def load_documents(load_path: str) -> List[Document]:
    """Load documents from disk."""
    with open(load_path, 'rb') as f:
        documents = pickle.load(f)
    print(f"Loaded {len(documents)} documents from {load_path}")
    return documents
