"""
RAG pipeline for the multilingual code-mix chatbot.
"""
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import os

from .config import RETRIEVER_TOP_K
from .loader.data_loader import load_sample_data
from .retriever.vector_store import build_vector_store, load_vector_store
from .generator.multilingual_generator import load_generator, build_prompt

class MultilingualRAG:
    """Multilingual RAG pipeline for the code-mix chatbot."""
    
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        vector_store_path: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        generator_model_name: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            documents: List of documents to index (optional)
            vector_store_path: Path to load/save the vector store (optional)
            embedding_model_name: Name of the embedding model to use (optional)
            generator_model_name: Name of the generator model to use (optional)
        """
        # Initialize the vector store
        if vector_store_path and os.path.exists(vector_store_path):
            self.vector_store = load_vector_store(
                vector_store_path,
                embedding_model_name=embedding_model_name
            )
        elif documents:
            self.vector_store = build_vector_store(
                documents,
                embedding_model_name=embedding_model_name,
                save_path=vector_store_path
            )
        else:
            # Use sample data if no documents or vector store path provided
            print("No documents or vector store path provided. Using sample data.")
            sample_docs = load_sample_data()
            self.vector_store = build_vector_store(
                sample_docs,
                embedding_model_name=embedding_model_name
            )
        
        # Initialize the generator
        self.generator = load_generator(model_name=generator_model_name)
        
        # Initialize the RAG chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.generator,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
        )
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing the query, response, and retrieved documents
        """
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(query, k=RETRIEVER_TOP_K)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Build prompt
        prompt = build_prompt(context, query)
        
        # Generate response
        response = self.generator.invoke(prompt)
        
        return {
            "query": query,
            "response": response,
            "context": context,
            "documents": docs
        }

def initialize_rag(
    documents: Optional[List[Document]] = None,
    vector_store_path: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
    generator_model_name: Optional[str] = None
) -> MultilingualRAG:
    """
    Initialize the RAG pipeline.
    
    Args:
        documents: List of documents to index (optional)
        vector_store_path: Path to load/save the vector store (optional)
        embedding_model_name: Name of the embedding model to use (optional)
        generator_model_name: Name of the generator model to use (optional)
        
    Returns:
        MultilingualRAG instance
    """
    return MultilingualRAG(
        documents=documents,
        vector_store_path=vector_store_path,
        embedding_model_name=embedding_model_name,
        generator_model_name=generator_model_name
    )
