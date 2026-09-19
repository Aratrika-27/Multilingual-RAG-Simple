"""
Test script for the RAG pipeline.
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader.data_loader import load_sample_data
from src.rag_pipeline import initialize_rag

def test_rag_pipeline():
    """Test the RAG pipeline with sample data."""
    print("Testing RAG pipeline with sample data...")
    
    # Load sample data
    documents = load_sample_data()
    
    # Initialize RAG pipeline
    rag = initialize_rag(documents=documents)
    
    # Test queries
    test_queries = [
        "What's the weather like tomorrow?",
        "कल का मौसम कैसा होगा?",  # Hindi: What will the weather be like tomorrow?
        "আগামীকাল আবহাওয়া কেমন হবে?",  # Bengali: What will the weather be like tomorrow?
        "Kal ka weather kaisa hoga?",  # Hindi-English mix
        "Doctor appointment के लिए reminder set करो",  # Hindi-English mix
        "Kolkata में best restaurants কোনগুলি?"  # Hindi-Bengali-English mix
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag.query(query)
            print(f"Response: {result['response']}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nRAG pipeline test completed.")

if __name__ == "__main__":
    test_rag_pipeline()
