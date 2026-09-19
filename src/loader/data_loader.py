"""
Data loader for the multilingual code-mix chatbot.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from pathlib import Path

def load_json_data(file_path: str) -> List[Dict]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_dataset(
    base_path: str, 
    locales: List[str], 
    splits: List[str],
    file_pattern: str = "{locale}_{split}.json"
) -> Tuple[List[Document], Dict[str, pd.DataFrame]]:
    """
    Load multilingual dataset from JSON files.
    
    Args:
        base_path: Base directory containing the data files
        locales: List of language locales to load (e.g., ["en", "hi", "bn"])
        splits: List of data splits to load (e.g., ["train", "eval", "test"])
        file_pattern: Pattern for file names, with {locale} and {split} placeholders
        
    Returns:
        Tuple containing:
        - List of Documents for the retriever
        - Dictionary mapping locales to DataFrames
    """
    all_docs = []
    locale_datasets = {}

    for locale in locales:
        dfs = []
        for split in splits:
            file_name = file_pattern.format(locale=locale, split=split)
            file_path = os.path.join(base_path, file_name)
            
            if os.path.exists(file_path):
                try:
                    data = load_json_data(file_path)
                    df = pd.DataFrame(data)
                    df['locale'] = locale
                    df['split'] = split
                    dfs.append(df)
                    
                    # Create documents for the retriever
                    for _, row in df.iterrows():
                        # Customize this based on your data structure
                        if 'text' in row:
                            content = row['text']
                        elif 'utterance' in row:
                            content = row['utterance']
                        else:
                            # Create content from available fields
                            content = " ".join([f"{k}: {v}" for k, v in row.items() 
                                              if k not in ['locale', 'split'] and isinstance(v, (str, int, float))])
                        
                        metadata = {
                            'locale': locale,
                            'split': split,
                            'source': file_path,
                            'id': row.get('id', f"{locale}_{split}_{len(all_docs)}")
                        }
                        
                        all_docs.append(Document(page_content=content, metadata=metadata))
                        
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        if dfs:
            locale_datasets[locale] = pd.concat(dfs, ignore_index=True)
    
    return all_docs, locale_datasets

def load_sample_data() -> List[Document]:
    """
    Load sample multilingual data for testing.
    
    Returns:
        List of Documents for the retriever
    """
    # Sample data in English, Hindi, and Bengali
    samples = [
        # English samples
        "The weather forecast predicts rain tomorrow in Delhi.",
        "Please set a reminder for my doctor's appointment on Friday.",
        "What are the best restaurants in Kolkata for dinner?",
        
        # Hindi samples
        "कल दिल्ली में बारिश होने की संभावना है।",
        "शुक्रवार को मेरे डॉक्टर अपॉइंटमेंट के लिए एक रिमाइंडर सेट करें।",
        "कोलकाता में डिनर के लिए सबसे अच्छे रेस्तरां कौन से हैं?",
        
        # Bengali samples
        "আগামীকাল দিল্লিতে বৃষ্টি হওয়ার সম্ভাবনা আছে।",
        "শুক্রবার আমার ডাক্তারের অ্যাপয়েন্টমেন্টের জন্য একটি রিমাইন্ডার সেট করুন।",
        "কলকাতায় ডিনারের জন্য সেরা রেস্তোরাঁ কোনগুলি?",
        
        # Code-mixed samples
        "Kal Delhi mein rain hone ki forecast hai.",
        "Friday ko mere doctor appointment ke liye ek reminder set karo.",
        "Kolkata mein dinner ke liye best restaurants কোনগুলি?",
    ]
    
    # Create documents
    docs = []
    for i, text in enumerate(samples):
        # Determine language (simplified)
        if any('\u0900' <= c <= '\u097F' for c in text):  # Devanagari (Hindi)
            locale = "hi"
        elif any('\u0980' <= c <= '\u09FF' for c in text):  # Bengali
            locale = "bn"
        else:
            locale = "en"
            
        metadata = {
            'locale': locale,
            'id': f"sample_{i}",
            'source': "sample_data"
        }
        
        docs.append(Document(page_content=text, metadata=metadata))
    
    return docs
