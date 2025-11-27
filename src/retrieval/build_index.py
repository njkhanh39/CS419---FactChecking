"""Build Index Module: Creates BM25 and FAISS indexes from corpus

This module handles Phase 1 of the fact-checking pipeline:
- Loads corpus (from data collection Phase 0)
- Splits documents into sentences (~500 sentences from 20 articles)
- Builds BM25 index for fast lexical retrieval
- Builds FAISS index for semantic retrieval
- Saves indexes and sentence store

Usage:
    from src.retrieval.build_index import IndexBuilder
    builder = IndexBuilder()
    builder.build_from_corpus_file('corpus_*.json')
"""

import json
import pickle
import faiss
import numpy as np
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    from ..config.paths import DATA_RAW_DIR, DATA_INDEX_DIR
except ImportError:
    from config.paths import DATA_RAW_DIR, DATA_INDEX_DIR


class IndexBuilder:
    """
    Builds BM25 and FAISS indexes from corpus for sentence-level retrieval
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize index builder
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.model = None
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitting (can be improved with spaCy/NLTK)
        
        Args:
            text: Document text
            
        Returns:
            List of sentences
        """
        # Simple regex-based sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out very short sentences (< 10 chars)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def load_corpus_from_json(self, corpus_path: str) -> List[Dict]:
        """
        Load corpus from JSON file (Phase 0 output format)
        
        Args:
            corpus_path: Path to corpus JSON file
            
        Returns:
            List of sentence dictionaries with metadata
        """
        print(f"Loading corpus from: {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        
        sentences = []
        corpus_documents = corpus_data.get('corpus', [])
        
        for doc_idx, doc in enumerate(corpus_documents):
            doc_text = doc.get('text', '')
            doc_sentences = self.split_into_sentences(doc_text)
            
            for sent_idx, sentence in enumerate(doc_sentences):
                sentences.append({
                    'sentence_id': len(sentences),
                    'doc_id': doc_idx,
                    'text': sentence,
                    'doc_title': doc.get('title', ''),
                    'doc_url': doc.get('url', ''),
                    'doc_date': doc.get('date', ''),
                    'doc_domain': doc.get('domain', ''),
                    'doc_author': doc.get('author', ''),
                })
        
        print(f"Extracted {len(sentences)} sentences from {len(corpus_documents)} documents")
        return sentences
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (can be improved with proper tokenizer)
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Lowercase and split on whitespace/punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def build_indices(self, sentences: List[Dict], save_dir: Optional[str] = None):
        """
        Build BM25 and FAISS indexes from sentences
        
        Args:
            sentences: List of sentence dictionaries
            save_dir: Directory to save indexes (defaults to DATA_INDEX_DIR)
        """
        if not sentences:
            print("Error: No sentences to index")
            return
        
        save_dir = save_dir or DATA_INDEX_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract text for indexing
        sentence_texts = [s['text'] for s in sentences]
        
        # Save sentence store (metadata for retrieval)
        print("1. Saving sentence store...")
        sentence_store_path = os.path.join(save_dir, 'sentence_store.pkl')
        with open(sentence_store_path, 'wb') as f:
            pickle.dump(sentences, f)
        print(f"   Saved {len(sentences)} sentences to sentence_store.pkl")
        
        # --- BUILD BM25 INDEX ---
        print("\n2. Building BM25 Index...")
        tokenized_corpus = [self.tokenize(text) for text in sentence_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        bm25_path = os.path.join(save_dir, 'bm25_index.pkl')
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25, f)
        print(f"   Saved BM25 index to bm25_index.pkl")
        
        # --- BUILD EMBEDDING INDEX (FAISS) ---
        print("\n3. Building Embedding Index (FAISS)...")
        if self.model is None:
            print(f"   Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        
        print(f"   Encoding {len(sentence_texts)} sentences...")
        embeddings = self.model.encode(
            sentence_texts, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            batch_size=32
        )
        
        # Ensure numpy array with float32 (required by FAISS)
        embeddings_array: np.ndarray = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        # Normalize embeddings for Cosine Similarity
        faiss.normalize_L2(embeddings_array)
        
        # Create FAISS index
        dimension: int = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine sim (normalized)
        index.add(embeddings_array)  # type: ignore[call-arg]
        
        # Save FAISS index
        faiss_path = os.path.join(save_dir, 'faiss_index.bin')
        faiss.write_index(index, faiss_path)
        print(f"   Saved FAISS index to faiss_index.bin")
        
        print(f"\n✓ All indexes built successfully!")
        print(f"  Location: {save_dir}")
        print(f"  Total sentences indexed: {len(sentences)}")
    
    def build_from_corpus_file(self, corpus_filename: str, save_dir: Optional[str] = None):
        """
        Build indexes from a corpus file in data/raw/
        
        Args:
            corpus_filename: Name of corpus file (e.g., 'corpus_*.json')
            save_dir: Directory to save indexes
        """
        corpus_path = os.path.join(DATA_RAW_DIR, corpus_filename)
        
        if not os.path.exists(corpus_path):
            print(f"Error: Corpus file not found: {corpus_path}")
            print(f"Available files in {DATA_RAW_DIR}:")
            if os.path.exists(DATA_RAW_DIR):
                for f in os.listdir(DATA_RAW_DIR):
                    if f.endswith('.json'):
                        print(f"  - {f}")
            return
        
        sentences = self.load_corpus_from_json(corpus_path)
        self.build_indices(sentences, save_dir)
    
    def find_latest_corpus(self) -> Optional[str]:
        """
        Find the most recent corpus file in data/raw/
        
        Returns:
            Filename of latest corpus, or None
        """
        if not os.path.exists(DATA_RAW_DIR):
            return None
        
        corpus_files = [f for f in os.listdir(DATA_RAW_DIR) if f.startswith('corpus_') and f.endswith('.json')]
        if not corpus_files:
            return None
        
        # Sort by modification time (most recent first)
        corpus_files.sort(key=lambda f: os.path.getmtime(os.path.join(DATA_RAW_DIR, f)), reverse=True)
        return corpus_files[0]


# Standalone function for backward compatibility
def build_indices_from_latest():
    """
    Build indexes from the most recent corpus file in data/raw/
    """
    builder = IndexBuilder()
    latest_corpus = builder.find_latest_corpus()
    
    if latest_corpus:
        print(f"Found latest corpus: {latest_corpus}")
        builder.build_from_corpus_file(latest_corpus)
    else:
        print(f"Error: No corpus files found in {DATA_RAW_DIR}")
        print("Please run data collection first:")
        print("  from src.data_collection import DataCollector")
        print("  collector = DataCollector()")
        print("  corpus = collector.collect_corpus(claim='...', num_urls=20)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use specified corpus file
        corpus_file = sys.argv[1]
        builder = IndexBuilder()
        builder.build_from_corpus_file(corpus_file)
    else:
        # Use latest corpus
        build_indices_from_latest()