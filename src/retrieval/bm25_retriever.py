"""BM25 Retriever: Fast lexical retrieval using BM25 algorithm

This implements Phase 1 (Funnel Stage 1) of the fact-checking pipeline:
- Fast candidate generation with high recall
- Uses BM25 for keyword-based ranking
- Retrieves top-50 sentences from ~500 total

TODO: Apply a fast filter before BM25 (reduce too short sentences, overlap sentences, etc.)

Usage:
    from src.retrieval.bm25_retriever import BM25Retriever
    retriever = BM25Retriever()
    results = retriever.retrieve(claim="...", top_k=50)
"""

import pickle
import os
import re
import numpy as np
from typing import List, Dict

try:
    from ..config.paths import DATA_INDEX_DIR
except ImportError:
    from config.paths import DATA_INDEX_DIR


class BM25Retriever:
    """
    BM25-based lexical retrieval for sentence-level search
    """
    
    def __init__(self, index_dir: str = DATA_INDEX_DIR):
        """
        Initialize BM25 retriever
        
        Args:
            index_dir: Directory containing BM25 index and sentence store
        """
        self.index_dir = index_dir
        self.bm25 = self._load_bm25()
        self.sentence_store = self._load_sentence_store()
    
    def _load_bm25(self):
        """Load BM25 index"""
        path = os.path.join(self.index_dir, 'bm25_index.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BM25 index not found at {path}. "
                "Please run build_index.py first."
            )
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _load_sentence_store(self):
        """Load sentence store (metadata)"""
        path = os.path.join(self.index_dir, 'sentence_store.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Sentence store not found at {path}. "
                "Please run build_index.py first."
            )
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text (must match tokenization in build_index.py)
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def retrieve(self, query: str, top_k: int = 50) -> List[Dict]:
        """
        Retrieve top-k sentences using BM25
        
        Args:
            query: Search query (user's claim)
            top_k: Number of results to return (default: 50 for Stage 1)
            
        Returns:
            List of sentence dictionaries with scores:
            [
                {
                    'sentence_id': int,
                    'text': str,
                    'score': float,
                    'doc_title': str,
                    'doc_url': str,
                    'doc_date': str,
                    'doc_domain': str,
                    'method': 'bm25'
                },
                ...
            ]
        """
        # Tokenize query
        tokenized_query = self.tokenize(query)
        
        # Get BM25 scores for all sentences
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices (descending order)
        top_k_actual = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k_actual]
        
        # Build results with metadata
        results = []
        for idx in top_indices:
            sentence = self.sentence_store[idx]
            results.append({
                'sentence_id': sentence['sentence_id'],
                'text': sentence['text'],
                'score': float(scores[idx]),
                'doc_id': sentence['doc_id'],
                'doc_title': sentence['doc_title'],
                'doc_url': sentence['doc_url'],
                'doc_date': sentence['doc_date'],
                'doc_domain': sentence['doc_domain'],
                'doc_author': sentence['doc_author'],
                'method': 'bm25'
            })
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the index
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'total_sentences': len(self.sentence_store),
            'index_type': 'BM25',
            'index_location': self.index_dir
        }


# Example usage
if __name__ == "__main__":
    retriever = BM25Retriever()
    
    # Test retrieval
    query = "Vietnam coffee export statistics"
    results = retriever.retrieve(query, top_k=10)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Source: {result['doc_domain']}")
        print()