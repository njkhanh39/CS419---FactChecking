"""Embedding Retriever: Semantic retrieval using sentence embeddings

This provides semantic search capabilities using FAISS:
- Used in Phase 2 (Funnel Stage 2) for semantic scoring
- Can also be used standalone for semantic retrieval
- Returns sentences with cosine similarity scores

Usage:
    from src.retrieval.embed_retriever import EmbeddingRetriever
    retriever = EmbeddingRetriever()
    results = retriever.retrieve(claim="...", top_k=10)
"""

import faiss
import pickle
import os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

try:
    from ..config.paths import DATA_INDEX_DIR
except ImportError:
    from config.paths import DATA_INDEX_DIR


class EmbeddingRetriever:
    """
    Semantic retrieval using sentence embeddings and FAISS
    """
    
    def __init__(self, index_dir: str = DATA_INDEX_DIR, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding retriever
        
        Args:
            index_dir: Directory containing FAISS index and sentence store
            model_name: Sentence transformer model name (must match build_index)
        """
        self.index_dir = index_dir
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = self._load_faiss()
        self.sentence_store = self._load_sentence_store()
    
    def _load_faiss(self):
        """Load FAISS index"""
        path = os.path.join(self.index_dir, 'faiss_index.bin')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FAISS index not found at {path}. "
                "Please run build_index.py first."
            )
        return faiss.read_index(path)
    
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
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve top-k sentences using semantic similarity
        
        Args:
            query: Search query (user's claim)
            top_k: Number of results to return
            
        Returns:
            List of sentence dictionaries with cosine similarity scores:
            [
                {
                    'sentence_id': int,
                    'text': str,
                    'score': float,  # Cosine similarity (0-1)
                    'doc_title': str,
                    'doc_url': str,
                    'doc_date': str,
                    'doc_domain': str,
                    'method': 'embedding'
                },
                ...
            ]
        """
        # Encode query to embedding
        query_vector = self.model.encode([query], convert_to_numpy=True)
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # Search FAISS index
        # distances = cosine similarity scores (because vectors are normalized)
        # indices = sentence IDs
        top_k_actual = min(top_k, len(self.sentence_store))
        distances, indices = self.index.search(query_vector, top_k_actual)
        
        # Build results with metadata
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = distances[0][i]
            
            # Safety check for invalid index
            if idx == -1 or idx >= len(self.sentence_store):
                continue
            
            sentence = self.sentence_store[idx]
            results.append({
                'sentence_id': sentence['sentence_id'],
                'text': sentence['text'],
                'score': float(score),  # Cosine similarity
                'doc_id': sentence['doc_id'],
                'doc_title': sentence['doc_title'],
                'doc_url': sentence['doc_url'],
                'doc_date': sentence['doc_date'],
                'doc_domain': sentence['doc_domain'],
                'doc_author': sentence['doc_author'],
                'method': 'embedding'
            })
        
        return results
    
    def compute_similarity(self, query: str, sentences: List[str]) -> np.ndarray:
        """
        Compute semantic similarity between query and list of sentences
        
        Args:
            query: Query text
            sentences: List of sentence texts
            
        Returns:
            Array of cosine similarity scores
        """
        # Encode query and sentences
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        sentence_embeddings = self.model.encode(sentences, convert_to_numpy=True)
        
        # Convert to float32 and normalize
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        sentence_embeddings = np.ascontiguousarray(sentence_embeddings, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        faiss.normalize_L2(sentence_embeddings)
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(sentence_embeddings, query_embedding.T).flatten()
        
        return similarities
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the index
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'total_sentences': len(self.sentence_store),
            'index_type': 'FAISS (IndexFlatIP)',
            'model': self.model_name,
            'embedding_dim': self.index.d,
            'index_location': self.index_dir
        }


# Example usage
if __name__ == "__main__":
    retriever = EmbeddingRetriever()
    
    # Test retrieval
    query = "Vietnam coffee export statistics"
    results = retriever.retrieve(query, top_k=10)
    
    print(f"Query: {query}")
    print(f"Model: {retriever.model_name}")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Similarity: {result['score']:.4f}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Source: {result['doc_domain']}")
        print()