import faiss
import pickle
import os
import numpy as np
from config.paths import DATA_INDEX_DIR
from sentence_transformers import SentenceTransformer

class EmbeddingRetriever:
    def __init__(self, index_dir= DATA_INDEX_DIR):
        self.index_dir = index_dir
        # Load model again (same one used in build_index)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = self._load_faiss()
        self.doc_store = self._load_doc_store()

    def _load_faiss(self):
        path = os.path.join(self.index_dir, 'faiss_index.bin')
        return faiss.read_index(path)

    def _load_doc_store(self):
        path = os.path.join(self.index_dir, 'doc_store.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)

    def retrieve(self, query, top_k=10):
        """
        Returns a list of dicts: {'text': ..., 'id': ..., 'score': ...}
        """
        # Encode query
        query_vector = self.model.encode([query])
        faiss.normalize_L2(query_vector)
        
        # Search FAISS
        # distances = cosine similarity scores (because we normalized)
        # indices = IDs of the documents
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            score = distances[0][i]
            
            if idx == -1: continue # Safety check for empty index
            
            doc_data = self.doc_store[idx]
            results.append({
                'id': doc_data['id'],
                'text': doc_data['text'],
                'score': float(score),
                'method': 'embedding'
            })
            
        return results