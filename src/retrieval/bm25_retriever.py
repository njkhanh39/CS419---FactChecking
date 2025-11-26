import pickle
import os
import numpy as np
from config.paths import DATA_INDEX_DIR

class BM25Retriever:
    def __init__(self, index_dir= DATA_INDEX_DIR ):
        self.index_dir = index_dir
        self.bm25 = self._load_bm25()
        self.doc_store = self._load_doc_store()

    def _load_bm25(self):
        path = os.path.join(self.index_dir, 'bm25_index.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_doc_store(self):
        path = os.path.join(self.index_dir, 'doc_store.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)

    def retrieve(self, query, top_k=10):
        """
        Returns a list of dicts: {'text': ..., 'id': ..., 'score': ...}
        """
        tokenized_query = query.split(" ")
        
        # Get scores for all docs
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top N indices
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            doc_data = self.doc_store[idx]
            results.append({
                'id': doc_data['id'],
                'text': doc_data['text'],
                'score': float(scores[idx]), # Convert numpy float to python float
                'method': 'bm25'
            })
            
        return results