import pandas as pd
import pickle
import faiss
import numpy as np
import os
from config.paths import DATA_RAW_DIR, DATA_INDEX_DIR
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# Paths based on your architecture

RAW_CORPUS_PATH = os.path.join(DATA_RAW_DIR, 'raw/corpus.csv')  # Assume you have a CSV with columns ['id', 'text']

def build_indices():
    print("1. Loading Data...")
    # Load your raw data
    # For this example, let's assume a CSV file. 
    # In reality, you might loop through text files.
    if not os.path.exists(RAW_CORPUS_PATH):
        print(f"Error: {RAW_CORPUS_PATH} not found. Please add a dataset first.")
        return

    df = pd.read_csv(RAW_CORPUS_PATH)
    documents = df['text'].tolist()
    doc_ids = df['id'].tolist()
    
    # Save the document mapping (ID -> Text) so we can retrieve text later
    with open(os.path.join(DATA_INDEX_DIR, 'doc_store.pkl'), 'wb') as f:
        pickle.dump(df.to_dict('records'), f)

    # --- BUILD BM25 INDEX ---
    print("2. Building BM25 Index...")
    tokenized_corpus = [doc.split(" ") for doc in documents] # Simple whitespace tokenizer
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(os.path.join(DATA_INDEX_DIR, 'bm25_index.pkl'), 'wb') as f:
        pickle.dump(bm25, f)

    # --- BUILD EMBEDDING INDEX (FAISS) ---
    print("3. Building Embedding Index (FAISS)...")
    # Load a lightweight model (as suggested in your PDF)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode all documents (this takes time depending on data size)
    embeddings = model.encode(documents, show_progress_bar=True)
    
    # Normalize embeddings for Cosine Similarity (FAISS uses Dot Product by default)
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # IP = Inner Product (Cosine sim if normalized)
    index.add(embeddings)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(DATA_INDEX_DIR, 'faiss_index.bin'))
    
    print("Done! Indices saved to data/index/")

if __name__ == "__main__":
    os.makedirs(DATA_INDEX_DIR, exist_ok=True)
    build_indices()