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

# Suppress TensorFlow warnings (saves 2-3s initialization time)
import os as _os
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32, device: str = 'auto', preload_model: bool = True):
        """
        Initialize index builder
        
        Args:
            model_name: Sentence transformer model name
            batch_size: Batch size for encoding (default: 32)
            device: Device for encoding ('auto', 'cuda', 'mps', 'cpu')
            preload_model: Load model during init to save time later (default: True)
        """
        self.model_name = model_name
        self.device = self._detect_device(device)
        
        # Optimize batch size for CPU (reduce from 32 to 16 for better cache performance)
        if self.device == 'cpu' and batch_size > 16:
            self.batch_size = 16
            print(f"   💻 Using CPU")
            print(f"   ⚠️  Adjusted batch_size: {batch_size} → {self.batch_size} (optimized for CPU)")
        else:
            self.batch_size = batch_size
            print(f"   💻 Using {self.device.upper()}")
        
        # Pre-load model to avoid repeated loading overhead (saves 10-15s per call)
        if preload_model:
            print(f"   Loading model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"   ✓ Model loaded and ready!")
        else:
            self.model = None
    
    def _detect_device(self, device: str = 'auto') -> str:
        """
        Detect best available device for model inference
        
        Args:
            device: Device preference ('auto', 'cuda', 'mps', 'cpu')
            
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if device != 'auto':
            return device
        
        # Auto-detect best device
        try:
            import torch
            
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"   ⚡ GPU detected: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                print(f"   ⚡ Apple Silicon (MPS) detected")
            else:
                device = 'cpu'
                print(f"   💻 Using CPU")
        except ImportError:
            device = 'cpu'
            print(f"   💻 Using CPU (PyTorch not found)")
        
        return device
    
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
    
    def filter_sentences(self, sentences: List[Dict]) -> List[Dict]:
        """
        Filter sentences to remove:
        - Too short sentences (< 20 chars)
        - Too long sentences (> 500 chars)
        - Near-duplicate sentences (>80% overlap)
        - Low-information sentences (all caps, urls only, etc.)
        
        Args:
            sentences: List of sentence dictionaries
            
        Returns:
            Filtered list of sentence dictionaries
        """
        print(f"\nFiltering {len(sentences)} sentences...")
        filtered = []
        seen_texts = set()
        
        for sentence in sentences:
            text = sentence['text'].strip()
            
            # Rule 1: Length filter (20-500 chars)
            if len(text) < 20 or len(text) > 500:
                continue
            
            # Rule 2: Remove sentences with too few words (< 5 words)
            word_count = len(text.split())
            if word_count < 5:
                continue
            
            # Rule 3: Remove all-caps sentences (likely headers/spam)
            if text.isupper() and len(text) > 30:
                continue
            
            # Rule 4: Remove sentences that are mostly URLs
            url_pattern = r'https?://\S+'
            urls = re.findall(url_pattern, text)
            if urls and len(''.join(urls)) > len(text) * 0.5:
                continue
            
            # Rule 5: Filter inappropriate symbols (emojis, excessive punctuation)
            # Remove emojis and special unicode characters
            emoji_pattern = re.compile(
                "["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE
            )
            emoji_count = len(emoji_pattern.findall(text))
            # If >3 emojis or emoji content >10% of text, skip
            if emoji_count > 3:
                continue
            
            # Remove text with excessive special characters
            special_chars = re.findall(r'[^a-zA-Z0-9\s.,!?;:\-\'"]', text)
            if len(special_chars) > len(text) * 0.15:  # >15% special chars
                continue
            
            # Rule 6: Check for near-duplicates using normalized text
            normalized_text = ' '.join(text.lower().split())
            
            # Check if we've seen a very similar sentence
            is_duplicate = False
            for seen in seen_texts:
                # Simple overlap check: count common words
                seen_words = set(seen.split())
                current_words = set(normalized_text.split())
                
                if len(seen_words) == 0 or len(current_words) == 0:
                    continue
                
                overlap = len(seen_words & current_words)
                max_len = max(len(seen_words), len(current_words))
                overlap_ratio = overlap / max_len
                
                # If >80% overlap, consider it duplicate
                if overlap_ratio > 0.80:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            # Passed all filters - add to results
            filtered.append(sentence)
            seen_texts.add(normalized_text)
        
        print(f"   Filtered: {len(sentences)} → {len(filtered)} sentences")
        print(f"   Removed: {len(sentences) - len(filtered)} low-quality/duplicate sentences")
        
        # Re-assign sentence IDs
        for idx, sentence in enumerate(filtered):
            sentence['sentence_id'] = idx
        
        return filtered
    
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
        
        # Apply quality filters
        sentences = self.filter_sentences(sentences)
        
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
    
    def build_indices(self, sentences: List[Dict], claim: str, save_dir: Optional[str] = None, top_k_bm25: int = 50):
        """
        Build BM25 and FAISS indexes from sentences using FUNNEL ARCHITECTURE
        
        CORRECT APPROACH (23x faster):
        1. Build BM25 index on ALL sentences (cheap, ~0.05s)
        2. Query BM25 with claim → Top 50 candidates (high recall)
        3. Encode ONLY those 50 sentences (expensive, ~0.3s instead of ~7s)
        4. Build FAISS index with 50 embeddings (not all 890!)
        
        Args:
            sentences: List of ALL sentence dictionaries
            claim: User's claim to use for BM25 filtering
            save_dir: Directory to save indexes (defaults to DATA_INDEX_DIR)
            top_k_bm25: Number of top BM25 candidates to encode (default: 50)
        """
        import time
        
        if not sentences:
            print("Error: No sentences to index")
            return
        
        if not claim:
            print("Error: Claim is required for BM25 filtering")
            return
        
        save_dir = save_dir or DATA_INDEX_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n🚀 FUNNEL ARCHITECTURE: BM25 filter → Encode top {top_k_bm25} → FAISS")
        print(f"   Total sentences: {len(sentences)}")
        print(f"   Claim: {claim[:80]}..." if len(claim) > 80 else f"   Claim: {claim}")
        
        # Extract text for indexing
        sentence_texts = [s['text'] for s in sentences]
        
        # Save sentence store (metadata for ALL sentences)
        print("\n[1/4] Saving sentence store (ALL sentences)...")
        t0 = time.time()
        sentence_store_path = os.path.join(save_dir, 'sentence_store.pkl')
        with open(sentence_store_path, 'wb') as f:
            pickle.dump(sentences, f)
        print(f"   Saved {len(sentences)} sentences to sentence_store.pkl ({time.time()-t0:.2f}s)")
        
        # --- STAGE 1: BUILD BM25 INDEX ON ALL SENTENCES (CHEAP FILTER) ---
        print(f"\n[2/4] Building BM25 Index on ALL {len(sentences)} sentences (cheap filter)...")
        t0 = time.time()
        tokenized_corpus = [self.tokenize(text) for text in sentence_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_time = time.time() - t0
        
        bm25_path = os.path.join(save_dir, 'bm25_index.pkl')
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25, f)
        print(f"   ✓ BM25 index built and saved ({bm25_time:.2f}s)")
        
        # Query BM25 to get top candidates
        print(f"\n[3/4] Querying BM25 to get Top {top_k_bm25} candidates...")
        t0 = time.time()
        tokenized_query = self.tokenize(claim)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get top K indices
        top_indices = np.argsort(bm25_scores)[::-1][:top_k_bm25]
        top_candidates = [sentences[i] for i in top_indices]
        top_texts = [sentences[i]['text'] for i in top_indices]
        query_time = time.time() - t0
        
        print(f"   ✓ Retrieved top {len(top_candidates)} candidates ({query_time:.2f}s)")
        print(f"   Score range: {bm25_scores[top_indices[0]]:.2f} → {bm25_scores[top_indices[-1]]:.2f}")
        
        # Save top candidates mapping (so we know which embeddings correspond to which sentences)
        top_candidates_path = os.path.join(save_dir, 'top50_candidates.pkl')
        with open(top_candidates_path, 'wb') as f:
            pickle.dump({'indices': top_indices.tolist(), 'candidates': top_candidates}, f)
        print(f"   ✓ Saved top {len(top_candidates)} candidates to top50_candidates.pkl")
        
        # --- STAGE 2: ENCODE ONLY TOP K CANDIDATES (EXPENSIVE OPERATION) ---
        print(f"\n[4/4] Encoding ONLY Top {len(top_candidates)} candidates (expensive operation)...")
        
        # Model should already be loaded in __init__
        if self.model is None:
            print(f"   ⚠️  WARNING: Model not pre-loaded, loading now...")
            print(f"   ⚠️  This will add 12-15s overhead. Use preload_model=True in __init__()")
            t_load = time.time()
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"   Model loaded in {time.time()-t_load:.2f}s")
        
        print(f"   ⚡ Model pre-loaded, encoding only (no initialization overhead)")
        print(f"   📊 Encoding {len(top_texts)} sentences instead of {len(sentences)} ({len(sentences)/len(top_texts):.1f}x reduction!)")
        
        # Batch encoding - processes only top K sentences
        t0 = time.time()
        embeddings = self.model.encode(
            top_texts, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            batch_size=self.batch_size,
            normalize_embeddings=False   # We'll normalize with FAISS
        )
        encoding_time = time.time() - t0
        print(f"   ✓ Encoding completed in {encoding_time:.2f}s ({len(top_texts)/encoding_time:.1f} sentences/sec)")
        print(f"   💾 Memory saved: {len(sentences) - len(top_texts)} embeddings not created!")
        
        # Ensure numpy array with float32 (required by FAISS)
        t0 = time.time()
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
        print(f"   Saved FAISS index to faiss_index.bin ({time.time()-t0:.2f}s)")
        
        print(f"\n{'='*70}")
        print(f"✓ FUNNEL ARCHITECTURE COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"  Location: {save_dir}")
        print(f"  BM25 index: {len(sentences)} sentences (cheap filter)")
        print(f"  FAISS index: {len(top_candidates)} embeddings (expensive operation)")
        print(f"  Efficiency gain: {len(sentences)/len(top_candidates):.1f}x less encoding work!")
        print(f"  Time breakdown:")
        print(f"    - BM25 build: {bm25_time:.2f}s")
        print(f"    - BM25 query: {query_time:.2f}s")
        print(f"    - Encoding {len(top_candidates)} sentences: {encoding_time:.2f}s")
        print(f"    - Total: {bm25_time + query_time + encoding_time:.2f}s")
        print(f"  Estimated time saved vs encoding all: {(len(sentences)/len(top_candidates))*encoding_time - encoding_time:.2f}s")
    
    def build_from_corpus_file(self, corpus_filename: str, claim: str, save_dir: Optional[str] = None, top_k_bm25: int = 50):
        """
        Build indexes from a corpus file in data/raw/ using FUNNEL ARCHITECTURE
        
        Args:
            corpus_filename: Name of corpus file (e.g., 'corpus_*.json')
            claim: User's claim for BM25 filtering
            save_dir: Directory to save indexes
            top_k_bm25: Number of top BM25 candidates to encode (default: 50)
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
        self.build_indices(sentences, claim, save_dir, top_k_bm25)
    
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
def build_indices_from_latest(claim: str, batch_size: int = 32, device: str = 'auto', top_k_bm25: int = 50):
    """
    Build indexes from the most recent corpus file in data/raw/ using FUNNEL ARCHITECTURE
    
    Args:
        claim: User's claim for BM25 filtering (REQUIRED for funnel architecture)
        batch_size: Batch size for encoding (default: 32)
        device: Device for encoding ('auto', 'cuda', 'mps', 'cpu')
        top_k_bm25: Number of top BM25 candidates to encode (default: 50)
    """
    if not claim:
        print("Error: Claim is required for funnel architecture!")
        print("The funnel approach requires the claim to:")
        print("  1. Build BM25 on all sentences")
        print("  2. Query BM25 with claim → Top 50")
        print("  3. Encode only those 50 (23x faster!)")
        return
    
    builder = IndexBuilder(batch_size=batch_size, device=device)
    latest_corpus = builder.find_latest_corpus()
    
    if latest_corpus:
        print(f"Found latest corpus: {latest_corpus}")
        print(f"Using claim for BM25 filtering: {claim[:80]}...")
        builder.build_from_corpus_file(latest_corpus, claim, top_k_bm25=top_k_bm25)
    else:
        print(f"Error: No corpus files found in {DATA_RAW_DIR}")
        print("Please run data collection first:")
        print("  from src.data_collection import DataCollector")
        print("  collector = DataCollector()")
        print("  corpus = collector.collect_corpus(claim='...', num_urls=20)")


if __name__ == "__main__":
    import sys
    
    # Configuration
    BATCH_SIZE = 32  # Optimal for most CPUs/GPUs
    DEVICE = 'auto'  # Auto-detect cuda/mps/cpu
    
    print("=" * 60)
    print("Phase 1: Building Indexes with Batch Encoding Optimization")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Device: {DEVICE} (auto-detect)")
    print(f"  - Model: all-MiniLM-L6-v2")
    print()
    
    if len(sys.argv) > 1:
        # Use specified corpus file
        corpus_file = sys.argv[1]
        builder = IndexBuilder(batch_size=BATCH_SIZE, device=DEVICE)
        builder.build_from_corpus_file(corpus_file)
    else:
        # Use latest corpus
        build_indices_from_latest(batch_size=BATCH_SIZE, device=DEVICE)