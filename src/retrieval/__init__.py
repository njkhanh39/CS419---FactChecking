"""Retrieval Module: Phase 1 - Fast candidate generation with BM25

This module handles the first stage of the funnel architecture:
- Build indexes from corpus (BM25 + FAISS)
- BM25 retrieval: ~500 sentences → Top 50 (high recall)
- Embedding retrieval: Semantic search support

Components:
-----------
- build_index.py: Build BM25 and FAISS indexes from corpus
- bm25_retriever.py: Fast lexical retrieval using BM25
- embed_retriever.py: Semantic retrieval using embeddings

Usage:
------
# Build indexes
from src.retrieval import IndexBuilder
builder = IndexBuilder()
builder.build_from_corpus_file('corpus_*.json')

# Retrieve candidates
from src.retrieval import BM25Retriever
retriever = BM25Retriever()
top_50 = retriever.retrieve(claim="...", top_k=50)
"""

from .build_index import IndexBuilder, build_indices_from_latest
from .bm25_retriever import BM25Retriever
from .embed_retriever import EmbeddingRetriever
from .retrieval_orchestrator import RetrievalOrchestrator

__all__ = [
    'IndexBuilder',
    'build_indices_from_latest',
    'BM25Retriever',
    'EmbeddingRetriever',
    'RetrievalOrchestrator'
]
