"""Retrieval Orchestrator: Complete two-stage funnel architecture

This orchestrates Phase 1 (Indexing & Retrieval) with hybrid ranking:
- Stage 1 (BM25): Fast candidate generation → Top 50 sentences (high recall)
- Stage 2 (Hybrid): Precise reranking → Top 10-12 sentences (high precision)
  - Combines: 0.6×Semantic + 0.2×Lexical + 0.2×Metadata

Usage:
    from src.retrieval.retrieval_orchestrator import RetrievalOrchestrator
    
    orchestrator = RetrievalOrchestrator()
    results = orchestrator.retrieve_and_rank(
        claim="Vietnam is the second largest coffee exporter",
        top_k=12
    )
    
    # Display results
    for result in results:
        print(f"Score: {result['combined_score']:.4f}")
        print(f"Text: {result['text']}")
"""

import os
from typing import List, Dict, Optional
from .bm25_retriever import BM25Retriever
from .embed_retriever import EmbeddingRetriever
from ..utils.metadata import MetadataHandler

try:
    from ..config.paths import DATA_INDEX_DIR
except ImportError:
    from config.paths import DATA_INDEX_DIR


class RetrievalOrchestrator:
    """
    Orchestrates two-stage funnel architecture for sentence retrieval
    
    Stage 1: BM25 retrieval (Top 50 - High Recall)
    Stage 2: Hybrid ranking (Top 10-12 - High Precision)
    """
    
    def __init__(
        self, 
        index_dir: str = DATA_INDEX_DIR,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.3,
        metadata_weight: float = 0.2,
        stage1_k: int = 50,
        trusted_domains: Optional[List[str]] = None
    ):
        """
        Initialize retrieval orchestrator
        
        Args:
            index_dir: Directory containing indexes
            semantic_weight: Weight for semantic similarity score
            lexical_weight: Weight for BM25 lexical score
            metadata_weight: Weight for metadata score
            stage1_k: Number of candidates from Stage 1 (default: 50)
            trusted_domains: List of trusted domains for authority scoring
        """
        self.index_dir = index_dir
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.metadata_weight = metadata_weight
        self.stage1_k = stage1_k
        
        # Initialize retrievers
        print("Initializing Retrieval Orchestrator...")
        print(f"  Index directory: {index_dir}")
        
        self.bm25_retriever = BM25Retriever(index_dir=index_dir)
        print("  ✓ BM25 retriever loaded")
        
        self.embed_retriever = EmbeddingRetriever(index_dir=index_dir)
        print("  ✓ Embedding retriever loaded")
        
        self.metadata_handler = MetadataHandler(trusted_domains=trusted_domains)
        print("  ✓ Metadata handler initialized")
        
        print(f"  Weights: Semantic={semantic_weight}, Lexical={lexical_weight}, Metadata={metadata_weight}")
        print()
    
    def retrieve_and_rank(
        self, 
        claim: str, 
        top_k: int = 12,
        claim_date: Optional[str] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Complete two-stage retrieval with hybrid ranking
        
        Args:
            claim: User's claim to fact-check
            top_k: Number of final sentences to return (default: 12)
            claim_date: Date associated with claim (for recency scoring)
            verbose: Print progress information
            
        Returns:
            List of top-k sentences with combined scores:
            [
                {
                    'sentence_id': int,
                    'text': str,
                    'combined_score': float,
                    'scores': {
                        'semantic': float,
                        'lexical': float,
                        'metadata': float,
                        'bm25_raw': float
                    },
                    'doc_title': str,
                    'doc_url': str,
                    'doc_date': str,
                    'doc_domain': str,
                    'doc_author': str
                },
                ...
            ]
        """
        if verbose:
            print(f"{'='*70}")
            print(f"RETRIEVAL ORCHESTRATOR - Two-Stage Funnel Architecture")
            print(f"{'='*70}")
            print(f"Claim: {claim}\n")
        
        # ========== STAGE 1: BM25 RETRIEVAL (High Recall) ==========
        if verbose:
            print(f"[Stage 1] BM25 Retrieval (High Recall)")
            print(f"          Target: Top {self.stage1_k} candidates")
        
        bm25_results = self.bm25_retriever.retrieve(claim, top_k=self.stage1_k)
        
        if verbose:
            print(f"          Retrieved: {len(bm25_results)} sentences")
            if bm25_results:
                print(f"          Score range: {bm25_results[0]['score']:.4f} → {bm25_results[-1]['score']:.4f}")
        
        if not bm25_results:
            if verbose:
                print("          ✗ No results found")
            return []
        
        # ========== STAGE 2: HYBRID RANKING (High Precision) ==========
        if verbose:
            print(f"\n[Stage 2] Hybrid Ranking (High Precision)")
            print(f"          Formula: {self.semantic_weight}×Semantic + {self.lexical_weight}×Lexical + {self.metadata_weight}×Metadata")
        
        # Extract sentence texts for semantic scoring
        sentence_texts = [result['text'] for result in bm25_results]
        
        # Compute semantic similarities
        if verbose:
            print(f"          Computing semantic scores...")
        semantic_scores = self.embed_retriever.compute_similarity(claim, sentence_texts)
        
        # Normalize BM25 scores to 0-1 range
        bm25_scores = [result['score'] for result in bm25_results]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        min_bm25 = min(bm25_scores) if bm25_scores else 0.0
        bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
        
        normalized_bm25 = [(score - min_bm25) / bm25_range for score in bm25_scores]
        
        # Compute metadata scores
        if verbose:
            print(f"          Computing metadata scores...")
        
        hybrid_results = []
        for idx, result in enumerate(bm25_results):
            # Create document dict for metadata scoring
            doc = {
                'url': result['doc_url'],
                'text': result['text'],
                'date': result['doc_date'],
                'domain': result['doc_domain']
            }
            
            # Calculate metadata score
            metadata_scores = self.metadata_handler.calculate_metadata_score(
                document=doc,
                claim=claim,
                claim_date=claim_date
            )
            
            # Combine scores
            combined_score = (
                self.semantic_weight * float(semantic_scores[idx]) +
                self.lexical_weight * normalized_bm25[idx] +
                self.metadata_weight * metadata_scores['combined']
            )
            
            hybrid_results.append({
                'sentence_id': result['sentence_id'],
                'text': result['text'],
                'combined_score': combined_score,
                'scores': {
                    'semantic': float(semantic_scores[idx]),
                    'lexical': normalized_bm25[idx],
                    'metadata': metadata_scores['combined'],
                    'bm25_raw': result['score']
                },
                'metadata_breakdown': {
                    'recency': metadata_scores['recency'],
                    'authority': metadata_scores['authority'],
                    'entity_overlap': metadata_scores['entity_overlap']
                },
                'doc_id': result['doc_id'],
                'doc_title': result['doc_title'],
                'doc_url': result['doc_url'],
                'doc_date': result['doc_date'],
                'doc_domain': result['doc_domain'],
                'doc_author': result['doc_author']
            })
        
        # Sort by combined score (descending)
        hybrid_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Select top-k
        final_results = hybrid_results[:top_k]
        
        if verbose:
            print(f"          Ranked and selected: Top {len(final_results)} sentences")
            if final_results:
                print(f"          Combined score range: {final_results[0]['combined_score']:.4f} → {final_results[-1]['combined_score']:.4f}")
        
        return final_results
    
    def display_results(self, results: List[Dict], show_scores: bool = True):
        """
        Display retrieval results in a readable format
        
        Args:
            results: List of result dictionaries from retrieve_and_rank
            show_scores: Whether to show detailed score breakdown
        """
        print(f"\n{'='*70}")
        print(f"RETRIEVAL RESULTS - Top {len(results)} Sentences")
        print(f"{'='*70}\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Combined Score: {result['combined_score']:.4f}")
            
            if show_scores:
                scores = result['scores']
                print(f"   ├─ Semantic:  {scores['semantic']:.4f} (weight: {self.semantic_weight})")
                print(f"   ├─ Lexical:   {scores['lexical']:.4f} (weight: {self.lexical_weight})")
                print(f"   └─ Metadata:  {scores['metadata']:.4f} (weight: {self.metadata_weight})")
                
                meta = result['metadata_breakdown']
                print(f"      ├─ Recency: {meta['recency']:.2f}")
                print(f"      ├─ Authority: {meta['authority']:.2f}")
                print(f"      └─ Entity Overlap: {meta['entity_overlap']:.2f}")
            
            print(f"\n   Text: {result['text']}")
            print(f"\n   Source: {result['doc_domain']}")
            print(f"   Title:  {result['doc_title'][:80]}...")
            print(f"   Date:   {result['doc_date']}")
            print(f"   URL:    {result['doc_url'][:60]}...")
            print()
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the retrieval system
        
        Returns:
            Dictionary with system statistics
        """
        bm25_stats = self.bm25_retriever.get_statistics()
        embed_stats = self.embed_retriever.get_statistics()
        
        return {
            'total_sentences': bm25_stats['total_sentences'],
            'stage1_candidates': self.stage1_k,
            'weights': {
                'semantic': self.semantic_weight,
                'lexical': self.lexical_weight,
                'metadata': self.metadata_weight
            },
            'bm25_index': bm25_stats['index_type'],
            'embedding_model': embed_stats['model'],
            'embedding_dim': embed_stats['embedding_dim'],
            'index_location': self.index_dir
        }


# Example usage
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = RetrievalOrchestrator()
    
    # Display statistics
    stats = orchestrator.get_statistics()
    print("System Statistics:")
    print(f"  Total sentences indexed: {stats['total_sentences']}")
    print(f"  Stage 1 candidates: {stats['stage1_candidates']}")
    print(f"  Embedding model: {stats['embedding_model']}")
    print()
    
    # Test retrieval
    claim = "Vietnam is the world's second largest coffee exporter"
    results = orchestrator.retrieve_and_rank(claim, top_k=12)
    
    # Display results
    orchestrator.display_results(results, show_scores=True)
