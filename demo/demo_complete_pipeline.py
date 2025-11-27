"""
Complete Pipeline Demo: Phase 0 + Phase 1 + Phase 2
====================================================

This script demonstrates the COMPLETE optimized pipeline:
- Phase 0: Data Collection (with concurrent scraping)
- Phase 1: Indexing (with batch encoding)
- Phase 2: Retrieval (BM25 + Hybrid ranking)

Expected total time: ~7 seconds per claim
- Phase 0: ~2-3 seconds (concurrent scraping)
- Phase 1: ~2 seconds (batch encoding)
- Phase 2: ~0.5 seconds (retrieval)

Prerequisites:
    - API key configured in src/config/api_keys.py
    - Or set SERPAPI_KEY environment variable

Usage:
    python demo_complete_pipeline.py
"""

import os
import time
from pathlib import Path

def main():
    print("=" * 80)
    print("COMPLETE OPTIMIZED PIPELINE DEMO")
    print("=" * 80)
    print()
    print("Testing: Phase 0 (Data Collection) + Phase 1 (Indexing) + Phase 2 (Retrieval)")
    print("With ALL optimizations enabled:")
    print("  ⚡ Concurrent scraping (10 workers)")
    print("  ⚡ Batch encoding (batch_size=32)")
    print("  ⚡ Strict timeouts (3s fail-fast)")
    print("  ⚡ Text-only headers")
    print()
    
    # Get API key
    api_key = None
    try:
        from src.config.api_keys import SERPAPI_KEY
        api_key = SERPAPI_KEY
    except ImportError:
        api_key = os.environ.get('SERPAPI_KEY')
    
    if not api_key:
        print("❌ ERROR: API key not found!")
        print()
        print("Please set up your API key:")
        print("  Option 1: Create src/config/api_keys.py with SERPAPI_KEY")
        print("  Option 2: Set environment variable: $env:SERPAPI_KEY='your_key'")
        print()
        print("Get API key from: https://serpapi.com/")
        return
    
    print(f"✓ API key configured\n")
    
    # Test claim
    claim = "Vietnam is the world's second largest coffee exporter"
    print(f"Test Claim: {claim}\n")
    
    # =========================================================================
    # PHASE 0: DATA COLLECTION (with concurrent scraping)
    # =========================================================================
    print("=" * 80)
    print("PHASE 0: DATA COLLECTION (Concurrent Scraping)")
    print("=" * 80)
    
    start_phase0 = time.time()
    
    from src.data_collection import DataCollector
    
    collector = DataCollector(search_api="serpapi", api_key=api_key)
    
    print("\nCollecting data...")
    corpus = collector.collect_corpus(
        claim=claim,
        num_urls=10,  # Optimized: 10 URLs instead of 20
        save=True
    )
    
    phase0_time = time.time() - start_phase0
    
    print(f"\n✓ Phase 0 complete in {phase0_time:.2f} seconds")
    print(f"  Documents collected: {len(corpus['corpus'])}")
    print(f"  Total words: {sum(len(doc['text'].split()) for doc in corpus['corpus']):,}")
    
    # =========================================================================
    # PHASE 1: INDEXING (with batch encoding)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: INDEXING (Batch Encoding)")
    print("=" * 80)
    
    start_phase1 = time.time()
    
    from src.retrieval.build_index import IndexBuilder
    
    print("\nBuilding indexes with batch encoding...")
    builder = IndexBuilder(
        batch_size=32,    # Batch encoding optimization
        device='auto'     # Auto-detect GPU/MPS/CPU
    )
    
    # Get the filename of the corpus we just created
    corpus_filename = collector.list_saved_corpora()[0]  # Most recent
    builder.build_from_corpus_file(corpus_filename)
    
    phase1_time = time.time() - start_phase1
    
    print(f"\n✓ Phase 1 complete in {phase1_time:.2f} seconds")
    
    # =========================================================================
    # PHASE 2: RETRIEVAL (BM25 + Hybrid Ranking)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: RETRIEVAL (Two-Stage Funnel)")
    print("=" * 80)
    
    start_phase2 = time.time()
    
    from src.retrieval import RetrievalOrchestrator
    
    print("\nInitializing retrieval orchestrator...")
    orchestrator = RetrievalOrchestrator(
        semantic_weight=0.5,
        lexical_weight=0.3,
        metadata_weight=0.2,
        stage1_k=50
    )
    
    print("\nRetrieving and ranking evidence...")
    results = orchestrator.retrieve_and_rank(
        claim=claim,
        top_k=12,
        verbose=True
    )
    
    phase2_time = time.time() - start_phase2
    
    print(f"\n✓ Phase 2 complete in {phase2_time:.2f} seconds")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if results:
        print(f"\nTop 3 Evidence Sentences:\n")
        for i, result in enumerate(results[:3], 1):
            print(f"{i}. [{result['combined_score']:.4f}] {result['text'][:150]}...")
            print(f"   Source: {result['metadata']['doc_domain']}")
            print(f"   Scores: Semantic={result['scores']['semantic']:.3f}, "
                  f"Lexical={result['scores']['lexical']:.3f}, "
                  f"Metadata={result['scores']['metadata']:.3f}\n")
    
    # =========================================================================
    # PERFORMANCE SUMMARY
    # =========================================================================
    total_time = phase0_time + phase1_time + phase2_time
    
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print()
    print(f"  Phase 0 (Data Collection):  {phase0_time:6.2f}s  (concurrent scraping)")
    print(f"  Phase 1 (Indexing):         {phase1_time:6.2f}s  (batch encoding)")
    print(f"  Phase 2 (Retrieval):        {phase2_time:6.2f}s  (BM25 + hybrid)")
    print(f"  {'─' * 50}")
    print(f"  TOTAL TIME:                 {total_time:6.2f}s")
    print()
    print("Target Performance (with optimizations):")
    print("  Phase 0: ~2-3 seconds  (vs 15s without optimization)")
    print("  Phase 1: ~2 seconds    (vs 15s without optimization)")
    print("  Phase 2: ~0.5 seconds")
    print("  Total:   ~5-7 seconds  (vs 31s without optimization)")
    print()
    
    if total_time <= 8:
        print("✅ EXCELLENT! Performance meets optimization targets!")
    elif total_time <= 12:
        print("⚠️  GOOD, but slightly slower than target (network latency?)")
    else:
        print("❌ SLOW - Check if optimizations are enabled")
    
    print()
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Check API key is configured")
        print("  2. Ensure all dependencies installed: pip install -r requirements.txt")
        print("  3. Check internet connection")
