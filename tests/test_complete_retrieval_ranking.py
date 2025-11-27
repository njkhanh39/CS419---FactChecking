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
    python -m tests.test_complete_retrieval_ranking
"""

import os
import sys
import time
from pathlib import Path

def main():
    # Redirect all output to test_result.txt
    output_file = Path("tests/test_result.txt")
    output_file.parent.mkdir(exist_ok=True)
    
    # Use Tee-like behavior: write to both file and console
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
        def close(self):
            # Don't close stdout/stderr, only file handles
            for f in self.files:
                if hasattr(f, 'name') and f.name not in ['<stdout>', '<stderr>']:
                    f.close()
    
    log_file = open(output_file, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(original_stdout, log_file)
    sys.stderr = TeeOutput(original_stderr, log_file)
    
    try:
        _run_pipeline()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        print(f"\n✓ Results saved to: {output_file.absolute()}")

def _run_pipeline():
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
    
    # Test claims - multiple examples to test pipeline
    test_claims = [
        "Vietnam is the world's second largest coffee exporter",
        "The Eiffel Tower is 330 meters tall",
        "Python was created by Guido van Rossum in 1991",
        "The Great Wall of China is visible from space",
        "Water boils at 100 degrees Celsius at sea level"
    ]
    
    print(f"Testing {len(test_claims)} claims:\n")
    for i, claim in enumerate(test_claims, 1):
        print(f"  {i}. {claim}")
    print()
    
    # =========================================================================
    # INITIALIZATION (After user enters query - NOT counted in phase times)
    # =========================================================================
    print("=" * 80)
    print("INITIALIZATION (one-time setup after query entered)")
    print("=" * 80)
    print("\nInitializing pipeline components...")
    
    init_start = time.time()
    
    # Initialize Phase 0 components (no special init needed - just imports)
    from src.data_collection import DataCollector
    
    # Initialize Phase 1 components (pre-load model to avoid counting it in Phase 1)
    from src.retrieval.build_index import IndexBuilder
    print("   Loading sentence embedding model...")
    builder = IndexBuilder(
        batch_size=32,
        device='auto',
        preload_model=True  # Pre-load model during init
    )
    
    # Initialize Phase 2 components (load indexes if needed)
    # RetrievalOrchestrator will be initialized after indexing
    
    init_time = time.time() - init_start
    
    print(f"\n✓ Initialization complete in {init_time:.2f} seconds")
    print("   (This time is NOT included in phase measurements)\n")
    
    # Initialize data collector
    collector = DataCollector(search_api="serpapi", api_key=api_key)
    
    # Clean up old corpus files to avoid duplicates
    import glob
    old_files = glob.glob(str(Path("data/raw/corpus_*.json")))
    if old_files:
        print(f"Cleaning up {len(old_files)} old corpus files...")
        for f in old_files:
            try:
                Path(f).unlink()
            except:
                pass
    
    # Import RetrievalOrchestrator (but don't initialize yet - need indexes first)
    from src.retrieval import RetrievalOrchestrator
    orchestrator = None  # Will be initialized after first indexing
    
    # =========================================================================
    # PROCESS MULTIPLE CLAIMS
    # =========================================================================
    
    all_results = []
    
    for claim_idx, claim in enumerate(test_claims, 1):
        print("=" * 80)
        print(f"CLAIM {claim_idx}/{len(test_claims)}: {claim}")
        print("=" * 80)
        print()
        
        # PHASE 0: DATA COLLECTION
        print(f"[Phase 0] Data Collection...")
        start_phase0 = time.time()
        corpus = collector.collect_corpus(claim=claim, num_urls=10, save=True)
        phase0_time = time.time() - start_phase0
        print(f"✓ Phase 0: {phase0_time:.2f}s ({len(corpus['corpus'])} docs, {sum(len(doc['text'].split()) for doc in corpus['corpus']):,} words)")
        
        # Get the just-saved corpus filename (most recent by modification time)
        corpus_dir = Path("data/raw")
        corpus_files = list(corpus_dir.glob("corpus_*.json"))
        if corpus_files:
            # Sort by modification time, get most recent
            latest_corpus = max(corpus_files, key=lambda f: f.stat().st_mtime)
            corpus_filename = latest_corpus.name
        else:
            raise FileNotFoundError("No corpus files found after collection!")
        
        # PHASE 1: INDEXING (with funnel architecture)
        print(f"[Phase 1] Indexing...")
        start_phase1 = time.time()
        builder.build_from_corpus_file(corpus_filename, claim)  # Pass claim for BM25 filtering
        phase1_time = time.time() - start_phase1
        print(f"✓ Phase 1: {phase1_time:.2f}s (BM25 + encoding top 50)")
        
        # CRITICAL: Reinitialize orchestrator AFTER EACH indexing to load new indexes
        # If we don't do this, orchestrator uses cached indexes from previous claim
        print(f"   Reinitializing Retrieval Orchestrator to load new indexes...")
        orchestrator = RetrievalOrchestrator(
            semantic_weight=0.5,
            lexical_weight=0.3,
            metadata_weight=0.2,
            stage1_k=50
        )
        
        # PHASE 2: RETRIEVAL
        print(f"[Phase 2] Retrieval...")
        start_phase2 = time.time()
        results = orchestrator.retrieve_and_rank(claim=claim, top_k=12, verbose=False)
        phase2_time = time.time() - start_phase2
        print(f"✓ Phase 2: {phase2_time:.2f}s")
        
        total_time = phase0_time + phase1_time + phase2_time
        print(f"\n✓ Total processing: {total_time:.2f}s")
        
        # Store results
        all_results.append({
            'claim': claim,
            'phase0_time': phase0_time,
            'phase1_time': phase1_time,
            'phase2_time': phase2_time,
            'total_time': total_time,
            'num_docs': len(corpus['corpus']),
            'top_evidence': results[:3] if results else []
        })
        
        # Show top evidence
        if results:
            print(f"\nTop 3 Evidence:")
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. [{result['combined_score']:.3f}] {result['text'][:100]}...")
        print()
    
    # =========================================================================
    # SUMMARY ACROSS ALL CLAIMS
    # =========================================================================
    
    print("=" * 80)
    print("PERFORMANCE SUMMARY - ALL CLAIMS")
    print("=" * 80)
    print()
    
    # Calculate averages
    avg_phase0 = sum(r['phase0_time'] for r in all_results) / len(all_results)
    avg_phase1 = sum(r['phase1_time'] for r in all_results) / len(all_results)
    avg_phase2 = sum(r['phase2_time'] for r in all_results) / len(all_results)
    avg_total = sum(r['total_time'] for r in all_results) / len(all_results)
    total_processing = sum(r['total_time'] for r in all_results)
    
    print(f"INITIALIZATION (one-time): {init_time:.2f}s")
    print()
    print("PER-CLAIM PERFORMANCE:")
    print(f"  Claims processed:           {len(all_results)}")
    print(f"  Average Phase 0:            {avg_phase0:.2f}s")
    print(f"  Average Phase 1:            {avg_phase1:.2f}s")
    print(f"  Average Phase 2:            {avg_phase2:.2f}s")
    print(f"  Average per claim:          {avg_total:.2f}s")
    print()
    print(f"  Total processing time:      {total_processing:.2f}s")
    print(f"  Total with init:            {init_time + total_processing:.2f}s")
    print(f"  Throughput:                 {len(all_results) / (total_processing/60):.1f} claims/min")
    print()
    
    print("BREAKDOWN BY CLAIM:")
    for i, r in enumerate(all_results, 1):
        print(f"  {i}. {r['claim'][:50]:50s} | {r['total_time']:5.2f}s (P0:{r['phase0_time']:4.1f}s P1:{r['phase1_time']:4.1f}s P2:{r['phase2_time']:4.1f}s)")
    print()
    
    if avg_total <= 12:
        print("✅ EXCELLENT! Average processing time is optimal!")
    elif avg_total <= 18:
        print("✅ GOOD! Average processing time meets targets!")
    else:
        print("⚠️ SLOWER THAN TARGET - Check network and encoding performance")
    
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
