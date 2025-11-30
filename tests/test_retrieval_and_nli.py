"""
Pipeline Demo: Phase 0 + Phase 1 + Phase 2 (NLI Integration)
============================================================

This script demonstrates the pipeline up to the NLI stage:
1. Data Collection (Scraping)
2. Indexing (BM25 + Embeddings)
3. Retrieval (Hybrid Ranking)
4. NLI Inference (RoBERTa-MNLI) - NEW!

Expected timing (per claim, excluding init):
- Phase 0: ~2-5s
- Phase 1: ~2s
- Phase 2 (Retrieval): ~0.5s
- Phase 3 (NLI): ~0.5s - 1.5s (depending on hardware)

Usage:
    python -m tests.test_retrieval_and_nli
"""

import os
import sys
import time
from pathlib import Path

def main():
    # Redirect output to file while keeping console output
    output_file = Path("tests/test_nli_result.txt")
    output_file.parent.mkdir(exist_ok=True)
    
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
    
    log_file = open(output_file, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(original_stdout, log_file)
    
    try:
        _run_pipeline()
    finally:
        sys.stdout = original_stdout
        log_file.close()
        print(f"\n✓ Results saved to: {output_file.absolute()}")

def _run_pipeline():
    print("=" * 80)
    print("RETRIEVAL + NLI PIPELINE DEMO")
    print("=" * 80)
    
    # 1. Check API Key
    api_key = None
    try:
        from src.config.api_keys import SERPAPI_KEY
        api_key = SERPAPI_KEY
    except ImportError:
        api_key = os.environ.get('SERPAPI_KEY')
    
    if not api_key:
        print("❌ ERROR: API key not found in src/config/api_keys.py")
        return

    # 2. Initialization (One-time setup)
    print("\n[INITIALIZATION] Loading models (This happens once per server start)...")
    init_start = time.time()

    # Init Phase 0 (Collector)
    from src.data_collection import DataCollector
    collector = DataCollector(search_api="serpapi", api_key=api_key)

    # Init Phase 1 (Index Builder - loads SentenceTransformer)
    from src.retrieval.build_index import IndexBuilder
    print("   Loading Sentence Transformer...")
    builder = IndexBuilder(preload_model=True)

    # Init Phase 1/2 (Retriever Orchestrator)
    from src.retrieval import RetrievalOrchestrator
    # We will instantiate this inside the loop to reload indexes, 
    # but imports are done here.

    # Init Phase 3 (NLI Model)
    print("   Loading NLI Model (RoBERTa-MNLI)...")
    from src.nli.batch_inference import get_model_instance, run_nli_inference
    # Force load model into memory now so it doesn't count against per-claim time
    get_model_instance() 

    init_time = time.time() - init_start
    print(f"✓ Initialization complete in {init_time:.2f}s\n")

    # 3. Test Claims
    test_claims = [
        "Vietnam is the world's second largest coffee exporter",
        "The Great Wall of China is visible from space with the naked eye",
        "Python is an interpreted programming language"
    ]

    for i, claim in enumerate(test_claims, 1):
        print("=" * 80)
        print(f"CLAIM {i}: {claim}")
        print("=" * 80)

        # --- Phase 0: Collection ---
        print(f"\n1. Data Collection...")
        t0 = time.time()
        corpus = collector.collect_corpus(claim=claim, num_urls=5, save=True)
        t0_end = time.time()
        
        # Find latest corpus file
        corpus_dir = Path("data/raw")
        latest_corpus = max(corpus_dir.glob("corpus_*.json"), key=lambda f: f.stat().st_mtime)
        print(f"   ✓ Collected {len(corpus['corpus'])} docs ({t0_end - t0:.2f}s)")

        # --- Phase 1: Indexing ---
        print(f"2. Indexing...")
        t1 = time.time()
        builder.build_from_corpus_file(latest_corpus.name, claim)
        t1_end = time.time()
        print(f"   ✓ Indexing complete ({t1_end - t1:.2f}s)")

        # --- Phase 2: Retrieval ---
        print(f"3. Retrieval (Hybrid)...")
        t2 = time.time()
        # Re-init orchestrator to pick up new indexes
        orchestrator = RetrievalOrchestrator(semantic_weight=0.5, lexical_weight=0.3, metadata_weight=0.2)
        ranked_results = orchestrator.retrieve_and_rank(claim=claim, top_k=10, verbose=False)
        t2_end = time.time()
        print(f"   ✓ Retrieved {len(ranked_results)} sentences ({t2_end - t2:.2f}s)")

        # --- Phase 3: NLI Inference ---
        print(f"4. NLI Inference...")
        t3 = time.time()
        nli_results = run_nli_inference(claim, ranked_results)
        t3_end = time.time()
        print(f"   ✓ Inference complete ({t3_end - t3:.2f}s)")

        # --- Display Results ---
        print("\nTOP EVIDENCE WITH NLI VERDICT:")
        print("-" * 80)
        print(f"{'VERDICT':<10} | {'CONF':<6} | {'TEXT (Truncated)':<60}")
        print("-" * 80)
        
    #    for item in nli_results[:5]: # Show top 5
        for item in nli_results: # Show all
            label = item['nli_label']
            conf = item['nli_confidence']
            text = item['text'].replace('\n', ' ')
            if len(text) > 55: text = text[:55] + "..."
            
            # Color code output if supported (simple ANSI)
            color = ""
            if label == "SUPPORT": color = "\033[92m" # Green
            elif label == "REFUTE": color = "\033[91m" # Red
            elif label == "NEUTRAL": color = "\033[93m" # Yellow
            reset = "\033[0m"
            
            print(f"{color}{label:<10}{reset} | {conf:.2f}   | {text}")
        print("-" * 80)
        
        total_time = (t0_end - t0) + (t1_end - t1) + (t2_end - t2) + (t3_end - t3)
        print(f"\nTotal Processing Time: {total_time:.2f}s")
        print("\n")

if __name__ == "__main__":
    main()