"""
End-to-End Fact-Checking Pipeline
==================================

This module provides the complete fact-checking pipeline that connects all phases:

Phase 0: Data Collection (Web Search + Scraping)
Phase 1: Indexing (BM25 + FAISS with Funnel Architecture)
Phase 2: Retrieval (Two-Stage Hybrid Ranking)
Phase 3: NLI Inference (RoBERTa-MNLI)
Phase 4: Aggregation (Scoring + Voting → Final Verdict)

Usage:
------
from src.pipeline.fact_check import FactChecker

# Initialize (one-time setup)
checker = FactChecker()

# Check a claim
result = checker.check_claim("Vietnam is the second largest coffee exporter")

# Display result
checker.display_result(result)

# Or run interactively
checker.run_interactive()
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import pipeline components
from ..data_collection import DataCollector
from ..retrieval import IndexBuilder, RetrievalOrchestrator
from ..nli import run_nli_inference
from ..aggregation import make_final_decision, display_verdict


class FactChecker:
    """
    Complete end-to-end fact-checking system
    
    Orchestrates all pipeline phases from user query to final verdict.
    """
    
    def __init__(
        self,
        search_api: str = "serpapi",
        num_urls: int = 10,
        top_k_retrieval: int = 12,
        aggregation_method: str = "hybrid",
        verbose: bool = True
    ):
        """
        Initialize FactChecker with all pipeline components
        
        Args:
            search_api: Search API to use ("serpapi" or "bing")
            num_urls: Number of URLs to scrape (default: 10)
            top_k_retrieval: Number of sentences to retrieve (default: 12)
            aggregation_method: Aggregation method ("hybrid", "scoring", "voting")
            verbose: Print detailed progress information
        """
        self.num_urls = num_urls
        self.top_k_retrieval = top_k_retrieval
        self.aggregation_method = aggregation_method
        self.verbose = verbose
        
        if self.verbose:
            print("\n" + "="*80)
            print("FACT-CHECKING PIPELINE INITIALIZATION")
            print("="*80)
            print()
        
        # ========== PHASE 0: Data Collection ==========
        if self.verbose:
            print("[Phase 0] Initializing Data Collection...")
        
        try:
            self.data_collector = DataCollector(search_api=search_api)
            if self.verbose:
                print("  ✓ Data collector ready")
        except Exception as e:
            logger.error(f"Failed to initialize Data Collector: {e}")
            raise RuntimeError(
                "Data Collection initialization failed. "
                "Please ensure API keys are configured in src/config/api_keys.py"
            )
        
        # ========== PHASE 1: Indexing ==========
        if self.verbose:
            print("\n[Phase 1] Initializing Index Builder...")
        
        try:
            self.index_builder = IndexBuilder(
                batch_size=16,
                device='auto',
                preload_model=True  # Pre-load model during init
            )
            if self.verbose:
                print("  ✓ Index builder ready (model pre-loaded)")
        except Exception as e:
            logger.error(f"Failed to initialize Index Builder: {e}")
            raise RuntimeError("Index Builder initialization failed.")
        
        # Phase 2: Retrieval Orchestrator will be initialized after first indexing
        self.retrieval_orchestrator: Optional[RetrievalOrchestrator] = None
        
        if self.verbose:
            print("\n[Phase 2] Retrieval Orchestrator")
            print("  ⏳ Will be initialized after first indexing")
        
        # ========== PHASE 3: NLI ==========
        if self.verbose:
            print("\n[Phase 3] Initializing NLI Model...")
        
        try:
            # Pre-load NLI model during initialization to avoid delay during inference
            from src.nli.batch_inference import get_model_instance
            import time
            nli_start = time.time()
            self.nli_model = get_model_instance()
            nli_time = time.time() - nli_start
            if self.verbose:
                print(f"  ✓ NLI model loaded and ready ({nli_time:.2f}s)")
        except Exception as e:
            logger.error(f"Failed to pre-load NLI model: {e}")
            if self.verbose:
                print("  ⚠️  NLI model will be loaded on first use")
            self.nli_model = None
        
        # ========== PHASE 4: Aggregation ==========
        if self.verbose:
            print("\n[Phase 4] Aggregation")
            print(f"  ✓ Method: {aggregation_method}")
        
        if self.verbose:
            print("\n" + "="*80)
            print("✓ INITIALIZATION COMPLETE")
            print("="*80)
            print()
    
    def check_claim(
        self, 
        claim: str,
        num_urls: Optional[int] = None,
        top_k: Optional[int] = None,
        method: Optional[str] = None,
        save_corpus: bool = True
    ) -> Dict[str, Any]:
        """
        Complete fact-checking pipeline for a single claim
        
        Args:
            claim: The claim to fact-check
            num_urls: Override default number of URLs to scrape
            top_k: Override default number of sentences to retrieve
            method: Override default aggregation method
            save_corpus: Whether to save collected corpus to disk
            
        Returns:
            Complete fact-checking result dictionary with verdict
        """
        num_urls = num_urls or self.num_urls
        top_k = top_k or self.top_k_retrieval
        method = method or self.aggregation_method
        
        if self.verbose:
            print("\n" + "="*80)
            print("FACT-CHECKING PIPELINE - EXECUTION")
            print("="*80)
            print(f"\nClaim: {claim}")
            print(f"Parameters: {num_urls} URLs, top-{top_k} retrieval, {method} aggregation")
            print()
        
        start_time = time.time()
        result = {
            'claim': claim,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'parameters': {
                'num_urls': num_urls,
                'top_k': top_k,
                'aggregation_method': method
            }
        }
        
        try:
            # ========== PHASE 0: DATA COLLECTION ==========
            if self.verbose:
                print("\n" + "─"*80)
                print("PHASE 0: DATA COLLECTION")
                print("─"*80)
            
            phase0_start = time.time()
            corpus = self.data_collector.collect_corpus(
                claim=claim,
                num_urls=num_urls,
                save=save_corpus
            )
            phase0_time = time.time() - phase0_start
            
            if not corpus or not corpus.get('corpus'):
                logger.error("Data collection failed: No documents retrieved")
                return {
                    **result,
                    'verdict': 'NOT ENOUGH INFO',
                    'confidence': 0.0,
                    'error': 'No documents could be retrieved from the web',
                    'phase_times': {'phase0': phase0_time}
                }
            
            result['phase0'] = {
                'time': phase0_time,
                'num_documents': len(corpus['corpus']),
                'corpus_file': corpus.get('metadata', {}).get('corpus_file', None),
                'urls': [doc.get('url', '') for doc in corpus['corpus']]
            }
            
            if self.verbose:
                print(f"\n✓ Phase 0 Complete: {phase0_time:.2f}s")
                print(f"  Documents collected: {len(corpus['corpus'])}")
            
            # ========== PHASE 1: INDEXING ==========
            if self.verbose:
                print("\n" + "─"*80)
                print("PHASE 1: INDEXING (BM25 + FAISS)")
                print("─"*80)
            
            phase1_start = time.time()
            
            # Get corpus filename
            corpus_file = corpus.get('metadata', {}).get('corpus_file')
            if not corpus_file:
                # Try to find most recent corpus file
                from pathlib import Path
                corpus_dir = Path("data/raw")
                corpus_files = list(corpus_dir.glob("corpus_*.json"))
                if corpus_files:
                    corpus_file = max(corpus_files, key=lambda f: f.stat().st_mtime).name
                else:
                    raise FileNotFoundError("No corpus file found")
            
            # Build indexes with funnel architecture
            self.index_builder.build_from_corpus_file(corpus_file, claim)
            phase1_time = time.time() - phase1_start
            
            result['phase1'] = {
                'time': phase1_time,
                'corpus_file': corpus_file
            }
            
            if self.verbose:
                print(f"\n✓ Phase 1 Complete: {phase1_time:.2f}s")
            
            # Initialize retrieval orchestrator after first indexing
            if self.retrieval_orchestrator is None:
                if self.verbose:
                    print("\n  Initializing Retrieval Orchestrator...")
                self.retrieval_orchestrator = RetrievalOrchestrator(
                    semantic_weight=0.5,
                    lexical_weight=0.3,
                    metadata_weight=0.2,
                    stage1_k=50
                )
            else:
                # Reinitialize to load new indexes
                if self.verbose:
                    print("\n  Reinitializing Retrieval Orchestrator for new indexes...")
                self.retrieval_orchestrator = RetrievalOrchestrator(
                    semantic_weight=0.5,
                    lexical_weight=0.3,
                    metadata_weight=0.2,
                    stage1_k=50
                )
            
            # ========== PHASE 2: RETRIEVAL ==========
            if self.verbose:
                print("\n" + "─"*80)
                print("PHASE 2: RETRIEVAL (Two-Stage Hybrid Ranking)")
                print("─"*80)
            
            phase2_start = time.time()
            ranked_evidence = self.retrieval_orchestrator.retrieve_and_rank(
                claim=claim,
                top_k=top_k,
                verbose=self.verbose
            )
            phase2_time = time.time() - phase2_start
            
            if not ranked_evidence:
                logger.error("Retrieval failed: No evidence found")
                return {
                    **result,
                    'verdict': 'NOT ENOUGH INFO',
                    'confidence': 0.0,
                    'error': 'No relevant evidence could be retrieved',
                    'phase_times': {
                        'phase0': phase0_time,
                        'phase1': phase1_time,
                        'phase2': phase2_time
                    }
                }
            
            result['phase2'] = {
                'time': phase2_time,
                'num_evidence': len(ranked_evidence),
                'retrieved_sentences': [
                    {
                        'text': ev.get('text', ''),
                        'score': ev.get('combined_score', 0),
                        'source': ev.get('doc_url', '')
                    }
                    for ev in ranked_evidence[:5]  # Top 5 for display
                ]
            }
            
            if self.verbose:
                print(f"\n✓ Phase 2 Complete: {phase2_time:.2f}s")
                print(f"  Evidence retrieved: {len(ranked_evidence)}")
            
            # ========== PHASE 3: NLI INFERENCE ==========
            if self.verbose:
                print("\n" + "─"*80)
                print("PHASE 3: NLI INFERENCE (RoBERTa-MNLI)")
                print("─"*80)
            
            phase3_start = time.time()
            nli_results = run_nli_inference(claim, ranked_evidence)
            phase3_time = time.time() - phase3_start
            
            if not nli_results:
                logger.error("NLI inference failed: No results")
                return {
                    **result,
                    'verdict': 'NOT ENOUGH INFO',
                    'confidence': 0.0,
                    'error': 'NLI inference failed',
                    'phase_times': {
                        'phase0': phase0_time,
                        'phase1': phase1_time,
                        'phase2': phase2_time,
                        'phase3': phase3_time
                    }
                }
            
            result['phase3'] = {
                'time': phase3_time,
                'num_results': len(nli_results)
            }
            
            if self.verbose:
                print(f"\n✓ Phase 3 Complete: {phase3_time:.2f}s")
                print(f"  NLI results: {len(nli_results)}")
                
                # Show label distribution
                support_count = sum(1 for r in nli_results if r.get('nli_label') == 'SUPPORT')
                refute_count = sum(1 for r in nli_results if r.get('nli_label') == 'REFUTE')
                neutral_count = sum(1 for r in nli_results if r.get('nli_label') == 'NEUTRAL')
                print(f"  Labels: {support_count} SUPPORT, {refute_count} REFUTE, {neutral_count} NEUTRAL")
            
            # ========== PHASE 4: AGGREGATION ==========
            if self.verbose:
                print("\n" + "─"*80)
                print("PHASE 4: AGGREGATION (Final Verdict)")
                print("─"*80)
            
            phase4_start = time.time()
            verdict = make_final_decision(nli_results, method=method)
            phase4_time = time.time() - phase4_start
            
            result['phase4'] = {
                'time': phase4_time
            }
            
            if self.verbose:
                print(f"\n✓ Phase 4 Complete: {phase4_time:.2f}s")
            
            # ========== FINAL RESULT ==========
            total_time = time.time() - start_time
            
            result.update({
                'verdict': verdict['verdict'],
                'confidence': verdict['confidence'],
                'explanation': verdict['explanation'],
                'evidence_summary': verdict['evidence_summary'],
                'scores': verdict['scores'],
                'voting': verdict['voting'],
                'top_evidence': verdict['top_evidence'],
                'all_evidence': nli_results,  # Include all NLI results
                'phase_times': {
                    'phase0_collection': phase0_time,
                    'phase1_indexing': phase1_time,
                    'phase2_retrieval': phase2_time,
                    'phase3_nli': phase3_time,
                    'phase4_aggregation': phase4_time,
                    'total': total_time
                }
            })
            
            if self.verbose:
                print("\n" + "="*80)
                print("PIPELINE COMPLETE")
                print("="*80)
                print(f"\n⏱️  Total Time: {total_time:.2f}s")
                print(f"  Phase 0 (Collection): {phase0_time:.2f}s")
                print(f"  Phase 1 (Indexing):   {phase1_time:.2f}s")
                print(f"  Phase 2 (Retrieval):  {phase2_time:.2f}s")
                print(f"  Phase 3 (NLI):        {phase3_time:.2f}s")
                print(f"  Phase 4 (Aggregation): {phase4_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return {
                **result,
                'verdict': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'phase_times': {
                    'total': time.time() - start_time
                }
            }
    
    def display_result(self, result: Dict[str, Any], detailed: bool = True):
        """
        Display fact-checking result in human-readable format
        
        Args:
            result: Result dictionary from check_claim()
            detailed: Show detailed breakdown
        """
        # Use the display_verdict function from aggregation module
        if result.get('error'):
            print("\n" + "="*80)
            print("FACT-CHECKING RESULT")
            print("="*80)
            print(f"\n❌ ERROR: {result['error']}")
            print("\nPlease check:")
            print("  1. API keys are configured (src/config/api_keys.py)")
            print("  2. Internet connection is working")
            print("  3. Required packages are installed (pip install -r requirements.txt)")
            print()
            return
        
        # Create verdict result for display_verdict function
        verdict_result = {
            'verdict': result['verdict'],
            'confidence': result['confidence'],
            'claim': result['claim'],
            'explanation': result['explanation'],
            'evidence_summary': result['evidence_summary'],
            'scores': result['scores'],
            'voting': result['voting'],
            'top_evidence': result['top_evidence'],
            'method': result['parameters']['aggregation_method']
        }
        
        display_verdict(verdict_result, verbose=detailed)
        
        if detailed:
            # Show timing information
            times = result.get('phase_times', {})
            print("\n" + "─"*80)
            print("PERFORMANCE BREAKDOWN")
            print("─"*80)
            print(f"  Phase 0 (Data Collection): {times.get('phase0_collection', 0):.2f}s")
            print(f"  Phase 1 (Indexing):        {times.get('phase1_indexing', 0):.2f}s")
            print(f"  Phase 2 (Retrieval):       {times.get('phase2_retrieval', 0):.2f}s")
            print(f"  Phase 3 (NLI):             {times.get('phase3_nli', 0):.2f}s")
            print(f"  Phase 4 (Aggregation):     {times.get('phase4_aggregation', 0):.2f}s")
            print(f"  ─────────────────────────────────")
            print(f"  Total:                     {times.get('total', 0):.2f}s")
            print()
    
    def run_interactive(self):
        """
        Run interactive fact-checking session
        
        Allows user to enter multiple claims and get immediate results.
        """
        print("\n" + "="*80)
        print("INTERACTIVE FACT-CHECKING SESSION")
        print("="*80)
        print("\nType your claims below. Type 'quit', 'exit', or press Ctrl+C to stop.")
        print()
        
        claim_count = 0
        
        try:
            while True:
                # Get claim from user
                try:
                    claim = input("\n📝 Enter claim to fact-check: ").strip()
                except EOFError:
                    break
                
                if not claim:
                    print("⚠️  Empty claim. Please enter a valid claim.")
                    continue
                
                if claim.lower() in ['quit', 'exit', 'q']:
                    break
                
                claim_count += 1
                print(f"\n[Claim {claim_count}] Processing: {claim}")
                
                # Check the claim
                result = self.check_claim(claim)
                
                # Display result
                self.display_result(result, detailed=True)
                
        except KeyboardInterrupt:
            print("\n\n⏸️  Session interrupted by user.")
        
        print("\n" + "="*80)
        print(f"SESSION COMPLETE - Checked {claim_count} claim(s)")
        print("="*80)
        print()


def main():
    """
    Main entry point for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fact-Checking Pipeline - Verify claims using web evidence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m src.pipeline.fact_check
  
  # Check single claim
  python -m src.pipeline.fact_check --claim "Vietnam is the second largest coffee exporter"
  
  # Customize parameters
  python -m src.pipeline.fact_check --claim "..." --urls 20 --top-k 15 --method voting
        """
    )
    
    parser.add_argument(
        '--claim',
        type=str,
        help='Claim to fact-check (if not provided, runs interactive mode)'
    )
    
    parser.add_argument(
        '--urls',
        type=int,
        default=10,
        help='Number of URLs to scrape (default: 10)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=12,
        help='Number of evidence sentences to retrieve (default: 12)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['hybrid', 'scoring', 'voting'],
        default='hybrid',
        help='Aggregation method (default: hybrid)'
    )
    
    parser.add_argument(
        '--search-api',
        type=str,
        choices=['serpapi', 'bing'],
        default='serpapi',
        help='Search API to use (default: serpapi)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed progress information'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize fact checker
        checker = FactChecker(
            search_api=args.search_api,
            num_urls=args.urls,
            top_k_retrieval=args.top_k,
            aggregation_method=args.method,
            verbose=not args.quiet
        )
        
        if args.claim:
            # Single claim mode
            result = checker.check_claim(args.claim)
            checker.display_result(result, detailed=True)
        else:
            # Interactive mode
            checker.run_interactive()
            
    except KeyboardInterrupt:
        print("\n\n⏸️  Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ FATAL ERROR: {e}")
        print("\nPlease check:")
        print("  1. API keys are configured in src/config/api_keys.py")
        print("  2. All dependencies are installed: pip install -r requirements.txt")
        print("  3. Python version is 3.8 or higher")
        sys.exit(1)


if __name__ == "__main__":
    main()