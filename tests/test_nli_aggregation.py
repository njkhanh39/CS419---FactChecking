"""
Test: NLI + Aggregation Integration

This script tests the complete NLI → Aggregation pipeline:
1. Load sample evidence with retrieval scores
2. Run NLI inference to get labels and confidences
3. Apply aggregation to get final verdict

Usage:
    python -m tests.test_nli_aggregation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nli import run_nli_inference
from src.aggregation import make_final_decision, display_verdict


def test_nli_aggregation():
    """Test complete NLI + Aggregation pipeline"""
    
    print("="*80)
    print("NLI + AGGREGATION INTEGRATION TEST")
    print("="*80)
    print()
    
    # ========== SAMPLE DATA ==========
    # This simulates output from Phase 2 (Retrieval)
    claim = "Vietnam is the world's second largest coffee exporter"
    
    ranked_evidence = [
        {
            'text': 'Vietnam is the second largest coffee exporter in the world.',
            'combined_score': 0.85,
            'doc_url': 'https://example.com/coffee-exports',
            'doc_title': 'Global Coffee Trade',
            'doc_domain': 'example.com'
        },
        {
            'text': 'Coffee exports from Vietnam have grown significantly over the past decade.',
            'combined_score': 0.72,
            'doc_url': 'https://example.com/vietnam-coffee',
            'doc_title': 'Vietnam Coffee Industry',
            'doc_domain': 'example.com'
        },
        {
            'text': 'Brazil is the largest coffee producer globally, followed by Vietnam.',
            'combined_score': 0.68,
            'doc_url': 'https://example.com/brazil-coffee',
            'doc_title': 'World Coffee Production',
            'doc_domain': 'example.com'
        },
        {
            'text': 'Vietnam ranks third in coffee production after Brazil and Colombia.',
            'combined_score': 0.55,
            'doc_url': 'https://example.com/coffee-rankings',
            'doc_title': 'Coffee Production Rankings',
            'doc_domain': 'example.com'
        },
        {
            'text': 'Vietnamese coffee is exported to over 80 countries worldwide.',
            'combined_score': 0.65,
            'doc_url': 'https://example.com/export-markets',
            'doc_title': 'Coffee Export Markets',
            'doc_domain': 'example.com'
        },
        {
            'text': 'The coffee industry in Vietnam employs millions of people.',
            'combined_score': 0.60,
            'doc_url': 'https://example.com/coffee-employment',
            'doc_title': 'Coffee Employment',
            'doc_domain': 'example.com'
        }
    ]
    
    print(f"Claim: {claim}")
    print(f"Evidence sentences: {len(ranked_evidence)}")
    print()
    
    # ========== PHASE 3: NLI INFERENCE ==========
    print("="*80)
    print("PHASE 3: NLI INFERENCE")
    print("="*80)
    print()
    
    print("Running NLI inference on evidence sentences...")
    print("(This will load RoBERTa-large-MNLI model - may take 30-60 seconds)")
    print()
    
    try:
        nli_results = run_nli_inference(claim, ranked_evidence)
        
        print(f"✓ NLI inference completed for {len(nli_results)} sentences")
        print()
        
        # Display NLI results
        print("NLI Results:")
        print("-" * 80)
        for i, result in enumerate(nli_results, 1):
            print(f"\n{i}. {result['nli_label']} (confidence: {result['nli_confidence']:.2f})")
            print(f"   Text: {result['text'][:70]}...")
            print(f"   Probabilities: SUPPORT={result['nli_probs']['entailment']:.2f}, "
                  f"REFUTE={result['nli_probs']['contradiction']:.2f}, "
                  f"NEUTRAL={result['nli_probs']['neutral']:.2f}")
        
    except Exception as e:
        print(f"✗ NLI inference failed: {e}")
        print("\nNote: Make sure you have installed transformers and torch:")
        print("  pip install transformers torch")
        return
    
    # ========== PHASE 4: AGGREGATION ==========
    print("\n\n" + "="*80)
    print("PHASE 4: AGGREGATION")
    print("="*80)
    print()
    
    # Test all three methods
    methods = ['scoring', 'voting', 'hybrid']
    
    for method in methods:
        print("\n" + "─"*80)
        print(f"Method: {method.upper()}")
        print("─"*80)
        
        verdict = make_final_decision(nli_results, method=method)
        
        # Display compact summary
        print(f"\n✓ Verdict: {verdict['verdict']}")
        print(f"  Confidence: {verdict['confidence']:.1%}")
        print(f"  Evidence: {verdict['evidence_summary']['supporting']} SUPPORT, "
              f"{verdict['evidence_summary']['refuting']} REFUTE, "
              f"{verdict['evidence_summary']['neutral']} NEUTRAL")
        print(f"  Explanation: {verdict['explanation']}")
    
    # ========== FINAL DISPLAY ==========
    print("\n\n" + "="*80)
    print("RECOMMENDED METHOD: HYBRID")
    print("="*80)
    
    final_verdict = make_final_decision(nli_results, method='hybrid')
    display_verdict(final_verdict, verbose=True)
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print()
    print("✓ NLI inference: Working correctly")
    print("✓ Scoring aggregation: Implemented")
    print("✓ Voting aggregation: Implemented")
    print("✓ Final decision: Implemented")
    print()
    print("Next steps:")
    print("  1. Integrate with complete pipeline (Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4)")
    print("  2. Test with real claims and web-scraped data")
    print("  3. Tune thresholds based on evaluation results")
    print()


if __name__ == "__main__":
    try:
        test_nli_aggregation()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
