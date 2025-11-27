"""
Demo Script: Complete Retrieval Pipeline
==========================================

This script demonstrates the complete two-stage retrieval pipeline:
1. Sentence filtering during indexing
2. Stage 1: BM25 retrieval (Top 50)
3. Stage 2: Hybrid ranking (Top 12)

Usage:
    python demo_retrieval.py

Prerequisites:
    1. Corpus collected: data/raw/corpus_*.json
    2. Indexes built: run build_index.py first
"""

from src.retrieval import RetrievalOrchestrator

def main():
    print("=" * 80)
    print("RETRIEVAL PIPELINE DEMO")
    print("=" * 80)
    print()
    
    # Initialize orchestrator
    print("Step 1: Initializing Retrieval Orchestrator...")
    orchestrator = RetrievalOrchestrator(
        semantic_weight=0.5,
        lexical_weight=0.3,
        metadata_weight=0.2,
        stage1_k=50  # Get top 50 from BM25 first
    )
    
    # Show system statistics
    stats = orchestrator.get_statistics()
    print("\nSystem Statistics:")
    print(f"  Total sentences indexed: {stats['total_sentences']}")
    print(f"  Stage 1 candidates (BM25): {stats['stage1_candidates']}")
    print(f"  Embedding model: {stats['embedding_model']}")
    print(f"  Embedding dimension: {stats['embedding_dim']}")
    print(f"  Weights: {stats['weights']}")
    print()
    
    # Test with example claims
    claims = [
        "Vietnam is the world's second largest coffee exporter",
        "Coffee production in Vietnam dropped by 40% in 2023",
        "Climate change affects coffee production in Southeast Asia"
    ]
    
    for i, claim in enumerate(claims, 1):
        print(f"\n{'#' * 80}")
        print(f"CLAIM {i}: {claim}")
        print(f"{'#' * 80}\n")
        
        # Retrieve and rank
        results = orchestrator.retrieve_and_rank(
            claim=claim,
            top_k=12,  # Final top 12 sentences
            verbose=True
        )
        
        # Display results
        if results:
            orchestrator.display_results(results, show_scores=True)
        else:
            print("No results found.\n")
        
        # Wait for user input before next claim
        if i < len(claims):
            input("\nPress Enter to continue to next claim...")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print("\n❌ ERROR: Indexes not found!")
        print(f"   {e}")
        print("\n📝 Please run these steps first:")
        print("   1. Collect data: python -m src.data_collection.collector")
        print("   2. Build indexes: python -m src.retrieval.build_index")
        print("\n   Then run this demo again.")
    except KeyboardInterrupt:
        print("\n\n⏸️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
