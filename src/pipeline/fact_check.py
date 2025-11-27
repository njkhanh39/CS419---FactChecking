"""
End-to-End Fact-Checking Pipeline

Orchestrates all phases:
  Phase 0: Data Collection (web search + scraping)
  Phase 1: Indexing & Retrieval (BM25 + Embedding with hybrid ranking)
  Phase 2: NLI Inference (RoBERTa-MNLI)
  Phase 3: Aggregation & Verdict

Usage:
    from src.pipeline.fact_check import FactChecker
    
    checker = FactChecker()
    result = checker.check_claim("Vietnam is the 2nd largest coffee exporter")
    print(result['verdict'], result['confidence'])
"""

# TODO: Implement FactChecker class
# TODO: Implement check_claim() method
# TODO: Integrate all phases: DataCollector → IndexBuilder → Retrieval → NLI → Aggregation