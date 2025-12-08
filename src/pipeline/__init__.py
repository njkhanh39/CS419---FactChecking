"""
Pipeline Module: End-to-End Fact-Checking Orchestration
========================================================

This module provides the complete fact-checking pipeline that connects all phases.

Main Component:
--------------
FactChecker - Orchestrates the complete pipeline:
  - Phase 0: Data Collection (Web Search + Scraping)
  - Phase 1: Indexing (BM25 + FAISS)
  - Phase 2: Retrieval (Two-Stage Hybrid Ranking)
  - Phase 3: NLI Inference (RoBERTa-MNLI)
  - Phase 4: Aggregation (Final Verdict)

Usage:
------
from src.pipeline import FactChecker

# Initialize
checker = FactChecker()

# Check a claim
result = checker.check_claim("Your claim here")

# Display result
checker.display_result(result)

# Or run interactively
checker.run_interactive()

Command-line Usage:
------------------
# Interactive mode
python -m src.pipeline.fact_check

# Single claim
python -m src.pipeline.fact_check --claim "Your claim here"
"""

from .fact_check import FactChecker

__all__ = ['FactChecker']
