"""
Aggregation Module: Convert NLI results to final verdict

This module provides three complementary approaches for aggregating NLI results:

1. **Scoring** (scoring.py):
   - Converts NLI labels to numerical scores
   - SUPPORT → +confidence, REFUTE → -confidence, NEUTRAL → 0
   - Aggregates scores using weighted sum, simple sum, or average

2. **Voting** (voting.py):
   - Simple majority voting or weighted voting
   - Counts label frequencies and determines winner
   - Handles tie-breaking using confidence scores

3. **Final Decision** (final_decision.py):
   - Combines scoring and voting for robust verdict
   - Applies decision thresholds and rules
   - Generates comprehensive fact-checking result

Usage:
    from src.aggregation import make_final_decision
    
    # After NLI inference
    verdict = make_final_decision(nli_results, method='hybrid')
    
    print(f"Verdict: {verdict['verdict']}")
    print(f"Confidence: {verdict['confidence']:.1%}")
"""

from .scoring import (
    label_to_score,
    calculate_evidence_scores,
    aggregate_scores
)

from .voting import (
    majority_vote,
    weighted_vote
)

from .final_decision import (
    make_final_decision,
    display_verdict
)

__all__ = [
    # Scoring
    'label_to_score',
    'calculate_evidence_scores',
    'aggregate_scores',
    
    # Voting
    'majority_vote',
    'weighted_vote',
    
    # Final Decision
    'make_final_decision',
    'display_verdict'
]
