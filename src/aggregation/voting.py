"""Voting Module: Simple majority voting for claim verification

This module provides an alternative aggregation method based on majority voting.
Instead of numerical scores, it counts the labels and uses the majority to decide.

Voting Logic:
- Count SUPPORT, REFUTE, NEUTRAL labels
- Majority wins (requires >50% for definitive verdict)
- Tie-breaking: Use confidence scores if counts are close

Usage:
    from src.aggregation.voting import majority_vote
    
    verdict = majority_vote(nli_results)
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def majority_vote(
    nli_results: List[Dict[str, Any]], 
    tie_threshold: float = 0.1,
    min_evidence: int = 3
) -> Dict[str, Any]:
    """
    Determine verdict using majority voting
    
    Args:
        nli_results (list): Evidence with NLI labels
            Expected format: [
                {
                    'text': str,
                    'nli_label': str,
                    'nli_confidence': float,
                    ...
                },
                ...
            ]
        tie_threshold (float): If label counts are within this ratio, consider it a tie (default: 0.1)
        min_evidence (int): Minimum evidence required for confident verdict (default: 3)
        
    Returns:
        dict: Voting result:
            {
                'verdict': str,              # "SUPPORTED", "REFUTED", "NOT ENOUGH INFO"
                'confidence': float,         # Confidence in verdict (0-1)
                'vote_counts': {
                    'support': int,
                    'refute': int,
                    'neutral': int
                },
                'vote_percentages': {
                    'support': float,
                    'refute': float,
                    'neutral': float
                },
                'avg_confidence': float,     # Average confidence of winning votes
                'total_evidence': int,
                'method': 'majority_vote'
            }
    """
    if not nli_results:
        logger.warning("No NLI results provided for voting.")
        return {
            'verdict': 'NOT ENOUGH INFO',
            'confidence': 0.0,
            'vote_counts': {'support': 0, 'refute': 0, 'neutral': 0},
            'vote_percentages': {'support': 0.0, 'refute': 0.0, 'neutral': 0.0},
            'avg_confidence': 0.0,
            'total_evidence': 0,
            'method': 'majority_vote'
        }
    
    # Count votes
    support_votes = []
    refute_votes = []
    neutral_votes = []
    
    for item in nli_results:
        label = item.get('nli_label', 'NEUTRAL').upper()
        confidence = item.get('nli_confidence', 0.0)
        
        if label == 'SUPPORT':
            support_votes.append(confidence)
        elif label == 'REFUTE':
            refute_votes.append(confidence)
        else:
            neutral_votes.append(confidence)
    
    num_support = len(support_votes)
    num_refute = len(refute_votes)
    num_neutral = len(neutral_votes)
    total = len(nli_results)
    
    # Calculate percentages
    support_pct = num_support / total
    refute_pct = num_refute / total
    neutral_pct = num_neutral / total
    
    logger.info(f"Vote counts: {num_support} SUPPORT ({support_pct:.1%}), "
                f"{num_refute} REFUTE ({refute_pct:.1%}), "
                f"{num_neutral} NEUTRAL ({neutral_pct:.1%})")
    
    # Determine verdict based on majority
    if total < min_evidence:
        # Not enough evidence
        verdict = 'NOT ENOUGH INFO'
        confidence = 0.0
        avg_confidence = 0.0
        logger.info(f"Not enough evidence: {total} < {min_evidence}")
    
    elif num_support > num_refute and num_support > num_neutral:
        # SUPPORT wins
        if support_pct > 0.5:
            verdict = 'SUPPORTED'
            confidence = support_pct
            avg_confidence = sum(support_votes) / len(support_votes) if support_votes else 0.0
            logger.info(f"Verdict: SUPPORTED (majority: {support_pct:.1%})")
        else:
            # No clear majority (e.g., 40% support, 35% refute, 25% neutral)
            verdict = 'NOT ENOUGH INFO'
            confidence = 0.5
            avg_confidence = sum(support_votes) / len(support_votes) if support_votes else 0.0
            logger.info(f"No clear majority: {support_pct:.1%} support < 50%")
    
    elif num_refute > num_support and num_refute > num_neutral:
        # REFUTE wins
        if refute_pct > 0.5:
            verdict = 'REFUTED'
            confidence = refute_pct
            avg_confidence = sum(refute_votes) / len(refute_votes) if refute_votes else 0.0
            logger.info(f"Verdict: REFUTED (majority: {refute_pct:.1%})")
        else:
            verdict = 'NOT ENOUGH INFO'
            confidence = 0.5
            avg_confidence = sum(refute_votes) / len(refute_votes) if refute_votes else 0.0
            logger.info(f"No clear majority: {refute_pct:.1%} refute < 50%")
    
    elif abs(num_support - num_refute) / total < tie_threshold:
        # Tie - use confidence scores to break tie
        avg_support_conf = sum(support_votes) / len(support_votes) if support_votes else 0.0
        avg_refute_conf = sum(refute_votes) / len(refute_votes) if refute_votes else 0.0
        
        if avg_support_conf > avg_refute_conf:
            verdict = 'SUPPORTED'
            confidence = 0.6  # Lower confidence for tie-break
            avg_confidence = avg_support_conf
            logger.info(f"Tie broken by confidence: SUPPORT ({avg_support_conf:.2f}) > REFUTE ({avg_refute_conf:.2f})")
        else:
            verdict = 'REFUTED'
            confidence = 0.6
            avg_confidence = avg_refute_conf
            logger.info(f"Tie broken by confidence: REFUTE ({avg_refute_conf:.2f}) > SUPPORT ({avg_support_conf:.2f})")
    
    else:
        # Neutral dominates or unclear
        verdict = 'NOT ENOUGH INFO'
        confidence = neutral_pct
        avg_confidence = sum(neutral_votes) / len(neutral_votes) if neutral_votes else 0.0
        logger.info(f"Verdict: NOT ENOUGH INFO (neutral dominates: {neutral_pct:.1%})")
    
    return {
        'verdict': verdict,
        'confidence': confidence,
        'vote_counts': {
            'support': num_support,
            'refute': num_refute,
            'neutral': num_neutral
        },
        'vote_percentages': {
            'support': support_pct,
            'refute': refute_pct,
            'neutral': neutral_pct
        },
        'avg_confidence': avg_confidence,
        'total_evidence': total,
        'method': 'majority_vote'
    }


def weighted_vote(
    nli_results: List[Dict[str, Any]], 
    min_evidence: int = 3
) -> Dict[str, Any]:
    """
    Weighted voting: Each vote is weighted by its confidence and retrieval score
    
    Args:
        nli_results (list): Evidence with NLI labels and combined_score
        min_evidence (int): Minimum evidence required (default: 3)
        
    Returns:
        dict: Weighted voting result (same format as majority_vote)
    """
    if not nli_results:
        logger.warning("No NLI results provided for weighted voting.")
        return {
            'verdict': 'NOT ENOUGH INFO',
            'confidence': 0.0,
            'vote_counts': {'support': 0, 'refute': 0, 'neutral': 0},
            'vote_percentages': {'support': 0.0, 'refute': 0.0, 'neutral': 0.0},
            'avg_confidence': 0.0,
            'total_evidence': 0,
            'method': 'weighted_vote'
        }
    
    # Calculate weighted votes
    support_weight = 0.0
    refute_weight = 0.0
    neutral_weight = 0.0
    total_weight = 0.0
    
    num_support = 0
    num_refute = 0
    num_neutral = 0
    
    for item in nli_results:
        label = item.get('nli_label', 'NEUTRAL').upper()
        confidence = item.get('nli_confidence', 0.0)
        retrieval_score = item.get('combined_score', 1.0)
        
        # Weight = confidence × retrieval_score
        weight = confidence * retrieval_score
        total_weight += weight
        
        if label == 'SUPPORT':
            support_weight += weight
            num_support += 1
        elif label == 'REFUTE':
            refute_weight += weight
            num_refute += 1
        else:
            neutral_weight += weight
            num_neutral += 1
    
    # Calculate weighted percentages
    support_pct = support_weight / total_weight if total_weight > 0 else 0.0
    refute_pct = refute_weight / total_weight if total_weight > 0 else 0.0
    neutral_pct = neutral_weight / total_weight if total_weight > 0 else 0.0
    
    logger.info(f"Weighted votes: SUPPORT {support_pct:.1%}, REFUTE {refute_pct:.1%}, NEUTRAL {neutral_pct:.1%}")
    
    # Determine verdict
    if len(nli_results) < min_evidence:
        verdict = 'NOT ENOUGH INFO'
        confidence = 0.0
    elif support_pct > refute_pct and support_pct > neutral_pct:
        verdict = 'SUPPORTED'
        confidence = support_pct
    elif refute_pct > support_pct and refute_pct > neutral_pct:
        verdict = 'REFUTED'
        confidence = refute_pct
    else:
        verdict = 'NOT ENOUGH INFO'
        confidence = max(support_pct, refute_pct, neutral_pct)
    
    return {
        'verdict': verdict,
        'confidence': confidence,
        'vote_counts': {
            'support': num_support,
            'refute': num_refute,
            'neutral': num_neutral
        },
        'vote_percentages': {
            'support': support_pct,
            'refute': refute_pct,
            'neutral': neutral_pct
        },
        'avg_confidence': confidence,
        'total_evidence': len(nli_results),
        'method': 'weighted_vote'
    }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample NLI results
    sample_nli_results = [
        {'text': 'Vietnam is 2nd largest exporter...', 'nli_label': 'SUPPORT', 'nli_confidence': 0.92, 'combined_score': 0.85},
        {'text': 'Coffee exports grew...', 'nli_label': 'SUPPORT', 'nli_confidence': 0.78, 'combined_score': 0.72},
        {'text': 'Brazil is largest...', 'nli_label': 'NEUTRAL', 'nli_confidence': 0.65, 'combined_score': 0.60},
        {'text': 'Vietnam ranks third...', 'nli_label': 'REFUTE', 'nli_confidence': 0.71, 'combined_score': 0.55},
        {'text': 'Vietnam coffee production...', 'nli_label': 'SUPPORT', 'nli_confidence': 0.85, 'combined_score': 0.68}
    ]
    
    print("="*70)
    print("VOTING MODULE - Example")
    print("="*70)
    print()
    
    # Majority vote
    print("[Method 1] Majority Vote")
    print("-" * 70)
    result = majority_vote(sample_nli_results)
    print(f"\nVerdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Votes: {result['vote_counts']['support']} SUPPORT, "
          f"{result['vote_counts']['refute']} REFUTE, "
          f"{result['vote_counts']['neutral']} NEUTRAL")
    print()
    
    # Weighted vote
    print("[Method 2] Weighted Vote")
    print("-" * 70)
    result = weighted_vote(sample_nli_results)
    print(f"\nVerdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Weighted: SUPPORT {result['vote_percentages']['support']:.1%}, "
          f"REFUTE {result['vote_percentages']['refute']:.1%}, "
          f"NEUTRAL {result['vote_percentages']['neutral']:.1%}")
