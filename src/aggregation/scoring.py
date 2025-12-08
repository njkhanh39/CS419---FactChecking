"""Scoring Module: Convert NLI labels to numerical scores

This module translates categorical NLI labels (SUPPORT, REFUTE, NEUTRAL) 
and their confidence scores into standardized numerical values for aggregation.

Scoring scheme:
- SUPPORT: +confidence (e.g., +0.9 for high confidence support)
- REFUTE: -confidence (e.g., -0.85 for high confidence refutation)
- NEUTRAL: 0 (neutral evidence doesn't contribute to verdict)

Usage:
    from src.aggregation.scoring import calculate_evidence_scores, aggregate_scores
    
    evidence_scores = calculate_evidence_scores(nli_results)
    total_score = aggregate_scores(evidence_scores)
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def label_to_score(label: str, confidence: float) -> float:
    """
    Convert NLI label and confidence to numerical score
    
    Args:
        label (str): NLI label - "SUPPORT", "REFUTE", or "NEUTRAL"
        confidence (float): Model confidence (0-1)
        
    Returns:
        float: Numerical score
            - SUPPORT: +confidence (e.g., +0.9)
            - REFUTE: -confidence (e.g., -0.85)
            - NEUTRAL: 0
    """
    label = label.upper()
    
    if label == "SUPPORT":
        return confidence
    elif label == "REFUTE":
        return -confidence
    elif label == "NEUTRAL":
        return 0.0
    else:
        logger.warning(f"Unknown NLI label: {label}. Treating as NEUTRAL.")
        return 0.0


def calculate_evidence_scores(nli_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate numerical scores for all evidence sentences
    
    Args:
        nli_results (list): Evidence with NLI labels from batch_inference
            Expected format: [
                {
                    'text': str,
                    'nli_label': str,
                    'nli_confidence': float,
                    'nli_probs': dict,
                    ...
                },
                ...
            ]
            
    Returns:
        list: Evidence enriched with numerical scores:
            [
                {
                    'text': str,
                    'nli_label': str,
                    'nli_confidence': float,
                    'nli_score': float,  # NEW: numerical score
                    'nli_probs': dict,
                    ...
                },
                ...
            ]
    """
    if not nli_results:
        logger.warning("No NLI results provided for scoring.")
        return []
    
    scored_evidence = []
    
    for item in nli_results:
        # Extract NLI information
        label = item.get('nli_label', 'NEUTRAL')
        confidence = item.get('nli_confidence', 0.0)
        
        # Calculate numerical score
        score = label_to_score(label, confidence)
        
        # Add score to evidence
        new_item = item.copy()
        new_item['nli_score'] = score
        
        scored_evidence.append(new_item)
        
        logger.debug(f"Evidence: {item['text'][:30]}... | {label} ({confidence:.2f}) → Score: {score:+.3f}")
    
    return scored_evidence


def aggregate_scores(scored_evidence: List[Dict[str, Any]], method: str = 'weighted_sum') -> Dict[str, Any]:
    """
    Aggregate individual evidence scores into overall statistics
    
    Args:
        scored_evidence (list): Evidence with nli_score from calculate_evidence_scores
        method (str): Aggregation method
            - 'weighted_sum': Sum all scores weighted by combined_score (default)
            - 'simple_sum': Simple sum of all nli_scores
            - 'average': Average of all nli_scores
            
    Returns:
        dict: Aggregated statistics:
            {
                'total_score': float,        # Overall score (-N to +N)
                'normalized_score': float,   # Normalized to [-1, 1]
                'num_evidence': int,         # Total evidence count
                'num_support': int,          # Supporting evidence count
                'num_refute': int,           # Refuting evidence count
                'num_neutral': int,          # Neutral evidence count
                'support_score': float,      # Sum of supporting scores
                'refute_score': float,       # Sum of refuting scores (negative)
                'method': str                # Aggregation method used
            }
    """
    if not scored_evidence:
        logger.warning("No scored evidence provided for aggregation.")
        return {
            'total_score': 0.0,
            'normalized_score': 0.0,
            'num_evidence': 0,
            'num_support': 0,
            'num_refute': 0,
            'num_neutral': 0,
            'support_score': 0.0,
            'refute_score': 0.0,
            'method': method
        }
    
    num_support = 0
    num_refute = 0
    num_neutral = 0
    support_score = 0.0
    refute_score = 0.0
    
    # Calculate aggregated score based on method
    if method == 'weighted_sum':
        # Weight each evidence by its retrieval score (combined_score)
        total_score = 0.0
        total_weight = 0.0
        
        for item in scored_evidence:
            nli_score = item.get('nli_score', 0.0)
            weight = item.get('combined_score', 1.0)  # Use retrieval score as weight
            
            total_score += nli_score * weight
            total_weight += weight
            
            # Count by label
            label = item.get('nli_label', 'NEUTRAL').upper()
            if label == 'SUPPORT':
                num_support += 1
                support_score += nli_score
            elif label == 'REFUTE':
                num_refute += 1
                refute_score += nli_score
            else:
                num_neutral += 1
        
        # Normalize by total weight
        if total_weight > 0:
            total_score = total_score / total_weight
    
    elif method == 'simple_sum':
        # Simple sum of all scores
        total_score = 0.0
        
        for item in scored_evidence:
            nli_score = item.get('nli_score', 0.0)
            total_score += nli_score
            
            label = item.get('nli_label', 'NEUTRAL').upper()
            if label == 'SUPPORT':
                num_support += 1
                support_score += nli_score
            elif label == 'REFUTE':
                num_refute += 1
                refute_score += nli_score
            else:
                num_neutral += 1
    
    elif method == 'average':
        # Average of all scores
        total_score = 0.0
        
        for item in scored_evidence:
            nli_score = item.get('nli_score', 0.0)
            total_score += nli_score
            
            label = item.get('nli_label', 'NEUTRAL').upper()
            if label == 'SUPPORT':
                num_support += 1
                support_score += nli_score
            elif label == 'REFUTE':
                num_refute += 1
                refute_score += nli_score
            else:
                num_neutral += 1
        
        total_score = total_score / len(scored_evidence)
    
    else:
        logger.error(f"Unknown aggregation method: {method}. Using simple_sum.")
        return aggregate_scores(scored_evidence, method='simple_sum')
    
    # Normalize total score to [-1, 1] range for consistent interpretation
    # For weighted_sum and average, score is already roughly normalized
    # For simple_sum, normalize by number of evidence
    if method == 'simple_sum':
        normalized_score = total_score / len(scored_evidence) if scored_evidence else 0.0
    else:
        normalized_score = max(-1.0, min(1.0, total_score))  # Clamp to [-1, 1]
    
    result = {
        'total_score': total_score,
        'normalized_score': normalized_score,
        'num_evidence': len(scored_evidence),
        'num_support': num_support,
        'num_refute': num_refute,
        'num_neutral': num_neutral,
        'support_score': support_score,
        'refute_score': refute_score,
        'method': method
    }
    
    logger.info(f"Aggregation ({method}): {num_support} SUPPORT, {num_refute} REFUTE, {num_neutral} NEUTRAL → Score: {total_score:+.3f} (normalized: {normalized_score:+.3f})")
    
    return result


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample NLI results
    sample_nli_results = [
        {
            'text': 'Vietnam is the second largest coffee exporter in the world.',
            'nli_label': 'SUPPORT',
            'nli_confidence': 0.92,
            'combined_score': 0.85
        },
        {
            'text': 'Coffee exports from Vietnam have grown significantly.',
            'nli_label': 'SUPPORT',
            'nli_confidence': 0.78,
            'combined_score': 0.72
        },
        {
            'text': 'Brazil is the largest coffee producer globally.',
            'nli_label': 'NEUTRAL',
            'nli_confidence': 0.65,
            'combined_score': 0.60
        },
        {
            'text': 'Vietnam ranks third in coffee production after Brazil and Colombia.',
            'nli_label': 'REFUTE',
            'nli_confidence': 0.71,
            'combined_score': 0.55
        }
    ]
    
    print("="*70)
    print("SCORING MODULE - Example")
    print("="*70)
    print()
    
    # Calculate scores
    print("[Step 1] Converting NLI labels to numerical scores...")
    scored = calculate_evidence_scores(sample_nli_results)
    print()
    
    for item in scored:
        print(f"  {item['nli_label']:8s} ({item['nli_confidence']:.2f}) → {item['nli_score']:+.3f} | {item['text'][:50]}")
    print()
    
    # Aggregate scores
    print("[Step 2] Aggregating scores...")
    print()
    
    for method in ['weighted_sum', 'simple_sum', 'average']:
        print(f"\nMethod: {method}")
        result = aggregate_scores(scored, method=method)
        print(f"  Total Score: {result['total_score']:+.3f}")
        print(f"  Normalized:  {result['normalized_score']:+.3f}")
        print(f"  Evidence: {result['num_support']} SUPPORT, {result['num_refute']} REFUTE, {result['num_neutral']} NEUTRAL")
