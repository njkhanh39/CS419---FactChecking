"""Final Decision Module: Make ultimate verdict for fact-checking

This module combines scoring and voting to produce the final fact-checking verdict.
It applies decision rules and thresholds to determine if a claim is SUPPORTED, REFUTED,
or if there's NOT ENOUGH INFO.

Decision Strategy:
1. Calculate numerical scores (from scoring.py)
2. Apply voting (from voting.py)
3. Combine both methods with decision thresholds
4. Generate comprehensive verdict with evidence summary

Usage:
    from src.aggregation.final_decision import make_final_decision
    
    verdict = make_final_decision(nli_results)
"""

import logging
from typing import List, Dict, Any, Optional
from .scoring import calculate_evidence_scores, aggregate_scores
from .voting import majority_vote, weighted_vote

logger = logging.getLogger(__name__)


def make_final_decision(
    nli_results: List[Dict[str, Any]],
    method: str = 'hybrid',
    score_threshold: float = 0.3,
    confidence_threshold: float = 0.6,
    min_evidence: int = 3
) -> Dict[str, Any]:
    """
    Make final fact-checking decision
    
    Args:
        nli_results (list): Evidence enriched with NLI labels from batch_inference
        method (str): Decision method
            - 'scoring': Use numerical scoring only
            - 'voting': Use majority voting only
            - 'hybrid': Combine both (default, most robust)
        score_threshold (float): Minimum normalized score for definitive verdict (default: 0.3)
        confidence_threshold (float): Minimum confidence for definitive verdict (default: 0.6)
        min_evidence (int): Minimum evidence required (default: 3)
        
    Returns:
        dict: Final verdict:
            {
                'verdict': str,              # "SUPPORTED", "REFUTED", "NOT ENOUGH INFO"
                'confidence': float,         # Overall confidence (0-1)
                'claim': str,                # Original claim (if available)
                'evidence_summary': {
                    'total': int,
                    'supporting': int,
                    'refuting': int,
                    'neutral': int
                },
                'scores': {
                    'total_score': float,
                    'normalized_score': float,
                    'support_score': float,
                    'refute_score': float
                },
                'voting': {
                    'verdict': str,
                    'confidence': float,
                    'percentages': dict
                },
                'top_evidence': list,        # Top 3-5 most relevant evidence
                'method': str,               # Decision method used
                'explanation': str           # Human-readable explanation
            }
    """
    if not nli_results:
        logger.warning("No NLI results provided for decision making.")
        return {
            'verdict': 'NOT ENOUGH INFO',
            'confidence': 0.0,
            'evidence_summary': {'total': 0, 'supporting': 0, 'refuting': 0, 'neutral': 0},
            'scores': {'total_score': 0.0, 'normalized_score': 0.0, 'support_score': 0.0, 'refute_score': 0.0},
            'voting': {'verdict': 'NOT ENOUGH INFO', 'confidence': 0.0, 'percentages': {}},
            'top_evidence': [],
            'method': method,
            'explanation': 'No evidence available for fact-checking.'
        }
    
    logger.info(f"Making final decision using '{method}' method on {len(nli_results)} evidence sentences")
    
    # Extract claim if available
    claim = nli_results[0].get('claim', 'Unknown claim')
    
    # ========== STEP 1: Calculate Scores ==========
    scored_evidence = calculate_evidence_scores(nli_results)
    aggregated_scores = aggregate_scores(scored_evidence, method='weighted_sum')
    
    # ========== STEP 2: Perform Voting ==========
    vote_result = weighted_vote(nli_results, min_evidence=min_evidence)
    
    # ========== STEP 3: Make Decision Based on Method ==========
    if method == 'scoring':
        # Decision based purely on numerical scores
        normalized_score = aggregated_scores['normalized_score']
        
        if aggregated_scores['num_evidence'] < min_evidence:
            verdict = 'NOT ENOUGH INFO'
            confidence = 0.0
            explanation = f"Insufficient evidence: only {aggregated_scores['num_evidence']} sentences found (minimum: {min_evidence})"
        
        elif normalized_score > score_threshold:
            verdict = 'SUPPORTED'
            confidence = min(1.0, (normalized_score / 1.0))  # Scale to 0-1
            explanation = f"Positive score ({normalized_score:+.3f}) indicates claim is supported by evidence"
        
        elif normalized_score < -score_threshold:
            verdict = 'REFUTED'
            confidence = min(1.0, abs(normalized_score / 1.0))
            explanation = f"Negative score ({normalized_score:+.3f}) indicates claim is refuted by evidence"
        
        else:
            verdict = 'NOT ENOUGH INFO'
            confidence = 0.5
            explanation = f"Score ({normalized_score:+.3f}) is inconclusive (threshold: ±{score_threshold})"
    
    elif method == 'voting':
        # Decision based purely on voting
        verdict = vote_result['verdict']
        confidence = vote_result['confidence']
        
        if verdict == 'SUPPORTED':
            explanation = f"Majority vote: {vote_result['vote_percentages']['support']:.1%} supporting evidence"
        elif verdict == 'REFUTED':
            explanation = f"Majority vote: {vote_result['vote_percentages']['refute']:.1%} refuting evidence"
        else:
            explanation = f"No clear majority or insufficient evidence"
    
    elif method == 'hybrid':
        # Hybrid: Combine scoring and voting (most robust)
        normalized_score = aggregated_scores['normalized_score']
        vote_verdict = vote_result['verdict']
        vote_confidence = vote_result['confidence']
        
        # Both methods must agree for high confidence verdict
        if aggregated_scores['num_evidence'] < min_evidence:
            verdict = 'NOT ENOUGH INFO'
            confidence = 0.0
            explanation = f"Insufficient evidence: {aggregated_scores['num_evidence']} < {min_evidence}"
        
        elif vote_verdict == 'SUPPORTED' and normalized_score > score_threshold:
            # Both agree on SUPPORTED
            verdict = 'SUPPORTED'
            confidence = (vote_confidence + min(1.0, normalized_score)) / 2  # Average confidence
            explanation = f"Both scoring ({normalized_score:+.3f}) and voting ({vote_confidence:.1%}) support the claim"
        
        elif vote_verdict == 'REFUTED' and normalized_score < -score_threshold:
            # Both agree on REFUTED
            verdict = 'REFUTED'
            confidence = (vote_confidence + min(1.0, abs(normalized_score))) / 2
            explanation = f"Both scoring ({normalized_score:+.3f}) and voting ({vote_confidence:.1%}) refute the claim"
        
        elif vote_verdict in ['SUPPORTED', 'REFUTED'] and vote_confidence > confidence_threshold:
            # Trust voting with high confidence even if score is weak
            verdict = vote_verdict
            confidence = vote_confidence * 0.8  # Reduce confidence slightly for disagreement
            explanation = f"Strong voting evidence ({vote_confidence:.1%}) overrides weak score ({normalized_score:+.3f})"
        
        elif abs(normalized_score) > score_threshold and abs(normalized_score) > 0.5:
            # Trust strong score even if voting is unclear
            verdict = 'SUPPORTED' if normalized_score > 0 else 'REFUTED'
            confidence = min(1.0, abs(normalized_score)) * 0.8
            explanation = f"Strong score ({normalized_score:+.3f}) overrides unclear voting"
        
        else:
            # Methods disagree or both weak
            verdict = 'NOT ENOUGH INFO'
            confidence = 0.5
            explanation = f"Methods disagree: score={normalized_score:+.3f}, vote={vote_verdict} ({vote_confidence:.1%})"
    
    else:
        logger.error(f"Unknown decision method: {method}. Using hybrid.")
        return make_final_decision(nli_results, method='hybrid', score_threshold=score_threshold,
                                  confidence_threshold=confidence_threshold, min_evidence=min_evidence)
    
    # ========== STEP 4: Select Top Evidence ==========
    # Sort by combined score (retrieval) and NLI confidence
    sorted_evidence = sorted(
        scored_evidence,
        key=lambda x: x.get('combined_score', 0) * x.get('nli_confidence', 0),
        reverse=True
    )
    
    top_evidence = []
    for item in sorted_evidence[:5]:  # Top 5
        top_evidence.append({
            'text': item['text'],
            'label': item['nli_label'],
            'confidence': item['nli_confidence'],
            'retrieval_score': item.get('combined_score', 0),
            'source': item.get('doc_url', 'Unknown')
        })
    
    # ========== STEP 5: Compile Final Result ==========
    result = {
        'verdict': verdict,
        'confidence': confidence,
        'claim': claim,
        'evidence_summary': {
            'total': aggregated_scores['num_evidence'],
            'supporting': aggregated_scores['num_support'],
            'refuting': aggregated_scores['num_refute'],
            'neutral': aggregated_scores['num_neutral']
        },
        'scores': {
            'total_score': aggregated_scores['total_score'],
            'normalized_score': aggregated_scores['normalized_score'],
            'support_score': aggregated_scores['support_score'],
            'refute_score': aggregated_scores['refute_score']
        },
        'voting': {
            'verdict': vote_result['verdict'],
            'confidence': vote_result['confidence'],
            'percentages': vote_result['vote_percentages']
        },
        'top_evidence': top_evidence,
        'method': method,
        'explanation': explanation
    }
    
    logger.info(f"Final Decision: {verdict} (confidence: {confidence:.2%})")
    logger.info(f"Explanation: {explanation}")
    
    return result


def display_verdict(verdict_result: Dict[str, Any], verbose: bool = True):
    """
    Display verdict in human-readable format
    
    Args:
        verdict_result (dict): Result from make_final_decision
        verbose (bool): Show detailed breakdown
    """
    print("\n" + "="*70)
    print("FACT-CHECKING VERDICT")
    print("="*70)
    
    # Verdict
    verdict = verdict_result['verdict']
    confidence = verdict_result['confidence']
    
    verdict_emoji = {
        'SUPPORTED': '✓',
        'REFUTED': '✗',
        'NOT ENOUGH INFO': '?'
    }
    
    print(f"\n{verdict_emoji.get(verdict, '?')} VERDICT: {verdict}")
    print(f"   Confidence: {confidence:.1%}")
    print(f"\n   {verdict_result['explanation']}")
    
    if verbose:
        # Evidence summary
        summary = verdict_result['evidence_summary']
        print(f"\n{'─'*70}")
        print("EVIDENCE SUMMARY")
        print(f"{'─'*70}")
        print(f"   Total evidence: {summary['total']}")
        print(f"   ✓ Supporting:   {summary['supporting']} ({summary['supporting']/summary['total']:.1%})")
        print(f"   ✗ Refuting:     {summary['refuting']} ({summary['refuting']/summary['total']:.1%})")
        print(f"   ○ Neutral:      {summary['neutral']} ({summary['neutral']/summary['total']:.1%})")
        
        # Scores
        scores = verdict_result['scores']
        print(f"\n{'─'*70}")
        print("SCORING")
        print(f"{'─'*70}")
        print(f"   Normalized Score: {scores['normalized_score']:+.3f}")
        print(f"   Support Score:    {scores['support_score']:+.3f}")
        print(f"   Refute Score:     {scores['refute_score']:+.3f}")
        
        # Voting
        voting = verdict_result['voting']
        print(f"\n{'─'*70}")
        print("VOTING")
        print(f"{'─'*70}")
        print(f"   Vote Result: {voting['verdict']} ({voting['confidence']:.1%})")
        print(f"   Support: {voting['percentages']['support']:.1%}")
        print(f"   Refute:  {voting['percentages']['refute']:.1%}")
        print(f"   Neutral: {voting['percentages']['neutral']:.1%}")
        
        # Top evidence
        print(f"\n{'─'*70}")
        print("TOP EVIDENCE")
        print(f"{'─'*70}")
        for i, evidence in enumerate(verdict_result['top_evidence'][:3], 1):
            print(f"\n   {i}. [{evidence['label']}] Confidence: {evidence['confidence']:.2f}")
            print(f"      {evidence['text'][:100]}...")
            print(f"      Source: {evidence['source'][:50]}...")
    
    print(f"\n{'='*70}\n")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample NLI results (from Phase 3)
    sample_nli_results = [
        {
            'text': 'Vietnam is the second largest coffee exporter in the world.',
            'nli_label': 'SUPPORT',
            'nli_confidence': 0.92,
            'combined_score': 0.85,
            'doc_url': 'https://example.com/coffee-exports'
        },
        {
            'text': 'Coffee exports from Vietnam have grown significantly over the years.',
            'nli_label': 'SUPPORT',
            'nli_confidence': 0.78,
            'combined_score': 0.72,
            'doc_url': 'https://example.com/vietnam-coffee'
        },
        {
            'text': 'Brazil is the largest coffee producer globally.',
            'nli_label': 'NEUTRAL',
            'nli_confidence': 0.65,
            'combined_score': 0.60,
            'doc_url': 'https://example.com/brazil-coffee'
        },
        {
            'text': 'Vietnam ranks third in coffee production after Brazil and Colombia.',
            'nli_label': 'REFUTE',
            'nli_confidence': 0.71,
            'combined_score': 0.55,
            'doc_url': 'https://example.com/coffee-rankings'
        },
        {
            'text': 'Vietnam coffee production increased 200% in the last decade.',
            'nli_label': 'SUPPORT',
            'nli_confidence': 0.85,
            'combined_score': 0.68,
            'doc_url': 'https://example.com/coffee-growth'
        },
        {
            'text': 'Vietnamese coffee is exported to over 80 countries worldwide.',
            'nli_label': 'SUPPORT',
            'nli_confidence': 0.80,
            'combined_score': 0.65,
            'doc_url': 'https://example.com/export-markets'
        }
    ]
    
    print("\n" + "="*70)
    print("FINAL DECISION MODULE - Example")
    print("="*70)
    
    # Test all three methods
    for method in ['scoring', 'voting', 'hybrid']:
        print(f"\n\n{'#'*70}")
        print(f"# Method: {method.upper()}")
        print(f"{'#'*70}")
        
        result = make_final_decision(sample_nli_results, method=method)
        display_verdict(result, verbose=True)
