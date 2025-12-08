# Aggregation Module Documentation

## Overview

The aggregation module is **Phase 4** of the fact-checking pipeline. It takes NLI results (Phase 3) and produces a final verdict: **SUPPORTED**, **REFUTED**, or **NOT ENOUGH INFO**.

## Architecture

```
Phase 3 (NLI) Output → Phase 4 (Aggregation) → Final Verdict
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              scoring.py          voting.py
                    ↓                   ↓
                    └─────────┬─────────┘
                              ↓
                      final_decision.py
                              ↓
                          VERDICT
```

## Module Components

### 1. scoring.py - Numerical Scoring

Converts categorical NLI labels to numerical scores for mathematical aggregation.

**Key Functions:**
- `label_to_score(label, confidence)` - Convert single label to score
- `calculate_evidence_scores(nli_results)` - Score all evidence
- `aggregate_scores(scored_evidence, method)` - Aggregate into total score

**Scoring Scheme:**
```
SUPPORT  → +confidence  (e.g., +0.92)
REFUTE   → -confidence  (e.g., -0.85)
NEUTRAL  → 0.0
```

**Aggregation Methods:**
- `weighted_sum`: Weight by retrieval score (recommended)
- `simple_sum`: Simple sum of all scores
- `average`: Average of all scores

**Example:**
```python
from src.aggregation.scoring import calculate_evidence_scores, aggregate_scores

# Convert NLI labels to scores
scored = calculate_evidence_scores(nli_results)

# Aggregate
result = aggregate_scores(scored, method='weighted_sum')
print(f"Total score: {result['total_score']:+.3f}")
print(f"Normalized: {result['normalized_score']:+.3f}")
```

### 2. voting.py - Majority Voting

Alternative method using democratic voting instead of scores.

**Key Functions:**
- `majority_vote(nli_results)` - Simple majority voting
- `weighted_vote(nli_results)` - Weighted by confidence × retrieval_score

**Voting Logic:**
1. Count votes for each label (SUPPORT, REFUTE, NEUTRAL)
2. Majority (>50%) wins
3. Tie-breaking: Use average confidence scores
4. Return verdict with confidence percentage

**Example:**
```python
from src.aggregation.voting import majority_vote, weighted_vote

# Simple majority
result = majority_vote(nli_results)
print(f"Verdict: {result['verdict']}")
print(f"Votes: {result['vote_counts']}")

# Weighted voting (recommended)
result = weighted_vote(nli_results)
print(f"Weighted percentages: {result['vote_percentages']}")
```

### 3. final_decision.py - Final Verdict

Combines scoring and voting to produce robust final verdict.

**Key Functions:**
- `make_final_decision(nli_results, method)` - Main decision function
- `display_verdict(verdict_result)` - Pretty-print verdict

**Decision Methods:**
- `scoring`: Use numerical scores only
- `voting`: Use majority vote only
- `hybrid`: Combine both (recommended, most robust)

**Hybrid Logic:**
```
IF both scoring and voting agree:
    → High confidence verdict
ELIF voting has >60% confidence:
    → Trust voting (more robust to outliers)
ELIF |score| > 0.5:
    → Trust strong score
ELSE:
    → NOT ENOUGH INFO (methods disagree)
```

**Example:**
```python
from src.aggregation import make_final_decision, display_verdict

# Make decision
verdict = make_final_decision(nli_results, method='hybrid')

# Display results
display_verdict(verdict, verbose=True)
```

## Input Format

The aggregation module expects NLI results from `src.nli.batch_inference`:

```python
nli_results = [
    {
        'text': 'Evidence sentence...',
        'nli_label': 'SUPPORT',           # or 'REFUTE' or 'NEUTRAL'
        'nli_confidence': 0.92,           # Model confidence (0-1)
        'nli_probs': {
            'entailment': 0.92,           # SUPPORT probability
            'contradiction': 0.05,        # REFUTE probability
            'neutral': 0.03               # NEUTRAL probability
        },
        'combined_score': 0.85,           # Retrieval score (from Phase 2)
        'doc_url': 'https://...',
        'doc_title': '...',
        # ... other metadata
    },
    # ... more evidence
]
```

## Output Format

The final verdict has this structure:

```python
{
    'verdict': 'SUPPORTED',              # or 'REFUTED' or 'NOT ENOUGH INFO'
    'confidence': 0.78,                  # Overall confidence (0-1)
    'claim': 'Original claim text',
    
    'evidence_summary': {
        'total': 6,
        'supporting': 4,
        'refuting': 1,
        'neutral': 1
    },
    
    'scores': {
        'total_score': 0.52,
        'normalized_score': 0.52,
        'support_score': 3.35,
        'refute_score': -0.85
    },
    
    'voting': {
        'verdict': 'SUPPORTED',
        'confidence': 0.67,
        'percentages': {
            'support': 0.67,
            'refute': 0.17,
            'neutral': 0.17
        }
    },
    
    'top_evidence': [
        {
            'text': 'Most relevant evidence...',
            'label': 'SUPPORT',
            'confidence': 0.92,
            'retrieval_score': 0.85,
            'source': 'https://...'
        },
        # ... top 5 evidence
    ],
    
    'method': 'hybrid',
    'explanation': 'Both scoring (+0.52) and voting (67%) support the claim'
}
```

## Integration with Pipeline

### Complete Pipeline Flow

```python
# Phase 0: Data Collection
from src.data_collection import DataCollector
collector = DataCollector()
corpus = collector.collect_corpus(claim, num_urls=20)

# Phase 1 & 2: Retrieval
from src.retrieval import RetrievalOrchestrator
orchestrator = RetrievalOrchestrator()
ranked_evidence = orchestrator.retrieve_and_rank(claim, top_k=12)

# Phase 3: NLI Inference
from src.nli import run_nli_inference
nli_results = run_nli_inference(claim, ranked_evidence)

# Phase 4: Aggregation (NEW!)
from src.aggregation import make_final_decision, display_verdict
verdict = make_final_decision(nli_results, method='hybrid')
display_verdict(verdict)
```

## Configuration & Tuning

### Default Thresholds

```python
# In final_decision.py
score_threshold = 0.3          # Minimum score for definitive verdict
confidence_threshold = 0.6     # Minimum voting confidence
min_evidence = 3               # Minimum evidence required
```

### Tuning Guidelines

**If getting too many "NOT ENOUGH INFO":**
- Lower `score_threshold` (e.g., 0.2)
- Lower `confidence_threshold` (e.g., 0.5)

**If getting false positives:**
- Raise `score_threshold` (e.g., 0.4)
- Raise `confidence_threshold` (e.g., 0.7)
- Increase `min_evidence` (e.g., 5)

**For more conservative verdicts:**
- Use `method='hybrid'` (requires agreement)
- Increase all thresholds

**For more aggressive verdicts:**
- Use `method='scoring'` or `method='voting'`
- Decrease thresholds

## Testing

### Unit Tests

Test individual components:

```bash
# Test scoring
python -m src.aggregation.scoring

# Test voting
python -m src.aggregation.voting

# Test final decision
python -m src.aggregation.final_decision
```

### Integration Test

Test complete NLI + Aggregation:

```bash
python -m tests.test_nli_aggregation
```

This will:
1. Load sample evidence
2. Run NLI inference (loads RoBERTa model)
3. Apply all three aggregation methods
4. Display comparative results

**Note:** First run takes ~60 seconds to download RoBERTa model.

## Consistency with NLI Module

✅ **Your friend's NLI implementation is excellent and fully compatible!**

**What we checked:**
1. ✅ Label format: Uses standard "SUPPORT", "REFUTE", "NEUTRAL"
2. ✅ Confidence scores: Returns 0-1 probabilities
3. ✅ Batch processing: Efficient batch inference
4. ✅ Output format: Enriches evidence with `nli_label`, `nli_confidence`, `nli_probs`
5. ✅ Model: Uses robust RoBERTa-large-MNLI

**Integration points:**
- NLI `run_nli_inference()` output → Aggregation `make_final_decision()` input
- No format conversion needed - direct compatibility!

## Example Usage

### Minimal Example

```python
from src.nli import run_nli_inference
from src.aggregation import make_final_decision

claim = "Vietnam is the second largest coffee exporter"
evidence = [...]  # From retrieval

# NLI
nli_results = run_nli_inference(claim, evidence)

# Aggregation
verdict = make_final_decision(nli_results)

print(f"Verdict: {verdict['verdict']}")
print(f"Confidence: {verdict['confidence']:.1%}")
```

### Complete Example

See `tests/test_nli_aggregation.py` for a full working example.

## Troubleshooting

### "No NLI results provided"
- Check that NLI inference ran successfully
- Verify `nli_results` is not empty

### "Unknown NLI label"
- Check label format is "SUPPORT", "REFUTE", or "NEUTRAL" (uppercase)
- NLI model should return these standard labels

### Low confidence verdicts
- May need more evidence (increase retrieval `top_k`)
- Check NLI model confidence scores
- Consider lowering decision thresholds

### Methods disagree
- Normal for borderline cases
- Review top evidence to understand disagreement
- May need to tune weights or thresholds

## Next Steps

1. ✅ **NLI Module** - Implemented by your friend
2. ✅ **Aggregation Module** - Just implemented
3. ⏭️ **End-to-End Pipeline** - Connect all phases
4. ⏭️ **Evaluation** - Test on benchmark dataset
5. ⏭️ **Tuning** - Optimize thresholds and weights

## References

- Implementation Guide: `IMPLEMENTATION_GUIDE.md`
- Architecture: `architecture.txt`
- NLI Module: `src/nli/help.txt`
- Test Script: `tests/test_nli_aggregation.py`
