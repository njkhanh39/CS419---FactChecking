# Quick Guide: Switch to Faster NLI Model

## TL;DR - Make NLI 3-5x Faster

**Current:** `FacebookAI/roberta-large-mnli` (355M params, ~24s for Phase 3)  
**Faster:** `cross-encoder/nli-distilroberta-base` (82M params, ~5-8s for Phase 3)

## Step-by-Step Instructions

### Option 1: Quick Fix (Change One Line)

1. **Open file:** `src/nli/batch_inference.py`

2. **Find line 12-13:**
```python
def get_model_instance():
    global _NLI_MODEL_INSTANCE
    if _NLI_MODEL_INSTANCE is None:
        # UPDATED: Use the correct official ID
        _NLI_MODEL_INSTANCE = NLIModel(model_name="FacebookAI/roberta-large-mnli")
    return _NLI_MODEL_INSTANCE
```

3. **Change to:**
```python
def get_model_instance():
    global _NLI_MODEL_INSTANCE
    if _NLI_MODEL_INSTANCE is None:
        # FASTER MODEL: 3-5x speedup, slight accuracy trade-off
        _NLI_MODEL_INSTANCE = NLIModel(model_name="cross-encoder/nli-distilroberta-base")
    return _NLI_MODEL_INSTANCE
```

4. **Restart API:**
```bash
# Stop current API (Ctrl+C)
# Start again
python -m src.api.api
```

5. **First run will download new model** (~330MB, one-time)

### Option 2: Make it Configurable

**Better approach:** Let users choose model via config

1. **Create config file:** `src/config/model_config.py`
```python
"""Model configuration for fact-checking pipeline"""

# NLI Model Selection
# Options:
#   "FacebookAI/roberta-large-mnli" - Most accurate, slowest (355M params)
#   "cross-encoder/nli-distilroberta-base" - Balanced, 3-5x faster (82M params)
#   "cross-encoder/nli-MiniLM2-L6-H768" - Fastest, 10x faster, less accurate (22M params)

NLI_MODEL_NAME = "cross-encoder/nli-distilroberta-base"  # Recommended default

# You can override via environment variable:
# export NLI_MODEL_NAME="FacebookAI/roberta-large-mnli"
```

2. **Update `batch_inference.py`:**
```python
from src.config.model_config import NLI_MODEL_NAME
import os

def get_model_instance():
    global _NLI_MODEL_INSTANCE
    if _NLI_MODEL_INSTANCE is None:
        model_name = os.getenv("NLI_MODEL_NAME", NLI_MODEL_NAME)
        _NLI_MODEL_INSTANCE = NLIModel(model_name=model_name)
    return _NLI_MODEL_INSTANCE
```

3. **Now you can switch models without code changes:**
```bash
# Use fast model (default)
python -m src.api.api

# Or use accurate model
export NLI_MODEL_NAME="FacebookAI/roberta-large-mnli"
python -m src.api.api
```

## Model Comparison

| Model | Params | Speed | Accuracy | Size | Recommendation |
|-------|--------|-------|----------|------|----------------|
| roberta-large-mnli | 355M | 1x (baseline) | 91% | 1.4GB | Production (accuracy critical) |
| **nli-distilroberta-base** | **82M** | **3-5x** | **88-89%** | **330MB** | **✅ Recommended (best balance)** |
| nli-MiniLM2-L6-H768 | 22M | 10x | 84-85% | 90MB | Development/Demo (speed critical) |

## Testing After Switch

```bash
# Test with a claim
streamlit run frontend/frontend.py

# Check timing in Performance Metrics
# Phase 3 should be ~5-8s instead of ~24s
```

## Troubleshooting

### "Model not found" error
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/transformers/*
python -m src.api.api
```

### Labels are wrong
Some models use different label mappings. If you see incorrect results:

1. **Check model card:** https://huggingface.co/cross-encoder/nli-distilroberta-base

2. **Update label mapping in `nli_model.py`:**
```python
# Standard mapping (most models)
self.label_map = {0: "REFUTE", 1: "NEUTRAL", 2: "SUPPORT"}

# If labels are swapped, try:
self.label_map = {0: "NEUTRAL", 1: "REFUTE", 2: "SUPPORT"}
# or
self.label_map = {0: "SUPPORT", 1: "NEUTRAL", 2: "REFUTE"}
```

3. **Test with known claim:**
```python
claim = "Paris is the capital of France"
# Should be SUPPORTED
```

## Rollback

If faster model has issues:

1. **Revert change in `batch_inference.py`:**
```python
_NLI_MODEL_INSTANCE = NLIModel(model_name="FacebookAI/roberta-large-mnli")
```

2. **Restart API**

## Next Steps

After switching to faster model:
1. Test accuracy on your evaluation dataset
2. If accuracy is acceptable, keep it
3. If not, consider:
   - Using GPU (10x speedup with large model)
   - Hybrid approach (fast model for screening, large model for final verdict)
   - Ensemble (combine both models' predictions)
