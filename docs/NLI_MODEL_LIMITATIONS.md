# NLI Model Limitations and Known Issues

## Summary

This document describes the limitations of current NLI models used in the fact-checking pipeline, including model selection issues and inherent accuracy constraints.

---

## 1. Model Selection Issues

### ❌ DeBERTa-v3-large-zeroshot (BROKEN - DO NOT USE)

**Model**: `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`

**Problem**: This is a **zero-shot classification model**, NOT a standard 3-class NLI model.

**Symptoms**:
- Outputs only **2 labels** (relevant/not-relevant) instead of 3 (entailment/neutral/contradiction)
- Causes error: `index 2 is out of bounds for axis 0 with size 2`
- All evidence defaults to NEUTRAL when inference fails

**Root Cause**: Zero-shot models use a different architecture optimized for binary classification, not NLI's 3-way classification.

**Status**: ❌ **DO NOT USE** - Incompatible with our pipeline

---

### ❌ microsoft/deberta-v3-base (TOKENIZER ISSUE)

**Model**: `microsoft/deberta-v3-base`

**Problem**: Tokenizer conversion failure with fast tokenizers.

**Error**:
```
Converting from SentencePiece and Tiktoken failed, if a converter for 
SentencePiece is available, provide a model path with a SentencePiece 
tokenizer.model file.
```

**Root Cause**: The model uses SentencePiece tokenizer which requires special conversion that fails in some environments.

**Status**: ❌ **DO NOT USE** - Tokenizer incompatibility

---

### ✅ FacebookAI/roberta-large-mnli (RECOMMENDED)

**Model**: `FacebookAI/roberta-large-mnli`

**Status**: ✅ **WORKS CORRECTLY**

**Pros**:
- Standard 3-class NLI model (entailment/neutral/contradiction)
- Stable tokenizer (RoBERTa tokenizer)
- ~90% accuracy on MNLI benchmark
- Compatible with ONNX optimization
- Compatible with INT8 quantization

**Cons**:
- Slower than base models (~5-8s CPU, ~0.5s GPU)
- 355M parameters (larger memory footprint)
- See "Inherent NLI Limitations" below

**Label Mapping**:
```python
0 = REFUTE (contradiction)
1 = NEUTRAL
2 = SUPPORT (entailment)
```

**Performance**:
- CPU (no optimization): ~10-12s per batch
- CPU + INT8 quantization: ~3-5s per batch (2-3x speedup, ~1-2% accuracy loss)
- CPU + ONNX: ~5-7s per batch (2x speedup)
- GPU: ~0.3-0.5s per batch (20x speedup)

---

## 2. Inherent NLI Model Limitations

Even with a working model like RoBERTa-large-mnli, NLI models have fundamental limitations:

### ❌ Poor Numerical Reasoning

**Problem**: Models cannot accurately detect numerical mismatches.

**Example**:
- **Claim**: "Donald Trump is the 46th president of the U.S."
- **Evidence**: "Donald Trump is the 45th and 47th president" (2017-21; 2025-)
- **Model Prediction**: SUPPORT (79.4% confidence) ❌
- **Correct Answer**: REFUTE (46th ≠ 45th/47th) ✓

**Why**: The model focuses on semantic similarity ("Donald Trump" + "president" + "U.S.") and ignores the specific number mismatch (46th vs 45th/47th).

### ❌ Weak Negation Detection

**Problem**: Models struggle with negative statements.

**Example**:
- **Claim**: "The vaccine is safe"
- **Evidence**: "The vaccine is **not** safe according to some reports"
- **Model Prediction**: SUPPORT (high confidence on keyword overlap) ❌
- **Correct Answer**: REFUTE (direct negation) ✓

### ❌ Context-Independent Inference

**Problem**: Models don't understand temporal context or world knowledge.

**Example**:
- **Claim**: "Biden is the current president" (checked in 2025)
- **Evidence**: "Trump is the current president" (written in 2025, Trump's second term)
- **Model Prediction**: REFUTE (name mismatch) ❌
- **Correct Answer**: Depends on when "current" refers to ✓

---

## 3. Recommended Solutions

### For Production Systems:

1. **Use FEVER-trained models** instead of general NLI:
   - `fever-nli` or `roberta-large-fever` 
   - Trained specifically on fact-checking data
   - Better numerical and negation handling

2. **Add rule-based numerical verification**:
   ```python
   def verify_numbers(claim, evidence):
       claim_nums = extract_numbers(claim)
       evidence_nums = extract_numbers(evidence)
       if claim_nums and evidence_nums:
           if not numbers_match(claim_nums, evidence_nums):
               return "REFUTE"
       return nli_model.predict(claim, evidence)
   ```

3. **Ensemble multiple models**:
   - Combine RoBERTa + numerical checker + entity matcher
   - Vote on final verdict

### For This Project (CS419):

**Current Configuration** (Best Balance):
```powershell
# Windows
$env:NLI_MODEL_NAME='FacebookAI/roberta-large-mnli'
$env:NLI_USE_QUANTIZATION='true'  # 2-3x speedup, ~1% accuracy loss
$env:NLI_USE_ONNX='false'         # Disable to avoid conversion issues

# Linux/Mac
export NLI_MODEL_NAME="FacebookAI/roberta-large-mnli"
export NLI_USE_QUANTIZATION="true"
export NLI_USE_ONNX="false"
```

**Performance**: ~3-5s per batch (12 sentences) on CPU with INT8 quantization

---

## 4. INT8 Quantization Impact

**Question**: Does INT8 quantization affect accuracy?

**Answer**: Yes, but minimally.

**Accuracy Impact**:
- ~1-2% accuracy drop on benchmark datasets
- Negligible for most real-world applications
- Still maintains ~88-89% accuracy (vs ~90% baseline)

**Speed Gain**:
- **2-3x faster** on CPU (10s → 3-5s)
- **75% memory reduction** (1.4GB → 350MB)
- **No speedup on GPU** (GPU is already fast)

**Recommendation**: ✅ **Enable INT8 on CPU**, disable on GPU

---

## 5. ONNX Runtime Issues

**Current Status**: ONNX conversion works but can be unstable.

**Known Issues**:
1. **Conversion time**: First run takes ~2 minutes to convert model
2. **Cache loading**: ~30-40s to load cached ONNX model (vs 5-8s PyTorch)
3. **Label count issues**: Some models export with wrong number of labels
4. **No `.eval()` method**: ONNX models don't have PyTorch methods

**When to Use ONNX**:
- ✅ GPU inference (CUDAExecutionProvider): ~2x speedup
- ✅ Production deployment (smaller model size)
- ❌ CPU inference: PyTorch + INT8 quantization is faster
- ❌ Development: Too slow to load, adds debugging complexity

**Recommendation**: **Disable ONNX for CPU**, only use for GPU deployment

---

## 6. Summary Table

| Model | Status | Accuracy | CPU Speed | GPU Speed | Issues |
|-------|--------|----------|-----------|-----------|--------|
| **RoBERTa-large-mnli** | ✅ Recommended | ~90% | 10s → 3-5s (INT8) | 0.5s | Weak numerical reasoning |
| DeBERTa-v3-large-zeroshot | ❌ Broken | N/A | N/A | N/A | Only 2 labels (not NLI) |
| DeBERTa-v3-base | ❌ Broken | N/A | N/A | N/A | Tokenizer conversion fails |
| distilroberta-base-v2 | ⚠️ Low accuracy | ~75% | 2s | 0.2s | Too inaccurate for fact-checking |

---

## 7. For CS419 Report

**Mention in Limitations Section**:

1. **NLI Model Selection Challenge**:
   - Tested 3 models (DeBERTa-large-zeroshot, DeBERTa-base, RoBERTa-large-mnli)
   - 2 models incompatible (zero-shot architecture, tokenizer issues)
   - Final choice: RoBERTa-large-mnli (stable, accurate, well-supported)

2. **Inherent NLI Limitations**:
   - **Numerical reasoning**: Cannot detect number mismatches (46th vs 45th)
   - **Negation detection**: Struggles with "not", "no", "never"
   - **Semantic-only**: Focuses on word overlap, ignores precise meaning

3. **Optimization Trade-offs**:
   - INT8 quantization: 2-3x speedup, ~1-2% accuracy loss (acceptable)
   - ONNX Runtime: Unstable on some models, slow cache loading
   - Final config: PyTorch + INT8 quantization (best balance)

4. **Real-World Impact**:
   - Example: "Trump is 46th president" vs "Trump is 45th/47th president"
   - Model incorrectly labels as SUPPORT (focuses on name/title match)
   - Would need rule-based numerical verification layer for production

---

## 8. Future Improvements

For a production fact-checking system:

1. **Use FEVER-specific models**: `roberta-large-fever` or `bart-large-fever`
2. **Add numerical verification**: Rule-based number extraction and comparison
3. **Entity resolution**: Use knowledge graphs (Wikidata) for entity verification
4. **Multi-model ensemble**: Combine NLI + numerical + entity matchers
5. **Fine-tune on domain data**: Train on fact-checking datasets (FEVER, LIAR, etc.)

---

**Last Updated**: December 9, 2025  
**Current Recommended Model**: FacebookAI/roberta-large-mnli + INT8 quantization
