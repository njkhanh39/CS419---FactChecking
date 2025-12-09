# End-to-End Pipeline - Quick Start Guide

## 🎯 Overview

The complete fact-checking pipeline is now ready! You can check any claim with a single command.

## 🚀 Usage

### Method 1: Interactive Mode (Recommended for Testing)

```bash
cd "e:\File\Code\Stuff Files\CS419 - IR\CS419---FactChecking"
python -m src.pipeline.fact_check
```

Then type your claims:
```
📝 Enter claim to fact-check: Vietnam is the second largest coffee exporter
[Processing...]
✓ VERDICT: SUPPORTED (78% confidence)
```

### Method 2: Single Claim (Command Line)

```bash
python -m src.pipeline.fact_check --claim "Vietnam is the second largest coffee exporter"
```

### Method 3: Python Code

```python
from src.pipeline import FactChecker

# Initialize
checker = FactChecker()

# Check a claim
result = checker.check_claim("Vietnam is the second largest coffee exporter")

# Display result
checker.display_result(result)

# Access result data
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']:.1%}")
```

## ⚙️ Configuration Options

### Basic Options

```bash
# Collect more URLs for better evidence
python -m src.pipeline.fact_check --claim "..." --urls 20

# Retrieve more sentences
python -m src.pipeline.fact_check --claim "..." --top-k 15

# Change aggregation method
python -m src.pipeline.fact_check --claim "..." --method voting  # or scoring, hybrid

# Quiet mode (less output)
python -m src.pipeline.fact_check --claim "..." --quiet
```

### Performance Optimization

**For GPU users** (10-20x faster NLI):
```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional: ONNX for maximum GPU speed
pip install onnxruntime-gpu optimum[onnxruntime-gpu]
export NLI_USE_ONNX="true"

# Run normally - GPU auto-detected
python -m src.pipeline.fact_check --claim "..."
```

**For CPU users** (2-5x faster):
```bash
# Option 1: ONNX Runtime (2-3x faster)
pip install optimum[onnxruntime]
export NLI_USE_ONNX="true"

# Option 2: INT8 Quantization (2-3x faster, 75% less memory)
export NLI_USE_QUANTIZATION="true"

# Option 3: Both (3-5x faster - best for CPU)
export NLI_USE_ONNX="true"
export ONNX_QUANTIZE="true"

python -m src.pipeline.fact_check --claim "..."
```

**Model selection** (default: DeBERTa-v3-large):
```bash
# Best accuracy (default)
export NLI_MODEL_NAME="MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

# Faster but slightly less accurate
export NLI_MODEL_NAME="microsoft/deberta-v3-base"

# Original model
export NLI_MODEL_NAME="FacebookAI/roberta-large-mnli"
```

### Advanced Configuration

```python
checker = FactChecker(
    search_api="serpapi",        # or "bing"
    num_urls=10,                 # Number of URLs to scrape
    top_k_retrieval=12,          # Top sentences to retrieve
    aggregation_method="hybrid", # "hybrid", "scoring", or "voting"
    verbose=True                 # Show detailed progress
)
```

## 📊 Pipeline Flow

```
User enters claim
    ↓
Phase 0: Data Collection (2-5s)
    → Web search for relevant URLs
    → Scrape content from websites
    → Save corpus with metadata
    ↓
Phase 1: Indexing (0.5-1s)
    → Build BM25 index (all sentences)
    → Filter top 50 with BM25
    → Encode only top 50 with embeddings
    → Build FAISS index (50 embeddings)
    ↓
Phase 2: Retrieval (0.3-0.5s)
    → Stage 1: BM25 → Top 50
    → Stage 2: Hybrid ranking → Top 12
    → Combine: semantic + lexical + metadata
    ↓
Phase 3: NLI Inference (0.3-12s depending on hardware/optimization)
    → Load DeBERTa-v3-large model (once, cached after)
    → Classify each sentence: SUPPORT/REFUTE/NEUTRAL
    → Return confidence scores
    → Performance:
      • GPU: 0.3-0.5s (20-30x faster)
      • CPU + ONNX + Quantization: 1-2s (5-10x faster)
      • CPU + ONNX: 2-4s (2-3x faster)
      • CPU baseline: 10-12s
    ↓
Phase 4: Aggregation (<0.1s)
    → Calculate numerical scores
    → Perform voting
    → Combine methods (hybrid)
    → Generate final verdict
    ↓
Display Result
```

## 📋 Output Format

The pipeline returns a comprehensive result dictionary:

```python
{
    'claim': 'Vietnam is the second largest coffee exporter',
    'verdict': 'SUPPORTED',           # or 'REFUTED' or 'NOT ENOUGH INFO'
    'confidence': 0.78,               # 0-1 confidence score
    'explanation': '...',             # Human-readable explanation
    
    'evidence_summary': {
        'total': 12,
        'supporting': 8,
        'refuting': 2,
        'neutral': 2
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
        'percentages': {...}
    },
    
    'top_evidence': [
        {
            'text': 'Evidence sentence...',
            'label': 'SUPPORT',
            'confidence': 0.92,
            'source': 'https://...'
        },
        # ... top 5 evidence
    ],
    
    'phase_times': {
        'phase0_collection': 3.45,
        'phase1_indexing': 0.87,
        'phase2_retrieval': 0.43,
        'phase3_nli': 5.12,
        'phase4_aggregation': 0.02,
        'total': 9.89
    }
}
```

## 🔧 Troubleshooting

### "API key not found"
```bash
# Set up API key in src/config/api_keys.py
# Copy from template:
cp src/config/config_template.py src/config/api_keys.py
# Edit and add your SERPAPI_KEY
```

### "Module not found"
```bash
# Install all dependencies
pip install -r requirements.txt
```

### "RoBERTa model download"
First time running NLI (Phase 3) will download the model (~1.3GB).
This happens automatically and only once.

### Slow performance
- Phase 0 (Collection): Depends on internet speed and website responsiveness
- Phase 3 (NLI): First run is slow (model download), subsequent runs are fast
- Use `--urls 5` for faster testing

## 💡 Tips

### For Development/Testing
```python
# Quick test with fewer URLs
checker = FactChecker(num_urls=5, verbose=True)
result = checker.check_claim("Test claim")
```

### For Production
```python
# More evidence, less verbose
checker = FactChecker(num_urls=20, verbose=False)
result = checker.check_claim("Production claim")
```

### Batch Processing
```python
checker = FactChecker()

claims = [
    "Vietnam is the second largest coffee exporter",
    "The Eiffel Tower is 330 meters tall",
    "Python was created in 1991"
]

results = []
for claim in claims:
    result = checker.check_claim(claim)
    results.append(result)
    print(f"{claim}: {result['verdict']}")
```

## 📚 See Also

- **Complete Documentation**: `docs/AGGREGATION_GUIDE.md`
- **Architecture**: `architecture.txt`
- **Implementation Guide**: `IMPLEMENTATION_GUIDE.md`
- **Test Scripts**: `tests/test_nli_aggregation.py`

## 🎉 Quick Test

Try this to verify everything works:

```bash
# Test with a simple claim
python -m src.pipeline.fact_check --claim "Water boils at 100 degrees Celsius"

# Expected output:
# ✓ VERDICT: SUPPORTED (high confidence)
# Evidence: Multiple sources confirming the boiling point
```

## 🔍 Example Session

```bash
$ python -m src.pipeline.fact_check

================================================================================
FACT-CHECKING PIPELINE INITIALIZATION
================================================================================

[Phase 0] Initializing Data Collection...
  ✓ Data collector ready

[Phase 1] Initializing Index Builder...
  ✓ Index builder ready (model pre-loaded)

[Phase 2] Retrieval Orchestrator
  ⏳ Will be initialized after first indexing

[Phase 3] NLI Inference
  ⏳ Model will be loaded on first use (singleton pattern)

[Phase 4] Aggregation
  ✓ Method: hybrid

================================================================================
✓ INITIALIZATION COMPLETE
================================================================================

================================================================================
INTERACTIVE FACT-CHECKING SESSION
================================================================================

Type your claims below. Type 'quit', 'exit', or press Ctrl+C to stop.

📝 Enter claim to fact-check: Vietnam is the second largest coffee exporter

[Claim 1] Processing: Vietnam is the second largest coffee exporter

[... pipeline processes ...]

================================================================================
FACT-CHECKING VERDICT
================================================================================

✓ VERDICT: SUPPORTED
   Confidence: 78.5%

   Both scoring (+0.52) and voting (67%) support the claim

[... detailed evidence and timing ...]
```
