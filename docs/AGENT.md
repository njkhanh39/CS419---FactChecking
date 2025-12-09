# AI Agent Context - Fact-Checking System

## Project Overview

This is a **fact-checking application** built for CS419 - Information Retrieval course. The system uses **Information Retrieval** and **Natural Language Inference (NLI)** to automatically verify user claims by collecting evidence from the web.

## System Architecture (5 Phases)

### **Phase 0: Data Collection** ✅ IMPLEMENTED
- **Location**: `src/data_collection/`
- **Purpose**: Gather raw evidence from the web
- **Flow**: User Claim → Search API (Google/Bing) → Scrape 10 URLs → Save Corpus (JSON)
- **Key Files**:
  - `web_search.py` - SerpApi/Bing Search integration
  - `web_scraper.py` - trafilatura/newspaper3k scraping with **concurrent execution**
  - `collector.py` - Complete pipeline orchestrator
- **Output**: `data/raw/corpus_*.json` with 10 documents + metadata
- **⚡ Performance**: 
  - **Baseline**: ~15s (sequential scraping, 10 docs)
  - **Optimized**: ~8-12s (parallel scraping with ThreadPoolExecutor)
  - **Key Optimizations**:
    - Concurrent execution (10 workers): 2-3x speedup
    - Strict timeouts (5s): Prevents blocking on slow sites
    - Text-only headers: 20% bandwidth reduction
  - **Multi-threading Safety**: ✅ Safe - low overhead, no system risk

### **Phase 1: Indexing & Retrieval** (Funnel Architecture) ✅ IMPLEMENTED
- **Location**: `src/retrieval/`
- **Purpose**: Two-stage funnel for efficient sentence retrieval
- **Flow**: 
  - **Stage 1 (BM25 - Cheap Filter)**: Build BM25 on ALL sentences (~890) → Query → Top 50 (high recall, fast)
  - **Stage 2 (Semantic - Expensive)**: Encode ONLY Top 50 → FAISS index (50 embeddings) → Hybrid ranking → Top 12 (high precision)
- **Key Files**:
  - `build_index.py` - Build BM25 index on ALL sentences, then encode only Top 50 for FAISS
  - `bm25_retriever.py` - BM25 lexical retrieval (Stage 1 - cheap filter)
  - `embed_retriever.py` - Semantic embedding retrieval (Stage 2 - on filtered 50)
  - `retrieval_orchestrator.py` - Complete two-stage funnel orchestrator
- **Hybrid Ranking Formula**: 
  ```
  Score = 0.6 × Semantic + 0.2 × Lexical (BM25) + 0.2 × Metadata
  ```
- **Metadata Components** (from `utils/metadata.py`):
  - Recency (date proximity)
  - Authority (trusted domains)
  - Entity overlap (named entities match)
- **⚡ Performance**:
  - **WRONG APPROACH**: Encode all 890 sentences → ~7s (wasteful!)
  - **CORRECT APPROACH**: BM25 filter → Encode only top 50 → ~0.3s (23x faster!)
  - **Key Optimizations**:
    - **Funnel architecture (CRITICAL)**: BM25 first (cheap), then encode only top 50 (not all 890!)
    - Batch encoding (batch_size=16-32): Process 50 sentences in 2-3 batches
    - Hardware acceleration (GPU/MPS): Additional 2-3x speedup
    - Fast model (MiniLM-L6-v2): 4x faster than roberta-large
  - **Memory Savings**: 50 embeddings instead of 890 (17x reduction)

### **Phase 2: NLI Inference** (The Brain) ✅ IMPLEMENTED
- **Location**: `src/nli/`
- **Purpose**: Determine if sentences support/refute the claim
- **Flow**: Top 12 sentences → NLI Model → Probabilities (Entailment/Contradiction/Neutral)
- **Key Files**:
  - `nli_model.py` - NLI model wrapper with optimization support
  - `batch_inference.py` - Batch processing for efficiency
- **Models** (tested):
  1. **RoBERTa-large-MNLI** (RECOMMENDED): `FacebookAI/roberta-large-mnli`
     - ✅ **Status**: Works correctly with 3-class NLI
     - Accuracy: ~90% on MNLI benchmark
     - Speed: 0.5s per batch (GPU), 10s (CPU), 3-5s (CPU + INT8)
     - **Limitation**: Weak numerical reasoning (see NLI_MODEL_LIMITATIONS.md)
  2. ❌ **DeBERTa-v3-large-zeroshot**: `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`
     - **Status**: BROKEN - Zero-shot model (2 labels instead of 3)
     - Error: "index 2 is out of bounds for axis 0 with size 2"
  3. ❌ **DeBERTa-v3-base**: `microsoft/deberta-v3-base`
     - **Status**: BROKEN - Tokenizer conversion failure
     - Error: "Converting from SentencePiece failed"
- **Optimizations**:
  - **GPU Acceleration**: Auto-detects CUDA/MPS (10-20x faster)
  - **INT8 Quantization**: CPU-only, 2-3x faster, 75% memory reduction, ~1-2% accuracy loss
  - **ONNX Runtime**: Not recommended (slow cache loading, unstable)
- **Recommended Configuration** (CPU):
  ```bash
  export NLI_MODEL_NAME="FacebookAI/roberta-large-mnli"
  export NLI_USE_QUANTIZATION="true"  # Enable INT8 for 2-3x speedup
  export NLI_USE_ONNX="false"         # Disable ONNX (PyTorch faster with INT8)
  ```
- **Output**: For each sentence → {label: "SUPPORT/REFUTE/NEUTRAL", confidence: 0.0-1.0, probabilities: {...}}

### **Phase 3: Aggregation & Verdict**
- **Location**: `src/aggregation/`
- **Purpose**: Combine evidence and produce final verdict
- **Flow**: NLI results → Scoring → Thresholding → Final Verdict
- **Key Files**:
  - `scoring.py` - Calculate scores from NLI outputs
  - `voting.py` - Majority voting aggregation
  - `final_decision.py` - Apply thresholds, generate verdict
- **Scoring Logic**:
  ```
  For each sentence:
    If SUPPORT: score = +1 × probability
    If REFUTE: score = -1 × probability
    If NEUTRAL: score = 0
  
  Final Score = Σ(all scores)
  ```
- **Verdict Thresholds**:
  ```
  S_final > 0.5    → SUPPORTED
  S_final < -0.5   → REFUTED
  Otherwise        → INSUFFICIENT EVIDENCE
  ```

### **Phase 4: End-to-End Pipeline**
- **Location**: `src/pipeline/`
- **Purpose**: Orchestrate all phases
- **Key File**: `fact_check.py`
- **Flow**: 
  ```
  claim → collect_data() → build_index() → 
  retrieve_and_rank() → nli_inference() → aggregate() → verdict
  ```

## Project Structure

```
CS419---FactChecking/
├── src/
│   ├── data_collection/    # Phase 0 ✅
│   ├── retrieval/          # Phase 1 (BM25, FAISS, Hybrid Ranking) ✅
│   ├── nli/                # Phase 2 (RoBERTa-MNLI)
│   ├── aggregation/        # Phase 3 (Scoring, voting)
│   ├── pipeline/           # Phase 4 (Orchestration)
│   ├── config/             # Configuration files
│   │   ├── paths.py        # Project paths
│   │   └── api_keys.py     # API keys (gitignored)
│   └── utils/              # Utilities
│       ├── metadata.py     # Metadata scoring ✅
│       ├── preprocessing.py
│       ├── evaluation.py
│       └── helpers.py
├── data/
│   ├── raw/                # Scraped corpus (JSON)
│   ├── processed/          # Cleaned data
│   ├── index/              # BM25 + FAISS indexes
│   └── samples/            # Example data
├── notebooks/              # Jupyter notebooks for testing
├── requirements.txt        # Python dependencies
└── architecture.txt        # Detailed structure

```

## Key Configuration (`src/config/`)

### `api_keys.py` (User-created, gitignored)
```python
SERPAPI_KEY = "..."           # Google Search via SerpApi
BING_API_KEY = "..."          # Bing Search API
DEFAULT_SEARCH_API = "serpapi"
```

### `paths.py` (Project paths)
```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DATA_RAW_DIR = DATA_DIR / 'raw'
DATA_INDEX_DIR = DATA_DIR / 'index'
```

### `config_template.py` (All settings)
- Search settings (num URLs, timeout, retries)
- BM25 parameters (k1=1.5, b=0.75)
- Ranking weights (semantic: 0.5, lexical: 0.3, metadata: 0.2)
- NLI settings (model, batch size)
- Aggregation thresholds (support: 0.5, refute: -0.5)
- Trusted domains for authority scoring

## Data Flow Example

```
User: "Vietnam is the world's second largest coffee exporter"
    ↓
[Phase 0] Web Search → 20 URLs
    ↓
[Phase 0] Scrape → 20 documents with metadata
    ↓
[Phase 1 - Stage 1] Extract ~500 sentences → BM25 → Top 50
    ↓
[Phase 1 - Stage 2] Hybrid Ranking:
    - Semantic: 0.85 (cosine similarity)
    - Lexical: 0.72 (BM25 score)
    - Metadata: 0.90 (recent, trusted, entities match)
    - Combined: 0.83
    → Top 10 sentences
    ↓
[Phase 2] NLI for each sentence:
    - Sentence 1: SUPPORT (0.95)
    - Sentence 2: SUPPORT (0.89)
    - Sentence 3: NEUTRAL (0.67)
    - ...
    ↓
[Phase 3] Aggregate:
    - Score = 0.95 + 0.89 + 0 + ... = 0.73
    - 0.73 > 0.5 → VERDICT: SUPPORTED
    ↓
Output:
{
  "verdict": "SUPPORTED",
  "confidence": 0.82,
  "final_score": 0.73,
  "evidence_summary": {
    "support": 7,
    "refute": 1,
    "neutral": 2
  },
  "top_evidence": [...]
}
```

## Important Implementation Notes

### 1. Corpus Format (Phase 0 Output)
```json
{
  "claim": "User's claim",
  "corpus": [
    {
      "url": "https://...",
      "text": "Full article text...",
      "title": "Article title",
      "date": "2023-12-15",
      "domain": "reuters.com",
      "author": "...",
      "description": "...",
      "extraction_method": "trafilatura"
    },
    ...
  ],
  "metadata": {
    "collection_date": "2024-01-01T10:00:00",
    "num_documents_scraped": 20
  }
}
```

### 2. Sentence Format (After Ranking)
```python
{
  "text": "Sentence text",
  "doc_id": 5,
  "doc_title": "Article title",
  "doc_url": "https://...",
  "doc_date": "2023-12-15",
  "scores": {
    "semantic": 0.85,
    "lexical": 0.72,
    "metadata": 0.90,
    "combined": 0.83
  }
}
```

### 3. NLI Output Format
```python
[
  {
    "sentence": "Vietnam exported 1.7M tons...",
    "label": "SUPPORT",  # or "REFUTE", "NEUTRAL"
    "confidence": 0.95,
    "probabilities": {
      "entailment": 0.95,
      "contradiction": 0.02,
      "neutral": 0.03
    }
  },
  ...
]
```

## Dependencies (requirements.txt)

### Core
- `numpy`, `pandas`, `python-dateutil`

### IR & Search
- `rank-bm25` - BM25 implementation
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Semantic embeddings

### NLI
- `transformers` - HuggingFace models
- `torch` - PyTorch backend
- `onnxruntime` - Optimized inference

### Data Collection
- `trafilatura` - Primary web scraper
- `newspaper3k` - Fallback scraper
- `google-search-results` - SerpApi client
- `requests`, `beautifulsoup4`, `lxml`

## Common Agent Tasks

### Task: Implement a Retrieval Module
1. Read `src/retrieval/help.txt` for context
2. Load index from `data/index/`
3. Accept claim as input
4. Return ranked results with scores
5. Follow BM25 parameters from `config_template.py`

### Task: Implement Hybrid Ranking in Retrieval
1. Read `src/retrieval/help.txt`
2. Import `MetadataHandler` from `src/utils/metadata.py`
3. Get BM25 scores from `bm25_retriever.py` (lexical)
4. Get semantic scores from `embed_retriever.py` (cosine similarity)
5. Get metadata scores from `MetadataHandler`
6. Combine with weights: 0.6 (semantic), 0.2 (lexical), 0.2 (metadata)
7. Return top-k sentences

### Task: Implement NLI Module
1. Read `src/nli/help.txt`
2. Load `roberta-large-mnli` or `roberta-base-mnli`
3. Create pairs: `(premise=sentence, hypothesis=claim)`
4. Batch process for efficiency
5. Map labels: entailment→SUPPORT, contradiction→REFUTE, neutral→NEUTRAL

### Task: Debug Data Collection
1. Check API keys in `src/config/api_keys.py`
2. Test with small num_urls (5) first
3. Check `data/raw/` for saved corpus
4. Verify JSON structure matches expected format

### Task: Integrate New Module
1. Check pipeline flow in `src/pipeline/help.txt`
2. Import from parent modules
3. Pass data in expected format (see above)
4. Handle errors gracefully
5. Add logging/print statements for debugging

## Testing Strategy

### Unit Tests
- Test each module independently
- Use example data from `data/samples/`
- Run module as `python -m src.module_name`

### Integration Tests
- Test pipeline stages together
- Example: data_collection → indexing → retrieval

### End-to-End Tests
- Use Jupyter notebooks: `notebooks/`
- Test complete pipeline with real claims
- Verify output format and verdict logic

## Code Style Guidelines

1. **Docstrings**: All functions must have docstrings with Args/Returns
2. **Type Hints**: Use type hints for function signatures
3. **Error Handling**: Use try/except with meaningful error messages
4. **Logging**: Print progress for long-running operations
5. **Configuration**: Use `config_template.py` values, don't hardcode

## Common Pitfalls

1. **API Keys**: Check they're set before making requests
2. **File Paths**: Use `pathlib` or `os.path.join`, not string concatenation
3. **Data Format**: Verify JSON structure matches expected format
4. **Empty Results**: Handle cases where search/scraping returns 0 results
5. **FAISS dtype**: Always use `np.float32` for embeddings
6. **Batch Size**: Reduce if running out of memory (NLI)

## Getting Help

- **Module-specific**: Read `help.txt` in each module directory
- **Architecture**: See `architecture.txt`
- **Usage**: See `README_NEW.md`
- **Setup**: See `IMPLEMENTATION_GUIDE.md`

## Current Status

- ✅ **Phase 0** (Data Collection): Fully implemented
- ✅ **Metadata Handler**: Fully implemented
- ✅ **Configuration**: Template and paths ready
- 🔄 **Phases 1-4**: Need implementation (your team's work)
- 🔄 **Phase 5**: Need integration

## Quick Reference: Module Relationships

```
data_collection (Phase 0)
    ↓ outputs corpus JSON
build_index (Phase 1)
    ↓ outputs indexes
bm25_retriever (Phase 1)
    ↓ outputs top 50 sentences
rank_sentences (Phase 2) + metadata.py
    ↓ outputs top 10 sentences
batch_inference (Phase 3)
    ↓ outputs NLI results
final_decision (Phase 4)
    ↓ outputs verdict
```

---

**Last Updated**: November 27, 2025  
**Project**: CS419 - Information Retrieval  
**Status**: Phase 0 Complete, Phases 1-4 In Progress
