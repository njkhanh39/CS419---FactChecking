# Fact-Checking System using Information Retrieval

An automated fact-checking system that retrieves evidence from the web and uses Natural Language Inference (NLI) to verify claims.

## 🎯 System Overview

This system implements a complete fact-checking pipeline:

**User Claim** → **Web Search** → **Evidence Collection** → **Sentence Ranking** → **NLI** → **Verdict**

### Pipeline Phases

1. **Phase 0: Data Collection**
   - Query generation from claim
   - Web search (Google/Bing via APIs)
   - Web scraping (10 articles with parallel processing)
   - Corpus creation with metadata

2. **Phase 1: Indexing & Retrieval** (Funnel Architecture - Two Stages)
   - **Stage 1 (BM25 - Cheap Filter)**: 
     - Build BM25 index on ALL sentences (~890 from 10 articles)
     - Query BM25 → Top 50 sentences (high recall, <0.1s)
   - **Stage 2 (Semantic - Expensive)**: 
     - Encode ONLY the Top 50 BM25 results (not all 890!)
     - Build FAISS index with 50 embeddings
     - Hybrid ranking → Top 12 (high precision)
       - Semantic: 0.5 (embedding similarity)
       - Lexical: 0.3 (BM25 score)
       - Metadata: 0.2 (recency, authority, entity overlap)
   - **Critical**: Don't encode all sentences! Use BM25 as cheap filter first (23x speedup)

3. **Phase 2: NLI Inference**
   - Create pairs: `[(Sentence_1, Claim), (Sentence_2, Claim), ...]`
   - Batch inference with DeBERTa-v3-large (default) or RoBERTa-MNLI
   - GPU acceleration, ONNX Runtime, and INT8 quantization support
   - Output: Probabilities (Entailment, Contradiction, Neutral)
   - Performance: 0.3-0.5s (GPU), 2-3s (CPU with optimizations)

4. **Phase 3: Aggregation & Verdict**
   - Sentence labeling (Support/Refute/Neutral)
   - Score calculation: `+1×P` (Support), `-1×P` (Refute), `0` (Neutral)
   - Final verdict: `S_final = Σ Score_i`
   - **SUPPORTED** if S_final > 0.5
   - **REFUTED** if S_final < -0.5
   - **INSUFFICIENT EVIDENCE** otherwise

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd CS419---FactChecking
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys** (for data collection)
```bash
# Copy template and add your keys
cp src/config/config_template.py src/config/api_keys.py
# Edit api_keys.py with your actual API keys
```

Required API keys:
- **SerpApi** (for Google Search): Get from [https://serpapi.com/](https://serpapi.com/)
- **Bing Search API** (alternative): Get from Azure Portal

### Performance Optimization (Optional)

**For GPU acceleration** (10-20x faster NLI):
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional: ONNX Runtime for maximum speed
pip install onnxruntime-gpu optimum[onnxruntime-gpu]
```

**For CPU optimization** (2-3x faster):
```bash
# Enable ONNX Runtime
export NLI_USE_ONNX="true"

# Or enable INT8 quantization
export NLI_USE_QUANTIZATION="true"

# Or both (ONNX with quantization - fastest CPU)
export NLI_USE_ONNX="true"
export ONNX_QUANTIZE="true"
```

See `docs/GPU_ACCELERATION_GUIDE.md` for detailed setup instructions.

## 🚀 Usage

### Quick Start: Complete Pipeline

```python
from src.data_collection import DataCollector
from src.pipeline.fact_check import FactChecker

# 1. Collect evidence from web
collector = DataCollector(search_api="serpapi")
corpus = collector.collect_corpus(
    claim="Vietnam is the world's largest coffee exporter",
    num_urls=10
)

# 2. Run fact-checking pipeline
checker = FactChecker()
result = checker.check_claim(
    claim="Vietnam is the world's largest coffee exporter",
    corpus=corpus
)

# 3. Display verdict
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Supporting evidence: {result['num_support']}")
print(f"Refuting evidence: {result['num_refute']}")
```

### Step-by-Step Usage

#### 1. Data Collection
```python
from src.data_collection import DataCollector

collector = DataCollector(search_api="serpapi")
corpus = collector.collect_corpus(claim="Your claim here", num_urls=20)
# Saves to: data/raw/corpus_*.json
```

#### 2. Build Index
```python
from src.retrieval.build_index import IndexBuilder

builder = IndexBuilder()
builder.build_bm25_index(corpus)
builder.build_embedding_index(corpus)
# Saves to: data/index/
```

#### 3. Retrieve Sentences with Hybrid Ranking
```python
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embed_retriever import EmbeddingRetriever
from src.utils.metadata import MetadataHandler

# Stage 1: BM25 (Top 50 - High Recall)
bm25_retriever = BM25Retriever()
candidate_sentences = bm25_retriever.retrieve(claim, top_k=50)

# Stage 2: Hybrid Ranking (Top 10 - High Precision)
embed_retriever = EmbeddingRetriever()
metadata_handler = MetadataHandler()

# Combine: 0.5×Semantic + 0.3×Lexical + 0.2×Metadata
final_sentences = hybrid_rank(claim, candidate_sentences, top_k=10)
```

#### 4. NLI Inference
```python
from src.nli.batch_inference import BatchInference

nli = BatchInference()
results = nli.predict_batch(claim, final_sentences)
# Results: [{"sentence": "...", "label": "SUPPORT", "confidence": 0.92}, ...]
```

#### 5. Aggregation
```python
from src.aggregation.final_decision import FinalDecision

aggregator = FinalDecision()
verdict = aggregator.decide(results)
print(verdict)  # {"verdict": "SUPPORTED", "score": 0.75, ...}
```

## 📂 Project Structure

Read `architecture.txt` for detailed structure.

Key directories:
- `src/data_collection/` - Web search and scraping (NEW)
- `src/retrieval/` - BM25 and embedding retrieval
- `src/sentence_ranker/` - Hybrid sentence ranking
- `src/nli/` - Natural Language Inference
- `src/aggregation/` - Scoring and verdict
- `src/pipeline/` - End-to-end orchestration
- `data/` - Data storage (raw, processed, index)
- `notebooks/` - Jupyter notebooks for testing

## 📖 Documentation

Each module has a `help.txt` file explaining its purpose and usage:
- `src/data_collection/help.txt` - Web search and scraping
- `src/retrieval/help.txt` - BM25 and embedding retrieval with hybrid ranking
- `src/nli/help.txt` - Natural Language Inference
- `src/aggregation/help.txt` - Scoring and verdict aggregation
- `src/pipeline/help.txt` - End-to-end orchestration

## ⚙️ Configuration

Edit `src/config/api_keys.py` to customize:
- API keys
- Retrieval parameters (BM25 k1, b)
- Ranking weights (semantic, lexical, metadata)
- NLI model selection
- Aggregation thresholds

## 🧪 Testing

Run notebooks for interactive testing:
```bash
jupyter notebook notebooks/exploratory_data.ipynb
```

## 🔧 Troubleshooting

### API Key Issues
- Ensure `src/config/api_keys.py` exists with valid keys
- Or set environment variables: `SERPAPI_API_KEY`, `BING_API_KEY`

### Scraping Failures
- Some websites block scrapers (403/404 errors)
- The system tries both trafilatura and newspaper3k
- Rate limiting: Adjust `SCRAPING_DELAY` in config

### Memory Issues
- Reduce batch size: `NLI_BATCH_SIZE` in config
- Use smaller model: `roberta-base-mnli` instead of `roberta-large-mnli`

## 📊 Example Output

```
Claim: "Vietnam is the world's second largest coffee producer in 2023"

Verdict: SUPPORTED
Confidence: 0.82
Final Score: 0.73

Evidence Summary:
  - Supporting: 7 sentences
  - Refuting: 1 sentence
  - Neutral: 2 sentences

Top Supporting Evidence:
1. [0.95] "Vietnam ranks as the world's second-largest coffee producer..."
2. [0.89] "In 2023, Vietnam exported approximately 1.7 million tons..."
3. [0.87] "After Brazil, Vietnam is the leading coffee exporter globally..."
```

## 🤝 Contributing

This is a student project for CS419 - Information Retrieval course.

## 👥 Team

Our team has 4 members:
1. Le Tien Dat, Student ID: 23125028
2. Nguyen Gia Khanh, Student ID: 23125007
3. Nguyen Thu Uyen, Student ID: 23125048
4. Cao Thanh Hieu, Student ID: 23125034

