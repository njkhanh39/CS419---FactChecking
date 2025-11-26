# Fact-Checking System

A comprehensive fact-checking application using Information Retrieval and Natural Language Inference techniques.

## Architecture

### Phase 1: Data Crawling
- Query generation from user claim
- Web search using SerpApi/Bing API (top 20 URLs)
- Content extraction using trafilatura/newspaper3k
- Corpus creation (20 documents)

### Phase 2: Indexing (Funnel Architecture)

**Stage 1: Candidate Generation (BM25)**
- Fast sparse retrieval
- High recall: 500+ sentences → Top 50
- Inverted index with BM25 scoring

**Stage 2: Semantic Reranking**
- Semantic similarity (embeddings + cosine)
- Entity/keyword matching with penalties
- Metadata scoring (recency, source authority)
- Combined score: Semantic×0.5 + BM25×0.3 + Metadata×0.2
- High precision: Top 50 → Top 10

### Phase 3: NLI Inference
- Batch inference with ONNX-optimized NLI model
- Pair construction: (Sentence, Claim) × 10
- Output: Entailment/Contradiction/Neutral probabilities

### Phase 4: Verdict & Aggregation
- Sentence labeling (Support/Refute/Neutral)
- Score calculation with probability weighting
- Global verdict with thresholds
- Confidence and explanation generation

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
# Edit .env with your API keys
```

## Usage

```bash
python main.py --claim "Your claim here"
```

## Project Structure

See `src/` directory for module organization.
