# Time Complexity Analysis: Fact-Checking Pipeline (Phase 0, 1, 2)

## Overview
This document analyzes the time complexity of the retrieval pipeline from data collection through hybrid ranking.

---

## 📊 Variables Definition

- **N** = Number of URLs to scrape (default: 20)
- **D** = Average document length (chars/words)
- **S** = Total number of sentences extracted (~300-500 after filtering)
- **K₁** = Stage 1 candidates (BM25 top-k, default: 50)
- **K₂** = Stage 2 final results (default: 10-12)
- **V** = Vocabulary size (unique words in corpus)
- **E** = Embedding dimension (384 for all-MiniLM-L6-v2)
- **Q** = Query length (words in claim)

---

## 🔍 Phase 0: Data Collection

### **Step 1: Web Search API Call**
```python
search_results = searcher.search(claim, num_results=N)
```
- **Complexity**: `O(N)` - API returns N URLs
- **Time**: ~2-5 seconds (network I/O, API processing)
- **Dominant Factor**: Network latency

### **Step 2: Web Scraping (N URLs)**
```python
documents = scraper.scrape_from_search_results(search_results, max_documents=N)
```
- **Per URL**:
  - Download HTML: `O(D)` - proportional to page size
  - Parse with trafilatura: `O(D)` - linear HTML parsing
  - Extract metadata: `O(D)` - regex operations
- **Total**: `O(N × D)`
- **Time**: ~1-3 seconds per URL × 20 URLs = **20-60 seconds**
- **Optimization**: Sequential scraping with delays (rate limiting)

### **Step 3: Save Corpus to JSON**
```python
json.dump(corpus, f, indent=2, ensure_ascii=False)
```
- **Complexity**: `O(N × D)` - serialize N documents
- **Time**: <1 second (disk I/O)

### **Phase 0 Total**
- **Time Complexity**: `O(N × D)`
- **Actual Time**: **~25-65 seconds** (dominated by network I/O)

---

## 🏗️ Phase 1: Indexing (One-time setup per corpus)

### **Step 1: Load Corpus & Split Sentences**
```python
sentences = load_corpus_from_json(corpus_path)
```
- **Load JSON**: `O(N × D)` - parse JSON
- **Split sentences**: `O(N × D)` - regex splitting on each document
- **Extract**: Creates S sentences (each with metadata)
- **Complexity**: `O(N × D + S)` ≈ `O(N × D)`

### **Step 2: Filter Sentences**
```python
filtered = filter_sentences(sentences)
```
For each of S sentences:
- Length check: `O(1)`
- Word count: `O(W)` where W = avg words per sentence
- Emoji detection: `O(L)` where L = sentence length
- **Duplicate check**: `O(S) × O(W)` - compare with all seen sentences
  - For each sentence: compare word sets with all previous sentences
  - Worst case: `O(S²W)` but with set operations: `O(S² × W)`

**Optimization**: Using set intersection for word overlap
- **Complexity**: `O(S² × W)` in worst case, but typically `O(S × W)` with early termination
- **Time**: 1-3 seconds for ~500 sentences

### **Step 3: Build BM25 Index**
```python
tokenized_corpus = [tokenize(text) for text in sentence_texts]
bm25 = BM25Okapi(tokenized_corpus)
```
- **Tokenization**: `O(S × W)` - split S sentences into words
- **BM25 construction**:
  - Build inverted index: `O(S × W)`
  - Calculate IDF scores: `O(V)` where V = vocabulary size
  - Calculate document lengths: `O(S)`
- **Total**: `O(S × W + V)`
- **Time**: ~2-5 seconds

### **Step 4: Build FAISS Index**
```python
embeddings = model.encode(sentence_texts)
index.add(embeddings)
```
- **Encoding sentences**:
  - Transformer forward pass per sentence: `O(W² × E)` (self-attention)
  - Batch processing (batch_size=32): `O(S/B × W² × E)` where B = batch size
  - **Effective**: `O(S × W² × E)` (BERT-based models)
- **FAISS IndexFlatIP.add()**:
  - Normalize vectors: `O(S × E)`
  - Add to index: `O(S × E)` - just storing vectors
- **Total**: `O(S × W² × E + S × E)` ≈ `O(S × W² × E)`
- **Time**: **~10-30 seconds** (GPU: ~3-8 seconds)

### **Step 5: Save to Disk**
```python
pickle.dump(bm25, f)
faiss.write_index(index, faiss_path)
```
- **Complexity**: `O(S × E)` - write embeddings
- **Time**: <1 second

### **Phase 1 Total**
- **Time Complexity**: `O(S² × W + S × W² × E)`
  - Dominated by: `O(S × W² × E)` (embedding computation)
- **Actual Time**: **~15-40 seconds** (one-time per corpus)
- **With GPU**: **~5-12 seconds**

---

## 🎯 Phase 2: Retrieval & Ranking (Per Query)

### **Initialization (One-time per session)**
```python
orchestrator = RetrievalOrchestrator()
```
- Load BM25 index: `O(S × V)` - unpickle
- Load FAISS index: `O(S × E)` - memory map
- Load sentence store: `O(S)` - unpickle metadata
- **Time**: ~2-5 seconds (cached in memory)

---

### **Stage 1: BM25 Retrieval**
```python
bm25_results = bm25_retriever.retrieve(claim, top_k=K₁)
```

1. **Tokenize query**: `O(Q)` - split claim into words
2. **BM25 scoring**:
   - Calculate score for each sentence: `O(Q × V)` per sentence
   - Total: `O(S × Q × V)` but typically `O(S × Q)` (sparse lookup)
3. **Top-K selection**:
   - Argsort: `O(S log S)`
   - Select top K₁: `O(K₁)`
4. **Build results with metadata**: `O(K₁)`

- **Complexity**: `O(S × Q + S log S)` ≈ `O(S log S)`
- **Time**: **~0.1-0.5 seconds** for S=500

---

### **Stage 2: Hybrid Ranking**

#### **2.1: Semantic Scoring**
```python
semantic_scores = embed_retriever.compute_similarity(claim, sentence_texts)
```
1. **Encode query**: `O(Q² × E)` - transformer forward pass
2. **Encode K₁ sentences**: `O(K₁ × W² × E)`
3. **Normalize vectors**: `O((1 + K₁) × E)`
4. **Compute cosine similarity**: 
   - Dot product: `O(K₁ × E)`
   - Total: `O((1 + K₁) × E)`

- **Complexity**: `O(Q² × E + K₁ × W² × E + K₁ × E)` ≈ `O(K₁ × W² × E)`
- **Time**: **~0.5-2 seconds** for K₁=50

#### **2.2: Metadata Scoring**
```python
metadata_scores = metadata_handler.calculate_metadata_score(doc, claim)
```
For each of K₁ sentences:
- Parse date: `O(1)` - dateutil.parser
- Calculate recency: `O(1)` - date arithmetic
- Check authority: `O(T)` where T = trusted domains (~30)
- Extract entities (claim): `O(Q)` - regex
- Extract entities (sentence): `O(W)` - regex
- Calculate overlap: `O(E₁ + E₂)` where E₁, E₂ = entity counts

- **Per sentence**: `O(Q + W + E₁ + E₂)` ≈ `O(W)`
- **Total**: `O(K₁ × W)`
- **Time**: **~0.1-0.3 seconds**

#### **2.3: Combine Scores & Sort**
```python
combined_score = 0.5×semantic + 0.3×lexical + 0.2×metadata
hybrid_results.sort(key=lambda x: x['combined_score'], reverse=True)
```
- Normalize BM25: `O(K₁)`
- Combine scores: `O(K₁)`
- Sort: `O(K₁ log K₁)`
- Select top K₂: `O(K₂)`

- **Complexity**: `O(K₁ log K₁)`
- **Time**: **<0.01 seconds**

### **Phase 2 Total (Per Query)**
- **Time Complexity**: `O(S log S + K₁ × W² × E + K₁ × W + K₁ log K₁)`
  - Dominated by: `O(K₁ × W² × E)` (semantic encoding)
- **Actual Time**: **~1-3 seconds per query**
- **With GPU**: **~0.3-0.8 seconds per query**

---

## 📈 Overall Pipeline Complexity

### **Total Time Complexity**
```
Phase 0 (Data Collection): O(N × D)
Phase 1 (Indexing):        O(S² × W + S × W² × E)
Phase 2 (Retrieval):       O(S log S + K₁ × W² × E)
```

### **Dominant Operations**
1. **Phase 0**: Network I/O (web scraping) - **not algorithmically reducible**
2. **Phase 1**: Embedding generation `O(S × W² × E)` - **one-time cost**
3. **Phase 2**: Semantic reranking `O(K₁ × W² × E)` - **per query**

---

## ⏱️ Real-World Performance

### **Typical Values**
- N = 20 documents
- D = 2000 words per document
- S = 400 sentences (after filtering from ~500)
- K₁ = 50 candidates
- K₂ = 12 final results
- V = 5000 unique words
- E = 384 dimensions
- W = 15 words per sentence
- Q = 10 words in query

### **Estimated Times (CPU)**
| Phase | Operation | Time | Frequency |
|-------|-----------|------|-----------|
| **Phase 0** | Web Search | 3s | **Per claim** |
| | Web Scraping (20 URLs) | 30s | **Per claim** |
| | Total | **~35s** | **Per claim** |
| **Phase 1** | Load & Filter | 2s | **Per claim** |
| | Build BM25 | 3s | **Per claim** |
| | Build FAISS | 25s | **Per claim** |
| | Total | **~30s** | **Per claim** |
| **Phase 2** | BM25 Retrieval | 0.2s | Per query |
| | Semantic Encoding | 1.5s | Per query |
| | Metadata Scoring | 0.2s | Per query |
| | Total | **~2s** | Per query |

---

## ⚠️ **CRITICAL ARCHITECTURAL INSIGHT**

### **Current Architecture: Per-Claim Pipeline**

According to the architecture documentation (AGENT.md, retrieval/help.txt):

> **"Each NEW claim requires fresh data collection (web search + scraping)"**
> 
> **"Each claim needs claim-specific evidence corpus (cannot reuse old data)"**

This means:
```
EVERY NEW CLAIM = Phase 0 + Phase 1 + Phase 2
                = 35s + 30s + 2s 
                = ~67 seconds PER CLAIM ⚠️
```

### **Why Can't We Reuse Data?**

**Problem**: Different claims need different evidence
- ☕ "Vietnam coffee exports" → Need coffee trade articles
- 🌡️ "Climate change in 2024" → Need climate science articles
- 💰 "Bitcoin price prediction" → Need cryptocurrency articles

**You cannot use coffee articles to verify climate claims!**

### **Reality Check**

| Scenario | Time per Claim | Reusable? |
|----------|----------------|-----------|
| **Testing same claim multiple times** | 2s (Phase 2 only) | ✅ YES - Indexes cached |
| **Testing similar claims** (e.g., "Vietnam coffee 2023" vs "Vietnam coffee 2024") | 67s (Full pipeline) | ❌ NO - Different search results |
| **Testing completely different claims** | 67s (Full pipeline) | ❌ NO - Need fresh evidence |

### **Actual User Experience**

```
User submits claim: "Vietnam is 2nd largest coffee exporter"
├─ [0-35s]  Collecting evidence from web... (Phase 0)
├─ [35-65s] Building search indexes... (Phase 1)
└─ [65-67s] Retrieving & ranking evidence... (Phase 2)
Total: ~67 seconds ⏱️

User submits new claim: "Climate change causes sea level rise"
├─ [0-35s]  Collecting NEW evidence... (Phase 0)
├─ [35-65s] Building NEW indexes... (Phase 1)  
└─ [65-67s] Retrieving & ranking... (Phase 2)
Total: ANOTHER ~67 seconds ⏱️
```

**This is the correct behavior by design!** Each claim needs fresh, relevant evidence.

---

## 🚀 Optimization Strategies

### **Already Implemented**
✅ Batch encoding (32 sentences at a time)
✅ FAISS for fast similarity search (vs. brute force)
✅ BM25 inverted index (vs. linear scan)
✅ Early termination in duplicate detection
✅ Set operations for entity overlap

---

## 💡 **How to Reduce Per-Claim Time**

### **Critical Optimizations (High Impact)**

#### **1. Phase 0: Parallel Web Scraping (30s → 10s)**
```python
# Current: Sequential scraping
for url in urls:
    document = scraper.scrape_url(url)  # ~1.5s each × 20 = 30s

# Optimized: Parallel scraping
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=10) as executor:
    documents = list(executor.map(scraper.scrape_url, urls))
    # ~3s for all 20 URLs in parallel!
```
**Impact**: ⬇️ **20 seconds saved per claim**

#### **2. Phase 1: GPU Acceleration (25s → 5s)**
```python
# Install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# FAISS will automatically use GPU if available
# sentence-transformers will use CUDA for encoding
```
**Impact**: ⬇️ **20 seconds saved per claim**

#### **3. Phase 1: Smaller Embedding Model (25s → 10s)**
```python
# Current: all-MiniLM-L6-v2 (384 dimensions, 6 layers)
model = SentenceTransformer('all-MiniLM-L6-v2')  # 25s

# Optimized: paraphrase-MiniLM-L3-v2 (128 dimensions, 3 layers)
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # 10s
```
**Trade-off**: 5% accuracy loss, 60% speed gain
**Impact**: ⬇️ **15 seconds saved per claim**

---

### **Combined Optimization Impact**

```
Original:  35s (Phase 0) + 30s (Phase 1) + 2s (Phase 2) = 67s
Optimized: 15s (Phase 0) + 10s (Phase 1) + 2s (Phase 2) = 27s

Speedup: 2.5x faster! ⚡
```

---

### **Alternative Architecture: Pre-Built Knowledge Base**

**Idea**: Build a large index ONCE, reuse for all claims

```python
# One-time setup (hours)
corpus = scrape_general_topics([
    "politics", "science", "technology", "health", 
    "economics", "climate", "sports"
])  # ~100,000 documents

build_indices(corpus)  # ~2 hours

# Per claim (seconds)
results = retrieve_and_rank(claim, top_k=12)  # 2s
```

**Pros**:
- ✅ 2 seconds per claim (vs 67s)
- ✅ No web scraping needed
- ✅ Works offline

**Cons**:
- ❌ Not claim-specific (lower precision)
- ❌ Outdated information (no fresh articles)
- ❌ Huge index size (~50GB for 100k docs)
- ❌ Poor recall for niche topics

**Verdict**: ❌ **Not suitable for fact-checking**
- Need fresh, claim-specific evidence
- General knowledge base has poor coverage

---

### **Hybrid Approach (Best of Both Worlds)**

```python
# Phase 0a: Quick check in pre-built index (2s)
if has_recent_evidence_in_cache(claim):
    return cached_results

# Phase 0b: Collect fresh evidence if needed (35s)
fresh_corpus = collect_corpus(claim, num_urls=20)

# Phase 1 & 2: Index and retrieve (32s)
return retrieve_and_rank(fresh_corpus, claim)
```

**For repeated/trending claims**: 2s (cached)
**For new/unique claims**: 67s (fresh)

---

### **Practical Recommendations**

For your CS419 project:

1. ✅ **Implement parallel scraping** (easiest, 20s savings)
2. ✅ **Use GPU if available** (20s savings)
3. ⚠️ **Keep current model** (accuracy > speed for demo)
4. ✅ **Add progress indicators** (users accept 60s if they see progress)

**Expected optimized time**: **~35-45 seconds per claim** (acceptable for academic demo)

---

## 📊 Scalability Analysis

### **How does performance scale?**

| Sentences (S) | BM25 (Stage 1) | Embedding (Stage 2) | Total |
|---------------|----------------|---------------------|-------|
| 100 | 0.05s | 0.3s | 0.35s |
| 400 | 0.2s | 1.5s | 1.7s |
| 1,000 | 0.5s | 4s | 4.5s |
| 10,000 | 5s | 40s | 45s |

**Conclusion**: Current architecture is optimal for S < 1000 sentences (20-40 documents).

For larger corpora (S > 10,000), consider:
- FAISS approximate search (IVF/HNSW)
- Two-stage embedding: lightweight model → full model
- Pre-filtering with metadata before semantic ranking

---

## 🎯 Summary & Conclusion

### **Reality of Per-Claim Architecture**

**Yes, the time complexity IS high** (~67 seconds per claim), but this is **CORRECT BY DESIGN**:

✅ **Why it must be this way:**
1. Each claim needs **claim-specific evidence** (coffee ≠ climate)
2. Fact-checking requires **fresh, recent sources** (not cached)
3. Web scraping is **inherently slow** (network I/O, not algorithmic)

❌ **Cannot be avoided:**
- Phase 0 (Data Collection) is mandatory per claim
- Phase 1 (Indexing) is mandatory per new corpus
- Only Phase 2 (Retrieval) can be cached for identical claims

---

### **Bottleneck Analysis**

| Component | Time | Can Optimize? | Max Speedup |
|-----------|------|---------------|-------------|
| 🌐 **Web scraping** | 30s | ✅ YES (parallel) | 3x → **10s** |
| 🧠 **Embedding** | 25s | ✅ YES (GPU) | 5x → **5s** |
| 📊 **BM25** | 3s | ⚠️ Minimal | 1.2x → **2.5s** |
| 🔍 **Filtering** | 2s | ⚠️ Minimal | 1.5x → **1.3s** |
| 🎯 **Retrieval** | 2s | ❌ Already fast | - |

**Realistic Optimized Time**: **~27 seconds per claim** (with GPU + parallel scraping)

---

### **Is This Acceptable?**

**For Academic Demo (CS419)**: ✅ **YES**
- Users expect delay for fact-checking
- Quality > speed for research project
- Can show progress indicators

**For Production System**: ⚠️ **DEPENDS**
- Acceptable for high-value claims (legal, medical)
- Not acceptable for real-time social media
- Consider pre-computing for trending topics

---

### **Comparison with Real Systems**

| System | Time per Claim | Approach |
|--------|----------------|----------|
| **Your System** | 67s (27s optimized) | Fresh evidence per claim |
| **Google Fact Check** | 2-5s | Pre-indexed knowledge base |
| **Snopes.com** | Hours/days | Manual human review |
| **ClaimBuster** | 10s | Hybrid (cache + fresh) |

Your system is **slower than Google** (pre-built index) but **faster than human fact-checkers** and provides **higher quality** than pure cached approaches.

---

### **Final Recommendation**

**DO NOT try to cache/reuse data across different claims!** This would:
- ❌ Reduce accuracy (wrong evidence)
- ❌ Miss recent information
- ❌ Violate the architecture design

**Instead, optimize the per-claim pipeline:**
1. ✅ Implement parallel scraping (**easiest**, 20s savings)
2. ✅ Use GPU if available (20s savings)
3. ✅ Add progress bars (UX improvement)
4. ✅ Accept 27-67s as the cost of quality fact-checking

**Bottom Line**: Your 67-second architecture is **appropriate for academic research** and **maintains high accuracy**. The complexity is a feature, not a bug! 🎯
