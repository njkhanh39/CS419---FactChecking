# Frontend and Pipeline Fixes - December 9, 2025

## Issues Fixed

### 1. ✅ Missing Refuting Evidence in Display

**Problem:** Frontend showed "0 Supporting, 0 Refuting, 7 Neutral" even though terminal showed "5 REFUTE, 7 NEUTRAL"

**Root Cause:** 
- Frontend was looking for `summary.get('support')` but backend returns `'supporting'`
- Same for `'refute'` vs `'refuting'`

**Fix:**
- Updated `frontend.py` lines to use correct keys: `'supporting'`, `'refuting'`, `'neutral'`

### 2. ✅ Top Evidence Not Verdict-Aware

**Problem:** Top evidence always showed highest-scoring sentences regardless of verdict. When verdict was REFUTED, it still showed SUPPORT/NEUTRAL evidence.

**Root Cause:** 
- `final_decision.py` sorted evidence only by combined score
- Didn't prioritize evidence matching the verdict

**Fix:**
- Modified `final_decision.py` STEP 4 to be verdict-aware:
  - If verdict = SUPPORTED: Show SUPPORT evidence first
  - If verdict = REFUTED: Show REFUTE evidence first  
  - If verdict = NOT ENOUGH INFO: Show by relevance

### 3. ✅ Missing URLs in Frontend

**Problem:** Frontend didn't show source URLs that were scraped

**Root Cause:**
- URLs were collected but not included in API response

**Fix:**
- Added URLs to `phase0` result in `fact_check.py`
- Frontend now displays all source URLs in "🔗 Sources Used" section

### 4. ✅ Missing Scoring/Voting Details

**Problem:** Frontend only showed overall confidence, not the breakdown of scoring vs voting

**Fix:**
- Added collapsible "📊 Detailed Analysis" section showing:
  - Scoring: total_score, normalized_score, support_score, refute_score
  - Voting: percentages for support/refute/neutral, voting verdict

### 5. ✅ No Adjustable Parameters

**Problem:** User couldn't control num_urls or top_k from frontend

**Fix:**
- Added sidebar with sliders for:
  - Number of URLs (5-20, default 10)
  - Number of sentences (8-20, default 12)
  - Estimated runtime calculator
  - Accuracy vs Speed indicators
- Updated API to accept these parameters
- Updated `ClaimRequest` model with `num_urls` and `top_k` fields

### 6. ✅ Generic "Agent is working" Message

**Problem:** Frontend showed single static message during processing

**Fix (Partial):**
- Added progressive update placeholders for future enhancement
- Added performance metrics display showing time for each phase
- **TODO:** Implement real-time progress updates using WebSockets or SSE

## Code Changes Summary

### `src/pipeline/fact_check.py`
- Added `urls` to phase0 result
- Added `retrieved_sentences` preview to phase2 result
- Added `all_evidence` (all NLI results) to final result

### `src/aggregation/final_decision.py`
- Rewrote STEP 4 to prioritize evidence based on verdict
- Added `doc_domain` and `score` to top_evidence items

### `frontend/frontend.py`
- Added sidebar with adjustable sliders
- Fixed evidence summary keys (`supporting`, `refuting`, `neutral`)
- Added "🔗 Sources Used" section
- Added "📊 Detailed Analysis" section with scoring & voting breakdown
- Added color-coded evidence cards (🟢 SUPPORT, 🔴 REFUTE, ⚪ NEUTRAL)
- Added performance metrics display

### `src/api/api.py`
- Updated `ClaimRequest` to accept `num_urls` and `top_k`
- Changed `verbose=True` to `verbose=False` for cleaner API logs
- Pass parameters from request to `checker.check_claim()`

## Remaining Issues

### Issue: "Donald Trump 45th president" Returns REFUTED Instead of SUPPORTED

**Diagnosis:**
The claim "Donald Trump is the 46th president" is actually INCORRECT - Trump was the 45th president. The system correctly identifies this as REFUTED.

However, if the user searches "Donald Trump is the 45th president" (correct claim), the system might still struggle if:
1. Web results contain mixed information about 45th/46th/47th presidents
2. NLI model gets confused by numerical differences
3. All evidence is classified as NEUTRAL due to uncertainty

**Potential Solutions:**
1. Lower `score_threshold` from 0.3 to 0.2 for more lenient SUPPORTED/REFUTED verdicts
2. Adjust `min_evidence` threshold for common knowledge claims
3. Add special handling for numerical fact-checking (future enhancement)
4. Improve query generation to be more precise

**Testing Needed:**
- Test with correct claim: "Donald Trump is the 45th president of the United States"
- Test with incorrect claim: "Donald Trump is the 46th president of the United States"
- Compare results to understand model behavior

## Future Enhancements (Not Yet Implemented)

### 1. Real-Time Progress Updates

**Goal:** Show live updates as each phase completes

**Approach:**
- Use WebSockets or Server-Sent Events (SSE)
- Update frontend progressively:
  - Phase 0: "🔍 Searching web... Found 8/10 URLs"
  - Phase 1: "🏗️ Building indexes... Encoded 125 sentences"
  - Phase 2: "📊 Ranking evidence... Retrieved top 12 sentences"
  - Phase 3: "🤖 Running NLI... Analyzed 5/12 sentences"
  - Phase 4: "🧮 Calculating verdict..."

**Implementation:**
```python
# API side (using SSE)
from sse_starlette.sse import EventSourceResponse

@app.post("/check/stream")
async def check_claim_stream(request: ClaimRequest):
    async def event_generator():
        yield {"event": "phase0_start", "data": "Searching web..."}
        # ... run phase 0 ...
        yield {"event": "phase0_complete", "data": {"urls": [...], "time": 3.2}}
        # ... continue for each phase ...
    
    return EventSourceResponse(event_generator())

# Frontend side
const eventSource = new EventSource("/check/stream");
eventSource.onmessage = (event) => {
    updatePhaseStatus(event.data);
};
```

### 2. Retrieved Sentences Preview

**Goal:** Show top retrieved sentences after Phase 2, before NLI

**Display:**
- Expandable section: "📄 Retrieved Sentences (12)"
- Show sentence text and retrieval score
- Update with NLI labels after Phase 3

### 3. Evidence Filtering

**Goal:** Let users filter evidence by label (SUPPORT/REFUTE/NEUTRAL)

**UI:**
- Checkbox filters above evidence list
- Count badges: `SUPPORT (3)`, `REFUTE (5)`, `NEUTRAL (4)`

### 4. Export Results

**Goal:** Download results as PDF or JSON

**Features:**
- JSON: Complete API response
- PDF: Formatted report with verdict, evidence, sources

## Testing Checklist

- [x] Test with "Vietnam coffee exporter" - should show correct evidence
- [x] Test with refuted claim - should show REFUTE evidence first
- [ ] Test with "Donald Trump 45th president" - verify correctness
- [x] Test adjusting num_urls slider - should update query
- [x] Test adjusting top_k slider - should update query
- [x] Verify all source URLs are displayed
- [x] Verify scoring/voting details are shown
- [ ] Test with various claim types (supported/refuted/unclear)

## Performance Notes

**Typical Runtimes:**
- Phase 0 (Data Collection): 2-5s (depends on num_urls and internet speed)
- Phase 1 (Indexing): 0.5-1s (depends on corpus size)
- Phase 2 (Retrieval): 0.3-0.5s (depends on top_k)
- Phase 3 (NLI): 2-10s (first run downloads model, subsequent runs are faster)
- Phase 4 (Aggregation): <0.1s

**Total:** ~8-15 seconds per query

## Deployment Notes

1. Make sure API is running: `python -m src.api`
2. Make sure frontend is running: `streamlit run frontend/frontend.py`
3. API runs on `http://localhost:8000`
4. Frontend connects to API at `http://localhost:8000/check`

## Known Limitations

1. **No streaming updates yet** - User sees "Agent is working" until complete
2. **No caching** - Each query rebuilds indexes (could cache for same claim)
3. **No error recovery** - If one phase fails, entire pipeline fails
4. **No batch processing** - Can only check one claim at a time
5. **Fixed aggregation method** - Always uses "hybrid" (could make configurable)
