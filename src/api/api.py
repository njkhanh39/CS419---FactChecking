# File: src/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.pipeline.fact_check import FactChecker
from sse_starlette.sse import EventSourceResponse
import uvicorn
import asyncio
import json
import time
from typing import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
import functools

# Define the data format coming from the frontend
class ClaimRequest(BaseModel):
    claim: str
    num_urls: int = 10  # Default to 10 URLs
    top_k: int = 12     # Default to 12 sentences
    method: str = "hybrid"  # Aggregation method: 'hybrid', 'voting', 'scoring'

app = FastAPI(title="Fact-Checking Agent API")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GLOBAL VARIABLE
# We initialize this OUTSIDE the function so it only loads the heavy AI models once.
print("\n" + "="*80)
print("⏳ INITIALIZING FACT-CHECKING PIPELINE")
print("="*80)
print("\nThis may take 10-15 seconds on first run (downloading models)...")
print("Subsequent runs will be much faster.\n")

checker = FactChecker(verbose=True)  # Show initialization progress

print("\n" + "="*80)
print("✅ PIPELINE READY - API Server Started")
print("="*80)
print(f"API endpoint: http://localhost:8000/check")
print(f"API docs: http://localhost:8000/docs")
print("="*80 + "\n")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is ready"""
    return {
        "status": "ready",
        "message": "Fact-checking pipeline is initialized and ready",
        "models_loaded": checker.nli_model is not None
    }

@app.post("/check/stream")
async def check_claim_stream(request: ClaimRequest):
    """
    Streaming endpoint that sends real-time progress updates via Server-Sent Events (SSE)
    """
    # Create thread pool executor for blocking operations
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()
    
    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            claim = request.claim
            num_urls = request.num_urls
            top_k = request.top_k
            method = getattr(request, 'method', 'hybrid')
            
            # Send initial status
            yield {
                "event": "status",
                "data": json.dumps({
                    "phase": "init",
                    "message": "Initializing fact-checking pipeline...",
                    "progress": 0
                })
            }
            
            # Import required modules
            from src.data_collection import DataCollector
            from src.retrieval import IndexBuilder, RetrievalOrchestrator
            from src.nli.batch_inference import run_nli_inference
            from src.aggregation.final_decision import make_final_decision
            
            result = {
                'claim': claim,
                'parameters': {
                    'num_urls': num_urls,
                    'top_k': top_k,
                    'aggregation_method': method
                }
            }
            
            # Phase 0: Data Collection
            yield {
                "event": "phase",
                "data": json.dumps({
                    "phase": 0,
                    "name": "Data Collection",
                    "message": f"Searching web for {num_urls} relevant articles...",
                    "progress": 10
                })
            }
            
            phase0_start = time.time()
            # Run blocking operation in thread pool
            corpus = await loop.run_in_executor(
                executor,
                functools.partial(checker.data_collector.collect_corpus, claim, num_urls=num_urls, save=True)
            )
            phase0_time = time.time() - phase0_start
            
            if corpus and corpus.get('corpus'):
                # Filter out None URLs
                urls = [doc.get('url', '') for doc in corpus['corpus'] if doc.get('url')]
                yield {
                    "event": "phase_complete",
                    "data": json.dumps({
                        "phase": 0,
                        "message": f"✓ Collected {len(corpus['corpus'])} documents",
                        "urls": urls[:5],  # Send first 5 URLs
                        "time": round(phase0_time, 2),
                        "progress": 25
                    })
                }
                result['phase0'] = {
                    'time': phase0_time,
                    'num_documents': len(corpus['corpus']),
                    'urls': urls
                }
            else:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "message": "No documents could be retrieved",
                        "phase": 0
                    })
                }
                return
            
            # Phase 1: Indexing
            yield {
                "event": "phase",
                "data": json.dumps({
                    "phase": 1,
                    "name": "Indexing",
                    "message": "Building search indexes (BM25 + FAISS)...",
                    "progress": 35
                })
            }
            
            phase1_start = time.time()
            corpus_file = corpus.get('metadata', {}).get('corpus_file')
            
            if not corpus_file:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "message": "Corpus file path is missing from metadata",
                        "phase": 1
                    })
                }
                return
            
            # Run blocking operation in thread pool
            await loop.run_in_executor(
                executor,
                functools.partial(checker.index_builder.build_from_corpus_file, corpus_file, claim)
            )
            phase1_time = time.time() - phase1_start
            
            yield {
                "event": "phase_complete",
                "data": json.dumps({
                    "phase": 1,
                    "message": "✓ Indexes built successfully",
                    "time": round(phase1_time, 2),
                    "progress": 45
                })
            }
            result['phase1'] = {'time': phase1_time}
            
            # Initialize orchestrator
            if checker.retrieval_orchestrator is None:
                checker.retrieval_orchestrator = RetrievalOrchestrator()
            else:
                checker.retrieval_orchestrator = RetrievalOrchestrator()
            
            # Phase 2: Retrieval
            yield {
                "event": "phase",
                "data": json.dumps({
                    "phase": 2,
                    "name": "Retrieval",
                    "message": f"Retrieving top {top_k} most relevant sentences...",
                    "progress": 55
                })
            }
            
            phase2_start = time.time()
            # Run blocking operation in thread pool
            ranked_evidence = await loop.run_in_executor(
                executor,
                functools.partial(checker.retrieval_orchestrator.retrieve_and_rank, claim=claim, top_k=top_k, verbose=False)
            )
            phase2_time = time.time() - phase2_start
            
            if ranked_evidence:
                preview_sentences = [
                    {"text": ev.get('text', '')[:100] + "...", "score": round(ev.get('combined_score', 0), 3)}
                    for ev in ranked_evidence[:3]
                ]
                yield {
                    "event": "phase_complete",
                    "data": json.dumps({
                        "phase": 2,
                        "message": f"✓ Retrieved {len(ranked_evidence)} sentences",
                        "preview": preview_sentences,
                        "time": round(phase2_time, 2),
                        "progress": 65
                    })
                }
                result['phase2'] = {'time': phase2_time, 'num_evidence': len(ranked_evidence)}
            else:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "message": "No relevant evidence found",
                        "phase": 2
                    })
                }
                return
            
            # Phase 3: NLI Inference
            yield {
                "event": "phase",
                "data": json.dumps({
                    "phase": 3,
                    "name": "NLI Analysis",
                    "message": f"Analyzing {len(ranked_evidence)} sentences with AI model...",
                    "progress": 70
                })
            }
            
            phase3_start = time.time()
            nli_results = await loop.run_in_executor(
                executor,
                functools.partial(run_nli_inference, claim, ranked_evidence)
            )
            phase3_time = time.time() - phase3_start
            
            if nli_results:
                support_count = sum(1 for r in nli_results if r.get('nli_label') == 'SUPPORT')
                refute_count = sum(1 for r in nli_results if r.get('nli_label') == 'REFUTE')
                neutral_count = sum(1 for r in nli_results if r.get('nli_label') == 'NEUTRAL')
                
                yield {
                    "event": "phase_complete",
                    "data": json.dumps({
                        "phase": 3,
                        "message": f"✓ NLI complete: {support_count} SUPPORT, {refute_count} REFUTE, {neutral_count} NEUTRAL",
                        "labels": {
                            "support": support_count,
                            "refute": refute_count,
                            "neutral": neutral_count
                        },
                        "time": round(phase3_time, 2),
                        "progress": 85
                    })
                }
                result['phase3'] = {'time': phase3_time, 'num_results': len(nli_results)}
            else:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "message": "NLI inference failed",
                        "phase": 3
                    })
                }
                return
            
            # Phase 4: Aggregation
            yield {
                "event": "phase",
                "data": json.dumps({
                    "phase": 4,
                    "name": "Final Verdict",
                    "message": "Computing final verdict from evidence...",
                    "progress": 90
                })
            }
            
            phase4_start = time.time()
            verdict = await loop.run_in_executor(
                executor,
                functools.partial(make_final_decision, nli_results, method=method)
            )
            phase4_time = time.time() - phase4_start
            
            result['phase4'] = {'time': phase4_time}
            
            # Compile final result
            total_time = phase0_time + phase1_time + phase2_time + phase3_time + phase4_time
            
            result.update({
                'verdict': verdict['verdict'],
                'confidence': verdict['confidence'],
                'explanation': verdict['explanation'],
                'evidence_summary': verdict['evidence_summary'],
                'scores': verdict['scores'],
                'voting': verdict['voting'],
                'top_evidence': verdict['top_evidence'],
                'all_evidence': nli_results,
                'phase_times': {
                    'phase0_collection': phase0_time,
                    'phase1_indexing': phase1_time,
                    'phase2_retrieval': phase2_time,
                    'phase3_nli': phase3_time,
                    'phase4_aggregation': phase4_time,
                    'total': total_time
                }
            })
            
            # Send final result
            yield {
                "event": "complete",
                "data": json.dumps({
                    "verdict": result['verdict'],
                    "confidence": result['confidence'],
                    "progress": 100,
                    "result": result
                })
            }
            
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({
                    "message": str(e),
                    "phase": "unknown"
                })
            }
    
    return EventSourceResponse(event_generator())

@app.post("/check")
async def check_claim_endpoint(request: ClaimRequest):
    try:
        print(f"\n{'='*60}")
        print(f"NEW REQUEST: {request.claim[:50]}...")
        print(f"Parameters: urls={request.num_urls}, sentences={request.top_k}")
        print(f"{'='*60}\n")
        
        # Run your existing pipeline logic with custom parameters
        result = checker.check_claim(
            claim=request.claim, 
            num_urls=request.num_urls,
            top_k=request.top_k,
            method=getattr(request, 'method', 'hybrid')
        )
        
        # Check if your pipeline caught an error internally
        if result.get('verdict') == 'ERROR':
            raise HTTPException(status_code=500, detail=result.get('error'))
        
        print(f"\n{'='*60}")
        print(f"REQUEST COMPLETE: {result['verdict']} ({result['confidence']:.1%})")
        print(f"{'='*60}\n")

        return result

    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)