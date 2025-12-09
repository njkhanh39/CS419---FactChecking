# File: frontend.py
import streamlit as st
import requests
import json
import time

# 1. Config
API_URL = "http://localhost:8000/check"
API_STREAM_URL = "http://localhost:8000/check/stream"
st.set_page_config(page_title="FactCheck AI", page_icon="🕵️‍♀️", layout="wide")

# 2. Header
st.title("🕵️‍♀️ Automated Fact-Checking Agent")
st.markdown("CS419 - Information Retrieval Project")

# 3. Advanced Settings (Sidebar)
with st.sidebar:
    st.header("⚙️ Settings")
    
    num_urls = st.slider(
        "Number of URLs to search",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="More URLs = Better accuracy but slower"
    )
    
    top_k = st.slider(
        "Number of sentences to analyze",
        min_value=8,
        max_value=20,
        value=12,
        step=1,
        help="More sentences = Better coverage but slower"
    )
    
    st.info(f"""
    **Estimated Runtime:**  
    🕐 ~{8 + num_urls * 0.5 + top_k * 0.3:.0f}s
    
    **Accuracy vs Speed:**  
    📊 URLs: {num_urls}/20  
    📄 Sentences: {top_k}/20
    """)

# 4. Input Form
with st.form("search_form"):
    claim_input = st.text_input("Enter a claim to verify:", 
        value="Donald Trump is the 46th president of the U.S.")
    submitted = st.form_submit_button("Check Fact")

# 5. Logic with Real-Time Progress Updates (SSE)
if submitted and claim_input:
    # Create containers for dynamic updates
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    phase_containers = {
        0: st.empty(),
        1: st.empty(),
        2: st.empty(),
        3: st.empty(),
        4: st.empty()
    }
    
    result_container = st.empty()
    
    try:
        # Try SSE streaming endpoint first (for real-time updates)
        import sseclient
        
        # Prepare request
        payload = {
            "claim": claim_input,
            "num_urls": num_urls,
            "top_k": top_k
        }
        
        # Use streaming endpoint
        response = requests.post(
            API_STREAM_URL,
            json=payload,
            stream=True,
            headers={'Accept': 'text/event-stream'}
        )
        
        if response.status_code == 200:
            client = sseclient.SSEClient(response)
            final_result = None
            
            for event in client.events():
                try:
                    if event.event == "status":
                        data = json.loads(event.data)
                        status_container.info(f"🚀 {data['message']}")
                        progress_bar.progress(data['progress'] / 100)
                    
                    elif event.event == "phase":
                        data = json.loads(event.data)
                        phase_num = data['phase']
                        phase_name = data['name']
                        message = data['message']
                        
                        status_container.info(f"⏳ Phase {phase_num}: {phase_name}")
                        phase_containers[phase_num].markdown(f"**Phase {phase_num}:** {message}")
                        progress_bar.progress(data['progress'] / 100)
                    
                    elif event.event == "phase_complete":
                        data = json.loads(event.data)
                        phase_num = data['phase']
                        message = data['message']
                        phase_time = data.get('time', 0)
                        
                        phase_containers[phase_num].success(f"{message} ({phase_time}s)")
                        progress_bar.progress(data['progress'] / 100)
                        
                        # Show additional details
                        if phase_num == 0 and 'urls' in data:
                            with phase_containers[phase_num]:
                                st.markdown("**Sample URLs:**")
                                for url in data['urls'][:3]:
                                    st.markdown(f"- {url}")
                        
                        elif phase_num == 2 and 'preview' in data:
                            with phase_containers[phase_num]:
                                st.markdown("**Top Retrieved Sentences:**")
                                for sent in data['preview']:
                                    st.markdown(f"- {sent['text']} (score: {sent['score']})")
                        
                        elif phase_num == 3 and 'labels' in data:
                            with phase_containers[phase_num]:
                                labels = data['labels']
                                st.markdown(f"📊 **Labels:** {labels['support']} SUPPORT, {labels['refute']} REFUTE, {labels['neutral']} NEUTRAL")
                    
                    elif event.event == "complete":
                        data = json.loads(event.data)
                        final_result = data['result']
                        progress_bar.progress(1.0)
                        status_container.success("✅ Fact-checking complete!")
                        break
                    
                    elif event.event == "error":
                        data = json.loads(event.data)
                        st.error(f"❌ Error in phase {data.get('phase', 'unknown')}: {data['message']}")
                        break
                
                except json.JSONDecodeError:
                    continue
            
            # Display final results
            if final_result:
                # Clear phase containers
                status_container.empty()
                for container in phase_containers.values():
                    container.empty()
                
                data = final_result
                
                # --- DISPLAY VERDICT ---
                verdict = data.get('verdict', 'UNKNOWN')
                confidence = data.get('confidence', 0.0)
                
                if verdict == "SUPPORTED":
                    st.success(f"### ✅ VERDICT: {verdict}")
                elif verdict == "REFUTED":
                    st.error(f"### ❌ VERDICT: {verdict}")
                else:
                    st.warning(f"### ⚠️ VERDICT: {verdict}")
                
                # Show detailed confidence breakdown
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Overall Confidence", f"{confidence:.1%}")
                with col2:
                    scores = data.get('scores', {})
                    st.metric("Score", f"{scores.get('normalized_score', 0):+.2f}")
                
                # Show voting and scoring details
                with st.expander("📊 Detailed Analysis (Scoring & Voting)", expanded=False):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**🔢 Scoring Method:**")
                        st.write(f"- Total Score: {scores.get('total_score', 0):+.3f}")
                        st.write(f"- Normalized: {scores.get('normalized_score', 0):+.3f}")
                        st.write(f"- Support Score: {scores.get('support_score', 0):+.3f}")
                        st.write(f"- Refute Score: {scores.get('refute_score', 0):+.3f}")
                    
                    with col_b:
                        voting = data.get('voting', {})
                        st.markdown("**🗳️ Voting Method:**")
                        vote_pct = voting.get('percentages', {})
                        st.write(f"- Support: {vote_pct.get('support', 0):.1%}")
                        st.write(f"- Refute: {vote_pct.get('refute', 0):.1%}")
                        st.write(f"- Neutral: {vote_pct.get('neutral', 0):.1%}")
                        st.write(f"- Verdict: {voting.get('verdict', 'N/A')}")

                # --- DISPLAY METRICS ---
                summary = data.get('evidence_summary', {})
                col1, col2, col3 = st.columns(3)
                col1.metric("Supporting Sentences", summary.get('supporting', 0))
                col2.metric("Refuting Sentences", summary.get('refuting', 0))
                col3.metric("Neutral Sentences", summary.get('neutral', 0))
                
                # --- DISPLAY SOURCE URLs ---
                st.divider()
                st.subheader("🔗 Sources Used")
                phase0_data = data.get('phase0', {})
                urls = phase0_data.get('urls', [])
                
                if urls:
                    col_urls = st.columns(2)
                    for idx, url in enumerate(urls):
                        with col_urls[idx % 2]:
                            st.markdown(f"**{idx+1}.** [{url}]({url})")
                else:
                    st.info("No URLs available")

                # --- DISPLAY EVIDENCE ---
                st.divider()
                st.subheader("📚 Top Retrieved Evidence")
                
                top_evidence = data.get('top_evidence', [])
                
                for i, ev in enumerate(top_evidence):
                    label = ev.get('label', 'N/A')
                    
                    # Color code the expander based on NLI label
                    if label == "SUPPORT":
                        label_icon = "🟢"
                    elif label == "REFUTE":
                        label_icon = "🔴"
                    else:
                        label_icon = "⚪"
                    
                    with st.expander(f"{label_icon} #{i+1}: {ev.get('doc_domain', 'Unknown Source')} ({label})"):
                        st.markdown(f"**Text:** _{ev.get('text', '')}_")
                        st.markdown(f"**Source URL:** [{ev.get('source', 'Unknown')}]({ev.get('source', '#')})")
                        st.markdown(f"**NLI Confidence:** {ev.get('confidence', 0):.1%}")
                        st.markdown(f"**Retrieval Score:** {ev.get('retrieval_score', 0):.3f}")
                
                # Show explanation
                st.divider()
                st.info(f"**Explanation:** {data.get('explanation', 'No explanation available')}")
                
                # Show timing
                with st.expander("⏱️ Performance Metrics"):
                    times = data.get('phase_times', {})
                    st.write(f"- Phase 0 (Data Collection): {times.get('phase0_collection', 0):.2f}s")
                    st.write(f"- Phase 1 (Indexing): {times.get('phase1_indexing', 0):.2f}s")
                    st.write(f"- Phase 2 (Retrieval): {times.get('phase2_retrieval', 0):.2f}s")
                    st.write(f"- Phase 3 (NLI): {times.get('phase3_nli', 0):.2f}s")
                    st.write(f"- Phase 4 (Aggregation): {times.get('phase4_aggregation', 0):.2f}s")
                    st.write(f"**Total: {times.get('total', 0):.2f}s**")
        
        else:
            st.error(f"Error: Server returned {response.status_code}")
    
    except ImportError:
        # Fallback to regular API if sseclient not installed
        st.warning("⚠️ Real-time updates not available. Install sseclient-py: `pip install sseclient-py`")
        st.info("Using fallback mode (no progress updates)...")
        
        status_container.info("🚀 Processing... This may take 30-60 seconds.")
        
        try:
            response = requests.post(
                API_URL,
                json={
                    "claim": claim_input,
                    "num_urls": num_urls,
                    "top_k": top_k
                },
                timeout=180
            )
            
            if response.status_code == 200:
                data = response.json()
                status_container.success("✅ Complete!")
                
                # Display results (same as above)
                # ... (reuse the result display code)
                
        except Exception as e:
            st.error(f"Connection Failed: {e}")
            st.info("Make sure the API is running: `python -m src.api.api`")
    
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Make sure the API is running: `python -m src.api.api`")
