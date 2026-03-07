"""
app.py — Streamlit frontend for the RAG Pipeline.

Pages:
  💬 Chat     — Ask questions with session memory
  📄 Documents — Upload and manage documents
  ℹ️  About    — Architecture overview
"""

import streamlit as st
import requests
import time
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal clean styling ──────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Clean chat bubbles */
    .user-msg {
        background: #f0f2f6;
        border-radius: 12px 12px 2px 12px;
        padding: 10px 14px;
        margin: 4px 0;
        margin-left: 20%;
        font-size: 0.95rem;
    }
    .assistant-msg {
        background: #e8f4fd;
        border-radius: 12px 12px 12px 2px;
        padding: 10px 14px;
        margin: 4px 0;
        margin-right: 20%;
        font-size: 0.95rem;
    }
    /* Source cards */
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #4a90d9;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.82rem;
        color: #555;
    }
    /* Metric badges */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        margin-right: 4px;
    }
    .badge-simple  { background: #d4edda; color: #155724; }
    .badge-complex { background: #fff3cd; color: #856404; }
    .badge-bm25    { background: #d1ecf1; color: #0c5460; }
    .badge-hybrid  { background: #e2d9f3; color: #4a235a; }
    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []


# ── API Helpers ────────────────────────────────────────────────────────────────

def api_ingest(file_bytes: bytes, filename: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/ingest",
            files={"file": (filename, file_bytes)},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is the backend running?"}
    except Exception as e:
        return {"error": str(e)}


def api_query(query: str, session_id: str, metadata_filter: dict = None) -> dict:
    payload = {"query": query, "session_id": session_id}
    if metadata_filter:
        payload["metadata_filter"] = metadata_filter
    try:
        r = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is the backend running?"}
    except Exception as e:
        return {"error": str(e)}


def api_new_session() -> str | None:
    try:
        r = requests.post(f"{API_BASE}/sessions", timeout=10)
        r.raise_for_status()
        return r.json().get("session_id")
    except Exception:
        return None


def api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "unreachable", "memory_db": "?"}


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 RAG Assistant")
    st.divider()

    # Navigation
    page = st.radio(
        "Navigate",
        ["💬 Chat", "📄 Documents", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()

    # Session info
    st.caption("SESSION")
    if st.session_state.session_id:
        st.code(st.session_state.session_id[:16] + "...", language=None)
        if st.button("🔄 New Session", use_container_width=True):
            new_id = api_new_session()
            if new_id:
                st.session_state.session_id = new_id
                st.session_state.messages = []
                st.success("New session started.")
                st.rerun()
    else:
        st.caption("No active session")

    st.divider()

    # Health status
    st.caption("BACKEND STATUS")
    health = api_health()
    status_icon = "🟢" if health.get("status") == "healthy" else "🔴"
    st.write(f"{status_icon} API: **{health.get('status', 'unknown')}**")

    db_ok = "🟢" if health.get("memory_db") == "ok" else "🔴"
    st.caption(f"{db_ok} Memory DB")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ══════════════════════════════════════════════════════════════════════════════

if page == "💬 Chat":

    st.header("💬 Chat with your Documents")

    # Auto-create session if none exists
    if not st.session_state.session_id:
        new_id = api_new_session()
        if new_id:
            st.session_state.session_id = new_id
        else:
            st.warning("⚠️ Backend not reachable. Start the API with `python main.py`")

    # Optional: filter by document
    with st.expander("🔧 Filters (optional)", expanded=False):
        filter_doc = st.text_input(
            "Filter by document name",
            placeholder="e.g. contract.pdf — leave blank to search all documents",
        )

    # Render chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info("💡 Upload documents in the **Documents** tab, then ask questions here.")
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-msg">👤 {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="assistant-msg">🤖 {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # Show metadata for assistant messages
                    if msg.get("meta"):
                        meta = msg["meta"]
                        complexity = meta.get("complexity", "")
                        route = meta.get("route_used", "")
                        latency = meta.get("latency_ms", 0)
                        faith = meta.get("faithfulness_score", 0)
                        warnings = meta.get("warnings", [])

                        badge_complexity = (
                            f'<span class="badge badge-simple">SIMPLE</span>'
                            if complexity == "SIMPLE"
                            else f'<span class="badge badge-complex">COMPLEX</span>'
                        )
                        badge_route = (
                            f'<span class="badge badge-bm25">BM25</span>'
                            if route == "BM25"
                            else f'<span class="badge badge-hybrid">HYBRID</span>'
                        )
                        st.markdown(
                            f'{badge_complexity}{badge_route}'
                            f'<span style="font-size:0.75rem;color:#888;"> '
                            f'⏱ {latency:.0f}ms &nbsp;|&nbsp; '
                            f'✅ Faithfulness: {faith:.0%}'
                            f'</span>',
                            unsafe_allow_html=True,
                        )

                        # Warnings
                        for w in warnings:
                            st.caption(f"⚠️ {w}")

                        # Sources
                        sources = meta.get("sources", [])
                        if sources:
                            with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                                for src in sources:
                                    st.markdown(
                                        f'<div class="source-card">'
                                        f'📄 <b>{src["document_name"]}</b> &nbsp;|&nbsp; '
                                        f'Page {src["page_number"]} &nbsp;|&nbsp; '
                                        f'{src["section_title"]}<br>'
                                        f'<i>{src["chunk_excerpt"][:150]}...</i>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                    st.write("")

    # Chat input
    query = st.chat_input("Ask a question about your documents...")

    if query:
        if not st.session_state.session_id:
            st.error("No active session. Please wait for backend connection.")
        else:
            # Add user message immediately
            st.session_state.messages.append({"role": "user", "content": query})

            # Build metadata filter
            metadata_filter = None
            if filter_doc.strip():
                metadata_filter = {"document_name": {"$eq": filter_doc.strip()}}

            # Call API with spinner
            with st.spinner("Thinking..."):
                result = api_query(
                    query,
                    st.session_state.session_id,
                    metadata_filter,
                )

            if "error" in result:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {result['error']}",
                    "meta": {},
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("answer", "No answer returned."),
                    "meta": {
                        "sources":           result.get("sources", []),
                        "complexity":        result.get("complexity", ""),
                        "route_used":        result.get("route_used", ""),
                        "faithfulness_score": result.get("faithfulness_score", 0),
                        "latency_ms":        result.get("latency_ms", 0),
                        "warnings":          result.get("warnings", []),
                    },
                })

            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📄 Documents":

    st.header("📄 Document Management")

    # Upload
    st.subheader("Upload Documents")
    uploaded = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are parsed with Docling, chunked, embedded, and stored in Pinecone.",
    )

    if uploaded:
        if st.button("⬆️ Ingest Selected Files", type="primary"):
            progress = st.progress(0, text="Starting ingestion...")
            results = []

            for i, file in enumerate(uploaded):
                progress.progress(
                    (i) / len(uploaded),
                    text=f"Processing {file.name}...",
                )
                result = api_ingest(file.read(), file.name)
                results.append((file.name, result))

                if "error" not in result:
                    if file.name not in st.session_state.ingested_docs:
                        st.session_state.ingested_docs.append(file.name)

            progress.progress(1.0, text="Done!")
            time.sleep(0.5)
            progress.empty()

            # Show results
            for fname, res in results:
                if "error" in res:
                    st.error(f"❌ **{fname}**: {res['error']}")
                elif res.get("skipped_duplicate"):
                    st.warning(f"⚠️ **{fname}**: Already ingested — skipped.")
                else:
                    st.success(
                        f"✅ **{fname}**: {res['total_chunks']} chunks ingested "
                        f"(ID: `{res['document_id']}`)"
                    )

    st.divider()

    # Ingested documents this session
    st.subheader("Ingested This Session")
    if st.session_state.ingested_docs:
        for doc in st.session_state.ingested_docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                suffix = Path(doc).suffix.upper().replace(".", "")
                icon = "📕" if suffix == "PDF" else "📘"
                st.write(f"{icon} {doc}")
            with col2:
                st.caption(suffix)
    else:
        st.caption("No documents ingested in this session yet.")

    st.divider()

    # Tips
    with st.expander("💡 Tips", expanded=False):
        st.markdown("""
        - **PDF**: Native text extraction + OCR for scanned pages
        - **DOCX**: Preserves headings and table structure via Docling
        - **Deduplication**: Re-uploading the same file is safe — it will be skipped
        - **Metadata filtering**: After ingestion, use the Chat filter to query a specific document
        - **Chunk size**: Configurable in `.env` — default 512 tokens
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":

    st.header("ℹ️ Architecture Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔄 Query Pipeline")
        st.markdown("""
        1. **Greeting filter** — instant canned response, no retrieval
        2. **Query validation** — reject malformed input
        3. **PII redaction** — scrub emails, phone numbers
        4. **Memory** — load conversation summary from SQLite
        5. **Coreference resolution** — resolve "it", "that"
        6. **Complexity classifier** → SIMPLE or COMPLEX
        7. **Routing**
           - SIMPLE → BM25 only (~80ms)
           - COMPLEX → Pinecone hybrid (~350ms)
        8. **Cross-encoder reranker** → top-5 chunks
        9. **Relevance threshold** — decline if score < 0.3
        10. **LLM generation** (GPT-4o)
        11. **Faithfulness check** — score answer vs context
        12. **Save to memory** — auto-summarized by GPT-4o-mini
        """)

    with col2:
        st.subheader("📥 Ingestion Pipeline")
        st.markdown("""
        1. **Docling parser** — structure-aware PDF/DOCX parsing
           - OCR for scanned documents
           - Table structure preservation
        2. **HybridChunker** — semantic chunking
           - Max 512 tokens per chunk
           - Never splits mid-sentence or mid-table
           - Preserves headings and section context
        3. **Content hash deduplication** — skip already-ingested docs
        5. **Pinecone upsert** — vectors + rich metadata
           - document_id, document_name
           - page_number, section_title
           - chunk_index, content_hash
        6. **BM25 index rebuild** — for simple query routing
        """)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🧠 Memory")
        st.markdown("""
        **Type**: `ConversationSummaryMemory`

        After every turn, GPT-4o-mini compresses the full
        conversation history into a rolling summary (~200 tokens).
        The summary stays flat in size regardless of how many
        turns have happened.

        **Storage**: `SQLChatMessageHistory` → SQLite

        Sessions are scoped by `session_id` and survive
        app restarts. Start a new session from the sidebar
        to clear context.
        """)

    with col4:
        st.subheader("🛡️ Guardrails")
        st.markdown("""
        **Pre-retrieval**
        - Greeting / chit-chat interception
        - Query length and gibberish check
        - PII detection and redaction
        - Complexity classification

        **Post-retrieval**
        - Relevance threshold (no hallucination on low scores)
        - Near-duplicate chunk removal
        - Token budget enforcement

        **Post-generation**
        - Faithfulness score (answer vs context overlap)
        - Answer completeness check
        """)

    st.divider()
