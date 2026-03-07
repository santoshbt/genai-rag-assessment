"""
main.py — FastAPI application entrypoint.

Starts:
  - FastAPI app with all routes
  - Prometheus metrics server (optional)
  - Structured logging
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api import router
from logger import setup_logging, start_metrics_server, logger
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    setup_logging()
    start_metrics_server()
    
    # Initialize BM25 index from existing ChromaDB data
    from ingestion import rebuild_bm25_index
    await rebuild_bm25_index()
    
    logger.info(
        "rag_pipeline_started",
        embedding_model=settings.HF_EMBEDDING_MODEL,
        llm_model=settings.HF_LLM_MODEL,
        vector_db="ChromaDB",
        chroma_collection=settings.CHROMA_COLLECTION_NAME,
    )
    yield
    # Shutdown
    logger.info("rag_pipeline_shutdown")


app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "Production-grade Retrieval-Augmented Generation pipeline.\n\n"
        "**Features:**\n"
        "- Structure-aware document parsing via Docling\n"
        "- Contextual chunking preserving headings, tables, sections\n"
        "- HuggingFace embeddings (sentence-transformers)\n"
        "- ChromaDB local vector store with metadata filtering\n"
        "- Hybrid retrieval: BM25 (simple) + vector similarity (complex)\n"
        "- Cross-encoder reranker\n"
        "- ConversationSummaryMemory persisted in SQLite\n"
        "- Pre and post retrieval guardrails\n"
        "- Prometheus observability"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
