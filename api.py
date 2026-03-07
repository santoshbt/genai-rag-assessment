"""
routers/api.py — FastAPI route definitions.

Endpoints:
  POST /ingest          — Upload and ingest a PDF/DOCX document
  POST /query           — Query the RAG pipeline
  POST /sessions        — Create a new session
  DELETE /sessions/{id} — Delete a session
  GET  /health          — Health check
"""

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from schemas import (
    QueryRequest, QueryResponse, IngestResponse,
    SessionCreateResponse, SessionDeleteResponse, HealthResponse
)
from ingestion import ingest_document
from memory import get_or_create_memory, clear_session
from pipeline import run_query
from logger import logger


router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


# ── Ingestion ──────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a PDF or DOCX document",
    description=(
        "Upload a document for ingestion. The system will:\n"
        "- Parse using Docling (structure-aware)\n"
        "- Chunk while preserving semantic boundaries\n"
        "- Embed with HuggingFace sentence-transformers\n"
        "- Store in ChromaDB with rich metadata\n"
        "- Skip if document already ingested (content hash deduplication)"
    ),
)
async def ingest_endpoint(file: UploadFile = File(...)):
    """Ingest a PDF or DOCX document into the RAG pipeline."""

    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Supported: {ALLOWED_EXTENSIONS}",
        )

    # Write to temp file for Docling processing
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await ingest_document(tmp_path, original_filename=file.filename)
        return result
    except Exception as e:
        logger.error("ingest_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )
    finally:
        os.unlink(tmp_path)


# ── Query ──────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG pipeline",
    description=(
        "Submit a question to be answered using the ingested documents.\n\n"
        "- Pass `session_id` to continue an existing conversation\n"
        "- Omit `session_id` to start a new session\n"
        "- Pass `metadata_filter` to restrict search to specific documents\n"
        "  Example: `{\"document_name\": {\"$eq\": \"contract.pdf\"}}`"
    ),
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Query the RAG pipeline with optional session continuity."""
    try:
        response = await run_query(request)
        return response
    except Exception as e:
        logger.error("query_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


# ── Session Management ─────────────────────────────────────────────────────────

@router.post(
    "/sessions",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation session",
)
async def create_session():
    """Create a new session. Returns session_id for use in subsequent queries."""
    session_id, _ = get_or_create_memory(None)
    return SessionCreateResponse(
        session_id=session_id,
        message="New session created. Pass this session_id in your /query requests.",
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=SessionDeleteResponse,
    summary="Delete a conversation session",
)
async def delete_session(session_id: str):
    """Clear all conversation history for a session."""
    try:
        clear_session(session_id)
        return SessionDeleteResponse(
            session_id=session_id,
            message="Session cleared successfully.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear session: {str(e)}",
        )


# ── Health Check ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check():
    """Check connectivity to ChromaDB and the memory DB."""
    from config import settings
    import sqlalchemy

    # ChromaDB
    chroma_status = "ok"
    try:
        from ingestion import chroma_collection
        _ = chroma_collection.count()
    except Exception as e:
        chroma_status = f"error: {str(e)[:50]}"

    # HuggingFace models (just check if loaded)
    hf_status = "ok"
    try:
        from ingestion import embedding_model
        _ = embedding_model
    except Exception as e:
        hf_status = f"error: {str(e)[:50]}"

    # Memory DB
    db_status = "ok"
    try:
        engine = sqlalchemy.create_engine(settings.MEMORY_DB_URL)
        with engine.connect():
            pass
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"

    overall = "healthy" if all(
        s == "ok" for s in [chroma_status, hf_status, db_status]
    ) else "degraded"

    return HealthResponse(
        status=overall,
        chromadb=chroma_status,
        huggingface=hf_status,
        memory_db=db_status,
    )
