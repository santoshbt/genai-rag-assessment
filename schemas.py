"""
models/schemas.py — All Pydantic request/response schemas and internal data models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────────

class QueryComplexity(str, Enum):
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"

class RetrievalRoute(str, Enum):
    BM25 = "BM25"
    HYBRID = "HYBRID"
    CACHE_L1 = "CACHE_L1"
    GREETING = "GREETING"


# ── Chunk & Document Models ────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    document_id: str
    document_name: str
    page_number: int = 0
    section_title: str = "UNKNOWN"
    chunk_index: int
    raw_text: str
    content_hash: str
    char_count: int
    token_estimate: int


class DocumentChunk(BaseModel):
    chunk_id: str
    embedding: Optional[List[float]] = None
    metadata: ChunkMetadata


# ── Ingestion ──────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id: str
    document_name: str
    total_chunks: int
    skipped_duplicate: bool = False
    message: str


# ── Query & Answer ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    metadata_filter: Optional[dict] = None           # e.g. {"document_name": "contract.pdf"}
    top_k: Optional[int] = None


class SourceReference(BaseModel):
    document_name: str
    page_number: int
    section_title: str
    chunk_excerpt: str
    reranker_score: float


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[SourceReference]
    complexity: QueryComplexity
    route_used: RetrievalRoute
    faithfulness_score: float
    latency_ms: float
    warnings: List[str] = []


# ── Session ────────────────────────────────────────────────────────────────────

class SessionCreateResponse(BaseModel):
    session_id: str
    message: str


class SessionDeleteResponse(BaseModel):
    session_id: str
    message: str


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    chromadb: str
    huggingface: str
    memory_db: str
