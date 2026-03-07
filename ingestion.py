"""
services/ingestion.py — Document ingestion using Docling for contextual chunking.

Handles:
  - PDF and DOCX parsing via Docling (structure-aware)
  - Context-aware chunking preserving headings, sections, tables
  - HuggingFace embedding generation
  - Deduplication via content hash
  - ChromaDB upsert with rich metadata
"""

import hashlib
import uuid
from pathlib import Path
from typing import List, Tuple

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from config import settings
from schemas import DocumentChunk, ChunkMetadata, IngestResponse
from logger import logger, DOCS_INGESTED, CHUNKS_CREATED, INGESTION_ERRORS, INGESTION_LATENCY, Timer


# ── Clients ────────────────────────────────────────────────────────────────────


# HuggingFace embedding model
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(settings.HF_EMBEDDING_MODEL)

# ChromaDB client
import chromadb
from chromadb.config import Settings as ChromaSettings

chroma_client = chromadb.PersistentClient(
    path=settings.CHROMA_PERSIST_DIR,
    settings=ChromaSettings(anonymized_telemetry=False)
)


def get_or_create_collection():
    """Get or create ChromaDB collection."""
    collection = chroma_client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info("chroma_collection_ready", name=settings.CHROMA_COLLECTION_NAME)
    return collection


chroma_collection = get_or_create_collection()


# ── Docling Parsing ────────────────────────────────────────────────────────────

def parse_document(file_path: str) -> Tuple[object, str]:
    """
    Parse a PDF or DOCX using Docling structure-aware converter.
    Returns (docling_document, document_name).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True                   # OCR for scanned PDFs
    pipeline_options.do_table_structure = True        # Preserve table structure

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
    )

    logger.info("parsing_document", file=path.name, format=suffix)
    result = converter.convert(str(path))
    return result.document, path.name


# ── Contextual Chunking ────────────────────────────────────────────────────────

def contextual_chunk(docling_doc, document_id: str, document_name: str) -> List[DocumentChunk]:
    """
    Chunk document using Docling's HybridChunker:
      - Respects semantic boundaries (sections, headings, paragraphs)
      - Never splits mid-sentence or mid-table
      - Attaches rich metadata to each chunk
      - Redacts PII before storing
    """
    from guardrails import detect_and_redact_pii
    
    chunker = HybridChunker(
        max_tokens=settings.CHUNK_MAX_TOKENS,
        merge_peers=True,                            # merge tiny adjacent chunks
    )

    chunks: List[DocumentChunk] = []
    chunk_index = 0

    for chunk in chunker.chunk(docling_doc):
        text = chunk.text.strip()

        # Discard empty or too-short chunks
        if len(text) < settings.CHUNK_MIN_CHARS:
            continue

        # Redact PII from chunk text
        redacted_text, pii_warnings = detect_and_redact_pii(text)
        if pii_warnings:
            logger.info("pii_redacted_in_chunk", chunk_index=chunk_index, warnings=pii_warnings)

        # Extract metadata from Docling chunk context
        page_number = 0
        section_title = "UNKNOWN"

        if chunk.meta:
            if hasattr(chunk.meta, 'page_no') and chunk.meta.page_no:
                page_number = chunk.meta.page_no
            if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                section_title = chunk.meta.headings[-1]  # innermost heading

        content_hash = hashlib.sha256(redacted_text.encode()).hexdigest()[:16]
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        token_estimate = len(redacted_text.split()) * 4 // 3   # rough token estimate

        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            metadata=ChunkMetadata(
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                section_title=section_title,
                chunk_index=chunk_index,
                raw_text=redacted_text,  # Store redacted text
                content_hash=content_hash,
                char_count=len(redacted_text),
                token_estimate=token_estimate,
            )
        ))
        chunk_index += 1

    logger.info("chunking_complete", document=document_name, chunks=len(chunks))
    return chunks


# ── Embedding ──────────────────────────────────────────────────────────────────

async def embed_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """
    Batch embed all chunks using HuggingFace sentence-transformers.
    Processes in batches for efficiency.
    """
    BATCH_SIZE = 32  # Adjust based on GPU memory
    texts = [c.metadata.raw_text for c in chunks]

    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        # sentence-transformers encode is synchronous, but fast
        embeddings = embedding_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
        logger.debug("embedding_batch", start=i, end=i + len(batch))

    for chunk, embedding in zip(chunks, all_embeddings):
        chunk.embedding = embedding

    return chunks


# ── Deduplication ──────────────────────────────────────────────────────────────

def document_already_ingested(document_id: str) -> bool:
    """
    Check if document_id already exists in ChromaDB.
    """
    try:
        results = chroma_collection.get(
            where={"document_id": document_id},
            limit=1
        )
        return len(results['ids']) > 0
    except Exception:
        return False


# ── ChromaDB Upsert ───────────────────────────────────────────────────────────

def upsert_to_chroma(chunks: List[DocumentChunk]) -> int:
    """
    Upsert embedded chunks into ChromaDB with full metadata.
    """
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for chunk in chunks:
        ids.append(chunk.chunk_id)
        embeddings.append(chunk.embedding)
        
        # ChromaDB metadata
        metadata = {
            "document_id":   chunk.metadata.document_id,
            "document_name": chunk.metadata.document_name,
            "page_number":   chunk.metadata.page_number,
            "section_title": chunk.metadata.section_title,
            "chunk_index":   chunk.metadata.chunk_index,
            "content_hash":  chunk.metadata.content_hash,
            "char_count":    chunk.metadata.char_count,
        }
        metadatas.append(metadata)
        documents.append(chunk.metadata.raw_text)

    # Upsert to ChromaDB
    chroma_collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    
    logger.info("chroma_upsert_complete", count=len(ids))
    return len(ids)


async def rebuild_bm25_index():
    """
    Rebuild BM25 index by fetching all chunks from ChromaDB.
    Called after each ingestion to keep BM25 in sync.
    """
    from retrieval import build_bm25_index
    
    try:
        # Get all documents from ChromaDB
        results = chroma_collection.get(
            include=['documents', 'metadatas']
        )
        
        chunks = []
        for i, doc_id in enumerate(results['ids']):
            chunks.append({
                "chunk_id": doc_id,
                "text": results['documents'][i],
                "metadata": results['metadatas'][i],
            })
        
        build_bm25_index(chunks)
        logger.info("bm25_index_rebuilt", corpus_size=len(chunks))
    except Exception as e:
        logger.error("bm25_rebuild_failed", error=str(e))


# ── Main Ingestion Orchestrator ────────────────────────────────────────────────

async def ingest_document(file_path: str, original_filename: str = None) -> IngestResponse:
    """
    Full ingestion pipeline:
      parse → chunk → embed → dedup check → upsert
    """
    with Timer() as t:
        path = Path(file_path)
        document_name = original_filename or path.name
        document_id = hashlib.sha256(document_name.encode()).hexdigest()[:12]

        # Deduplication check
        if document_already_ingested(document_id):
            logger.warning("duplicate_document", document=document_name)
            return IngestResponse(
                document_id=document_id,
                document_name=document_name,
                total_chunks=0,
                skipped_duplicate=True,
                message=f"Document '{document_name}' already ingested. Skipping."
            )

        try:
            # 1. Parse with Docling
            docling_doc, doc_name = parse_document(file_path)

            # 2. Contextual chunking
            chunks = contextual_chunk(docling_doc, document_id, doc_name)

            if not chunks:
                raise ValueError(f"No valid chunks extracted from {document_name}")

            # 3. Embed
            chunks = await embed_chunks(chunks)

            # 4. Upsert to ChromaDB
            upserted = upsert_to_chroma(chunks)

            # 5. Rebuild BM25 index to include new document
            await rebuild_bm25_index()

            DOCS_INGESTED.inc()
            CHUNKS_CREATED.inc(upserted)

            logger.info(
                "ingestion_complete",
                document=document_name,
                chunks=upserted,
                latency_ms=round(t.elapsed_ms, 2),
            )

            return IngestResponse(
                document_id=document_id,
                document_name=document_name,
                total_chunks=upserted,
                skipped_duplicate=False,
                message=f"Successfully ingested {upserted} chunks."
            )

        except Exception as e:
            INGESTION_ERRORS.inc()
            logger.error("ingestion_failed", document=document_name, error=str(e))
            raise
