"""
services/retrieval.py — Retrieval service.

Routing:
  SIMPLE query  → BM25 only (no embedding inference)
  COMPLEX query → Hybrid: ChromaDB vector similarity + metadata filter

Post-retrieval:
  Cross-encoder reranker → top-k final chunks
"""

from typing import List, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import settings
from logger import logger, RETRIEVAL_SCORES, LOW_RELEVANCE
from guardrails import check_relevance_threshold, deduplicate_chunks


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
chroma_collection = chroma_client.get_or_create_collection(
    name=settings.CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# Cross-encoder reranker — loaded once at startup
reranker = CrossEncoder(settings.RERANKER_MODEL)

# In-memory BM25 index — rebuilt from ChromaDB at startup
# For production, persist this index to disk
_bm25_index: Optional[BM25Okapi] = None
_bm25_corpus: List[dict] = []           # list of chunk metadata dicts


# ── BM25 Index Management ──────────────────────────────────────────────────────

def build_bm25_index(chunks: List[dict]):
    """
    Build/rebuild BM25 index from chunk list.
    Called after ingestion to keep BM25 in sync with ChromaDB.
    chunks: list of {"text": str, "metadata": dict}
    """
    global _bm25_index, _bm25_corpus
    
    if not chunks:
        logger.warning("bm25_build_skipped_empty_corpus")
        return
    
    _bm25_corpus = chunks
    tokenized = [c["text"].lower().split() for c in chunks]
    _bm25_index = BM25Okapi(tokenized)
    logger.info("bm25_index_built", corpus_size=len(chunks))


def _get_bm25_results(query: str, top_k: int) -> List[dict]:
    """Run BM25 sparse retrieval. Returns top_k chunk metadata dicts."""
    if _bm25_index is None or not _bm25_corpus:
        logger.warning("bm25_index_not_ready")
        return []

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            chunk = _bm25_corpus[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)

    logger.debug("bm25_retrieval", results=len(results))
    return results


# ── Dense Retrieval via ChromaDB ──────────────────────────────────────────────

async def _embed_query(query: str) -> List[float]:
    """Embed query using HuggingFace sentence-transformers."""
    # sentence-transformers encode is synchronous but fast
    embedding = embedding_model.encode(query, convert_to_numpy=True, show_progress_bar=False)
    return embedding.tolist()


async def _chroma_vector_search(
    query_embedding: List[float],
    top_k: int,
    metadata_filter: Optional[dict] = None,
) -> List[dict]:
    """
    Query ChromaDB with vector similarity + optional metadata filter.
    """
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ['documents', 'metadatas', 'distances']
    }
    if metadata_filter:
        query_kwargs["where"] = metadata_filter

    results = chroma_collection.query(**query_kwargs)

    chunks = []
    if results['ids'] and len(results['ids'][0]) > 0:
        for i in range(len(results['ids'][0])):
            # ChromaDB returns distances, convert to similarity score
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            
            chunks.append({
                "chunk_id":      results['ids'][0][i],
                "text":          results['documents'][0][i],
                "vector_score":  similarity,
                "metadata":      results['metadatas'][0][i]
            })

    logger.debug("chroma_retrieval", results=len(chunks))
    return chunks


# ── Reranker ───────────────────────────────────────────────────────────────────

def rerank_chunks(query: str, chunks: List[dict], top_k: int) -> Tuple[List[dict], List[float]]:
    """
    Cross-encoder reranker: scores each (query, chunk_text) pair.
    Returns (reranked_chunks, scores) sorted by score descending.
    """
    if not chunks:
        return [], []

    pairs = [(query, c["text"]) for c in chunks]
    raw_scores = reranker.predict(pairs)
    
    # Normalize scores using sigmoid to get 0-1 range
    import math
    scores = [1 / (1 + math.exp(-score)) for score in raw_scores]

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    final_chunks = []
    final_scores = []
    for score, chunk in top:
        chunk["reranker_score"] = round(float(score), 4)
        final_chunks.append(chunk)
        final_scores.append(float(score))
        RETRIEVAL_SCORES.observe(score)

    logger.info("reranking_complete", top_k=len(final_chunks), top_score=final_scores[0] if final_scores else 0)
    return final_chunks, final_scores


# ── Main Retrieval Orchestrator ────────────────────────────────────────────────

async def retrieve(
    query: str,
    retrieval_query: str,         # stopword-stripped + coreference-resolved query
    complexity: str,
    metadata_filter: Optional[dict] = None,
) -> Tuple[List[dict], List[float], str, str]:
    """
    Route query based on complexity:
      SIMPLE  → BM25 only  (no embedding inference)
      COMPLEX → ChromaDB vector similarity + optional metadata filter

    Returns:
      (chunks, reranker_scores, route_used, no_answer_message)
      no_answer_message is non-empty if relevance check fails
    """

    # ── SIMPLE ROUTE: BM25 only ──────────────────────────────
    if complexity == "SIMPLE":
        logger.info("routing_simple_bm25", query=query[:60])
        raw_chunks = _get_bm25_results(retrieval_query, top_k=settings.BM25_TOP_K)

        if not raw_chunks:
            # Fallback to vector search if BM25 index not ready
            logger.warning("bm25_fallback_to_vector")
            query_emb = await _embed_query(retrieval_query)
            raw_chunks = await _chroma_vector_search(
                query_emb, settings.PINECONE_TOP_K, metadata_filter
            )

        chunks, scores = rerank_chunks(query, raw_chunks, settings.PINECONE_FINAL_TOP_K)
        route = "BM25"

    # ── COMPLEX ROUTE: Hybrid (vector + metadata filter) ─────
    else:
        logger.info("routing_complex_hybrid", query=query[:60])
        query_emb = await _embed_query(retrieval_query)
        raw_chunks = await _chroma_vector_search(
            query_emb,
            settings.PINECONE_TOP_K,
            metadata_filter,
        )

        # Merge BM25 results for richer candidate pool
        bm25_chunks = _get_bm25_results(retrieval_query, top_k=settings.BM25_TOP_K)

        # Combine and deduplicate by chunk_id before reranking
        seen_ids = {c["chunk_id"] for c in raw_chunks}
        for bc in bm25_chunks:
            cid = bc.get("chunk_id", "")
            if cid and cid not in seen_ids:
                raw_chunks.append(bc)
                seen_ids.add(cid)

        chunks, scores = rerank_chunks(query, raw_chunks, settings.PINECONE_FINAL_TOP_K)
        route = "HYBRID"

    # ── Relevance threshold check ─────────────────────────────
    is_relevant, no_answer_msg = check_relevance_threshold(scores)
    if not is_relevant:
        LOW_RELEVANCE.inc()
        logger.warning("low_relevance_blocking", top_score=max(scores) if scores else 0, threshold=settings.RELEVANCE_THRESHOLD)
        return [], [], route, no_answer_msg

    # ── Context deduplication ─────────────────────────────────
    # Use text embeddings for dedup only if chunks are few enough
    if len(chunks) > 1:
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            texts = [c["text"] for c in chunks]
            vec = TfidfVectorizer().fit_transform(texts).toarray()
            chunks = deduplicate_chunks(chunks, vec.tolist())
        except Exception:
            pass  # skip dedup if vectorization fails

    return chunks, scores, route, ""
