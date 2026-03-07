"""
pipeline.py — Main RAG pipeline orchestrator.

Ties together:
  Guardrails → Cache → Memory → Routing → Retrieval → Generation → Post-guardrails
"""

import time
from typing import Optional

from schemas import QueryRequest, QueryResponse, SourceReference, QueryComplexity, RetrievalRoute
from guardrails import (
    check_greeting,
    validate_query,
    detect_and_redact_pii,
    classify_query_complexity,
    strip_stopwords,
    resolve_coreferences,
    compute_faithfulness_score,
    check_answer_completeness,
)
from memory import get_or_create_memory, get_running_summary, get_last_assistant_message, save_turn
from retrieval import retrieve
from generation import generate_answer
from logger import logger, QUERIES_TOTAL, QUERY_LATENCY, FAITHFULNESS_SCORES, Timer


async def run_query(request: QueryRequest) -> QueryResponse:
    """
    Full RAG query pipeline:

      1.  Greeting filter          → short-circuit, no retrieval
      2.  Query validation         → reject malformed input
      3.  PII detection & redaction
      4.  Memory: restore session  → get running summary
      5.  Coreference resolution   → resolve "it", "that" etc
      6.  Complexity classification → SIMPLE or COMPLEX
      7.  Stopword stripping       → cleaner retrieval query
      8.  Retrieval routing        → BM25 (simple) or Hybrid (complex)
      9.  Reranking                → cross-encoder top-5
      10. Relevance threshold check
      11. Context deduplication
      12. Prompt construction + LLM generation
      13. Post-generation guardrails (faithfulness, completeness)
      14. Save turn to memory      → summary updated automatically
      15. Return structured response
    """
    start_time = time.perf_counter()
    warnings = []
    query = request.query.strip()

    # ── Step 1: Greeting filter ───────────────────────────────
    is_greeting, canned = check_greeting(query)
    if is_greeting:
        return QueryResponse(
            answer=canned,
            session_id=request.session_id or "none",
            sources=[],
            complexity=QueryComplexity.SIMPLE,
            route_used=RetrievalRoute.GREETING,
            faithfulness_score=1.0,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            warnings=[],
        )

    # ── Step 2: Query validation ──────────────────────────────
    is_valid, reason = validate_query(query)
    if not is_valid:
        return QueryResponse(
            answer=reason,
            session_id=request.session_id or "none",
            sources=[],
            complexity=QueryComplexity.SIMPLE,
            route_used=RetrievalRoute.BM25,
            faithfulness_score=0.0,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            warnings=[reason],
        )

    # ── Step 3: PII detection ─────────────────────────────────
    query, pii_warnings = detect_and_redact_pii(query)
    warnings.extend(pii_warnings)

    # ── Step 4: Memory — restore or create session ────────────
    session_id, memory = get_or_create_memory(request.session_id)
    conversation_summary = get_running_summary(memory)
    last_assistant_msg = get_last_assistant_message(memory)

    # ── Step 5: Coreference resolution ───────────────────────
    retrieval_query = resolve_coreferences(query, last_assistant_msg)

    # ── Step 6: Complexity classification ────────────────────
    complexity = classify_query_complexity(query)

    # ── Step 7: Stopword stripping (retrieval query only) ────
    retrieval_query_stripped = strip_stopwords(retrieval_query)

    logger.info(
        "pipeline_start",
        session_id=session_id,
        complexity=complexity,
        query_preview=query[:60],
    )

    # ── Steps 8–11: Retrieval + Reranking ────────────────────
    with Timer() as retrieval_timer:
        chunks, reranker_scores, route_str, no_answer_msg = await retrieve(
            query=query,
            retrieval_query=retrieval_query_stripped,
            complexity=complexity,
            metadata_filter=request.metadata_filter,
        )

    # If relevance threshold failed → return gracefully
    if no_answer_msg:
        return QueryResponse(
            answer=no_answer_msg,
            session_id=session_id,
            sources=[],
            complexity=QueryComplexity(complexity),
            route_used=RetrievalRoute(route_str),
            faithfulness_score=0.0,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            warnings=warnings + ["Low relevance — answer withheld to prevent hallucination."],
        )

    # ── Step 12: Answer generation ────────────────────────────
    with Timer() as gen_timer:
        answer = await generate_answer(
            query=query,
            chunks=chunks,
            conversation_summary=conversation_summary,
            complexity=complexity,
        )

    # ── Step 13: Post-generation guardrails ───────────────────
    context_texts = [c.get("text", "") for c in chunks]
    faithfulness = compute_faithfulness_score(answer, context_texts)
    completeness_warnings = check_answer_completeness(answer, query)
    warnings.extend(completeness_warnings)

    FAITHFULNESS_SCORES.observe(faithfulness)

    if faithfulness < 0.3:
        warnings.append(f"Low faithfulness score ({faithfulness}) — answer may not be fully grounded.")
        logger.warning("low_faithfulness", score=faithfulness)

    # ── Step 14: Save turn to memory ──────────────────────────
    # LangChain auto-summarizes after save_context
    save_turn(memory, query, answer)

    # ── Step 15: Build response ───────────────────────────────
    sources = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        sources.append(SourceReference(
            document_name=meta.get("document_name", "Unknown"),
            page_number=meta.get("page_number", 0),
            section_title=meta.get("section_title", "UNKNOWN"),
            chunk_excerpt=chunk.get("text", "")[:200] + "...",
            reranker_score=round(chunk.get("reranker_score", 0.0), 4),
        ))

    total_latency = round((time.perf_counter() - start_time) * 1000, 2)

    QUERIES_TOTAL.labels(complexity=complexity, route=route_str).inc()
    QUERY_LATENCY.labels(route=route_str).observe(total_latency / 1000)

    logger.info(
        "pipeline_complete",
        session_id=session_id,
        complexity=complexity,
        route=route_str,
        faithfulness=faithfulness,
        latency_ms=total_latency,
        sources=len(sources),
    )

    return QueryResponse(
        answer=answer,
        session_id=session_id,
        sources=sources,
        complexity=QueryComplexity(complexity),
        route_used=RetrievalRoute(route_str),
        faithfulness_score=faithfulness,
        latency_ms=total_latency,
        warnings=warnings,
    )
