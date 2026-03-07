"""
observability/logger.py — Structured logging + Prometheus metrics.
"""

import structlog
import time
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from config import settings

# ── Structured Logger Setup ────────────────────────────────────────────────────

def setup_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer() if settings.LOG_LEVEL == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.LOG_LEVEL)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

logger = structlog.get_logger()


# ── Prometheus Metrics ─────────────────────────────────────────────────────────

# Ingestion
DOCS_INGESTED = Counter("rag_documents_ingested_total", "Total documents ingested")
CHUNKS_CREATED = Counter("rag_chunks_created_total", "Total chunks created")
INGESTION_ERRORS = Counter("rag_ingestion_errors_total", "Ingestion errors")
INGESTION_LATENCY = Histogram("rag_ingestion_latency_seconds", "Ingestion latency")

# Queries
QUERIES_TOTAL = Counter("rag_queries_total", "Total queries", ["complexity", "route"])
QUERY_LATENCY = Histogram("rag_query_latency_seconds", "Query latency", ["route"])
CACHE_HITS = Counter("rag_cache_hits_total", "Cache hits", ["level"])
GREETING_INTERCEPTS = Counter("rag_greeting_intercepts_total", "Greeting intercepts")

# Retrieval
RETRIEVAL_SCORES = Histogram("rag_retrieval_reranker_scores", "Reranker scores")
LOW_RELEVANCE = Counter("rag_low_relevance_queries_total", "Queries below relevance threshold")

# Guardrails
FAITHFULNESS_SCORES = Histogram("rag_faithfulness_scores", "Faithfulness scores")
GUARDRAIL_BLOCKS = Counter("rag_guardrail_blocks_total", "Guardrail blocks", ["reason"])

# Memory
ACTIVE_SESSIONS = Gauge("rag_active_sessions", "Active sessions")


def start_metrics_server():
    """Start Prometheus metrics HTTP server."""
    if settings.ENABLE_PROMETHEUS:
        start_http_server(settings.PROMETHEUS_PORT)
        logger.info("prometheus_started", port=settings.PROMETHEUS_PORT)


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
