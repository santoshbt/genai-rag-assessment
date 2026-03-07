"""
guardrails/guardrails.py — Pre and post retrieval guardrails.

Pre-retrieval:
  - Greeting / chit-chat filter
  - Query validation (length, gibberish)
  - PII detection
  - Query complexity classification → routing decision

Post-retrieval:
  - Relevance threshold check
  - Context deduplication
  - Faithfulness scoring
  - Answer completeness check
"""

import re
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config import settings
from logger import logger, GUARDRAIL_BLOCKS, GREETING_INTERCEPTS


# ── Greeting & Chit-chat Filter ────────────────────────────────────────────────

GREETING_PATTERNS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "how are you doing", "thanks", "thank you", "ok",
    "okay", "bye", "goodbye", "got it", "sure", "great", "cool",
    "who are you", "what can you do", "help", "what is this",
    "nice to meet you", "good day",
}

CANNED_RESPONSES = {
    "greeting":  "Hello! I'm your document assistant. Ask me anything about your uploaded documents.",
    "thanks":    "You're welcome! Feel free to ask anything else about the documents.",
    "identity":  "I'm a document assistant. Upload documents and ask me questions about their content.",
    "farewell":  "Goodbye! Come back anytime to query your documents.",
}

def check_greeting(query: str) -> Tuple[bool, str]:
    """
    Returns (is_greeting, canned_response).
    Runs in <1ms — pure string ops, no model inference.
    """
    normalized = query.lower().strip().rstrip("!?.").strip()

    for pattern in GREETING_PATTERNS:
        if normalized == pattern or normalized.startswith(pattern + " "):
            GREETING_INTERCEPTS.inc()
            if any(w in normalized for w in ["bye", "goodbye"]):
                return True, CANNED_RESPONSES["farewell"]
            if any(w in normalized for w in ["thank", "thanks"]):
                return True, CANNED_RESPONSES["thanks"]
            if any(w in normalized for w in ["who are you", "what can you do", "what is this"]):
                return True, CANNED_RESPONSES["identity"]
            return True, CANNED_RESPONSES["greeting"]

    return False, ""


# ── Query Validation ───────────────────────────────────────────────────────────

def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validate query is substantive and processable.
    Returns (is_valid, reason_if_invalid).
    """
    stripped = query.strip()

    if len(stripped) == 0:
        return False, "Query is empty."

    word_count = len(stripped.split())
    if word_count < 3:
        return False, "Query is too short. Please ask a complete question."

    # Gibberish detection: ratio of non-alpha chars too high
    alpha_ratio = sum(c.isalpha() for c in stripped) / len(stripped)
    if alpha_ratio < 0.4:
        return False, "Query appears to contain invalid characters."

    return True, ""


# ── PII Detection ──────────────────────────────────────────────────────────────

PII_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
    (r"\b\d{10}\b",                                             "phone_number"),
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",                       "aadhaar"),
    (r"\b\d{3}-\d{2}-\d{4}\b",                                  "ssn"),
    (r"\b(?:\d[ -]*?){13,16}\b",                                 "credit_card"),
]

def detect_and_redact_pii(query: str) -> Tuple[str, List[str]]:
    """
    Detect PII in query, redact it, and return warnings.
    Returns (redacted_query, pii_warnings).
    """
    warnings = []
    redacted = query

    for pattern, pii_type in PII_PATTERNS:
        matches = re.findall(pattern, redacted)
        if matches:
            warnings.append(f"PII detected and redacted: {pii_type}")
            redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
            GUARDRAIL_BLOCKS.labels(reason=f"pii_{pii_type}").inc()
            logger.warning("pii_detected", type=pii_type)

    return redacted, warnings


# ── Query Complexity Classifier ────────────────────────────────────────────────

COMPLEXITY_SIGNALS = {
    "compare", "difference", "contrast", "explain why", "how does",
    "relationship", "summarize", "analyze", "evaluate", "pros and cons",
    "versus", "vs", "impact", "implications", "discuss", "elaborate",
}

COMPOUND_SIGNALS = {
    "and also", "additionally", "furthermore", "as well as",
    "along with", "in addition", "moreover",
}

TEMPORAL_SIGNALS = {
    "when", "how many", "how much", "total", "average",
    "between", "from", "since", "before", "after",
}

def classify_query_complexity(query: str) -> str:
    """
    Rule-based complexity scoring — no model inference.
    Returns 'SIMPLE' or 'COMPLEX'.
    """
    score = 0
    lower = query.lower()
    words = lower.split()
    word_count = len(words)

    # Length signals
    if word_count > 15:   score += 2
    elif word_count > 8:  score += 1

    # Multi-question
    if query.count("?") > 1:   score += 2

    # Analytical keywords
    if any(sig in lower for sig in COMPLEXITY_SIGNALS):   score += 2

    # Compound intent
    if any(sig in lower for sig in COMPOUND_SIGNALS):     score += 1

    # Temporal/numeric reasoning
    if any(sig in lower for sig in TEMPORAL_SIGNALS):     score += 1

    # Ambiguous pronouns (likely follow-up on complex topic)
    if any(w in words for w in ["it", "they", "this", "that"]) and word_count > 5:
        score += 1

    complexity = "COMPLEX" if score >= settings.SIMPLE_QUERY_COMPLEXITY_THRESHOLD else "SIMPLE"
    logger.debug("query_complexity", score=score, complexity=complexity)
    return complexity


# ── Stopword Stripping ─────────────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "about",
    "tell", "me", "please", "you", "what", "which", "where", "how",
    "give", "find", "show", "i", "my", "your", "their", "our",
}

def strip_stopwords(query: str) -> str:
    """Strip stopwords for retrieval only — original query sent to LLM."""
    words = query.lower().split()
    stripped = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return " ".join(stripped) if stripped else query


# ── Coreference Resolution ─────────────────────────────────────────────────────

REFERENCE_TRIGGERS = {
    "it", "that", "this", "they", "same", "above",
    "mentioned", "previously", "earlier", "again", "those",
}

def resolve_coreferences(query: str, last_assistant_message: str) -> str:
    """
    Prepend last assistant response to query if coreferences detected.
    Resolved query used ONLY for retrieval — original sent to LLM.
    """
    tokens = set(query.lower().split())
    if tokens & REFERENCE_TRIGGERS and last_assistant_message:
        # Take first 200 chars of last response as resolution context
        context = last_assistant_message[:200]
        resolved = context + " " + query
        logger.debug("coreference_resolved", original=query)
        return resolved
    return query


# ── Post-Retrieval: Relevance Check ───────────────────────────────────────────

def check_relevance_threshold(reranker_scores: List[float]) -> Tuple[bool, str]:
    """
    If top reranked chunk is below threshold, don't generate.
    Returns (is_relevant, message).
    """
    if not reranker_scores:
        return False, "No relevant content found in the documents for this query."

    if max(reranker_scores) < settings.RELEVANCE_THRESHOLD:
        GUARDRAIL_BLOCKS.labels(reason="low_relevance").inc()
        logger.warning("low_relevance", top_score=max(reranker_scores))
        return False, (
            "I don't have enough relevant information in the documents "
            "to answer this question confidently."
        )
    return True, ""


# ── Post-Retrieval: Context Deduplication ─────────────────────────────────────

def deduplicate_chunks(chunks: List[dict], embeddings: List[List[float]]) -> List[dict]:
    """
    Remove near-duplicate chunks (cosine similarity > 0.95).
    Returns deduplicated chunk list.
    """
    if len(chunks) <= 1:
        return chunks

    kept = [0]
    emb_array = np.array(embeddings)

    for i in range(1, len(chunks)):
        is_duplicate = False
        for j in kept:
            sim = cosine_similarity([emb_array[i]], [emb_array[j]])[0][0]
            if sim > 0.95:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(i)

    deduplicated = [chunks[i] for i in kept]
    removed = len(chunks) - len(deduplicated)
    if removed > 0:
        logger.debug("chunks_deduplicated", removed=removed)
    return deduplicated


# ── Post-Generation: Faithfulness Score ───────────────────────────────────────

def compute_faithfulness_score(answer: str, context_chunks: List[str]) -> float:
    """
    Simple faithfulness check: what fraction of answer sentences
    can be traced back to retrieved context via keyword overlap.
    Returns score 0.0–1.0.
    """
    if not answer or not context_chunks:
        return 0.0

    combined_context = " ".join(context_chunks).lower()
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 20]

    if not sentences:
        return 1.0

    grounded = 0
    for sentence in sentences:
        words = set(sentence.lower().split()) - STOPWORDS
        if not words:
            continue
        # Check if at least 40% of meaningful words appear in context
        overlap = sum(1 for w in words if w in combined_context)
        if overlap / len(words) >= 0.4:
            grounded += 1

    score = round(grounded / len(sentences), 3)
    logger.debug("faithfulness_score", score=score)
    return score


# ── Post-Generation: Answer Completeness ──────────────────────────────────────

def check_answer_completeness(answer: str, query: str) -> List[str]:
    """Flag potentially incomplete answers."""
    warnings = []
    word_count = len(answer.split())
    query_words = len(query.split())

    if word_count < settings.MIN_ANSWER_WORDS and query_words > 8:
        warnings.append("Answer may be incomplete for the complexity of the question.")
        logger.warning("answer_too_short", words=word_count)

    return warnings
