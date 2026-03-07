"""
services/generation.py — Answer generation using LLM.

Prompt Engineering:
  - System prompt enforces grounding (no hallucination)
  - Injects conversation summary (memory) for context continuity
  - Injects retrieved chunks with source labels
  - Requests citations in answer

Context Management:
  - Enforces max token budget for prompt
  - Trims least-relevant chunks if budget exceeded
"""

from typing import List, Tuple

from config import settings
from logger import logger


# LLM setup based on provider
if settings.LLM_PROVIDER == "groq":
    # Groq API (fast inference)
    from groq import AsyncGroq
    groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    logger.info("using_groq_api", model=settings.GROQ_MODEL)
    
elif settings.LLM_PROVIDER == "local":
    # Local HuggingFace model
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(
        settings.HF_LLM_MODEL,
        token=settings.HF_API_TOKEN if settings.HF_API_TOKEN else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.HF_LLM_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=settings.HF_API_TOKEN if settings.HF_API_TOKEN else None
    )
    
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=settings.LLM_MAX_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
        do_sample=True,
    )
    logger.info("local_llm_loaded", model=settings.HF_LLM_MODEL)
    
elif settings.LLM_PROVIDER == "hf_api":
    # HuggingFace Inference API
    from huggingface_hub import InferenceClient
    llm_client = InferenceClient(token=settings.HF_API_TOKEN)
    logger.info("using_hf_inference_api", model=settings.HF_LLM_MODEL)


# ── Prompt Templates ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document assistant with memory of the current conversation.

STRICT RULES:
1. Answer ONLY based on the provided Document Context below.
2. If the answer is not in the context, say: "I cannot find this information in the provided documents."
3. Never make up facts, figures, dates, or names not present in the context.
4. Always cite your sources using the format [Source: <document_name>, Page <page>].
5. Use the Conversation Summary to resolve follow-up questions and avoid repeating yourself.
6. Be concise, precise and do not exceed more than 5 lines.
7. Prefer bullet points for lists, prose for explanations."""


def _build_context_block(chunks: List[dict]) -> Tuple[str, int]:
    """
    Build the context block from retrieved chunks.
    Returns (context_string, estimated_token_count).
    Each chunk labeled with its source for citation.
    """
    lines = []
    total_tokens = 0

    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        doc_name = meta.get("document_name", "Unknown")
        page = meta.get("page_number", "?")
        section = meta.get("section_title", "")
        text = chunk.get("text", "")

        section_label = f" | {section}" if section and section != "UNKNOWN" else ""
        header = f"[{i}] Source: {doc_name}, Page {page}{section_label}"
        entry = f"{header}\n{text}"
        lines.append(entry)
        total_tokens += len(text.split()) * 4 // 3

    return "\n\n---\n\n".join(lines), total_tokens


def _trim_chunks_to_budget(chunks: List[dict], budget_tokens: int) -> List[dict]:
    """
    Trim least-relevant chunks (last in reranked list) until within token budget.
    """
    trimmed = list(chunks)
    while trimmed:
        _, tokens = _build_context_block(trimmed)
        if tokens <= budget_tokens:
            break
        trimmed.pop()   # remove least-relevant chunk
        logger.debug("chunk_trimmed_for_token_budget", remaining=len(trimmed))
    return trimmed


def build_prompt(
    query: str,
    chunks: List[dict],
    conversation_summary: str,
    complexity: str,
) -> List[dict]:
    """
    Build the full message list for the chat completion.

    Structure:
      [system]  → grounding rules
      [user]    → summary + context + question

    Token budget management:
      Reserve space for system + query + summary, fill rest with context.
    """
    # Estimate tokens for fixed parts
    system_tokens  = len(SYSTEM_PROMPT.split()) * 4 // 3
    query_tokens   = len(query.split()) * 4 // 3
    summary_tokens = len(conversation_summary.split()) * 4 // 3 if conversation_summary else 0
    overhead       = system_tokens + query_tokens + summary_tokens + 200  # buffer

    context_budget = settings.MAX_PROMPT_TOKENS - overhead

    # Trim chunks if needed
    usable_chunks = _trim_chunks_to_budget(chunks, context_budget)
    context_block, _ = _build_context_block(usable_chunks)

    # Build user message
    user_parts = []

    if conversation_summary:
        user_parts.append(
            f"Conversation Summary (what we discussed so far):\n{conversation_summary}"
        )

    user_parts.append(
        f"Document Context:\n{context_block}"
    )

    instruction = (
        "Answer concisely with citations." if complexity == "SIMPLE"
        else "Provide a thorough answer with citations. Use bullet points where appropriate."
    )
    user_parts.append(f"Question: {query}\n\n{instruction}")

    user_message = "\n\n" + "\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    logger.debug(
        "prompt_built",
        chunks_used=len(usable_chunks),
        complexity=complexity,
        has_summary=bool(conversation_summary),
    )
    return messages


async def generate_answer(
    query: str,
    chunks: List[dict],
    conversation_summary: str,
    complexity: str,
) -> str:
    """
    Generate a grounded answer using configured LLM provider.
    Returns the answer string.
    """
    messages = build_prompt(query, chunks, conversation_summary, complexity)
    
    if settings.LLM_PROVIDER == "groq":
        response = await groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        answer = response.choices[0].message.content.strip()
        model_used = settings.GROQ_MODEL
        
    elif settings.LLM_PROVIDER == "local":
        # Local HuggingFace inference
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
        prompt += "<|assistant|>\n"
        
        outputs = llm_pipeline(
            prompt,
            max_new_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            do_sample=True,
            return_full_text=False,
        )
        answer = outputs[0]["generated_text"].strip()
        model_used = settings.HF_LLM_MODEL
        
    elif settings.LLM_PROVIDER == "hf_api":
        # HuggingFace Inference API
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
        prompt += "<|assistant|>\n"
        
        response = llm_client.text_generation(
            prompt,
            model=settings.HF_LLM_MODEL,
            max_new_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )
        answer = response.strip()
        model_used = settings.HF_LLM_MODEL
    
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")
    
    logger.info(
        "answer_generated",
        words=len(answer.split()),
        model=model_used,
        provider=settings.LLM_PROVIDER,
    )
    return answer
