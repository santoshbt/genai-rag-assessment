"""
memory/memory.py — Conversation memory using LangChain ConversationSummaryMemory
                   with Groq LLM for summarization, persisted in SQLite.
"""

import uuid
from typing import Optional
from pathlib import Path

from langchain_classic.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_groq import ChatGroq

from config import settings
from logger import logger, ACTIVE_SESSIONS


# Ensure db directory exists
Path("db").mkdir(exist_ok=True)


def _build_summary_llm():
    """
    Lightweight LLM used for generating conversation summaries.
    Uses Groq for fast, free summarization.
    """
    if settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
        return ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=0,
            groq_api_key=settings.GROQ_API_KEY,
        )
    else:
        logger.warning("groq_not_configured_for_memory", provider=settings.LLM_PROVIDER)
        return None


def get_or_create_memory(session_id: Optional[str]) -> tuple[str, ConversationSummaryMemory]:
    """
    If session_id provided → restore existing session from SQLite.
    If None → create a new session with fresh UUID.
    Returns (session_id, memory).
    """
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info("new_session_created", session_id=session_id)
    else:
        logger.info("session_resumed", session_id=session_id)

    # SQLite-persisted message history
    chat_history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string=settings.MEMORY_DB_URL,
    )

    summary_llm = _build_summary_llm()
    
    if summary_llm:
        memory = ConversationSummaryMemory(
            llm=summary_llm,
            memory_key="chat_history",
            return_messages=True,
            chat_memory=chat_history,
            human_prefix="User",
            ai_prefix="Assistant",
            input_key="question",
            output_key="answer",
        )
    else:
        # Fallback: use ConversationBufferMemory (no summarization)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=chat_history,
            human_prefix="User",
            ai_prefix="Assistant",
            input_key="question",
            output_key="answer",
        )
        logger.warning("using_buffer_memory_fallback")

    ACTIVE_SESSIONS.inc()
    return session_id, memory


def get_running_summary(memory: ConversationSummaryMemory) -> str:
    """Return the current conversation summary."""
    return memory.buffer if hasattr(memory, 'buffer') else ""


def get_last_assistant_message(memory: ConversationSummaryMemory) -> str:
    """Extract the most recent assistant message from memory."""
    messages = memory.chat_memory.messages
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            return msg.content
        if msg.__class__.__name__ == "AIMessage":
            return msg.content
    return ""


def clear_session(session_id: str):
    """Delete all messages for a session from SQLite."""
    chat_history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string=settings.MEMORY_DB_URL,
    )
    chat_history.clear()
    ACTIVE_SESSIONS.dec()
    logger.info("session_cleared", session_id=session_id)


def save_turn(
    memory: ConversationSummaryMemory,
    user_message: str,
    assistant_message: str,
):
    """
    Save a conversation turn to memory.
    LangChain will auto-summarize after saving.
    """
    memory.save_context(
        {"question": user_message},
        {"answer": assistant_message},
    )
    logger.debug("turn_saved_to_memory")
