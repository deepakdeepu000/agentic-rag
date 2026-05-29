"""
state.py — Canonical typed state for the agentic RAG graph.

Rules:
- Always TypedDict + Annotated reducers — never bare dict.
- messages uses add_messages so LangGraph can merge updates safely.
- All other fields are plain last-write-wins.
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Retrieval ─────────────────────────────────────────────────────
    query: str
    retrieved_chunks: list[dict]        # [{content, metadata, score}]
    retrieval_attempts: int             # Hard stop at max_retrieval_attempts

    # ── Tool tracking ─────────────────────────────────────────────────
    tool_calls_made: list[str]          # Names of MCP tools already called
    # tool_results: dict[str, str]        # tool_call_id -> result content
    route: str                          # Router decision: retrieve/web_search/both/direct
    retrieval_done: bool                # Retrieval completion flag for routing
    web_search_done: bool               # Web search completion flag for routing

    # ── Control flow ──────────────────────────────────────────────────
    iterations: int                     # Coordinator loop counter
    max_iterations: int                 # Hard stop (default 10)
    final_answer: str | None

    # ── Persistent memory ─────────────────────────────────────────────
    session_id: str                     # UUID; ties rows to this session
    session_summary: str                # Injected as SystemMessage[0] at startup
    messages_since_last_summary: int    # Counter; triggers summarizer node
    summarize_every: int                # Auto-detected per message length
