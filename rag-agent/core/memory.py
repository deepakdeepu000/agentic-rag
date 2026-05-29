"""
memory.py — SQLite session store, summarizer node, and session startup loader.

Design:
- SQLiteSessionStore wraps all blocking sqlite3 calls with asyncio.to_thread
  so the async LangGraph graph is never blocked.
- summarizer_node fires every summarize_every messages to trim context window
  bloat; it merges (not replaces) prior summary so history is never lost.
- load_session_into_state is the single entry point before every graph invoke.
"""
import asyncio
import logging
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from core.config import Settings
from core.llm_factory import get_chat_model
from core.state import RAGState
from utils.mcp_result_utils import extract_mcp_text


logger = logging.getLogger(__name__)


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class SessionRecord:
    session_id: str
    title: str
    summary: str
    msg_count: int
    created_at: str
    updated_at: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rows_to_messages(rows: list[tuple[str, str]]) -> list:
    """Convert (role, content) DB rows back to LangChain message objects."""
    mapping = {
        "human": HumanMessage,
        "ai": AIMessage,
        "system": SystemMessage,
    }
    result = []
    for role, content in rows:
        cls = mapping.get(role)
        if cls:
            result.append(cls(content=content))
    return result


def get_summarize_threshold(messages: list) -> int:
    """
    Auto-detect summarize_every based on average message length.
    Long messages (code, docs) → summarize every 4.
    Short Q&A turns         → summarize every 8.
    """
    if not messages:
        return 8
    avg_len = sum(len(m.content) for m in messages[-4:]) / min(len(messages), 4)
    return 4 if avg_len > 800 else 8


def _clean_session_title(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^(session\s*title\s*:\s*|title\s*:\s*)", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip('"\'`')
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("-:;,. ")
    return cleaned[:80]


SESSION_TITLE_SYSTEM_PROMPT = """You generate short, descriptive session titles.

Given the user's first question, return a concise title with these rules:
- 3 to 8 words maximum
- specific to the topic
- no quotes, bullets, numbering, markdown, or trailing punctuation
- output only the title text
"""


# ── SQLite Session Store ──────────────────────────────────────────────────────

class SQLiteSessionStore:
    """Async-safe SQLite session store. All blocking calls wrapped in to_thread."""

    def __init__(self, db_path: str = "./sessions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    title       TEXT NOT NULL DEFAULT '',
                    summary     TEXT NOT NULL DEFAULT '',
                    msg_count   INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
            """)
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "title" not in columns:
                conn.execute(
                    "ALTER TABLE sessions ADD COLUMN title TEXT NOT NULL DEFAULT ''"
                )
            conn.commit()

    # ── Sync internals ────────────────────────────────────────────────

    def _create_session_sync(self, session_id: str) -> SessionRecord:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, title) VALUES (?, '')",
                (session_id,),
            )
            conn.commit()
        return self._load_session_sync(session_id)

    def _load_session_sync(self, session_id: str) -> SessionRecord | None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT session_id, title, summary, msg_count, created_at, updated_at "
                "FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return SessionRecord(*row) if row else None

    def _list_sessions_sync(self, limit: int = 20) -> list[SessionRecord]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT session_id, title, summary, msg_count, created_at, updated_at "
                "FROM sessions ORDER BY updated_at DESC, created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [SessionRecord(*row) for row in rows]

    def _update_title_sync(self, session_id: str, title: str):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE session_id = ?",
                (title, session_id),
            )
            conn.commit()

    def _update_summary_sync(self, session_id: str, summary: str):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE session_id = ?",
                (summary, session_id),
            )
            conn.commit()

    def _save_message_sync(self, session_id: str, role: str, content: str):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            conn.execute(
                "UPDATE sessions SET msg_count = msg_count + 1, "
                "updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()

    def _get_messages_sync(self, session_id: str, limit: int = 20) -> list[tuple]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return list(reversed(rows))  # oldest-first

    # ── Async public API ──────────────────────────────────────────────

    async def create_session(self, session_id: str) -> SessionRecord:
        return await asyncio.to_thread(self._create_session_sync, session_id)

    async def load_session(self, session_id: str) -> SessionRecord | None:
        return await asyncio.to_thread(self._load_session_sync, session_id)

    async def list_sessions(self, limit: int = 20) -> list[SessionRecord]:
        return await asyncio.to_thread(self._list_sessions_sync, limit)

    async def update_title(self, session_id: str, title: str):
        await asyncio.to_thread(self._update_title_sync, session_id, title)

    async def update_summary(self, session_id: str, summary: str):
        await asyncio.to_thread(self._update_summary_sync, session_id, summary)

    async def save_message(self, session_id: str, role: str, content: str):
        await asyncio.to_thread(self._save_message_sync, session_id, role, content)

    async def get_recent_messages(self, session_id: str, limit: int = 10) -> list:
        rows = await asyncio.to_thread(self._get_messages_sync, session_id, limit)
        return _rows_to_messages(rows)


# ── Session startup loader ────────────────────────────────────────────────────

async def load_session_into_state(
    session_id: str,
    store: SQLiteSessionStore,
    settings: Settings,
) -> dict:
    """
    Load (or create) a session and build the initial partial RAGState.
    Call this BEFORE every graph.ainvoke().
    """
    session = await store.load_session(session_id)
    if not session:
        session = await store.create_session(session_id)

    recent_msgs = await store.get_recent_messages(session_id, limit=10)
    summarize_every = get_summarize_threshold(recent_msgs)

    # Session context injected as SystemMessage[0] — never as human/ai
    init_messages: list = []
    if session.summary:
        init_messages.append(
            SystemMessage(content=f"[SESSION CONTEXT]\n{session.summary}")
        )
    init_messages.extend(recent_msgs)

    return {
        "session_id": session_id,
        "session_summary": session.summary,
        "messages": init_messages,
        "messages_since_last_summary": 0,
        "summarize_every": summarize_every,
        # Standard defaults
        "query": "",
        "retrieved_chunks": [],
        "retrieval_attempts": 0,
        "tool_calls_made": [],
        "route": "direct",
        "retrieval_done": False,
        "web_search_done": False,
        "iterations": 0,
        "max_iterations": settings.max_iterations,
        "final_answer": None,
    }


async def generate_session_title(settings: Settings, user_question: str) -> str:
    llm = get_chat_model(settings, purpose="summarizer")
    response = await llm.ainvoke(
        [
            SystemMessage(content=SESSION_TITLE_SYSTEM_PROMPT),
            HumanMessage(content=f"User question:\n{user_question}"),
        ]
    )

    raw_content = getattr(response, "content", response)
    pieces = extract_mcp_text(raw_content)
    title_text = pieces[0] if pieces else str(raw_content)
    return _clean_session_title(title_text)


async def maybe_generate_session_title(
    session_id: str,
    store: SQLiteSessionStore,
    settings: Settings,
    user_question: str,
    title_provider: Callable[[Settings, str], Awaitable[str]] = generate_session_title,
) -> str:
    session = await store.load_session(session_id)
    if not session:
        return ""
    if session.title.strip():
        return session.title
    if session.msg_count != 1:
        return ""

    title = _clean_session_title(await title_provider(settings, user_question))
    if not title:
        return ""

    await store.update_title(session_id, title)
    logger.info("Generated session title for %s: %s", session_id, title)
    return title


# ── Summarizer node prompts ───────────────────────────────────────────────────

SUMMARIZER_SYSTEM_PROMPT = """You are a conversation memory assistant.

Given:
1. The PRIOR SUMMARY (may be empty for the first summary)
2. The RECENT MESSAGES to compress

Produce a single concise paragraph (3-6 sentences) that:
- Covers what the user asked about and what was found/answered
- Mentions key topics, document sources cited, and decisions made
- Merges naturally with the prior summary (do not repeat the same facts)
- Is written in past tense, third person ("The user asked...", "The assistant retrieved...")
- Does NOT include greetings, filler, or meta-commentary

Output ONLY the merged summary paragraph. Nothing else."""


# ── Summarizer node ───────────────────────────────────────────────────────────

async def summarizer_node(state: RAGState, config: RunnableConfig) -> dict:
    """
    Compress current messages into a rolling session_summary, save to DB,
    trim active messages to last 2 (+ summary SystemMessage) to fight
    context-window bloat.
    """
    store: SQLiteSessionStore = config["configurable"]["session_store"]
    settings: Settings = config["configurable"]["settings"]
    logger.info(
        "Summarizer node invoked: messages=%d session=%s",
        len(state["messages"]),
        state["session_id"],
    )

    llm = get_chat_model(settings, purpose="summarizer")

    prior = state["session_summary"] or "(none)"
    recent_text = "\n".join(
        f"{type(m).__name__}: {m.content[:400]}"
        for m in state["messages"]
        if not isinstance(m, SystemMessage)
    )

    response = await llm.ainvoke([
        SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"PRIOR SUMMARY:\n{prior}\n\n"
            f"RECENT MESSAGES:\n{recent_text}"
        )),
    ])

    new_summary = response.content.strip()
    await store.update_summary(state["session_id"], new_summary)
    logger.info("Summarizer updated session summary")

    # Keep only last 2 non-system messages to prevent context bloat
    non_system = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    kept = non_system[-2:] if len(non_system) >= 2 else non_system

    return {
        "session_summary": new_summary,
        "messages_since_last_summary": 0,
        "messages": [
            SystemMessage(content=f"[SESSION CONTEXT]\n{new_summary}")
        ] + kept,
    }
