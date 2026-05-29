from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from core.config import Settings
from core.memory import SQLiteSessionStore, load_session_into_state
from core.state import RAGState
from nodes.graph import build_graph
from utils.mcp_result_utils import extract_mcp_text
from schemas import ChatResponse
from services.session_service import SessionService

logger = logging.getLogger(__name__)


class ChatService:
    """Coordinates graph execution and session persistence for the FastAPI layer."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.graph = None
        self.session_store: SQLiteSessionStore | None = None
        self.session_service: SessionService | None = None
        self._run_config: RunnableConfig | None = None

    @classmethod
    async def create(cls, settings: Settings) -> "ChatService":
        self = cls(settings)
        graph, _, session_store = await build_graph(settings)
        self.graph = graph
        self.session_store = session_store
        self.session_service = SessionService(session_store)
        return self

    def configure_request(self, session_id: str) -> RunnableConfig:
        if self.session_store is None:
            raise RuntimeError("ChatService is not initialized")
        return RunnableConfig(
            configurable={
                "session_store": self.session_store,
                "settings": self.settings,
                "thread_id": session_id,
            }
        )

    async def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        if self.graph is None or self.session_store is None or self.session_service is None:
            raise RuntimeError("ChatService is not initialized")

        session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        session_record, created_new_session = await self.session_service.ensure_session(session_id)

        # Load a compact, session-aware state from SQLite before every turn.
        state = await load_session_into_state(session_id, self.session_store, self.settings)

        # Persist the human turn immediately so the session is durable even if graph execution fails.
        await self.session_store.save_message(session_id, "human", message)

        # Append the current user turn to the in-memory graph state.
        state["messages"].append(HumanMessage(content=message))
        state["query"] = message
        state["messages_since_last_summary"] += 1

        logger.info("Chat request: session_id=%s created=%s message_len=%d", session_id, created_new_session, len(message))

        result = await self.graph.ainvoke(state, config=self.configure_request(session_id))
        answer = self._extract_answer(result)

        await self.session_store.save_message(session_id, "ai", answer)

        summary = str(result.get("session_summary") or state.get("session_summary") or session_record.summary or "")
        metadata = {
            "route": result.get("route", state.get("route", "direct")),
            "iterations": result.get("iterations", state.get("iterations", 0)),
            "retrieval_attempts": result.get("retrieval_attempts", state.get("retrieval_attempts", 0)),
            "retrieval_done": result.get("retrieval_done", state.get("retrieval_done", False)),
            "web_search_done": result.get("web_search_done", state.get("web_search_done", False)),
            "tool_calls_made": result.get("tool_calls_made", state.get("tool_calls_made", [])),
        }

        return ChatResponse(
            session_id=session_id,
            response=answer,
            created_new_session=created_new_session,
            summary=summary,
            route=str(metadata["route"]),
            iterations=int(metadata["iterations"] or 0),
            retrieval_attempts=int(metadata["retrieval_attempts"] or 0),
            tool_calls_made=list(metadata["tool_calls_made"] or []),
            metadata=metadata,
        )

    def _extract_answer(self, result: dict[str, Any]) -> str:
        messages = result.get("messages") or []
        if messages:
            final_msg = messages[-1]
            raw_answer = getattr(final_msg, "content", None)
            if raw_answer is None:
                raw_answer = getattr(final_msg, "text", None)
            if raw_answer is None:
                raw_answer = getattr(final_msg, "answer", None)
            if raw_answer is None:
                raw_answer = final_msg
        else:
            raw_answer = result.get("final_answer") or "No response generated."

        parts = extract_mcp_text(raw_answer)
        return "\n\n".join(parts) if parts else str(raw_answer)
