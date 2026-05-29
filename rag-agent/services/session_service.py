from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from dataclasses import asdict
from typing import Any

from core.memory import SQLiteSessionStore, SessionRecord
from schemas import SessionDetailResponse, SessionListItem, SessionMessage


class SessionService:
    """Session read/write helpers built on top of the project's SQLiteSessionStore."""

    def __init__(self, store: SQLiteSessionStore):
        self.store = store
        self.db_path = store.db_path

    async def ensure_session(self, session_id: str) -> tuple[SessionRecord, bool]:
        existing = await self.store.load_session(session_id)
        if existing is not None:
            return existing, False
        created = await self.store.create_session(session_id)
        return created, True

    async def list_sessions(self, limit: int = 20) -> list[SessionListItem]:
        records = await self.store.list_sessions(limit=limit)
        return [SessionListItem(**asdict(record)) for record in records]

    async def get_session_detail(self, session_id: str, limit: int | None = None) -> SessionDetailResponse | None:
        return await asyncio.to_thread(self._get_session_detail_sync, session_id, limit)

    async def get_all_messages(self, session_id: str) -> list[SessionMessage]:
        detail = await self.get_session_detail(session_id)
        return detail.messages if detail else []

    def _get_session_detail_sync(self, session_id: str, limit: int | None = None) -> SessionDetailResponse | None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row

            session_row = conn.execute(
                "SELECT session_id, summary, msg_count, created_at, updated_at "
                "FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if session_row is None:
                return None

            params: list[Any] = [session_id]
            sql = (
                "SELECT id, role, content, created_at "
                "FROM messages WHERE session_id = ? ORDER BY id ASC"
            )
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)

            rows = conn.execute(sql, tuple(params)).fetchall()

        messages = [
            SessionMessage(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

        return SessionDetailResponse(
            session_id=session_row["session_id"],
            summary=session_row["summary"],
            msg_count=session_row["msg_count"],
            created_at=session_row["created_at"],
            updated_at=session_row["updated_at"],
            messages=messages,
        )
