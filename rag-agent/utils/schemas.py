from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    session_id: str | None = Field(default=None, description="Existing session ID to continue")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    created_new_session: bool = False
    summary: str = ""
    route: str = "direct"
    iterations: int = 0
    retrieval_attempts: int = 0
    tool_calls_made: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionCreateResponse(BaseModel):
    session_id: str
    created: bool = True


class SessionListItem(BaseModel):
    session_id: str
    summary: str = ""
    msg_count: int = 0
    created_at: str
    updated_at: str


class SessionListResponse(BaseModel):
    items: list[SessionListItem]
    count: int


class SessionMessage(BaseModel):
    id: int
    role: Literal["human", "ai", "system", "tool"]
    content: str
    created_at: str


class SessionDetailResponse(BaseModel):
    session_id: str
    summary: str = ""
    msg_count: int = 0
    created_at: str
    updated_at: str
    messages: list[SessionMessage] = Field(default_factory=list)
