from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from schemas import (
    ChatRequest,
    ChatResponse,
    SessionCreateResponse,
    SessionDetailResponse,
    SessionListResponse,
)
from services.chat_service import ChatService
from services.session_service import SessionService

router = APIRouter(prefix="/api/v1", tags=["agent"])


def get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Chat service is not ready")
    return service


def get_session_service(request: Request) -> SessionService:
    service = getattr(request.app.state, "session_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Session service is not ready")
    return service


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/sessions", response_model=SessionCreateResponse)
async def create_session(
    session_service: SessionService = Depends(get_session_service),
):
    import uuid

    session_id = f"session-{uuid.uuid4().hex[:8]}"
    await session_service.ensure_session(session_id)
    return SessionCreateResponse(session_id=session_id, created=True)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    session_service: SessionService = Depends(get_session_service),
):
    items = await session_service.list_sessions(limit=limit)
    return SessionListResponse(items=items, count=len(items))


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
):
    detail = await session_service.get_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return detail


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    try:
        return await chat_service.chat(payload.message, payload.session_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
