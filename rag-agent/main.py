from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from services.chat_service import ChatService
from services.session_service import SessionService
from core.config import Settings
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    setup_logging(settings)

    logger.info("Initializing FastAPI agent service")
    chat_service = await ChatService.create(settings)
    session_service = chat_service.session_service

    app.state.settings = settings
    app.state.chat_service = chat_service
    app.state.session_service = session_service

    logger.info("Agent service ready")
    yield

    logger.info("Shutting down FastAPI agent service")


app = FastAPI(
    title="Agentic RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)
