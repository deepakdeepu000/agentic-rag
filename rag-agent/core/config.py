"""
config.py — All settings loaded from environment / .env file.
Never hard-code secrets; always read from here.
"""
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Ollama ─────────────────────────────────────────────────────────
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.2:3b", alias="OLLAMA_MODEL")
    # Smaller/faster model for the summarizer node (keeps costs low)
    ollama_summarizer_model: str = Field("llama3.2:3b", alias="OLLAMA_SUMMARIZER_MODEL")

    # ── Chroma ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field("./chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field("docs", alias="CHROMA_COLLECTION")
    # Optional comma-separated list to query multiple collections in isolation.
    chroma_collections: list[str] = Field(default_factory=list, alias="CHROMA_COLLECTIONS")

    # ── Embeddings ────────────────────────────────────────────────────
    # EMBEDDING_PROVIDER: "huggingface" (default) | "ollama"
    #   huggingface — local sentence-transformers model, runs fully offline
    #   ollama      — uses a model already pulled in your Ollama instance
    embedding_provider: str = Field("ollama", alias="EMBEDDING_PROVIDER")

    # HuggingFace provider settings
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field("cpu", alias="EMBEDDING_DEVICE")

    # Ollama provider settings
    ollama_embedding_model: str = Field("mxbai-embed-large", alias="OLLAMA_EMBEDDING_MODEL")

    # ── SQLite session store ──────────────────────────────────────────
    sqlite_db_path: str = Field("./sessions.db", alias="SQLITE_DB_PATH")

    # ── Remote MCP (SSE) ──────────────────────────────────────────────
    # Prefer REMOTE_MCP_SERVERS for multiple endpoints.
    # Keep REMOTE_MCP_URL for backward compatibility with a single SSE endpoint.
    remote_mcp_servers: list[dict[str, str]] = Field([],
        alias="REMOTE_MCP_SERVERS",
    )
    remote_mcp_url: str = Field("", alias="REMOTE_MCP_URL")
    remote_mcp_token: str = Field("", alias="REMOTE_MCP_TOKEN")
    # Optional extra header (e.g. tenant ID)
    remote_mcp_tenant_id: str = Field("", alias="REMOTE_MCP_TENANT_ID")

    # ── LLM provider routing ──────────────────────────────────────────
    llm_provider: str = Field("gemini", alias="LLM_PROVIDER")
    coordinator_temperature: float = Field(0, alias="COORDINATOR_TEMPERATURE")
    summarizer_temperature: float = Field(0, alias="SUMMARIZER_TEMPERATURE")
    router_temperature: float = Field(0, alias="ROUTER_TEMPERATURE")

    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")

    gemini_model: str = Field("gemini-3.1-flash-lite", alias="GEMINI_MODEL")
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")

    # Optional override for router-only model
    router_model: str = Field("", alias="ROUTER_MODEL")

    # ── Web search tool configuration ─────────────────────────────────
    web_search_tool_name: str = Field("web_search", alias="WEB_SEARCH_TOOL_NAME")

    # ── Graph control ─────────────────────────────────────────────────
    max_iterations: int = Field(5, alias="MAX_ITERATIONS")
    max_retrieval_attempts: int = Field(3, alias="MAX_RETRIEVAL_ATTEMPTS")
    retrieval_k: int = Field(5, alias="RETRIEVAL_K")

    # ── Logging ───────────────────────────────────────────────────────
    log_file_path: str = Field("./rag_debug.log", alias="LOG_FILE_PATH")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @field_validator("chroma_collections", mode="before")
    @classmethod
    def _parse_chroma_collections(cls, value):
        if value is None or value == "":
            return []
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            return [part for part in parts if part]
        return value
