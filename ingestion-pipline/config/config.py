import os
import logging
from dataclasses import dataclass, field
from typing import Dict

log = logging.getLogger(__name__)


def _env(key: str, default: str) -> str:
    """Read an env var; log when a non-default value is active."""
    val = os.environ.get(key, default)
    if val != default:
        log.debug("Config override via env: %s=%s", key, val)
    return val


@dataclass
class ChunkConfig:
    """
    Token-aware chunk config.

    target_tokens: preferred chunk size
    overlap_tokens: semantic overlap between adjacent chunks
    min_tokens: do not emit chunks smaller than this unless necessary
    max_tokens: hard limit before forced sub-splitting
    splitter: kept for compatibility with your existing code
    """
    target_tokens: int
    overlap_tokens: int
    min_tokens: int = 80
    max_tokens: int = 650
    embedding_max_tokens: int = 512
    splitter: str = "semantic"

    @property
    def chunk_size(self) -> int:
        return self.target_tokens

    @property
    def chunk_overlap(self) -> int:
        return self.overlap_tokens


@dataclass
class RetrievalConfig:
    """Query-time scoring and reranking settings."""
    use_hybrid: bool = True
    dense_weight: float = 0.70
    sparse_weight: float = 0.30
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 8
    min_final_score: float = 0.10


@dataclass
class IngestionConfig:
    # --- paths -----------------------------------------------------------
    watch_folder: str = field(default_factory=lambda: _env("INGESTION_WATCH_FOLDER", "./data"))
    chroma_persist_dir: str = field(default_factory=lambda: _env("INGESTION_CHROMA_PERSIST_DIR", "./chroma_db"))
    state_db_path: str = field(default_factory=lambda: _env("INGESTION_STATE_DB_PATH", "./ingestion_state.db"))

    # --- file handling ---------------------------------------------------
    supported_extensions: tuple = (
        ".pdf", ".docx", ".txt", ".md",
        ".html", ".csv",
    )
    stability_wait_seconds: float = field(default_factory=lambda: float(_env("INGESTION_STABILITY_WAIT", "2.0")))
    stability_polls: int = field(default_factory=lambda: int(_env("INGESTION_STABILITY_POLLS", "3")))

    # --- embedding -------------------------------------------------------
    embedding_model: str = field(default_factory=lambda: _env("INGESTION_EMBEDDING_MODEL", "mxbai-embed-large"))
    embedding_batch_size: int = field(default_factory=lambda: int(_env("INGESTION_EMBEDDING_BATCH_SIZE", "1")))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))

    # --- vector store ----------------------------------------------------
    chroma_collection: str = field(default_factory=lambda: _env("INGESTION_CHROMA_COLLECTION", "documents"))

    # --- pipeline --------------------------------------------------------
    max_retries: int = field(default_factory=lambda: int(_env("INGESTION_MAX_RETRIES", "3")))

    # --- observability ---------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # --- retrieval -------------------------------------------------------
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # --- per-doc-type chunk configs -------------------------------------
    chunk_configs: Dict[str, ChunkConfig] = field(default_factory=lambda: {
        # Dense prose
        "pdf":   ChunkConfig(target_tokens=450, overlap_tokens=80, min_tokens=120, max_tokens=650),
        "docx":  ChunkConfig(target_tokens=450, overlap_tokens=80, min_tokens=120, max_tokens=650),

        # Mixed text / markdown / html
        "txt":   ChunkConfig(target_tokens=220, overlap_tokens=60, min_tokens=80,  max_tokens=300, embedding_max_tokens=512),
        "md":    ChunkConfig(target_tokens=220, overlap_tokens=60, min_tokens=80,  max_tokens=300, embedding_max_tokens=512),
        "html":  ChunkConfig(target_tokens=220, overlap_tokens=60, min_tokens=80,  max_tokens=300, embedding_max_tokens=512),

        # Structured rows
        "csv":   ChunkConfig(target_tokens=140, overlap_tokens=20, min_tokens=40,  max_tokens=220, embedding_max_tokens=512),

        # OCR / image text
        "image": ChunkConfig(target_tokens=220, overlap_tokens=40, min_tokens=60,  max_tokens=320, embedding_max_tokens=512),
    })

    def log_summary(self) -> None:
        """Emit a structured summary of the active configuration at startup."""
        log.info(
            "IngestionConfig loaded | watch=%s chroma=%s state_db=%s model=%s batch=%d retries=%d log_level=%s",
            self.watch_folder,
            self.chroma_persist_dir,
            self.state_db_path,
            self.embedding_model,
            self.embedding_batch_size,
            self.max_retries,
            self.log_level,
        )