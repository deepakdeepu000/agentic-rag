# ingestion/models.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class RawDocument:
    """Output of extractor — one per file."""
    file_path: str
    file_hash: str          # SHA-256 of raw file bytes
    doc_type: str           # pdf, docx, txt, etc.
    mime_type: str
    filename: str
    raw_text: str
    page_count: Optional[int] = None
    extracted_at: str = ""  # ISO timestamp

@dataclass
class Chunk:
    """One chunk ready for embedding and upsert."""
    chunk_id: str           # deterministic: sha256(file_hash + chunk_index)
    file_hash: str
    file_path: str
    filename: str
    doc_type: str
    chunk_index: int
    total_chunks: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    metadata_boost: float = 0.0