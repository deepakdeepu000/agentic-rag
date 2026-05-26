

import hashlib
import logging
import re
from typing import List, Tuple

try:
    import tiktoken
except Exception:
    tiktoken = None

from .models import RawDocument, Chunk
from config.config import IngestionConfig, ChunkConfig

log = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")

_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER

    if tiktoken is None:
        _ENCODER = False
        return None

    try:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = False
        return None

    return _ENCODER


def _token_len(text: str) -> int:
    if not text:
        return 0

    enc = _get_encoder()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass

    return max(1, len(text.split()))


def _chunk_id(file_hash: str, section_path: str, text: str) -> str:
    """
    Stable ID:
    same file + same section path + same normalized text -> same ID.
    """
    digest = hashlib.sha256(f"{file_hash}::{section_path}::{text}".encode("utf-8")).hexdigest()
    return digest[:32]


def _split_blocks(text: str) -> List[str]:
    return [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]


def _extract_sections(text: str) -> List[Tuple[List[str], str]]:
    """
    Split markdown-like documents into heading-aware sections.
    For plain text, returns one section with an empty path.
    """
    lines = text.splitlines()
    sections: List[Tuple[List[str], str]] = []

    heading_stack: List[str] = []
    buffer: List[str] = []

    def flush_buffer():
        if buffer:
            body = "\n".join(buffer).strip()
            if body:
                sections.append((heading_stack.copy(), body))
            buffer.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush_buffer()
            level = len(m.group(1))
            title = m.group(2).strip()

            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
            continue

        buffer.append(line)

    flush_buffer()

    if not sections:
        return [([], text.strip())]

    return sections


def _context_prefix(doc: RawDocument, section_path: List[str]) -> str:
    parts = [f"Document: {doc.filename}"]
    if section_path:
        parts.append("Section: " + " > ".join(section_path))
    return "\n".join(parts)


def _tail_for_overlap(text: str, overlap_tokens: int) -> str:
    """
    Take a semantic tail from the previous chunk.
    """
    if overlap_tokens <= 0 or not text.strip():
        return ""

    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        words = text.split()
        return " ".join(words[-overlap_tokens:]).strip()

    tail: List[str] = []
    total = 0

    for sent in reversed(sentences):
        n = _token_len(sent)
        if not tail or total + n <= overlap_tokens:
            tail.insert(0, sent)
            total += n
        else:
            break

    return " ".join(tail).strip()


def _split_overlong_block(block: str, target_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Split a very large paragraph into smaller semantic windows.
    """
    sentences = [s.strip() for s in _SENTENCE_RE.split(block) if s.strip()]

    if len(sentences) <= 1:
        words = block.split()
        if not words:
            return []

        step = max(1, target_tokens - overlap_tokens)
        out = []
        for start in range(0, len(words), step):
            part = " ".join(words[start:start + target_tokens]).strip()
            if part:
                out.append(part)
        return out

    out: List[str] = []
    current: List[str] = []

    for sent in sentences:
        candidate = " ".join(current + [sent]).strip()
        if current and _token_len(candidate) > target_tokens:
            chunk = " ".join(current).strip()
            if chunk:
                out.append(chunk)
            tail = _tail_for_overlap(chunk, overlap_tokens)
            current = [tail, sent] if tail else [sent]
        else:
            current.append(sent)

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            out.append(chunk)

    return out


def _split_text_to_limit(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Split text into pieces that stay within a hard token ceiling."""
    text = text.strip()
    if not text:
        return []

    max_tokens = max(1, max_tokens)
    overlap_tokens = max(0, overlap_tokens)

    enc = _get_encoder()
    if enc:
        try:
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return [text]

            step = max(1, max_tokens - overlap_tokens)
            pieces: List[str] = []
            for start in range(0, len(tokens), step):
                part = enc.decode(tokens[start:start + max_tokens]).strip()
                if part:
                    pieces.append(part)
            return pieces
        except Exception:
            pass

    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    step = max(1, max_tokens - overlap_tokens)
    pieces: List[str] = []
    for start in range(0, len(words), step):
        part = " ".join(words[start:start + max_tokens]).strip()
        if part:
            pieces.append(part)
    return pieces


def _split_with_prefix(prefix: str, body: str, hard_limit: int, overlap_tokens: int) -> List[str]:
    """Ensure the final chunk text, including the document prefix, stays under the hard limit."""
    prefix = prefix.strip()
    body = body.strip()
    full_text = f"{prefix}\n\n{body}".strip()

    if _token_len(full_text) <= hard_limit:
        return [full_text]

    if hard_limit <= 1:
        return [full_text]

    prefix_tokens = _token_len(prefix)
    body_limit = max(1, hard_limit - prefix_tokens - 8)
    body_parts = _split_text_to_limit(body, body_limit, overlap_tokens)

    if len(body_parts) == 1 and body_parts[0].strip() == body:
        tighter_limit = max(1, body_limit // 2)
        if tighter_limit < body_limit:
            body_parts = _split_text_to_limit(body, tighter_limit, overlap_tokens)

    result: List[str] = []
    for part in body_parts:
        candidate = f"{prefix}\n\n{part}".strip()
        if _token_len(candidate) <= hard_limit:
            result.append(candidate)
        else:
            result.extend(_split_with_prefix(prefix, part, max(1, hard_limit // 2), overlap_tokens))

    return result if result else [full_text]


def _make_section_chunks(
    doc: RawDocument,
    cfg: ChunkConfig,
    section_path: List[str],
    section_text: str,
) -> List[Tuple[str, str]]:
    """
    Return [(section_path_str, chunk_text), ...]
    """
    prefix = _context_prefix(doc, section_path)
    blocks = _split_blocks(section_text)

    emitted: List[Tuple[str, str]] = []
    current_parts: List[str] = []
    carry = ""
    has_new_content = False
    hard_limit = cfg.embedding_max_tokens

    def current_body() -> str:
        parts = [p for p in current_parts if p.strip()]
        return "\n\n".join(parts).strip()

    def flush():
        nonlocal carry, current_parts, has_new_content
        body = current_body()
        if has_new_content and body:
            section_path_str = " > ".join(section_path) if section_path else ""
            emitted.extend(
                (section_path_str, chunk_text)
                for chunk_text in _split_with_prefix(prefix, body, hard_limit, cfg.overlap_tokens)
            )
            carry = _tail_for_overlap(body, cfg.overlap_tokens)
        current_parts = [carry] if carry else []
        has_new_content = False

    for block in blocks:
        pieces = [block] if _token_len(block) <= cfg.max_tokens else _split_overlong_block(block, cfg.target_tokens, cfg.overlap_tokens)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            candidate = "\n\n".join([p for p in current_parts + [piece] if p]).strip()
            if current_parts and has_new_content and _token_len(candidate) > cfg.target_tokens:
                flush()

            current_parts.append(piece)
            has_new_content = True

            # Safety: if a piece is still too large, force it to stand alone.
            if _token_len(piece) > cfg.max_tokens:
                flush()

    flush()
    return emitted


def chunk_document(doc: RawDocument, config: IngestionConfig) -> List[Chunk]:
    """
    Token-aware, heading-aware, overlap-preserving chunking.
    Assumes doc.raw_text has already been cleaned.
    """
    cfg = config.chunk_configs.get(doc.doc_type, config.chunk_configs["txt"])

    if not doc.raw_text or not doc.raw_text.strip():
        log.warning("No text to chunk for %s", doc.file_path)
        return []

    log.debug(
        "Chunking %s: doc_type=%s target=%d overlap=%d max=%d",
        doc.filename, doc.doc_type, cfg.target_tokens, cfg.overlap_tokens, cfg.max_tokens,
    )

    sections = _extract_sections(doc.raw_text)

    raw_chunks: List[Tuple[str, str]] = []
    for section_path, section_text in sections:
        raw_chunks.extend(_make_section_chunks(doc, cfg, section_path, section_text))

    if not raw_chunks:
        log.warning("No chunks produced for %s", doc.file_path)
        return []

    total = len(raw_chunks)
    chunks: List[Chunk] = []

    for i, (section_path_str, text) in enumerate(raw_chunks):
        body = text.strip()
        if not body:
            continue

        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc.file_hash, section_path_str, body),
                file_hash=doc.file_hash,
                file_path=doc.file_path,
                filename=doc.filename,
                doc_type=doc.doc_type,
                chunk_index=i,
                total_chunks=total,
                text=body,
                metadata={
                    "filename": doc.filename,
                    "file_path": doc.file_path,
                    "file_hash": doc.file_hash,
                    "doc_type": doc.doc_type,
                    "mime_type": doc.mime_type,
                    "chunk_index": i,
                    "total_chunks": total,
                    "page_count": doc.page_count,
                    "extracted_at": doc.extracted_at,
                    "section_path": section_path_str,
                    "chunk_strategy": "semantic_heading_token",
                    "target_tokens": cfg.target_tokens,
                    "overlap_tokens": cfg.overlap_tokens,
                },
            )
        )

    log.info(
        "%s: %d chunks produced (target=%d overlap=%d max=%d)",
        doc.filename, len(chunks), cfg.target_tokens, cfg.overlap_tokens, cfg.max_tokens,
    )
    return chunks