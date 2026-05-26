"""
Ingestion pipeline orchestrator.

Coordinates: extract → clean → chunk → embed → upsert.
Each step is logged with timing so slow stages are easy to identify.
"""
import logging
import time
from contextlib import contextmanager
from typing import Optional

from config.config import IngestionConfig
from ingestion.store_state import StateStore
from ingestion.extractor import extract
from core.cleaner import clean
from core.chunker import chunk_document
from ingestion.embedder import embed_chunks
from ingestion.chroma_store import (
    collection_name_for_path,
    get_or_create_collection_by_name,
    upsert_chunks,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

@contextmanager
def _timed(label: str):
    """Context manager that logs wall-clock time for a pipeline stage."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        log.debug("Stage '%s' completed in %.3fs", label, elapsed)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    def __init__(self, config: IngestionConfig) -> None:
        self.config = config
        self.state = StateStore(config.state_db_path)
        self.collections = {}
        log.info("IngestionPipeline ready")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, file_path: str, file_hash: str) -> None:
        """
        Full pipeline for a single file.
        Idempotent: same (file_path, file_hash) pair is a no-op.
        """
        if self.state.is_already_ingested(file_hash):
            log.info("Duplicate — already ingested, skipping: %s (%s…)", file_path, file_hash[:12])
            return

        log.info("Pipeline start: %s (hash=%s…)", file_path, file_hash[:12])
        t_start = time.perf_counter()

        for attempt in range(1, self.config.max_retries + 1):
            self.state.upsert(file_path, file_hash, "processing", attempt)
            success = self._run_once(file_path, file_hash, attempt)
            if success:
                elapsed = time.perf_counter() - t_start
                log.info(
                    "Pipeline complete: %s — attempt=%d total_time=%.2fs",
                    file_path, attempt, elapsed,
                )
                return
            if attempt < self.config.max_retries:
                backoff = 2 ** attempt
                log.warning(
                    "Pipeline attempt %d/%d failed for %s — backing off %ds",
                    attempt, self.config.max_retries, file_path, backoff,
                )
                time.sleep(backoff)

        log.error(
            "Pipeline exhausted %d retries for %s",
            self.config.max_retries, file_path,
        )

    def retry_failed(self) -> None:
        """Re-process all files currently in 'failed' state under the retry cap."""
        failed = self.state.get_failed_for_retry(self.config.max_retries)
        if not failed:
            log.debug("retry_failed: nothing eligible")
            return
        log.info("retry_failed: re-queuing %d file(s)", len(failed))
        for file_path, file_hash, _ in failed:
            self.process(file_path, file_hash)

    def health(self) -> dict:
        """Return a dict summarising pipeline state — for diagnostics."""
        stats = self.state.get_stats()
        chroma_count = {}
        for collection_name, collection in self.collections.items():
            try:
                chroma_count[collection_name] = collection.count()
            except Exception as exc:
                chroma_count[collection_name] = f"error: {exc}"
        return {
            "state_store": stats,
            "chroma_chunk_count": chroma_count,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_once(self, file_path: str, file_hash: str, attempt: int) -> bool:
        """
        Execute one full pipeline pass.
        Returns True on success, False on recoverable failure.
        Calls state.upsert() to record terminal outcomes.
        """
        try:
            # 1 ── Extract ──────────────────────────────────────────────
            with _timed("extract"):
                doc = extract(file_path, file_hash)

            if not doc or not doc.raw_text.strip():
                log.warning(
                    "Empty extraction for %s (attempt %d) — marking failed",
                    file_path, attempt,
                )
                self.state.upsert(
                    file_path, file_hash, "failed", attempt, "empty extraction"
                )
                return False   # no point retrying an empty file

            log.debug(
                "Extract OK: %s — type=%s chars=%d pages=%s",
                file_path, doc.doc_type, len(doc.raw_text), doc.page_count,
            )

            # 2 ── Clean ────────────────────────────────────────────────
            with _timed("clean"):
                doc.raw_text = clean(doc.raw_text)
            log.debug("Clean OK: chars_after=%d", len(doc.raw_text))

            # 3 ── Chunk ────────────────────────────────────────────────
            with _timed("chunk"):
                chunks = chunk_document(doc, self.config)

            if not chunks:
                log.warning("No chunks produced for %s", file_path)
                self.state.upsert(
                    file_path, file_hash, "failed", attempt, "no chunks produced"
                )
                return False

            log.debug("Chunk OK: %d chunks for %s", len(chunks), file_path)

            # 4 ── Embed ────────────────────────────────────────────────
            with _timed("embed"):
                chunks = embed_chunks(chunks, self.config)

            # 5 ── Upsert ───────────────────────────────────────────────
            with _timed("upsert"):
                collection = self._get_collection(file_path)
                n = upsert_chunks(chunks, collection)

            self.state.upsert(
                file_path, file_hash, "done", attempt, chunk_count=n
            )
            log.info("✓ Ingested: %s → %d chunks (attempt %d)", file_path, n, attempt)
            return True

        except Exception as exc:
            log.exception(
                "Pipeline error on attempt %d for %s: %s",
                attempt, file_path, exc,
            )
            self.state.upsert(file_path, file_hash, "failed", attempt, str(exc))
            return False

    def _get_collection(self, file_path: str):
        collection_name = collection_name_for_path(
            file_path,
            self.config.watch_folder,
            self.config.chroma_collection,
        )
        if collection_name not in self.collections:
            self.collections[collection_name] = get_or_create_collection_by_name(
                self.config,
                collection_name,
            )
        return self.collections[collection_name]