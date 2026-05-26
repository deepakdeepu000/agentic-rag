"""
Entry-point for the document-ingestion service.

Start-up sequence
-----------------
1. Configure structured logging (level from env LOG_LEVEL).
2. Load IngestionConfig (fields can be overridden via environment variables).
3. Boot the IngestionPipeline.
4. Scan the watch folder for pre-existing files (startup catch-up).
5. Start the file-system watcher.
6. Run a background thread that retries failed files every 5 minutes.
7. Log a periodic diagnostics summary every 60 seconds.
8. Block until interrupted; cleanly stop the observer on shutdown.
"""
import signal
import threading
import time
import logging
from utils.logging import _configure_logging

from config.config import IngestionConfig
from services.pipeline import IngestionPipeline
from retrival.watcher import start_watcher


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

def _retry_worker(pipeline: IngestionPipeline, interval: int = 300) -> None:
    """Periodically re-queue failed files. Runs as a daemon thread."""
    log.info("Retry worker started (interval=%ds)", interval)
    while True:
        time.sleep(interval)
        log.debug("Retry worker: checking for failed files …")
        try:
            pipeline.retry_failed()
        except Exception:
            log.exception("Retry worker encountered an unhandled error")


def _diagnostics_worker(pipeline: IngestionPipeline, interval: int = 60) -> None:
    """Log a health summary every `interval` seconds. Runs as a daemon thread."""
    log.info("Diagnostics worker started (interval=%ds)", interval)
    while True:
        time.sleep(interval)
        try:
            health = pipeline.health()
            log.info(
                "DIAGNOSTICS | state=%s | chroma_chunks=%s",
                health["state_store"],
                health["chroma_chunk_count"],
            )
        except Exception:
            log.exception("Diagnostics worker encountered an error")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Config ──────────────────────────────────────────────────────
    config = IngestionConfig()
    _configure_logging(config.log_level)    # re-apply now we have config
    config.log_summary()

    # ── 2. Pipeline ────────────────────────────────────────────────────
    pipeline = IngestionPipeline(config)

    # ── 3. File-ready callback ─────────────────────────────────────────
    def on_file_ready(file_path: str, file_hash: str) -> None:
        try:
            pipeline.process(file_path, file_hash)
        except Exception:
            log.exception("Unhandled error in on_file_ready for %s", file_path)

    # ── 4 + 5. Start watcher (includes startup scan) ───────────────────
    observer = start_watcher(config, on_file_ready)

    # ── 6. Background: retry loop ─────────────────────────────────────
    threading.Thread(
        target=_retry_worker,
        args=(pipeline,),
        daemon=True,
        name="retry-worker",
    ).start()

    # ── 7. Background: diagnostics ────────────────────────────────────
    threading.Thread(
        target=_diagnostics_worker,
        args=(pipeline,),
        daemon=True,
        name="diagnostics-worker",
    ).start()

    # ── 8. Graceful shutdown on SIGTERM (Docker stop) ─────────────────
    def _handle_sigterm(signum, frame):
        log.info("SIGTERM received — shutting down …")
        observer.stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    log.info("Ingestion service running. Press Ctrl-C to stop.")
    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — stopping observer …")
        observer.stop()

    observer.join()
    log.info("Observer stopped. Exiting.")


if __name__ == "__main__":
    main()