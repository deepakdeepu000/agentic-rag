# retrival/watcher.py
import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Callable

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config.config import IngestionConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def file_hash(path: str) -> str:
    """SHA-256 of file bytes. Used for dedup and chunk-ID generation."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def is_file_stable(path: str, wait: float, polls: int) -> bool:
    """
    Return True only when the file size is identical across `polls`
    consecutive checks separated by `wait` seconds.

    FIX (was broken): the original code re-assigned prev_size on every
    size change, which meant it never actually confirmed two consecutive
    equal measurements — it only checked the *last* pair. The corrected
    logic requires `polls` successive stable readings, not just one final
    match after any number of changes.
    """
    stable_count = 0
    prev_size: int = -1

    for _ in range(polls * 2):          # enough iterations to converge
        try:
            cur_size = os.path.getsize(path)
        except FileNotFoundError:
            log.debug("Stability check: file vanished — %s", path)
            return False

        if cur_size == prev_size:
            stable_count += 1
            log.debug(
                "Stability check: size=%d stable_count=%d/%d path=%s",
                cur_size, stable_count, polls, path,
            )
            if stable_count >= polls:
                return True
        else:
            # Size changed — reset the stable counter
            log.debug(
                "Stability check: size changed %d → %d, resetting count — %s",
                prev_size, cur_size, path,
            )
            stable_count = 0
            prev_size = cur_size

        time.sleep(wait)

    log.warning("Stability check: timed out after %d iterations — %s", polls * 2, path)
    return False


def _is_supported(path: str, config: IngestionConfig) -> bool:
    """Return True when the file's suffix is in the configured allow-list."""
    ext = Path(path).suffix.lower()
    supported = ext in config.supported_extensions
    if not supported:
        log.debug("Skipping unsupported extension '%s': %s", ext, path)
    return supported


# ---------------------------------------------------------------------------
# Event handler
# ---------------------------------------------------------------------------

class IngestionEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler that feeds newly created / modified files into
    the ingestion pipeline after confirming they are stable.
    """

    def __init__(
        self,
        config: IngestionConfig,
        on_file_ready: Callable[[str, str], None],
    ) -> None:
        super().__init__()
        self.config = config
        self.on_file_ready = on_file_ready   # callback(file_path, file_hash)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle(self, path: str) -> None:
        log.debug("Event triggered for: %s", path)

        if not _is_supported(path, self.config):
            return

        if not os.path.isfile(path):
            log.debug("Path is not a regular file, skipping: %s", path)
            return

        log.info("File detected, checking stability: %s", path)

        if not is_file_stable(
            path,
            self.config.stability_wait_seconds,
            self.config.stability_polls,
        ):
            log.warning("File not stable after stability check, skipping: %s", path)
            return

        try:
            fh = file_hash(path)
        except OSError as exc:
            log.error("Failed to hash file %s: %s", path, exc)
            return

        log.info(
            "File ready for ingestion: %s (hash=%s…)", path, fh[:12]
        )
        self.on_file_ready(path, fh)

    # ------------------------------------------------------------------
    # Watchdog callbacks
    # ------------------------------------------------------------------

    def on_created(self, event) -> None:
        if not event.is_directory:
            log.debug("on_created: %s", event.src_path)
            self._handle(event.src_path)

    def on_modified(self, event) -> None:
        if not event.is_directory:
            log.debug("on_modified: %s", event.src_path)
            self._handle(event.src_path)

    def on_moved(self, event) -> None:
        """Handle files renamed/moved into the watched directory."""
        if not event.is_directory:
            log.debug("on_moved: %s → %s", event.src_path, event.dest_path)
            self._handle(event.dest_path)


# ---------------------------------------------------------------------------
# Startup scan  (FIX: was completely missing)
# ---------------------------------------------------------------------------

def scan_existing_files(
    config: IngestionConfig,
    on_file_ready: Callable[[str, str], None],
) -> int:
    """
    Walk the watch folder at startup and submit every supported file that
    is already present.  This ensures files dropped while the service was
    offline are not silently ignored.

    Returns the number of files submitted.
    """
    watch_path = Path(config.watch_folder)
    if not watch_path.exists():
        log.warning("Watch folder does not exist yet, skipping startup scan: %s", watch_path)
        return 0

    submitted = 0
    log.info("Startup scan: walking %s for existing files …", watch_path)

    for file_path in watch_path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in config.supported_extensions:
            log.debug("Startup scan: skipping unsupported %s", file_path)
            continue

        try:
            fh = file_hash(str(file_path))
        except OSError as exc:
            log.error("Startup scan: failed to hash %s: %s", file_path, exc)
            continue

        log.info("Startup scan: submitting %s (hash=%s…)", file_path, fh[:12])
        on_file_ready(str(file_path), fh)
        submitted += 1

    log.info("Startup scan complete: %d file(s) submitted", submitted)
    return submitted


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def start_watcher(
    config: IngestionConfig,
    on_file_ready: Callable[[str, str], None],
) -> Observer:
    """
    1. Scan any files already present in the watch folder.
    2. Start a watchdog Observer for live additions / modifications.

    Returns the running Observer so the caller can join / stop it.
    """
    watch_path = Path(config.watch_folder)
    watch_path.mkdir(parents=True, exist_ok=True)

    # Process files that arrived before this process started
    scan_existing_files(config, on_file_ready)

    handler = IngestionEventHandler(config, on_file_ready)
    observer = Observer()
    observer.schedule(handler, str(watch_path), recursive=True)
    observer.start()

    log.info(
        "Watcher active: folder=%s  extensions=%s",
        watch_path,
        ", ".join(sorted(config.supported_extensions)),
    )
    return observer