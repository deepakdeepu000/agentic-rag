"""logging_utils.py - Shared logging setup for terminal and file output."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from core.config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure root logging for console and file output."""
    log_path = Path(settings.log_file_path)
    if log_path.parent and log_path.parent != Path("."):
        log_path.parent.mkdir(parents=True, exist_ok=True)

    console_level_name = (settings.log_level or "INFO").upper()
    console_level = getattr(logging, console_level_name, logging.INFO)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)
        elif isinstance(handler, logging.StreamHandler):
            handler.setLevel(console_level)