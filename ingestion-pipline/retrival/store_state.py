import sqlite3
import logging
import threading
from datetime import datetime, timezone
from typing import List, Tuple, Optional

log = logging.getLogger(__name__)

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ingestion_state (
    file_path     TEXT NOT NULL,
    file_hash     TEXT NOT NULL,
    status        TEXT NOT NULL,    -- 'pending' | 'processing' | 'done' | 'failed'
    attempt_count INTEGER DEFAULT 0,
    last_attempt  TEXT,             -- ISO-8601 UTC
    error_msg     TEXT,
    chunk_count   INTEGER,
    PRIMARY KEY (file_path, file_hash)
);
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_state_status ON ingestion_state(status);
"""


class StateStore:
    """
    Thread-safe SQLite state store.

    Uses a threading.Lock around every write so multiple pipeline threads
    don't stomp on each other when running concurrent retries.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=10,
        )
        # WAL mode: readers don't block writers and vice-versa
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute(CREATE_TABLE)
        self.conn.execute(CREATE_INDEX)
        self.conn.commit()
        log.info("StateStore initialised: %s", db_path)

    # ------------------------------------------------------------------
    # Reads (no lock needed — SQLite WAL allows concurrent reads)
    # ------------------------------------------------------------------

    def is_already_ingested(self, file_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM ingestion_state WHERE file_hash=? AND status='done'",
            (file_hash,),
        ).fetchone()
        result = row is not None
        log.debug("is_already_ingested(%s…) → %s", file_hash[:12], result)
        return result

    def get_failed_for_retry(self, max_retries: int) -> List[Tuple[str, str, int]]:
        rows = self.conn.execute(
            "SELECT file_path, file_hash, attempt_count "
            "FROM ingestion_state "
            "WHERE status='failed' AND attempt_count < ?",
            (max_retries,),
        ).fetchall()
        if rows:
            log.info("get_failed_for_retry: %d file(s) eligible", len(rows))
        return rows

    def get_stats(self) -> dict:
        """Return status counts — useful for health-check / metrics."""
        rows = self.conn.execute(
            "SELECT status, COUNT(*) FROM ingestion_state GROUP BY status"
        ).fetchall()
        stats = {row[0]: row[1] for row in rows}
        log.debug("StateStore stats: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert(
        self,
        file_path: str,
        file_hash: str,
        status: str,
        attempt_count: int = 0,
        error_msg: Optional[str] = None,
        chunk_count: Optional[int] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        log.debug(
            "StateStore.upsert: path=%s hash=%s… status=%s attempt=%d",
            file_path, file_hash[:12], status, attempt_count,
        )
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO ingestion_state
                    (file_path, file_hash, status, attempt_count,
                     last_attempt, error_msg, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path, file_hash) DO UPDATE SET
                    status        = excluded.status,
                    attempt_count = excluded.attempt_count,
                    last_attempt  = excluded.last_attempt,
                    error_msg     = excluded.error_msg,
                    chunk_count   = excluded.chunk_count
                """,
                (
                    file_path, file_hash, status, attempt_count,
                    now, error_msg, chunk_count,
                ),
            )
            self.conn.commit()

    def close(self) -> None:
        log.info("Closing StateStore connection: %s", self.db_path)
        self.conn.close()