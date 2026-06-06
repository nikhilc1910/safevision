"""
SQLite access layer. Plain sqlite3 — no ORM, no migrations library.
Three tables, no joins — SQLAlchemy would add abstraction without adding anything useful.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from vision.events import ViolationEvent

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class ViolationStore:
    def __init__(self, db_path: str | Path = "safevision.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA_PATH.read_text())

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert(self, event: ViolationEvent, shift_id: str | None = None) -> int:
        """Insert one violation. Returns the new row id."""
        ts = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat()
        x1, y1, x2, y2 = event.bbox if event.bbox else (None, None, None, None)

        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO violations
                    (camera_id, zone_id, vtype, confidence,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                     timestamp_utc, shift_id, frame_snap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event.camera_id, event.zone_id, event.vtype, event.confidence,
                 int(x1) if x1 else None, int(y1) if y1 else None,
                 int(x2) if x2 else None, int(y2) if y2 else None,
                 ts, shift_id, event.frame_snap),
            )
            return cur.lastrowid

    def recent(self, limit: int = 50, camera_id: str | None = None) -> list[dict]:
        """Fetch most recent violations as plain dicts for display."""
        query  = "SELECT camera_id, zone_id, vtype, confidence, timestamp_utc FROM violations"
        params: list = []

        if camera_id:
            query  += " WHERE camera_id = ?"
            params.append(camera_id)

        query += " ORDER BY timestamp_utc DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(r) for r in rows]

    def mark_false_positive(self, violation_id: int) -> None:
        """Operator-marked FP — feeds active learning loop in v2.0."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE violations SET false_positive = 1 WHERE id = ?",
                (violation_id,),
            )
