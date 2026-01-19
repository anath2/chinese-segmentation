import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db_path() -> Path:
    """
    Returns the sqlite DB path.

    Defaults to: <repo>/app/data/transcriber.db
    Override with: TRANSCRIBER_DB_PATH
    """
    env = os.getenv("TRANSCRIBER_DB_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parent / "data" / "transcriber.db").resolve()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def db_conn() -> Iterator[sqlite3.Connection]:
    path = get_db_path()
    _ensure_parent_dir(path)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        # Reasonable defaults for small local apps.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db_conn() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)"
        )
        current = conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations"
        ).fetchone()["v"]

        migrations: list[tuple[int, str]] = [
            (1, _migration_001()),
        ]

        for version, sql in migrations:
            if version > int(current):
                conn.executescript(sql)
                conn.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))


def _migration_001() -> str:
    # Keep it as executescript-friendly SQL.
    return """
CREATE TABLE IF NOT EXISTS texts (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  source_type TEXT NOT NULL, -- 'text' | 'ocr' | future
  raw_text TEXT NOT NULL,
  normalized_text TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS segments (
  id TEXT PRIMARY KEY,
  text_id TEXT NOT NULL,
  paragraph_idx INTEGER NOT NULL,
  seg_idx INTEGER NOT NULL,
  segment_text TEXT NOT NULL,
  pinyin TEXT NOT NULL DEFAULT '',
  english TEXT NOT NULL DEFAULT '',
  provider_meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY(text_id) REFERENCES texts(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_segments_text_id ON segments(text_id);

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  text_id TEXT,
  segment_id TEXT,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  FOREIGN KEY(text_id) REFERENCES texts(id) ON DELETE SET NULL,
  FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_text_id ON events(text_id);

CREATE TABLE IF NOT EXISTS vocab_items (
  id TEXT PRIMARY KEY,
  headword TEXT NOT NULL,
  pinyin TEXT NOT NULL DEFAULT '',
  english TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'unknown', -- unknown|learning|known
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_vocab_items_key
  ON vocab_items(headword, pinyin, english);

CREATE TABLE IF NOT EXISTS vocab_occurrences (
  id TEXT PRIMARY KEY,
  vocab_item_id TEXT NOT NULL,
  text_id TEXT,
  segment_id TEXT,
  snippet TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  FOREIGN KEY(vocab_item_id) REFERENCES vocab_items(id) ON DELETE CASCADE,
  FOREIGN KEY(text_id) REFERENCES texts(id) ON DELETE SET NULL,
  FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_vocab_occ_vocab_item_id ON vocab_occurrences(vocab_item_id);
CREATE INDEX IF NOT EXISTS idx_vocab_occ_text_id ON vocab_occurrences(text_id);

CREATE TABLE IF NOT EXISTS srs_state (
  vocab_item_id TEXT PRIMARY KEY,
  due_at TEXT,
  interval_days REAL NOT NULL DEFAULT 0,
  ease REAL NOT NULL DEFAULT 2.5,
  reps INTEGER NOT NULL DEFAULT 0,
  lapses INTEGER NOT NULL DEFAULT 0,
  last_reviewed_at TEXT,
  FOREIGN KEY(vocab_item_id) REFERENCES vocab_items(id) ON DELETE CASCADE
);
"""


@dataclass(frozen=True)
class TextRecord:
    id: str
    created_at: str
    source_type: str
    raw_text: str
    normalized_text: str
    metadata: dict[str, Any]


def create_text(*, raw_text: str, source_type: str, metadata: dict[str, Any] | None) -> TextRecord:
    text_id = uuid.uuid4().hex
    created_at = _utc_now_iso()
    normalized_text = raw_text.strip()
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO texts (id, created_at, source_type, raw_text, normalized_text, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (text_id, created_at, source_type, raw_text, normalized_text, metadata_json),
        )

    return TextRecord(
        id=text_id,
        created_at=created_at,
        source_type=source_type,
        raw_text=raw_text,
        normalized_text=normalized_text,
        metadata=json.loads(metadata_json),
    )


def get_text(text_id: str) -> TextRecord | None:
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM texts WHERE id = ?", (text_id,)).fetchone()
        if row is None:
            return None
        return TextRecord(
            id=row["id"],
            created_at=row["created_at"],
            source_type=row["source_type"],
            raw_text=row["raw_text"],
            normalized_text=row["normalized_text"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )


def create_event(
    *,
    event_type: str,
    text_id: str | None,
    segment_id: str | None,
    payload: dict[str, Any] | None,
) -> str:
    event_id = uuid.uuid4().hex
    ts = _utc_now_iso()
    payload_json = json.dumps(payload or {}, ensure_ascii=False)
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO events (id, ts, text_id, segment_id, event_type, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_id, ts, text_id, segment_id, event_type, payload_json),
        )
    return event_id


def save_vocab_item(
    *,
    headword: str,
    pinyin: str,
    english: str,
    text_id: str | None,
    segment_id: str | None,
    snippet: str | None,
) -> str:
    """
    Upsert-like behavior based on (headword, pinyin, english).
    Returns vocab_item_id.
    """
    now = _utc_now_iso()
    snippet = snippet or ""

    with db_conn() as conn:
        # Insert or ignore; then select.
        vocab_item_id = uuid.uuid4().hex
        conn.execute(
            """
            INSERT OR IGNORE INTO vocab_items (id, headword, pinyin, english, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'unknown', ?, ?)
            """,
            (vocab_item_id, headword, pinyin, english, now, now),
        )
        row = conn.execute(
            """
            SELECT id FROM vocab_items
            WHERE headword = ? AND pinyin = ? AND english = ?
            """,
            (headword, pinyin, english),
        ).fetchone()
        if row is None:
            # Extremely unlikely; fall back to the generated id.
            resolved_id = vocab_item_id
        else:
            resolved_id = row["id"]
            conn.execute(
                "UPDATE vocab_items SET updated_at = ? WHERE id = ?",
                (now, resolved_id),
            )

        occ_id = uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO vocab_occurrences (id, vocab_item_id, text_id, segment_id, snippet, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (occ_id, resolved_id, text_id, segment_id, snippet, now),
        )

    return resolved_id


def update_vocab_status(*, vocab_item_id: str, status: str) -> None:
    if status not in {"unknown", "learning", "known"}:
        raise ValueError("Invalid status")
    now = _utc_now_iso()
    with db_conn() as conn:
        conn.execute(
            "UPDATE vocab_items SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, vocab_item_id),
        )
