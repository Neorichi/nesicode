"""SQLite persistence (one file, WAL mode, a single shared connection guarded by a lock)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS user_keys (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    secret TEXT NOT NULL,
    PRIMARY KEY (user_id, provider)
);
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT 'general',
    executor_model TEXT NOT NULL,
    improver_model TEXT NOT NULL,
    check_command TEXT NOT NULL DEFAULT '',
    max_steps INTEGER NOT NULL DEFAULT 60,
    harness_version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS harness_versions (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    files TEXT NOT NULL,
    source TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    run_id INTEGER,
    created_at REAL NOT NULL,
    UNIQUE (project_id, version)
);
CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    thread_id INTEGER REFERENCES threads(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    task TEXT NOT NULL,
    status TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}',
    result TEXT,
    final_text TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    harness_version INTEGER,
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS runs_project ON runs(project_id, kind, created_at);
CREATE INDEX IF NOT EXISTS runs_thread ON runs(thread_id, created_at);
CREATE TABLE IF NOT EXISTS events (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS vault_secrets (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ref TEXT NOT NULL,
    blob TEXT NOT NULL,
    hint TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_used_at REAL,
    PRIMARY KEY (user_id, ref)
);
CREATE TABLE IF NOT EXISTS user_providers (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    label TEXT NOT NULL,
    kind TEXT NOT NULL,
    base_url TEXT NOT NULL,
    models TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (user_id, name)
);
CREATE TABLE IF NOT EXISTS user_connectors (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    label TEXT NOT NULL,
    template TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    command TEXT NOT NULL DEFAULT '',
    args TEXT NOT NULL DEFAULT '[]',
    url TEXT NOT NULL DEFAULT '',
    fields TEXT NOT NULL DEFAULT '[]',
    enabled INTEGER NOT NULL DEFAULT 1,
    tools TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    UNIQUE (user_id, name)
);
CREATE TABLE IF NOT EXISTS browser_pairings (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS telegram_links (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    bot_username TEXT NOT NULL DEFAULT '',
    chat_id INTEGER,
    chat_name TEXT NOT NULL DEFAULT '',
    pair_code TEXT,
    pair_created REAL,
    active_project INTEGER,
    active_thread INTEGER,
    update_offset INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS model_limits (
    user_id INTEGER NOT NULL,
    spec TEXT NOT NULL,
    context INTEGER NOT NULL,
    output INTEGER,
    source TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, spec)
);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    domain TEXT NOT NULL,
    cue TEXT NOT NULL,
    lesson TEXT NOT NULL,
    evidence TEXT NOT NULL,
    confidence TEXT NOT NULL DEFAULT '',
    run_id INTEGER,
    created_at REAL NOT NULL
);
"""


MIGRATIONS = [
    # per-user global settings, e.g. {"share_workspace": false}
    ("users", "settings", "TEXT NOT NULL DEFAULT '{}'"),
    # 1 = the conversation runs in its own cage; 0 = it shares the project workspace
    ("threads", "isolated", "INTEGER NOT NULL DEFAULT 1"),
    # sha256 manifest of the cage when it was created (to detect the conversation's own changes)
    ("threads", "manifest", "TEXT NOT NULL DEFAULT '{}'"),
    # attachments sent with a chat message: [{"name", "path", "kind", "size"}]
    ("runs", "attachments", "TEXT NOT NULL DEFAULT '[]'"),
    # per-project vision model override (empty = NESICODE_VISION_MODEL)
    ("projects", "vision_model", "TEXT NOT NULL DEFAULT ''"),
    # per-project image-generation model (empty = NESICODE_IMAGE_MODEL)
    ("projects", "image_model", "TEXT NOT NULL DEFAULT ''"),
    # speech-to-text model for the microphone button ('' = macOS built-in recognizer when available)
    ("projects", "transcribe_model", "TEXT NOT NULL DEFAULT ''"),
    # per-conversation execution mode ('' = the user's default) and tools allowed "always" in it
    ("threads", "exec_mode", "TEXT NOT NULL DEFAULT ''"),
    # main folder picked by the user on disk ('' = nesicode keeps the files in its data dir)
    ("projects", "folder", "TEXT NOT NULL DEFAULT ''"),
    ("threads", "allowed_tools", "TEXT NOT NULL DEFAULT '[]'"),
]


class DB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA)
            for table, column, decl in MIGRATIONS:
                cols = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
                if column not in cols:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, params)

    def one(self, sql: str, params: tuple | dict = ()) -> dict | None:
        with self.lock:
            row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def insert(self, sql: str, params: tuple | dict = ()) -> int:
        with self.lock:
            return int(self.conn.execute(sql, params).lastrowid)

    # ---- events -------------------------------------------------------------
    def add_event(self, run_id: int, event: dict) -> int:
        with self.lock:
            row = self.conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id=?", (run_id,)).fetchone()
            seq = int(row[0])
            self.conn.execute(
                "INSERT INTO events (run_id, seq, data) VALUES (?, ?, ?)",
                (run_id, seq, json.dumps({**event, "ts": time.time()}, ensure_ascii=False)),
            )
            return seq

    def events_after(self, run_id: int, after: int, limit: int = 500) -> list[tuple[int, dict]]:
        rows = self.all(
            "SELECT seq, data FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?", (run_id, after, limit)
        )
        return [(r["seq"], json.loads(r["data"])) for r in rows]


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
