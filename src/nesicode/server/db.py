"""SQLite persistence (one file, WAL mode, a single shared connection guarded by a lock)."""

from __future__ import annotations

import json
import re
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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
CREATE TABLE IF NOT EXISTS bots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    instructions TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    policy TEXT NOT NULL DEFAULT 'ask',
    max_steps INTEGER NOT NULL DEFAULT 30,
    daily_cap INTEGER NOT NULL DEFAULT 24,
    notify TEXT NOT NULL DEFAULT 'telegram',
    thread_id INTEGER,
    enabled INTEGER NOT NULL DEFAULT 1,
    fail_streak INTEGER NOT NULL DEFAULT 0,
    paused_reason TEXT NOT NULL DEFAULT '',
    last_run_at REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS bot_triggers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id INTEGER NOT NULL REFERENCES bots(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    spec TEXT NOT NULL DEFAULT '',
    tz TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL DEFAULT '',
    config TEXT NOT NULL DEFAULT '{}',
    next_at REAL,
    last_at REAL,
    last_hash TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS bot_triggers_due ON bot_triggers(enabled, next_at);
CREATE TABLE IF NOT EXISTS bot_memory (
    bot_id INTEGER NOT NULL REFERENCES bots(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (bot_id, key)
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
    # runs started by a bot (and what woke it up)
    ("runs", "bot_id", "INTEGER"),
    ("runs", "trigger", "TEXT NOT NULL DEFAULT ''"),
    # 1 = the user renamed the conversation: never retitle it automatically
    ("threads", "title_locked", "INTEGER NOT NULL DEFAULT 0"),
    # when the agent may use a connector: mention (only if the user names it) | auto | ask (approve every use)
    ("user_connectors", "usage", "TEXT NOT NULL DEFAULT 'mention'"),
    # project profile: general | security | webdev | data | docs (a harness overlay, see harness/profiles.py)
    ("projects", "profile", "TEXT NOT NULL DEFAULT 'general'"),
    # the bot this conversation belongs to (set by the bot itself; ids can be reused, so never infer it from bots.thread_id)
    ("threads", "bot_id", "INTEGER"),
    # per-conversation execution mode ('' = the user's default) and tools allowed "always" in it
    ("threads", "exec_mode", "TEXT NOT NULL DEFAULT ''"),
    # main folder picked by the user on disk ('' = nesicode keeps the files in its data dir)
    ("projects", "folder", "TEXT NOT NULL DEFAULT ''"),
    ("threads", "allowed_tools", "TEXT NOT NULL DEFAULT '[]'"),
]


# Ids that leave the app (URLs, live streams, jobs in memory, Telegram anchors, webhook secrets 'bot:<id>:webhook')
# must never be handed out twice: without AUTOINCREMENT SQLite gives a new row the highest free id, so deleting the
# newest conversation/run/project/bot let the next one take its id and receive what was meant for the deleted one.
AUTOINC_TABLES = ("projects", "threads", "runs", "bots", "bot_triggers")
# Columns (with no foreign key) that may still name a deleted row: a migrated table's sequence starts above them too.
_DANGLING = {
    "projects": ("SELECT MAX(active_project) FROM telegram_links", "SELECT MAX(project_id) FROM lessons"),
    "threads": ("SELECT MAX(active_thread) FROM telegram_links", "SELECT MAX(thread_id) FROM bots"),
    "runs": ("SELECT MAX(run_id) FROM harness_versions", "SELECT MAX(run_id) FROM lessons"),
    "bots": ("SELECT MAX(bot_id) FROM runs", "SELECT MAX(bot_id) FROM threads",
             "SELECT MAX(CAST(substr(ref, 5, instr(substr(ref, 5), ':') - 1) AS INTEGER)) FROM vault_secrets "
             "WHERE ref LIKE 'bot:%:%'"),
}
_HEAD = re.compile(r'^\s*CREATE\s+TABLE\s+(?:"\w+"|`\w+`|\[\w+\]|\w+)\s*\(', re.I)
_PK = re.compile(r"\bid\s+INTEGER\s+PRIMARY\s+KEY\b(?!\s+AUTOINCREMENT)", re.I)


def _autoinc_sql(table: str, sql: str) -> str | None:
    """The CREATE TABLE of `table` (as stored, with every column added since) for a copy named _autoinc_<table> whose
    id is AUTOINCREMENT — None when the statement is not the expected shape (then the table is left alone)."""
    if not _HEAD.match(sql) or len(_PK.findall(sql)) != 1:
        return None
    return _PK.sub("id INTEGER PRIMARY KEY AUTOINCREMENT", _HEAD.sub(f'CREATE TABLE "_autoinc_{table}" (', sql, 1), 1)


def _rebuild_autoinc(conn: sqlite3.Connection, table: str, new_sql: str) -> None:
    """SQLite's documented table rebuild (inside the caller's transaction, foreign keys off): new table, copy, drop,
    rename, then the table's own indexes and triggers again (telegram.py's telegram_unanchor_* among them)."""
    extra = [r[0] for r in conn.execute(
        "SELECT sql FROM sqlite_master WHERE tbl_name=? AND type IN ('index', 'trigger') AND sql IS NOT NULL", (table,))]
    cols = ", ".join(f'"{r[1]}"' for r in conn.execute(f'PRAGMA table_info("{table}")'))
    conn.execute(new_sql)
    conn.execute(f'INSERT INTO "_autoinc_{table}" ({cols}) SELECT {cols} FROM "{table}"')
    conn.execute(f'DROP TABLE "{table}"')
    conn.execute(f'ALTER TABLE "_autoinc_{table}" RENAME TO "{table}"')
    for sql in extra:
        conn.execute(sql)
    floor = [conn.execute(f'SELECT MAX(id) FROM "{table}"').fetchone()[0]]
    for q in _DANGLING.get(table, ()):
        try:
            floor.append(conn.execute(q).fetchone()[0])
        except sqlite3.Error:  # a table or column of a much older database: nothing to protect there
            pass
    top = max((int(v) for v in floor if isinstance(v, int)), default=0)
    if not conn.execute("UPDATE sqlite_sequence SET seq=MAX(seq, ?) WHERE name=?", (top, table)).rowcount:
        conn.execute("INSERT INTO sqlite_sequence (name, seq) VALUES (?, ?)", (table, top))


def migrate_autoincrement(conn: sqlite3.Connection, path: Path | None) -> list[str]:
    """Give AUTOINCREMENT ids to the AUTOINC_TABLES of a database created before it (idempotent: a database that has
    it — every new one — is not touched). One transaction for all the tables: a failure leaves the file as it was.
    The first time, the file is copied to <db>.pre-autoinc.bak. Returns the tables migrated."""

    def todo() -> dict[str, str]:
        out = {}
        for t in AUTOINC_TABLES:
            row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()
            if row and row[0] and not re.search(r"\bAUTOINCREMENT\b", row[0], re.I) and (new := _autoinc_sql(t, row[0])):
                out[t] = new
        return out

    if not todo():
        return []
    if path is not None and str(path) != ":memory:":
        bak = Path(f"{path}.pre-autoinc.bak")
        if not bak.exists():
            dst = sqlite3.connect(bak)
            try:
                conn.backup(dst)  # a consistent copy, WAL included (a plain file copy could miss recent writes)
            finally:
                dst.close()
    before = {tuple(r) for r in conn.execute("PRAGMA foreign_key_check")}
    conn.execute("PRAGMA foreign_keys=OFF")  # (a no-op inside a transaction: set before BEGIN)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            tables = todo()  # again under the write lock: another process may have just done it
            for table, new_sql in tables.items():
                _rebuild_autoinc(conn, table, new_sql)
            broken = {tuple(r) for r in conn.execute("PRAGMA foreign_key_check")} - before
            if broken:
                raise sqlite3.IntegrityError(f"the rebuild would break foreign keys: {list(broken)[:5]}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
    return list(tables)


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
            try:
                if done := migrate_autoincrement(self.conn, path):
                    print(f"nesicode: ids are never reused now in {', '.join(done)} (backup: {path}.pre-autoinc.bak)")
            except Exception as exc:  # never keep the app from starting: it works as before, and retries next time
                print("nesicode: could not migrate the database to AUTOINCREMENT ids (retried at the next start):", exc)

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
