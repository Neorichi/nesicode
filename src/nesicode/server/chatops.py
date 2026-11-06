"""Chat operations shared by the web API and other front-ends (Telegram)."""

from __future__ import annotations

import json
import re
import time

from ..agent.vision import image_mime
from ..agent.workspace import ATTACHMENTS_DIR
from .db import DB, loads
from .jobs import Jobs, create_cage, thread_ws


def user_prefs(db: DB, user_id: int) -> dict:
    row = db.one("SELECT settings FROM users WHERE id=?", (user_id,))
    return loads(row["settings"], {}) if row else {}


def ensure_thread(db: DB, project: dict, user_id: int, first_text: str) -> dict:
    """A new conversation in `project` (caged unless the user shares the project folder)."""
    isolated = 0 if user_prefs(db, user_id).get("share_workspace") else 1
    now = time.time()
    title = re.sub(r"\s+", " ", first_text).strip()[:60] or "Nueva conversación"
    tid = db.insert(
        "INSERT INTO threads (project_id, title, isolated, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (project["id"], title, isolated, now, now),
    )
    if isolated:
        db.execute("UPDATE threads SET manifest=? WHERE id=?", (json.dumps(create_cage(project, tid)), tid))
    return db.one("SELECT * FROM threads WHERE id=?", (tid,))


def save_attachment(project: dict, thread: dict, name: str, data: bytes) -> dict:
    target = thread_ws(project, thread) / ATTACHMENTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "archivo"
    stem, dot, ext = clean.rpartition(".")
    if not dot:
        stem, ext = clean, ""
    candidate, n = clean, 1
    while (target / candidate).exists():
        n += 1
        candidate = f"{stem}-{n}.{ext}" if ext else f"{stem}-{n}"
    f = target / candidate
    f.write_bytes(data)
    head = data[:8192]
    kind = "image" if image_mime(f.name) else "pdf" if f.suffix.lower() == ".pdf" else ("binary" if b"\0" in head else "text")
    return {"name": f.name, "path": f"{ATTACHMENTS_DIR}/{candidate}", "kind": kind, "size": len(data)}


def post_message(db: DB, jobs: Jobs, project: dict, thread: dict, text: str, attachments: list[dict]) -> dict:
    rid = db.insert(
        "INSERT INTO runs (project_id, thread_id, kind, task, status, created_at, attachments) "
        "VALUES (?, ?, 'chat', ?, 'waiting', ?, ?)",
        (project["id"], thread["id"], text, time.time(), json.dumps(attachments)),
    )
    db.execute("UPDATE threads SET updated_at=? WHERE id=?", (time.time(), thread["id"]))
    jobs.enqueue_chat(rid, thread["id"])
    return db.one("SELECT * FROM runs WHERE id=?", (rid,))
