"""Seed a local dev database with a demo user and an offline (mock) project.

    NESICODE_DATA_DIR=./data uv run python scripts/seed_dev.py

Demo login: demo@nesicode.test / demo-pass-123  (local development only)
"""

import json
import time

from nesicode.harness import base_harness
from nesicode.server import auth
from nesicode.server.app import db
from nesicode.server.jobs import project_ws

EMAIL, PASSWORD = "demo@nesicode.test", "demo-pass-123"

user = db.one("SELECT id FROM users WHERE email=?", (EMAIL,))
if user:
    print("demo user already exists")
else:
    uid = db.insert(
        "INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (EMAIL, "Demo", auth.hash_password(PASSWORD), time.time()),
    )
    pid = db.insert(
        "INSERT INTO projects (user_id, name, domain, executor_model, improver_model, check_command, max_steps, "
        "harness_version, created_at) VALUES (?, 'demo-mock', 'demo', 'mock/demo', 'mock/demo', "
        "'test -f NESICODE_RESULT.md', 60, 1, ?)",
        (uid, time.time()),
    )
    db.execute(
        "INSERT INTO harness_versions (project_id, version, files, source, note, created_at) VALUES (?, 1, ?, 'base', ?, ?)",
        (pid, json.dumps(base_harness()), "Harness base de nesicode", time.time()),
    )
    ws = project_ws(db.one("SELECT * FROM projects WHERE id=?", (pid,)))
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "README.md").write_text("# demo\n\nProyecto de ejemplo para nesicode.\n")
    print(f"created {EMAIL} with project demo-mock (#{pid})")
