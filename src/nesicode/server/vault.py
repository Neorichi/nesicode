"""Secret vault for API keys.

Envelope encryption: every secret gets its own random 256-bit data key; the secret is sealed
with AES-256-GCM under that data key, and the data key is sealed with AES-256-GCM under the
master key. The associated data binds each ciphertext to (user, reference), so a row copied
to another user or provider fails to decrypt. Plaintext never leaves this module except to
make a model call, and the API only ever shows a hint (last 4 characters).

The master key never lives in the database or in the data directory:
  1. NESICODE_VAULT_KEY (base64, 32 bytes) from the environment / your secret manager, or
  2. ~/.nesicode/vault.key (mode 0600), created on first use.
Both are outside every agent cage (the OS sandbox denies the user's home and the repo).
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .db import DB

VERSION = 1
_lock = threading.Lock()
_master: bytes | None = None


def _master_key() -> bytes:
    global _master
    with _lock:
        if _master is not None:
            return _master
        env = os.environ.get("NESICODE_VAULT_KEY", "").strip()
        if env:
            key = base64.b64decode(env)
        else:
            path = Path(os.environ.get("NESICODE_VAULT_KEY_FILE", "~/.nesicode/vault.key")).expanduser()
            if path.exists():
                key = base64.b64decode(path.read_text(encoding="utf-8").strip())
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.parent.chmod(0o700)
                key = AESGCM.generate_key(bit_length=256)
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(base64.b64encode(key).decode())
        if len(key) != 32:
            raise RuntimeError("vault master key must be 32 bytes (base64)")
        _master = key
        return key


def _aad(user_id: int, ref: str) -> bytes:
    return f"nesicode-vault:v{VERSION}:{int(user_id)}:{ref}".encode()


def seal(user_id: int, ref: str, plaintext: str) -> str:
    data_key = AESGCM.generate_key(bit_length=256)
    n1, n2 = secrets.token_bytes(12), secrets.token_bytes(12)
    aad = _aad(user_id, ref)
    wrapped = AESGCM(_master_key()).encrypt(n1, data_key, aad)
    body = AESGCM(data_key).encrypt(n2, plaintext.encode(), aad)
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return json.dumps({"v": VERSION, "n1": b64(n1), "k": b64(wrapped), "n2": b64(n2), "c": b64(body)})


def unseal(user_id: int, ref: str, blob: str) -> str:
    d = json.loads(blob)
    aad = _aad(user_id, ref)
    data_key = AESGCM(_master_key()).decrypt(base64.b64decode(d["n1"]), base64.b64decode(d["k"]), aad)
    return AESGCM(data_key).decrypt(base64.b64decode(d["n2"]), base64.b64decode(d["c"]), aad).decode()


def hint(value: str) -> str:
    return "••••" + value[-4:] if len(value) >= 8 else "••••"


class Vault:
    def __init__(self, db: DB):
        self.db = db

    def put(self, user_id: int, ref: str, value: str) -> str:
        blob = seal(user_id, ref, value)
        self.db.execute(
            "INSERT INTO vault_secrets (user_id, ref, blob, hint, created_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, ref) DO UPDATE SET blob=excluded.blob, hint=excluded.hint, created_at=excluded.created_at",
            (user_id, ref, blob, hint(value), time.time()),
        )
        return hint(value)

    def get(self, user_id: int, ref: str) -> str | None:
        row = self.db.one("SELECT blob FROM vault_secrets WHERE user_id=? AND ref=?", (user_id, ref))
        if not row:
            return None
        try:
            value = unseal(user_id, ref, row["blob"])
        except Exception:
            return None  # tampered, moved, or sealed under another master key
        self.db.execute("UPDATE vault_secrets SET last_used_at=? WHERE user_id=? AND ref=?", (time.time(), user_id, ref))
        return value

    def hint(self, user_id: int, ref: str) -> str | None:
        row = self.db.one("SELECT hint FROM vault_secrets WHERE user_id=? AND ref=?", (user_id, ref))
        return row["hint"] if row else None

    def delete(self, user_id: int, ref: str) -> None:
        self.db.execute("DELETE FROM vault_secrets WHERE user_id=? AND ref=?", (user_id, ref))
