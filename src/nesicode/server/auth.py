"""Password hashing (scrypt), cookie sessions, and encryption of per-user API keys."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from ..config import settings
from .db import DB

SESSION_COOKIE = "nesi_session"
SESSION_DAYS = 30
_N, _R, _P = 2**14, 8, 1
# Accounts made automatically for local mode (no login) have no password anyone knows: this marker
# (never a valid hash) lets the first sign-up claim them when the server switches to NESICODE_AUTH=true.
UNCLAIMED_PREFIX = "!local$"
LOCAL_EMAIL = "local@nesicode.local"  # the default local account (also recognised in data from older versions)


def unclaimed_hash() -> str:
    return UNCLAIMED_PREFIX + secrets.token_urlsafe(16)


def is_unclaimed(user: dict) -> bool:
    """A local-mode account whose password nobody has set (so nobody can log into it)."""
    return str(user.get("password_hash") or "").startswith(UNCLAIMED_PREFIX) or user.get("email") == LOCAL_EMAIL


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt_b64, dk_b64 = stored.split("$")
        dk = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt_b64), n=int(n), r=int(r), p=int(p), dklen=32
        )
        return hmac.compare_digest(dk, base64.b64decode(dk_b64))
    except (ValueError, TypeError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(db: DB, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    db.execute(
        "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
        (_token_hash(token), user_id, time.time() + SESSION_DAYS * 86400),
    )
    return token


def session_user(db: DB, token: str | None) -> dict | None:
    if not token:
        return None
    row = db.one(
        "SELECT u.id, u.email, u.name FROM sessions s JOIN users u ON u.id=s.user_id "
        "WHERE s.token_hash=? AND s.expires_at>?",
        (_token_hash(token), time.time()),
    )
    return row


def delete_session(db: DB, token: str | None) -> None:
    if token:
        db.execute("DELETE FROM sessions WHERE token_hash=?", (_token_hash(token),))


def _fernet() -> Fernet:
    """Key from NESICODE_SECRET_KEY, or a generated one persisted in the data dir."""
    secret = settings.secret_key
    if not secret:
        path: Path = settings.data_dir / ".secret_key"
        if path.exists():
            secret = path.read_text(encoding="utf-8").strip()
        else:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            secret = secrets.token_urlsafe(48)
            path.write_text(secret, encoding="utf-8")
            path.chmod(0o600)
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def decrypt(value: str) -> str | None:
    try:
        return _fernet().decrypt(value.encode()).decode()
    except InvalidToken:
        return None


def user_keys(db: DB, user_id: int) -> dict:
    """The user's provider context for model calls: their own providers (endpoint + key from the
    vault). Built-in server providers are resolved from the server environment by the registry."""
    from .vault import Vault

    v = Vault(db)
    ctx: dict = {}
    for p in db.all("SELECT name, kind, base_url FROM user_providers WHERE user_id=?", (user_id,)):
        ctx[p["name"]] = {"kind": p["kind"], "base_url": p["base_url"], "api_key": v.get(user_id, f"provider:{p['name']}") or ""}
    return ctx


def migrate_legacy_keys(db: DB) -> int:
    """Move keys saved by the old Fernet store (user_keys table) into the vault as providers."""
    import json as _json

    from ..llm import server_provider_config
    from .vault import Vault

    v = Vault(db)
    moved = 0
    for row in db.all("SELECT user_id, provider, secret FROM user_keys"):
        value = decrypt(row["secret"])
        cfg = server_provider_config(row["provider"])
        if value and cfg and not db.one(
            "SELECT id FROM user_providers WHERE user_id=? AND name=?", (row["user_id"], row["provider"])
        ):
            now = time.time()
            db.insert(
                "INSERT INTO user_providers (user_id, name, label, kind, base_url, models, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (row["user_id"], row["provider"], row["provider"], cfg["kind"], cfg["base_url"], _json.dumps([]), now, now),
            )
            v.put(row["user_id"], f"provider:{row['provider']}", value)
            moved += 1
        db.execute("DELETE FROM user_keys WHERE user_id=? AND provider=?", (row["user_id"], row["provider"]))
    return moved
