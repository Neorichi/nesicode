"""Export / import a user's configuration (providers + keys, connectors, preferences, project
template, Telegram bot) as a password-encrypted file, to move it to another computer.

File format (JSON, .nesicode):
  {"format": "nesicode-config", "version": 1, "created": <ts>, "kdf": "scrypt", "n": 2**15, "r": 8, "p": 1,
   "salt": b64, "nonce": b64, "ciphertext": b64}
The plaintext is a JSON document; it is sealed with AES-256-GCM under a key derived from the password.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .db import DB, loads
from .vault import Vault

FORMAT = "nesicode-config"
VERSION = 1
N, R, P = 2**15, 8, 1


class ConfigError(ValueError):
    pass


def _key(password: str, salt: bytes, n: int = N, r: int = R, p: int = P) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32, maxmem=128 * 1024 * 1024)


def seal(doc: dict, password: str) -> bytes:
    salt, nonce = secrets.token_bytes(16), secrets.token_bytes(12)
    ct = AESGCM(_key(password, salt)).encrypt(nonce, json.dumps(doc).encode(), FORMAT.encode())
    b = lambda x: base64.b64encode(x).decode()  # noqa: E731
    return json.dumps({"format": FORMAT, "version": VERSION, "created": time.time(), "kdf": "scrypt",
                       "n": N, "r": R, "p": P, "salt": b(salt), "nonce": b(nonce), "ciphertext": b(ct)},
                      indent=1).encode()


def unseal(data: bytes, password: str) -> dict:
    try:
        env = json.loads(data)
    except Exception as exc:
        raise ConfigError("El archivo no es una configuración de Nesicode") from exc
    if env.get("format") != FORMAT:
        raise ConfigError("El archivo no es una configuración de Nesicode")
    try:
        key = _key(password, base64.b64decode(env["salt"]), int(env.get("n", N)), int(env.get("r", R)), int(env.get("p", P)))
        plain = AESGCM(key).decrypt(base64.b64decode(env["nonce"]), base64.b64decode(env["ciphertext"]), FORMAT.encode())
    except Exception as exc:
        raise ConfigError("Contraseña incorrecta o archivo dañado") from exc
    return json.loads(plain)


# ------------------------------------------------------------------ build / apply
def build(db: DB, user_id: int, *, project_id: int | None, include_telegram: bool) -> dict:
    v = Vault(db)
    providers = []
    for p in db.all("SELECT * FROM user_providers WHERE user_id=? ORDER BY name", (user_id,)):
        providers.append({"name": p["name"], "label": p["label"], "kind": p["kind"], "base_url": p["base_url"],
                          "models": loads(p["models"], []), "api_key": v.get(user_id, f"provider:{p['name']}") or ""})
    connectors = []
    for c in db.all("SELECT * FROM user_connectors WHERE user_id=? ORDER BY id", (user_id,)):
        fields = loads(c["fields"], [])
        secrets_ = {f["key"]: v.get(user_id, f"connector:{c['name']}:{f['key']}") or "" for f in fields if f.get("secret")}
        connectors.append({k: c[k] for k in ("name", "label", "template", "kind", "command", "url", "enabled")}
                          | {"args": loads(c["args"], []), "fields": fields, "secrets": secrets_})
    row = db.one("SELECT settings FROM users WHERE id=?", (user_id,))
    doc: dict = {"providers": providers, "connectors": connectors, "settings": loads(row["settings"], {}) if row else {}}
    if project_id:
        p = db.one("SELECT * FROM projects WHERE id=? AND user_id=?", (project_id, user_id))
        if p:
            h = db.one("SELECT files FROM harness_versions WHERE project_id=? AND version=?", (p["id"], p["harness_version"]))
            doc["project_template"] = {
                "from": p["name"],
                **{k: p.get(k) for k in ("executor_model", "improver_model", "vision_model", "image_model",
                                         "transcribe_model", "check_command", "max_steps", "domain")},
                "harness": loads(h["files"], {}) if h else {},
            }
    if include_telegram:
        token = v.get(user_id, "telegram:bot")
        if token:
            doc["telegram"] = {"token": token}
    return doc


def summary(doc: dict) -> dict:
    t = doc.get("project_template")
    return {
        "providers": [p["name"] for p in doc.get("providers", [])],
        "connectors": [c["label"] for c in doc.get("connectors", [])],
        "settings": sorted(doc.get("settings", {}).keys()),
        "project_template": (t or {}).get("from"),
        "telegram": bool(doc.get("telegram")),
    }


def apply(db: DB, user_id: int, doc: dict) -> dict:
    """Merge an imported configuration into this user's account (same names are overwritten)."""
    v = Vault(db)
    now = time.time()
    for p in doc.get("providers", []):
        db.execute(
            "INSERT INTO user_providers (user_id, name, label, kind, base_url, models, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, name) DO UPDATE SET label=excluded.label, "
            "kind=excluded.kind, base_url=excluded.base_url, models=excluded.models, updated_at=excluded.updated_at",
            (user_id, p["name"], p.get("label") or p["name"], p["kind"], p["base_url"], json.dumps(p.get("models", [])), now, now),
        )
        if p.get("api_key"):
            v.put(user_id, f"provider:{p['name']}", p["api_key"])
    for c in doc.get("connectors", []):
        db.execute(
            "INSERT INTO user_connectors (user_id, name, label, template, kind, command, args, url, fields, enabled, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, name) DO UPDATE SET label=excluded.label, "
            "template=excluded.template, kind=excluded.kind, command=excluded.command, args=excluded.args, url=excluded.url, "
            "fields=excluded.fields, enabled=excluded.enabled",
            (user_id, c["name"], c["label"], c.get("template", ""), c["kind"], c.get("command", ""),
             json.dumps(c.get("args", [])), c.get("url", ""), json.dumps(c.get("fields", [])), int(c.get("enabled", 1)), now),
        )
        for key, value in (c.get("secrets") or {}).items():
            if value:
                v.put(user_id, f"connector:{c['name']}:{key}", value)
    row = db.one("SELECT settings FROM users WHERE id=?", (user_id,))
    settings = {**(loads(row["settings"], {}) if row else {}), **(doc.get("settings") or {})}
    if doc.get("project_template"):
        settings["project_template"] = doc["project_template"]
    db.execute("UPDATE users SET settings=? WHERE id=?", (json.dumps(settings), user_id))
    if doc.get("telegram", {}).get("token"):
        v.put(user_id, "telegram:bot", doc["telegram"]["token"])
        db.execute(
            "INSERT INTO telegram_links (user_id, bot_username, created_at) VALUES (?, '', ?) "
            "ON CONFLICT(user_id) DO UPDATE SET chat_id=NULL, chat_name='', update_offset=0",
            (user_id, now),
        )
    return summary(doc)
