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
import re
import secrets
import time
from typing import Any, Callable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .db import DB, loads
from .vault import Vault

FORMAT = "nesicode-config"
VERSION = 1
N, R, P = 2**15, 8, 1
ROLE_KEYS = ("executor_model", "improver_model", "vision_model", "image_model", "transcribe_model")


class ConfigError(ValueError):
    pass


def _key(password: str, salt: bytes, n: int = N, r: int = R, p: int = P) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32, maxmem=128 * 1024 * 1024)


def _kdf_params(env: dict) -> tuple[int, int, int]:
    """The scrypt cost comes from the (untrusted) file: accept only what this app writes, give or take
    one step, so a crafted file cannot keep the CPU busy for an hour before the password is even checked."""
    try:
        n, r, p = int(env.get("n", N)), int(env.get("r", R)), int(env.get("p", P))
    except (TypeError, ValueError) as exc:
        raise ConfigError("El archivo no es una configuración de Nesicode") from exc
    if not (2**10 <= n <= 2**16 and n & (n - 1) == 0 and 1 <= r <= 8 and p == 1):
        raise ConfigError("Archivo de configuración no válido (parámetros de cifrado fuera de rango)")
    return n, r, p


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
    if not isinstance(env, dict) or env.get("format") != FORMAT:
        raise ConfigError("El archivo no es una configuración de Nesicode")
    n, r, p = _kdf_params(env)
    try:
        key = _key(password, base64.b64decode(env["salt"]), n, r, p)
        plain = AESGCM(key).decrypt(base64.b64decode(env["nonce"]), base64.b64decode(env["ciphertext"]), FORMAT.encode())
        doc = json.loads(plain)
    except Exception as exc:
        raise ConfigError("Contraseña incorrecta o archivo dañado") from exc
    if not isinstance(doc, dict):
        raise ConfigError("Contraseña incorrecta o archivo dañado")
    return doc


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
        connectors.append({k: c[k] for k in ("name", "label", "template", "kind", "command", "url", "enabled", "usage")}
                          | {"args": loads(c["args"], []), "fields": fields, "secrets": secrets_})
    row = db.one("SELECT settings FROM users WHERE id=?", (user_id,))
    doc: dict = {"providers": providers, "connectors": connectors, "settings": loads(row["settings"], {}) if row else {}}
    # Models by role always travel: from the chosen project, else the one used most recently.
    src = db.one("SELECT * FROM projects WHERE id=? AND user_id=?", (project_id, user_id)) if project_id else None
    src = src or db.one(
        "SELECT p.* FROM projects p LEFT JOIN threads t ON t.project_id=p.id WHERE p.user_id=? "
        "GROUP BY p.id ORDER BY MAX(COALESCE(t.updated_at, p.created_at)) DESC LIMIT 1", (user_id,))
    if src:
        doc["role_models"] = {k: src.get(k) or "" for k in ROLE_KEYS} | {"from": src["name"]}
    if project_id:
        p = db.one("SELECT * FROM projects WHERE id=? AND user_id=?", (project_id, user_id))
        if p:
            h = db.one("SELECT files FROM harness_versions WHERE project_id=? AND version=?", (p["id"], p["harness_version"]))
            # (never the folders the agent may read, read_dirs: they are this machine's, and a file must never widen
            # what an agent may read — validate_import and create_project take named keys only, so none come in either)
            doc["project_template"] = {
                "from": p["name"],
                **{k: p.get(k) for k in ("executor_model", "improver_model", "vision_model", "image_model",
                                         "transcribe_model", "check_command", "max_steps", "domain", "profile")},
                "harness": loads(h["files"], {}) if h else {},
            }
    doc["bots"] = []
    for b in db.all("SELECT b.*, p.name AS project_name FROM bots b JOIN projects p ON p.id=b.project_id WHERE b.user_id=?", (user_id,)):
        trig = db.all("SELECT kind, spec, tz, prompt, config FROM bot_triggers WHERE bot_id=? AND kind IN ('cron', 'interval', 'url')",
                      (b["id"],))
        doc["bots"].append({k: b[k] for k in ("name", "description", "instructions", "model", "policy", "max_steps",
                                              "daily_cap", "notify", "project_name")}
                           | {"enabled": bool(b["enabled"]),
                              "triggers": [{**t, "config": loads(t["config"], {})} for t in trig],
                              "memory": db.all("SELECT key, value FROM bot_memory WHERE bot_id=?", (b["id"],))})
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
        "role_models": {k: v for k, v in (doc.get("role_models") or {}).items() if v and k in ROLE_KEYS},
        "bots": [b["name"] for b in doc.get("bots") or []],
        # what the names stand for: where providers point, what connectors run (and the environment variables their
        # fields set), the preferences' values
        "details": {
            "providers": [{"name": p["name"], "kind": p.get("kind"), "base_url": p.get("base_url")} for p in doc.get("providers", [])],
            "connectors": [{"name": c["name"], "kind": c.get("kind"), "enabled": bool(c.get("enabled", 1)),
                            "command": " ".join([c.get("command") or "", *c.get("args", [])]).strip(), "url": c.get("url") or "",
                            "usage": c.get("usage") or "mention",
                            "env": sorted({f["env"] for f in c.get("fields") or [] if isinstance(f, dict) and f.get("env")
                                           and isinstance(f["env"], str)})}
                           for c in doc.get("connectors", [])],
            "settings": {k: v for k, v in (doc.get("settings") or {}).items() if k not in ("project_template", "pending_bots")},
            "check_command": (t or {}).get("check_command") or "",
        },
    }


# ------------------------------------------------------------------ validation of an imported file
# Whoever sealed the file chose its contents: an import passes the same checks as the API (and never
# loosens the approvals), so a shared "team setup" cannot plant a command, redirect a stored key or
# switch approvals off behind the user's back.
PROVIDER_RX = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")  # = the API's NAME_RX
CONNECTOR_RX = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
FIELD_KEY_RX = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
HEADER_RX = re.compile(r"^[A-Za-z0-9-]{1,60}$")
ENV_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,60}$")
TELEGRAM_RX = re.compile(r"^\d+:[A-Za-z0-9_-]{20,}$")
KEY_RX = re.compile(r"[\x21-\x7e]+")  # = the API's: an API key is visible ASCII, no spaces
BAD_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "cookie", "upgrade"}
# Preferences an import may make stricter but never looser (strictest first).
STRICTER: dict[str, list] = {"exec_mode": ["ask_all", "ask_bash", "auto"], "browser_usage": ["ask", "mention", "auto"],
                             "share_workspace": [False, True]}


def _items(value) -> list:
    return value if isinstance(value, list) else []


def _text(value, limit: int) -> str | None:
    return value if isinstance(value, str) and len(value) <= limit else None


def validate_import(doc: dict, *, auth_enabled: bool, check_url: Callable[[str], str], settings_model: Any = None,
                    current_settings: dict | None = None) -> tuple[dict, dict]:
    """Clean copy of an imported document plus what was dropped or changed and why ({skipped, notes})."""
    from ..connectors import catalog_entry

    skipped: list[str] = []
    notes: list[str] = []
    out: dict = {}

    def url_or_none(url, what: str, exact: bool = False) -> str | None:
        try:
            checked = check_url(str(url or ""))
        except Exception as exc:
            skipped.append(f"{what}: {getattr(exc, 'detail', None) or exc}")
            return None
        # a connector keeps its URL exactly as written (GitHub's MCP endpoint ends in '/'); providers are normalised
        return str(url).strip() if exact else checked

    providers = []
    for p in _items(doc.get("providers")):
        name = p.get("name") if isinstance(p, dict) else None
        if not (isinstance(name, str) and PROVIDER_RX.match(name) and p.get("kind") in ("openai", "anthropic")):
            skipped.append(f"proveedor {str(name)[:40]!r}: nombre o tipo no válido")
            continue
        if (base := url_or_none(p.get("base_url"), f"proveedor «{name}»")) is None:
            continue
        models = sorted({m.strip() for m in _items(p.get("models")) if isinstance(m, str) and 0 < len(m.strip()) <= 200})
        key = (_text(p.get("api_key"), 500) or "").strip()
        if key and not KEY_RX.fullmatch(key):  # (the API's rule: no HTTP header could carry it)
            skipped.append(f"proveedor «{name}»: su API key no es válida (espacios, saltos de línea o acentos); se importa sin ella")
            key = ""
        providers.append({"name": name, "label": (_text(p.get("label"), 60) or "").strip() or name, "kind": p["kind"],
                          "base_url": base, "models": models[:500], "api_key": key})
    out["providers"] = providers

    connectors = []
    for c in _items(doc.get("connectors")):
        if not isinstance(c, dict):
            continue
        name, label, kind, template = c.get("name"), c.get("label"), c.get("kind"), c.get("template") or ""
        tag = f"conector «{str(label or name)[:40]}»"
        entry = catalog_entry(template) if isinstance(template, str) and template not in ("", "custom") else None
        command, args = c.get("command") or "", c.get("args") or []
        if not (isinstance(name, str) and CONNECTOR_RX.match(name) and (_text(label, 60) or "").strip()
                and kind in ("stdio", "http", "builtin") and isinstance(template, str)
                and (template in ("", "custom") or (entry and entry["kind"] == kind))
                and isinstance(command, str) and len(command) <= 1000 and isinstance(args, list) and len(args) <= 100
                and all(isinstance(a, str) and len(a) <= 1000 for a in args)):
            skipped.append(f"{tag}: datos no válidos o plantilla desconocida")
            continue
        if auth_enabled and (kind == "stdio" or (entry or {}).get("local_only")):
            skipped.append(f"{tag}: los conectores locales solo están disponibles en modo local (sin login)")
            continue
        if kind == "builtin":
            if not entry:
                skipped.append(f"{tag}: conector integrado desconocido")
                continue
            command, args = entry["id"], []
        if kind == "stdio" and not command.strip():
            skipped.append(f"{tag}: sin comando")
            continue
        url = ""
        if kind == "http" or (kind == "builtin" and c.get("url")):
            if (entry or {}).get("local_only"):  # (local mode only) the user's own Burp & co. live on this machine
                try:
                    _basic_url(c.get("url"))
                except ConfigError as exc:
                    skipped.append(f"{tag}: {exc}")
                    continue
                url = str(c.get("url")).strip()
            elif (url := url_or_none(c.get("url"), tag, exact=True)) is None:
                continue
        fields = []
        for f in _items(c.get("fields")):
            if not (isinstance(f, dict) and isinstance(f.get("key"), str) and FIELD_KEY_RX.match(f["key"])):
                continue
            g: dict = {"key": f["key"], "label": _text(f.get("label"), 100) or f["key"], "secret": bool(f.get("secret"))}
            if isinstance(f.get("header"), str) and HEADER_RX.match(f["header"]) and f["header"].lower() not in BAD_HEADERS:
                g["header"] = f["header"]
            if (prefix := _text(f.get("prefix"), 40)) is not None and not re.search(r"[\r\n]", prefix):
                g["prefix"] = prefix
            if isinstance(f.get("env"), str) and ENV_RX.match(f["env"]):
                g["env"] = f["env"]
            if not g["secret"] and (value := _text(f.get("value"), 2000)) is not None:
                g["value"] = value
            fields.append(g)
        # exactly a catalog entry (same command; its fields, nothing from the file): nothing to review
        catalog_stdio = bool(entry) and kind == "stdio" and command == entry.get("command") and args == entry.get("args", [])
        if catalog_stdio:
            fields = [{k: v for k, v in f.items() if k in ("key", "label", "secret", "header", "prefix", "env")}
                      for f in entry.get("fields", []) if not f.get("url_param")]
        elif kind == "stdio":
            # Any other command from a file is reviewed as the command line it shows, so nothing else of the file may
            # reach its process: environment variables would run other code than that (NODE_OPTIONS=--import=…,
            # PYTHONPATH, PATH, npm_config_registry…). It comes as the API makes one — no fields (a header means
            # nothing to a local command) —, and the review names what was left out.
            if dropped := sorted({f["env"] for f in fields if f.get("env")}):
                notes.append(f"{tag}: se importa sin las variables de entorno que traía ({', '.join(dropped)[:200]}): "
                             "con ellas su comando podría ejecutar código distinto del que revisas")
            fields = []
        secret_keys = {f["key"] for f in fields if f.get("secret")}
        raw = c.get("secrets") if isinstance(c.get("secrets"), dict) else {}
        enabled = bool(c.get("enabled", True))
        if kind == "stdio" and enabled and not catalog_stdio:  # a command that came in a file never starts by itself
            enabled = False
            notes.append(f"{tag}: importado DESACTIVADO. Ejecuta «{' '.join([command, *args])[:200]}»: revísalo y actívalo "
                         "en Conectores si te fías de él")
        connectors.append({"name": name, "label": label.strip(), "template": template, "kind": kind, "command": command,
                           "args": list(args), "url": url, "fields": fields, "enabled": int(enabled),
                           "usage": c.get("usage") if c.get("usage") in ("mention", "auto", "ask") else "mention",
                           "secrets": {k: v for k, v in raw.items() if k in secret_keys and isinstance(v, str) and len(v) <= 4000}})
    out["connectors"] = connectors

    cur = current_settings or {}
    allowed = set(getattr(settings_model, "model_fields", None) or {})
    clean: dict = {}
    for k, v in (doc.get("settings") if isinstance(doc.get("settings"), dict) else {}).items():
        if k == "hidden_providers":
            clean[k] = [x for x in _items(v) if isinstance(x, str) and PROVIDER_RX.match(x)][:50]
            continue
        if k not in allowed or v is None:
            continue  # project_template / role_models / pending_bots travel in their own keys; the rest is not ours
        try:
            v = getattr(settings_model.model_validate({k: v}), k)
        except Exception:
            skipped.append(f"preferencia {k}: valor no válido")
            continue
        order = STRICTER.get(k)
        if order and cur.get(k) in order and v in order and order.index(v) > order.index(cur[k]):
            notes.append(f"preferencia {k}: se mantiene «{cur[k]}» (el archivo pedía «{v}», que relaja las aprobaciones; "
                         "cámbialo tú en Ajustes si lo quieres)")
            continue
        clean[k] = v
    out["settings"] = clean

    t = doc.get("project_template")
    if isinstance(t, dict):
        tpl: dict = {"from": (_text(t.get("from"), 100) or "").strip() or "otro equipo"}
        for k in (*ROLE_KEYS, "check_command", "domain"):
            if (value := _text(t.get(k), 1000 if k == "check_command" else 200)) is not None:
                tpl[k] = value
        if isinstance(t.get("profile"), str) and re.match(r"^[a-z0-9_-]{1,30}$", t["profile"]):
            tpl["profile"] = t["profile"]
        if isinstance(t.get("max_steps"), int) and not isinstance(t["max_steps"], bool) and 5 <= t["max_steps"] <= 300:
            tpl["max_steps"] = t["max_steps"]
        h = t.get("harness")
        if isinstance(h, dict) and len(h) <= 500 and all(isinstance(a, str) and isinstance(b, str) for a, b in h.items()):
            tpl["harness"] = h
        out["project_template"] = tpl
    if isinstance(doc.get("role_models"), dict):
        out["role_models"] = {k: v for k, v in doc["role_models"].items() if k in ROLE_KEYS and _text(v, 200)} | (
            {"from": doc["role_models"]["from"]} if _text(doc["role_models"].get("from"), 100) else {})

    bots = []
    for b in _items(doc.get("bots")):
        if not (isinstance(b, dict) and (_text(b.get("name"), 60) or "").strip()):
            continue
        b = {k: b.get(k) for k in ("name", "description", "instructions", "model", "policy", "max_steps", "daily_cap",
                                   "notify", "project_name", "enabled", "triggers")} | {
            "memory": [m for m in _items(b.get("memory")) if isinstance(m, dict) and _text(m.get("key"), 200)
                       and _text(m.get("value"), 20_000) is not None][:200],
            # the kinds an export writes; a webhook is only ever opened by the user (it shows its secret once)
            "triggers": [x for x in _items(b.get("triggers")) if isinstance(x, dict) and x.get("kind") in ("cron", "interval", "url")][:20]}
        if b.get("policy") == "auto":  # unattended runs without approvals: only the user can switch that on
            b["policy"] = "ask"
            notes.append(f"bot «{b['name']}»: importado con «pedir aprobación»; el archivo pedía «auto» (cámbialo tú si quieres)")
        # Like a command from a file, instructions from a file never run by themselves: on a schedule a bot reads the
        # projects and fetches URLs unattended (those tools never ask), so it arrives switched off until reviewed.
        if b["triggers"] and b.get("enabled") is not False:
            b["enabled"] = False
            notes.append(f"bot «{b['name']}»: importado DESACTIVADO (se ejecutaría solo con las instrucciones del archivo); "
                         "revísalo y actívalo en Bots si te fías de él")
        bots.append(b)
    if bots:
        out["bots"] = bots
    tg = doc.get("telegram")
    if isinstance(tg, dict) and isinstance(tg.get("token"), str) and TELEGRAM_RX.match(tg["token"]):
        out["telegram"] = {"token": tg["token"]}
    return out, {"skipped": skipped, "notes": notes}


def redirected_keys(db: DB, user_id: int, doc: dict) -> tuple[list[str], list[str]]:
    """Providers of a (validated) document that would point a stored key at another endpoint: their names,
    and a note per name. Such keys are dropped on import (the file did not bring one of its own)."""
    names, notes = [], []
    for p in doc.get("providers", []):
        old = db.one("SELECT kind, base_url FROM user_providers WHERE user_id=? AND name=?", (user_id, p["name"]))
        if old and (old["base_url"].rstrip("/"), old["kind"]) != (p["base_url"], p["kind"]) and not p.get("api_key"):
            names.append(p["name"])
            notes.append(f"proveedor «{p['name']}»: cambia de dirección ({p['base_url']}); su clave guardada se borra, "
                         "vuelve a ponerla si te fías de esa dirección")
    return names, notes


def stricter_usages(db: DB, user_id: int, doc: dict) -> tuple[dict[str, str], list[str]]:
    """Connectors of a (validated) document whose usage would be looser than the one they have now — e.g.
    'ask' (each tool call needs the user's approval) → 'auto': {name: usage to keep}, and a note per name.
    Same order as browser_usage: an import never switches approvals off."""
    order, keep, notes = STRICTER["browser_usage"], {}, []
    for c in doc.get("connectors", []):
        old = db.one("SELECT usage FROM user_connectors WHERE user_id=? AND name=?", (user_id, c["name"]))
        if old and old["usage"] in order and c.get("usage") in order and order.index(c["usage"]) > order.index(old["usage"]):
            keep[c["name"]] = old["usage"]
            notes.append(f"conector «{c['label']}»: se mantiene «{old['usage']}» (el archivo pedía «{c['usage']}», que relaja "
                         "las aprobaciones; cámbialo tú en Conectores si lo quieres)")
    return keep, notes


def _basic_url(url: str) -> str:
    url = str(url or "").strip().rstrip("/")
    if not re.match(r"^https?://[^\s/]+", url):
        raise ConfigError("la URL debe ser http(s)://")
    return url


def apply(db: DB, user_id: int, doc: dict, *, auth_enabled: bool | None = None, check_url: Callable[[str], str] | None = None,
          settings_model: Any = None, current_settings: dict | None = None) -> dict:
    """Merge an imported configuration into this user's account (same names are overwritten). The
    document is validated first (see validate_import); the result lists what was skipped and why."""
    from ..config import settings as app_settings

    v = Vault(db)
    row = db.one("SELECT settings FROM users WHERE id=?", (user_id,))
    doc, report = validate_import(
        doc, auth_enabled=app_settings.auth_enabled if auth_enabled is None else auth_enabled,
        check_url=check_url or _basic_url, settings_model=settings_model,
        current_settings=current_settings if current_settings is not None else (loads(row["settings"], {}) if row else {}))
    now = time.time()
    dropped, notes = redirected_keys(db, user_id, doc)
    report["notes"] += notes
    for name in dropped:
        v.delete(user_id, f"provider:{name}")  # never send a stored key to the endpoint a file chose
    kept, notes = stricter_usages(db, user_id, doc)
    report["notes"] += notes
    for c in doc.get("connectors", []):
        c["usage"] = kept.get(c["name"], c["usage"])
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
        old = db.one("SELECT * FROM user_connectors WHERE user_id=? AND name=?", (user_id, c["name"]))
        if old and (old["kind"], old["command"], loads(old["args"], []), old["url"]) != (c["kind"], c["command"], c["args"], c["url"]):
            for f in loads(old["fields"], []):  # its stored tokens must not follow it to a new server or command
                if f.get("secret") and not c["secrets"].get(f.get("key")):
                    v.delete(user_id, f"connector:{c['name']}:{f.get('key')}")
        db.execute(
            "INSERT INTO user_connectors (user_id, name, label, template, kind, command, args, url, fields, enabled, created_at, usage) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, name) DO UPDATE SET label=excluded.label, "
            "template=excluded.template, kind=excluded.kind, command=excluded.command, args=excluded.args, url=excluded.url, "
            "fields=excluded.fields, enabled=excluded.enabled, usage=excluded.usage",
            (user_id, c["name"], c["label"], c.get("template", ""), c["kind"], c.get("command", ""),
             json.dumps(c.get("args", [])), c.get("url", ""), json.dumps(c.get("fields", [])), int(c.get("enabled", 1)), now,
             c.get("usage") or "mention"),
        )
        for key, value in (c.get("secrets") or {}).items():
            if value:
                v.put(user_id, f"connector:{c['name']}:{key}", value)
    row = db.one("SELECT settings FROM users WHERE id=?", (user_id,))
    settings = {**(loads(row["settings"], {}) if row else {}), **(doc.get("settings") or {})}
    if doc.get("project_template"):
        settings["project_template"] = doc["project_template"]
    if doc.get("role_models"):
        settings["role_models"] = {k: v for k, v in doc["role_models"].items() if k in ROLE_KEYS and v}
    if doc.get("bots"):  # created by the app once a project exists (see import_bots)
        settings["pending_bots"] = doc["bots"]
    db.execute("UPDATE users SET settings=? WHERE id=?", (json.dumps(settings), user_id))
    if doc.get("telegram", {}).get("token"):
        v.put(user_id, "telegram:bot", doc["telegram"]["token"])
        db.execute(
            "INSERT INTO telegram_links (user_id, bot_username, created_at) VALUES (?, '', ?) "
            "ON CONFLICT(user_id) DO UPDATE SET chat_id=NULL, chat_name='', update_offset=0",
            (user_id, now),
        )
    return {**summary(doc), **report}


def import_bots(db: DB, bots, user_id: int) -> list[str]:
    """Create the bots waiting in settings['pending_bots'] (same project name if it exists, else the first project)."""
    row = db.one("SELECT settings FROM users WHERE id=?", (user_id,))
    settings = loads(row["settings"], {}) if row else {}
    pending = settings.get("pending_bots") or []
    projects = db.all("SELECT id, name FROM projects WHERE user_id=? ORDER BY id", (user_id,))
    if not pending or not projects or bots is None:
        return []
    created = []
    for b in pending:
        project = next((p for p in projects if p["name"] == b.get("project_name")), projects[0])
        existing = db.one("SELECT id FROM bots WHERE user_id=? AND name=?", (user_id, b["name"]))
        data = {k: b.get(k) for k in ("name", "description", "instructions", "model", "policy", "max_steps", "daily_cap", "notify")}
        data |= {"project_id": project["id"], "enabled": b.get("enabled", True), "triggers": b.get("triggers") or []}
        try:
            saved = bots.save(user_id, data, existing["id"] if existing else None)
        except Exception as exc:  # a bad trigger must not block the rest
            print("nesicode: could not import bot", b.get("name"), exc)
            continue
        if not existing:  # a new bot may reuse a deleted bot's id: never its old webhook secret
            Vault(db).delete(user_id, f"bot:{saved['id']}:webhook")
        for m in b.get("memory") or []:
            db.execute("INSERT INTO bot_memory (bot_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
                       "ON CONFLICT(bot_id, key) DO UPDATE SET value=excluded.value", (saved["id"], m["key"], m["value"], time.time()))
        created.append(saved["name"])
    settings.pop("pending_bots", None)
    db.execute("UPDATE users SET settings=? WHERE id=?", (json.dumps(settings), user_id))
    return created
