"""nesicode web API (FastAPI) + static SPA."""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import secrets
import shutil
import sys
import subprocess
import threading
import time
import zipfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agent.workspace import IGNORE_DIRS
from ..config import settings
from ..harness import COMPONENTS, base_harness, component_of, diff_summary, inventory, normalize, unified_diff, validate
from ..llm import (
    KINDS, PROVIDERS, TEMPLATES, LLMError, get_provider, list_models, provider_available, server_provider_config, split_any,
)
from .vault import Vault
from . import auth
from .db import DB, loads
from ..agent.vision import image_mime
from ..agent.workspace import ATTACHMENTS_DIR, apply_cage, changes_since, manifest
from .jobs import (
    FINAL, Jobs, create_cage, current_harness, project_root, project_ws, run_dir, save_harness_version,
    thread_dir, thread_ws, user_root,
)

db = DB(settings.db_path)
vault = Vault(db)
auth.migrate_legacy_keys(db)


def _upgrade_base_harness() -> None:
    """Projects still on the untouched previous base harness get the new built-in `documents` skill."""
    from ..harness.base import DOCUMENTS_SKILL  # noqa: F401

    new = base_harness()
    old = {k: v for k, v in new.items() if k != "skills/documents/SKILL.md"}
    old["guardrails.md"] = "\n".join(l for l in old["guardrails.md"].split("\n")[1:] if "Never type passwords" not in l)
    old["system_prompt.md"] = old["system_prompt.md"].replace("You are Nesicode,", "You are nesicode,")
    for p in db.all("SELECT * FROM projects"):
        files = current_harness(db, p)
        if files in (old, {**old, "system_prompt.md": new["system_prompt.md"]}):
            save_harness_version(db, p["id"], new, "base", "Nesicode: skill `documents` (gráficos, pptx, xlsx, docx)")


_upgrade_base_harness()


def _add_credentials_rule() -> None:
    """Every project's current harness gets the no-credentials guardrail (once)."""
    rule = [l for l in base_harness()["guardrails.md"].splitlines() if "Never type passwords" in l][0]
    for p in db.all("SELECT * FROM projects"):
        files = current_harness(db, p)
        if "Never type passwords" in files.get("guardrails.md", ""):
            continue
        g = files.get("guardrails.md", "").rstrip("\n")
        files = {**files, "guardrails.md": (g + "\n" if g else "") + rule + "\n"}
        save_harness_version(db, p["id"], files, "base", "Nesicode: regla de seguridad (no escribir credenciales ni iniciar sesión)")


_add_credentials_rule()
jobs = Jobs(db)
app = FastAPI(title="nesicode", version="0.1.0")

def _web_dist() -> Path:
    if os.environ.get("NESICODE_WEB_DIST"):
        return Path(os.environ["NESICODE_WEB_DIST"])
    if getattr(sys, "frozen", False):  # PyInstaller bundle
        return Path(getattr(sys, "_MEIPASS", ".")) / "web_dist"
    return Path(__file__).resolve().parents[3] / "web" / "dist"


WEB_DIST = _web_dist()
DESKTOP_TOKEN = os.environ.get("NESICODE_DESKTOP_TOKEN", "")
DESKTOP_COOKIE = "nesi_desktop"


@app.middleware("http")
async def desktop_guard(request: Request, call_next):
    """Desktop mode has no login, so a random per-launch token (handed to the app window in the
    first URL, then kept as an HttpOnly cookie) stops other local programs from using the API."""
    if not DESKTOP_TOKEN or (request.url.path.startswith("/hooks/") and request.method == "POST"):
        return await call_next(request)  # bot webhooks carry their own secret
    if request.query_params.get("token") == DESKTOP_TOKEN:
        resp = RedirectResponse(request.url.path or "/")
        resp.set_cookie(DESKTOP_COOKIE, DESKTOP_TOKEN, httponly=True, samesite="strict", path="/")
        return resp
    if not secrets.compare_digest(request.cookies.get(DESKTOP_COOKIE, ""), DESKTOP_TOKEN):
        return JSONResponse({"detail": "nesicode desktop: abre la app desde su ventana"}, status_code=403)
    return await call_next(request)


# --------------------------------------------------------------------- helpers
def local_user() -> dict:
    """The account used when NESICODE_AUTH is off (single-user mode, no login screen)."""
    if settings.local_user:
        row = db.one("SELECT id, email, name FROM users WHERE email=?", (settings.local_user.strip().lower(),))
        if row:
            return row
    row = db.one("SELECT id, email, name FROM users ORDER BY id LIMIT 1")
    if row and not settings.local_user:
        return row
    email = (settings.local_user or "local@nesicode.local").strip().lower()
    try:
        import getpass

        name = getpass.getuser() if not settings.local_user else email.split("@")[0]
    except Exception:
        name = email.split("@")[0]
    uid = db.insert(
        "INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (email, name, auth.hash_password(secrets.token_urlsafe(32)), time.time()),
    )
    return db.one("SELECT id, email, name FROM users WHERE id=?", (uid,))


def current_user(request: Request) -> dict:
    if not settings.auth_enabled:
        return local_user()
    user = auth.session_user(db, request.cookies.get(auth.SESSION_COOKIE))
    if not user:
        raise HTTPException(401, "No has iniciado sesión")
    return user


def get_project(user: dict, project_id: int) -> dict:
    p = db.one("SELECT * FROM projects WHERE id=? AND user_id=?", (project_id, user["id"]))
    if not p:
        raise HTTPException(404, "Proyecto no encontrado")
    return p


def get_run(user: dict, run_id: int) -> tuple[dict, dict]:
    run = db.one("SELECT * FROM runs WHERE id=?", (run_id,))
    if not run:
        raise HTTPException(404, "Ejecución no encontrada")
    return run, get_project(user, run["project_id"])


def check_model(spec: str, user: dict | None = None) -> str:
    try:
        split_any(spec, auth.user_keys(db, user["id"]) if user else None)
    except LLMError as exc:
        raise HTTPException(422, str(exc)) from exc
    return spec


def run_view(run: dict, *, full: bool = False) -> dict:
    out = {k: run[k] for k in ("id", "project_id", "thread_id", "kind", "task", "status", "final_text",
                                "error", "harness_version", "created_at", "finished_at")}
    out["bot_id"], out["trigger"] = run.get("bot_id"), run.get("trigger") or ""
    out["params"] = loads(run["params"], {})
    out["attachments"] = loads(run.get("attachments"), [])
    result = loads(run["result"], {}) or {}
    if full:
        out["result"] = result
    else:
        out["summary"] = {k: result.get(k) for k in ("selected", "harness_changed", "adopted_version", "usage", "steps") if k in result}
    return out


def _set_cookie(resp: Response, token: str) -> None:
    resp.set_cookie(
        auth.SESSION_COOKIE, token, httponly=True, samesite="lax", secure=settings.cookie_secure,
        max_age=auth.SESSION_DAYS * 86400, path="/",
    )


# ------------------------------------------------------------------------ auth
class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=8, max_length=200)
    name: str = Field(default="", max_length=100)


@app.post("/api/auth/register")
def register(body: Credentials, response: Response):
    if not settings.allow_signup and db.one("SELECT 1 AS x FROM users LIMIT 1"):
        raise HTTPException(403, "El registro está desactivado")
    email = body.email.strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(422, "Email no válido")
    if db.one("SELECT id FROM users WHERE email=?", (email,)):
        raise HTTPException(409, "Ya existe una cuenta con ese email")
    uid = db.insert(
        "INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (email, body.name.strip() or email.split("@")[0], auth.hash_password(body.password), time.time()),
    )
    _set_cookie(response, auth.create_session(db, uid))
    return db.one("SELECT id, email, name FROM users WHERE id=?", (uid,))


@app.post("/api/auth/login")
def login(body: Credentials, response: Response):
    user = db.one("SELECT * FROM users WHERE email=?", (body.email.strip().lower(),))
    if not user or not auth.verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Email o contraseña incorrectos")
    _set_cookie(response, auth.create_session(db, user["id"]))
    return {"id": user["id"], "email": user["email"], "name": user["name"]}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    auth.delete_session(db, request.cookies.get(auth.SESSION_COOKIE))
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
    return {**user, "auth": settings.auth_enabled}


# ------------------------------------------------------------ providers & vault
NAME_RX = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")
_server_models: dict[str, tuple[float, list[str]]] = {}


def check_base_url(url: str) -> str:
    """Endpoints must be http(s). With NESICODE_ALLOW_PRIVATE_PROVIDERS=0 (multi-user servers),
    private/loopback addresses are refused so users cannot probe the server's network."""
    import ipaddress
    import socket
    from urllib.parse import urlparse

    url = url.strip().rstrip("/")
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise HTTPException(422, "El endpoint debe ser una URL http(s)://")
    if os.environ.get("NESICODE_ALLOW_PRIVATE_PROVIDERS", "1") == "0":
        try:
            addrs = {ai[4][0] for ai in socket.getaddrinfo(u.hostname, None)}
        except OSError as exc:
            raise HTTPException(422, f"No se puede resolver {u.hostname}") from exc
        for a in addrs:
            ip = ipaddress.ip_address(a)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise HTTPException(422, "Este servidor no permite endpoints en redes privadas")
    return url


def provider_rows(user_id: int) -> list[dict]:
    rows = db.all("SELECT * FROM user_providers WHERE user_id=? ORDER BY name", (user_id,))
    out = []
    for r in rows:
        h = vault.hint(user_id, f"provider:{r['name']}")
        out.append({
            "name": r["name"], "label": r["label"], "kind": r["kind"], "base_url": r["base_url"],
            "models": loads(r["models"], []), "key_hint": h, "has_key": bool(h), "source": "user",
            "available": True,
        })
    mine = {r["name"] for r in rows}
    hidden = set(user_settings(user_id).get("hidden_providers", []))
    for name, info in PROVIDERS.items():
        if name in mine or name in hidden or not provider_available(name, {}):
            continue
        _warm_server_models(name)
        if name in ("ollama", "compat") and not os.environ.get(info.base_url_env):
            continue
        cfg = server_provider_config(name) or {"kind": "openai", "base_url": ""}
        cached = _server_models.get(name, (0, []))[1]
        out.append({
            "name": name, "label": info.label + " (servidor)", "kind": cfg["kind"], "base_url": cfg["base_url"],
            "models": cached or list(m.split("/", 1)[1] for m in info.examples), "key_hint": "servidor" if info.key_env else None,
            "has_key": bool(info.key_env), "source": "server", "available": True,
        })
    return out


def _discover_server(name: str) -> list[str]:
    if name == "mock":
        return ["demo"]
    cfg = server_provider_config(name)
    if not cfg:
        raise LLMError("proveedor desconocido")
    models = list_models(cfg["kind"], cfg["base_url"], cfg["api_key"])  # the server key never leaves
    _server_models[name] = (time.time(), models)
    return models


_warming: set[str] = set()


def _warm_server_models(name: str) -> None:
    """Fetch a server provider's model list in the background (cached for 10 minutes)."""
    if time.time() - _server_models.get(name, (0, []))[0] < 600 or name in _warming:
        return
    _warming.add(name)

    def run():
        try:
            _discover_server(name)
        except Exception:
            _server_models[name] = (time.time(), _server_models.get(name, (0, []))[1])
        finally:
            _warming.discard(name)

    threading.Thread(target=run, daemon=True).start()


def hidden_server_providers(user_id: int) -> list[dict]:
    hidden = user_settings(user_id).get("hidden_providers", [])
    return [{"name": n, "label": PROVIDERS[n].label} for n in hidden if n in PROVIDERS and provider_available(n, {})]


@app.get("/api/models/limits")
def get_model_limits(spec: str, refresh: bool = False, user: dict = Depends(current_user)):
    from .limits import model_limits

    return {"spec": spec, **model_limits(db, user["id"], spec, refresh=refresh)}


@app.get("/api/providers")
def list_providers(user: dict = Depends(current_user)):
    return {"providers": provider_rows(user["id"]), "hidden": hidden_server_providers(user["id"]),
            "can_adopt": not settings.auth_enabled, "templates": TEMPLATES, "kinds": list(KINDS),
            "defaults": {"executor": settings.default_executor_model, "improver": settings.default_improver_model,
                         "vision": settings.vision_model, "image": settings.image_model}}


class ProviderBody(BaseModel):
    name: str = Field(min_length=1, max_length=31)
    label: str = Field(default="", max_length=60)
    kind: str = Field(pattern="^(openai|anthropic)$")
    base_url: str = Field(min_length=1, max_length=300)
    api_key: str | None = Field(default=None, max_length=500)
    models: list[str] = Field(default_factory=list, max_length=500)


class ProviderPatch(BaseModel):
    label: str | None = Field(default=None, max_length=60)
    kind: str | None = Field(default=None, pattern="^(openai|anthropic)$")
    base_url: str | None = Field(default=None, max_length=300)
    api_key: str | None = Field(default=None, max_length=500)  # None = keep, "" = remove
    models: list[str] | None = Field(default=None, max_length=500)


def _clean_models(models: list[str]) -> list[str]:
    return sorted({m.strip() for m in models if m.strip() and len(m.strip()) <= 200})


# ---- automatic role assignment from a provider's models (e.g. Nesi Labs: newest nesiN, nesi-vision)
NESI_RX = re.compile(r"^nesi(\d*)$")
VISION_RX = re.compile(r"^nesi[-_]?vision", re.I)


def _nesi_rank(model: str) -> int:
    m = NESI_RX.match(model)
    return (int(m.group(1)) if m.group(1) else 1) if m else -1


def suggest_roles(models: list[str]) -> dict[str, str]:
    """Which of these models fit each role: the newest nesiN for chat, nesi-vision for vision."""
    out: dict[str, str] = {}
    nesi = [m for m in models if NESI_RX.match(m)]
    if nesi:
        out["executor_model"] = max(nesi, key=_nesi_rank)
    vision = sorted(m for m in models if VISION_RX.match(m))
    if vision:
        out["vision_model"] = max(vision, key=lambda m: (len(re.findall(r"\d+", m)) and int(re.findall(r"\d+", m)[-1]), m))
    return out


def _refresh_limits_bg(user_id: int, provider: str, models: list[str]) -> None:
    from .limits import refresh_provider_limits

    threading.Thread(target=refresh_provider_limits, args=(db, user_id, provider, models), daemon=True).start()


def auto_assign(user: dict, provider: str, models: list[str]) -> list[dict]:
    """Fill or upgrade project roles with this provider's best models. Never overrides a working model
    from another provider; upgrades chat only from an older nesiN of this same provider."""
    sug = suggest_roles(models)
    if not sug:
        return []
    ctx = auth.user_keys(db, user["id"])

    def works(spec: str) -> bool:
        try:
            name, _ = split_any(spec, ctx)
            return provider_available(name, ctx)
        except LLMError:
            return False

    changes = []
    for p in db.all("SELECT * FROM projects WHERE user_id=?", (user["id"],)):
        upd = {}
        for col, model in sug.items():
            cur = p.get(col) or ""
            new = f"{provider}/{model}"
            if cur == new:
                continue
            cur_prov, _, cur_model = cur.partition("/")
            older = cur_prov == provider and _nesi_rank(cur_model) >= 0 and _nesi_rank(cur_model) < _nesi_rank(model)
            if not cur or not works(cur) or older:
                upd[col] = new
        if "executor_model" in sug and not works(p.get("improver_model") or ""):
            upd["improver_model"] = f"{provider}/{sug['executor_model']}"
        if upd:
            db.execute(f"UPDATE projects SET {', '.join(f'{k}=?' for k in upd)} WHERE id=?", (*upd.values(), p["id"]))
            changes.append({"project": p["name"], **upd})
    return changes


@app.post("/api/providers")
def create_provider(body: ProviderBody, user: dict = Depends(current_user)):
    name = body.name.strip().lower()
    if not NAME_RX.match(name):
        raise HTTPException(422, "Nombre: minúsculas, dígitos, - o _ (sin espacios)")
    if db.one("SELECT id FROM user_providers WHERE user_id=? AND name=?", (user["id"], name)):
        raise HTTPException(409, f"Ya tienes un proveedor llamado {name}")
    base = check_base_url(body.base_url)
    now = time.time()
    db.insert(
        "INSERT INTO user_providers (user_id, name, label, kind, base_url, models, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user["id"], name, body.label.strip() or name, body.kind, base, json.dumps(_clean_models(body.models)), now, now),
    )
    if body.api_key and body.api_key.strip():
        vault.put(user["id"], f"provider:{name}", body.api_key.strip())
    models = _clean_models(body.models)
    if not models:
        try:
            models = list_models(body.kind, base, (body.api_key or "").strip())
            db.execute("UPDATE user_providers SET models=? WHERE user_id=? AND name=?", (json.dumps(models), user["id"], name))
        except LLMError:
            models = []
    assigned = auto_assign(user, name, models)
    _refresh_limits_bg(user["id"], name, models)
    return {**next(p for p in provider_rows(user["id"]) if p["name"] == name), "assigned": assigned}


def _own_provider(user: dict, name: str) -> dict:
    row = db.one("SELECT * FROM user_providers WHERE user_id=? AND name=?", (user["id"], name))
    if not row:
        raise HTTPException(404, "Proveedor no encontrado")
    return row


@app.patch("/api/providers/{name}")
def patch_provider(name: str, body: ProviderPatch, user: dict = Depends(current_user)):
    _own_provider(user, name)
    fields: dict = {}
    if body.label is not None:
        fields["label"] = body.label.strip() or name
    if body.kind is not None:
        fields["kind"] = body.kind
    if body.base_url is not None:
        fields["base_url"] = check_base_url(body.base_url)
    if body.models is not None:
        fields["models"] = json.dumps(_clean_models(body.models))
    if fields:
        fields["updated_at"] = time.time()
        db.execute(
            f"UPDATE user_providers SET {', '.join(f'{k}=?' for k in fields)} WHERE user_id=? AND name=?",
            (*fields.values(), user["id"], name),
        )
    if body.api_key is not None:
        if body.api_key.strip():
            vault.put(user["id"], f"provider:{name}", body.api_key.strip())
        else:
            vault.delete(user["id"], f"provider:{name}")
    return next(p for p in provider_rows(user["id"]) if p["name"] == name)


@app.delete("/api/providers/{name}")
def delete_provider(name: str, user: dict = Depends(current_user)):
    _own_provider(user, name)
    db.execute("DELETE FROM user_providers WHERE user_id=? AND name=?", (user["id"], name))
    vault.delete(user["id"], f"provider:{name}")
    return {"ok": True}


def _set_hidden(user: dict, name: str, hide: bool) -> list[str]:
    prefs = user_settings(user["id"])
    hidden = [h for h in prefs.get("hidden_providers", []) if h != name] + ([name] if hide else [])
    prefs["hidden_providers"] = hidden
    db.execute("UPDATE users SET settings=? WHERE id=?", (json.dumps(prefs), user["id"]))
    return hidden


@app.post("/api/providers/{name}/hide")
def hide_provider(name: str, user: dict = Depends(current_user)):
    if name not in PROVIDERS:
        raise HTTPException(404, "Solo se pueden ocultar proveedores del servidor")
    return {"hidden": _set_hidden(user, name, True)}


@app.post("/api/providers/{name}/unhide")
def unhide_provider(name: str, user: dict = Depends(current_user)):
    return {"hidden": _set_hidden(user, name, False)}


@app.post("/api/providers/{name}/adopt")
def adopt_provider(name: str, user: dict = Depends(current_user)):
    """Turn a server provider (.env) into one of the user's own, so it can be edited or deleted.
    The server key is copied into the user's vault, so this is only allowed in single-user mode."""
    if settings.auth_enabled:
        raise HTTPException(403, "Solo en modo local (NESICODE_AUTH=false): crea un proveedor propio con tu key")
    cfg = server_provider_config(name)
    if not cfg or name == "mock" or not provider_available(name, {}):
        raise HTTPException(404, "Proveedor no encontrado")
    if db.one("SELECT id FROM user_providers WHERE user_id=? AND name=?", (user["id"], name)):
        raise HTTPException(409, "Ya es tuyo")
    now = time.time()
    models = _server_models.get(name, (0, []))[1]
    db.insert(
        "INSERT INTO user_providers (user_id, name, label, kind, base_url, models, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user["id"], name, PROVIDERS[name].label, cfg["kind"], cfg["base_url"], json.dumps(models), now, now),
    )
    if cfg["api_key"]:
        vault.put(user["id"], f"provider:{name}", cfg["api_key"])
    _set_hidden(user, name, True)  # deleting your copy later must not resurrect the server one
    assigned = auto_assign(user, name, models)
    _refresh_limits_bg(user["id"], name, models)
    return {**next(p for p in provider_rows(user["id"]) if p["name"] == name), "assigned": assigned}


class ProviderTest(BaseModel):
    kind: str = Field(pattern="^(openai|anthropic)$")
    base_url: str = Field(min_length=1, max_length=300)
    api_key: str | None = Field(default=None, max_length=500)
    name: str | None = None  # reuse the stored key of this provider when api_key is omitted


@app.post("/api/providers/test")
def test_provider(body: ProviderTest, user: dict = Depends(current_user)):
    key = (body.api_key or "").strip()
    if not key and body.name:
        key = vault.get(user["id"], f"provider:{body.name}") or ""
    try:
        models = list_models(body.kind, check_base_url(body.base_url), key)
    except LLMError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "models": models}


@app.post("/api/providers/{name}/discover")
def discover_models(name: str, user: dict = Depends(current_user)):
    row = db.one("SELECT * FROM user_providers WHERE user_id=? AND name=?", (user["id"], name))
    try:
        if row:
            models = list_models(row["kind"], row["base_url"], vault.get(user["id"], f"provider:{name}") or "")
            db.execute("UPDATE user_providers SET models=?, updated_at=? WHERE id=?", (json.dumps(models), time.time(), row["id"]))
        else:
            if name not in PROVIDERS or not provider_available(name, {}):
                raise HTTPException(404, "Proveedor no encontrado")
            models = _discover_server(name)
    except LLMError as exc:
        raise HTTPException(502, f"No se pudieron obtener los modelos: {exc}") from exc
    _refresh_limits_bg(user["id"], name, models)
    return {"models": models, "assigned": auto_assign(user, name, models)}


# ------------------------------------------------------- export / import config
_exports: dict[str, tuple[float, int, bytes]] = {}  # one-time download tokens


class ExportBody(BaseModel):
    password: str = Field(min_length=8, max_length=200)
    project_id: int | None = None
    include_telegram: bool = False


@app.post("/api/config/export")
def config_export(body: ExportBody, user: dict = Depends(current_user)):
    """Encrypt the user's configuration with their password; returns a one-time download URL."""
    from .config_io import build, seal, summary

    doc = build(db, user["id"], project_id=body.project_id, include_telegram=body.include_telegram)
    blob = seal(doc, body.password)
    token = secrets.token_urlsafe(24)
    now = time.time()
    for k in [k for k, (ts, _, _) in _exports.items() if now - ts > 600]:
        _exports.pop(k, None)
    _exports[token] = (now, user["id"], blob)
    name = time.strftime("nesicode-config-%Y%m%d.nesicode")
    return {"url": f"/api/config/export/{token}", "filename": name, "summary": summary(doc)}


@app.get("/api/config/export/{token}")
def config_export_download(token: str, user: dict = Depends(current_user)):
    item = _exports.pop(token, None)
    if not item or item[1] != user["id"] or time.time() - item[0] > 600:
        raise HTTPException(404, "El enlace de exportación caducó: vuelve a exportar")
    return Response(item[2], media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{time.strftime("nesicode-config-%Y%m%d.nesicode")}"'})


@app.post("/api/config/import")
async def config_import(request: Request, file: UploadFile = File(...), user: dict = Depends(current_user)):
    from .config_io import ConfigError, apply, unseal

    form = await request.form()
    password = str(form.get("password") or "")
    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(413, "Archivo demasiado grande")
    try:
        doc = unseal(data, password)
    except ConfigError as exc:
        raise HTTPException(422, str(exc)) from exc
    result = apply(db, user["id"], doc)
    result["projects_updated"] = _apply_role_models(user, doc.get("role_models") or {})
    from .config_io import import_bots

    result["bots_created"] = import_bots(db, bots, user["id"])  # waits for a project if there is none yet
    if doc.get("telegram"):
        telegram.restart(user["id"])
    for p in doc.get("providers", []):
        _refresh_limits_bg(user["id"], p["name"], p.get("models", []))
    return {"imported": result}


def _model_works(spec: str, user: dict) -> bool:
    ctx = auth.user_keys(db, user["id"])
    try:
        name, _ = split_any(spec, ctx)
        return provider_available(name, ctx)
    except LLMError:
        return False


def _apply_role_models(user: dict, roles: dict) -> list[str]:
    """Imported models by role replace the ones in this user's projects (when the model is usable here)."""
    from .config_io import ROLE_KEYS

    usable = {k: v for k, v in roles.items() if k in ROLE_KEYS and v and _model_works(v, user)}
    changed = []
    for p in db.all("SELECT * FROM projects WHERE user_id=?", (user["id"],)):
        upd = {k: v for k, v in usable.items() if p.get(k) != v}
        if upd:
            db.execute(f"UPDATE projects SET {', '.join(f'{k}=?' for k in upd)} WHERE id=?", (*upd.values(), p["id"]))
            changed.append(p["name"])
    return changed


@app.get("/api/config/template")
def config_template(user: dict = Depends(current_user)):
    t = user_settings(user["id"]).get("project_template")
    return {"template": {k: v for k, v in t.items() if k != "harness"} | {"harness_files": len(t.get("harness", {}))} if t else None}


# ------------------------------------------------------------------ Telegram
from .telegram import TelegramManager  # noqa: E402

telegram = TelegramManager(db, jobs)
if not os.environ.get("NESICODE_NO_TELEGRAM"):
    threading.Thread(target=telegram.start_all, daemon=True, name="nesi-telegram-start").start()
threading.Thread(target=jobs.resume_waiting, daemon=True, name="nesi-resume-queue").start()

from .bots import BotManager, CronError, local_tz, next_runs  # noqa: E402

bots = BotManager(db, jobs, telegram)
jobs.bots = bots
bots.start()  # scheduler thread (off with NESICODE_NO_SCHEDULER)


# ------------------------------------------------------------------------ bots
def _bot_view(b: dict, full: bool = False) -> dict:
    trig = db.all("SELECT * FROM bot_triggers WHERE bot_id=? ORDER BY id", (b["id"],))
    last = db.one("SELECT id, status, trigger, created_at, finished_at FROM runs WHERE bot_id=? ORDER BY id DESC LIMIT 1", (b["id"],))
    project = db.one("SELECT name FROM projects WHERE id=?", (b["project_id"],))
    nxt = min((t["next_at"] for t in trig if t["next_at"] and t["kind"] in ("cron", "interval", "once", "url") and t["enabled"]),
              default=None)
    out = {**{k: b[k] for k in ("id", "project_id", "name", "description", "instructions", "model", "policy", "max_steps",
                               "daily_cap", "notify", "thread_id", "fail_streak", "paused_reason", "last_run_at",
                               "created_at", "updated_at")},
           "enabled": bool(b["enabled"]), "project": project["name"] if project else "", "next_at": nxt if b["enabled"] else None,
           "last_run": last, "webhook": any(t["kind"] == "webhook" for t in trig),
           "triggers": [{k: t[k] for k in ("id", "kind", "spec", "tz", "prompt", "next_at", "last_at", "enabled")}
                        | {"config": loads(t["config"], {})} for t in trig if t["kind"] != "webhook"]}
    if full:
        out["memory"] = db.all("SELECT key, value, updated_at FROM bot_memory WHERE bot_id=? ORDER BY key", (b["id"],))
        out["runs"] = db.all("SELECT id, status, trigger, created_at, finished_at, substr(final_text, 1, 300) AS summary, result "
                             "FROM runs WHERE bot_id=? ORDER BY id DESC LIMIT 30", (b["id"],))
        for r in out["runs"]:
            r["usage"] = (loads(r.pop("result"), {}) or {}).get("usage")
        out["windows"] = os.name == "nt"
    return out


def _bot(user: dict, bot_id: int) -> dict:
    b = bots.bot(bot_id, user["id"])
    if not b:
        raise HTTPException(404, "Bot no encontrado")
    return b


class TriggerIn(BaseModel):
    kind: str = Field(pattern="^(cron|interval|url)$")
    spec: str = Field(max_length=2000)
    tz: str = Field(default="", max_length=64)
    prompt: str = Field(default="", max_length=4000)
    config: dict = Field(default_factory=dict)


class BotIn(BaseModel):
    project_id: int | None = None
    name: str | None = Field(default=None, max_length=60)
    description: str | None = Field(default=None, max_length=300)
    instructions: str | None = Field(default=None, max_length=20_000)
    model: str | None = Field(default=None, max_length=200)
    policy: str | None = Field(default=None, pattern="^(ask|readonly|auto)$")
    notify: str | None = Field(default=None, pattern="^(telegram|none)$")
    max_steps: int | None = Field(default=None, ge=3, le=200)
    daily_cap: int | None = Field(default=None, ge=1, le=500)
    enabled: bool | None = None
    triggers: list[TriggerIn] | None = None


def _save_bot(user: dict, body: BotIn, bot_id: int | None = None) -> dict:
    from ..agent.workspace import ToolError

    data = body.model_dump(exclude_none=True)
    if "triggers" in data:
        data["triggers"] = [t for t in data["triggers"]]
    if data.get("model"):
        check_model(data["model"], user)
    try:
        b = bots.save(user["id"], data, bot_id)
    except (ValueError, CronError, ToolError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return _bot_view(b, full=True)


@app.get("/api/bots")
def list_bots(user: dict = Depends(current_user)):
    return {"bots": [_bot_view(b) for b in db.all("SELECT * FROM bots WHERE user_id=? ORDER BY name", (user["id"],))],
            "tz": local_tz(), "telegram": bool(telegram.bridges.get(user["id"]) and telegram.bridges[user["id"]].link().get("chat_id"))}


@app.post("/api/bots")
def create_bot_api(body: BotIn, user: dict = Depends(current_user)):
    if not body.project_id or not body.name:
        raise HTTPException(422, "Faltan el nombre y el proyecto")
    return _save_bot(user, body)


@app.get("/api/bots/{bot_id}")
def get_bot(bot_id: int, user: dict = Depends(current_user)):
    return _bot_view(_bot(user, bot_id), full=True)


@app.patch("/api/bots/{bot_id}")
def patch_bot(bot_id: int, body: BotIn, user: dict = Depends(current_user)):
    _bot(user, bot_id)
    return _save_bot(user, body, bot_id)


@app.delete("/api/bots/{bot_id}")
def delete_bot(bot_id: int, user: dict = Depends(current_user)):
    b = _bot(user, bot_id)
    for r in db.all("SELECT id FROM runs WHERE bot_id=? AND status IN ('waiting', 'queued', 'running')", (bot_id,)):
        jobs.cancel(r["id"])
        db.execute("UPDATE runs SET status='cancelled', finished_at=? WHERE id=? AND status='waiting'", (time.time(), r["id"]))
    vault.delete(user["id"], f"bot:{bot_id}:webhook")
    db.execute("UPDATE runs SET bot_id=NULL WHERE bot_id=?", (b["id"],))  # ids can be reused: detach the history
    db.execute("DELETE FROM bots WHERE id=?", (b["id"],))  # triggers and memory cascade; its conversation stays
    return {"ok": True}


class BotRun(BaseModel):
    text: str = Field(default="", max_length=20_000)


@app.post("/api/bots/{bot_id}/run")
def run_bot(bot_id: int, body: BotRun, user: dict = Depends(current_user)):
    b = _bot(user, bot_id)
    run = bots.fire(b["id"], "manual", body.text, force=True)
    if not run:
        raise HTTPException(409, "No se pudo lanzar el bot")
    return run_view(run)


@app.delete("/api/bots/{bot_id}/memory/{key}")
def delete_bot_memory(bot_id: int, key: str, user: dict = Depends(current_user)):
    _bot(user, bot_id)
    db.execute("DELETE FROM bot_memory WHERE bot_id=? AND key=?", (bot_id, key))
    return {"ok": True}


class CronPreview(BaseModel):
    kind: str = Field(pattern="^(cron|interval)$")
    spec: str = Field(max_length=200)
    tz: str = Field(default="", max_length=64)


@app.post("/api/bots/preview")
def preview_schedule(body: CronPreview, user: dict = Depends(current_user)):
    try:
        return {"next": next_runs(body.kind, body.spec, body.tz or local_tz(), 3)}
    except CronError as exc:
        raise HTTPException(422, str(exc)) from exc


class BotDraft(BaseModel):
    project_id: int
    description: str = Field(min_length=3, max_length=4000)


@app.post("/api/bots/draft")
def draft_bot(body: BotDraft, user: dict = Depends(current_user)):
    """Turn 'every morning at 8 tell me…' into a bot config (name, instructions, triggers) the user then reviews."""
    from ..llm import ModelOptions
    from .bots import validate_trigger

    p = get_project(user, body.project_id)
    tz = local_tz() or "local"
    prompt = (
        "Turn the user's request into a JSON bot configuration for Nesicode. Reply with ONLY a JSON object:\n"
        '{"name": short name, "description": one line, "instructions": complete self-contained brief for an unattended '
        'agent (what to check, how to report, when to stay silent by answering NOTHING_NEW), written in the user\'s language, '
        '"triggers": [{"kind": "cron", "spec": "min hour dom mon dow", "tz": "' + tz + '"} | {"kind": "interval", "spec": "30" (minutes) or "2h"} '
        '| {"kind": "url", "spec": "https://…", "config": {"every": 30}}] (empty list if it should only run on demand), '
        '"policy": "readonly" if it only needs to read/fetch/report, else "ask"}\n\nUser request:\n' + body.description
    )
    try:
        provider, model = get_provider(p["executor_model"], auth.user_keys(db, user["id"]))
        stop = threading.Event()
        timer = threading.Timer(40, stop.set)  # a slow or hung model must not block the form: fall back after 40 s
        timer.start()
        try:
            text, _ = provider.complete(model, "You produce strict JSON configs.", prompt,
                                        ModelOptions(effort="low", role="drafter", max_tokens=2000, cancel=stop))
        finally:
            timer.cancel()
        m = re.search(r"\{.*\}", text, re.S)
        draft = json.loads(m.group(0)) if m else {}
    except Exception:
        draft = {}
    triggers = []
    for t in draft.get("triggers") or []:
        try:
            triggers.append(validate_trigger(t))
        except Exception:
            continue
    words = re.sub(r"\s+", " ", body.description).strip()
    return {
        "project_id": p["id"],
        "name": str(draft.get("name") or words[:40]).strip()[:60],
        "description": str(draft.get("description") or words[:120])[:300],
        "instructions": str(draft.get("instructions") or body.description),
        "triggers": triggers,
        "policy": draft.get("policy") if draft.get("policy") in ("ask", "readonly", "auto") else "ask",
        "notify": "telegram",
        "ai": bool(draft),
    }


# --- webhook: POST /hooks/{bot_id} with header X-Nesicode-Token (or /hooks/{bot_id}/{token})
_hook_hits: dict[int, list[float]] = {}


@app.post("/api/bots/{bot_id}/webhook")
def enable_webhook(bot_id: int, request: Request, user: dict = Depends(current_user)):
    """Create (or rotate) the bot's webhook secret. It is shown only now."""
    b = _bot(user, bot_id)
    token = secrets.token_urlsafe(32)
    vault.put(user["id"], f"bot:{b['id']}:webhook", token)
    if not db.one("SELECT id FROM bot_triggers WHERE bot_id=? AND kind='webhook'", (b["id"],)):
        db.execute("INSERT INTO bot_triggers (bot_id, kind, created_at) VALUES (?, 'webhook', ?)", (b["id"], time.time()))
    base = f"{request.url.scheme}://{request.url.netloc}"
    return {"url": f"{base}/hooks/{b['id']}", "token": token, "url_with_token": f"{base}/hooks/{b['id']}/{token}"}


@app.delete("/api/bots/{bot_id}/webhook")
def disable_webhook(bot_id: int, user: dict = Depends(current_user)):
    b = _bot(user, bot_id)
    vault.delete(user["id"], f"bot:{b['id']}:webhook")
    db.execute("DELETE FROM bot_triggers WHERE bot_id=? AND kind='webhook'", (b["id"],))
    return {"ok": True}


async def _hook(bot_id: int, token: str, request: Request):
    host = (request.headers.get("host") or "").split(":")[0].strip("[]")
    if DESKTOP_TOKEN and host not in ("127.0.0.1", "localhost", "::1"):
        raise HTTPException(403, "host no permitido")  # DNS-rebinding guard for the desktop app
    b = db.one("SELECT * FROM bots WHERE id=?", (bot_id,))
    expected = vault.get(b["user_id"], f"bot:{bot_id}:webhook") if b else None
    if not b or not expected or not token or not secrets.compare_digest(token, expected):
        raise HTTPException(404, "not found")
    now = time.time()
    hits = [t for t in _hook_hits.get(bot_id, []) if now - t < 60]
    if len(hits) >= 10:
        raise HTTPException(429, "demasiadas llamadas: máx. 10 por minuto")
    _hook_hits[bot_id] = hits + [now]
    body = await request.body()
    if len(body) > 64 * 1024:
        raise HTTPException(413, "cuerpo demasiado grande (máx. 64 KB)")
    run = bots.fire(bot_id, "webhook", "", payload=body.decode("utf-8", "replace"))
    if not run:
        raise HTTPException(409, "el bot está en pausa, ocupado o sin ejecuciones disponibles hoy")
    return JSONResponse({"run_id": run["id"], "status": run["status"]}, status_code=202)


@app.post("/hooks/{bot_id}")
async def hook(bot_id: int, request: Request):
    return await _hook(bot_id, request.headers.get("x-nesicode-token", ""), request)


@app.post("/hooks/{bot_id}/{token}")
async def hook_with_token(bot_id: int, token: str, request: Request):
    return await _hook(bot_id, token, request)


def _telegram_view(user_id: int) -> dict:
    row = db.one("SELECT * FROM telegram_links WHERE user_id=?", (user_id,)) or {}
    return {
        "configured": bool(row) and bool(vault.hint(user_id, "telegram:bot")),
        "bot": row.get("bot_username", ""),
        "running": user_id in telegram.bridges,
        "error": telegram.errors.get(user_id, ""),
        "linked": bool(row.get("chat_id")),
        "chat_name": row.get("chat_name", ""),
    }


class TelegramAttach(BaseModel):
    attach: bool


@app.post("/api/threads/{thread_id}/telegram")
def thread_telegram(thread_id: int, body: TelegramAttach, user: dict = Depends(current_user)):
    """Attach (or detach) this conversation to the user's Telegram chat from the app."""
    t, p = get_thread(user, thread_id)
    row = db.one("SELECT * FROM telegram_links WHERE user_id=?", (user["id"],))
    if not row or not row["chat_id"]:
        raise HTTPException(422, "Telegram no está vinculado: configúralo en Conectores (MCP) → Telegram")
    bridge = telegram.bridges.get(user["id"])
    if body.attach:
        db.execute("UPDATE telegram_links SET active_thread=?, active_project=? WHERE user_id=?", (t["id"], p["id"], user["id"]))
        if bridge:
            bridge._attach(row["chat_id"], t["id"])
    elif row["active_thread"] == t["id"]:
        db.execute("UPDATE telegram_links SET active_thread=NULL WHERE user_id=?", (user["id"],))
        if bridge:
            bridge.send(row["chat_id"], f"🔓 Has soltado «{t['title']}» desde la app.")
    return {"attached": body.attach}


@app.get("/api/telegram")
def telegram_status(user: dict = Depends(current_user)):
    return _telegram_view(user["id"])


class TelegramToken(BaseModel):
    token: str = Field(min_length=20, max_length=200)


@app.put("/api/telegram")
def telegram_set_token(body: TelegramToken, user: dict = Depends(current_user)):
    from .telegram import TelegramClient

    token = body.token.strip()
    if not re.match(r"^\d+:[A-Za-z0-9_-]{20,}$", token):
        raise HTTPException(422, "Eso no parece un token de bot (formato 123456:ABC…). Cópialo tal cual de @BotFather.")
    try:
        me = TelegramClient(token).call("getMe")
    except Exception as exc:
        raise HTTPException(422, f"Telegram no acepta ese token: {exc}") from exc
    vault.put(user["id"], "telegram:bot", token)
    db.execute(
        "INSERT INTO telegram_links (user_id, bot_username, created_at) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET bot_username=excluded.bot_username, chat_id=NULL, chat_name='', "
        "update_offset=0",
        (user["id"], me.get("username", ""), time.time()),
    )
    telegram.restart(user["id"])
    return _telegram_view(user["id"])


@app.post("/api/telegram/pair")
def telegram_pair(user: dict = Depends(current_user)):
    row = db.one("SELECT * FROM telegram_links WHERE user_id=?", (user["id"],))
    if not row:
        raise HTTPException(422, "Primero guarda el token de tu bot")
    code = secrets.token_urlsafe(12).replace("-", "").replace("_", "")[:16]
    db.execute("UPDATE telegram_links SET pair_code=?, pair_created=? WHERE user_id=?", (code, time.time(), user["id"]))
    if user["id"] not in telegram.bridges:
        telegram.restart(user["id"])
    return {"code": code, "link": f"https://t.me/{row['bot_username']}?start={code}", "command": f"/start {code}"}


@app.delete("/api/telegram")
def telegram_remove(user: dict = Depends(current_user)):
    telegram.stop(user["id"])
    vault.delete(user["id"], "telegram:bot")
    db.execute("DELETE FROM telegram_links WHERE user_id=?", (user["id"],))
    return {"ok": True}


@app.post("/api/telegram/unlink")
def telegram_unlink(user: dict = Depends(current_user)):
    db.execute("UPDATE telegram_links SET chat_id=NULL, chat_name='' WHERE user_id=?", (user["id"],))
    return _telegram_view(user["id"])


# ---------------------------------------------------------------- your Chrome
EXTENSION_SRC = (Path(getattr(sys, "_MEIPASS", "")) / "chrome_extension") if getattr(sys, "frozen", False) \
    else Path(__file__).resolve().parents[3] / "extension"


def _pairing_hash(token: str) -> str:
    import hashlib

    return hashlib.sha256(token.encode()).hexdigest()


@app.get("/api/browser")
def browser_status(request: Request, user: dict = Depends(current_user)):
    from ..browser import hub

    paired = db.one("SELECT created_at FROM browser_pairings WHERE user_id=?", (user["id"],))
    return {**hub.status(user["id"]), "paired": bool(paired), "extension_dir": str(EXTENSION_SRC),
            "local_mode": not settings.auth_enabled}


@app.post("/api/browser/pairing")
def browser_pairing(request: Request, user: dict = Depends(current_user)):
    """New connection code for the extension (replaces the previous one)."""
    import base64

    token = secrets.token_urlsafe(32)
    db.execute(
        "INSERT INTO browser_pairings (user_id, token_hash, created_at) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET token_hash=excluded.token_hash, created_at=excluded.created_at",
        (user["id"], _pairing_hash(token), time.time()),
    )
    base = f"{request.url.scheme}://{request.url.netloc}"
    code = base64.urlsafe_b64encode(json.dumps({"server": base, "token": token}).encode()).decode()
    return {"code": "NESI-" + code}


@app.post("/api/browser/extension/install")
def install_extension(user: dict = Depends(current_user)):
    """Copy the extension to ~/Documents/Nesicode Chrome so it can be loaded with 'Load unpacked'."""
    if settings.auth_enabled:
        raise HTTPException(403, "Solo en modo local")
    if not (EXTENSION_SRC / "manifest.json").exists():
        raise HTTPException(500, "La extensión no viene incluida en esta instalación")
    target = Path.home() / "Documents" / "Nesicode Chrome"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(EXTENSION_SRC, target)
    return {"path": str(target)}


@app.get("/api/browser/extension.zip")
def extension_zip(user: dict = Depends(current_user)):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in EXTENSION_SRC.rglob("*"):
            if f.is_file():
                z.write(f, f"nesicode-chrome/{f.relative_to(EXTENSION_SRC)}")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="nesicode-chrome.zip"'})


@app.websocket("/api/browser/ws")
async def browser_ws(ws: WebSocket):
    from ..browser import hub

    token = ws.query_params.get("token", "")
    row = db.one("SELECT user_id FROM browser_pairings WHERE token_hash=?", (_pairing_hash(token),)) if token else None
    if not row:
        await ws.close(code=4401)
        return
    await ws.accept()
    user_id = row["user_id"]
    try:
        hello = await asyncio.wait_for(ws.receive_json(), 10)
        hub.attach(user_id, ws, {"browser": str(hello.get("browser", ""))[:120], "version": str(hello.get("version", ""))[:20]})
        await ws.send_json({"type": "welcome", "user": user_id})
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
            else:
                hub.on_message(msg)
    except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
        pass
    finally:
        hub.detach(user_id, ws)


# ------------------------------------------------------------ MCP connectors
def _connector_view(user_id: int) -> list[dict]:
    from ..connectors import manager
    from .connectors_store import connector_configs

    out = []
    for row, cfg in connector_configs(db, user_id):
        fields = []
        for f in loads(row["fields"], []):
            hint = vault.hint(user_id, f"connector:{row['name']}:{f['key']}") if f.get("secret") else None
            fields.append({"key": f["key"], "label": f.get("label", f["key"]), "secret": bool(f.get("secret")), "hint": hint})
        out.append({
            "name": row["name"], "label": row["label"], "template": row["template"], "kind": row["kind"],
            "command": " ".join([row["command"], *loads(row["args"], [])]).strip(), "url": row["url"],
            "enabled": bool(row["enabled"]), "usage": row.get("usage") or "mention", "tools": loads(row["tools"], []), "fields": fields,
            "status": manager().status(user_id, cfg) if row["enabled"] else "off",
        })
    return out


@app.get("/api/connectors")
def list_connectors(user: dict = Depends(current_user)):
    from ..connectors import CATALOG, missing_requirements

    mine = _connector_view(user["id"])
    added = {c["template"] for c in mine}
    local_ok = not settings.auth_enabled
    catalog = [
        {**{k: v for k, v in c.items() if k not in ("command", "args")},
         "missing": missing_requirements(c), "allowed": local_ok or (c["kind"] in ("http", "builtin") and not c.get("local_only")), "added": c["id"] in added}
        for c in CATALOG
    ]
    return {"catalog": catalog, "connectors": mine, "local_allowed": local_ok}


def _start_and_list(user: dict, name: str) -> list[str]:
    """(Re)start a connector and return its tool names; raises HTTP 422 with the real error."""
    from ..connectors import manager
    from .connectors_store import connector_configs

    for row, cfg in connector_configs(db, user["id"]):
        if row["name"] == name:
            manager().stop(user["id"], name)
            try:
                tools = [t.name for t in manager().tools(user["id"], cfg, timeout=150)]
            except Exception as exc:
                msg = f"No se pudo arrancar {row['label']}: {exc}"
                if row["template"] == "burp" and re.search(r"not found|404|405|refused|connect", str(exc), re.I):
                    msg += (". En esa dirección no responde el MCP de Burp: comprueba en Burp → pestaña MCP que está "
                            "activado y qué puerto usa (Advanced options). Si otro programa ocupa ese puerto, cámbialo en "
                            "Burp (p. ej. 9877) y pon aquí la misma dirección.")
                raise HTTPException(422, msg) from exc
            db.execute("UPDATE user_connectors SET tools=? WHERE id=?", (json.dumps(tools), row["id"]))
            return tools
    raise HTTPException(404, "Conector no encontrado")


class ConnectorAdd(BaseModel):
    template: str = Field(min_length=1, max_length=40)
    values: dict[str, str] = Field(default_factory=dict)


class ConnectorCustom(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    kind: str = Field(pattern="^(stdio|http)$")
    command: str = Field(default="", max_length=1000)
    url: str = Field(default="", max_length=500)
    token: str = Field(default="", max_length=2000)


def _unique_connector_name(user_id: int, base: str) -> str:
    base = re.sub(r"[^a-z0-9_-]+", "-", base.lower()).strip("-")[:24] or "mcp"
    name, n = base, 1
    while db.one("SELECT id FROM user_connectors WHERE user_id=? AND name=?", (user_id, name)):
        n += 1
        name = f"{base}-{n}"
    return name


def _insert_connector(user: dict, name: str, label: str, template: str, kind: str, command: str,
                      args: list[str], url: str, fields: list[dict], secrets_: dict[str, str]) -> dict:
    if kind == "stdio" and settings.auth_enabled:
        raise HTTPException(403, "Los conectores locales solo están disponibles en modo local (sin login)")
    db.insert(
        "INSERT INTO user_connectors (user_id, name, label, template, kind, command, args, url, fields, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (user["id"], name, label, template, kind, command, json.dumps(args), url, json.dumps(fields), time.time()),
    )
    for key, value in secrets_.items():
        if value:
            vault.put(user["id"], f"connector:{name}:{key}", value)
    try:
        tools = _start_and_list(user, name)
    except HTTPException:
        db.execute("DELETE FROM user_connectors WHERE user_id=? AND name=?", (user["id"], name))
        for key in secrets_:
            vault.delete(user["id"], f"connector:{name}:{key}")
        raise
    return {"name": name, "tools": tools}


@app.post("/api/connectors")
def add_connector(body: ConnectorAdd, user: dict = Depends(current_user)):
    from ..connectors import catalog_entry, missing_requirements

    entry = catalog_entry(body.template)
    if not entry:
        raise HTTPException(404, "Conector desconocido")
    if missing := missing_requirements(entry):
        raise HTTPException(422, f"Falta instalar en este equipo: {', '.join(missing)}")
    if entry.get("local_only") and settings.auth_enabled:
        raise HTTPException(403, "Este conector solo está disponible en modo local (sin login)")
    url = entry.get("url", "")
    for f in entry["fields"]:
        if f.get("url_param"):
            v = re.sub(r"^https?://", "", body.values.get(f["key"], "").strip()).strip("/")
            if not re.fullmatch(r"[A-Za-z0-9.-]+(:\d{1,5})?", v):
                raise HTTPException(422, f"{f['label']}: escribe solo el nombre del servidor")
            url = url.replace("{" + f["key"] + "}", v)
    fields = [{k: v for k, v in f.items() if k in ("key", "label", "secret", "header", "prefix", "env")}
              for f in entry["fields"] if not f.get("url_param")]
    for f in entry["fields"]:
        if not body.values.get(f["key"], "").strip() and not f.get("optional"):
            raise HTTPException(422, f"Falta: {f['label']}")
    secrets_ = {f["key"]: body.values[f["key"]].strip() for f in entry["fields"] if f.get("secret")}
    for f in fields:
        if not f.get("secret"):
            f["value"] = body.values.get(f["key"], "")
    return _insert_connector(user, _unique_connector_name(user["id"], entry["id"]), entry["label"], entry["id"],
                             entry["kind"], entry["id"] if entry["kind"] == "builtin" else entry.get("command", ""),
                             list(entry.get("args", [])),
                             url, fields, secrets_)


@app.post("/api/connectors/custom")
def add_custom_connector(body: ConnectorCustom, user: dict = Depends(current_user)):
    import shlex

    name = _unique_connector_name(user["id"], body.label)
    if body.kind == "stdio":
        parts = shlex.split(body.command)
        if not parts:
            raise HTTPException(422, "Escribe el comando que arranca el servidor MCP")
        return _insert_connector(user, name, body.label, "custom", "stdio", parts[0], parts[1:], "", [], {})
    url = check_base_url(body.url)
    fields = [{"key": "token", "label": "Token", "secret": True, "header": "Authorization", "prefix": "Bearer "}] if body.token else []
    return _insert_connector(user, name, body.label, "custom", "http", "", [], url, fields,
                             {"token": body.token.strip()} if body.token else {})


class ConnectorPatch(BaseModel):
    enabled: bool | None = None
    usage: str | None = Field(default=None, pattern="^(mention|auto|ask)$")


@app.patch("/api/connectors/{name}")
def patch_connector(name: str, body: ConnectorPatch, user: dict = Depends(current_user)):
    from ..connectors import manager

    if not db.one("SELECT id FROM user_connectors WHERE user_id=? AND name=?", (user["id"], name)):
        raise HTTPException(404, "Conector no encontrado")
    if body.usage is not None:
        db.execute("UPDATE user_connectors SET usage=? WHERE user_id=? AND name=?", (body.usage, user["id"], name))
    if body.enabled is not None:
        db.execute("UPDATE user_connectors SET enabled=? WHERE user_id=? AND name=?", (int(body.enabled), user["id"], name))
        if not body.enabled:
            manager().stop(user["id"], name)
    return {"ok": True}


@app.post("/api/connectors/{name}/test")
def test_connector(name: str, user: dict = Depends(current_user)):
    return {"tools": _start_and_list(user, name)}


@app.delete("/api/connectors/{name}")
def delete_connector(name: str, user: dict = Depends(current_user)):
    from ..connectors import manager

    row = db.one("SELECT * FROM user_connectors WHERE user_id=? AND name=?", (user["id"], name))
    if not row:
        raise HTTPException(404, "Conector no encontrado")
    manager().stop(user["id"], name)
    for f in loads(row["fields"], []):
        vault.delete(user["id"], f"connector:{name}:{f['key']}")
    db.execute("DELETE FROM user_connectors WHERE id=?", (row["id"],))
    return {"ok": True}


# -------------------------------------------------------------------- projects
class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    domain: str = Field(default="general", max_length=60)
    executor_model: str = ""
    improver_model: str = ""
    check_command: str = Field(default="", max_length=1000)
    max_steps: int = Field(default=60, ge=5, le=300)
    git_url: str = Field(default="", max_length=500)
    vision_model: str = Field(default="", max_length=200)
    image_model: str = Field(default="", max_length=200)
    transcribe_model: str = Field(default="", max_length=200)
    folder: str = Field(default="", max_length=1000)  # an existing folder on this machine (local mode only)
    use_template: bool = False  # start from the imported project template (models, check command, harness)
    profile: str = Field(default="general", pattern="^[a-z0-9_-]{1,30}$")


class ProjectPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    domain: str | None = Field(default=None, max_length=60)
    executor_model: str | None = None
    improver_model: str | None = None
    check_command: str | None = Field(default=None, max_length=1000)
    max_steps: int | None = Field(default=None, ge=5, le=300)
    vision_model: str | None = Field(default=None, max_length=200)
    image_model: str | None = Field(default=None, max_length=200)
    transcribe_model: str | None = Field(default=None, max_length=200)


def project_view(p: dict) -> dict:
    return {**p, "workspace": str(project_ws(p)), "external": bool(p.get("folder"))}


@app.get("/api/projects")
def list_projects(user: dict = Depends(current_user)):
    return [project_view(p) for p in db.all("SELECT * FROM projects WHERE user_id=? ORDER BY created_at DESC", (user["id"],))]


def check_folder(path: str) -> str:
    """A project folder on this machine. Only in local mode: on a shared server it would let a user
    point the agent at anyone's files."""
    if settings.auth_enabled:
        raise HTTPException(403, "Elegir carpetas del disco solo está disponible en modo local (sin login)")
    p = Path(os.path.expanduser(path.strip())).resolve()
    if not p.is_dir():
        raise HTTPException(422, f"La carpeta no existe: {p}")
    data = settings.data_dir.resolve()
    if p == data or data in p.parents or p in data.parents:
        raise HTTPException(422, "Esa carpeta es la de datos de nesicode; elige otra")
    if p == Path(p.anchor) or p == Path.home().resolve():
        raise HTTPException(422, "Elige una carpeta de proyecto, no la raíz del disco ni tu carpeta personal")
    return str(p)


def default_model(user: dict, preferred: str, role: str = "executor_model") -> str:
    """The best suggestion from the user's providers (e.g. newest nesiN), else the configured default
    if usable, else the first model of a provider they have."""
    for p in provider_rows(user["id"]):
        sug = suggest_roles(p["models"]).get(role)
        if sug and p["name"] != "mock":
            return f"{p['name']}/{sug}"
    if role != "executor_model":
        return preferred
    ctx = auth.user_keys(db, user["id"])
    try:
        name, _ = split_any(preferred, ctx)
        if provider_available(name, ctx):
            return preferred
    except LLMError:
        pass
    for p in provider_rows(user["id"]):
        if p["models"] and p["name"] != "mock":
            return f"{p['name']}/{p['models'][0]}"
    return preferred


@app.post("/api/folders/pick")
def pick_folder_hint(user: dict = Depends(current_user)):
    """Folder picking happens in the desktop window (native dialog); the browser types a path."""
    return {"local_mode": not settings.auth_enabled, "home": str(Path.home()) if not settings.auth_enabled else ""}


@app.post("/api/projects")
def create_project(body: ProjectCreate, user: dict = Depends(current_user)):
    folder = check_folder(body.folder) if body.folder.strip() else ""
    tpl = user_settings(user["id"]).get("project_template") if body.use_template else None
    if tpl:
        for k in ("executor_model", "improver_model", "vision_model", "image_model", "transcribe_model", "check_command", "domain"):
            if tpl.get(k) and getattr(body, k) in ("", "general"):
                setattr(body, k, tpl[k])
        if tpl.get("max_steps"):
            body.max_steps = int(tpl["max_steps"])
    for k, v in (user_settings(user["id"]).get("role_models") or {}).items():
        if hasattr(body, k) and not getattr(body, k) and v and _model_works(v, user):
            setattr(body, k, v)
    ex = check_model(body.executor_model or default_model(user, settings.default_executor_model), user)
    im = check_model(body.improver_model or default_model(user, settings.default_improver_model), user)
    if body.domain in ("", "general") and body.profile not in ("", "general"):
        from ..harness.profiles import PROFILES as _P

        body.domain = (_P.get(body.profile) or {}).get("domain", "general")
    domain = re.sub(r"[^a-z0-9_-]+", "-", (body.domain or "general").strip().lower()).strip("-") or "general"
    vm = check_model(body.vision_model.strip(), user) if body.vision_model.strip() else default_model(user, "", "vision_model")
    gm = check_model(body.image_model.strip(), user) if body.image_model.strip() else ""
    tm = check_model(body.transcribe_model.strip(), user) if body.transcribe_model.strip() else ""
    pid = db.insert(
        "INSERT INTO projects (user_id, name, domain, executor_model, improver_model, check_command, max_steps, "
        "harness_version, created_at, vision_model, image_model, folder, transcribe_model) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
        (user["id"], body.name.strip(), domain, ex, im, body.check_command.strip(), body.max_steps, time.time(), vm, gm,
         folder, tm),
    )
    from ..harness.profiles import PROFILES, detect_profile, profile_harness

    profile = body.profile if body.profile in PROFILES else "general"
    harness, note = profile_harness(profile), f"Harness base de nesicode · perfil {PROFILES[profile]['label']}"
    if tpl and tpl.get("harness") and not validate(tpl["harness"]):
        harness, note = tpl["harness"], f"Importado de «{tpl.get('from', 'otro equipo')}»"
        profile = detect_profile(harness)
    db.execute(
        "INSERT INTO harness_versions (project_id, version, files, source, note, created_at) VALUES (?, 1, ?, 'base', ?, ?)",
        (pid, json.dumps(harness), note, time.time()),
    )
    db.execute("UPDATE projects SET profile=? WHERE id=?", (profile, pid))
    p = db.one("SELECT * FROM projects WHERE id=?", (pid,))
    ws = project_ws(p)
    ws.mkdir(parents=True, exist_ok=True)
    if body.git_url.strip() and not folder:
        url = body.git_url.strip()
        if not re.match(r"^https://[\w.-]+/[\w./~-]+$", url):
            raise HTTPException(422, "Solo se admiten URLs git https://")
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(ws)], capture_output=True, text=True, timeout=300,
            env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"},
        )
        if proc.returncode != 0:
            db.execute("DELETE FROM projects WHERE id=?", (pid,))
            shutil.rmtree(ws, ignore_errors=True)
            raise HTTPException(422, f"git clone falló: {proc.stderr[-500:]}")
    from .config_io import import_bots

    import_bots(db, bots, user["id"])  # bots imported before any project existed land here
    return project_view(p)


@app.get("/api/projects/{project_id}")
def read_project(project_id: int, user: dict = Depends(current_user)):
    return project_view(get_project(user, project_id))


@app.patch("/api/projects/{project_id}")
def patch_project(project_id: int, body: ProjectPatch, user: dict = Depends(current_user)):
    get_project(user, project_id)
    fields = body.model_dump(exclude_none=True)
    for key in ("executor_model", "improver_model", "vision_model", "image_model", "transcribe_model"):
        if fields.get(key):
            check_model(fields[key], user)
    if fields:
        sets = ", ".join(f"{k}=?" for k in fields)
        db.execute(f"UPDATE projects SET {sets} WHERE id=?", (*fields.values(), project_id))
    return project_view(get_project(user, project_id))


@app.get("/api/profiles")
def list_profiles(user: dict = Depends(current_user)):
    from ..harness.profiles import profile_list

    return profile_list()


class ProfileChange(BaseModel):
    profile: str = Field(pattern="^[a-z0-9_-]{1,30}$")


@app.post("/api/projects/{project_id}/profile")
def change_profile(project_id: int, body: ProfileChange, user: dict = Depends(current_user)):
    """Switch the project's profile: a new harness version with the new profile's blocks and skills
    (everything else the harness learned is kept; older versions stay in the history)."""
    from ..harness.profiles import PROFILES, apply_profile

    p = get_project(user, project_id)
    if body.profile not in PROFILES:
        raise HTTPException(422, "Perfil desconocido")
    files = apply_profile(current_harness(db, p), body.profile)
    errors = validate(files)
    if errors:
        raise HTTPException(422, "; ".join(errors))
    v = save_harness_version(db, project_id, files, "manual", f"Perfil: {PROFILES[body.profile]['label']}")
    db.execute("UPDATE projects SET profile=? WHERE id=?", (body.profile, project_id))
    if p.get("domain") in ("general", "", None) or p.get("domain") == (PROFILES.get(p.get("profile") or "general") or {}).get("domain"):
        db.execute("UPDATE projects SET domain=? WHERE id=?", (PROFILES[body.profile]["domain"], project_id))
    return {**project_view(get_project(user, project_id)), "harness_version": v}


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: int, user: dict = Depends(current_user)):
    p = get_project(user, project_id)
    for r in db.all("SELECT id FROM runs WHERE project_id=?", (project_id,)):
        jobs.cancel(r["id"])
    db.execute("DELETE FROM projects WHERE id=?", (project_id,))
    shutil.rmtree(project_root(p), ignore_errors=True)
    return {"ok": True}


# ----------------------------------------------------------------- workspace
@app.get("/api/projects/{project_id}/tree")
def workspace_tree(project_id: int, user: dict = Depends(current_user)):
    root = project_ws(get_project(user, project_id))
    out: list[dict] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        out.append({"path": str(rel), "dir": p.is_dir(), "size": 0 if p.is_dir() else p.stat().st_size})
        if len(out) >= 3000:
            break
    return out


@app.get("/api/projects/{project_id}/file")
def workspace_file(project_id: int, path: str, user: dict = Depends(current_user)):
    root = project_ws(get_project(user, project_id)).resolve()
    f = (root / path).resolve()
    if root not in f.parents or not f.is_file():
        raise HTTPException(404, "Fichero no encontrado")
    data = f.read_bytes()[:300_000]
    if b"\0" in data[:8192]:
        return {"path": path, "binary": True, "content": ""}
    return {"path": path, "binary": False, "content": data.decode("utf-8", "replace")}


@app.post("/api/projects/{project_id}/upload")
async def upload(project_id: int, file: UploadFile = File(...), user: dict = Depends(current_user)):
    root = project_ws(get_project(user, project_id)).resolve()
    data = await file.read()
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(413, "Máximo 200 MB")
    name = Path(file.filename or "upload").name
    if name.endswith(".zip"):
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise HTTPException(422, "ZIP no válido") from exc
        count = 0
        for info in zf.infolist():
            target = (root / info.filename).resolve()
            if root not in target.parents and target != root:
                raise HTTPException(422, f"Ruta peligrosa en el ZIP: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
        return {"ok": True, "files": count}
    (root / name).write_bytes(data)
    return {"ok": True, "files": 1}


# ---------------------------------------------------------------------- harness
class HarnessSave(BaseModel):
    files: dict[str, str]
    note: str = Field(default="", max_length=500)


@app.get("/api/harness/components")
def harness_components():
    return COMPONENTS


@app.get("/api/projects/{project_id}/harness")
def read_harness(project_id: int, user: dict = Depends(current_user)):
    p = get_project(user, project_id)
    files = current_harness(db, p)
    return {
        "version": p["harness_version"], "files": files, "inventory": inventory(files),
        "components": {path: component_of(path) for path in files}, "errors": validate(files),
    }


@app.get("/api/projects/{project_id}/harness/versions")
def harness_versions(project_id: int, user: dict = Depends(current_user)):
    get_project(user, project_id)
    rows = db.all(
        "SELECT version, source, note, run_id, created_at, files FROM harness_versions WHERE project_id=? ORDER BY version DESC",
        (project_id,),
    )
    for r in rows:
        r["inventory"] = inventory(loads(r.pop("files"), {}))
    return rows


@app.get("/api/projects/{project_id}/harness/versions/{version}")
def harness_version(project_id: int, version: int, user: dict = Depends(current_user)):
    get_project(user, project_id)
    row = db.one("SELECT * FROM harness_versions WHERE project_id=? AND version=?", (project_id, version))
    if not row:
        raise HTTPException(404, "Versión no encontrada")
    files = loads(row["files"], {})
    prev = db.one(
        "SELECT files FROM harness_versions WHERE project_id=? AND version<? ORDER BY version DESC LIMIT 1",
        (project_id, version),
    )
    prev_files = loads(prev["files"], {}) if prev else {}
    return {**{k: row[k] for k in ("version", "source", "note", "run_id", "created_at")}, "files": files,
            "diff": unified_diff(prev_files, files), "changes": diff_summary(prev_files, files)}


@app.post("/api/projects/{project_id}/harness/validate")
def harness_validate(project_id: int, body: HarnessSave, user: dict = Depends(current_user)):
    get_project(user, project_id)
    return {"errors": validate(normalize(body.files))}


@app.post("/api/projects/{project_id}/harness")
def harness_save(project_id: int, body: HarnessSave, user: dict = Depends(current_user)):
    p = get_project(user, project_id)
    files = normalize({k.strip(): v for k, v in body.files.items()})
    errors = validate(files)
    if errors:
        return JSONResponse({"errors": errors}, status_code=422)
    if files == current_harness(db, p):
        return {"version": p["harness_version"], "unchanged": True}
    v = save_harness_version(db, project_id, files, "manual", body.note.strip() or "Edición manual")
    return {"version": v}


@app.post("/api/projects/{project_id}/harness/restore/{version}")
def harness_restore(project_id: int, version: int, user: dict = Depends(current_user)):
    get_project(user, project_id)
    row = db.one("SELECT files FROM harness_versions WHERE project_id=? AND version=?", (project_id, version))
    if not row:
        raise HTTPException(404, "Versión no encontrada")
    v = save_harness_version(db, project_id, loads(row["files"], {}), "restore", f"Restaurada la v{version}")
    return {"version": v}


# ---------------------------------------------------------------------- chat
THREAD_COLS = "t.id, t.project_id, t.title, t.isolated, t.exec_mode, t.allowed_tools, t.created_at, t.updated_at"
MAX_ATTACHMENT = 25 * 1024 * 1024


class ThreadCreate(BaseModel):
    title: str = Field(default="Nueva conversación", max_length=200)


class MessageCreate(BaseModel):
    text: str = Field(min_length=1, max_length=40_000)
    attachments: list[str] = Field(default_factory=list, max_length=20)


class UserSettings(BaseModel):
    language: str | None = Field(default=None, pattern="^(es|en)$")
    exec_mode: str | None = Field(default=None, pattern="^(auto|ask_bash|ask_all)$")
    share_workspace: bool | None = None
    auto_harness: bool | None = None
    auto_harness_daily: int | None = Field(default=None, ge=1, le=50)
    auto_harness_min_steps: int | None = Field(default=None, ge=3, le=300)
    auto_harness_min_errors: int | None = Field(default=None, ge=1, le=50)
    browser_usage: str | None = Field(default=None, pattern="^(mention|auto|ask)$")


SETTINGS_DEFAULTS = {
    "language": "es",
    "exec_mode": "auto",
    "share_workspace": False,
    "auto_harness": False,
    "auto_harness_daily": 3,
    "auto_harness_min_steps": 20,
    "auto_harness_min_errors": 3,
    "hidden_providers": ["mock"],
    "browser_usage": "mention",
}


def user_settings(user_id: int) -> dict:
    row = db.one("SELECT settings FROM users WHERE id=?", (user_id,))
    return {**SETTINGS_DEFAULTS, **(loads(row["settings"], {}) if row else {})}


@app.get("/api/me/settings")
def get_settings(user: dict = Depends(current_user)):
    return user_settings(user["id"])


@app.put("/api/me/settings")
def put_settings(body: UserSettings, user: dict = Depends(current_user)):
    merged = {**user_settings(user["id"]), **body.model_dump(exclude_none=True)}
    db.execute("UPDATE users SET settings=? WHERE id=?", (json.dumps(merged), user["id"]))
    return merged


@app.get("/api/projects/{project_id}/threads")
def list_threads(project_id: int, user: dict = Depends(current_user)):
    get_project(user, project_id)
    return db.all(f"SELECT {THREAD_COLS} FROM threads t WHERE t.project_id=? ORDER BY t.updated_at DESC", (project_id,))


@app.get("/api/threads")
def all_threads(user: dict = Depends(current_user)):
    return db.all(
        f"SELECT {THREAD_COLS}, p.name AS project_name FROM threads t JOIN projects p ON p.id=t.project_id "
        "WHERE p.user_id=? ORDER BY t.updated_at DESC LIMIT 200",
        (user["id"],),
    )


@app.post("/api/projects/{project_id}/threads")
def create_thread(project_id: int, body: ThreadCreate, user: dict = Depends(current_user)):
    p = get_project(user, project_id)
    isolated = 0 if user_settings(user["id"])["share_workspace"] else 1
    now = time.time()
    tid = db.insert(
        "INSERT INTO threads (project_id, title, isolated, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (project_id, body.title.strip() or "Nueva conversación", isolated, now, now),
    )
    if isolated:
        db.execute("UPDATE threads SET manifest=? WHERE id=?", (json.dumps(create_cage(p, tid)), tid))
    return db.one(f"SELECT {THREAD_COLS} FROM threads t WHERE t.id=?", (tid,))


def get_thread(user: dict, thread_id: int) -> tuple[dict, dict]:
    t = db.one("SELECT * FROM threads WHERE id=?", (thread_id,))
    if not t:
        raise HTTPException(404, "Conversación no encontrada")
    return t, get_project(user, t["project_id"])


def thread_out(t: dict) -> dict:
    return {k: v for k, v in t.items() if k != "manifest"}


@app.get("/api/threads/{thread_id}")
def read_thread(thread_id: int, user: dict = Depends(current_user)):
    t, _ = get_thread(user, thread_id)
    runs = db.all("SELECT * FROM runs WHERE thread_id=? ORDER BY created_at, id", (thread_id,))
    msgs = []
    for r in runs:
        view = run_view(r)
        view["events"] = [{"seq": s, **e} for s, e in db.events_after(r["id"], 0, limit=5000)]
        msgs.append(view)
    tg = db.one("SELECT chat_id, active_thread FROM telegram_links WHERE user_id=?", (user["id"],)) or {}
    b = db.one("SELECT * FROM bots WHERE thread_id=? AND user_id=?", (thread_id, user["id"]))
    return {"thread": {**thread_out(t), "effective_mode": jobs.exec_mode(t, user["id"]),
                       "telegram": {"linked": bool(tg.get("chat_id")), "attached": tg.get("active_thread") == t["id"]},
                       "bot": _bot_view(b) if b else None},
            "runs": msgs}


class ThreadPatch(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    exec_mode: str | None = Field(default=None, pattern="^(|auto|ask_bash|ask_all)$")  # '' = my default
    reset_allowed: bool = False


@app.patch("/api/threads/{thread_id}")
def patch_thread(thread_id: int, body: ThreadPatch, user: dict = Depends(current_user)):
    get_thread(user, thread_id)
    if body.title is not None:
        db.execute("UPDATE threads SET title=?, title_locked=1 WHERE id=?", (body.title.strip() or "Conversación", thread_id))
    if body.exec_mode is not None:
        db.execute("UPDATE threads SET exec_mode=? WHERE id=?", (body.exec_mode, thread_id))
    if body.reset_allowed:
        db.execute("UPDATE threads SET allowed_tools='[]' WHERE id=?", (thread_id,))
    t = db.one(f"SELECT {THREAD_COLS} FROM threads t WHERE t.id=?", (thread_id,))
    return {**t, "effective_mode": jobs.exec_mode(t, user["id"])}


class Decision(BaseModel):
    call_id: str = Field(min_length=1, max_length=200)
    decision: str = Field(pattern="^(allow|always|deny)$")


@app.post("/api/runs/{run_id}/approve")
def approve_call(run_id: int, body: Decision, user: dict = Depends(current_user)):
    get_run(user, run_id)
    if not jobs.decide(run_id, body.call_id, body.decision):
        raise HTTPException(409, "Esa petición ya no está esperando respuesta")
    return {"ok": True}


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: int, user: dict = Depends(current_user)):
    t, p = get_thread(user, thread_id)
    for r in db.all("SELECT id FROM runs WHERE thread_id=?", (thread_id,)):
        jobs.cancel(r["id"])
    db.execute("DELETE FROM threads WHERE id=?", (thread_id,))
    shutil.rmtree(thread_dir(p, thread_id), ignore_errors=True)
    return {"ok": True}


# ---- the conversation's cage: its files, its own changes, applying them to the project
def _in(root: Path, rel: str) -> Path:
    root = root.resolve()
    f = (root / rel).resolve()
    if f != root and root not in f.parents:
        raise HTTPException(404, "Fichero no encontrado")
    return f


@app.get("/api/threads/{thread_id}/tree")
def thread_tree(thread_id: int, user: dict = Depends(current_user)):
    t, p = get_thread(user, thread_id)
    root = thread_ws(p, t)
    out = []
    if root.is_dir():
        for f in sorted(root.rglob("*")):
            rel = f.relative_to(root)
            if any(part in IGNORE_DIRS for part in rel.parts):
                continue
            out.append({"path": str(rel), "dir": f.is_dir(), "size": 0 if f.is_dir() else f.stat().st_size})
            if len(out) >= 3000:
                break
    return out


def _inline_local_images(body: bytes, page: Path, root: Path) -> bytes:
    """<img src="chart.png"> next to the page → data: URI, so a sandboxed page still shows its local images."""
    import base64 as _b64

    html = body.decode("utf-8", "replace")

    def repl(m):
        quote, ref = m.group(2), m.group(3)
        if re.match(r"^(https?:|data:|blob:|#|/)", ref):
            return m.group(0)
        try:
            target = (page.parent / ref.split("?")[0]).resolve()
            target.relative_to(root.resolve())  # never outside the conversation's workspace
        except (ValueError, OSError):
            return m.group(0)
        mime = image_mime(target.name)
        if not mime or not target.is_file() or target.stat().st_size > 5 * 1024 * 1024:
            return m.group(0)
        return f"{m.group(1)}{quote}data:{mime};base64,{_b64.b64encode(target.read_bytes()).decode()}{quote}"

    return re.sub(r"""(<img\b[^>]*?\bsrc=)(["'])([^"']+)\2""", repl, html, flags=re.I).encode()


HTML_SANDBOX_CSP = (
    # generated pages run in an opaque origin: no cookies, no access to the app or its API, no network calls
    "sandbox allow-scripts allow-popups allow-modals allow-downloads; default-src 'none'; "
    "script-src 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://unpkg.com; "
    "style-src 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; font-src data: https://fonts.gstatic.com; "
    "img-src data: blob: https:; media-src data: blob:; connect-src 'none'; frame-ancestors 'self'"
)


@app.get("/api/threads/{thread_id}/file")
def thread_file(thread_id: int, path: str, download: int = 0, view: int = 0, user: dict = Depends(current_user)):
    t, p = get_thread(user, thread_id)
    f = _in(thread_ws(p, t), path)
    if not f.is_file():
        raise HTTPException(404, "Fichero no encontrado")
    if download:
        import mimetypes

        return FileResponse(f, media_type=mimetypes.guess_type(f.name)[0] or "application/octet-stream", filename=f.name)
    if view and f.suffix.lower() in (".html", ".htm", ".svg"):
        if f.stat().st_size > 15 * 1024 * 1024:
            raise HTTPException(413, "Demasiado grande para verlo aquí: descárgalo")
        body = f.read_bytes()
        if f.suffix.lower() in (".html", ".htm"):
            body = _inline_local_images(body, f, thread_ws(p, t))
        return Response(body, media_type="image/svg+xml" if f.suffix.lower() == ".svg" else "text/html; charset=utf-8",
                        headers={"Content-Security-Policy": HTML_SANDBOX_CSP, "X-Content-Type-Options": "nosniff",
                                 "Cache-Control": "no-store"})
    if image_mime(f.name):
        return FileResponse(f, media_type=image_mime(f.name))
    data = f.read_bytes()[:300_000]
    if b"\0" in data[:8192]:
        return {"path": path, "binary": True, "content": ""}
    return {"path": path, "binary": False, "content": data.decode("utf-8", "replace")}


@app.get("/api/threads/{thread_id}/changes")
def thread_changes(thread_id: int, user: dict = Depends(current_user)):
    t, p = get_thread(user, thread_id)
    if not t["isolated"]:
        return {"isolated": False, "added": [], "modified": [], "removed": [], "diff": ""}
    ch = changes_since(loads(t["manifest"], {}), thread_ws(p, t), project_ws(p))
    return {"isolated": True, **ch}


@app.post("/api/threads/{thread_id}/apply")
def thread_apply(thread_id: int, user: dict = Depends(current_user)):
    t, p = get_thread(user, thread_id)
    if not t["isolated"]:
        raise HTTPException(422, "Esta conversación ya trabaja directamente sobre el proyecto")
    if db.one("SELECT id FROM runs WHERE thread_id=? AND status IN ('queued', 'running')", (thread_id,)):
        raise HTTPException(409, "Espera a que termine la respuesta en curso")
    cage = thread_ws(p, t)
    with jobs._project_lock(p["id"]):
        applied = apply_cage(loads(t["manifest"], {}), cage, project_ws(p))
    # Re-baseline: later applies only carry changes made after this one.
    db.execute("UPDATE threads SET manifest=? WHERE id=?", (json.dumps(manifest(cage)), thread_id))
    return applied


@app.post("/api/threads/{thread_id}/attachments")
async def upload_attachments(thread_id: int, files: list[UploadFile] = File(...), user: dict = Depends(current_user)):
    t, p = get_thread(user, thread_id)
    target = thread_ws(p, t) / ATTACHMENTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    out = []
    for up in files[:20]:
        data = await up.read()
        if len(data) > MAX_ATTACHMENT:
            raise HTTPException(413, f"{up.filename}: máximo 25 MB por archivo")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(up.filename or "archivo").name).strip("._") or "archivo"
        stem, dot, ext = name.rpartition(".")
        if not dot:
            stem, ext = name, ""
        candidate, n = name, 1
        while (target / candidate).exists():
            n += 1
            candidate = f"{stem}-{n}.{ext}" if ext else f"{stem}-{n}"
        (target / candidate).write_bytes(data)
        out.append(_attachment_meta(target / candidate, f"{ATTACHMENTS_DIR}/{candidate}"))
    return out


def _attachment_meta(f: Path, rel: str) -> dict:
    head = f.read_bytes()[:8192]
    kind = "image" if image_mime(f.name) else "pdf" if f.suffix.lower() == ".pdf" else ("binary" if b"\0" in head else "text")
    return {"name": f.name, "path": rel, "kind": kind, "size": f.stat().st_size}


@app.post("/api/threads/{thread_id}/messages")
def send_message(thread_id: int, body: MessageCreate, user: dict = Depends(current_user)):
    t, p = get_thread(user, thread_id)
    root = thread_ws(p, t)
    atts = []
    for rel in body.attachments:
        if not rel.startswith(ATTACHMENTS_DIR + "/"):
            raise HTTPException(422, f"Adjunto no válido: {rel}")
        f = _in(root, rel)
        if not f.is_file():
            raise HTTPException(422, f"El adjunto ya no existe: {rel}")
        atts.append(_attachment_meta(f, rel))
    first = db.one("SELECT COUNT(*) AS n FROM runs WHERE thread_id=?", (thread_id,))["n"] == 0
    bot = db.one("SELECT id FROM bots WHERE thread_id=? AND user_id=?", (thread_id, user["id"]))
    rid = db.insert(
        "INSERT INTO runs (project_id, thread_id, kind, task, status, created_at, attachments, bot_id, trigger) "
        "VALUES (?, ?, 'chat', ?, 'waiting', ?, ?, ?, ?)",
        (p["id"], thread_id, body.text, time.time(), json.dumps(atts), bot["id"] if bot else None, "chat" if bot else ""),
    )
    if first and t["title"] == "Nueva conversación":
        title = re.sub(r"\s+", " ", body.text).strip()[:60]
        db.execute("UPDATE threads SET title=? WHERE id=?", (title, thread_id))
    db.execute("UPDATE threads SET updated_at=? WHERE id=?", (time.time(), thread_id))
    jobs.enqueue_chat(rid, thread_id)  # starts now, or waits its turn behind the reply in progress
    return run_view(db.one("SELECT * FROM runs WHERE id=?", (rid,)))


# ------------------------------------------------------------- microphone
@app.get("/api/projects/{project_id}/transcribe")
def transcribe_info(project_id: int, user: dict = Depends(current_user)):
    from ..speech import native_available

    p = get_project(user, project_id)
    engine = "model" if p.get("transcribe_model") else "macos" if native_available() else None
    return {"engine": engine, "model": p.get("transcribe_model") or ""}


@app.post("/api/projects/{project_id}/transcribe")
async def transcribe_audio(project_id: int, request: Request, audio: UploadFile = File(...),
                           user: dict = Depends(current_user)):
    """Microphone button: audio in, text out (the user reviews it before sending)."""
    import tempfile

    from ..speech import SpeechError, native_available, transcribe_native

    p = get_project(user, project_id)
    form = await request.form()
    lang = str(form.get("lang") or user_settings(user["id"]).get("lang") or "es")[:10]
    data = await audio.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "Audio demasiado largo (máx. 25 MB)")
    if len(data) < 2000:
        return {"text": "", "engine": "none"}
    spec = p.get("transcribe_model") or ""

    def run() -> dict:
        if spec:
            provider, model = get_provider(spec, auth.user_keys(db, user["id"]))
            return {"text": provider.transcribe(model, data, audio.filename or "audio.wav", lang.split("-")[0]), "engine": spec}
        if native_available():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data)
            try:
                return {"text": transcribe_native(f.name, lang), "engine": "macos"}
            finally:
                os.unlink(f.name)
        raise HTTPException(422, "No hay transcripción configurada: elige un modelo de «Transcripción» en Ajustes del proyecto "
                                 "(p. ej. openai/whisper-1 o groq/whisper-large-v3). En la app de Mac funciona sin configurar nada.")

    try:
        return await asyncio.to_thread(run)
    except (LLMError, SpeechError) as exc:
        raise HTTPException(422, str(exc)) from exc


# ----------------------------------------------------------------- harnessing
class HarnessingCreate(BaseModel):
    task: str = Field(min_length=1, max_length=40_000)
    k: int = Field(default=1, ge=1, le=4)
    mode: str = Field(default="parallel", pattern="^(parallel|sequential)$")
    apply_changes: bool = False
    adopt_harness: bool = True


@app.post("/api/projects/{project_id}/harnessing")
def start_harnessing(project_id: int, body: HarnessingCreate, user: dict = Depends(current_user)):
    p = get_project(user, project_id)
    rid = db.insert(
        "INSERT INTO runs (project_id, kind, task, status, params, created_at) VALUES (?, 'harness', ?, 'queued', ?, ?)",
        (p["id"], body.task, json.dumps(body.model_dump(exclude={"task"})), time.time()),
    )
    jobs.submit_harnessing(rid)
    return run_view(db.one("SELECT * FROM runs WHERE id=?", (rid,)))


@app.post("/api/runs/{run_id}/improve")
def improve_run(run_id: int, user: dict = Depends(current_user)):
    """Option A: 'Mejorar harness con esta tarea' from a chat answer."""
    run, p = get_run(user, run_id)
    if run["kind"] != "chat" or run["status"] in ("queued", "running"):
        raise HTTPException(422, "Solo se puede mejorar a partir de una respuesta de chat terminada")
    if db.one("SELECT id FROM runs WHERE project_id=? AND kind='harness' AND status IN ('queued', 'running')", (p["id"],)):
        raise HTTPException(409, "Ya hay un harnessing en curso en este proyecto")
    rid = jobs.improve_from_chat(run, p, auto=False)
    return run_view(db.one("SELECT * FROM runs WHERE id=?", (rid,)))


@app.get("/api/projects/{project_id}/runs")
def list_runs(project_id: int, kind: str = "harness", user: dict = Depends(current_user)):
    get_project(user, project_id)
    rows = db.all("SELECT * FROM runs WHERE project_id=? AND kind=? ORDER BY id DESC LIMIT 100", (project_id, kind))
    return [run_view(r) for r in rows]


@app.get("/api/runs/{run_id}")
def read_run(run_id: int, user: dict = Depends(current_user)):
    run, _ = get_run(user, run_id)
    view = run_view(run, full=True)
    view["events"] = [{"seq": s, **e} for s, e in db.events_after(run_id, 0, limit=5000)]
    return view


@app.get("/api/runs/{run_id}/rollouts/{label}")
def read_rollout(run_id: int, label: str, user: dict = Depends(current_user)):
    _, p = get_run(user, run_id)
    if not re.match(r"^[a-z]+[0-9]+$", label):
        raise HTTPException(404, "Rollout no encontrado")
    d = run_dir(p, run_id) / "rollouts" / label
    if not d.is_dir():
        raise HTTPException(404, "Rollout no encontrado")

    def read(name: str) -> str:
        f = d / name
        return f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""

    return {
        "label": label, "trajectory": read("trajectory.txt"), "diff": read("diff.patch"),
        "check": read("check.txt"), "harness": loads(read("harness.json"), {}),
        "events": loads(read("events.json"), []),
    }


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: int, user: dict = Depends(current_user)):
    run, _ = get_run(user, run_id)
    if run["status"] in FINAL:
        return {"ok": False}
    if run["status"] in ("queued", "waiting"):
        db.execute("UPDATE runs SET status='cancelled', finished_at=? WHERE id=?", (time.time(), run_id))
    return {"ok": jobs.cancel(run_id) or run["status"] in ("queued", "waiting")}


@app.post("/api/runs/{run_id}/send-now")
def send_run_now(run_id: int, user: dict = Depends(current_user)):
    """A queued message jumps the queue: the reply in progress stops and this one starts."""
    run, _ = get_run(user, run_id)
    if run["status"] != "waiting":
        raise HTTPException(409, "Ese mensaje ya no está en la cola")
    return {"ok": jobs.send_now(run_id)}


@app.post("/api/runs/{run_id}/adopt")
def adopt_run(run_id: int, user: dict = Depends(current_user)):
    run, p = get_run(user, run_id)
    result = loads(run["result"], {}) or {}
    files = result.get("adopted_harness")
    if run["kind"] != "harness" or not files:
        raise HTTPException(422, "Esta ejecución no tiene un harness adoptable")
    v = save_harness_version(db, p["id"], files, "harnessing", f"Adoptado de la ejecución #{run_id}", run_id)
    result["adopted_version"] = v
    db.execute("UPDATE runs SET result=? WHERE id=?", (json.dumps(result, ensure_ascii=False), run_id))
    return {"version": v}


@app.post("/api/runs/{run_id}/apply")
def apply_run(run_id: int, user: dict = Depends(current_user)):
    run, p = get_run(user, run_id)
    if run["kind"] != "harness" or run["status"] != "done":
        raise HTTPException(422, "Solo se pueden aplicar ejecuciones de harnessing terminadas")
    try:
        return {"changes": jobs.apply_later(run_id, p)}
    except FileNotFoundError as exc:
        raise HTTPException(410, str(exc)) from exc


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: int, request: Request, after: int = 0, user: dict = Depends(current_user)):
    get_run(user, run_id)

    async def gen():
        last = after
        gen_seen, idx = -1, 0
        idle = 0
        while True:
            if await request.is_disconnected():
                return
            sent = False
            for seq, ev in db.events_after(run_id, last):
                last = seq
                sent = True
                yield f"id: {seq}\nevent: event\ndata: {json.dumps({'seq': seq, **ev}, ensure_ascii=False)}\n\n"
            live = jobs.live.get(run_id)
            if live is not None:
                if live.gen != gen_seen:
                    gen_seen, idx = live.gen, 0
                items = live.items[idx:]
                if items:
                    idx += len(items)
                    sent = True
                    yield f"event: live\ndata: {json.dumps(items, ensure_ascii=False)}\n\n"
            status = db.one("SELECT status FROM runs WHERE id=?", (run_id,))
            if status and status["status"] in FINAL and not sent:
                yield f"event: end\ndata: {json.dumps({'status': status['status']})}\n\n"
                return
            idle = 0 if sent else idle + 1
            if idle and idle % 50 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.15)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------- lessons
@app.get("/api/lessons")
def list_lessons(domain: str | None = None, user: dict = Depends(current_user)):
    if domain:
        return db.all("SELECT * FROM lessons WHERE user_id=? AND domain=? ORDER BY created_at DESC", (user["id"], domain))
    return db.all("SELECT * FROM lessons WHERE user_id=? ORDER BY created_at DESC", (user["id"],))


@app.delete("/api/lessons/{lesson_id}")
def delete_lesson(lesson_id: int, user: dict = Depends(current_user)):
    db.execute("DELETE FROM lessons WHERE id=? AND user_id=?", (lesson_id, user["id"]))
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"ok": True, "sandbox": settings.sandbox, "auth": settings.auth_enabled}


# ---------------------------------------------------------------------- SPA
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        f = (WEB_DIST / path).resolve()
        if path and WEB_DIST in f.parents and f.is_file():
            return FileResponse(f)
        return FileResponse(WEB_DIST / "index.html")
