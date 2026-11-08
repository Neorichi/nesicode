"""nesicode web API (FastAPI) + static SPA."""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import secrets
import shutil
import sqlite3
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
from ..agent.workspace import ATTACHMENTS_DIR, apply_cage, manifest
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


# ------------------------------------------------- cross-site requests and DNS rebinding
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
EXTENSION_SCHEMES = {"chrome-extension", "moz-extension", "safari-web-extension"}


def _split_host(value: str) -> tuple[str, int | None]:
    """'Example.com:8420' -> ('example.com', 8420); '[::1]' -> ('::1', None); garbage -> ('', None)."""
    value = (value or "").strip().lower()
    if value.startswith("["):
        host, _, rest = value[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else ""
    elif value.count(":") == 1:
        host, _, port = value.partition(":")
    else:
        host, port = value, ""
    try:
        return host.rstrip("."), (int(port) if port else None)
    except ValueError:
        return "", None


def _env_set(name: str) -> set[str]:
    return {x.strip().lower().rstrip("/") for x in os.environ.get(name, "").split(",") if x.strip()}


def host_allowed(host_header: str) -> bool:
    """Can a browser have sent this Host on purpose? A DNS-rebinding page always arrives with ITS own
    domain (evil.example) as Host, so a domain name must be listed in NESICODE_ALLOWED_HOSTS. IP
    literals, single-label names (localhost, a LAN machine name), *.localhost and *.local cannot be
    pointed at this server by a remote attacker."""
    import ipaddress

    host, _ = _split_host(host_header)
    if not host_header.strip() or host == "localhost" or "." not in host or host.endswith((".localhost", ".local")):
        return True  # (no Host at all: not a browser)
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    allowed = {_split_host(h)[0] if not h.startswith(".") else h for h in _env_set("NESICODE_ALLOWED_HOSTS")}
    return "*" in allowed or host in allowed or any(a.startswith(".") and host.endswith(a) for a in allowed)


def _origin_parts(url: str) -> tuple[str, str, int | None] | None:
    from urllib.parse import urlsplit

    try:
        u = urlsplit(url.strip())
        if not u.scheme or not u.hostname:
            return None
        return u.scheme.lower(), u.hostname.lower().rstrip("."), u.port
    except ValueError:
        return None


def same_origin(url: str, hosts: list[str]) -> bool:
    """Does `url` (an Origin or Referer) point at the host this request was sent to?"""
    o = _origin_parts(url)
    if not o:
        return False
    scheme, host, port = o
    default = {"http": 80, "https": 443}.get(scheme)
    for h in hosts:
        name, p = _split_host(h)
        if name and name == host and (p or default) == (port or default):
            return True
    return False


def _request_hosts(headers) -> list[str]:
    # X-Forwarded-Host (reverse proxies) cannot be set by a cross-site page: custom headers need a CORS preflight
    return [headers.get("host", ""), headers.get("x-forwarded-host", "").split(",")[0]]


def cross_site_reason(method: str, headers) -> str:
    """Why this state-changing request must be refused as cross-site ('' = it is fine). Browsers send
    Origin (or at least Referer / Sec-Fetch-Site) with every POST/PUT/PATCH/DELETE; other programs
    (curl, scripts, the desktop shell) send none of them."""
    if method.upper() in SAFE_METHODS:
        return ""
    hosts, trusted = _request_hosts(headers), _env_set("NESICODE_ALLOWED_ORIGINS")
    origin = headers.get("origin")
    o = origin.strip().lower().rstrip("/") if origin is not None else None
    if o is not None and o in trusted:
        return ""
    site = (headers.get("sec-fetch-site") or "").strip().lower()
    if site and o != "null":
        # Fetch metadata is computed by the browser itself (no page can set or fake it) against the URL it
        # really requested, so it stays right behind a proxy that rewrites Host (nginx's plain proxy_pass)
        return "" if site in ("same-origin", "none") else (o or f"Sec-Fetch-Site: {site}")
    if o is not None:  # older browsers: compare Origin with the host the request was sent to
        return "" if o != "null" and same_origin(o, hosts) else (o or "null")
    referer = headers.get("referer")
    if referer:
        r = _origin_parts(referer)
        as_origin = f"{r[0]}://{r[1]}" + (f":{r[2]}" if r[2] else "") if r else ""
        return "" if same_origin(referer, hosts) or as_origin in trusted else referer[:100]
    return ""


@app.middleware("http")
async def request_guard(request: Request, call_next):
    """No login (`nesicode serve`, desktop) must not mean 'any website can drive the API': refuse
    state-changing requests that come from another site (CSRF) and, without login, Host names that
    only a DNS-rebinding page would send. Bot webhooks are exempt (external callers with their own
    secret, often through a tunnel's domain)."""
    if request.method == "POST" and request.url.path.startswith("/hooks/"):
        return await call_next(request)
    host = request.headers.get("host", "")
    if (not settings.auth_enabled or os.environ.get("NESICODE_ALLOWED_HOSTS")) and not host_allowed(host):
        return JSONResponse({"detail": f"Nombre de servidor no permitido: {host[:100]}. Abre Nesicode en http://127.0.0.1 "
                                       "o http://localhost, o añade ese nombre a NESICODE_ALLOWED_HOSTS."}, status_code=403)
    if reason := cross_site_reason(request.method, request.headers):
        return JSONResponse({"detail": f"Petición rechazada: viene de otra web ({reason}). Si es tu propio dominio, "
                                       "añádelo a NESICODE_ALLOWED_ORIGINS."}, status_code=403)
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
    email = (settings.local_user or auth.LOCAL_EMAIL).strip().lower()
    try:
        import getpass

        name = getpass.getuser() if not settings.local_user else email.split("@")[0]
    except Exception:
        name = email.split("@")[0]
    uid = db.insert(  # no usable password: the first sign-up after switching to NESICODE_AUTH=true claims it
        "INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (email, name, auth.unclaimed_hash(), time.time()),
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
    users = db.all("SELECT id, email, password_hash FROM users ORDER BY id")
    claim = None
    if users and not settings.allow_signup:
        if any(not auth.is_unclaimed(u) for u in users):
            raise HTTPException(403, "El registro está desactivado")
        # Only local-mode accounts exist (made while NESICODE_AUTH was off; nobody knows their password):
        # this first sign-up is the owner and takes that account, with its projects, keys and settings.
        claim = users[0]
    email = body.email.strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(422, "Email no válido")
    if email == auth.LOCAL_EMAIL:
        raise HTTPException(422, "Ese email está reservado: usa el tuyo")
    same = db.one("SELECT id, email, password_hash FROM users WHERE email=?", (email,))
    if claim and same:
        claim = same  # an unclaimed account with this very email (NESICODE_LOCAL_USER)
    elif same:
        raise HTTPException(409, "Ya existe una cuenta con ese email")
    name, pw = body.name.strip() or email.split("@")[0], auth.hash_password(body.password)
    if claim:
        uid = claim["id"]
        # atomic: of two sign-ups racing for the same unclaimed account, only the first one takes it
        if db.execute("UPDATE users SET email=?, name=?, password_hash=? WHERE id=? AND password_hash=?",
                      (email, name, pw, uid, claim["password_hash"])).rowcount != 1:
            raise HTTPException(403, "El registro está desactivado")
    else:
        uid = db.insert("INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                        (email, name, pw, time.time()))
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


def _non_public_ip(addr: str) -> bool:
    """Loopback, private, link-local (cloud metadata), CGNAT 100.64/10, multicast, reserved…: anything
    that is not a public unicast address (IPv4-mapped IPv6 addresses are judged as the IPv4 inside)."""
    import ipaddress

    ip = ipaddress.ip_address(addr.split("%", 1)[0])
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_multicast or ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved


def check_base_url(url: str) -> str:
    """Endpoints must be http(s). With NESICODE_ALLOW_PRIVATE_PROVIDERS=0 (multi-user servers),
    non-public addresses are refused so users cannot probe the server's network. Every URL built from
    user input goes through here: providers, custom and catalog connectors, git clones, imports."""
    from urllib.parse import urlparse

    url = url.strip().rstrip("/")
    try:
        u = urlparse(url)
        host = u.hostname
        u.port  # noqa: B018 — raises ValueError for a bad port
    except ValueError as exc:
        raise HTTPException(422, "El endpoint debe ser una URL http(s)://") from exc
    if u.scheme not in ("http", "https") or not host:
        raise HTTPException(422, "El endpoint debe ser una URL http(s)://")
    if private_networks_forbidden():
        public_addresses(host)
    return url


def private_networks_forbidden() -> bool:
    return os.environ.get("NESICODE_ALLOW_PRIVATE_PROVIDERS", "1") == "0"


def public_addresses(host: str) -> list[str]:
    """The addresses `host` resolves to, all public (else HTTP 422)."""
    import socket

    try:
        addrs = sorted({ai[4][0].split("%", 1)[0] for ai in socket.getaddrinfo(host, None)})
    except OSError as exc:
        raise HTTPException(422, f"No se puede resolver {host}") from exc
    if not addrs or any(_non_public_ip(a) for a in addrs):
        raise HTTPException(422, "Este servidor no permite endpoints en redes privadas")
    return addrs


def git_pin(url: str) -> list[str]:
    """git -c options that make it connect only to addresses checked here. git resolves the name again by
    itself, and a DNS-rebinding name could answer a public address to us and a private one to git."""
    from urllib.parse import urlparse

    if not private_networks_forbidden():
        return []
    u = urlparse(url)
    addrs = public_addresses(u.hostname or "")
    return ["-c", f"http.curloptResolve={u.hostname}:{u.port or 443}:" + ",".join(f"[{a}]" if ":" in a else a for a in addrs)]


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


def fetch_models(kind: str, base_url: str, api_key: str) -> list[str]:
    """list_models, with any malformed answer (an HTML page, `{"data": 5}`…) reported as LLMError
    instead of escaping as a 500: the usual cause is the website's URL instead of the API's."""
    try:
        return list_models(kind, base_url, api_key)
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"la dirección no responde como una API de modelos ({type(exc).__name__}). "
                       "¿Es la URL de la API (p. ej. https://…/v1) y no la de la web?") from exc


def _discover_server(name: str) -> list[str]:
    if name == "mock":
        return ["demo"]
    cfg = server_provider_config(name)
    if not cfg:
        raise LLMError("proveedor desconocido")
    models = fetch_models(cfg["kind"], cfg["base_url"], cfg["api_key"])  # the server key never leaves
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
    models, warning = _clean_models(body.models), ""
    if not models:  # ask the endpoint first: nothing is stored until we know what it answered
        try:
            models = fetch_models(body.kind, base, (body.api_key or "").strip())
        except LLMError as exc:
            models, warning = [], f"No se pudieron obtener los modelos: {exc}"
    now = time.time()
    try:
        db.insert(
            "INSERT INTO user_providers (user_id, name, label, kind, base_url, models, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user["id"], name, body.label.strip() or name, body.kind, base, json.dumps(models), now, now),
        )
    except sqlite3.IntegrityError as exc:  # the same form sent twice while the models were being listed
        raise HTTPException(409, f"Ya tienes un proveedor llamado {name}") from exc
    try:
        if body.api_key and body.api_key.strip():
            vault.put(user["id"], f"provider:{name}", body.api_key.strip())
        assigned = auto_assign(user, name, models)
        view = next(p for p in provider_rows(user["id"]) if p["name"] == name)
    except Exception:  # never leave a half-created provider behind ("Ya tienes un proveedor llamado …")
        db.execute("DELETE FROM user_providers WHERE user_id=? AND name=?", (user["id"], name))
        vault.delete(user["id"], f"provider:{name}")
        raise
    _refresh_limits_bg(user["id"], name, models)
    return {**view, "assigned": assigned, **({"warning": warning} if warning else {})}


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
        models = fetch_models(body.kind, check_base_url(body.base_url), key)
    except LLMError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "models": models}


@app.post("/api/providers/{name}/discover")
def discover_models(name: str, user: dict = Depends(current_user)):
    row = db.one("SELECT * FROM user_providers WHERE user_id=? AND name=?", (user["id"], name))
    try:
        if row:
            models = fetch_models(row["kind"], row["base_url"], vault.get(user["id"], f"provider:{name}") or "")
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
    form = await request.form()
    password = str(form.get("password") or "")
    data = await file.read(5 * 1024 * 1024 + 1)
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(413, "Archivo demasiado grande")
    if str(form.get("preview") or "") in ("1", "true"):  # dry run: what WOULD be imported (commands, URLs, preferences)
        return {"preview": await asyncio.to_thread(_preview_config, user, data, password)}
    # scrypt (and the DB work) off the event loop: other users' requests and live streams keep flowing
    return {"imported": await asyncio.to_thread(_import_config, user, data, password)}


def _unseal_or_422(data: bytes, password: str) -> dict:
    from .config_io import ConfigError, unseal

    try:
        return unseal(data, password)
    except ConfigError as exc:
        raise HTTPException(422, str(exc)) from exc


def _preview_config(user: dict, data: bytes, password: str) -> dict:
    from .config_io import redirected_keys, stricter_usages, summary, validate_import

    doc, report = validate_import(_unseal_or_422(data, password), auth_enabled=settings.auth_enabled, check_url=check_base_url,
                                  settings_model=UserSettings, current_settings=user_settings(user["id"]))
    report["notes"] += redirected_keys(db, user["id"], doc)[1]
    kept, notes = stricter_usages(db, user["id"], doc)
    report["notes"] += notes
    for c in doc.get("connectors", []):
        c["usage"] = kept.get(c["name"], c["usage"])
    return {**summary(doc), **report}


def _import_config(user: dict, data: bytes, password: str) -> dict:
    from ..connectors import manager
    from .config_io import apply, import_bots

    doc = _unseal_or_422(data, password)
    # the same rules as the API: multi-user stdio ban, URL checks, validated preferences, no looser approvals
    result = apply(db, user["id"], doc, auth_enabled=settings.auth_enabled, check_url=check_base_url,
                   settings_model=UserSettings, current_settings=user_settings(user["id"]))
    for c in result["details"]["connectors"]:  # an overwritten connector must not keep running its old config
        try:
            manager().stop(user["id"], c["name"])
        except Exception as exc:
            print("nesicode: could not stop connector", c["name"], exc)
    result["projects_updated"] = _apply_role_models(user, result["role_models"])  # the validated ones
    result["bots_created"] = import_bots(db, bots, user["id"])  # waits for a project if there is none yet
    if result["telegram"]:
        telegram.restart(user["id"])
    for p in result["details"]["providers"]:
        row = db.one("SELECT models FROM user_providers WHERE user_id=? AND name=?", (user["id"], p["name"]))
        _refresh_limits_bg(user["id"], p["name"], loads(row["models"], []) if row else [])
    return result


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
bots.repair_links()  # links from before threads.bot_id existed
bots.start()  # scheduler thread (off with NESICODE_NO_SCHEDULER)
# Runs of deleted conversations kept while their worker finished (see delete_thread): gone when it ends
# (registered after the bots' hook, so that one still sees the row), and leftovers of a previous launch now.
jobs.on_finish.append(lambda run_id: _drop_detached_run(run_id))
db.execute("DELETE FROM runs WHERE kind='chat' AND thread_id IS NULL")


# ------------------------------------------------------------------------ bots
def _bot_view(b: dict, full: bool = False) -> dict:
    linked = db.one("SELECT id FROM threads WHERE id=? AND bot_id=? AND project_id=?", (b.get("thread_id"), b["id"], b["project_id"]))
    if not linked:
        bots.ensure_thread(b)  # its conversation was deleted earlier: give it a fresh one, never a dead link
        b = db.one("SELECT * FROM bots WHERE id=?", (b["id"],))
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
    if bot_id is None:  # a new bot may reuse a deleted bot's id: never its old webhook secret
        vault.delete(user["id"], f"bot:{b['id']}:webhook")
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
    db.execute("UPDATE threads SET bot_id=NULL WHERE bot_id=?", (b["id"],))  # …and the conversation (a plain chat now)
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
    tz = local_tz()  # '' on Windows: then the example has no tz at all (it means this computer's local time)
    cron = '{"kind": "cron", "spec": "min hour dom mon dow"' + (f', "tz": "{tz}"' if tz else "") + "}"
    prompt = (
        "Turn the user's request into a JSON bot configuration for Nesicode. Reply with ONLY a JSON object:\n"
        '{"name": short name, "description": one line, "instructions": complete self-contained brief for an unattended '
        'agent (what to check, how to report, when to stay silent by answering NOTHING_NEW), written in the user\'s language, '
        '"triggers": [' + cron + ' | {"kind": "interval", "spec": "30" (minutes) or "2h"} '
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
    for t in draft.get("triggers") if isinstance(draft.get("triggers"), list) else []:
        if not isinstance(t, dict):
            continue
        if str(t.get("tz") or "").strip().lower() in ("local", "localtime", "system", "local time"):
            t = {**t, "tz": tz}  # models copy placeholders: 'local' means this computer's zone ('' = its local time)
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
    # the webhook trigger row lives and dies with the bot (cascade): a secret left behind by a deleted bot whose
    # id was reused never opens the new one
    b = db.one("SELECT b.* FROM bots b WHERE b.id=? AND EXISTS (SELECT 1 FROM bot_triggers t WHERE t.bot_id=b.id "
               "AND t.kind='webhook')", (bot_id,))
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


class ExtensionSocket:
    """The extension's socket as the browser hub sees it. It remembers the ids of the commands sent
    through it, and only replies to those are passed on: another user's extension (or anyone guessing
    the ids) can never answer this user's commands with made-up page contents."""

    MAX_PENDING = 1000

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.sent: dict[str, None] = {}  # insertion-ordered set of command ids awaiting a reply

    def _remember(self, data) -> None:
        if isinstance(data, dict) and data.get("id") is not None:
            self.sent[str(data["id"])] = None
            while len(self.sent) > self.MAX_PENDING:  # commands that timed out never get a reply
                self.sent.pop(next(iter(self.sent)))

    async def send_json(self, data, *args, **kwargs):
        self._remember(data)
        return await self.ws.send_json(data, *args, **kwargs)

    async def send_text(self, data: str):
        try:
            self._remember(json.loads(data))
        except ValueError:
            pass
        return await self.ws.send_text(data)

    def claim(self, msg) -> bool:
        """Is this a reply to a command sent through THIS socket (each id answers once)?"""
        rid = str(msg.get("id")) if isinstance(msg, dict) and msg.get("id") is not None else None
        return rid is not None and self.sent.pop(rid, False) is None

    def __getattr__(self, name):
        return getattr(self.ws, name)


def extension_origin_ok(origin: str | None, headers) -> bool:
    """The extension connects from chrome-extension://…; other programs send no Origin. A web page
    (another site) has no business here, even with a stolen pairing code."""
    if not origin:
        return True
    o = origin.strip().lower().rstrip("/")
    return (o.split("://", 1)[0] in EXTENSION_SCHEMES or o in _env_set("NESICODE_ALLOWED_ORIGINS")
            or (o != "null" and same_origin(o, _request_hosts(headers))))


@app.websocket("/api/browser/ws")
async def browser_ws(ws: WebSocket):
    from ..browser import hub

    token = ws.query_params.get("token", "")
    row = db.one("SELECT user_id FROM browser_pairings WHERE token_hash=?", (_pairing_hash(token),)) if token else None
    if not row or not extension_origin_ok(ws.headers.get("origin"), ws.headers):
        await ws.close(code=4401)
        return
    await ws.accept()
    user_id = row["user_id"]
    sock = ExtensionSocket(ws)
    try:
        hello = await asyncio.wait_for(ws.receive_json(), 10)
        hello = hello if isinstance(hello, dict) else {}
        hub.attach(user_id, sock, {"browser": str(hello.get("browser", ""))[:120], "version": str(hello.get("version", ""))[:20]})
        await ws.send_json({"type": "welcome", "user": user_id})
        while True:
            msg = await ws.receive_json()
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
            elif sock.claim(msg):  # only answers to commands this very socket received
                hub.on_message(msg)
    except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError, ValueError):
        pass
    finally:
        hub.detach(user_id, sock)


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


def _host_only_connector(row: dict) -> bool:
    """Runs a command on this machine or talks to its local services: never allowed on a multi-user server."""
    from ..connectors import catalog_entry

    return row["kind"] == "stdio" or bool((catalog_entry(row.get("template") or "") or {}).get("local_only"))


def _disable_host_only_connectors() -> None:
    """Multi-user servers: such connectors may exist from local mode or from an import made before imports
    were checked; they stay switched off (the chat only starts enabled connectors)."""
    if settings.auth_enabled:
        for row in db.all("SELECT id, kind, template FROM user_connectors WHERE enabled=1"):
            if _host_only_connector(row):
                db.execute("UPDATE user_connectors SET enabled=0 WHERE id=?", (row["id"],))


_disable_host_only_connectors()
LOCAL_ONLY_MSG = "Los conectores locales solo están disponibles en modo local (sin login)"


def _start_and_list(user: dict, name: str) -> list[str]:
    """(Re)start a connector and return its tool names; raises HTTP 422 with the real error."""
    from ..connectors import manager
    from .connectors_store import connector_configs

    for row, cfg in connector_configs(db, user["id"]):
        if row["name"] == name:
            if settings.auth_enabled and _host_only_connector(row):
                raise HTTPException(403, LOCAL_ONLY_MSG)
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
        raise HTTPException(403, LOCAL_ONLY_MSG)
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
    if url and not entry.get("local_only"):
        # the host comes from the user: same private-network rule as custom endpoints. Only checked, never
        # rewritten: the catalog's exact URL is the endpoint (GitHub's ends in '/', and MCP clients do not
        # follow the 301 a server may answer for the other spelling)
        check_base_url(url)
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


def split_command(command: str, windows: bool | None = None) -> list[str]:
    """A command line as the user typed it → argv. POSIX shlex would eat Windows backslashes
    (C:\\Users\\ana\\mcp.js → C:Usersanamcp.js), so Windows uses its own rules (CommandLineToArgvW):
    double quotes group words and vanish, backslashes are literal unless they come before a quote."""
    import shlex

    if not (os.name == "nt" if windows is None else windows):
        return shlex.split(command)
    args: list[str] = []
    cur: list[str] = []
    quoted = have = False
    i = 0
    while i < len(command):
        ch = command[i]
        if ch == "\\":
            n = len(command[i:]) - len(command[i:].lstrip("\\"))
            i += n
            if i < len(command) and command[i] == '"':
                cur.append("\\" * (n // 2) + ('"' if n % 2 else ""))
                i += n % 2  # an odd run escapes the quote; an even one leaves it to open/close a group
            else:
                cur.append("\\" * n)
            have = True
        elif ch == '"':
            if quoted and command[i + 1:i + 2] == '"':  # "" inside quotes is a literal quote
                cur.append('"')
                i += 1
            else:
                quoted = not quoted
            have = True
            i += 1
        elif ch in " \t" and not quoted:
            if have:
                args.append("".join(cur))
            cur, have = [], False
            i += 1
        else:
            cur.append(ch)
            have = True
            i += 1
    if have:
        args.append("".join(cur))
    return args


@app.post("/api/connectors/custom")
def add_custom_connector(body: ConnectorCustom, user: dict = Depends(current_user)):
    name = _unique_connector_name(user["id"], body.label)
    if body.kind == "stdio":
        try:
            parts = split_command(body.command)
        except ValueError as exc:  # an unclosed quote
            raise HTTPException(422, f"Comando no válido: {exc}") from exc
        if not parts or not parts[0].strip():
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

    row = db.one("SELECT * FROM user_connectors WHERE user_id=? AND name=?", (user["id"], name))
    if not row:
        raise HTTPException(404, "Conector no encontrado")
    if body.enabled and settings.auth_enabled and _host_only_connector(row):
        raise HTTPException(403, LOCAL_ONLY_MSG)
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


GIT_ENV_KEYS = {"PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "GIT_SSL_CAINFO", "HTTP_PROXY", "HTTPS_PROXY",
                "NO_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "no_proxy", "all_proxy"}


def git_env(environ: dict | None = None, windows: bool | None = None) -> dict[str, str]:
    """git's environment: this machine's PATH (plus, on Windows, SystemRoot & co., without which networking
    does not even start), proxy and CA settings — and never a terminal prompt for credentials."""
    from ..agent.workspace import WINDOWS_ENV

    windows = os.name == "nt" if windows is None else windows
    keep = GIT_ENV_KEYS | (WINDOWS_ENV if windows else set())
    env = {k: v for k, v in (os.environ if environ is None else environ).items() if k.upper() in keep or k in keep}
    if not env.get("PATH"):
        env["PATH"] = os.defpath if windows else "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def run_git(cmd: list[str], timeout: int = 300) -> tuple[bool, str]:
    """(ok, stderr) of a git command run with git_env()."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=git_env())
        return proc.returncode == 0, proc.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


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
        try:
            if not re.match(r"^https://[\w.-]+/[\w./~-]+$", url):
                raise HTTPException(422, "Solo se admiten URLs git https://")
            check_base_url(url)  # no clones from the server's own network on multi-user servers
            pin = git_pin(url)  # …not even if the name resolves differently when git looks it up again
        except HTTPException:
            db.execute("DELETE FROM projects WHERE id=?", (pid,))
            shutil.rmtree(project_root(p), ignore_errors=True)
            raise
        # No redirects (they could lead to a private address) and no stored credentials of the server's account.
        # Multi-user: links in a cloned repo could point at other users' files; check them out as plain files.
        cmd = ["git", "-c", "http.followRedirects=false", "-c", "credential.helper=", *pin,
               *(["-c", "core.symlinks=false"] if settings.auth_enabled else []), "clone", "--depth", "1", url, str(ws)]
        ok, err = run_git(cmd)
        if not ok:
            db.execute("DELETE FROM projects WHERE id=?", (pid,))
            shutil.rmtree(project_root(p), ignore_errors=True)
            for path, label in ((str(ws), "<proyecto>"), (str(settings.data_dir), "<datos>")):
                err = err.replace(path, label)  # git's messages name the server's own folders
            raise HTTPException(422, f"git clone falló: {err[-500:]}")
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


THREAD_DELETE_GRACE, PROJECT_DELETE_GRACE = 1.5, 5.0  # seconds a deleted conversation's/project's runs get to stop


def halt_runs(where: str, args: tuple, grace: float) -> dict[int, str]:
    """Stop these runs (as the Stop button does) and give their workers up to `grace` seconds to wrap up.
    Returns {id: status} of those still in a worker after that: 'running' (a step that ignores Stop: an MCP
    call, a connector starting…) or 'queued' (its job has not started yet: every worker is busy). Their rows
    must outlive that job: run ids are reused by SQLite, and a job that runs or finishes later works on its
    id — on whatever new run (maybe another user's) got it meanwhile."""
    ids = [r["id"] for r in db.all(f"SELECT id FROM runs WHERE {where} AND status IN ('waiting', 'queued', 'running')", args)]
    if not ids:
        return {}
    for rid in ids:
        jobs.cancel(rid)
    marks = ",".join("?" * len(ids))
    db.execute(f"UPDATE runs SET status='cancelled', finished_at=? WHERE status='waiting' AND id IN ({marks})", (time.time(), *ids))
    deadline = time.time() + grace
    while True:
        busy = {r["id"]: r["status"] for r in db.all(f"SELECT id, status FROM runs WHERE status IN ('queued', 'running') "
                                                     f"AND id IN ({marks})", tuple(ids))}
        if "running" not in busy.values() or time.time() >= deadline:
            return busy
        time.sleep(0.05)


def _drop_detached_run(run_id: int) -> None:
    """The reply of a deleted conversation has finished: its run, kept only so its id could not be reused
    while the worker was alive, goes now (see delete_thread)."""
    db.execute("DELETE FROM runs WHERE id=? AND kind='chat' AND thread_id IS NULL", (run_id,))


def _forget_telegram_targets(user_id: int, project_id: int | None = None, thread_ids: list[int] | tuple = ()) -> None:
    """Telegram must not stay attached to a deleted conversation/project: the ids get reused."""
    for tid in thread_ids:
        db.execute("UPDATE telegram_links SET active_thread=NULL WHERE user_id=? AND active_thread=?", (user_id, tid))
    if project_id is not None:
        db.execute("UPDATE telegram_links SET active_project=NULL, active_thread=NULL WHERE user_id=? AND active_project=?",
                   (user_id, project_id))


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: int, user: dict = Depends(current_user)):
    p = get_project(user, project_id)
    # (a run still queued behind busy workers is not waited for: the job queue must notice that its run is
    # gone — jobs-runtime#1 — before it would run whatever new run got the id)
    if "running" in halt_runs("project_id=?", (project_id,), grace=PROJECT_DELETE_GRACE).values():
        raise HTTPException(409, "Hay una respuesta de este proyecto que aún no se ha detenido (p. ej. esperando a un "
                                 "conector). Vuelve a intentarlo en unos segundos.")
    for b in db.all("SELECT id FROM bots WHERE project_id=? AND user_id=?", (project_id, user["id"])):
        vault.delete(user["id"], f"bot:{b['id']}:webhook")  # vault rows do not cascade with the bot
    thread_ids = [t["id"] for t in db.all("SELECT id FROM threads WHERE project_id=?", (project_id,))]
    db.execute("DELETE FROM projects WHERE id=?", (project_id,))
    _forget_telegram_targets(user["id"], project_id, thread_ids)
    shutil.rmtree(project_root(p), ignore_errors=True)
    return {"ok": True}


# ------------------------------------- files inside a workspace, never through a symbolic link
# Workspaces hold links the server must not follow: a cloned repo's, the agent's (`ln -s` works even in its
# sandbox, and in docker mode the link is resolved on the host), the user's. Following one on a write or
# on a read meant for the user lets a project reach outside itself — another user's files on a shared server.
# So paths below a root are opened one component at a time with O_NOFOLLOW (openat-style).
_OPENAT = hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY") and \
    {os.open, os.mkdir, os.unlink, os.stat} <= os.supports_dir_fd
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)  # a FIFO planted in the workspace must not hang a server thread
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NONBLOCK | getattr(os, "O_BINARY", 0)
READ_FLAGS = os.O_RDONLY | _NONBLOCK | getattr(os, "O_BINARY", 0)


class UnsafePath(Exception):
    """A path that leaves its root or goes through a symbolic link (or something that is not a file/folder)."""


def rel_parts(rel: str) -> list[str]:
    parts = [p for p in str(rel).replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or ".." in parts or str(rel).startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", str(rel)):
        raise UnsafePath(str(rel))
    return parts


def _is_link_at(name: str, dir_fd: int) -> bool:
    import stat as _stat

    try:
        return _stat.S_ISLNK(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode)
    except OSError:
        return False


def _open_dirs(root: Path, names: list[str], mkdirs: bool) -> int:
    """fd of root/names…, entered one folder at a time without following links (created if `mkdirs`)."""
    dfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in names:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            for attempt in (0, 1):
                try:
                    nfd = os.open(name, flags, dir_fd=dfd)
                    break
                except FileNotFoundError:
                    if not mkdirs or attempt:
                        raise
                    try:
                        os.mkdir(name, 0o755, dir_fd=dfd)
                    except FileExistsError:
                        pass
                except OSError as exc:
                    if _is_link_at(name, dfd):
                        raise UnsafePath(name) from exc
                    raise
            os.close(dfd)
            dfd = nfd
        return dfd
    except BaseException:
        os.close(dfd)
        raise


def _plain_path(root: Path, parts: list[str], mkdirs: bool) -> Path:
    """No openat (Windows, where links need admin rights): check each component instead."""
    cur = Path(root)
    for name in parts[:-1]:
        cur = cur / name
        if cur.is_symlink():
            raise UnsafePath(name)
        if mkdirs:
            cur.mkdir(exist_ok=True)
    if (cur / parts[-1]).is_symlink():
        raise UnsafePath(parts[-1])
    return cur / parts[-1]


def open_in(root: Path, rel: str, flags: int, *, mkdirs: bool = False, mode: int = 0o644) -> int:
    """os.open(root/rel) without following a symbolic link anywhere below root: UnsafePath if there is
    one. Only regular files come back (a FIFO, socket or device is UnsafePath too)."""
    import stat as _stat

    parts = rel_parts(rel)
    if _OPENAT:
        dfd = _open_dirs(root, parts[:-1], mkdirs)
        try:
            fd = os.open(parts[-1], flags | os.O_NOFOLLOW, mode, dir_fd=dfd)
        except OSError as exc:
            if _is_link_at(parts[-1], dfd):
                raise UnsafePath(rel) from exc
            raise
        finally:
            os.close(dfd)
    else:
        fd = os.open(_plain_path(root, parts, mkdirs), flags, mode)
    if not _stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise UnsafePath(rel)
    return fd


def unlink_in(root: Path, rel: str, *, only_link: bool = False) -> bool:
    """Remove root/rel itself — a link is removed, never followed. False if there was nothing (or, with
    only_link, if it is not a link)."""
    parts = rel_parts(rel)
    if not _OPENAT:
        p = Path(root, *parts[:-1])
        if any(Path(root, *parts[:i]).is_symlink() for i in range(1, len(parts))):
            raise UnsafePath(rel)
        target = p / parts[-1]
        if only_link and not target.is_symlink():
            return False
        target.unlink(missing_ok=True)
        return True
    try:
        dfd = _open_dirs(root, parts[:-1], False)
    except FileNotFoundError:
        return False
    try:
        if only_link and not _is_link_at(parts[-1], dfd):
            return False
        os.unlink(parts[-1], dir_fd=dfd)
        return True
    except FileNotFoundError:
        return False
    finally:
        os.close(dfd)


def mkdirs_in(root: Path, rel: str) -> None:
    """mkdir -p root/rel, never through a link."""
    parts = rel_parts(rel)
    if _OPENAT:
        os.close(_open_dirs(root, parts, True))
    else:
        _plain_path(root, [*parts, "x"], True)


def write_in(root: Path, rel: str, data: bytes) -> None:
    """Write root/rel (folders created) without following links: a link in the folders is UnsafePath; a
    link AT rel is replaced by the file, never written through."""
    try:
        fd = open_in(root, rel, WRITE_FLAGS, mkdirs=True)
    except UnsafePath:
        if not unlink_in(root, rel, only_link=True):
            raise
        fd = open_in(root, rel, WRITE_FLAGS | os.O_EXCL, mkdirs=True)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def read_in(root: Path, rel: str, limit: int) -> bytes:
    """The first `limit` bytes of root/rel (never the whole file), not through a link."""
    with os.fdopen(open_in(root, rel, READ_FLAGS), "rb") as fh:
        return fh.read(limit)


def resolved_rel(root: Path, path: str) -> tuple[Path, str]:
    """(root resolved, path relative to it with internal links resolved) or 404 if it leaves root."""
    try:
        root = root.resolve()
        f = (root / path).resolve()
    except (OSError, ValueError, RuntimeError) as exc:  # a NUL byte, a link loop…
        raise HTTPException(404, "Fichero no encontrado") from exc
    if f == root or root not in f.parents:
        raise HTTPException(404, "Fichero no encontrado")
    return root, str(f.relative_to(root))


def _sorted_entries(path) -> list:
    try:
        with os.scandir(path) as it:
            return sorted(it, key=lambda e: e.name)
    except OSError:
        return []


def list_tree(root: Path, cap: int = 3000) -> list[dict]:
    """Files and folders under root in path order. Links are listed, never entered or followed; ignored
    folders (node_modules, .git…) are skipped, not walked; one odd entry (a dangling link, a file deleted
    meanwhile, no permission) never breaks the listing."""
    out: list[dict] = []
    stack, prefixes = [iter(_sorted_entries(root))], [""]
    while stack and len(out) < cap:
        e = next(stack[-1], None)
        if e is None:
            stack.pop()
            prefixes.pop()
            continue
        if e.name in IGNORE_DIRS:
            continue
        rel = prefixes[-1] + e.name
        try:
            link = e.is_symlink()
            is_dir = not link and e.is_dir(follow_symlinks=False)
            size = 0 if is_dir or link else e.stat(follow_symlinks=False).st_size
        except OSError:
            continue
        out.append({"path": rel, "dir": is_dir, "size": size, **({"link": True} if link else {})})
        if is_dir:
            stack.append(iter(_sorted_entries(e.path)))
            prefixes.append(rel + "/")
    return out


def _view_text(root: Path, path: str) -> dict:
    """What the file viewers return: at most the first 300 KB, never read whole (multi-GB logs/datasets)."""
    root, rel = resolved_rel(root, path)
    try:
        data = read_in(root, rel, 300_000)
    except (UnsafePath, OSError) as exc:
        raise HTTPException(404, "Fichero no encontrado") from exc
    if b"\0" in data[:8192]:
        return {"path": path, "binary": True, "content": ""}
    return {"path": path, "binary": False, "content": data.decode("utf-8", "replace")}


# ----------------------------------------------------------------- workspace
@app.get("/api/projects/{project_id}/tree")
def workspace_tree(project_id: int, user: dict = Depends(current_user)):
    return list_tree(project_ws(get_project(user, project_id)))


@app.get("/api/projects/{project_id}/file")
def workspace_file(project_id: int, path: str, user: dict = Depends(current_user)):
    return _view_text(project_ws(get_project(user, project_id)), path)


@app.post("/api/projects/{project_id}/upload")
async def upload(project_id: int, file: UploadFile = File(...), user: dict = Depends(current_user)):
    root = project_ws(get_project(user, project_id)).resolve()
    data = await file.read(200 * 1024 * 1024 + 1)
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(413, "Máximo 200 MB")
    name = Path((file.filename or "upload").replace("\\", "/")).name or "upload"
    return await asyncio.to_thread(_store_upload, root, name, data)


def _store_upload(root: Path, name: str, data: bytes) -> dict:
    if not name.endswith(".zip"):
        try:
            write_in(root, name, data)
        except (UnsafePath, OSError) as exc:
            raise HTTPException(422, f"No se puede escribir {name}: {exc}") from exc
        return {"ok": True, "files": 1}
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise HTTPException(422, "ZIP no válido") from exc
    for info in zf.infolist():
        try:
            rel_parts(info.filename)
        except UnsafePath as exc:
            raise HTTPException(422, f"Ruta peligrosa en el ZIP: {info.filename}") from exc
    count = 0
    for info in zf.infolist():
        try:
            if info.is_dir():
                mkdirs_in(root, info.filename)
                continue
            try:
                fd = open_in(root, info.filename, WRITE_FLAGS, mkdirs=True)
            except UnsafePath:  # a link AT that name is replaced; one in its folders is refused below
                if not unlink_in(root, info.filename, only_link=True):
                    raise
                fd = open_in(root, info.filename, WRITE_FLAGS | os.O_EXCL, mkdirs=True)
            with zf.open(info) as src, os.fdopen(fd, "wb") as dst:  # links inside the ZIP become plain files
                shutil.copyfileobj(src, dst)
        except (UnsafePath, OSError) as exc:
            raise HTTPException(422, f"Ruta peligrosa en el ZIP: {info.filename}") from exc
        count += 1
    return {"ok": True, "files": count}


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
    b = db.one("SELECT * FROM bots WHERE id=? AND user_id=? AND project_id=?", (t.get("bot_id"), user["id"], t["project_id"])) if t.get("bot_id") else None
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
    busy = halt_runs("thread_id=?", (thread_id,), grace=THREAD_DELETE_GRACE)
    # a bot IS its conversation: deleting the conversation deletes the bot (triggers, memory, webhook secret)
    for b in db.all("SELECT id FROM bots WHERE id=? AND user_id=?", (t.get("bot_id") or -1, user["id"])):
        vault.delete(user["id"], f"bot:{b['id']}:webhook")
        db.execute("UPDATE runs SET bot_id=NULL WHERE bot_id=?", (b["id"],))
        db.execute("DELETE FROM bots WHERE id=?", (b["id"],))
    for rid in busy:  # still in a worker (or waiting for one): detach it so its id stays taken until its job ends
        db.execute("UPDATE runs SET thread_id=NULL WHERE id=? AND status IN ('queued', 'running')", (rid,))
    db.execute("DELETE FROM threads WHERE id=?", (thread_id,))
    _forget_telegram_targets(user["id"], thread_ids=[thread_id])
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
    return list_tree(root) if root.is_dir() else []


INLINE_IMAGES_BUDGET = 32 * 1024 * 1024  # data: URIs one page may get in all (a page can repeat one image endlessly)


def _inline_local_images(body: bytes, page: Path, root: Path) -> bytes:
    """<img src="chart.png"> next to the page → data: URI, so a sandboxed page still shows its local images.
    Each image is read once (at most 5 MB) and the page gets INLINE_IMAGES_BUDGET of them in all: a 1 KB page
    repeating `<img src="big.png">` must not make a response of gigabytes."""
    import base64 as _b64

    html = body.decode("utf-8", "replace")
    root = root.resolve()
    limit, left = 5 * 1024 * 1024, [INLINE_IMAGES_BUDGET]
    uris: dict[str, str | None] = {}  # rel -> data: URI (only ones that fit the budget are kept)

    def repl(m):
        quote, ref = m.group(2), m.group(3)
        if left[0] <= 0 or re.match(r"^(https?:|data:|blob:|#|/)", ref):
            return m.group(0)
        try:
            target = (page.parent / ref.split("?")[0]).resolve()
            rel = str(target.relative_to(root))  # never outside the conversation's workspace
        except (ValueError, OSError):
            return m.group(0)
        mime = image_mime(target.name)
        if not mime:
            return m.group(0)
        if rel not in uris:
            try:
                data = read_in(root, rel, limit + 1)
            except (UnsafePath, OSError):
                data = b""
            uri = f"data:{mime};base64,{_b64.b64encode(data).decode()}" if data and len(data) <= limit else None
            uris[rel] = uri if uri and len(uri) <= left[0] else None
        uri = uris[rel]
        if uri is None or len(uri) > left[0]:
            return m.group(0)
        left[0] -= len(uri)
        return f"{m.group(1)}{quote}{uri}{quote}"

    return re.sub(r"""(<img\b[^>]*?\bsrc=)(["'])([^"']+)\2""", repl, html, flags=re.I).encode()


def stream_fd(fd: int, media_type: str, filename: str | None = None) -> StreamingResponse:
    """Send a file already opened safely (open_in): no second lookup of its path that a link swapped in
    meanwhile could redirect."""
    from urllib.parse import quote

    fh = os.fdopen(fd, "rb")  # closed by the generator, or when collected if the client never reads
    size = os.fstat(fd).st_size

    def chunks():
        with fh:
            left = size
            while left > 0 and (block := fh.read(min(256 * 1024, left))):
                left -= len(block)
                yield block

    headers = {"Content-Length": str(size), "X-Content-Type-Options": "nosniff"}
    if filename:
        q = quote(filename)
        headers["Content-Disposition"] = f'attachment; filename="{filename}"' if q == filename else f"attachment; filename*=utf-8''{q}"
    return StreamingResponse(chunks(), media_type=media_type, headers=headers)


def _diff_side(root: Path, rel: str) -> tuple[bool, str | None]:
    """(exists, text) of root/rel for a diff. text None = binary, too big, or behind a link (never read)."""
    from ..agent.workspace import MAX_DIFF_FILE

    try:
        with os.fdopen(open_in(root, rel, READ_FLAGS), "rb") as fh:
            data = fh.read(MAX_DIFF_FILE + 1)
    except FileNotFoundError:
        return False, ""
    except (UnsafePath, OSError):
        return True, None
    if len(data) > MAX_DIFF_FILE or b"\0" in data[:8192]:
        return True, None
    return True, data.decode("utf-8", "replace")


def cage_changes(base: dict, cage: Path, project: Path, max_chars: int = 80_000) -> dict:
    """What a conversation changed since its starting manifest, diffed against the project. Same output as
    workspace.changes_since, but the project side is read without following links: a linked folder in the
    project (`d -> another user's project`) must not print someone else's .env as the 'old' side."""
    import difflib

    now = manifest(cage)
    added = sorted(p for p in now if p not in base)
    modified = sorted(p for p in now if p in base and now[p] != base[p])
    removed = sorted(p for p in base if p not in now)
    chunks: list[str] = []
    for rel in added + modified + removed:
        (had, ta), (has, tb) = _diff_side(project, rel), _diff_side(cage, rel)
        if ta is None or tb is None:
            chunks.append(f"Binary or large file changed: {rel}\n")
            continue
        chunks.extend(difflib.unified_diff(ta.splitlines(keepends=True), tb.splitlines(keepends=True),
                                           fromfile=f"a/{rel}" if had else "/dev/null", tofile=f"b/{rel}" if has else "/dev/null"))
    text = "".join(c if c.endswith("\n") else c + "\n" for c in chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (diff truncated)\n"
    return {"added": added, "modified": modified, "removed": removed, "diff": text}


def prepare_targets(root: Path, writes: list[str], removes: list[str]) -> None:
    """Before copying changes onto the project (apply_cage / apply_changes write with plain open()): a path
    whose folders go through a link — or meet a file, FIFO… where a folder should be — refuses the whole
    apply; a link AT a path about to be written is removed first, so the copy replaces the link instead of
    writing wherever it points. Call it holding the project lock (shared-workspace runs hold it too)."""
    import stat as _stat

    blocked, links, writing = [], [], set(writes)
    for rel in sorted(writing | set(removes)):
        try:
            parts = rel_parts(rel)
        except UnsafePath:
            blocked.append(rel)
            continue
        cur, bad = root, False
        for name in parts[:-1]:
            cur = cur / name
            try:
                if not _stat.S_ISDIR(os.lstat(cur).st_mode):
                    bad = True
            except FileNotFoundError:
                break  # created as a real folder by the copy
            except OSError:
                bad = True
            if bad:
                break
        if not bad:
            try:
                mode = os.lstat(root / rel).st_mode
                if _stat.S_ISLNK(mode):
                    if rel in writing:  # (a link that is removed goes itself: unlink never follows it)
                        links.append(rel)
                elif not _stat.S_ISREG(mode):
                    bad = True
            except FileNotFoundError:
                pass
            except OSError:
                bad = True
        if bad:
            blocked.append(rel)
    if blocked:
        raise HTTPException(422, "No se ha aplicado nada: en el proyecto estas rutas pasan por un enlace simbólico o no "
                                 "son ficheros normales: " + ", ".join(blocked[:10]) + ("…" if len(blocked) > 10 else ""))
    for rel in links:
        unlink_in(root, rel, only_link=True)


HTML_SANDBOX_CSP = (
    # generated pages run in an opaque origin: no cookies, no access to the app or its API, no network calls
    "sandbox allow-scripts allow-popups allow-modals allow-downloads; default-src 'none'; "
    "script-src 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://unpkg.com; "
    "style-src 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; font-src data: https://fonts.gstatic.com; "
    "img-src data: blob: https:; media-src data: blob:; connect-src 'none'; frame-ancestors 'self'"
)


@app.get("/api/threads/{thread_id}/file")
def thread_file(thread_id: int, path: str, download: int = 0, view: int = 0, user: dict = Depends(current_user)):
    import mimetypes

    t, p = get_thread(user, thread_id)
    root, rel = resolved_rel(thread_ws(p, t), path)
    name, suffix = Path(rel).name, Path(rel).suffix.lower()
    try:
        fd = open_in(root, rel, READ_FLAGS)  # the very file checked above, even if a link is swapped in meanwhile
    except (UnsafePath, OSError) as exc:
        raise HTTPException(404, "Fichero no encontrado") from exc
    if download:
        return stream_fd(fd, mimetypes.guess_type(name)[0] or "application/octet-stream", filename=name)
    if view and suffix in (".html", ".htm", ".svg"):
        with os.fdopen(fd, "rb") as fh:
            body = fh.read(15 * 1024 * 1024 + 1)
        if len(body) > 15 * 1024 * 1024:
            raise HTTPException(413, "Demasiado grande para verlo aquí: descárgalo")
        if suffix in (".html", ".htm"):
            body = _inline_local_images(body, root / rel, root)
        return Response(body, media_type="image/svg+xml" if suffix == ".svg" else "text/html; charset=utf-8",
                        headers={"Content-Security-Policy": HTML_SANDBOX_CSP, "X-Content-Type-Options": "nosniff",
                                 "Cache-Control": "no-store"})
    if image_mime(name):
        return stream_fd(fd, image_mime(name))
    with os.fdopen(fd, "rb") as fh:
        data = fh.read(300_000)  # a multi-GB log must not be loaded to show its first 300 KB
    if b"\0" in data[:8192]:
        return {"path": path, "binary": True, "content": ""}
    return {"path": path, "binary": False, "content": data.decode("utf-8", "replace")}


@app.get("/api/threads/{thread_id}/changes")
def thread_changes(thread_id: int, user: dict = Depends(current_user)):
    t, p = get_thread(user, thread_id)
    if not t["isolated"]:
        return {"isolated": False, "added": [], "modified": [], "removed": [], "diff": ""}
    return {"isolated": True, **cage_changes(loads(t["manifest"], {}), thread_ws(p, t), project_ws(p))}


@app.post("/api/threads/{thread_id}/apply")
def thread_apply(thread_id: int, user: dict = Depends(current_user)):
    t, p = get_thread(user, thread_id)
    if not t["isolated"]:
        raise HTTPException(422, "Esta conversación ya trabaja directamente sobre el proyecto")
    if db.one("SELECT id FROM runs WHERE thread_id=? AND status IN ('queued', 'running')", (thread_id,)):
        raise HTTPException(409, "Espera a que termine la respuesta en curso")
    cage, target, base = thread_ws(p, t), project_ws(p), loads(t["manifest"], {})
    with jobs._project_lock(p["id"]):
        now = manifest(cage)
        prepare_targets(target, [r for r in now if base.get(r) != now[r]], [r for r in base if r not in now])
        applied = apply_cage(base, cage, target)
    # Re-baseline: later applies only carry changes made after this one.
    db.execute("UPDATE threads SET manifest=? WHERE id=?", (json.dumps(manifest(cage)), thread_id))
    return applied


@app.post("/api/threads/{thread_id}/attachments")
async def upload_attachments(thread_id: int, files: list[UploadFile] = File(...), user: dict = Depends(current_user)):
    t, p = get_thread(user, thread_id)
    root = thread_ws(p, t)
    blobs = []
    for up in files[:20]:
        data = await up.read(MAX_ATTACHMENT + 1)
        if len(data) > MAX_ATTACHMENT:
            raise HTTPException(413, f"{up.filename}: máximo 25 MB por archivo")
        blobs.append((up.filename, data))
    return await asyncio.to_thread(_store_attachments, root, blobs)


def _store_attachments(root: Path, blobs: list[tuple[str | None, bytes]]) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    try:  # .attachments may be a link copied from the project: replace it, never write through it
        unlink_in(root, ATTACHMENTS_DIR, only_link=True)
        mkdirs_in(root, ATTACHMENTS_DIR)
    except (UnsafePath, OSError) as exc:
        raise HTTPException(422, f"No se pueden guardar adjuntos aquí: {exc}") from exc
    out = []
    for filename, data in blobs:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path((filename or "archivo").replace("\\", "/")).name).strip("._") or "archivo"
        stem, dot, ext = name.rpartition(".")
        if not dot:
            stem, ext = name, ""
        candidate, n = name, 1
        while True:  # O_EXCL: the first free name, atomically
            try:
                fd = open_in(root, f"{ATTACHMENTS_DIR}/{candidate}", WRITE_FLAGS | os.O_EXCL)
                break
            except FileExistsError:
                n += 1
                candidate = f"{stem}-{n}.{ext}" if ext else f"{stem}-{n}"
            except (UnsafePath, OSError) as exc:
                raise HTTPException(422, f"No se puede guardar {name}: {exc}") from exc
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        out.append(_attachment_meta(root / ATTACHMENTS_DIR / candidate, f"{ATTACHMENTS_DIR}/{candidate}", data[:8192], len(data)))
    return out


def _attachment_meta(f: Path, rel: str, head: bytes | None = None, size: int | None = None) -> dict:
    if head is None or size is None:
        with open(f, "rb") as fh:
            head, size = fh.read(8192), os.fstat(fh.fileno()).st_size
    kind = "image" if image_mime(f.name) else "pdf" if f.suffix.lower() == ".pdf" else ("binary" if b"\0" in head else "text")
    return {"name": f.name, "path": rel, "kind": kind, "size": size}


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
    bot = db.one("SELECT id FROM bots WHERE id=? AND user_id=? AND project_id=?", (t.get("bot_id"), user["id"], p["id"])) if t.get("bot_id") else None
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
    from ..agent.workspace import diff_workspaces

    selected = (loads(run["result"], {}) or {}).get("selected")
    rd = run_dir(p, run_id)
    after = rd / "rollouts" / str(selected) / "ws"
    if selected and after.is_dir():  # the same links/odd-files check as a conversation's apply
        lock = jobs._project_lock(p["id"])
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Hay una respuesta en curso trabajando en este proyecto: espera a que termine")
        try:
            d = diff_workspaces(rd / "source", after, max_chars=0)
            prepare_targets(project_ws(p), d["added"] + d["modified"], d["removed"])
        finally:
            lock.release()
    try:
        return {"changes": jobs.apply_later(run_id, p)}
    except FileNotFoundError as exc:
        raise HTTPException(410, str(exc)) from exc


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: int, request: Request, after: int = 0, user: dict = Depends(current_user)):
    run, _ = await asyncio.to_thread(get_run, user, run_id)
    ident = (run["project_id"], run["created_at"])  # run ids are reused once deleted: this pins THIS run

    def poll(last: int, gen_seen: int, idx: int):
        events = db.events_after(run_id, last)
        live = jobs.live.get(run_id)
        items, gen = [], gen_seen
        if live is not None:
            if live.gen != gen_seen:
                gen, idx = live.gen, 0
            items = list(live.items[idx:])
        # Checked AFTER reading: if the row is still the same run now, what we read was its own data.
        row = db.one("SELECT status, project_id, created_at FROM runs WHERE id=?", (run_id,))
        if not row or (row["project_id"], row["created_at"]) != ident:
            return None
        return events, items, gen, idx, row["status"]

    async def gen():
        last = after
        gen_seen, idx = -1, 0
        idle = 0
        while True:
            if await request.is_disconnected():
                return
            got = await asyncio.to_thread(poll, last, gen_seen, idx)
            if got is None:  # deleted (its conversation or project): nothing more will ever come from this run
                yield f"event: end\ndata: {json.dumps({'status': 'deleted'})}\n\n"
                return
            events, items, gen_seen, idx, status = got
            sent = False
            for seq, ev in events:
                last = seq
                sent = True
                yield f"id: {seq}\nevent: event\ndata: {json.dumps({'seq': seq, **ev}, ensure_ascii=False)}\n\n"
            if items:
                idx += len(items)
                sent = True
                yield f"event: live\ndata: {json.dumps(items, ensure_ascii=False)}\n\n"
            if status in FINAL and not sent:
                yield f"event: end\ndata: {json.dumps({'status': status})}\n\n"
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
