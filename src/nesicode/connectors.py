"""MCP connectors: a one-click catalog plus a manager that keeps each MCP server session alive.

Local (stdio) servers run OUTSIDE the agent cage (they need the real Chrome, apps, accounts), so the
server only allows them in local mode; remote (HTTP) servers are allowed everywhere. Every MCP call
goes through the same approval modes as bash.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import threading
from dataclasses import dataclass, field

# ------------------------------------------------------------------ catalog
CATALOG: list[dict] = [
    {
        "id": "chrome", "label": "Chrome aparte (DevTools)", "icon": "chrome",
        "description": "Abre un Chrome nuevo y separado, sin tus sesiones, para pruebas: consola, red, rendimiento, capturas. Para usar tu Chrome de siempre, usa la extensión.",
        "kind": "stdio", "command": "npx", "args": ["-y", "chrome-devtools-mcp@latest"], "requires": ["npx"], "fields": [],
        "hint": "This is a SEPARATE automated Chrome without the user's logins, not the user's browser. "
                "Pass the pageId from new_page/list_pages to every page tool (call list_pages if ids changed). "
                "For take_screenshot omit filePath: the image is returned to you and shown to the user.",
    },
    {
        "id": "playwright", "label": "Navegador (Playwright)", "icon": "browser",
        "description": "Un navegador automático con su propio perfil: navegar, clicar, escribir, extraer datos de webs.",
        "kind": "stdio", "command": "npx", "args": ["-y", "@playwright/mcp@latest"], "requires": ["npx"], "fields": [],
        "hint": "Navigate first, then use browser_snapshot to get element refs before clicking or typing.",
    },
    {
        "id": "github", "label": "GitHub", "icon": "github",
        "description": "Issues, pull requests, repositorios y código de tu cuenta de GitHub.",
        "kind": "http", "url": "https://api.githubcopilot.com/mcp/", "requires": [],
        "fields": [{"key": "token", "label": "Token personal de GitHub", "secret": True,
                    "header": "Authorization", "prefix": "Bearer ",
                    "help": "github.com → Settings → Developer settings → Personal access tokens"}],
    },
    {
        "id": "office", "label": "Office (Word, Excel, PowerPoint)", "icon": "office",
        "description": "Crear, leer y editar documentos Word, hojas de Excel (con fórmulas y gráficos) y presentaciones PowerPoint. Integrado en Nesicode: no hay que instalar nada.",
        "kind": "builtin", "requires": [], "fields": [],
        "hint": "Built into Nesicode: paths are relative to the workspace. The file is shared with the user automatically.",
    },
    {
        "id": "burp", "label": "Burp Suite", "icon": "burp",
        "description": "Tu Burp Suite (Pro o Community): historial del proxy, Repeater, enviar peticiones, Collaborator (solo Pro), codificar/decodificar. Antes instala en Burp la extensión oficial «MCP Server» (BApp Store) y actívala en su pestaña MCP.",
        "kind": "http", "url": "http://{address}/mcp", "requires": [], "local_only": True,
        "fields": [{"key": "address", "label": "Dirección del servidor MCP de Burp", "url_param": True, "default": "127.0.0.1:9876",
                    "help": "La que muestra la pestaña MCP de Burp. Necesita Burp 2023 o posterior. Si otro programa usa el 9876, cambia el puerto en Burp y aquí."}],
        "hint": "This is the user's own Burp Suite, for authorized security testing of their own targets only. "
                "Burp may pop up its own approval before sending requests or reading history: if a call waits, tell the user to approve it in Burp.",
    },
    {
        "id": "burp-rest", "label": "Burp Suite (REST API, versiones antiguas)", "icon": "burp",
        "description": "Para Burp Suite Professional antiguo (2.x, 2020–2022) que no puede usar la extensión MCP: lanzar escaneos activos, ver su progreso y las vulnerabilidades encontradas. En Burp: User options → Misc → REST API → «Service running» y crea una API key.",
        "kind": "builtin", "requires": [], "local_only": True, "url": "http://{address}",
        "fields": [{"key": "address", "label": "Dirección de la REST API de Burp", "url_param": True, "default": "127.0.0.1:1337",
                    "help": "La «Service URL» de User options → Misc → REST API (sin la API key)."},
                   {"key": "api_key", "label": "API key de la REST API", "secret": True, "env": "BURP_API_KEY", "optional": True,
                    "help": "Créala en User options → Misc → REST API → API keys. Solo se ve al crearla."}],
        "hint": "The user's own Burp Suite Pro via its REST API: scans only (no proxy history). Use only on targets the user "
                "is authorized to test; the user approves each scan.",
    },
    {
        "id": "burp-dast", "label": "Burp Suite DAST", "icon": "burp",
        "description": "Burp Suite DAST (antes Enterprise): sitios, escaneos e incidencias con los permisos de tu usuario de API.",
        "kind": "http", "url": "https://{host}/mcp/v1", "requires": [],
        "fields": [{"key": "host", "label": "Dirección de tu Burp Suite DAST (p. ej. dast.miempresa.com)", "url_param": True},
                   {"key": "token", "label": "API key de un usuario de API", "secret": True,
                    "header": "Authorization", "prefix": "Bearer ",
                    "help": "Burp Suite DAST → Team → Add a new user → API user"}],
    },
    {
        "id": "ms-roadmap", "label": "Microsoft 365 roadmap y Azure", "icon": "microsoft",
        "description": "Pregunta por novedades de Microsoft 365 (Teams, Outlook, Excel…) y de Azure: qué sale, cuándo y qué se retira. Servidor oficial de Microsoft, público y gratuito.",
        "kind": "http", "url": "https://www.microsoft.com/releasecommunications/mcp", "requires": [], "fields": [],
    },
]


def catalog_entry(template_id: str) -> dict | None:
    return next((c for c in CATALOG if c["id"] == template_id), None)


def missing_requirements(entry: dict) -> list[str]:
    return [r for r in entry.get("requires", []) if not shutil.which(r)]


# ------------------------------------------------------------------ runtime
@dataclass
class ConnectorConfig:
    name: str
    kind: str  # stdio | http
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def key(self, user_id: int) -> str:
        blob = json.dumps([user_id, self.name, self.kind, self.command, self.args, self.env, self.url, self.headers],
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class McpTool:
    name: str
    description: str
    schema: dict


@dataclass
class McpResult:
    text: str
    images: list[tuple[bytes, str]]
    is_error: bool


def _describe(exc: BaseException) -> str:
    """The innermost real error (anyio wraps failures in ExceptionGroups)."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


class _Connection:
    def __init__(self, cfg: ConnectorConfig):
        self.cfg = cfg
        self.session = None
        self.ready = asyncio.Event()
        self.stop = asyncio.Event()
        self.error: str = ""
        self.tools: list[McpTool] = []
        self.task: asyncio.Task | None = None

    async def run(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        try:
            if self.cfg.kind == "stdio":
                env = {**os.environ, **self.cfg.env}
                params = StdioServerParameters(command=self.cfg.command, args=self.cfg.args, env=env)
                async with stdio_client(params) as (read, write):
                    await self._serve(ClientSession(read, write))
            else:
                from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

                async with create_mcp_http_client(headers=self.cfg.headers) as http:
                    async with streamable_http_client(self.cfg.url, http_client=http) as streams:
                        read, write = streams[0], streams[1]
                        await self._serve(ClientSession(read, write))
        except BaseException as exc:  # noqa: BLE001 - surface any startup failure to the user
            if not self.error:
                self.error = _describe(exc)[:800]
        finally:
            self.session = None
            self.ready.set()

    async def _serve(self, session) -> None:
        async with session:
            await session.initialize()
            listed = await session.list_tools()
            self.tools = [
                McpTool(t.name, (t.description or "")[:1500],
                        getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {"type": "object", "properties": {}})
                for t in listed.tools
            ]
            self.session = session
            self.ready.set()
            await self.stop.wait()


class McpManager:
    """Owns an asyncio loop in a background thread; sync callers (agent threads) use it safely."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True, name="nesi-mcp")
        self.thread.start()
        self.conns: dict[str, _Connection] = {}
        self.lock = threading.Lock()

    def _run(self, coro, timeout: float):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _ensure(self, key: str, cfg: ConnectorConfig, timeout: float) -> _Connection:
        conn = self.conns.get(key)
        if conn and (conn.session is None and conn.ready.is_set()):
            self.conns.pop(key, None)  # died: start again
            conn = None
        if conn is None:
            conn = _Connection(cfg)
            self.conns[key] = conn
            conn.task = asyncio.ensure_future(conn.run())
        await asyncio.wait_for(conn.ready.wait(), timeout)
        if conn.session is None:
            self.conns.pop(key, None)
            raise RuntimeError(conn.error or "the MCP server stopped")
        return conn

    def tools(self, user_id: int, cfg: ConnectorConfig, timeout: float = 120) -> list[McpTool]:
        if cfg.kind == "builtin":
            return builtin_tools(cfg.command, cfg)
        return self._run(self._ensure(cfg.key(user_id), cfg, timeout), timeout + 5).tools

    def call(self, user_id: int, cfg: ConnectorConfig, tool: str, args: dict, timeout: float = 180) -> McpResult:
        async def go():
            conn = await self._ensure(cfg.key(user_id), cfg, 120)
            res = await conn.session.call_tool(tool, args or {}, read_timeout_seconds=timeout)
            texts, images = [], []
            for c in getattr(res, "content", None) or []:
                t = getattr(c, "type", "")
                if t == "text":
                    texts.append(c.text)
                elif t == "image":
                    images.append((base64.b64decode(c.data), getattr(c, "mime_type", None) or getattr(c, "mimeType", None) or "image/png"))
                elif t == "resource":
                    r = c.resource
                    texts.append(getattr(r, "text", None) or f"[resource {getattr(r, 'uri', '')}]")
                else:
                    texts.append(str(getattr(c, "uri", "") or c))
            sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
            if sc and not texts:
                texts.append(json.dumps(sc, ensure_ascii=False)[:20000])
            return McpResult("\n".join(texts).strip(), images, bool(getattr(res, "is_error", None) or getattr(res, "isError", False)))

        return self._run(go(), timeout + 130)

    def stop(self, user_id: int, name: str) -> None:
        async def go():
            for key, conn in list(self.conns.items()):
                if conn.cfg.name == name:
                    conn.stop.set()
                    self.conns.pop(key, None)

        self._run(go(), 10)

    def status(self, user_id: int, cfg: ConnectorConfig) -> str:
        if cfg.kind == "builtin":
            return "running"
        conn = self.conns.get(cfg.key(user_id))
        if not conn:
            return "stopped"
        return "running" if conn.session is not None else ("error" if conn.error else "starting")


def builtin_module(which: str):
    if which == "office":
        from . import office

        return office
    if which == "burp-rest":
        from . import burp_rest

        return burp_rest
    raise RuntimeError(f"unknown built-in connector {which!r}")


def builtin_tools(which: str, cfg: "ConnectorConfig | None" = None) -> list[McpTool]:
    """Connectors that live inside Nesicode (no external server). `which` is the connector's template."""
    mod = builtin_module(which)
    if cfg is not None and hasattr(mod, "ping"):
        mod.ping(cfg)  # e.g. Burp REST: fail now (with a clear message) if it is not reachable
    return [McpTool(name, spec.description, spec.parameters) for name, (spec, _) in mod.TOOLS.items()]


_manager: McpManager | None = None


def manager() -> McpManager:
    global _manager
    if _manager is None:
        _manager = McpManager()
    return _manager


def tool_name(connector: str, tool: str) -> str:
    """Model-safe tool name (^[a-zA-Z0-9_-]{1,64}$)."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"mcp__{connector}__{tool}")
    return name[:64]
