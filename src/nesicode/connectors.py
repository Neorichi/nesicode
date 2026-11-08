"""MCP connectors: a one-click catalog plus a manager that keeps each MCP server session alive.

Local (stdio) servers run OUTSIDE the agent cage (they need the real Chrome, apps, accounts), so the
server only allows them in local mode; remote (HTTP) servers are allowed everywhere. Every MCP call
goes through the same approval modes as bash.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import collections
import concurrent.futures
import functools
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
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
        "id": "virustotal", "label": "VirusTotal", "icon": "virustotal",
        "description": "Consulta en VirusTotal hashes de archivos, URLs, dominios e IPs (veredicto de más de 70 antivirus). Integrado: solo necesitas tu API key gratuita de virustotal.com. Subir un archivo o una URL siempre te pide permiso, porque se comparte con VirusTotal.",
        "kind": "builtin", "requires": [],
        "fields": [{"key": "api_key", "label": "API key de VirusTotal", "secret": True, "env": "VT_API_KEY",
                    "help": "virustotal.com → tu perfil → API key (la gratuita permite 4 consultas por minuto)."}],
        "hint": "Lookups are cheap and private; vt_scan_url / vt_scan_file share data with VirusTotal (the user approves).",
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


# Words that count as "the user asked for this connector" (besides its name), for catalog connectors. Only
# specific words: generic ones ('issue', 'tab', 'word', 'roadmap') switched connectors on in ordinary coding
# talk ("convert tabs to spaces", "there's an issue with the login", "count the words").
ALIASES = {
    "chrome": ["chrome devtools", "devtools", "chrome aparte"],
    "playwright": ["playwright"],
    "github": ["github", "pull request"],
    "office": ["office", "excel", "powerpoint", "docx", "xlsx", "xlsm", "pptx", "hoja de calculo", "presentacion"],
    "burp": ["burp"], "burp-rest": ["burp"], "burp-dast": ["burp", "dast"],
    "ms-roadmap": ["microsoft 365", "m365", "azure update"],
    "virustotal": ["virustotal", "virus total"],
    "browser": ["chrome", "mi navegador", "my browser", "navegador del usuario"],
}
# Generic words that count only in a phrase that clearly names the product ('hazme un Word', 'in Word',
# 'Word document', 'roadmap de Microsoft'). Regexes over the normalised text (see _norm).
PHRASES = {
    # singular only: the plural after these words is the English noun ('in words', 'mark my words', 'in your words')
    "office": [r"(?:un|una|el|al|del|de|en|con|usa|usar|use|using|in|into|with|mi|my|tu|your|este|ese) word",
               r"words? (?:docs?|documents?|files?|templates?|documentos?|archivos?|ficheros?|plantillas?)"],
    "ms-roadmap": [r"(?:microsoft|m365|365|azure) roadmaps?",
                   r"roadmaps? (?:de |del |of |for )?(?:microsoft|m365|office 365|azure)"],
}
# Words that don't identify a custom connector on their own ('Linear de mi equipo', 'Servidor Jira', 'MCP local').
# The second group: the words of Nesicode's own tool names, which the mention check sees in almost every conversation
# (connector_context lists the tools used: 'search', 'read_file', 'view_image'…), and everyday words of coding and
# office talk. 'Brave Search' must not switch on whenever the agent searched the code, nor 'Jira Cloud' at "the cloud".
_GENERIC = frozenset("""
    a al an and at con de del e el en for in la las lo los mi mis my o of on or our para por su sus the their to tu tus
    u un una unas unos with y your api base cliente client company conector connector cuenta account data datos dev
    empresa equipo home interno internal local mcp new nueva nuevo oficina personal prod produccion production prueba
    pruebas remote remoto server servicio service servidor team test trabajo work casa
    back bash bot chart click close create delegate delete dir edit fetch file files get history http image images list
    load memory notify once open read replace report scan schedule screenshot scroll search set share skill tab tabs
    type url view write
    archivo archivos buscador busqueda cloud code codigo doc docs document documento documentos documents error errores
    errors fichero ficheros google herramienta herramientas microsoft tool tools web
    as be by do if is it me no se si so te up us vs we ya yo
""".split())


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def _norm(text: str) -> str:
    """Lower case, no accents, any run of non-letters → one space; letters of every script are kept (数据库)."""
    return " " + " ".join(re.findall(r"[^\W_]+", _fold(text or ""))) + " "


def _wide(word: str) -> bool:
    """CJK/kana/hangul: written without spaces, so it is matched inside longer runs of text."""
    return any(unicodedata.east_asian_width(c) in "WF" for c in word)


def _phrase_re(words: list[str]) -> re.Pattern:
    if any(_wide(w) for w in words):
        return re.compile(re.escape(" ".join(words)))
    # whole words, each with an optional plural ('hojas de calculo', 'presentaciones', 'Excels')
    return re.compile(" " + " ".join(re.escape(w) + ("(?:e?s)?" if len(w) >= 3 and w[-1].isalpha() else "")
                                     for w in words) + " ")


def trigger_words(name: str = "", label: str = "", template: str = "") -> list[str]:
    """The words that switch an 'only when I ask' connector on (for the UI / the agent's suggestion)."""
    custom = template not in ALIASES
    whole = label.split("(")[0]
    phrases = [whole, name] if custom else list(ALIASES[template]) + [name]
    if custom:  # plus each distinctive word of its label and name ('Confluence empresa' → 'confluence')
        for src in (label, name):
            words = [w for w in _norm(src).split() if w not in _GENERIC]
            short_only = not any(len(w) >= 3 or _wide(w) for w in words)
            # 2-letter words count when they are all there is ('DB', 'Mi DB') or carry a digit ('AWS S3')
            phrases += [w for w in words if len(w) >= 3 or _wide(w) or (len(w) == 2 and (short_only or any(c.isdigit() for c in w)))]
    out: list[str] = []
    for i, ph in enumerate(phrases):
        words = _norm(ph).split()
        # a lone generic word only counts when it is the whole label the user gave it ('Datos', 'API'), never the
        # fallback name 'mcp', which would match every MCP tool name in the context
        if not words or (len(words) == 1 and words[0] in _GENERIC and not (custom and i == 0 and words[0] != "mcp")):
            continue
        size = len("".join(words))
        if size < 2 or (size == 2 and (len(words) > 1 or not (custom or _wide(words[0])))):
            continue
        if (p := " ".join(words)) not in out:
            out.append(p)
    return out


@functools.lru_cache(maxsize=512)
def _patterns(name: str, label: str, template: str) -> tuple[re.Pattern, ...]:
    pats = [_phrase_re(p.split()) for p in trigger_words(name, label, template)]
    return tuple(pats + [re.compile(" " + r + " ") for r in PHRASES.get(template, [])])


def mentioned(text: str, *, name: str = "", label: str = "", template: str = "") -> bool:
    """Did the user name this connector? ('usa burp', 'en un Excel', 'con VirusTotal', 'busca en Confluence'…)"""
    t = _norm(text)
    return any(p.search(t) for p in _patterns(name or "", label or "", template or ""))


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


# What a local MCP server inherits from Nesicode's own environment. Its launch environment is readable by any process
# of the same user — the agent's commands in their cage included (macOS: sysctl KERN_PROCARGS2, which the sandbox cannot
# block) — so it only gets what toolchains (npx, uvx, node, python…) need to start, never Nesicode's settings
# (NESICODE_*, the desktop token) nor anything named like a secret (the provider keys loaded from .env…). What the
# connector itself configures (cfg.env: its own token…) goes on top.
_ENV_KEEP = frozenset("""
    PATH HOME USER LOGNAME SHELL LANG LANGUAGE TZ TERM TMPDIR TMP TEMP __CF_USER_TEXT_ENCODING
    DISPLAY WAYLAND_DISPLAY
    HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY http_proxy https_proxy no_proxy all_proxy
    SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE NODE_EXTRA_CA_CERTS
    NVM_DIR NVM_BIN NVM_INC VOLTA_HOME PYENV_ROOT PYENV_VERSION JAVA_HOME GOPATH GOROOT CARGO_HOME RUSTUP_HOME
    BUN_INSTALL PNPM_HOME DENO_DIR DENO_INSTALL ASDF_DIR ASDF_DATA_DIR MISE_DATA_DIR SDKMAN_DIR
    HOMEBREW_PREFIX HOMEBREW_CELLAR HOMEBREW_REPOSITORY
    UV_CACHE_DIR UV_TOOL_DIR UV_TOOL_BIN_DIR UV_PYTHON_INSTALL_DIR NPM_CONFIG_PREFIX NPM_CONFIG_CACHE
    SYSTEMROOT SYSTEMDRIVE WINDIR COMSPEC PATHEXT APPDATA LOCALAPPDATA PROGRAMDATA PROGRAMFILES PROGRAMFILES(X86)
    PROGRAMW6432 COMMONPROGRAMFILES COMMONPROGRAMFILES(X86) COMMONPROGRAMW6432 USERPROFILE HOMEDRIVE HOMEPATH
    USERNAME PUBLIC ALLUSERSPROFILE OS NUMBER_OF_PROCESSORS PROCESSOR_ARCHITECTURE PROCESSOR_IDENTIFIER
""".split())
_ENV_KEEP_PREFIXES = ("LC_", "XDG_")
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION", re.I)


def server_env(own: dict[str, str] | None = None, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a local (stdio) MCP server starts with: the toolchain part of `base` (os.environ) plus `own`
    (the connector's cfg.env), which may set anything."""
    base = os.environ if base is None else base
    nt = os.name == "nt"  # Windows names are case-insensitive (SystemRoot, ProgramFiles(x86)…)
    keep = {}
    for k, v in base.items():
        n = k.upper() if nt else k
        if k.upper().startswith("NESICODE_") or _SECRET_NAME.search(k) or (v or "").startswith("()"):  # (bash functions)
            continue
        if n in _ENV_KEEP or n.startswith(_ENV_KEEP_PREFIXES):
            keep[k] = v
    return {**keep, **(own or {})}


def _describe(exc: BaseException) -> str:
    """The innermost real error (anyio wraps failures in ExceptionGroups)."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


# The server no longer knows our session (it restarted): the request was rejected without running. The whole
# message must be one of the transports' own answers ('Session not found', 'Bad Request: No valid session ID
# provided', 'Not Found: Invalid or expired session ID', 'Bad Request: Server not initialized'…): a tool's error
# that mentions a session ('invalid session token for Jira') ran, and must not be repeated.
_SESSION_GONE = re.compile(r"(?:(?:bad request|not found|error)\s*:\s*)?(?:session not found|no valid session(?: id)?"
                           r"(?: provided)?|(?:unknown|invalid|invalid or expired|expired) session(?: id)?|session (?:id )?"
                           r"(?:has )?expired|session terminated|server not initialized)(?:[\s:=(]+[\w.-]{6,}\)?)?\W*", re.I)


def _lost(exc: BaseException) -> str:
    """'gone' when the server dropped our session, 'closed' when the connection died, 'suspect' when an HTTP server
    answered an error status without a JSON-RPC error (python-sdk 1.x servers answer an unknown session with a plain
    '400 Bad Request', but a crash after running the tool looks the same: never repeat it, reconnect next time), ''
    for any other error."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    msg = str(getattr(exc, "message", "") or exc).strip()
    if _SESSION_GONE.fullmatch(msg):
        return "gone"
    if msg.lower() == "connection closed" or type(exc).__name__ in ("ClosedResourceError", "BrokenResourceError"):
        return "closed"
    if msg.lower() == "server returned an error response":
        return "suspect"
    return ""


class _StderrTail:
    """A stdio server's stderr: still copied to ours (terminal / nesicode.log), and its last lines kept so a
    failed start says why ('Error: falta la variable NOTION_TOKEN') instead of 'Connection closed'."""

    # stack frames and code lines (indented) and the traceback header: noise next to the error line itself
    _NOISE = re.compile(r"^\s|^Traceback \(most recent call last\)")

    def __init__(self, name: str):
        rfd, wfd = os.pipe()
        self.file = os.fdopen(wfd, "wb", buffering=0)  # handed to the server process as its stderr
        self.lines: collections.deque[str] = collections.deque(maxlen=40)
        self.thread = threading.Thread(target=self._pump, args=(rfd,), daemon=True, name=f"mcp-stderr-{name}")
        self.thread.start()

    def _pump(self, rfd: int) -> None:
        part = b""
        with os.fdopen(rfd, "rb", buffering=0) as f:
            while chunk := f.read(65536):
                try:
                    if sys.stderr is not None:
                        sys.stderr.write(chunk.decode("utf-8", "replace"))
                except Exception:  # noqa: BLE001 - logging must never block the server's pipe
                    pass
                lines = (part + chunk).split(b"\n")
                part = lines.pop()[:4096]  # bounded, even for a line that never ends
                self.lines.extend(ln.decode("utf-8", "replace").rstrip()[:400] for ln in lines if ln.strip())
        if part.strip():
            self.lines.append(part.decode("utf-8", "replace").rstrip()[:400])

    def close_writer(self) -> None:
        try:
            self.file.close()  # the server has its own copy; ours must close so the reader sees the end
        except OSError:
            pass

    async def tail(self) -> str:
        for _ in range(20):  # the server is gone: give the reader a moment to drain the pipe
            if not self.thread.is_alive():
                break
            await asyncio.sleep(0.025)
        raw = list(self.lines)
        lines = [ln for ln in raw if not self._NOISE.search(ln)] or raw
        return "\n".join(lines[-10:])[-1200:]


class _Connection:
    def __init__(self, cfg: ConnectorConfig, user_id: int = 0):
        self.cfg = cfg
        self.user_id = user_id
        self.born = time.monotonic()  # its start began: everyone waiting for it shares one startup deadline
        self.session = None
        self.ready = asyncio.Event()
        self.stop = asyncio.Event()  # set by stop()/_drop(): shuts the server down, even while it is starting
        self.error: str = ""
        self.tools: list[McpTool] = []
        self.task: asyncio.Task | None = None
        self.stderr: _StderrTail | None = None

    async def run(self) -> None:
        import anyio
        from mcp.client.stdio import StdioServerParameters, stdio_client

        try:
            async with anyio.create_task_group() as tg:
                # stop() cancels this scope: an anyio cancel, so stdio_client's shielded shutdown still kills the
                # process tree, also while the server is starting and has not answered initialize
                tg.start_soon(self._stop_when_asked, tg.cancel_scope)
                if self.cfg.kind == "stdio":
                    self.stderr = _StderrTail(self.cfg.name)
                    params = StdioServerParameters(command=self.cfg.command, args=self.cfg.args, env=server_env(self.cfg.env))
                    async with stdio_client(params, errlog=self.stderr.file) as (read, write):
                        self.stderr.close_writer()
                        await self._serve(read, write, watch=True)
                else:
                    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

                    async with create_mcp_http_client(headers=self.cfg.headers) as http:
                        async with streamable_http_client(self.cfg.url, http_client=http) as streams:
                            await self._serve(streams[0], streams[1])
                tg.cancel_scope.cancel()  # the server went away by itself
        except BaseException as exc:  # noqa: BLE001 - surface any startup failure to the user
            if not self.error and not self.stop.is_set():
                self.error = _describe(exc)[:800]
        finally:
            self.session = None
            if self.stderr is not None:
                self.stderr.close_writer()
                if self.error and not self.stop.is_set() and (tail := await self.stderr.tail()):
                    self.error = f"{self.error}\n{tail}"[:2000]
            self.ready.set()

    async def _stop_when_asked(self, scope) -> None:
        await self.stop.wait()
        scope.cancel()

    async def _serve(self, read, write, watch: bool = False) -> None:
        import anyio
        from mcp import ClientSession

        closed = anyio.Event()
        async with anyio.create_task_group() as tg:
            if watch:  # stdio_client doesn't end when the server process dies: notice it ourselves
                read = _relay_until_eof(tg, read, closed)
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                self.tools = [
                    McpTool(t.name, (t.description or "")[:1500],
                            getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {"type": "object", "properties": {}})
                    for t in listed.tools
                ]
                self.session = session
                self.ready.set()
                await (closed.wait() if watch else anyio.sleep_forever())
                self.session = None
                self.error = "el servidor MCP se cerró (terminó o falló)"
            tg.cancel_scope.cancel()


def _relay_until_eof(tg, src, closed):
    """Pass the server's messages on to the session and set `closed` when its stream ends (the process exited)."""
    import anyio

    send, recv = anyio.create_memory_object_stream(0)

    async def relay():
        try:
            async with send:
                async for item in src:
                    await send.send(item)
        except (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream):
            pass
        finally:
            closed.set()

    tg.start_soon(relay)
    return recv


BACKOFF = 300.0  # s: a server that did not answer at startup is not waited for again by messages for this long


class McpManager:
    """Owns an asyncio loop in a background thread; sync callers (agent threads) use it safely.

    A connection is keyed by (user, full config). tools() starts it and records it as started for that user's
    connector; stop() (Probar, Desactivar, Eliminar) shuts down only that user's connections and revokes that, so
    a conversation already running can't bring a disabled connector back. A crashed server, or one that restarted
    and forgot our session, is reconnected on its next use. A server that never answered at startup is stopped and
    remembered for BACKOFF s (a new config, or Probar / Desactivar → stop(), forgets it): messages skip it meanwhile."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True, name="nesi-mcp")
        self.thread.start()
        self.conns: dict[str, _Connection] = {}
        self.started: dict[tuple[int, str], str] = {}  # (user, connector name) -> key of the config tools() started
        self.dying: set[_Connection] = set()  # dropped, their server still shutting down (stop() waits for them too)
        self.failed: dict[str, tuple[float, str, int, str]] = {}  # key -> (retry after, why, user, name): startups that hung
        self.closed = False  # shutdown(): the app is closing, nothing starts any more
        self.lock = threading.Lock()

    def _run(self, coro, timeout: float, cancel: threading.Event | None = None):
        cf = asyncio.run_coroutine_threadsafe(coro, self.loop)
        deadline = time.monotonic() + timeout
        try:
            while not concurrent.futures.wait([cf], timeout=min(0.25 if cancel is not None else timeout,
                                                                max(0.0, deadline - time.monotonic()))).done:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("cancelado: el usuario pulsó Detener")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"sin respuesta en {timeout:.0f} s")
            return cf.result()
        except BaseException:
            cf.cancel()  # a no-op if it finished; else the coroutine stops too (and its MCP request is cancelled)
            raise

    def _drop(self, key: str, conn: _Connection) -> None:
        """Forget a connection and shut its server down (in the background)."""
        if self.conns.get(key) is conn:
            self.conns.pop(key, None)
        conn.stop.set()
        if conn.task is not None and not conn.task.done() and conn not in self.dying:
            self.dying.add(conn)
            conn.task.add_done_callback(lambda _task: self.dying.discard(conn))

    def _hung_lately(self, key: str) -> str:
        """Why this config is not waited for now ('' = go ahead): it did not answer at startup a moment ago."""
        until, why, *_ = self.failed.get(key) or (0.0, "")
        left = until - time.monotonic()
        if left <= 0:
            self.failed.pop(key, None)
            return ""
        return (f"{why} hace un momento; no se vuelve a esperar hasta dentro de {max(1, round(left / 60))} min "
                "(«Probar» en Integraciones lo reintenta ya)")

    async def _ensure(self, key: str, cfg: ConnectorConfig, user_id: int, timeout: float,
                      backoff: bool = False) -> _Connection:
        if self.closed:
            raise RuntimeError("Nesicode se está cerrando: no se arrancan conectores")
        if backoff and (why := self._hung_lately(key)):
            raise RuntimeError(why)
        conn = self.conns.get(key)
        if conn is not None and conn.ready.is_set() and conn.session is None:
            self._drop(key, conn)  # it died (crashed, closed, failed to start): start a new one
            conn = None
        if conn is None:
            conn = _Connection(cfg, user_id)
            self.conns[key] = conn
            conn.task = asyncio.ensure_future(conn.run())
        try:  # the deadline counts from its start: a message that gave up waiting (Stop) does not reset it
            if not conn.ready.is_set():  # (wait_for with no time left fails even when the event is already set)
                await asyncio.wait_for(conn.ready.wait(), max(0.0, conn.born + timeout - time.monotonic()))
        except TimeoutError:
            self._drop(key, conn)  # it never answered: don't leave its process running
            why = f"el servidor MCP no respondió en {timeout:.0f} s al arrancar"
            self.failed[key] = (time.monotonic() + BACKOFF, why, user_id, cfg.name)
            tail = await conn.stderr.tail() if conn.stderr is not None else ""
            raise RuntimeError(why + (f"\n{tail}" if tail else "")) from None
        if conn.session is None or conn.stop.is_set():  # also a stop() that came in while it was starting
            raise RuntimeError(conn.error or "el servidor MCP se detuvo")
        self.failed.pop(key, None)
        return conn

    def tools(self, user_id: int, cfg: ConnectorConfig, timeout: float = 120,
              cancel: threading.Event | None = None) -> list[McpTool]:
        """Start a connector (or reuse it) and list its tools — also one that hung lately (Probar, adding it)."""
        if cfg.kind == "builtin":
            return builtin_tools(cfg.command, cfg)
        key = cfg.key(user_id)

        async def go():
            conn = await self._ensure(key, cfg, user_id, timeout)
            self.started[(user_id, cfg.name)] = key
            return conn.tools

        return self._run(go(), timeout + 5, cancel)

    def tools_many(self, user_id: int, cfgs: list[ConnectorConfig], timeout: float = 90,
                   cancel: threading.Event | None = None) -> list:
        """Start the MCP servers (stdio / http) a message is offered, all at once; returns, in order, each one's tools
        or the exception that keeps it out. One that did not answer at startup a moment ago is not waited for again
        (BACKOFF): a hung server costs one timeout per few minutes, not one per message. `cancel` (Stop) ends the
        wait within a quarter of a second (the servers go on starting, for the next message)."""

        async def one(cfg):
            key = cfg.key(user_id)
            try:
                conn = await self._ensure(key, cfg, user_id, timeout, backoff=True)
            except Exception as exc:  # noqa: BLE001 - reported per connector
                return exc
            self.started[(user_id, cfg.name)] = key
            return conn.tools

        async def go():
            return list(await asyncio.gather(*(one(c) for c in cfgs)))

        return self._run(go(), timeout + 5, cancel) if cfgs else []

    def call(self, user_id: int, cfg: ConnectorConfig, tool: str, args: dict, timeout: float = 180,
             cancel: threading.Event | None = None) -> McpResult:
        """Call a tool. `cancel` (e.g. the run's Stop) abandons the call and cancels its MCP request."""
        key = cfg.key(user_id)

        async def go():
            for attempt in (1, 2):
                if self.started.get((user_id, cfg.name)) != key:
                    raise RuntimeError("the user stopped, turned off or removed this connector in Nesicode: it is not "
                                       "started again during this message")
                conn = await self._ensure(key, cfg, user_id, 120, backoff=True)
                try:
                    res = await conn.session.call_tool(tool, args or {}, read_timeout_seconds=timeout)
                except Exception as exc:
                    if not (lost := _lost(exc)):
                        raise
                    if lost == "gone" and attempt == 1:
                        self._drop(key, conn)
                        continue  # the server restarted and rejected the call unrun: once more, on a new session
                    # dead (or doubtful): it shows as 'error' and the next call starts a fresh one (_ensure replaces it)
                    conn.session = None
                    conn.stop.set()
                    if lost == "closed":  # the call may have run: never repeat it on our own
                        tail = await conn.stderr.tail() if conn.stderr is not None else ""
                        raise RuntimeError("the connection to the MCP server closed during the call (the server exited, "
                                           "crashed or restarted); the next call reconnects" + (f"\n{tail}" if tail else "")) from exc
                    raise
                return _result(res)

        return self._run(go(), timeout + 130, cancel)

    def stop(self, user_id: int, name: str) -> None:
        """Stop this user's connector (Probar, Desactivar, Eliminar), also a server that is still starting.
        Until tools() starts it again, conversations already running can't restart it."""
        async def go():
            self.started.pop((user_id, name), None)
            for key, f in list(self.failed.items()):  # Probar / Desactivar: the next start waits for it again
                if f[2:] == (user_id, name):
                    self.failed.pop(key, None)
            for key, conn in list(self.conns.items()):
                if conn.user_id == user_id and conn.cfg.name == name:
                    self._drop(key, conn)
            # also the ones dropped before (a start that timed out, a lost session): they may still be shutting down
            tasks = [c.task for c in self.dying if c.user_id == user_id and c.cfg.name == name and not c.task.done()]
            if tasks:  # wait (bounded) for the processes to exit: a restart must not overlap the old one
                await asyncio.wait(tasks, timeout=8)

        self._run(go(), 10)

    def shutdown(self, timeout: float = 8.0) -> None:
        """Stop every server (the app is closing) all at once, waiting at most `timeout` s for them to exit; nothing
        starts on this manager afterwards. Safe to call again."""
        self.closed = True
        if self.loop.is_closed() or not self.thread.is_alive():
            return

        async def go():
            self.started.clear()
            for key, conn in list(self.conns.items()):
                self._drop(key, conn)
            tasks = [c.task for c in list(self.dying) if c.task is not None and not c.task.done()]
            if tasks:
                await asyncio.wait(tasks, timeout=timeout)

        try:
            self._run(go(), timeout + 1)
        except Exception as exc:  # noqa: BLE001 - never block the exit
            print("nesicode: could not stop every MCP server:", exc)

    def status(self, user_id: int, cfg: ConnectorConfig) -> str:
        if cfg.kind == "builtin":
            return "running"
        conn = self.conns.get(cfg.key(user_id))
        if not conn:
            return "stopped"
        if conn.session is not None:
            return "running"
        return "error" if conn.ready.is_set() else "starting"  # ready without a session: it failed or died


def _result(res) -> McpResult:
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


def builtin_module(which: str):
    if which == "office":
        from . import office

        return office
    if which == "burp-rest":
        from . import burp_rest

        return burp_rest
    if which == "virustotal":
        from . import virustotal

        return virustotal
    raise RuntimeError(f"unknown built-in connector {which!r}")


def builtin_tools(which: str, cfg: "ConnectorConfig | None" = None) -> list[McpTool]:
    """Connectors that live inside Nesicode (no external server). `which` is the connector's template."""
    mod = builtin_module(which)
    if cfg is not None and hasattr(mod, "ping"):
        mod.ping(cfg)  # e.g. Burp REST: fail now (with a clear message) if it is not reachable
    return [McpTool(name, spec.description, spec.parameters) for name, (spec, _) in mod.TOOLS.items()]


_manager: McpManager | None = None
_manager_lock = threading.Lock()


def manager() -> McpManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = McpManager()
        return _manager


def shutdown_all(timeout: float = 8.0) -> None:
    """Stop every MCP server process this app started (the desktop window closed, `serve` stopped), in parallel and
    in at most ~`timeout` s; a server that ignores its stdin closing is killed. Safe to call more than once. A later
    manager() starts afresh — and whatever it starts is stopped at exit too (atexit below)."""
    global _manager
    with _manager_lock:
        m, _manager = _manager, None
    if m is not None:
        m.shutdown(timeout)


atexit.register(shutdown_all)  # (after the jobs' own exit hook has stopped the runs that use them)


def tool_name(connector: str, tool: str) -> str:
    """Model-safe tool name (^[a-zA-Z0-9_-]{1,64}$)."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"mcp__{connector}__{tool}")
    return name[:64]
