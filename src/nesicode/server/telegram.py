"""Telegram bot bridge: manage your Nesicode conversations from Telegram.

The user creates a bot with @BotFather and pastes its token (stored in the vault). Nesicode
long-polls Telegram (no public URL needed), pairs ONE chat with a one-time /start code, and from
then on only that chat is obeyed. Messages go to the active conversation; answers, images, files
and approval requests (with inline buttons) come back to Telegram.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

import httpx

from ..agent.vision import image_mime
from ..agent.workspace import ATTACHMENTS_DIR
from .db import DB, loads
from .jobs import FINAL, Jobs, create_cage, thread_ws
from .vault import Vault

API = "https://api.telegram.org"
PAIR_TTL = 15 * 60
MAX_TEXT = 3900
RETRIES = 3            # a Bot API call is retried this many times on rate limits and network blips
MAX_RETRY_WAIT = 30    # …waiting at most this long each time (Telegram's retry_after, capped)
MAX_DOWNLOAD = 20 * 1024 * 1024  # the Bot API hands bots files up to 20 MB…
MAX_UPLOAD = 50 * 1024 * 1024    # …and takes up to 50 MB from them (sendPhoto: 10 MB)
MAX_PHOTO = 10 * 1024 * 1024
ALBUM_WAIT = 1.5       # photos sent together arrive as separate messages: wait this long for the rest of the album
RECONNECT_WAIT = 1     # first wait after Telegram didn't answer (then it doubles, up to 60 s)
POLL = 1.0             # how often a relayed run is checked for news
# message kinds with nothing Nesicode can use (text, files and voice notes are handled)
UNSUPPORTED = ("sticker", "location", "venue", "contact", "poll", "dice", "game", "story", "invoice", "paid_media")

HELP = (
    "Nesicode en Telegram\n\n"
    "Escribe cualquier mensaje y lo envío a la conversación anclada (o empiezo una nueva).\n\n"
    "/projects – elegir proyecto\n"
    "/chats – ver conversaciones y anclarte a una\n"
    "/attach <id> – anclarte a una conversación: lo que escribas va ahí y ves en directo lo que pasa en ella, "
    "aunque la uses desde la app\n"
    "/detach – soltar la conversación anclada\n"
    "/new [texto] – empezar una conversación nueva\n"
    "/current – dónde estoy\n"
    "/stop – detener la respuesta en curso\n"
    "/bots – tus bots (pulsa uno para lanzarlo)\n"
    "/bot <nombre> [texto] – lanzar un bot ahora, con una instrucción opcional\n"
    "/help – esta ayuda\n\n"
    "También puedes enviarme fotos y archivos: los adjunto a la conversación."
)
COMMANDS = [
    {"command": "chats", "description": "List chats and attach to one"},
    {"command": "projects", "description": "Choose project"},
    {"command": "new", "description": "Start a new chat"},
    {"command": "attach", "description": "Attach to a chat by id"},
    {"command": "detach", "description": "Detach from the current chat"},
    {"command": "current", "description": "Current project and chat"},
    {"command": "bots", "description": "Your bots: run one now"},
    {"command": "stop", "description": "Stop the running answer"},
    {"command": "help", "description": "Help"},
]


class TelegramError(RuntimeError):
    """The Bot API said ok=false: `code` is its error_code (0 = not even a JSON answer), `retry_after` its wait (429)."""

    def __init__(self, text: str, code: int = 0, retry_after: float = 0):
        super().__init__(text)
        self.code, self.retry_after = code, retry_after


def transient(exc: BaseException) -> bool:
    """Worth retrying: network trouble, a rate limit (429), Telegram's own 5xx, or something that isn't the Bot API
    answering (a captive portal or proxy page). A rejected token (401) or a bad request (400) is not."""
    if isinstance(exc, TelegramError):
        return exc.code in (0, 429) or exc.code >= 500
    return isinstance(exc, httpx.TransportError)


class TelegramClient:
    def __init__(self, token: str, api: str = API, transport: httpx.BaseTransport | None = None,
                 stop: threading.Event | None = None):
        self.base = f"{api}/bot{token}"
        self.http = httpx.Client(timeout=httpx.Timeout(70.0, connect=15.0), transport=transport)
        self.stop = stop  # set when the bridge closes: no more retries

    def call(self, method: str, **params) -> dict | list | bool:
        """One Bot API call. Rate limits (honouring retry_after), Telegram's 5xx and network blips are retried a few
        times (`_retries`): one hiccup must not lose a bot's report or an approval request."""
        files = params.pop("_files", None)
        # getMe/getUpdates answer "is Telegram there?": their callers have their own backoff (or want a quick answer)
        retries = params.pop("_retries", 0 if method in ("getMe", "getUpdates") else RETRIES)
        body = {k: v for k, v in params.items() if v is not None}
        for attempt in range(retries + 1):
            try:
                return self._call_once(method, body, files)
            except Exception as exc:
                if attempt >= retries or not transient(exc):
                    raise
                if self._pause(min(getattr(exc, "retry_after", 0) or 2 ** attempt, MAX_RETRY_WAIT)):
                    raise  # the bridge is closing: give up now
        raise AssertionError("unreachable")

    def _pause(self, seconds: float) -> bool:
        """Wait before a retry. True when the bridge closed meanwhile."""
        if self.stop is None:
            time.sleep(seconds)
            return False
        return self.stop.wait(seconds)

    def _call_once(self, method: str, body: dict, files) -> dict | list | bool:
        if files:
            r = self.http.post(f"{self.base}/{method}", data=body, files=files)
        else:
            r = self.http.post(f"{self.base}/{method}", json=body)
        try:
            data = r.json()
        except ValueError:
            raise TelegramError(f"Telegram {method}: respuesta inesperada (HTTP {r.status_code})") from None
        if not data.get("ok"):
            p = data.get("parameters") or {}
            raise TelegramError(data.get("description") or f"Telegram {method} failed",
                                int(data.get("error_code") or r.status_code or 0), float(p.get("retry_after") or 0))
        return data["result"]

    def download(self, file_id: str) -> bytes:
        info = self.call("getFile", file_id=file_id)
        url = self.base.replace("/bot", "/file/bot", 1) + "/" + info["file_path"]
        r = self.http.get(url)
        r.raise_for_status()  # never save an error page as the user's file
        return r.content


def _chunks(text: str, size: int = MAX_TEXT) -> list[str]:
    text = text.strip() or "(sin texto)"
    return [text[i : i + size] for i in range(0, len(text), size)]


class TelegramBridge:
    """One bot for one Nesicode user. Runs its own polling thread."""

    def __init__(self, db: DB, jobs: Jobs, user_id: int, token: str, api: str = API, transport=None):
        self.db, self.jobs, self.user_id = db, jobs, user_id
        self.stop = threading.Event()
        self.tg = TelegramClient(token, api, transport, stop=self.stop)
        self.thread: threading.Thread | None = None
        self.me: dict = {}
        self.error = ""  # why Telegram isn't answering right now ('' = all good)
        self.followed: set[int] = set()  # runs being relayed to Telegram right now (see _claim)
        self._followed_lock = threading.Lock()
        self._albums: dict[str, dict] = {}
        self._album_lock = threading.Lock()

    # ------------------------------------------------------------ state
    def link(self) -> dict:
        return self.db.one("SELECT * FROM telegram_links WHERE user_id=?", (self.user_id,)) or {}

    def _set(self, **fields) -> None:
        if fields:
            self.db.execute(
                f"UPDATE telegram_links SET {', '.join(f'{k}=?' for k in fields)} WHERE user_id=?",
                (*fields.values(), self.user_id),
            )

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Connect and start polling. A token Telegram rejects raises right away; no network yet (the app started at
        login before Wi-Fi, a blip) is not fatal: the polling thread keeps retrying until Telegram answers."""
        try:
            self._hello()
        except Exception as exc:
            if not transient(exc):
                raise
            self.error = f"Sin conexión con Telegram ({exc}): reintentando…"
        self.thread = threading.Thread(target=self._loop, daemon=True, name=f"nesi-telegram-{self.user_id}")
        self.thread.start()
        threading.Thread(target=self._mirror, daemon=True, name=f"nesi-telegram-mirror-{self.user_id}").start()

    def _hello(self) -> None:
        self.me = self.tg.call("getMe")
        self.error = ""
        try:
            self.tg.call("setMyCommands", commands=COMMANDS, _retries=0)
        except Exception:
            pass

    def close(self) -> None:
        self.stop.set()

    def _loop(self) -> None:
        backoff = RECONNECT_WAIT
        while not self.stop.is_set():
            try:
                if not self.me:  # Telegram wasn't reachable at start: keep knocking (with the same backoff)
                    self._hello()
                offset = self.link().get("update_offset") or 0
                updates = self.tg.call("getUpdates", offset=offset, timeout=50,
                                       allowed_updates=["message", "callback_query"])
                backoff, self.error = RECONNECT_WAIT, ""
                for u in updates:
                    self._set(update_offset=u["update_id"] + 1)
                    try:
                        self.handle(u)
                    except Exception as exc:  # one bad update must not stop the bot
                        chat = (u.get("message") or (u.get("callback_query") or {}).get("message") or {}).get("chat", {})
                        if chat.get("id") and chat["id"] == self.link().get("chat_id"):
                            self._safe_send(chat["id"], f"⚠️ Error: {exc}")
            except Exception as exc:
                self.error = (f"Sin conexión con Telegram ({exc}): reintentando…" if transient(exc)
                              else f"Telegram no acepta el bot: {exc}")
                self.stop.wait(min(backoff, 60))
                backoff *= 2

    def _claim(self, run_id: int) -> bool:
        """True when nobody relays this run yet (then it gets relayed once). The claim ends with its relay (_follow):
        SQLite reuses the ids of deleted runs, and a new run with an old id must be relayed too. Never keyed by the
        run's created_at: «send now» moves a queued message by changing it, which would relay that run twice."""
        with self._followed_lock:
            if run_id in self.followed:
                return False
            self.followed.add(run_id)
            return True

    def _mirror(self) -> None:
        """While attached to a chat, show runs that were started from the app too."""
        while not self.stop.wait(2):
            try:
                link = self.link()
                chat_id, tid = link.get("chat_id"), link.get("active_thread")
                if not chat_id or not tid:
                    continue
                for r in self.db.all("SELECT id FROM runs WHERE thread_id=? AND kind='chat' AND status IN ('queued','running')", (tid,)):
                    thread = self._thread()
                    project = self.db.one("SELECT * FROM projects WHERE id=?", (thread["project_id"],)) if thread else None
                    if not project or not self._claim(r["id"]):
                        continue
                    task = self.db.one("SELECT task FROM runs WHERE id=?", (r["id"],))["task"]
                    self._safe_send(chat_id, f"💻 Desde la app, en «{thread['title']}»:\n{task[:1500]}")
                    threading.Thread(target=self._follow, args=(chat_id, r["id"], project, thread, None), daemon=True).start()
            except Exception:
                pass

    # ------------------------------------------------------------ sending
    def send(self, chat_id: int, text: str, buttons: list[list[dict]] | None = None) -> dict:
        last = {}
        parts = _chunks(text)
        for i, part in enumerate(parts):
            markup = {"inline_keyboard": buttons} if buttons and i == len(parts) - 1 else None
            last = self.tg.call("sendMessage", chat_id=chat_id, text=part, reply_markup=markup,
                                disable_web_page_preview=True)
        return last

    def _safe_send(self, chat_id: int, text: str, buttons: list[list[dict]] | None = None) -> dict:
        """send() for relays and notices: a message that still can't go out after the retries is logged and skipped,
        so the run's next messages (its final answer, an approval request) are still relayed."""
        try:
            return self.send(chat_id, text, buttons)
        except Exception as exc:
            print(f"nesicode telegram: could not send a message: {exc}")
            return {}

    def edit(self, chat_id: int, message_id: int, text: str) -> None:
        try:
            self.tg.call("editMessageText", chat_id=chat_id, message_id=message_id, text=text[:MAX_TEXT])
        except Exception:
            pass

    # ------------------------------------------------------------ updates
    def handle(self, u: dict) -> None:
        if "callback_query" in u:
            return self._callback(u["callback_query"])
        msg = u.get("message") or {}
        chat_id = msg.get("chat", {}).get("id")
        text = (msg.get("text") or msg.get("caption") or "").strip()
        link = self.link()
        if text.startswith("/start"):
            code = text.split(maxsplit=1)[1].strip() if " " in text else ""
            return self._pair(chat_id, code, msg)
        if not link.get("chat_id") or chat_id != link["chat_id"]:
            return  # not paired with this chat: ignore everything silently
        cmd = text.split()[0].split("@")[0].lower() if text.startswith("/") else ""
        rest = text[len(text.split()[0]):].strip() if cmd else ""
        if cmd == "/help":
            return self.send(chat_id, HELP)
        if cmd == "/projects":
            return self._projects(chat_id)
        if cmd == "/chats":
            return self._threads(chat_id)
        if cmd == "/attach":
            if not rest.isdigit():
                return self._threads(chat_id)
            return self._attach(chat_id, int(rest))
        if cmd == "/detach":
            t = self._thread()
            self._set(active_thread=None)
            return self.send(chat_id, f"🔓 Has soltado «{t['title']}»." if t else "No estabas anclado a ninguna conversación.")
        if cmd == "/new":
            self._set(active_thread=None)
            if rest:
                return self._message(chat_id, rest, msg)
            return self.send(chat_id, "🆕 Vale: lo próximo que escribas empezará una conversación nueva.")
        if cmd == "/current":
            return self._current(chat_id)
        if cmd == "/stop":
            return self._stop(chat_id)
        if cmd == "/bots":
            return self._bots(chat_id)
        if cmd == "/bot":
            if not rest:
                return self._bots(chat_id)
            return self._bot_command(chat_id, rest)
        if cmd:
            return self.send(chat_id, "No conozco ese comando. /help")
        if msg.get("media_group_id"):
            return self._album(chat_id, msg)
        return self._message(chat_id, text, msg)

    def _pair(self, chat_id: int, code: str, msg: dict) -> None:
        link = self.link()
        if link.get("chat_id") == chat_id:
            return self.send(chat_id, "✅ Ya estás vinculado.\n\n" + HELP)
        ok = (link.get("pair_code") and code and secrets.compare_digest(code, link["pair_code"])
              and time.time() - (link.get("pair_created") or 0) < PAIR_TTL)
        if not ok:
            return self.send(chat_id, "Para vincular este chat, usa el enlace que te da Nesicode (Conectores (MCP) → Telegram).")
        who = msg.get("from", {})
        name = " ".join(x for x in (who.get("first_name"), who.get("last_name")) if x) or who.get("username") or str(chat_id)
        self._set(chat_id=chat_id, chat_name=name, pair_code=None)
        self.send(chat_id, f"✅ Vinculado con Nesicode, {name}.\n\n" + HELP)

    # ------------------------------------------------------------ navigation
    def _projects(self, chat_id: int) -> None:
        rows = self.db.all("SELECT id, name FROM projects WHERE user_id=? ORDER BY created_at DESC LIMIT 20", (self.user_id,))
        if not rows:
            return self.send(chat_id, "No tienes proyectos todavía: crea uno en la app.")
        active = self.link().get("active_project")
        buttons = [[{"text": ("● " if p["id"] == active else "") + p["name"], "callback_data": f"p:{p['id']}"}] for p in rows]
        self.send(chat_id, "Elige proyecto:", buttons)

    def _threads(self, chat_id: int) -> None:
        project = self._project()
        if not project:
            return self.send(chat_id, "Primero elige un proyecto: /projects")
        rows = self.db.all(
            "SELECT t.id, t.title, EXISTS(SELECT 1 FROM runs r WHERE r.thread_id=t.id AND r.status IN ('queued','running')) AS busy "
            "FROM threads t WHERE t.project_id=? ORDER BY t.updated_at DESC LIMIT 12", (project["id"],))
        active = self.link().get("active_thread")
        buttons = [[{"text": ("📌 " if t["id"] == active else "⏳ " if t["busy"] else "") + f"#{t['id']} {t['title'][:55]}",
                     "callback_data": f"t:{t['id']}"}] for t in rows]
        buttons.append([{"text": "🆕 Nueva conversación", "callback_data": "t:new"}])
        if active:
            buttons.append([{"text": "🔓 Soltar la anclada", "callback_data": "t:detach"}])
        self.send(chat_id, f"Conversaciones de «{project['name']}» (📌 anclada · ⏳ trabajando). Toca una para anclarte:", buttons)

    def _attach(self, chat_id: int, thread_id: int) -> None:
        t = self.db.one("SELECT t.* FROM threads t JOIN projects p ON p.id=t.project_id WHERE t.id=? AND p.user_id=?",
                        (thread_id, self.user_id))
        if not t:
            return self.send(chat_id, "No encuentro esa conversación. /chats")
        self._set(active_thread=t["id"], active_project=t["project_id"])
        last = self.db.one("SELECT task, final_text, status FROM runs WHERE thread_id=? AND kind='chat' ORDER BY id DESC LIMIT 1", (t["id"],))
        tail = ""
        if last:
            tail = f"\n\nÚltimo mensaje: {last['task'][:300]}"
            if last["final_text"]:
                tail += f"\n\nÚltima respuesta:\n{last['final_text'][:1500]}"
            if last["status"] in ("queued", "running"):
                tail += "\n\n⏳ Está trabajando ahora mismo: te iré mostrando lo que haga."
        self.send(chat_id, f"📌 Anclado a «{t['title']}» (#{t['id']}). Lo que escribas va aquí y verás en directo lo que pase, "
                           f"también si la usas desde la app. /detach para soltarla.{tail}")

    def _current(self, chat_id: int) -> None:
        project = self._project()
        thread = self._thread()
        self.send(chat_id, f"Proyecto: {project['name'] if project else '—'}\n"
                           f"Conversación anclada: {('#' + str(thread['id']) + ' ' + thread['title']) if thread else '— (lo próximo que escribas empezará una nueva)'}")

    def _project(self) -> dict | None:
        link = self.link()
        p = None
        if link.get("active_project"):
            p = self.db.one("SELECT * FROM projects WHERE id=? AND user_id=?", (link["active_project"], self.user_id))
        if not p:
            p = self.db.one("SELECT * FROM projects WHERE user_id=? ORDER BY created_at DESC LIMIT 1", (self.user_id,))
            if p:
                self._set(active_project=p["id"])
        return p

    def _thread(self) -> dict | None:
        link = self.link()
        if not link.get("active_thread"):
            return None
        return self.db.one(
            "SELECT t.* FROM threads t JOIN projects p ON p.id=t.project_id WHERE t.id=? AND p.user_id=?",
            (link["active_thread"], self.user_id),
        )

    def _callback(self, cq: dict) -> None:
        chat_id = cq.get("message", {}).get("chat", {}).get("id")
        if chat_id != self.link().get("chat_id"):
            return
        data = cq.get("data", "")
        answer = ""
        if data.startswith("p:"):
            p = self.db.one("SELECT * FROM projects WHERE id=? AND user_id=?", (int(data[2:]), self.user_id))
            if p:
                self._set(active_project=p["id"], active_thread=None)
                answer = f"Proyecto: {p['name']}"
                self.send(chat_id, f"📁 Proyecto «{p['name']}». Escribe para empezar una conversación nueva, o /chats para anclarte a una.")
        elif data == "t:new":
            self._set(active_thread=None)
            answer = "Nueva conversación"
            self.send(chat_id, "🆕 Lo próximo que escribas empezará una conversación nueva.")
        elif data == "t:detach":
            self._set(active_thread=None)
            answer = "Soltada"
            self.send(chat_id, "🔓 Conversación soltada.")
        elif data.startswith("t:"):
            self._attach(chat_id, int(data[2:]))
            answer = "Anclado"
        elif data.startswith("b:"):
            bot = self.db.one("SELECT * FROM bots WHERE id=? AND user_id=?", (int(data[2:]), self.user_id)) \
                if data[2:].isdigit() else None
            if bot:
                self._run_bot(chat_id, bot, "")
                answer = "Lanzado"
            else:
                answer = "Ese bot ya no existe"
        elif data.startswith("ap:"):
            _, run_id, decision, call_id = data.split(":", 3)
            # only this user's runs: callback data comes from the client and a modified one can send anything
            mine = self.db.one("SELECT r.id FROM runs r JOIN projects p ON p.id=r.project_id WHERE r.id=? AND p.user_id=?",
                               (int(run_id), self.user_id))
            if mine and call_id.startswith("@") and call_id[1:].isdigit():  # the request's event holds the full call id
                ev = dict(self.db.events_after(int(run_id), int(call_id[1:]) - 1, limit=1))
                e = ev.get(int(call_id[1:])) or {}
                call_id = e.get("id", "") if e.get("type") == "approval_request" else ""
            ok = bool(mine and call_id) and self.jobs.decide(int(run_id), call_id, decision)
            answer = {"allow": "Permitido", "always": "Permitido siempre", "deny": "Denegado"}.get(decision, decision) if ok \
                else "Esa petición ya no está esperando"
            if ok:
                self.edit(chat_id, cq["message"]["message_id"], cq["message"].get("text", "") + f"\n\n→ {answer}")
        try:
            self.tg.call("answerCallbackQuery", callback_query_id=cq["id"], text=answer[:180] or None)
        except Exception:
            pass

    def _stop(self, chat_id: int) -> None:
        thread = self._thread()
        if not thread:
            return self.send(chat_id, "No hay nada en marcha.")
        # /stop also empties the queue: otherwise the next queued message would start right away
        waiting = self.db.all("SELECT id FROM runs WHERE thread_id=? AND status='waiting'", (thread["id"],))
        for w in waiting:
            self.db.execute("UPDATE runs SET status='cancelled', finished_at=? WHERE id=?", (time.time(), w["id"]))
        run = self.db.one("SELECT id FROM runs WHERE thread_id=? AND status IN ('queued','running')", (thread["id"],))
        stopped = bool(run and self.jobs.cancel(run["id"]))
        if stopped or waiting:
            self.send(chat_id, "⏹ Detenido." + (f" Quitados {len(waiting)} mensajes de la cola." if waiting else ""))
        else:
            self.send(chat_id, "No hay nada en marcha.")

    # ------------------------------------------------------------ chatting
    def _message(self, chat_id: int, text: str, msg: dict, more: list[dict] | None = None) -> None:
        """A message from the user (or an album: `msg` plus `more`) → one run in the anchored conversation."""
        from .chatops import ensure_thread, post_message, save_attachment

        msgs = [msg, *(more or [])]
        files, voice = _files_in(msgs), msg.get("voice")
        if not text and not files and not voice:
            if any(k in m for m in msgs for k in UNSUPPORTED):
                self.send(chat_id, "No sé qué hacer con ese tipo de mensaje: escríbeme, mándame una nota de voz o "
                                   "envíame fotos y archivos.")
            return  # a service message (pinned message, timer changed…): nothing to answer, no run
        project = self._project()
        if not project:
            return self.send(chat_id, "No tienes proyectos todavía: crea uno en la app.")
        if voice:  # what the user said, through the project's transcription model
            heard = self._hear(chat_id, project, voice)
            if heard is None:
                return
            text = f"{text}\n\n{heard}" if text else heard
        if big := [name for _, name, size in files if size > MAX_DOWNLOAD]:
            self._safe_send(chat_id, f"📎 {', '.join(big)}: Telegram no deja a los bots descargar archivos de más de "
                                     f"{MAX_DOWNLOAD // 2**20} MB. Pásamelos desde la app.")
            files = [f for f in files if f[2] <= MAX_DOWNLOAD]
            if not text and not files:
                return
        blobs = [(name, self.tg.download(file_id)) for file_id, name, _ in files]  # before creating anything
        thread = self._thread()
        if thread and thread["project_id"] != project["id"]:
            thread = None
        thread = thread or ensure_thread(self.db, project, self.user_id, text or "Telegram")
        self._set(active_thread=thread["id"], active_project=project["id"])
        attachments = [save_attachment(project, thread, name, data) for name, data in blobs]
        # the bot's own conversation: the user is talking to that bot (its instructions, memory, rules), as in the app
        bot = self.db.one("SELECT id FROM bots WHERE id=? AND user_id=? AND project_id=?",
                          (thread["bot_id"], self.user_id, project["id"])) if thread.get("bot_id") else None
        task = text or ("Mira el archivo adjunto." if len(attachments) == 1 else "Mira los archivos adjuntos.")
        run = post_message(self.db, self.jobs, project, thread, task, attachments,
                           bot_id=bot["id"] if bot else None, trigger="chat" if bot else "")
        self._claim(run["id"])
        status = self._safe_send(chat_id, "📥 En cola: empiezo cuando termine lo anterior (/stop para pararlo)."
                                 if run["status"] == "waiting" else "⏳ Trabajando…")
        threading.Thread(target=self._follow, args=(chat_id, run["id"], project, thread, status.get("message_id")),
                         daemon=True).start()

    def _album(self, chat_id: int, msg: dict) -> None:
        """Photos or files sent together arrive as separate messages sharing a media_group_id: gather them for a
        moment and start ONE run with all of them (the caption usually comes with the first)."""
        key = str(msg["media_group_id"])
        with self._album_lock:
            album = self._albums.setdefault(key, {"msgs": [], "timer": None})
            album["msgs"].append(msg)
            if album["timer"]:
                album["timer"].cancel()
            album["timer"] = threading.Timer(ALBUM_WAIT, self._album_done, args=(chat_id, key))
            album["timer"].daemon = True
            album["timer"].start()

    def _album_done(self, chat_id: int, key: str) -> None:
        with self._album_lock:
            album = self._albums.pop(key, None)
        if not album or self.stop.is_set():
            return
        msgs = sorted(album["msgs"], key=lambda m: m.get("message_id", 0))
        text = next((m["caption"].strip() for m in msgs if (m.get("caption") or "").strip()), "")
        try:
            self._message(chat_id, text, msgs[0], msgs[1:])
        except Exception as exc:
            self._safe_send(chat_id, f"⚠️ Error: {exc}")

    def _hear(self, chat_id: int, project: dict, voice: dict) -> str | None:
        """A voice note as text, through the project's transcription model (like the app's microphone button).
        None when it can't be heard: the user is told why."""
        spec = project.get("transcribe_model") or ""
        if not spec:
            self._safe_send(chat_id, "🎤 Para entender notas de voz, elige un modelo de «Transcripción» en los ajustes del "
                                     "proyecto (p. ej. openai/whisper-1). Mientras, escríbeme el mensaje.")
            return None
        if int(voice.get("file_size") or 0) > MAX_DOWNLOAD:
            self._safe_send(chat_id, "🎤 Esa nota de voz es demasiado larga para Telegram: escríbeme el mensaje.")
            return None
        try:
            from ..llm import get_provider
            from .auth import user_keys
            from .chatops import user_prefs

            provider, model = get_provider(spec, user_keys(self.db, self.user_id))
            lang = str(user_prefs(self.db, self.user_id).get("language") or "es").split("-")[0]
            heard = provider.transcribe(model, self.tg.download(voice["file_id"]), "nota-de-voz.ogg", lang).strip()
        except Exception as exc:
            self._safe_send(chat_id, f"🎤 No he podido transcribir la nota de voz: {exc}")
            return None
        if not heard:
            self._safe_send(chat_id, "🎤 No he entendido nada en la nota de voz.")
            return None
        self._safe_send(chat_id, f"🎤 «{heard[:1000]}»")  # what was understood, so a mishearing is caught early
        return heard

    def _follow(self, chat_id: int, run_id: int, project: dict, thread: dict, status_id: int | None, bot_name: str = "") -> None:
        """Mirror a run to Telegram: progress, approvals, images, files and the final answer. A message Telegram still
        refuses after the retries is skipped, never the rest of the relay."""
        try:
            self._relay(chat_id, run_id, project, thread, status_id, bot_name)
        except Exception as exc:  # e.g. the conversation was deleted mid-run
            print(f"nesicode telegram: stopped relaying run {run_id}: {exc}")
        finally:
            with self._followed_lock:
                self.followed.discard(run_id)

    def _relay(self, chat_id: int, run_id: int, project: dict, thread: dict, status_id: int | None, bot_name: str) -> None:
        from ..agent.workspace import Workspace

        ws = Workspace(thread_ws(project, thread))
        seen, steps, last_edit, tool = 0, 0, 0.0, ""
        while True:
            for seq, e in self.db.events_after(run_id, seen):
                seen = seq
                t = e.get("type")
                if t == "tool_call" and not e.get("depth"):
                    steps += 1
                    tool = e.get("name", "")
                elif t == "approval_request":
                    args = e.get("args") or {}
                    what = args.get("command") or args.get("url") or args.get("path") or json.dumps(args, ensure_ascii=False)[:300]
                    # callback_data holds 64 bytes: point at this event (it has the full call id), never cut the id
                    data = lambda d, s=seq: f"ap:{run_id}:{d}:@{s}"  # noqa: E731
                    row = [{"text": "✅ Permitir", "callback_data": data("allow")}]
                    if e.get("always", True):
                        row.append({"text": "♾ Siempre", "callback_data": data("always")})
                    row.append({"text": "⛔ Denegar", "callback_data": data("deny")})
                    self._safe_send(chat_id, f"🔐 Nesicode quiere usar {e.get('name')}:\n{what}", [row])
                elif t == "bot_finished" and not bot_name and not e.get("depth"):
                    # a chat with the bot from here: its finish sends no separate notice, so this is where the user sees it
                    self._safe_send(chat_id, f"✅ Tarea cumplida: el bot se ha desactivado — {e.get('text', '')}")
                elif t == "image":
                    self._send_file(chat_id, ws, e.get("path", ""), e.get("caption", ""), photo=True)
                elif t == "file":
                    self._send_file(chat_id, ws, e.get("path", ""), e.get("caption", ""), photo=False)
            if status_id and steps and time.time() - last_edit > 3:
                self.edit(chat_id, status_id, f"⏳ Trabajando… (paso {steps}: {tool})")
                last_edit = time.time()
            run = self.db.one("SELECT status, final_text, error, thread_id FROM runs WHERE id=?", (run_id,))
            if not run or run["thread_id"] != thread["id"]:
                return  # deleted with its conversation (and maybe its id already given to another run)
            if run["status"] in FINAL and not self.db.events_after(run_id, seen):
                break
            time.sleep(POLL)
        if status_id:
            self.edit(chat_id, status_id, f"✓ Hecho en {steps} pasos" if run["status"] in ("done", "max_steps")
                      else f"Estado: {run['status']}")
        final = run["final_text"] or run["error"] or f"(terminado: {run['status']})"
        if bot_name:
            if run["status"] == "done" and final.strip().upper().startswith("NOTHING_NEW"):
                return  # the bot had nothing to report: stay silent
            if run["status"] == "cancelled":
                return
            final = f"🤖 {bot_name}" + ("" if run["status"] in ("done", "max_steps") else f" · {run['status']}") + f"\n\n{final}"
        self._safe_send(chat_id, final)

    def follow_bot(self, bot: dict, run_id: int, project: dict, thread: dict) -> None:
        chat_id = self.link().get("chat_id")
        if not chat_id or not self._claim(run_id):
            return
        threading.Thread(target=self._follow, args=(chat_id, run_id, project, thread, None, bot["name"]),
                         daemon=True, name="nesi-tg-bot").start()

    def _bots(self, chat_id: int) -> None:
        rows = self.db.all("SELECT id, name, enabled, completed_at FROM bots WHERE user_id=? ORDER BY name", (self.user_id,))
        if not rows:
            return self.send(chat_id, "No tienes bots. Créalos en la app (Bots) o pídeselo al agente: «hazme un bot que…».")
        mark = lambda r: "▶ " if r["enabled"] else "✅ " if r["completed_at"] else "⏸ "  # noqa: E731 (✅ = its task is done)
        buttons = [[{"text": mark(r) + r["name"][:50], "callback_data": f"b:{r['id']}"}] for r in rows]
        self.send(chat_id, "🤖 Tus bots — pulsa uno para lanzarlo ahora, o escribe /bot <nombre> <instrucción>:", buttons)

    def _bot_command(self, chat_id: int, rest: str) -> None:
        """/bot <name or id> [text]. Names can have spaces: the longest bot name the text starts with (whole words, any
        case) wins, so '/bot Post Semanal resume…' is «Post Semanal», never «Post Diario». When only the start of a
        name was given and several bots have it, they are offered as buttons instead of guessing."""
        if not getattr(self.jobs, "bots", None):
            return self.send(chat_id, "Los bots no están disponibles.")
        rows = self.db.all("SELECT * FROM bots WHERE user_id=? ORDER BY name, id", (self.user_id,))
        words = rest.split()
        folded = [w.casefold() for w in words]
        named = [(len(n), b) for b in rows if (n := b["name"].casefold().split()) and folded[:len(n)] == n]
        n = max((k for k, _ in named), default=0)
        found = [b for k, b in named if k == n]
        if not found and words[0].isdigit():
            found, n = [b for b in rows if b["id"] == int(words[0])], 1
        if not found:
            found, n = [b for b in rows if b["name"].casefold().startswith(folded[0])], 1
        if not found:
            return self.send(chat_id, "No encuentro ese bot. /bots")
        if len(found) > 1:
            buttons = [[{"text": "▶ " + b["name"][:50], "callback_data": f"b:{b['id']}"}] for b in found[:20]]
            return self.send(chat_id, f"Hay varios bots que encajan con «{' '.join(words[:max(n, 1)])}»: pulsa el que "
                                      "quieras, o escribe su nombre completo: /bot <nombre> <instrucción>", buttons)
        parts = rest.split(None, n)
        self._run_bot(chat_id, found[0], parts[n] if len(parts) > n else "")

    def _run_bot(self, chat_id: int, bot: dict, text: str) -> None:
        bots = getattr(self.jobs, "bots", None)
        if not bots:
            return self.send(chat_id, "Los bots no están disponibles.")
        run = bots.fire(bot["id"], "telegram", text or "Haz tu tarea ahora.", force=True)
        if not run:
            return self.send(chat_id, f"No he podido lanzar «{bot['name']}».")
        self.send(chat_id, f"▶ Lanzado «{bot['name']}»" + (" (en cola)" if run["status"] == "waiting" else "") + "…")
        if bot["notify"] != "telegram":  # still show the answer here: the user asked from Telegram
            project = self.db.one("SELECT * FROM projects WHERE id=?", (bot["project_id"],))
            thread = self.db.one("SELECT * FROM threads WHERE id=?", (run["thread_id"],))
            self.follow_bot(bot, run["id"], project, thread)

    def _send_file(self, chat_id: int, ws, rel: str, caption: str, photo: bool) -> None:
        """Send a file the run produced. A picture Telegram won't take as a photo (a tall screenshot, too big) goes as
        a document; one it won't take at all (over 50 MB) or that still fails is named in a message, never dropped.
        Read like every server read of a workspace (Workspace.open_file): no link followed below it, even one swapped in
        after the path was checked (the server would send a file from outside the cage), never blocking on a FIFO (the
        relay would hang for good), regular files only; the size is the opened file's, the bytes are the ones read."""
        try:
            p = ws.resolve(rel)
            with ws.open_file(p) as fh:
                size = os.fstat(fh.fileno()).st_size
                data = fh.read(MAX_UPLOAD + 1) if size <= MAX_UPLOAD else b""  # checked before reading it into memory
        except Exception:
            return
        caption = (caption or p.name)[:1000]
        size = max(size, len(data))  # (it may have grown meanwhile)
        if size > MAX_UPLOAD:
            self._safe_send(chat_id, f"📎 «{p.name}» ocupa {size / 2**20:.0f} MB y Telegram no deja a los bots enviar más de "
                                     f"{MAX_UPLOAD // 2**20} MB: lo tienes en la conversación, en la app.")
            return
        if photo and image_mime(p.name) and size <= MAX_PHOTO:
            try:
                self.tg.call("sendPhoto", chat_id=chat_id, caption=caption, _files={"photo": (p.name, data)})
                return
            except Exception:  # PHOTO_INVALID_DIMENSIONS and the like: as a document it goes through
                pass
        try:
            self.tg.call("sendDocument", chat_id=chat_id, caption=caption, _files={"document": (p.name, data)})
        except Exception as exc:
            self._safe_send(chat_id, f"📎 No he podido enviarte «{p.name}» por Telegram ({exc}): lo tienes en la "
                                     "conversación, en la app.")


def _files_in(msgs: list[dict]) -> list[tuple[str, str, int]]:
    """(file_id, name, size) of what the messages carry that can be attached. Voice notes are transcribed instead."""
    out, seen = [], set()
    for m in msgs:
        mid = m.get("message_id", 0)
        found = [(m["photo"][-1], f"foto-{mid}.jpg")] if m.get("photo") else []
        for kind, default in (("document", f"archivo-{mid}"), ("audio", f"audio-{mid}.mp3"), ("video", f"video-{mid}.mp4"),
                              ("video_note", f"video-{mid}.mp4"), ("animation", f"animacion-{mid}.mp4")):
            if m.get(kind):
                found.append((m[kind], m[kind].get("file_name") or default))
        for f, name in found:
            key = f.get("file_unique_id") or f.get("file_id")
            if f.get("file_id") and key not in seen:  # an animation also comes as a document: attach it once
                seen.add(key)
                out.append((f["file_id"], name, int(f.get("file_size") or 0)))
    return out


def unanchor_deleted(db: DB) -> None:
    """Telegram is anchored to a conversation (and a project) by id, and SQLite gives the ids of deleted rows to new
    ones: when the anchored conversation or project is deleted, drop the anchor — otherwise the next conversation that
    gets that id would receive Telegram's messages and be mirrored there. The database does it, so every way of
    deleting (a conversation, a whole project) is covered."""
    db.execute("CREATE TRIGGER IF NOT EXISTS telegram_unanchor_thread AFTER DELETE ON threads BEGIN "
               "UPDATE telegram_links SET active_thread=NULL WHERE active_thread=OLD.id; END")
    db.execute("CREATE TRIGGER IF NOT EXISTS telegram_unanchor_project AFTER DELETE ON projects BEGIN "
               "UPDATE telegram_links SET active_project=NULL WHERE active_project=OLD.id; END")
    # anchors left dangling before this existed
    db.execute("UPDATE telegram_links SET active_thread=NULL WHERE active_thread NOT IN (SELECT id FROM threads)")
    db.execute("UPDATE telegram_links SET active_project=NULL WHERE active_project NOT IN (SELECT id FROM projects)")


class TelegramManager:
    """Starts/stops one bridge per user that has configured a bot."""

    def __init__(self, db: DB, jobs: Jobs, api: str = API, transport=None):
        self.db, self.jobs = db, jobs
        self.api, self.transport = api, transport  # another Bot API endpoint (tests use a fake one)
        self.vault = Vault(db)
        self.bridges: dict[int, TelegramBridge] = {}
        self._errors: dict[int, str] = {}
        unanchor_deleted(db)

    @property
    def errors(self) -> dict[int, str]:
        """Per user: why Telegram isn't working — a token it rejects, or no connection yet (then it keeps retrying)."""
        out = dict(self._errors)
        for uid, b in list(self.bridges.items()):
            if b.error:
                out[uid] = b.error
        return out

    def start_all(self) -> None:
        for row in self.db.all("SELECT user_id FROM telegram_links"):
            self.restart(row["user_id"])

    def restart(self, user_id: int) -> dict | None:
        old = self.bridges.pop(user_id, None)
        if old:
            old.close()
        token = self.vault.get(user_id, "telegram:bot")
        if not token:
            return None
        bridge = TelegramBridge(self.db, self.jobs, user_id, token, self.api, self.transport)
        try:
            bridge.start()  # unreachable is fine (it retries in the background); a rejected token is not
        except Exception as exc:
            self._errors[user_id] = str(exc)
            return None
        self._errors.pop(user_id, None)
        self.bridges[user_id] = bridge
        return bridge.me

    def stop(self, user_id: int) -> None:
        b = self.bridges.pop(user_id, None)
        if b:
            b.close()

    # ------------------------------------------------------------ for bots
    def notify(self, user_id: int, text: str) -> bool:
        b = self.bridges.get(user_id)
        chat_id = b.link().get("chat_id") if b else None
        if not chat_id:
            return False
        try:
            b.send(chat_id, text[:4000])
        except Exception as exc:  # still failing after the retries: the caller hears "not sent", not an exception
            print(f"nesicode telegram: notice not sent: {exc}")
            return False
        return True

    def follow_bot_run(self, bot: dict, run_id: int, project: dict, thread: dict) -> None:
        b = self.bridges.get(bot["user_id"])
        if b:
            b.follow_bot(bot, run_id, project, thread)
