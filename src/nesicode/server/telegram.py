"""Telegram bot bridge: manage your Nesicode conversations from Telegram.

The user creates a bot with @BotFather and pastes its token (stored in the vault). Nesicode
long-polls Telegram (no public URL needed), pairs ONE chat with a one-time /start code, and from
then on only that chat is obeyed. Messages go to the active conversation; answers, images, files
and approval requests (with inline buttons) come back to Telegram.
"""

from __future__ import annotations

import json
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


class TelegramClient:
    def __init__(self, token: str, api: str = API, transport: httpx.BaseTransport | None = None):
        self.base = f"{api}/bot{token}"
        self.http = httpx.Client(timeout=httpx.Timeout(70.0, connect=15.0), transport=transport)

    def call(self, method: str, **params) -> dict | list | bool:
        files = params.pop("_files", None)
        if files:
            r = self.http.post(f"{self.base}/{method}", data={k: v for k, v in params.items() if v is not None}, files=files)
        else:
            r = self.http.post(f"{self.base}/{method}", json={k: v for k, v in params.items() if v is not None})
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(data.get("description") or f"Telegram {method} failed")
        return data["result"]

    def download(self, file_id: str) -> bytes:
        info = self.call("getFile", file_id=file_id)
        url = self.base.replace("/bot", "/file/bot", 1) + "/" + info["file_path"]
        return self.http.get(url).content


def _chunks(text: str, size: int = MAX_TEXT) -> list[str]:
    text = text.strip() or "(sin texto)"
    return [text[i : i + size] for i in range(0, len(text), size)]


class TelegramBridge:
    """One bot for one Nesicode user. Runs its own polling thread."""

    def __init__(self, db: DB, jobs: Jobs, user_id: int, token: str, api: str = API, transport=None):
        self.db, self.jobs, self.user_id = db, jobs, user_id
        self.tg = TelegramClient(token, api, transport)
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.me: dict = {}
        self.followed: set[int] = set()  # run ids already mirrored to Telegram

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
        self.me = self.tg.call("getMe")
        try:
            self.tg.call("setMyCommands", commands=COMMANDS)
        except Exception:
            pass
        self.thread = threading.Thread(target=self._loop, daemon=True, name=f"nesi-telegram-{self.user_id}")
        self.thread.start()
        threading.Thread(target=self._mirror, daemon=True, name=f"nesi-telegram-mirror-{self.user_id}").start()

    def close(self) -> None:
        self.stop.set()

    def _loop(self) -> None:
        backoff = 1
        while not self.stop.is_set():
            try:
                offset = self.link().get("update_offset") or 0
                updates = self.tg.call("getUpdates", offset=offset, timeout=50,
                                       allowed_updates=["message", "callback_query"])
                backoff = 1
                for u in updates:
                    self._set(update_offset=u["update_id"] + 1)
                    try:
                        self.handle(u)
                    except Exception as exc:  # one bad update must not stop the bot
                        chat = (u.get("message") or (u.get("callback_query") or {}).get("message") or {}).get("chat", {})
                        if chat.get("id") and chat["id"] == self.link().get("chat_id"):
                            self.send(chat["id"], f"⚠️ Error: {exc}")
            except Exception:
                self.stop.wait(min(backoff, 60))
                backoff *= 2

    def _mirror(self) -> None:
        """While attached to a chat, show runs that were started from the app too."""
        while not self.stop.wait(2):
            try:
                link = self.link()
                chat_id, tid = link.get("chat_id"), link.get("active_thread")
                if not chat_id or not tid:
                    continue
                for r in self.db.all("SELECT id FROM runs WHERE thread_id=? AND kind='chat' AND status IN ('queued','running')", (tid,)):
                    if r["id"] in self.followed:
                        continue
                    self.followed.add(r["id"])
                    thread = self._thread()
                    project = self.db.one("SELECT * FROM projects WHERE id=?", (thread["project_id"],)) if thread else None
                    if not project:
                        continue
                    task = self.db.one("SELECT task FROM runs WHERE id=?", (r["id"],))["task"]
                    self.send(chat_id, f"💻 Desde la app, en «{thread['title']}»:\n{task[:1500]}")
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
            name, _, text = rest.partition(" ")
            return self._run_bot(chat_id, name, text)
        if cmd:
            return self.send(chat_id, "No conozco ese comando. /help")
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
            self._run_bot(chat_id, data[2:], "")
            answer = "Lanzado"
        elif data.startswith("ap:"):
            _, run_id, decision, call_id = data.split(":", 3)
            ok = self.jobs.decide(int(run_id), call_id, decision)
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
    def _message(self, chat_id: int, text: str, msg: dict) -> None:
        from .chatops import ensure_thread, post_message, save_attachment

        project = self._project()
        if not project:
            return self.send(chat_id, "No tienes proyectos todavía: crea uno en la app.")
        thread = self._thread()
        if thread and thread["project_id"] != project["id"]:
            thread = None
        thread = thread or ensure_thread(self.db, project, self.user_id, text or "Telegram")
        self._set(active_thread=thread["id"], active_project=project["id"])
        attachments = []
        files = []
        if msg.get("photo"):
            files.append((msg["photo"][-1]["file_id"], f"foto-{msg['message_id']}.jpg"))
        if msg.get("document"):
            files.append((msg["document"]["file_id"], msg["document"].get("file_name") or f"archivo-{msg['message_id']}"))
        for file_id, name in files:
            attachments.append(save_attachment(project, thread, name, self.tg.download(file_id)))
        run = post_message(self.db, self.jobs, project, thread, text or "Mira el archivo adjunto.", attachments)
        self.followed.add(run["id"])
        status = self.send(chat_id, "📥 En cola: empiezo cuando termine lo anterior (/stop para pararlo)."
                           if run["status"] == "waiting" else "⏳ Trabajando…")
        threading.Thread(target=self._follow, args=(chat_id, run["id"], project, thread, status.get("message_id")),
                         daemon=True).start()

    def _follow(self, chat_id: int, run_id: int, project: dict, thread: dict, status_id: int | None, bot_name: str = "") -> None:
        """Mirror a run to Telegram: progress, approvals, images, files and the final answer."""
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
                    data = lambda d: f"ap:{run_id}:{d}:{e['id']}"[:64]  # noqa: E731
                    row = [{"text": "✅ Permitir", "callback_data": data("allow")}]
                    if e.get("always", True):
                        row.append({"text": "♾ Siempre", "callback_data": data("always")})
                    row.append({"text": "⛔ Denegar", "callback_data": data("deny")})
                    self.send(chat_id, f"🔐 Nesicode quiere usar {e.get('name')}:\n{what}", [row])
                elif t == "image":
                    self._send_file(chat_id, ws, e.get("path", ""), e.get("caption", ""), photo=True)
                elif t == "file":
                    self._send_file(chat_id, ws, e.get("path", ""), e.get("caption", ""), photo=False)
            if status_id and steps and time.time() - last_edit > 3:
                self.edit(chat_id, status_id, f"⏳ Trabajando… (paso {steps}: {tool})")
                last_edit = time.time()
            run = self.db.one("SELECT status, final_text, error FROM runs WHERE id=?", (run_id,))
            if run["status"] in FINAL and not self.db.events_after(run_id, seen):
                break
            time.sleep(1)
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
        self.send(chat_id, final)

    def follow_bot(self, bot: dict, run_id: int, project: dict, thread: dict) -> None:
        chat_id = self.link().get("chat_id")
        if not chat_id or run_id in self.followed:
            return
        self.followed.add(run_id)
        threading.Thread(target=self._follow, args=(chat_id, run_id, project, thread, None, bot["name"]),
                         daemon=True, name="nesi-tg-bot").start()

    def _bots(self, chat_id: int) -> None:
        rows = self.db.all("SELECT id, name, enabled FROM bots WHERE user_id=? ORDER BY name", (self.user_id,))
        if not rows:
            return self.send(chat_id, "No tienes bots. Créalos en la app (Bots) o pídeselo al agente: «hazme un bot que…».")
        buttons = [[{"text": ("▶ " if r["enabled"] else "⏸ ") + r["name"][:50], "callback_data": f"b:{r['id']}"}] for r in rows]
        self.send(chat_id, "🤖 Tus bots — pulsa uno para lanzarlo ahora, o escribe /bot <nombre> <instrucción>:", buttons)

    def _run_bot(self, chat_id: int, name_or_id: str, text: str) -> None:
        bots = getattr(self.jobs, "bots", None)
        if not bots:
            return self.send(chat_id, "Los bots no están disponibles.")
        q = name_or_id.strip().lower()
        rows = self.db.all("SELECT * FROM bots WHERE user_id=?", (self.user_id,))
        bot = next((b for b in rows if str(b["id"]) == q or b["name"].lower() == q), None) or \
            next((b for b in rows if b["name"].lower().startswith(q)), None)
        if not bot:
            return self.send(chat_id, "No encuentro ese bot. /bots")
        run = bots.fire(bot["id"], "telegram", text or "Haz tu tarea ahora.", force=True)
        if not run:
            return self.send(chat_id, f"No he podido lanzar «{bot['name']}».")
        self.send(chat_id, f"▶ Lanzado «{bot['name']}»" + (" (en cola)" if run["status"] == "waiting" else "") + "…")
        if bot["notify"] != "telegram":  # still show the answer here: the user asked from Telegram
            project = self.db.one("SELECT * FROM projects WHERE id=?", (bot["project_id"],))
            thread = self.db.one("SELECT * FROM threads WHERE id=?", (run["thread_id"],))
            self.follow_bot(bot, run["id"], project, thread)

    def _send_file(self, chat_id: int, ws, rel: str, caption: str, photo: bool) -> None:
        try:
            p = ws.resolve(rel)
            data = p.read_bytes()
        except Exception:
            return
        method, field = ("sendPhoto", "photo") if photo and image_mime(p.name) and len(data) < 10_000_000 else ("sendDocument", "document")
        try:
            self.tg.call(method, chat_id=chat_id, caption=(caption or p.name)[:1000], _files={field: (p.name, data)})
        except Exception:
            pass


class TelegramManager:
    """Starts/stops one bridge per user that has configured a bot."""

    def __init__(self, db: DB, jobs: Jobs):
        self.db, self.jobs = db, jobs
        self.vault = Vault(db)
        self.bridges: dict[int, TelegramBridge] = {}
        self.errors: dict[int, str] = {}

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
        bridge = TelegramBridge(self.db, self.jobs, user_id, token)
        try:
            bridge.start()
        except Exception as exc:
            self.errors[user_id] = str(exc)
            return None
        self.errors.pop(user_id, None)
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
        b.send(chat_id, text[:4000])
        return True

    def follow_bot_run(self, bot: dict, run_id: int, project: dict, thread: dict) -> None:
        b = self.bridges.get(bot["user_id"])
        if b:
            b.follow_bot(bot, run_id, project, thread)
