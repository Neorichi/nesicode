"""Background execution of chat turns and harnessing runs, with a live event bus."""

from __future__ import annotations

import base64
import json
import re
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..agent import Agent, Workspace, apply_changes, copy_workspace
from ..agent.workspace import ATTACHMENTS_DIR
from ..config import settings
from ..harnessing import HarnessingConfig, HarnessingRun
from .auth import user_keys
from .db import DB, loads

FINAL = {"done", "error", "cancelled", "interrupted", "max_steps", "refusal"}


# Three isolation layers on disk, each nested in the previous one:
#   workspaces/<user>/                          layer 1: user (API ownership checks, always)
#   workspaces/<user>/<project>/project/         layer 2: the project's base files
#   workspaces/<user>/<project>/threads/<id>/ws/ layer 3: one cage per conversation
#   workspaces/<user>/<project>/runs/<id>/       harnessing rollouts (each rollout is its own cage)
# Every bash command runs under the OS sandbox with only its own cage readable/writable.


def user_root(user_id: int) -> Path:
    return settings.workspaces_dir / str(int(user_id))


def project_root(project: dict) -> Path:
    return user_root(project["user_id"]) / str(int(project["id"]))


def project_ws(project: dict) -> Path:
    """The project's main folder: a folder the user picked on disk, or nesicode's own copy."""
    if project.get("folder"):
        return Path(project["folder"])
    return project_root(project) / "project"


def thread_dir(project: dict, thread_id: int) -> Path:
    return project_root(project) / "threads" / str(int(thread_id))


def thread_ws(project: dict, thread: dict) -> Path:
    """Where a conversation works: its own cage, or the shared project dir if not isolated."""
    if thread.get("isolated", 1):
        return thread_dir(project, thread["id"]) / "ws"
    return project_ws(project)


def run_dir(project: dict, run_id: int) -> Path:
    return project_root(project) / "runs" / str(int(run_id))


def create_cage(project: dict, thread_id: int) -> dict[str, str]:
    """Copy the project into a fresh conversation cage; return its starting manifest."""
    from ..agent.workspace import manifest

    ws = thread_dir(project, thread_id) / "ws"
    src = project_ws(project)
    src.mkdir(parents=True, exist_ok=True)
    copy_workspace(src, ws)
    return manifest(ws)


def migrate_layout(db: DB) -> None:
    """Move pre-isolation data (files directly under <user>/<project>, runs under data/runs)."""
    for p in db.all("SELECT * FROM projects"):
        root = project_root(p)
        if root.is_dir() and not p.get("folder") and not (root / "project").exists():
            entries = list(root.iterdir())
            (root / "project").mkdir()
            for e in entries:
                if e.name not in ("project", "threads", "runs"):
                    shutil.move(str(e), str(root / "project" / e.name))
        for t in db.all("SELECT id FROM threads WHERE project_id=? AND isolated=1", (p["id"],)):
            if not (thread_dir(p, t["id"]) / "ws").exists():
                db.execute("UPDATE threads SET manifest=? WHERE id=?", (json.dumps(create_cage(p, t["id"])), t["id"]))
        for r in db.all("SELECT id FROM runs WHERE project_id=? AND kind='harness'", (p["id"],)):
            old = settings.runs_dir / str(r["id"])
            if old.is_dir() and not run_dir(p, r["id"]).exists():
                run_dir(p, r["id"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(run_dir(p, r["id"])))


def current_harness(db: DB, project: dict) -> dict:
    row = db.one(
        "SELECT files FROM harness_versions WHERE project_id=? AND version=?",
        (project["id"], project["harness_version"]),
    )
    return loads(row["files"], {}) if row else {}


def save_harness_version(db: DB, project_id: int, files: dict, source: str, note: str, run_id: int | None = None) -> int:
    with db.lock:
        row = db.one("SELECT COALESCE(MAX(version), 0) + 1 AS v FROM harness_versions WHERE project_id=?", (project_id,))
        v = int(row["v"])
        db.execute(
            "INSERT INTO harness_versions (project_id, version, files, source, note, run_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (project_id, v, json.dumps(files, ensure_ascii=False), source, note, run_id, time.time()),
        )
        db.execute("UPDATE projects SET harness_version=? WHERE id=?", (v, project_id))
    return v


def attachments_note(ws: Workspace, attachments: list[dict], vision: str, keys: dict, emit) -> str:
    """Tell the agent what was attached; describe images up front with the vision model."""
    from ..agent.vision import describe_image

    lines = ["The user attached these files (paths relative to the workspace):"]
    for a in attachments:
        lines.append(f"- `{a['path']}` ({a['kind']}, {a['size']:,} bytes)" + (" — read it with read_file" if a["kind"] in ("text", "pdf") else ""))
        if a["kind"] != "image":
            continue
        if not vision:
            lines.append("  (no vision model is configured, so you cannot see this image; say so if it matters)")
            continue
        emit({"type": "tool_call", "id": f"att-{a['path']}", "name": "view_image", "args": {"path": a["path"]}})
        try:
            text = describe_image(ws, a["path"], vision, keys, "Describe this image in detail for a software engineer.")
            emit({"type": "tool_result", "id": f"att-{a['path']}", "name": "view_image", "output": text, "is_error": False})
            lines.append("  Image description (from the vision model):\n    " + text.replace("\n", "\n    "))
        except Exception as exc:
            emit({"type": "tool_result", "id": f"att-{a['path']}", "name": "view_image", "output": str(exc), "is_error": True})
            lines.append(f"  (the vision model failed on this image: {exc})")
    return "\n".join(lines)


def _write_thread_history(db: DB, ws: Workspace, run: dict, archive: str) -> None:
    """Earlier turns of this conversation, complete with their steps, so `search_history` can find
    anything that only survives as a summary in the model's context."""
    from ..agent import trajectory_text

    rows = db.all("SELECT id, task, final_text FROM runs WHERE thread_id=? AND id<? AND kind='chat' ORDER BY id",
                  (run["thread_id"], run["id"]))
    if not rows:
        return
    parts = ["# Historial de esta conversación (turnos anteriores)"]
    for r in rows[-40:]:
        events = [e for _, e in db.events_after(r["id"], 0, limit=5000)]
        parts.append(f"## Turno #{r['id']}\n### Usuario\n{r['task']}\n### Pasos\n{trajectory_text(events, 60_000)}\n"
                     f"### Respuesta final\n{r['final_text']}")
    try:
        from ..agent.context import write_context_file

        write_context_file(ws, "history.md", "\n\n".join(parts), archive)
    except Exception:
        pass


def browser_tools(user_id: int) -> dict:
    """The user's own Chrome (Nesicode extension), when it is connected."""
    from ..browser import TOOLS, hub

    if not hub.connected(user_id):
        return {}
    last = {"tab": None}

    def make(name: str):
        def call(args: dict, agent) -> str:
            from ..agent.workspace import ToolError

            a = dict(args or {})
            if a.get("tab_id") is None and last["tab"] is not None and name != "chrome_tabs":
                a["tab_id"] = last["tab"]
            try:
                res = hub.call(user_id, name.removeprefix("chrome_"), a, timeout=60)
            except Exception as exc:
                raise ToolError(str(exc)) from exc
            if res.get("tab_id") is not None:
                last["tab"] = res["tab_id"]
            if name == "chrome_screenshot" and res.get("image"):
                data = base64.b64decode(res.pop("image").split(",", 1)[-1])
                rel = f".chrome/screenshot-{int(time.time() * 1000)}.jpg"
                target = agent.ws.resolve(rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                agent.emit({"type": "image", "path": rel, "caption": res.get("title", "Chrome")})
                res["saved_to"] = rel
            return json.dumps(res, ensure_ascii=False)[:20000]

        return call

    return {name: (spec, make(name)) for name, spec in TOOLS.items()}


def builtin_agent_tools(connector: str, label: str, which: str, cfg=None) -> dict:
    """Built-in connectors (Office, Burp REST) run in-process on the conversation's own workspace."""
    from ..connectors import builtin_module, tool_name
    from ..llm import ToolSpec

    try:
        mod = builtin_module(which)
    except RuntimeError:
        return {}
    TOOLS = mod.TOOLS
    takes_cfg = which != "office"

    def make(fn):
        def call(args: dict, agent) -> str:
            text, written = fn(args or {}, agent.ws, cfg) if takes_cfg else fn(args or {}, agent.ws)
            if written:
                p = agent.ws.resolve(written)
                agent.emit({"type": "file", "path": written, "size": p.stat().st_size, "caption": label})
                text += " Already shared with the user as a download (no need for share_file)."
            return text

        return call

    out = {}
    for name, (spec, fn) in TOOLS.items():
        full = tool_name(connector, name)
        out[full] = (ToolSpec(full, f"[{label}] {spec.description}", spec.parameters), make(fn))
    return out


PATH_ARG = re.compile(r"(file|path|dir|folder|output|document|workbook|presentation|docx|xlsx|pptx)", re.I)


def absolutize_paths(args: dict, ws) -> dict:
    """External servers don't run in the conversation's folder: turn relative file paths the agent
    passes into absolute paths inside its cage, so files land (and are read) where they belong."""
    out = dict(args or {})
    for key, value in out.items():
        if not isinstance(value, str) or not PATH_ARG.search(key):
            continue
        v = value.strip()
        if not v or v.startswith(("/", "~", "\\")) or "://" in v or re.match(r"^[A-Za-z]:[\\/]", v):
            continue
        try:
            out[key] = str(ws.resolve(v))
        except Exception:
            pass  # e.g. "../.." escaping the cage: leave it, the server decides
    return out


def _mcp_caller(user_id: int, cfg, tool: str):
    def call(args: dict, agent) -> str:
        from ..agent.workspace import ToolError
        from ..connectors import manager

        args = absolutize_paths(args, agent.ws)
        try:
            res = manager().call(user_id, cfg, tool, args)
        except Exception as exc:
            raise ToolError(f"{cfg.name}: {exc}") from exc
        text = res.text or "(no text output)"
        for i, (data, mime) in enumerate(res.images[:4]):  # screenshots etc. land in the cage and in the chat
            ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(mime, "png")
            rel = f".mcp/{cfg.name}-{int(time.time() * 1000)}-{i}.{ext}"
            target = agent.ws.resolve(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            agent.emit({"type": "image", "path": rel, "caption": f"{cfg.name} · {tool}"})
            text += f"\n[image saved to {rel}]"
        if res.is_error:
            raise ToolError(text[:4000])
        return text[:20000]

    return call


def needs_improvement(res, max_steps: int, prefs: dict) -> str:
    """Why a finished chat answer deserves a harnessing run, or '' if it went fine."""
    if res.status not in ("done", "max_steps"):
        return ""  # API errors, refusals and cancellations are not the harness's fault
    if res.status == "max_steps":
        return "agotó el límite de pasos"
    errors = sum(1 for e in res.events if e.get("type") == "tool_result" and e.get("is_error") and not e.get("depth"))
    if errors >= int(prefs.get("auto_harness_min_errors", 3)):
        return f"{errors} errores de herramientas"
    limit = int(prefs.get("auto_harness_min_steps", max(15, max_steps // 2)))
    if res.steps >= limit:
        return f"necesitó {res.steps} pasos"
    return ""


EXEC_MODES = ("auto", "ask_bash", "ask_all")
APPROVAL_TIMEOUT = 30 * 60
WRITE_TOOLS = {"write_file", "edit_file", "generate_image"}


ALWAYS_ASK = ("__burp_scan", "create_bot", "__vt_scan_file", "__vt_scan_url")  # active attacks against a target: the user approves each one, in every mode


def needs_approval(mode: str, tool: str) -> bool:
    if tool.endswith(ALWAYS_ASK) or tool == "create_bot":
        return True
    if tool.startswith(("mcp__", "chrome_")):  # act outside the cage (the user's real browser, accounts…)
        return mode in ("ask_bash", "ask_all")
    if mode == "ask_all":
        return tool == "bash" or tool in WRITE_TOOLS or tool == "delegate"
    if mode == "ask_bash":
        return tool == "bash"
    return False


class Approval:
    def __init__(self, call_id: str):
        self.call_id = call_id
        self.event = threading.Event()
        self.decision = ""


class Live:
    """Transient (non-persisted) events such as token deltas, per run."""

    def __init__(self):
        self.gen = 0
        self.items: list[dict] = []


class Jobs:
    def __init__(self, db: DB):
        self.db = db
        self.pool = ThreadPoolExecutor(max_workers=settings.max_jobs, thread_name_prefix="nesi-job")
        self.cancels: dict[int, threading.Event] = {}
        self.live: dict[int, Live] = {}
        self.approvals: dict[int, Approval] = {}  # run_id -> the tool call waiting for the user
        self.project_locks: dict[int, threading.Lock] = {}
        self.on_finish: list = [self._auto_title]  # callbacks(run_id) after any run ends (bots use it too)
        self.bots = None  # BotManager, set by the app
        self._lock = threading.Lock()
        db.execute(
            "UPDATE runs SET status='interrupted', finished_at=? WHERE status IN ('queued', 'running')",
            (time.time(),),
        )
        migrate_layout(db)

    def _project_lock(self, project_id: int) -> threading.Lock:
        with self._lock:
            return self.project_locks.setdefault(project_id, threading.Lock())

    def emitter(self, run_id: int):
        live = self.live.setdefault(run_id, Live())

        def emit(event: dict) -> None:
            event = dict(event)
            persist = event.pop("_persist", True)
            if persist:
                self.db.add_event(run_id, event)
                if event.get("type") in ("assistant", "done", "error"):
                    live.gen += 1
                    live.items = []
            else:
                live.items.append(event)

        return emit

    def cancel(self, run_id: int) -> bool:
        ev = self.cancels.get(run_id)
        if ev:
            ev.set()
            return True
        return False

    def _finish(self, run_id: int, status: str, **fields) -> None:
        sets = ["status=?", "finished_at=?"] + [f"{k}=?" for k in fields]
        self.db.execute(
            f"UPDATE runs SET {', '.join(sets)} WHERE id=?",
            (status, time.time(), *fields.values(), run_id),
        )
        self.cancels.pop(run_id, None)
        self.live.pop(run_id, None)
        for cb in list(self.on_finish):
            try:
                cb(run_id)
            except Exception as exc:
                print("nesicode: on_finish hook failed:", exc)
        row = self.db.one("SELECT thread_id, kind FROM runs WHERE id=?", (run_id,))
        if row and row["kind"] == "chat" and row["thread_id"]:
            self.advance(row["thread_id"])

    # ------------------------------------------------------------ titles
    TITLE_AT = (1, 3)  # retitle after the 1st and 3rd answered turn (more context by then)

    def _auto_title(self, run_id: int) -> None:
        run = self.db.one("SELECT id, thread_id, kind, status, bot_id, project_id FROM runs WHERE id=?", (run_id,))
        if not run or run["kind"] != "chat" or run["status"] not in ("done", "max_steps") or run["bot_id"] or not run["thread_id"]:
            return
        thread = self.db.one("SELECT * FROM threads WHERE id=?", (run["thread_id"],))
        if not thread or thread.get("title_locked") or thread["title"].startswith("🤖"):
            return
        n = self.db.one("SELECT COUNT(*) AS n FROM runs WHERE thread_id=? AND kind='chat' AND status IN ('done', 'max_steps')",
                        (thread["id"],))["n"]
        if n not in self.TITLE_AT:
            return
        threading.Thread(target=self._make_title, args=(thread["id"], run["project_id"]), daemon=True, name="nesi-title").start()

    def _make_title(self, thread_id: int, project_id: int) -> None:
        from ..llm import ModelOptions, get_provider

        try:
            project = self.db.one("SELECT * FROM projects WHERE id=?", (project_id,))
            spec = project["executor_model"]
            if spec.startswith("mock/"):
                return
            turns = self.db.all("SELECT task, final_text FROM runs WHERE thread_id=? AND kind='chat' AND status IN ('done', 'max_steps') "
                                "ORDER BY created_at, id LIMIT 4", (thread_id,))
            convo = "\n\n".join(f"User: {t['task'][:1500]}\nAssistant: {(t['final_text'] or '')[:1500]}" for t in turns)
            provider, model = get_provider(spec, user_keys(self.db, project["user_id"]))
            text, _ = provider.complete(
                model, "You name conversations.",
                "Write a short title (3 to 7 words) for this conversation, in the same language the user writes in. "
                "It must say what the conversation is about (the topic or goal), not quote the first message. "
                "Reply with the title only: no quotes, no trailing period, no emoji.\n\n" + convo,
                ModelOptions(effort="low", role="titler", max_tokens=60),
            )
            title = re.sub(r"\s+", " ", text.strip().splitlines()[0] if text.strip() else "").strip(" \"'«»*#.")[:80]
            if len(title) >= 3:
                # the user may have renamed it meanwhile: only replace an unlocked title
                self.db.execute("UPDATE threads SET title=? WHERE id=? AND title_locked=0", (title, thread_id))
        except Exception as exc:  # a title is never worth an error
            print("nesicode: auto-title failed:", exc)

    # ------------------------------------------------------------ per-conversation queue
    def enqueue_chat(self, run_id: int, thread_id: int) -> str:
        """Start the message now, or leave it 'waiting' behind the reply in progress in this conversation."""
        with self._lock:
            busy = self.db.one("SELECT id FROM runs WHERE thread_id=? AND id<>? AND status IN ('queued', 'running')",
                               (thread_id, run_id))
            status = "waiting" if busy else "queued"
            self.db.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))
        if status == "queued":
            self.submit_chat(run_id)
        return status

    def advance(self, thread_id: int) -> None:
        """A reply finished: start the oldest queued message of the conversation, if any."""
        with self._lock:
            if self.db.one("SELECT id FROM runs WHERE thread_id=? AND status IN ('queued', 'running')", (thread_id,)):
                return
            nxt = self.db.one("SELECT id FROM runs WHERE thread_id=? AND status='waiting' ORDER BY created_at, id LIMIT 1",
                              (thread_id,))
            if not nxt:
                return
            self.db.execute("UPDATE runs SET status='queued' WHERE id=?", (nxt["id"],))
        self.submit_chat(nxt["id"])

    def send_now(self, run_id: int) -> bool:
        """Jump the queue: stop the reply in progress and start this queued message next."""
        with self._lock:
            run = self.db.one("SELECT * FROM runs WHERE id=?", (run_id,))
            if not run or run["status"] != "waiting":
                return False
            cur = self.db.one("SELECT id, created_at FROM runs WHERE thread_id=? AND status IN ('queued', 'running') "
                              "ORDER BY id DESC LIMIT 1", (run["thread_id"],))
            first = self.db.one("SELECT MIN(created_at) AS t FROM runs WHERE thread_id=? AND status='waiting'",
                                (run["thread_id"],))["t"]
            base = cur["created_at"] if cur else first - 1
            # right after the current reply, ahead of everything else in the queue (and shown in that order)
            self.db.execute("UPDATE runs SET created_at=? WHERE id=?", (min(base + 1e-3, first - 1e-3), run_id))
        if cur:
            if not self.cancel(cur["id"]):  # queued but not started yet: just drop it back into the queue
                self.db.execute("UPDATE runs SET status='waiting' WHERE id=? AND status='queued'", (cur["id"],))
                self.advance(run["thread_id"])
        else:
            self.advance(run["thread_id"])
        return True

    def resume_waiting(self) -> None:
        """After a restart: queued messages carry on."""
        for row in self.db.all("SELECT DISTINCT thread_id FROM runs WHERE status='waiting' AND thread_id IS NOT NULL"):
            self.advance(row["thread_id"])

    # ------------------------------------------------------------------ chat
    def submit_chat(self, run_id: int) -> None:
        self.cancels[run_id] = threading.Event()
        self.pool.submit(self._safe, self._chat, run_id)

    def submit_harnessing(self, run_id: int) -> None:
        self.cancels[run_id] = threading.Event()
        self.pool.submit(self._safe, self._harnessing, run_id)

    def _safe(self, fn, run_id: int) -> None:
        try:
            fn(run_id)
        except Exception as exc:  # never lose a run silently
            self.emitter(run_id)({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
            self._finish(run_id, "error", error=traceback.format_exc()[-4000:])

    def _chat(self, run_id: int) -> None:
        db = self.db
        run = db.one("SELECT * FROM runs WHERE id=?", (run_id,))
        project = db.one("SELECT * FROM projects WHERE id=?", (run["project_id"],))
        cancel = self.cancels[run_id]
        emit = self.emitter(run_id)
        history = [
            (r["task"], r["final_text"])
            for r in db.all(
                "SELECT task, final_text FROM runs WHERE thread_id=? AND id<? AND kind='chat' "
                "AND status IN ('done', 'max_steps') ORDER BY id",
                (run["thread_id"], run_id),
            )
        ][-20:]
        thread = db.one("SELECT * FROM threads WHERE id=?", (run["thread_id"],))
        keys = user_keys(db, project["user_id"])
        vision = project.get("vision_model") or settings.vision_model
        image = project.get("image_model") or settings.image_model
        ws = Workspace(thread_ws(project, thread))
        from .limits import learn_limit, model_limits

        lim = model_limits(db, project["user_id"], project["executor_model"])
        # A shared workspace is serialised per project; a cage is already one-run-at-a-time.
        lock = threading.Lock() if thread["isolated"] else self._project_lock(project["id"])
        with lock:
            db.execute("UPDATE runs SET status='running', harness_version=? WHERE id=?", (project["harness_version"], run_id))
            task = run["task"]
            _write_thread_history(db, ws, run, f".nesicode/chats/{thread['id']}")
            attachments = loads(run["attachments"], [])
            if attachments:
                task += "\n\n" + attachments_note(ws, attachments, vision, keys, emit)
            harness = current_harness(db, project)
            bot = db.one("SELECT * FROM bots WHERE id=?", (run["bot_id"],)) if run.get("bot_id") else None
            info: dict = {}
            recent = " ".join([run["task"]] + [h[0] for h in history[-3:]] + ([bot["instructions"]] if bot else []))
            tools = self.connector_tools(project["user_id"], emit, context=recent, info=info)
            tools.update(self.chat_tools(project, emit))
            note = ""
            if info["on_request"]:
                note = ("# Connectors available on request\nThe user has these connectors but wants them used ONLY when they "
                        "explicitly ask for them by name: " + ", ".join(sorted(set(info["on_request"]))) + ". Do the task with "
                        "your other tools. If one of them would clearly help, say so at the end and suggest the user ask for it "
                        "(e.g. 'use Burp to …'); never pretend you used it.")
            model, steps, system_base = project["executor_model"], project["max_steps"], None
            if bot and self.bots:
                tools.update(self.bots.agent_tools(bot, run_id))
                model = bot["model"] or model
                steps = min(steps, bot["max_steps"])
                system_base = harness.get("system_prompt.md", "").strip() + "\n\n" + self.bots.system_section(bot, run.get("trigger") or "")
            agent = Agent(
                model=model,
                harness=harness,
                workspace=ws,
                emit=emit,
                user_keys=keys,
                max_steps=steps,
                cancel=cancel,
                approve=self._approver(run_id, thread, project, emit, cancel, bot, attended=run.get("trigger") == "chat",
                                       ask_tools=info["ask"]),
                system_note=note,
                vision_model=vision,
                image_model=image,
                system_base=system_base,
                extra_tools=tools,
                context_limit=lim["context"],
                output_limit=lim["output"],
                on_limit=lambda n: learn_limit(db, project["user_id"], project["executor_model"], n),
                archive=f".nesicode/chats/{thread['id']}",
            )
            res = agent.run(task, history)
        try:
            if not bot:  # unattended bot runs never trigger the auto-improver
                self._auto_improve(run, project, thread, res, emit)
        except Exception as exc:  # never break the chat because of the auto-improver
            emit({"type": "auto_harness_error", "text": f"{type(exc).__name__}: {exc}"})
        self._finish(
            run_id, res.status, final_text=res.final_text, error=res.error,
            result=json.dumps({"usage": res.usage.to_dict(), "steps": res.steps}),
        )
        db.execute("UPDATE threads SET updated_at=? WHERE id=?", (time.time(), run["thread_id"]))

    # ----------------------------------------------------------- connectors
    def chat_tools(self, project: dict, emit) -> dict:
        """Tools every conversation gets from the app itself (not from connectors)."""
        if not self.bots:
            return {}
        from ..llm import ToolSpec

        bots = self.bots

        def create_bot(args: dict, agent) -> str:
            from ..agent.workspace import ToolError

            data = {k: args.get(k) for k in ("name", "description", "instructions", "policy", "notify", "max_steps", "daily_cap")}
            data["project_id"] = project["id"]
            data["triggers"] = args.get("triggers") or []
            try:
                bot = bots.save(project["user_id"], data)
            except (ValueError, ToolError) as exc:
                raise ToolError(f"no se pudo crear el bot: {exc}") from exc
            agent.emit({"type": "bot_created", "bot_id": bot["id"], "name": bot["name"]})
            nxt = self.db.one("SELECT MIN(next_at) AS t FROM bot_triggers WHERE bot_id=? AND next_at IS NOT NULL", (bot["id"],))["t"]
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(nxt)) if nxt else "solo a mano / Telegram / webhook"
            return f"Bot «{bot['name']}» creado (id {bot['id']}). Próxima ejecución: {when}. Se gestiona en la página Bots."

        trig = {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["cron", "interval", "url"]},
            "spec": {"type": "string", "description": "cron: 5 fields 'min hour dom mon dow' (e.g. '0 8 * * 1-5'); "
                                                       "interval: minutes or 2h/1d (min 5 min); url: the page to watch"},
            "tz": {"type": "string", "description": "IANA time zone for cron, e.g. Europe/Madrid (default: this computer's)"},
            "prompt": {"type": "string", "description": "what to do on each firing (optional)"},
            "config": {"type": "object", "description": "url: {every: minutes between checks}"}}, "required": ["kind", "spec"]}
        spec = ToolSpec(
            "create_bot",
            "Create a Nesicode bot: a standing instruction the agent runs on its own on a schedule (cron/interval), when "
            "a web page changes (url), from Telegram (/bot <name>) or by hand. Use it when the user asks for something "
            "recurring or automatic ('every morning…', 'tell me when…'). The user approves it before it is created. "
            "Write `instructions` as a complete, self-contained brief (what to check, how to report, what counts as news). "
            "policy: ask (default: asks the user before writing files, running commands or using connectors), readonly, auto.",
            {"type": "object", "properties": {
                "name": {"type": "string"}, "description": {"type": "string"}, "instructions": {"type": "string"},
                "triggers": {"type": "array", "items": trig},
                "policy": {"type": "string", "enum": ["ask", "readonly", "auto"]},
                "notify": {"type": "string", "enum": ["telegram", "none"]},
                "max_steps": {"type": "integer"}, "daily_cap": {"type": "integer"}},
             "required": ["name", "instructions"]},
        )
        return {"create_bot": (spec, create_bot)}

    def connector_tools(self, user_id: int, emit, context: str | None = None, info: dict | None = None) -> dict:
        """Tools of the user's enabled MCP connectors, as agent tools named mcp__<connector>__<tool>.

        `context` (the user's recent words): connectors in 'mention' mode are only offered when named there;
        the others are listed in info['on_request'] so the agent can suggest them. info['ask'] collects the
        tool names that need the user's OK on every use. context=None → everything (connector tests, etc.)."""
        from ..connectors import catalog_entry, manager, mentioned, tool_name
        from ..llm import ToolSpec
        from .connectors_store import connector_configs

        info = info if info is not None else {}
        info.setdefault("on_request", [])
        info.setdefault("ask", set())
        out: dict = {}
        browser = browser_tools(user_id)
        usage = (loads((self.db.one("SELECT settings FROM users WHERE id=?", (user_id,)) or {}).get("settings"), {}) or {}).get("browser_usage", "mention")
        if browser:
            if context is None or usage != "mention" or mentioned(context, template="browser"):
                out.update(browser)
                if usage == "ask":
                    info["ask"].update(browser)
            else:
                info["on_request"].append("the user's own Chrome (Nesicode extension)")
        own_chrome = bool(browser)
        for row, cfg in connector_configs(self.db, user_id, enabled_only=True):
            if own_chrome and row["template"] == "chrome":
                continue  # the user's own Chrome is connected: don't offer a second, separate Chrome
            mode = row.get("usage") or "mention"
            if context is not None and mode == "mention" and not mentioned(context, name=row["name"], label=row["label"],
                                                                          template=row["template"]):
                info["on_request"].append(row["label"])
                continue
            before = set(out)
            if cfg.kind == "builtin":
                out.update(builtin_agent_tools(row["name"], row["label"], cfg.command, cfg))
                if mode == "ask":
                    info["ask"].update(set(out) - before)
                continue
            try:
                tools = manager().tools(user_id, cfg, timeout=90)
            except Exception as exc:
                emit({"type": "connector_error", "connector": row["label"], "text": str(exc)[:300]})
                continue
            hint = (catalog_entry(row["template"]) or {}).get("hint", "")
            for tool in tools:
                name = tool_name(row["name"], tool.name)
                desc = f"[{row['label']}] {tool.description}".strip()[:1600] + (f" ({hint})" if hint else "")
                out[name] = (ToolSpec(name, desc, tool.schema), _mcp_caller(user_id, cfg, tool.name))
            if mode == "ask":
                info["ask"].update(set(out) - before)
        return out

    # ------------------------------------------------------------ approvals
    def exec_mode(self, thread: dict, user_id: int) -> str:
        if thread.get("exec_mode") in EXEC_MODES:
            return thread["exec_mode"]
        prefs = loads((self.db.one("SELECT settings FROM users WHERE id=?", (user_id,)) or {}).get("settings"), {})
        return prefs.get("exec_mode") if prefs.get("exec_mode") in EXEC_MODES else "auto"

    def _approver(self, run_id: int, thread: dict, project: dict, emit, cancel: threading.Event, bot: dict | None = None,
                  attended: bool = False, ask_tools: set | None = None):
        def approve(call) -> tuple[bool, str]:
            if bot:
                from .bots import bot_denies

                # the user talking to the bot in its conversation is there to approve: read-only binds unattended runs
                if reason := bot_denies("ask" if attended else bot["policy"], call.name):
                    return False, reason
            t = self.db.one("SELECT * FROM threads WHERE id=?", (thread["id"],)) or thread
            mode = self.exec_mode(t, project["user_id"])  # read live: the user may switch modes mid-run
            remembered = call.name in loads(t.get("allowed_tools"), []) and not call.name.endswith(ALWAYS_ASK)
            if (not needs_approval(mode, call.name) and call.name not in (ask_tools or ())) or remembered:
                return True, ""
            pending = Approval(call.id)
            self.approvals[run_id] = pending
            each_time = call.name.endswith(ALWAYS_ASK)
            emit({"type": "approval_request", "id": call.id, "name": call.name, "args": call.arguments, "mode": mode,
                  "always": not each_time})
            deadline = time.time() + APPROVAL_TIMEOUT
            while not pending.event.wait(0.5):
                if cancel.is_set():
                    pending.decision = "cancelled"
                    break
                if time.time() > deadline:
                    pending.decision = "timeout"
                    break
            self.approvals.pop(run_id, None)
            decision = pending.decision or "deny"
            emit({"type": "approval_result", "id": call.id, "decision": decision})
            if decision == "always" and each_time:
                decision = "allow"  # approved this once; the next one asks again
            if decision == "always":
                allowed = sorted(set(loads(t.get("allowed_tools"), [])) | {call.name})
                self.db.execute("UPDATE threads SET allowed_tools=? WHERE id=?", (json.dumps(allowed), thread["id"]))
            return decision in ("allow", "always"), {"deny": "denied", "timeout": "no answer in 30 min",
                                                      "cancelled": "run cancelled"}.get(decision, decision)

        return approve

    def decide(self, run_id: int, call_id: str, decision: str) -> bool:
        pending = self.approvals.get(run_id)
        if not pending or pending.call_id != call_id:
            return False
        pending.decision = decision
        pending.event.set()
        return True

    # ------------------------------------------------- improve from a chat
    def create_harnessing(self, project: dict, task: str, params: dict) -> int:
        rid = self.db.insert(
            "INSERT INTO runs (project_id, kind, task, status, params, created_at) VALUES (?, 'harness', ?, 'queued', ?, ?)",
            (project["id"], task, json.dumps(params), time.time()),
        )
        self.submit_harnessing(rid)
        return rid

    def improve_from_chat(self, run: dict, project: dict, *, auto: bool, reason: str = "") -> int:
        """Option A/B: re-run a chat request as a Harness² task (same text, same attachments)."""
        task = run["task"]
        atts = loads(run.get("attachments"), [])
        if atts:
            task += "\n\nAttached files (paths relative to the workspace):\n" + "\n".join(
                f"- `{a['path']}` ({a['kind']})" for a in atts
            )
        params = {
            "k": 1, "mode": "parallel", "apply_changes": False, "adopt_harness": True,
            "source": {"thread_id": run["thread_id"], "run_id": run["id"], "auto": auto, "reason": reason},
        }
        return self.create_harnessing(project, task, params)

    def _auto_improve(self, run: dict, project: dict, thread: dict, res, emit) -> None:
        """Option B: launch harnessing on its own when a chat answer went badly (opt-in, capped)."""
        prefs = loads((self.db.one("SELECT settings FROM users WHERE id=?", (project["user_id"],)) or {}).get("settings"), {})
        if not prefs.get("auto_harness"):
            return
        reason = needs_improvement(res, project["max_steps"], prefs)
        if not reason:
            return
        if self.db.one("SELECT id FROM runs WHERE project_id=? AND kind='harness' AND status IN ('queued', 'running')", (project["id"],)):
            emit({"type": "auto_harness_skipped", "reason": reason, "why": "ya hay un harnessing en curso en este proyecto"})
            return
        cap = int(prefs.get("auto_harness_daily", 3))
        today = self.db.one(
            "SELECT COUNT(*) AS n FROM runs r JOIN projects p ON p.id=r.project_id WHERE p.user_id=? AND r.kind='harness' "
            "AND r.created_at>? AND r.params LIKE '%\"auto\": true%'",
            (project["user_id"], time.time() - 86400),
        )["n"]
        if today >= cap:
            emit({"type": "auto_harness_skipped", "reason": reason, "why": f"límite diario alcanzado ({cap})"})
            return
        rid = self.improve_from_chat(run, project, auto=True, reason=reason)
        emit({"type": "auto_harness", "run_id": rid, "reason": reason})

    # ----------------------------------------------------------- harnessing
    def _harnessing(self, run_id: int) -> None:
        db = self.db
        run = db.one("SELECT * FROM runs WHERE id=?", (run_id,))
        project = db.one("SELECT * FROM projects WHERE id=?", (run["project_id"],))
        params = loads(run["params"], {})
        cancel = self.cancels[run_id]
        emit = self.emitter(run_id)
        db.execute("UPDATE runs SET status='running', harness_version=? WHERE id=?", (project["harness_version"], run_id))
        lessons = db.all(
            "SELECT cue, lesson, evidence, confidence FROM lessons WHERE user_id=? AND domain=? "
            "ORDER BY created_at DESC LIMIT 12",
            (project["user_id"], project["domain"]),
        )
        from .limits import model_limits

        _lim = model_limits(db, project["user_id"], project["executor_model"])
        cfg = HarnessingConfig(
            task=run["task"],
            executor_model=project["executor_model"],
            improver_model=project["improver_model"],
            k=int(params.get("k", 1)),
            mode=params.get("mode", "parallel"),
            domain=project["domain"],
            check_command=project["check_command"],
            max_steps=project["max_steps"],
            rollout_width=settings.rollout_width,
            context_limit=_lim["context"],
            output_limit=_lim["output"],
            improver_context=model_limits(db, project["user_id"], project["improver_model"])["context"],
        )
        rd = run_dir(project, run_id)
        extra = []
        src = params.get("source") or {}
        if src.get("thread_id"):
            t = db.one("SELECT * FROM threads WHERE id=? AND project_id=?", (src["thread_id"], project["id"]))
            if t:
                extra.append((thread_ws(project, t) / ATTACHMENTS_DIR, ATTACHMENTS_DIR))
        hr = HarnessingRun(
            cfg, base_harness=current_harness(db, project), source_ws=project_ws(project), run_dir=rd,
            lessons=lessons, emit=emit, user_keys=user_keys(db, project["user_id"]), cancel=cancel, seed=run_id,
            extra_files=extra,
        )
        try:
            result = hr.run()
        except RuntimeError as exc:
            if str(exc) == "cancelled":
                emit({"type": "cancelled"})
                self._finish(run_id, "cancelled")
                return
            raise
        selected = result["selected"]
        result["applied"] = None
        if params.get("apply_changes"):
            with self._project_lock(project["id"]):
                result["applied"] = hr.apply_selected(selected, project_ws(project))
            emit({"type": "applied", "changes": result["applied"]})
        result["adopted_version"] = None
        if result["harness_changed"] and params.get("adopt_harness", True):
            v = save_harness_version(
                db, project["id"], result["adopted_harness"], "harnessing",
                f"Harness² run #{run_id}: adoptado `{selected}`", run_id,
            )
            result["adopted_version"] = v
            emit({"type": "adopted", "version": v, "label": selected})
        exp = (result.get("reflection") or {}).get("experience")
        if exp:
            db.insert(
                "INSERT INTO lessons (user_id, project_id, domain, cue, lesson, evidence, confidence, run_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (project["user_id"], project["id"], project["domain"], exp["cue"], exp["lesson"],
                 exp["evidence"], exp.get("confidence", ""), run_id, time.time()),
            )
            emit({"type": "lesson", **exp})
        # Keep only the selected rollout's workspace (for a later "apply"), drop the rest.
        for label, ro in hr.rollouts.items():
            if label != selected and ro.ws_path:
                shutil.rmtree(ro.ws_path, ignore_errors=True)
        emit({"type": "done", "status": "done", "usage": result["usage"]})
        sel = result["rollouts"].get(selected, {})
        self._finish(run_id, "done", result=json.dumps(result, ensure_ascii=False), final_text=sel.get("final_text", ""))

    def apply_later(self, run_id: int, project: dict) -> dict:
        run = self.db.one("SELECT result FROM runs WHERE id=?", (run_id,))
        result = loads(run["result"], {})
        selected = result.get("selected")
        rd = run_dir(project, run_id)
        after = rd / "rollouts" / str(selected) / "ws"
        if not selected or not after.exists():
            raise FileNotFoundError("the selected rollout's workspace is no longer available")
        with self._project_lock(project["id"]):
            changes = apply_changes(rd / "source", after, project_ws(project))
        result["applied"] = changes
        self.db.execute("UPDATE runs SET result=? WHERE id=?", (json.dumps(result, ensure_ascii=False), run_id))
        return changes
