"""Background execution of chat turns and harnessing runs, with a live event bus."""

from __future__ import annotations

import atexit
import base64
import functools
import json
import os
import re
import secrets
import shutil
import threading
import time
import traceback
import weakref
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from ..agent import Agent, Workspace, copy_workspace
from ..agent.workspace import ATTACHMENTS_DIR, ToolError
from ..config import settings
from ..harness import normalize, validate
from ..harnessing import HarnessingConfig, HarnessingRun, apply_rollout
from ..harnessing.engine import remove_tree
from .auth import user_keys
from .db import DB, loads

FINAL = {"done", "error", "cancelled", "interrupted", "max_steps", "refusal"}


class ProjectBusy(RuntimeError):
    """The project's folder is in use by another conversation (a shared one holds its lock for its whole run): what
    serves an HTTP request answers this at once (the API turns it into a 409) instead of hanging until it ends."""


BUSY = "El proyecto está ocupado por otra conversación; inténtalo cuando termine."
LOCK_TRY = 2.0  # seconds an «Aplicar» waits for the project's lock before answering ProjectBusy


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
    """Earlier turns of this conversation (in queue order, only the ones that ran), complete with their steps, so
    `search_history` can find anything that only survives as a summary in the model's context."""
    from ..agent import trajectory_text

    rows = _earlier_turns(db, run, limit=40)
    if not rows:
        return
    parts = ["# Historial de esta conversación (turnos anteriores)"]
    for r in rows:
        events = _turn_events(db, r["id"], 5_000)
        parts.append(f"## Turno #{r['id']} ({r['status']})\n### Usuario\n{r['task']}\n### Pasos\n"
                     f"{trajectory_text(events, 200_000, obs_chars=6_000)}\n### Respuesta final\n{r['final_text']}")
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
    # The tab this conversation used last: commands without tab_id act on it — except chrome_open, which opens a NEW
    # tab unless told otherwise (it must never navigate away the user's tab) and chrome_tabs.
    last = {"tab": None}

    def make(name: str):
        def call(args: dict, agent) -> str:
            a = dict(args or {})
            injected = a.get("tab_id") is None and last["tab"] is not None and name not in ("chrome_tabs", "chrome_open")
            if injected:
                a["tab_id"] = last["tab"]
            cancel = getattr(agent, "cancel", None)
            try:  # Stop abandons the wait and cancels the command (the outer wait also covers a call stuck elsewhere)
                res = _interruptible(cancel, lambda: hub.call(user_id, name.removeprefix("chrome_"), a, timeout=60,
                                                              cancel=cancel))
            except _Stopped:
                raise
            except Exception as exc:
                gone = _TAB_GONE.search(str(exc)) and a.get("tab_id") is not None and a.get("tab_id") == last["tab"]
                if gone:  # the tab used last was closed or taken out of the group: never aim at it again
                    last["tab"] = None
                if gone and injected:
                    raise ToolError(f"{exc} (it was the tab used last, id {a['tab_id']}, which is no longer available: "
                                    "call chrome_tabs to see the open tabs, or open a page with chrome_open)") from exc
                raise ToolError(str(exc)) from exc
            if name == "chrome_close":
                if res.get("closed") is not None and res.get("closed") == last["tab"]:
                    last["tab"] = None  # closed: the next command must not aim at it
            elif res.get("tab_id") is not None:
                last["tab"] = res["tab_id"]
            if name == "chrome_screenshot" and res.get("image"):
                data = base64.b64decode(res.pop("image").split(",", 1)[-1])
                rel = f".chrome/screenshot-{int(time.time() * 1000)}.jpg"
                agent.ws.write_bytes(agent.ws.resolve(rel), data)  # never through a link, never into a FIFO
                agent.emit({"type": "image", "path": rel, "caption": res.get("title", "Chrome")})
                res["saved_to"] = rel
            return _connector_output(agent, name, _browser_json(res))

        return call

    return {name: (spec, make(name)) for name, spec in TOOLS.items()}


def builtin_agent_tools(connector: str, label: str, which: str, cfg=None, alive=None) -> dict:
    """Built-in connectors (Office, Burp REST) run in-process on the conversation's own workspace.
    `alive()` -> '' while the connector may be used, else why not (the user turned it off or removed it meanwhile)."""
    from ..connectors import builtin_module, tool_name
    from ..llm import ToolSpec

    try:
        mod = builtin_module(which)
    except RuntimeError:
        return {}
    TOOLS = mod.TOOLS
    takes_cfg = which != "office"
    remote = which != "office"  # network calls (VirusTotal polls an analysis for minutes): Stop must not wait for them

    def make(fn):
        def call(args: dict, agent) -> str:
            if alive is not None and (why := alive()):
                raise ToolError(f"{label}: {why}")
            go = (lambda: fn(args or {}, agent.ws, cfg)) if takes_cfg else (lambda: fn(args or {}, agent.ws))
            text, written = _interruptible(getattr(agent, "cancel", None), go) if remote else go()
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


FINISHED = ("done", "max_steps")
STOPPED = ("cancelled", "error", "interrupted", "refusal")
HISTORY_TURNS = 200  # earlier turns handed to the agent (it keeps the first one and the newest that fit)


def _earlier_turns(db, run: dict, limit: int = HISTORY_TURNS) -> list[dict]:
    """This conversation's chat turns queued before `run`, in queue order, that ended — without the messages the user
    removed from the queue (cancelled before they ever started: no events). Over `limit`: the first + the newest."""
    statuses = FINISHED + STOPPED
    sql = ("SELECT r.id, r.task, r.final_text, r.status FROM runs r WHERE r.thread_id=? AND r.id<>? AND r.kind='chat' "
           "AND (r.created_at<? OR (r.created_at=? AND r.id<?)) "
           f"AND r.status IN ({','.join('?' * len(statuses))}) "
           "AND (r.status<>'cancelled' OR EXISTS(SELECT 1 FROM events e WHERE e.run_id=r.id)) ")
    args = (run["thread_id"], run["id"], run["created_at"], run["created_at"], run["id"], *statuses)
    newest = db.all(sql + "ORDER BY r.created_at DESC, r.id DESC LIMIT ?", (*args, limit))[::-1]  # oldest first
    if len(newest) < limit:
        return newest
    first = db.all(sql + "ORDER BY r.created_at, r.id LIMIT 1", args)
    if first and first[0]["id"] != newest[0]["id"]:
        return first + newest[1:]  # the conversation's first turn always stays (it usually holds the main request)
    return newest


def _turn_events(db, run_id: int, limit: int = 4_000) -> list[dict]:
    """A turn's events: all of them, or (a very long turn) its first 500 and its last ones."""
    n = db.one("SELECT COUNT(*) AS n FROM events WHERE run_id=?", (run_id,))["n"]
    if n <= limit:
        return [e for _, e in db.events_after(run_id, 0, limit=limit)]
    head = [e for _, e in db.events_after(run_id, 0, limit=500)]
    tail = db.all("SELECT data FROM events WHERE run_id=? ORDER BY seq DESC LIMIT ?", (run_id, limit - 500))
    return head + [json.loads(r["data"]) for r in reversed(tail)]


def conversation_history(db, run: dict) -> tuple[list[tuple[str, str]], dict | None]:
    """Earlier turns of this conversation for the next message, oldest first.

    Finished turns bring their answer (+ a digest of their key steps for the last few that used tools); stopped turns
    (Stop, error, app closed) bring a digest of what they did, so "continue" picks up the work; a turn that compacted
    brings its last checkpoint summary; the last turn that DID WORK brings its full step trail (an empty stopped turn
    after it never takes that slot). Messages removed from the queue never appear. Returns (history, the last turn
    that did work if it was stopped)."""
    from ..agent.context import redact, run_digest
    from ..agent.loop import trajectory_text

    rows = _earlier_turns(db, run)
    detail = max(0, len(rows) - 20)  # the newest 20 turns bring their steps; older ones only their answer
    events = {r["id"]: _turn_events(db, r["id"]) for r in rows[detail:]}
    steps = {rid: sum(1 for e in ev if e.get("type") == "tool_call" and not e.get("depth")) for rid, ev in events.items()}
    worked = [i for i in range(detail, len(rows)) if steps[rows[i]["id"]] or (rows[i]["final_text"] or "").strip()]
    last_work = worked[-1] if worked else None
    history: list[tuple[str, str]] = []
    interrupted = None
    for i, r in enumerate(rows):
        answer = (r["final_text"] or "").strip()
        if i < detail:
            history.append((r["task"], answer or f"[this turn ended as {r['status']} without an answer]"))
            continue
        ev, n = events[r["id"]], steps[r["id"]]
        if r["status"] in STOPPED:
            if not n and not answer:
                answer = f"[this turn was {r['status']} before doing any work]"
            else:
                digest = run_digest(ev)
                partial = f"\n[Partial answer] {answer[:800]}" if answer and answer[:200] not in digest else ""
                answer = (f"[This turn was {r['status']} at step {n} and did NOT finish. What it had done so far:]\n"
                          + digest + partial)
            if i == last_work:
                interrupted = {"id": r["id"], "status": r["status"], "steps": n}
        elif n and len(rows) - i <= 3:
            answer += "\n\n[Key steps of this turn]\n" + run_digest(ev, 2_500)
        checkpoint = next((e.get("summary") for e in reversed(ev) if e.get("type") == "compacted" and not e.get("depth")
                           and e.get("summary")), "")
        if checkpoint and (i == last_work or len(rows) - i <= 3):
            answer += ("\n\n[This turn compacted its context; its last checkpoint summary:]\n"
                       + redact(checkpoint[:6_000]))
        if i == last_work and n:  # the last turn that worked travels with its full trail (bounded)
            answer += "\n\n[PREVIOUS TURN — STEPS]\n" + redact(trajectory_text(ev, 40_000, obs_chars=4_000))
        history.append((r["task"], answer))
    if interrupted and last_work is not None and last_work < len(rows) - 1:
        interrupted["later_empty"] = len(rows) - 1 - last_work
    return history, interrupted


def connector_context(db, run: dict, history: list[tuple[str, str]], bot: dict | None = None) -> str:
    """Text that decides which 'only when I ask' connectors this message gets: the user's message, the last
    turns (both sides) and the tools already used in this conversation's recent turns (the ones queued before it)."""
    used = {e.get("name", "") for r in _earlier_turns(db, run)[-5:] for e in _turn_events(db, r["id"])
            if e.get("type") == "tool_call"}
    return " ".join([run["task"]] + [h[0] + " " + h[1][:2000] for h in history[-3:]] + sorted(used)
                    + ([bot["instructions"]] if bot else []))


def continuity_note(thread_id: int, interrupted: dict | None) -> str:
    """System-prompt note when the last turn that did work was stopped before finishing ('' otherwise: the agent
    itself describes the earlier turns when there are any)."""
    if not interrupted:
        return ""
    hist = f".nesicode/chats/{thread_id}/context/history.md"
    note = ("# The previous turn did not finish\n"
            f"The last turn that did work was {interrupted['status']} at step {interrupted['steps']} without finishing. "
            "Its digest and steps are in the earlier turns: if the user asks to continue, resume right after its last "
            "step — do not start over and do not redo what it already did. If something you need is not there, call "
            f"`search_history` (its full steps and outputs are archived in `{hist}`) before exploring files.")
    if interrupted.get("later_empty"):
        note += " The user's later message(s) after it were stopped before doing any work."
    return note + (" Values shown as <redacted> are hidden in that recap only: find the exact value with search_history "
                   "and read that line with read_file — never use a placeholder or a guessed value.")


# Argument names that hold a local file/folder path, judged by their LAST word (camelCase, snake_case and kebab-case
# alike): filePath, output_path, outputDir, paths, file_name… — but never fileId, folderId, direction, profile,
# output_format, redirect… (those are not paths: rewriting them broke real servers).
_PATH_WORDS = {"path", "paths", "file", "files", "filename", "filenames", "dir", "dirs", "directory", "directories",
               "folder", "folders",
               # the same written as one lowercase word (excel-mcp-server's `filepath`…) — never xpath, jsonpath, profile
               "filepath", "filepaths", "pathname", "dirpath", "dirname", "folderpath", "outfile", "infile", "logfile",
               "outdir", "indir", "workdir", "basedir", "rootdir", "outpath", "savepath"}
# Words that hold a path only when the value is a file name ("Informe Q3.docx", "out.png"), not an id, a format
# ("json") or a text.
_DOC_WORDS = {"document", "workbook", "presentation", "spreadsheet", "docx", "xlsx", "pptx", "output"}
_DOC_FILE = re.compile(r"^[^/\\\n]+(?:[/\\][^/\\\n]+)*\.(?:docx?|docm|xlsx?|xlsm|pptx?|odt|ods|odp|pdf|csv|txt|md|json|"
                       r"html?|xml|ya?ml|png|jpe?g|gif|webp|svg|zip|har|log)$", re.I)


def _path_kind(key: str) -> str:
    """'path' when the argument `key` holds a file path, 'doc' when it does if the value is a file name, else ''."""
    words = [w.lower() for w in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", str(key))]
    if not words:
        return ""
    if words[-1] in _PATH_WORDS or (words[-1] == "name" and len(words) > 1 and words[-2] == "file"):
        return "path"
    return "doc" if words[-1] in _DOC_WORDS else ""


def absolutize_paths(args: dict, ws) -> dict:
    """Local servers don't run in the conversation's folder: turn relative file paths the agent passes into absolute
    paths inside its cage, so files land (and are read) where they belong. Only arguments whose name says they are a
    path are touched (also each item of a list such as `paths`); see _mcp_caller for which servers get this."""

    def fix(v):
        if not isinstance(v, str):
            return v
        s = v.strip()
        if not s or s.startswith(("/", "~", "\\")) or "://" in s or re.match(r"^[A-Za-z]:[\\/]", s):
            return v
        try:
            return str(ws.resolve(s))
        except Exception:
            return v  # e.g. "../.." escaping the cage: leave it, the server decides

    out = dict(args or {})
    for key, value in out.items():
        kind = _path_kind(key)
        if kind == "doc" and not (isinstance(value, str) and _DOC_FILE.match(value.strip())):
            continue
        if kind and isinstance(value, str):
            out[key] = fix(value)
        elif kind == "path" and isinstance(value, list):
            out[key] = [fix(v) for v in value]
    return out


def _mcp_caller(user_id: int, cfg, tool: str, alive=None):
    """An MCP tool as an agent tool. `alive()` -> '' while the connector may be used, else why not."""

    def call(args: dict, agent) -> str:
        from ..connectors import manager

        if alive is not None and (why := alive()):
            raise ToolError(f"{cfg.name}: {why}")
        if cfg.kind == "stdio":
            # a local server shares this computer's files; a remote one (GitHub…) gets the arguments as written:
            # its `path` is a path in a repository, and a local absolute path would only leak the user's home folder
            args = absolutize_paths(args, agent.ws)
        cancel = getattr(agent, "cancel", None)
        try:  # Stop cancels the MCP request itself, not only our wait for it
            res = _interruptible(cancel, lambda: manager().call(user_id, cfg, tool, args, cancel=cancel))
        except _Stopped:
            raise
        except Exception as exc:
            raise ToolError(f"{cfg.name}: {exc}") from exc
        text = res.text or "(no text output)"
        for i, (data, mime) in enumerate(res.images[:4]):  # screenshots etc. land in the cage and in the chat
            ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(mime, "png")
            rel = f".mcp/{cfg.name}-{int(time.time() * 1000)}-{i}.{ext}"
            agent.ws.write_bytes(agent.ws.resolve(rel), data)  # never through a link, never into a FIFO
            agent.emit({"type": "image", "path": rel, "caption": f"{cfg.name} · {tool}"})
            text += f"\n[image saved to {rel}]"
        if res.is_error:
            raise ToolError(text if len(text) <= 4000 else text[:4000] + f"\n[... error output cut: {len(text):,} chars in total]")
        return _connector_output(agent, f"{cfg.name}-{tool}", text)

    return call


class _Stopped(ToolError):
    """The user pressed Stop while a connector call was in flight (the call is abandoned, not waited for)."""


def _interruptible(cancel: threading.Event | None, fn):
    """Run a blocking connector call (an MCP request, a Chrome command, VirusTotal polling…) in a helper thread and
    wait for it in short slices, so Stop returns at once instead of when the call ends (up to minutes). The abandoned
    call finishes, or times out, in the background and its result is dropped."""
    if cancel is None:
        return fn()
    if cancel.is_set():
        raise _Stopped("stopped by the user")
    box: dict = {}
    done = threading.Event()

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=target, daemon=True, name="nesi-connector-call").start()
    while not done.wait(0.2):
        if cancel.is_set():
            raise _Stopped("stopped by the user")
    if "error" in box:
        if cancel.is_set():  # a call that honours Stop itself ended because of it: the same answer either way
            raise _Stopped("stopped by the user") from box["error"]
        raise box["error"]
    return box["value"]


# Chrome's answer when a tab is gone or was taken out of the agent's group (extension/background.js assertAllowed).
_TAB_GONE = re.compile(r"not in the\W{1,3}Nesicode\W{1,3}group", re.I)


def _browser_json(res: dict) -> str:
    """A chrome_* result for the model. chrome_read puts the interactive elements (what the agent acts on) before the
    page text, so a long page never pushes the refs out of view; nothing is cut here (see _connector_output)."""
    if isinstance(res.get("elements"), list):
        head = {k: res[k] for k in ("tab_id", "title", "url") if k in res}
        rest = {k: v for k, v in res.items() if k not in head and k not in ("elements", "text")}
        res = {**head, **rest, "elements": res["elements"], **({"text": res["text"]} if "text" in res else {})}
    return json.dumps(res, ensure_ascii=False)


CONNECTOR_SUB_CAP = 20_000      # chars of one connector result that a sub-agent (no Layer-0 saving) keeps in context
CONNECTOR_HARD_CAP = 2_000_000  # beyond this even the main agent gets a cut copy (the complete output stays on disk)


def _connector_output(agent, tool: str, text: str) -> str:
    """A connector's result, never cut silently. The main agent saves big outputs to the conversation's archive and
    keeps head + tail with the real size (the loop's Layer 0), so it gets the whole text. A sub-agent doesn't do that,
    so a big result is cut here — saying so, with its real size and where the complete output was saved."""
    cap = CONNECTOR_SUB_CAP if getattr(agent, "depth", 0) else CONNECTOR_HARD_CAP
    if len(text) <= cap:
        return text
    where = "the rest could not be saved"
    try:
        from ..agent.context import save_output

        rel = save_output(agent.ws, "complete", tool, text, getattr(agent, "archive", ".nesicode"))
        where = f"the COMPLETE output is saved at {rel} — read it with read_file (offset/limit) or search it"
    except Exception:
        pass
    return text[:cap] + f"\n\n[... output cut here: showing the first {cap:,} of {len(text):,} chars; {where} ...]"


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


MCP_START_TIMEOUT = 90  # s a message waits for its MCP servers to start (all at once; a hung one is then skipped a while)
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


_REASONING_CLOSE = re.compile(r"</(?:think|thinking|analysis|reasoning)>", re.I)
_REASONING_OPEN = re.compile(r"<(?:think|thinking|analysis|reasoning)>", re.I)


def clean_title(text: str) -> str:
    """The title in a model's reply, without inline reasoning (<think>…</think>: reasoning models served without a
    reasoning parser put it in the text), quotes or markdown. '' when there is none (reasoning cut by max_tokens)."""
    text = text or ""
    closes = list(_REASONING_CLOSE.finditer(text))
    if closes:
        text = text[closes[-1].end():]  # the answer follows the last reasoning block (its opening tag may be in the prompt)
    elif (m := _REASONING_OPEN.search(text)):
        text = text[:m.start()]  # an unfinished reasoning block: there is no answer after it
    line = next((ln for ln in text.splitlines() if ln.strip()), "")
    title = re.sub(r"\s+", " ", line).strip(" \"'«»*#.`")
    title = re.sub(r"^(?:t[ií]tulo|title)\s*:\s*", "", title, flags=re.I).strip(" \"'«»*#.`")[:80]
    return "" if len(title) < 3 or title.startswith("<") else title


def merge_harness(base: dict, ours: dict, theirs: dict) -> tuple[dict, list[str]]:
    """Three-way merge of harness trees, file by file. base = the version a harnessing run started from, ours = the
    current version (the user may have edited it meanwhile), theirs = the run's result. A file changed on one side
    takes that side; changed differently on both, it is a conflict. Returns (merged, conflicting paths)."""
    base, ours, theirs = normalize(base), normalize(ours), normalize(theirs)
    merged: dict = {}
    conflicts: list[str] = []
    for path in sorted(set(base) | set(ours) | set(theirs)):
        b, o, t = base.get(path), ours.get(path), theirs.get(path)
        if t == b or o == t:
            v = o
        elif o == b:
            v = t
        else:
            conflicts.append(path)
            continue
        if v is not None:
            merged[path] = v
    return merged, conflicts


def _version_files(db: DB, project_id: int, version: int) -> dict:
    row = db.one("SELECT files FROM harness_versions WHERE project_id=? AND version=?", (project_id, version))
    return loads(row["files"], {}) if row else {}


KEEP_HARNESS_COPIES = 3  # finished, not yet applied harnessing runs per project whose copies stay for «Aplicar» later
GONE = ("Ya no se puede aplicar: las copias del proyecto de esta ejecución ya no existen (sus cambios ya se aplicaron, "
        "o se borraron para liberar espacio: solo se guardan las de las últimas ejecuciones sin aplicar).")


def _has_copies(rd: Path) -> bool:
    return not rd.is_symlink() and (rd / "source").is_dir()


def _drop_copies(rd: Path) -> None:
    """Delete a harnessing run's workspace copies (its snapshot, every rollout's workspace — a whole copy of the project
    each — and their private HOMEs with tool caches); trajectories, diffs and checks — what the UI shows — stay."""
    if rd.is_symlink():
        return
    remove_tree(rd / "source")
    rollouts = rd / "rollouts"
    if rollouts.is_dir() and not rollouts.is_symlink():
        for d in rollouts.iterdir():
            if d.is_dir() and not d.is_symlink():
                remove_tree(d / "ws")
                remove_tree(d / ".ws.home")


def _fully_applied(rd: Path, selected: str, applied: dict | None) -> bool:
    """After an apply: is everything of the result in the project now — applied, already there, kept beside the user's
    file as a conflict's side file, or a removal the user's later edit overrides? Then its copies are no longer worth
    their disk. A change that could not be written (or whose side file could not) keeps them, for another «Aplicar»."""
    if not isinstance(applied, dict) or not applied:
        return False
    for c in applied.get("conflicts") or []:
        if c.get("reason") == "error":
            return False
        written = os.path.lexists(rd / "rollouts" / str(selected) / "ws" / str(c.get("path", "")))
        if c.get("reason") in ("changed", "exists") and written and not c.get("side_file"):
            return False
    return True


def _fresh_archive(db: DB, ws: Workspace, run: dict, thread_id: int) -> None:
    """In a shared project folder a deleted conversation's archive (.nesicode/chats/<id>) stays behind, and thread ids
    were reused after a delete before they became AUTOINCREMENT (db.migrate_autoincrement; an archive from back then, or
    a database the migration could not convert, can still meet a new conversation with its id). Before a conversation's
    first turn that does any work, an archive found there is therefore someone else's: remove it, so search_history
    never serves a deleted conversation."""
    if db.one("SELECT 1 FROM runs r WHERE r.thread_id=? AND r.id<>? AND EXISTS (SELECT 1 FROM events e WHERE e.run_id=r.id) "
              "LIMIT 1", (thread_id, run["id"])):
        return
    chain = [ws.root / ".nesicode", ws.root / ".nesicode" / "chats", ws.root / ".nesicode" / "chats" / str(int(thread_id))]
    if any(p.is_symlink() for p in chain) or not chain[-1].is_dir():
        return
    shutil.rmtree(chain[-1], ignore_errors=True)


class Approval:
    """A tool call waiting for the user. `id` is a fresh server-side nonce: the model's own call id can repeat (servers
    that omit ids get 'call_0' on every step), so an old «Permitir» can never answer a newer request."""

    def __init__(self, call_id: str, job: "_Job | None" = None):
        self.id = secrets.token_hex(6)
        self.call_id = call_id
        self.job = job
        self.event = threading.Event()
        self.decision = ""


class Live:
    """Transient (non-persisted) events such as token deltas, per run."""

    def __init__(self):
        self.gen = 0
        self.items: list[dict] = []


class _Job:
    """One submission of a run to the pool.

    A run can be put back in its queue, and in a database still without AUTOINCREMENT ids (db.migrate_autoincrement could
    not convert it) a deleted run's id is given to the next one, so whatever a job does — starting the run, emitting
    events, waiting for an approval, finishing — first checks that this job still owns the run (Jobs._owns). A stale
    job does nothing."""

    def __init__(self, run_id: int, kind: str, thread_id: int | None = None):
        self.run_id = run_id
        self.kind = kind
        self.thread_id = thread_id
        self.cancel = threading.Event()
        self.live = Live()
        self.started = False
        self.finished = False


class _Pool:
    """Runs jobs on their own daemon threads, at most `max_workers` of them working at once.

    A job that has to wait a long time without working (an approval, a shared project's lock) steps out of its slot
    with `idle()`, so a run waiting for the user never starves the other conversations; it takes a slot again — before
    jobs that have not started — when it resumes. Daemon threads: closing the app never waits for a run (Jobs.shutdown
    stops them first)."""

    def __init__(self, max_workers: int):
        self._max_workers = max(1, int(max_workers))  # the bot scheduler reads it
        self._cond = threading.Condition()
        self._queue: deque = deque()
        self._busy = 0  # slots taken
        self._resuming = 0  # idle jobs waiting to take a slot back (served first)
        self._alive = 0  # job threads not finished yet
        self._closed = False
        self._local = threading.local()

    def submit(self, fn, *args) -> None:
        with self._cond:
            if self._closed:
                raise RuntimeError("cannot schedule new jobs after shutdown")
            self._queue.append((fn, args))
            self._pump()

    def _pump(self) -> None:  # with self._cond held
        while self._queue and not self._resuming and self._busy < self._max_workers:
            fn, args = self._queue.popleft()
            self._busy += 1
            self._alive += 1
            try:
                threading.Thread(target=self._work, args=(fn, args), daemon=True, name="nesi-job").start()
            except RuntimeError as exc:  # no thread now (e.g. the interpreter is exiting): it stays queued
                self._busy -= 1
                self._alive -= 1
                self._queue.appendleft((fn, args))
                print("nesicode: could not start a job:", exc)
                return

    def _work(self, fn, args) -> None:
        self._local.slot = True
        try:
            fn(*args)
        finally:
            with self._cond:
                if getattr(self._local, "slot", False):
                    self._busy -= 1
                self._local.slot = False
                self._alive -= 1
                self._cond.notify_all()
                self._pump()

    @contextmanager
    def idle(self, cancel: threading.Event | None = None):
        """Give this job's slot to someone else while it waits (no-op outside a pool job)."""
        if not getattr(self._local, "slot", False):
            yield
            return
        with self._cond:
            self._busy -= 1
            self._local.slot = False
            self._cond.notify_all()
            self._pump()
        try:
            yield
        finally:
            with self._cond:
                self._resuming += 1
                try:
                    while self._busy >= self._max_workers and not self._closed and not (cancel is not None and cancel.is_set()):
                        self._cond.wait(0.25)
                finally:
                    self._resuming -= 1
                self._busy += 1
                self._local.slot = True
                self._pump()

    def shutdown(self) -> list:
        """Start nothing more; return the jobs that never started."""
        with self._cond:
            self._closed = True
            dropped = list(self._queue)
            self._queue.clear()
            self._cond.notify_all()
        return dropped

    def join(self, timeout: float) -> bool:
        """Wait (up to `timeout` s) until every job thread has ended."""
        end = time.monotonic() + timeout
        with self._cond:
            while self._alive and (left := end - time.monotonic()) > 0:
                self._cond.wait(min(left, 0.25))
            return not self._alive


_INSTANCES: "weakref.WeakSet[Jobs]" = weakref.WeakSet()


def _stop_all_at_exit() -> None:
    """The app is closing (desktop window closed, `serve` stopped): stop every run and give them a few seconds to
    record how they ended, instead of keeping a hidden process alive until they finish (approvals wait 30 min)."""
    for jobs in list(_INSTANCES):
        try:
            jobs.shutdown(wait=5)
        except Exception as exc:  # never block the exit
            print("nesicode: shutdown failed:", exc)


try:  # before Python joins non-daemon threads (the harnessing engine's rollout pools are such threads)
    threading._register_atexit(_stop_all_at_exit)  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001 - very old/new Pythons: a plain exit hook still stops the daemon job threads
    atexit.register(_stop_all_at_exit)


class Jobs:
    def __init__(self, db: DB, max_workers: int | None = None):
        self.db = db
        self.pool = _Pool(max_workers or settings.max_jobs)
        self.cancels: dict[int, threading.Event] = {}
        self.live: dict[int, Live] = {}
        self.approvals: dict[int, Approval] = {}  # run_id -> the tool call waiting for the user
        self.project_locks: dict[int, threading.Lock] = {}
        self.on_finish: list = [self._auto_title]  # callbacks(run_id) after any run ends (bots use it too)
        self.bots = None  # BotManager, set by the app
        self._lock = threading.RLock()
        self._jobs: dict[int, _Job] = {}  # run_id -> the job that owns it (queued or running)
        self._closing = False
        # A message accepted but never started goes back to its conversation's queue (resume_waiting starts the queues
        # in order); a run that was working when the app closed is 'interrupted'.
        db.execute("UPDATE runs SET status='waiting' WHERE status='queued' AND kind='chat' AND thread_id IS NOT NULL")
        db.execute(
            "UPDATE runs SET status='interrupted', finished_at=? WHERE status IN ('queued', 'running')",
            (time.time(),),
        )
        migrate_layout(db)
        _INSTANCES.add(self)
        threading.Thread(target=self._prune_all_copies, daemon=True, name="nesi-prune").start()

    def _project_lock(self, project_id: int) -> threading.Lock:
        with self._lock:
            return self.project_locks.setdefault(project_id, threading.Lock())

    def _acquire(self, lock: threading.Lock, cancel: threading.Event) -> bool:
        """Take a project's lock. Waiting for it holds no worker, and Stop ends the wait (False)."""
        if lock.acquire(blocking=False):
            return True
        with self.pool.idle(cancel):
            while not lock.acquire(timeout=0.25):
                if cancel.is_set():
                    return False
        return True

    def bot_slots(self) -> int:
        """Unattended bot runs may use every worker but one: a chat always finds a free worker."""
        return max(1, self.pool._max_workers - 1)

    # ------------------------------------------------------------ jobs and runs
    def _owns(self, job: _Job) -> bool:
        return self._jobs.get(job.run_id) is job

    def _register(self, job: _Job) -> None:
        """(with _lock) `job` now owns its run; an earlier submission of the same id is told to stop."""
        old = self._jobs.get(job.run_id)
        if old is not None and old is not job:
            old.cancel.set()
        self._jobs[job.run_id] = job
        self.cancels[job.run_id] = job.cancel

    def _forget(self, job: _Job) -> None:
        """(with _lock) Drop what this job registered — only what is still its own."""
        if self._jobs.get(job.run_id) is job:
            del self._jobs[job.run_id]
        if self.cancels.get(job.run_id) is job.cancel:
            del self.cancels[job.run_id]
        if self.live.get(job.run_id) is job.live:
            del self.live[job.run_id]
        pending = self.approvals.get(job.run_id)
        if pending is not None and pending.job is job:
            del self.approvals[job.run_id]

    def _new_run(self, run_id: int) -> None:
        """(with _lock) A row was just created with this id: anything still registered under it belongs to a deleted run
        whose id SQLite reused — it must stop and never touch the new run."""
        old = self._jobs.pop(run_id, None)
        if old is not None:
            old.cancel.set()
        ev = self.cancels.pop(run_id, None)
        if ev is not None:
            ev.set()
        self.live.pop(run_id, None)
        self.approvals.pop(run_id, None)

    def _claim(self, job: _Job) -> dict | None:
        """Start the run this job was submitted for — only if the job still owns it and it is still 'queued' (not
        withdrawn, put back in its queue or deleted meanwhile). Returns the run, or None: then do nothing at all."""
        with self._lock:
            if not self._owns(job):
                return None
            claimed = not self._closing and self.db.execute(
                "UPDATE runs SET status='running' WHERE id=? AND status='queued'", (job.run_id,)).rowcount == 1
            run = self.db.one("SELECT * FROM runs WHERE id=?", (job.run_id,)) if claimed else None
            if not run:
                self._forget(job)
                return None
            job.started = True
            job.thread_id = run.get("thread_id")
            self.live[job.run_id] = job.live
            return run

    def emitter(self, run_id: int, job: _Job | None = None):
        live = job.live if job is not None else self.live.setdefault(run_id, Live())

        def emit(event: dict) -> None:
            if job is not None and not self._owns(job):
                return  # the run is no longer this job's (deleted and its id reused, or finished): never write into it
            event = dict(event)
            persist = event.pop("_persist", True)
            if persist:
                kind = event.get("type")
                # The streamed deltas end here: the final text is in this event, or (a retry of a reply that had
                # started streaming) they belong to a failed attempt that a reconnecting client must not see again.
                # Cleared BEFORE the event is stored: a stream never reads the event together with the old deltas.
                if kind in ("assistant", "done", "error") or (kind == "llm_retry" and event.get("partial")
                                                              and not event.get("depth")):
                    live.gen += 1
                    live.items = []
                self.db.add_event(run_id, event)
            else:
                live.items.append(event)

        return emit

    def shutdown(self, wait: float = 0.0) -> None:
        """Stop everything — when the app closes (automatically, see _stop_all_at_exit) and in the tests. Running runs
        are told to stop (they end 'interrupted'), pending approvals end, nothing new starts; messages that had not
        started stay in their queue for the next start. `wait`: seconds to give the running ones to end cleanly."""
        with self._lock:
            self._closing = True
            for ev in list(self.cancels.values()):
                ev.set()
            for job in list(self._jobs.values()):
                job.cancel.set()
            try:
                self.db.execute("UPDATE runs SET status='waiting' WHERE status='queued' AND kind='chat' AND thread_id IS NOT NULL")
            except Exception as exc:  # the database may already be gone at exit
                print("nesicode: could not keep the queued messages:", exc)
        self.pool.shutdown()
        if wait > 0:
            self.pool.join(wait)

    def cancel(self, run_id: int) -> bool:
        """Stop a run. A running one is told to stop (it ends 'cancelled'). One that has not started — 'queued' for a
        worker or 'waiting' in its conversation — is withdrawn at once (its job, if any, will find it no longer owns the
        run and do nothing), and the conversation's queue moves on."""
        with self._lock:
            ev = self.cancels.get(run_id)
            if ev is not None:
                ev.set()
            row = self.db.one("SELECT thread_id, kind, status FROM runs WHERE id=?", (run_id,))
            job = self._jobs.get(run_id)
            started = job is not None and job.started
            withdrawn = False
            if row and not started and row["status"] in ("queued", "waiting"):
                self.db.execute("UPDATE runs SET status='cancelled', finished_at=? WHERE id=? AND status IN ('queued', 'waiting')",
                                (time.time(), run_id))
                withdrawn = True
            if job is not None and not started:
                self._forget(job)
            if row and row["kind"] == "chat" and row["thread_id"]:
                self.advance(row["thread_id"])
        return ev is not None or withdrawn

    def _finish(self, run_id: int, status: str, *, job: _Job | None = None, **fields) -> None:
        with self._lock:
            if job is not None:
                if job.finished or not self._owns(job):
                    return  # deleted and replaced, or already finished: never touch that row again
                job.finished = True
            if status == "cancelled" and self._closing:
                status = "interrupted"  # stopped because the app is closing, not by the user
            sets = ["status=?", "finished_at=?"] + [f"{k}=?" for k in fields]
            now = time.time()
            self.db.execute(
                f"UPDATE runs SET {', '.join(sets)} WHERE id=?",
                (status, now, *fields.values(), run_id),
            )
            # however it ended (an answer, Stop, a crash in _safe, deleted meanwhile…): its conversation changed, so a
            # chat list that is not open (it watches updated_at) sees it
            self.db.execute("UPDATE threads SET updated_at=? WHERE id=(SELECT thread_id FROM runs WHERE id=?)", (now, run_id))
            if job is not None:
                self._forget(job)
            else:  # finished by hand (no job of its own): drop whatever is registered under the id
                stale = self._jobs.pop(run_id, None)
                if stale is not None:
                    stale.cancel.set()
                self.cancels.pop(run_id, None)
                self.live.pop(run_id, None)
        for cb in list(self.on_finish):
            try:
                cb(run_id)
            except Exception as exc:
                print("nesicode: on_finish hook failed:", exc)
        try:  # the run has ended whatever happens next: starting the next message must never turn it into an error
            row = self.db.one("SELECT thread_id, kind FROM runs WHERE id=?", (run_id,))
            if row and row["kind"] == "chat" and row["thread_id"]:
                self.advance(row["thread_id"])
            self._start_held_bots()
        except Exception as exc:
            print("nesicode: could not start the next queued message:", exc)

    # ------------------------------------------------------------ titles
    TITLE_AT = (1, 3)  # retitle after the 1st and 3rd answered turn (more context by then)

    def _auto_title(self, run_id: int) -> None:
        if self._closing:
            return  # the app is closing: no new model calls
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
            prompt = ("Write a short title (3 to 7 words) for this conversation, in the same language the user writes in. "
                      "It must say what the conversation is about (the topic or goal), not quote the first message. "
                      "Reply with the title only: no quotes, no trailing period, no emoji.\n\n" + convo)
            text, _ = provider.complete(model, "You name conversations.", prompt,
                                        ModelOptions(effort="low", role="titler", max_tokens=60))
            title = clean_title(text)
            if not title and (_REASONING_OPEN.search(text or "") or _REASONING_CLOSE.search(text or "")):
                # a reasoning model spent the few tokens thinking: one more try with room for the thinking
                text, _ = provider.complete(model, "You name conversations.", prompt,
                                            ModelOptions(effort="low", role="titler", max_tokens=1024))
                title = clean_title(text)
            if title:
                # the user may have renamed it meanwhile: only replace an unlocked title
                self.db.execute("UPDATE threads SET title=? WHERE id=? AND title_locked=0", (title, thread_id))
        except Exception as exc:  # a title is never worth an error
            print("nesicode: auto-title failed:", exc)

    # ------------------------------------------------------------ per-conversation queue
    def enqueue_chat(self, run_id: int, thread_id: int) -> str:
        """A new message joins the back of its conversation's queue, and starts at once if nothing is ahead of it."""
        with self._lock:
            self._new_run(run_id)
            self.db.execute("UPDATE runs SET status='waiting' WHERE id=?", (run_id,))
            self.advance(thread_id)
            row = self.db.one("SELECT status FROM runs WHERE id=?", (run_id,))
        return row["status"] if row else "waiting"

    def _thread_busy(self, thread_id: int) -> bool:
        """(with _lock) Is a reply of this conversation queued or running? The jobs in memory count too: a row's status
        can be changed under a running job (the API marks a run cancelled before it has stopped). But only while their
        run still exists in this conversation: a job stuck in a call that ignores Stop outlives a deleted conversation,
        and a new conversation that gets the same id (SQLite reuses it) must not wait for it."""
        for j in list(self._jobs.values()):
            if j.thread_id == thread_id and j.started and not j.finished and \
                    self.db.one("SELECT 1 FROM runs WHERE id=? AND thread_id=?", (j.run_id, thread_id)):
                return True
        return bool(self.db.one("SELECT 1 FROM runs WHERE thread_id=? AND status IN ('queued', 'running') LIMIT 1", (thread_id,)))

    def _may_start(self, run: dict) -> bool:
        """(with _lock) Unattended bot runs (schedules, catch-up when the app opens, webhooks, Telegram, «Ejecutar
        ahora») never take the last worker: they wait in their queue until a bot slot is free."""
        if not run.get("bot_id") or run.get("trigger") == "chat":
            return True
        n = self.db.one("SELECT COUNT(*) AS n FROM runs WHERE bot_id IS NOT NULL AND trigger<>'chat' "
                        "AND status IN ('queued', 'running')")["n"]
        return n < self.bot_slots()

    def advance(self, thread_id: int) -> None:
        """Start the oldest waiting message of the conversation, if nothing of it is queued or running (and, for an
        unattended bot run, if a bot slot is free)."""
        with self._lock:
            if self._closing or self._thread_busy(thread_id):
                return
            nxt = self.db.one("SELECT * FROM runs WHERE thread_id=? AND status='waiting' ORDER BY created_at, id LIMIT 1",
                              (thread_id,))
            if not nxt or not self._may_start(nxt):
                return
            self.db.execute("UPDATE runs SET status='queued' WHERE id=?", (nxt["id"],))
            job = _Job(nxt["id"], "chat", thread_id)
            self._register(job)
            try:
                self.pool.submit(self._safe, self._chat, job)
            except RuntimeError:  # shutting down: it stays in its queue for the next start
                self._forget(job)
                self.db.execute("UPDATE runs SET status='waiting' WHERE id=? AND status='queued'", (nxt["id"],))

    def _start_held_bots(self) -> None:
        """Bot runs held back by the bots' share of workers start as workers free up (oldest first)."""
        with self._lock:
            if self._closing:
                return
            for r in self.db.all("SELECT thread_id, MIN(created_at) AS t FROM runs WHERE status='waiting' AND bot_id IS NOT NULL "
                                 "AND trigger<>'chat' AND thread_id IS NOT NULL GROUP BY thread_id ORDER BY t"):
                self.advance(r["thread_id"])

    def send_now(self, run_id: int) -> bool:
        """Jump the queue: this waiting message goes next. The reply in progress is stopped; one that had not started
        yet is not lost — it goes back to the queue, right after this message."""
        stop = None
        with self._lock:
            run = self.db.one("SELECT * FROM runs WHERE id=?", (run_id,))
            if not run or run["status"] != "waiting":
                return False
            cur = self.db.one("SELECT id, status, created_at FROM runs WHERE thread_id=? AND status IN ('queued', 'running') "
                              "ORDER BY created_at DESC, id DESC LIMIT 1", (run["thread_id"],))
            first = self.db.one("SELECT MIN(created_at) AS t FROM runs WHERE thread_id=? AND status='waiting'",
                                (run["thread_id"],))["t"]
            base = cur["created_at"] if cur else first - 1
            # right after the current reply, ahead of everything else in the queue (and shown in that order)
            at = min(base + 1e-3, first - 1e-3)
            self.db.execute("UPDATE runs SET created_at=? WHERE id=?", (at, run_id))
            job = self._jobs.get(cur["id"]) if cur else None
            if cur and cur["status"] == "queued" and not (job and job.started):
                # accepted but not started (all workers busy): nothing to stop — it goes next after this message
                self.db.execute("UPDATE runs SET status='waiting', created_at=? WHERE id=? AND status='queued'",
                                ((at + first) / 2, cur["id"]))
                if job is not None:
                    self._forget(job)  # its job, still waiting for a worker, now does nothing
                self.advance(run["thread_id"])
            elif cur:
                stop = cur["id"]  # when it has stopped, its _finish starts this message
            else:
                self.advance(run["thread_id"])
        if stop is not None:
            self.cancel(stop)
        return True

    def resume_waiting(self) -> None:
        """After a restart: queued messages carry on, oldest conversation first. A bot run needs the bot manager (its
        tools and model), which the app attaches right after starting this: give it a moment."""
        deadline = time.time() + 5
        while self.bots is None and time.time() < deadline and \
                self.db.one("SELECT 1 FROM runs WHERE status='waiting' AND bot_id IS NOT NULL LIMIT 1"):
            time.sleep(0.05)
        for row in self.db.all("SELECT thread_id, MIN(created_at) AS t FROM runs WHERE status='waiting' AND thread_id IS NOT NULL "
                               "GROUP BY thread_id ORDER BY t"):
            self.advance(row["thread_id"])

    # ------------------------------------------------------------------ chat
    def submit_chat(self, run_id: int) -> None:
        """Run a 'queued' chat message now (the conversation queue normally does this, see advance)."""
        self._submit(run_id, "chat", self._chat)

    def submit_harnessing(self, run_id: int) -> None:
        with self._lock:
            self._new_run(run_id)
        self._submit(run_id, "harness", self._harnessing)

    def _submit(self, run_id: int, kind: str, fn) -> None:
        job = _Job(run_id, kind)
        with self._lock:
            self._register(job)
            try:
                self.pool.submit(self._safe, fn, job)
            except RuntimeError:  # shutting down: a chat waits for the next start, a harnessing run is 'interrupted' then
                self._forget(job)
                print(f"nesicode: closing, run {run_id} will not start now")

    def _safe(self, fn, job: _Job) -> None:
        try:
            fn(job)
        except Exception as exc:  # never lose a run silently — nor touch one this job no longer owns
            if job.finished or not self._owns(job):
                print(f"nesicode: a job for run {job.run_id} failed after it ended or lost the run:", exc)
                return
            error = traceback.format_exc()[-4000:]
            try:  # (its row may be gone: the conversation was deleted while the job was stuck)
                self.emitter(job.run_id, job)({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
            except Exception as exc2:
                print(f"nesicode: could not record the error of run {job.run_id}:", exc2)
            self._finish(job.run_id, "error", job=job, error=error)
        finally:
            with self._lock:  # an ended job owns nothing any more, whatever happened (else its conversation looks busy)
                if self._owns(job):
                    job.finished = True
                    self._forget(job)

    def _chat(self, job: _Job) -> None:
        run = self._claim(job)
        if run is None:
            return  # withdrawn, put back in its queue or deleted before it started: nothing to do
        db, run_id, cancel = self.db, job.run_id, job.cancel
        emit = self.emitter(run_id, job)
        project = db.one("SELECT * FROM projects WHERE id=?", (run["project_id"],))
        thread = db.one("SELECT * FROM threads WHERE id=?", (run["thread_id"],))
        if not project or not thread:  # deleted while it waited for a worker
            self._finish(run_id, "cancelled", job=job)
            return
        history, interrupted = conversation_history(db, run)
        keys = user_keys(db, project["user_id"])
        vision = project.get("vision_model") or settings.vision_model
        image = project.get("image_model") or settings.image_model
        ws = Workspace(thread_ws(project, thread))
        bot = db.one("SELECT * FROM bots WHERE id=?", (run["bot_id"],)) if run.get("bot_id") else None
        from .limits import learn_limit, model_limits

        # The model that really runs (a bot can have its own): its context limits — and a limit learned from its errors
        # is its own, never the project executor's.
        model = (bot["model"] if bot and self.bots else "") or project["executor_model"]
        lim = model_limits(db, project["user_id"], model)
        # A shared workspace is serialised per project (waiting for it holds no worker and Stop works meanwhile);
        # a cage is already one-run-at-a-time.
        lock = None if thread["isolated"] else self._project_lock(project["id"])
        if lock is not None and not self._acquire(lock, cancel):
            emit({"type": "cancelled"})
            self._finish(run_id, "cancelled", job=job)
            return
        try:
            db.execute("UPDATE runs SET harness_version=? WHERE id=?", (project["harness_version"], run_id))
            task = run["task"]
            archive = f".nesicode/chats/{thread['id']}"
            _fresh_archive(db, ws, run, thread["id"])
            _write_thread_history(db, ws, run, archive)
            attachments = loads(run["attachments"], [])
            if attachments:
                task += "\n\n" + attachments_note(ws, attachments, vision, keys, emit)
            harness = current_harness(db, project)
            info: dict = {}
            # "only when I ask": the user's recent words, the agent's recent answers and the connectors already used in
            # this conversation keep a connector available (no need to repeat its name every message)
            recent = connector_context(db, run, history, bot)
            tools = self.connector_tools(project["user_id"], emit, context=recent, info=info, cancel=cancel)
            tools.update(self.chat_tools(project, emit))
            note = continuity_note(thread["id"], interrupted)
            if thread["isolated"]:
                where = project.get("folder") or "the project folder"
                note += ("\n\n# Where your files go\nYou work in an ISOLATED COPY of the project (" + where + "), not in it: "
                         "files you create or clone stay in this conversation until the user presses «Archivos → Aplicar al "
                         "proyecto». Always work inside your workspace (never /tmp or other folders), and when you create files the "
                         "user may expect in their folder, tell them where they are and that they appear in " + where +
                         " after «Aplicar».")
            if info["on_request"]:
                note += ("\n\n" if note else "") + ("# Connectors available on request\nThe user has these connectors but wants them used ONLY when they "
                        "explicitly ask for them by name: " + ", ".join(sorted(set(info["on_request"]))) + ". Do the task with "
                        "your other tools. If one of them would clearly help, say so at the end and suggest the user ask for it "
                        "(e.g. 'use Burp to …'); never pretend you used it.")
            if info["skipped"]:
                note += ("\n\n" if note else "") + ("# Connectors that could not start\nThese connectors are switched on but "
                         "did not start for this message, so their tools are missing: " + "; ".join(info["skipped"]) +
                         ". Never pretend you used them. If the task needs one, do what you can without it and tell the user "
                         "it was unavailable and why (they can press «Probar» on it in Integraciones to retry now).")
            steps, system_base = project["max_steps"], None
            if bot and self.bots:
                tools.update(self.bots.agent_tools(bot, run_id))
                steps = min(steps, bot["max_steps"])
                system_base = (harness.get("system_prompt.md", "").strip() + "\n\n"
                               + self.bots.system_section(bot, run.get("trigger") or "", run_id=run_id))
            agent = Agent(
                model=model,
                harness=harness,
                workspace=ws,
                emit=emit,
                user_keys=keys,
                max_steps=steps,
                cancel=cancel,
                approve=self._approver(job, thread, project, emit, bot, attended=run.get("trigger") == "chat",
                                       ask_tools=info["ask"]),
                system_note=note,
                vision_model=vision,
                image_model=image,
                system_base=system_base,
                extra_tools=tools,
                context_limit=lim["context"],
                output_limit=lim["output"],
                on_limit=lambda n, m=model: learn_limit(db, project["user_id"], m, n),
                archive=archive,
            )
            res = agent.run(task, history)
        finally:
            if lock is not None:
                lock.release()
        try:
            if not bot:  # unattended bot runs never trigger the auto-improver
                self._auto_improve(run, project, thread, res, emit)
        except Exception as exc:  # never break the chat because of the auto-improver
            emit({"type": "auto_harness_error", "text": f"{type(exc).__name__}: {exc}"})
        self._finish(  # (it also marks the conversation as updated)
            run_id, res.status, job=job, final_text=res.final_text, error=res.error,
            result=json.dumps({"usage": res.usage.to_dict(), "steps": res.steps}),
        )

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

    def connector_tools(self, user_id: int, emit, context: str | None = None, info: dict | None = None,
                        cancel: threading.Event | None = None) -> dict:
        """Tools of the user's enabled MCP connectors, as agent tools named mcp__<connector>__<tool>.

        `context` (the user's recent words): connectors in 'mention' mode are only offered when named there;
        the others are listed in info['on_request'] so the agent can suggest them. info['ask'] collects the
        tool names that need the user's OK on every use; info['skipped'] says which offered connectors could not
        start and why (the agent is told). context=None → everything (connector tests, etc.).
        The offered MCP servers start all at once; one that hung at startup recently is skipped at once (see
        McpManager.tools_many), so a hung server costs one timeout per few minutes, not one per message.
        `cancel`: the run's Stop — starting a slow connector (up to 90 s) does not delay it."""
        from ..connectors import catalog_entry, manager, mentioned, tool_name
        from ..llm import ToolSpec
        from .connectors_store import connector_configs

        info = info if info is not None else {}
        info.setdefault("on_request", [])
        info.setdefault("ask", set())
        info.setdefault("skipped", [])
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
        offered = []
        for row, cfg in connector_configs(self.db, user_id, enabled_only=True):
            if own_chrome and row["template"] == "chrome":
                continue  # the user's own Chrome is connected: don't offer a second, separate Chrome
            mode = row.get("usage") or "mention"
            if context is not None and mode == "mention" and not mentioned(context, name=row["name"], label=row["label"],
                                                                          template=row["template"]):
                info["on_request"].append(row["label"])
                continue
            offered.append((row, cfg, mode))
        mcp = [cfg for _, cfg, _ in offered if cfg.kind != "builtin"]
        started: list = []
        if mcp:
            try:
                started = _interruptible(cancel, lambda: manager().tools_many(user_id, mcp, timeout=MCP_START_TIMEOUT,
                                                                              cancel=cancel))
            except Exception as exc:  # Stop (the run ends right away) or the manager itself failed
                if cancel is not None and cancel.is_set():
                    mcp = []
                started = [exc] * len(mcp)
        results = iter(started)
        for row, cfg, mode in offered:  # in the user's order (the tool list stays stable between messages)
            before = set(out)
            alive = functools.partial(self._connector_off, user_id, row["name"])
            if cfg.kind == "builtin":
                out.update(builtin_agent_tools(row["name"], row["label"], cfg.command, cfg, alive=alive))
            else:
                tools = next(results, None)
                if tools is None:
                    continue  # stopped before it started
                if isinstance(tools, BaseException):
                    why = str(tools)[:300]
                    emit({"type": "connector_error", "connector": row["label"], "text": why})
                    info["skipped"].append(f"{row['label']} ({why.splitlines()[0][:200] if why else type(tools).__name__})")
                    continue
                hint = (catalog_entry(row["template"]) or {}).get("hint", "")
                for tool in tools:
                    name = tool_name(row["name"], tool.name)
                    desc = f"[{row['label']}] {tool.description}".strip()[:1600] + (f" ({hint})" if hint else "")
                    out[name] = (ToolSpec(name, desc, tool.schema), _mcp_caller(user_id, cfg, tool.name, alive=alive))
            if mode == "ask":
                info["ask"].update(set(out) - before)
        return out

    def _connector_off(self, user_id: int, name: str) -> str:
        """'' while the user's connector `name` is switched on, else why it can't be used any more. Checked on every
        call: a connector turned off or removed mid-conversation stops working at once, not at the next message."""
        row = self.db.one("SELECT enabled FROM user_connectors WHERE user_id=? AND name=?", (user_id, name))
        if row is None:
            return "the user removed this connector from Nesicode during this conversation: it can no longer be used"
        if not row["enabled"]:
            return ("the user turned this connector off in Nesicode during this conversation: it can't be used until they "
                    "turn it on again (don't retry it)")
        return ""

    # ------------------------------------------------------------ approvals
    def exec_mode(self, thread: dict, user_id: int) -> str:
        if thread.get("exec_mode") in EXEC_MODES:
            return thread["exec_mode"]
        prefs = loads((self.db.one("SELECT settings FROM users WHERE id=?", (user_id,)) or {}).get("settings"), {})
        return prefs.get("exec_mode") if prefs.get("exec_mode") in EXEC_MODES else "auto"

    def _approver(self, job: _Job, thread: dict, project: dict, emit, bot: dict | None = None,
                  attended: bool = False, ask_tools: set | None = None):
        run_id, cancel = job.run_id, job.cancel

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
            if cancel.is_set() or not self._owns(job):
                return False, "run cancelled"
            pending = Approval(call.id, job)
            with self._lock:
                self.approvals[run_id] = pending
            each_time = call.name.endswith(ALWAYS_ASK)
            # `id` is the request's own nonce (the app's and Telegram's buttons send it back); `call_id` is the model's
            emit({"type": "approval_request", "id": pending.id, "call_id": call.id, "name": call.name, "args": call.arguments,
                  "mode": mode, "always": not each_time})
            deadline = time.time() + APPROVAL_TIMEOUT
            with self.pool.idle(cancel):  # waiting for the user is not work: the worker serves other conversations meanwhile
                while not pending.event.wait(0.5):
                    if cancel.is_set() or time.time() > deadline:
                        with self._lock:  # an answer arriving at this very moment either wins or is refused, never lost
                            if not pending.event.is_set():
                                pending.decision = "cancelled" if cancel.is_set() else "timeout"
                                pending.event.set()
            with self._lock:
                if self.approvals.get(run_id) is pending:
                    del self.approvals[run_id]
            decision = pending.decision or "deny"
            emit({"type": "approval_result", "id": pending.id, "call_id": call.id, "decision": decision})
            if decision == "always" and each_time:
                decision = "allow"  # approved this once; the next one asks again
            if decision == "always":
                allowed = sorted(set(loads(t.get("allowed_tools"), [])) | {call.name})
                self.db.execute("UPDATE threads SET allowed_tools=? WHERE id=?", (json.dumps(allowed), thread["id"]))
            return decision in ("allow", "always"), {"deny": "denied", "timeout": "no answer in 30 min",
                                                      "cancelled": "run cancelled"}.get(decision, decision)

        return approve

    def decide(self, run_id: int, approval_id: str, decision: str) -> bool:
        """The user's answer to the approval request whose event `id` is `approval_id` (a per-request nonce, so a stale
        button — e.g. an old Telegram message — never answers a newer request that reuses the model's call id)."""
        if decision not in ("allow", "always", "deny"):
            return False
        with self._lock:
            pending = self.approvals.get(run_id)
            if not pending or pending.event.is_set() or str(approval_id) != pending.id:
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
        # The rollouts re-run this request several times with nobody to approve anything: never for a conversation
        # where the user gates commands, nor for a turn where something needed their approval.
        t = self.db.one("SELECT * FROM threads WHERE id=?", (thread["id"],)) or thread
        if self.exec_mode(t, project["user_id"]) != "auto":
            emit({"type": "auto_harness_skipped", "reason": reason,
                  "why": "la conversación pide permiso antes de ejecutar comandos y el harnessing los repetiría sin preguntar"})
            return
        # (the approver writes its requests straight to the run's events, not into the agent's own list)
        if any(e.get("type") == "approval_request" for e in res.events) or self.db.one(
                "SELECT 1 FROM events WHERE run_id=? AND data LIKE ? LIMIT 1", (run["id"], '%"type": "approval_request"%')):
            emit({"type": "auto_harness_skipped", "reason": reason,
                  "why": "esta respuesta necesitó tu permiso para alguna acción y el harnessing la repetiría sin preguntar"})
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
    def _gated(self, project: dict, src: dict) -> str:
        """Why a harnessing run must not re-run a chat request ('' = it may). Its rollouts run every command without
        asking anyone, so a request from a conversation that asks before running commands is never re-run that way."""
        t = self.db.one("SELECT * FROM threads WHERE id=? AND project_id=?", (src.get("thread_id"), project["id"])) or {}
        mode = self.exec_mode(t, project["user_id"])
        if mode == "auto":
            return ""
        label = {"ask_bash": "pedir permiso para comandos", "ask_all": "pedir permiso para todo"}.get(mode, mode)
        return (f"Esta conversación está en modo «{label}»: el harnessing repetiría su tarea varias veces ejecutando "
                "comandos sin poder pedirte permiso, así que no se ha lanzado. Si quieres mejorar el harness con ella, pon "
                "la conversación en modo auto y vuelve a pulsar «Mejorar harness con esta tarea».")

    def _harnessing(self, job: _Job) -> None:
        run = self._claim(job)
        if run is None:
            return  # withdrawn or deleted before it started
        db, run_id, cancel = self.db, job.run_id, job.cancel
        emit = self.emitter(run_id, job)
        project = db.one("SELECT * FROM projects WHERE id=?", (run["project_id"],))
        if not project:
            self._finish(run_id, "cancelled", job=job)
            return
        params = loads(run["params"], {})
        base_version = project["harness_version"]
        db.execute("UPDATE runs SET harness_version=? WHERE id=?", (base_version, run_id))
        src = params.get("source") or {}
        if src.get("thread_id") and (why := self._gated(project, src)):
            emit({"type": "error", "text": why})
            self._finish(run_id, "error", job=job, error=why)
            return
        lessons = db.all(
            "SELECT cue, lesson, evidence, confidence FROM lessons WHERE user_id=? AND domain=? "
            "ORDER BY created_at DESC LIMIT 12",
            (project["user_id"], project["domain"]),
        )
        from .limits import model_limits

        _lim = model_limits(db, project["user_id"], project["executor_model"])
        _imp = model_limits(db, project["user_id"], project["improver_model"])  # its window AND its output cap
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
            improver_context=_imp["context"],
            improver_output=_imp["output"],
        )
        rd = run_dir(project, run_id)
        extra = []
        if src.get("thread_id"):
            t = db.one("SELECT * FROM threads WHERE id=? AND project_id=?", (src["thread_id"], project["id"]))
            if t:
                extra.append((thread_ws(project, t) / ATTACHMENTS_DIR, ATTACHMENTS_DIR))
        hr = HarnessingRun(
            cfg, base_harness=_version_files(db, project["id"], base_version), source_ws=project_ws(project), run_dir=rd,
            lessons=lessons, emit=emit, user_keys=user_keys(db, project["user_id"]), cancel=cancel, seed=run_id,
            extra_files=extra,
        )

        def stopped(result: dict | None = None) -> None:
            # Stop — even during the last rollouts, the selection or the reflection: from here on nothing is applied,
            # adopted or learned (what was already applied stays recorded in the result)
            emit({"type": "cancelled"})
            self._finish(run_id, "cancelled", job=job,
                         **({"result": json.dumps(result, ensure_ascii=False)} if result else {}))

        done = applied_all = False
        try:
            try:
                result = hr.run()
            except RuntimeError as exc:
                if str(exc) != "cancelled":
                    raise
                return stopped()
            if cancel.is_set():
                return stopped()
            selected = result["selected"]
            result["applied"] = None
            if params.get("apply_changes"):
                lock = self._project_lock(project["id"])
                if not self._acquire(lock, cancel):
                    return stopped()
                try:
                    if cancel.is_set():
                        return stopped()
                    result["applied"] = hr.apply_selected(selected, project_ws(project))
                finally:
                    lock.release()
                emit({"type": "applied", "changes": result["applied"]})
                applied_all = _fully_applied(rd, selected, result["applied"])
            if cancel.is_set():  # Stop pressed meanwhile (also while it waited for the project): adopt and learn nothing
                return stopped(result if result["applied"] else None)
            result["adopted_version"] = None
            if result["harness_changed"] and params.get("adopt_harness", True):
                v, why, merged = self._adopt(project["id"], base_version, result["adopted_harness"], run_id, selected)
                result["adopted_version"] = v
                if v:
                    emit({"type": "adopted", "version": v, "label": selected, **({"merged": True} if merged else {})})
                else:
                    result["adopt_skipped"] = why
                    emit({"type": "adopt_skipped", "label": selected, "reason": why})
            if cancel.is_set():
                return stopped(result)
            exp = (result.get("reflection") or {}).get("experience")
            if exp:
                db.insert(
                    "INSERT INTO lessons (user_id, project_id, domain, cue, lesson, evidence, confidence, run_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (project["user_id"], project["id"], project["domain"], exp["cue"], exp["lesson"],
                     exp["evidence"], exp.get("confidence", ""), run_id, time.time()),
                )
                emit({"type": "lesson", **exp})
            # Keep only the selected rollout's workspace (for a later "apply"), drop the rest (run() already did).
            for label, ro in hr.rollouts.items():
                if label != selected and ro.ws_path:
                    remove_tree(ro.ws_path)
            emit({"type": "done", "status": "done", "usage": result["usage"]})
            sel = result["rollouts"].get(selected, {})
            self._finish(run_id, "done", job=job, result=json.dumps(result, ensure_ascii=False),
                         final_text=sel.get("final_text", ""))
            done = True
        finally:
            # cancelled or failed: nothing can be applied later; applied: nothing is left to apply. Either way no copy of
            # the project is worth its disk any more (the trajectories, diffs and checks the UI shows stay).
            if not done or applied_all:
                _drop_copies(rd)
            if done:
                self._prune_copies(project)

    def _adopt(self, project_id: int, base_version: int, adopted: dict, run_id: int, label: str) -> tuple[int | None, str, bool]:
        """Make the run's harness current without losing what changed while it worked (a manual edit, a profile switch,
        another run): if the current version is no longer the one the run started from, the run's own edits are merged
        file by file onto it; when both changed the same file nothing is adopted (the run can still be adopted by
        hand). Returns (new version or None, why not, merged)."""
        note = f"Harness² run #{run_id}: adoptado `{label}`"
        with self.db.lock:  # nobody saves a version between reading the current one and saving ours
            row = self.db.one("SELECT harness_version FROM projects WHERE id=?", (project_id,))
            if not row:
                return None, "el proyecto ya no existe", False
            cur = int(row["harness_version"])
            if cur == int(base_version):
                return save_harness_version(self.db, project_id, adopted, "harnessing", note, run_id), "", False
            base = _version_files(self.db, project_id, base_version)
            ours = _version_files(self.db, project_id, cur)
            merged, conflicts = merge_harness(base, ours, adopted)
            if conflicts:
                return None, (f"el harness cambió mientras trabajaba (ahora es la v{cur}) y los dos cambiaron "
                              f"{', '.join(conflicts[:5])}: no se adopta solo para no deshacer tus cambios; "
                              "puedes adoptarlo a mano"), False
            if merged == normalize(ours):
                return None, f"sus cambios ya están en la versión actual (v{cur})", False
            if problems := validate(merged):
                return None, (f"el harness cambió mientras trabajaba (ahora es la v{cur}) y la combinación no es válida: "
                              f"{problems[0][:300]}"), False
            v = save_harness_version(self.db, project_id, merged, "harnessing", note + f" (fusionado con la v{cur})", run_id)
            return v, "", True

    def _prune_copies(self, project: dict) -> None:
        """Workspace copies are only kept for the last few successful runs of a project (for «Aplicar» later)."""
        lock = self._project_lock(project["id"])
        if not lock.acquire(blocking=False):
            return  # an «Aplicar» is using them right now: prune next time
        try:
            kept = 0
            for r in self.db.all("SELECT id, status FROM runs WHERE project_id=? AND kind='harness' "
                                 "AND status NOT IN ('queued', 'running') ORDER BY id DESC", (project["id"],)):
                rd = run_dir(project, r["id"])
                if not rd.is_dir():
                    continue
                if r["status"] == "done" and _has_copies(rd) and kept < KEEP_HARNESS_COPIES:
                    kept += 1  # (an applied run has no copies left: it never takes the place of one not applied yet)
                    continue
                _drop_copies(rd)
        finally:
            lock.release()

    def _prune_all_copies(self) -> None:
        """At start: runs cut short (closed app, cancel, error) and old ones no longer keep whole copies of the project."""
        for p in self.db.all("SELECT * FROM projects"):
            try:
                self._prune_copies(p)
            except Exception as exc:
                print("nesicode: could not prune harnessing copies:", exc)

    def _try_lock(self, lock: threading.Lock, cancel: threading.Event | None = None, wait: float = LOCK_TRY) -> bool:
        """Take a project's lock for something that serves a request: at most `wait` s, in short slices (Stop or the
        app closing end the wait). False: another conversation is working in the project (it may for long)."""
        deadline = time.monotonic() + wait
        while not lock.acquire(timeout=max(0.0, min(0.1, deadline - time.monotonic()))):
            if time.monotonic() >= deadline or self._closing or (cancel is not None and cancel.is_set()):
                return False
        return True

    def apply_later(self, run_id: int, project: dict, cancel: threading.Event | None = None) -> dict:
        """«Aplicar» on a finished harnessing run: the chosen result's file changes go into the project, three-way — a
        file the user changed since the run started is never overwritten or deleted (it comes back in `conflicts`, the
        result's version beside it as <name>.nesicode-new) and nothing is written through a link (apply_rollout).
        Then the run's copies of the project go (kept when something could not be written, for another try).
        It serves an HTTP request, so it never waits long for a busy project: ProjectBusy after LOCK_TRY s.
        FileNotFoundError (with a message for the user) when the copies are gone: applied already, or cleaned up."""
        row = self.db.one("SELECT result FROM runs WHERE id=?", (run_id,))
        result = loads(row["result"], {}) if row else {}
        selected = result.get("selected")
        rd = run_dir(project, run_id)
        after = rd / "rollouts" / str(selected) / "ws"
        lock = self._project_lock(project["id"])  # (pruning old copies takes it too)
        if not self._try_lock(lock, cancel):
            raise ProjectBusy(BUSY)
        try:
            if not selected or not _has_copies(rd) or after.is_symlink() or not after.is_dir():
                raise FileNotFoundError(GONE)
            try:
                changes = apply_rollout(rd / "source", after, project_ws(project))
            except FileNotFoundError as exc:
                raise FileNotFoundError(GONE) from exc
            result["applied"] = changes
            self.db.execute("UPDATE runs SET result=? WHERE id=?", (json.dumps(result, ensure_ascii=False), run_id))
            if _fully_applied(rd, selected, changes):
                _drop_copies(rd)
        finally:
            lock.release()
        return changes
