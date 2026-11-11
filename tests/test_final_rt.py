"""Final review, runtime area: titles from reasoning models, Stop while an image is described, closing the app with a
stuck run, Telegram files and attachments through links planted in the cage, the live buffer of a reasoning run, link
chains in «Aplicar» and the data folder of `nesicode desktop` from source. Offline regression tests (fake providers, a
fake Telegram client, temp dirs; the desktop test runs a subprocess whose server listens on 127.0.0.1)."""

import base64
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import httpx
import pytest

from nesicode.agent import Agent, Workspace
from nesicode.agent.workspace import ToolError, apply_cage, manifest
from nesicode.harness import base_harness
from nesicode.harnessing.apply import apply_rollout
from nesicode.llm.base import Conversation, Provider, ToolCall, Turn, Usage
from nesicode.server.chatops import ensure_thread, post_message, save_attachment
from nesicode.server.db import DB
from nesicode.server.jobs import FINAL, Jobs, project_ws, thread_ws

POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX only (FIFOs, symlinks)")
_UIDS = iter(range(950_000, 960_000))
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100


# ====================================================================== helpers
def _until(cond, timeout=10.0, step=0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(step)
    return bool(cond())


def _quick(fn, timeout=5.0):
    """Run fn in a thread: it must return (or raise) within `timeout` s — opening a FIFO would block it for good."""
    box: dict = {}

    def go():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"{fn} hung (did it open a FIFO?)"
    if "error" in box:
        raise box["error"]
    return box.get("value")


class Env:
    """A private Jobs on its own database (users with high ids, so their folders never meet other tests')."""

    def __init__(self, tmp_path, workers=1, share=False):
        self.db = DB(tmp_path / f"f-{time.time_ns()}.db")
        self.jobs = Jobs(self.db, max_workers=workers)
        self.uid = next(_UIDS)
        self.db.execute("INSERT INTO users (id, email, name, password_hash, created_at, settings) VALUES (?, ?, 'T', 'x', ?, ?)",
                        (self.uid, f"f{self.uid}@example.com", time.time(), json.dumps({"share_workspace": share})))

    def project(self, model="mock/demo", **cols) -> dict:
        pid = self.db.insert("INSERT INTO projects (user_id, name, executor_model, improver_model, max_steps, created_at) "
                             "VALUES (?, 'P', ?, 'mock/demo', 20, ?)", (self.uid, model, time.time()))
        self.db.execute("INSERT INTO harness_versions (project_id, version, files, source, created_at) VALUES (?, 1, ?, 'base', ?)",
                        (pid, json.dumps(base_harness()), time.time()))
        for k, v in cols.items():
            self.db.execute(f"UPDATE projects SET {k}=? WHERE id=?", (v, pid))
        p = self.db.one("SELECT * FROM projects WHERE id=?", (pid,))
        project_ws(p).mkdir(parents=True, exist_ok=True)
        return p

    def thread(self, project) -> dict:
        return ensure_thread(self.db, project, self.uid, "chat")

    def post(self, project, thread, text, attachments=None) -> dict:
        return post_message(self.db, self.jobs, project, thread, text, attachments or [])

    def run(self, rid) -> dict:
        return self.db.one("SELECT * FROM runs WHERE id=?", (rid,))

    def events(self, rid) -> list[dict]:
        return [e for _, e in self.db.events_after(rid, 0, limit=5000)]

    def wait(self, rid, timeout=30) -> dict:
        assert _until(lambda: (self.run(rid) or {}).get("status") in FINAL, timeout), (self.run(rid), self.events(rid))
        return self.run(rid)

    def title(self, thread) -> str:
        return self.db.one("SELECT title FROM threads WHERE id=?", (thread["id"],))["title"]

    def close(self) -> None:
        self.jobs.shutdown(wait=3)


@pytest.fixture()
def env(tmp_path):
    made = []

    def make(workers=1, share=False):
        e = Env(tmp_path, workers, share)
        made.append(e)
        return e

    yield make
    for e in made:
        e.close()


class _Quiet(Conversation):
    def add_user(self, text):
        pass

    def add_assistant_text(self, text):
        pass

    def add_tool_results(self, results):
        pass


class _Prov(Provider):
    name = "fake"

    def __init__(self, conv_cls):
        self.conv_cls = conv_cls

    def conversation(self, model, system, tools, options):
        return self.conv_cls(model, system, tools, options)


@pytest.fixture()
def models(monkeypatch):
    """Fake models for the agent loop: models['mock/x'] = factory returning a Provider."""
    import nesicode.agent.loop as loop

    real, table = loop.get_provider, {}

    def get(spec, keys=None):
        if spec in table:
            return table[spec](), spec.split("/", 1)[1]
        return real(spec, keys)

    monkeypatch.setattr(loop, "get_provider", get)
    return table


class _HangingVision(Provider):
    """A vision / image endpoint that accepts the request and does not answer (until the test lets it)."""

    name = "hang"

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def conversation(self, *a):
        raise NotImplementedError

    def vision(self, model, prompt, image, mime):
        self.entered.set()
        self.release.wait(30)
        return "too late", Usage()

    def generate_image(self, model, prompt):
        self.entered.set()
        self.release.wait(30)
        return PNG, "image/png"


# ====================================================================== llm#7: titles from reasoning models
def _titler(reply):
    """A provider whose one-shot replies are reply(max_tokens) -> Turn; Provider.complete() is the real one."""
    budgets: list[int] = []

    class Conv(_Quiet):
        def step(self, on_event=None):
            budgets.append(self.options.max_tokens)
            return reply(self.options.max_tokens)

    return _Prov(Conv), budgets


def _answered(e, p, t) -> None:
    e.db.execute("INSERT INTO runs (project_id, thread_id, kind, task, status, final_text, created_at) "
                 "VALUES (?, ?, 'chat', 'cambia las reglas de precios del inventario', 'done', 'Hecho.', ?)",
                 (p["id"], t["id"], time.time()))


def test_title_retries_when_the_reasoning_came_in_its_own_field(env, monkeypatch):
    """nesi3 through LiteLLM (reasoning_content), DeepSeek, vLLM/SGLang parsers, Claude thinking: the reasoning is not
    in the text, so a 60-token budget spent reasoning leaves an empty text (finish 'length') and no <think> tag."""
    import nesicode.llm as llm

    def reply(max_tokens):
        if max_tokens <= 60:
            return Turn("", [], "max_tokens", Usage(300, 60), thinking="The user wants a short title about pricing")
        return Turn("Reglas de precios del inventario", [], "end_turn", Usage(300, 12), thinking="ok")

    prov, budgets = _titler(reply)
    monkeypatch.setattr(llm, "get_provider", lambda spec, keys=None: (prov, "nesi3"))
    e = env()
    p = e.project("nesilabs/nesi3")
    t = e.thread(p)
    _answered(e, p, t)
    e.jobs._make_title(t["id"], p["id"])
    assert e.title(t) == "Reglas de precios del inventario"
    assert budgets == [60, 1024]


def test_title_asks_once_when_the_first_reply_has_it_and_gives_up_after_one_retry(env, monkeypatch):
    import nesicode.llm as llm

    prov, budgets = _titler(lambda n: Turn("Migración a Postgres", [], "end_turn", Usage(1, 5)))
    monkeypatch.setattr(llm, "get_provider", lambda spec, keys=None: (prov, "m"))
    e = env()
    p = e.project("compat/m")
    t = e.thread(p)
    _answered(e, p, t)
    e.jobs._make_title(t["id"], p["id"])
    assert e.title(t) == "Migración a Postgres" and budgets == [60]

    prov, budgets = _titler(lambda n: Turn("", [], "max_tokens", Usage(1, n), thinking="thinking…"))
    monkeypatch.setattr(llm, "get_provider", lambda spec, keys=None: (prov, "m"))
    t2 = e.thread(p)
    _answered(e, p, t2)
    before = e.title(t2)
    e.jobs._make_title(t2["id"], p["id"])
    assert e.title(t2) == before and budgets == [60, 1024]  # one more try, never a loop


# ====================================================================== runtime#2: Stop while an image is described
def test_stop_ends_an_image_description_at_once(tmp_path, monkeypatch):
    from nesicode.agent import vision

    hang = _HangingVision()
    monkeypatch.setattr(vision, "get_provider", lambda spec, keys=None: (hang, "v"))
    ws = Workspace(tmp_path / "ws")
    (ws.root / "a.png").write_bytes(PNG)
    ws.cancel = threading.Event()  # the agent's Stop (Agent sets ws.cancel)
    try:
        threading.Timer(0.3, ws.cancel.set).start()
        t0 = time.monotonic()
        with pytest.raises(ToolError, match="stopped"):
            _quick(lambda: vision.describe_image(ws, "a.png", "hang/v", None, ""))
        assert time.monotonic() - t0 < 2.5
        t0 = time.monotonic()
        with pytest.raises(ToolError, match="stopped"):  # already stopped: no call at all
            _quick(lambda: vision.generate_image(ws, "un gato", "", "hang/i", None))
        assert time.monotonic() - t0 < 1
        cancel = threading.Event()  # an explicit cancel (the chat's attachments) works the same
        threading.Timer(0.3, cancel.set).start()
        with pytest.raises(ToolError, match="stopped"):
            _quick(lambda: vision.describe_image(Workspace(ws.root), "a.png", "hang/v", None, "", cancel=cancel))
    finally:
        hang.release.set()


def test_stop_during_an_attachment_description_ends_the_run_and_never_holds_the_project(env, monkeypatch):
    """A shared conversation describes an attached image with a vision endpoint that does not answer: other shared
    conversations of the project go on meanwhile, and Stop ends the run at once (not after ~20 min of timeouts)."""
    from nesicode.agent import vision

    hang = _HangingVision()
    monkeypatch.setattr(vision, "get_provider", lambda spec, keys=None: (hang, "v"))
    e = env(workers=2, share=True)
    p = e.project(vision_model="hang/v")
    t1, t2 = e.thread(p), e.thread(p)
    assert not t1["isolated"] and not t2["isolated"]
    try:
        att = save_attachment(p, t1, "captura.png", PNG)
        r1 = e.post(p, t1, "mira la captura", [att])
        assert hang.entered.wait(10)
        r2 = e.post(p, t2, "hola")  # the description holds no lock of the project
        assert e.wait(r2["id"], 15)["status"] == "done"
        t0 = time.monotonic()
        e.jobs.cancel(r1["id"])
        assert e.wait(r1["id"], 5)["status"] == "cancelled"
        assert time.monotonic() - t0 < 3
        assert not any(ev["type"] == "start" for ev in e.events(r1["id"]))  # stopped before the agent started
    finally:
        hang.release.set()


def test_stop_while_the_agent_views_an_image(tmp_path, monkeypatch, models):
    from nesicode.agent import vision

    hang = _HangingVision()
    monkeypatch.setattr(vision, "get_provider", lambda spec, keys=None: (hang, "v"))

    class Viewer(_Quiet):
        def step(self, on_event=None):
            return Turn("", [ToolCall("v1", "view_image", {"path": "a.png"})], "tool_use", Usage(1, 1))

    models["mock/viewer"] = lambda: _Prov(Viewer)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "a.png").write_bytes(PNG)
    cancel = threading.Event()
    agent = Agent(model="mock/viewer", harness=base_harness(), workspace=ws, cancel=cancel, vision_model="hang/v")
    box: dict = {}
    th = threading.Thread(target=lambda: box.setdefault("res", agent.run("mira a.png")), daemon=True)
    try:
        th.start()
        assert hang.entered.wait(10)
        cancel.set()
        th.join(5)
        assert not th.is_alive(), "Stop waited for the vision call"
        assert box["res"].status == "cancelled"
    finally:
        hang.release.set()


# ====================================================================== runtime#3: closing waits once
def test_closing_waits_once_for_a_run_that_ignores_stop(env, models):
    """stop_runtime's wait and then the exit hook's for the same stuck run: the app closes after the first one."""
    gate, entered = threading.Event(), threading.Event()

    class Stuck(_Quiet):
        def step(self, on_event=None):
            entered.set()
            gate.wait(30)  # a call that ignores Stop
            return Turn("tarde", [], "end_turn", Usage(1, 1))

    models["mock/stuck"] = lambda: _Prov(Stuck)
    e = env()
    p = e.project("mock/stuck")
    r = e.post(p, e.thread(p), "hola")
    try:
        assert entered.wait(10)
        t0 = time.monotonic()
        e.jobs.shutdown(wait=0)  # the signal handler's halt
        e.jobs.shutdown(wait=1)  # the server's stop_runtime
        e.jobs.shutdown(wait=1)  # the interpreter's exit hook (_stop_all_at_exit)
        e.jobs.shutdown(wait=0.5)
        took = time.monotonic() - t0
        assert 0.9 <= took < 1.7, took
    finally:
        gate.set()
    assert e.wait(r["id"])["status"] in FINAL


# ====================================================================== runtime#6: Telegram attachments
@POSIX
def test_telegram_attachments_never_go_through_links_planted_in_the_cage(env, tmp_path):
    e = env()
    p = e.project()
    t = e.thread(p)
    cage = thread_ws(p, t)
    outside = tmp_path / "OUTSIDE"  # a folder of the user's, outside the conversation's cage
    outside.mkdir()
    # the agent turned .attachments into a link to it
    shutil.rmtree(cage / ".attachments", ignore_errors=True)
    os.symlink(outside, cage / ".attachments")
    a = save_attachment(p, t, "foto.jpg", b"\xff\xd8\xff photo")
    assert a == {"name": "foto.jpg", "path": ".attachments/foto.jpg", "kind": "image", "size": 9}
    assert list(outside.iterdir()) == []
    assert not (cage / ".attachments").is_symlink()
    assert (cage / ".attachments" / "foto.jpg").read_bytes() == b"\xff\xd8\xff photo"
    # a dangling link where the next file goes (it would create a file outside): the next free name is used
    os.symlink(outside / "planted.txt", cage / ".attachments" / "nota.txt")
    b = save_attachment(p, t, "nota.txt", b"text the user sent")
    assert b["path"] == ".attachments/nota-2.txt" and b["kind"] == "text"
    assert not os.path.lexists(outside / "planted.txt")
    assert (cage / ".attachments" / "nota-2.txt").read_bytes() == b"text the user sent"
    assert os.readlink(cage / ".attachments" / "nota.txt") == str(outside / "planted.txt")  # left alone
    assert save_attachment(p, t, "foto.jpg", b"\x00\x01")["path"] == ".attachments/foto-2.jpg"
    assert save_attachment(p, t, "../../x y.pdf", b"%PDF")["path"] == ".attachments/x_y.pdf"
    # .attachments is a plain file there: refused, nothing written anywhere
    shutil.rmtree(cage / ".attachments")
    (cage / ".attachments").write_text("mine")
    with pytest.raises(Exception, match="adjunto"):
        save_attachment(p, t, "foto.jpg", b"x")
    assert (cage / ".attachments").read_text() == "mine" and list(outside.iterdir()) == []


@POSIX
def test_a_telegram_photo_lands_in_the_cage_even_with_a_planted_link(env, tmp_path, monkeypatch):
    import nesicode.server.telegram as tgmod
    from nesicode.server.telegram import TelegramBridge

    monkeypatch.setattr(tgmod, "POLL", 0.05)
    photo = b"\xff\xd8\xff\xe0 photo the user sent from Telegram"

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.startswith("/file/"):
            return httpx.Response(200, content=photo)
        method = req.url.path.rsplit("/", 1)[-1]
        result = {"getFile": {"file_path": "photos/f.jpg"}, "getUpdates": []}.get(method, {"message_id": 1})
        return httpx.Response(200, json={"ok": True, "result": result})

    e = env()
    p = e.project()
    t = e.thread(p)
    cage = thread_ws(p, t)
    outside = tmp_path / "OUTSIDE"
    outside.mkdir()
    shutil.rmtree(cage / ".attachments", ignore_errors=True)
    os.symlink(outside, cage / ".attachments")
    e.db.execute("INSERT OR REPLACE INTO telegram_links (user_id, bot_username, chat_id, active_project, "
                 "active_thread, created_at) VALUES (?, 'nesi_test_bot', 42, ?, ?, ?)",
                 (e.uid, p["id"], t["id"], time.time()))
    bridge = TelegramBridge(e.db, e.jobs, e.uid, "123:abc", api="http://tg", transport=httpx.MockTransport(handler))
    try:
        bridge.handle({"update_id": 101, "message": {"message_id": 101, "chat": {"id": 42}, "from": {"first_name": "R"},
                                                     "photo": [{"file_id": "f1", "file_size": len(photo)}]}})
        assert list(outside.iterdir()) == []
        assert (cage / ".attachments" / "foto-101.jpg").read_bytes() == photo
        run = e.db.one("SELECT * FROM runs WHERE thread_id=? ORDER BY id DESC LIMIT 1", (t["id"],))
        assert json.loads(run["attachments"])[0]["path"] == ".attachments/foto-101.jpg"
        e.wait(run["id"])
    finally:
        bridge.close()


# ====================================================================== security#5: files sent to Telegram
class _TG:
    """Records the Bot API calls of a bridge (nothing leaves the process)."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def call(self, method, **params):
        self.calls.append((method, params))
        return {"message_id": len(self.calls)}

    def sent_files(self) -> list[bytes]:
        return [data for m, prm in self.calls if m in ("sendDocument", "sendPhoto")
                for _name, data in (prm.get("_files") or {}).values()]


def _bridge(e):
    from nesicode.server.telegram import TelegramBridge

    b = TelegramBridge(e.db, e.jobs, e.uid, "123:abc", api="http://tg")
    b.tg = _TG()
    return b


@POSIX
def test_telegram_never_blocks_on_a_fifo_the_agent_shared(env, tmp_path):
    b = _bridge(env())
    ws = Workspace(tmp_path / "ws")
    os.mkfifo(ws.root / "informe.pdf")  # shared as a regular file, swapped for a FIFO before the relay read it
    _quick(lambda: b._send_file(42, ws, "informe.pdf", "", photo=False))
    assert b.tg.sent_files() == []


@POSIX
def test_telegram_never_sends_a_file_through_a_link_swapped_in_after_the_check(env, tmp_path, monkeypatch):
    b = _bridge(env())
    ws = Workspace(tmp_path / "ws")
    (ws.root / "d").mkdir()
    (ws.root / "d" / "id_rsa").write_text("inside the cage")
    outside = tmp_path / "home" / ".ssh"
    outside.mkdir(parents=True)
    (outside / "id_rsa").write_text("PRIVATE KEY (outside the cage)")
    real = ws.resolve

    def racing(rel):  # a background loop in the cage swaps d for a link right after the path was checked
        p = real(rel)
        if not (ws.root / "d").is_symlink():
            os.rename(ws.root / "d", ws.root / "d.old")
            os.symlink(outside, ws.root / "d")
        return p

    monkeypatch.setattr(ws, "resolve", racing)
    b._send_file(42, ws, "d/id_rsa", "", photo=False)
    assert not any(b"PRIVATE KEY" in data for data in b.tg.sent_files())
    # without the swap the file goes, as a document (and an image as a photo)
    monkeypatch.setattr(ws, "resolve", real)
    (ws.root / "ok.txt").write_text("hola")
    (ws.root / "shot.png").write_bytes(PNG)
    b._send_file(42, ws, "ok.txt", "", photo=False)
    b._send_file(42, ws, "shot.png", "captura", photo=True)
    assert b.tg.sent_files()[-2:] == [b"hola", PNG]
    assert [m for m, _ in b.tg.calls][-2:] == ["sendDocument", "sendPhoto"]


# ====================================================================== runtime#7: the live buffer
def test_the_live_buffer_keeps_one_step_of_deltas(env, models):
    """nesi3 reasons in every step and calls tools without text: the reasoning of finished steps is on record
    ('thinking' events), so a (re)connecting client never gets every step's deltas again — only the newest meter."""
    steps, deltas, seen = 15, 100, {}  # (the project allows 20 steps)
    e = env()

    class Reasoner(_Quiet):
        n = 0

        def step(self, on_event=None):
            Reasoner.n += 1
            if Reasoner.n > steps:  # just before the final answer: what a client connecting now gets
                rid = next(iter(e.jobs.live))
                seen["items"] = list(e.jobs.live[rid].items)
                return Turn("Hecho.", [], "end_turn", Usage(10, 10))
            for i in range(deltas):
                on_event("thinking", f"razono {i} ")
            call = ToolCall(f"c{Reasoner.n}", "search", {"pattern": f"paso{Reasoner.n}"})
            return Turn("", [call], "tool_use", Usage(10, 10), thinking="razono " * deltas)

    models["mock/reasoner"] = lambda: _Prov(Reasoner)
    p = e.project("mock/reasoner")
    r = e.post(p, e.thread(p), "trabaja")
    assert e.wait(r["id"])["status"] == "done"
    items = seen["items"]
    assert len(items) <= 3, [i["type"] for i in items]
    assert items and items[-1]["type"] == "context"  # the meter a reconnecting client shows
    assert sum(1 for ev in e.events(r["id"]) if ev["type"] == "thinking") == steps


def test_the_live_buffer_restarts_when_the_deltas_are_on_record(env):
    e = env()
    p = e.project()
    t = e.thread(p)
    rid = e.db.insert("INSERT INTO runs (project_id, thread_id, kind, task, status, created_at) "
                      "VALUES (?, ?, 'chat', 'x', 'running', ?)", (p["id"], t["id"], time.time()))
    emit = e.jobs.emitter(rid)
    live = e.jobs.live[rid]
    ctx = {"type": "context", "used": 900, "limit": 131072}
    emit({**ctx, "_persist": False})
    for i in range(50):
        emit({"type": "thinking_delta", "text": f"r{i} ", "_persist": False})
    gen = live.gen
    emit({"type": "thinking", "text": "r0 … r49"})
    assert live.gen == gen + 1 and live.items == [ctx]  # the step's reasoning is on record
    emit({"type": "tool_call", "id": "c1", "name": "list_dir", "args": {}})
    assert live.gen == gen + 2 and live.items == [ctx]
    emit({"type": "text_delta", "text": "Hola", "_persist": False})
    emit({"type": "assistant", "text": "Hola"})
    assert live.items == [ctx]


# ====================================================================== security#4: link chains in «Aplicar»
def _links(root: Path, links: dict) -> None:
    for name, target in links.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, root / name)


def _inside(root: Path, rel: str) -> bool:
    real, top = os.path.realpath(root / rel), os.path.realpath(root)
    return real == top or real.startswith(top + os.sep)


def _project(tmp_path: Path, name: str) -> Path:
    proj = tmp_path / "home" / name
    proj.mkdir(parents=True)
    (proj / "README.md").write_text("hi\n")
    (tmp_path / "secret.txt").write_text("SECRET OUTSIDE THE PROJECT\n")
    return proj


@POSIX
def test_apply_never_creates_a_link_chain_that_leaves_the_project(tmp_path):
    chain = {"a": "y/z/../../secret.txt", "y": ".", "z": ".", "b": "d/../secret.txt", "d": "."}
    # a conversation's «Aplicar»
    proj = _project(tmp_path, "p1")
    cage = tmp_path / "cage"
    shutil.copytree(proj, cage, symlinks=True)
    base = manifest(cage)
    _links(cage, chain)
    out = apply_cage(base, cage, proj)
    assert sorted(out["added"]) == ["d", "y", "z"]
    assert sorted(s.split(":")[0] for s in out["skipped"]) == ["a", "b"]
    assert not os.path.lexists(proj / "a") and not os.path.lexists(proj / "b")
    assert all(_inside(proj, rel) for rel in out["added"])
    # a harnessing run's «Aplicar»
    proj2 = _project(tmp_path, "p2")
    before, after = tmp_path / "before", tmp_path / "after"
    shutil.copytree(proj2, before, symlinks=True)
    shutil.copytree(proj2, after, symlinks=True)
    _links(after, chain)
    res = apply_rollout(before, after, proj2)
    assert sorted(res["added"]) == ["d", "y", "z"]
    assert sorted((c["path"], c["reason"], c["side_file"]) for c in res["conflicts"]) == [
        ("a", "unsafe", None), ("b", "unsafe", None)]
    assert not os.path.lexists(proj2 / "a") and not os.path.lexists(proj2 / "b")


@POSIX
def test_apply_checks_its_links_again_once_all_of_them_exist(tmp_path):
    """A link that is inside in the conversation's copy and inside when written can still leave the project once a later
    link of the same apply exists, through a link the user made in the project meanwhile: it is taken out again."""
    proj = _project(tmp_path, "p3")
    cage = tmp_path / "cage3"
    shutil.copytree(proj, cage, symlinks=True)
    base = manifest(cage)
    _links(proj, {"z": "."})  # the user's own link, made after the conversation copied the project
    _links(cage, {"a": "y/z/../secret.txt", "y": "."})
    out = apply_cage(base, cage, proj)
    assert out["added"] == ["y"] and [s.split(":")[0] for s in out["skipped"]] == ["a"]
    assert not os.path.lexists(proj / "a") and os.readlink(proj / "z") == "."
    # the same through a harnessing run
    proj4 = _project(tmp_path, "p4")
    before, after = tmp_path / "before4", tmp_path / "after4"
    shutil.copytree(proj4, before, symlinks=True)
    shutil.copytree(proj4, after, symlinks=True)
    _links(proj4, {"z": "."})
    _links(after, {"a": "y/z/../secret.txt", "y": "."})
    res = apply_rollout(before, after, proj4)
    assert res["added"] == ["y"] and [(c["path"], c["reason"]) for c in res["conflicts"]] == [("a", "unsafe")]
    assert not os.path.lexists(proj4 / "a")


@POSIX
def test_apply_keeps_ordinary_relative_links(tmp_path):
    """.bin links (../pkg/bin/x) through a linked package (pnpm's layout; node_modules itself is never applied), links
    into sibling folders, a link to a linked folder: all still apply."""
    proj = _project(tmp_path, "p5")
    cage = tmp_path / "cage5"
    shutil.copytree(proj, cage, symlinks=True)
    base = manifest(cage)
    real = cage / "vendor" / ".store" / "pkg@1" / "pkg" / "bin"
    real.mkdir(parents=True)
    (real / "cli.js").write_text("#!/usr/bin/env node\n")
    _links(cage, {"vendor/pkg": ".store/pkg@1/pkg", "vendor/.bin/pkg": "../pkg/bin/cli.js",
                  "docs/readme": "../README.md", "latest": "vendor/pkg"})
    out = apply_cage(base, cage, proj)
    assert out["skipped"] == []
    for rel in ("vendor/pkg", "vendor/.bin/pkg", "docs/readme", "latest"):
        assert rel in out["added"] and _inside(proj, rel)
    assert (proj / "vendor" / ".bin" / "pkg").read_text() == "#!/usr/bin/env node\n"


# ====================================================================== platform#5: `nesicode desktop` from source
@pytest.mark.skipif(os.name == "nt", reason="the app folder is under %APPDATA% there (same code path)")
def test_nesicode_desktop_from_source_uses_the_app_data_folder(tmp_path):
    """`uv run nesicode desktop` must use the desktop app's folder and its .env, like the packaged app — not ./data
    and the current folder's .env (the CLI used to load the settings before the desktop could say it is the desktop)."""
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    cwd.mkdir()
    app = (home / "Library" / "Application Support" / "nesicode" if sys.platform == "darwin"
           else tmp_path / "xdg" / "nesicode")
    app.mkdir(parents=True)
    (app / ".env").write_text("NESICODE_MAX_JOBS=7\nNESICODE_EXECUTOR_MODEL=mock/demo\n")
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent('''
        import json, sys, types
        calls = {}
        webview = types.ModuleType("webview")  # a window that closes at once
        webview.windows = []
        webview.create_window = lambda *a, **k: calls.setdefault("window", True)
        webview.start = lambda *a, **k: calls.setdefault("start", k)
        sys.modules["webview"] = webview
        import nesicode.cli as cli
        import nesicode.desktop as desktop
        desktop._adopt_login_path = lambda: None  # (no login shell in a test)
        desktop.PREFERRED_PORT = int(sys.argv[1])
        cli.main(["desktop"])
        import nesicode.config as config
        s = config.settings
        print("RESULT " + json.dumps({"desktop": bool(config.IS_DESKTOP), "data_dir": str(s.data_dir),
                                       "app": str(config.app_data_dir()), "max_jobs": s.max_jobs,
                                       "executor": s.default_executor_model,
                                       "storage": calls.get("start", {}).get("storage_path", "")}))
    '''))
    import nesicode

    envv = {k: v for k, v in os.environ.items() if not k.startswith("NESICODE_") and not k.endswith("_API_KEY")}
    envv.update(HOME=str(home), XDG_DATA_HOME=str(tmp_path / "xdg"), PYTHONPATH=str(Path(nesicode.__file__).parents[1]),
                NESICODE_VAULT_KEY=base64.b64encode(os.urandom(32)).decode(), NESICODE_NO_TELEGRAM="1",
                NESICODE_NO_SCHEDULER="1")
    proc = subprocess.run([sys.executable, str(probe), "8671"], cwd=cwd, env=envv, capture_output=True, text=True,
                          timeout=120)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, (proc.returncode, proc.stdout[-2000:], proc.stderr[-4000:])
    got = json.loads(line[len("RESULT "):])
    assert got["desktop"] is True
    assert Path(got["data_dir"]) == Path(got["app"]) / "data" and Path(got["app"]).resolve() == app.resolve()
    assert (got["max_jobs"], got["executor"]) == (7, "mock/demo")  # the app folder's .env was read
    assert Path(got["storage"]).parent.resolve() == app.resolve()
    assert (app / "data" / "nesicode.db").exists() and not (cwd / "data").exists()
