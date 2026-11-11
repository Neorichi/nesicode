"""v0.4.7, server runtime: a conversation's «Aplicar» is three-way; git history made in its copy is reported, never
silently lost; auto-harnessing re-runs only self-contained turns, never adopts or learns from a follow-up and never
runs rollouts without a step cap; a full context is not blamed on a step limit; a very long run still shows its end
when reopened; harnessing counts as unattended work in the worker reservation; the project's verification command
reaches the chat (and bot) agent.

Offline: the mock model and small fakes, the app's TestClient (conftest-isolated data) or a private Jobs + database."""

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import nesicode.server.jobs as jobs_mod
from nesicode.agent import copy_workspace
from nesicode.agent.loop import AgentResult
from nesicode.agent.workspace import apply_cage, manifest
from nesicode.harness import base_harness
from nesicode.llm.base import Conversation, Provider, Turn, Usage
from nesicode.server.app import app, db, jobs
from nesicode.server.chatops import ensure_thread, post_message
from nesicode.server.db import DB
from nesicode.server.jobs import FINAL, Jobs, needs_improvement, thread_ws

ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")
_UIDS = iter(range(970_000, 980_000))


# ------------------------------------------------------------------ helpers
def _user(prefix="v"):
    c = TestClient(app)
    r = c.post("/api/auth/register", json={"email": f"{prefix}{time.time_ns()}@example.com", "password": "secret123"})
    assert r.status_code == 200, r.text
    return c


def _project(c, **kw):
    r = c.post("/api/projects", json={"name": "P", "executor_model": "mock/demo", "improver_model": "mock/demo", **kw})
    assert r.status_code == 200, r.text
    p = r.json()
    Path(p["workspace"]).mkdir(parents=True, exist_ok=True)
    return p


def _thread(c, p):
    t = c.post(f"/api/projects/{p['id']}/threads", json={}).json()
    row = db.one("SELECT * FROM threads WHERE id=?", (t["id"],))
    return t, thread_ws(db.one("SELECT * FROM projects WHERE id=?", (p["id"],)), row)


def _wait(c, rid, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = c.get(f"/api/runs/{rid}").json()
        if run["status"] not in ("queued", "running", "waiting"):
            return run
        time.sleep(0.1)
    raise AssertionError(f"run {rid} did not finish")


def _chat(c, tid, text):
    return _wait(c, c.post(f"/api/threads/{tid}/messages", json={"text": text}).json()["id"])


def _git(cwd, *args):
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}
    p = subprocess.run([GIT, "-C", str(cwd), *args], capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout


class _Quiet(Conversation):
    def add_user(self, text): pass  # noqa: E704
    def add_assistant_text(self, text): pass  # noqa: E704
    def add_tool_results(self, results): pass  # noqa: E704


class _Prov(Provider):
    name = "fake"

    def __init__(self, conv_cls):
        self.conv_cls = conv_cls

    def conversation(self, model, system, tools, options):
        return self.conv_cls(model, system, tools, options)


class _BotsStub:
    """What the bot manager adds to a bot run (enough to run one in a private Jobs)."""

    def agent_tools(self, bot, run_id):
        return {}

    def system_section(self, bot, trigger="", run_id=None):
        return "You are a bot."


class _Env:
    """A private Jobs on its own database (users with high ids: their folders never meet the app's)."""

    def __init__(self, tmp_path, workers):
        self.db = DB(tmp_path / f"jobs-{time.time_ns()}.db")
        self.jobs = Jobs(self.db, max_workers=workers)
        self.jobs.bots = _BotsStub()
        self.uid = next(_UIDS)
        self.db.execute("INSERT INTO users (id, email, name, password_hash, created_at) VALUES (?, ?, 'T', 'x', ?)",
                        (self.uid, f"u{self.uid}@example.com", time.time()))

    def project(self, **kw) -> dict:
        pid = self.db.insert("INSERT INTO projects (user_id, name, executor_model, improver_model, max_steps, check_command, "
                             "created_at) VALUES (?, 'P', 'mock/demo', 'mock/demo', 20, ?, ?)",
                             (self.uid, kw.get("check_command", ""), time.time()))
        self.db.execute("INSERT INTO harness_versions (project_id, version, files, source, created_at) VALUES (?, 1, ?, 'base', ?)",
                        (pid, json.dumps(base_harness()), time.time()))
        return self.db.one("SELECT * FROM projects WHERE id=?", (pid,))

    def bot(self, project, model="") -> dict:
        bid = self.db.insert("INSERT INTO bots (user_id, project_id, name, model, policy, created_at, updated_at) "
                             "VALUES (?, ?, 'B', ?, 'auto', ?, ?)", (self.uid, project["id"], model, time.time(), time.time()))
        return self.db.one("SELECT * FROM bots WHERE id=?", (bid,))

    def post(self, project, text, **kw) -> dict:
        t = ensure_thread(self.db, project, self.uid, "chat")
        return post_message(self.db, self.jobs, project, t, text, [], **kw)

    def run(self, rid) -> dict:
        return self.db.one("SELECT * FROM runs WHERE id=?", (rid,))

    def wait(self, rid, timeout=20) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (run := self.run(rid)) and run["status"] in FINAL:
                return run
            time.sleep(0.05)
        raise AssertionError(f"run {rid} did not finish: {self.run(rid)}")


@pytest.fixture()
def env(tmp_path):
    made = []

    def make(workers=2):
        e = _Env(tmp_path, workers)
        made.append(e)
        return e

    yield make
    for e in made:
        e.jobs.shutdown(wait=3)


@pytest.fixture()
def models(monkeypatch):
    """Extra fake models for the agent loop: models['mock/x'] = factory returning a Provider."""
    import nesicode.agent.loop as loop

    real, table = loop.get_provider, {}
    monkeypatch.setattr(loop, "get_provider", lambda spec, keys=None: (table[spec](), spec.split("/", 1)[1])
                        if spec in table else real(spec, keys))
    return table


@pytest.fixture()
def prompts(monkeypatch):
    """The system prompt of every mock conversation started meanwhile."""
    from nesicode.llm import mock

    seen: list[str] = []
    real = mock.MockConversation.__init__

    def spy(self, model, system, tools, options):
        seen.append(system)
        real(self, model, system, tools, options)

    monkeypatch.setattr(mock.MockConversation, "__init__", spy)
    return seen


class _Gate:
    """Something that lasts until the test releases it (it holds its worker, like a long reply)."""

    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Semaphore(0)

    def wait_entered(self, timeout=10):
        assert self.entered.acquire(timeout=timeout), "the gated work never started"

    def hold(self):
        self.entered.release()
        self.release.wait(20)

    def provider(self):
        gate = self

        class Conv(_Quiet):
            def step(self, on_event=None):
                gate.hold()
                return Turn("ok", [], "end_turn", Usage(1, 1))

        return _Prov(Conv)


# ====================================================================== runtime-1: a three-way «Aplicar»
def test_applying_a_second_conversation_never_reverts_what_the_first_applied():
    """Two conversations fix different functions of app.py; «Aplicar» A, then B: A's fix stays, B's version of the file
    goes next to it (app.py.nesicode-new), the conflict stays listed in B's Changes until it is resolved."""
    c = _user("r1a")
    p = _project(c)
    ws = Path(p["workspace"])
    (ws / "app.py").write_text("def a():\n    return 1  # bug A\n\n\ndef b():\n    return 2  # bug B\n")
    ta, ca = _thread(c, p)
    tb, cb = _thread(c, p)
    (ca / "app.py").write_text((ca / "app.py").read_text().replace("# bug A", "# FIX FROM A"))
    (cb / "app.py").write_text((cb / "app.py").read_text().replace("# bug B", "# FIX FROM B"))
    (cb / "nuevo.py").write_text("x = 1\n")  # a change of B nobody else touched: it applies as always
    ra = c.post(f"/api/threads/{ta['id']}/apply").json()
    assert ra["modified"] == ["app.py"] and ra["conflicts"] == []
    # before B applies, its Changes panel already says app.py changed in the project too
    assert c.get(f"/api/threads/{tb['id']}/changes").json()["conflicts"] == [{"path": "app.py", "reason": "changed"}]
    rb = c.post(f"/api/threads/{tb['id']}/apply")
    assert rb.status_code == 200, rb.text
    body = rb.json()
    assert body["conflicts"] == [{"path": "app.py", "reason": "changed", "side_file": "app.py.nesicode-new"}]
    assert body["added"] == ["nuevo.py"] and body["modified"] == [] and (ws / "nuevo.py").read_text() == "x = 1\n"
    text = (ws / "app.py").read_text()
    assert "FIX FROM A" in text and "FIX FROM B" not in text
    assert "FIX FROM B" in (ws / "app.py.nesicode-new").read_text()
    ch = c.get(f"/api/threads/{tb['id']}/changes").json()
    assert (ch["added"], ch["modified"], ch["conflicts"]) == ([], ["app.py"], [{"path": "app.py", "reason": "changed"}])
    # the user takes B's version by hand: the next «Aplicar» finds it there, and nothing is left pending
    shutil.move(ws / "app.py.nesicode-new", ws / "app.py")
    again = c.post(f"/api/threads/{tb['id']}/apply").json()
    assert again["modified"] == ["app.py"] and again["conflicts"] == [] and not (ws / "app.py.nesicode-new").exists()
    ch = c.get(f"/api/threads/{tb['id']}/changes").json()
    assert (ch["added"], ch["modified"], ch["removed"], ch["conflicts"]) == ([], [], [], [])


def test_apply_keeps_what_the_user_changed_in_the_project_meanwhile():
    """The user keeps editing their folder while the conversation works: their README section and the new line of a
    file the agent deleted survive «Aplicar»; a file only the agent changed applies."""
    c = _user("r1b")
    p = _project(c)
    ws = Path(p["workspace"])
    (ws / "README.md").write_text("# Proyecto\n")
    (ws / "notes.txt").write_text("nota 1\n")
    (ws / "keep.txt").write_text("igual\n")
    t, cage = _thread(c, p)
    (cage / "README.md").write_text("# Proyecto\n\nSección del agente\n")
    (cage / "notes.txt").unlink()
    (cage / "keep.txt").write_text("cambiado por el agente\n")
    (ws / "README.md").write_text("# Proyecto\n\n## Sección del usuario\n")
    (ws / "notes.txt").write_text("nota 1\nnota 2 del usuario\n")
    body = c.post(f"/api/threads/{t['id']}/apply").json()
    assert body["modified"] == ["keep.txt"] and body["removed"] == [] and body["skipped"] == []
    assert sorted((x["path"], x["reason"], x["side_file"]) for x in body["conflicts"]) == [
        ("README.md", "changed", "README.md.nesicode-new"), ("notes.txt", "changed", None)]
    assert (ws / "README.md").read_text() == "# Proyecto\n\n## Sección del usuario\n"
    assert (ws / "README.md.nesicode-new").read_text() == "# Proyecto\n\nSección del agente\n"
    assert (ws / "notes.txt").read_text() == "nota 1\nnota 2 del usuario\n"
    assert (ws / "keep.txt").read_text() == "cambiado por el agente\n"
    ch = c.get(f"/api/threads/{t['id']}/changes").json()
    assert (ch["modified"], ch["removed"]) == (["README.md"], ["notes.txt"])  # still pending: never silently dropped


@pytest.mark.skipif(os.name == "nt", reason="symbolic links")
def test_apply_keeps_a_link_the_user_made_where_the_conversation_changed_a_file():
    """A link at a changed path used to be removed before the copy: now it goes only when it is what the conversation
    started from — the user's own link (made meanwhile) stays, the conversation's file beside it."""
    c = _user("r1c")
    p = _project(c)
    ws = Path(p["workspace"])
    (ws / "config.json").write_text('{"env": "dev"}\n')
    (ws / "shared.json").write_text('{"env": "shared"}\n')
    t, cage = _thread(c, p)
    (cage / "config.json").write_text('{"env": "agent"}\n')
    (ws / "config.json").unlink()
    (ws / "config.json").symlink_to("shared.json")
    body = c.post(f"/api/threads/{t['id']}/apply").json()
    assert body["conflicts"] == [{"path": "config.json", "reason": "changed", "side_file": "config.json.nesicode-new"}]
    assert os.readlink(ws / "config.json") == "shared.json" and (ws / "shared.json").read_text() == '{"env": "shared"}\n'
    assert (ws / "config.json.nesicode-new").read_text() == '{"env": "agent"}\n'


def test_apply_refuses_while_the_project_folder_is_unavailable():
    """An external project folder on a disk that is not connected: nothing is judged or applied (a removal used to look
    done and left the Changes panel for good), and everything applies once the folder is back."""
    c = _user("r1e")
    p = _project(c)
    ws = Path(p["workspace"])
    (ws / "viejo.txt").write_text("x\n")
    t, cage = _thread(c, p)
    (cage / "viejo.txt").unlink()
    (cage / "nuevo.txt").write_text("n\n")
    away = ws.with_name(ws.name + "-desconectado")
    ws.rename(away)
    try:
        r = c.post(f"/api/threads/{t['id']}/apply")
        assert r.status_code == 409 and "no está disponible" in r.json()["detail"]
        ch = c.get(f"/api/threads/{t['id']}/changes").json()
        assert (ch["added"], ch["removed"], ch["conflicts"]) == (["nuevo.txt"], ["viejo.txt"], [])
    finally:
        away.rename(ws)
    body = c.post(f"/api/threads/{t['id']}/apply").json()
    assert (body["added"], body["removed"], body["conflicts"]) == (["nuevo.txt"], ["viejo.txt"], [])
    assert not (ws / "viejo.txt").exists() and (ws / "nuevo.txt").read_text() == "n\n"


def _case_insensitive(d: Path) -> bool:
    (d / "CaseProbe").write_text("x")
    try:
        return (d / "caseprobe").exists()
    finally:
        (d / "CaseProbe").unlink()


def test_a_case_only_rename_never_overwrites_the_users_edit(tmp_path):
    """button.tsx -> Button.tsx in the conversation while the user edited button.tsx: on a case-insensitive disk the two
    names are one file, and writing Button.tsx would replace the user's edit."""
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem")
    proj, cage = tmp_path / "project", tmp_path / "cage"
    (proj / "src").mkdir(parents=True)
    (proj / "src" / "button.tsx").write_text("v1\n")
    copy_workspace(proj, cage)
    base = manifest(cage)
    os.rename(cage / "src" / "button.tsx", cage / "src" / "Button.tsx")
    (proj / "src" / "button.tsx").write_text("v2 del usuario\n")
    out = apply_cage(base, cage, proj)
    assert sorted(os.listdir(proj / "src")) == ["Button.tsx.nesicode-new", "button.tsx"]
    assert (proj / "src" / "button.tsx").read_text() == "v2 del usuario\n"
    assert {(x["path"], x["reason"]) for x in out["conflicts"]} == {("src/button.tsx", "changed"), ("src/Button.tsx", "exists")}


# ====================================================================== runtime-2: git history in the copy
@pytest.mark.skipif(not GIT, reason="needs git")
def test_commits_branches_and_stashes_made_in_a_conversation_are_reported():
    c = _user("r2")
    p = _project(c)
    ws = Path(p["workspace"])
    _git(ws, "init", "-q", "-b", "main")
    (ws / "README.md").write_text("hola\n")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", "inicial")
    t, cage = _thread(c, p)
    assert c.get(f"/api/threads/{t['id']}/changes").json()["git"] is None  # nothing done in it yet
    (cage / "README.md").write_text("hola, mejorado\n")
    _git(cage, "commit", "-q", "-am", "agente: mejora README")
    ch = c.get(f"/api/threads/{t['id']}/changes").json()
    assert ch["modified"] == ["README.md"]
    assert ch["git"] == {"commits": 1, "branches": ["main"], "stash": False, "uncounted": False}
    assert c.post(f"/api/threads/{t['id']}/apply").json()["modified"] == ["README.md"]
    assert _git(ws, "log", "--format=%s").split() == ["inicial"]  # the project's history is untouched…
    assert c.get(f"/api/threads/{t['id']}/changes").json()["git"]["commits"] == 1  # …and the panel keeps saying so
    # work committed on a side branch, then back on main: no file to apply at all, but it is not silently lost
    _git(cage, "checkout", "-q", "-b", "fix")
    (cage / "fix.py").write_text("x = 1\n")
    _git(cage, "add", "fix.py")
    _git(cage, "commit", "-q", "-m", "fix")
    _git(cage, "checkout", "-q", "main")
    ch = c.get(f"/api/threads/{t['id']}/changes").json()
    assert (ch["added"], ch["modified"]) == ([], [])
    assert ch["git"] == {"commits": 2, "branches": ["fix", "main"], "stash": False, "uncounted": False}
    (cage / "README.md").write_text("a medias\n")
    _git(cage, "stash", "-q")
    assert c.get(f"/api/threads/{t['id']}/changes").json()["git"]["stash"] is True
    # `git gc` packs the new commits: they can no longer be counted, but the moved branches are still reported
    _git(cage, "stash", "drop", "-q")
    _git(cage, "gc", "-q")
    assert c.get(f"/api/threads/{t['id']}/changes").json()["git"] == {
        "commits": 0, "branches": ["fix", "main"], "stash": False, "uncounted": True}


def test_the_agent_is_told_about_git_and_the_projects_verification_command(prompts):
    """An isolated conversation's agent learns that «Aplicar» never carries git history, and the project's
    «Comando de verificación» reaches it (it was only ever used by harnessing)."""
    c = _user("h3")
    cmd = "PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/unit"
    p = _project(c, check_command=cmd)
    t, _ = _thread(c, p)
    assert _chat(c, t["id"], "arregla el bug")["status"] == "done"
    system = next(s for s in prompts if "arregla" not in s and "# Where your files go" in s)
    assert "# Verification" in system and cmd in system
    assert "never git history" in system and "don't commit unless the user asks" in system


def test_a_bot_run_gets_the_projects_verification_command(env, prompts):
    e = env(2)
    p = e.project(check_command="npm test --silent")
    b = e.bot(p)
    r = e.post(p, "revisa el proyecto", bot_id=b["id"], trigger="cron#1")
    assert e.wait(r["id"])["status"] == "done"
    assert any("You are a bot." in s and "npm test --silent" in s and "# Verification" in s for s in prompts)


# ====================================================================== runtime-3: follow-ups and unattended rollouts
def test_auto_harness_never_reruns_a_follow_up_turn(monkeypatch):
    c = _user("r3a")
    p = _project(c)
    launched: list[str] = []
    monkeypatch.setattr(jobs, "improve_from_chat", lambda run, project, *, auto, reason="": launched.append(run["task"]) or 999)
    c.put("/api/me/settings", json={"auto_harness": True, "auto_harness_min_steps": 3, "auto_harness_daily": 0})
    t, _ = _thread(c, p)
    first = _chat(c, t["id"], "Escribe el plan del proyecto en plan.md")
    assert [x["type"] for x in first["events"] if x["type"].startswith("auto_harness")] == ["auto_harness"]
    second = _chat(c, t["id"], "continúa")
    skipped = [x for x in second["events"] if x["type"] == "auto_harness_skipped"]
    assert launched == ["Escribe el plan del proyecto en plan.md"]
    assert skipped and "continúa la conversación" in skipped[0]["why"]


def test_improving_a_follow_up_carries_the_conversation_and_adopts_or_learns_nothing():
    c = _user("r3b")
    p = _project(c)
    t, _ = _thread(c, p)
    _chat(c, t["id"], "Escribe el plan del proyecto en plan.md")
    follow = _chat(c, t["id"], "continúa")
    hr = c.post(f"/api/runs/{follow['id']}/improve").json()
    assert hr["params"]["adopt_harness"] is False and hr["params"]["learn"] is False
    assert hr["params"]["source"]["follow_up"] is True
    assert "User: Escribe el plan del proyecto en plan.md" in hr["task"] and hr["task"].endswith("\ncontinúa")
    done = _wait(c, hr["id"])
    assert done["status"] == "done", done["error"]
    assert done["result"]["adopted_version"] is None
    assert not db.all("SELECT id FROM lessons WHERE run_id=?", (hr["id"],))
    assert c.get(f"/api/projects/{p['id']}").json()["harness_version"] == p["harness_version"]


def test_harnessing_rollouts_never_run_without_a_step_cap(monkeypatch):
    seen: list[int] = []

    class Recorder:
        def __init__(self, cfg, **_kw):
            seen.append(cfg.max_steps)

        def run(self):
            raise RuntimeError("cancelled")

    monkeypatch.setattr(jobs_mod, "HarnessingRun", Recorder)
    c = _user("r3c")
    for cap in (0, 40):
        p = _project(c, max_steps=cap)
        _wait(c, c.post(f"/api/projects/{p['id']}/harnessing", json={"task": "x", "k": 1}).json()["id"])
    assert seen == [150, 40]  # (unattended: nobody can press Stop on a rollout)


# ====================================================================== runtime-5: why a run stopped
def _result(status, steps, stop=""):
    res = AgentResult("…", status, steps, Usage(), [])
    if stop:
        res.stop_reason = stop  # (AgentResult.stop_reason, from the agent loop)
    return res


def test_a_full_context_with_no_step_cap_is_not_blamed_on_a_step_limit():
    prefs = {"auto_harness_min_steps": 20, "auto_harness_min_errors": 3}
    assert needs_improvement(_result("max_steps", 2), 0, prefs) == ""  # no cap, 2 steps: the thresholds decide
    assert needs_improvement(_result("max_steps", 2, "context_full"), 60, prefs) == ""
    assert needs_improvement(_result("max_steps", 30, "context_full"), 0, prefs) == "se llenó el contexto tras 30 pasos"
    assert needs_improvement(_result("max_steps", 60, "step_cap"), 60, prefs) == "agotó el límite de pasos"
    assert needs_improvement(_result("max_steps", 5), 60, prefs) == "agotó el límite de pasos"  # (a loop that doesn't say)
    assert needs_improvement(_result("done", 25), 0, prefs) == "necesitó 25 pasos"


def test_a_run_remembers_why_it_stopped(monkeypatch):
    from nesicode.agent.loop import Agent

    def run(self, task, history=None):
        self.emit({"type": "assistant", "text": "llevo la mitad"})
        return _result("max_steps", 7, "context_full")

    monkeypatch.setattr(Agent, "run", run)
    c = _user("r5")
    p = _project(c, max_steps=0)
    t, _ = _thread(c, p)
    assert _chat(c, t["id"], "migra todo")["status"] == "max_steps"
    view = c.get(f"/api/threads/{t['id']}").json()["runs"][0]
    assert view["summary"]["stop_reason"] == "context_full" and view["summary"]["steps"] == 7


# ====================================================================== runtime-6: a very long run, reopened
def test_reopening_a_very_long_run_shows_its_end_and_final_answer():
    c = _user("r6")
    p = _project(c)
    t, _ = _thread(c, p)
    now = time.time()
    rid = db.insert("INSERT INTO runs (project_id, thread_id, kind, task, status, final_text, created_at, finished_at) "
                    "VALUES (?, ?, 'chat', 'migra todo', 'done', 'RESPUESTA FINAL: todo migrado', ?, ?)",
                    (p["id"], t["id"], now, now))
    events = []
    for i in range(1, 1801):
        events += [{"type": "tool_call", "id": f"c{i}", "name": "bash", "args": {"command": f"echo {i}"}},
                   {"type": "tool_result", "id": f"c{i}", "name": "bash", "output": str(i), "is_error": False},
                   {"type": "assistant", "text": f"progreso {i}"}]
    events += [{"type": "assistant", "text": "RESPUESTA FINAL: todo migrado"}, {"type": "done", "status": "done", "steps": 1800}]
    with db.lock:
        db.conn.execute("BEGIN")
        db.conn.executemany("INSERT INTO events (run_id, seq, data) VALUES (?, ?, ?)",
                            [(rid, n, json.dumps({**e, "ts": now})) for n, e in enumerate(events, 1)])
        db.conn.execute("COMMIT")
    for got in (c.get(f"/api/threads/{t['id']}").json()["runs"][0]["events"], c.get(f"/api/runs/{rid}").json()["events"]):
        assert len(got) == 5_001 and got[0]["seq"] == 1 and got[499]["seq"] == 500
        assert got[500] == {"type": "omitted", "count": len(events) - 5_000}
        assert got[-2]["text"] == "RESPUESTA FINAL: todo migrado" and got[-1]["type"] == "done"
        assert got[-1]["seq"] == len(events)  # a stream resumes right after it


# ====================================================================== runtime-7: unattended work and the last worker
def test_a_harnessing_run_and_bot_runs_always_leave_a_worker_for_chats(env, monkeypatch):
    gate = _Gate()

    class GatedRun:
        def __init__(self, cfg, **_kw):
            pass

        def run(self):
            gate.hold()
            raise RuntimeError("cancelled")

    monkeypatch.setattr(jobs_mod, "HarnessingRun", GatedRun)
    e = env(2)  # one worker is always kept for chats
    p = e.project()
    hr = e.jobs.create_harnessing(p, "tarea", {"k": 1})
    gate.wait_entered()
    bot = e.post(p, "tarea programada", bot_id=e.bot(p)["id"], trigger="cron#1")
    try:
        assert e.run(bot["id"])["status"] == "waiting"  # held: the harnessing run holds the unattended worker
        chat = e.post(p, "hola")
        assert e.wait(chat["id"], timeout=10)["status"] == "done"
    finally:
        gate.release.set()
    assert e.wait(hr)["status"] == "cancelled" and e.wait(bot["id"])["status"] == "done"


def test_a_harnessing_run_waits_while_bots_hold_the_unattended_workers(env, models, monkeypatch):
    gate = _Gate()
    models["mock/gate"] = gate.provider
    ran: list[str] = []

    class QuickRun:
        def __init__(self, cfg, **_kw):
            self.task = cfg.task

        def run(self):
            ran.append(self.task)
            raise RuntimeError("cancelled")

    monkeypatch.setattr(jobs_mod, "HarnessingRun", QuickRun)
    e = env(2)
    p = e.project()
    bot = e.post(p, "tarea programada", bot_id=e.bot(p, model="mock/gate")["id"], trigger="cron#1")
    gate.wait_entered()
    hr = e.jobs.create_harnessing(p, "mejora", {"k": 1})
    try:
        time.sleep(0.3)
        assert e.run(hr)["status"] == "queued" and ran == []  # held back, not holding a worker
        chat = e.post(p, "hola")
        assert e.wait(chat["id"], timeout=10)["status"] == "done"
    finally:
        gate.release.set()
    assert e.wait(bot["id"])["status"] == "done"
    assert e.wait(hr)["status"] == "cancelled" and ran == ["mejora"]  # it started once the bot run ended


# ====================================================================== the web UI side (node + esbuild)
WEB = ROOT / "web"
SRC = WEB / "src"
NODE = shutil.which("node")
ESBUILD = WEB / "node_modules" / "esbuild"
needs_node = pytest.mark.skipif(not NODE or not ESBUILD.is_dir(), reason="needs node and web/node_modules (esbuild)")
STUB_MARKDOWN = "export default function Markdown({ text }: { text: string }) { return <div className=\"md\">{text}</div> }\n"


def run_ts(tmp_path: Path, body: str) -> dict:
    """Bundle `body` for node (with `L` = logic.ts, setLang, html(node) = static HTML in a router) and run it."""
    entry = tmp_path / "entry.tsx"
    entry.write_text(
        f"import * as L from {json.dumps(str(SRC / 'logic.ts'))}\n"
        f"import {{ setLang }} from {json.dumps(str(SRC / 'i18n.ts'))}\n"
        "import { renderToStaticMarkup } from 'react-dom/server'\n"
        "import { MemoryRouter, Route, Routes } from 'react-router-dom'\n"
        "const html = (node: unknown) => renderToStaticMarkup(<MemoryRouter initialEntries={['/p/1/c/2']}><Routes>"
        "<Route path='/p/:pid/c/:tid' element={node as any} /></Routes></MemoryRouter>)\n"
        "void html; void setLang\n"
        "const out: Record<string, unknown> = {}\n"
        f"{body}\nconsole.log(JSON.stringify(out))\n", encoding="utf-8")
    stub = tmp_path / "stub_markdown.tsx"
    stub.write_text(STUB_MARKDOWN, encoding="utf-8")
    bundle, build = tmp_path / "bundle.cjs", tmp_path / "build.cjs"
    build.write_text(
        f"require({json.dumps(str(ESBUILD))}).build({{entryPoints: [{json.dumps(str(entry))}], bundle: true, "
        f"platform: 'node', format: 'cjs', outfile: {json.dumps(str(bundle))}, logLevel: 'error', jsx: 'automatic', "
        f"nodePaths: [{json.dumps(str(WEB / 'node_modules'))}], plugins: [{{name: 'stub-markdown', setup(b) {{ "
        f"b.onResolve({{filter: /[\\\\/]Markdown$/}}, () => ({{path: {json.dumps(str(stub))}}})) }}}}]}})"
        ".catch((e) => { console.error(e); process.exit(1) })\n", encoding="utf-8")
    for args in ([NODE, str(build)], [NODE, str(bundle)]):
        p = subprocess.run(args, capture_output=True, encoding="utf-8", timeout=90, cwd=tmp_path)
        assert p.returncode == 0, p.stderr[-4000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@needs_node
def test_the_ui_shows_a_long_runs_end_its_final_answer_and_why_it_stopped(tmp_path):
    final = "RESPUESTA FINAL: todo migrado"
    note = ("[El contexto del modelo está lleno y no se puede compactar más: el trabajo se detuvo aquí. Escribe «continúa» "
            "para seguir desde el historial guardado.]")  # (the agent loop's own note)
    out = run_ts(tmp_path, f"""
      import Transcript from {json.dumps(str(SRC / 'components' / 'Transcript.tsx'))}
      import {{ statusLabel }} from {json.dumps(str(SRC / 'api.ts'))}
      const said = [{{ type: 'assistant', text: 'progreso 1' }}, {{ type: 'omitted', count: 432 }}]
      out.cut = L.unshownFinal(said as any, {json.dumps(final)})
      out.carried = L.unshownFinal([...said, {{ type: 'assistant', text: {json.dumps(final)} }}] as any, {json.dumps(final)})
      out.parts = L.unshownFinal([{{ type: 'assistant', text: 'uno ' }}, {{ type: 'assistant', text: 'dos' }}] as any, 'uno dos')
      out.note = L.unshownFinal([{{ type: 'assistant', text: 'llevo la mitad' }}] as any, {json.dumps("llevo la mitad\n\n" + note)})
      // the agent loop draws that note itself ('stopped'): never twice
      const ended = [{{ type: 'assistant', text: 'llevo la mitad' }}, {{ type: 'stopped', reason: 'context_full', text: {json.dumps(note)} }},
                     {{ type: 'done', status: 'max_steps', steps: 7, stop_reason: 'context_full' }}]
      out.noteOnce = L.unshownFinal(ended as any, {json.dumps("llevo la mitad\n\n" + note)})
      out.ended = html(<Transcript events={{ended as any}} status="max_steps" finalText={{{json.dumps("llevo la mitad\n\n" + note)}}} />)
      out.page = html(<Transcript events={{said as any}} status="done" finalText={{{json.dumps(final)}}} />)
      out.full = statusLabel('max_steps', 'context_full')
      out.cap = statusLabel('max_steps', 'step_cap')
      out.stuck = statusLabel('max_steps', 'stuck')
      out.odd = statusLabel('max_steps', 'nuevo')
      setLang('en')
      out.fullEn = statusLabel('max_steps', 'context_full')
      out.omittedEn = html(<Transcript events={{said as any}} status="done" />)
      out.endedEn = html(<Transcript events={{ended as any}} status="max_steps" />)
      setLang('es')
    """)
    assert out["cut"] == final and out["carried"] == "" and out["parts"] == "" and out["note"] == note
    assert final in out["page"] and "432 entradas intermedias de esta respuesta no se muestran" in out["page"]
    assert out["full"] == "Contexto lleno" and out["cap"] == "Límite de pasos" and out["fullEn"] == "Context full"
    assert out["stuck"] == "Detenido: se repetía sin avanzar" and out["odd"] == "Límite de pasos"
    assert "432 intermediate entries of this reply are not shown here" in out["omittedEn"]
    assert out["noteOnce"] == "" and out["ended"].count("el trabajo se detuvo aquí") == 1
    assert "Contexto lleno" in out["ended"]  # (from the 'done' event when the page has no summary yet)
    assert "the work stopped here" in out["endedEn"] and "Context full" in out["endedEn"]


@needs_node
def test_the_ui_words_a_conversations_conflicts_and_its_git_history(tmp_path):
    out = run_ts(tmp_path, f"""
      import ApplyReport from {json.dumps(str(SRC / 'components' / 'ApplyReport.tsx'))}
      const r = {{ added: [], modified: [], removed: [], skipped: [],
                   conflicts: [{{ path: 'app.py', reason: 'changed', side_file: 'app.py.nesicode-new' }}] }}
      out.chat = html(<ApplyReport result={{r}} chat />)
      out.run = html(<ApplyReport result={{r}} />)
      out.git = L.gitNotes({{ commits: 2, branches: ['fix', 'main'], stash: true }})
      out.none = L.gitNotes(null).concat(L.gitNotes({{ commits: 0, branches: [], stash: false }}))
      out.packed = L.gitNotes({{ commits: 0, branches: ['main'], stash: false, uncounted: true }})
      setLang('en')
      out.gitEn = L.gitNotes({{ commits: 1, branches: [], stash: false }})
      out.chatEn = L.conflictWhy('changed', true)
      setLang('es')
    """)
    assert "cambió en el proyecto después de que esta conversación lo copiara" in out["chat"]
    assert "<code>app.py.nesicode-new</code>" in out["chat"]
    assert "cambió en el proyecto después de empezar la ejecución" in out["run"]  # (a harnessing run's, as before)
    assert len(out["git"]) == 2 and out["git"][0].startswith("2 commit(s) hechos en esta conversación no llegan al proyecto")
    assert "Ramas con esos commits: fix, main." in out["git"][0] and "git stash" in out["git"][1]
    assert out["none"] == []
    assert out["packed"] == ["El historial de git de la copia de esta conversación cambió: «Aplicar» copia solo los "
                             "ficheros tal como están ahora en ella, nunca los commits ni las ramas. Ramas con esos "
                             "commits: main."]
    assert out["gitEn"] == ["1 commit(s) made in this conversation do not reach the project: Apply only copies the files "
                            "as they are now in its copy, never the git history. Commit in your project after applying."]
    assert out["chatEn"] == "changed in the project after this conversation copied it"
