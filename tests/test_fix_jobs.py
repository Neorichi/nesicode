"""Jobs runtime: queue, runs, approvals, harnessing orchestration and connector calls — offline regression tests.

Most tests use a private Jobs + database (few workers, users with high ids so their folders never meet the app's);
the ones that go through the HTTP API share the app's Jobs. Model calls are the mock provider or small fakes."""

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time

import pytest
from fastapi.testclient import TestClient

from nesicode.agent import Workspace
from nesicode.agent.workspace import ToolError
from nesicode.harness import base_harness
from nesicode.llm.base import Conversation, LLMError, Provider, ToolCall, Turn, Usage
from nesicode.server.app import app
from nesicode.server.chatops import ensure_thread, post_message
from nesicode.server.db import DB
from nesicode.server.jobs import FINAL, Jobs, project_ws, run_dir, thread_dir

_UIDS = iter(range(910_000, 920_000))


# ------------------------------------------------------------------ helpers
class Env:
    """A private Jobs on its own database."""

    def __init__(self, tmp_path, workers=1, share=False):
        self.db = DB(tmp_path / f"jobs-{time.time_ns()}.db")
        self.jobs = Jobs(self.db, max_workers=workers)
        self.uid = next(_UIDS)
        self.db.execute("INSERT INTO users (id, email, name, password_hash, created_at, settings) VALUES (?, ?, 'T', 'x', ?, ?)",
                        (self.uid, f"u{self.uid}@example.com", time.time(), json.dumps({"share_workspace": share})))

    def project(self, model="mock/demo", **kw) -> dict:
        pid = self.db.insert("INSERT INTO projects (user_id, name, executor_model, improver_model, max_steps, created_at) "
                             "VALUES (?, 'P', ?, 'mock/demo', ?, ?)", (self.uid, model, kw.get("max_steps", 20), time.time()))
        self.db.execute("INSERT INTO harness_versions (project_id, version, files, source, created_at) VALUES (?, 1, ?, 'base', ?)",
                        (pid, json.dumps(base_harness()), time.time()))
        return self.db.one("SELECT * FROM projects WHERE id=?", (pid,))

    def thread(self, project, exec_mode="") -> dict:
        t = ensure_thread(self.db, project, self.uid, "chat")
        if exec_mode:
            self.db.execute("UPDATE threads SET exec_mode=? WHERE id=?", (exec_mode, t["id"]))
        return self.db.one("SELECT * FROM threads WHERE id=?", (t["id"],))

    def bot(self, project, model="") -> dict:
        bid = self.db.insert("INSERT INTO bots (user_id, project_id, name, model, policy, created_at, updated_at) "
                             "VALUES (?, ?, 'B', ?, 'auto', ?, ?)", (self.uid, project["id"], model, time.time(), time.time()))
        return self.db.one("SELECT * FROM bots WHERE id=?", (bid,))

    def post(self, project, thread, text, **kw) -> dict:
        return post_message(self.db, self.jobs, project, thread, text, [], **kw)

    def delete_thread(self, project, thread) -> None:
        """What DELETE /api/threads/{id} does."""
        for r in self.db.all("SELECT id FROM runs WHERE thread_id=?", (thread["id"],)):
            self.jobs.cancel(r["id"])
        self.db.execute("DELETE FROM threads WHERE id=?", (thread["id"],))
        shutil.rmtree(thread_dir(project, thread["id"]), ignore_errors=True)

    def run(self, rid) -> dict:
        return self.db.one("SELECT * FROM runs WHERE id=?", (rid,))

    def events(self, rid) -> list[dict]:
        return [e for _, e in self.db.events_after(rid, 0, limit=5000)]

    def types(self, rid) -> list[str]:
        return [e["type"] for e in self.events(rid)]

    def wait(self, rid, timeout=15) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            run = self.run(rid)
            if run and run["status"] in FINAL:
                return run
            time.sleep(0.05)
        raise AssertionError(f"run {rid} did not finish: {self.run(rid)} {self.types(rid)}")

    def wait_event(self, rid, etype, n=1, timeout=15) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            found = [e for e in self.events(rid) if e["type"] == etype]
            if len(found) >= n:
                return found[n - 1]
            time.sleep(0.05)
        raise AssertionError(f"no {etype} #{n} in run {rid}: {self.types(rid)}")

    def close(self) -> None:
        self.jobs.shutdown(wait=3)


class Gate:
    """A model call that lasts until the test releases it (it holds its worker, like a long reply)."""

    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Semaphore(0)

    def wait_entered(self, n=1, timeout=10):
        for _ in range(n):
            assert self.entered.acquire(timeout=timeout), "the gated model call never started"

    def provider(self):
        gate = self

        class Conv(_Quiet):
            def step(self, on_event=None):
                gate.entered.release()
                gate.release.wait(20)
                return Turn("ok", [], "end_turn", Usage(1, 1))

        return _Prov(Conv)


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

    def system_section(self, bot, trigger=""):
        return "You are a bot."


class _AgentStub:
    def __init__(self, tmp_path, depth=0):
        self.ws = Workspace(tmp_path / f"ws{time.time_ns()}")
        self.cancel = threading.Event()
        self.depth = depth
        self.archive = ".nesicode/chats/7"
        self.events = []

    def emit(self, e):
        self.events.append(e)


@pytest.fixture()
def models(monkeypatch):
    """Extra fake models for the agent loop: models['mock/x'] = factory returning a Provider."""
    import nesicode.agent.loop as loop

    real = loop.get_provider
    table: dict = {}

    def get(spec, keys=None):
        if spec in table:
            return table[spec](), spec.split("/", 1)[1]
        return real(spec, keys)

    monkeypatch.setattr(loop, "get_provider", get)
    return table


@pytest.fixture()
def env(tmp_path):
    envs = []

    def make(workers=1, share=False):
        e = Env(tmp_path, workers, share)
        envs.append(e)
        return e

    yield make
    for e in envs:
        e.close()


# ------------------------------------------------------------------ jobs-runtime#1: reused run ids
@pytest.mark.parametrize("workers", [1, 2])
def test_stale_job_never_runs_a_reused_run_id(env, models, workers):
    e = env(workers)
    gate = Gate()
    models["mock/gate"] = gate.provider
    slow, fast = e.project("mock/gate"), e.project()
    try:
        blockers = [e.post(slow, e.thread(slow), f"largo {i}") for i in range(workers)]
        gate.wait_entered(workers)  # every worker is busy with a long reply
        x = e.thread(fast)
        b = e.post(fast, x, "B")
        assert e.run(b["id"])["status"] == "queued"
        e.delete_thread(fast, x)  # the user deletes that conversation and writes in a new one: SQLite reuses B's id
        c = e.post(fast, e.thread(fast), "C")
        assert c["id"] == b["id"]
        gate.release.set()
        done = e.wait(c["id"])
        types = e.types(c["id"])
        assert done["status"] == "done", (done, types)
        assert types.count("start") == 1 and types.count("done") == 1 and "error" not in types
        for r in blockers:
            assert e.wait(r["id"])["status"] == "done"
    finally:
        gate.release.set()


def test_stuck_job_of_a_deleted_conversation_never_writes_into_the_new_run(env, models):
    e = env(2)
    gate = Gate()
    models["mock/gate"] = gate.provider
    slow, fast = e.project("mock/gate"), e.project()
    try:
        x = e.thread(slow)
        old = e.post(slow, x, "se queda colgado")
        gate.wait_entered()  # running, inside a call that does not honour Stop
        e.delete_thread(slow, x)
        c = e.post(fast, e.thread(fast), "nuevo")
        assert c["id"] == old["id"]
        assert e.wait(c["id"])["status"] == "done"
        gate.release.set()  # the old job wakes up and finishes… into nothing
        assert e.jobs.pool.join(10)
        types = e.types(c["id"])
        assert e.run(c["id"])["status"] == "done" and types.count("start") == 1 and types.count("done") == 1, types
    finally:
        gate.release.set()


# ------------------------------------------------------------------ jobs-runtime#11 / #10: the conversation queue
@pytest.mark.parametrize("api_marks_first", [False, True])
def test_cancelling_a_queued_reply_moves_the_queue_in_order(env, models, api_marks_first):
    e = env(1)
    gate = Gate()
    models["mock/gate"] = gate.provider
    slow, fast = e.project("mock/gate"), e.project()
    try:
        blocker = e.post(slow, e.thread(slow), "ocupa el único worker")
        gate.wait_entered()
        t = e.thread(fast)
        m1, m2 = e.post(fast, t, "M1"), e.post(fast, t, "M2")
        assert (e.run(m1["id"])["status"], e.run(m2["id"])["status"]) == ("queued", "waiting")
        if api_marks_first:  # POST /cancel marks it before telling the jobs
            e.db.execute("UPDATE runs SET status='cancelled', finished_at=? WHERE id=?", (time.time(), m1["id"]))
        assert e.jobs.cancel(m1["id"]) or api_marks_first
        assert e.run(m1["id"])["status"] == "cancelled" and e.run(m2["id"])["status"] == "queued"
        m3 = e.post(fast, t, "M3")
        assert e.run(m3["id"])["status"] == "waiting"  # behind M2, never ahead of it
        gate.release.set()
        r2, r3 = e.wait(m2["id"]), e.wait(m3["id"])
        assert r2["status"] == r3["status"] == "done" and r2["finished_at"] <= r3["finished_at"]
        assert e.events(m1["id"]) == [] and e.run(m1["id"])["status"] == "cancelled"  # never started, never flipped back
        e.wait(blocker["id"])
    finally:
        gate.release.set()


def test_send_now_puts_a_reply_that_had_not_started_back_in_the_queue(env, models):
    e = env(1)
    gate = Gate()
    models["mock/gate"] = gate.provider
    slow, fast = e.project("mock/gate"), e.project()
    try:
        e.post(slow, e.thread(slow), "ocupa el único worker")
        gate.wait_entered()
        t = e.thread(fast)
        m1, m2 = e.post(fast, t, "M1"), e.post(fast, t, "urgente")
        assert e.jobs.send_now(m2["id"])
        assert (e.run(m1["id"])["status"], e.run(m2["id"])["status"]) == ("waiting", "queued")
        gate.release.set()
        r1, r2 = e.wait(m1["id"]), e.wait(m2["id"])
        assert r1["status"] == r2["status"] == "done" and r2["finished_at"] <= r1["finished_at"]
        assert "cancelled" not in e.types(m1["id"]) and e.types(m1["id"]).count("start") == 1
        order = [r["task"] for r in e.db.all("SELECT task FROM runs WHERE thread_id=? ORDER BY created_at, id", (t["id"],))]
        assert order == ["urgente", "M1"]
    finally:
        gate.release.set()


# ------------------------------------------------------------------ jobs-runtime#12: restart
def test_restart_resumes_accepted_messages_in_order(env):
    e = env(1)
    p = e.project()
    t, other = e.thread(p), e.thread(p)
    now = time.time()

    def row(thread, task, status, at):
        return e.db.insert("INSERT INTO runs (project_id, thread_id, kind, task, status, created_at) VALUES (?, ?, 'chat', ?, ?, ?)",
                           (p["id"], thread["id"], task, status, at))

    first = row(t, "PRIMERO (aceptado, sin empezar)", "queued", now - 2)
    second = row(t, "SEGUNDO (en cola detrás)", "waiting", now - 1)
    working = row(other, "trabajando al cerrar", "running", now - 3)
    restarted = Jobs(e.db, max_workers=1)  # the app starts again on the same data
    try:
        assert e.run(first)["status"] == "waiting" and e.run(working)["status"] == "interrupted"
        restarted.resume_waiting()
        r1, r2 = e.wait(first), e.wait(second)
        assert r1["status"] == r2["status"] == "done" and r1["finished_at"] <= r2["finished_at"]
    finally:
        restarted.shutdown(wait=3)


# ------------------------------------------------------------------ jobs-runtime#2: closing the app
def test_closing_never_overwrites_a_delivered_answer_and_keeps_the_queue(env, models):
    e = env(1)
    gate = Gate()
    models["mock/gate"] = gate.provider
    p = e.project("mock/gate")
    t = e.thread(p)
    try:
        m1, m2 = e.post(p, t, "M1"), e.post(p, t, "M2")
        gate.wait_entered()
        assert all(th.daemon for th in threading.enumerate() if th.name == "nesi-job")  # never keeps the process alive
        e.jobs.shutdown(wait=0.2)  # the window closes while M1 is inside a call that ignores Stop
        gate.release.set()
        r1 = e.wait(m1["id"])
        assert r1["status"] == "done" and "error" not in e.types(m1["id"])  # its answer arrived: never re-marked 'error'
        time.sleep(0.3)
        assert e.run(m2["id"])["status"] == "waiting"  # left for the next start, not stranded as 'queued'
    finally:
        gate.release.set()


def test_closing_interrupts_a_run_waiting_for_approval(env):
    e = env(1)
    p = e.project()
    r = e.post(p, e.thread(p, exec_mode="ask_all"), "escribe")
    e.wait_event(r["id"], "approval_request")
    e.jobs.shutdown(wait=5)
    run = e.wait(r["id"])
    assert run["status"] == "interrupted"


def test_process_exits_promptly_with_a_run_waiting_for_approval(tmp_path):
    script = textwrap.dedent("""
        import json, os, sys, time
        from nesicode.harness import base_harness
        from nesicode.server.chatops import ensure_thread, post_message
        from nesicode.server.db import DB
        from nesicode.server.jobs import Jobs

        db = DB(__import__("pathlib").Path(sys.argv[1]))
        jobs = Jobs(db, max_workers=1)
        db.execute("INSERT INTO users (id, email, name, password_hash, created_at) VALUES (1, 'a@b.c', 'A', 'x', 0)")
        pid = db.insert("INSERT INTO projects (user_id, name, executor_model, improver_model, created_at) "
                        "VALUES (1, 'P', 'mock/demo', 'mock/demo', 0)")
        db.execute("INSERT INTO harness_versions (project_id, version, files, source, created_at) VALUES (?, 1, ?, 'base', 0)",
                   (pid, json.dumps(base_harness())))
        p = db.one("SELECT * FROM projects WHERE id=?", (pid,))
        t = ensure_thread(db, p, 1, "x")
        db.execute("UPDATE threads SET exec_mode='ask_all' WHERE id=?", (t["id"],))
        r1 = post_message(db, jobs, p, t, "escribe", [])
        r2 = post_message(db, jobs, p, t, "luego esto", [])
        for _ in range(200):
            if any(e.get("type") == "approval_request" for _, e in db.events_after(r1["id"], 0)):
                break
            time.sleep(0.05)
        print("WAITING", time.time(), flush=True)
    """)
    path = tmp_path / "close_app.py"
    path.write_text(script)
    data = tmp_path / "data"
    envv = {**os.environ, "NESICODE_DATA_DIR": str(data), "NESICODE_NO_DOTENV": "1", "NESICODE_SANDBOX": "local"}
    started = time.time()
    proc = subprocess.run([sys.executable, str(path), str(tmp_path / "exit.db")], env=envv, capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
    waiting_at = float(re.search(r"WAITING ([0-9.]+)", proc.stdout).group(1))
    assert time.time() - waiting_at < 10, f"the process lingered {time.time() - waiting_at:.1f}s (started {started})"
    db = DB(tmp_path / "exit.db")
    statuses = [r["status"] for r in db.all("SELECT status FROM runs ORDER BY id")]
    assert statuses == ["interrupted", "waiting"]  # the next start resumes the queued message


# ------------------------------------------------------------------ jobs-runtime#7: workers
def test_approval_waits_do_not_hold_a_worker(env):
    e = env(1)
    p = e.project()
    a = e.post(p, e.thread(p, exec_mode="ask_all"), "escribe")  # waits for the user's OK to write
    req = e.wait_event(a["id"], "approval_request")
    b = e.post(p, e.thread(p), "otra conversación")
    assert e.wait(b["id"], timeout=10)["status"] == "done"  # the only worker was free for it
    assert e.run(a["id"])["status"] == "running"
    assert e.jobs.decide(a["id"], req["id"], "allow")
    assert e.wait(a["id"])["status"] == "done"


def test_bot_firings_always_leave_a_worker_for_chats(env, models):
    e = env(2)  # bots may take 1 of the 2 workers
    e.jobs.bots = _BotsStub()
    gate = Gate()
    models["mock/gate"] = gate.provider
    p = e.project()
    try:
        runs = []
        for i in range(3):  # e.g. three schedules caught up when the app opens
            b = e.bot(p, model="mock/gate")
            runs.append(e.post(p, e.thread(p), "tarea programada", bot_id=b["id"], trigger=f"cron#{i}"))
        gate.wait_entered()
        statuses = [e.run(r["id"])["status"] for r in runs]
        assert statuses.count("running") == 1 and statuses.count("waiting") == 2, statuses
        chat = e.post(p, e.thread(p), "hola")
        assert e.wait(chat["id"], timeout=10)["status"] == "done"
        gate.release.set()
        for r in runs:  # the held ones start as workers free up
            assert e.wait(r["id"])["status"] == "done"
    finally:
        gate.release.set()


# ------------------------------------------------------------------ jobs-runtime#9 (= loop-context#5, llm-providers#11)
def test_bot_with_its_own_model_uses_and_learns_its_own_limits(env, models, monkeypatch):
    import nesicode.server.jobs as jobs_mod

    calls = {"n": 0}

    class Small(_Quiet):
        def step(self, on_event=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise LLMError("This model's maximum context length is 8192 tokens. However, you requested 9000 tokens.")
            return Turn("resumen hecho", [], "end_turn", Usage(10, 10))

    models["mock/small"] = lambda: _Prov(Small)
    seen = []

    class Spy(jobs_mod.Agent):
        def __init__(self, **kw):
            seen.append((kw["model"], kw["context_limit"], kw["output_limit"]))
            super().__init__(**kw)

    monkeypatch.setattr(jobs_mod, "Agent", Spy)
    e = env(1)
    e.jobs.bots = _BotsStub()
    p = e.project("mock/demo")
    now = time.time()
    e.db.execute("INSERT INTO model_limits (user_id, spec, context, output, source, updated_at) VALUES (?, 'mock/demo', 1000000, 128000, 'table', ?)",
                 (e.uid, now))
    e.db.execute("INSERT INTO model_limits (user_id, spec, context, output, source, updated_at) VALUES (?, 'mock/small', 32768, 4096, 'provider', ?)",
                 (e.uid, now))
    bot = e.bot(p, model="mock/small")
    r = e.post(p, e.thread(p), "resume lo nuevo", bot_id=bot["id"], trigger="cron#1")
    assert e.wait(r["id"])["status"] == "done", e.types(r["id"])
    assert [s for s in seen if s[0] == "mock/small"] == [("mock/small", 32768, 4096)]  # the bot's window, not the executor's 1M
    rows = {x["spec"]: (x["context"], x["source"]) for x in e.db.all("SELECT * FROM model_limits WHERE user_id=?", (e.uid,))}
    assert rows["mock/small"] == (8192, "learned") and rows["mock/demo"] == (1000000, "table")


# ------------------------------------------------------------------ jobs-runtime#14: approval ids
def test_an_old_approval_button_never_answers_a_newer_request(env, models):
    class SameIds(_Quiet):  # an OpenAI-compatible server that omits ids: every step's call is 'call_0'
        n = 0

        def step(self, on_event=None):
            SameIds.n += 1
            if SameIds.n == 1:
                return Turn("", [ToolCall("call_0", "bash", {"command": "ls -la"})], "tool_use", Usage(1, 1))
            if SameIds.n == 2:
                return Turn("", [ToolCall("call_0", "bash", {"command": "echo destructivo > BORRADO"})], "tool_use", Usage(1, 1))
            return Turn("listo", [], "end_turn", Usage(1, 1))

    models["mock/ids"] = lambda: _Prov(SameIds)
    e = env(1)
    p = e.project("mock/ids")
    r = e.post(p, e.thread(p, exec_mode="ask_bash"), "dos comandos")
    first = e.wait_event(r["id"], "approval_request")
    assert first["call_id"] == "call_0" and first["id"] != "call_0"
    assert e.jobs.decide(r["id"], first["id"], "allow")
    second = e.wait_event(r["id"], "approval_request", n=2)
    assert second["call_id"] == "call_0" and second["id"] != first["id"]
    assert not e.jobs.decide(r["id"], first["id"], "allow")  # the stale button (e.g. an old Telegram message)
    assert not e.jobs.decide(r["id"], "call_0", "allow")  # the model's id is not an answer
    assert e.jobs.decide(r["id"], second["id"], "deny")
    assert e.wait(r["id"])["status"] == "done"
    results = [x for x in e.events(r["id"]) if x["type"] == "approval_result"]
    assert [(x["id"], x["decision"]) for x in results] == [(first["id"], "allow"), (second["id"], "deny")]


# ------------------------------------------------------------------ jobs-runtime#8: shared archives
def test_a_new_conversation_never_inherits_a_deleted_ones_archive(env):
    from nesicode.agent.context import search_history

    e = env(1, share=True)
    p = e.project()
    t1 = e.thread(p)
    assert not t1["isolated"]
    e.wait(e.post(p, t1, "primer chat")["id"])
    stale = project_ws(p) / ".nesicode" / "chats" / str(t1["id"]) / "context" / "history.md"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("el servidor de staging es 10.0.0.7 y la clave del cliente ACME es hunter2\n")
    e.delete_thread(p, t1)  # the shared folder keeps that conversation's archive
    t2 = e.thread(p)
    assert t2["id"] == t1["id"]
    e.wait(e.post(p, t2, "hola")["id"])
    assert not stale.exists()
    found = search_history(Workspace(project_ws(p)), {"query": "ACME"}, f".nesicode/chats/{t2['id']}")
    assert "hunter2" not in found
    # the conversation's own archive is kept from its first working turn on
    own = project_ws(p) / ".nesicode" / "chats" / str(t2["id"]) / "context" / "nota.md"
    own.parent.mkdir(parents=True, exist_ok=True)
    own.write_text("mío")
    e.wait(e.post(p, t2, "sigue")["id"])
    assert own.exists()


# ------------------------------------------------------------------ jobs-runtime#20: titles
def test_auto_title_ignores_inline_reasoning(env, monkeypatch):
    import nesicode.llm as llm
    from nesicode.server.jobs import clean_title

    assert clean_title("<think>\nThe user is asking…\n</think>\n\nRefactor del módulo de pagos") == "Refactor del módulo de pagos"
    assert clean_title("<think>\nThe user is asking about payments and") == ""  # cut by max_tokens: no title in it
    assert clean_title("Okay, payments.\n</think>\n\n«Migración a Postgres».") == "Migración a Postgres"  # opening tag in the template
    assert clean_title("Título: Informe de ventas") == "Informe de ventas"
    assert clean_title('"Plan de pruebas"') == "Plan de pruebas"
    budgets = []

    class Reasoner(Provider):
        name = "reasoner"

        def conversation(self, *a):
            raise NotImplementedError

        def complete(self, model, system, prompt, options):
            budgets.append(options.max_tokens)
            if options.max_tokens < 200:
                return "<think>\nThe user wants a title about payments, so", Usage()
            return "<think>Payments refactor.</think>\n\nRefactor del módulo de pagos", Usage()

    monkeypatch.setattr(llm, "get_provider", lambda spec, keys=None: (Reasoner(), "r1"))
    e = env(1)
    p = e.project("compat/r1")
    t = e.thread(p)
    e.db.execute("INSERT INTO runs (project_id, thread_id, kind, task, status, final_text, created_at) "
                 "VALUES (?, ?, 'chat', 'refactoriza pagos', 'done', 'hecho', ?)", (p["id"], t["id"], time.time()))
    e.jobs._make_title(t["id"], p["id"])
    assert e.db.one("SELECT title FROM threads WHERE id=?", (t["id"],))["title"] == "Refactor del módulo de pagos"
    assert budgets == [60, 1024]


# ------------------------------------------------------------------ connectors#2: argument paths
def test_absolutize_touches_only_real_path_arguments(tmp_path):
    from nesicode.server.jobs import absolutize_paths

    ws = Workspace(tmp_path / "ws")
    args = {"path": "README.md", "paths": ["a.pdf", "/abs/b.pdf"], "fileName": "x.png", "file_path": "d/e.txt",
            "outputDir": "out", "direction": "desc", "profile": "default", "fileId": "1AbC", "folderId": "f9",
            "documentId": "d1", "output_format": "json", "redirect": "follow", "document": "informe.docx",
            "workbook": "Ventas del Q3", "filename": "https://x.y/z.png"}
    out = absolutize_paths(args, ws)
    root = ws.root
    assert out["path"] == str(root / "README.md") and out["paths"] == [str(root / "a.pdf"), "/abs/b.pdf"]
    assert out["fileName"] == str(root / "x.png") and out["file_path"] == str(root / "d" / "e.txt")
    assert out["outputDir"] == str(root / "out") and out["document"] == str(root / "informe.docx")
    for key in ("direction", "profile", "fileId", "folderId", "documentId", "output_format", "redirect", "workbook", "filename"):
        assert out[key] == args[key], key
    # a document's name may have spaces; `output` is a path only when it names a file
    out = absolutize_paths({"presentation": "Informe trimestral.pptx", "output": "resultado.png"}, ws)
    assert out == {"presentation": str(root / "Informe trimestral.pptx"), "output": str(root / "resultado.png")}
    assert absolutize_paths({"output": "json", "document": "texto con un punto. Y más"}, ws) == \
        {"output": "json", "document": "texto con un punto. Y más"}


class _FakeManager:
    def __init__(self, text="ok", delay=0.0):
        self.text, self.delay, self.calls = text, delay, []

    def call(self, user_id, cfg, tool, args, timeout=180):
        from nesicode.connectors import McpResult

        self.calls.append((cfg.name, tool, args))
        time.sleep(self.delay)
        return McpResult(self.text, [], False)

    def tools(self, user_id, cfg, timeout=120):
        time.sleep(self.delay)
        return []


def test_remote_connectors_get_their_arguments_as_written(tmp_path, monkeypatch):
    import nesicode.connectors as conn
    from nesicode.server.jobs import _mcp_caller

    fake = _FakeManager()
    monkeypatch.setattr(conn, "manager", lambda: fake)
    agent = _AgentStub(tmp_path)
    github = conn.ConnectorConfig(name="github", kind="http", url="https://api.githubcopilot.com/mcp/")
    _mcp_caller(1, github, "get_file_contents")({"owner": "o", "repo": "r", "path": "README.md"}, agent)
    _mcp_caller(1, github, "list_issues")({"owner": "o", "repo": "r", "direction": "desc"}, agent)
    assert fake.calls[0][2]["path"] == "README.md" and fake.calls[1][2]["direction"] == "desc"
    local = conn.ConnectorConfig(name="playwright", kind="stdio", command="npx")
    _mcp_caller(1, local, "browser_take_screenshot")({"filename": "shot.png"}, agent)
    assert fake.calls[2][2]["filename"] == str(agent.ws.root / "shot.png")


# ------------------------------------------------------------------ connectors#4: big outputs
def test_big_connector_outputs_are_never_cut_silently(tmp_path, monkeypatch):
    import nesicode.connectors as conn
    from nesicode.agent import Agent
    from nesicode.server.jobs import _mcp_caller

    big = "".join(f"línea {i:05d} " + "x" * 50 + "\n" for i in range(1000)) + "ÚLTIMA LÍNEA\n"
    monkeypatch.setattr(conn, "manager", lambda: _FakeManager(big))
    cfg = conn.ConnectorConfig(name="pw", kind="stdio", command="npx")
    call = _mcp_caller(1, cfg, "browser_snapshot")
    # the main agent gets the whole output; its loop saves it complete and keeps head + tail in context
    assert call({}, _AgentStub(tmp_path)) == big
    agent = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(tmp_path / "main"), archive=".nesicode/chats/7")
    agent.extra_tools = {"mcp__pw__browser_snapshot": (None, call)}
    res = agent._call_tool(ToolCall("c1", "mcp__pw__browser_snapshot", {}), Usage())
    saved = [agent.ws.root / m for m in re.findall(r"(\.nesicode/\S+?\.txt)", res.content)]
    assert any(f.is_file() and f.read_text(encoding="utf-8") == big for f in saved), res.content[-600:]
    assert "20,000 chars in total" not in res.content
    # a sub-agent keeps less in context, but is told so — with the real size and the complete copy on disk
    sub = _AgentStub(tmp_path, depth=1)
    out = call({}, sub)
    assert len(out) < 21_000 and f"{len(big):,}" in out and "COMPLETE output is saved at" in out
    rel = re.search(r"saved at (\S+)", out).group(1)
    assert (sub.ws.root / rel).read_text(encoding="utf-8") == big


class _FakeHub:
    def __init__(self, open_tabs=(101,), default=303, delay=0.0):
        self.open = set(open_tabs) | {default}
        self.default, self.delay, self.sent = default, delay, []

    def connected(self, user_id):
        return True

    def call(self, user_id, cmd, args, timeout=45):
        self.sent.append((cmd, dict(args)))
        time.sleep(self.delay)
        tab = args.get("tab_id", self.default)
        if cmd == "open":
            new = 202 if "tab_id" not in args else tab
            self.open.add(new)
            return {"tab_id": new, "title": "nueva", "url": args["url"]}
        if tab not in self.open:
            raise RuntimeError("That tab is not in the “Nesicode” group: the user must drag it into the group so Nesicode can use it")
        if cmd == "close":
            self.open.discard(tab)
            return {"closed": tab}
        if cmd == "read":
            elements = [{"ref": f"e{i}", "tag": "a", "label": f"enlace {i}", "href": f"https://example.com/{i}"} for i in range(1, 301)]
            return {"tab_id": tab, "title": "t", "url": "https://example.com", "text": "texto " * 2500, "elements": elements}
        return {"tab_id": tab}


def test_chrome_read_keeps_every_element_and_valid_json(tmp_path, monkeypatch):
    import nesicode.browser as browser
    from nesicode.server.jobs import browser_tools

    monkeypatch.setattr(browser, "hub", _FakeHub())
    out = browser_tools(1)["chrome_read"][1]({}, _AgentStub(tmp_path))
    data = json.loads(out)  # never cut in the middle of the JSON
    assert data["elements"][-1]["ref"] == "e300" and out.index('"elements"') < out.index('"text"')


# ------------------------------------------------------------------ connectors#5: Chrome tabs
def test_chrome_open_opens_a_new_tab_and_a_closed_tab_is_forgotten(tmp_path, monkeypatch):
    import nesicode.browser as browser
    from nesicode.server.jobs import browser_tools

    hub = _FakeHub(open_tabs=(101,))
    monkeypatch.setattr(browser, "hub", hub)
    tools = browser_tools(1)
    agent = _AgentStub(tmp_path)

    def use(name, args=None):
        return tools[name][1](args or {}, agent)

    use("chrome_read", {"tab_id": 101})  # the user's own tab, shared with the agent
    use("chrome_open", {"url": "https://empresa.com"})
    assert hub.sent[-1] == ("open", {"url": "https://empresa.com"})  # a new tab, never the user's tab 101
    use("chrome_close")  # closes the tab it opened (the last one used)
    assert hub.sent[-1] == ("close", {"tab_id": 202})
    use("chrome_read")
    assert hub.sent[-1] == ("read", {})  # no stale id after the close
    use("chrome_read", {"tab_id": 101})
    hub.open.discard(101)  # the user closes it or drags it out of the group
    with pytest.raises(ToolError, match="no longer available"):
        use("chrome_read")
    use("chrome_read")
    assert hub.sent[-1] == ("read", {})


# ------------------------------------------------------------------ connectors#19: Stop during connector calls
def test_stop_interrupts_slow_connector_calls(env, tmp_path, monkeypatch):
    import nesicode.browser as browser
    import nesicode.connectors as conn
    import nesicode.virustotal as vt
    from nesicode.llm import ToolSpec
    from nesicode.server.jobs import _mcp_caller, browser_tools, builtin_agent_tools

    def stopped_quickly(fn):
        agent = _AgentStub(tmp_path)
        threading.Timer(0.3, agent.cancel.set).start()
        t0 = time.monotonic()
        with pytest.raises(ToolError, match="stopped by the user"):
            fn(agent)
        assert time.monotonic() - t0 < 2

    monkeypatch.setattr(conn, "manager", lambda: _FakeManager(delay=4))
    cfg = conn.ConnectorConfig(name="burp", kind="http", url="http://127.0.0.1:9876/mcp")
    stopped_quickly(lambda a: _mcp_caller(1, cfg, "send_http1_request")({}, a))
    monkeypatch.setattr(browser, "hub", _FakeHub(delay=4))
    stopped_quickly(lambda a: browser_tools(1)["chrome_open"][1]({"url": "https://x.y"}, a))

    def polling(args, ws, cfg):  # like vt_scan_file waiting for the analysis
        time.sleep(4)
        return "clean", None

    monkeypatch.setattr(vt, "TOOLS", {"vt_scan_file": (ToolSpec("vt_scan_file", "scan", {"type": "object"}), polling)})
    tools = builtin_agent_tools("virustotal", "VirusTotal", "virustotal", conn.ConnectorConfig(name="virustotal", kind="builtin"))
    stopped_quickly(lambda a: tools["mcp__virustotal__vt_scan_file"][1]({}, a))
    # starting a connector that hangs does not delay Stop either
    e = env(1)
    e.db.execute("INSERT INTO user_connectors (user_id, name, label, kind, command, created_at) VALUES (?, 'lento', 'Lento', 'stdio', 'x', ?)",
                 (e.uid, time.time()))
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    tools = e.jobs.connector_tools(e.uid, lambda ev: None, context=None, cancel=cancel)
    assert not any(name.startswith("mcp__lento") for name in tools)
    assert time.monotonic() - t0 < 2


# ------------------------------------------------------------------ via the API (the app's Jobs)
def _wait_run(client, run_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] not in ("queued", "running", "waiting"):
            return run
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def _wait_api_event(client, run_id, etype, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = [x for x in client.get(f"/api/runs/{run_id}").json()["events"] if x["type"] == etype]
        if found:
            return found[0]
        time.sleep(0.1)
    raise AssertionError(f"no {etype}")


@pytest.fixture()
def alice():
    c = TestClient(app)
    assert c.post("/api/auth/register", json={"email": f"jobs{time.time_ns()}@example.com", "password": "secret123"}).status_code == 200
    return c


def _project(client, **kw):
    r = client.post("/api/projects", json={"name": "Jobs", "executor_model": "mock/demo", "improver_model": "mock/demo", **kw})
    assert r.status_code == 200, r.text
    return r.json()


# jobs-runtime#3
def test_auto_improve_never_reruns_a_gated_conversation(alice):
    from nesicode.agent.loop import AgentResult
    from nesicode.server.app import db, jobs

    p = _project(alice)
    alice.put("/api/me/settings", json={"auto_harness": True, "auto_harness_min_steps": 3, "auto_harness_daily": 5})
    try:
        t = alice.post(f"/api/projects/{p['id']}/threads", json={}).json()
        alice.patch(f"/api/threads/{t['id']}", json={"exec_mode": "ask_all"})
        run = alice.post(f"/api/threads/{t['id']}/messages", json={"text": "despliega a producción"}).json()
        req = _wait_api_event(alice, run["id"], "approval_request")
        assert alice.post(f"/api/runs/{run['id']}/approve", json={"call_id": req["id"], "decision": "allow"}).status_code == 200
        done = _wait_run(alice, run["id"])
        assert done["status"] == "done"
        assert not any(x["type"] == "auto_harness" for x in done["events"])
        skipped = [x for x in done["events"] if x["type"] == "auto_harness_skipped"]
        assert skipped and "permiso" in skipped[0]["why"]
        # «Mejorar harness con esta tarea» on that conversation: refused before any rollout runs a command
        hr = _wait_run(alice, alice.post(f"/api/runs/{run['id']}/improve").json()["id"])
        assert hr["status"] == "error" and "permiso" in hr["error"]
        assert not any(x["type"] == "rollout" for x in hr["events"])
        # auto mode, but this turn needed the user's approval for something: not re-run on its own either
        t2 = alice.post(f"/api/projects/{p['id']}/threads", json={}).json()
        row = db.one("SELECT * FROM runs WHERE id=?", (run["id"],))
        events = []
        res = AgentResult("hecho", "done", 30, Usage(), [{"type": "approval_request", "id": "n1", "name": "mcp__burp__burp_scan"}])
        jobs._auto_improve({**row, "thread_id": t2["id"]}, db.one("SELECT * FROM projects WHERE id=?", (p["id"],)),
                           db.one("SELECT * FROM threads WHERE id=?", (t2["id"],)), res, events.append)
        assert [x["type"] for x in events] == ["auto_harness_skipped"] and "permiso" in events[0]["why"]
    finally:
        alice.put("/api/me/settings", json={"auto_harness": False})


# jobs-runtime#5 (+ #19: a cancelled run keeps no copy of the project)
def test_stop_during_the_last_stage_applies_adopts_and_learns_nothing(alice, monkeypatch):
    from nesicode.harnessing.engine import HarnessingRun
    from nesicode.server.app import db, jobs

    orig = HarnessingRun._select

    def select_then_stop(self, labels):
        jobs.cancel(self.seed)  # the user presses «Cancelar» while the final stage / selection runs (seed = run id)
        return orig(self, labels)

    monkeypatch.setattr(HarnessingRun, "_select", select_then_stop)
    p = _project(alice, check_command="test -f NESICODE_RESULT.md")
    run = alice.post(f"/api/projects/{p['id']}/harnessing", json={"task": "Write a report", "k": 1, "apply_changes": True}).json()
    run = _wait_run(alice, run["id"])
    assert run["status"] == "cancelled"
    assert not {"applied", "adopted", "lesson", "done"} & {x["type"] for x in run["events"]}
    assert not any(f["path"] == "NESICODE_RESULT.md" for f in alice.get(f"/api/projects/{p['id']}/tree").json())
    assert alice.get(f"/api/projects/{p['id']}").json()["harness_version"] == p["harness_version"]
    assert not db.all("SELECT id FROM lessons WHERE run_id=?", (run["id"],))
    rd = run_dir(db.one("SELECT * FROM projects WHERE id=?", (p["id"],)), run["id"])
    assert not (rd / "source").exists() and not list(rd.glob("rollouts/*/ws"))
    assert (rd / "rollouts" / "base1" / "trajectory.txt").exists()  # what the UI shows stays


# jobs-runtime#6 (= harness-harnessing#3)
def test_adoption_never_reverts_harness_edits_made_during_the_run(alice, monkeypatch):
    from nesicode.harnessing.engine import HarnessingRun
    from nesicode.server.app import db
    from nesicode.server.jobs import current_harness, save_harness_version

    edit: dict = {}

    def user_edits_then_select(self, labels):
        p = db.one("SELECT * FROM projects WHERE id=?", (edit["pid"],))
        files = dict(current_harness(db, p))
        files[edit["path"]] = files.get(edit["path"], "") + edit["line"]
        save_harness_version(db, p["id"], files, "manual", "edición del usuario durante la ejecución")
        return "agg1", "the improved candidate wins (the normal good outcome)"

    monkeypatch.setattr(HarnessingRun, "_select", user_edits_then_select)

    # the user edited a file the run did not touch: both changes end up in the adopted version
    p = _project(alice)
    edit.update(pid=p["id"], path="tool_guidance.md", line="- REGLA DEL USUARIO\n")
    run = _wait_run(alice, alice.post(f"/api/projects/{p['id']}/harnessing", json={"task": "Write a report", "k": 1}).json()["id"])
    assert run["status"] == "done" and run["result"]["adopted_version"], run["result"].get("adopt_skipped")
    files = alice.get(f"/api/projects/{p['id']}/harness").json()["files"]
    assert "REGLA DEL USUARIO" in files["tool_guidance.md"]
    assert "read it back once" in files["guardrails.md"] and "skills/verify-writes/SKILL.md" in files  # the run's edits

    # the user edited the same file the run changed: nothing is adopted on its own, the user's version stays current
    p2 = _project(alice)
    edit.update(pid=p2["id"], path="memory.md", line="- HECHO DEL USUARIO\n")
    run = _wait_run(alice, alice.post(f"/api/projects/{p2['id']}/harnessing", json={"task": "Write a report", "k": 1}).json()["id"])
    assert run["status"] == "done" and run["result"]["adopted_version"] is None and run["result"]["adopt_skipped"]
    assert any(x["type"] == "adopt_skipped" for x in run["events"])
    v = alice.get(f"/api/projects/{p2['id']}").json()["harness_version"]
    assert v == 2 and "HECHO DEL USUARIO" in alice.get(f"/api/projects/{p2['id']}/harness").json()["files"]["memory.md"]


# jobs-runtime#19
def test_harnessing_copies_do_not_pile_up(env):
    e = env(1)
    p = e.project()

    def fake_run(status):
        rid = e.db.insert("INSERT INTO runs (project_id, kind, task, status, created_at) VALUES (?, 'harness', 't', ?, ?)",
                          (p["id"], status, time.time()))
        rd = run_dir(p, rid)
        for rel in ("source/big.bin", "rollouts/base1/ws/big.bin", "rollouts/base1/trajectory.txt"):
            (rd / rel).parent.mkdir(parents=True, exist_ok=True)
            (rd / rel).write_text("x")
        return rd

    old_done = [fake_run("done") for _ in range(2)]
    cut_short = fake_run("running")  # the app closed while it worked
    new_done = [fake_run("done") for _ in range(3)]
    restarted = Jobs(e.db, max_workers=1)  # a restart prunes in the background
    deadline = time.time() + 10
    while time.time() < deadline and any((rd / "source").exists() for rd in old_done + [cut_short]):
        time.sleep(0.05)
    restarted.shutdown()
    for rd in old_done + [cut_short]:
        assert not (rd / "source").exists() and not (rd / "rollouts/base1/ws").exists(), rd
        assert (rd / "rollouts/base1/trajectory.txt").exists()
    for rd in new_done:  # the last few successful runs keep their copies for «Aplicar» later
        assert (rd / "source/big.bin").exists() and (rd / "rollouts/base1/ws/big.bin").exists()
