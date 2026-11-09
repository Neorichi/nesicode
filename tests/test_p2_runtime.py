"""Phase 2, group Q — runtime wiring: jobs, connectors, workspace, harnessing and database ids. Offline regression tests
(fake providers, local stdio MCP servers started with this Python or with node, temp dirs; nothing leaves 127.0.0.1)."""

import json
import os
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from nesicode.agent import Workspace
from nesicode.agent.workspace import ToolError, copy_workspace
from nesicode.harness import base_harness
from nesicode.llm.base import Conversation, LLMError, Provider, Turn, Usage
from nesicode.server import db as dbmod
from nesicode.server.chatops import ensure_thread, post_message
from nesicode.server.db import DB
from nesicode.server.jobs import FINAL, Jobs, project_ws, run_dir

POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX only (FIFOs, process groups, symlinks)")
_UIDS = iter(range(930_000, 940_000))


# ====================================================================== helpers
def _until(cond, timeout=10.0, step=0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(step)
    return bool(cond())


def _alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) is TerminateProcess there: it would kill the process it asks about
        import _winapi

        try:
            handle = _winapi.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        except OSError as exc:
            return exc.winerror == 5  # access denied: it exists (not ours); 87 (invalid parameter): no such process
        try:
            return _winapi.GetExitCodeProcess(handle) == 259  # STILL_ACTIVE (else it has exited, its handle still held)
        finally:
            _winapi.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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

    def __init__(self, tmp_path, workers=2, share=False):
        self.db = DB(tmp_path / f"q-{time.time_ns()}.db")
        self.jobs = Jobs(self.db, max_workers=workers)
        self.uid = next(_UIDS)
        self.db.execute("INSERT INTO users (id, email, name, password_hash, created_at, settings) VALUES (?, ?, 'T', 'x', ?, ?)",
                        (self.uid, f"q{self.uid}@example.com", time.time(), json.dumps({"share_workspace": share})))

    def project(self, model="mock/demo", improver="mock/demo") -> dict:
        pid = self.db.insert("INSERT INTO projects (user_id, name, executor_model, improver_model, max_steps, created_at) "
                             "VALUES (?, 'P', ?, ?, 20, ?)", (self.uid, model, improver, time.time()))
        self.db.execute("INSERT INTO harness_versions (project_id, version, files, source, created_at) VALUES (?, 1, ?, 'base', ?)",
                        (pid, json.dumps(base_harness()), time.time()))
        p = self.db.one("SELECT * FROM projects WHERE id=?", (pid,))
        project_ws(p).mkdir(parents=True, exist_ok=True)
        return p

    def thread(self, project) -> dict:
        return ensure_thread(self.db, project, self.uid, "chat")

    def bot(self, project) -> dict:
        bid = self.db.insert("INSERT INTO bots (user_id, project_id, name, policy, created_at, updated_at) "
                             "VALUES (?, ?, 'Vigía', 'auto', ?, ?)", (self.uid, project["id"], time.time(), time.time()))
        return self.db.one("SELECT * FROM bots WHERE id=?", (bid,))

    def post(self, project, thread, text, **kw) -> dict:
        return post_message(self.db, self.jobs, project, thread, text, [], **kw)

    def run(self, rid) -> dict:
        return self.db.one("SELECT * FROM runs WHERE id=?", (rid,))

    def events(self, rid) -> list[dict]:
        return [e for _, e in self.db.events_after(rid, 0, limit=5000)]

    def wait(self, rid, timeout=30) -> dict:
        assert _until(lambda: (self.run(rid) or {}).get("status") in FINAL, timeout), (self.run(rid), self.events(rid))
        return self.run(rid)

    def close(self) -> None:
        self.jobs.shutdown(wait=3)


@pytest.fixture()
def env(tmp_path):
    made = []

    def make(workers=2, share=False):
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


def _spy_model(models, name="mock/spy") -> list[str]:
    """A model that answers at once and records the system prompt of every conversation it is given."""
    systems: list[str] = []

    class Spy(_Quiet):
        def __init__(self, model, system, tools, options):
            super().__init__(model, system, tools, options)
            systems.append(system)

        def step(self, on_event=None):
            return Turn("hecho", [], "end_turn", Usage(1, 1))

    models[name] = lambda: _Prov(Spy)
    return systems


class _AgentStub:
    def __init__(self, root: Path):
        self.ws = Workspace(root)
        self.cancel = threading.Event()
        self.depth = 0
        self.archive = ".nesicode/chats/7"
        self.events: list[dict] = []

    def emit(self, e):
        self.events.append(e)


def _tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


# ---------------------------------------------------------------------- local MCP servers
MCP_SERVER = textwrap.dedent('''
    import json, os, signal, sys, time
    mode = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            f.write(str(os.getpid()))
    if mode == "mute":  # never answers initialize and ignores its stdin closing
        while True:
            time.sleep(0.1)
    if mode == "stubborn":  # ... and SIGTERM too
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(0.1)
    from mcp.server.mcpserver import MCPServer
    app = MCPServer("q")

    @app.tool()
    def env() -> str:
        """The names of the server's environment variables."""
        return json.dumps(sorted(os.environ))

    @app.tool()
    def pid() -> str:
        """The server's pid."""
        return str(os.getpid())

    app.run()
''')

NODE_SERVER = textwrap.dedent('''
    // a minimal MCP server over stdio (JSON-RPC lines), no SDK needed
    const rl = require("readline").createInterface({input: process.stdin});
    const send = (m) => process.stdout.write(JSON.stringify(m) + "\\n");
    rl.on("line", (line) => {
      let m;
      try { m = JSON.parse(line); } catch (e) { return; }
      if (m.id === undefined) return;
      if (m.method === "initialize") {
        send({jsonrpc: "2.0", id: m.id, result: {protocolVersion: m.params.protocolVersion, capabilities: {tools: {}},
              serverInfo: {name: "node-q", version: "1"}}});
      } else if (m.method === "tools/list") {
        send({jsonrpc: "2.0", id: m.id, result: {tools: [{name: "env", description: "env names",
              inputSchema: {type: "object", properties: {}}}]}});
      } else if (m.method === "tools/call") {
        send({jsonrpc: "2.0", id: m.id, result: {content: [{type: "text", text: JSON.stringify(Object.keys(process.env).sort())}]}});
      } else {
        send({jsonrpc: "2.0", id: m.id, error: {code: -32601, message: "unknown method"}});
      }
    });
''')


@pytest.fixture()
def mcp_script(tmp_path):
    path = tmp_path / "srv_q.py"
    path.write_text(MCP_SERVER)
    return str(path)


def _new_manager():
    from nesicode.connectors import McpManager

    return McpManager()


def _close_manager(m) -> None:
    try:
        m.shutdown(8)
    finally:
        m.loop.call_soon_threadsafe(m.loop.stop)
        m.thread.join(5)


@pytest.fixture()
def mgr():
    m = _new_manager()
    yield m
    _close_manager(m)


@pytest.fixture()
def fast_kill(monkeypatch):
    import mcp.client.stdio as mstdio

    monkeypatch.setattr(mstdio, "PROCESS_TERMINATION_TIMEOUT", 0.3)  # the 2 s grace before SIGTERM, shortened


def _stdio(name, server, *args, **env):
    from nesicode.connectors import ConnectorConfig

    return ConnectorConfig(name=name, kind="stdio", command=sys.executable, args=[server, *args], env=env)


# ====================================================================== Q16 — ids are never reused
TELEGRAM_TRIGGERS = (
    "CREATE TRIGGER IF NOT EXISTS telegram_unanchor_thread AFTER DELETE ON threads BEGIN "
    "UPDATE telegram_links SET active_thread=NULL WHERE active_thread=OLD.id; END",
    "CREATE TRIGGER IF NOT EXISTS telegram_unanchor_project AFTER DELETE ON projects BEGIN "
    "UPDATE telegram_links SET active_project=NULL WHERE active_project=OLD.id; END",
)


def _old_db(path: Path) -> None:
    """A database as 6e070d9 (and every version before this fix) left it: no AUTOINCREMENT, the columns added since by
    ALTER TABLE, Telegram's triggers, rows linked by foreign keys — and leftovers of deleted rows."""
    old = dbmod.SCHEMA.replace(" AUTOINCREMENT", "")
    assert dbmod.SCHEMA.count(" AUTOINCREMENT") == len(dbmod.AUTOINC_TABLES) and "AUTOINCREMENT" not in old
    c = sqlite3.connect(path, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    c.executescript(old)
    for table, column, decl in dbmod.MIGRATIONS:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    for sql in TELEGRAM_TRIGGERS:
        c.execute(sql)
    now = time.time()
    c.execute("INSERT INTO users (id, email, name, password_hash, created_at) VALUES (1, 'a@x.com', 'A', 'h', ?)", (now,))
    for i in (1, 2, 3):
        c.execute("INSERT INTO projects (user_id, name, executor_model, improver_model, created_at, profile) "
                  "VALUES (1, ?, 'mock/demo', 'mock/demo', ?, 'general')", (f"p{i}", now))
        c.execute("INSERT INTO harness_versions (project_id, version, files, source, created_at) VALUES (?, 1, '{}', 'base', ?)",
                  (i, now))
    for pid in (1, 1, 2):
        c.execute("INSERT INTO threads (project_id, title, created_at, updated_at) VALUES (?, 't', ?, ?)", (pid, now, now))
    for i, (pid, tid) in enumerate([(1, 1), (1, 1), (1, 2), (2, 3), (2, 3)], 1):
        c.execute("INSERT INTO runs (project_id, thread_id, kind, task, status, created_at) VALUES (?, ?, 'chat', ?, 'done', ?)",
                  (pid, tid, f"tarea {i}", now + i))
        c.execute("INSERT INTO events (run_id, seq, data) VALUES (?, 1, '{\"type\": \"start\"}')", (i,))
    c.execute("INSERT INTO runs (project_id, kind, task, status, created_at) VALUES (1, 'harness', 'h', 'done', ?)", (now,))
    for name in ("b1", "b2"):
        c.execute("INSERT INTO bots (user_id, project_id, name, created_at, updated_at) VALUES (1, 1, ?, ?, ?)", (name, now, now))
    c.execute("INSERT INTO bot_triggers (bot_id, kind, spec, created_at) VALUES (1, 'interval', '60', ?)", (now,))
    c.execute("INSERT INTO bot_triggers (bot_id, kind, spec, created_at) VALUES (2, 'once', '', ?)", (now,))
    c.execute("INSERT INTO runs (project_id, thread_id, kind, task, status, created_at, bot_id) "
              "VALUES (1, 1, 'chat', 'del bot', 'done', ?, 2)", (now,))
    c.execute("INSERT INTO vault_secrets (user_id, ref, blob, hint, created_at) VALUES (1, 'bot:2:webhook', 'x', 'h', ?)", (now,))
    c.execute("INSERT INTO telegram_links (user_id, active_project, active_thread, created_at) VALUES (1, 2, 3, ?)", (now,))
    c.execute("DELETE FROM bots WHERE id=2")  # the newest bot is deleted: its runs and its webhook secret stay behind
    c.close()


TABLES = ("users", "projects", "threads", "runs", "events", "bots", "bot_triggers", "harness_versions", "telegram_links")


def _snapshot(path: Path) -> dict:
    c = sqlite3.connect(path)
    try:
        return {
            "rows": {t: c.execute(f"SELECT * FROM {t} ORDER BY rowid").fetchall() for t in TABLES},
            "cols": {t: c.execute(f"PRAGMA table_info({t})").fetchall() for t in dbmod.AUTOINC_TABLES},
            "objects": sorted(r for r in c.execute("SELECT type, name, tbl_name FROM sqlite_master WHERE type IN ('index', 'trigger')")),
            "sql": {t: c.execute("SELECT sql FROM sqlite_master WHERE name=?", (t,)).fetchone()[0] for t in dbmod.AUTOINC_TABLES},
        }
    finally:
        c.close()


def test_an_old_database_migrates_to_ids_that_are_never_reused(tmp_path):
    path = tmp_path / "nesicode.db"
    _old_db(path)
    before = _snapshot(path)
    db = DB(path)
    after = _snapshot(path)
    for t in dbmod.AUTOINC_TABLES:
        assert "AUTOINCREMENT" in after["sql"][t] and "AUTOINCREMENT" not in before["sql"][t]
    assert after["rows"] == before["rows"] and after["cols"] == before["cols"]  # same data, same columns and constraints
    assert after["objects"] == before["objects"]  # indexes and Telegram's triggers are all back
    assert db.one("PRAGMA foreign_keys")["foreign_keys"] == 1 and not db.all("PRAGMA foreign_key_check")
    seq = {r["name"]: r["seq"] for r in db.all("SELECT * FROM sqlite_sequence")}
    assert seq["runs"] == 7 and seq["threads"] == 3 and seq["projects"] == 3
    assert seq["bots"] == 2  # the deleted bot's id is still named by its runs and its webhook secret: never handed out again
    # the copy made before touching the file is the database as it was
    bak = Path(f"{path}.pre-autoinc.bak")
    assert bak.exists() and _snapshot(bak)["rows"] == before["rows"] and _snapshot(bak)["sql"] == before["sql"]
    # deleting the newest row no longer frees its id
    now = time.time()
    db.execute("DELETE FROM runs WHERE id=7")
    assert db.insert("INSERT INTO runs (project_id, kind, task, status, created_at) VALUES (1, 'harness', 'x', 'done', ?)", (now,)) == 8
    db.execute("DELETE FROM threads WHERE id=3")  # cascades, and Telegram's trigger drops the anchor
    assert db.one("SELECT active_thread FROM telegram_links WHERE user_id=1")["active_thread"] is None
    assert not db.all("SELECT * FROM runs WHERE thread_id=3") and not db.all("SELECT * FROM events WHERE run_id IN (4, 5)")
    assert db.insert("INSERT INTO threads (project_id, title, created_at, updated_at) VALUES (1, 'n', ?, ?)", (now, now)) == 4
    assert db.insert("INSERT INTO bots (user_id, project_id, name, created_at, updated_at) VALUES (1, 1, 'nuevo', ?, ?)",
                     (now, now)) == 3
    db.execute("DELETE FROM projects WHERE id=3")
    assert db.insert("INSERT INTO projects (user_id, name, executor_model, improver_model, created_at) "
                     "VALUES (1, 'n', 'm/m', 'm/m', ?)", (now,)) == 4
    db.execute("DELETE FROM projects WHERE id=1")  # foreign keys still cascade through the rebuilt tables
    assert not db.all("SELECT id FROM threads WHERE project_id=1") and not db.all("SELECT id FROM bots WHERE project_id=1")
    assert not db.all("SELECT * FROM bot_triggers") and not db.all("SELECT * FROM events WHERE run_id IN (1, 2, 3)")


def test_a_second_start_is_a_no_op(tmp_path):
    path = tmp_path / "nesicode.db"
    _old_db(path)
    DB(path).conn.close()
    bak = Path(f"{path}.pre-autoinc.bak")
    stamp, first = bak.stat().st_mtime_ns, _snapshot(path)
    seq = sqlite3.connect(path).execute("SELECT * FROM sqlite_sequence ORDER BY name").fetchall()
    db = DB(path)
    assert _snapshot(path) == first and bak.stat().st_mtime_ns == stamp
    assert db.all("SELECT name, seq FROM sqlite_sequence ORDER BY name") == [{"name": n, "seq": s} for n, s in seq]
    assert dbmod.migrate_autoincrement(db.conn, path) == []


def test_a_failed_migration_leaves_the_database_as_it_was(tmp_path, monkeypatch, capsys):
    path = tmp_path / "nesicode.db"
    _old_db(path)
    before = _snapshot(path)
    real, done = dbmod._rebuild_autoinc, []

    def fail_on_the_third(conn, table, sql):
        if len(done) == 2:
            raise sqlite3.OperationalError("disk I/O error (simulated)")
        real(conn, table, sql)
        done.append(table)

    monkeypatch.setattr(dbmod, "_rebuild_autoinc", fail_on_the_third)
    db = DB(path)  # the app still starts
    assert "could not migrate" in capsys.readouterr().out
    assert done == ["projects", "threads"] and _snapshot(path) == before  # the two finished rebuilds were rolled back too
    assert db.one("PRAGMA foreign_keys")["foreign_keys"] == 1 and not db.all("PRAGMA foreign_key_check")
    assert not db.one("SELECT 1 AS x FROM sqlite_master WHERE name LIKE '_autoinc_%'")
    db.conn.close()
    monkeypatch.setattr(dbmod, "_rebuild_autoinc", real)
    db = DB(path)  # the next start does it
    assert all("AUTOINCREMENT" in db.one("SELECT sql FROM sqlite_master WHERE name=?", (t,))["sql"] for t in dbmod.AUTOINC_TABLES)
    assert _snapshot(path)["rows"] == before["rows"]


def test_a_new_database_never_reuses_ids(tmp_path):
    path = tmp_path / "fresh.db"
    db = DB(path)
    assert not Path(f"{path}.pre-autoinc.bak").exists()  # nothing to migrate, nothing to back up
    now = time.time()
    db.execute("INSERT INTO users (id, email, name, password_hash, created_at) VALUES (1, 'a@x.com', 'A', 'h', ?)", (now,))
    new = {
        "projects": lambda: db.insert("INSERT INTO projects (user_id, name, executor_model, improver_model, created_at) "
                                      "VALUES (1, 'p', 'm/m', 'm/m', ?)", (now,)),
        "threads": lambda: db.insert("INSERT INTO threads (project_id, title, created_at, updated_at) VALUES (1, 't', ?, ?)",
                                     (now, now)),
        "runs": lambda: db.insert("INSERT INTO runs (project_id, kind, task, status, created_at) VALUES (1, 'chat', 't', "
                                  "'done', ?)", (now,)),
        "bots": lambda: db.insert("INSERT INTO bots (user_id, project_id, name, created_at, updated_at) VALUES (1, 1, 'b', ?, ?)",
                                  (now, now)),
        "bot_triggers": lambda: db.insert("INSERT INTO bot_triggers (bot_id, kind, created_at) VALUES (1, 'once', ?)", (now,)),
    }
    first = {t: make() for t, make in new.items()}
    second = {t: new[t]() for t in reversed(list(new))}  # (children first: the project and the bot must exist)
    for t in reversed(list(new)):
        db.execute(f"DELETE FROM {t} WHERE id=?", (second[t],))
    for t, make in new.items():
        assert make() == second[t] + 1 > first[t], t


# ====================================================================== Q14 — hook truncation
def test_truncate_hook_cuts_for_every_positive_max_chars():
    from nesicode.agent.hooks import Hooks, truncate

    for n in [0, 1, 2, 3, 10, 39, 40, 41, 60, 999, 20_010, 50_000]:
        text = "".join(chr(97 + i % 26) for i in range(n))
        for limit in [1, 2, 3, 5, 30, 36, 37, 38, 40, 41, 50, 100, 20_000]:
            out = truncate(text, limit)
            if n <= limit:
                assert out == text
                continue
            assert len(out) <= limit < n, (n, limit, out)  # never longer than asked, never longer than before
            if "trimmed by hook" in out:
                head, rest = out.split("\n... [", 1)
                count, tail = rest.split(" chars trimmed by hook] ...\n", 1)
                assert text.startswith(head) and text.endswith(tail) and int(count) == n - len(head) - len(tail)
            else:
                assert out == text[:limit]
    hooks = Hooks({"post_tool": [{"tool": "*", "action": "truncate", "max_chars": 1}]})
    assert hooks.after("bash", "abc") == "a" and hooks.after("bash", "x") == "x"
    assert len(hooks.finish("bash", "Error: " + "y" * 500, ["careful"])) == 1
    hooks = Hooks({"post_tool": [{"tool": "*", "action": "truncate", "max_chars": 20_000}]})  # the base harness rule
    out = hooks.after("read_file", "z" * 20_010)
    assert len(out) <= 20_000 and "chars trimmed by hook" in out


# ====================================================================== Q13 — Ctrl+C never orphans a command
@POSIX
def test_keyboard_interrupt_while_a_command_runs_kills_it(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "ws")
    marker = ws.root / "child.pid"
    real_wait = subprocess.Popen.wait

    def wait(self, timeout=None):
        if marker.exists() and marker.read_text().strip():  # the command (and its child) is running: Ctrl+C now
            raise KeyboardInterrupt
        return real_wait(self, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        ws.run("sleep 30 & echo $! > child.pid; wait", timeout=60)
    assert time.monotonic() - t0 < 10
    child = int(marker.read_text())
    assert _until(lambda: not _alive(child), 5), "the command's child outlived the interrupt"


# ====================================================================== Q12 — FIFOs, sockets, devices never hang a walk
@pytest.fixture()
def fifos():
    made: list[Path] = []

    def make(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.mkfifo(path)
        made.append(path)
        return path

    yield make
    for p in made:  # whatever a failing test left blocked on one gets its other end, so no thread stays stuck
        for flags in (os.O_WRONLY | os.O_NONBLOCK, os.O_RDONLY | os.O_NONBLOCK):
            try:
                os.close(os.open(p, flags))
            except OSError:
                pass


def _unix_socket(path: Path) -> socket.socket:
    """A socket file at `path` (bound at a short path, then moved: temp paths are longer than sun_path allows)."""
    import tempfile

    short = Path(tempfile.mkdtemp(prefix="q", dir="/tmp"))
    s = socket.socket(socket.AF_UNIX)
    try:
        s.bind(str(short / "s"))
        os.rename(short / "s", path)
    finally:
        shutil.rmtree(short, ignore_errors=True)
    return s


@POSIX
def test_walks_diffs_copies_and_applies_never_open_a_fifo_or_socket(tmp_path, fifos):
    from nesicode.agent.workspace import _walk, apply_cage, apply_changes, changes_since, diff_workspaces, manifest
    from nesicode.harnessing import apply_rollout

    cage = tmp_path / "cage"
    _tree(cage, {"a.txt": "a\n", "sub/b.txt": "b\n"})
    base = manifest(cage)
    before, project = tmp_path / "before", tmp_path / "project"
    copy_workspace(cage, before)
    copy_workspace(cage, project)
    fifos(cage / "pipe")
    fifos(cage / "sub" / "pipe2")
    sock = _unix_socket(cage / "s.sock")
    try:
        (cage / "a.txt").write_text("a2\n")
        m = _quick(lambda: manifest(cage))
        assert "a.txt" in m and not {"pipe", "sub/pipe2", "s.sock"} & set(m)
        assert sorted(p.name for p in _quick(lambda: list(_walk(cage)))) == ["a.txt", "b.txt"]
        d = _quick(lambda: diff_workspaces(before, cage))
        assert d["modified"] == ["a.txt"] and not d["added"] and not d["removed"]
        assert _quick(lambda: changes_since(base, cage, before))["modified"] == ["a.txt"]
        copy = tmp_path / "copy"
        _quick(lambda: copy_workspace(cage, copy))
        assert (copy / "a.txt").read_text() == "a2\n" and not os.path.lexists(copy / "pipe")
        assert not os.path.lexists(copy / "s.sock")
        # FIFOs in the project where the result changed a file (and where its side file goes): reported or
        # replaced, never opened
        (cage / "sub" / "b.txt").write_text("b2\n")
        os.unlink(project / "sub" / "b.txt")
        fifos(project / "sub" / "b.txt")
        fifos(project / "sub" / "b.txt.nesicode-new")
        res = _quick(lambda: apply_rollout(before, cage, project))  # (a FIFO in the target means "changed" since)
        assert [(c["path"], c["side_file"]) for c in res["conflicts"]] == [("sub/b.txt", "sub/b.txt.nesicode-new")]
        assert res["modified"] == ["a.txt"] and (project / "sub" / "b.txt.nesicode-new").read_text() == "b2\n"
        assert stat.S_ISFIFO(os.lstat(project / "sub" / "b.txt").st_mode)  # the user's own FIFO is left alone
        out = _quick(lambda: apply_cage(base, cage, project))  # a conversation's «Aplicar» replaces it
        assert "sub/b.txt" in out["modified"] and (project / "sub" / "b.txt").read_text() == "b2\n"
        other = tmp_path / "other"
        other.mkdir()
        assert sorted(_quick(lambda: apply_changes(before, cage, other))["modified"]) == ["a.txt", "sub/b.txt"]
        assert sorted(os.listdir(other)) == ["a.txt", "sub"] and os.listdir(other / "sub") == ["b.txt"]
    finally:
        sock.close()


@POSIX
def test_a_copy_never_opens_what_was_swapped_in_after_the_listing(tmp_path, fifos):
    """copy_workspace lists a folder, then copies each file: anything that stops being a regular file in between (a
    FIFO, a socket, a link) is skipped without being opened."""
    from nesicode.agent.workspace import _copy_regular

    src = tmp_path / "src"
    _tree(src, {"keep.txt": "hola\n"})
    os.chmod(src / "keep.txt", 0o640)
    os.utime(src / "keep.txt", ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    out = tmp_path / "out"
    out.mkdir()
    _quick(lambda: _copy_regular(str(src / "keep.txt"), str(out / "keep.txt")))
    st = os.stat(out / "keep.txt")
    assert (out / "keep.txt").read_text() == "hola\n" and stat.S_IMODE(st.st_mode) == 0o640
    assert st.st_mtime_ns == 1_600_000_000_000_000_000
    fifos(src / "pipe")
    os.symlink("/etc/hosts", src / "link")
    sock = _unix_socket(src / "s.sock")
    try:
        for name in ("pipe", "link", "s.sock", "gone"):
            _quick(lambda name=name: _copy_regular(str(src / name), str(out / name)))
            assert not os.path.lexists(out / name), name
    finally:
        sock.close()
    # the destination is never written through a link either
    target = tmp_path / "victim.txt"
    target.write_text("intact\n")
    os.symlink(target, out / "keep2.txt")
    with pytest.raises(OSError):
        _quick(lambda: _copy_regular(str(src / "keep.txt"), str(out / "keep2.txt")))
    assert target.read_text() == "intact\n"


@POSIX
def test_attachments_copied_into_a_snapshot_never_follow_a_link(tmp_path, fifos):
    from nesicode.agent.workspace import copy_files_into

    outside = tmp_path / "outside"
    _tree(outside, {"secret.txt": "clave\n"})
    snap = tmp_path / "snap"
    snap.mkdir()
    cage = tmp_path / "cage"
    att = cage / ".attachments"
    _tree(att, {"foto.txt": "foto\n", "sub/doc.txt": "doc\n"})
    os.symlink(outside / "secret.txt", att / "fuera.txt")
    os.symlink("foto.txt", att / "dentro.txt")
    fifos(att / "pipe")
    skipped = _quick(lambda: copy_files_into(att, snap, ".attachments"))
    got = snap / ".attachments"
    assert (got / "foto.txt").read_text() == "foto\n" and (got / "sub/doc.txt").read_text() == "doc\n"
    assert os.readlink(got / "dentro.txt") == "foto.txt"  # a link inside stays a link
    assert not os.path.lexists(got / "fuera.txt") and not os.path.lexists(got / "pipe")
    assert {s.split(":")[0] for s in skipped} == {".attachments/fuera.txt", ".attachments/pipe"}
    # the agent turned its whole attachments folder into a link to the user's files: nothing is copied
    shutil.rmtree(att)
    os.symlink(outside, att)
    snap2 = tmp_path / "snap2"
    snap2.mkdir()
    assert copy_files_into(att, snap2, ".attachments") and not os.path.lexists(snap2 / ".attachments")
    # the project's own .attachments in the snapshot is a link out of it: never written through
    shutil.rmtree(cage)
    _tree(att, {"x.txt": "x\n"})
    os.symlink(outside, snap2 / ".attachments")
    assert copy_files_into(att, snap2, ".attachments") and sorted(os.listdir(outside)) == ["secret.txt"]


@POSIX
def test_connector_files_are_never_a_fifo_or_a_link(tmp_path, fifos, monkeypatch):
    """Office runs in-process (Stop can't interrupt it): a FIFO named like its document would block the run for good."""
    from nesicode import burp_rest, office, virustotal
    from nesicode.connectors import ConnectorConfig

    ws = Workspace(tmp_path / "ws")
    fifos(ws.root / "informe.docx")
    fifos(ws.root / "datos.xlsx")
    fifos(ws.root / "muestra.exe")
    with pytest.raises(ToolError, match="not a regular file"):
        _quick(lambda: office.word_write({"path": "informe.docx", "blocks": ["hola"]}, ws))
    with pytest.raises(ToolError, match="not a regular file"):
        _quick(lambda: office.excel_write({"path": "datos.xlsx", "rows": [[1]]}, ws))
    with pytest.raises(ToolError):
        _quick(lambda: office.office_read({"path": "datos.xlsx"}, ws))
    with pytest.raises(ToolError):  # swapped in after the checks: still never opened
        _quick(lambda: office._src(ws, ws.root / "informe.docx"))
    with pytest.raises(ToolError):
        _quick(lambda: office._store(ws, ws.root / "informe.docx", lambda buf: buf.write(b"PK")))
    with pytest.raises(ToolError):
        _quick(lambda: virustotal.vt_scan_file({"path": "muestra.exe"}, ws, ConnectorConfig("vt", "builtin", env={"VT_API_KEY": "k"})))
    # Office still writes and reads its documents (in memory, through the workspace)
    _quick(lambda: office.word_write({"path": "ok.docx", "title": "T", "blocks": ["primer párrafo"]}, ws))
    _quick(lambda: office.excel_write({"path": "ok.xlsx", "rows": [["a", 1], ["b", "=B1*2"]]}, ws))
    _quick(lambda: office.pptx_write({"path": "ok.pptx", "slides": [{"title": "Hola", "bullets": ["uno"]}]}, ws))
    assert "primer párrafo" in office.office_read({"path": "ok.docx"}, ws)[0]
    assert "=B1*2" in office.office_read({"path": "ok.xlsx"}, ws)[0] and "Hola" in office.office_read({"path": "ok.pptx"}, ws)[0]
    office.word_replace({"path": "ok.docx", "replacements": [{"find": "primer", "replace": "segundo"}]}, ws)
    assert "segundo párrafo" in office.office_read({"path": "ok.docx"}, ws)[0]
    # Burp's full JSON lands under a predictable name: a FIFO planted there is refused, not waited on
    fifos(ws.root / "burp" / "scan-7.json")

    class _Resp:
        status_code, headers, text = 200, {}, ""

        def json(self):
            return {"scan_status": "succeeded", "issue_events": [{"type": "issue_found", "issue": {"name": "XSS", "severity": "high"}}]}

    monkeypatch.setattr(burp_rest.httpx, "request", lambda *a, **k: _Resp())
    cfg = ConnectorConfig("burp-rest", "builtin", url="http://127.0.0.1:1337")
    with pytest.raises(ToolError):
        _quick(lambda: burp_rest.burp_scan_status({"task_id": "7", "save_json": True}, ws, cfg))
    os.unlink(ws.root / "burp" / "scan-7.json")
    text, rel = _quick(lambda: burp_rest.burp_scan_status({"task_id": "7", "save_json": True}, ws, cfg))
    assert rel == "burp/scan-7.json" and json.loads((ws.root / rel).read_text())["scan_status"] == "succeeded"


# ====================================================================== Q1 / Q5 / Q15 — «Aplicar»
def _harness_run(e: Env, p: dict, selected="agg1") -> tuple[int, Path, Path]:
    """A finished harnessing run whose snapshot is the project as it is now; returns (run id, run dir, chosen ws)."""
    rid = e.db.insert("INSERT INTO runs (project_id, kind, task, status, result, created_at) VALUES (?, 'harness', 't', 'done', ?, ?)",
                      (p["id"], json.dumps({"selected": selected}), time.time()))
    rd = run_dir(p, rid)
    copy_workspace(project_ws(p), rd / "source")
    copy_workspace(rd / "source", rd / "rollouts" / selected / "ws")
    (rd / "rollouts" / selected / "trajectory.txt").write_text("pasos")
    return rid, rd, rd / "rollouts" / selected / "ws"


@POSIX
def test_aplicar_never_overwrites_later_edits_nor_writes_through_a_link(env, tmp_path):
    from nesicode.server.jobs import GONE

    e = env()
    p = e.project()
    proj = project_ws(p)
    _tree(proj, {"app.py": "v1\n", "lib.py": "lib v1\n", "notas.md": "notas\n"})
    rid, rd, ws = _harness_run(e, p)
    _tree(ws, {"app.py": "v2 del harnessing\n", "lib.py": "lib v2\n", "nuevo.md": "nuevo\n", "enlace/evil.txt": "pwned\n"})
    os.unlink(ws / "notas.md")
    os.symlink("/etc", ws / "fuera")  # the rollout also made a link out of the project
    # days later, before «Aplicar», the user keeps working in the project
    (proj / "app.py").write_text("v1\nARREGLO DEL USUARIO\n")
    (proj / "notas.md").write_text("notas\nIMPORTANTE\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, proj / "enlace")
    changes = e.jobs.apply_later(rid, p)
    assert (proj / "app.py").read_text() == "v1\nARREGLO DEL USUARIO\n"  # never overwritten…
    assert (proj / "app.py.nesicode-new").read_text() == "v2 del harnessing\n"  # …the result's version waits beside it
    assert (proj / "notas.md").read_text() == "notas\nIMPORTANTE\n"  # a later edit is never deleted
    assert (proj / "lib.py").read_text() == "lib v2\n" and (proj / "nuevo.md").read_text() == "nuevo\n"
    assert not list(outside.iterdir()) and not os.path.lexists(proj / "fuera")  # nothing through or out of a link
    reasons = {c["path"]: (c["reason"], c["side_file"]) for c in changes["conflicts"]}
    assert reasons == {"app.py": ("changed", "app.py.nesicode-new"), "notas.md": ("changed", None),
                       "enlace/evil.txt": ("unsafe", None), "fuera": ("unsafe", None)}
    assert json.loads(e.run(rid)["result"])["applied"] == changes
    # (review) enlace/evil.txt is nowhere in the project — its folder is a link there now — so the run's copies stay
    # for another «Aplicar» instead of losing that file
    assert (rd / "source").is_dir() and ws.is_dir()
    os.unlink(proj / "enlace")  # the user removes that link: «Aplicar» again puts the file in a real folder
    again = e.jobs.apply_later(rid, p)
    assert "enlace/evil.txt" in again["added"] and (proj / "enlace" / "evil.txt").read_text() == "pwned\n"
    assert {c["path"]: c["reason"] for c in again["conflicts"]} == {"app.py": "changed", "notas.md": "changed",
                                                                     "fuera": "unsafe"}
    assert not list(outside.iterdir()) and not os.path.lexists(proj / "fuera")
    # everything is in the project now (or beside it; a link out of it is refused whatever happens): the run's copies
    # are gone, what the UI shows stays
    assert not (rd / "source").exists() and not ws.exists() and (rd / "rollouts/agg1/trajectory.txt").exists()
    with pytest.raises(FileNotFoundError) as gone:
        e.jobs.apply_later(rid, p)
    assert str(gone.value) == GONE


@POSIX
def test_aplicar_keeps_the_copies_when_something_could_not_be_written(env):
    e = env()
    p = e.project()
    proj = project_ws(p)
    _tree(proj, {"a.txt": "a\n"})
    rid, rd, ws = _harness_run(e, p)
    _tree(ws, {"datos/tabla.csv": "x,y\n", "a.txt": "a2\n"})
    (proj / "datos").write_text("un fichero donde el resultado necesita una carpeta\n")
    changes = e.jobs.apply_later(rid, p)
    assert [(c["path"], c["reason"]) for c in changes["conflicts"]] == [("datos/tabla.csv", "error")]
    assert (rd / "source").is_dir() and ws.is_dir()  # kept: «Aplicar» can be tried again once the way is clear
    os.unlink(proj / "datos")
    again = e.jobs.apply_later(rid, p)
    assert not again["conflicts"] and (proj / "datos/tabla.csv").read_text() == "x,y\n" and "a.txt" in again["modified"]
    assert not (rd / "source").exists()


def test_aplicar_never_waits_for_a_busy_project(env, models):
    """A shared conversation holds the project's lock for its whole run (an approval can take 30 min): «Aplicar» answers
    at once that the project is busy instead of hanging the request."""
    from nesicode.server.jobs import BUSY, ProjectBusy

    gate, entered = threading.Event(), threading.Event()

    class Slow(_Quiet):
        def step(self, on_event=None):
            entered.set()
            gate.wait(20)
            return Turn("ok", [], "end_turn", Usage(1, 1))

    models["mock/slow"] = lambda: _Prov(Slow)
    e = env(2, share=True)
    p = e.project("mock/slow")
    t = e.thread(p)
    assert not t["isolated"]
    rid, rd, ws = _harness_run(e, p)
    (ws / "nuevo.md").write_text("nuevo\n")
    chat = e.post(p, t, "trabaja en la carpeta compartida")
    try:
        assert entered.wait(10)
        t0 = time.monotonic()
        with pytest.raises(ProjectBusy) as busy:
            e.jobs.apply_later(rid, p)
        assert 1.5 < time.monotonic() - t0 < 4 and str(busy.value) == BUSY == \
            "El proyecto está ocupado por otra conversación; inténtalo cuando termine."
        stop = threading.Event()
        stop.set()
        t0 = time.monotonic()
        with pytest.raises(ProjectBusy):
            e.jobs.apply_later(rid, p, cancel=stop)  # a wait that is cancelled ends at once
        assert time.monotonic() - t0 < 0.5
        assert not (project_ws(p) / "nuevo.md").exists() and (rd / "source").is_dir()
    finally:
        gate.set()
    assert e.wait(chat["id"])["status"] == "done"
    assert "nuevo.md" in e.jobs.apply_later(rid, p)["added"]  # free again: applied


def test_applied_runs_never_take_the_place_of_copies_not_applied_yet(env):
    from nesicode.server.jobs import KEEP_HARNESS_COPIES

    e = env()
    p = e.project()
    (project_ws(p) / "a.txt").write_text("a\n")
    runs = [_harness_run(e, p) for _ in range(KEEP_HARNESS_COPIES + 1)]
    for _, _, ws in runs:
        (ws / "a.txt").write_text("cambiado\n")
    newest = runs[-1]
    e.jobs.apply_later(newest[0], p)
    assert not (newest[1] / "source").exists()
    e.jobs._prune_copies(p)
    assert all((rd / "source").is_dir() for _, rd, _ in runs[:-1])  # the 3 runs not applied keep their copies
    later = _harness_run(e, p)  # one more: now the oldest not applied gives way
    e.jobs._prune_copies(p)
    assert not (runs[0][1] / "source").exists() and not runs[0][2].exists()
    assert all((rd / "source").is_dir() for _, rd, _ in runs[1:-1] + [later])


# ====================================================================== Q1 / Q2 / Q15 — the harnessing run itself
def _harnessing(e: Env, p: dict, **params) -> int:
    return e.jobs.create_harnessing(p, "Write a report", {"k": 1, "mode": "parallel", **params})


@pytest.mark.parametrize("meanwhile", ["file", "link"])
def test_auto_apply_is_three_way_and_drops_the_copies(env, monkeypatch, tmp_path, meanwhile):
    from nesicode.harnessing.engine import HarnessingRun

    victim = tmp_path / "fuera.md"
    victim.write_text("fuera del proyecto\n")
    real = HarnessingRun.run

    def run_then_user_edits(self):
        result = real(self)
        target = self.source_ws / "NESICODE_RESULT.md"
        if meanwhile == "file":
            target.write_text("lo escribió el usuario mientras tanto\n")
        else:
            os.symlink(victim, target)  # a link to a file outside the project, where the result wants to write
        return result

    monkeypatch.setattr(HarnessingRun, "run", run_then_user_edits)
    e = env()
    p = e.project()
    (project_ws(p) / "app.py").write_text("print(1)\n")
    rid = _harnessing(e, p, apply_changes=True, adopt_harness=False)
    run = e.wait(rid, timeout=60)
    assert run["status"] == "done", e.events(rid)
    applied = json.loads(run["result"])["applied"]
    proj = project_ws(p)
    assert [(c["path"], c["reason"], c["side_file"]) for c in applied["conflicts"]] == \
        [("NESICODE_RESULT.md", "exists", "NESICODE_RESULT.md.nesicode-new")]
    assert (proj / "NESICODE_RESULT.md.nesicode-new").exists()
    assert victim.read_text() == "fuera del proyecto\n"  # never written through the link
    if meanwhile == "file":
        assert (proj / "NESICODE_RESULT.md").read_text() == "lo escribió el usuario mientras tanto\n"
    else:
        assert os.readlink(proj / "NESICODE_RESULT.md") == str(victim)
    rd = run_dir(p, rid)
    assert not (rd / "source").exists() and not list(rd.glob("rollouts/*/ws"))  # applied: no copy is kept
    assert list(rd.glob("rollouts/*/trajectory.txt"))


# ---------------------------------------------------------------------- (review) the project's folder is not there
def _folder_project(e: Env, folder: Path) -> dict:
    """A project that lives in a folder of the user's (e.g. on an external disk)."""
    p = e.project()
    e.db.execute("UPDATE projects SET folder=? WHERE id=?", (str(folder), p["id"]))
    return e.db.one("SELECT * FROM projects WHERE id=?", (p["id"],))


@POSIX
def test_aplicar_while_the_project_folder_is_not_there_keeps_the_result(env, tmp_path):
    """The folder is not there when «Aplicar» runs (a disk that is not connected, a folder moved): nothing is written,
    every change comes back as not written, and the run's copies stay until it is back — they used to be deleted (every
    path looked like it left the project: 'unsafe'), and the result was lost for good."""
    e = env()
    disk = tmp_path / "disco"
    folder = disk / "proyecto"
    _tree(folder, {"app.py": "v1\n", "viejo.md": "viejo\n"})
    p = _folder_project(e, folder)
    rid, rd, ws = _harness_run(e, p)
    _tree(ws, {"app.py": "v2\n", "nuevo.md": "nuevo\n"})
    os.unlink(ws / "viejo.md")
    disk.rename(tmp_path / "desconectado")
    changes = e.jobs.apply_later(rid, p)
    assert not changes["added"] and not changes["modified"] and not changes["removed"]
    assert sorted((c["path"], c["reason"], c["side_file"]) for c in changes["conflicts"]) == \
        [("app.py", "error", None), ("nuevo.md", "error", None), ("viejo.md", "error", None)]
    assert (rd / "source").is_dir() and ws.is_dir() and not disk.exists()  # kept; nothing made in the folder's place
    (tmp_path / "desconectado").rename(disk)  # connected again
    again = e.jobs.apply_later(rid, p)
    assert (again["added"], again["modified"], again["removed"], again["conflicts"]) == (["nuevo.md"], ["app.py"], ["viejo.md"], [])
    assert (folder / "app.py").read_text() == "v2\n" and not (folder / "viejo.md").exists()
    assert not (rd / "source").exists() and not ws.exists()


def test_auto_apply_while_the_project_folder_is_not_there_keeps_the_result(env, monkeypatch, tmp_path):
    from nesicode.harnessing.engine import HarnessingRun

    disk = tmp_path / "disco"
    real = HarnessingRun.run

    def run_then_unplug(self):
        result = real(self)
        disk.rename(tmp_path / "desconectado")  # the disk goes away while the run finishes
        return result

    monkeypatch.setattr(HarnessingRun, "run", run_then_unplug)
    e = env()
    folder = disk / "proyecto"
    _tree(folder, {"app.py": "print(1)\n"})
    p = _folder_project(e, folder)
    rid = _harnessing(e, p, apply_changes=True, adopt_harness=False)
    run = e.wait(rid, timeout=60)
    assert run["status"] == "done", e.events(rid)
    applied = json.loads(run["result"])["applied"]
    assert applied["conflicts"] and {c["reason"] for c in applied["conflicts"]} == {"error"} and not applied["added"]
    rd = run_dir(p, rid)
    assert (rd / "source").is_dir() and list(rd.glob("rollouts/*/ws"))  # kept for «Aplicar»
    (tmp_path / "desconectado").rename(disk)
    assert "NESICODE_RESULT.md" in e.jobs.apply_later(rid, p)["added"] and (folder / "NESICODE_RESULT.md").is_file()


@pytest.mark.skipif(not hasattr(os, "chflags") or not hasattr(stat, "UF_IMMUTABLE"), reason="needs chflags (macOS/BSD)")
def test_a_file_the_system_refuses_to_write_is_not_reported_as_a_link(env):
    """(review) EPERM is the apply's own answer to a result's link out of the project ('unsafe') — but also the
    system's for a locked file (macOS 'uchg'). That one is an 'error': the result's version waits beside it and the run's
    copies stay for another «Aplicar»."""
    from nesicode.harnessing import apply_rollout

    e = env()
    p = e.project()
    proj = project_ws(p)
    _tree(proj, {"app.py": "v1\n", "viejo.md": "viejo\n"})
    rid, rd, ws = _harness_run(e, p)
    (ws / "app.py").write_text("v2\n")
    os.unlink(ws / "viejo.md")
    os.symlink("/etc/hosts", ws / "fuera")
    for name in ("app.py", "viejo.md"):
        os.chflags(proj / name, stat.UF_IMMUTABLE)
    try:
        res = e.jobs.apply_later(rid, p)
        assert apply_rollout(rd / "source", ws, proj)["conflicts"] == res["conflicts"]  # (the same, run again)
    finally:
        for name in ("app.py", "viejo.md"):
            os.chflags(proj / name, 0)
    assert {c["path"]: (c["reason"], c["side_file"]) for c in res["conflicts"]} == {
        "viejo.md": ("error", None), "fuera": ("unsafe", None), "app.py": ("error", "app.py.nesicode-new")}
    assert (proj / "app.py").read_text() == "v1\n" and (proj / "app.py.nesicode-new").read_text() == "v2\n"
    assert (rd / "source").is_dir() and ws.is_dir()  # kept: once unlocked, «Aplicar» again does it
    again = e.jobs.apply_later(rid, p)
    assert again["modified"] == ["app.py"] and again["removed"] == ["viejo.md"] and (proj / "app.py").read_text() == "v2\n"
    assert not (rd / "source").exists()


def test_an_applied_run_shows_done_only_once_its_copies_are_gone(env, monkeypatch):
    """Once a run is 'done', «Aplicar» is offered: its copies must not be half deleted under an «Aplicar» then (a file
    missing from the copy of the result looks like the result deleted it). An applied run drops them before."""
    e = env()
    p = e.project()
    (project_ws(p) / "app.py").write_text("print(1)\n")
    seen, real = [], Jobs._finish

    def finish(self, run_id, status, **kw):
        if status == "done":
            rd = run_dir(p, run_id)
            seen.append((rd / "source").exists() or bool(list(rd.glob("rollouts/*/ws"))))
        return real(self, run_id, status, **kw)

    monkeypatch.setattr(Jobs, "_finish", finish)
    rid = _harnessing(e, p, apply_changes=True, adopt_harness=False)
    assert e.wait(rid, timeout=60)["status"] == "done", e.events(rid)
    assert json.loads(e.run(rid)["result"])["applied"]["added"] == ["NESICODE_RESULT.md"]
    assert seen == [False]


def test_harnessing_sizes_the_improver_with_its_own_output_limit(env, monkeypatch):
    import nesicode.server.limits as limits
    from nesicode.harnessing.engine import HarnessingRun

    seen = []

    def fake_limits(db, uid, spec, refresh=False):
        return {"context": 64_000, "output": 7_777, "source": "table"} if spec == "mock/improver" else \
            {"context": 200_000, "output": 32_000, "source": "table"}

    def capture(self):
        seen.append(self.cfg)
        raise RuntimeError("cancelled")

    monkeypatch.setattr(limits, "model_limits", fake_limits)
    monkeypatch.setattr(HarnessingRun, "run", capture)
    e = env()
    p = e.project(improver="mock/improver")
    e.wait(_harnessing(e, p))
    assert [(c.improver_context, c.improver_output, c.context_limit, c.output_limit) for c in seen] == \
        [(64_000, 7_777, 200_000, 32_000)]


def test_stop_while_waiting_for_the_project_applies_adopts_and_learns_nothing(env):
    e = env()
    p = e.project()
    lock = e.jobs._project_lock(p["id"])
    lock.acquire()  # a shared conversation is working in the project
    try:
        rid = _harnessing(e, p, apply_changes=True, adopt_harness=True)
        assert _until(lambda: any(x.get("phase") == "done" for x in e.events(rid)), 60), e.events(rid)
        time.sleep(0.3)  # hr.run() returned: the run now waits for the project
        assert e.run(rid)["status"] == "running"
        t0 = time.monotonic()
        e.jobs.cancel(rid)
        assert e.wait(rid, timeout=5)["status"] == "cancelled" and time.monotonic() - t0 < 2
    finally:
        lock.release()
    assert not {"applied", "adopted", "lesson", "done"} & {x["type"] for x in e.events(rid)}
    assert not (project_ws(p) / "NESICODE_RESULT.md").exists()
    assert e.db.one("SELECT harness_version FROM projects WHERE id=?", (p["id"],))["harness_version"] == p["harness_version"]
    assert not e.db.all("SELECT id FROM lessons WHERE run_id=?", (rid,))
    assert not (run_dir(p, rid) / "source").exists()


def test_stop_pressed_while_adopting_learns_nothing(env, monkeypatch):
    real = Jobs._adopt

    def adopt_then_stop(self, project_id, base_version, adopted, run_id, label):
        out = real(self, project_id, base_version, adopted, run_id, label)
        self.cancel(run_id)
        return out

    monkeypatch.setattr(Jobs, "_adopt", adopt_then_stop)

    from nesicode.harnessing.engine import HarnessingRun

    real_run = HarnessingRun.run

    def run_with_a_change(self):
        result = real_run(self)
        result.update(harness_changed=True, reflection={"experience": {"cue": "c", "lesson": "l", "evidence": "e"}},
                      adopted_harness={**result["adopted_harness"], "memory.md": "aprendido\n"})
        return result

    monkeypatch.setattr(HarnessingRun, "run", run_with_a_change)
    e = env()
    p = e.project()
    rid = _harnessing(e, p, adopt_harness=True)
    run = e.wait(rid, timeout=60)
    assert run["status"] == "cancelled", e.events(rid)
    assert not {"lesson", "done"} & {x["type"] for x in e.events(rid)}
    assert not e.db.all("SELECT id FROM lessons WHERE run_id=?", (rid,))


# ====================================================================== Q7 / Q8 / Q9 — jobs
def test_a_bot_is_told_when_its_previous_run_really_started(env, models, monkeypatch):
    from datetime import datetime

    import nesicode.server.jobs as jobs_mod
    from nesicode.server.bots import BotManager, _zone_or_local, local_tz

    systems = _spy_model(models)
    reached, go, real_note = threading.Event(), threading.Event(), jobs_mod.continuity_note

    def held_note(*a, **k):  # the run stops here, just before its system prompt is put together
        reached.set()
        go.wait(10)
        return real_note(*a, **k)

    monkeypatch.setattr(jobs_mod, "continuity_note", held_note)
    e = env(3)
    e.jobs.bots = BotManager(e.db, e.jobs)
    p = e.project("mock/spy")
    bot = e.bot(p)
    t, other = e.thread(p), e.thread(p)
    three_days_ago = time.time() - 3 * 86400
    e.db.execute("INSERT INTO runs (project_id, thread_id, kind, task, status, created_at, bot_id, trigger) "
                 "VALUES (?, ?, 'chat', 'antes', 'done', ?, ?, 'cron#1')", (p["id"], t["id"], three_days_ago, bot["id"]))
    e.db.execute("UPDATE bots SET last_run_at=? WHERE id=?", (time.time(), bot["id"]))  # fire() stamps it before the run
    r = e.post(p, t, "¿qué hay de nuevo?", bot_id=bot["id"], trigger="cron#1")
    assert reached.wait(10)
    # meanwhile a newer run of the same bot is running too (a webhook): it must not be taken for this one
    e.db.execute("INSERT INTO runs (project_id, thread_id, kind, task, status, created_at, bot_id, trigger) "
                 "VALUES (?, ?, 'chat', 'webhook', 'running', ?, ?, 'webhook')", (p["id"], other["id"], time.time(), bot["id"]))
    go.set()
    assert e.wait(r["id"])["status"] == "done", e.events(r["id"])
    expected = datetime.fromtimestamp(three_days_ago, _zone_or_local(local_tz())).strftime("%Y-%m-%d %H:%M")
    assert f"Previous run started: {expected}." in systems[-1]


def test_a_retried_reply_never_replays_its_failed_attempt(env):
    e = env(1)
    p = e.project()
    t = e.thread(p)
    rid = e.db.insert("INSERT INTO runs (project_id, thread_id, kind, task, status, created_at) "
                      "VALUES (?, ?, 'chat', 'x', 'running', ?)", (p["id"], t["id"], time.time()))
    emit = e.jobs.emitter(rid)
    live = e.jobs.live[rid]
    stored, real_add = {}, e.db.add_event

    def spy(run_id, event):
        stored[event["type"]] = (live.gen, list(live.items))  # what a stream reading this event would get with it
        return real_add(run_id, event)

    e.db.add_event = spy
    emit({"type": "text_delta", "text": "Hola, est", "_persist": False})
    gen = live.gen
    emit({"type": "llm_retry", "attempt": 2, "max": 3, "wait": 1.0, "reason": "stalled", "partial": True})
    assert live.items == [] and live.gen == gen + 1 and stored["llm_retry"] == (gen + 1, [])
    emit({"type": "text_delta", "text": "Hola, esto es", "_persist": False})
    emit({"type": "llm_retry", "attempt": 2, "max": 3, "partial": True, "depth": 1})  # a sub-agent's: not this text
    emit({"type": "llm_retry", "attempt": 2, "max": 3, "partial": False})  # nothing had been streamed yet
    assert [i["text"] for i in live.items] == ["Hola, esto es"] and live.gen == gen + 1
    emit({"type": "assistant", "text": "Hola, esto es todo"})
    assert stored["assistant"] == (gen + 2, []) and live.items == []


def test_every_way_a_run_ends_marks_its_conversation_updated(env, models, monkeypatch):
    """A chat list that is not open notices a conversation changed by its updated_at: an answer, a Stop and a crash
    (the _safe path) all count, not only the normal end."""
    import nesicode.server.jobs as jobs_mod

    go, entered = threading.Event(), threading.Event()

    class Held(_Quiet):
        def step(self, on_event=None):
            entered.set()
            while not go.wait(0.05):
                if self.options.cancel is not None and self.options.cancel.is_set():
                    raise LLMError("cancelled")  # what a real provider does when Stop aborts the request
            return Turn("ok", [], "end_turn", Usage(1, 1))

    models["mock/held"] = lambda: _Prov(Held)
    e = env(1)
    p = e.project("mock/held")
    t = e.thread(p)

    def updated() -> float:
        return e.db.one("SELECT updated_at FROM threads WHERE id=?", (t["id"],))["updated_at"]

    def ended(how) -> str:
        go.clear()
        entered.clear()
        r = e.post(p, t, "hola")
        assert entered.wait(10)
        e.db.execute("UPDATE threads SET updated_at=1 WHERE id=?", (t["id"],))  # long ago
        how(r["id"])
        status = e.wait(r["id"])["status"]
        assert updated() > 1, status
        return status

    assert ended(lambda rid: go.set()) == "done"
    assert ended(lambda rid: e.jobs.cancel(rid)) == "cancelled"
    crash = threading.Event()

    def broken_note(*a, **k):  # something in the run blows up after it started: _safe records the error
        crash.wait(10)
        raise RuntimeError("se rompió")

    monkeypatch.setattr(jobs_mod, "continuity_note", broken_note)
    r = e.post(p, t, "otra")
    e.db.execute("UPDATE threads SET updated_at=1 WHERE id=?", (t["id"],))
    crash.set()
    assert e.wait(r["id"])["status"] == "error" and updated() > 1


# ====================================================================== Q6 / Q10 — calls at run time
class _CancelAwareHub:
    """A Chrome command that runs until the run's Stop cancels it (it honours `cancel`, like the real hub)."""

    def __init__(self):
        self.got = self.ended = None

    def connected(self, user_id):
        return True

    def call(self, user_id, cmd, args, timeout=45, cancel=None):
        self.got = cancel
        try:
            if cancel is None or not cancel.wait(10):
                return {"tab_id": 1}
            raise RuntimeError("cancelado: el usuario pulsó Detener")
        finally:
            self.ended = time.monotonic()


class _CancelAwareManager:
    def __init__(self, tools=()):
        from nesicode.connectors import McpTool

        self.tool_list = [McpTool(n, "", {"type": "object", "properties": {}}) for n in tools]
        self.got = self.ended = None
        self.calls = []

    def tools_many(self, user_id, cfgs, timeout=90, cancel=None):
        return [list(self.tool_list) for _ in cfgs]

    def call(self, user_id, cfg, tool, args, timeout=180, cancel=None):
        from nesicode.connectors import McpResult

        self.calls.append(tool)
        self.got = cancel
        try:
            if tool == "slow" and cancel is not None and cancel.wait(10):
                raise RuntimeError("cancelado: el usuario pulsó Detener")
            return McpResult("ok", [], False)
        finally:
            self.ended = time.monotonic()


def test_stop_cancels_chrome_and_mcp_calls_themselves(tmp_path, monkeypatch):
    import nesicode.browser as browser
    import nesicode.connectors as conn
    from nesicode.server.jobs import _mcp_caller, browser_tools

    hub, manager = _CancelAwareHub(), _CancelAwareManager()
    monkeypatch.setattr(browser, "hub", hub)
    monkeypatch.setattr(conn, "manager", lambda: manager)
    cfg = conn.ConnectorConfig(name="burp", kind="http", url="http://127.0.0.1:9876/mcp")
    for fake, call in ((hub, lambda a: browser_tools(1)["chrome_read"][1]({}, a)),
                       (manager, lambda a: _mcp_caller(1, cfg, "slow")({}, a))):
        agent = _AgentStub(tmp_path / f"ws{time.time_ns()}")
        threading.Timer(0.3, agent.cancel.set).start()
        t0 = time.monotonic()
        with pytest.raises(ToolError, match="stopped by the user"):
            call(agent)
        assert time.monotonic() - t0 < 1.5
        assert fake.got is agent.cancel  # the call itself got Stop…
        assert _until(lambda: fake.ended is not None, 2)  # …and ended, instead of running on in the background


def test_a_connector_switched_off_mid_run_stops_working_at_once(env, tmp_path, monkeypatch):
    import nesicode.connectors as conn

    manager = _CancelAwareManager(tools=("ping",))
    monkeypatch.setattr(conn, "manager", lambda: manager)
    e = env(1)
    now = time.time()
    e.db.execute("INSERT INTO user_connectors (user_id, name, label, template, kind, command, created_at, usage) "
                 "VALUES (?, 'office', 'Office', 'office', 'builtin', 'office', ?, 'auto')", (e.uid, now))
    e.db.execute("INSERT INTO user_connectors (user_id, name, label, template, kind, command, created_at, usage) "
                 "VALUES (?, 'eco', 'Eco', 'custom', 'stdio', 'eco-server', ?, 'auto')", (e.uid, now))
    tools = e.jobs.connector_tools(e.uid, lambda ev: None, context=None)
    agent = _AgentStub(tmp_path / "ws")
    write, ping = tools["mcp__office__word_write"][1], tools["mcp__eco__ping"][1]
    assert "Wrote" in write({"path": "a.docx", "blocks": ["hola"]}, agent) and ping({}, agent) == "ok"
    e.db.execute("UPDATE user_connectors SET enabled=0 WHERE user_id=?", (e.uid,))  # «Desactivar» mid-conversation
    with pytest.raises(ToolError, match="turned this connector off"):
        write({"path": "b.docx", "blocks": ["no"]}, agent)
    with pytest.raises(ToolError, match="turned this connector off"):
        ping({}, agent)
    assert manager.calls == ["ping"] and not (agent.ws.root / "b.docx").exists()
    e.db.execute("DELETE FROM user_connectors WHERE user_id=?", (e.uid,))  # «Eliminar»
    with pytest.raises(ToolError, match="removed"):
        write({"path": "b.docx", "blocks": ["no"]}, agent)
    e.db.execute("INSERT INTO user_connectors (user_id, name, label, template, kind, command, created_at, usage) "
                 "VALUES (?, 'office', 'Office', 'office', 'builtin', 'office', ?, 'auto')", (e.uid, now))
    assert "Wrote" in write({"path": "c.docx", "blocks": ["otra vez"]}, agent)  # on again: it works again


# ====================================================================== Q3 — MCP servers never inherit secrets
def test_mcp_server_environment_keeps_toolchains_and_drops_secrets():
    from nesicode.connectors import server_env

    base = {
        "PATH": "/usr/bin:/bin", "HOME": "/Users/u", "USER": "u", "LANG": "es_ES.UTF-8", "LC_ALL": "es_ES.UTF-8",
        "TMPDIR": "/tmp/u", "SHELL": "/bin/zsh", "XDG_CONFIG_HOME": "/Users/u/.config", "NVM_DIR": "/Users/u/.nvm",
        "HTTPS_PROXY": "http://proxy:3128", "no_proxy": "localhost", "SSL_CERT_FILE": "/etc/ssl/cert.pem",
        "JAVA_HOME": "/jdk", "PYENV_ROOT": "/Users/u/.pyenv",
        # never: Nesicode's own settings and token, anything named like a secret, everything else
        "NESICODE_DESKTOP_TOKEN": "tok", "NESICODE_DATA_DIR": "/data", "OPENAI_API_KEY": "sk-x", "ANTHROPIC_API_KEY": "sk-y",
        "GITHUB_TOKEN": "ghp", "AWS_SECRET_ACCESS_KEY": "s", "PGPASSWORD": "p", "MY_COOKIE": "c", "XDG_SESSION_ID": "3",
        "SSH_AUTH_SOCK": "/tmp/agent", "DATABASE_URL": "postgres://u:p@h/db", "PYTHONPATH": "/src", "BASH_FUNC_x%%": "() { :; }",
    }
    env = server_env({"GITHUB_TOKEN": "del conector", "NOTION_API_KEY": "k"}, base)
    keep = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "SHELL", "XDG_CONFIG_HOME", "NVM_DIR", "HTTPS_PROXY", "no_proxy",
            "SSL_CERT_FILE", "JAVA_HOME", "PYENV_ROOT"}
    assert set(env) == keep | {"GITHUB_TOKEN", "NOTION_API_KEY"}
    assert env["GITHUB_TOKEN"] == "del conector" and all(env[k] == base[k] for k in keep)


@pytest.mark.parametrize("tool", ["npx", "uvx", "node", "python3"])
def test_toolchains_still_start_with_that_environment(tool):
    """What local MCP servers are launched with (npx …, uvx …) still finds and runs its toolchain."""
    from nesicode.connectors import server_env

    if not (exe := shutil.which(tool)):
        pytest.skip(f"{tool} is not installed")
    # started the way the MCP stdio client starts it: the command found on the PATH first (on Windows `npx` is npx.cmd,
    # which CreateProcess never finds by its bare name; a .cmd runs in cmd.exe, which needs ComSpec, PATHEXT, SystemRoot)
    r = subprocess.run([exe, "--version"], env=server_env({"npm_config_update_notifier": "false"}),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and (r.stdout or r.stderr).strip(), r


def _env_names(m, cfg) -> set[str]:
    m.tools(1, cfg, timeout=60)
    return set(json.loads(m.call(1, cfg, "env", {}).text))


SECRETS = {"OPENAI_API_KEY": "sk-FAKE", "NESICODE_DESKTOP_TOKEN": "FAKE-desktop", "SOME_SECRET": "s", "GH_TOKEN": "g"}
# What a server needs to find its toolchain and its home: Windows has no HOME (USERPROFILE is the home there), and a
# .cmd launcher (npx.cmd) needs cmd.exe to run at all
NEEDED = ({"PATH", "USERPROFILE", "SYSTEMROOT", "COMSPEC", "PATHEXT", "APPDATA", "LOCALAPPDATA", "TEMP"} if os.name == "nt"
          else {"PATH", "HOME"})


def test_a_python_mcp_server_starts_without_the_servers_secrets(mcp_script, mgr, monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    names = _env_names(mgr, _stdio("py", mcp_script, CONNECTOR_TOKEN="suyo"))
    assert not set(SECRETS) & names and NEEDED | {"CONNECTOR_TOKEN"} <= names


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_a_node_mcp_server_found_on_path_starts_without_the_servers_secrets(tmp_path, mgr, monkeypatch):
    from nesicode.connectors import ConnectorConfig

    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    script = tmp_path / "srv.js"
    script.write_text(NODE_SERVER)
    cfg = ConnectorConfig(name="nodo", kind="stdio", command="node", args=[str(script)], env={"CONNECTOR_TOKEN": "suyo"})
    names = _env_names(mgr, cfg)  # `node` itself is found through the PATH it inherits
    assert not set(SECRETS) & names and NEEDED | {"CONNECTOR_TOKEN"} <= names


# ====================================================================== Q4 — a hung MCP server
def test_a_hung_mcp_server_costs_one_timeout_per_backoff_window(mcp_script, mgr, tmp_path, fast_kill):
    pidfile = tmp_path / "mute.pid"
    mute = _stdio("lento", mcp_script, "mute", str(pidfile))
    t0 = time.monotonic()
    [res] = mgr.tools_many(1, [mute], timeout=1)
    assert isinstance(res, RuntimeError) and "no respondió en 1 s" in str(res) and 0.9 < time.monotonic() - t0 < 4
    pid = int(pidfile.read_text())
    assert _until(lambda: not _alive(pid), 8)  # dropped and stopped, not kept for the next message
    pidfile.unlink()
    t0 = time.monotonic()
    [res] = mgr.tools_many(1, [mute], timeout=1)  # the next message: no second wait
    assert time.monotonic() - t0 < 0.5 and "no se vuelve a esperar" in str(res) and "Probar" in str(res)
    assert not pidfile.exists() and not mgr.conns
    other = _stdio("lento", mcp_script, "mute", str(tmp_path / "other.pid"))  # a new config is a new server
    assert "no respondió en 1 s" in str(mgr.tools_many(1, [other], timeout=1)[0])
    mgr.stop(1, "lento")  # «Probar» / «Desactivar»: it is waited for again
    t0 = time.monotonic()
    [res] = mgr.tools_many(1, [mute], timeout=1)
    assert time.monotonic() - t0 >= 0.9 and "no respondió en 1 s" in str(res)


def test_offered_mcp_servers_start_together_and_stop_returns_at_once(mcp_script, mgr, tmp_path, fast_kill):
    a = _stdio("a", mcp_script, "mute", str(tmp_path / "a.pid"))
    b = _stdio("b", mcp_script, "mute", str(tmp_path / "b.pid"))
    good = _stdio("good", mcp_script)
    t0 = time.monotonic()
    res = mgr.tools_many(1, [a, b, good], timeout=3)
    assert time.monotonic() - t0 < 4.9  # in parallel: one timeout (one after the other: 6 s and a startup)
    assert isinstance(res[0], RuntimeError) and isinstance(res[1], RuntimeError) and {"env", "pid"} <= {t.name for t in res[2]}
    c = _stdio("c", mcp_script, "mute", str(tmp_path / "c.pid"))
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="Detener"):
        mgr.tools_many(1, [c], timeout=30, cancel=cancel)
    assert time.monotonic() - t0 < 1.2
    # the startup deadline belongs to the server, not to each wait: the next message only waits what is left of it
    while time.monotonic() - t0 < 1.5:
        time.sleep(0.05)
    t1 = time.monotonic()
    [res] = mgr.tools_many(1, [c], timeout=4)
    assert "no respondió en 4 s" in str(res) and time.monotonic() - t1 < 3.6  # (4.5 s if each wait started over)


def test_the_agent_is_told_which_connectors_could_not_start(env, models, mcp_script, tmp_path, monkeypatch, fast_kill):
    import nesicode.connectors as conn
    import nesicode.server.jobs as jobs_mod

    m = _new_manager()
    monkeypatch.setattr(conn, "_manager", m)
    monkeypatch.setattr(jobs_mod, "MCP_START_TIMEOUT", 1)
    systems = _spy_model(models)
    e = env(1)
    p = e.project("mock/spy")
    t = e.thread(p)
    e.db.execute("INSERT INTO user_connectors (user_id, name, label, template, kind, command, args, created_at, usage) "
                 "VALUES (?, 'lento', 'Servidor lento', 'custom', 'stdio', ?, ?, ?, 'auto')",
                 (e.uid, sys.executable, json.dumps([mcp_script, "mute", str(tmp_path / "m.pid")]), time.time()))
    try:
        t0 = time.monotonic()
        r1 = e.post(p, t, "hola")
        assert e.wait(r1["id"])["status"] == "done"
        first = time.monotonic() - t0
        err = [x for x in e.events(r1["id"]) if x["type"] == "connector_error"]
        assert [x["connector"] for x in err] == ["Servidor lento"] and "no respondió en 1 s" in err[0]["text"]
        assert "# Connectors that could not start" in systems[-1] and "Servidor lento" in systems[-1]
        t0 = time.monotonic()
        r2 = e.post(p, t, "y ahora")
        assert e.wait(r2["id"])["status"] == "done"
        assert time.monotonic() - t0 < first - 0.6  # the second message does not wait for it again
        err = [x for x in e.events(r2["id"]) if x["type"] == "connector_error"]
        assert "no se vuelve a esperar" in err[0]["text"] and "Servidor lento" in systems[-1]
    finally:
        _close_manager(m)


# ====================================================================== Q11 — MCP servers never outlive the app
def test_shutdown_all_stops_every_mcp_server_at_once(mcp_script, tmp_path, monkeypatch):
    import nesicode.connectors as conn

    m = _new_manager()
    monkeypatch.setattr(conn, "_manager", m)
    try:
        good = _stdio("good", mcp_script, "serve", str(tmp_path / "good.pid"))
        m.tools(1, good, timeout=60)
        stubborn = _stdio("terco", mcp_script, "stubborn", str(tmp_path / "terco.pid"))  # still starting; ignores SIGTERM
        threading.Thread(target=lambda: pytest.raises(RuntimeError, m.tools, 1, stubborn, 60), daemon=True).start()
        assert _until(lambda: (tmp_path / "terco.pid").exists() and (tmp_path / "terco.pid").read_text().strip(), 15)
        pids = [int((tmp_path / f).read_text()) for f in ("good.pid", "terco.pid")]
        t0 = time.monotonic()
        conn.shutdown_all()
        assert time.monotonic() - t0 < 9.5
        assert _until(lambda: not any(_alive(pid) for pid in pids), 2), pids
        assert conn._manager is None and m.closed
        t0 = time.monotonic()
        conn.shutdown_all()  # again: nothing to do
        assert time.monotonic() - t0 < 0.5
        with pytest.raises(RuntimeError, match="cerrando"):  # a run still finishing can't start one on it
            m.tools(1, good, timeout=5)
    finally:
        _close_manager(m)
