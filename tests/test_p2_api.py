"""Phase-2 regression tests for the HTTP API, desktop shell, limits and docs (group R: R1–R15).
Offline: fake providers, an in-process HTTP server on 127.0.0.1, temp dirs, timeouts."""

import http.server
import io
import json
import os
import signal
import socket
import stat
import threading
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nesicode.server import app as appmod
from nesicode.server.app import app, db, jobs, run_dir, thread_ws

ROOT = Path(__file__).resolve().parents[1]


def _user(prefix="r"):
    c = TestClient(app)
    r = c.post("/api/auth/register", json={"email": f"{prefix}{time.time_ns()}@example.com", "password": "secret123"})
    assert r.status_code == 200, r.text
    c.me = r.json()
    return c


def _project(c, **kw):
    r = c.post("/api/projects", json={"name": "P", "executor_model": "mock/demo", "improver_model": "mock/demo", **kw})
    assert r.status_code == 200, r.text
    p = r.json()
    Path(p["workspace"]).mkdir(parents=True, exist_ok=True)
    return p


def _row(table, row_id):
    return db.one(f"SELECT * FROM {table} WHERE id=?", (row_id,))


def _thread(c, project):
    t = c.post(f"/api/projects/{project['id']}/threads", json={}).json()
    return t, thread_ws(_row("projects", project["id"]), _row("threads", t["id"]))


def _harness(c, project):
    return c.get(f"/api/projects/{project['id']}/harness").json()["files"]


# ------------------------------------------------------------------ R1: leaving Blank, decided by the stored profile
def test_leaving_blank_after_its_opening_line_was_rewritten():
    from nesicode.harness.profiles import BLANK_SYSTEM

    c = _user("r1")
    p = _project(c, profile="blank")
    files = _harness(c, p)
    assert files["system_prompt.md"].startswith(BLANK_SYSTEM) and files["guardrails.md"] == ""
    # an improvement rewrote Blank's opening line and learned a fact
    files = {**files, "system_prompt.md": "You are a careful agent working inside a project workspace.\n",
             "memory.md": "- El proyecto usa pnpm.\n"}
    assert c.post(f"/api/projects/{p['id']}/harness", json={"files": files}).status_code == 200
    r = c.post(f"/api/projects/{p['id']}/profile", json={"profile": "general"})
    assert r.status_code == 200, r.text
    new = _harness(c, p)
    assert "Never type passwords" in new["guardrails.md"]  # the base harness is back (it left Blank)…
    assert "El proyecto usa pnpm" in new["memory.md"]  # …and what it learned stayed
    assert not new["system_prompt.md"].startswith("You are a careful agent")


def test_a_version_restored_after_switching_to_blank_is_judged_by_its_own_text():
    """Security → Blank → «Restaurar» the security version (projects.profile still says blank) → Web: the result is the
    restored harness switched to Web, never the security text grafted onto the base harness."""
    from nesicode.harness.profiles import PROFILES, apply_profile

    c = _user("r1b")
    p = _project(c, profile="security")
    v1 = _row("projects", p["id"])["harness_version"]
    assert c.post(f"/api/projects/{p['id']}/profile", json={"profile": "blank"}).status_code == 200
    assert c.post(f"/api/projects/{p['id']}/harness/restore/{v1}").status_code == 200
    restored = _harness(c, p)
    assert _row("projects", p["id"])["profile"] == "blank"
    assert c.post(f"/api/projects/{p['id']}/profile", json={"profile": "webdev"}).status_code == 200
    new = _harness(c, p)
    assert new == apply_profile(restored, "webdev")
    assert PROFILES["security"]["intro"] not in new["system_prompt.md"]


# ------------------------------------------------------------------ R2: old profile texts upgraded once at startup
def test_startup_upgrades_old_profile_texts_once_and_keeps_what_was_learned():
    from nesicode.harness import profiles

    c = _user("r2")
    p = _project(c, profile="security")
    other = _project(c, profile="general")
    files = _harness(c, p)
    legacy = {**files, "skills/security-report/SKILL.md": profiles.SEC_REPORT_0_3,
              "system_prompt.md": files["system_prompt.md"].replace(profiles.SEC_SYSTEM.strip(), profiles.SEC_SYSTEM_0_3.strip()),
              "memory.md": files.get("memory.md", "") + "- El cliente es ACME; alcance: *.acme.test.\n"}
    assert legacy["system_prompt.md"] != files["system_prompt.md"]
    assert c.post(f"/api/projects/{p['id']}/harness", json={"files": legacy}).status_code == 200
    others = len(c.get(f"/api/projects/{other['id']}/harness/versions").json())

    appmod._refresh_profiles()
    new = _harness(c, p)
    assert new["skills/security-report/SKILL.md"] == profiles.SKILL_TEXT["security-report"]
    assert profiles.SEC_SYSTEM.strip() in new["system_prompt.md"] and profiles.SEC_SYSTEM_0_3.strip() not in new["system_prompt.md"]
    assert "El cliente es ACME" in new["memory.md"]
    versions = c.get(f"/api/projects/{p['id']}/harness/versions").json()
    assert versions[0]["note"] == appmod.PROFILE_REFRESH_NOTE
    appmod._refresh_profiles()  # a later startup: nothing to do
    assert len(c.get(f"/api/projects/{p['id']}/harness/versions").json()) == len(versions)
    assert len(c.get(f"/api/projects/{other['id']}/harness/versions").json()) == others


# ------------------------------------------------------------------ R3: clean shutdown
def test_stop_runtime_stops_new_work_then_runs_and_mcp_servers_once(monkeypatch):
    import nesicode.connectors as connectors

    calls = []

    class Jobs:
        def shutdown(self, wait=0.0):
            calls.append(("jobs", wait))

    class Bots:
        def stop(self):
            calls.append(("bots",))

    class Telegram:
        bridges = {7: object()}

        def stop(self, uid):
            calls.append(("telegram", uid))

    monkeypatch.setattr(appmod, "jobs", Jobs())
    monkeypatch.setattr(appmod, "bots", Bots())
    monkeypatch.setattr(appmod, "telegram", Telegram())
    monkeypatch.setattr(appmod, "_stopped", False)
    monkeypatch.setattr(connectors, "shutdown_all", lambda: calls.append(("mcp",)), raising=False)
    appmod.stop_runtime(wait=0.5)
    # nothing new can start, then the runs stop, then (meanwhile) the MCP servers, then the runs get their seconds
    assert calls[:3] == [("bots",), ("telegram", 7), ("jobs", 0)]
    assert sorted(calls[3:]) == [("jobs", 0.5), ("mcp",)]
    appmod.stop_runtime()  # idempotent (the lifespan, the desktop window and the exit hook may all call it)
    assert len(calls) == 5


def test_app_lifespan_shutdown_and_exit_hook_stop_the_runtime(monkeypatch):
    import threading as _threading

    stopped, original = [], appmod.stop_runtime
    monkeypatch.setattr(appmod, "stop_runtime", lambda *a, **k: stopped.append(1))
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
        assert not stopped
    assert stopped == [1]  # the server's shutdown of this app
    hooks = getattr(_threading, "_threading_atexits", None)
    if hooks is not None:  # (CPython) also at interpreter exit, which a forced server exit does not skip
        assert any(getattr(h, "func", h) is original for h in hooks)  # (registered as a functools.partial)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_ctrl_c_on_the_server_does_not_wait_for_a_reply_followed_in_a_browser(tmp_path):
    """`nesicode serve` with a browser tab following a reply that waits for an approval: the first Ctrl+C stops it
    at once (the run is kept as interrupted) instead of the server waiting up to 30 minutes for that event stream."""
    import http.client
    import sqlite3
    import subprocess
    import sys

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    script = tmp_path / "serve.py"
    script.write_text("import sys, uvicorn\n"
                      "uvicorn.run('nesicode.server.app:app', host='127.0.0.1', port=int(sys.argv[1]), log_level='warning')\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("NESICODE_") and not k.endswith("_API_KEY")}
    env.update({"PYTHONPATH": str(ROOT / "src"), "NESICODE_NO_DOTENV": "1", "NESICODE_DATA_DIR": str(tmp_path / "data"),
                "NESICODE_AUTH": "false", "NESICODE_EXECUTOR_MODEL": "mock/demo", "NESICODE_IMPROVER_MODEL": "mock/demo",
                "NESICODE_SANDBOX": "local", "NESICODE_NO_TELEGRAM": "1", "NESICODE_NO_SCHEDULER": "1",
                "NESICODE_VAULT_KEY": os.environ["NESICODE_VAULT_KEY"]})
    proc = subprocess.Popen([sys.executable, str(script), str(port)], env=env, cwd=tmp_path,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def api(method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
        data = conn.getresponse().read()
        conn.close()
        return json.loads(data)

    try:
        deadline = time.time() + 30
        while True:
            try:
                api("GET", "/api/health")
                break
            except OSError:
                assert time.time() < deadline, "the server did not start"
                time.sleep(0.2)
        p = api("POST", "/api/projects", {"name": "S"})
        t = api("POST", f"/api/projects/{p['id']}/threads", {})
        api("PATCH", f"/api/threads/{t['id']}", {"exec_mode": "ask_all"})
        run = api("POST", f"/api/threads/{t['id']}/messages", {"text": "escribe el informe"})
        deadline = time.time() + 15
        while not any(e["type"] == "approval_request" for e in api("GET", f"/api/runs/{run['id']}")["events"]):
            assert time.time() < deadline, "no approval request"
            time.sleep(0.1)
        stream = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
        stream.request("GET", f"/api/runs/{run['id']}/stream")
        resp = stream.getresponse()
        ended = threading.Event()
        threading.Thread(target=lambda: (resp.read(), ended.set()), daemon=True).start()  # a browser tab following it
        time.sleep(0.5)
        t0 = time.time()
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=15)  # (before: still waiting for that stream)
        assert time.time() - t0 < 10 and ended.wait(5)
        status = sqlite3.connect(tmp_path / "data" / "nesicode.db").execute("SELECT status FROM runs WHERE id=?",
                                                                             (run["id"],)).fetchone()[0]
        assert status == "interrupted"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(5)


def test_desktop_window_close_stops_the_runtime_before_the_server():
    from nesicode import desktop

    order = []

    class Server:
        def __setattr__(self, name, value):
            order.append((name, value))

    class Thread:
        def join(self, timeout=None):
            order.append(("join", timeout))

    desktop._close(Server(), Thread(), lambda: order.append("stop_runtime"))
    assert order == ["stop_runtime", ("should_exit", True), ("join", 5)]


# ------------------------------------------------------------------ R4: no request hangs on the project lock
def _harness_run(project, files: dict[str, str], *, result: dict | None = None, harness_version=None) -> int:
    rid = db.insert("INSERT INTO runs (project_id, kind, task, status, result, harness_version, created_at) "
                    "VALUES (?, 'harness', 't', 'done', ?, ?, ?)",
                    (project["id"], json.dumps(result or {"selected": "x1"}), harness_version, time.time()))
    rd = run_dir(_row("projects", project["id"]), rid)
    (rd / "source").mkdir(parents=True)
    for rel, text in files.items():
        f = rd / "rollouts" / "x1" / "ws" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return rid


def test_apply_endpoints_answer_409_instead_of_waiting_for_a_busy_project(monkeypatch):
    c = _user("r4")
    p = _project(c)
    t, cage = _thread(c, p)
    (cage / "nuevo.txt").write_text("hola\n")
    rid = _harness_run(p, {"de_la_ejecucion.txt": "x\n"})
    lock = jobs._project_lock(p["id"])
    assert lock.acquire(timeout=2)  # a shared conversation is working (it holds the lock for its whole run)
    try:
        for url in (f"/api/threads/{t['id']}/apply", f"/api/runs/{rid}/apply"):
            t0 = time.time()
            r = c.post(url)
            assert r.status_code == 409 and r.json()["detail"] == appmod.BUSY_MSG, (url, r.text)
            assert time.time() - t0 < appmod.LOCK_WAIT + 2
    finally:
        lock.release()
    assert not (Path(p["workspace"]) / "nuevo.txt").exists()
    assert c.post(f"/api/threads/{t['id']}/apply").status_code == 200  # free again: it applies
    assert (Path(p["workspace"]) / "nuevo.txt").read_text() == "hola\n"

    def busy(run_id, project):  # the jobs side gives up on a lock taken meanwhile
        raise appmod.ProjectBusy(appmod.BUSY_MSG)

    monkeypatch.setattr(appmod.jobs, "apply_later", busy)
    r = c.post(f"/api/runs/{rid}/apply")
    assert r.status_code == 409 and r.json()["detail"] == appmod.BUSY_MSG


# ------------------------------------------------------------------ R5: adopt, improve, shared archives
def test_manual_adopt_merges_onto_later_edits_and_never_reverts_them():
    c = _user("r5a")
    p = _project(c)
    base_v = _row("projects", p["id"])["harness_version"]
    base = _harness(c, p)
    run_files = {**base, "guardrails.md": base["guardrails.md"] + "- Ejecuta los tests antes de terminar.\n"}
    rid = _harness_run(p, {}, result={"selected": "x1", "adopted_harness": run_files}, harness_version=base_v)
    # meanwhile the user edited another file of the harness
    mine = {**base, "memory.md": base.get("memory.md", "") + "- Despliegue: viernes.\n"}
    assert c.post(f"/api/projects/{p['id']}/harness", json={"files": mine}).status_code == 200
    r = c.post(f"/api/runs/{rid}/adopt")
    assert r.status_code == 200 and r.json().get("merged") is True, r.text
    now = _harness(c, p)
    assert "Ejecuta los tests" in now["guardrails.md"] and "Despliegue: viernes" in now["memory.md"]
    assert json.loads(_row("runs", rid)["result"])["adopted_version"] == r.json()["version"]
    # pressed again: its changes are already there, no duplicate version
    again = c.post(f"/api/runs/{rid}/adopt").json()
    assert again["unchanged"] is True and again["version"] == _row("projects", p["id"])["harness_version"]


def test_manual_adopt_with_a_conflict_is_refused_and_changes_nothing():
    c = _user("r5b")
    p = _project(c)
    base_v = _row("projects", p["id"])["harness_version"]
    base = _harness(c, p)
    rid = _harness_run(p, {}, harness_version=base_v,
                       result={"selected": "x1", "adopted_harness": {**base, "memory.md": "- versión de la ejecución\n"}})
    assert c.post(f"/api/projects/{p['id']}/harness", json={"files": {**base, "memory.md": "- mi versión\n"}}).status_code == 200
    before = _row("projects", p["id"])["harness_version"]
    r = c.post(f"/api/runs/{rid}/adopt")
    assert r.status_code == 409 and "memory.md" in r.json()["detail"] and f"v{base_v}" in r.json()["detail"]
    assert _row("projects", p["id"])["harness_version"] == before and "mi versión" in _harness(c, p)["memory.md"]


def test_improve_from_a_conversation_that_asks_before_commands_is_a_409():
    c = _user("r5c")
    p = _project(c)
    t, _ = _thread(c, p)
    assert c.patch(f"/api/threads/{t['id']}", json={"exec_mode": "ask_bash"}).status_code == 200
    rid = db.insert("INSERT INTO runs (project_id, thread_id, kind, task, status, created_at) VALUES (?, ?, 'chat', 'x', 'done', ?)",
                    (p["id"], t["id"], time.time()))
    r = c.post(f"/api/runs/{rid}/improve")
    assert r.status_code == 409 and "permiso" in r.json()["detail"]
    assert not db.one("SELECT id FROM runs WHERE project_id=? AND kind='harness'", (p["id"],))


def test_deleting_a_shared_conversation_removes_its_archive_never_through_a_link(tmp_path):
    c = _user("r5d")
    assert c.put("/api/me/settings", json={"share_workspace": True}).status_code == 200
    p = _project(c)
    ws = Path(p["workspace"])
    t1 = c.post(f"/api/projects/{p['id']}/threads", json={}).json()
    t2 = c.post(f"/api/projects/{p['id']}/threads", json={}).json()
    assert _row("threads", t1["id"])["isolated"] == 0
    arch = ws / ".nesicode" / "chats" / str(t1["id"]) / "context"
    arch.mkdir(parents=True)
    (arch / "compact-1.md").write_text("resumen\n")
    (ws / ".nesicode" / "chats" / str(t2["id"])).mkdir()
    assert c.delete(f"/api/threads/{t1['id']}").status_code == 200
    assert not (ws / ".nesicode" / "chats" / str(t1["id"])).exists()
    assert (ws / ".nesicode" / "chats" / str(t2["id"])).is_dir()  # only that conversation's
    # an agent of the shared folder replaced chats/ with a link to somewhere else: nothing there is touched
    outside = tmp_path / "outside"
    (outside / str(t2["id"])).mkdir(parents=True)
    (outside / str(t2["id"]) / "keep.txt").write_text("keep\n")
    import shutil

    shutil.rmtree(ws / ".nesicode" / "chats")
    (ws / ".nesicode" / "chats").symlink_to(outside, target_is_directory=True)
    assert c.delete(f"/api/threads/{t2['id']}").status_code == 200
    assert (outside / str(t2["id"]) / "keep.txt").read_text() == "keep\n"
    c.put("/api/me/settings", json={"share_workspace": False})


# ------------------------------------------------------------------ R6: no half-created rows
def test_a_cage_that_cannot_be_made_leaves_no_conversation(monkeypatch):
    c = _user("r6a")
    p = _project(c)
    before = db.one("SELECT COUNT(*) AS n FROM threads WHERE project_id=?", (p["id"],))["n"]
    secret_dir = str(appmod.settings.data_dir)

    def broken(project, thread_id):
        (appmod.thread_dir(project, thread_id) / "ws").mkdir(parents=True)
        raise OSError(f"[Errno 13] Permission denied: '{secret_dir}/workspaces/x/locked.db'")

    monkeypatch.setattr(appmod, "create_cage", broken)
    r = c.post(f"/api/projects/{p['id']}/threads", json={})
    assert r.status_code == 422 and "Permission denied" in r.json()["detail"] and secret_dir not in r.json()["detail"]
    assert db.one("SELECT COUNT(*) AS n FROM threads WHERE project_id=?", (p["id"],))["n"] == before
    assert not any((appmod.project_root(_row("projects", p["id"])) / "threads").glob("*/ws"))


def test_provider_keys_are_checked_before_anything_is_stored(monkeypatch):
    monkeypatch.setattr(appmod, "_refresh_limits_bg", lambda *a: None)  # (offline: no limits lookup)
    c = _user("r6b")
    body = {"name": "nl", "kind": "openai", "base_url": "https://api.example.invalid/v1", "models": ["m"],
            "api_key": "sk-abc\ndef"}
    r = c.post("/api/providers", json=body)
    assert r.status_code == 422 and "API key" in r.json()["detail"]
    assert not db.one("SELECT id FROM user_providers WHERE user_id=? AND name='nl'", (c.me["id"],))
    assert appmod.vault.get(c.me["id"], "provider:nl") is None
    r = c.post("/api/providers", json={**body, "api_key": "  sk-abc123  "})  # surrounding spaces are only trimmed
    assert r.status_code == 200 and appmod.vault.get(c.me["id"], "provider:nl") == "sk-abc123"
    assert c.patch("/api/providers/nl", json={"api_key": "sk nueva"}).status_code == 422
    assert appmod.vault.get(c.me["id"], "provider:nl") == "sk-abc123"
    assert c.post("/api/providers/test", json={"kind": "openai", "base_url": "https://x.invalid/v1", "api_key": "ñ"}).status_code == 422


def test_saved_and_connect_time_address_rules_are_the_same():
    from nesicode.llm.transport import address_problem
    from nesicode.server.limits import endpoint_problem

    for addr in ("127.0.0.1", "10.1.2.3", "169.254.169.254", "100.64.0.1", "0.0.0.0", "::1", "fd00::1", "::ffff:10.0.0.1",
                 "224.0.0.1", "fe80::1%en0", "192.0.2.10", "93.184.216.34", "2606:4700::1111"):
        assert appmod._non_public_ip(addr) == bool(endpoint_problem(addr)), addr
        if address_problem(addr, strict=True):  # never laxer than the LLM layer's own rule
            assert endpoint_problem(addr), addr
    assert not endpoint_problem("93.184.216.34") and endpoint_problem("100.64.0.1")


# ------------------------------------------------------------------ R7: user endpoints vetted at connect time
class _Endpoint:
    """An OpenAI-like server on 127.0.0.1 that counts the requests it really gets."""

    def __init__(self):
        hits = self.hits = []

        class H(http.server.BaseHTTPRequestHandler):
            def _send(self, body: dict, status=200):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                hits.append(self.path)
                models = self.path.endswith("/models")
                self._send({"data": [{"id": "m1", "context_length": 4096}]} if models else {}, 200 if models else 404)

            def do_POST(self):  # noqa: N802
                hits.append(self.path)
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self._send({"text": "hola"})

            def log_message(self, *a):
                pass

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture()
def rebound(monkeypatch):
    """A user provider whose name (rebind.test) passed the check when it was saved and now resolves to 127.0.0.1."""
    ep = _Endpoint()
    real = socket.getaddrinfo

    def fake(host, port, *args, **kw):
        if host == "rebind.test":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port or 0))]
        return real(host, port, *args, **kw)

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    c = _user("r7")
    now = time.time()
    db.insert("INSERT INTO user_providers (user_id, name, label, kind, base_url, models, created_at, updated_at) "
              "VALUES (?, 'reb', 'reb', 'openai', ?, '[\"m1\"]', ?, ?)", (c.me["id"], f"http://rebind.test:{ep.port}/v1", now, now))
    appmod.vault.put(c.me["id"], "provider:reb", "sk-secret-key")
    p = _project(c)
    assert c.patch(f"/api/projects/{p['id']}", json={"transcribe_model": "reb/whisper"}).status_code == 200
    yield c, p, ep
    ep.close()


def _transcribe(c, p):
    return c.post(f"/api/projects/{p['id']}/transcribe", files={"audio": ("a.wav", b"RIFF" + b"\0" * 4000, "audio/wav")},
                  data={"lang": "es"})


def test_user_endpoints_are_refused_at_connect_time_after_a_dns_change(rebound, monkeypatch):
    c, p, ep = rebound
    monkeypatch.setenv("NESICODE_ALLOW_PRIVATE_PROVIDERS", "0")
    r = c.post("/api/providers/reb/discover")
    assert r.status_code == 502 and "refused to connect" in r.json()["detail"], r.text
    lim = c.get("/api/models/limits", params={"spec": "reb/m1", "refresh": True}).json()
    assert lim["source"] in ("table", "default")  # the provider was never asked
    r = _transcribe(c, p)
    assert r.status_code == 422 and "refused to connect" in r.json()["detail"], r.text
    assert ep.hits == []  # nothing (no key, no audio) ever reached the private address
    # the operator's own (built-in) providers are never guarded
    assert appmod.fetch_models("openai", f"http://127.0.0.1:{ep.port}/v1", "", mine=False) == ["m1"]


def test_the_same_requests_go_through_where_private_networks_are_allowed(rebound, monkeypatch):
    c, p, ep = rebound
    monkeypatch.setenv("NESICODE_ALLOW_PRIVATE_PROVIDERS", "1")
    assert c.post("/api/providers/reb/discover").json()["models"] == ["m1"]
    assert c.get("/api/models/limits", params={"spec": "reb/m1", "refresh": True}).json()["source"] == "provider"
    assert _transcribe(c, p).json()["text"] == "hola"
    assert "/v1/audio/transcriptions" in ep.hits


# ------------------------------------------------------------------ R8: the SPA's CSP
def test_the_app_page_sends_a_content_security_policy_and_no_referrer():
    csp = appmod.spa_csp()
    rules = dict((d.split(" ", 1) + [""])[:2] for d in csp.split("; "))
    assert rules["default-src"] == "'self'" and rules["script-src"] == "'self'"  # no inline or injected scripts
    assert "'unsafe-inline'" not in csp and "'unsafe-eval'" not in csp
    assert rules["img-src"].split()[:3] == ["'self'", "data:", "blob:"]
    assert rules["connect-src"] == "'self'" and rules["object-src"] == "'none'" and rules["frame-ancestors"] == "'none'"
    assert rules["base-uri"] == "'none'" and rules["frame-src"] == "'self'"
    assert "'unsafe-eval'" in appmod.spa_csp(desktop=True).split("; ")[1]  # pywebview's bridge builds its API with new Function
    assert appmod.SPA_HEADERS["Referrer-Policy"] == "no-referrer"
    if appmod.WEB_DIST.is_dir():  # (a checkout with the UI built)
        r = TestClient(app).get("/")
        assert r.headers["content-security-policy"] == csp and r.headers["referrer-policy"] == "no-referrer"


def test_generated_page_previews_send_no_referrer():
    c = _user("r8")
    p = _project(c)
    t, cage = _thread(c, p)
    (cage / "informe.html").write_text("<h1>hola</h1>")
    r = c.get(f"/api/threads/{t['id']}/file", params={"path": "informe.html", "view": 1})
    assert r.status_code == 200 and r.headers["referrer-policy"] == "no-referrer" and "sandbox" in r.headers["content-security-policy"]


# ------------------------------------------------------------------ R9: removing a pending follow-up
def test_a_pending_one_off_follow_up_can_be_removed():
    c = _user("r9")
    p = _project(c)
    b = c.post("/api/bots", json={"project_id": p["id"], "name": "Vigía", "instructions": "x",
                                  "triggers": [{"kind": "interval", "spec": "60"}]}).json()
    once = db.insert("INSERT INTO bot_triggers (bot_id, kind, prompt, next_at, created_at) VALUES (?, 'once', 'revisa', ?, ?)",
                     (b["id"], time.time() + 3600, time.time()))  # what bot_schedule_once stores
    c.post(f"/api/bots/{b['id']}/webhook")
    hook = db.one("SELECT id FROM bot_triggers WHERE bot_id=? AND kind='webhook'", (b["id"],))["id"]
    other = _user("r9o")
    assert other.delete(f"/api/bots/{b['id']}/triggers/{once}").status_code == 404  # not their bot
    assert c.delete(f"/api/bots/{b['id']}/triggers/{hook}").status_code == 404  # webhooks: their own endpoint
    b2 = c.post("/api/bots", json={"project_id": p["id"], "name": "Otro", "instructions": "y"}).json()
    assert c.delete(f"/api/bots/{b2['id']}/triggers/{once}").status_code == 404  # not that bot's trigger
    r = c.delete(f"/api/bots/{b['id']}/triggers/{once}")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert not _row("bot_triggers", once) and _row("bot_triggers", hook)
    assert [t["kind"] for t in c.get(f"/api/bots/{b['id']}").json()["triggers"]] == ["interval"]
    assert c.delete(f"/api/bots/{b['id']}/triggers/{once}").status_code == 404


# ------------------------------------------------------------------ R10: connectors UI
def test_connector_names_transliterate_labels():
    assert appmod.connector_slug("Navegación") == "navegacion"
    assert appmod.connector_slug("Straße Ærø  Łódź") == "strasse-aero-lodz"
    assert appmod.connector_slug("数据库") == "" and appmod.connector_slug("__Mi API__") == "mi-api"
    c = _user("r10a")
    names = []
    for label in ("数据库", "数据库", "MCP", "Navegación", "Navegación"):
        names.append(appmod._unique_connector_name(c.me["id"], label))
        db.insert("INSERT INTO user_connectors (user_id, name, label, kind, created_at) VALUES (?, ?, ?, 'http', ?)",
                  (c.me["id"], names[-1], label, time.time()))
    assert names == ["mcp-1", "mcp-2", "mcp-3", "navegacion", "navegacion-2"]


def test_connectors_list_their_trigger_words(monkeypatch):
    from nesicode.connectors import mentioned

    c = _user("r10b")
    monkeypatch.setattr(appmod, "_start_and_list", lambda user, name: ["buscar"])
    monkeypatch.setattr(appmod, "check_base_url", lambda url: url)
    for label in ("Navegación interna", "数据库"):
        r = c.post("/api/connectors/custom", json={"label": label, "kind": "http", "url": "https://mcp.example.com/mcp"})
        assert r.status_code == 200, r.text
    data = c.get("/api/connectors").json()
    mine = {x["label"]: x for x in data["connectors"]}
    assert mine["Navegación interna"]["name"] == "navegacion-interna"
    assert "navegacion" in mine["Navegación interna"]["trigger_words"]
    assert mine["数据库"]["name"] == "mcp-1" and "数据库" in mine["数据库"]["trigger_words"]
    assert all(isinstance(x["trigger_words"], list) for x in data["catalog"])
    assert "burp" in next(x for x in data["catalog"] if x["id"] == "burp")["trigger_words"]
    assert isinstance(c.get("/api/browser").json()["trigger_words"], list)
    # its tool names keep it on; other connectors' tools never switch it on
    assert mentioned("sigue con mcp__mcp-1__buscar", name="mcp-1", label="数据库", template="custom")
    assert not mentioned("mcp__office__word_write mcp__burp__send", name="mcp-1", label="数据库", template="custom")


# ------------------------------------------------------------------ R11: forgetting a learned limit
def test_a_learned_limit_can_be_forgotten():
    from nesicode.server.limits import forget_learned, learn_limit

    c = _user("r11")
    spec = "mock/demo"
    learn_limit(db, c.me["id"], spec, 8192)
    assert c.get("/api/models/limits", params={"spec": spec}).json() == {"spec": spec, "context": 8192,
                                                                         "output": 16000, "source": "learned"}
    r = c.delete("/api/models/limits", params={"spec": spec})
    assert r.status_code == 200 and r.json()["source"] != "learned" and r.json()["context"] != 8192
    assert set(r.json()) == {"spec", "context", "output", "source"}
    assert c.get("/api/models/limits", params={"spec": spec}).json() == r.json()
    assert forget_learned(db, c.me["id"], spec) is False  # nothing learned any more
    other = _user("r11o")
    learn_limit(db, other.me["id"], spec, 4096)
    c.delete("/api/models/limits", params={"spec": spec})
    assert other.get("/api/models/limits", params={"spec": spec}).json()["context"] == 4096  # only one's own


# ------------------------------------------------------------------ R12: deleting conversations, webhooks
def test_deleting_a_conversation_only_deletes_the_bot_that_really_owns_it():
    c = _user("r12a")
    p = _project(c)
    b = c.post("/api/bots", json={"project_id": p["id"], "name": "Guardia", "instructions": "x"}).json()
    t, _ = _thread(c, p)
    db.execute("UPDATE threads SET bot_id=? WHERE id=?", (b["id"], t["id"]))  # a stale link (ids are reused)
    assert c.delete(f"/api/threads/{t['id']}").status_code == 200
    assert _row("bots", b["id"])  # the bot lives on: its conversation is another one
    own = _row("bots", b["id"])["thread_id"]
    assert c.delete(f"/api/threads/{own}").status_code == 200
    assert not _row("bots", b["id"])  # its own conversation takes it along


def test_a_webhook_needs_its_trigger_row_not_only_a_secret():
    c = _user("r12b")
    p = _project(c)
    b = c.post("/api/bots", json={"project_id": p["id"], "name": "Hook", "instructions": "x"}).json()
    token = c.post(f"/api/bots/{b['id']}/webhook").json()["token"]
    db.execute("DELETE FROM bot_triggers WHERE bot_id=? AND kind='webhook'", (b["id"],))
    assert appmod.vault.get(c.me["id"], f"bot:{b['id']}:webhook") == token  # the secret is still there…
    r = TestClient(app).post(f"/hooks/{b['id']}", headers={"X-Nesicode-Token": token}, content=b"{}")
    assert r.status_code == 404  # …but without its trigger row the hook does not exist
    assert not db.one("SELECT id FROM runs WHERE bot_id=?", (b["id"],))


# ------------------------------------------------------------------ R13: ZIP uploads
def _zip(entries) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            z.writestr(name, data)
    return buf.getvalue()


def _upload(c, p, data: bytes):
    return c.post(f"/api/projects/{p['id']}/upload", files={"file": ("p.zip", data, "application/zip")})


def test_zip_uploads_are_capped_and_never_unpack_links_or_leave_the_project(monkeypatch, tmp_path):
    c = _user("r13")
    p = _project(c)
    ws = Path(p["workspace"])
    monkeypatch.setattr(appmod, "ZIP_MAX_ENTRIES", 10)
    r = _upload(c, p, _zip([(f"f{i}.txt", "x") for i in range(11)]))
    assert r.status_code == 413 and "entradas" in r.json()["detail"] and not (ws / "f0.txt").exists()
    monkeypatch.setattr(appmod, "ZIP_MAX_BYTES", 1024 * 1024)
    r = _upload(c, p, _zip([("bomba.bin", b"\0" * (2 * 1024 * 1024))]))  # 2 MB of zeros: a few KB packed
    assert r.status_code == 413 and not (ws / "bomba.bin").exists()
    for bad in ("../fuera.txt", "/etc/fuera.txt", "a/../../fuera.txt"):
        r = _upload(c, p, _zip([("ok.txt", "x"), (bad, "y")]))
        assert r.status_code == 422, bad
    assert not (ws.parent / "fuera.txt").exists() and not (ws / "ok.txt").exists()  # refused before writing
    link = zipfile.ZipInfo("enlace")
    link.create_system, link.external_attr = 3, (stat.S_IFLNK | 0o777) << 16
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(link, str(tmp_path / "secreto"))
        z.writestr("src/app.py", "print('hola')\n")
    r = _upload(c, p, buf.getvalue())
    assert r.status_code == 200 and r.json()["files"] == 1 and r.json()["skipped"][0].startswith("enlace:")
    assert not os.path.lexists(ws / "enlace") and (ws / "src" / "app.py").read_text() == "print('hola')\n"


# ------------------------------------------------------------------ R14: documented settings
def test_settings_are_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for var in ("NESICODE_LLM_FIRST_BYTE_TIMEOUT", "NESICODE_LLM_IDLE_TIMEOUT", "NESICODE_ALLOWED_HOSTS",
                "NESICODE_ALLOWED_ORIGINS", "NESICODE_ALLOW_PRIVATE_PROVIDERS"):
        assert var in readme and var in example, var
    assert "KERN_PROCARGS2" in readme


# ------------------------------------------------------------------ R15: what Apply skipped stays listed
def test_changes_that_could_not_be_applied_stay_listed_until_resolved(tmp_path):
    c = _user("r15")
    p = _project(c)
    t, cage = _thread(c, p)
    (cage / "listo.txt").write_text("hecho\n")
    (cage / "fuera").symlink_to(tmp_path)  # a link out of the project is never created there
    r = c.post(f"/api/threads/{t['id']}/apply")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added"] == ["listo.txt"] and [s.split(":")[0] for s in body["skipped"]] == ["fuera"]
    ch = c.get(f"/api/threads/{t['id']}/changes").json()
    assert ch["added"] == ["fuera"] and not ch["modified"] and not ch["removed"]  # still pending, listo.txt is done
    assert not os.path.lexists(Path(p["workspace"]) / "fuera")
    (cage / "fuera").unlink()  # undone in the conversation: nothing pending any more
    ch = c.get(f"/api/threads/{t['id']}/changes").json()
    assert ch["added"] == [] and ch["modified"] == [] and ch["removed"] == []
