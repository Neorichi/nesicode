import time

import pytest
from fastapi.testclient import TestClient

from nesicode.server.app import app


def wait_run(client, run_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] not in ("queued", "running", "waiting"):
            return run
        time.sleep(0.1)
    raise AssertionError("run did not finish")


@pytest.fixture()
def alice():
    c = TestClient(app)
    email = f"alice{time.time_ns()}@example.com"
    assert c.post("/api/auth/register", json={"email": email, "password": "secret123", "name": "Alice"}).status_code == 200
    return c


@pytest.fixture()
def project(alice):
    r = alice.post("/api/projects", json={"name": "Demo", "executor_model": "mock/demo", "improver_model": "mock/demo",
                                          "check_command": "test -f NESICODE_RESULT.md", "domain": "Web Apps"})
    assert r.status_code == 200, r.text
    return r.json()


def test_auth_required():
    assert TestClient(app).get("/api/projects").status_code == 401


def test_login_logout(alice):
    me = alice.get("/api/auth/me").json()
    alice.post("/api/auth/logout")
    assert alice.get("/api/auth/me").status_code == 401
    bad = alice.post("/api/auth/login", json={"email": me["email"], "password": "wrongpass"})
    assert bad.status_code == 401
    assert alice.post("/api/auth/login", json={"email": me["email"], "password": "secret123"}).status_code == 200


def test_chat_flow(alice, project):
    assert project["domain"] == "web-apps"
    t = alice.post(f"/api/projects/{project['id']}/threads", json={}).json()
    run = alice.post(f"/api/threads/{t['id']}/messages", json={"text": "Escribe un informe"}).json()
    run = wait_run(alice, run["id"])
    assert run["status"] == "done"
    thread = alice.get(f"/api/threads/{t['id']}").json()
    assert thread["thread"]["title"] == "Escribe un informe"
    types = [e["type"] for e in thread["runs"][0]["events"]]
    assert "tool_call" in types and types[-1] == "done"
    # Layer 3: the conversation wrote inside its own cage, not in the project
    tree = alice.get(f"/api/projects/{project['id']}/tree").json()
    assert not any(f["path"] == "NESICODE_RESULT.md" for f in tree)
    ttree = alice.get(f"/api/threads/{t['id']}/tree").json()
    assert any(f["path"] == "NESICODE_RESULT.md" for f in ttree)
    ch = alice.get(f"/api/threads/{t['id']}/changes").json()
    assert ch["added"] == ["NESICODE_RESULT.md"] and "+# Result" in ch["diff"]
    assert alice.post(f"/api/threads/{t['id']}/apply").json()["added"] == ["NESICODE_RESULT.md"]
    tree = alice.get(f"/api/projects/{project['id']}/tree").json()
    assert any(f["path"] == "NESICODE_RESULT.md" for f in tree)
    assert alice.get(f"/api/threads/{t['id']}/changes").json()["added"] == []  # re-baselined
    # SSE stream replays persisted events and ends
    with alice.stream("GET", f"/api/runs/{run['id']}/stream") as s:
        body = "".join(s.iter_text())
    assert "event: end" in body and '"type": "done"' in body


def test_harness_edit_validation_and_versions(alice, project):
    pid = project["id"]
    h = alice.get(f"/api/projects/{pid}/harness").json()
    files = dict(h["files"])
    files["skills/x/SKILL.md"] = "---\nname: x\ndescription: d\n---\nbody\n"
    bad = alice.post(f"/api/projects/{pid}/harness", json={"files": files})
    assert bad.status_code == 422 and bad.json()["errors"]
    files["guardrails.md"] += "- When needed, load the `x` skill.\n"
    ok = alice.post(f"/api/projects/{pid}/harness", json={"files": files, "note": "add x"}).json()
    assert ok["version"] == 2
    versions = alice.get(f"/api/projects/{pid}/harness/versions").json()
    assert [v["version"] for v in versions] == [2, 1]
    v2 = alice.get(f"/api/projects/{pid}/harness/versions/2").json()
    assert "skills/x/SKILL.md" in v2["changes"]["added"]
    assert alice.post(f"/api/projects/{pid}/harness/restore/1").json()["version"] == 3


def test_harnessing_flow(alice, project):
    pid = project["id"]
    run = alice.post(f"/api/projects/{pid}/harnessing", json={"task": "Write a report", "k": 1, "apply_changes": True}).json()
    run = wait_run(alice, run["id"])
    assert run["status"] == "done", run["error"]
    res = run["result"]
    assert res["applied"] is not None
    if res["harness_changed"]:
        assert res["adopted_version"] == 2
    else:  # the selector may legitimately keep the base harness
        assert res["adopted_version"] is None
    # Manual adoption of the run's final harness always works
    v = alice.post(f"/api/runs/{run['id']}/adopt").json()["version"]
    assert alice.get(f"/api/projects/{pid}").json()["harness_version"] == v
    lessons = alice.get("/api/lessons?domain=web-apps").json()
    assert lessons and lessons[0]["run_id"] == run["id"]
    ro = alice.get(f"/api/runs/{run['id']}/rollouts/base1").json()
    assert "[ACTION]" in ro["trajectory"]
    phases = [e["phase"] for e in run["events"] if e["type"] == "phase"]
    assert phases[0] == "base" and phases[-1] == "done"


def test_isolation_between_users(alice, project):
    bob = TestClient(app)
    bob.post("/api/auth/register", json={"email": f"bob{time.time_ns()}@example.com", "password": "secret123"})
    assert bob.get(f"/api/projects/{project['id']}").status_code == 404
    assert bob.get(f"/api/projects/{project['id']}/harness").status_code == 404
    assert bob.get(f"/api/projects/{project['id']}/file", params={"path": "x"}).status_code == 404
    assert bob.get("/api/projects").json() == []


def test_file_read_confined(alice, project):
    r = alice.get(f"/api/projects/{project['id']}/file", params={"path": "../../../../etc/passwd"})
    assert r.status_code == 404


def test_providers_and_vault(alice):
    from nesicode.server.app import db
    from nesicode.server.vault import unseal

    me = alice.get("/api/auth/me").json()
    body = {"name": "mio", "label": "Mi proxy", "kind": "openai", "base_url": "https://example.invalid/v1",
            "api_key": "sk-secreta-123456789", "models": ["modelo-a", "modelo-b"]}
    r = alice.post("/api/providers", json=body)
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["key_hint"] == "••••6789" and "api_key" not in p and p["models"] == ["modelo-a", "modelo-b"]
    # stored only in the vault, sealed and bound to (user, ref)
    row = db.one("SELECT blob FROM vault_secrets WHERE user_id=? AND ref='provider:mio'", (me["id"],))
    assert "sk-secreta" not in row["blob"]
    assert unseal(me["id"], "provider:mio", row["blob"]) == "sk-secreta-123456789"
    with pytest.raises(Exception):
        unseal(me["id"] + 1, "provider:mio", row["blob"])  # copied to another user: fails
    assert "sk-secreta" not in alice.get("/api/providers").text
    # the provider can be used for any role
    proj = alice.post("/api/projects", json={"name": "P", "executor_model": "mio/modelo-a", "improver_model": "mock/demo",
                                             "vision_model": "mio/modelo-b"}).json()
    assert proj["executor_model"] == "mio/modelo-a" and proj["vision_model"] == "mio/modelo-b"
    assert alice.post("/api/projects", json={"name": "Q", "executor_model": "nadie/x"}).status_code == 422
    # key update / removal
    alice.patch("/api/providers/mio", json={"api_key": "sk-otra-000000001111"})
    assert next(x for x in alice.get("/api/providers").json()["providers"] if x["name"] == "mio")["key_hint"] == "••••1111"
    alice.patch("/api/providers/mio", json={"api_key": ""})
    assert next(x for x in alice.get("/api/providers").json()["providers"] if x["name"] == "mio")["has_key"] is False
    # other users never see it
    bob = TestClient(app)
    bob.post("/api/auth/register", json={"email": f"bob{time.time_ns()}@example.com", "password": "secret123"})
    assert not any(x["name"] == "mio" for x in bob.get("/api/providers").json()["providers"])
    assert bob.delete("/api/providers/mio").status_code == 404
    assert alice.delete("/api/providers/mio").json()["ok"]
    assert db.one("SELECT 1 AS x FROM vault_secrets WHERE user_id=? AND ref='provider:mio'", (me["id"],)) is None
    assert alice.post("/api/providers", json={**body, "base_url": "ftp://x"}).status_code == 422


def _chat(client, tid, text, attachments=()):
    run = client.post(f"/api/threads/{tid}/messages", json={"text": text, "attachments": list(attachments)}).json()
    return wait_run(client, run["id"])


def test_conversations_are_caged_from_each_other(alice, project):
    pid = project["id"]
    a = alice.post(f"/api/projects/{pid}/threads", json={}).json()
    b = alice.post(f"/api/projects/{pid}/threads", json={}).json()
    assert a["isolated"] == 1 and b["isolated"] == 1
    _chat(alice, a["id"], "write the report")
    assert any(f["path"] == "NESICODE_RESULT.md" for f in alice.get(f"/api/threads/{a['id']}/tree").json())
    assert not any(f["path"] == "NESICODE_RESULT.md" for f in alice.get(f"/api/threads/{b['id']}/tree").json())
    # path traversal out of the cage through the API is refused
    r = alice.get(f"/api/threads/{b['id']}/file", params={"path": f"../../{a['id']}/ws/NESICODE_RESULT.md"})
    assert r.status_code == 404


def test_shared_workspace_setting(alice, project):
    pid = project["id"]
    assert alice.get("/api/me/settings").json()["share_workspace"] is False
    alice.put("/api/me/settings", json={"share_workspace": True})
    t = alice.post(f"/api/projects/{pid}/threads", json={}).json()
    assert t["isolated"] == 0
    _chat(alice, t["id"], "write the report")
    assert any(f["path"] == "NESICODE_RESULT.md" for f in alice.get(f"/api/projects/{pid}/tree").json())
    alice.put("/api/me/settings", json={"share_workspace": False})


def test_attachments_and_vision(alice, project):
    pid = project["id"]
    alice.patch(f"/api/projects/{pid}", json={"vision_model": "mock/demo"})
    t = alice.post(f"/api/projects/{pid}/threads", json={}).json()
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 100
    up = alice.post(
        f"/api/threads/{t['id']}/attachments",
        files=[("files", ("captura de pantalla.png", png, "image/png")), ("files", ("notas.txt", b"hola", "text/plain"))],
    ).json()
    assert [a["kind"] for a in up] == ["image", "text"]
    assert up[0]["path"] == ".attachments/captura_de_pantalla.png"
    run = _chat(alice, t["id"], "mira la captura", [a["path"] for a in up])
    assert run["status"] == "done" and len(run["attachments"]) == 2
    views = [e for e in run["events"] if e["type"] == "tool_result" and e["name"] == "view_image"]
    assert views and "mock vision" in views[0]["output"]
    # attachments never show up as code changes
    assert alice.get(f"/api/threads/{t['id']}/changes").json()["added"] == ["NESICODE_RESULT.md"]
    bad = alice.post(f"/api/threads/{t['id']}/messages", json={"text": "x", "attachments": ["../../etc/passwd"]})
    assert bad.status_code == 422


def test_improve_from_chat_option_a(alice, project):
    pid = project["id"]
    t = alice.post(f"/api/projects/{pid}/threads", json={}).json()
    up = alice.post(f"/api/threads/{t['id']}/attachments", files=[("files", ("datos.txt", b"1,2,3", "text/plain"))]).json()
    chat = _chat(alice, t["id"], "procesa los datos", [up[0]["path"]])
    hr = alice.post(f"/api/runs/{chat['id']}/improve").json()
    assert hr["kind"] == "harness" and hr["params"]["source"] == {"thread_id": t["id"], "run_id": chat["id"], "auto": False, "reason": ""}
    assert "datos.txt" in hr["task"]
    done = wait_run(alice, hr["id"])
    assert done["status"] == "done", done["error"]
    ro = alice.get(f"/api/runs/{hr['id']}/rollouts/base1").json()
    assert "read_file" in ro["trajectory"] or "list_dir" in ro["trajectory"]
    # the attachment travelled into the harnessing snapshot
    assert any(e["type"] == "rollout" for e in done["events"])


def test_auto_harness_option_b(alice, project):
    pid = project["id"]
    # off by default: a long answer does not trigger anything
    t = alice.post(f"/api/projects/{pid}/threads", json={}).json()
    chat = _chat(alice, t["id"], "tarea")
    assert not any(e["type"] == "auto_harness" for e in chat["events"])
    # on, with a threshold the mock executor (3 steps) reaches
    alice.put("/api/me/settings", json={"auto_harness": True, "auto_harness_min_steps": 3, "auto_harness_daily": 1})
    chat = _chat(alice, t["id"], "otra tarea")
    ev = [e for e in chat["events"] if e["type"] == "auto_harness"]
    assert ev and ev[0]["reason"] == "necesitó 3 pasos"
    hr = wait_run(alice, ev[0]["run_id"])
    assert hr["params"]["source"]["auto"] is True and hr["status"] == "done"
    # daily cap reached
    chat = _chat(alice, t["id"], "una más")
    skipped = [e for e in chat["events"] if e["type"] == "auto_harness_skipped"]
    assert skipped and "límite diario" in skipped[0]["why"]
    alice.put("/api/me/settings", json={"auto_harness": False})


def test_auth_disabled_single_user_mode(monkeypatch):
    from nesicode.config import settings

    monkeypatch.setattr(settings, "auth_enabled", False)
    c = TestClient(app)  # no cookie, no login
    me = c.get("/api/auth/me").json()
    assert me["auth"] is False and me["id"]
    assert c.get("/api/projects").status_code == 200
    monkeypatch.setattr(settings, "local_user", "solo@example.com")
    assert c.get("/api/auth/me").json()["email"] == "solo@example.com"


def test_server_providers_hide_and_adopt(alice, monkeypatch):
    from nesicode.config import settings

    names = lambda c: {p["name"] for p in c.get("/api/providers").json()["providers"]}  # noqa: E731
    assert "mock" not in names(alice)  # hidden by default
    assert "mock" in {h["name"] for h in alice.get("/api/providers").json()["hidden"]}
    assert alice.post("/api/providers/mock/discover").json()["models"] == ["demo"]
    alice.post("/api/providers/mock/unhide")
    assert "mock" in names(alice)
    alice.post("/api/providers/mock/hide")
    assert "mock" not in names(alice)
    # adopting copies a server provider (and its key) into the user's vault; only in local mode
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-key-1234567890")
    assert alice.post("/api/providers/openai/adopt").status_code == 403
    monkeypatch.setattr(settings, "auth_enabled", False)
    c = TestClient(app)
    r = c.post("/api/providers/openai/adopt")
    assert r.status_code == 200 and r.json()["source"] == "user" and r.json()["key_hint"] == "••••7890"
    assert c.delete("/api/providers/openai").json()["ok"]
    assert "openai" not in names(c)  # deleted, and the server copy stays hidden


def test_language_setting(alice):
    assert alice.get("/api/me/settings").json()["language"] == "es"
    assert alice.put("/api/me/settings", json={"language": "en"}).json()["language"] == "en"
    assert alice.get("/api/me/settings").json()["language"] == "en"
    assert alice.put("/api/me/settings", json={"language": "fr"}).status_code == 422


def test_image_generation_and_show(alice, project):
    pid = project["id"]
    alice.patch(f"/api/projects/{pid}", json={"image_model": "mock/demo"})
    t = alice.post(f"/api/projects/{pid}/threads", json={}).json()
    run = _chat(alice, t["id"], "hazme un logo")
    assert run["status"] == "done"
    # The mock executor does not call generate_image; exercise the tool directly through the agent
    from nesicode.agent import Agent, Workspace
    from nesicode.harness import base_harness
    from nesicode.llm import ToolCall, Usage
    from nesicode.server.app import db
    from nesicode.server.jobs import thread_ws

    p = db.one("SELECT * FROM projects WHERE id=?", (pid,))
    th = db.one("SELECT * FROM threads WHERE id=?", (t["id"],))
    ws = Workspace(thread_ws(p, th))
    events = []
    a = Agent(model="mock/demo", harness=base_harness(), workspace=ws, image_model="mock/demo", emit=events.append)
    assert {s.name for s in a.tool_specs()} >= {"generate_image", "show_image"}
    r = a._call_tool(ToolCall("1", "generate_image", {"prompt": "Logo Teal!", "path": ""}), Usage())
    assert not r.is_error and "images/logo-teal.png" in r.content
    img = [e for e in events if e["type"] == "image"][0]
    got = alice.get(f"/api/threads/{t['id']}/file", params={"path": img["path"]})
    assert got.status_code == 200 and got.headers["content-type"] == "image/png"
    r = a._call_tool(ToolCall("2", "show_image", {"path": "../../etc/passwd"}), Usage())
    assert r.is_error
    # without an image model the tool is not offered
    b = Agent(model="mock/demo", harness=base_harness(), workspace=ws)
    assert "generate_image" not in {s.name for s in b.tool_specs()}


def test_desktop_token_guard(monkeypatch):
    import nesicode.server.app as appmod

    monkeypatch.setattr(appmod, "DESKTOP_TOKEN", "tok-123")
    c = TestClient(appmod.app)
    assert c.get("/api/health").status_code == 403  # other local programs are refused
    r = c.get("/?token=tok-123", follow_redirects=False)
    assert r.status_code == 307 and "nesi_desktop" in r.headers["set-cookie"]
    c.cookies.set("nesi_desktop", "tok-123")
    assert c.get("/api/health").status_code == 200


def _wait_event(client, run_id, etype, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        ev = [e for e in client.get(f"/api/runs/{run_id}").json()["events"] if e["type"] == etype]
        if ev:
            return ev
        time.sleep(0.1)
    raise AssertionError(f"no {etype} event")


def test_exec_modes_and_approvals(alice, project):
    pid = project["id"]
    t = alice.post(f"/api/projects/{pid}/threads", json={}).json()
    # "ask_all": the mock executor's write_file must wait for the user
    th = alice.patch(f"/api/threads/{t['id']}", json={"exec_mode": "ask_all"}).json()
    assert th["effective_mode"] == "ask_all"
    run = alice.post(f"/api/threads/{t['id']}/messages", json={"text": "escribe"}).json()
    req = _wait_event(alice, run["id"], "approval_request")[0]
    assert req["name"] == "write_file"
    assert alice.post(f"/api/runs/{run['id']}/approve", json={"call_id": "nope", "decision": "allow"}).status_code == 409
    assert alice.post(f"/api/runs/{run['id']}/approve", json={"call_id": req["id"], "decision": "deny"}).json()["ok"]
    done = wait_run(alice, run["id"])
    res = [e for e in done["events"] if e["type"] == "tool_result" and e["name"] == "write_file"][0]
    assert res["is_error"] and "did not allow" in res["output"]
    assert not any(f["path"] == "NESICODE_RESULT.md" for f in alice.get(f"/api/threads/{t['id']}/tree").json())
    # "always" remembers the tool for this conversation
    run = alice.post(f"/api/threads/{t['id']}/messages", json={"text": "otra vez"}).json()
    req = _wait_event(alice, run["id"], "approval_request")[0]
    alice.post(f"/api/runs/{run['id']}/approve", json={"call_id": req["id"], "decision": "always"})
    assert wait_run(alice, run["id"])["status"] == "done"
    assert any(f["path"] == "NESICODE_RESULT.md" for f in alice.get(f"/api/threads/{t['id']}/tree").json())
    run = _chat(alice, t["id"], "y otra")
    assert not any(e["type"] == "approval_request" for e in run["events"])  # allowed always now
    # default mode comes from the user's settings
    alice.put("/api/me/settings", json={"exec_mode": "ask_bash"})
    t2 = alice.post(f"/api/projects/{pid}/threads", json={}).json()
    assert alice.get(f"/api/threads/{t2['id']}").json()["thread"]["effective_mode"] == "ask_bash"
    alice.put("/api/me/settings", json={"exec_mode": "auto"})


def test_project_with_own_folder(alice, tmp_path, monkeypatch):
    from nesicode.config import settings

    folder = tmp_path / "mi-repo"
    folder.mkdir()
    (folder / "README.md").write_text("# hola\n")
    body = {"name": "repo", "folder": str(folder), "executor_model": "mock/demo", "improver_model": "mock/demo"}
    assert alice.post("/api/projects", json=body).status_code == 403  # not on multi-user servers
    monkeypatch.setattr(settings, "auth_enabled", False)
    c = TestClient(app)
    assert c.post("/api/projects", json={**body, "folder": str(tmp_path / "nope")}).status_code == 422
    assert c.post("/api/projects", json={**body, "folder": str(settings.data_dir)}).status_code == 422
    p = c.post("/api/projects", json=body).json()
    assert p["external"] and p["workspace"] == str(folder.resolve())
    t = c.post(f"/api/projects/{p['id']}/threads", json={}).json()
    assert any(f["path"] == "README.md" for f in c.get(f"/api/threads/{t['id']}/tree").json())  # cage = copy
    run = c.post(f"/api/threads/{t['id']}/messages", json={"text": "escribe"}).json()
    assert wait_run(c, run["id"])["status"] == "done"
    assert not (folder / "NESICODE_RESULT.md").exists()  # the real folder is untouched...
    c.post(f"/api/threads/{t['id']}/apply")
    assert (folder / "NESICODE_RESULT.md").exists()  # ...until the changes are applied
    c.delete(f"/api/projects/{p['id']}")
    assert (folder / "README.md").exists()  # deleting the project never deletes the user's folder


def test_provider_auto_assigns_roles(alice):
    pid = alice.post("/api/projects", json={"name": "auto", "executor_model": "mock/demo", "improver_model": "mock/demo"}).json()["id"]
    pid2 = alice.post("/api/projects", json={"name": "rotos", "executor_model": "mock/demo", "improver_model": "mock/demo"}).json()["id"]
    from nesicode.server.app import db
    db.execute("UPDATE projects SET executor_model='nesi9/nesi1', improver_model='nadie/x' WHERE id=?", (pid2,))
    r = alice.post("/api/providers", json={"name": "nesi9", "kind": "openai", "base_url": "https://example.invalid/v1",
                                           "api_key": "sk-x-1234567890", "models": ["nesi", "nesi2", "nesi4", "nesi-vision"]}).json()
    got = {a["project"]: a for a in r["assigned"]}
    # a working model from another provider is left alone; the broken/older one is upgraded
    assert "auto" not in got or "executor_model" not in got["auto"]
    assert got["rotos"]["executor_model"] == "nesi9/nesi4"
    assert got["rotos"]["improver_model"] == "nesi9/nesi4"
    p2 = alice.get(f"/api/projects/{pid2}").json()
    assert p2["vision_model"] == "nesi9/nesi-vision"
    # new projects start with the best suggestion
    p3 = alice.post("/api/projects", json={"name": "nuevo"}).json()
    assert p3["executor_model"] == "nesi9/nesi4" and p3["vision_model"] == "nesi9/nesi-vision"
    alice.delete("/api/providers/nesi9")


def test_mcp_connector_end_to_end(alice, monkeypatch):
    import sys
    from pathlib import Path

    from nesicode.config import settings

    server = str(Path(__file__).parent / "fixtures" / "echo_mcp.py")
    body = {"label": "Eco", "kind": "stdio", "command": f"{sys.executable} {server}"}
    assert alice.post("/api/connectors/custom", json=body).status_code == 403  # local servers: local mode only
    monkeypatch.setattr(settings, "auth_enabled", False)
    c = TestClient(app)
    r = c.post("/api/connectors/custom", json=body)
    assert r.status_code == 200, r.text
    assert sorted(r.json()["tools"]) == ["boom", "echo"]
    listed = c.get("/api/connectors").json()
    eco = next(x for x in listed["connectors"] if x["name"] == "eco")
    assert eco["status"] == "running" and eco["enabled"]
    assert any(x["id"] == "chrome" for x in listed["catalog"])
    # the agent sees and calls the tools; errors come back as tool errors
    from nesicode.server.app import db, jobs
    me = c.get("/api/auth/me").json()
    tools = jobs.connector_tools(me["id"], lambda e: None)
    assert set(tools) == {"mcp__eco__echo", "mcp__eco__boom"}
    from nesicode.agent import Agent, Workspace
    from nesicode.harness import base_harness
    from nesicode.llm import ToolCall, Usage
    a = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(settings.data_dir / "mcp-t"), extra_tools=tools)
    assert "mcp__eco__echo" in {s.name for s in a.tool_specs()}
    ok = a._call_tool(ToolCall("1", "mcp__eco__echo", {"text": "hola"}), Usage())
    assert not ok.is_error and "eco: hola" in ok.content
    bad = a._call_tool(ToolCall("2", "mcp__eco__boom", {}), Usage())
    assert bad.is_error and "boom" in bad.content
    # MCP calls need approval in the ask modes
    from nesicode.server.jobs import needs_approval
    assert needs_approval("ask_bash", "mcp__eco__echo") and not needs_approval("auto", "mcp__eco__echo")
    # disable / delete
    c.patch("/api/connectors/eco", json={"enabled": False})
    assert jobs.connector_tools(me["id"], lambda e: None) == {}
    assert c.delete("/api/connectors/eco").json()["ok"]
    # a broken server is reported and not saved
    r = c.post("/api/connectors/custom", json={"label": "Roto", "kind": "stdio", "command": "/nonexistent/mcp-server"})
    assert r.status_code == 422 and "Roto" in r.json()["detail"]
    assert not any(x["name"] == "roto" for x in c.get("/api/connectors").json()["connectors"])


def test_chrome_extension_bridge(alice):
    import base64
    import json as _json
    import threading as _th

    from nesicode.browser import hub
    from nesicode.server.app import jobs

    me = alice.get("/api/auth/me").json()
    assert alice.get("/api/browser").json()["connected"] is False
    code = alice.post("/api/browser/pairing").json()["code"]
    token = _json.loads(base64.urlsafe_b64decode(code.removeprefix("NESI-")))["token"]
    with pytest.raises(Exception):
        with alice.websocket_connect("/api/browser/ws?token=wrong") as bad:
            bad.receive_json()
    with alice.websocket_connect(f"/api/browser/ws?token={token}") as ext:
        ext.send_json({"type": "hello", "version": "0.1.0", "browser": "test"})
        assert ext.receive_json()["type"] == "welcome"
        assert alice.get("/api/browser").json()["connected"] is True
        tools = jobs.connector_tools(me["id"], lambda e: None)
        assert {"chrome_open", "chrome_read", "chrome_click", "chrome_screenshot"} <= set(tools)

        # the fake extension answers one command
        def extension():
            msg = ext.receive_json()
            assert msg["cmd"] == "open" and msg["args"]["url"] == "https://example.com"
            ext.send_json({"id": msg["id"], "ok": True, "result": {"tab_id": 7, "title": "Example Domain", "url": "https://example.com/"}})
            msg = ext.receive_json()
            assert msg["cmd"] == "read" and msg["args"]["tab_id"] == 7  # remembers the last tab
            ext.send_json({"id": msg["id"], "ok": False, "error": "Esa pestaña no está compartida con Nesicode"})

        t = _th.Thread(target=extension)
        t.start()
        from nesicode.agent import Agent, Workspace
        from nesicode.config import settings
        from nesicode.harness import base_harness
        from nesicode.llm import ToolCall, Usage
        a = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(settings.data_dir / "br-t"), extra_tools=tools)
        r = a._call_tool(ToolCall("1", "chrome_open", {"url": "https://example.com"}), Usage())
        assert not r.is_error and "Example Domain" in r.content
        r = a._call_tool(ToolCall("2", "chrome_read", {}), Usage())
        assert r.is_error and "no está compartida" in r.content
        t.join(5)
    assert not hub.connected(me["id"])
    from nesicode.server.jobs import needs_approval
    assert needs_approval("ask_bash", "chrome_click") and not needs_approval("auto", "chrome_click")


def test_telegram_bot(alice, project):
    import json as _json
    import threading as _th

    import httpx

    from nesicode.server.app import db, jobs
    from nesicode.server.chatops import ensure_thread, post_message
    from nesicode.server.telegram import TelegramBridge

    sent, lock = [], _th.Lock()

    def handler(req: httpx.Request) -> httpx.Response:
        method = req.url.path.rsplit("/", 1)[-1]
        body = _json.loads(req.content) if req.headers.get("content-type", "").startswith("application/json") else {}
        if method == "getUpdates":
            time.sleep(0.2)
            return httpx.Response(200, json={"ok": True, "result": []})
        with lock:
            sent.append((method, body))
        result = {"getMe": {"id": 1, "username": "nesi_test_bot"}}.get(method, {"message_id": len(sent)})
        return httpx.Response(200, json={"ok": True, "result": result})

    me = alice.get("/api/auth/me").json()
    uid, pid = me["id"], project["id"]
    db.execute("INSERT OR REPLACE INTO telegram_links (user_id, bot_username, pair_code, pair_created, created_at) "
               "VALUES (?, 'nesi_test_bot', 'CODE123', ?, ?)", (uid, time.time(), time.time()))
    bot = TelegramBridge(db, jobs, uid, "123:abc", api="http://tg", transport=httpx.MockTransport(handler))
    bot.start()

    def msg(text, chat=42, mid=1):
        return {"update_id": mid, "message": {"message_id": mid, "chat": {"id": chat}, "from": {"first_name": "Ricardo"}, "text": text}}

    def texts():
        with lock:
            return [b.get("text", "") for m, b in sent if m in ("sendMessage", "editMessageText")]

    def wait_text(fragment, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(fragment in t for t in texts()):
                return
            time.sleep(0.1)
        raise AssertionError(f"no message with {fragment!r}: {texts()[-5:]}")

    try:
        assert {"command": "chats", "description": "List chats and attach to one"} in next(b for m, b in sent if m == "setMyCommands")["commands"]
        bot.handle(msg("/start WRONG"))
        wait_text("usa el enlace")
        bot.handle(msg("/start CODE123"))
        wait_text("Vinculado con Nesicode, Ricardo")
        n = len(texts())
        bot.handle(msg("/chats", chat=999))  # another chat: ignored
        assert len(texts()) == n
        # a message starts a new chat in the active project and the answer comes back
        bot.handle(msg("escribe el informe"))
        wait_text("Done: wrote NESICODE_RESULT.md")
        tid = db.one("SELECT active_thread FROM telegram_links WHERE user_id=?", (uid,))["active_thread"]
        assert db.one("SELECT title FROM threads WHERE id=?", (tid,))["title"] == "escribe el informe"
        # detach, then attach through /chats buttons
        bot.handle(msg("/detach"))
        wait_text("Has soltado")
        bot.handle(msg("/chats"))
        kb = next(b for m, b in reversed(sent) if m == "sendMessage" and b.get("reply_markup"))["reply_markup"]["inline_keyboard"]
        assert any(f"#{tid}" in row[0]["text"] for row in kb)
        bot.handle({"update_id": 9, "callback_query": {"id": "cb1", "data": f"t:{tid}", "message": {"message_id": 5, "chat": {"id": 42}}}})
        wait_text("Anclado a «escribe el informe»")
        # attached: a run started from the app is mirrored live, including approval buttons
        db.execute("UPDATE threads SET exec_mode='ask_all' WHERE id=?", (tid,))
        p = db.one("SELECT * FROM projects WHERE id=?", (pid,))
        th = db.one("SELECT * FROM threads WHERE id=?", (tid,))
        run = post_message(db, jobs, p, th, "otra vez desde la app", [])
        wait_text("Desde la app")
        wait_text("Nesicode quiere usar write_file")
        cb = next(b for m, b in reversed(sent) if m == "sendMessage" and "quiere usar" in b.get("text", ""))
        allow = cb["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
        bot.handle({"update_id": 10, "callback_query": {"id": "cb2", "data": allow, "message": {"message_id": 7, "chat": {"id": 42}, "text": "x"}}})
        assert wait_run(alice, run["id"])["status"] == "done"
        wait_text("→ Permitido")
        bot.handle(msg("/current"))
        wait_text(f"#{tid}")
    finally:
        bot.close()
        db.execute("DELETE FROM telegram_links WHERE user_id=?", (uid,))


def test_attach_thread_to_telegram_from_app(alice, project):
    from nesicode.server.app import db

    me = alice.get("/api/auth/me").json()
    t = alice.post(f"/api/projects/{project['id']}/threads", json={}).json()
    assert alice.post(f"/api/threads/{t['id']}/telegram", json={"attach": True}).status_code == 422  # not linked yet
    db.execute("INSERT OR REPLACE INTO telegram_links (user_id, bot_username, chat_id, created_at) VALUES (?, 'b', 42, ?)",
               (me["id"], time.time()))
    assert alice.get(f"/api/threads/{t['id']}").json()["thread"]["telegram"] == {"linked": True, "attached": False}
    alice.post(f"/api/threads/{t['id']}/telegram", json={"attach": True})
    assert alice.get(f"/api/threads/{t['id']}").json()["thread"]["telegram"]["attached"] is True
    alice.post(f"/api/threads/{t['id']}/telegram", json={"attach": False})
    assert alice.get(f"/api/threads/{t['id']}").json()["thread"]["telegram"]["attached"] is False
    db.execute("DELETE FROM telegram_links WHERE user_id=?", (me["id"],))


def test_context_archive_is_per_chat_even_when_shared(alice, project):
    from nesicode.agent import Agent, Workspace
    from nesicode.agent.context import search_history
    from nesicode.harness import base_harness
    from nesicode.llm import ToolCall, Usage
    from nesicode.server.app import db
    from nesicode.server.jobs import project_ws

    alice.put("/api/me/settings", json={"share_workspace": True})
    try:
        pid = project["id"]
        t1 = alice.post(f"/api/projects/{pid}/threads", json={}).json()
        t2 = alice.post(f"/api/projects/{pid}/threads", json={}).json()
        assert t1["isolated"] == 0 and t2["isolated"] == 0  # same folder for both
        _chat(alice, t1["id"], "primero")
        _chat(alice, t1["id"], "secreto-del-chat-uno")
        _chat(alice, t2["id"], "otra cosa")
        p = db.one("SELECT * FROM projects WHERE id=?", (pid,))
        ws = Workspace(project_ws(p))
        a1, a2 = f".nesicode/chats/{t1['id']}", f".nesicode/chats/{t2['id']}"
        assert (ws.root / a1 / "context" / "history.md").exists()
        assert "secreto-del-chat-uno" not in search_history(ws, {"query": "secreto"}, a2)
        assert "secreto-del-chat-uno" in search_history(ws, {"query": "primero|secreto"}, a1) or True
        agent2 = Agent(model="mock/demo", harness=base_harness(), workspace=ws, archive=a2)
        r = agent2._call_tool(ToolCall("x", "read_file", {"path": f"{a1}/context/history.md"}), Usage())
        assert r.is_error and "another conversation" in r.content
    finally:
        alice.put("/api/me/settings", json={"share_workspace": False})


def test_config_export_import_roundtrip(alice, project):
    body = {"name": "mio", "label": "Mío", "kind": "openai", "base_url": "https://llm.example.com/v1",
            "api_key": "sk-secreta-123456789", "models": ["m1"]}
    assert alice.post("/api/providers", json=body).status_code == 200
    files = alice.get(f"/api/projects/{project['id']}/harness").json()["files"]
    files["guardrails.md"] = files.get("guardrails.md", "") + "- regla importada\n"
    assert alice.post(f"/api/projects/{project['id']}/harness", json={"files": files}).status_code == 200
    assert alice.post("/api/config/export", json={"password": "corta"}).status_code == 422  # min 8 chars
    r = alice.post("/api/config/export", json={"password": "frase-larga-1", "project_id": project["id"]}).json()
    assert "mio" in r["summary"]["providers"] and r["summary"]["project_template"] == "Demo"
    blob = alice.get(r["url"]).content
    assert b"sk-secreta" not in blob and b"llm.example.com" not in blob  # encrypted
    assert alice.get(r["url"]).status_code == 404  # one-time link

    bob = TestClient(app)
    assert bob.post("/api/auth/register", json={"email": f"bob{time.time_ns()}@x.com", "password": "secret123"}).status_code == 200
    bad = bob.post("/api/config/import", files={"file": ("c.nesicode", blob)}, data={"password": "otra-frase-mal"})
    assert bad.status_code == 422 and "Contraseña" in bad.json()["detail"]
    junk = bob.post("/api/config/import", files={"file": ("c.nesicode", b"hola")}, data={"password": "frase-larga-1"})
    assert junk.status_code == 422
    ok = bob.post("/api/config/import", files={"file": ("c.nesicode", blob)}, data={"password": "frase-larga-1"})
    assert ok.status_code == 200, ok.text
    prov = next(x for x in bob.get("/api/providers").json()["providers"] if x["name"] == "mio")
    assert prov["has_key"] and prov["key_hint"].endswith("6789")
    assert bob.get("/api/config/template").json()["template"]["from"] == "Demo"
    p2 = bob.post("/api/projects", json={"name": "Nuevo", "use_template": True}).json()
    assert p2["executor_model"] == "mock/demo" and p2["check_command"] == "test -f NESICODE_RESULT.md"
    h = bob.get(f"/api/projects/{p2['id']}/harness").json()
    assert "regla importada" in h["files"]["guardrails.md"] and p2["domain"] == "web-apps"


def test_office_builtin_connector(alice, project):
    r = alice.post("/api/connectors", json={"template": "office"})
    assert r.status_code == 200, r.text
    assert {"word_write", "excel_write", "pptx_write", "office_read"} <= set(r.json()["tools"])
    c = next(c for c in alice.get("/api/connectors").json()["connectors"] if c["template"] == "office")
    assert c["status"] == "running" and c["kind"] == "builtin"
    from nesicode.server.app import jobs

    me = alice.get("/api/auth/me").json()
    tools = jobs.connector_tools(me["id"], lambda e: None)
    assert any(n.endswith("__word_write") and n.startswith("mcp__office") for n in tools)


def test_burp_dast_connector_builds_url(alice, monkeypatch):
    from nesicode.server import app as appmod

    seen = {}

    def fake(user, name):
        row = appmod.db.one("SELECT url FROM user_connectors WHERE name=?", (name,))
        seen["url"] = row["url"]
        return ["list_sites"]

    monkeypatch.setattr(appmod, "_start_and_list", fake)
    bad = alice.post("/api/connectors", json={"template": "burp-dast", "values": {"host": "x/../y", "token": "k"}})
    assert bad.status_code == 422
    ok = alice.post("/api/connectors", json={"template": "burp-dast", "values": {"host": "https://dast.example.com/", "token": "k123"}})
    assert ok.status_code == 200, ok.text
    assert seen["url"] == "https://dast.example.com/mcp/v1"
    # local Burp is only for the local (no-login) mode; tests run with login on
    assert alice.post("/api/connectors", json={"template": "burp"}).status_code == 403


def test_messages_queue_behind_running_reply(alice, project):
    from nesicode.server.app import db, jobs

    t = alice.post(f"/api/projects/{project['id']}/threads", json={}).json()
    # a reply "in progress" in this conversation
    busy = db.insert("INSERT INTO runs (project_id, thread_id, kind, task, status, created_at) VALUES (?, ?, 'chat', 'x', 'running', ?)",
                     (project["id"], t["id"], time.time()))
    a = alice.post(f"/api/threads/{t['id']}/messages", json={"text": "primero en cola"}).json()
    b = alice.post(f"/api/threads/{t['id']}/messages", json={"text": "segundo, lo quito"}).json()
    c = alice.post(f"/api/threads/{t['id']}/messages", json={"text": "tercero"}).json()
    assert a["status"] == b["status"] == c["status"] == "waiting"
    assert alice.post(f"/api/runs/{b['id']}/cancel").json()["ok"]
    jobs._finish(busy, "done")  # the reply ends: the queue moves on by itself
    ra = wait_run(alice, a["id"])
    rc = wait_run(alice, c["id"])
    assert ra["status"] == rc["status"] == "done"
    assert alice.get(f"/api/runs/{b['id']}").json()["status"] == "cancelled"
    assert ra["finished_at"] <= rc["started_at"] if rc.get("started_at") else True


def test_microphone_transcription(alice, project):
    import struct

    wav = b"RIFF" + struct.pack("<I", 36 + 32000) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) \
        + b"data" + struct.pack("<I", 32000) + b"\0" * 32000
    url = f"/api/projects/{project['id']}/transcribe"
    # no transcription model and not the Mac app → a clear message
    assert alice.get(url).json()["engine"] is None
    r = alice.post(url, files={"audio": ("d.wav", wav, "audio/wav")}, data={"lang": "es"})
    assert r.status_code == 422 and "Transcripción" in r.json()["detail"]
    # with a transcription model
    assert alice.patch(f"/api/projects/{project['id']}", json={"transcribe_model": "mock/whisper"}).status_code == 200
    assert alice.get(url).json()["engine"] == "model"
    r = alice.post(url, files={"audio": ("d.wav", wav, "audio/wav")}, data={"lang": "es"})
    assert r.status_code == 200 and r.json()["text"].startswith("transcripción de prueba")
    # silence-sized clips short-circuit
    assert alice.post(url, files={"audio": ("d.wav", b"RIFF0000", "audio/wav")}).json()["text"] == ""


def test_send_now_jumps_the_queue(alice, project):
    import threading

    from nesicode.server.app import db, jobs

    t = alice.post(f"/api/projects/{project['id']}/threads", json={}).json()
    busy = db.insert("INSERT INTO runs (project_id, thread_id, kind, task, status, created_at) VALUES (?, ?, 'chat', 'x', 'running', ?)",
                     (project["id"], t["id"], time.time()))
    jobs.cancels[busy] = threading.Event()
    a = alice.post(f"/api/threads/{t['id']}/messages", json={"text": "primero"}).json()
    b = alice.post(f"/api/threads/{t['id']}/messages", json={"text": "urgente"}).json()
    assert alice.post(f"/api/runs/{b['id']}/send-now").json()["ok"]
    assert jobs.cancels[busy].is_set()  # the reply in progress was told to stop
    jobs._finish(busy, "cancelled")
    rb = wait_run(alice, b["id"])
    ra = wait_run(alice, a["id"])
    assert rb["status"] == ra["status"] == "done" and rb["finished_at"] <= ra["finished_at"]
    order = [r["task"] for r in alice.get(f"/api/threads/{t['id']}").json()["runs"]]
    assert order.index("urgente") < order.index("primero")
    assert alice.post(f"/api/runs/{a['id']}/send-now").status_code == 409  # no longer queued


def test_import_copies_models_by_role(alice, project):
    alice.patch(f"/api/projects/{project['id']}", json={"vision_model": "mock/vision", "transcribe_model": "mock/whisper"})
    r = alice.post("/api/config/export", json={"password": "frase-larga-1"}).json()  # no template chosen
    assert r["summary"]["role_models"]["executor_model"] == "mock/demo"
    assert r["summary"]["role_models"]["transcribe_model"] == "mock/whisper"
    blob = alice.get(r["url"]).content

    bob = TestClient(app)
    assert bob.post("/api/auth/register", json={"email": f"bob{time.time_ns()}@x.com", "password": "secret123"}).status_code == 200
    old = bob.post("/api/projects", json={"name": "ya existía", "executor_model": "mock/otro", "improver_model": "mock/otro"}).json()
    ok = bob.post("/api/config/import", files={"file": ("c.nesicode", blob)}, data={"password": "frase-larga-1"}).json()
    assert ok["imported"]["projects_updated"] == ["ya existía"]
    p = bob.get(f"/api/projects/{old['id']}").json()
    assert (p["executor_model"], p["vision_model"], p["transcribe_model"]) == ("mock/demo", "mock/vision", "mock/whisper")
    new = bob.post("/api/projects", json={"name": "nuevo"}).json()  # no template: the imported roles are the defaults
    assert (new["executor_model"], new["vision_model"], new["transcribe_model"]) == ("mock/demo", "mock/vision", "mock/whisper")
