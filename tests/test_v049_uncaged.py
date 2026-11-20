"""«Sin jaula» (0.4.9): a project setting that runs its chat commands as the user, outside any sandbox.

A user asked their agent to stand up a CTF with Docker. Inside the macOS sandbox, with its private HOME, the agent could
not see the user's Docker context nor reach its socket, and it spent three replies and 10 GB trying to build a Lima VM of
its own; another agent with no sandbox ran `docker compose up` at once. Now a project can run its chats' commands as the
user (their HOME and login environment, without Nesicode's own variables), only where Nesicode serves one person, never
for bots, never from a configuration file. A caged agent that hits Docker is told at once that it is out of reach.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nesicode.agent import Workspace, loop, tools
from nesicode.agent import workspace as wsmod
from nesicode.config import settings
from nesicode.server.app import app, db
from nesicode.server.config_io import seal, unseal

posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX shell syntax")


@pytest.fixture()
def user_env(tmp_path, monkeypatch):
    """The user's login environment as the server would read it (never the real one in a test)."""
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    env = {"HOME": str(real_home), "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "USERVAR": "mine",
           "DOCKER_HOST": "unix:///tmp/test-docker.sock", "NESICODE_DESKTOP_TOKEN": "tok", "NESICODE_DATA_DIR": "/x",
           "SECRET_FROM_DOTENV": "k"}
    monkeypatch.setattr(wsmod, "_USER_ENV", None)
    monkeypatch.setattr(wsmod, "_login_environment", lambda: dict(env))
    monkeypatch.setattr(wsmod, "DOTENV_KEYS", {"SECRET_FROM_DOTENV"})
    return real_home


# ---------------------------------------------------------------------------------------------- the workspace
@posix_only
def test_uncaged_commands_run_as_the_user_with_their_environment(tmp_path, user_env):
    ws = Workspace(tmp_path / "ws", uncaged=True)
    assert ws.mode() == "host"
    code, out = ws.run('echo "$HOME|${DOCKER_HOST:-}|${NESICODE_DESKTOP_TOKEN:-gone}|${NESICODE_DATA_DIR:-gone}|'
                       '${SECRET_FROM_DOTENV:-gone}|${USERVAR:-}|${TERM:-}"; touch "$HOME/written-outside"')
    assert code == 0, out
    assert out.splitlines()[0] == f"{user_env}|unix:///tmp/test-docker.sock|gone|gone|gone|mine|dumb"
    assert (user_env / "written-outside").exists()  # outside the workspace: no sandbox (on macOS it would be refused)

    caged = Workspace(tmp_path / "ws2")
    assert caged.mode() == wsmod.sandbox_mode() and caged.mode() != "host"
    code, out = caged.run('echo "$HOME|${USERVAR:-none}"')
    assert out.splitlines()[0] == f"{caged.home}|none"  # the sandbox's private HOME and minimal environment, as before


def test_the_users_environment_is_read_once_and_falls_back_to_this_process(monkeypatch):
    calls = []
    monkeypatch.setattr(wsmod, "_USER_ENV", None)
    monkeypatch.setattr(wsmod, "_login_environment", lambda: calls.append(1) or None)
    first = wsmod.user_environment()
    assert first.get("PATH") == os.environ.get("PATH") and wsmod.user_environment() == first and calls == [1]


@pytest.mark.skipif(os.name == "nt", reason="no login shell on Windows")
def test_the_login_shell_output_is_read_past_whatever_its_startup_files_print(monkeypatch):
    def fake(stdout):
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout, stderr=b"")

    good = b"Welcome!\nnvm: using v22\n" + wsmod._ENV_MARK + b"PATH=/usr/bin:/bin\0HOME=/Users/me\0WITH_EQ=a=b\0\0"
    monkeypatch.setattr(wsmod.subprocess, "run", fake(good))
    assert wsmod._login_environment() == {"PATH": "/usr/bin:/bin", "HOME": "/Users/me", "WITH_EQ": "a=b"}
    monkeypatch.setattr(wsmod.subprocess, "run", fake(b"zsh: no such builtin"))
    assert wsmod._login_environment() is None  # (then this process's own environment)
    monkeypatch.setattr(wsmod.subprocess, "run", fake(wsmod._ENV_MARK + b"HOME=/Users/me\0"))
    assert wsmod._login_environment() is None  # no PATH: not a usable environment


def test_hints_and_prompt_follow_the_workspace(tmp_path, user_env, monkeypatch):
    ws = Workspace(tmp_path / "ws", uncaged=True)
    assert ws.outside_hint() == wsmod.HOST_HINT
    with pytest.raises(wsmod.ToolError) as e:  # the file tools still stay inside the workspace
        ws.resolve("../elsewhere.txt")
    assert "with bash you reach the user's other folders" in str(e.value)
    monkeypatch.setattr(tools, "sandbox_mode", lambda: "macos")  # (the app's own setting: the sandbox is on)
    assert not tools._refused_by_sandbox("ls ~/Downloads", "Operation not permitted", mode="host")
    assert tools._refused_by_sandbox("ls ~/Downloads", "Operation not permitted")
    from nesicode.harness import base_harness

    monkeypatch.setattr(loop, "sandbox_mode", lambda: "macos")
    uncaged = loop.build_system(base_harness(), ws)
    caged = loop.build_system(base_harness(), Workspace(tmp_path / "ws2"))
    assert loop.UNCAGED in uncaged and loop.OUT_OF_REACH not in uncaged
    assert loop.OUT_OF_REACH in caged and loop.UNCAGED not in caged and "Docker are out of reach" in caged


@pytest.mark.parametrize("command, output", [
    ("docker version", "failed to connect to the docker API at unix:///var/run/docker.sock; check if the path is correct"),
    ("docker compose up -d", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?"),
    ("colima start --cpu 2", 'level=fatal msg="instance name \\"colima\\" too long: must be less than UNIX_PATH_MAX=104'),
    ("colima status", 'level=fatal msg="colima is not running"'),
])
def test_a_caged_agent_is_told_at_once_that_docker_is_out_of_reach(tmp_path, monkeypatch, command, output):
    class Ws:
        root, home, read_dirs = tmp_path, tmp_path / "h", ()

        def __init__(self, uncaged=False):
            self.uncaged = uncaged

        def run(self, command, timeout):
            return 1, output

        def seen_read_dirs(self):
            return []

    monkeypatch.setattr(tools, "sandbox_mode", lambda: "macos")
    said = tools.bash(Ws(), {"command": command})
    assert said.endswith(tools.DOCKER_NOTE) and "«Sin jaula»" in said and "do not try" in said
    assert tools.DOCKER_NOTE not in tools.bash(Ws(uncaged=True), {"command": command})  # (no sandbox: a real failure)
    assert tools.DOCKER_NOTE not in tools.bash(Ws(), {"command": "cat notes.md"})
    monkeypatch.setattr(tools, "sandbox_mode", lambda: "none")
    assert tools.DOCKER_NOTE not in tools.bash(Ws(), {"command": command})


# ---------------------------------------------------------------------------------------------- the setting
def _local(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "local_user", f"nc{time.time_ns()}@example.com")
    return TestClient(app)


def _project(c, **kw) -> dict:
    r = c.post("/api/projects", json={"name": "P", "executor_model": "mock/demo", "improver_model": "mock/demo", **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_the_setting_is_a_projects_and_local_only(monkeypatch):
    c = _local(monkeypatch)
    p = _project(c)
    assert p["uncaged"] is False and p["uncaged_allowed"] is True
    url = f"/api/projects/{p['id']}"
    assert c.patch(url, json={"uncaged": True}).json()["uncaged"] is True
    assert db.one("SELECT uncaged FROM projects WHERE id=?", (p["id"],))["uncaged"] == 1
    assert c.patch(url, json={"name": "renamed"}).json()["uncaged"] is True  # (other changes leave it as it is)
    assert c.patch(url, json={"uncaged": False}).json()["uncaged"] is False
    c.patch(url, json={"uncaged": True})

    monkeypatch.setattr(settings, "auth_enabled", True)  # the same data, served to several people: never
    from nesicode.server.app import project_view

    row = db.one("SELECT * FROM projects WHERE id=?", (p["id"],))
    assert project_view(row)["uncaged"] is False and project_view(row)["uncaged_allowed"] is False
    s = TestClient(app)
    assert s.post("/api/auth/register", json={"email": f"nc{time.time_ns()}@example.com", "password": "secret123"}).status_code == 200
    q = _project(s)
    r = s.patch(f"/api/projects/{q['id']}", json={"uncaged": True})
    assert r.status_code == 403 and "modo local" in r.json()["detail"]


def test_a_configuration_file_never_carries_it_nor_sets_it(monkeypatch):
    c = _local(monkeypatch)
    p = _project(c)
    c.patch(f"/api/projects/{p['id']}", json={"uncaged": True})
    r = c.post("/api/config/export", json={"password": "frase-larga-1", "project_id": p["id"]})
    assert r.status_code == 200, r.text
    doc = unseal(c.get(r.json()["url"]).content, "frase-larga-1")
    assert doc["project_template"]["from"] == "P" and "uncaged" not in json.dumps(doc)
    doc["project_template"]["uncaged"] = True
    r = c.post("/api/config/import", files={"file": ("c.nesicode", seal(doc, "frase-larga-1"))}, data={"password": "frase-larga-1"})
    assert r.status_code == 200, r.text
    assert _project(c, use_template=True, name="Q")["uncaged"] is False


def _wait(c, run_id: int, timeout: float = 90) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = c.get(f"/api/runs/{run_id}").json()
        if r["status"] in ("done", "error", "cancelled", "interrupted", "max_steps", "refusal"):
            return r
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def test_chats_run_without_the_sandbox_and_bots_never(monkeypatch, user_env):
    from nesicode.llm.mock import MockProvider
    from nesicode.server.app import bots

    seen: list[str] = []

    class Recorder(MockProvider):
        def conversation(self, model, system, tools_, options):
            seen.append(system)
            return super().conversation(model, system, tools_, options)

    monkeypatch.setattr(loop, "get_provider", lambda spec, keys=None: (Recorder(), "demo"))
    c = _local(monkeypatch)
    p = _project(c)
    c.patch(f"/api/projects/{p['id']}", json={"uncaged": True})
    t = c.post(f"/api/projects/{p['id']}/threads", json={}).json()
    run = c.post(f"/api/threads/{t['id']}/messages", json={"text": "levanta el docker", "attachments": []}).json()
    assert _wait(c, run["id"])["status"] == "done"
    assert seen and all(loop.UNCAGED in s for s in seen)

    seen.clear()
    b = c.post("/api/bots", json={"project_id": p["id"], "name": "Resumen", "instructions": "Resume lo nuevo.",
                                  "policy": "auto", "triggers": [{"kind": "cron", "spec": "0 8 * * 1-5", "tz": "Europe/Madrid"}]})
    assert b.status_code == 200, b.text
    fired = bots.fire(b.json()["id"], "cron#test")
    assert fired and _wait(c, fired["id"])["status"] == "done"
    assert seen and not any(loop.UNCAGED in s for s in seen)  # a bot answers other people: its commands stay caged
