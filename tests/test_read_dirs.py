"""Folders the agent may READ (never change), per project — opt-in, off by default.

Seen live: the user wanted the agent to use an APK their browser had saved in ~/Downloads; the sandbox kept it out (as
it must keep out the rest of their home), so the only way in was attaching it (25 MB max) or copying it into the project.
Now a project names a few folders the agent may read: checked by the server (never the disk root, the home or a folder
that holds it, another user's home, Nesicode's own folders, a place with keys), allowed by the macOS sandbox after its
walls and before the secrets (which stay out even inside an allowed folder), mounted read-only in docker mode, and
named to the agent, who copies what it needs into its workspace with bash.

A fake home stands in for the user's (HOME, which the sandbox and the checks read): the real one is never looked at. The
sandbox runs are real on macOS (skipped without sandbox-exec)."""

import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nesicode.agent import Workspace, loop, tools
from nesicode.agent import workspace as wsmod
from nesicode.agent.tools import _refused_by_sandbox, _refused_in_read_dirs
from nesicode.agent.workspace import ToolError
from nesicode.config import settings
from nesicode.harness import base_harness
from nesicode.server import read_dirs as rd
from nesicode.server.app import app, db
from nesicode.server.config_io import seal

MACOS = sys.platform == "darwin" and shutil.which("sandbox-exec") is not None
macos_only = pytest.mark.skipif(not MACOS, reason="needs the macOS sandbox (sandbox-exec)")
posix_only = pytest.mark.skipif(os.name == "nt", reason="symbolic links need admin rights on Windows")
FINAL = {"done", "error", "cancelled", "interrupted", "max_steps", "refusal"}


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A user's home at <tmp>/Users/me, with another user's next to it and the shared folder; returned canonical. The
    server runs in local mode (as the desktop app does): the setting exists only there."""
    monkeypatch.setattr(settings, "auth_enabled", False)
    users = tmp_path / "Users"
    me = users / "me"
    for d in ("Downloads", "Documents/x", ".ssh", ".config/gh", ".config/myapp", "Library/Keychains",
              "Library/Mobile Documents/com~apple~CloudDocs"):
        (me / d).mkdir(parents=True)
    (users / "other" / "Downloads").mkdir(parents=True)
    (users / "Shared" / "fotos").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(me))
    monkeypatch.setenv("USERPROFILE", str(me))  # (Windows reads the home from there)
    return rd.canonical(me)


def _c(p) -> str:
    return str(rd.canonical(p))


def _refused(raw, why: str) -> None:
    with pytest.raises(rd.ReadDirError) as e:
        rd.check_read_dir(str(raw))
    assert why in str(e.value), (raw, str(e.value))


# ---------------------------------------------------------------------------------------------------- what may be read
def test_folders_the_agent_may_be_let_read(home, tmp_path):
    usb = tmp_path / "Volumes" / "USB" / "apks"  # (an external disk's folder)
    usb.mkdir(parents=True)
    for p in (home / "Downloads", home / "Documents" / "x", home / ".config" / "myapp", usb,
              home / "Library" / "Mobile Documents" / "com~apple~CloudDocs", tmp_path / "Users" / "Shared" / "fotos"):
        assert rd.check_read_dir(str(p)) == _c(p)
    assert rd.check_read_dir("~/Downloads") == _c(home / "Downloads")  # what «Añadir Descargas» sends
    assert rd.check_read_dir(f"  {home / 'Downloads'}  ") == _c(home / "Downloads")


def test_folders_the_agent_is_never_let_read(home, tmp_path, monkeypatch):
    _refused(Path(str(tmp_path)).anchor, "raíz del disco")
    for p in (home, home.parent, tmp_path):  # the home, and every folder that holds it
        _refused(p, "no tu carpeta personal")
    _refused(tmp_path / "Users" / "other" / "Downloads", "de otro usuario")
    _refused(tmp_path / "Users" / "other", "de otro usuario")
    for p in (home / ".ssh", home / ".config" / "gh", home / "Library" / "Keychains"):
        _refused(p, "claves o credenciales")
    for p in (home / ".config", home / "Library"):
        _refused(p, "datos privados de tus apps")
    # Nesicode's own: its data (every project and conversation), a source checkout (.env) — and what holds them
    srv = tmp_path / "srv"
    (srv / "data" / "workspaces").mkdir(parents=True)
    monkeypatch.setattr(settings, "data_dir", srv / "data")
    for p in (srv / "data", srv / "data" / "workspaces", srv):
        _refused(p, "es de Nesicode")
    if (wsmod.REPO_DIR / "pyproject.toml").is_file():
        for p in (wsmod.REPO_DIR, wsmod.REPO_DIR / "src", wsmod.REPO_DIR.parent):
            _refused(p, "es de Nesicode")
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    _refused(tmp_path / "nope", "no existe")
    _refused(tmp_path / "a.txt", "no existe")  # a file is no folder
    _refused("relativa/carpeta", "ruta completa")
    _refused("", "no válida")
    _refused("/x\n/y", "no válida")
    if os.name != "nt":  # (and NESICODE_SANDBOX_DENY, a list of POSIX paths separated by ':')
        _refused("/System/Library" if sys.platform == "darwin" else "/proc/self", "del sistema")
        monkeypatch.setenv("NESICODE_SANDBOX_DENY", str(home / "Documents"))
        _refused(home / "Documents" / "x", "vetada")


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX permissions (root reads everything)")
def test_a_folder_the_server_cannot_read_is_refused_with_the_reason(home):
    """What the server cannot read, no caged command can (on a Mac, also a folder its privacy settings keep Nesicode
    out of: adding it is when the system asks the user)."""
    locked = home / "Documents" / "cerrada"
    locked.mkdir()
    locked.chmod(0)
    try:
        _refused(locked, "no tiene permiso para leer esa carpeta")
    finally:
        locked.chmod(0o755)
    assert rd.check_read_dir(str(locked)) == _c(locked)


@posix_only
def test_links_are_judged_where_they_lead(home, tmp_path):
    (tmp_path / "todo").symlink_to(home)
    _refused(tmp_path / "todo", "no tu carpeta personal")
    (home / "Downloads" / "llaves").symlink_to(home / ".ssh")
    _refused(home / "Downloads" / "llaves", "claves o credenciales")
    (tmp_path / "dl").symlink_to(home / "Downloads")
    assert rd.check_read_dir(str(tmp_path / "dl")) == _c(home / "Downloads")  # stored where it really is
    # ~/.ssh kept in a dotfiles folder: that folder is a secret place too
    dot = tmp_path / "dotfiles" / "ssh"
    dot.mkdir(parents=True)
    shutil.rmtree(home / ".ssh")
    (home / ".ssh").symlink_to(dot)
    _refused(dot, "claves o credenciales")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS spellings")
def test_on_macos_every_spelling_of_a_folder_is_judged_as_the_folder(home):
    if not Path(str(home).upper()).exists():
        pytest.skip("case-sensitive volume")
    _refused(str(home).upper(), "no tu carpeta personal")
    assert rd.check_read_dir(str(home / "downloads")) == _c(home / "Downloads")  # its real case: what the sandbox sees
    data = "/System/Volumes/Data" + str(home)  # the firmlink alias of the data volume
    if Path(data).is_dir():
        _refused(data, "no tu carpeta personal")
        assert rd.check_read_dir(data + "/Downloads") == _c(home / "Downloads")


def test_at_most_ten_each_once_and_only_in_local_mode(home, monkeypatch):
    dl = _c(home / "Downloads")
    assert rd.check_read_dirs([str(home / "Downloads"), "~/Downloads", dl]) == [dl]
    many = []
    for i in range(rd.MAX_READ_DIRS + 1):
        (home / "Documents" / f"d{i}").mkdir()
        many.append(str(home / "Documents" / f"d{i}"))
    assert len(rd.check_read_dirs(many[:-1])) == rd.MAX_READ_DIRS
    with pytest.raises(rd.ReadDirError, match="Como mucho 10 carpetas"):
        rd.check_read_dirs(many)
    with pytest.raises(rd.ReadDirError, match="no válida"):
        rd.check_read_dirs("~/Downloads")
    monkeypatch.setattr(settings, "auth_enabled", True)  # a shared server: the folders would be the server's
    with pytest.raises(rd.ReadDirError) as e:
        rd.check_read_dirs([dl])
    assert e.value.status == 403
    assert rd.check_read_dirs([]) == []  # (emptying it is always fine)
    assert rd.usable_read_dirs({"id": 1, "read_dirs": json.dumps([dl])}) == []


@posix_only
def test_a_run_gets_only_the_folders_that_still_pass(home, tmp_path):
    dl = _c(home / "Downloads")
    usb, moved = tmp_path / "usb", tmp_path / "moved"
    usb.mkdir()
    moved.mkdir()
    stored = [dl, _c(usb), _c(moved), str(home), 7]
    usb.rmdir()  # the disk was unplugged
    moved.rmdir()
    moved.symlink_to(home / "Documents")  # replaced by a link: it leads somewhere else now
    assert rd.usable_read_dirs({"id": 1, "read_dirs": json.dumps(stored)}) == [dl]
    assert rd.usable_read_dirs({"id": 1, "read_dirs": "not json"}) == []
    assert rd.usable_read_dirs({"id": 1}) == []


# ---------------------------------------------------------------------------------------------------- the API
def _local(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "local_user", f"rd{time.time_ns()}@example.com")
    return TestClient(app)


def _project(c, **kw) -> dict:
    r = c.post("/api/projects", json={"name": "P", "executor_model": "mock/demo", "improver_model": "mock/demo", **kw})
    assert r.status_code == 200, r.text
    return r.json()


def _wait(c, run_id: int, timeout: float = 60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = c.get(f"/api/runs/{run_id}").json()
        if run["status"] in FINAL:
            return run
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_the_setting_is_kept_with_the_project_and_checked_by_the_server(home, monkeypatch):
    c = _local(monkeypatch)
    p = _project(c)
    assert p["read_dirs"] == [] and p["read_dirs_mode"] == wsmod.sandbox_mode()  # off by default
    url, dl = f"/api/projects/{p['id']}", _c(home / "Downloads")
    r = c.patch(url, json={"read_dirs": ["~/Downloads"]})
    assert r.status_code == 200 and r.json()["read_dirs"] == [dl]
    assert c.get(url).json()["read_dirs"] == [dl] and [x["read_dirs"] for x in c.get("/api/projects").json()
                                                       if x["id"] == p["id"]] == [[dl]]
    assert json.loads(db.one("SELECT read_dirs FROM projects WHERE id=?", (p["id"],))["read_dirs"]) == [dl]
    # the settings form saves the rest without it: the folders stay
    assert c.patch(url, json={"name": "Q", "max_steps": 40}).json()["read_dirs"] == [dl]
    # a folder the agent may not read: the server says why, and nothing changes
    r = c.patch(url, json={"read_dirs": [dl, str(home)]})
    assert r.status_code == 422 and "no tu carpeta personal" in r.json()["detail"]
    r = c.patch(url, json={"read_dirs": [dl, str(home / "nope")]})
    assert r.status_code == 422 and r.json()["detail"] == f"La carpeta no existe: {home / 'nope'}"
    assert c.get(url).json()["read_dirs"] == [dl]
    assert c.patch(url, json={"read_dirs": "~/Downloads"}).status_code == 422
    assert c.patch(url, json={"read_dirs": []}).json()["read_dirs"] == []


def test_a_shared_server_never_lets_the_agent_read_its_folders(home, monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)  # login on: a multi-user server
    c = TestClient(app)
    assert c.post("/api/auth/register", json={"email": f"rd{time.time_ns()}@example.com", "password": "secret123"}).status_code == 200
    p = _project(c)
    assert p["read_dirs_mode"] == "off"
    r = c.patch(f"/api/projects/{p['id']}", json={"read_dirs": [str(home / "Downloads")]})
    assert r.status_code == 403 and "modo local" in r.json()["detail"]
    assert c.patch(f"/api/projects/{p['id']}", json={"read_dirs": []}).status_code == 200


def test_a_configuration_file_never_carries_them_nor_sets_them(home, monkeypatch):
    """An exported project template leaves them out (they are this machine's), and an imported one that names some
    never widens what an agent may read: projects made from it start with none."""
    c = _local(monkeypatch)
    p = _project(c)
    c.patch(f"/api/projects/{p['id']}", json={"read_dirs": ["~/Downloads"]})
    r = c.post("/api/config/export", json={"password": "frase-larga-1", "project_id": p["id"]})
    assert r.status_code == 200, r.text
    from nesicode.server.config_io import unseal

    doc = unseal(c.get(r.json()["url"]).content, "frase-larga-1")
    assert doc["project_template"]["from"] == "P" and "read_dirs" not in json.dumps(doc)
    doc["project_template"]["read_dirs"] = [str(home / "Documents" / "x")]
    r = c.post("/api/config/import", files={"file": ("c.nesicode", seal(doc, "frase-larga-1"))}, data={"password": "frase-larga-1"})
    assert r.status_code == 200, r.text
    q = _project(c, use_template=True, name="Q")
    assert q["read_dirs"] == [] and "read_dirs" not in json.dumps(db.one("SELECT settings FROM users WHERE id=?",
                                                                            (c.get("/api/auth/me").json()["id"],)))


def test_chat_runs_get_the_folders_and_harnessing_rollouts_never(home, monkeypatch):
    from nesicode.llm.mock import MockProvider

    seen: list[tuple[str, str]] = []

    class Recorder(MockProvider):
        def conversation(self, model, system, tools_, options):
            seen.append((options.role, system))
            return super().conversation(model, system, tools_, options)

    monkeypatch.setattr(loop, "get_provider", lambda spec, keys=None: (Recorder(), "demo"))
    c = _local(monkeypatch)
    p = _project(c)
    dl = _c(home / "Downloads")
    c.patch(f"/api/projects/{p['id']}", json={"read_dirs": [dl]})
    t = c.post(f"/api/projects/{p['id']}/threads", json={}).json()
    run = c.post(f"/api/threads/{t['id']}/messages", json={"text": "usa el apk de Descargas", "attachments": []}).json()
    assert _wait(c, run["id"])["status"] == "done"
    chat = [s for role, s in seen if role == "executor"]
    assert chat and all(f"these folders of the user: `{dl}`" in s for s in chat)
    seen.clear()
    h = c.post(f"/api/projects/{p['id']}/harnessing", json={"task": "Write a report", "k": 1}).json()
    assert _wait(c, h["id"], 120)["status"] == "done"
    rollouts = [s for role, s in seen if role == "executor"]
    assert rollouts and not any("these folders of the user" in s or dl in s for s in rollouts)


# ---------------------------------------------------------------------------------------------------- the sandbox
def _line(profile: str, needle: str) -> int:
    lines = profile.splitlines()
    return next(i for i, line in enumerate(lines) if needle in line)


def test_the_profile_lets_them_be_read_after_the_walls_and_before_the_secrets(home, tmp_path):
    dl = _c(home / "Downloads")
    cage, own = tmp_path / "ws", tmp_path / "ws-home"
    cage.mkdir()
    own.mkdir()
    prof = wsmod.macos_profile(cage, own, (), [dl])
    allow = f"(allow file-read* (subpath {wsmod._q(dl)}))"
    assert allow in prof.splitlines()
    first = _line(prof, "(deny file-read* file-write*")  # the home, /Users, /Volumes, the temp dirs…
    secrets = _line(prof, f"(subpath {wsmod._q(Path(str(home)) / '.ssh')})")
    after = _line(prof, wsmod._q(Path(str(home)) / ".cargo" / "credentials"))  # the denies after the toolchains
    walls_last = _line(prof, f"(allow file-read* file-write* (subpath {wsmod._q(cage.resolve())})")
    assert first < prof.splitlines().index(allow) < secrets < after < walls_last
    assert not [line for line in prof.splitlines() if wsmod._q(dl) in line and "write" in line]  # never writable
    for s in [home / s for s in ("Library/Keychains", ".config/gh", ".aws", ".nesicode", ".netrc")] + \
             [settings.data_dir.resolve(), wsmod.vault_key_file()]:
        assert f"(subpath {wsmod._q(Path(os.path.abspath(s)))})" in prof.splitlines()[secrets], s
    # the secrets are always kept out; with no folders there is nothing more
    plain = wsmod.macos_profile(cage, own)
    assert allow not in plain and plain.splitlines()[secrets - 1].startswith("(deny file-read* file-write*")
    assert len(plain.splitlines()) == len(prof.splitlines()) - 1


def test_docker_mounts_them_read_only_and_hides_the_secrets_in_them(tmp_path, monkeypatch):
    outer = tmp_path / "outer, with a comma"
    me = outer / "me"  # (a home inside the folder: refused by the checks, and hidden anyway)
    (me / ".ssh").mkdir(parents=True)
    (me / ".netrc").write_text("machine x password y\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(me))
    monkeypatch.setenv("USERPROFILE", str(me))
    d = str(rd.canonical(outer))
    argv = wsmod.docker_argv(tmp_path / "ws", "ls", "nesicode-x", [d])
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--mount"]
    seen = wsmod.container_path
    assert f'type=bind,"source={d}","destination={seen(d)}",readonly' in mounts
    ssh, netrc = str(Path(d) / "me" / ".ssh"), str(Path(d) / "me" / ".netrc")
    assert f'type=tmpfs,"destination={seen(ssh)}",tmpfs-size=65536' in mounts
    empty = settings.data_dir.resolve() / wsmod.EMPTY_FILE
    assert f'type=bind,source={empty},"destination={seen(netrc)}",readonly' in mounts
    assert empty.is_file() and empty.stat().st_size == 0
    assert argv[-3:] == ["bash", "-lc", "ls"] and "--mount" not in wsmod.docker_argv(tmp_path / "ws", "ls", "n")
    # a folder that is itself a place the agent may never read is never mounted (the server never lets it be named)
    assert "--mount" not in wsmod.docker_argv(tmp_path / "ws", "ls", "n", [str(rd.canonical(me / ".ssh"))])
    assert wsmod.container_path(r"C:\Users\me\Downloads", windows=True) == "/c/Users/me/Downloads"
    assert wsmod.container_path("/Users/me/Downloads", windows=False) == "/Users/me/Downloads"


@macos_only
def test_a_caged_command_reads_an_allowed_folder_and_nothing_more(home, tmp_path):
    allowed = home / "Downloads"
    (allowed / "app.apk").write_bytes(b"APK-BYTES")
    (allowed / "sub").mkdir()
    (allowed / "sub" / "nota.txt").write_text("nota", encoding="utf-8")
    sibling = home / "Documents" / "x"
    (sibling / "privado.txt").write_text("PRIVADO", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "fuera.txt").write_text("FUERA", encoding="utf-8")
    (home / ".ssh" / "id_ed25519").write_text("CLAVE", encoding="utf-8")
    # links in the allowed folder that lead out of it: the sandbox judges where they lead
    (allowed / "a_file").symlink_to(elsewhere / "fuera.txt")
    (allowed / "a_dir").symlink_to(elsewhere)
    (allowed / "a_key").symlink_to(home / ".ssh" / "id_ed25519")
    ws = Workspace(tmp_path / "ws", read_dirs=[rd.check_read_dir(str(allowed))])
    q = lambda p: shlex.quote(str(p))  # noqa: E731
    code, out = ws.run(f"cat {q(allowed / 'app.apk')}; echo; ls {q(allowed / 'sub')}")
    assert code == 0 and out.split() == ["APK-BYTES", "nota.txt"], out
    code, out = ws.run(f"cp {q(allowed / 'app.apk')} . && cat app.apk")  # what the agent is told to do
    assert code == 0 and out == "APK-BYTES" and (ws.root / "app.apk").read_bytes() == b"APK-BYTES"
    for cmd in (f"echo x > {q(allowed / 'nuevo.txt')}", f"rm {q(allowed / 'app.apk')}", f"mkdir {q(allowed / 'd')}",
                f"mv {q(allowed / 'app.apk')} {q(allowed / 'b.apk')}", f"touch {q(allowed / 'app.apk')}",
                f"cp app.apk {q(allowed / 'copia.apk')}"):
        code, out = ws.run(cmd + " 2>&1")
        assert code != 0 and "not permitted" in out, (cmd, out)
    assert sorted(x.name for x in allowed.iterdir()) == ["a_dir", "a_file", "a_key", "app.apk", "sub"]
    assert (allowed / "app.apk").read_bytes() == b"APK-BYTES"
    for p in (sibling / "privado.txt", elsewhere / "fuera.txt", home / ".ssh" / "id_ed25519",
              allowed / "a_file", allowed / "a_dir" / "fuera.txt", allowed / "a_key"):
        code, out = ws.run(f"cat {q(p)} 2>&1")
        assert code != 0 and "not permitted" in out and "FUERA" not in out and "CLAVE" not in out, (p, out)
    # what the agent reads back: no wall where it may read, the wall named elsewhere, read-only where it tried to write
    ok = tools.bash(ws, {"command": f"cat {q(allowed / 'app.apk')}; ls {q(allowed / 'nada.apk')}"})
    assert "APK-BYTES" in ok and "No such file" in ok and "[sandbox]" not in ok, ok
    wall = tools.bash(ws, {"command": f"cat {q(sibling / 'privado.txt')}"})
    assert "[sandbox] That is out of your reach" in wall and f"`{_c(allowed)}`" in wall, wall
    ro = tools.bash(ws, {"command": f"echo x > {q(allowed / 'nuevo.txt')}"})
    assert tools.READ_ONLY_NOTE in ro and "out of your reach" not in ro, ro


@macos_only
def test_secrets_stay_out_of_reach_inside_an_allowed_folder(tmp_path, monkeypatch):
    """The checks never let a folder that holds the home be read; the sandbox keeps the secrets out even so."""
    outer = tmp_path / "outer"
    me = outer / "me"
    secrets = {".ssh/id_ed25519": "K-SSH", ".config/gh/hosts.yml": "K-GH", "Library/Keychains/login.keychain-db": "K-KC",
               ".aws/credentials": "K-AWS", ".netrc": "K-NETRC", ".nesicode/vault.key": "K-VAULT",
               "Library/Application Support/Firefox/Profiles/x/logins.json": "K-FF"}
    for rel, text in {**secrets, "docs/leeme.txt": "publico", ".config/myapp/x.txt": "publico"}.items():
        (me / rel).parent.mkdir(parents=True, exist_ok=True)
        (me / rel).write_text(text, encoding="utf-8")
    (outer / "otro.txt").write_text("publico", encoding="utf-8")
    monkeypatch.setenv("HOME", str(me))
    ws = Workspace(tmp_path / "ws", read_dirs=[str(rd.canonical(outer))])
    for rel in ("otro.txt", "me/docs/leeme.txt", "me/.config/myapp/x.txt"):
        code, out = ws.run(f"cat {shlex.quote(str(outer / rel))}")
        assert code == 0 and out == "publico", (rel, out)
    for rel, text in secrets.items():
        code, out = ws.run(f"cat {shlex.quote(str(me / rel))} 2>&1")
        assert code != 0 and "not permitted" in out and text not in out, (rel, out)
    assert "K-" not in ws.run(f"grep -r K- {shlex.quote(str(outer))} 2>/dev/null")[1]


@macos_only
@posix_only
def test_a_key_kept_in_a_dotfiles_folder_stays_out_when_that_folder_may_be_read(tmp_path, monkeypatch):
    me, docs = tmp_path / "Users" / "me", tmp_path / "Users" / "me" / "Documents"
    (docs / "dotfiles" / "ssh").mkdir(parents=True)
    (docs / "dotfiles" / "ssh" / "id_ed25519").write_text("K-SSH", encoding="utf-8")
    (docs / "dotfiles" / "zshrc").write_text("publico", encoding="utf-8")
    (me / ".ssh").symlink_to(docs / "dotfiles" / "ssh")
    monkeypatch.setenv("HOME", str(me))
    ws = Workspace(tmp_path / "ws", read_dirs=[rd.check_read_dir(str(docs))])
    assert ws.run(f"cat {shlex.quote(str(docs / 'dotfiles' / 'zshrc'))}")[1] == "publico"
    code, out = ws.run(f"cat {shlex.quote(str(docs / 'dotfiles' / 'ssh' / 'id_ed25519'))} 2>&1")
    assert code != 0 and "not permitted" in out and "K-SSH" not in out


# ---------------------------------------------------------------------------------------------------- the agent
def test_the_agent_is_told_which_folders_it_may_read_only_when_there_are_some(tmp_path, monkeypatch):
    plain = Workspace(tmp_path / "a")
    ws = Workspace(tmp_path / "b", read_dirs=["/Users/me/Downloads", "/Volumes/USB/apks"])
    monkeypatch.setattr(loop, "sandbox_mode", lambda: "macos")
    before = loop.build_system(base_harness(), plain)
    assert loop.OUT_OF_REACH in before and "folders of the user" not in before and loop.OUT_OF_REACH_BESIDES not in before
    now = loop.build_system(base_harness(), ws)
    assert ("You may also READ (never change) these folders of the user: `/Users/me/Downloads`, `/Volumes/USB/apks`. "
            "Your file tools work only inside the workspace: copy what you need from them with bash "
            "(e.g. `cp '/Users/me/Downloads/<file>' .`)") in now
    assert loop.OUT_OF_REACH not in now and loop.OUT_OF_REACH_BESIDES in now
    assert len(now) - len(before) < 500  # short: small windows are tight
    # no OS sandbox (Windows): what the user allows is still said, the walls are not promised
    monkeypatch.setattr(loop, "sandbox_mode", lambda: "none")
    bare = loop.build_system(base_harness(), ws)
    assert "these folders of the user" in bare and loop.OUT_OF_REACH_BESIDES not in bare and loop.OUT_OF_REACH not in bare
    assert "folders of the user" not in loop.build_system(base_harness(), plain)
    # docker: the paths its commands see (where they are mounted: the same ones, or a Windows host's at /c/…)
    monkeypatch.setattr(loop, "sandbox_mode", lambda: "docker")
    assert loop.OUT_OF_REACH_BESIDES in loop.build_system(base_harness(), ws)
    assert wsmod.seen_paths([r"C:\Users\me\Downloads"], "docker") == (
        ["/c/Users/me/Downloads"] if os.name == "nt" else [r"C:\Users\me\Downloads"])
    assert wsmod.seen_paths(["/Users/me/Downloads"], "macos") == ["/Users/me/Downloads"]


def test_the_out_of_reach_hint_never_fires_for_what_the_agent_may_read(monkeypatch):
    monkeypatch.setattr(tools, "sandbox_mode", lambda: "macos")
    dl = ["/Users/me/Downloads"]
    # a read there that printed those words (a log), a file that is not there, a quoted name with spaces: no wall
    assert not _refused_by_sandbox("cat /Users/me/Downloads/log.txt", "fallo: Operation not permitted", dl)
    assert not _refused_by_sandbox('ls "/Users/me/Downloads/no.apk"', "ls: /Users/me/Downloads/no.apk: No such file or directory", dl)
    assert not _refused_by_sandbox("cp '/Users/me/Downloads/Mi App.apk' .", "cp: Mi App.apk: No such file or directory", dl)
    assert not _refused_by_sandbox("cat /Users/me/Downloads/My\\ App.apk", "No such file or directory", ["/Users/me/Downloads"])
    # the rest of the user's folders are still a wall, also next to an allowed one
    assert _refused_by_sandbox("cat /Users/me/Documents/a.txt", "cat: /Users/me/Documents/a.txt: Operation not permitted", dl)
    assert _refused_by_sandbox("cp /Users/me/Downloads/a ~/Documents/", "cp: ~/Documents/: No such file or directory", dl)
    assert _refused_by_sandbox("ls /Users/me/DownloadsViejos", "ls: /Users/me/DownloadsViejos: Operation not permitted", dl)
    assert _refused_by_sandbox("ls ~/Downloads", "ls: /cage/home/Downloads: No such file or directory", dl)  # ~ is its own
    assert _refused_by_sandbox("cat /Users/me/Downloads/x", "Operation not permitted")  # (none allowed: as before)
    # a write in an allowed folder (macOS: not permitted; docker: a read-only mount)
    assert _refused_in_read_dirs("echo x > /Users/me/Downloads/a", "bash: /Users/me/Downloads/a: Operation not permitted", dl)
    assert _refused_in_read_dirs("rm /Users/me/Downloads/a", "rm: cannot remove '/Users/me/Downloads/a': Read-only file system", dl)
    assert not _refused_in_read_dirs("echo x > out/a", "bash: out/a: Operation not permitted", dl)
    assert not _refused_in_read_dirs("echo x > /Users/me/Downloads/a", "Operation not permitted", [])
    monkeypatch.setattr(tools, "sandbox_mode", lambda: "none")
    assert not _refused_in_read_dirs("echo x > /Users/me/Downloads/a", "Operation not permitted", dl)


def test_file_tools_say_how_to_get_a_file_from_a_folder_the_agent_may_read(tmp_path):
    allowed = (tmp_path / "dl").resolve()
    allowed.mkdir()
    ws = Workspace(tmp_path / "ws", read_dirs=[str(allowed)])
    with pytest.raises(ToolError) as e:
        ws.resolve(str(allowed / "app.apk"))
    assert "you may only READ, with bash" in str(e.value) and f"`cp {shlex.quote(str(allowed / 'app.apk'))} .`" in str(e.value)
    with pytest.raises(ToolError) as e:
        ws.resolve(str(tmp_path / "elsewhere" / "x"))
    assert f"You may READ (never change) only these folders of the user, with bash: `{allowed}`" in str(e.value)
    with pytest.raises(ToolError, match="attaches them to the chat"):
        Workspace(tmp_path / "ws2").resolve(str(allowed / "app.apk"))  # (no folders: as before)


# ---------------------------------------------------------------------------------------------------- the page
def test_the_settings_card_lists_them_and_shows_the_servers_reasons_in_both_languages(home, tmp_path):
    from test_final_web import ESBUILD, NODE, SRC, run_ts

    if not NODE or not ESBUILD.is_dir():
        pytest.skip("needs node and web/node_modules (esbuild)")
    reasons = []
    for raw in (home, home / "nope", home / ".ssh", Path(str(tmp_path)).anchor, home / "Library",
                tmp_path / "Users" / "other"):
        with pytest.raises(rd.ReadDirError) as e:
            rd.check_read_dir(str(raw))
        reasons.append(str(e.value))
    with pytest.raises(rd.ReadDirError) as e:
        rd.check_read_dirs([str(home / "Downloads")] * 1 + ["x"])
    reasons.append(str(e.value))
    reasons.append("Como mucho 10 carpetas")
    reasons.append(f"Nesicode no tiene permiso para leer esa carpeta (en el Mac: Ajustes del Sistema → Privacidad y "
                   f"seguridad → Archivos y carpetas): {home / 'Downloads'}")
    out = run_ts(tmp_path, f"""
      import {{ ReadDirsCard }} from {json.dumps(str(SRC / 'pages' / 'ProjectSettings.tsx'))}
      import {{ translateError }} from {json.dumps(str(SRC / 'i18n.ts'))}
      const base = {{ id: 1, name: 'P', domain: 'general', executor_model: 'm', improver_model: 'm', check_command: '',
                      max_steps: 60, harness_version: 1, created_at: 0, workspace: '/w', vision_model: '', image_model: '' }}
      const card = (extra: any) => html(<ReadDirsCard project={{{{ ...base, ...extra }}}} onSaved={{() => {{}}}} />)
      out.mac = card({{ read_dirs: ['/Users/me/Downloads', '/Volumes/USB/apks'], read_dirs_mode: 'macos' }})
      out.empty = card({{ read_dirs: [], read_dirs_mode: 'macos' }})
      out.none = card({{ read_dirs: [], read_dirs_mode: 'none' }})
      out.off = card({{ read_dirs: [], read_dirs_mode: 'off' }})
      setLang('en')
      out.en = card({{ read_dirs: ['/Users/me/Downloads'], read_dirs_mode: 'macos' }})
      out.reasons = {json.dumps(reasons)}.map(translateError)
      setLang('es')
    """)
    mac = out["mac"]
    assert "Carpetas que el agente puede leer" in mac and "lo que lea se envía al modelo" in mac
    assert mac.count(">Quitar<") == 2 and "<code>/Users/me/Downloads</code>" in mac and "<code>/Volumes/USB/apks</code>" in mac
    assert "Añadir Descargas" not in mac and ">Añadir<" in mac  # Downloads is there already
    assert "Añadir Descargas" in out["empty"] and "el agente solo ve la carpeta de la conversación" in out["empty"]
    assert "no van en una jaula" in out["none"] and "no van en una jaula" not in out["empty"]
    assert "Solo en la app de escritorio / modo local" in out["off"] and "<input" not in out["off"]
    en = out["en"]
    assert "Folders the agent can read" in en and "what it reads is sent to the model" in en and ">Remove<" in en
    for es, translated in zip(reasons, out["reasons"]):
        assert translated != es and not any(w in translated for w in ("carpeta", "Esa ", "Elige", "Como mucho")), translated
