"""Process trees end with Stop on every OS (Windows has no process groups: a command runs in a job object there), and a
local MCP server started through a launcher script — npx.cmd on Windows — starts with the scrubbed environment and is
stopped with everything it started. Offline: local scripts only."""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from nesicode.agent import Workspace
from nesicode.agent import workspace as wsmod

NT = os.name == "nt"


def _alive(pid: int) -> bool:
    if NT:  # (os.kill(pid, 0) is TerminateProcess there)
        import _winapi

        try:
            handle = _winapi.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        except OSError as exc:
            return exc.winerror == 5  # access denied: it exists; else no such process
        try:
            return _winapi.GetExitCodeProcess(handle) == 259  # STILL_ACTIVE
        finally:
            _winapi.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a killed orphan may linger a moment as a zombie until launchd/init reaps it
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return bool(out) and not out.startswith("Z")
    except OSError:
        return True


def _wait_dead(pid: int, timeout: float = 8.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


def _read_pid(path: Path, timeout: float = 15.0) -> int:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if text := path.read_text().strip():
                return int(text)
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError(f"{path.name} never written")


def _kill(pid: int) -> None:
    try:
        os.kill(pid, 9 if not NT else 15)  # (Windows: a TerminateProcess either way)
    except OSError:
        pass


# ====================================================================== the agent's commands
BG = "import os, time\nopen('bg.pid', 'w').write(str(os.getpid()))\ntime.sleep(60)\n"
# leaves bg.py running on its own (its parent exits at once: an orphan, as `server &`, a daemon, Start-Process) and
# off the output pipe, so the call itself never waits for it
START = textwrap.dedent("""
    import subprocess, sys
    subprocess.Popen([sys.executable, "bg.py"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
""")
SLEEP = "import time\ntime.sleep(60)\n"


def _python(ws: Workspace) -> str:
    """How a command of this workspace's shell runs our scripts with Python."""
    if not NT:
        return "python3"  # (the cage's toolchain: the test's own interpreter may be in a folder the sandbox denies)
    exe = sys.executable
    return f'& "{exe}"' if wsmod.shell_name() == "powershell" else '"' + exe.replace("\\", "/") + '"'


def _scripts(ws: Workspace) -> None:
    for name, text in (("bg.py", BG), ("start.py", START), ("sleep.py", SLEEP)):
        (ws.root / name).write_text(text, encoding="utf-8")


def test_what_a_command_left_running_outlives_the_call_but_not_stop_nor_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(wsmod, "EOF_GRACE", 0.5)
    ws = Workspace(tmp_path / "ws")
    _scripts(ws)
    code, out = ws.run(f"{_python(ws)} start.py")
    pid = _read_pid(ws.root / "bg.pid")
    try:
        assert code == 0, out
        time.sleep(0.6)
        assert _alive(pid)  # survives the call (a dev server the next command can reach)...
        ws.cancel = threading.Event()
        ws.cancel.set()  # ...but not the user's Stop: its parent is long gone, and it still goes
        assert _wait_dead(pid), "Stop did not end what the command left running"
    finally:
        _kill(pid)

    ws2 = Workspace(tmp_path / "ws2")
    _scripts(ws2)
    assert ws2.run(f"{_python(ws2)} start.py")[0] == 0
    pid2 = _read_pid(ws2.root / "bg.pid")
    try:
        time.sleep(0.3)
        assert _alive(pid2)
        ws2.kill_background()  # what the run's end calls
        assert _wait_dead(pid2)
    finally:
        _kill(pid2)


@pytest.mark.parametrize("how", ["stop", "timeout"])
def test_stop_and_timeout_end_an_orphan_the_running_command_left(tmp_path, how):
    """start.py exits at once and leaves bg.py: on Windows, taskkill /T from the shell no longer finds it (its parent
    is gone); the command's job does."""
    ws = Workspace(tmp_path / "ws")
    _scripts(ws)
    py = _python(ws)
    if how == "stop":
        ws.cancel = threading.Event()

        def stop_once_started():
            _read_pid(ws.root / "bg.pid")
            ws.cancel.set()

        threading.Thread(target=stop_once_started, daemon=True).start()
    t0 = time.monotonic()
    code, out = ws.run(f"{py} start.py; {py} sleep.py", timeout=60 if how == "stop" else 8)
    pid = _read_pid(ws.root / "bg.pid")
    try:
        assert code == (130 if how == "stop" else 124), out
        assert time.monotonic() - t0 < 20
        assert _wait_dead(pid), f"{how} did not end the orphan"
    finally:
        _kill(pid)


def test_an_interrupted_wait_never_leaves_the_command_or_its_orphan_running(tmp_path, monkeypatch):
    """A raw KeyboardInterrupt (a second Ctrl+C in the CLI) while a command runs: nothing it started survives."""
    ws = Workspace(tmp_path / "ws")
    _scripts(ws)
    real_wait = subprocess.Popen.wait

    def wait(self, timeout=None):
        if (ws.root / "bg.pid").exists() and (ws.root / "bg.pid").read_text().strip():
            raise KeyboardInterrupt
        return real_wait(self, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    py = _python(ws)
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        ws.run(f"{py} start.py; {py} sleep.py", timeout=60)
    monkeypatch.undo()
    pid = _read_pid(ws.root / "bg.pid")
    try:
        assert time.monotonic() - t0 < 20
        assert _wait_dead(pid), "the orphan outlived the interrupt"
    finally:
        _kill(pid)


@pytest.mark.skipif(not NT, reason="Windows job objects")
def test_a_command_job_is_let_go_and_closed_when_nothing_is_left(tmp_path):
    ws = Workspace(tmp_path / "ws")
    assert wsmod._WinJob.ready()
    assert ws.run("echo hola")[1].strip() == "hola"  # started suspended, let go in its job: it runs
    end = time.monotonic() + 5
    while ws._bg.groups and time.monotonic() < end:  # (its console host may end a moment after the shell)
        time.sleep(0.1)
    assert not ws._bg.groups  # nothing left running: its job is closed, nothing is watched any more


# ====================================================================== a local MCP server behind a launcher script
MCP_SERVER = textwrap.dedent('''
    import json, os, sys, threading, time
    with open(sys.argv[1], "w") as f:
        f.write(str(os.getpid()))
    # a server that does not exit when its stdin closes (Chrome, a child of npx…): only a kill of the tree ends it
    threading.Thread(target=time.sleep, args=(3600,)).start()
    from mcp.server.mcpserver import MCPServer
    app = MCPServer("w")

    @app.tool()
    def env() -> str:
        """The names of the server's environment variables."""
        return json.dumps(sorted(os.environ))

    app.run()
''')
SECRETS = {"OPENAI_API_KEY": "sk-FAKE", "NESICODE_DESKTOP_TOKEN": "FAKE-desktop", "GH_TOKEN": "g"}
NEEDED = ({"PATH", "USERPROFILE", "SYSTEMROOT", "COMSPEC", "PATHEXT", "APPDATA", "LOCALAPPDATA", "TEMP"} if NT
          else {"PATH", "HOME"})


@pytest.fixture()
def mgr(monkeypatch):
    import mcp.client.stdio as mstdio

    from nesicode.connectors import McpManager

    monkeypatch.setattr(mstdio, "PROCESS_TERMINATION_TIMEOUT", 0.5)  # the grace after stdin closes, shortened
    m = McpManager()
    yield m
    try:
        m.shutdown(8)
    finally:
        m.loop.call_soon_threadsafe(m.loop.stop)
        m.thread.join(5)


def _launcher(tmp_path: Path) -> Path:
    """`wserver` on the PATH, as npx is: npx.cmd (run by cmd.exe) on Windows, a shell script elsewhere — a process
    between Nesicode and the server, as npx's node is."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = tmp_path / "server.py"
    script.write_text(MCP_SERVER, encoding="utf-8")
    if NT:
        (bin_dir / "wserver.cmd").write_bytes(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n'.encode())
    else:
        sh = bin_dir / "wserver"
        sh.write_text(f'#!/bin/sh\n"{sys.executable}" "{script}" "$@"\n')
        sh.chmod(0o755)
    return bin_dir


def test_a_server_started_through_a_launcher_script_gets_its_environment_and_stop_ends_its_tree(tmp_path, mgr,
                                                                                                  monkeypatch):
    from nesicode.connectors import ConnectorConfig

    bin_dir = _launcher(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    pidfile = tmp_path / "server.pid"
    # the bare name, as the catalog's «npx»: found on the PATH (Windows: through PATHEXT, wserver.cmd)
    cfg = ConnectorConfig(name="lanzado", kind="stdio", command="wserver", args=[str(pidfile)], env={"CONNECTOR_TOKEN": "x"})
    assert {t.name for t in mgr.tools(1, cfg, timeout=60)} == {"env"}
    names = set(json.loads(mgr.call(1, cfg, "env", {}).text))
    assert not set(SECRETS) & names and NEEDED | {"CONNECTOR_TOKEN"} <= names
    pid = _read_pid(pidfile)
    try:
        assert _alive(pid)
        mgr.stop(1, "lanzado")  # Desactivar / Eliminar / Probar: the server goes, not only its launcher
        assert _wait_dead(pid), "the server outlived stop()"
        assert mgr.status(1, cfg) == "stopped"
    finally:
        _kill(pid)


def test_closing_the_app_ends_a_launched_server_and_what_it_started(tmp_path, mgr, monkeypatch):
    from nesicode.connectors import ConnectorConfig

    bin_dir = _launcher(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    pidfile = tmp_path / "server.pid"
    cfg = ConnectorConfig(name="lanzado", kind="stdio", command="wserver", args=[str(pidfile)])
    mgr.tools(1, cfg, timeout=60)
    pid = _read_pid(pidfile)
    try:
        mgr.shutdown(8)  # the desktop window closed / `nesicode serve` stopped
        assert _wait_dead(pid), "the server outlived the app"
    finally:
        _kill(pid)
