"""v0.4.7, the cage: stop_process stops what an earlier command left running (in the macOS sandbox a later `kill`
cannot), rustup-managed Rust works in the cage, and timeout, Stop and the run's end also reach what a command started
with setsid(). Offline: local scripts only."""

import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from nesicode.agent import ToolError, Workspace
from nesicode.agent import tools
from nesicode.agent import workspace as wsmod

NT = os.name == "nt"
MACOS = sys.platform == "darwin" and wsmod.sandbox_mode() == "macos"
posix_only = pytest.mark.skipif(NT, reason="POSIX shell, process groups and setsid")


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


def _kill(*pids: int) -> None:
    for pid in pids:
        if pid <= 1:  # (0 would be the test's own process group: never)
            continue
        try:
            os.kill(pid, 9 if not NT else 15)  # (Windows: a TerminateProcess either way)
        except OSError:
            pass


def _python(ws: Workspace) -> str:
    """How a command of this workspace's shell runs our scripts with Python (as in test_proc_trees)."""
    if not NT:
        return "python3"  # (the cage's toolchain: the test's own interpreter may be in a folder the sandbox denies)
    exe = sys.executable
    return f'& "{exe}"' if wsmod.shell_name() == "powershell" else '"' + exe.replace("\\", "/") + '"'


# ====================================================================== tools-1: stop_process
BG = "import os, time\nopen('bg.pid', 'w').write(str(os.getpid()))\ntime.sleep(60)\n"
# leaves bg.py running on its own (its parent exits at once: an orphan, as `server &`) and off the output pipe
START = textwrap.dedent("""
    import subprocess, sys
    subprocess.Popen([sys.executable, "bg.py"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
""")


def test_stop_process_lists_and_stops_what_an_earlier_command_left_running(tmp_path, monkeypatch):
    """Every OS: the process a command left running is listed, and stopped on request (Windows: its job goes)."""
    monkeypatch.setattr(wsmod, "EOF_GRACE", 0.5)
    ws = Workspace(tmp_path / "ws")
    assert tools.stop_process(ws, {}) == "Nothing your commands started is running in the background."
    (ws.root / "bg.py").write_text(BG, encoding="utf-8")
    (ws.root / "start.py").write_text(START, encoding="utf-8")
    assert ws.run(f"{_python(ws)} start.py")[0] == 0
    pid = _read_pid(ws.root / "bg.pid")
    try:
        listing = tools.stop_process(ws, {})
        line = next((ln for ln in listing.splitlines() if ln.startswith(f"pid {pid}: ")), "")
        assert line and (NT or line.endswith(" bg.py")), listing  # (Windows lists the program's name)
        out = tools.stop_process(ws, {"pid": pid})
        assert out.startswith("Stopped: ") and f"pid {pid} (" in out, out
        assert _wait_dead(pid)
        end = time.monotonic() + 5
        while "Nothing" not in (listing := tools.stop_process(ws, {})) and time.monotonic() < end:
            time.sleep(0.1)  # (Windows: a console host may end a moment after)
        assert listing == "Nothing your commands started is running in the background."
    finally:
        _kill(pid)


@posix_only
def test_a_later_kill_is_refused_by_the_cage_but_stop_process_is_not(tmp_path):
    ws = Workspace(tmp_path / "ws")
    pid = int(ws.run("sleep 60 >/dev/null 2>&1 & echo $!")[1].strip())
    try:
        if MACOS:  # why the tool exists: each command runs in a sandbox of its own, which only signals its own processes
            out = ws.run(f"kill {pid} 2>&1; echo rc=$?")[1]
            assert "not permitted" in out and _alive(pid), out
        assert f"pid {pid}: sleep 60" in tools.stop_process(ws, {})
        assert tools.stop_process(ws, {"pid": str(pid)}) == f"Stopped: pid {pid} (sleep 60)"  # (a pid sent as a string)
        assert _wait_dead(pid)
    finally:
        _kill(pid)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, timeout: float = 15.0) -> str:
    end = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return r.read().decode()
        except OSError:
            if time.monotonic() > end:
                raise
            time.sleep(0.1)


@posix_only
def test_a_server_restarted_after_an_edit_serves_the_new_code(tmp_path):
    """The failure: the old server could not be stopped, the new one died on 'Address already in use' (in its log) and
    the agent kept testing the old code."""
    ws = Workspace(tmp_path / "ws")
    port = _free_port()
    serve = f"python3 -m http.server {port} --bind 127.0.0.1 --directory site > server.log 2>&1 & echo $!"
    (ws.root / "site").mkdir()
    (ws.root / "site" / "index.html").write_text("v1")
    pid = int(ws.run(serve)[1].strip())
    pid2 = 0
    try:
        assert _get(f"http://127.0.0.1:{port}/") == "v1"
        (ws.root / "site" / "index.html").write_text("v2")  # the edit; the server must be restarted to pick it up
        assert tools.stop_process(ws, {"pid": pid}).startswith(f"Stopped: pid {pid} ")
        pid2 = int(ws.run(serve)[1].strip())
        assert _get(f"http://127.0.0.1:{port}/") == "v2"
        assert "Address already in use" not in (ws.root / "server.log").read_text()
    finally:
        _kill(pid, pid2)


@posix_only
def test_stop_process_only_stops_what_this_workspaces_commands_started(tmp_path):
    ws, other = Workspace(tmp_path / "ws"), Workspace(tmp_path / "other")
    outsider = subprocess.Popen(["sleep", "60"])  # a process of the user's (the server's own child here)
    theirs = int(other.run("sleep 61 >/dev/null 2>&1 & echo $!")[1].strip())  # another conversation's
    mine = int(ws.run("sleep 64 >/dev/null 2>&1 & echo $!")[1].strip())
    try:
        for pid in (outsider.pid, theirs, os.getpid(), 1):
            with pytest.raises(ToolError, match="not one of the processes your commands left running"):
                tools.stop_process(ws, {"pid": pid})
        with pytest.raises(ToolError, match="`pid`"):
            tools.stop_process(ws, {"pid": 0})
        with pytest.raises(ToolError, match="integer"):
            tools.stop_process(ws, {"pid": "the server"})
        assert tools.stop_process(ws, {"all": True}) == f"Stopped: pid {mine} (sleep 64)"
        assert _wait_dead(mine)
        assert outsider.poll() is None and _alive(theirs)
    finally:
        outsider.kill()
        outsider.wait()
        _kill(theirs, mine)


STUBBORN = textwrap.dedent("""
    import os, signal, subprocess, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen(["sleep", "62"])
    open("pids", "w").write(f"{os.getpid()} {child.pid}")
    time.sleep(60)
""")


@posix_only
def test_stop_process_stops_what_it_started_and_kills_what_ignores_sigterm(tmp_path, monkeypatch):
    monkeypatch.setattr(wsmod, "STOP_GRACE", 0.5)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "stubborn.py").write_text(STUBBORN)
    ws.run("python3 stubborn.py >/dev/null 2>&1 &")
    end = time.monotonic() + 15
    while len((ws.root / "pids").read_text().split() if (ws.root / "pids").exists() else ()) < 2:
        assert time.monotonic() < end, "stubborn.py never started"
        time.sleep(0.05)
    pid, child = map(int, (ws.root / "pids").read_text().split())
    other = int(ws.run("sleep 63 >/dev/null 2>&1 & echo $!")[1].strip())
    try:
        listing = tools.stop_process(ws, {})
        assert f"pid {child}: sleep 62  (started by {pid})" in listing and f"pid {other}: sleep 63" in listing, listing
        t0 = time.monotonic()
        out = tools.stop_process(ws, {"pid": pid})
        assert time.monotonic() - t0 < 5
        assert out.startswith("Stopped: ") and f"pid {pid} (" in out and f"pid {child} (sleep 62)" in out, out
        assert _wait_dead(pid) and _wait_dead(child)
        assert _alive(other)  # a process of another command: not part of what was stopped
        assert tools.stop_process(ws, {"pid": "all"}) == f"Stopped: pid {other} (sleep 63)"
        assert _wait_dead(other)
    finally:
        _kill(pid, child, other)


def test_the_agent_offers_stop_process_and_bash_points_to_it(tmp_path):
    from nesicode.agent import Agent
    from nesicode.harness import base_harness
    from nesicode.llm import ToolCall, Usage

    a = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(tmp_path / "ws"))
    specs = {s.name: s for s in a.tool_specs()}
    assert "stop_process" in specs and "stop_process" in specs["bash"].description
    assert "required" not in specs["stop_process"].parameters  # no argument: the list
    r = a._call_tool(ToolCall("1", "stop_process", {}), Usage())
    assert not r.is_error and "Nothing your commands started is running" in r.content
    assert a._call_tool(ToolCall("2", "stop_process", {"pid": os.getpid()}), Usage()).is_error


# ====================================================================== tools-7: what left the command's group
HELPER = "import os, time\nos.setsid()\nopen('helper.pid', 'w').write(str(os.getpid()))\ntime.sleep(60)\n"


@posix_only
@pytest.mark.parametrize("how", ["stop", "timeout"])
def test_stop_and_timeout_end_a_child_that_called_setsid(tmp_path, how):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "helper.py").write_text(HELPER)
    if how == "stop":
        ws.cancel = threading.Event()
        threading.Thread(target=lambda: (_read_pid(ws.root / "helper.pid"), ws.cancel.set()), daemon=True).start()
    try:
        code, out = ws.run("python3 helper.py & echo started; sleep 60", timeout=60 if how == "stop" else 2)
        assert code == (130 if how == "stop" else 124), out
        pid = _read_pid(ws.root / "helper.pid")
        assert _wait_dead(pid), f"{how} left the detached child running"
    finally:
        if (ws.root / "helper.pid").exists():
            _kill(_read_pid(ws.root / "helper.pid"))


# a background launcher (a test runner, a dev server) that detaches a child of its own (a browser) and stays, until told
LAUNCH = textwrap.dedent("""
    import os, subprocess, sys, time
    subprocess.Popen([sys.executable, "-c", "import os, time; open('child.pid', 'w').write(str(os.getpid())); time.sleep(60)"],
                     start_new_session=True)
    open("launcher.pid", "w").write(str(os.getpid()))
    while not os.path.exists("go"):
        time.sleep(0.05)
""")


@posix_only
@pytest.mark.parametrize("how", ["run end", "stop", "launcher gone"])
def test_what_a_background_process_detached_ends_with_the_run_and_with_stop(tmp_path, monkeypatch, how):
    monkeypatch.setattr(wsmod, "EOF_GRACE", 0.5)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "launch.py").write_text(LAUNCH)
    code, out = ws.run("python3 launch.py >/dev/null 2>&1 & while [ ! -s child.pid ]; do sleep 0.05; done")
    child, launcher = _read_pid(ws.root / "child.pid"), _read_pid(ws.root / "launcher.pid")
    try:
        assert code == 0, out
        time.sleep(0.3)
        assert _alive(child) and _alive(launcher)  # both outlive the call (a server the next command can reach)
        if how == "launcher gone":  # its parent ends first: only what was recorded when the call ended still finds it
            (ws.root / "go").write_text("")
            assert _wait_dead(launcher)
            assert _alive(child)
        if how == "stop":
            ws.cancel = threading.Event()
            ws.cancel.set()
        else:
            ws.kill_background()  # what the run's end calls
        assert _wait_dead(child), f"{how}: the detached child outlived it"
        assert _wait_dead(launcher)
    finally:
        _kill(child, launcher)


def test_the_process_table_falls_back_to_proc_without_ps(tmp_path, monkeypatch):
    """A slim Linux container has no `ps`: /proc is read instead (a name may hold spaces and parentheses)."""
    proc = tmp_path / "proc"
    for pid, stat, cmdline in (("100", "100 (python3) S 1 100 100 0 -1 4194304", b"python3\0app.py\0"),
                               ("101", "101 (we ird) (name) Z 100 100 100 0 -1 4194304", b""),
                               ("102", "garbage", b"x")):
        (proc / pid).mkdir(parents=True)
        (proc / pid / "stat").write_text(stat)
        (proc / pid / "cmdline").write_bytes(cmdline)
    (proc / "self").mkdir()
    (proc / "103").mkdir()  # (gone while listed: no stat)
    monkeypatch.setenv("PATH", str(tmp_path / "no-ps-here"))
    table = wsmod._process_table(commands=True, proc_root=str(proc))
    assert table == {100: wsmod._Proc(1, 100, False, "python3 app.py"), 101: wsmod._Proc(100, 100, True, "")}
    assert wsmod._process_table(proc_root=str(tmp_path / "nothing")) == {}


# ====================================================================== tools-5: rustup-managed Rust
def test_the_cage_points_rustup_at_the_users_toolchains(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # (Windows: expanduser reads this one)
    monkeypatch.delenv("RUSTUP_HOME", raising=False)
    monkeypatch.setenv("CARGO_HOME", str(home / ".cargo"))
    ws = Workspace(tmp_path / "ws")
    env = ws._local_env("none")
    assert "RUSTUP_HOME" not in env  # no rustup: nothing to point at
    (home / ".rustup").mkdir()
    env = ws._local_env("none")
    assert env["RUSTUP_HOME"] == os.path.join(str(home), ".rustup")
    assert "CARGO_HOME" not in env  # cargo's own home stays the private one (its registry cache is written there)
    custom = tmp_path / "rustup-elsewhere"
    custom.mkdir()
    monkeypatch.setenv("RUSTUP_HOME", str(custom))  # the user's own setting wins
    assert ws._local_env("none")["RUSTUP_HOME"] == str(custom)


_RUSTUP = Path(os.path.expanduser("~/.rustup"))
_CARGO_BIN = Path(os.path.expanduser("~/.cargo/bin"))


@posix_only
@pytest.mark.skipif(not ((_CARGO_BIN / "cargo").exists() and _RUSTUP.is_dir() and any((_RUSTUP / "toolchains").glob("*"))),
                    reason="rustup-managed Rust is not installed")
def test_rustup_managed_rust_builds_in_the_cage(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", f"{_CARGO_BIN}{os.pathsep}{os.environ['PATH']}")
    ws = Workspace(tmp_path / "ws")
    code, out = ws.run("cargo new -q --vcs none hello && cd hello && cargo build -q && ./target/debug/hello", timeout=300)
    assert code == 0 and "Hello, world!" in out, out
    assert not any(ws.home.glob(".rustup/toolchains/*"))  # no toolchain downloaded into the private home
