"""Regression tests for the agent tools, the workspace and the OS sandbox (audit group C) — offline."""

import gc
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from pathlib import Path

import pytest

from nesicode.agent import Workspace, ToolError
from nesicode.agent import workspace as wsmod
from nesicode.agent.hooks import Hooks
from nesicode.agent.tools import bash, edit_file, list_dir, read_file, search, write_file
from nesicode.agent.workspace import (
    _walk, apply_cage, apply_changes, changes_since, copy_workspace, diff_workspaces, manifest,
)
from nesicode.config import settings
from nesicode.harness import base_harness

MACOS = sys.platform == "darwin" and wsmod.sandbox_mode() == "macos"
macos_only = pytest.mark.skipif(not MACOS, reason="macOS sandbox-exec cage")
posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX shell and process groups")


def _finishes(fn, timeout=10.0):
    """Run fn in a thread; fail (instead of hanging the suite) if it does not return in time."""
    box = {}

    def target():
        try:
            box["v"] = fn()
        except BaseException as exc:  # re-raised below, in the test's thread
            box["e"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"{getattr(fn, '__name__', fn)} did not finish in {timeout}s"
    if "e" in box:
        raise box["e"]
    return box.get("v")


class _HttpServer:
    """A local web server in ANOTHER process (ports this process listens on are nesicode's own: blocked)."""

    def __enter__(self):
        code = ("import http.server, sys\n"
                "h = http.server.SimpleHTTPRequestHandler; h.log_message = lambda *a: None\n"
                "s = http.server.HTTPServer(('127.0.0.1', 0), h); print(s.server_address[1], flush=True); s.serve_forever()")
        self.proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        self.port = int(self.proc.stdout.readline())
        return self

    def __exit__(self, *exc):
        self.proc.kill()
        self.proc.wait()


CURL = "curl -s -o /dev/null -w '%{{http_code}}' --max-time 3 -I http://127.0.0.1:{}/; echo \" rc=$?\""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a killed child of launchd/init may linger a moment as a zombie
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return bool(out) and not out.startswith("Z")
    except OSError:
        return True


def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


def _short_dir() -> Path:
    """A directory with a short path (unix sockets are limited to ~104 bytes)."""
    return Path(tempfile.mkdtemp(prefix="nf"))


# ------------------------------------------------------------------ #1 symlink escapes
def test_walkers_never_follow_symlinks_out_of_the_cage(tmp_path):
    outside = tmp_path / "other-conversation"
    outside.mkdir()
    (outside / "secret.env").write_text("OPENAI_API_KEY=sk-other-conversation-secret\n")
    (outside / "deep").mkdir()
    (outside / "deep" / "more.txt").write_text("SECRET too\n")
    ws = Workspace(tmp_path / "ws")
    (ws.root / "app.py").write_text("print('hi')\n")
    os.symlink(outside / "secret.env", ws.root / "notes.txt")  # what a caged `ln -s` can do
    os.symlink(outside, ws.root / "otherdir")

    assert search(ws, {"pattern": "SECRET|sk-"}) == "no matches"
    listing = list_dir(ws, {"path": ".", "depth": 3})
    assert "notes.txt -> " in listing and "otherdir -> " in listing
    assert "deep" not in listing and "more.txt" not in listing  # never descended into the linked dir
    assert [line for line in listing.splitlines() if " B)" in line] == ["app.py (12 B)"]  # no size of a link target
    assert sorted(p.name for p in _walk(ws.root)) == ["app.py"]
    m = manifest(ws.root)
    assert m["notes.txt"].startswith("symlink:") and m["otherdir"].startswith("symlink:")
    for tool, args in ((read_file, {"path": "notes.txt"}), (edit_file, {"path": "notes.txt", "old": "x", "new": "y"}),
                       (write_file, {"path": "otherdir/x.txt", "content": "x"}), (search, {"pattern": ".", "path": "notes.txt"})):
        with pytest.raises(ToolError, match="outside the workspace"):
            tool(ws, args)
    assert not (outside / "x.txt").exists()


@posix_only
def test_list_dir_never_lists_through_a_folder_swapped_for_a_link(tmp_path, monkeypatch):
    # review: list_dir checked realpath() and THEN scandir'ed the path: a background process swapping the folder
    # for a link to ~ in between got the home's names and sizes listed
    outside = tmp_path / "home"
    outside.mkdir()
    (outside / "id_rsa").write_text("key")
    ws = Workspace(tmp_path / "ws")
    (ws.root / "d").mkdir()
    (ws.root / "d" / "mine.txt").write_text("m")
    real_scandir, calls = os.scandir, []

    def swapping(p="."):
        calls.append(p)
        if len(calls) == 2:  # about to list d: swap it for a link to the outside dir
            os.rename(ws.root / "d", ws.root / "d.real")
            os.symlink(outside, ws.root / "d")
        return real_scandir(p)

    monkeypatch.setattr(os, "scandir", swapping)
    out = list_dir(ws, {"path": ".", "depth": 3})
    assert len(calls) >= 2 and "id_rsa" not in out and "mine.txt" in out


def test_server_file_access_refuses_links_swapped_in_after_resolve(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("SECRET\n")
    ws = Workspace(tmp_path / "ws")
    (ws.root / "d").mkdir()
    (ws.root / "d" / "secret.py").write_text("mine\n")
    p = ws.resolve("d/secret.py")  # resolved while d/ is a real directory...
    shutil.rmtree(ws.root / "d")
    os.symlink(outside, ws.root / "d")  # ...then a background process swaps in a link
    with pytest.raises(ToolError):
        ws.read_bytes(p)
    with pytest.raises(ToolError):
        ws.write_bytes(p, b"pwned")
    assert (outside / "secret.py").read_text() == "SECRET\n"
    os.symlink(outside / "secret.py", ws.root / "f.txt")
    with pytest.raises(ToolError, match="symbolic link"):
        ws.read_bytes(ws.root / "f.txt")


# ------------------------------------------------------------------ #2 install_harness through a .nesicode link
@pytest.mark.parametrize("linked", [".nesicode", ".nesicode/scripts"])
def test_install_harness_never_follows_links(tmp_path, linked):
    outside = tmp_path / "users-dotclaude"
    (outside / "skills" / "my-skill").mkdir(parents=True)
    (outside / "skills" / "my-skill" / "SKILL.md").write_text("keep me")
    (outside / "scripts").mkdir()
    (outside / "scripts" / "backup.sh").write_text("keep me too")
    ws = Workspace(tmp_path / "ws")
    if linked == ".nesicode":
        os.symlink(outside, ws.root / ".nesicode")
    else:
        (ws.root / ".nesicode").mkdir()
        os.symlink(outside / "scripts", ws.root / ".nesicode" / "scripts")
    h = dict(base_harness(), **{"scripts/check.py": "print('ok')\n"})
    ws.install_harness(h)
    assert (outside / "skills" / "my-skill" / "SKILL.md").read_text() == "keep me"
    assert (outside / "scripts" / "backup.sh").read_text() == "keep me too"
    assert not (outside / "scripts" / "check.py").exists()
    nesi = ws.root / ".nesicode"
    assert not nesi.is_symlink() and not (nesi / "scripts").is_symlink()
    assert (nesi / "scripts" / "check.py").read_bytes() == b"print('ok')\n"  # LF bytes, also on Windows
    assert os.access(nesi / "scripts" / "check.py", os.X_OK)


# ------------------------------------------------------------------ #3 / #8 the macOS cage
@macos_only
def test_cage_env_is_minimal(tmp_path, monkeypatch):
    monkeypatch.setenv("NESI_FAKE_SECRET_KEY", "sk-FAKE-in-the-server-env")
    ws = Workspace(tmp_path / "ws")
    code, out = ws.run("env")
    assert code == 0 and "sk-FAKE" not in out and "NESICODE_PATH" not in out
    # Other processes' environments: `ps` is refused. (A raw sysctl(KERN_PROCARGS2) cannot be blocked by Seatbelt,
    # which is why caged commands, and every process nesicode starts, must carry no secrets in their environment.)
    for cmd in ("ps eww $PPID", "/bin/ps -E -p $PPID"):
        assert "not permitted" in ws.run(cmd + " 2>&1")[1], cmd


@macos_only
def test_cage_cannot_reach_other_processes(tmp_path):
    ws = Workspace(tmp_path / "ws")
    # nesicode's own API port (any TCP port this process listens on) is unreachable; other local ports work
    api = socket.socket()
    api.bind(("127.0.0.1", 0))
    api.listen(1)
    try:
        with _HttpServer() as srv:
            assert ws.run(CURL.format(srv.port))[1].startswith("200")
        assert "rc=7" in ws.run(CURL.format(api.getsockname()[1]))[1]  # refused, not a 3 s timeout (rc=28)
    finally:
        api.close()
    # unix sockets of other programs (tmux, ssh-agent, docker) are refused; the cage's own still work
    d = _short_dir()
    other = socket.socket(socket.AF_UNIX)
    other.bind(str(d / "s"))
    other.listen(1)
    try:
        code, out = ws.run(f"python3 -c \"import socket; s=socket.socket(socket.AF_UNIX); s.connect('{d / 's'}')\" 2>&1")
        assert code != 0 and "not permitted" in out
        own = ("import socket; a=socket.socket(socket.AF_UNIX); a.bind('own.sock'); a.listen(1); "
               "b=socket.socket(socket.AF_UNIX); b.connect('own.sock'); print('own ok')")
        assert "own ok" in ws.run(f"python3 -c \"{own}\" 2>&1")[1]
    finally:
        other.close()
        shutil.rmtree(d, ignore_errors=True)
    profile = wsmod.macos_profile(ws.root, ws.home)
    for rule in ('(deny mach-lookup (global-name "com.apple.pasteboard.1"))', "(deny lsopen)", "(deny appleevent-send)",
                 "(deny network-outbound (remote unix-socket))"):
        assert rule in profile
    assert ws.run("pbpaste >/dev/null 2>&1; echo rc=$?")[1].strip() != "rc=0"
    assert ws.run("open -g -j -a /usr/bin/true >/dev/null 2>&1; echo rc=$?")[1].strip() != "rc=0"


@macos_only
def test_cage_read_policy_and_private_temp(tmp_path):
    base = settings.data_dir / "workspaces" / "_fix_tools"
    a, b = Workspace(base / "1" / "ws"), Workspace(base / "2" / "ws")
    try:
        for cmd in ("ls /Users/Shared", 'ls "$(getconf DARWIN_USER_TEMP_DIR)"', "ls /private/tmp", "ls /Volumes",
                    f"ls {os.path.expanduser('~')}"):
            assert "not permitted" in a.run(cmd + " 2>&1")[1], cmd
        if Path(os.path.expanduser("~/.local/share")).is_dir():
            assert "not permitted" in a.run(f"ls {os.path.expanduser('~/.local/share')} 2>&1")[1]
        for var in ("/opt/homebrew/var", "/usr/local/var"):  # review: local databases of Homebrew services
            if Path(var).is_dir():
                assert "not permitted" in a.run(f"ls {var} 2>&1")[1], var
        # mktemp ignores TMPDIR on macOS: it now lands in the conversation's private temp dir, out of other cages' reach
        code, f = a.run("f=$(mktemp) && echo original > $f && echo $f")
        f = f.strip().splitlines()[-1]
        assert code == 0 and f.startswith(str(a.home)), f
        assert "not permitted" in b.run(f"cat {f} 2>&1; echo TAMPERED > {f} 2>&1")[1]
        assert Path(f).read_text() == "original\n"
        assert a.run("d=$(mktemp -d) && test -d $d && mktemp -t foo")[0] == 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


@macos_only
def test_toolchains_still_work_in_the_cage(tmp_path):
    ws = Workspace(tmp_path / "ws")
    assert ws.run("python3 -c 'import ssl, json; print(42)'")[1].strip().endswith("42")
    assert int(ws.run("ls /usr/bin | wc -l")[1].strip()) > 50
    code, out = ws.run("command -v git >/dev/null || { echo SKIP; exit 0; }; git init -q . && git status --short && echo OK")
    assert code == 0 and ("OK" in out or "SKIP" in out), out
    for tool, cmd in (("node", "node -e 1"), ("npm", "npm --version"), ("uv", "uv --version")):
        code, out = ws.run(f"command -v {tool} >/dev/null || {{ echo SKIP; exit 0; }}; {cmd}", timeout=60)
        assert code == 0, (tool, out)
    with _HttpServer() as srv:
        assert ws.run(CURL.format(srv.port))[1].startswith("200")


# ------------------------------------------------------------------ #15 / live stop-continue#5, full-window#9: PATH
@posix_only
@pytest.mark.parametrize("mode", ["auto", "none"])
def test_login_shell_keeps_the_users_path_order(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(settings, "sandbox", mode)
    ws = Workspace(tmp_path / "ws")
    tc = ws.root / ".toolchain" / "bin"  # stands in for ~/.local/bin (inside the cage so the sandbox allows it)
    tc.mkdir(parents=True)
    (tc / "python3").write_text("#!/bin/sh\necho user toolchain python\n")
    (tc / "python3").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tc}:/usr/bin:/bin:/usr/sbin:/sbin")
    code, out = ws.run("command -v python3; python3")
    assert out.splitlines()[0] == str(tc / "python3"), out  # not /usr/bin/python3 moved first by path_helper
    assert "user toolchain python" in out


def test_cage_path_drops_denied_dirs():
    home = Path(os.path.expanduser("~")).resolve()
    ws = Workspace(Path(tempfile.mkdtemp()) / "ws")
    path = os.pathsep.join([str(wsmod.REPO_DIR / ".venv" / "bin"), str(home / ".local" / "bin"), str(home / "secret" / "bin"),
                            "/usr/bin", str(ws.root / "bin")])
    got = wsmod._cage_path(path, ws.root, ws.home).split(os.pathsep)
    assert got[0] == str(wsmod._cage_bin(ws.home))
    assert str(home / ".local" / "bin") in got and "/usr/bin" in got and str(ws.root / "bin") in got
    assert str(wsmod.REPO_DIR / ".venv" / "bin") not in got and str(home / "secret" / "bin") not in got


@posix_only
def test_pipeline_failure_is_reported(tmp_path):
    ws = Workspace(tmp_path / "ws")
    out = bash(ws, {"command": "false | cat"})
    assert out.startswith("exit code: 0") and "[pipeline exit codes: 1 0]" in out  # tail's 0 no longer hides it
    out = bash(ws, {"command": "printf 'a\\nb\\n' | head -1"})
    assert out == "exit code: 0\na\n"
    assert bash(ws, {"command": "true | false"}).startswith("exit code: 1")
    assert bash(ws, {"command": "! false | false"}).startswith("exit code: 0")
    assert bash(ws, {"command": "exit 3"}).startswith("exit code: 3")
    assert bash(ws, {"command": "false | true; exit 5"}).startswith("exit code: 5")


@posix_only
def test_pipeline_report_never_lands_in_the_agents_files(tmp_path):
    # review: the report used to be shell code appended AFTER the command, so an unterminated heredoc swallowed it
    # into the file being written (a markdown table full of `|`), and a trailing `\` joined it to the last line.
    ws = Workspace(tmp_path / "ws")
    out = bash(ws, {"command": "cat > t.md <<'EOF'\n| a | b |\n|---|---|"})
    assert out.startswith("exit code: 0") and (ws.root / "t.md").read_text() == "| a | b |\n|---|---|\n"
    plain = ws.run("echo a | cat \\")  # what bash itself does with a trailing backslash...
    assert bash(ws, {"command": "echo a | cat \\"}).startswith(f"exit code: {plain[0]}\n{plain[1]}")  # ...unchanged
    errors = [ln for ln in ws.run("echo a | cat\nif then")[1].splitlines() if "syntax error" in ln]
    assert errors and errors == [ln for ln in bash(ws, {"command": "echo a | cat\nif then"}).splitlines()
                                 if "syntax error" in ln]  # bash's line numbers are still the agent's
    assert bash(ws, {"command": "trap 'echo mine' EXIT; false | cat"}) == "exit code: 0\nmine\n"  # the agent's own trap wins


# ------------------------------------------------------------------ #4 / #7 / #21 command lifetime
@posix_only
def test_runaway_output_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(wsmod, "OUTPUT_CAP", 3_000_000)
    ws = Workspace(tmp_path / "ws")
    t0 = time.monotonic()
    code, out = ws.run("yes", timeout=60)
    assert time.monotonic() - t0 < 10 and code == 137 and "more than" in out
    assert len(out) < 2 * wsmod.OUTPUT_KEEP + 500
    code, out = ws.run("python3 -c \"print('START'); print('x' * 400000); print('END')\"")
    assert code == 0 and out.startswith("START") and out.rstrip().endswith("END") and "bytes of output omitted" in out


@posix_only
@pytest.mark.parametrize("how", ["stop", "timeout"])
def test_stop_and_timeout_do_not_wait_for_detached_children(tmp_path, how):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "helper.py").write_text(
        "import os, time\nos.setsid()\nopen('helper.pid', 'w').write(str(os.getpid()))\ntime.sleep(20)\n")
    if how == "stop":
        ws.cancel = threading.Event()
        threading.Timer(0.6, ws.cancel.set).start()
    t0 = time.monotonic()
    try:
        code, out = ws.run("python3 helper.py & echo started; sleep 60", timeout=1 if how == "timeout" else 60)
        assert time.monotonic() - t0 < 5, "waited for the detached child"
        assert code == (130 if how == "stop" else 124) and "started" in out
    finally:
        pid_file = ws.root / "helper.pid"
        for _ in range(30):
            if pid_file.exists() and pid_file.read_text():
                break
            time.sleep(0.1)
        if pid_file.exists() and pid_file.read_text():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except OSError:
                pass


@posix_only
def test_background_processes_are_cleaned_up(tmp_path, monkeypatch):
    monkeypatch.setattr(wsmod, "EOF_GRACE", 0.5)
    ws = Workspace(tmp_path / "ws")
    t0 = time.monotonic()
    code, out = ws.run("sleep 30 & echo $!")  # keeps the output pipe: must not block until the timeout
    pid = int(out.strip().splitlines()[-1])
    assert code == 0 and time.monotonic() - t0 < 5
    assert _alive(pid)  # survives the call (a dev server the next call can curl)...
    ws.cancel = threading.Event()
    ws.cancel.set()  # ...but not the user's Stop
    assert _wait_dead(pid), "Stop did not kill the background process"

    ws2 = Workspace(tmp_path / "ws2")
    pid2 = int(ws2.run("sleep 30 >/dev/null 2>&1 & echo $!")[1].strip())
    assert _alive(pid2)
    ws2.kill_background()  # what the run's end calls
    assert _wait_dead(pid2)

    ws3 = Workspace(tmp_path / "ws3")
    pid3 = int(ws3.run("sleep 30 >/dev/null 2>&1 & echo $!")[1].strip())
    del ws3  # the run is over and its workspace released
    gc.collect()
    assert _wait_dead(pid3)


@posix_only
def test_background_watcher_never_leaves_a_group_unwatched():
    # review: a group added while the watcher was exiting (prune() had just found none) got no watcher: Stop did not
    # kill it and its id was never pruned, so a later kill could hit a process group that reused the id.
    bg = wsmod._Background()
    assert bg.add(10**9) is True  # nothing watched yet: start a watcher
    assert bg.add(10**9 + 1) is False  # one is running
    assert bg.prune() is False  # both ids are gone: the watcher stops...
    assert bg.add(10**9 + 2) is True  # ...so the next group starts a new one
    if os.geteuid() != 0:
        bg.groups.add(1)  # pid 1 (launchd/init): exists, but not ours to signal (an id reused by another user)
        bg.prune()
        assert 1 not in bg.groups


# ------------------------------------------------------------------ #17 Docker
@posix_only
def test_docker_timeout_kills_the_container(tmp_path, monkeypatch):
    fake = tmp_path / "fakebin"
    fake.mkdir()
    log = tmp_path / "log"
    log.mkdir()
    (fake / "docker").write_text(
        "#!/bin/bash\n"
        'if [ "$1" = run ]; then echo "$@" > "$FAKE_DOCKER_LOG/run"; exec sleep 30; fi\n'
        'if [ "$1" = kill ]; then echo "$2" >> "$FAKE_DOCKER_LOG/kill"; fi\n')
    (fake / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    monkeypatch.setattr(settings, "sandbox", "docker")
    ws = Workspace(tmp_path / "ws")
    t0 = time.monotonic()
    code, out = ws.run("echo hi", timeout=1)
    assert code == 124 and time.monotonic() - t0 < 6
    argv = (log / "run").read_text().split()
    name = argv[argv.index("--name") + 1]
    assert "--init" in argv and (log / "kill").read_text().split() == [name]
    if sys.platform.startswith("linux"):
        assert f"{os.getuid()}:{os.getgid()}" in argv


# ------------------------------------------------------------------ #18 Windows encodings
def test_children_write_utf8(tmp_path):
    ws = Workspace(tmp_path / "ws")
    env = ws._local_env("none")
    assert env["PYTHONUTF8"] == "1" and env["PYTHONIOENCODING"] == "utf-8"
    ps = wsmod.powershell_argv("Get-ChildItem")
    assert "OutputEncoding" in ps[-1] and ps[-1].endswith("Get-ChildItem")
    if os.name != "nt":
        assert "✓ año" in ws.run("python3 -c \"print('✓ año')\"")[1]


# ------------------------------------------------------------------ #19 private HOME of a user folder
@posix_only
def test_home_of_a_user_folder_lives_in_the_data_dir(tmp_path):
    parent = tmp_path / "volume-root"
    proj = parent / "proj"
    proj.mkdir(parents=True)
    ws = Workspace(proj)
    assert str(ws.home).startswith(str(settings.data_dir.resolve() / "homes"))
    parent.chmod(0o555)  # like /Volumes/X or a container's /app
    try:
        assert ws.run("echo hello") == (0, "hello\n")
        assert sorted(os.listdir(parent)) == ["proj"]  # no .proj.home litter next to the project
    finally:
        parent.chmod(0o755)
    cage = Workspace(settings.data_dir / "workspaces" / "9" / "9" / "threads" / "9" / "ws")
    assert cage.home == cage.root.parent / ".ws.home"  # a cage's home still goes away with its conversation


# ------------------------------------------------------------------ #5 / #6 encodings and line endings
def test_edit_file_keeps_legacy_encodings_and_refuses_binary(tmp_path):
    ws = Workspace(tmp_path / "ws")
    csv = "Nombre;Ciudad\nJosé;Málaga\nNuño;Logroño\nTOTAL;0\n".encode("cp1252")
    (ws.root / "datos.csv").write_bytes(csv)
    shown = read_file(ws, {"path": "datos.csv"})
    assert "José;Málaga" in shown and "cp1252" in shown
    assert edit_file(ws, {"path": "datos.csv", "old": "TOTAL;0", "new": "TOTAL;2"}).startswith("edited")
    assert (ws.root / "datos.csv").read_bytes() == csv.replace(b"TOTAL;0", b"TOTAL;2")
    edit_file(ws, {"path": "datos.csv", "old": "José", "new": "Josefa"})  # the model copies what read_file showed
    assert b"Josefa;M\xe1laga" in (ws.root / "datos.csv").read_bytes()
    with pytest.raises(ToolError, match="cp1252"):
        edit_file(ws, {"path": "datos.csv", "old": "TOTAL;2", "new": "TOTAL ✓"})
    assert b"TOTAL;2" in (ws.root / "datos.csv").read_bytes()

    db = b"SQLite format 3\x00" + b"\x00" * 100 + b"debug_modeoff" + b"\x00" * 50
    (ws.root / "app.db").write_bytes(db)
    with pytest.raises(ToolError, match="binary"):
        edit_file(ws, {"path": "app.db", "old": "debug_modeoff", "new": "debug_modeon!"})
    assert (ws.root / "app.db").read_bytes() == db

    mostly = "ok = 'ñandú'\nraw = '".encode() + b"\xff\xfe" + b"'\nx = 1\n"  # mostly UTF-8, two stray bytes
    (ws.root / "m.py").write_bytes(mostly)
    edit_file(ws, {"path": "m.py", "old": "x = 1", "new": "x = 2"})
    assert (ws.root / "m.py").read_bytes() == mostly.replace(b"x = 1", b"x = 2")

    (ws.root / "bom.txt").write_bytes(b"\xef\xbb\xbfhola\n")
    edit_file(ws, {"path": "bom.txt", "old": "hola", "new": "adiós"})
    assert (ws.root / "bom.txt").read_bytes() == "﻿adiós\n".encode()
    (ws.root / "u16.txt").write_bytes("línea\r\n".encode("utf-16"))
    edit_file(ws, {"path": "u16.txt", "old": "línea", "new": "fila"})
    assert (ws.root / "u16.txt").read_bytes() == "fila\r\n".encode("utf-16")


def test_line_endings_are_preserved(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "run.bat").write_bytes(b"@echo off\r\nset A=1\r\necho %A%\r\n")
    edit_file(ws, {"path": "run.bat", "old": "set A=1\necho %A%", "new": "set A=2\necho %A%\necho done"})
    assert (ws.root / "run.bat").read_bytes() == b"@echo off\r\nset A=2\r\necho %A%\r\necho done\r\n"
    (ws.root / "s.py").write_bytes(b"s = 'a\rb'\nt = 1\n")
    edit_file(ws, {"path": "s.py", "old": "t = 1", "new": "t = 2"})
    assert (ws.root / "s.py").read_bytes() == b"s = 'a\rb'\nt = 2\n"  # a lone CR is not a line ending
    write_file(ws, {"path": "new.sh", "content": "#!/bin/sh\necho hi\n"})
    assert (ws.root / "new.sh").read_bytes() == b"#!/bin/sh\necho hi\n"  # exactly as given, on every OS
    write_file(ws, {"path": "run.bat", "content": "@echo off\necho rewritten\n"})
    assert (ws.root / "run.bat").read_bytes() == b"@echo off\r\necho rewritten\r\n"


# ------------------------------------------------------------------ #16 / #20 read_file
def test_read_file_offsets_and_line_numbers(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "ten.txt").write_text("".join(f"line {i}\n" for i in range(1, 11)))
    assert read_file(ws, {"path": "ten.txt", "offset": 50}) == "(no lines at offset 50: the file has 10 lines)"
    (ws.root / "empty.txt").write_text("")
    assert read_file(ws, {"path": "empty.txt"}) == "(empty file)"
    (ws.root / "ff.py").write_text("a = 1\n\x0c\nb = ' '\nc = 3\nTARGET = 5\n")
    assert search(ws, {"pattern": "TARGET"}) == "ff.py:5: TARGET = 5"  # what grep -n says
    assert "     5\tTARGET = 5" in read_file(ws, {"path": "ff.py"})


def test_read_file_streams_huge_files(tmp_path, monkeypatch):
    import tracemalloc

    from nesicode.agent import tools

    monkeypatch.setattr(tools, "MAX_TEXT_FILE", 100_000)
    ws = Workspace(tmp_path / "ws")
    with open(ws.root / "big.log", "w") as f:
        for i in range(1, 200_001):
            f.write(f"{i} some log line\n")
        f.write("Z" * 300_000 + "\n")  # one huge line
    tracemalloc.start()
    out = read_file(ws, {"path": "big.log", "offset": 100, "limit": 3})
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert out.startswith("   100\t100 some log line\n   101\t101 some log line\n   102\t102 some log line")
    assert "(199899 more lines; use offset=103)" in out and peak < 5_000_000  # 200,001 lines in all
    last = read_file(ws, {"path": "big.log", "offset": 200_001, "limit": 1})
    assert "line truncated" in last and len(last) < 100_000


def test_tool_arguments_are_parsed_strictly(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "f.py").write_text("a = 1\na = 1\n")
    assert "1\ta = 1" in read_file(ws, {"path": "f.py", "offset": None, "limit": None})
    assert list_dir(ws, {"path": ".", "depth": None})
    with pytest.raises(ToolError, match="2 times"):
        edit_file(ws, {"path": "f.py", "old": "a = 1", "new": "a = 2", "replace_all": "false"})
    assert edit_file(ws, {"path": "f.py", "old": "a = 1", "new": "a = 2", "replace_all": "true"}).endswith("(2 replacement(s))")
    with pytest.raises(ToolError, match="content"):
        write_file(ws, {"path": "f.py"})
    assert (ws.root / "f.py").read_text() == "a = 2\na = 2\n"
    write_file(ws, {"path": "f.py", "content": ""})  # explicit empty content is still allowed
    assert (ws.root / "f.py").read_text() == ""
    if os.name != "nt":
        assert bash(ws, {"command": "echo ok", "timeout": None}).startswith("exit code: 0")


# ------------------------------------------------------------------ #10 / #11 / #12 search
def test_search_is_linear_on_minified_lines(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "dist").mkdir()
    (ws.root / "dist" / "bundle.min.js").write_text("function x(){return 1};" * 25_000 + "\n")  # one 575k-char line
    t0 = time.monotonic()
    assert search(ws, {"pattern": ".*TODO.*"}) == "no matches"
    assert search(ws, {"pattern": "function.*import"}) == "no matches"
    assert time.monotonic() - t0 < 8
    hit = search(ws, {"pattern": "return 1"})
    assert hit.startswith("dist/bundle.min.js:1: ") and len(hit) < 300


def test_search_windows_have_no_fake_line_ends(tmp_path):
    # review: a long line is matched in windows; at a window's end `$`, `\b` and lookaheads saw a fake end of line
    import random
    import re

    from nesicode.agent.tools import SEARCH_WINDOW, _match

    ws = Workspace(tmp_path / "ws")
    line = "x" * (SEARCH_WINDOW - 3) + "foo" + "barbaz" * 2000
    (ws.root / "min.js").write_text(line + "\n")
    for pat in ("foo$", r"foo\b", "foo(?!bar)"):
        assert search(ws, {"pattern": pat}) == "no matches", pat
    for pat in ("foobar", "baz$", "foo.*", "x+foo"):
        assert search(ws, {"pattern": pat}).startswith("min.js:1: "), pat
    rnd = random.Random(7)
    for _ in range(60):
        text = "".join(rnd.choice("abc ") for _ in range(rnd.randint(SEARCH_WINDOW + 1, 3 * SEARCH_WINDOW)))
        for pat in ("ab$", r"ab\b", "a(?!b)c", "c.*a", r"\bca", "b+$", "[ab]{3}$", "^ab"):
            rx = re.compile(pat)
            assert bool(_match(rx, text, time.monotonic() + 10)) == bool(rx.search(text)), pat


def test_search_refuses_exponential_patterns_and_huge_brace_globs(tmp_path):
    # review: `(a+)+$` on one 41-char line froze the whole server (re holds the GIL), windows or not
    from nesicode.agent.tools import _globs, _nested_repeat

    ws = Workspace(tmp_path / "ws")
    (ws.root / "a.txt").write_text("a" * 14 + "b\n")  # short: even unguarded these finish (a regression fails, never hangs)
    for pat in (r"(a+)+$", r"(\w+\s?)+$", r"(\s*\w+)*:", r"(.*)*x", r"((?:,\w+)*)+$", r"(a+){10}$"):
        with pytest.raises(ToolError, match="exponential"):
            search(ws, {"pattern": pat})
    for pat in (r"(a+)+", r"(\w+\.)+Foo", r"(ab+)+$", r"(foo|bar)+", r"(\s*,\s*\w+)*$", r"(?:\r?\n)+", r"(a{3})+$",
                r"(a++)+$", r"(?>a+)+$", r"import .* from", r".*TODO.*", r"def \w+\(self\):$"):
        assert not _nested_repeat(pat), pat
    t0 = time.monotonic()
    assert len(_globs("{a,b}" * 30 + ".py")) == 64 and time.monotonic() - t0 < 1  # was 2^30 expansions


def test_search_stops_on_cancel_and_globs_like_ripgrep(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "src").mkdir()
    (ws.root / "App.tsx").write_text("useThing()\n")
    (ws.root / "src" / "api.ts").write_text("useThing()\n")
    (ws.root / "main.py").write_text("useThing()\n")
    (ws.root / "src" / "util.py").write_text("useThing()\n")
    assert sorted(h.split(":")[0] for h in search(ws, {"pattern": "useThing", "glob": "*.{ts,tsx}"}).splitlines()) == \
        ["App.tsx", "src/api.ts"]
    assert sorted(h.split(":")[0] for h in search(ws, {"pattern": "useThing", "glob": "**/*.py"}).splitlines()) == \
        ["main.py", "src/util.py"]
    (ws.root / "huge.sql").write_text("useThing()\n" + "x" * 2_100_000)
    out = search(ws, {"pattern": "useThing"})
    assert "huge.sql" in out and "not searched" in out  # never skipped silently
    ws.cancel = threading.Event()
    ws.cancel.set()
    assert "stopped by the user" in search(ws, {"pattern": "useThing"})


@posix_only
def test_fifos_and_device_links_never_hang(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "a.txt").write_text("hello\n")
    base = manifest(ws.root)
    os.mkfifo(ws.root / "test.pipe")
    os.symlink("/dev/zero", ws.root / "zeros.txt")
    assert _finishes(lambda: search(ws, {"pattern": "hello"})) == "a.txt:1: hello"
    assert _finishes(lambda: manifest(ws.root))
    assert _finishes(lambda: changes_since(base, ws.root, tmp_path / "proj"))["removed"] == []
    assert _finishes(lambda: list_dir(ws, {}))
    assert _finishes(lambda: diff_workspaces(ws.root, ws.root))["modified"] == []
    with pytest.raises(ToolError):
        _finishes(lambda: ws.read_bytes(ws.root / "test.pipe"))


# ------------------------------------------------------------------ #13 copy_workspace
@posix_only
def test_copy_workspace_skips_sockets_and_fifos(tmp_path):
    proj = _short_dir()
    try:
        (proj / "app.py").write_text("print(1)\n")
        (proj / ".git").mkdir()
        os.mkfifo(proj / "tmp.fifo")
        s = socket.socket(socket.AF_UNIX)
        s.bind(str(proj / ".git" / "fsmonitor.ipc"))
        os.symlink("app.py", proj / "link.py")
        copy_workspace(proj, tmp_path / "cage")  # used to raise shutil.Error after copying everything else
        s.close()
        assert (tmp_path / "cage" / "app.py").read_text() == "print(1)\n"
        assert os.readlink(tmp_path / "cage" / "link.py") == "app.py"
        assert not os.path.lexists(tmp_path / "cage" / "tmp.fifo")
        assert not os.path.lexists(tmp_path / "cage" / ".git" / "fsmonitor.ipc")
    finally:
        shutil.rmtree(proj, ignore_errors=True)


# ------------------------------------------------------------------ #9 apply: big files and links are not deletions
def _cage(tmp_path):
    proj = tmp_path / "project"
    proj.mkdir()
    cage = tmp_path / "cage"
    return proj, cage


@posix_only
def test_apply_cage_keeps_big_files_and_links(tmp_path, monkeypatch):
    monkeypatch.setattr(wsmod, "MAX_HASH", 1000)
    proj, cage = _cage(tmp_path)
    (proj / "events.csv").write_text("x" * 900)
    (proj / "settings.json").write_text('{"env": "prod"}')
    (proj / "keep.txt").write_text("keep")
    copy_workspace(proj, cage)
    base = manifest(cage)
    with open(cage / "events.csv", "a") as f:
        f.write("y" * 500)  # grows past the hash limit
    (cage / "settings.dev.json").write_text('{"env": "dev"}')
    (cage / "settings.json").unlink()
    os.symlink("settings.dev.json", cage / "settings.json")
    ch = changes_since(base, cage, proj)
    assert ch["removed"] == [] and ch["added"] == ["settings.dev.json"]
    assert ch["modified"] == ["events.csv", "settings.json"] and "Symlink changed: settings.json" in ch["diff"]
    res = apply_cage(base, cage, proj)
    assert res["skipped"] == [] and (proj / "events.csv").read_text() == "x" * 900 + "y" * 500
    assert os.readlink(proj / "settings.json") == "settings.dev.json" and (proj / "keep.txt").exists()
    # a legacy manifest (no version key) never recorded links or big files: they are not 'added'...
    legacy = {k: v for k, v in manifest(cage).items() if k and not v.startswith(("symlink:", "large:"))}
    ch = changes_since(legacy, cage, proj)
    assert ch["added"] == [] and ch["removed"] == []  # ...nor deleted


@posix_only
def test_a_folder_replaced_by_a_link_is_never_listed(tmp_path, monkeypatch):
    # review: the "still exists?" check of removed paths listed THROUGH a folder the agent turned into a link
    proj, cage = _cage(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "b.txt").write_text("someone else's\n")
    (proj / "a").mkdir()
    (proj / "a" / "b.txt").write_text("mine\n")
    copy_workspace(proj, cage)
    base = manifest(cage)
    shutil.rmtree(cage / "a")
    os.symlink(outside, cage / "a")
    listed: list[str] = []
    real = os.listdir
    monkeypatch.setattr(os, "listdir", lambda p=".": (listed.append(os.fspath(p)), real(p))[1])
    ch = changes_since(base, cage, proj)
    assert ch["removed"] == ["a/b.txt"] and ch["added"] == ["a"]
    assert not any(os.path.realpath(p).startswith(str(outside.resolve())) for p in listed), listed
    d = diff_workspaces(proj, cage, max_chars=0)
    assert d["removed"] == ["a/b.txt"]
    monkeypatch.setattr(os, "listdir", real)
    res = apply_cage(base, cage, proj)
    assert (outside / "b.txt").read_text() == "someone else's\n" and not (proj / "a" / "b.txt").exists()
    assert any(s.startswith("a:") for s in res["skipped"])  # the link points out of the project: not created


@posix_only
def test_apply_never_writes_or_links_outside_the_project(tmp_path):
    proj, cage = _cage(tmp_path)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "secret.py").write_text("SECRET\n")
    (victim / ".env").write_text("OPENAI_API_KEY=sk-victim\n")
    (proj / "app.py").write_text("v1\n")
    os.symlink(victim / "secret.py", proj / "pwn")  # links in the project (e.g. from a cloned repo)
    os.symlink(victim, proj / "d")
    copy_workspace(proj, cage)
    base = manifest(cage)
    (cage / "pwn").unlink()
    (cage / "pwn").write_text("replaced through the link\n")
    (cage / "d").unlink()
    (cage / "d").mkdir()
    (cage / "d" / ".env").write_text("mine\n")
    os.symlink("/etc/hosts", cage / "abs_link")
    ch = changes_since(base, cage, proj)
    assert "sk-victim" not in ch["diff"] and "SECRET" not in ch["diff"]
    res = apply_cage(base, cage, proj)
    assert (victim / "secret.py").read_text() == "SECRET\n" and (victim / ".env").read_text() == "OPENAI_API_KEY=sk-victim\n"
    assert not (proj / "pwn").is_symlink() and (proj / "pwn").read_text() == "replaced through the link\n"
    assert not os.path.lexists(proj / "abs_link")
    assert any(s.startswith("abs_link:") for s in res["skipped"]) and any(s.startswith("d/.env") for s in res["skipped"])


def _case_insensitive(d: Path) -> bool:
    (d / "CaseProbe").write_text("x")
    try:
        return (d / "caseprobe").exists()
    finally:
        (d / "CaseProbe").unlink()


def test_case_only_rename_does_not_delete_the_file(tmp_path):
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem")
    proj, cage = _cage(tmp_path)
    (proj / "src").mkdir()
    (proj / "src" / "button.tsx").write_text("export const Button = 1\n")
    copy_workspace(proj, cage)
    snap = tmp_path / "snapshot"
    copy_workspace(proj, snap)
    base = manifest(cage)
    os.rename(cage / "src" / "button.tsx", cage / "src" / "Button.tsx")
    ch = changes_since(base, cage, proj)
    assert ch["added"] == ["src/Button.tsx"] and ch["removed"] == ["src/button.tsx"]
    apply_cage(base, cage, proj)  # chat 'Aplicar'
    assert os.listdir(proj / "src") == ["Button.tsx"]
    assert (proj / "src" / "Button.tsx").read_text() == "export const Button = 1\n"
    proj2 = tmp_path / "project2"
    copy_workspace(snap, proj2)
    d = apply_changes(snap, cage, proj2)  # harnessing apply
    assert d["removed"] == ["src/button.tsx"] and os.listdir(proj2 / "src") == ["Button.tsx"]


# ------------------------------------------------------------------ #14 / #22 / harness#12 hooks
BLOCKED = ["rm -rf /", "rm -rf ~", "rm -rf $HOME", "rm -rf /*", "rm -fr /", "rm -r -f ~", "rm -f -r $HOME",
           "rm --recursive --force /", "rm -Rf ~/", "rm -rf ~/*", 'rm -rf "$HOME"', "rm -rf ${HOME}/", "cd /tmp && rm -rf ~",
           "sudo rm -rf / --no-preserve-root", "rm -rf ~ && echo done", "rm -rf build /"]
ALLOWED = ["rm -rf build/", "rm -rf ./build", "rm -rf ~/project/build", "rm -rf /tmp/x", "git rm -r --cached .", "ls /",
           "rm -rf $HOME_BACKUP", "echo ~",
           # review: rm's arguments end at the line; a later `cd ~` / `ls /` is not one of them
           "rm -rf dist\ncd ~", "rm -f out.txt\nls /", "rm -rf node_modules\necho $HOME", "rm -rf /tmp/x\ncd /"]
BLOCKED += ["cd x\nrm -rf ~", "echo hi\nrm -rf $HOME/*", "rm\t-rf\t/"]


def test_base_rm_rule_blocks_root_and_home():
    hooks = Hooks(json.loads(base_harness()["plugins/hooks.json"]))
    for cmd in BLOCKED:
        assert hooks.before("bash", {"command": cmd})[0], cmd
    for cmd in ALLOWED:
        assert hooks.before("bash", {"command": cmd})[0] is None, cmd


def test_hooks_match_each_argument_and_survive_bad_rules():
    hooks = Hooks({"pre_tool": [{"tool": "bash", "match": "^ls$", "action": "block", "message": "no ls"},
                                {"tool": "*", "match": "(", "action": "block"},  # invalid: never matches, never crashes
                                {"tool": "write_file", "match": '"path": "secret', "action": "warn", "message": "careful"}],
                   "post_tool": [{"tool": "*", "action": "truncate", "max_chars": 1}]})
    assert hooks.before("bash", {"command": "ls"})[0] == "no ls"
    assert hooks.before("bash", {"command": "ls -la"})[0] is None
    assert hooks.before("write_file", {"path": "secret.txt", "content": "x"}) == (None, ["careful"])  # JSON rules still work
    assert len(hooks.after("read_file", "x" * 1000)) < 60


def test_hooks_apply_to_error_results():
    hooks = Hooks({"pre_tool": [{"tool": "edit_file", "action": "warn", "message": "PRE-WARN"}],
                   "post_tool": [{"tool": "edit_file", "match": "was not found", "action": "append", "message": "re-read the file"}]})
    _, warnings = hooks.before("edit_file", {"path": "a.py", "old": "x", "new": "y"})
    out = hooks.finish("edit_file", "Error: `old` was not found; read the file", warnings)
    assert out.startswith("[hook] PRE-WARN\nError:") and out.endswith("[hint] re-read the file")


def test_hooks_apply_to_error_results_in_the_loop(tmp_path):
    from nesicode.agent import Agent
    from nesicode.llm import ToolCall, Usage

    h = dict(base_harness(), **{"plugins/hooks.json": json.dumps({
        "pre_tool": [{"tool": "edit_file", "action": "warn", "message": "PRE-WARN"}],
        "post_tool": [{"tool": "edit_file", "match": "was not found", "action": "append", "message": "HINT"}]})})
    ws = Workspace(tmp_path / "ws")
    (ws.root / "a.py").write_text("x = 1\n")
    agent = Agent(model="mock/demo", harness=h, workspace=ws)
    res = agent._call_tool(ToolCall("1", "edit_file", {"path": "a.py", "old": "nope", "new": "y"}), Usage())
    assert res.is_error and "PRE-WARN" in res.content and "[hint] HINT" in res.content


# ------------------------------------------------------------------ #23 vision
def _png(path: Path, w: int, h: int) -> None:
    raw = b"".join(b"\x00" + os.urandom(w * 3) for _ in range(h))  # noise: does not compress

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


class _FakeVision:
    def __init__(self):
        self.calls = []

    def vision(self, model, question, data, mime):
        self.calls.append((len(data), mime, data[:4]))
        return "a picture", None


def test_view_image_limits_sniffing_and_webp(tmp_path, monkeypatch):
    from nesicode.agent import vision

    fake = _FakeVision()
    monkeypatch.setattr(vision, "get_provider", lambda spec, keys: (fake, "m"))
    assert vision.image_mime("shot.WEBP") == "image/webp" and vision.image_mime("a.txt") is None
    ws = Workspace(tmp_path / "ws")
    (ws.root / "photo.png").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # a JPEG named .png
    assert vision.describe_image(ws, "photo.png", "v/m", None, "") == "a picture"
    assert fake.calls[-1][1] == "image/jpeg"
    monkeypatch.setattr(vision, "MAX_SOURCE_BYTES", 50)
    with pytest.raises(ToolError, match="too large"):
        vision.describe_image(ws, "photo.png", "v/m", None, "")
    monkeypatch.setattr(vision, "MAX_SOURCE_BYTES", 60_000_000)
    _png(ws.root / "retina.png", 1300, 1000)  # ~3.9 MB of raw image bytes: over the ~3.7 MB the API accepts
    assert (ws.root / "retina.png").stat().st_size > vision.MAX_IMAGE_BYTES
    if MACOS:
        assert vision.describe_image(ws, "retina.png", "v/m", None, "what is it?") == "a picture"
        size, mime, head = fake.calls[-1]
        assert size <= vision.MAX_IMAGE_BYTES and mime == "image/jpeg" and head[:3] == b"\xff\xd8\xff"
    else:
        with pytest.raises(ToolError, match="smaller copy"):
            vision.describe_image(ws, "retina.png", "v/m", None, "")
