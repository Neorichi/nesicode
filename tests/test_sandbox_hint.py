"""When the sandbox keeps something out, the agent is told to ask the user for the file — not to hunt for a way in.

Seen live: an agent that needed an APK the user's Chrome had downloaded tried ~/Downloads («Operation not permitted»),
then relaunched Finder «cold», spending steps on a wall that is there on purpose."""

import shutil
import tempfile
import sys
from pathlib import Path

import pytest

from nesicode.agent import Workspace
from nesicode.agent import loop
from nesicode.agent import tools
from nesicode.agent.tools import _refused_by_sandbox
from nesicode.agent.workspace import ToolError
from nesicode.harness import base_harness

MAC_SANDBOX = sys.platform == "darwin" and shutil.which("sandbox-exec")


def test_the_system_prompt_says_what_is_out_of_reach_only_when_a_sandbox_keeps_it_out(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    monkeypatch.setattr(loop, "sandbox_mode", lambda: "macos")
    assert loop.OUT_OF_REACH in loop.build_system(base_harness(), ws)
    monkeypatch.setattr(loop, "sandbox_mode", lambda: "none")  # (Windows: no OS sandbox, so no such promise)
    assert loop.OUT_OF_REACH not in loop.build_system(base_harness(), ws)


def test_file_tools_refusing_a_path_outside_say_how_a_file_gets_in(tmp_path):
    ws = Workspace(tmp_path / "ws")
    with pytest.raises(ToolError, match="attaches them to the chat"):
        ws.resolve("../../Downloads/app.apk")


def test_the_hint_is_for_refusals_of_the_users_folders_and_apps_only(monkeypatch):
    monkeypatch.setattr(tools, "sandbox_mode", lambda: "macos")
    assert _refused_by_sandbox("cp ~/Downloads/app.apk .", "cp: /Users/x/Downloads/app.apk: Operation not permitted")
    assert _refused_by_sandbox("ls $HOME/Documents", "ls: Documents: Operation not permitted")
    assert _refused_by_sandbox("open -a Finder ~/Downloads", "LSOpenURLsWithRole() failed with error -54")
    assert not _refused_by_sandbox("cat missing.txt", "cat: missing.txt: No such file or directory")
    assert not _refused_by_sandbox("python3 build.py", "PermissionError: Operation not permitted: 'out/x'")
    monkeypatch.setattr(tools, "sandbox_mode", lambda: "none")
    assert not _refused_by_sandbox("cp ~/Downloads/app.apk .", "cp: Operation not permitted")


@pytest.mark.skipif(not MAC_SANDBOX, reason="needs the macOS sandbox (sandbox-exec)")
def test_a_real_refusal_gets_the_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "sandbox_mode", lambda: "macos")
    ws = Workspace(tmp_path / "ws")
    for command in ("ls ~/Downloads", f"ls {Path.home() / 'Downloads'}"):  # the cage's own HOME, then the real one
        out = tools.bash(ws, {"command": command})
        assert "exit code: 0" not in out and "[sandbox]" in out and "attaches them to the chat" in out, out
    plain = tools.bash(ws, {"command": "cat missing.txt"})
    assert "[sandbox]" not in plain


def test_the_hint_does_not_depend_on_the_exit_code(monkeypatch):
    # Seen live: `ls /Users/me/Downloads/ 2>&1 | grep apk; echo "exit $?"` ends 0 although ls was refused.
    monkeypatch.setattr(tools, "sandbox_mode", lambda: "macos")
    assert _refused_by_sandbox('ls -la /Users/me/Downloads/ 2>&1 | head -3; echo "---exit $?"',
                               "ls: /Users/me/Downloads/: Operation not permitted\n---exit 0")
    assert not _refused_by_sandbox("ls ~/.npm; echo done", "ls: /cage/home/.npm: No such file or directory\ndone")


def test_the_system_prompt_asks_not_to_get_past_bot_protection(tmp_path):
    assert loop.NO_BOT_EVASION in loop.build_system(base_harness(), Workspace(tmp_path))


@pytest.mark.skipif(not MAC_SANDBOX or not Path("/private/var/run/mDNSResponder").exists(),
                    reason="needs the macOS sandbox and mDNSResponder")
def test_caged_commands_can_resolve_host_names_but_reach_no_other_socket(tmp_path):
    # 0.4.0–0.4.2: mDNSResponder's socket was refused, so curl/pip/npm/git could not resolve any host name. Offline:
    # connect to that socket itself (what getaddrinfo does) and to a foreign one, which must stay refused.
    import socket
    import threading

    ws = Workspace(tmp_path / "ws")
    foreign = Path(tempfile.mkdtemp(prefix="nesi-sock-", dir="/tmp")) / "x.sock"
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(str(foreign))
    srv.listen(1)
    def serve():
        try:
            srv.accept()
        except OSError:  # (closed at the end of the test)
            pass

    threading.Thread(target=serve, daemon=True).start()
    probe = ("import socket\n"
             f"for p in ('/private/var/run/mDNSResponder', {str(foreign.resolve())!r}):\n"
             "    s = socket.socket(socket.AF_UNIX)\n"
             "    try:\n"
             "        s.connect(p); print(p, 'ok')\n"
             "    except OSError as e:\n"
             "        print(p, 'refused', e.errno)\n")
    (ws.root / "probe.py").write_text(probe, encoding="utf-8")
    out = tools.bash(ws, {"command": "python3 probe.py"})
    srv.close()
    assert "/private/var/run/mDNSResponder ok" in out, out
    assert f"{foreign.resolve()} refused" in out, out
