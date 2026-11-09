"""When the sandbox keeps something out, the agent is told to ask the user for the file — not to hunt for a way in.

Seen live: an agent that needed an APK the user's Chrome had downloaded tried ~/Downloads («Operation not permitted»),
then relaunched Finder «cold», spending steps on a wall that is there on purpose."""

import shutil
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
