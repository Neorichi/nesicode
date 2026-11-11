"""v0.4.7: the places where the fix groups' work joins (offline)."""

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from nesicode.agent import Workspace
from nesicode.agent import context as ctxm
from nesicode.agent import tools

WEB = Path(__file__).resolve().parents[1] / "web" / "src"


def test_search_history_never_freezes_the_server_on_a_pathological_regex(tmp_path, monkeypatch):
    """search_history runs its regex through the search tool's matcher: a pattern it cannot prove harmless is matched
    in a child process killed at the deadline. In this process, `(a|aa)+$` on 42 a's and a b backtracks for minutes
    inside one re call, with the GIL held, so the per-line deadline checks never get a chance to run."""
    monkeypatch.setattr(ctxm, "HISTORY_SEARCH_BUDGET", 2.0)
    ws = Workspace(tmp_path / "ws")
    d = ws.root / ".nesicode" / "context"
    d.mkdir(parents=True)
    (d / "history.md").write_text("a" * 42 + "b\nuna línea normal\n", encoding="utf-8")
    t = time.monotonic()
    out = ctxm.search_history(ws, {"query": "(a|aa)+$"})
    assert time.monotonic() - t < 12, out
    assert "stopped" in out or "no match" in out.lower() or "plain text" in out


def test_a_kill_the_cage_refuses_points_to_stop_process(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "ws")
    refused = "bash: line 1: kill: (4242) - Operation not permitted\n"
    monkeypatch.setattr(ws, "run", lambda command, timeout=120: (1, refused))
    assert "stop_process" in tools.bash(ws, {"command": "kill 4242"})
    assert "stop_process" in tools.bash(ws, {"command": "sleep 1; pkill -f 'http.server'"})
    assert "stop_process" not in tools.bash(ws, {"command": "cat /etc/master.passwd"})  # another refusal: no hint


def test_the_tool_specs_carry_the_line_prefix_and_the_read_before_write_rules():
    """In the tools' own descriptions (always current): the shipped tool_guidance.md is copied into every project and
    upgraded by exact text, so the rules live where no migration is needed."""
    assert "without read_file's line-number prefix" in tools.SCHEMAS["edit_file"].description
    assert "must have been read in this run first" in tools.SCHEMAS["write_file"].description


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_git_state_warns_that_earlier_changes_may_be_the_users(tmp_path):
    ws = Workspace(tmp_path / "ws")
    subprocess.run(["git", "init", "-q"], cwd=ws.root, check=True)
    (ws.root / "notas.txt").write_text("trabajo del usuario\n", encoding="utf-8")
    out = ctxm.git_state(ws)
    assert out.startswith("## Git state (fresh;") and "never revert or discard those" in out and "notas.txt" in out


def test_stop_process_has_a_label_and_an_english_entry():
    assert "stop_process: t('Parar proceso')" in (WEB / "components" / "Transcript.tsx").read_text(encoding="utf-8")
    i18n = (WEB / "i18n.ts").read_text(encoding="utf-8")
    assert "'Parar proceso': 'Stop process'" in i18n
    assert "'Nesicode: regla de seguridad (no descartar cambios sin confirmar ni hacer commits que no se pidan)'" in i18n
