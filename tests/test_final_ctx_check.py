"""Final review, area ctx — checker's regression tests for what the fixes still left open.

Offline: temp workspaces and the test database; regex timings use inputs that took seconds before the fix."""

from __future__ import annotations

import json
import random
import re
import threading
import time

import pytest
from test_p2_context import _client, _run

from nesicode.agent import Agent, Workspace
from nesicode.agent import context as ctxm
from nesicode.agent.hooks import Hooks
from nesicode.harness import base_harness
from nesicode.harness.base import LEGACY_RM_MATCHES, RM_MATCH

ARCHIVE = ".nesicode/chats/1"


# ====================================================================== security#2: the base rm rule, really linear
@pytest.mark.parametrize("word", ['"rm', "'rm", "(rm", "{rm", "`rm", 'x"rm', "x(rm", "rm"])
def test_the_base_rm_rule_stays_linear_when_rm_follows_a_quote_or_a_bracket(word):
    """A match may start at an `rm` right after a quote, `(`, `{` or a backtick: the argument scan must not cross
    those either (`echo "rm "rm …` of 20 KB took 1.7-3.4 s, quadratic, with the GIL held)."""
    hooks = Hooks(json.loads(base_harness()["plugins/hooks.json"]))
    cmd = "echo " + f"{word} " * 6_000
    t0 = time.perf_counter()
    assert hooks.before("bash", {"command": cmd})[0] is None
    assert time.perf_counter() - t0 < 0.5, word


def test_the_base_rm_rule_keeps_the_verdicts_of_the_earlier_one():
    hooks = Hooks(json.loads(base_harness()["plugins/hooks.json"]))
    for cmd in ['rm -rf "rm" /', "x=$(rm -rf /)", '(rm -rf "$HOME")', 'echo "rm "rm; rm -rf ~/', "a(rm -rf ~"]:
        assert hooks.before("bash", {"command": cmd})[0], cmd
    for cmd in ['echo "rm "rm "rm', 'rm -rf "(rm" build/', "rm -rf x(rm build", "echo {rm ~x"]:
        assert hooks.before("bash", {"command": cmd})[0] is None, cmd
    # the same verdict as the rule it replaces, on random commands made of the pieces that matter
    old, new = re.compile(LEGACY_RM_MATCHES[1]), re.compile(RM_MATCH)
    toks = ["rm", "rm", "-rf", "-f", "--", "/", "/*", "~", "~/", "$HOME", "${HOME}/*", "x", "build/", "rmdir", '"/"',
            '"rm', "'rm", "(rm", "{rm", "`rm", 'x"rm', "x(rm", ";", "&&", "|", "(", ")", "`", "\\", "\n", '"', "'", "$("]
    rnd = random.Random(5)
    for _ in range(5_000):
        cmd = "".join(rnd.choice(toks) + rnd.choice([" ", " ", "\t", ""]) for _ in range(rnd.randint(1, 9)))
        for text in (cmd, json.dumps({"command": cmd})):
            assert bool(new.search(text)) == bool(old.search(text)), text


def test_a_harness_stored_with_the_previous_spelling_of_the_rule_runs_the_current_one(tmp_path):
    h = base_harness()
    cfg = json.loads(h["plugins/hooks.json"])
    cfg["pre_tool"][0]["match"] = LEGACY_RM_MATCHES[-1]  # (the spelling that still went quadratic on `"rm` words)
    h["plugins/hooks.json"] = json.dumps(cfg)
    a = Agent(model="mock/demo", harness=h, workspace=Workspace(tmp_path / "ws"))
    assert a.hooks.pre[0]["match"] == RM_MATCH
    t0 = time.perf_counter()
    assert a.hooks.before("bash", {"command": "echo " + '"rm ' * 6_000})[0] is None
    assert time.perf_counter() - t0 < 0.5


# ====================================================================== ctx#3: history.md is searched whole
def test_history_md_stays_within_what_search_history_reads_with_multibyte_text(tmp_path, monkeypatch):
    """The cap of history.md is in bytes, like the read limit of search_history: the newest turns are at the end of
    the file, so a file over that limit hid exactly them (CJK text is 3 bytes a char). Scaled down: limits / 500."""
    from nesicode.server import jobs
    from nesicode.server.app import db

    monkeypatch.setattr(ctxm, "ARCHIVE_READ_MAX", 100_000)
    monkeypatch.setattr(jobs, "HISTORY_MAX_BYTES", 60_000)
    monkeypatch.setattr(jobs, "HISTORY_CLIPPED_CHARS", 300)
    _, p, t = _client()
    now = time.time()
    for i in range(1, 41):
        _run(p, t, f"mensaje {i}", "done", f"答复{i}: " + "漢字" * 1_000 + f" MARCA-{i}-FIN", at=now - 1000 + i)
    cur = _run(p, t, "sigue", "running", at=now)
    ws = Workspace(tmp_path / "ws")
    archive = f".nesicode/chats/{t['id']}"
    jobs._write_thread_history(db, ws, cur, archive)
    raw = (ws.root / archive / "context" / "history.md").read_bytes()
    assert len(raw) <= ctxm.ARCHIVE_READ_MAX, len(raw)
    assert raw.decode().count("## Turno #") == 40  # every turn is still there (the oldest ones clipped)
    for needle in ("MARCA-40-FIN", "MARCA-39-FIN", "MARCA-1-FIN", "MARCA-5-FIN"):
        assert needle in ctxm.search_history(ws, {"query": needle}, archive), needle


# ====================================================================== ctx#1: Stop between and inside long lines
@pytest.mark.parametrize("shape", ["many long lines", "one huge line"])
def test_stop_ends_search_history_soon_on_long_lines(tmp_path, shape):
    """`(.*)token` is allowed (no nested repeat) and costs ~50 ms on a 4,000-char line: Stop was only looked at every
    256 lines (~12 s) and never inside the windows of one long line."""
    rnd = random.Random(1)
    words = "the quick admin panel login users settings export report invoice order product".split()
    ws = Workspace(tmp_path / "ws")
    out = ws.root / ARCHIVE / "outputs"
    out.mkdir(parents=True)
    if shape == "many long lines":
        text = "\n".join(" ".join(rnd.choice(words) for _ in range(800))[:3_900] for _ in range(300))
    else:
        text = " ".join(rnd.choice(words) for _ in range(80_000))[:400_000]
    (out / "1-bash-c1.txt").write_text(text + "\n")
    ws.cancel = threading.Event()
    threading.Timer(0.3, ws.cancel.set).start()
    t0 = time.monotonic()
    res = ctxm.search_history(ws, {"query": r"(.*)token"}, ARCHIVE)
    took = time.monotonic() - t0
    assert took < 2.0, f"search_history returned {took:.1f} s after it started (Stop at 0.3 s)"
    assert "stopped by the user" in res, res[-200:]
