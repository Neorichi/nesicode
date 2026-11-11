"""v0.4.7, round 3: what the judges of the live runs still found (offline)."""

import json

import pytest

from nesicode.agent import Agent, Workspace
from nesicode.agent import loop as loopmod
from nesicode.agent import tools
from nesicode.harness import base_harness
from nesicode.llm import transport
from nesicode.llm.openai_compat import CUT, OpenAICompatProvider
from nesicode.server import jobs

from test_fix_llm import DONE, chunk, completion, oa_conv, respond, stream
from test_fix_llm import srv  # noqa: F401  (a fixture)


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(transport, "BACKOFF_BASE", 0.01)


def _call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_a_cut_reply_never_runs_its_last_call_even_when_the_server_closed_its_json(srv, tmp_path, monkeypatch):
    """The live case (multiturn T3): nesi3's reply stopped at the 16,384-token cap inside a write_file, and the
    endpoint returned the cut arguments as a closed JSON object. With `path` first, the harness would have written a
    truncated file and reported «created»: the cut call is answered with the cut explanation instead, never run."""
    cut = {"index": 0, **_call("call_w", "write_file", {"path": "tools/admin_client.py",
                                                        "content": "class Client:\n    def list_users(self, limit: int"})}
    s = srv(stream(chunk({"tool_calls": [cut]}), chunk(finish="length"),
                   chunk(choices=False, usage={"prompt_tokens": 900, "completion_tokens": 16_384}), DONE),
            stream(chunk({"content": "Hecho."}), chunk(finish="stop"),
                   chunk(choices=False, usage={"prompt_tokens": 950, "completion_tokens": 3}), DONE))
    monkeypatch.setattr(loopmod, "get_provider",
                        lambda spec, keys=None: (OpenAICompatProvider("nesilabs", s.url, "sk-test"), "nesi3"))
    ws = Workspace(tmp_path / "ws")
    events = []
    Agent(model="x/nesi3", harness=base_harness(), workspace=ws, emit=events.append, max_steps=0,
          context_limit=131_072, output_limit=16_384).run("escribe el cliente")
    assert not (ws.root / "tools" / "admin_client.py").exists()
    result = next(e for e in events if e.get("type") == "tool_result" and e.get("id", "").endswith("call_w")
                  or e.get("type") == "tool_result" and e.get("name") == "write_file")
    assert result["is_error"] and "cut off" in result["output"], result


def test_only_the_last_call_of_a_cut_reply_is_cut(srv):
    whole = _call("c1", "read_file", {"path": "a.py"})
    last = _call("c2", "write_file", {"path": "b.py", "content": "def f(:"})
    s = srv(respond(200, completion("", tool_calls=[whole, last], finish="length", usage=(900, 1_000))))
    _, conv = oa_conv(s)
    turn = conv.step()
    assert [c.parse_error for c in turn.tool_calls] == ["", CUT] and turn.stop_reason == "max_tokens"


def test_edit_file_takes_more_spellings_and_a_number(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "client.py").write_text("PORT = 8765\n", encoding="utf-8")
    tools.edit_file(ws, {"path": "client.py", "old_text": 8765, "new_text": 9876})
    assert (ws.root / "client.py").read_text(encoding="utf-8") == "PORT = 9876\n"
    with pytest.raises(tools.ToolError, match="unknown parameter"):
        tools.edit_file(ws, {"path": "client.py", "old": "9876", "replacement": "1"})


def test_bash_says_when_a_stage_of_a_pipeline_failed(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "ws")
    monkeypatch.setattr(ws, "run", lambda command, timeout=120: (0, "FAILED (failures=1)\n[pipeline exit codes: 1 0]\n"))
    out = tools.bash(ws, {"command": "python3 -m unittest 2>&1 | tail -3"})
    assert out.startswith("exit code: 0 (but a stage of the pipeline failed: exit codes 1 0)\n"), out
    monkeypatch.setattr(ws, "run", lambda command, timeout=120: (0, "OK\n"))
    assert tools.bash(ws, {"command": "echo ok | cat"}).startswith("exit code: 0\n")


def test_the_bash_spec_says_where_a_later_kill_is_refused():
    assert "a `kill` from a later command is always refused" in tools.SCHEMAS["bash"].description


def test_a_long_conversation_is_not_cut_to_200_turns_before_the_budget_decides():
    assert jobs.HISTORY_TURNS >= 1_000
