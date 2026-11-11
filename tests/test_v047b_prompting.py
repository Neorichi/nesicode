"""v0.4.7, second round, prompting: what live runs of nesi3 (131,072-token window, no step cap) got wrong that a line of
the code-level prompt can steer. A bug was explained to the user with an output the model had predicted ("-0,95 €")
instead of the one the tests printed ("-1,95 €"), and the replaced logic was checked only with the tests' own inputs; a
reasoning-only reply drafted a whole file until the output cap cut it, and the file was then written out again; a
`grep -r` from bash returned mostly the conversation's own archive (.nesicode/), which the search tool skips.

Offline: a fake provider that records each request's system prompt, temp workspaces."""

from __future__ import annotations

import os
import re

import pytest

from nesicode.agent import Agent, Workspace, loop, tools
from nesicode.agent.workspace import HARNESS_DIR, IGNORE_DIRS
from nesicode.harness import base_harness
from nesicode.llm.base import Conversation, Provider, ToolCall, Turn, Usage

posix_only = pytest.mark.skipif(os.name == "nt", reason="grep from the bash tool")


class _Conv(Conversation):
    def __init__(self, prov, model, system, tools, options):
        super().__init__(model, system, tools, options)
        self.prov = prov

    def add_user(self, text):
        pass

    def add_assistant_text(self, text):
        pass

    def add_tool_results(self, results):
        pass

    def step(self, on_event=None):
        role = self.options.role
        self.prov.requests.append({"role": role, "system": self.system})
        return self.prov.policy(role, sum(r["role"] == role for r in self.prov.requests))


class _Prov(Provider):
    name = "fake"

    def __init__(self, policy):
        self.policy, self.requests = policy, []

    def conversation(self, model, system, tools, options):
        return _Conv(self, model, system, tools, options)

    def system(self, role: str) -> str:
        return next(r["system"] for r in self.requests if r["role"] == role)


@pytest.fixture()
def fake(monkeypatch):
    def install(policy):
        prov = _Prov(policy)
        monkeypatch.setattr(loop, "get_provider", lambda spec, keys=None: (prov, "m"))
        return prov
    return install


def _section(system: str, head: str) -> str:
    start = system.index(head)
    nxt = system.find("\n# ", start + len(head))
    return system[start: nxt if nxt >= 0 else len(system)]


QUOTE_REAL = "quote outputs, values and errors as your tools really returned them, never as you predicted them"
OLD_LOGIC = "When you replace logic, list what the old code also handled (guards, special cases, input formats)"


def test_every_final_answer_quotes_real_outputs_and_an_uncapped_run_checks_what_replaced_logic_handled(tmp_path):
    ws = Workspace(tmp_path / "ws")
    capped = _section(loop.build_system(base_harness(), ws, max_steps=60), "# When your turn ends")
    uncapped = _section(loop.build_system(base_harness(), ws, max_steps=0), "# When your turn ends")
    for text in (capped, uncapped):  # (an answer forced by the step cap reports what was verified too)
        assert f"In your final answer, {QUOTE_REAL}." in text
    # with no step cap there is room to check more than the tests did, before the final verification
    assert OLD_LOGIC in uncapped and "check each with a quick run of your own, not only the tests" in uncapped
    assert uncapped.index(OLD_LOGIC) < uncapped.index("Before your final answer, verify the result")
    assert OLD_LOGIC not in capped and "no step limit" not in capped


def test_the_environment_says_not_to_draft_files_in_the_reasoning_and_that_nesicode_is_not_the_projects(tmp_path):
    ws = Workspace(tmp_path / "ws")
    for max_steps in (0, 60):
        env = _section(loop.build_system(base_harness(), ws, max_steps=max_steps), "# Environment")
        assert "Never draft whole files in your reasoning" in env and "each reply's output cap" in env
        assert "write them with write_file and refine them with edit_file" in env
        assert f"`{HARNESS_DIR}/` is Nesicode's own folder (conversation archives" in env
        assert "not part of the project: leave it out of project-wide searches and bulk edits from the shell" in env
        assert f"(e.g. `grep -r --exclude-dir={HARNESS_DIR}`)" in env and "the `search` tool already skips it" in env
        # next to the notes-file line, which says that file lives in this conversation's archive
        assert env.index("Your notes file is") < env.index(f"`{HARNESS_DIR}/` is Nesicode's own folder")
    # what the lines name is real: the agent's own tools, and the search tool's skipped folders
    assert {"write_file", "edit_file"} <= {spec.name for spec in Agent(model="mock/demo", harness=base_harness(),
                                                                       workspace=ws).tool_specs()}
    assert HARNESS_DIR in IGNORE_DIRS


def test_the_uncapped_run_and_its_subagent_get_the_lines_their_settings_call_for(tmp_path, fake):
    """The top-level run (no step cap, the user's setting) gets all of them; its subagent (25 steps) shares the
    workspace and writes files too, so it gets the environment lines and the final-answer rule, not the uncapped
    ones."""
    h = base_harness()
    h["sub_agents/fixer/AGENT.md"] = "---\nname: fixer\ndescription: arregla\n---\nArregla."

    def policy(role, n):
        if role == "executor" and n == 1:
            return Turn("", [ToolCall("d1", "delegate", {"agent": "fixer", "task": "arregla money.py"})], "tool_use",
                        Usage(1, 1))
        return Turn("fin", [], "end_turn", Usage(1, 1))
    prov = fake(policy)
    res = Agent(model="fake/m", harness=h, workspace=Workspace(tmp_path / "ws"), archive=".nesicode/chats/3",
                max_steps=0, context_limit=131_072).run("arregla el bug de format_eur")
    assert res.status == "done"
    top, sub = prov.system("executor"), prov.system("subagent")
    for system in (top, sub):
        env = _section(system, "# Environment")
        assert "Never draft whole files in your reasoning" in env and "--exclude-dir=.nesicode" in env
        assert QUOTE_REAL in _section(system, "# When your turn ends")
    assert OLD_LOGIC in top and OLD_LOGIC not in sub


@posix_only
def test_the_shell_search_the_prompt_advises_leaves_the_archive_out_as_the_search_tool_does(tmp_path):
    """Seen live: `grep -rn "8765" .` as a final check returned 12,958 chars, mostly lines of the conversation's own
    archived history. The prompt's example, run as given, finds the project's line and nothing of the archive."""
    ws = Workspace(tmp_path / "ws")
    (ws.root / "client.py").write_text("PORT = 8765\n")
    hist = ws.root / HARNESS_DIR / "chats" / "1" / "context"
    hist.mkdir(parents=True)
    (hist / "history.md").write_text("1. **Orion escucha en el puerto 8765**.\n[ACTION] write_file " + "x" * 4_000
                                     + " 8765\n")
    env = _section(loop.build_system(base_harness(), ws), "# Environment")
    advised = re.search(r"e\.g\. `(grep [^`]+)`", env)
    assert advised, env
    assert f"{HARNESS_DIR}/chats/1/context/history.md" in tools.bash(ws, {"command": "grep -rn 8765 ."})  # as live
    out = tools.bash(ws, {"command": f"{advised.group(1)} -n 8765 ."})
    assert out.startswith("exit code: 0") and "client.py:1:PORT = 8765" in out and HARNESS_DIR not in out, out
    found = tools.search(ws, {"pattern": "8765"})  # what the line says of the search tool
    assert "client.py" in found and HARNESS_DIR not in found, found
