"""v0.4.7, group loop — the run loop with no step cap (max_steps = 0): empty replies, the failure breaker, loop
detection, and how a run asks for its final answer (and says why it ended: AgentResult.stop_reason).

Offline: a scripted OpenAI-format provider (its notes ride with the newest tool result, like OpenAICompatConversation),
the real OpenAI-compatible provider against a scripted SSE server on 127.0.0.1, temp workspaces."""

from __future__ import annotations

import json

import pytest
from test_p2_context import DONE, oc, quick_retries, server, sse  # noqa: F401

from nesicode.agent import Agent, ToolError, Workspace
from nesicode.agent import context as ctxm
from nesicode.agent import loop as loopmod
from nesicode.harness import base_harness
from nesicode.llm.base import Conversation, LLMError, Provider, ToolCall, Turn, Usage

CUT = "output was cut off at max_tokens; resend a smaller call"
SUMMARY = "## Goal\nSeguir.\n## Next Step\nContinuar."
INTERIM = "Voy a ejecutar los tests."


# ====================================================================== a scripted provider
class Conv(Conversation):
    def __init__(self, prov, model, system, tools, options):
        super().__init__(model, system, tools, options)
        self.prov = prov
        self.messages = [{"role": "system", "content": system}]

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def add_assistant_text(self, text):
        self.messages.append({"role": "assistant", "content": text or "(no answer)"})

    def add_tool_results(self, results):
        for r in results:
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": r.content or "(empty)"})

    def add_note(self, text):  # (as OpenAICompatConversation: with the newest tool result, else a user turn)
        if self.messages[-1]["role"] == "tool":
            self.messages[-1]["content"] += "\n\n" + text
        else:
            self.add_user(text)

    def step(self, on_event=None):
        inp = sum(len(json.dumps(m, ensure_ascii=False)) for m in self.messages) // 4
        rec = {"role": self.options.role, "input": inp, "messages": json.loads(json.dumps(self.messages))}
        self.prov.log.append(rec)
        if self.prov.window and inp + (self.options.max_tokens or 0) > self.prov.window:
            raise LLMError(f"HTTP 400: This model's maximum context length is {self.prov.window} tokens. However, you "
                           f"requested {inp + self.options.max_tokens} tokens ({inp} in the messages, ...).")
        turn = self.prov.policy(self, rec)
        turn.usage = Usage(inp, turn.usage.output_tokens or 20)
        msg = {"role": "assistant", "content": turn.text}
        if turn.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                 for c in turn.tool_calls]
        if turn.thinking:
            msg["reasoning_content"] = turn.thinking
        self.messages.append(msg)
        return turn


class Prov(Provider):
    name = "fake"

    def __init__(self, policy, window=None):
        self.policy, self.window, self.log = policy, window, []

    def conversation(self, model, system, tools, options):
        return Conv(self, model, system, tools, options)

    def requests(self, role="executor"):
        return [r for r in self.log if r["role"] == role]


@pytest.fixture()
def scripted(monkeypatch):
    def install(policy, window=None):
        prov = Prov(policy, window)
        monkeypatch.setattr(loopmod, "get_provider", lambda spec, keys=None: (prov, "m"))
        return prov
    return install


def agent(ws, events=None, **kw):
    kw.setdefault("context_limit", 131_072)
    kw.setdefault("output_limit", 16_384)
    kw.setdefault("max_steps", 0)
    return Agent(model="fake/m", harness=kw.pop("harness", None) or base_harness(), workspace=ws,
                 emit=(events.append if events is not None else None), **kw)


def call(cid, name, **args):
    return ToolCall(cid, name, args)


def nudged(rec):
    return "Do not call any more tools" in json.dumps(rec["messages"][-1])


def tool_results(events):
    return [e for e in events if e["type"] == "tool_result" and not e.get("depth")]


def _with_explorer():
    h = base_harness()
    h["sub_agents/explorer/AGENT.md"] = "---\nname: explorer\ndescription: Explores code\n---\nYou explore code.\n"
    return h


# ====================================================================== 1. empty replies (loop-1, llm-2)
@pytest.mark.parametrize("finish", ["end_turn", "tool_use"])  # reasoning only / a call the server could not parse
def test_an_empty_reply_is_asked_for_again_and_an_interim_line_is_never_the_answer(tmp_path, scripted, finish):
    def policy(conv, rec):
        n = len(prov.requests())
        if n == 1:
            return Turn(INTERIM, [call("c1", "list_dir", path=".")], "tool_use")
        if n == 2:
            return Turn("", [], finish, thinking="3 tests fallan en app.py:42")
        return Turn("Los 3 tests fallan en app.py:42.", [], "end_turn")
    prov = scripted(policy)
    events = []
    res = agent(Workspace(tmp_path / "ws"), events).run("ejecuta los tests")
    assert (res.status, res.final_text, res.stop_reason) == ("done", "Los 3 tests fallan en app.py:42.", "")
    third = prov.requests()[2]["messages"]
    # the empty reply was taken back (an empty assistant message is rejected by some APIs), and the note rides with the
    # tool result: no new user turn that would make interleaved-thinking templates drop this loop's reasoning
    assert third[-1]["role"] == "tool" and "came back empty" in third[-1]["content"]
    assert [m["content"] for m in third if m["role"] == "assistant"] == [INTERIM]
    assert [e["text"] for e in events if e["type"] == "thinking"] == ["3 tests fallan en app.py:42"]  # still archived


def test_replies_that_stay_empty_end_the_run_with_an_explicit_error(tmp_path, scripted):
    def policy(conv, rec):
        if len(prov.requests()) == 1:
            return Turn(INTERIM, [call("c1", "list_dir", path=".")], "tool_use")
        return Turn("", [], "end_turn", thinking="pienso")
    prov = scripted(policy)
    events = []
    res = agent(Workspace(tmp_path / "ws"), events).run("ejecuta los tests")
    assert (res.status, res.stop_reason, res.final_text) == ("error", "empty_replies", "")  # never the interim line
    assert not res.delivered and "respuestas vacías" in res.error and "continúa" in res.error
    assert len(prov.requests()) == 2 + loopmod.EMPTY_RETRIES
    assert [e["text"] for e in events if e["type"] == "error"] == [res.error]


def test_an_answer_given_with_a_last_tool_call_is_asked_for_again_after_an_empty_reply(tmp_path, scripted):
    def policy(conv, rec):
        n = len(prov.requests())
        if n == 1:
            return Turn("Informe listo en informe.md.", [call("c1", "write_file", path="informe.md", content="# I\n")],
                        "tool_use")
        if n == 2:
            return Turn("", [], "end_turn")  # (it thinks it has already answered)
        return Turn("Informe listo en informe.md.", [], "end_turn")
    prov = scripted(policy)
    res = agent(Workspace(tmp_path / "ws")).run("haz el informe")
    assert (res.status, res.final_text) == ("done", "Informe listo en informe.md.")
    assert "write your final answer for the user now" in prov.requests()[2]["messages"][-1]["content"]


def test_an_empty_reply_from_the_real_compatible_provider_keeps_the_loops_reasoning(tmp_path, server):
    usage = oc(usage={"prompt_tokens": 50, "completion_tokens": 5})
    first = sse(oc({"reasoning_content": "R1: leo el listado"}), oc({"content": INTERIM}),
                oc({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                    "function": {"name": "list_dir", "arguments": '{"path": "."}'}}]}),
                oc(finish="tool_calls"), usage, DONE)
    reasoning_only = sse(oc({"reasoning_content": "3 tests fallan en app.py:42; "}), oc({"reasoning_content": "lo digo"}),
                         oc(finish="stop"), usage, DONE)
    final = sse(oc({"content": "Los 3 tests fallan en app.py:42."}), oc(finish="stop"), usage, DONE)
    s = server(first, reasoning_only, final)
    a = Agent(model="nesilabs/nesi3", harness=base_harness(), workspace=Workspace(tmp_path / "ws"), max_steps=0,
              user_keys={"nesilabs": {"kind": "openai", "base_url": s.url, "api_key": "k"}})
    res = a.run("ejecuta los tests")
    assert (res.status, res.final_text) == ("done", "Los 3 tests fallan en app.py:42."), res.final_text
    assert len(s.requests) == 3
    msgs = s.requests[2]["messages"]
    replies = [m for m in msgs if m["role"] == "assistant"]
    assert len(replies) == 1 and replies[0]["reasoning_content"] == "R1: leo el listado"
    assert msgs[-1]["role"] == "tool" and "came back empty" in msgs[-1]["content"]


def test_a_subagent_that_only_returns_empty_replies_hands_the_parent_an_error_not_its_interim_line(tmp_path, scripted):
    def policy(conv, rec):
        if rec["role"] == "subagent":
            if len(prov.requests("subagent")) == 1:
                return Turn("Voy a leer src/app.py.", [call("s1", "list_dir", path=".")], "tool_use")
            return Turn("", [], "end_turn", thinking="hallazgos solo en el razonamiento")
        if any(m["role"] == "tool" for m in rec["messages"]):
            return Turn("Fin.", [], "end_turn")
        return Turn("Delego.", [call("d1", "delegate", agent="explorer", task="explora")], "tool_use")
    prov = scripted(policy)
    events = []
    res = agent(Workspace(tmp_path / "ws"), events, harness=_with_explorer()).run("explora")
    out = next(e for e in tool_results(events) if e["name"] == "delegate")
    assert res.status == "done"
    assert out["is_error"] and "empty replies" in out["output"] and "Voy a leer" not in out["output"], out


# ====================================================================== 2. the failure breaker (loop-2, tools-3, harness-1)
def test_unrelated_edit_misses_never_disable_edit_file(tmp_path):
    from nesicode.llm import Usage as U

    ws = Workspace(tmp_path / "ws")
    for f in "abcd":
        (ws.root / f"{f}.py").write_text(f"def {f}():\n\treturn '{f}'\n")
    a = agent(ws)
    for f in "abc":
        assert not a._call_tool(call(f"r{f}", "read_file", path=f"{f}.py"), U()).is_error
        assert a._call_tool(call(f"e{f}", "edit_file", path=f"{f}.py", old=f"    return '{f}'", new="x"), U()).is_error
    res = a._call_tool(call("ed", "edit_file", path="d.py", old="\treturn 'd'", new="\treturn 'D'"), U())
    assert not res.is_error, res.content
    assert (ws.root / "d.py").read_text() == "def d():\n\treturn 'D'\n"


def test_denials_hook_blocks_and_unreadable_arguments_never_disable_a_tool(tmp_path):
    from nesicode.llm import Usage as U

    h = dict(base_harness(), **{"plugins/hooks.json": json.dumps(
        {"pre_tool": [{"tool": "bash", "match": "^rm ", "action": "block", "message": "Blocked: no rm."}]})})
    asked, user = [], {"allows": False}

    def approve(c):
        asked.append(c.arguments.get("command") or c.name)
        return user["allows"], "denied"
    a = Agent(model="mock/demo", harness=h, workspace=Workspace(tmp_path / "ws"), approve=approve,
              extra_tools={"mcp__gh__get": (None, lambda args, ag: "ok " + str(args.get("q")))})
    for cmd in ("npm install", "git push", "curl https://x.test"):
        assert "did not allow" in a._call_tool(call(cmd, "bash", command=cmd), U()).content
    user["allows"] = True
    res = a._call_tool(call("b4", "bash", command="echo hola"), U())
    assert not res.is_error and "hola" in res.content, res.content
    assert asked == ["npm install", "git push", "curl https://x.test", "echo hola"]  # the user was asked every time
    user["allows"] = False
    for i in range(3):  # (a connector too: the user's decisions are not failures of its backend)
        assert "did not allow" in a._call_tool(call(f"g{i}", "mcp__gh__get", q="x"), U()).content
    user["allows"] = True
    assert a._call_tool(call("g4", "mcp__gh__get", q="x"), U()).content == "ok x"
    a.approve = None
    for i in range(3):
        assert "no rm" in a._call_tool(call(f"rm{i}", "bash", command=f"rm -rf build{i}"), U()).content
        bad = ToolCall(f"w{i}", "write_file", {}, parse_error="invalid JSON arguments: Invalid \\escape: line 1 column 58")
        assert a._call_tool(bad, U()).is_error
    assert not a._call_tool(call("w9", "write_file", path="x.txt", content="hola\n"), U()).is_error
    assert (a.ws.root / "x.txt").read_text() == "hola\n"


def test_a_connector_failing_different_calls_the_same_way_stays_on(tmp_path):
    from nesicode.llm import Usage as U

    def get_file(args, ag):
        if args["path"] != "src/app.py":
            raise ToolError("Not Found")
        return "print('hola')"
    a = agent(Workspace(tmp_path / "ws"), extra_tools={"mcp__gh__get_file": (None, get_file)})
    outs = [a._call_tool(call(f"c{i}", "mcp__gh__get_file", path=p), U()) for i, p in enumerate(("a.py", "lib/b.py", "c.md"))]
    assert all(o.is_error and "Not Found" in o.content for o in outs)
    assert a._call_tool(call("c4", "mcp__gh__get_file", path="src/app.py"), U()).content == "print('hola')"


def test_a_connector_failing_the_same_call_the_same_way_is_switched_off_with_guidance_to_carry_on(tmp_path):
    from nesicode.llm import Usage as U

    a = agent(Workspace(tmp_path / "ws"),
              extra_tools={"mcp__vt__report": (None, lambda args, ag: (_ for _ in ()).throw(ToolError("401: bad API key")))})
    outs = [a._call_tool(call(f"c{i}", "mcp__vt__report", hash="ab12"), U()) for i in range(4)]
    assert "disabled for this run" in outs[3].content and "401: bad API key" in outs[3].content
    assert "Stop retrying" not in outs[3].content and "carry on with your other tools" in outs[3].content


# ====================================================================== 3. loop detection with no step cap (loop-3, harness-2)
def test_the_same_call_with_the_same_result_is_named_then_the_run_stops_with_a_summary(tmp_path, scripted):
    def policy(conv, rec):
        if nudged(rec):
            return Turn("Resumen: leí app.py; no encontré el fallo.", [], "end_turn")
        n = len(prov.requests())
        if n > 60:  # (v0.4.6 never stopped it)
            return Turn("me rindo", [], "end_turn")
        return Turn("", [call(f"c{n}", "read_file", path="app.py")], "tool_use")
    prov = scripted(policy)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "app.py").write_text("print('hola')\n")
    events = []
    res = agent(ws, events).run("encuentra el fallo")
    outs = [e["output"] for e in tool_results(events)]
    assert len(outs) == loopmod.REPEAT_STOP_AT and res.steps == loopmod.REPEAT_STOP_AT + 1
    third = prov.requests()[3]["messages"][-1]
    assert third["role"] == "tool" and "this exact call" in third["content"] and "3 times" in third["content"]
    last = prov.requests()[-1]["messages"][-1]
    assert last["role"] == "tool" and "same `read_file` call 6 times" in last["content"]  # (harness-6: with the result)
    assert (res.status, res.stop_reason) == ("max_steps", "stuck")
    assert res.final_text.startswith("Resumen: leí app.py") and "«continúa»" in res.final_text
    assert [e for e in events if e["type"] == "stopped" and e["reason"] == "stuck"]
    assert next(e for e in events if e["type"] == "done")["stop_reason"] == "stuck"


def test_rerunning_the_tests_after_each_edit_is_not_a_repetition(tmp_path, scripted):
    def policy(conv, rec):
        n = len(prov.requests())
        if n > 24:
            return Turn("Sigue fallando; lo dejo así.", [], "end_turn")
        if n % 2:
            return Turn("", [call(f"e{n}", "edit_file", path="app.py", old=f"v = {n - 2}", new=f"v = {n}")], "tool_use")
        return Turn("", [call(f"t{n}", "bash", command="echo '1 failed'")], "tool_use")  # (each edit leaves it failing)
    prov = scripted(policy)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "app.py").write_text("v = -1\n")
    events = []
    res = agent(ws, events).run("arregla el test")
    assert (res.status, res.stop_reason, res.steps) == ("done", "", 25)
    assert not any("this exact call" in json.dumps(r["messages"]) for r in prov.requests())


def test_the_same_failing_test_rerun_with_nothing_changed_is_a_repetition_whatever_its_timing(tmp_path):
    from nesicode.llm import ToolResult

    a = agent(Workspace(tmp_path / "ws"))
    run = call("t", "bash", command="pytest -q")
    counts = [a._repeats(run, ToolResult("t", f"exit code: 1\n3 failed, 2 passed in {t}s")) for t in ("0.51", "0.62", "1.03")]
    assert counts == [1, 2, 3]
    assert a._repeats(run, ToolResult("t", "exit code: 1\n2 failed, 3 passed in 0.40s")) == 1  # a different result


@pytest.mark.parametrize("why", ["cut", "denied", "disabled", "unknown"])
def test_rounds_in_which_no_call_can_run_stop_the_run(tmp_path, scripted, why):
    def policy(conv, rec):
        if nudged(rec):
            return Turn("Resumen: no pude escribir el fichero.", [], "end_turn")
        n = len(prov.requests())
        if n > 40:
            return Turn("me rindo", [], "end_turn")
        if why == "cut":  # the same huge write, cut by the output cap every time
            return Turn("", [ToolCall(f"w{n}", "write_file", {}, parse_error=CUT)], "max_tokens", Usage(0, 16_384))
        if why == "denied":
            return Turn("", [call(f"b{n}", "bash", command=f"rm -rf build{n}")], "tool_use")
        if why == "unknown":  # a tool it does not have
            return Turn("", [call(f"p{n}", "python", code=f"print({n})")], "tool_use")
        return Turn("", [call(f"m{n}", "mcp__x__do", n=n)], "tool_use")
    prov = scripted(policy)
    tools = {"mcp__x__do": (None, lambda args, ag: (_ for _ in ()).throw(ToolError("backend down (request #1)")))}
    events = []
    res = agent(Workspace(tmp_path / "ws"), events, extra_tools=tools,
                approve=(lambda c: (False, "denied")) if why == "denied" else None).run("escribe index.html")
    assert (res.status, res.stop_reason) == ("max_steps", "stuck")
    assert res.final_text.startswith("Resumen: no pude escribir el fichero.")
    expected = loopmod.STUCK_ROUNDS + (3 if why == "disabled" else 0)  # (3 real failures before it is switched off)
    assert len(tool_results(events)) == expected


# ====================================================================== 4. asking for the final answer (harness-5, harness-6)
def test_a_step_cap_asks_for_the_final_answer_with_the_tool_results(tmp_path, scripted):
    def policy(conv, rec):
        if nudged(rec):
            return Turn("Resumen final.", [], "end_turn")
        n = len(prov.requests())
        return Turn("", [call(f"c{n}", "list_dir", path=f"d{n}")], "tool_use")
    prov = scripted(policy)
    ws = Workspace(tmp_path / "ws")
    for i in range(1, 4):
        (ws.root / f"d{i}").mkdir()
    res = agent(ws, max_steps=3).run("mira")
    last = prov.requests()[-1]["messages"]
    assert last[-1]["role"] == "tool" and "step budget" in last[-1]["content"]  # not a new user turn (harness-6)
    assert (res.status, res.stop_reason, res.final_text) == ("max_steps", "step_cap", "Resumen final.")


def test_a_tool_call_after_the_final_answer_request_is_answered_with_that_request_again(tmp_path, scripted):
    def policy(conv, rec):
        n = len(prov.requests())
        if n <= 2:
            return Turn(f"Miro d{n}.", [call(f"c{n}", "list_dir", path=".")], "tool_use")
        if n == 3:  # it ignores the request and calls one more tool (with no text at all)
            return Turn("", [call("c3", "list_dir", path=".")], "tool_use")
        return Turn("Resumen final: d1 y d2 vistos.", [], "end_turn")
    prov = scripted(policy)
    res = agent(Workspace(tmp_path / "ws"), max_steps=2).run("mira")
    assert (res.status, res.stop_reason, res.final_text) == ("max_steps", "step_cap", "Resumen final: d1 y d2 vistos.")
    last = prov.requests()[-1]["messages"]
    assert last[-1]["role"] == "tool" and "no more tools can run" in last[-1]["content"] and "step budget" in last[-1]["content"]
    assert [m.get("tool_calls") for m in last if m["role"] == "assistant"][-1][0]["id"] == "c2"  # the ignored reply is gone


def test_no_step_cap_never_speaks_of_a_step_budget_when_the_context_fills_up(tmp_path, scripted):
    state = {"refused": False}

    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return Turn(SUMMARY, [], "end_turn", Usage(0, 200))
        last = json.dumps(rec["messages"][-1])
        if "context window is full" in last and not state["refused"]:
            state["refused"] = True  # the provider rejects that request as over its window: compaction, then retry
            raise LLMError("HTTP 400: This model's maximum context length is 20000 tokens. However, you requested "
                           "21000 tokens (20000 in the messages, ...).")
        if "Do not call any more tools" in json.dumps(rec["messages"][-2:]):
            return Turn("Resumen: hice A y B; falta C.", [], "end_turn")
        n = len(prov.requests())
        return Turn(f"Notas largas {n}: " + "detalle " * 1_200, [call(f"c{n}", "bash", command=f"echo {n}")], "tool_use")
    prov = scripted(policy)
    a = agent(Workspace(tmp_path / "ws"), context_limit=20_000, output_limit=4_000)
    a._compactions = ctxm.MAX_COMPACTIONS  # no compaction left: the context fills up for good
    res = a.run("documenta el módulo")
    assert state["refused"]
    seen = [json.dumps(r["messages"]) for r in prov.requests()]
    assert not any("step budget" in s for s in seen)
    assert "context window is full" in json.dumps(prov.requests()[-1]["messages"][-1])  # re-sent after the compaction
    assert (res.status, res.stop_reason) == ("max_steps", "context_full")
    assert res.final_text.startswith("Resumen: hice A y B") and "«continúa»" in res.final_text


def test_a_subagent_stopped_for_repeating_itself_is_reported_as_such(tmp_path, scripted):
    def policy(conv, rec):
        if rec["role"] == "subagent":
            if nudged(rec):
                return Turn("Resumen parcial del explorador.", [], "end_turn")
            n = len(prov.requests("subagent"))
            return Turn("", [call(f"s{n}", "list_dir", path=".")], "tool_use")
        if any(m["role"] == "tool" for m in rec["messages"]):
            return Turn("Fin.", [], "end_turn")
        return Turn("Delego.", [call("d1", "delegate", agent="explorer", task="explora")], "tool_use")
    prov = scripted(policy)
    events = []
    agent(Workspace(tmp_path / "ws"), events, harness=_with_explorer()).run("explora")
    out = next(e for e in tool_results(events) if e["name"] == "delegate")
    assert not out["is_error"] and "repeating itself" in out["output"] and "Resumen parcial del explorador." in out["output"]
    assert "continúa" not in out["output"]  # (the note for the user is not the parent model's business)
