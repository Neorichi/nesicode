"""Group A regressions — the context engine (compaction, masking, continuity). Offline: a fake OpenAI-format
provider that rejects over-window requests the way vLLM does, temp workspaces, the test database."""

from __future__ import annotations

import json
import re
import threading
import time
from types import SimpleNamespace

import pytest

from nesicode.agent import Agent, Workspace
from nesicode.agent import context as ctxm
from nesicode.harness import base_harness
from nesicode.llm import ModelOptions
from nesicode.llm.base import Conversation, LLMError, Provider, ToolCall, ToolResult, ToolSpec, Turn, Usage

RULE = "NUNCA toques config/prod.yml"
SUMMARY = "## Goal\nKeep going.\n## Constraints & Preferences\n- (see the request)\n## Next Step\nContinue."


# ------------------------------------------------------------------ fake provider
class FakeConv(Conversation):
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
            self.messages.append({"role": "tool", "tool_call_id": r.call_id,
                                  "content": ("ERROR: " if r.is_error else "") + (r.content or "(empty)")})

    def mask_tool_results(self, keep_last, marker, *, keep_chars=0, apply=True):
        idx = [i for i, m in enumerate(self.messages) if m["role"] == "tool"]
        last = max((i for i, m in enumerate(self.messages) if m["role"] == "assistant"), default=-1)
        freed = 0
        for k in self.maskable([len(self.messages[i]["content"]) for i in idx], sum(1 for i in idx if i > last),
                               keep_last, keep_chars):
            m = self.messages[idx[k]]
            new = marker(m["tool_call_id"], m["content"])
            if new is not None and len(new) < len(m["content"]):
                freed += len(m["content"]) - len(new)
                if apply:
                    m["content"] = new
        return freed

    def ntokens(self):
        chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in self.messages)
        chars += sum(len(json.dumps({"name": t.name, "description": t.description, "parameters": t.parameters}))
                     for t in self.tools or [] if t is not None)
        return int(chars / self.prov.cpt)

    def step(self, on_event=None):
        inp = self.ntokens()
        rec = {"role": self.options.role, "input": inp, "max_tokens": self.options.max_tokens, "rejected": False,
               "messages": json.loads(json.dumps(self.messages, default=str)), "options": self.options, "conv": self}
        self.prov.log.append(rec)
        if self.prov.window and inp + (self.options.max_tokens or 0) > self.prov.window:
            rec["rejected"] = True
            raise LLMError(f"compat request failed: HTTP 400: This model's maximum context length is {self.prov.window} "
                           f"tokens. However, you requested {inp + self.options.max_tokens} tokens.")
        turn = self.prov.policy(self, rec)
        turn.usage = Usage(inp, turn.usage.output_tokens or 40)
        msg = {"role": "assistant", "content": turn.text}
        if turn.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                 for c in turn.tool_calls]
        self.messages.append(msg)
        return turn


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, policy, window=None, cpt=4.0):
        self.policy, self.window, self.cpt, self.log = policy, window, cpt, []

    def conversation(self, model, system, tools, options):
        return FakeConv(self, model, system, tools, options)

    def executor(self, accepted=True):
        return [r for r in self.log if r["role"] == "executor" and (not accepted or not r["rejected"])]


@pytest.fixture()
def fake(monkeypatch):
    import nesicode.agent.loop as loop

    def install(policy, window=None, cpt=4.0):
        prov = FakeProvider(policy, window, cpt)
        monkeypatch.setattr(loop, "get_provider", lambda spec, keys=None: (prov, "m"))
        return prov
    return install


def summarizer(rec, text=SUMMARY):
    return Turn(text, [], "end_turn", Usage(0, 200))


def _has(rec, needle):
    return any(needle in (m.get("content") or "") for m in rec["messages"])


def _post(prov):
    return [r for r in prov.executor() if _has(r, ctxm.SUMMARY_PREFIX)]


def _agent(ws, events, **kw):
    kw.setdefault("output_limit", 16_000)
    return Agent(model="fake/m", harness=base_harness(), workspace=ws, emit=events.append, **kw)


# ------------------------------------------------------------------ 1. earlier turns survive compaction
def test_earlier_turn_rule_survives_threshold_compaction(tmp_path, fake):
    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return summarizer(rec)
        if _has(rec, ctxm.SUMMARY_PREFIX):
            return Turn("Listo.", [], "end_turn", Usage(0, 10))
        n = len(prov.executor())
        return Turn("escribo", [ToolCall(f"w{n}", "write_file", {"path": f"p{n}.txt", "content": "y" * 12_000})],
                    "tool_use", Usage(0, 3000))
    prov = fake(policy)
    events = []
    history = [(f"Regla fija para toda la conversación: {RULE} y trabaja en la rama release-7.", "Entendido."),
               ("haz el paso 1", "Paso 1 hecho.")]
    res = _agent(Workspace(tmp_path / "ws"), events, context_limit=64_000).run("sigue", history)
    assert res.status == "done"
    assert [e for e in events if e["type"] == "compacted" and e["reason"] == "threshold"]
    summ = [r for r in prov.log if r["role"] == "summarizer"]
    assert summ and RULE in summ[0]["messages"][-1]["content"]  # the summarizer sees the earlier turns
    assert "# EARLIER TURNS OF THIS CONVERSATION" in summ[0]["messages"][-1]["content"]
    assert "Pending User Requests" in summ[0]["messages"][-1]["content"] and "TODAY'S DATE" in summ[0]["messages"][-1]["content"]
    post = _post(prov)
    assert post and _has(post[0], RULE)  # ...and so does the model after the compaction
    roles = [m["role"] for m in post[0]["messages"]]
    assert roles[:5] == ["system", "user", "assistant", "user", "assistant"] and roles[-1] == "user"
    assert "earlier turns of this conversation" in post[0]["messages"][0]["content"].lower()


def test_earlier_turns_survive_the_learned_limit_path(tmp_path, fake):
    prov = fake(lambda conv, rec: summarizer(rec) if rec["role"] == "summarizer" else Turn("A | B", [], "end_turn", Usage(0, 20)),
                window=16_384)
    events, learned = [], []
    history = [(f"Regla: {RULE}.", "Entendido."), ("dame la respuesta A", "Respuesta A: " + "alfa " * 3000),
               ("y la B", "Respuesta B: " + "beta " * 3000)]
    a = _agent(Workspace(tmp_path / "ws"), events, context_limit=128_000, on_limit=learned.append)
    res = a.run("resume ambos en una tabla", history)
    assert res.status == "done" and learned == [16_384] and a.context_limit == 16_384
    assert len(prov.executor(accepted=False)) - len(prov.executor()) == 1  # one rejection, then it fits
    ok = prov.executor()[0]
    assert _has(ok, RULE) and _has(ok, "Respuesta A") and _has(ok, "Respuesta B")
    assert ok["input"] + ok["max_tokens"] <= 16_384
    assert any(e["type"] == "compacted" and e["reason"] == "context exceeded" for e in events)


# ------------------------------------------------------------------ 2-3. small windows, real accounting, no thrash
@pytest.mark.parametrize("window,cpt", [(16_384, 4.0), (16_384, 3.3), (32_768, 4.0), (32_768, 3.3)])
def test_small_windows_complete_without_errors_or_thrash(tmp_path, fake, window, cpt):
    import random

    rnd = random.Random(7)

    def code(n):
        lines = []
        while sum(len(x) + 1 for x in lines) < n:
            lines.append(f"    def m_{rnd.randint(0, 99999)}(self, v: int) -> int:  # case {rnd.randint(0, 999)}")
            lines.append(f"        return self.cache.get(v, {rnd.randint(0, 9999)}) + f(v * {rnd.randint(1, 99)})")
        return "\n".join(lines)

    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return summarizer(rec)
        n = len(prov.executor())
        if n >= 16:
            return Turn("Hecho.", [], "end_turn", Usage(0, 10))
        if n % 5 == 4:
            return Turn("Escribo.", [ToolCall(f"w{n}", "write_file", {"path": f"out/r{n}.py", "content": code(3_000)})],
                        "tool_use", Usage(0, 900))
        return Turn("Leo.", [ToolCall(f"r{n}", "read_file", {"path": f"src/mod{n % 5}.py"})], "tool_use", Usage(0, 60))
    prov = fake(policy, window=window, cpt=cpt)
    ws = Workspace(tmp_path / "ws")
    for i in range(5):
        (ws.root / "src").mkdir(exist_ok=True)
        (ws.root / "src" / f"mod{i}.py").write_text(code(9_000))
    events = []
    history = [(f"Regla: {RULE}.", "Entendido. " + "detalle " * 300)]
    res = _agent(ws, events, context_limit=window, max_steps=60).run("refactoriza src/", history)
    assert res.status == "done", res.error
    assert not [r for r in prov.log if r["rejected"]]  # never over the window because of our own payload
    comp = [e for e in events if e["type"] == "compacted"]
    assert len(comp) <= 7  # not after every step (16 steps)
    assert all(e["after"] <= window * 0.5 for e in comp)
    assert all(_has(r, RULE) for r in _post(prov))
    ctx = [e for e in events if e["type"] == "context"]
    assert ctx[0]["reason"] == "new_message" and sum(e.get("reason") == "compacted" for e in ctx) == len(comp)


def test_first_request_is_checked_before_sending(tmp_path, fake):
    prov = fake(lambda conv, rec: Turn("visto", [], "end_turn", Usage(0, 10)), window=32_768)
    ws = Workspace(tmp_path / "ws")
    events = []
    huge = "analiza esto:\n" + "palabra " * 12_000 + "\nFIN-DEL-TEXTO"
    res = _agent(ws, events, context_limit=32_768).run(huge)
    assert res.status == "done" and not [r for r in prov.log if r["rejected"]]
    sent = prov.executor()[0]["messages"][-1]["content"]
    assert "FIN-DEL-TEXTO" in sent and "the FULL message is saved in" in sent
    rel = re.search(r"saved in (\S+request\.md)", sent).group(1)
    assert (ws.root / rel).read_text() == huge  # nothing lost
    first = next(e for e in events if e["type"] == "context")
    assert first["used"] > ctxm.tools_chars(Agent(model="fake/m", harness=base_harness(), workspace=ws).tool_specs()) // 4


def test_hopeless_context_stops_compacting_and_ends_cleanly(tmp_path, fake):
    # 25 huge connector schemas fill most of the window: compaction cannot help, so it stops trying
    tools = {f"tool{i}": (ToolSpec(f"tool{i}", "x" * 3_000, {"type": "object", "properties": {}}), lambda a, g: "ok")
             for i in range(25)}

    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return summarizer(rec)
        return Turn("sigo", [ToolCall(f"b{len(prov.log)}", "bash", {"command": "echo " + "q" * 2000})], "tool_use", Usage(0, 40))
    prov = fake(policy, window=32_768)
    events = []
    res = _agent(Workspace(tmp_path / "ws"), events, context_limit=32_768, extra_tools=tools, max_steps=60).run("tarea")
    assert res.status == "max_steps" and res.steps < 30
    assert len([e for e in events if e["type"] == "compacted"]) <= 3
    assert "context window is full" in prov.log[-1]["messages"][-1]["content"]
    assert not [r for r in prov.log if r["rejected"]]


# ------------------------------------------------------------------ 4. truncated replies
def test_output_cap_cut_call_is_explained_without_compaction(tmp_path, fake):
    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return summarizer(rec)
        n = len(prov.executor())
        if n <= 3:  # the model insists on one huge write: cut every time
            return Turn("", [ToolCall(f"w{n}", "write_file", {}, parse_error="invalid JSON arguments: Unterminated string")],
                        "max_tokens", Usage(0, rec["max_tokens"]))
        if n == 4:
            return Turn("", [ToolCall("ok", "write_file", {"path": "index.html", "content": "<p>1</p>"})], "tool_use", Usage(0, 50))
        return Turn("Hecho en partes.", [], "end_turn", Usage(0, 20))
    prov = fake(policy, window=200_000)
    events = []
    a = _agent(Workspace(tmp_path / "ws"), events, context_limit=200_000, output_limit=64_000)
    res = a.run("crea una landing en index.html")
    assert res.status == "done"
    assert not [e for e in events if e["type"] == "compacted"]
    assert not [r for r in prov.log if r["role"] == "summarizer"]
    outs = [e["output"] for e in events if e["type"] == "tool_result"]
    assert "cut off by the output limit (16,384 tokens per reply)" in outs[0] and "write_file the first part" in outs[0]
    assert "reply number 3 in a row" in outs[2] and "16,384" in outs[2]
    assert "write_file" not in a._fail_streak  # cut calls are not tool failures (no 3-strikes disable)
    assert outs[-1].startswith("created index.html")


def test_cut_text_reply_is_continued_not_done(tmp_path, fake):
    parts = ["La primera parte del informe", " y la segunda", " y el final."]

    def policy(conv, rec):
        n = len(prov.executor()) - 1
        return Turn(parts[n], [], "max_tokens" if n < 2 else "end_turn", Usage(0, rec["max_tokens"]))
    prov = fake(policy, window=200_000)
    events = []
    res = _agent(Workspace(tmp_path / "ws"), events, context_limit=200_000).run("escribe el informe")
    assert res.status == "done" and res.final_text == "".join(parts)
    note = prov.executor()[1]["messages"][-1]["content"]
    assert "cut off by the output limit" in note and "no recap" in note


def test_endlessly_cut_text_reply_is_not_reported_as_done(tmp_path, fake):
    prov = fake(lambda conv, rec: Turn("bla ", [], "max_tokens", Usage(0, rec["max_tokens"])), window=200_000)
    res = _agent(Workspace(tmp_path / "ws"), [], context_limit=200_000).run("escribe")
    assert res.status == "error" and res.final_text == "bla " * 4 and len(prov.executor()) == 4


def test_cut_reply_with_no_room_compacts_once_then_explains(tmp_path, fake):
    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return summarizer(rec)
        if any(e["type"] == "tool_result" and "created" in e["output"] for e in events):
            return Turn("fin", [], "end_turn", Usage(0, 10))
        if rec["max_tokens"] < 12_000:  # the reply needs more than the room left: cut
            return Turn("", [ToolCall(f"w{len(prov.log)}", "write_file", {}, parse_error="invalid JSON arguments")],
                        "max_tokens", Usage(0, rec["max_tokens"]))
        return Turn("", [ToolCall("ok", "write_file", {"path": "a.txt", "content": "z"})], "tool_use", Usage(0, 50))
    prov = fake(policy, window=32_768)
    events = []
    history = [("contexto previo", "x" * 50_000)]
    res = _agent(Workspace(tmp_path / "ws"), events, context_limit=32_768).run("escribe a.txt", history)
    reasons = [e["reason"] for e in events if e["type"] == "compacted"]
    assert reasons.count("truncated reply") <= 1 and res.status in ("done", "max_steps")
    assert not [r for r in prov.log if r["rejected"]]


# ------------------------------------------------------------------ 5. summarizer
def test_stop_during_compaction_returns_at_once(tmp_path, fake):
    started = threading.Event()

    def policy(conv, rec):
        if rec["role"] == "summarizer":
            assert rec["options"].cancel is not None and getattr(rec["options"], "max_attempts", None) == 2
            started.set()
            while not rec["options"].cancel.is_set():  # a slow summary that honours cancel
                time.sleep(0.01)
            raise LLMError("cancelled by the user")
        return Turn("leo", [ToolCall(f"r{len(prov.log)}", "bash", {"command": "echo hola"})], "tool_use", Usage(0, 30))
    prov = fake(policy)
    events = []
    a = _agent(Workspace(tmp_path / "ws"), events, context_limit=4_000, output_limit=1_000)
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("res", a.run("tarea " + "y" * 9_000)), daemon=True)
    th.start()
    assert started.wait(10)
    t0 = time.time()
    a.cancel.set()
    th.join(5)
    assert time.time() - t0 < 1.0 and box["res"].status == "cancelled"
    assert not [e for e in events if e["type"] == "compacted"]  # no fallback checkpoint after a Stop


@pytest.mark.parametrize("bad", [
    Turn("<think>\nOkay, the user wants a summary. Let me go through", [], "max_tokens", Usage(0, 8000)),
    Turn("## Goal\nPort the config.\n## Completed Actions\n1. READ", [], "max_tokens", Usage(0, 8000)),
    Turn("", [], "end_turn", Usage(0, 10)),
])
def test_bad_summaries_are_retried_then_a_bounded_fallback_is_used(tmp_path, fake, bad):
    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return Turn(bad.text, [], bad.stop_reason, Usage(0, 100))
        n = len(prov.executor())
        if n >= 14:
            return Turn("fin", [], "end_turn", Usage(0, 10))
        return Turn(f"Valor confirmado: build_salt = q7Rk-55Lw-91Zt (paso {n})",
                    [ToolCall(f"c{n}", "bash", {"command": "echo " + "v" * 3_000})], "tool_use", Usage(0, 40))
    prov = fake(policy)
    events = []
    res = _agent(Workspace(tmp_path / "ws"), events, context_limit=8_000, output_limit=4_000).run("tarea")
    assert res.status == "done"
    comp = [e for e in events if e["type"] == "compacted"]
    assert len(comp) >= 2 and all(e.get("fallback") for e in comp)
    summ = [r for r in prov.log if r["role"] == "summarizer"]
    assert len(summ) == 2 * len(comp)  # each compaction: one try + one retry with a smaller transcript
    assert all("<think>" not in e["summary"] for e in comp)
    assert "build_salt = q7Rk-55Lw-91Zt" in comp[-1]["summary"]  # what the agent stated survives, verbatim
    sizes = [len(e["summary"]) for e in comp]
    assert max(sizes) < 8_000 * 4 * 0.1  # bounded: no snowball on repeated failures


def test_summarizer_sees_outputs_as_the_model_saw_them(tmp_path, fake):
    config = "\n".join(f"setting_{i} = value-{i}" for i in range(90)) + "\nbuild_salt = q7Rk-55Lw-91Zt\n" + \
             "\n".join(f"option_{i} = {i}" for i in range(90))

    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return summarizer(rec)
        n = len(prov.executor())
        if n == 1:
            return Turn("Leo la config.", [ToolCall("cfg", "read_file", {"path": "app.ini"})], "tool_use", Usage(0, 30))
        if n == 2:
            prov.window = 8_000  # the next request overflows the real window: learned limit + compaction
            return Turn("Leo más.", [ToolCall("l2", "bash", {"command": "echo " + "l" * 3_000})], "tool_use", Usage(0, 30))
        return Turn("fin", [], "end_turn", Usage(0, 10))
    prov = fake(policy)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "app.ini").write_text(config)
    res = _agent(ws, [], context_limit=128_000).run("revisa app.ini")
    assert res.status == "done"
    prompt = [r for r in prov.log if r["role"] == "summarizer"][0]["messages"][-1]["content"]
    i = prompt.find("build_salt = q7Rk-55Lw-91Zt")  # the middle of a ~5k output is no longer cut to 1200+1200
    assert i > 0 and prompt.find("[OBSERVATION]") < i and "not included here" not in prompt


def test_transcript_elides_oldest_outputs_first_and_says_so():
    events = []
    for i in range(6):
        events += [{"type": "tool_call", "id": f"c{i}", "name": "read_file", "args": {"path": f"f{i}.txt"}},
                   {"type": "tool_result", "id": f"c{i}", "output": f"OUT{i} " + "x" * 3_000}]
    events.append({"type": "tool_call", "id": "sub", "name": "read_file", "args": {}, "depth": 1, "agent": "rev"})
    saved = {"c0": ".nesicode/outputs/c0.txt"}
    text = ctxm.transcript(events, budget=12_000, saved=saved)
    assert len(text) <= 12_000 and "(rev)" not in text  # sub-agent internals stay out of the parent's compaction
    assert "OUT5 " + "x" * 3_000 in text and "OUT4 " + "x" * 3_000 in text  # the newest stay whole
    # older ones keep a fair share each, and every cut says so (with the saved copy)
    assert "OUT0 " in text and "[part of the output not included here" in text and ".nesicode/outputs/c0.txt" in text
    assert "do not state what it contained" in text
    tight = ctxm.transcript(events, budget=8_000, saved=saved)  # no room for a fair share: the oldest go whole
    assert "OUT0" not in tight and "[output not included here" in tight and "OUT5 " + "x" * 3_000 in tight


# ------------------------------------------------------------------ 6. masking
def test_openai_masking_dry_run_size_budget_latest_round_and_markers():
    from nesicode.llm.openai_compat import OpenAICompatConversation, OpenAICompatProvider

    conv = OpenAICompatConversation(OpenAICompatProvider("x", "http://127.0.0.1:9"), "m", "sys", [], ModelOptions())
    conv.add_user("go")
    for i in range(6):
        conv.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]})
        conv.add_tool_results([ToolResult(f"c{i}", "z" * 2_000)])
    conv.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "n1"}, {"id": "n2"}]})
    conv.add_tool_results([ToolResult("n1", "y" * 2_000), ToolResult("n2", "y" * 2_000)])  # not answered yet
    marker = ctxm.mask_marker("read_file", ".nesicode/outputs/x.txt", False, "config/app.ini lines 1–120",
                              'read_file(path="config/app.ini", offset=1, limit=120)')
    before = json.dumps(conv.messages)
    would = conv.mask_tool_results(0, marker, keep_chars=0, apply=False)
    assert would > 0 and json.dumps(conv.messages) == before  # dry run: nothing changes
    freed = conv.mask_tool_results(1, marker, keep_chars=6_500)
    tools = [m["content"] for m in conv.messages if m["role"] == "tool"]
    assert tools[-2:] == ["y" * 2_000] * 2  # the unanswered round is never masked
    assert tools[-3] == "z" * 2_000  # kept by size: newer results (unanswered ones included) up to keep_chars
    assert all(t.startswith("[old read_file output cleared") for t in tools[:5])
    assert freed == sum(2_000 - len(t) for t in tools[:5])
    assert "config/app.ini lines 1–120" in tools[0] and 'offset=1, limit=120' in tools[0] and ".nesicode/outputs/x.txt" in tools[0]


def test_anthropic_masking_works_on_tool_result_blocks_and_drops_thinking():
    from nesicode.llm.anthropic_provider import AnthropicConversation

    conv = AnthropicConversation(None, False, "claude-x", "sys", [], ModelOptions())
    conv.add_user("go")
    for i in range(4):
        conv.messages.append({"role": "assistant", "content": [SimpleNamespace(type="thinking", thinking="hmm"),
                                                               SimpleNamespace(type="tool_use", id=f"c{i}")]})
        conv.add_tool_results([ToolResult(f"c{i}", "z" * 2_000)])
    conv.messages[-1]["content"][0]["content"] = [{"type": "text", "text": "w" * 2_000}]  # blocks, not a string
    conv.messages.append({"role": "assistant", "content": [SimpleNamespace(type="tool_use", id="n1")]})
    conv.add_tool_results([ToolResult("n1", "y" * 2_000)])
    marker = ctxm.mask_marker("bash", ".nesicode/outputs/b.txt", False)
    assert conv.mask_tool_results(0, marker, apply=False) > 0
    assert any(getattr(b, "type", "") == "thinking" for m in conv.messages if m["role"] == "assistant"
               for b in m["content"])  # dry run: untouched
    freed = conv.mask_tool_results(0, marker)
    results = [b for m in conv.messages if m["role"] == "user" and isinstance(m["content"], list) for b in m["content"]]
    assert freed > 0 and all(r["content"].startswith("[old bash output cleared") for r in results[:4])
    assert results[-1]["content"] == "y" * 2_000  # the round the model has not answered
    # an edited history invalidates every later thinking block (a 400 on enforced accounts): none is replayed
    assert not any(getattr(b, "type", "") == "thinking" for m in conv.messages if m["role"] == "assistant"
                   for b in m["content"])


def test_masking_keeps_the_newest_results_by_size_and_never_unseen_ones(tmp_path, fake):
    def policy(conv, rec):
        n = len(prov.executor())
        if n >= 12:
            return Turn("fin", [], "end_turn", Usage(0, 10))
        return Turn("leo", [ToolCall(f"r{n}", "read_file", {"path": f"f{n}.txt"})], "tool_use", Usage(0, 20))
    prov = fake(policy)
    ws = Workspace(tmp_path / "ws")
    for i in range(12):
        (ws.root / f"f{i}.txt").write_text("\n".join(f"dato {i}-{j}" for j in range(400)))
    events = []
    _agent(ws, events, context_limit=40_000).run("lee todo")
    assert [e for e in events if e["type"] == "masked"]
    last = prov.executor()[-1]["messages"]
    tools = [m["content"] for m in last if m["role"] == "tool"]
    kept = [t for t in tools if not t.startswith("[old ")]
    assert len(kept) >= 2 and sum(len(t) for t in kept[:-2]) <= 40_000 * 4 * ctxm.MASK_KEEP_SHARE
    cleared = [t for t in tools if t.startswith("[old ")]
    assert cleared and all(re.search(r"f\d+\.txt lines 1–\d+", t) and "read_file(path=" in t for t in cleared)


# ------------------------------------------------------------------ 7. inline cut of read outputs
def test_read_file_inline_cut_names_the_next_offset(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "logs").mkdir()
    (ws.root / "logs/app.log").write_text("\n".join(f"2026-10-14 INFO event {i} " + "k" * 80 for i in range(316)))
    a = Agent(model="mock/demo", harness=base_harness(), workspace=ws, context_limit=32_768)
    r = a._call_tool(ToolCall("c1", "read_file", {"path": "logs/app.log", "offset": 1, "limit": 120}), Usage())
    m = re.search(r"showing lines 1–(\d+) of 316; lines (\d+)–316 are NOT shown\. Continue with "
                  r"read_file\(path=\"logs/app\.log\", offset=(\d+), limit=(\d+)\); full output saved to (\S+)\]", r.content)
    assert m, r.content[-400:]
    shown, first_hidden, offset, _limit, saved = m.groups()
    assert int(first_hidden) == int(shown) + 1 == int(offset)
    assert r.content.count("\n") == int(shown)  # nothing hidden in the middle: lines 1..shown, then the marker
    assert (ws.root / saved).read_text().count("\n") >= 119
    other = ctxm.inline_output("\n".join(f"row {i}" for i in range(5_000)), ".nesicode/outputs/o.txt", 8_000)
    assert re.search(r"lines \d+–\d+ of 5000 \(chars [\d,]+–[\d,]+ of [\d,]+\) are NOT shown", other)
    assert "search on it" not in other


# ------------------------------------------------------------------ 8. what the checkpoint brings back
def test_checkpoint_reinjects_touched_files_skills_pending_round_and_nudge(tmp_path, fake):
    script = [("skill", [ToolCall("s1", "load_skill", {"name": "documents"})]),
              ("lee", [ToolCall("r1", "read_file", {"path": "notes.md"})]),
              ("edita", [ToolCall("e1", "edit_file", {"path": "app.py", "old": "PORT = 1", "new": "PORT = 8765"})]),
              ("más", [ToolCall("b1", "bash", {"command": "echo " + "u" * 9_000})])]

    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return summarizer(rec)
        n = len(prov.executor()) - 1  # (this request is already logged)
        if n < len(script):
            return Turn(script[n][0], script[n][1], "tool_use", Usage(0, 30))
        return Turn("fin", [], "end_turn", Usage(0, 10))
    prov = fake(policy)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "notes.md").write_text("nota importante\n")
    (ws.root / "app.py").write_text("PORT = 1\n")
    (ws.root / "untouched.py").write_text("NO-DEBE-APARECER\n")
    (ws.root / "secret.txt").write_text("FUERA-DE-LA-JAULA\n")
    (ws.root / "link.md").symlink_to(ws.root / "secret.txt")
    events = []
    a = _agent(ws, events, context_limit=9_000, max_steps=4)
    a._touched["link.md"] = "read"  # a symlink is never followed
    res = a.run("porta la config")
    assert res.status == "max_steps"
    msgs = _post(prov)[0]["messages"]
    post = next(m["content"] for m in msgs if ctxm.SUMMARY_PREFIX in (m.get("content") or ""))
    assert "## Files you read or edited in this run" in post and "PORT = 8765" in post and "nota importante" in post
    assert "do not read them again" in post and "NO-DEBE-APARECER" not in post and "FUERA-DE-LA-JAULA" not in post
    assert "## Results of your last tool calls (you have not seen them yet" in post and "u" * 2_000 in post
    assert "documents" in post and "load_skill" in post
    assert FINAL_NUDGE_TEXT in msgs[-1]["content"]  # the step budget was used up after the compaction: nudged


FINAL_NUDGE_TEXT = "what you changed, what you verified, and what is left."


def test_final_nudge_survives_a_compaction_on_the_next_iteration(tmp_path, fake):
    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return summarizer(rec)
        n = len(prov.executor(accepted=False))
        if n == 3:  # right after the nudge, the provider rejects the request: learned limit, compaction
            prov.window = 12_000
        if _has(rec, "Do not call any more tools"):
            return Turn("Resumen final: hecho X, verificado Y.", [], "end_turn", Usage(0, 30))
        return Turn("sigo", [ToolCall(f"b{n}", "bash", {"command": "echo " + "p" * 4_000})], "tool_use", Usage(0, 30))
    prov = fake(policy)
    events = []
    res = _agent(Workspace(tmp_path / "ws"), events, context_limit=64_000, max_steps=3).run("tarea")
    assert any(e["type"] == "compacted" for e in events)
    assert res.status == "max_steps" and res.final_text.startswith("Resumen final")


def test_anchors_have_no_invented_files_or_code_errors():
    events = [{"type": "tool_call", "id": "w", "name": "write_file",
               "args": {"path": "shapes.py", "content": "class T:\n    def __init__(self, c):\n        self.c = c\n"
                                                        "        raise ValueError('bad side')\n"}},
              {"type": "tool_result", "id": "w", "output": "created shapes.py"},
              {"type": "tool_call", "id": "r", "name": "read_file", "args": {"path": "docs/sdk.md"}},
              {"type": "tool_result", "id": "r", "output": "    25\treturn r.json()['access_token']\n    26\t- login failed? see FAQ"},
              {"type": "tool_call", "id": "t", "name": "bash", "args": {"command": "python -m pytest -q 2>&1 | tail -3"}},
              {"type": "tool_result", "id": "t", "output": "exit code: 0\nE   ModuleNotFoundError: No module named 'httpx'\n1 failed, 3 passed"}]
    a = ctxm.extract_anchors("return r.json()['x'] and self.c = 1; see src/app.py and `http://localhost:8080/v2`")
    assert "r.json" not in a.get("files", []) and "src/app.py" in a["files"]
    assert a["urls"] == ["http://localhost:8080/v2"]
    errs = ctxm.error_lines(events)
    assert any("No module named 'httpx'" in e for e in errs) and any("1 failed" in e for e in errs)
    assert not any("ValueError" in e or "login failed" in e for e in errs)


# ------------------------------------------------------------------ 9. archive
def test_archive_names_never_collide_across_runs_and_events_keep_the_summary(tmp_path, fake):
    long_summary = SUMMARY + "\n" + "detalle del resumen " * 400

    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return Turn(long_summary, [], "end_turn", Usage(0, 900))
        if _has(rec, ctxm.SUMMARY_PREFIX):
            return Turn("fin", [], "end_turn", Usage(0, 10))
        return Turn("leo", [ToolCall(f"b{len(prov.log)}", "bash", {"command": "echo " + "t" * 7_000})], "tool_use", Usage(0, 30))
    prov = fake(policy)
    ws = Workspace(tmp_path / "ws")
    paths = []
    for turn in ("TURNO-UNO", "TURNO-DOS"):
        events = []
        _agent(ws, events, context_limit=6_000, archive=".nesicode/chats/3").run(turn)
        comp = [e for e in events if e["type"] == "compacted"]
        assert comp and len(comp[0]["summary"]) > 4_000  # the full summary travels in the event (capped at 20k)
        paths.append((comp[0]["path"], comp[0]["transcript"]))
    assert paths[0] != paths[1]
    t1, t2 = (ws.root / paths[0][1]).read_text(), (ws.root / paths[1][1]).read_text()
    assert "TURNO-UNO" in t1 and "TURNO-DOS" in t2 and t1.startswith("# Transcript 1")
    from nesicode.agent.context import search_history
    assert "TURNO-UNO" in search_history(ws, {"query": "TURNO-UNO"}, ".nesicode/chats/3")


def test_only_saved_outputs_are_masked():
    m = ctxm.mask_marker("bash", None, False)
    assert m("c", "x" * 350) is None and m("c", "x" * 5_000) is None  # not saved: never cleared
    m2 = ctxm.mask_marker("bash", ".nesicode/outputs/c.txt", False)
    assert m2("c", "x" * 350) is None and m2("c", "x" * 5_000).startswith("[old bash output cleared")


# ------------------------------------------------------------------ 10. cross-turn continuity (jobs.py)
def _client():
    from fastapi.testclient import TestClient
    from nesicode.server.app import app

    c = TestClient(app)
    assert c.post("/api/auth/register", json={"email": f"a{time.time_ns()}@x.com", "password": "secret123"}).status_code == 200
    p = c.post("/api/projects", json={"name": "p", "executor_model": "mock/demo", "improver_model": "mock/demo"}).json()
    t = c.post(f"/api/projects/{p['id']}/threads", json={}).json()
    return c, p, t


def _run(p, t, task, status, final="", events=(), at=None):
    from nesicode.server.app import db

    rid = db.insert("INSERT INTO runs (project_id, thread_id, kind, task, status, final_text, created_at) "
                    "VALUES (?, ?, 'chat', ?, ?, ?, ?)", (p["id"], t["id"], task, status, final, at or time.time()))
    for e in events:
        db.add_event(rid, e)
    return db.one("SELECT * FROM runs WHERE id=?", (rid,))


def _work(n=3, prefix="c"):
    out = [{"type": "start"}]
    for i in range(n):
        out += [{"type": "tool_call", "id": f"{prefix}{i}", "name": "bash", "args": {"command": f"echo paso{i}"}},
                {"type": "tool_result", "id": f"{prefix}{i}", "output": f"exit code: 0\npaso{i} hecho", "is_error": False}]
    return out


def test_history_skips_removed_messages_and_gives_the_trail_to_the_turn_that_worked():
    from nesicode.server.app import db
    from nesicode.server.jobs import continuity_note, conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "refactoriza el módulo de pagos", "cancelled", "", _work(5) + [{"type": "cancelled"}], at=now - 40)
    _run(p, t, "borra la carpeta tests entera", "cancelled", at=now - 30)  # removed from the queue: never ran
    _run(p, t, "¿y esto?", "cancelled", "", [{"type": "start"}, {"type": "cancelled"}], at=now - 20)  # stopped at once
    history, interrupted = conversation_history(db, _run(p, t, "sigue", "waiting", at=now))
    assert [h[0] for h in history] == ["refactoriza el módulo de pagos", "¿y esto?"]
    assert "[PREVIOUS TURN — STEPS]" in history[0][1] and "paso4" in history[0][1]
    assert history[1][1] == "[this turn was cancelled before doing any work]"
    assert interrupted["id"] and interrupted["steps"] == 5 and interrupted["later_empty"] == 1
    note = continuity_note(t["id"], interrupted)
    assert "at step 5" in note and "do not start over" in note and "later message" in note
    assert continuity_note(t["id"], None) == ""  # nothing interrupted: the agent itself describes earlier turns


def test_compacted_turn_brings_its_checkpoint_to_the_next_message():
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    events = _work(4) + [{"type": "compacted", "summary": "## Goal\nPortar la config.\n## Next Step\nEditar app.py (CHECKPOINT-X)",
                          "n": 1}] + _work(2, "d")
    _run(p, t, "porta la config", "max_steps", "Hice parte.", events, at=now - 10)
    history, _ = conversation_history(db, _run(p, t, "sigue", "waiting", at=now))
    assert "CHECKPOINT-X" in history[0][1] and "last checkpoint summary" in history[0][1]


def test_history_md_and_connector_context_use_queue_order_and_only_earlier_runs(tmp_path):
    from nesicode.server.app import db
    from nesicode.server.jobs import _write_thread_history, connector_context, conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "PRIMERO", "done", "a", [{"type": "start"}, {"type": "tool_call", "id": "x", "name": "mcp__vt__report", "args": {}}],
         at=now - 30)
    cur = _run(p, t, "ACTUAL", "running", at=now - 20)
    _run(p, t, "LUEGO (encolado después)", "done", "b",  # created after the current one (a later queued message)
         [{"type": "start"}, {"type": "tool_call", "id": "y", "name": "mcp__burp__scan", "args": {}}], at=now)
    _run(p, t, "QUITADO", "cancelled", at=now - 25)  # removed from the queue
    history, _ = conversation_history(db, cur)
    assert [h[0] for h in history] == ["PRIMERO"]
    ctx = connector_context(db, cur, history)
    assert "mcp__vt__report" in ctx and "mcp__burp__scan" not in ctx
    ws = Workspace(tmp_path / "ws")
    _write_thread_history(db, ws, cur, ".nesicode/chats/1")
    md = (ws.root / ".nesicode/chats/1/context/history.md").read_text()
    assert "PRIMERO" in md and "LUEGO" not in md and "QUITADO" not in md


def test_long_conversations_keep_their_first_turn():
    from nesicode.server.app import db
    from nesicode.server.jobs import _earlier_turns, conversation_history

    _, p, t = _client()
    now = time.time()
    for i in range(30):
        _run(p, t, f"turno {i}", "done", f"respuesta {i}", [{"type": "start"}], at=now - 100 + i)
    cur = _run(p, t, "sigue", "waiting", at=now)
    assert [r["task"] for r in _earlier_turns(db, cur, limit=5)] == ["turno 0", "turno 26", "turno 27", "turno 28", "turno 29"]
    history, _ = conversation_history(db, cur)
    assert history[0][0] == "turno 0" and history[-1][0] == "turno 29" and len(history) == 30


def test_fit_history_keeps_the_first_message_and_says_what_was_omitted(tmp_path):
    a = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(tmp_path), context_limit=16_000)
    history = [(f"Reglas: {RULE}. Responde en español.", "ok")] + [(f"pregunta {i} " + "q" * 2_000, "r" * 6_000) for i in range(30)]
    kept = a._fit_history(history, "system")
    assert kept[0][0].startswith("Reglas:") and RULE in kept[0][0]
    assert re.search(r"\[\d+ earlier turn\(s\) between this one and the next shown are omitted here — search_history finds "
                     r"them\]", kept[0][1])
    assert kept[-1][0].startswith("pregunta 29")  # the newest turn stays
    assert sum(len(u) + len(x) for u, x in kept) <= (int(16_000 * 0.3) - 1) * 4 + 200
    # a huge first message stays too (head + tail)
    kept = a._fit_history([("R" * 200_000 + "COLA", "ok"), ("b", "c")], "system", budget=500)
    assert kept[0][0].endswith("COLA") and len(kept[0][0]) < 3_000


def test_agent_mentions_earlier_turns_only_when_there_are_some(tmp_path, fake):
    prov = fake(lambda conv, rec: Turn("hola", [], "end_turn", Usage(0, 5)))
    _agent(Workspace(tmp_path / "a"), [], context_limit=32_000).run("hola")
    system = prov.log[-1]["messages"][0]["content"]
    assert "earlier turns" not in system.lower() and "history.md" not in system
    assert "private reasoning is not kept between steps" in system  # facts go in visible text or a notes file
    _agent(Workspace(tmp_path / "b"), [], context_limit=32_000).run("sigue", [("antes", "respuesta")])
    system = prov.log[-1]["messages"][0]["content"]
    assert "# Earlier turns of this conversation" in system and "If something you need is not shown" in system


# ------------------------------------------------------------------ redact / digest
def test_redact_formats_and_no_over_redaction():
    from nesicode.agent.context import redact

    keep = ["Author: Ricardo Sanchez <rs@example.com>", "authentication: required", "HTTP/1.1 403 Forbidden"]
    for s in keep:
        assert redact(s) == s
    cases = {
        "echo sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWx": "AbCdEfGhIj",
        "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 pushed": "AbCdEfGhIjKl",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA1234\n-----END RSA PRIVATE KEY-----": "MIIEpAIB",
        "curl -u admin:S3cr3tPassw0rd https://x": "S3cr3tPassw0rd",
        "mysql -u root -pS3cr3tPassw0rd db": "S3cr3tPassw0rd",
        'password: "correct horse battery staple"': "horse battery",
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY": "wJalrX",
        "AKIAIOSFODNN7EXAMPLE": "IOSFODNN7",
    }
    for s, secret in cases.items():
        assert secret not in redact(s), redact(s)
    assert redact('-H "Cookie: session=s3ss-alice-0001; theme=dark"') == '-H "Cookie: session=<redacted>; theme=<redacted>"'
    assert redact("Set-Cookie: csrftoken=AbC123secretTOKEN; Path=/") == "Set-Cookie: csrftoken=<redacted>; Path=/"


def test_run_digest_reads_bash_and_http_honestly():
    from nesicode.agent.context import run_digest

    events = [
        {"type": "tool_call", "id": "1", "name": "bash", "args": {"command": "python -m pytest -q 2>&1 | tail -20"}},
        {"type": "tool_result", "id": "1", "output": "exit code: 0\nE   ModuleNotFoundError: No module named 'pytest'\n1 error in 0.1s"},
        {"type": "tool_call", "id": "2", "name": "read_file", "args": {"path": "logs/billing.log"}},
        {"type": "tool_result", "id": "2", "output": "   237\tworker: shard 9 crashed with exit status 139\n   238\tstatus: 200 ok"},
        {"type": "tool_call", "id": "3", "name": "write_file", "args": {"path": "tri.py", "content": "raise ValueError('x')\nself.c = 1"}},
        {"type": "tool_result", "id": "3", "output": "created tri.py"},
        {"type": "tool_call", "id": "4", "name": "bash", "args": {"command": "curl -sS -i https://vulnbank.org/api -H 'X-Api-Key: k3y-val-123456'"}},
        {"type": "tool_result", "id": "4", "output": "exit code: 0\nHTTP/1.1 403 Forbidden\nserver: nginx"},
        {"type": "tool_call", "id": "5", "name": "bash", "args": {"command": "sleep 120"}},
        {"type": "tool_result", "id": "5", "output": "exit code: 130\n[stopped by the user]"},
        {"type": "assistant", "text": "Config confirmado: build_salt = q7Rk-55Lw-91Zt (línea 73)"},
        {"type": "assistant", "text": "Sigo con los tests."},
    ]
    d = run_digest(events)
    assert "exit code 0 · E   ModuleNotFoundError: No module named 'pytest' … 1 error in 0.1s" in d
    assert "HTTP 139" not in d and "HTTP 130" not in d and "HTTP 200" not in d
    assert "HTTP 403" in d  # a real HTTP call
    assert "exit code 130" in d
    assert "Errors (verbatim)" in d and "ValueError" not in d and "self.c" not in d
    assert "build_salt = q7Rk-55Lw-91Zt" in d  # what it said along the way
    assert "k3y-val-123456" not in d and "<redacted>" in d and "search_history" in d  # and where the real value is


# ------------------------------------------------------------------ 11. limits
def test_learned_limit_survives_ttl_and_refresh(monkeypatch):
    import nesicode.server.limits as limits
    from nesicode.server.app import db

    uid = int(time.time_ns() % 1_000_000_000)
    reported = {"value": None}
    monkeypatch.setattr(limits, "fetch_limits", lambda *a, **k: reported["value"])
    monkeypatch.setattr(limits, "server_provider_config", lambda name: {"kind": "openai", "base_url": "http://127.0.0.1:9"})
    spec = "compat/modelo-local"
    assert limits.model_limits(db, uid, spec)["source"] == "default"
    limits.learn_limit(db, uid, spec, 32_768)
    db.execute("UPDATE model_limits SET updated_at=updated_at-?", (25 * 3600,))  # a day later
    lim = limits.model_limits(db, uid, spec)
    assert lim["context"] == 32_768 and lim["source"] == "learned" and lim["output"]
    reported["value"] = {"context": 200_000, "output": 8_000}  # a bigger (or table/probe) value never replaces it
    limits.refresh_provider_limits(db, uid, "compat", ["modelo-local"])
    assert limits.model_limits(db, uid, spec)["context"] == 32_768
    assert limits.model_limits(db, uid, spec, refresh=True)["context"] == 32_768
    reported["value"] = {"context": 16_384, "output": 4_000}  # the provider reports a SMALLER window: it wins
    assert limits.model_limits(db, uid, spec, refresh=True) == {"context": 16_384, "output": 4_000, "source": "provider"}


# ------------------------------------------------------------------ 12. sub-agents
def test_subagent_gets_limits_and_the_conversation_archive(tmp_path, fake):
    def policy(conv, rec):
        if rec["role"] == "subagent":
            n = sum(1 for r in prov.log if r["role"] == "subagent")
            if n == 1:
                return Turn("leo", [ToolCall("s1", "read_file", {"path": ".nesicode/chats/7/outputs/big.txt"})], "tool_use", Usage(0, 10))
            return Turn("informe del subagente", [], "end_turn", Usage(0, 10))
        if not any(e["type"] == "tool_result" for e in events):
            return Turn("delego", [ToolCall("d1", "delegate", {"agent": "reviewer", "task": "analiza el output"})], "tool_use", Usage(0, 10))
        return Turn("fin", [], "end_turn", Usage(0, 10))
    prov = fake(policy)
    ws = Workspace(tmp_path / "ws")
    big = ws.root / ".nesicode/chats/7/outputs/big.txt"
    big.parent.mkdir(parents=True)
    big.write_text("linea\n" * 3_000)
    h = base_harness()
    h["sub_agents/reviewer/AGENT.md"] = "---\nname: reviewer\ndescription: revisa\n---\nRevisa."
    events = []
    a = Agent(model="fake/m", harness=h, workspace=ws, emit=events.append, context_limit=128_000, output_limit=16_000,
              archive=".nesicode/chats/7")
    res = a.run("revisa el output guardado")
    assert res.status == "done"
    sub = [r for r in prov.log if r["role"] == "subagent"]
    assert sub and all(r["max_tokens"] <= 16_000 for r in sub)  # never the 64k default
    sub_results = [e for e in events if e["type"] == "tool_result" and e.get("depth")]
    assert sub_results and not sub_results[0]["is_error"]  # its own conversation's archive is readable
    assert "full output saved to .nesicode/chats/7/outputs/" in sub_results[0]["output"]  # Layer 0 at depth > 0


# ------------------------------------------------------------------ 13. small harness nudges + retries
def test_read_only_streak_gets_one_reminder(tmp_path, fake):
    def policy(conv, rec):
        n = len(prov.executor())
        if n >= 25:
            return Turn("fin", [], "end_turn", Usage(0, 5))
        return Turn("leo", [ToolCall(f"l{n}", "list_dir", {"path": "."})], "tool_use", Usage(0, 5))
    prov = fake(policy)
    _agent(Workspace(tmp_path / "ws"), [], context_limit=128_000).run("explora")
    tools = [m["content"] for m in prov.executor()[-1]["messages"] if m["role"] == "tool"]
    assert sum("tool calls in a row without writing" in t for t in tools) == 1
    assert "tool calls in a row without writing" in tools[19]


def test_provider_retries_become_visible_events(tmp_path, fake):
    def policy(conv, rec):
        notify = getattr(rec["options"], "notify", None)
        assert callable(notify)
        notify({"type": "llm_retry", "attempt": 2, "max": 5, "wait": 8, "reason": "read timeout"})
        return Turn("hola", [], "end_turn", Usage(0, 5))
    fake(policy)
    events = []
    _agent(Workspace(tmp_path / "ws"), events, context_limit=32_000).run("hola")
    retry = [{k: v for k, v in e.items() if k != "_persist"} for e in events if e["type"] == "llm_retry"]
    assert retry == [{"type": "llm_retry", "attempt": 2, "max": 5, "wait": 8, "reason": "read timeout"}]
    assert next(e for e in events if e["type"] == "llm_retry")["_persist"] is True  # stored with the run


def test_window_scaled_budgets():
    assert ctxm.min_output(16_384) < 8_192 and ctxm.safety_tokens(16_384) < 4_096
    assert ctxm.min_output(200_000) == 8_192 and ctxm.safety_tokens(200_000) == 4_096
    assert ctxm.inline_limit(16_384) == 8_000 and ctxm.inline_limit(131_072) == 24_000 and ctxm.inline_limit(None) == 8_000
    assert ctxm.MAX_COMPACTIONS >= 40
    # the "no room" trigger no longer fires before the 67% summary threshold on small windows
    for w in (16_384, 24_576, 32_768):
        assert w - ctxm.safety_tokens(w) - ctxm.min_output(w) > ctxm.SUMMARY_AT * w


# ------------------------------------------------------------------ review fixes
def test_huge_message_stays_whole_in_the_archive_after_a_learned_limit(tmp_path, fake):
    # configured for 128k, the real window is 32k: the message is cut twice (128k, then 32k), but the archived copy
    # must always be the ORIGINAL message, and the context must point to it
    prov = fake(lambda conv, rec: Turn("visto", [], "end_turn", Usage(0, 10)), window=32_768)
    ws = Workspace(tmp_path / "ws")
    huge = "INICIO\n" + "\n".join(f"linea {i} " + "x" * 60 for i in range(3_000)) + "\nMEDIO-UNICO-42\n" + \
           "\n".join(f"linea b{i} " + "y" * 60 for i in range(3_000)) + "\nFIN"
    res = _agent(ws, [], context_limit=128_000).run(huge)
    assert res.status == "done" and len([r for r in prov.log if r["rejected"]]) == 1
    sent = prov.executor()[0]["messages"][-1]["content"]
    rel = re.search(r"saved in (\S+request\.md)", sent).group(1)
    assert (ws.root / rel).read_text() == huge and "MEDIO-UNICO-42" not in sent


def test_denser_text_than_estimated_recovers_after_a_context_error(tmp_path, fake):
    # 3.2 chars/token (code, Spanish...) on a window smaller than configured: the re-fit must learn the density
    # instead of re-sending the same oversized request until the run fails
    prov = fake(lambda conv, rec: Turn("ok", [], "end_turn", Usage(0, 10)), window=16_917, cpt=3.2)
    history = [(f"turno {i} " + "u" * 2_000, "resp " + "a" * 15_000) for i in range(4)]
    res = _agent(Workspace(tmp_path / "ws"), [], context_limit=32_768, output_limit=4_096).run("tarea " + "t" * 30_000, history)
    assert res.status == "done", res.error
    assert len([r for r in prov.log if r["rejected"]]) <= 2
    assert ctxm.prompt_tokens("However, you requested 17857 tokens (14946 in the messages, 2911 in the completion).") == 14_946
    assert ctxm.prompt_tokens("prompt is too long: 1,050,000 tokens > 1000000 maximum") == 1_050_000
    assert ctxm.prompt_tokens("rate limited") is None


def test_summary_that_quotes_reasoning_tags_is_kept_intact():
    ok = "## Goal\nFix the parser: an unclosed <think> tag and the regex `<think>.*?</think>`.\n## Next Step\nRun tests."
    assert ctxm.clean_summary(ok) == (ok, "")
    assert ctxm.clean_summary("<think>\nplan\n</think>\n" + ok) == (ok, "")  # a leading reasoning block still goes
    assert ctxm.clean_summary("Let me write it.</think>\n" + ok) == (ok, "")  # opening tag eaten by a reasoning parser
    assert ctxm.clean_summary("  <think>\ncut while thinking", "max_tokens")[1] == "unclosed reasoning block"


def test_subagents_can_search_the_conversation_archive(tmp_path):
    sub = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(tmp_path), depth=1, archive=".nesicode/chats/7")
    assert "search_history" in [s.name for s in sub.tool_specs()]  # its checkpoints tell it to use it
