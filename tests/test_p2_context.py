"""Phase 2, group P — context and continuity: reasoning replay, progress nudges, history/trail, masking notice, safe
archive, interrupted replies, hooks on errors, background processes, connector context, Claude stall watchdog.

Offline: fake providers, a scripted HTTP server on 127.0.0.1 (SSE, stalls), temp workspaces and the test database."""

from __future__ import annotations

import json
import os
import random
import re
import select
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nesicode.agent import Agent, Workspace, trajectory_text
from nesicode.agent import context as ctxm
from nesicode.agent import loop as loopmod
from nesicode.harness import base_harness
from nesicode.llm import ModelOptions, ToolResult, transport
from nesicode.llm.base import Conversation, LLMError, Provider, ToolCall, Turn, Usage
from nesicode.llm.mock import MockConversation
from nesicode.llm.openai_compat import OpenAICompatConversation, OpenAICompatProvider

posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX shell, process groups and symlinks")
SUMMARY = "## Goal\nSeguir.\n## Next Step\nContinuar."


# ====================================================================== fake provider (OpenAI-format messages)
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

    @property
    def replays_reasoning(self):
        return self.prov.replay

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

    def step(self, on_event=None):
        chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in self.messages)
        chars += sum(len(json.dumps({"name": t.name, "description": t.description, "parameters": t.parameters}))
                     for t in self.tools or [] if t is not None)
        inp = int(chars / self.prov.cpt)
        rec = {"role": self.options.role, "input": inp, "rejected": False, "options": self.options,
               "messages": json.loads(json.dumps(self.messages, default=str))}
        self.prov.log.append(rec)
        if self.prov.window and inp + (self.options.max_tokens or 0) > self.prov.window:
            rec["rejected"] = True
            raise LLMError(f"HTTP 400: This model's maximum context length is {self.prov.window} tokens. However, you "
                           f"requested {inp + self.options.max_tokens} tokens ({inp} in the messages, ...).")
        turn = self.prov.policy(self, rec, on_event)
        turn.usage = Usage(inp, turn.usage.output_tokens or 40)
        msg = {"role": "assistant", "content": turn.text}
        if turn.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in turn.tool_calls]
        if self.prov.replay and turn.thinking:
            msg["reasoning_content"] = turn.thinking
        self.messages.append(msg)
        return turn


class Prov(Provider):
    name = "fake"

    def __init__(self, policy, window=None, cpt=4.0, replay=False, conv=None):
        self.policy, self.window, self.cpt, self.replay, self.log = policy, window, cpt, replay, []
        self.conv_cls = conv or Conv

    def conversation(self, model, system, tools, options):
        return self.conv_cls(self, model, system, tools, options)

    def executor(self):
        return [r for r in self.log if r["role"] == "executor" and not r["rejected"]]


@pytest.fixture()
def fake(monkeypatch):
    def install(policy, **kw):
        prov = Prov(policy, **kw)
        monkeypatch.setattr(loopmod, "get_provider", lambda spec, keys=None: (prov, "m"))
        return prov
    return install


def agent(ws, events=None, **kw):
    kw.setdefault("output_limit", 16_000)
    return Agent(model="fake/m", harness=kw.pop("harness", None) or base_harness(), workspace=ws,
                 emit=(events.append if events is not None else None), **kw)


def code(rnd, n):
    lines = []
    while sum(len(x) + 1 for x in lines) < n:
        lines.append(f"    def m_{rnd.randint(0, 99999)}(self, v: int) -> int:  # case {rnd.randint(0, 999)}")
        lines.append(f"        return self.cache.get(v, {rnd.randint(0, 9999)}) + f(v * {rnd.randint(1, 99)})")
    return "\n".join(lines)


def tool_texts(rec):
    return [m["content"] for m in rec["messages"] if m["role"] == "tool"]


# ====================================================================== scripted HTTP server (127.0.0.1)
class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass  # clients that hang up (Stop, the stall watchdog) are expected


class FakeServer:
    def __init__(self, *script):
        self.script, self.requests, self.disconnects = list(script), [], []
        self.closing = threading.Event()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                fake.requests.append(json.loads(raw) if raw else None)
                fn = fake.script.pop(0) if fake.script else json_reply(500, {"error": {"message": "no more script"}})
                fn(self, fake.requests[-1])

        self.httpd = _Server(("127.0.0.1", 0), Handler)
        self.httpd.fake = self
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.closing.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def server():
    made = []

    def make(*script):
        made.append(FakeServer(*script))
        return made[-1]
    yield make
    for s in made:
        s.close()


@pytest.fixture(autouse=True)
def quick_retries(monkeypatch):
    monkeypatch.setattr(transport, "BACKOFF_BASE", 0.01)


def json_reply(status, payload):
    def handler(req, body):
        data = json.dumps(payload).encode()
        req.send_response(status)
        req.send_header("Content-Type", "application/json")
        req.send_header("Content-Length", str(len(data)))
        req.end_headers()
        req.wfile.write(data)
    return handler


def _stall(req, limit=10.0):
    fake = req.server.fake
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit and not fake.closing.is_set():
        readable, _, _ = select.select([req.connection], [], [], 0.05)
        if readable:
            try:
                data = req.connection.recv(1)
            except OSError:
                data = b""
            if not data:
                fake.disconnects.append(time.monotonic())
                break
    req.close_connection = True


def hang(req, body):  # never answers, not even the headers
    _stall(req)


def sse(*chunks, then="end"):
    """Chunks are bytes (a number = a pause in seconds); then: end | stall (open and silent)."""
    def handler(req, body):
        req.send_response(200)
        req.send_header("Content-Type", "text/event-stream")
        req.send_header("Transfer-Encoding", "chunked")
        req.end_headers()
        try:
            for c in chunks:
                if isinstance(c, (int, float)):
                    time.sleep(c)
                    continue
                req.wfile.write(b"%x\r\n%s\r\n" % (len(c), c))
                req.wfile.flush()
        except OSError:
            req.server.fake.disconnects.append(time.monotonic())
            req.close_connection = True
            return
        if then == "stall":
            _stall(req)
        else:
            req.wfile.write(b"0\r\n\r\n")
    return handler


def oc(delta=None, finish=None, usage=None):  # one OpenAI-style SSE chunk
    c = {"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    return ("data: " + json.dumps(c) + "\n\n").encode()


DONE = b"data: [DONE]\n\n"
WRITE = [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": '{"path": "a"}'}}]


def call_reply(reasoning, key="reasoning_content", cid="call_1"):
    calls = [dict(WRITE[0], id=cid)]
    return sse(oc({key: reasoning}), oc({"tool_calls": calls}), oc(finish="tool_calls"),
               oc(usage={"prompt_tokens": 50, "completion_tokens": 5}), DONE)


def text_reply(text):
    return sse(oc({"content": text}), oc(finish="stop"), DONE)


def assistants(body):
    return [m for m in body["messages"] if m["role"] == "assistant"]


# ====================================================================== P1 reasoning replay
def test_reasoning_goes_back_only_within_the_current_tool_loop_under_the_servers_name(server):
    s = server(sse(oc({"reasoning": "El bug está en "}), oc({"reasoning": "streamer.ts:238."}),
                   oc({"tool_calls": WRITE}), oc(finish="tool_calls"), DONE),
               text_reply("hecho"))
    prov = OpenAICompatProvider("nesilabs", s.url, "k")
    conv = prov.conversation("nesi3", "sys", [], ModelOptions(max_tokens=500))
    conv.add_user("turno anterior")
    conv.add_assistant_text("respuesta anterior")  # an earlier turn (chat history): never carries reasoning
    conv.add_user("go")
    turn = conv.step()
    assert turn.thinking == "El bug está en streamer.ts:238." and conv.replays_reasoning
    conv.add_tool_results([ToolResult(turn.tool_calls[0].id, "ok")])
    assert conv.step().text == "hecho"
    history, loop_reply = assistants(s.requests[1])
    assert not any(k in history for k in ("reasoning", "reasoning_content"))
    assert loop_reply["reasoning"] == "El bug está en streamer.ts:238." and "reasoning_content" not in loop_reply


@pytest.mark.parametrize("refusal", [
    {"error": {"message": "1 validation error for ChatCompletionRequest\nmessages.2.reasoning_content\n  Extra inputs are "
                          "not permitted"}},
    {"detail": [{"type": "extra_forbidden", "msg": "Extra inputs are not permitted"}]},  # the field is not even named
])
def test_a_server_refusing_replayed_reasoning_gets_it_stripped_retried_once_and_never_again(server, refusal):
    s = server(call_reply("R1 " * 50), json_reply(400, refusal), call_reply("R2", cid="call_2"), text_reply("fin"))
    prov = OpenAICompatProvider("compat", s.url, "k")
    conv = prov.conversation("glm", "sys", [], ModelOptions(max_tokens=500))
    conv.add_user("go")
    t1 = conv.step()
    conv.add_tool_results([ToolResult(t1.tool_calls[0].id, "ok")])
    t2 = conv.step()  # refused once (400), stripped, sent again
    assert t2.thinking == "R2" and prov.reasoning_refused and not conv.replays_reasoning
    assert conv.take_dropped() == len("R1 " * 50) and conv.take_dropped() == 0  # reported once, to the loop
    conv.add_tool_results([ToolResult(t2.tool_calls[0].id, "ok")])
    assert conv.step().text == "fin"
    assert len(s.requests) == 4 and assistants(s.requests[1])[0]["reasoning_content"] == "R1 " * 50
    for body in s.requests[2:]:  # the retry and every later request: no reasoning anywhere
        assert not any(k in m for m in assistants(body) for k in ("reasoning", "reasoning_content"))


def test_an_unrelated_400_is_not_mistaken_for_a_reasoning_refusal(server):
    s = server(call_reply("R1"), json_reply(400, {"error": {"message": "Invalid tool_call_id: call_9 is unknown"}}))
    prov = OpenAICompatProvider("compat", s.url, "k")
    conv = prov.conversation("m", "sys", [], ModelOptions(max_tokens=500))
    conv.add_user("go")
    t1 = conv.step()
    conv.add_tool_results([ToolResult(t1.tool_calls[0].id, "ok")])
    with pytest.raises(LLMError, match="Invalid tool_call_id"):
        conv.step()
    assert not prov.reasoning_refused and len(s.requests) == 2


def _loop_conv(name="nesilabs"):
    conv = OpenAICompatConversation(OpenAICompatProvider(name, "http://127.0.0.1:9"), "m", "sys", [], ModelOptions())
    conv.add_user("go")
    for i, why in enumerate(("PENSAMIENTO-A " * 40, "PENSAMIENTO-B " * 40, "PENSAMIENTO-C " * 40)):
        conv.messages.append({"role": "assistant", "content": "", "reasoning_content": why,
                              "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "x", "arguments": "{}"}}]})
        conv.add_tool_results([ToolResult(f"c{i}", "z" * 3_000)])
    return conv


def test_masking_drops_the_reasoning_of_old_replies_in_the_same_pass():
    conv = _loop_conv()
    marker = ctxm.mask_marker("bash", ".nesicode/outputs/x.txt", False)
    before = json.dumps(conv.messages)
    would = conv.mask_tool_results(1, marker, apply=False)
    assert json.dumps(conv.messages) == before  # dry run
    freed = conv.mask_tool_results(1, marker)
    replies = [m for m in conv.messages if m["role"] == "assistant"]
    assert "reasoning_content" not in replies[0] and "reasoning_content" not in replies[1]
    assert replies[2]["reasoning_content"].startswith("PENSAMIENTO-C")  # issued the round not answered yet: kept
    tools = [m["content"] for m in conv.messages if m["role"] == "tool"]
    assert tools[-1] == "z" * 3_000 and all(t.startswith("[old bash output cleared") for t in tools[:2])
    reasoning = 2 * len("PENSAMIENTO-A " * 40)
    assert freed == would == sum(3_000 - len(t) for t in tools[:2]) + reasoning


def test_deepseek_keeps_its_reasoning_when_masking():
    conv = _loop_conv("deepseek")
    conv.mask_tool_results(1, ctxm.mask_marker("bash", ".nesicode/outputs/x.txt", False))
    assert all(m.get("reasoning_content") for m in conv.messages if m["role"] == "assistant")


def test_the_loop_counts_replayed_reasoning_in_its_context_estimate(tmp_path, fake):
    def run(replay):
        def policy(conv, rec, on_event):
            if len(prov.executor()) == 1:
                return Turn("miro", [ToolCall("t1", "list_dir", {"path": "."})], "tool_use", Usage(0, 30),
                            thinking="r" * 20_000)
            return Turn("listo", [], "end_turn", Usage(0, 10))
        prov = fake(policy, replay=replay)
        events = []
        agent(Workspace(tmp_path / f"ws{replay}"), events, context_limit=200_000).run("mira")
        return [e["used"] for e in events if e["type"] == "context"][-1]
    assert 4_500 <= run(True) - run(False) <= 5_500  # 20,000 chars of reasoning ≈ 5,000 tokens


class Refusing(Conv):
    """Its server refuses replayed reasoning on the 2nd request: the history loses it (like OpenAICompatProvider)."""
    dropped = 0

    def take_dropped(self):
        n, self.dropped = self.dropped, 0
        return n

    def step(self, on_event=None):
        if self.prov.replay and len(self.prov.executor()) == 1:
            for m in self.messages:
                self.dropped += len(m.pop("reasoning_content", ""))
            self.prov.replay = False
        return super().step(on_event)


def test_reasoning_a_server_refused_leaves_the_context_estimate(tmp_path, fake):
    def run(replay):
        def policy(conv, rec, on_event):
            n = len(prov.executor())
            if n <= 2:
                return Turn("miro", [ToolCall(f"t{n}", "list_dir", {"path": "."})], "tool_use", Usage(0, 30),
                            thinking="r" * 20_000 if n == 1 else "")
            return Turn("listo", [], "end_turn", Usage(0, 10))
        prov = fake(policy, replay=replay, conv=Refusing, cpt=3.3)
        a = agent(Workspace(tmp_path / f"ws{replay}"), [], context_limit=200_000)
        a.run("mira")
        return a._conv_chars, a._cpt
    (chars, cpt), (chars0, cpt0) = run(True), run(False)
    assert abs(chars - chars0) < 100 and abs(cpt - cpt0) < 0.05  # the 20,000 refused chars no longer count (nor skew
    # the chars-per-token ratio that sizes a rebuilt conversation)


def test_system_prompt_and_base_guidance_say_reasoning_may_be_dropped(tmp_path):
    from nesicode.agent import build_system

    guidance = base_harness()["tool_guidance.md"]
    assert "Your private reasoning and old tool outputs may be dropped to save context" in guidance
    system = build_system(base_harness(), Workspace(tmp_path))
    assert "may be dropped between steps" in system and "is not kept between steps" not in system


# ====================================================================== P2 [THINKING] and the 32k cap
def test_reasoning_is_kept_up_to_32k_with_its_end_and_reaches_transcripts_and_the_summarizer(tmp_path, fake):
    long = "x" * 39_000 + " CONCLUSION: bug en streamer.ts:238 (BUG-238)"

    def policy(conv, rec, on_event):
        if rec["role"] == "summarizer":
            return Turn(SUMMARY, [], "end_turn", Usage(0, 100))
        n = len(prov.executor())
        if n <= 6:
            return Turn("leo", [ToolCall(f"b{n}", "bash", {"command": "echo " + "q" * 9_000})], "tool_use", Usage(0, 30),
                        thinking=long if n == 1 else f"paso {n}: sigo")
        return Turn("fin", [], "end_turn", Usage(0, 10))
    prov = fake(policy)
    events = []
    ws = Workspace(tmp_path / "ws")
    agent(ws, events, context_limit=16_000).run("revisa")
    first = next(e for e in events if e["type"] == "thinking")
    assert len(first["text"]) <= loopmod.THINKING_CAP + 80 and first["text"].endswith("(BUG-238)")
    comp = [e for e in events if e["type"] == "compacted"]
    assert comp
    transcript = (ws.root / comp[0]["transcript"]).read_text()
    assert "[THINKING]" in transcript and "BUG-238" in transcript
    prompt = next(r for r in prov.log if r["role"] == "summarizer")["messages"][-1]["content"]
    assert "[THINKING]" in prompt and "BUG-238" in prompt and "from reasoning, unverified" in prompt
    text = trajectory_text(events, 10**9, thinking_chars=3_000)
    line = next(ln for ln in text.splitlines() if ln.startswith("[THINKING]") and "BUG-238" in ln)
    assert len(line) < 3_200 and "earlier chars of this reasoning omitted" in line  # the END is kept
    assert "[THINKING]" not in trajectory_text(events, 10**9)  # off by default (harnessing rollouts)


# ====================================================================== DB helpers (conversation tests)
def _client():
    from fastapi.testclient import TestClient

    from nesicode.server.app import app

    c = TestClient(app)
    assert c.post("/api/auth/register", json={"email": f"p2{time.time_ns()}@x.com", "password": "secret123"}).status_code == 200
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


def _long_turn(steps=60, out_lines=300):
    ev = [{"type": "start"}]
    for i in range(steps):
        ev.append({"type": "thinking", "text": f"razonamiento del paso {i}: " + "pienso " * 400 + f"CONCLUSION-{i}"})
        name, args = ("list_dir", {"path": "."}) if i == steps - 3 else ("read_file", {"path": f"src/f{i}.py"})
        ev.append({"type": "tool_call", "id": f"c{i}", "name": name, "args": args})
        out = "\n".join(f"{n:>6}\tlinea {n} del fichero f{i}.py " + "k" * 30 for n in range(1, out_lines))
        ev.append({"type": "tool_result", "id": f"c{i}", "name": name, "output": out, "is_error": False})
    ev.append({"type": "assistant", "text": "ÚLTIMAS PALABRAS: ahora reviso streamer.ts"})
    return ev


# ====================================================================== P3 history.md
def test_history_md_keeps_every_step_and_the_reasoning_of_a_long_turn(tmp_path):
    from nesicode.server.app import db
    from nesicode.server.jobs import _write_thread_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "revisa el monorepo", "cancelled", "", _long_turn(steps=140, out_lines=400) + [{"type": "cancelled"}],
         at=now - 10)
    cur = _run(p, t, "continúa", "running", at=now)
    ws = Workspace(tmp_path / "ws")
    _write_thread_history(db, ws, cur, ".nesicode/chats/5")
    md = (ws.root / ".nesicode/chats/5/context/history.md").read_text()
    assert "middle of trajectory omitted" not in md
    assert '"src/f0.py"' in md and '"src/f139.py"' in md and md.count("[ACTION]") == 140  # no step dropped
    assert md.count("[THINKING]") == 140 and "CONCLUSION-0" in md and "CONCLUSION-139" in md  # the end of each
    assert "lines (" in md and "of this output omitted here" in md  # long outputs: head/tail lines
    assert len(md) < 1_500_000


# ====================================================================== P4 the previous turn's trail
def test_previous_turn_trail_is_tail_first_window_scaled_with_reasoning_and_no_repeated_last_words():
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "revisa el monorepo", "cancelled", "ÚLTIMAS PALABRAS: ahora reviso streamer.ts",
         _long_turn() + [{"type": "cancelled"}], at=now - 10)
    nxt = _run(p, t, "continúa", "waiting", at=now)
    db.execute("INSERT OR REPLACE INTO model_limits (user_id, spec, context, output, source, updated_at) "
               "VALUES (?, 'mock/demo', 16384, 4096, 'learned', ?)", (p["user_id"], now))
    history, interrupted = conversation_history(db, nxt)  # the window comes from the known limits (16k)
    answer = history[0][1]
    trail = answer[answer.index(loopmod.TRAIL_MARK):]
    assert len(trail) <= 16_384 * 0.25 * 3.5 + 200  # ~25% of the window for a stopped turn
    assert trail.rstrip().endswith("ÚLTIMAS PALABRAS: ahora reviso streamer.ts")  # it ends where the turn stopped
    assert "CONCLUSION-59" in trail and "[THINKING]" in trail  # its last reasoning
    assert re.search(r"the first \d+ tool call\(s\) of this turn are not shown here", trail)
    assert '"src/f0.py"' not in trail and "more lines of this listing" in trail  # oldest out; list_dir dump shrunk
    assert answer.count("ÚLTIMAS PALABRAS") == 1 and "Its last words" not in answer and "[Partial answer]" not in answer
    assert interrupted["steps"] == 60
    big, _ = conversation_history(db, nxt, window=131_072)  # a bigger window: a longer trail (more steps)
    assert len(big[0][1]) > len(answer) * 3


def test_a_finished_turn_does_not_repeat_its_answer_in_its_trail():
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    events = [{"type": "start"}, {"type": "tool_call", "id": "a", "name": "bash", "args": {"command": "ls"}},
              {"type": "tool_result", "id": "a", "name": "bash", "output": "exit code: 0\nREADME.md"},
              {"type": "assistant", "text": "INFORME FINAL: todo correcto."}, {"type": "done", "status": "done", "steps": 2}]
    _run(p, t, "revisa", "done", "INFORME FINAL: todo correcto.", events, at=now - 5)
    history, _ = conversation_history(db, _run(p, t, "gracias", "waiting", at=now), window=32_768)
    assert history[0][1].count("INFORME FINAL") == 1 and loopmod.TRAIL_MARK in history[0][1]


def test_trimming_an_earlier_turn_keeps_the_newest_steps_of_its_trail():
    text = "Respuesta.\n\n" + loopmod.TRAIL_MARK + "\n" + "\n".join(f"[ACTION] paso {i}" for i in range(2_000))
    out = loopmod._trim_answer(text, 6_000)
    assert len(out) <= 6_000 and out.startswith("Respuesta.") and out.rstrip().endswith("[ACTION] paso 1999")
    assert "older steps of this turn omitted" in out and "[ACTION] paso 0\n" not in out


# ====================================================================== P5 masking notice
def test_masking_tells_the_model_once_how_many_outputs_were_cleared(tmp_path, fake):
    def policy(conv, rec, on_event):
        n = len(prov.executor())
        if n >= 14:
            return Turn("fin", [], "end_turn", Usage(0, 10))
        return Turn("leo", [ToolCall(f"r{n}", "read_file", {"path": f"f{n}.txt"})], "tool_use", Usage(0, 20))
    prov = fake(policy)
    ws = Workspace(tmp_path / "ws")
    for i in range(14):
        (ws.root / f"f{i}.txt").write_text("\n".join(f"dato {i}-{j}" for j in range(400)))
    events = []
    agent(ws, events, context_limit=40_000).run("lee todo")
    masked = [e for e in events if e["type"] == "masked"]
    assert masked
    notes = [m["content"] for m in prov.executor()[-1]["messages"] if m["role"] == "user" and "to free context" in m["content"]]
    assert len(notes) == len(masked)  # one line per masking, no more
    n = int(re.match(r"\[harness\] (\d+) earlier tool outputs? (?:was|were) cleared", notes[0]).group(1))
    first_after = next(r for r in prov.executor() if any("to free context" in (m["content"] or "") for m in r["messages"]))
    assert n == sum(t.startswith("[old ") for t in tool_texts(first_after)) and n > 0
    assert "Record any conclusion you have not written down yet" in notes[0] and "before re-reading" in notes[0]


# ====================================================================== P6 progress nudge
RULES = ("Revisa el monorepo. Reglas: (1) no modifiques nada salvo REVIEW.md; (2) formato WR-<n> | archivo:línea; "
         "(3) AÑADE CADA HALLAZGO A REVIEW.md EN CUANTO LO ENCUENTRES; (4) no instales nada; (5) en español.")


def test_progress_reminder_quotes_the_users_rules_and_comes_again_after_a_masking(tmp_path, fake):
    rnd = random.Random(5)

    def policy(conv, rec, on_event):
        n = len(prov.executor())
        if n >= 45:
            return Turn("fin", [], "end_turn", Usage(0, 10))
        return Turn("leo", [ToolCall(f"r{n}", "read_file", {"path": f"src/f{n % 9}.py", "offset": 1 + n})], "tool_use",
                    Usage(0, 20))
    prov = fake(policy)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "src").mkdir()
    for i in range(9):
        (ws.root / "src" / f"f{i}.py").write_text(code(rnd, 9_000))
    events = []
    history = [(RULES + " " + "contexto " * 300, "Entendido.")]
    agent(ws, events, context_limit=32_768, max_steps=80).run("continúa", history)
    reminded = {}  # tool_call_id -> text
    for rec in prov.executor():
        for m in rec["messages"]:
            if m["role"] == "tool" and "tool calls in a row without writing" in m["content"]:
                reminded.setdefault(m["tool_call_id"], m["content"])
    ids = sorted(reminded, key=lambda cid: int(cid[1:]))
    assert len(ids) >= 2 and ids[0] == "r20"  # the 20th read-only call
    assert all(int(b[1:]) - int(a[1:]) >= loopmod.READ_STREAK // 2 for a, b in zip(ids, ids[1:]))  # never back to back
    first = reminded[ids[0]]
    note = first[first.index("[harness]"):]
    assert "Write what you have now" in note and "keep results on disk as you go" in note
    assert "AÑADE CADA HALLAZGO A REVIEW.md" in note and "current message: «continúa»" in note
    quoted = re.findall(r"«(.*?)»", note, re.S)
    assert sum(len(q) for q in quoted) <= loopmod.REQUEST_QUOTE
    # the second one came after a masking (or compaction) while the read-only streak went on
    order = [e for e in events if e["type"] in ("tool_call", "masked", "compacted") and not e.get("depth")]
    at = next(i for i, e in enumerate(order) if e.get("id") == ids[1])
    first_at = next(i for i, e in enumerate(order) if e.get("id") == ids[0])
    assert any(e["type"] in ("masked", "compacted") for e in order[first_at:at])


# ====================================================================== P8 digest noise
def test_digest_takes_http_codes_only_from_status_lines_of_network_tools():
    from nesicode.agent.context import run_digest

    events = []
    for i, (name, args, out) in enumerate([
        ("read_file", {"path": "queue.ts"}, "   12\tif (response.status === 401) retry();\n   13\t\"status\": 404,"),
        ("bash", {"command": "grep -rn http src/ | head"}, "exit code: 0\nsrc/a.ts:3: HTTP/1.1 500 in a comment"),
        ("bash", {"command": "cat fixtures/resp.json"}, 'exit code: 0\n{"status": 503, "statusCode": 502}'),
        ("bash", {"command": "cd app && TOKEN=x curl -s -i http://127.0.0.1:3000/health"}, "exit code: 0\nHTTP/1.1 418 I'm a teapot"),
        ("bash", {"command": "curl -s http://127.0.0.1:3000/api"}, 'exit code: 0\n{"status": 500}'),
        ("mcp__burp__send_http1_request", {"url": "https://vulnbank.org"}, "HTTP/2 302 Found\nlocation: /login"),
    ]):
        events += [{"type": "tool_call", "id": str(i), "name": name, "args": args},
                   {"type": "tool_result", "id": str(i), "name": name, "output": out}]
    d = run_digest(events)
    tags = re.findall(r"HTTP (\d{3})", d.split("Identifiers seen")[0].replace("HTTP/1.1", "").replace("HTTP/2", ""))
    assert sorted(set(tags)) == ["302", "418"]
    assert "exit code 0" in d


# ====================================================================== P9 search_history ranking
def test_search_history_fallback_ranks_by_distinct_words_and_prefers_the_agents_words(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ctx = ws.root / ".nesicode/chats/4/context"
    out = ws.root / ".nesicode/chats/4/outputs"
    ctx.mkdir(parents=True)
    out.mkdir(parents=True)
    lines = ["## Turno #1 (cancelled)", "### Pasos", "[ACTION] list_dir {\"path\": \"packages\"}", "[OBSERVATION] packages/"]
    lines += [f"  world-postgres/src/file{i}.ts (12 B)" for i in range(60)]
    lines += ["[ASSISTANT] Hallazgo: en world-postgres, invocations no libera el job-lease al fallar"]
    (ctx / "history.md").write_text("\n".join(lines))
    (out / "1-bash-grep.txt").write_text("world-postgres/src/invocations.ts:12: const lease = jobLease() // job-lease\n")
    res = ctxm.search_history(ws, {"query": "WR- hallazgos REVIEW.md world-postgres invocations job-lease"},
                              ".nesicode/chats/4")
    hits = [ln for ln in res.splitlines() if re.match(r"\.nesicode/chats/4/\S+:\d+: ", ln)]
    assert "lines ranked by how many of the words they contain" in res and len(hits) == ctxm.RANKED_HITS
    assert "[ASSISTANT] Hallazgo" in hits[0] and hits[0].startswith(".nesicode/chats/4/context/history.md:")
    assert "invocations.ts:12" in hits[1]  # same words, but a tool-output dump: after the agent's own line


# ====================================================================== P10 read_file states its budget
def test_read_file_description_states_the_per_call_budget(tmp_path):
    def desc(limit, harness=None):
        a = Agent(model="mock/demo", harness=harness or base_harness(), workspace=Workspace(tmp_path), context_limit=limit)
        return next(s.description for s in a.tool_specs() if s.name == "read_file")
    assert "about 4,000 characters" in desc(16_384) and "stops at a line boundary" in desc(32_768)
    assert "offset" in desc(32_768)
    assert "about 24,000 characters" in desc(131_072)  # the base harness's 20k hook for every tool spares read_file
    hooks = lambda rules: dict(base_harness(), **{"plugins/hooks.json": json.dumps(rules)})  # noqa: E731
    assert "about 24,000 characters" in desc(131_072, hooks({}))
    assert "about 4,400 characters" in desc(131_072, hooks({"post_tool": [{"tool": "read_file", "action": "truncate",
                                                                          "max_chars": 5_000}]}))


# ====================================================================== P11 small windows
@pytest.mark.parametrize("window,most", [(16_384, 7), (32_768, 2)])
def test_small_windows_compact_rarely_and_rebuild_to_about_a_third(tmp_path, fake, window, most):
    rnd = random.Random(3)

    def policy(conv, rec, on_event):
        if rec["role"] == "summarizer":
            return Turn("## Goal\nRefactor.\n## Constraints & Preferences\n- NUNCA toques config/prod.yml\n## Next Step\n"
                        "Continue.", [], "end_turn", Usage(0, 300))
        n = len(prov.executor())
        if n >= 40:
            return Turn("Hecho.", [], "end_turn", Usage(0, 10))
        if n % 6 == 5:
            return Turn("Escribo.", [ToolCall(f"w{n}", "write_file", {"path": f"out/r{n}.py", "content": code(rnd, 2_500)})],
                        "tool_use", Usage(0, 700))
        return Turn("Leo.", [ToolCall(f"r{n}", "read_file", {"path": f"src/m{n % 7}.py", "offset": 1 + 40 * (n // 7)})],
                    "tool_use", Usage(0, 60))
    prov = fake(policy, window=window, cpt=3.3)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "src").mkdir()
    for i in range(7):
        (ws.root / "src" / f"m{i}.py").write_text(code(rnd, 20_000))
    events = []
    res = agent(ws, events, context_limit=window, max_steps=200).run(
        "sigue con el refactor", [("Regla: NUNCA toques config/prod.yml. Prepara el plan.", "Entendido. " + "detalle " * 300)])
    assert res.status == "done" and not [r for r in prov.log if r["rejected"]]
    steps, at = 0, []
    for e in events:
        steps += e["type"] == "tool_call"
        if e["type"] == "compacted":
            at.append(steps)
    assert len(at) <= most, at  # phase 1: 15 compactions in 40 steps at 16k (every 2 steps)
    assert all(b - a >= 4 for a, b in zip(at, at[1:])), at
    assert all(e["after"] <= window * 0.36 for e in events if e["type"] == "compacted")


# ====================================================================== P12 interrupted replies
class _Streaming(Conversation):
    """Streams scripted deltas through on_event; then the script decides: stop (sets cancel), fail, or retry."""

    def __init__(self, script, agent_ref, *a):
        super().__init__(*a)
        self.script, self.agent_ref = script, agent_ref

    def add_user(self, text):
        pass

    def add_assistant_text(self, text):
        pass

    def add_tool_results(self, results):
        pass

    def step(self, on_event=None):
        for item in self.script:
            if item == "STOP":
                self.agent_ref[0].cancel.set()
            elif item == "FAIL":
                raise LLMError("fake request failed: connection lost mid-reply (after 5 attempts)")
            elif item == "RETRY":
                self.options.notify({"type": "llm_retry", "attempt": 2, "max": 5, "wait": 0, "reason": "stalled",
                                     "partial": True})
            else:
                on_event(*item)
        return Turn("nunca", [], "end_turn", Usage(10, 10))


def _streaming_run(tmp_path, monkeypatch, script):
    ref = [None]

    class P(Provider):
        name = "fake"

        def conversation(self, model, system, tools, options):
            return _Streaming(script, ref, model, system, tools, options)
    monkeypatch.setattr(loopmod, "get_provider", lambda spec, keys=None: (P(), "m"))
    events = []
    ref[0] = agent(Workspace(tmp_path / "ws"), events, context_limit=64_000)
    return ref[0].run("busca el bug"), [{k: v for k, v in e.items() if k != "_persist"} for e in events]


def test_a_reply_cut_by_stop_is_persisted_before_the_cancelled_event(tmp_path, monkeypatch):
    res, events = _streaming_run(tmp_path, monkeypatch, [
        ("thinking", "El bug está en streamer.ts:238 "), ("text", "Encontré un bug "), ("text", "en streamer.ts"),
        "STOP", ("text", " (no llega)")])
    assert res.status == "cancelled"
    kinds = [e["type"] for e in events]
    i = kinds.index("cancelled")
    assert events[i - 1] == {"type": "assistant", "text": "Encontré un bug en streamer.ts", "interrupted": True}
    assert events[i - 2] == {"type": "thinking", "text": "El bug está en streamer.ts:238 ", "interrupted": True}
    # the next turn knows it
    d = ctxm.run_digest(events)
    assert "Its last words (cut off mid-reply when the turn was stopped): Encontré un bug en streamer.ts" in d
    assert "[ASSISTANT] (interrupted mid-reply) Encontré un bug en streamer.ts" in trajectory_text(events)


def test_a_reply_cut_by_an_error_is_persisted_and_a_retry_restarts_it(tmp_path, monkeypatch):
    res, events = _streaming_run(tmp_path, monkeypatch, [
        ("text", "versión que se reintenta"), "RETRY", ("text", "Segunda versión, a medias"), "FAIL"])
    assert res.status == "error"
    kinds = [e["type"] for e in events]
    i = kinds.index("error")
    assert events[i - 1] == {"type": "assistant", "text": "Segunda versión, a medias", "interrupted": True}
    assert not any(e["type"] == "assistant" and "se reintenta" in e["text"] for e in events)


def test_a_complete_reply_is_never_duplicated_as_interrupted(tmp_path, fake):
    def policy(conv, rec, on_event):
        on_event("text", "hola")
        return Turn("hola", [], "end_turn", Usage(0, 5))
    fake(policy)
    events = []
    agent(Workspace(tmp_path / "ws"), events, context_limit=32_000).run("saluda")
    assert not any(e.get("interrupted") for e in events)


def test_the_next_turn_gets_the_interrupted_text_from_the_history():
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "busca el bug", "cancelled", "", [
        {"type": "start"}, {"type": "tool_call", "id": "a", "name": "read_file", "args": {"path": "streamer.ts"}},
        {"type": "tool_result", "id": "a", "name": "read_file", "output": "   238\tconst x = y;"},
        {"type": "assistant", "text": "Encontré un bug en streamer.ts:238", "interrupted": True},
        {"type": "cancelled"}], at=now - 5)
    history, _ = conversation_history(db, _run(p, t, "continúa", "waiting", at=now), window=32_768)
    assert "[ASSISTANT] (interrupted mid-reply) Encontré un bug en streamer.ts:238" in history[0][1]


def test_a_turn_stopped_mid_answer_without_tools_still_reaches_the_next_turn():
    from nesicode.server.app import db
    from nesicode.server.jobs import continuity_note, conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "explícame el diseño", "cancelled", "", [
        {"type": "start"}, {"type": "assistant", "text": "El diseño tiene tres capas: la primera", "interrupted": True},
        {"type": "cancelled"}], at=now - 5)
    history, interrupted = conversation_history(db, _run(p, t, "continúa", "waiting", at=now))
    assert "before doing any work" not in history[0][1]
    assert "Its last words (cut off mid-reply when the turn was stopped): El diseño tiene tres capas" in history[0][1]
    assert interrupted and "do not start over" in continuity_note(t["id"], interrupted)


# ====================================================================== P13 hooks on error results
def test_every_error_result_goes_through_the_hooks(tmp_path):
    h = dict(base_harness(), **{"plugins/hooks.json": json.dumps({
        "pre_tool": [{"tool": "*", "action": "warn", "message": "PRE-WARN"}],
        "post_tool": [{"tool": "*", "match": "rejected|did not allow|Error", "action": "append", "message": "HINT"}]})})
    ws = Workspace(tmp_path / "ws")
    a = Agent(model="mock/demo", harness=h, workspace=ws, extra_tools={
        "boom": (None, lambda args, ag: (_ for _ in ()).throw(RuntimeError("tool bug")))})
    from nesicode.llm import Usage as U

    res = a._call_tool(ToolCall("1", "boom", {}), U())  # a tool bug (generic exception)
    assert res.is_error and "PRE-WARN" in res.content and "RuntimeError: tool bug" in res.content and "[hint] HINT" in res.content
    res = a._call_tool(ToolCall("2", "write_file", {}, parse_error="invalid JSON arguments"), U())
    assert res.is_error and "[hint] HINT" in res.content
    a.approve = lambda call: (False, "no")
    res = a._call_tool(ToolCall("3", "bash", {"command": "ls"}), U())
    assert res.is_error and "PRE-WARN" in res.content and "[hint] HINT" in res.content


# ====================================================================== P14 background processes
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return bool(out) and not out.startswith("Z")
    except OSError:
        return True


def _wait(cond, timeout=5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


@posix_only
def test_background_processes_end_with_the_top_level_run_only(tmp_path, fake):
    def policy(conv, rec, on_event):
        if not any(m["role"] == "tool" for m in rec["messages"]):
            return Turn("arranco", [ToolCall("b1", "bash", {"command": "sleep 30 >/dev/null 2>&1 & echo $!"})], "tool_use",
                        Usage(0, 10))
        return Turn("listo", [], "end_turn", Usage(0, 10))
    fake(policy)
    events = []
    ws = Workspace(tmp_path / "ws")
    agent(ws, events, context_limit=32_000).run("arranca un servidor")
    pid = int(next(e for e in events if e["type"] == "tool_result")["output"].split()[-1])
    assert _wait(lambda: not _alive(pid)), "the run ended but its background process lives on"

    events = []
    ws2 = Workspace(tmp_path / "ws2")
    agent(ws2, events, context_limit=32_000, depth=1, label="sub").run("arranca")  # a sub-agent: the parent goes on
    pid2 = int(next(e for e in events if e["type"] == "tool_result")["output"].split()[-1])
    try:
        assert _alive(pid2)
    finally:
        ws2.kill_background()
    assert _wait(lambda: not _alive(pid2))


# ====================================================================== P15 planted links in the archive
@posix_only
def test_search_history_never_follows_links_planted_in_the_archive(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRETO-FUERA de la jaula\n")
    ws = Workspace(tmp_path / "ws")
    mine = ws.root / ".nesicode/chats/1"
    (mine / "outputs").mkdir(parents=True)
    (mine / "outputs" / "propio.txt").write_text("SECRETO-PROPIO de este chat\n")
    (mine / "outputs" / "leak.txt").symlink_to(outside / "secret.txt")
    (mine / "context").symlink_to(outside, target_is_directory=True)  # a whole folder planted as a link
    other = ws.root / ".nesicode/chats/2/context"
    other.mkdir(parents=True)
    (other / "history.md").write_text("SECRETO-OTRO-CHAT\n")
    (mine / "outputs" / "cross.txt").symlink_to(other / "history.md")
    found = ctxm.search_history(ws, {"query": "SECRETO"}, ".nesicode/chats/1")
    assert "SECRETO-PROPIO" in found and "FUERA" not in found and "OTRO-CHAT" not in found
    # the conversation's archive itself replaced by a link to another chat's archive
    alias = ws.root / ".nesicode/chats/3"
    alias.symlink_to(ws.root / ".nesicode/chats/2", target_is_directory=True)
    assert "OTRO-CHAT" not in ctxm.search_history(ws, {"query": "SECRETO"}, ".nesicode/chats/3")


@posix_only
def test_archive_writes_and_checkpoint_reads_never_go_through_links(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "src").mkdir()
    (ws.root / "src" / "app.py").write_text("PORT = 1\n")
    ctx = ws.root / ".nesicode/chats/1/context"
    ctx.mkdir(parents=True)
    (ctx / "history.md").symlink_to(ws.root / "src" / "app.py")  # the next turn's history.md would overwrite app.py
    with pytest.raises(Exception):
        ctxm.write_context_file(ws, "history.md", "HISTORIAL", ".nesicode/chats/1")
    assert (ws.root / "src" / "app.py").read_text() == "PORT = 1\n"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "conf.py").write_text("FUERA = 1\n")
    (ws.root / "lib").symlink_to(outside, target_is_directory=True)
    block = ctxm.files_block(ws, [("lib/conf.py", "read"), ("src/app.py", "edited")], 10_000)
    assert "PORT = 1" in block and "FUERA" not in block
    (ws.root / ".nesicode/chats/1/outputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(Exception):
        ctxm.save_output(ws, "x", "bash", "SALIDA", ".nesicode/chats/1", "x")  # never written through the planted link
    assert [p.name for p in outside.iterdir()] == ["conf.py"]


# ====================================================================== P16 connector context
def test_connector_context_uses_the_users_words_and_connector_tools_only():
    from nesicode.connectors import mentioned
    from nesicode.server.app import db
    from nesicode.server.jobs import connector_context, conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "revisa la web de vulnbank.org", "done", "Hecho. Si quieres un escaneo activo, pídeme que use Burp.", [
        {"type": "start"}, {"type": "tool_call", "id": "1", "name": "search", "args": {"pattern": "x"}},
        {"type": "tool_call", "id": "2", "name": "read_file", "args": {"path": "a"}},
        {"type": "tool_call", "id": "3", "name": "mcp__vt__vt_domain_report", "args": {"domain": "vulnbank.org"}}],
        at=now - 10)
    nxt = _run(p, t, "vale, sigue", "waiting", at=now)
    history, _ = conversation_history(db, nxt)
    ctx = connector_context(db, nxt, history)
    assert "Burp" not in ctx and not mentioned(ctx, name="burp", label="Burp Suite", template="burp")
    assert "mcp__vt__vt_domain_report" in ctx and "read_file" not in ctx.split() and "search" not in ctx.split()
    assert "revisa la web de vulnbank.org" in ctx and "vale, sigue" in ctx
    asked = _run(p, t, "usa burp para escanearla", "done", "ok", [{"type": "start"}], at=now - 5)
    nxt2 = _run(p, t, "sigue", "waiting", at=now + 1)
    ctx2 = connector_context(db, nxt2, conversation_history(db, nxt2)[0])
    assert asked["task"] in ctx2 and mentioned(ctx2, name="burp", label="Burp Suite", template="burp")


# ====================================================================== P17 mock provider masking
def test_mock_conversation_masks_with_the_budgeted_signature():
    conv = MockConversation("m", "", [], ModelOptions())
    conv.add_user("go")
    for i in range(5):
        conv.messages.append(("assistant", f"paso {i}"))
        conv.add_tool_results([ToolResult(f"c{i}", "y" * 2_000)])
    marker = ctxm.mask_marker("read_file", ".nesicode/outputs/x.txt", False)
    before = list(conv.messages)
    assert conv.mask_tool_results(0, marker, apply=False) > 0 and conv.messages == before
    freed = conv.mask_tool_results(1, marker, keep_chars=4_500)
    tools = [m[1] for m in conv.messages if m[0] == "tool"]
    assert tools[-1] == "y" * 2_000 and tools[-2] == "y" * 2_000  # pending + one by size (keep_chars)
    assert all(t.startswith("[old read_file output cleared") for t in tools[:3]) and freed > 5_000
    assert not hasattr(loopmod, "inspect")  # the loop's old-signature fallback is gone


# ====================================================================== P18 Claude stalls
def aev(name, data):
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


A_START = aev("message_start", {"type": "message_start", "message": {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "content": [], "stop_reason": None,
    "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}})
A_END = (aev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": {"output_tokens": 5}}) + aev("message_stop", {"type": "message_stop"}))
PING = aev("ping", {"type": "ping"})


def a_text(text):
    return (aev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
            + aev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                          "delta": {"type": "text_delta", "text": text}})
            + aev("content_block_stop", {"type": "content_block_stop", "index": 0}))


def claude(s, **opts):
    from nesicode.llm.anthropic_provider import AnthropicProvider

    conv = AnthropicProvider(api_key="sk-ant-test", base_url=s.url, isolated=True).conversation(
        "claude-sonnet-5-5", "sys", [], ModelOptions(max_tokens=200, **opts))
    conv.add_user("hola")
    return conv


def test_a_stalled_claude_stream_is_aborted_and_retried(server, monkeypatch):
    monkeypatch.setenv("NESICODE_LLM_IDLE_TIMEOUT", "0.6")
    monkeypatch.setenv("NESICODE_LLM_FIRST_BYTE_TIMEOUT", "0.6")
    s = server(sse(A_START, a_text("a medias"), then="stall"), sse(A_START, a_text("respuesta completa"), A_END))
    notes = []
    t0 = time.monotonic()
    turn = claude(s, notify=notes.append).step(on_event=lambda kind, text: None)
    assert turn.text == "respuesta completa" and time.monotonic() - t0 < 6 and len(s.requests) == 2
    assert len(notes) == 1 and "stalled" in notes[0]["reason"] and notes[0]["partial"] is True
    assert _wait(lambda: bool(s.disconnects), 3), "the stalled connection was not closed"


def test_claude_pings_keep_a_slow_reply_alive(server, monkeypatch):
    monkeypatch.setenv("NESICODE_LLM_IDLE_TIMEOUT", "0.6")
    monkeypatch.setenv("NESICODE_LLM_FIRST_BYTE_TIMEOUT", "0.6")
    s = server(sse(A_START, 0.25, PING, 0.25, PING, 0.25, PING, 0.25, PING, 0.25, PING, a_text("ok"), A_END))
    assert claude(s).step().text == "ok" and len(s.requests) == 1  # 1.25 s with pings only: not a stall


def test_claude_that_never_answers_is_retried_then_fails(server, monkeypatch):
    monkeypatch.setenv("NESICODE_LLM_FIRST_BYTE_TIMEOUT", "0.5")
    s = server(hang, hang)
    t0 = time.monotonic()
    with pytest.raises(LLMError, match="did not start"):
        claude(s, max_attempts=2).step()
    assert time.monotonic() - t0 < 6 and len(s.requests) == 2


def test_stop_during_a_claude_stall_returns_at_once(server, monkeypatch):
    monkeypatch.setenv("NESICODE_LLM_IDLE_TIMEOUT", "30")
    s = server(sse(A_START, a_text("pensando"), then="stall"))
    cancel = threading.Event()
    threading.Timer(0.4, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(LLMError, match="cancelled"):
        claude(s, cancel=cancel).step(on_event=lambda kind, text: None)
    assert time.monotonic() - t0 < 2


# ====================================================================== review fixes
class _Scripted(OpenAICompatProvider):
    """The real OpenAI-compatible conversation, its server scripted: every reply reasons and reads a file."""

    def __init__(self, replies: int):
        super().__init__("nesilabs", "http://127.0.0.1:9", "k")
        self.replies, self.bodies = replies, []

    def chat(self, body, options, on_event=None):
        self.bodies.append(json.loads(json.dumps(body["messages"])))
        n = len(self.bodies)
        usage = {"prompt_tokens": sum(len(json.dumps(m)) for m in body["messages"]) // 3 + 3_000, "completion_tokens": 50}
        if n > self.replies:
            return {"choices": [{"message": {"role": "assistant", "content": "fin"}, "finish_reason": "stop"}],
                    "usage": usage}, False
        call = {"id": f"c{n}", "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": f"f{n % 5}.txt"})}}
        msg = {"role": "assistant", "content": "", "reasoning_content": f"RAZONAMIENTO-{n} " + "pienso " * 300,
               "tool_calls": [call]}
        return {"choices": [{"message": msg, "finish_reason": "tool_calls"}], "usage": usage}, False


def test_masking_notice_rides_with_the_newest_results_so_replayed_reasoning_survives(tmp_path, monkeypatch):
    # Qwen3/GLM-style chat templates keep the reasoning of the assistant messages after the LAST user message only: a
    # notice sent as a new user turn would make the server drop every reasoning replayed so far, at each masking.
    prov = _Scripted(12)
    monkeypatch.setattr(loopmod, "get_provider", lambda spec, keys=None: (prov, "nesi3"))
    ws = Workspace(tmp_path / "ws")
    for i in range(5):
        (ws.root / f"f{i}.txt").write_text("\n".join(f"line {i}-{j} " + "data " * 10 for j in range(300)))
    events = []
    agent(ws, events, context_limit=40_000, output_limit=8_000).run("lee los ficheros", [("reglas", "vale")])
    assert any(e["type"] == "masked" for e in events)
    noticed = [b for b in prov.bodies if any("to free context" in str(m.get("content")) for m in b)]
    assert noticed
    for body in noticed:
        last_user = max(i for i, m in enumerate(body) if m["role"] == "user")
        assert "to free context" not in str(body[last_user]["content"])  # never a user turn of its own
        assert body[last_user]["content"] == "lee los ficheros"
        kept = [m for m in body[last_user + 1:] if m["role"] == "assistant" and m.get("reasoning_content")]
        assert len(kept) >= 2  # what an interleaved-thinking template still renders
    first = noticed[0]
    assert first[-1]["role"] == "tool" and "[harness]" in first[-1]["content"]  # with the newest (unseen) results
    conv = OpenAICompatConversation(OpenAICompatProvider("x", "http://127.0.0.1:9"), "m", "s", [], ModelOptions())
    conv.add_user("hola")
    conv.add_note("[harness] nota")  # no tool results to ride with: a user message
    assert conv.messages[-1] == {"role": "user", "content": "[harness] nota"}


def test_a_long_single_line_output_keeps_its_head_and_tail_in_trajectories():
    data = json.dumps({"items": [{"id": i, "name": f"item-{i}"} for i in range(500)]})
    for out in (data + "\n", "exit code: 0\n" + data + "\n", data):
        cut = loopmod._shorten(out, 2_500)
        assert len(cut) <= 2_600 and "item-0" in cut and "item-499" in cut and "omitted here" in cut
    events = [{"type": "tool_call", "id": "1", "name": "bash", "args": {"command": "curl -s https://api.x/items"}},
              {"type": "tool_result", "id": "1", "name": "bash", "output": "exit code: 0\n" + data + "\n"}]
    for text in (trajectory_text(events), trajectory_text(events, 10_000, fit="tail"),
                 trajectory_text(events, 10_000, fit="shrink")):
        assert "item-0" in text and "item-499" in text  # (harnessing rollouts, the trail, history.md)
    lines = "\n".join(f"{n:>6}\tlinea {n}" for n in range(1, 2_000))
    assert re.search(r"\[\.\.\. [\d,]+ lines \([\d,]+ chars\) of this output omitted here \.\.\.\]", loopmod._shorten(lines, 2_500))


def test_digest_sees_curl_behind_timeout_loops_and_wrappers():
    from nesicode.agent.context import run_digest

    commands = ["timeout 20 curl -sI https://a.org", "for h in a b; do curl -sI https://$h.org; done",
                "if curl -sI https://c.org; then echo up; fi", "xargs -n1 curl -sI < urls.txt", "env A=1 curl -I https://d.org"]
    for i, cmd in enumerate(commands):
        d = run_digest([{"type": "tool_call", "id": "1", "name": "bash", "args": {"command": cmd}},
                        {"type": "tool_result", "id": "1", "name": "bash", "output": f"exit code: 0\nHTTP/2 20{i}\nserver: x"}])
        assert f"HTTP 20{i}" in d, cmd
    d = run_digest([{"type": "tool_call", "id": "1", "name": "bash", "args": {"command": "wget -S -O /dev/null https://e.org"}},
                    {"type": "tool_result", "id": "1", "name": "bash",
                     "output": "exit code: 0\n--2026-11-08--  https://e.org/\n  HTTP/1.1 301 Moved\n  HTTP/1.1 200 OK"}])
    assert "HTTP 200 301" in d  # wget -S indents its status lines
    d = run_digest([{"type": "tool_call", "id": "1", "name": "bash", "args": {"command": "cat fixtures/raw.txt"}},
                    {"type": "tool_result", "id": "1", "name": "bash", "output": "exit code: 0\nHTTP/1.1 500 Oops"}])
    assert "HTTP 500" not in d  # a file that holds a raw response is not a network call


@posix_only
def test_archive_listing_without_dir_fds_refuses_a_link_pointing_out_of_the_workspace(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("SECRETO-FUERA\n")
    ws = Workspace(tmp_path / "ws")
    mine = ws.root / ".nesicode/chats/1"
    (mine / "outputs").mkdir(parents=True)
    (mine / "outputs" / "propio.txt").write_text("propio\n")
    (mine / "context").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(Workspace, "open_dir", lambda self, p: None)  # the Windows path: no directory fds
    files = [ws.rel(p) for p in ctxm._archive_files(ws, ".nesicode/chats/1")]
    assert files == [".nesicode/chats/1/outputs/propio.txt"]


def test_a_turn_stopped_while_still_reasoning_brings_the_end_of_its_reasoning():
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "revisa el monorepo", "cancelled", "", _long_turn(steps=8) + [{"type": "cancelled"}], at=now - 10)
    _run(p, t, "continúa", "cancelled", "", [
        {"type": "start"},
        {"type": "thinking", "text": "x" * 5_000 + " el bug está en queue.ts:413 (BUG-413)", "interrupted": True},
        {"type": "cancelled"}], at=now - 5)
    history, interrupted = conversation_history(db, _run(p, t, "sigue", "waiting", at=now), window=32_768)
    worked, thinking = history[0][1], history[1][1]
    assert loopmod.TRAIL_MARK in worked and interrupted["steps"] == 8  # the turn that worked keeps the trail slot
    assert interrupted.get("later_empty") == 1
    assert thinking.startswith("[this turn was cancelled while still reasoning, before doing any work")
    assert thinking.endswith("(BUG-413)") and len(thinking) < 1_800 and "unverified" in thinking


def test_a_trail_whose_last_reply_alone_is_too_big_keeps_where_it_stopped():
    events = [{"type": "tool_call", "id": "1", "name": "read_file", "args": {"path": "a"}},
              {"type": "tool_result", "id": "1", "name": "read_file", "output": "ok"},
              {"type": "assistant", "text": "INICIO " + "informe " * 3_000 + "DONDE-PARO", "interrupted": True}]
    trail = trajectory_text(events, 5_000, fit="tail")
    assert len(trail) <= 5_000 and trail.endswith("DONDE-PARO")
    assert "[ASSISTANT] (interrupted mid-reply) INICIO" in trail and "first 1 tool call(s)" in trail
