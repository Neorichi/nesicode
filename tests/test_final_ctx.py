"""Final review, area ctx: context management, the history archive, the LLM layer and the base harness's hook.

Offline: fake providers, scripted HTTP servers on 127.0.0.1 (ephemeral ports), temp workspaces and the test database.
A regex that would freeze the interpreter runs in a child process with a timeout, so a regression fails instead of
hanging the suite."""

from __future__ import annotations

import base64
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from test_p2_context import (DONE, _client, _run, agent, code, fake, hang, json_reply, oc, quick_retries,  # noqa: F401
                             server, sse)

from nesicode.agent import Agent, Workspace
from nesicode.agent import context as ctxm
from nesicode.agent.hooks import Hooks
from nesicode.agent.tools import read_file
from nesicode.harness import base_harness
from nesicode.llm import LLMError, ModelOptions, build_provider, fetch_limits, openai_compat, parse_limit, transport
from nesicode.llm.base import ToolCall, Turn, Usage
from nesicode.llm.openai_compat import OpenAICompatProvider

ARCHIVE = ".nesicode/chats/1"
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
USAGE = {"prompt_tokens": 10, "completion_tokens": 5}


# ====================================================================== a scripted server for GET and POST
class Srv:
    """Each request gets the next handler of `script` (then `default`); handlers are fn(req, body) like the ones of
    test_p2_context (sse, json_reply, hang). Requests are recorded with their method, path and arrival time."""

    def __init__(self, *script, default=None):
        self.script, self.default = list(script), default
        self.requests: list[dict] = []
        self.disconnects: list[float] = []
        self.closing = threading.Event()
        me = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _any(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = None
                me.requests.append({"method": self.command, "path": self.path, "body": body, "t": time.monotonic()})
                fn = me.script.pop(0) if me.script else me.default
                (fn or json_reply(404, {"error": {"message": "not found"}}))(self, body)

            do_GET = do_POST = _any

        class S(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass

        self.httpd = S(("127.0.0.1", 0), H)
        self.httpd.fake = self
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.closing.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def srv():
    made = []

    def make(*script, default=None):
        made.append(Srv(*script, default=default))
        return made[-1]
    yield make
    for s in made:
        s.close()


def tool_call_reply(name, args, cid="c1"):
    return sse(oc({"role": "assistant", "tool_calls": [{"index": 0, "id": cid, "type": "function",
                                                         "function": {"name": name, "arguments": json.dumps(args)}}]}),
               oc(finish="tool_calls"), oc(usage=USAGE), DONE)


def answer(text, finish="stop"):
    return sse(oc({"role": "assistant", "content": text}), oc(finish=finish), oc(usage=USAGE), DONE)


# ====================================================================== ctx#1 / security#3: search_history's regex
def _search_in_child(root, query, timeout=20.0):
    """search_history in a child process: (seconds the search took, its output); None if it was still running."""
    import nesicode

    code_ = ("import sys, time, json\nfrom nesicode.agent import Workspace\nfrom nesicode.agent import context as c\n"
             "t0 = time.monotonic()\nout = c.search_history(Workspace(sys.argv[1]), {'query': sys.argv[2]}, sys.argv[3])\n"
             "print(json.dumps([time.monotonic() - t0, out]))\n")
    src = os.path.dirname(os.path.dirname(nesicode.__file__))  # the child imports the very code under test
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}
    try:
        p = subprocess.run([sys.executable, "-c", code_, str(root), query, ARCHIVE], capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


JSON_LINE = '{"items":[' + ",".join('{"id":%d,"name":"item-%d"}' % (i, i) for i in range(6_000)) + "]}"


@pytest.mark.parametrize("query,line", [
    (r"(\w+\s?)+token", "revisa la configuración del despliegue de staging hoy"),  # the shape `search` refuses
    (r"(a+)+$", "a" * 40 + "!"),  # exponential: hours on 40 chars
    (r".*staging_token.*", JSON_LINE),  # quadratic on one long line (a saved MCP / chrome_* result is one-line JSON)
    (r".*password", JSON_LINE),
])
def test_search_history_never_freezes_on_a_regex(tmp_path, query, line):
    out = tmp_path / "ws" / ARCHIVE / "outputs"
    out.mkdir(parents=True)
    (out / "1-mcp__api-c1.txt").write_text(line + "\n")
    res = _search_in_child(tmp_path / "ws", query)
    assert res is not None, f"search_history({query!r}) was still running after 20 s"
    took, text = res
    assert took < 5, (took, text)


def test_search_history_still_finds_the_line_and_shows_where_it_matched(tmp_path):
    ws = Workspace(tmp_path / "ws")
    out = ws.root / ARCHIVE / "outputs"
    out.mkdir(parents=True)
    line = JSON_LINE[:60_000] + ',"staging_token":"tok-77"' + JSON_LINE[60_000:]
    (out / "1-mcp__api-c1.txt").write_text("cabecera\n" + line + "\n")
    hit = ctxm.search_history(ws, {"query": r".*staging_token.*"}, ARCHIVE)
    assert re.search(r"1-mcp__api-c1\.txt:2: .*staging_token", hit), hit[:400]  # the match itself, not the line's start
    # a nested repeat is searched as plain text (it would take exponential time as a regex)
    (out / "2-bash-c2.txt").write_text("nota\nel patrón (\\w+\\s?)+token salió en el log\n")
    hit = ctxm.search_history(ws, {"query": r"(\w+\s?)+token"}, ARCHIVE)
    assert "2-bash-c2.txt:2:" in hit and "plain text" in hit, hit


def test_search_history_stops_when_the_user_presses_stop(tmp_path):
    ws = Workspace(tmp_path / "ws")
    out = ws.root / ARCHIVE / "outputs"
    out.mkdir(parents=True)
    for i in range(3):
        (out / f"{i}-bash-c{i}.txt").write_text("valor=1\n" * 50)
    ws.cancel = threading.Event()
    ws.cancel.set()
    assert "stopped by the user" in ctxm.search_history(ws, {"query": "valor"}, ARCHIVE)


# ====================================================================== ctx#4: file:line is a read_file offset
def test_search_history_line_numbers_are_read_file_offsets(tmp_path):
    ws = Workspace(tmp_path / "ws")
    out = ws.root / ARCHIVE / "outputs"
    out.mkdir(parents=True)
    progress = "".join(f"\rDownloading {p}%" for p in range(0, 101, 10))  # curl/pip progress bars: bare \r
    (out / "1-bash-pip-c1.txt").write_bytes(f"exit code: 0\n{progress}\nSTAGING_API_KEY=sk-live-AAAABBBBCCCC1234\n"
                                            "salto\x0cde página y separador\nOTRO_VALOR=42\n".encode("utf-8"))
    for query in ("STAGING_API_KEY", "OTRO_VALOR", "valor otro"):  # (the last one has no exact match: ranked)
        hit = ctxm.search_history(ws, {"query": query}, ARCHIVE)
        rel, n = re.search(r"^(\S+?):(\d+):", hit, re.M).groups()
        line = read_file(ws, {"path": rel, "offset": int(n), "limit": 1})
        assert ("sk-live-AAAABBBBCCCC1234" if query == "STAGING_API_KEY" else "OTRO_VALOR=42") in line, (query, hit, line)


# ====================================================================== ctx#7: a read_file cut keeps the real length
def test_inline_cut_of_a_legacy_encoded_file_names_its_real_length_and_keeps_the_encoding(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "legacy.c").write_bytes("".join(f"/* línea {i} café */ int v{i};\n" for i in range(1, 5001)).encode("cp1252"))
    out = read_file(ws, {"path": "legacy.c"})
    assert out.endswith("(encoding: cp1252)")
    cut = ctxm.inline_output(out, ".nesicode/outputs/x.txt", 19_400, "read_file", {"path": "legacy.c"})
    assert re.search(r"showing lines 1–\d+ of 5000; lines \d+–5000 are NOT shown", cut), cut[-400:]
    assert "(encoding: cp1252)" in cut
    # a huge file whose total is unknown ("more lines may follow"): never presented as the file's end
    big = "\n".join(f"{i:>6}\t" + "x" * 90 for i in range(1, 2001)) + "\n... (more lines may follow; use offset=2001)" \
        + "\n(large file: 900,000,000 bytes; lines longer than 20,000 characters are cut)"
    cut = ctxm.inline_output(big, ".nesicode/outputs/y.txt", 8_000, "read_file", {"path": "huge.log"})
    assert "of 2000" not in cut and "more lines may follow" not in cut.split("[...")[0]
    assert re.search(r"of more than 2000", cut) and "(large file:" in cut, cut[-500:]


# ====================================================================== ctx#3: every omitted turn is searchable
def test_turns_the_context_omits_are_findable_with_search_history(tmp_path):
    from nesicode.server.app import db
    from nesicode.server.jobs import _write_thread_history, conversation_history

    _, p, t = _client()
    now = time.time()
    for i in range(1, 51):
        _run(p, t, f"mensaje {i}" + (" REGLA-T5: nunca toques deploy/prod.yml" if i == 5 else ""), "done",
             f"respuesta {i}" + (" VALOR-T7=9931" if i == 7 else ""), [{"type": "assistant", "text": f"respuesta {i}"}],
             at=now - 1000 + i)
    cur = _run(p, t, "¿qué regla te di sobre prod.yml?", "running", at=now)
    history, _ = conversation_history(db, cur, window=131_072)
    ws = Workspace(tmp_path / "ws")
    archive = f".nesicode/chats/{t['id']}"
    _write_thread_history(db, ws, cur, archive)
    fitted = Agent(model="mock/demo", harness=base_harness(), workspace=ws, context_limit=131_072,
                   archive=archive)._fit_history(history, "s")
    assert not any("REGLA-T5" in u for u, _ in fitted)
    assert "search_history finds them" in fitted[0][1]  # the promise...
    for needle in ("REGLA-T5", "VALOR-T7"):  # ...holds
        assert needle in ctxm.search_history(ws, {"query": needle}, archive)
    text = (ws.root / archive / "context" / "history.md").read_text()
    assert text.count("## Turno #") == 50 and "mensaje 50" in text


# ====================================================================== ctx#6: the previous turn's checkpoint
def test_previous_turn_checkpoint_keeps_its_next_step_and_says_where_the_rest_is():
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    summary = ("## Goal\nRevisar.\n## Completed Actions\n"
               + "\n".join(f"{i}. READ packages/core/src/file{i}.ts — revisado [tool: read_file]" for i in range(1, 400))
               + "\n## Active State\nEn curso: queue.ts\n## Next Step\nREANUDAR en queue.ts:413")
    path = f".nesicode/chats/{t['id']}/context/20261108-101010-compact-1.md"
    _run(p, t, "revisa", "cancelled", "", [
        {"type": "tool_call", "id": "a", "name": "read_file", "args": {"path": "x"}},
        {"type": "tool_result", "id": "a", "name": "read_file", "output": "ok"},
        {"type": "compacted", "summary": summary, "n": 1, "path": path}, {"type": "cancelled"}], at=now - 5)
    history, _ = conversation_history(db, _run(p, t, "continúa", "running", at=now), window=131_072)
    answer_ = history[0][1]
    assert "REANUDAR en queue.ts:413" in answer_ and "## Goal" in answer_
    assert "omitted" in answer_ and path in answer_  # the cut is said, and where the whole checkpoint is


# ====================================================================== ctx#2: the summarizer request fits the window
def test_summarizer_request_fits_the_window_on_dense_content(tmp_path, fake):
    rnd = random.Random(5)

    def policy(conv, rec, on_event):
        if rec["role"] == "summarizer":
            return Turn("## Goal\nAuditar.\n## Next Step\nSeguir.", [], "end_turn", Usage(0, 300))
        n = len(prov.executor())
        if n >= 45:
            return Turn("Fin.", [], "end_turn", Usage(0, 10))
        text = f"Hallazgo {n}: " + " ".join(f"campo_{rnd.randint(0, 99)}={rnd.randint(0, 9999)}" for _ in range(250))
        return Turn(text, [ToolCall(f"r{n}{j}", "read_file", {"path": f"src/m{(2 * n + j) % 7}.py"}) for j in range(2)],
                    "tool_use", Usage(0, 300))
    prov = fake(policy, window=131_072, cpt=2.7)  # denser than the 3.0 floor of the executor's estimate
    ws = Workspace(tmp_path / "ws")
    (ws.root / "src").mkdir()
    for i in range(7):
        (ws.root / "src" / f"m{i}.py").write_text(code(rnd, 60_000))
    events = []
    agent(ws, events, context_limit=131_072, output_limit=32_768, max_steps=60).run("audita las respuestas de la API")
    summ = [r for r in prov.log if r["role"] == "summarizer"]
    assert summ, "no compaction happened"
    assert not [r for r in summ if r["rejected"]], f"{sum(r['rejected'] for r in summ)}/{len(summ)} over the window"
    assert not [e for e in events if e["type"] == "compacted" and e.get("fallback")]


def test_a_summarizer_request_the_window_rejects_is_retried_smaller(tmp_path, fake):
    from test_p2_context import Conv

    class DenseSummaries(Conv):  # the transcript the summarizer gets is much denser than the conversation
        def step(self, on_event=None):
            saved = self.prov.cpt
            self.prov.cpt = 2.0 if self.options.role == "summarizer" else saved
            try:
                return super().step(on_event)
            finally:
                self.prov.cpt = saved

    rnd = random.Random(7)

    def policy(conv, rec, on_event):
        if rec["role"] == "summarizer":
            return Turn("## Goal\nAuditar.\n## Next Step\nSeguir.", [], "end_turn", Usage(0, 300))
        n = len(prov.executor())
        if n >= 30:
            return Turn("Fin.", [], "end_turn", Usage(0, 10))
        said = f"Hallazgo {n}: " + " ".join(f"campo_{i}={rnd.randint(0, 9999)}" for i in range(200))  # never masked
        return Turn(said, [ToolCall(f"r{n}", "read_file", {"path": f"src/m{n % 7}.py"})], "tool_use", Usage(0, 600))
    prov = fake(policy, window=32_768, cpt=3.5, conv=DenseSummaries)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "src").mkdir()
    for i in range(7):
        (ws.root / "src" / f"m{i}.py").write_text(code(rnd, 30_000))
    events = []
    res = agent(ws, events, context_limit=32_768, output_limit=8_000, max_steps=40).run("audita")
    summ = [r for r in prov.log if r["role"] == "summarizer"]
    assert res.status == "done" and summ
    assert summ[0]["rejected"] and not summ[1]["rejected"], [r["rejected"] for r in summ]  # retried, and it fit
    assert not [e for e in events if e["type"] == "compacted" and e.get("fallback")]


# ====================================================================== ctx#5: replayed reasoning on small windows
class _OfflineCompat(OpenAICompatProvider):
    """The real OpenAI-compatible conversation (its masking and reasoning replay), with chat() answered offline:
    tokens = JSON of messages + tools / cpt; a request over the window is rejected like vLLM does."""

    def __init__(self, policy, window, cpt=3.3):
        super().__init__("local-vllm", "http://127.0.0.1:9/v1")
        self.policy, self.window, self.cpt, self.log = policy, window, cpt, []

    def chat(self, body, options, on_event=None):
        chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in body["messages"]) + len(json.dumps(body.get("tools") or []))
        inp = int(chars / self.cpt)
        rec = {"role": options.role, "input": inp, "rejected": False, "body": body}
        self.log.append(rec)
        if inp + (body.get("max_tokens") or 0) > self.window:
            rec["rejected"] = True
            raise LLMError(f"HTTP 400: This model's maximum context length is {self.window} tokens. However, you requested "
                           f"{inp + body['max_tokens']} tokens ({inp} in the messages, {body['max_tokens']} in the completion).")
        msg, finish, out = self.policy(rec)
        return {"choices": [{"message": msg, "finish_reason": finish}], "usage": {"prompt_tokens": inp, "completion_tokens": out}}, False


# (16k: 8 since the always-on safety lines of the system prompt — what the sandbox keeps out, never get past a site's
# bot protection — added ~90 tokens; 7 before them, 9 before P1/P11. The gaps of >= 4 calls are what matters.)
@pytest.mark.parametrize("window,most", [(16_384, 8), (32_768, 2)])
def test_small_windows_with_a_reasoning_model_compact_rarely(tmp_path, monkeypatch, window, most):
    """P11's scenario and bounds (test_p2_context::test_small_windows_compact_rarely_and_rebuild_to_about_a_third) with
    2,500 chars of reasoning in every reply, replayed (P1): before, 9 compactions at 16k and 6 at 32k."""
    from nesicode.agent import loop as loopmod

    rnd = random.Random(3)
    thinking = ("razono " * 400)[:2_500]

    def policy(rec):
        if rec["role"] == "summarizer":
            return {"role": "assistant", "content": "## Goal\nRefactor.\n## Constraints & Preferences\n- NUNCA toques "
                    "config/prod.yml\n## Next Step\nContinue."}, "stop", 300
        n = sum(1 for r in prov.log if r["role"] == "executor" and not r["rejected"]) - 1
        if n >= 40:
            return {"role": "assistant", "content": "Hecho.", "reasoning_content": thinking}, "stop", 10
        if n % 6 == 5:
            call = {"name": "write_file", "arguments": json.dumps({"path": f"out/r{n}.py", "content": code(rnd, 2_500)})}
        else:
            call = {"name": "read_file", "arguments": json.dumps({"path": f"src/m{n % 7}.py", "offset": 1 + 40 * (n // 7)})}
        return {"role": "assistant", "content": "Sigo.", "reasoning_content": thinking,
                "tool_calls": [{"id": f"c{n}", "type": "function", "function": call}]}, "tool_calls", 700
    prov = _OfflineCompat(policy, window)
    monkeypatch.setattr(loopmod, "get_provider", lambda spec, keys=None: (prov, "m"))
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
    assert len(at) <= most and all(b - a >= 4 for a, b in zip(at, at[1:])), at
    # the reasoning still goes back with the newest reply (interleaved thinking), even past the masking threshold
    last = [m for m in prov.log[-1]["body"]["messages"] if m.get("role") == "assistant"]
    assert last[-1].get("reasoning_content") == thinking


def test_strip_reasoning_keeps_the_newest_replies_and_never_touches_deepseek():
    from test_p2_context import _loop_conv

    conv = _loop_conv()
    freed = conv.strip_reasoning(1)
    replies = [m for m in conv.messages if m["role"] == "assistant"]
    assert freed == 2 * len("PENSAMIENTO-A " * 40) and "reasoning_content" in replies[-1]
    assert not any("reasoning_content" in m for m in replies[:-1]) and conv.strip_reasoning(1) == 0
    ds = _loop_conv("deepseek")
    assert ds.strip_reasoning(1) == 0 and all("reasoning_content" in m for m in ds.messages if m["role"] == "assistant")


# ====================================================================== ctx#8: a sub-agent that did not finish
def _with_reviewer():
    h = base_harness()
    h["sub_agents/reviewer/AGENT.md"] = "---\nname: reviewer\ndescription: Reviews code\n---\nYou review code.\n"
    return h


def _delegate_result(events):
    return next(e for e in events if e["type"] == "tool_result" and e.get("name") == "delegate" and not e.get("depth"))


def test_a_refusing_subagent_is_not_reported_as_an_answer(tmp_path, fake):
    def policy(conv, rec, on_event):
        if rec["role"] == "subagent":
            return Turn("", [], "refusal", Usage(0, 5))
        if any(m["role"] == "tool" for m in rec["messages"]):
            return Turn("Fin.", [], "end_turn", Usage(0, 5))
        return Turn("Delego.", [ToolCall("d1", "delegate", {"agent": "reviewer", "task": "revisa a.py"})], "tool_use", Usage(0, 5))
    fake(policy)
    events = []
    agent(Workspace(tmp_path / "ws"), events, harness=_with_reviewer()).run("revisa")
    res = _delegate_result(events)
    assert res["is_error"] and "refused" in res["output"], res


def test_a_stopped_subagent_is_not_reported_as_an_answer(tmp_path, fake):
    cancel = threading.Event()

    def policy(conv, rec, on_event):
        if rec["role"] == "subagent":
            n = sum(1 for r in prov.log if r["role"] == "subagent")
            if n == 3:
                cancel.set()  # Stop while the sub-agent works
            return Turn(f"sub paso {n}: leo src/a{n}.py", [ToolCall(f"s{n}", "list_dir", {"path": "."})], "tool_use",
                        Usage(0, 5))
        return Turn("Delego.", [ToolCall("d1", "delegate", {"agent": "reviewer", "task": "revisa"})], "tool_use", Usage(0, 5))
    prov = fake(policy)
    events = []
    res = agent(Workspace(tmp_path / "ws"), events, harness=_with_reviewer(), cancel=cancel).run("revisa")
    out = _delegate_result(events)
    assert res.status == "cancelled"
    assert out["is_error"] and "stopped by the user" in out["output"] and "did not finish" in out["output"], out


def test_a_subagent_out_of_steps_is_marked_partial(tmp_path, fake):
    def policy(conv, rec, on_event):
        if rec["role"] == "subagent":
            if "step budget" in json.dumps(rec["messages"][-1]):
                return Turn("Resumen parcial del revisor.", [], "end_turn", Usage(0, 5))
            n = sum(1 for r in prov.log if r["role"] == "subagent")
            return Turn(f"paso {n}", [ToolCall(f"s{n}", "bash", {"command": f"echo paso {n}"})], "tool_use", Usage(0, 5))
        if any(m["role"] == "tool" for m in rec["messages"]):
            return Turn("Fin.", [], "end_turn", Usage(0, 5))
        return Turn("Delego.", [ToolCall("d1", "delegate", {"agent": "reviewer", "task": "revisa"})], "tool_use", Usage(0, 5))
    prov = fake(policy)
    events = []
    agent(Workspace(tmp_path / "ws"), events, harness=_with_reviewer()).run("revisa")
    out = _delegate_result(events)
    assert not out["is_error"] and "partial" in out["output"] and "Resumen parcial del revisor." in out["output"], out


# ====================================================================== llm#2: Stop while a tool waits for a model
@pytest.mark.parametrize("tool,kind", [("view_image", "openai"), ("generate_image", "openai"), ("view_image", "anthropic")])
def test_stop_ends_a_vision_or_image_request_at_once_and_sends_nothing_after_it(tmp_path, srv, monkeypatch, tool, kind):
    monkeypatch.setattr(openai_compat, "NON_STREAM_READ_TIMEOUT", 3.0)
    monkeypatch.setenv("NESICODE_LLM_FIRST_BYTE_TIMEOUT", "1")
    args = {"path": "pic.png"} if tool == "view_image" else {"prompt": "a cat"}
    main = srv(tool_call_reply(tool, args), default=answer("done"))
    side = srv(default=hang)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "pic.png").write_bytes(PNG)
    cancel = threading.Event()
    keys = {"main": {"kind": "openai", "base_url": main.url, "api_key": "k"},
            "side": {"kind": kind, "base_url": side.url, "api_key": "k"}}
    a = Agent(model="main/m", harness=base_harness(), workspace=ws, user_keys=keys, cancel=cancel,
              vision_model="side/v" if tool == "view_image" else "", image_model="side/i" if tool == "generate_image" else "")
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("r", a.run("mira")), daemon=True)
    th.start()
    deadline = time.monotonic() + 10
    while not side.requests and th.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert side.requests, "the vision/image endpoint never got the request"
    stop = time.monotonic()
    cancel.set()
    th.join(15)
    took = time.monotonic() - stop
    assert out["r"].status == "cancelled"
    assert took < 2.0, f"the run ended {took:.1f} s after Stop"
    assert len(side.requests) == 1, f"{len(side.requests)} requests (one sent after Stop)"


def test_stop_ends_the_download_of_a_generated_image(srv):
    side = srv(default=hang)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(LLMError, match="cancelled"):
        transport.fetch_image(side.url + "/img.png", strict=False, cancel=cancel)
    assert time.monotonic() - t0 < 2.0


# ====================================================================== llm#3: finish reasons
PARTIAL = "El informe completo es: 1) los logs muestran"


def _zai(tmp_path, *script):
    s = Srv(*script, default=answer("Respuesta completa."))
    a = Agent(model="zai/glm-5", harness=base_harness(), workspace=Workspace(tmp_path / "ws"),
              user_keys={"zai": {"kind": "openai", "base_url": s.url, "api_key": "k"}})
    try:
        return a.run("Escribe el informe"), s.requests
    finally:
        s.close()


@pytest.mark.parametrize("finish", ["network_error", "abort", "error"])
def test_a_reply_the_provider_failed_is_retried_not_taken_as_the_answer(tmp_path, finish):
    res, reqs = _zai(tmp_path, answer(PARTIAL, finish))
    assert res.status == "done" and res.final_text == "Respuesta completa." and len(reqs) == 2, (res.final_text, len(reqs))


@pytest.mark.parametrize("finish", ["network_error", "error"])
def test_a_plain_json_reply_the_provider_failed_is_retried(tmp_path, finish):
    plain = json_reply(200, {"id": "x", "object": "chat.completion", "model": "glm", "usage": USAGE, "choices": [
        {"index": 0, "message": {"role": "assistant", "content": PARTIAL}, "finish_reason": finish}]})
    res, reqs = _zai(tmp_path, plain)
    assert res.final_text == "Respuesta completa." and len(reqs) == 2, (res.final_text, len(reqs))


def test_a_reply_blocked_by_the_providers_filter_is_a_refusal(tmp_path):
    res, reqs = _zai(tmp_path, answer(PARTIAL, "sensitive"))
    assert res.status == "refusal" and len(reqs) == 1


def test_a_reply_cut_by_a_full_context_is_not_the_final_answer(tmp_path):
    res, reqs = _zai(tmp_path, answer(PARTIAL, "model_length"))  # Mistral: the context filled up mid-reply
    assert len(reqs) == 2 and res.final_text.endswith("Respuesta completa."), (res.final_text, len(reqs))


# ====================================================================== llm#4: streamed replies are bounded
def test_one_endless_sse_line_is_refused_quickly():
    dec = openai_compat._SSE()
    piece = b"x" * 65_536
    t0 = time.monotonic()
    with pytest.raises(LLMError, match="line"):
        for _ in range(12 * 16):  # 12 MB without an end of line
            dec.feed(piece)
    assert time.monotonic() - t0 < 2.0


def test_a_long_sse_line_is_parsed_in_linear_time():
    data = json.dumps({"choices": [{"index": 0, "delta": {"content": "y" * 5_000_000}}]}).encode()
    raw = b"data: " + data + b"\n\n"
    dec = openai_compat._SSE()
    t0 = time.monotonic()
    events = []
    for i in range(0, len(raw), 65_536):
        events += dec.feed(raw[i:i + 65_536])
    assert time.monotonic() - t0 < 1.0
    assert len(events) == 1 and len(json.loads(events[0][1])["choices"][0]["delta"]["content"]) == 5_000_000


def test_a_streamed_reply_larger_than_the_cap_is_refused(srv, monkeypatch):
    monkeypatch.setattr(openai_compat, "MAX_BODY", 2 * 2**20)
    s = srv(sse(*[oc({"content": "z" * 10_000}) for _ in range(300)], oc(finish="stop"), DONE))
    conv = OpenAICompatProvider("u", s.url, "k").conversation("m", "sys", [], ModelOptions(max_tokens=100))
    conv.add_user("hola")
    with pytest.raises(LLMError, match="larger than"):
        conv.step()
    assert len(s.requests) == 1  # not retried


# ====================================================================== llm#5: vLLM / SGLang limits
def test_vllm_and_sglang_limits_are_read_and_learned(srv):
    assert parse_limit("The input (40000 tokens) is longer than the model's context length (32768 tokens).") == 32768
    vllm = ("The decoder prompt (length 40000) is longer than the maximum model length of 32768. Make sure that "
            "`max_model_len` is no smaller than the number of text tokens.")
    assert parse_limit(vllm) == 32768 and ctxm.prompt_tokens(vllm) == 40000
    assert ctxm.prompt_tokens("The input (40000 tokens) is longer than the model's context length (32768 tokens).") == 40000
    s = srv(json_reply(404, {"detail": "Not Found"}),
            json_reply(200, {"object": "list", "data": [{"id": "qwen3", "object": "model", "max_model_len": 32768}]}))
    assert fetch_limits("openai", s.url + "/v1", "", "qwen3") == {"context": 32768, "output": None}
    assert [r["path"] for r in s.requests] == ["/model/info", "/v1/models"]


# ====================================================================== llm#6: vision replies in content parts
def test_a_vision_reply_in_content_parts_is_read_as_text(tmp_path, srv):
    from nesicode.agent.vision import describe_image

    parts = [{"type": "thinking", "thinking": [{"type": "text", "text": "pienso..."}]}, {"type": "text", "text": "Un cuadrado rojo."}]
    s = srv(json_reply(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": parts},
                                          "finish_reason": "stop"}], "usage": USAGE}))
    ws = Workspace(tmp_path / "ws")
    (ws.root / "pic.png").write_bytes(PNG)
    keys = {"vis": {"kind": "openai", "base_url": s.url, "api_key": "k"}}
    assert describe_image(ws, "pic.png", "vis/magistral", keys, "¿qué es?") == "Un cuadrado rojo."


# ====================================================================== llm#8: a keyless Anthropic row
def test_a_keyless_anthropic_row_says_why_instead_of_a_bare_401(srv):
    for base in ("https://api.anthropic.com", "", "https://api.anthropic.com/"):
        with pytest.raises(LLMError, match="no API key"):
            build_provider("anthropic", "anthropic", base, "")
    # a keyless gateway of the user's own (Anthropic-compatible) still works without credentials; a 401 says why
    s = srv(json_reply(401, {"type": "error", "error": {"type": "authentication_error", "message": "x-api-key header is required"}}))
    prov = build_provider("anthropic", "gw", s.url, "")
    conv = prov.conversation("claude-x", "sys", [], ModelOptions(max_tokens=100))
    conv.add_user("hola")
    with pytest.raises(LLMError, match="no API key saved"):
        conv.step()


# ====================================================================== security#2: the base rm rule is linear
def _rm_rule() -> str:
    return json.loads(base_harness()["plugins/hooks.json"])["pre_tool"][0]["match"]


def test_the_base_rm_rule_takes_linear_time_and_keeps_its_verdicts():
    from test_fix_tools import ALLOWED, BLOCKED

    hooks = Hooks(json.loads(base_harness()["plugins/hooks.json"]))
    for cmd in BLOCKED + ["rm -rf rm /", "echo x; rm -rf /*", "(rm -rf ${HOME})", 'bash -c "rm -rf /"']:
        assert hooks.before("bash", {"command": cmd})[0], cmd
    for cmd in ALLOWED + ["rm -rf ./tmp ~/x", "rm -f a b c", "echo rm rm rm"]:
        assert hooks.before("bash", {"command": cmd})[0] is None, cmd
    for cmd in ("echo " + "rm " * 5_000, "echo " + "rm -f x " * 3_000):
        t0 = time.perf_counter()
        assert hooks.before("bash", {"command": cmd})[0] is None
        assert time.perf_counter() - t0 < 0.5, cmd[:40]


@pytest.mark.parametrize("legacy", [
    r"""(?:^|[\s;&|({`"'])rm(?:\s+[^\s;&|]+)*?\s+["']?(?:/\*?|~/?\*?|\$\{?HOME\}?/?\*?)["']?(?=$|[\s;&|)`"'])""",
    r"""(?:^|[\s;&|({`"'])rm(?:[ \t]+[^\s;&|\\]+)*?[ \t]+["']?(?:/\*?|~/?\*?|\$\{?HOME\}?/?\*?)["']?(?=$|[\s;&|)`"'])""",
])
def test_a_stored_harness_with_an_earlier_spelling_of_the_rule_runs_the_linear_one(tmp_path, legacy):
    h = base_harness()
    cfg = json.loads(h["plugins/hooks.json"])
    cfg["pre_tool"][0]["match"] = legacy
    h["plugins/hooks.json"] = json.dumps(cfg)
    a = Agent(model="mock/demo", harness=h, workspace=Workspace(tmp_path / "ws"))
    assert a.hooks.pre[0]["match"] == _rm_rule() != legacy
    t0 = time.perf_counter()
    assert a.hooks.before("bash", {"command": "echo " + "rm " * 5_000})[0] is None
    assert time.perf_counter() - t0 < 0.5
    assert a.hooks.before("bash", {"command": "rm -rf ~"})[0]


# ====================================================================== security#6: "proxied" as httpx decides it
NO_PROXY_CASES = ["", "localhost,*", "*", "localhost", "localhost.:8633", "127.0.0.1", ".example.com", "example.com",
                  "https://api.example.com", "api.example.com:8443", "10.0.0.0/8"]
URLS = ["http://localhost.:8633/v1", "http://localhost:8633/v1", "https://example.com/v1", "https://api.example.com/v1",
        "https://api.example.com:8443/v1", "http://127.0.0.1:8633/v1", "http://10.1.2.3/v1"]


def _routed_by_proxy(client, url_cls, url: str) -> bool:
    """The oracle: the transport the very client that makes the request picks for this URL."""
    with client as c:
        return c._transport_for_url(url_cls(url)) is not c._transport


@pytest.mark.parametrize("no_proxy", NO_PROXY_CASES)
def test_proxied_agrees_with_the_http_client_that_connects(monkeypatch, no_proxy):
    import anthropic
    import httpx2

    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", no_proxy)
    for url in URLS:
        assert transport.proxied(url) == _routed_by_proxy(httpx.Client(), httpx.URL, url), (no_proxy, url)
        sdk = _routed_by_proxy(anthropic.DefaultHttpxClient(), httpx2.URL, url)  # (the client AnthropicProvider uses)
        assert transport.proxied(url, "anthropic") == sdk, (no_proxy, url, "anthropic")


def test_a_private_endpoint_is_refused_when_no_proxy_lists_a_star(monkeypatch, srv):
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "localhost,*")  # httpx: no proxy at all (it connects directly)
    s = srv(default=answer("secreto interno"))
    conv = OpenAICompatProvider("u", s.url, "k", guard_private=True).conversation("m", "sys", [], ModelOptions(max_tokens=50))
    conv.add_user("hola")
    with pytest.raises(LLMError, match="private networks are not allowed"):
        conv.step()
    assert not s.requests
