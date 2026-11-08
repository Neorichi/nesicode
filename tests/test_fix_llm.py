"""Group B (LLM transport) regression tests: streaming, retries, cancel, message validity, Anthropic, limits, images.

Everything is offline: a scripted HTTP server on 127.0.0.1 plays the provider (SSE, stalls, dropped connections,
429s, HTML pages...)."""

import json
import select
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nesicode.llm import LLMError, ModelOptions, ToolResult, ToolSpec, Usage, build_provider, get_provider, list_models
from nesicode.llm import transport
from nesicode.llm.anthropic_provider import AnthropicProvider, _cost, claude_version, thinking_params
from nesicode.llm.openai_compat import CUT, OpenAICompatProvider, _SSE

TOOLS = [ToolSpec("write_file", "write", {"type": "object", "properties": {"path": {"type": "string"},
                                                                          "content": {"type": "string"}}})]


# ---------------------------------------------------------------------- fake provider
class _Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass  # clients that hang up mid-response (Stop, aborted attempts) are expected here


class FakeServer:
    """Answers each request with the next scripted handler (then `default`). A handler gets (request handler, body)
    and either writes the response itself (returns None) or returns (status, headers, payload)."""

    def __init__(self, *script, default=None):
        self.script, self.default = list(script), default
        self.requests: list[dict] = []
        self.disconnects: list[float] = []  # when a stalled client hung up (monotonic)
        self.closing = threading.Event()
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _handle(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw
                with fake._lock:
                    fake.requests.append({"method": self.command, "path": self.path, "headers": dict(self.headers),
                                          "body": body, "t": time.monotonic()})
                    fn = fake.script.pop(0) if fake.script else fake.default
                res = fn(self, body) if fn else (500, {}, b"{}")
                if res is None:
                    return
                status, headers, payload = res
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                for k, v in {"Content-Type": "application/json", **headers}.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _handle

        self.httpd = _Server(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.httpd.fake = self
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.closing.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def _stall(req, limit=10.0):
    """Keep the connection open and silent; record when the client hangs up."""
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


def respond(status, payload, headers=None):
    return lambda req, body: (status, headers or {}, payload)


def stream(*chunks, delay=0.0, then="end", ctype="text/event-stream"):
    """Chunks are bytes (a number = a pause in seconds). then: end (proper chunked terminator) | abort (connection
    cut mid-body) | stall (open and silent)."""
    def handler(req, body):
        req.send_response(200)
        req.send_header("Content-Type", ctype)
        req.send_header("Transfer-Encoding", "chunked")
        req.end_headers()
        try:
            for c in chunks:
                if isinstance(c, (int, float)):
                    time.sleep(c)
                    continue
                if delay:
                    time.sleep(delay)
                req.wfile.write(b"%x\r\n%s\r\n" % (len(c), c))
                req.wfile.flush()
        except OSError:  # the client hung up (Stop)
            req.server.fake.disconnects.append(time.monotonic())
            req.close_connection = True
            return None
        if then == "abort":
            req.connection.shutdown(2)
            req.close_connection = True
        elif then == "stall":
            _stall(req)
        else:
            req.wfile.write(b"0\r\n\r\n")
        return None
    return handler


def hang(req, body):  # accepts the request and never answers (a hung backend)
    _stall(req)


@pytest.fixture
def srv():
    servers = []

    def make(*script, default=None):
        s = FakeServer(*script, default=default)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(transport, "BACKOFF_BASE", 0.01)


# OpenAI-style SSE
def chunk(delta=None, finish=None, usage=None, choices=True, crlf=False):
    c = {"id": "c1", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}] if choices else []}
    if usage:
        c["usage"] = usage
    return ("data: " + json.dumps(c) + ("\r\n\r\n" if crlf else "\n\n")).encode()


DONE = b"data: [DONE]\n\n"


def completion(content="", tool_calls=None, finish="stop", reasoning=None, usage=(7, 3)):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


def oa_conv(server, name="nesilabs", model="nesi3", tools=TOOLS, **opts):
    prov = OpenAICompatProvider(name, server.url, "sk-test")
    conv = prov.conversation(model, "sys", tools, ModelOptions(max_tokens=1000, **opts))
    conv.add_user("go")
    return prov, conv


# ---------------------------------------------------------------------- OpenAI-compatible: streaming
def test_openai_streams_text_reasoning_split_tool_calls_and_usage(srv):
    s = srv(stream(
        b": keep-alive comment\n\n",
        chunk({"role": "assistant"}),
        chunk({"reasoning_content": "Thinking "}),
        chunk({"reasoning_content": "hard"}, crlf=True),
        chunk({"content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                               "function": {"name": "write_file", "arguments": '{"path": "a.py", '}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"content": "x = 1"}'}}]}),
        chunk({"tool_calls": [{"index": 1, "id": "call_b", "function": {"name": "write_file", "arguments": ""}}]}),
        chunk({"tool_calls": [{"index": 1, "function": {"arguments": '{"path": "b.py"}'}}]}),
        chunk(finish="tool_calls"),
        chunk(choices=False, usage={"prompt_tokens": 123, "completion_tokens": 45}),
        DONE,
    ))
    _, conv = oa_conv(s)
    events = []
    turn = conv.step(on_event=lambda kind, text: events.append((kind, text)))
    assert turn.text == "Hello" and turn.thinking == "Thinking hard"
    assert [(c.id, c.name, c.arguments, c.parse_error) for c in turn.tool_calls] == [
        ("call_a", "write_file", {"path": "a.py", "content": "x = 1"}, ""),
        ("call_b", "write_file", {"path": "b.py"}, ""),
    ]
    assert turn.stop_reason == "tool_use" and (turn.usage.input_tokens, turn.usage.output_tokens) == (123, 45)
    assert events == [("thinking", "Thinking "), ("thinking", "hard"), ("text", "Hel"), ("text", "lo")]
    sent = s.requests[0]["body"]
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert sent["max_tokens"] == 1000 and "max_completion_tokens" not in sent
    replay = conv.messages[-1]
    assert replay["role"] == "assistant" and "reasoning_content" not in replay  # not DeepSeek: unchanged history
    assert [(tc["id"], tc["function"]["arguments"]) for tc in replay["tool_calls"]] == [
        ("call_a", '{"path": "a.py", "content": "x = 1"}'), ("call_b", '{"path": "b.py"}')]


def test_sse_decoder_edge_cases():
    d = _SSE()
    out = d.feed(b'data: {"a": 1}\r')  # CRLF split across two chunks
    out += d.feed(b'\n\r\ndata:{"b": 2}\n\n')  # no space after "data:"
    out += d.feed(b": comment\nevent: error\ndata: {\"c\":\ndata: 3}\n\n")  # named event, multi-line data
    out += d.feed(b'data: {"d": 4}\ndata: {"e": 5}\n')  # blank line between events forgotten
    out += d.feed(b'{"f": 6}\n\n')  # a bare JSON line
    out += d.feed(b"data: [DONE]")  # last event without its blank line
    out += d.flush()
    assert out == [("", '{"a": 1}'), ("", '{"b": 2}'), ("error", '{"c":\n3}'), ("", '{"d": 4}'), ("", '{"e": 5}'),
                   ("", '{"f": 6}'), ("", "[DONE]")]


def test_openai_stalled_stream_times_out_retries_and_notifies(srv, monkeypatch):
    monkeypatch.setenv("NESICODE_LLM_IDLE_TIMEOUT", "0.3")
    monkeypatch.setenv("NESICODE_LLM_FIRST_BYTE_TIMEOUT", "0.3")
    s = srv(stream(chunk({"content": "partial "}), then="stall"),
            stream(chunk({"content": "full answer"}), chunk(finish="stop"), DONE))
    notes = []
    _, conv = oa_conv(s, notify=notes.append)
    shown = []
    t0 = time.monotonic()
    turn = conv.step(on_event=lambda kind, text: shown.append(text))
    assert turn.text == "full answer" and time.monotonic() - t0 < 3
    assert len(s.requests) == 2 and shown == ["partial ", "full answer"]
    assert len(notes) == 1 and notes[0]["type"] == "llm_retry" and notes[0]["attempt"] == 2
    assert notes[0]["max"] == transport.DEFAULT_ATTEMPTS and notes[0]["partial"] is True
    assert "stalled" in notes[0]["reason"]
    time.sleep(0.2)
    assert s.disconnects, "the stalled request was not closed"


def test_openai_role_only_chunk_keeps_the_first_byte_limit(srv, monkeypatch):
    """A first chunk with only the role, then a long silence (prompt processing, hidden reasoning): that is still
    waiting for the reply to start, not a stalled reply."""
    monkeypatch.setenv("NESICODE_LLM_FIRST_BYTE_TIMEOUT", "1.5")
    monkeypatch.setenv("NESICODE_LLM_IDLE_TIMEOUT", "0.2")
    s = srv(stream(chunk({"role": "assistant"}), 0.45, chunk({"content": "hi"}), chunk(finish="stop"), DONE))
    _, conv = oa_conv(s)
    assert conv.step().text == "hi" and len(s.requests) == 1


def test_openai_no_first_byte_times_out_then_gives_up(srv, monkeypatch):
    monkeypatch.setenv("NESICODE_LLM_FIRST_BYTE_TIMEOUT", "0.2")
    s = srv(default=hang)
    _, conv = oa_conv(s, max_attempts=2)
    t0 = time.monotonic()
    with pytest.raises(LLMError, match="did not start.*after 2 attempts"):
        conv.step()
    assert time.monotonic() - t0 < 2.5 and len(s.requests) == 2
    assert conv.messages[-1]["role"] == "user"  # nothing appended


def test_openai_429_honours_retry_after(srv):
    s = srv(respond(429, {"error": {"message": "slow down"}}, {"Retry-After": "1"}),
            stream(chunk({"content": "ok"}), chunk(finish="stop"), DONE))
    notes = []
    _, conv = oa_conv(s, notify=notes.append)
    t0 = time.monotonic()
    assert conv.step().text == "ok"
    assert time.monotonic() - t0 >= 1.0 and notes[0]["wait"] >= 1.0 and "429" in notes[0]["reason"]


def test_openai_never_sleeps_after_the_last_attempt(srv):
    s = srv(default=respond(503, {"error": {"message": "busy"}}, {"retry-after-ms": "300"}))
    _, conv = oa_conv(s, max_attempts=2)
    t0 = time.monotonic()
    with pytest.raises(LLMError, match="HTTP 503"):
        conv.step()
    elapsed = time.monotonic() - t0
    assert len(s.requests) == 2 and 0.3 <= elapsed < 0.6  # one wait (between the attempts), none after the last


def test_retry_after_parsing():
    import email.utils

    assert transport.retry_after_seconds({"retry-after": "20"}) == 20
    assert transport.retry_after_seconds({"retry-after-ms": "1500"}) == 1.5
    date = email.utils.formatdate(time.time() + 30, usegmt=True)
    assert 25 < transport.retry_after_seconds({"retry-after": date}) <= 31
    assert transport.retry_after_seconds({"retry-after": "garbage"}) is None
    assert transport.retry_after_seconds({}) is None


def test_openai_huge_retry_after_fails_fast(srv):
    s = srv(default=respond(429, {"error": {"message": "daily quota"}}, {"Retry-After": "86400"}))
    _, conv = oa_conv(s)
    t0 = time.monotonic()
    with pytest.raises(LLMError, match="asks to wait 86400 s"):
        conv.step()
    assert len(s.requests) == 1 and time.monotonic() - t0 < 1


def test_openai_non_json_200_is_an_llm_error(srv):
    html = b"<!doctype html><html><body>OpenRouter - models</body></html>"
    s = srv(default=respond(200, html, {"Content-Type": "text/html"}))
    _, conv = oa_conv(s)
    with pytest.raises(LLMError, match="OpenAI-compatible"):
        conv.step()
    assert len(s.requests) == 1  # a misconfigured URL is not retried
    with pytest.raises(LLMError, match="no es JSON"):
        list_models("openai", s.url, "k")
    s2 = srv(default=respond(200, {"data": None}))
    with pytest.raises(LLMError, match="lista de modelos"):
        list_models("openai", s2.url, "k")
    s3 = srv(default=respond(200, html, {"Content-Type": "text/html"}))
    with pytest.raises(LLMError, match="OpenAI-compatible"):
        OpenAICompatProvider("x", s3.url).post("/chat/completions", {"model": "m"})


def test_openai_stream_options_rejected_then_remembered(srv):
    s = srv(respond(400, {"error": {"message": "Unrecognized request argument supplied: stream_options"}}),
            stream(chunk({"content": "one"}), chunk(finish="stop"), DONE),
            stream(chunk({"content": "two"}), chunk(finish="stop"), DONE))
    _, conv = oa_conv(s)
    assert conv.step().text == "one"
    conv.add_user("again")
    assert conv.step().text == "two"
    bodies = [r["body"] for r in s.requests]
    assert "stream_options" in bodies[0] and "stream_options" not in bodies[1] and "stream_options" not in bodies[2]
    assert all(b["stream"] is True for b in bodies)


def test_openai_streaming_rejected_falls_back_to_plain_json(srv):
    s = srv(respond(400, {"error": {"message": "streaming is not supported by this deployment"}}),
            respond(200, completion("plain")))
    _, conv = oa_conv(s)
    shown = []
    assert conv.step(on_event=lambda k, t: shown.append(t)).text == "plain"
    assert "stream" not in s.requests[1]["body"] and shown == ["plain"]


def test_openai_server_ignoring_stream_answers_json(srv):
    s = srv(respond(200, completion("whole reply", usage=(11, 2))), respond(200, completion("second")))
    prov, conv = oa_conv(s)
    shown = []
    turn = conv.step(on_event=lambda k, t: shown.append(t))
    assert turn.text == "whole reply" and shown == ["whole reply"] and turn.usage.input_tokens == 11
    conv.add_user("more")
    assert conv.step().text == "second" and prov.stream is False  # it only answers whole replies: plain requests
    assert "stream" not in s.requests[1]["body"]


def test_openai_cancel_during_stalled_stream_closes_the_connection(srv):
    s = srv(stream(chunk({"content": "thinking..."}), then="stall"))
    cancel = threading.Event()
    _, conv = oa_conv(s, cancel=cancel)
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(LLMError, match="cancelled by the user"):
        conv.step(on_event=lambda k, t: None)
    stopped = time.monotonic()
    assert stopped - t0 < 1.5
    deadline = time.monotonic() + 2
    while not s.disconnects and time.monotonic() < deadline:
        time.sleep(0.02)
    assert s.disconnects and s.disconnects[0] - stopped < 1.0  # the server saw the hang-up: it stops generating


def test_openai_cancel_while_waiting_for_headers(srv):
    s = srv(default=hang)
    cancel = threading.Event()
    _, conv = oa_conv(s, cancel=cancel)
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(LLMError, match="cancelled"):
        conv.step()
    assert time.monotonic() - t0 < 1.5
    deadline = time.monotonic() + 2
    while not s.disconnects and time.monotonic() < deadline:
        time.sleep(0.02)
    assert s.disconnects  # the helper thread's socket was shut down, not left blocked for minutes


def test_on_event_exception_propagates_unchanged_and_aborts(srv):
    class Stop(Exception):
        pass

    s = srv(stream(chunk({"content": "a"}), chunk({"content": "b"}), then="stall"))
    _, conv = oa_conv(s)

    def on_event(kind, text):
        raise Stop()

    with pytest.raises(Stop):
        conv.step(on_event=on_event)
    deadline = time.monotonic() + 2
    while not s.disconnects and time.monotonic() < deadline:
        time.sleep(0.02)
    assert s.disconnects


def test_openai_mid_stream_error_event_is_retried(srv):
    s = srv(stream(chunk({"content": "par"}), b'data: {"error": {"message": "upstream overloaded", "code": 503}}\n\n'),
            stream(chunk({"content": "done"}), chunk(finish="stop"), DONE))
    notes = []
    _, conv = oa_conv(s, notify=notes.append)
    assert conv.step(on_event=lambda k, t: None).text == "done"
    assert notes[0]["partial"] is True and "overloaded" in notes[0]["reason"]


def test_openai_context_error_in_stream_is_not_retried(srv):
    s = srv(default=stream(b'data: {"error": {"message": "This model\'s maximum context length is 8192 tokens. '
                           b'However, you requested 9000 tokens"}}\n\n'))
    _, conv = oa_conv(s)
    with pytest.raises(LLMError, match="maximum context length is 8192") as err:
        conv.step()
    from nesicode.llm import parse_limit

    assert parse_limit(str(err.value)) == 8192 and len(s.requests) == 1


def test_openai_dropped_connection_mid_stream_is_retried(srv):
    s = srv(stream(chunk({"content": "half"}), then="abort"),
            stream(chunk({"content": "whole"}), chunk(finish="stop"), DONE))
    _, conv = oa_conv(s)
    assert conv.step().text == "whole" and len(s.requests) == 2


def test_openai_stream_ending_without_finish_is_retried(srv):
    s = srv(stream(chunk({"content": "half"})), stream(chunk({"content": "whole"}), chunk(finish="stop")))
    _, conv = oa_conv(s)
    assert conv.step().text == "whole" and len(s.requests) == 2  # the 2nd ends without [DONE] but with a finish


def test_openai_disconnect_after_finish_keeps_the_reply(srv):
    s = srv(stream(chunk({"content": "complete"}), chunk(finish="stop"), then="abort"))
    _, conv = oa_conv(s)
    assert conv.step().text == "complete" and len(s.requests) == 1


def test_openai_length_cut_tool_call_gets_parse_error(srv):
    s = srv(stream(chunk({"tool_calls": [{"index": 0, "id": "call_w", "function": {
        "name": "write_file", "arguments": '{"path": "app.py", "content": "def main():\\n    retu'}}]}),
        chunk(finish="length"), DONE))
    _, conv = oa_conv(s)
    turn = conv.step()
    assert turn.stop_reason == "max_tokens"
    call = turn.tool_calls[0]
    assert call.parse_error == CUT and call.arguments == {}
    assert conv.messages[-1]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_openai_missing_or_repeated_ids_become_unique_and_are_replayed(srv):
    reply = completion(tool_calls=[{"type": "function", "function": {"name": "write_file", "arguments": "{}"}},
                                   {"id": "", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}],
                       finish="tool_calls")
    again = completion(tool_calls=[{"id": "call_0", "type": "function", "function": {"name": "write_file",
                                                                                     "arguments": "{}"}}],
                       finish="tool_calls")
    s = srv(respond(200, reply), respond(200, again), respond(200, again), respond(200, completion("bye")))
    _, conv = oa_conv(s)
    ids = []
    for _ in range(3):
        turn = conv.step()
        ids += [c.id for c in turn.tool_calls]
        assert [tc["id"] for tc in conv.messages[-1]["tool_calls"]] == [c.id for c in turn.tool_calls]
        conv.add_tool_results([ToolResult(c.id, "ok") for c in turn.tool_calls])
    assert len(ids) == 4 and len(set(ids)) == 4 and all(i.startswith("call_") and i != "call_0" or i == "call_0"
                                                        for i in ids)
    assert ids.count("call_0") == 1  # the server reused "call_0": the second one got a fresh id
    conv.step()
    sent = s.requests[-1]["body"]["messages"]
    calls = {tc["id"] for m in sent if m["role"] == "assistant" for tc in m.get("tool_calls", [])}
    results = {m["tool_call_id"] for m in sent if m["role"] == "tool"}
    assert calls == results == set(ids)


def test_openai_malformed_arguments_are_replayed_as_an_empty_object(srv):
    bad = completion(tool_calls=[{"id": "call_x", "type": "function", "extra_content": {"google": {"sig": "abc"}},
                                  "function": {"name": "write_file", "arguments": '{"path": "a.py", "x": '}}],
                     finish="tool_calls")
    s = srv(respond(200, bad), respond(200, completion("fixed")))
    _, conv = oa_conv(s)
    turn = conv.step()
    assert turn.tool_calls[0].parse_error.startswith("invalid JSON arguments")
    conv.add_tool_results([ToolResult("call_x", "Tool call rejected", True)])
    conv.step()
    replayed = [m for m in s.requests[-1]["body"]["messages"] if m["role"] == "assistant"][0]["tool_calls"][0]
    assert replayed["function"]["arguments"] == "{}" and replayed["id"] == "call_x"
    assert replayed["extra_content"] == {"google": {"sig": "abc"}}  # provider fields (e.g. Gemini signatures) kept


def test_max_completion_tokens_by_endpoint_and_model(srv):
    assert OpenAICompatProvider("openai-work", "https://api.openai.com/v1").max_tokens_key("gpt-4o") == \
        "max_completion_tokens"
    assert OpenAICompatProvider("azure", "https://acme.openai.azure.com/openai/v1").max_tokens_key("gpt-4o") == \
        "max_completion_tokens"
    nesi = OpenAICompatProvider("nesilabs", "https://api.nesilabs.com/v1")
    assert nesi.max_tokens_key("nesi3") == "max_tokens"
    assert nesi.max_tokens_key("gpt-5-mini") == nesi.max_tokens_key("o3") == "max_completion_tokens"
    assert OpenAICompatProvider("openrouter", "https://openrouter.ai/api/v1").max_tokens_key("openai/gpt-5") == \
        "max_tokens"
    # a server that wants the other name says so once; it is remembered
    s = srv(respond(400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with this model. "
                                               "Use 'max_completion_tokens' instead."}}),
            stream(chunk({"content": "ok"}), chunk(finish="stop"), DONE))
    _, conv = oa_conv(s, model="my-reasoner")
    assert conv.step().text == "ok"
    assert "max_tokens" in s.requests[0]["body"] and s.requests[1]["body"]["max_completion_tokens"] == 1000


def test_deepseek_gets_its_reasoning_content_back(srv):
    s = srv(respond(200, completion(tool_calls=[{"id": "c1", "type": "function", "function": {
        "name": "write_file", "arguments": "{}"}}], finish="tool_calls", reasoning="let me think")),
        respond(200, completion("done")))
    prov = OpenAICompatProvider("deepseek", s.url, "k")
    conv = prov.conversation("deepseek-v4-pro", "sys", TOOLS, ModelOptions())
    conv.add_user("earlier question")
    conv.add_assistant_text("earlier answer")
    conv.add_user("go")
    turn = conv.step()
    conv.add_tool_results([ToolResult(turn.tool_calls[0].id, "ok")])
    conv.step()
    sent = [m for m in s.requests[-1]["body"]["messages"] if m["role"] == "assistant"]
    assert [m.get("reasoning_content") for m in sent] == ["", "let me think"]


def test_content_filter_is_a_refusal(srv):
    s = srv(stream(chunk({"content": ""}), chunk(finish="content_filter"), DONE))
    _, conv = oa_conv(s)
    assert conv.step().stop_reason == "refusal"


def test_broken_notify_never_breaks_the_request(srv):
    s = srv(respond(502, b"bad gateway", {"Content-Type": "text/plain"}),
            stream(chunk({"content": "ok"}), chunk(finish="stop"), DONE))

    def notify(info):
        raise RuntimeError("listener bug")

    _, conv = oa_conv(s, notify=notify)
    assert conv.step().text == "ok"


def test_complete_honours_cancel(srv):
    cancel = threading.Event()
    cancel.set()
    s = srv(default=hang)
    prov = OpenAICompatProvider("x", s.url)
    with pytest.raises(LLMError, match="cancelled"):
        prov.complete("m", "sys", "hi", ModelOptions(cancel=cancel))
    assert not s.requests
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(LLMError, match="cancelled"):
        prov.complete("m", "sys", "summarize", ModelOptions(cancel=cancel, max_attempts=2))
    assert time.monotonic() - t0 < 1.5


def test_usage_counts_cache_tokens():
    u = Usage(10, 2)
    u.add(Usage(100, 5, 0.5, cache_read_tokens=80, cache_write_tokens=10))
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (110, 7, 80, 10)
    assert u.to_dict()["cache_read_tokens"] == 80
    assert Usage(1, 1).to_dict() == {"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0}


# ---------------------------------------------------------------------- private networks and images
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def test_generated_image_url_cannot_reach_internal_hosts(srv):
    internal = srv(default=respond(200, b"AWS_SECRET_ACCESS_KEY=internal-only\n", {"Content-Type": "text/plain"}))
    redirect = srv(default=respond(302, b"", {"Location": "http://169.254.169.254/latest/meta-data/"}))
    images = srv(respond(200, {"data": [{"url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/r"}]}),
                 respond(200, {"data": [{"url": redirect.url + "/img.png"}]}),
                 respond(200, {"data": [{"url": internal.url + "/latest/meta-data"}]}),
                 respond(200, {"data": [{"url": internal.url + "/img.png"}]}))
    prov = OpenAICompatProvider("img", images.url)
    with pytest.raises(LLMError, match="link-local"):  # cloud metadata: always refused, nothing is fetched
        prov.generate_image("m", "a cat")
    with pytest.raises(LLMError, match="link-local"):  # ...also through a redirect
        prov.generate_image("m", "a cat")
    with pytest.raises(LLMError, match="PNG, JPEG, WebP or GIF"):  # not an image: never saved
        prov.generate_image("m", "a cat")
    internal.default = respond(200, PNG, {"Content-Type": "image/png"})
    assert prov.generate_image("m", "a cat") == (PNG, "image/png")
    # multi-user servers (NESICODE_ALLOW_PRIVATE_PROVIDERS=0): private and loopback addresses are refused too
    before = len(internal.requests)
    with pytest.raises(LLMError, match="private or local"):
        transport.fetch_image(internal.url + "/img.png", strict=True)
    assert len(internal.requests) == before
    with pytest.raises(LLMError, match="larger than"):
        transport.fetch_image(internal.url + "/img.png", strict=False, max_bytes=16)


def test_user_endpoint_is_checked_at_connect_time(srv, monkeypatch):
    """NESICODE_ALLOW_PRIVATE_PROVIDERS=0: a user endpoint whose name now resolves to a private address is refused
    when connecting, before the key or the prompt are sent (DNS can change after the endpoint was saved)."""
    s = srv(default=respond(200, completion("leak")))
    monkeypatch.setenv("NESICODE_ALLOW_PRIVATE_PROVIDERS", "0")
    prov = build_provider("openai", "mine", s.url, "sk-user-key")
    conv = prov.conversation("m", "sys", [], ModelOptions())
    conv.add_user("hi")
    with pytest.raises(LLMError, match="private"):
        conv.step()
    assert not s.requests
    monkeypatch.setenv("NESICODE_ALLOW_PRIVATE_PROVIDERS", "1")
    assert build_provider("openai", "mine", s.url, "k").post("/chat/completions", {"model": "m"})["choices"]


# ---------------------------------------------------------------------- Anthropic
def aev(name, data):
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


def a_start(usage=None, model="claude-sonnet-5-5"):
    return aev("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None,
        "stop_sequence": None, "usage": usage or {"input_tokens": 10, "output_tokens": 1}}})


def a_text(text, i=0):
    return (aev("content_block_start", {"type": "content_block_start", "index": i,
                                        "content_block": {"type": "text", "text": ""}})
            + aev("content_block_delta", {"type": "content_block_delta", "index": i,
                                          "delta": {"type": "text_delta", "text": text}})
            + aev("content_block_stop", {"type": "content_block_stop", "index": i}))


def a_tool_start(i=0, tid="toolu_1", name="write_file"):
    return aev("content_block_start", {"type": "content_block_start", "index": i, "content_block": {
        "type": "tool_use", "id": tid, "name": name, "input": {}}})


def a_json(part, i=0):
    return aev("content_block_delta", {"type": "content_block_delta", "index": i,
                                       "delta": {"type": "input_json_delta", "partial_json": part}})


def a_stop_block(i=0):
    return aev("content_block_stop", {"type": "content_block_stop", "index": i})


def a_end(reason, output=50):
    return (aev("message_delta", {"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None},
                                  "usage": {"output_tokens": output}})
            + aev("message_stop", {"type": "message_stop"}))


def a_conv(server, model="claude-sonnet-5-5", tools=TOOLS, **opts):
    prov = AnthropicProvider(api_key="sk-ant-user", base_url=server.url, isolated=True)
    conv = prov.conversation(model, "sys", tools, ModelOptions(max_tokens=2000, **opts))
    conv.add_user("go")
    return conv


def test_user_anthropic_provider_never_receives_the_server_credentials(srv, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-OPERATOR-SECRET")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "OPERATOR-TOKEN")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Gateway-Auth: OPERATOR-GATEWAY")
    s = srv(default=stream(a_start(), a_text("hi"), a_end("end_turn")))
    for key in ("", "sk-ant-mine"):
        prov, model = get_provider("mine/claude-sonnet-5-5",
                                   {"mine": {"kind": "anthropic", "base_url": s.url, "api_key": key}})
        conv = prov.conversation(model, "sys", [], ModelOptions(max_tokens=100))
        conv.add_user("hola")
        assert conv.step().text == "hi"
        headers = s.requests[-1]["headers"]
        assert "OPERATOR" not in json.dumps(headers) and "X-Gateway-Auth" not in headers
        auth = {k.lower(): v for k, v in headers.items()}.get("authorization")
        assert auth == (f"Bearer {key}" if key else None)  # no key: no credentials at all
    # an Anthropic-compatible endpoint saved with the official URL is not redirected by ANTHROPIC_BASE_URL either
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.operator.internal")
    prov, _ = get_provider("mine/claude-x", {"mine": {"kind": "anthropic", "base_url": "https://api.anthropic.com",
                                                      "api_key": ""}})
    assert str(prov.client.base_url).startswith("https://api.anthropic.com") and prov.client.api_key == ""
    # the server's own provider keeps using the server's key
    builtin, _ = get_provider("anthropic/claude-opus-5-5")
    assert builtin.client.api_key == "sk-ant-OPERATOR-SECRET"


def test_anthropic_overloaded_error_mid_stream_is_retried(srv):
    s = srv(stream(a_start(), aev("error", {"type": "error", "error": {"type": "overloaded_error",
                                                                       "message": "Overloaded"}})),
            stream(a_start(), a_text("recovered"), a_end("end_turn")))
    notes = []
    conv = a_conv(s, notify=notes.append)
    assert conv.step().text == "recovered" and len(s.requests) == 2
    assert "overloaded" in notes[0]["reason"]


def test_anthropic_dropped_connection_is_retried(srv):
    s = srv(stream(a_start(), a_tool_start(), a_json('{"path": "a.py", "content": "print(1)'), then="abort"),
            stream(a_start(), a_text("ok"), a_end("end_turn")))
    conv = a_conv(s)
    assert conv.step().text == "ok" and len(s.requests) == 2


def test_anthropic_incomplete_stream_is_never_used(srv):
    """A stream that ends without message_stop (a proxy closed it): the SDK would return write_file{path} without
    its content and the file would be emptied. It is retried, and never appended to the history."""
    s = srv(default=stream(a_start(), a_tool_start(), a_json('{"path": "app.py", "content": "def main():\\n    retu')))
    conv = a_conv(s, max_attempts=2)
    with pytest.raises(LLMError, match="ended before it was complete"):
        conv.step()
    assert len(s.requests) == 2 and [m["role"] for m in conv.messages] == ["user"]


def test_anthropic_context_window_exceeded_marks_the_cut_call(srv):
    s = srv(stream(a_start(), a_tool_start(), a_json('{"path": "app.py", "content": "def main():\\n    retu'),
                   a_stop_block(), a_end("model_context_window_exceeded")))
    turn = a_conv(s).step()
    assert turn.stop_reason == "max_tokens" and turn.tool_calls[0].parse_error == CUT


def test_anthropic_max_tokens_marks_only_incomplete_calls(srv):
    s = srv(stream(a_start(), a_tool_start(0, "toolu_a"), a_json('{"path": "a.py", "content": "ok"}', 0), a_stop_block(0),
                   a_tool_start(1, "toolu_b"), a_json('{"path": "b.py", "content": "unfini', 1), a_stop_block(1),
                   a_end("max_tokens")))
    turn = a_conv(s).step()
    assert turn.stop_reason == "max_tokens"
    assert [(c.id, c.parse_error) for c in turn.tool_calls] == [("toolu_a", ""), ("toolu_b", CUT)]


def test_anthropic_stop_during_tool_input_streaming(srv):
    parts = [a_json('{"path": "big.py", "content": "')] + [a_json("x = 1\\n" * 20) for _ in range(30)]
    s = srv(stream(a_start(), a_tool_start(), *parts, a_json('"}'), a_stop_block(), a_end("tool_use"), delay=0.15))
    cancel = threading.Event()
    conv = a_conv(s, cancel=cancel)
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(LLMError, match="cancelled"):
        conv.step(on_event=lambda k, t: None)  # only input_json_delta events: on_event is never called
    stopped = time.monotonic()
    assert stopped - t0 < 1.5
    deadline = time.monotonic() + 2
    while not s.disconnects and time.monotonic() < deadline:
        time.sleep(0.02)
    assert s.disconnects and s.disconnects[0] - stopped < 1.0


def test_anthropic_cache_breakpoints_and_usage(srv):
    usage = {"input_tokens": 100, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 200,
             "output_tokens": 1}
    s = srv(stream(a_start(usage, model="claude-opus-5-5"), a_text("ok"), a_end("end_turn", 50)))
    conv = a_conv(s, model="claude-opus-5-5")
    conv.add_assistant_text("earlier")
    conv.add_user("now")
    turn = conv.step()
    assert turn.usage.input_tokens == 1300  # the whole prompt: uncached + cache reads + cache writes
    assert (turn.usage.cache_read_tokens, turn.usage.cache_write_tokens) == (1000, 200)
    assert turn.usage.cost_usd == pytest.approx((100 * 4 + 200 * 4 * 1.25 + 1000 * 0.2 + 50 * 20) / 1e6)
    sent = s.requests[0]["body"]
    assert sent["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert sent["messages"][-1]["content"] == [{"type": "text", "text": "now", "cache_control": {"type": "ephemeral"}}]
    assert json.dumps(sent).count("cache_control") == 2
    assert all(isinstance(m["content"], str) or not isinstance(m["content"], list) or
               all("cache_control" not in b for b in m["content"] if isinstance(b, dict)) for m in conv.messages)
    conv.add_tool_results([ToolResult("toolu_1", "result")])
    params = conv._params()
    assert params["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in conv.messages[-1]["content"][-1]  # the stored history is never changed


def test_anthropic_non_claude_models_on_compatible_endpoints_are_sent_as_before(srv):
    conv = a_conv(srv(), model="deepseek-chat")
    params = conv._params()
    assert params["system"] == "sys" and "cache_control" not in json.dumps(params, default=str)


def test_anthropic_prices_and_thinking_by_model_version():
    assert _cost("claude-opus-4-5-20251101", 1_000_000, 0, 0, 0) == 5.0
    assert _cost("claude-sonnet-4-6", 0, 0, 0, 1_000_000) == 15.0
    assert _cost("claude-mythos-5-1", 0, 1_000_000, 0, 0) == 0.25
    assert _cost("claude-haiku-5-5", 50_000, 0, 0, 1_000_000) == pytest.approx(0.005 + 0.5)
    assert _cost("claude-haiku-5-5", 150_000, 0, 0, 1_000_000) == pytest.approx(0.075 + 2.5)  # >100K prompt rate
    assert _cost("some-other-model", 10**6, 0, 0, 10**6) == 0.0
    for model in ("claude-opus-4-5-20251101", "claude-opus-4-1-20250805", "claude-sonnet-4-20250514",
                  "claude-sonnet-4-5", "claude-haiku-4-5-20251001", "claude-3-7-sonnet-20250219"):
        assert "thinking" not in thinking_params(model, "high"), model
    assert thinking_params("claude-opus-4-5-20251101", "high") == {"output_config": {"effort": "high"}}
    assert thinking_params("claude-sonnet-4-5", "high") == {}
    for model in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-5-5", "claude-haiku-5-5", "claude-fable-5-1"):
        assert thinking_params(model, "high")["thinking"]["type"] == "adaptive", model
    assert thinking_params("claude-sonnet-4-6", "xhigh")["output_config"] == {"effort": "high"}
    assert claude_version("claude-3-opus-20240229") == (3, 0) and claude_version("gpt-5") is None


def test_anthropic_auth_error_is_not_retried(srv):
    s = srv(default=respond(401, {"type": "error", "error": {"type": "authentication_error",
                                                             "message": "invalid x-api-key"}}))
    with pytest.raises(LLMError, match="authentication failed"):
        a_conv(s).step()
    assert len(s.requests) == 1


# ---------------------------------------------------------------------- limits
def test_parse_limit_more_provider_formats():
    from nesicode.llm import parse_limit

    assert parse_limit("The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).") \
        == 1048576
    assert parse_limit("Prompt contains 140000 tokens and 0 draft tokens, too large for model with 131072 maximum "
                       "context length") == 131072
    assert parse_limit('{"error":{"code":400,"message":"the request exceeds the available context size, try '
                       'increasing it","type":"exceed_context_size_error","n_prompt_tokens":10000,"n_ctx":8192}}') == 8192
    assert parse_limit("input length and `max_tokens` exceed context limit: 195000 + 64000 > 200000, decrease input "
                       "length or `max_tokens` and try again") == 200000
    assert parse_limit("Input tokens exceed the configured limit of 272000 tokens.") == 272000
    assert parse_limit("prompt is too long: 1050000 tokens > 1000000 maximum") == 1000000
    assert parse_limit("This model's maximum prompt length is 131072 but the request contains 300011 tokens.") == 131072
    assert parse_limit("However, the model is loaded with context length of only 4096 tokens, which is not enough.") \
        == 4096
    assert parse_limit('{"error":{"code":"1261","message":"Prompt exceeds max length"}}') is None


def test_accepted_probe_is_not_stored_nor_resent(srv):
    from nesicode.llm import probe_limit

    ok = srv(default=respond(200, completion("x")))
    assert probe_limit(ok.url, {}, "big-model") is None  # accepted: no made-up 600k window
    assert probe_limit(ok.url, {}, "big-model") is None and len(ok.requests) == 1  # and never sent again
    rejected = srv(default=respond(400, {"error": {"message": "This model's maximum context length is 131072 tokens"}}))
    assert probe_limit(rejected.url, {}, "small-model") == {"context": 131072, "output": None}


# ---------------------------------------------------------------------- reviewer additions
def test_openai_plain_text_200_is_not_retried(srv):
    """Neither SSE nor JSON nor HTML (a web server's 'OK', a proxy notice): a wrong URL, not a cut stream."""
    s = srv(default=respond(200, b"Hello from a web server", {"Content-Type": "text/plain"}))
    _, conv = oa_conv(s)
    with pytest.raises(LLMError, match="OpenAI-compatible API.*Hello from a web server"):
        conv.step()
    assert len(s.requests) == 1
    with pytest.raises(LLMError, match="OpenAI-compatible"):
        OpenAICompatProvider("x", s.url).post("/images/generations", {"model": "m"})
    assert len(s.requests) == 2
    # a real event stream that is cut short is still retried (no Content-Type check makes it "not an API")
    s2 = srv(stream(b": still working\n\n"), stream(chunk({"content": "ok"}), chunk(finish="stop"), DONE))
    _, conv = oa_conv(s2)
    assert conv.step().text == "ok" and len(s2.requests) == 2


def test_openai_length_cut_last_call_without_arguments_gets_parse_error(srv):
    """The cap hit right after the last call's name: its arguments are missing, it must not run with {}."""
    s = srv(stream(chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {
        "name": "write_file", "arguments": '{"path": "a.py", "content": "x"}'}}]}),
        chunk({"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "write_file", "arguments": ""}}]}),
        chunk(finish="length"), DONE))
    _, conv = oa_conv(s)
    turn = conv.step()
    assert turn.stop_reason == "max_tokens"
    assert [(c.id, c.parse_error) for c in turn.tool_calls] == [("c1", ""), ("c2", CUT)]


def test_parse_limit_ignores_limits_that_are_not_tokens():
    from nesicode.llm import parse_limit

    assert parse_limit("Request body exceeds the configured limit of 10485760 bytes") is None  # a gateway's size cap
    assert parse_limit("Input tokens exceed the configured limit of 272000 tokens.") == 272000


def test_openai_error_bodies_that_cannot_succeed_are_not_retried(srv):
    # 200 + an error object (one-api / new-api style gateways) of a kind another attempt cannot fix
    s = srv(default=respond(200, {"error": {"message": "Invalid schema for function 'write_file'",
                                            "type": "invalid_request_error", "code": None}}))
    _, conv = oa_conv(s)
    with pytest.raises(LLMError, match="Invalid schema"):
        conv.step()
    assert len(s.requests) == 1
    # Google-style errors come wrapped in a list; mid-stream they must surface, not look like a cut stream
    s2 = srv(default=stream(b'data: [{"error": {"code": 400, "message": "Unknown name \\"foo\\"", '
                            b'"status": "INVALID_ARGUMENT"}}]\n\n'))
    _, conv = oa_conv(s2)
    with pytest.raises(LLMError, match='Unknown name "foo"'):
        conv.step()
    assert len(s2.requests) == 1
    # transient ones are still retried (LiteLLM: code "500" as a string)
    s3 = srv(stream(b'data: {"error": {"message": "upstream hiccup", "type": "None", "code": "500"}}\n\n'),
             stream(chunk({"content": "ok"}), chunk(finish="stop"), DONE))
    _, conv = oa_conv(s3)
    assert conv.step().text == "ok" and len(s3.requests) == 2


def test_image_url_is_vetted_even_through_a_proxy(srv, monkeypatch):
    """With HTTP_PROXY set the socket talks to the proxy, so the target is checked before anything is sent."""
    proxy = srv(default=respond(200, PNG, {"Content-Type": "image/png"}))
    for name in ("NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "http_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTP_PROXY", proxy.url)
    assert transport.proxied("http://169.254.169.254/latest/meta-data/")
    with pytest.raises(LLMError, match="link-local"):
        transport.fetch_image("http://169.254.169.254/latest/meta-data/iam/security-credentials/r", strict=False)
    with pytest.raises(LLMError, match="private or local"):
        transport.fetch_image("http://127.0.0.1:9/img.png", strict=True)
    assert not proxy.requests  # nothing reached the proxy
    assert transport.fetch_image("http://127.0.0.1:9/img.png", strict=False) == (PNG, "image/png")
    assert proxy.requests[-1]["path"] == "http://127.0.0.1:9/img.png"  # fetched through the proxy
