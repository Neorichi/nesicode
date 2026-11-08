"""Any OpenAI-compatible chat-completions endpoint (OpenAI, OpenRouter, Ollama, Gemini, Z.ai/GLM, vLLM, LM Studio,
LiteLLM proxies...). Replies are streamed (SSE): a long answer is never cut by a total timeout, Stop works mid-reply
and stalls are detected; servers that cannot stream get a plain JSON request. Tool calls via the `tools` field."""

from __future__ import annotations

import codecs
import json
import re
import uuid
from urllib.parse import urlparse

import httpx

from .base import (
    Conversation,
    EventSink,
    LLMError,
    ModelOptions,
    Provider,
    ToolCall,
    ToolResult,
    Turn,
    Usage,
)
from .transport import (
    CONNECT_TIMEOUT,
    RETRY_STATUS,
    Call,
    Retryable,
    address_problem,
    fetch_image,
    first_byte_timeout,
    idle_timeout,
    image_mime,
    proxied,
    pump,
    retry_after_seconds,
    with_retries,
)

CUT = "output was cut off at max_tokens; resend a smaller call"
NON_STREAM_READ_TIMEOUT = 600.0  # a plain JSON request only answers once the whole reply is ready
MAX_BODY = 64 * 1024 * 1024
STOP = {"stop": "end_turn", "tool_calls": "tool_use", "function_call": "tool_use", "length": "max_tokens",
        "model_context_window_exceeded": "max_tokens", "content_filter": "refusal"}


class OpenAICompatConversation(Conversation):
    def __init__(self, provider: "OpenAICompatProvider", model, system, tools, options):
        super().__init__(model, system, tools, options)
        self.provider = provider
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_text(self, text: str) -> None:
        msg = {"role": "assistant", "content": text or "(no answer)"}
        if self.provider.replays_reasoning:
            msg["reasoning_content"] = ""  # DeepSeek's thinking mode rejects history without it (that reasoning is gone)
        self.messages.append(msg)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            content = ("ERROR: " if r.is_error else "") + (r.content or "(empty)")
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": content})

    def mask_tool_results(self, keep_last: int, marker) -> int:
        tool_msgs = [m for m in self.messages if m.get("role") == "tool"]
        freed = 0
        for m in tool_msgs[: max(0, len(tool_msgs) - keep_last)]:
            new = marker(m.get("tool_call_id", ""), m.get("content", ""))
            if new is not None and len(new) < len(m.get("content", "")):
                freed += len(m["content"]) - len(new)
                m["content"] = new
        return freed

    def step(self, on_event: EventSink | None = None) -> Turn:
        body: dict = {"model": self.model, "messages": self.messages}
        if self.options.max_tokens:
            # Reserve output explicitly: vLLM otherwise trims it silently near the limit. (The provider names the
            # field the way this endpoint and model expect.)
            body["max_tokens"] = self.options.max_tokens
        if self.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in self.tools
            ]
        data, streamed = self.provider.chat(body, self.options, on_event)
        turn = self._take(data)
        if turn.text and on_event and not streamed:
            on_event("text", turn.text)
        return turn

    def _take(self, data: dict) -> Turn:
        """A chat completion (streamed and reassembled, or plain) -> Turn; the reply is appended to the history."""
        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            if not isinstance(msg, dict):
                raise TypeError("message is not an object")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(f"unexpected response from {self.provider.name}: {str(data)[:300]}") from exc
        text = _text(msg.get("content"))
        finish = choice.get("finish_reason") or "stop"
        calls: list[ToolCall] = []
        replay: list[dict] = []
        tool_calls = [tc for tc in msg.get("tool_calls") or [] if isinstance(tc, dict)]
        for n, tc in enumerate(tool_calls):
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = str(fn.get("name") or "")
            args, err, raw = _arguments(fn.get("arguments"))
            # Cut by the output cap: invalid arguments, or a last call that has its name but no arguments yet
            if finish == "length" and (err or (n == len(tool_calls) - 1 and fn.get("arguments") in (None, ""))):
                err = CUT
            cid = self.provider.unique_call_id(tc.get("id"))
            calls.append(ToolCall(id=cid, name=name, arguments=args, parse_error=err))
            # What goes back into the history: our id (the tool results carry it) and valid JSON arguments (strict
            # servers reject every later request otherwise). Other fields, e.g. Gemini's signatures, are kept.
            replay.append({**tc, "id": cid, "type": tc.get("type") or "function",
                           "function": {**fn, "name": name, "arguments": raw}})
        thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
        thinking = thinking if isinstance(thinking, str) else ""
        assistant: dict = {"role": "assistant", "content": text}
        if replay:
            assistant["tool_calls"] = replay
        if self.provider.replays_reasoning:
            assistant["reasoning_content"] = thinking
        self.messages.append(assistant)
        u = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        details = u.get("prompt_tokens_details") if isinstance(u.get("prompt_tokens_details"), dict) else {}
        usage = Usage(_int(u.get("prompt_tokens")), _int(u.get("completion_tokens")),
                      cache_read_tokens=_int(details.get("cached_tokens")))
        stop = STOP.get(finish, finish)
        if calls and stop == "end_turn":
            stop = "tool_use"
        return Turn(text=text, tool_calls=calls, stop_reason=stop, usage=usage, thinking=thinking)


class OpenAICompatProvider(Provider):
    # What this endpoint accepts, learned from its errors (there is one provider instance per run).
    stream = True  # SSE streaming (switched off if the server rejects it)
    stream_usage = True  # stream_options.include_usage (switched off if rejected)
    tokens_key = ""  # "max_tokens" | "max_completion_tokens" once a 400 has told us which one
    guard_private = False  # vet the address actually connected to (user endpoints on multi-user servers)

    def __init__(self, name: str, base_url: str, api_key: str = "", extra_headers: dict | None = None, *,
                 guard_private: bool = False):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.guard_private = guard_private

    @property
    def replays_reasoning(self) -> bool:
        """DeepSeek's thinking mode answers 400 unless assistant messages carry their reasoning_content back. Other
        servers don't need it (it would only add tokens), so it stays off for them."""
        return "deepseek" in f"{self.name} {self.base_url}".lower()

    def max_tokens_key(self, model: str) -> str:
        """OpenAI's own API and its reasoning models (o-series, gpt-5) take max_completion_tokens; most other servers
        only know max_tokens. Decided by endpoint and model family (not by the provider's name), or learned from a 400."""
        if self.tokens_key:
            return self.tokens_key
        host = (urlparse(self.base_url).hostname or "").lower()
        if host == "api.openai.com" or host.endswith((".openai.azure.com", ".cognitiveservices.azure.com")):
            return "max_completion_tokens"
        return "max_completion_tokens" if re.match(r"(o\d|gpt-5)", model.lower()) else "max_tokens"

    def unique_call_id(self, cid) -> str:
        """The server's tool-call id, or a fresh one when it is missing, empty or already used in this run: ids key
        the tool results, approvals and the UI, so they must never repeat."""
        ids = self.__dict__.setdefault("_ids", set())
        cid = str(cid or "").strip()
        if not cid or cid in ids:
            cid = "call_" + uuid.uuid4().hex[:24]
        ids.add(cid)
        return cid

    def _policy(self):
        if not self.guard_private or proxied(self.base_url):
            return None
        return lambda ip: address_problem(ip, strict=True)

    # ------------------------------------------------------------------ requests
    def chat(self, body: dict, options: ModelOptions, on_event: EventSink | None = None) -> tuple[dict, bool]:
        """POST /chat/completions with retries, streamed when the server allows it. Returns (response, streamed);
        a streamed reply is reassembled into the non-streaming shape."""
        cancel = getattr(options, "cancel", None)
        timeouts = [0]

        def attempt(n: int) -> tuple[dict, bool]:
            last: _Status | None = None
            for _ in range(4):  # protocol fallbacks (stream_options, streaming, max-tokens field) cost no attempt
                req = self._prepare(body)
                try:
                    if self.stream:
                        return self._stream(req, cancel, on_event)
                    return self._post_once("/chat/completions", req, cancel, timeouts), False
                except _Status as exc:
                    if not self._adapt(exc, req):
                        raise exc.error(self.name) from None
                    last = exc
            raise last.error(self.name)  # type: ignore[union-attr]

        return with_retries(attempt, label=self.name, cancel=cancel, notify=getattr(options, "notify", None),
                            attempts=getattr(options, "max_attempts", None))

    def _prepare(self, body: dict) -> dict:
        req = {k: v for k, v in body.items() if k not in ("max_tokens", "max_completion_tokens")}
        limit = body.get("max_tokens") or body.get("max_completion_tokens")
        if limit:
            req[self.max_tokens_key(str(body.get("model", "")))] = limit
        if self.stream:
            req["stream"] = True
            if self.stream_usage:
                req["stream_options"] = {"include_usage": True}
        return req

    def _adapt(self, exc: "_Status", req: dict) -> bool:
        """Learn from a 400/422 what this server does not accept (each switch happens once). True = send again."""
        from . import parse_limit  # the package imports this module

        if exc.status not in (400, 422) or parse_limit(exc.text):
            return False
        text = exc.text.lower()
        if req.get("stream_options") and "stream_options" in text:
            self.stream_usage = False
            return True
        if req.get("stream") and re.search(r"\bstream(ing)?\b", text):
            self.stream = False
            return True
        key = next((k for k in ("max_completion_tokens", "max_tokens") if k in req), "")
        if key and not self.tokens_key and key in text and re.search(
                r"unsupported|unrecognized|not supported|not allowed|not permitted|extra|unknown", text):
            self.tokens_key = "max_tokens" if key == "max_completion_tokens" else "max_completion_tokens"
            return True
        return False

    def _stream(self, req: dict, cancel, on_event) -> tuple[dict, bool]:
        content = _dumps(req)
        call = Call(self._policy())
        first, idle = first_byte_timeout(), idle_timeout()
        url = self.base_url + "/chat/completions"

        def produce(call: Call, emit) -> tuple[dict, bool]:
            # httpx's read timeout is only a backstop: pump() enforces the first-byte and idle limits precisely
            limit = httpx.Timeout(connect=CONNECT_TIMEOUT, read=max(first, idle) + 60, write=120.0, pool=CONNECT_TIMEOUT)
            with httpx.Client(timeout=limit) as client:
                with client.stream("POST", url, content=content, headers=self.headers,
                                   extensions={"trace": call.trace}) as resp:
                    call.touch()
                    if resp.status_code >= 400:
                        raise _Status(resp.status_code, _read_text(resp, call), resp.headers)
                    return _read_reply(resp, call, emit, self.name)

        data, streamed = self._run(produce, call, cancel, on_event, first, idle)
        if not streamed:
            # The server ignores `stream`: it only answers once the whole reply is ready, so from now on use the plain
            # request and its long read timeout (the first-byte limit would cut long replies).
            self.stream = False
        return data, streamed

    def _post_once(self, path: str, req: dict, cancel, timeouts: list[int] | None = None) -> dict:
        content = _dumps(req)
        call = Call(self._policy())
        url = self.base_url + path

        def produce(call: Call, emit) -> dict:
            limit = httpx.Timeout(connect=CONNECT_TIMEOUT, read=NON_STREAM_READ_TIMEOUT, write=120.0,
                                  pool=CONNECT_TIMEOUT)
            with httpx.Client(timeout=limit) as client:
                with client.stream("POST", url, content=content, headers=self.headers,
                                   extensions={"trace": call.trace}) as resp:
                    call.touch()
                    if resp.status_code >= 400:
                        raise _Status(resp.status_code, _read_text(resp, call), resp.headers)
                    return _read_reply(resp, call, emit, self.name)[0]

        try:
            return self._run(produce, call, cancel)
        except _Timeout as exc:
            if timeouts is not None:
                timeouts[0] += 1
                if timeouts[0] > 1:  # the whole reply did not fit in the read timeout twice: a third try won't either
                    raise LLMError(f"{self.name} request failed: {exc.reason} (twice; no reply within "
                                   f"{NON_STREAM_READ_TIMEOUT:.0f} s)") from exc
            raise

    def _run(self, produce, call: Call, cancel, on_event=None, first=None, idle=None):
        """pump() + the mapping of transport failures to retryable / final errors."""
        try:
            return pump(produce, call, cancel=cancel, on_event=on_event, first_byte=first, idle=idle)
        except Retryable as exc:
            exc.partial = exc.partial or call.emitted > 0
            raise
        except (httpx.UnsupportedProtocol, httpx.InvalidURL) as exc:
            raise LLMError(f"{self.name}: invalid endpoint URL {self.base_url!r}: {exc}") from exc
        except httpx.LocalProtocolError as exc:  # refused before sending: retrying cannot help
            raise LLMError(f"{self.name}: invalid request ({exc}); check the API key for stray spaces or line "
                           "breaks") from exc
        except httpx.TransportError as exc:
            if call.blocked:
                raise LLMError(f"{self.name}: {call.blocked}") from exc
            detail = str(exc) or type(exc).__name__
            if isinstance(exc, httpx.TimeoutException):
                raise _Timeout(f"timed out: {detail}", partial=call.emitted > 0) from exc
            where = "connection lost mid-reply" if call.data else "connection error"
            raise Retryable(f"{where}: {detail}", partial=call.emitted > 0) from exc
        except httpx.HTTPError as exc:  # e.g. a body that cannot be decoded
            raise Retryable(f"bad response: {exc}", partial=call.emitted > 0) from exc
        except _Incomplete as exc:
            raise Retryable(str(exc), partial=call.emitted > 0) from exc

    def post(self, path: str, body: dict, cancel=None, *, notify=None, max_attempts: int | None = None) -> dict:
        """Plain JSON POST with retries (images, vision, and chat on servers that cannot stream)."""
        timeouts = [0]

        def attempt(n: int) -> dict:
            try:
                return self._post_once(path, body, cancel, timeouts)
            except _Status as exc:
                raise exc.error(self.name) from None

        return with_retries(attempt, label=self.name, cancel=cancel, notify=notify, attempts=max_attempts)

    def conversation(self, model, system, tools, options: ModelOptions) -> Conversation:
        return OpenAICompatConversation(self, model, system, tools, options)

    def transcribe(self, model: str, audio: bytes, filename: str, language: str = "") -> str:
        data = {"model": model}
        if language:
            data["language"] = language
        headers = {k: v for k, v in self.headers.items() if k.lower() != "content-type"}
        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=15.0)) as client:
                r = client.post(self.base_url + "/audio/transcriptions", data=data,
                                files={"file": (filename, audio, "audio/wav")}, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name}: could not reach the transcription endpoint: {exc}") from exc
        if r.status_code >= 400:
            raise LLMError(f"{self.name} transcription failed: HTTP {r.status_code}: {r.text[:500]}")
        try:
            return str(r.json().get("text", "")).strip()
        except ValueError:
            return r.text.strip()

    def generate_image(self, model: str, prompt: str) -> tuple[bytes, str]:
        import base64
        import binascii

        body = {"model": model, "prompt": prompt, "n": 1, "response_format": "b64_json"}
        for attempt in range(3):
            try:
                data = self.post("/images/generations", body)
                break
            except LLMError as exc:
                text = str(exc).lower()
                wrong_backend = "language model" in text or "not available on this endpoint" in text or "does not support image" in text
                if not wrong_backend:
                    raise
                if attempt == 2:  # a load-balanced group keeps landing on a text-only model
                    raise LLMError(
                        f"NOT RETRYABLE — the image model '{model}' of {self.name} is answering with a text-only model "
                        "(it cannot create images). Do not try again with other prompts: tell the user to choose another "
                        "model in Project settings → Models by role → Image generation (e.g. an xAI grok-imagine or "
                        "OpenAI gpt-image model), or to fix that model group at the provider."
                    ) from exc
        try:
            item = data["data"][0]
            if not isinstance(item, dict):
                raise TypeError("not an object")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected image response from {self.name}: {str(data)[:300]}") from exc
        if item.get("b64_json"):
            try:
                raw = base64.b64decode(item["b64_json"])
            except (binascii.Error, ValueError, TypeError) as exc:
                raise LLMError(f"{self.name} returned an image that is not valid base64") from exc
            return raw, image_mime(raw) or str(item.get("mime_type") or "image/png")
        if item.get("url"):
            # The URL comes from the provider: same network policy as endpoints (checked at connect time, no
            # redirects into private networks), a size cap, and the bytes must really be an image.
            return fetch_image(str(item["url"]), strict=self.guard_private)
        raise LLMError(f"{self.name} returned no image")

    def vision(self, model: str, prompt: str, image: bytes, mime: str) -> tuple[str, Usage]:
        import base64

        url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        data = self.post("/chat/completions", {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}},
        ]}]})
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected vision response from {self.name}: {str(data)[:300]}") from exc
        u = data.get("usage") or {}
        return text, Usage(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0))


# ---------------------------------------------------------------------- helpers
class _Status(Exception):
    """An HTTP error status from the server (with up to 64 KB of its body)."""

    def __init__(self, status: int, text: str, headers=None):
        super().__init__(f"HTTP {status}")
        self.status, self.text = status, text
        self.retry_after = retry_after_seconds(headers)

    def error(self, name: str) -> Exception:
        what = f"HTTP {self.status}: {self.text[:1500]}"
        if self.status in RETRY_STATUS:
            return Retryable(what, self.retry_after)
        return LLMError(f"{name} request failed: {what}")


class _Timeout(Retryable):
    """A read timeout (a plain JSON request is sent again after one only once)."""


class _Incomplete(Exception):
    """The reply stream ended before it was complete (no finish_reason and no [DONE])."""


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _text(content) -> str:
    if isinstance(content, list):  # some servers return content parts
        return "".join(p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return content if isinstance(content, str) else ""


def _arguments(raw) -> tuple[dict, str, str]:
    """A tool call's arguments -> (arguments, parse error, JSON text to replay in the history)."""
    if raw is None or raw == "":
        return {}, "", "{}"
    if isinstance(raw, dict):
        return dict(raw), "", json.dumps(raw, ensure_ascii=False)
    if not isinstance(raw, str):
        return {}, "arguments must be a JSON object", "{}"
    try:
        args = json.loads(raw)
    except ValueError as exc:
        return {}, f"invalid JSON arguments: {exc}", "{}"
    if not isinstance(args, dict):
        return {}, "arguments must be a JSON object", "{}"
    return args, "", raw


def _dumps(body: dict) -> bytes:
    try:
        return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except UnicodeEncodeError:  # lone surrogates (undecodable bytes in a file): escape them instead of crashing
        return json.dumps(body, separators=(",", ":")).encode("ascii")


def _read_text(resp: httpx.Response, call: Call, limit: int = 65536) -> str:
    buf = bytearray()
    try:
        for chunk in resp.iter_bytes():
            call.touch()
            buf += chunk
            if len(buf) >= limit:
                break
    except httpx.HTTPError:
        pass
    return bytes(buf[:limit]).decode("utf-8", "replace")


def _not_api(name: str, ctype: str, sample: bytes) -> str:
    snippet = sample[:160].decode("utf-8", "replace").strip()
    return (f"{name}: the endpoint did not answer like an OpenAI-compatible API ({ctype or 'not JSON'}: {snippet!r}); "
            "check the base URL (it usually ends in /v1)")


# Error kinds another attempt cannot fix (OpenAI-style "type"/"code", Google-style "status")
_FINAL_ERRORS = re.compile(r"invalid_request|invalid_argument|authenticat|permission|forbidden|not_found|"
                           r"invalid_api_key|failed_precondition", re.I)


def _is_error_list(data) -> bool:
    """Google-style errors come wrapped in a list: [{"error": {...}}]."""
    return isinstance(data, list) and any(isinstance(p, dict) and p.get("error") for p in data)


def _reply_error(payload, name: str) -> Exception:
    """An error object inside a 200 reply (mid-stream or as plain JSON): transient ones are retried."""
    from . import parse_limit

    if _is_error_list(payload):
        payload = next(p for p in payload if isinstance(p, dict) and p.get("error"))
    err = payload.get("error") if isinstance(payload, dict) and payload.get("error") else payload
    message = str((err.get("message") or err.get("type") or err) if isinstance(err, dict) else err)
    raw_code = (err.get("code") or err.get("status")) if isinstance(err, dict) else None
    kind = " ".join(str(err.get(k) or "") for k in ("type", "status", "code")) if isinstance(err, dict) else ""
    try:
        code = int(raw_code)
    except (TypeError, ValueError):
        code = None
    text = f"error in the reply: {message}"[:1500]
    if parse_limit(message) or (code is not None and 400 <= code < 500 and code not in RETRY_STATUS) \
            or _FINAL_ERRORS.search(kind):
        return LLMError(f"{name} request failed: {text}")
    return Retryable(text)


def _json_reply(raw: bytes, name: str) -> dict:
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise LLMError(_not_api(name, "", raw)) from exc
    if (isinstance(data, dict) and data.get("error") and not data.get("choices")) or _is_error_list(data):
        raise _reply_error(data, name)
    if not isinstance(data, dict):
        raise LLMError(f"unexpected response from {name}: {str(data)[:300]}")
    return data


def _read_reply(resp: httpx.Response, call: Call, emit, name: str) -> tuple[dict, bool]:
    """Read a chat-completion reply: SSE (reassembled; deltas emitted as they arrive) or a plain JSON body (servers
    that ignore `stream`). HTML or other non-API answers raise LLMError (wrong URL, proxy, captive portal)."""
    ctype = resp.headers.get("content-type", "").lower()
    decoder, reply = _SSE(), _Reply(name)
    head, mode, raw, sample = b"", "", bytearray(), b""
    try:
        for chunk in resp.iter_bytes():
            call.touch()
            if not mode:
                head += chunk
                start = head.lstrip()[:1]
                if not start:
                    continue
                if "html" in ctype or start == b"<":
                    raise LLMError(_not_api(name, ctype, head))
                mode = "json" if start in (b"{", b"[") and "event-stream" not in ctype else "sse"
                chunk, head, sample = head, b"", head[:200]
            if mode == "json":
                raw += chunk
                if len(raw) > MAX_BODY:
                    raise LLMError(f"{name}: the reply is larger than {MAX_BODY // 2**20} MB")
                continue
            for event, data in decoder.feed(chunk):
                reply.add(event, data, emit, call)
            if reply.done:
                break
        else:
            for event, data in decoder.flush():
                reply.add(event, data, emit, call)
    except httpx.TransportError:
        if mode != "sse" or call.reason or reply.finish is None:
            raise
        # the server dropped the connection after the reply was complete (its finish_reason arrived): keep it
    if mode == "json":
        return _json_reply(bytes(raw), name), False
    if not mode:
        raise _Incomplete("the server sent an empty reply")
    if not reply.complete:
        if not reply.seen and "event-stream" not in ctype:
            # neither SSE nor JSON (plain text from a wrong URL, a proxy, a captive portal): retrying cannot help
            raise LLMError(_not_api(name, ctype, sample))
        raise _Incomplete("the reply stream ended before it was complete")
    return reply.result(), True


class _SSE:
    """Server-sent events -> (event, data) pairs. Handles LF/CRLF/CR line ends (also split across chunks),
    comments, multi-line data, and servers that leave out the blank line between events."""

    _EOL = re.compile(r"\r\n|\r|\n")

    def __init__(self):
        self._utf8 = codecs.getincrementaldecoder("utf-8")("replace")
        self._buf = ""
        self._event = ""
        self._data: list[str] = []

    def feed(self, chunk: bytes) -> list[tuple[str, str]]:
        self._buf += self._utf8.decode(chunk)
        return self._lines(final=False)

    def flush(self) -> list[tuple[str, str]]:
        self._buf += self._utf8.decode(b"", final=True)
        out = self._lines(final=True)
        if self._buf:
            out += self._line(self._buf)
            self._buf = ""
        return out + self._dispatch()

    def _lines(self, final: bool) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        buf, pos = self._buf, 0
        while (m := self._EOL.search(buf, pos)) is not None:
            if m.group() == "\r" and m.end() == len(buf) and not final:
                break  # maybe the first half of a CRLF split across chunks
            out += self._line(buf[pos:m.start()])
            pos = m.end()
        self._buf = buf[pos:]
        return out

    def _line(self, line: str) -> list[tuple[str, str]]:
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return []  # comment / keep-alive
        if line[:1] in ("{", "["):
            field, value = "data", line  # a bare JSON line
        else:
            field, sep, value = line.partition(":")
            if not sep:
                return []
            if value.startswith(" "):
                value = value[1:]
        if field == "event":
            self._event = value.strip()
            return []
        if field != "data":
            return []  # id, retry
        out = self._dispatch() if self._data and _whole("\n".join(self._data)) else []
        self._data.append(value)
        return out

    def _dispatch(self) -> list[tuple[str, str]]:
        if not self._data:
            self._event = ""
            return []
        out = [(self._event, "\n".join(self._data))]
        self._event, self._data = "", []
        return out


def _whole(data: str) -> bool:
    if data.strip() == "[DONE]":
        return True
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


class _Reply:
    """A streamed chat completion, reassembled into the non-streaming shape."""

    def __init__(self, name: str):
        self.name = name
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.calls: dict[int, dict] = {}
        self.ids: dict[str, int] = {}
        self.last = -1  # the call that index-less deltas continue
        self.finish: str | None = None
        self.usage: dict | None = None
        self.done = False
        self.seen = 0  # SSE data events received (0 at the end: the body was not an event stream at all)

    @property
    def complete(self) -> bool:
        return self.done or self.finish is not None

    def add(self, event: str, data: str, emit, call: Call) -> None:
        self.seen += 1
        if data.strip() == "[DONE]":
            self.done = True
            return
        try:
            chunk = json.loads(data)
        except ValueError:
            return  # keep-alive or junk between events
        if _is_error_list(chunk):
            raise _reply_error(chunk, self.name)
        if not isinstance(chunk, dict):
            return
        if event == "error" or chunk.get("error"):
            raise _reply_error(chunk, self.name)
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]  # usually only in the last chunk (stream_options.include_usage)
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict) or (choice.get("index") or 0) != 0:
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, list):
                    content = _text(content)
                if isinstance(content, str) and content:
                    self.text.append(content)
                    emit("text", content)
                    call.data = True  # output has started: from now on the idle limit applies (not on a role-only
                    # first chunk: hidden reasoning may follow it, and it keeps the longer first-byte limit)
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(thought, str) and thought:
                    self.reasoning.append(thought)
                    emit("thinking", thought)
                    call.data = True
                for tc in delta.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        self._tool(tc)
                        call.data = True
            if choice.get("finish_reason"):
                if choice["finish_reason"] == "error":
                    raise Retryable("the provider reported an error in the middle of the reply")
                self.finish = str(choice["finish_reason"])

    def _tool(self, tc: dict) -> None:
        idx, cid = tc.get("index"), str(tc.get("id") or "")
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        if cid and cid in self.ids:
            idx = self.ids[cid]
        elif isinstance(idx, int):
            cur = self.calls.get(idx)
            if cur is not None and cid and cur["id"] and cur["id"] != cid:
                idx = max(self.calls) + 1  # a server that numbers every call 0
        elif cid or self.last < 0 or (fn.get("name") and self._closed(self.last)):
            idx = max(self.calls, default=-1) + 1  # no index: a new call...
        else:
            idx = self.last  # ...or the continuation of the previous one
        cur = self.calls.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        self.last = idx
        if cid and not cur["id"]:
            cur["id"] = cid
            self.ids[cid] = idx
        if tc.get("type"):
            cur["type"] = tc["type"]
        if fn.get("name") and not cur["function"]["name"]:
            cur["function"]["name"] = str(fn["name"])
        args = fn.get("arguments")
        if isinstance(args, str):
            cur["function"]["arguments"] += args
        elif isinstance(args, dict):
            cur["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
        for key, value in fn.items():
            if key not in ("name", "arguments") and value is not None:
                cur["function"].setdefault(key, value)
        for key, value in tc.items():
            if key not in ("index", "id", "type", "function") and value is not None:
                cur.setdefault(key, value)  # e.g. Gemini's extra_content (thought signatures)

    def _closed(self, idx: int) -> bool:
        try:
            return isinstance(json.loads(self.calls[idx]["function"]["arguments"] or "x"), dict)
        except ValueError:
            return False

    def result(self) -> dict:
        message: dict = {"role": "assistant", "content": "".join(self.text)}
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        if self.calls:
            message["tool_calls"] = [self.calls[i] for i in sorted(self.calls)]
        return {"choices": [{"index": 0, "message": message, "finish_reason": self.finish}], "usage": self.usage or {}}
