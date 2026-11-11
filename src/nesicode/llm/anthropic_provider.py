"""Claude via the official Anthropic SDK (streaming manual tool loop).

Each request streams on a helper thread (`transport.pump`), so Stop aborts it at once: while connecting, while Claude
streams a large tool input, and between retries. A stalled request (no bytes at all — the server's pings count as
life — for the first-byte / idle limits) is aborted and retried, like the OpenAI-compatible path. The retries are ours
(the SDK's are off: its backoff cannot be cancelled and it does not cover errors that arrive mid-stream). Prompt
caching: a breakpoint on the system prompt (which caches the tools too) and a rolling one on the last message."""

from __future__ import annotations

import json
import os
import re
import threading

import anthropic
import httpx2

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
    scoped,
)
from .transport import (
    CONNECT_TIMEOUT,
    RETRY_STATUS,
    Call,
    Retryable,
    address_problem,
    first_byte_timeout,
    idle_timeout,
    proxied,
    pump,
    retry_after_seconds,
    with_retries,
)

# $ per 1M tokens: (input, output, cache read). Cache writes (5-minute TTL) cost 1.25x input. Unknown models cost 0;
# dated ids (claude-opus-4-5-20251101) use the price of their family.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-mythos-5-1": (10.0, 50.0, 0.25),
    "claude-fable-5": (10.0, 50.0, 1.0),
    "claude-mythos-5": (10.0, 50.0, 1.0),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
    "claude-opus-4-7": (5.0, 25.0, 0.50),
    "claude-opus-4-6": (5.0, 25.0, 0.50),
    "claude-opus-4-5": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30),
    "claude-sonnet-4-5": (3.0, 15.0, 0.30),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
}
LONG_PROMPT_PRICES = {"claude-haiku-5-5": (100_000, (0.50, 2.50, 0.05))}  # prompts over N tokens cost more
CACHE_WRITE = 1.25
EPHEMERAL = {"type": "ephemeral"}
CACHEABLE = ("text", "image", "document", "tool_result")
CUT = "output was cut off at max_tokens; resend a smaller call"

# Models that accept the server-side `fallbacks: "default"` refusal fallback.
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_VERSION = re.compile(r"claude-(?:opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?!\d)")
_VERSION_OLD = re.compile(r"claude-(\d+)(?:[-.](\d{1,2}))?-(?:opus|sonnet|haiku)")  # claude-3-7-sonnet-20250219


def claude_version(model: str) -> tuple[int, int] | None:
    """(major, minor) of a Claude model id: claude-opus-4-5-20251101 -> (4, 5), claude-sonnet-4-20250514 -> (4, 0),
    claude-3-7-sonnet-20250219 -> (3, 7), claude-opus-5-5 -> (5, 5). None for names that are not Claude ids."""
    m = _VERSION.search(model or "") or _VERSION_OLD.search(model or "")
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


def thinking_params(model: str, effort: str) -> dict:
    """Adaptive thinking and effort only where the model takes them: Claude 4.6+ and 5.x (and names that are not
    Claude ids, as before). Older models answer 400 to both, except Opus 4.5, which takes effort low/medium/high."""
    version = claude_version(model)
    if version is None or version >= (4, 6):
        out: dict = {"thinking": {"type": "adaptive", "display": "summarized"}}
        if effort:
            # xhigh arrived with Opus 4.7: Opus/Sonnet 4.6 stop at high/max
            out["output_config"] = {"effort": "high" if effort == "xhigh" and version and version < (4, 7) else effort}
        return out
    if version == (4, 5) and "opus" in model and effort in ("low", "medium", "high"):
        return {"output_config": {"effort": effort}}
    return {}


def _price_key(model: str) -> str | None:
    m = (model or "").lower()
    m = m[m.find("claude-"):] if "claude-" in m else m  # "anthropic/claude-…", "us.anthropic.claude-…"
    if m in PRICES:
        return m
    return max((k for k in PRICES if m.startswith((k + "-", k + "@"))), key=len, default=None)


def _cost(model: str, fresh: int, read: int, write: int, output: int) -> float:
    key = _price_key(model)
    if key is None:
        return 0.0
    pin, pout, pread = PRICES[key]
    if key in LONG_PROMPT_PRICES and fresh + read + write > LONG_PROMPT_PRICES[key][0]:
        pin, pout, pread = LONG_PROMPT_PRICES[key][1]
    return (fresh * pin + write * pin * CACHE_WRITE + read * pread + output * pout) / 1_000_000


def _billed(final, requested: str) -> float:
    """$ of a reply, at the prices of the model that ran each part (one we have no price for: the requested model's).
    With a refusal fallback, usage.iterations lists every attempt, the declined ones too (billed), while the top-level
    usage covers only the one that served the reply."""

    def cost(model, u) -> float:
        model = str(model or "")
        fresh, read, write, output = (int(getattr(u, k, 0) or 0) for k in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"))
        return _cost(model if _price_key(model) else requested, fresh, read, write, output)

    attempts = getattr(final.usage, "iterations", None)
    if attempts:
        return sum(cost(getattr(a, "model", None), a) for a in attempts)
    return cost(getattr(final, "model", None), final.usage)


def _kind(block) -> str:
    return str(getattr(block, "type", "") or "")


def _kept(blocks: list) -> list[int]:
    """Indexes of the reply's blocks that make up the turn. A refusal fallback mid-output leaves the declined model's
    partial before a `fallback` block, and of it only the text counts (the fallback model continued from it). Its
    thinking and tool calls are left out, as the API requires when the turn is sent back: those calls never run, and
    no tool_result is sent for them. So are its server-tool calls without their result and any other internal block.
    The `fallback` blocks stay where they are (the API checks the turn against them)."""
    last = max((i for i, b in enumerate(blocks) if _kind(b) == "fallback"), default=-1)
    if last < 0:
        return list(range(len(blocks)))
    head = blocks[:last]
    calls = {b.id for b in head if _kind(b) == "server_tool_use" and getattr(b, "id", None)}
    paired = {r for b in head if (r := getattr(b, "tool_use_id", None)) in calls}

    def keep(b) -> bool:
        kind = _kind(b)
        if kind == "fallback" or (kind == "text" and getattr(b, "text", "")):
            return True
        if kind == "server_tool_use":
            return getattr(b, "id", None) in paired
        return getattr(b, "tool_use_id", None) in paired  # the result of a server-tool call that is kept

    return [i for i, b in enumerate(head) if keep(b)] + list(range(last, len(blocks)))


def _rolling_breakpoint(messages: list[dict]) -> list[dict]:
    """A copy of the history whose last block carries a cache breakpoint. The history itself is never changed, so a
    request has at most two breakpoints (the system prompt and this one) and the next request reads this prefix."""
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else None
    if isinstance(content, str) and content:
        blocks = [{"type": "text", "text": content, "cache_control": EPHEMERAL}]
    elif isinstance(content, list) and content and isinstance(content[-1], dict) \
            and content[-1].get("type") in CACHEABLE:
        blocks = [*content[:-1], {**content[-1], "cache_control": EPHEMERAL}]
    else:
        return messages
    return [*messages[:-1], {**last, "content": blocks}]


def _whole_json(raw: str) -> bool:
    if not raw.strip():
        return True  # no input deltas: the input came whole with the block (or is empty)
    try:
        return isinstance(json.loads(raw), dict)
    except ValueError:
        return False


KEYLESS = ("this provider has no API key saved: add one in Ajustes del proyecto → Proveedores (providers you define "
           "never use the server's ANTHROPIC_API_KEY)")


def _status_error(exc: anthropic.APIStatusError, partial: bool, keyless: bool = False) -> Exception:
    status, kind = exc.status_code, exc.type
    message = exc.api_message or exc.message
    if status == 401:
        return LLMError(f"Anthropic authentication failed: {message}" + (f" — {KEYLESS}" if keyless else ""))
    retry_after = retry_after_seconds(exc.response.headers)
    if status == 429 or kind == "rate_limit_error":
        return Retryable(f"rate limit: {message}", retry_after, partial)
    # status 200 = an error event inside the stream (e.g. overloaded_error under load)
    if status in RETRY_STATUS or status >= 500 or (status == 200 and kind in (None, "overloaded_error", "api_error",
                                                                               "timeout_error")):
        where = "in the reply stream" if status == 200 else str(status)
        return Retryable(f"API error {where} ({kind or 'error'}): {message}", retry_after, partial)
    return LLMError(f"Anthropic API error {status}: {message}")


class AnthropicConversation(Conversation):
    def __init__(self, client: anthropic.Anthropic, first_party: bool, model, system, tools, options, policy=None,
                 keyless: bool = False):
        super().__init__(model, system, tools, options)
        self.client = client
        self.first_party = first_party
        self.policy = policy  # vets the address connected to (user endpoints on multi-user servers)
        self.keyless = keyless  # a provider the user defined without a key (a 401 then says so)
        self.messages: list[dict] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_text(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text or "(no answer)"})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        # All results of one assistant turn go back in a single user message.
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.call_id,
                        "content": r.content or "(empty)",
                        **({"is_error": True} if r.is_error else {}),
                    }
                    for r in results
                ],
            }
        )

    def mask_tool_results(self, keep_last: int, marker, *, keep_chars: int = 0, apply: bool = True) -> int:
        """Masks the content of old tool_result blocks (string or text blocks; results with images are left alone).
        Editing an earlier turn invalidates every later thinking block (Claude's history-editing check: a 400 on
        enforced accounts), so when anything is masked every thinking block is dropped from the history too — the
        model then simply continues without its earlier reasoning."""
        slots = []  # (message index, block index) of every tool_result, oldest first
        last_reply = max((i for i, m in enumerate(self.messages) if m.get("role") == "assistant"), default=-1)
        for i, m in enumerate(self.messages):
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                slots += [(i, j) for j, b in enumerate(m["content"]) if isinstance(b, dict) and b.get("type") == "tool_result"]

        def text_of(block: dict) -> str | None:
            c = block.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list) and all(isinstance(x, dict) and x.get("type") == "text" for x in c):
                return "".join(x.get("text", "") for x in c)
            return None  # images or other blocks: not masked

        sizes = [len(text_of(self.messages[i]["content"][j]) or "") for i, j in slots]
        pending = sum(1 for i, _ in slots if i > last_reply)
        freed = 0
        for k in self.maskable(sizes, pending, keep_last, keep_chars):
            i, j = slots[k]
            block = self.messages[i]["content"][j]
            old = text_of(block)
            if old is None:
                continue
            new = marker(block.get("tool_use_id", ""), old)
            if new is not None and len(new) < len(old):
                freed += len(old) - len(new)
                if apply:
                    self.messages[i]["content"][j] = {**block, "content": new}
        if apply and freed:
            for m in self.messages:
                if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                    kept = [b for b in m["content"] if (b.get("type") if isinstance(b, dict) else getattr(b, "type", None))
                            not in ("thinking", "redacted_thinking")]
                    m["content"] = kept or [{"type": "text", "text": "(no visible reply)"}]
        return freed

    def _params(self) -> dict:
        params: dict = {
            "model": self.model,
            "max_tokens": self.options.max_tokens,
            "messages": self.messages,
        }
        # Prompt caching (Claude ids or the official API; other Anthropic-compatible servers may not take it)
        cache = self.first_party or claude_version(self.model) is not None
        if self.system:
            params["system"] = [{"type": "text", "text": self.system, "cache_control": EPHEMERAL}] if cache else self.system
        if cache and self.messages:
            params["messages"] = _rolling_breakpoint(self.messages)
        if self.tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in self.tools
            ]
        params.update(thinking_params(self.model, self.options.effort))
        betas: list[str] = []
        if self.first_party and self.model in FALLBACK_MODELS:
            betas.append(FALLBACK_BETA)
            params["fallbacks"] = "default"
        if betas:
            params["betas"] = betas
        return params

    def step(self, on_event: EventSink | None = None) -> Turn:
        params = self._params()
        return with_retries(lambda n: self._attempt(params, on_event), label="Anthropic",
                            cancel=getattr(self.options, "cancel", None), notify=getattr(self.options, "notify", None),
                            attempts=getattr(self.options, "max_attempts", None))

    def _attempt(self, params: dict, on_event: EventSink | None) -> Turn:
        call = Call(self.policy)
        inputs: dict[int, list[str]] = {}  # tool-input JSON as it arrived, per content block
        closed: set[int] = set()
        state = {"complete": False}

        def produce(call: Call, emit):
            _local.call = call
            try:
                with self.client.beta.messages.stream(**params) as stream:
                    call.touch()
                    for event in stream:  # Stop is checked on every event, tool-input deltas included
                        call.touch()
                        if call.reason:
                            break
                        if event.type == "content_block_delta":
                            call.data = True
                            delta = event.delta
                            if delta.type == "text_delta":
                                emit("text", delta.text)
                            elif delta.type == "thinking_delta":
                                emit("thinking", delta.thinking)
                            elif delta.type == "input_json_delta":
                                inputs.setdefault(event.index, []).append(delta.partial_json)
                        elif event.type == "content_block_stop":
                            closed.add(event.index)
                        elif event.type == "message_stop":
                            state["complete"] = True
                            break  # the last event: a server that leaves the stream open must not look stalled
                    try:
                        return stream.current_message_snapshot
                    except AssertionError:  # no message_start at all
                        return None
            finally:
                _local.call = None

        partial = lambda: call.emitted > 0  # noqa: E731
        try:
            # first-byte / idle watchdog: a dead connection is aborted and retried instead of waiting 600 s per read
            final = pump(produce, call, cancel=getattr(self.options, "cancel", None), on_event=on_event,
                         first_byte=first_byte_timeout(), idle=idle_timeout())
        except (LLMError, Retryable):
            raise
        except anthropic.APIStatusError as exc:
            raise _status_error(exc, partial(), self.keyless) from exc
        except (anthropic.APIConnectionError, httpx2.TransportError) as exc:
            if call.blocked:
                raise LLMError(f"Anthropic: {call.blocked}") from exc
            if isinstance(exc.__cause__, httpx2.LocalProtocolError):  # refused before sending: retrying cannot help
                raise LLMError(f"Anthropic: invalid request ({exc.__cause__}); check the API key for stray spaces "
                               "or line breaks") from exc
            if isinstance(exc, anthropic.APIConnectionError):  # before the reply started (timeouts included)
                raise Retryable(f"could not reach the API: {exc.__cause__ or exc}", partial=partial()) from exc
            raise Retryable(f"connection lost mid-reply: {exc or type(exc).__name__}", partial=partial()) from exc
        except ValueError as exc:  # the SDK could not parse a tool input
            if "tool parameter JSON" not in str(exc):
                raise
            raise Retryable(f"invalid tool input in the reply: {str(exc)[:200]}", partial=partial()) from exc
        except RuntimeError as exc:
            if "Unexpected event order" not in str(exc):
                raise
            raise Retryable(f"malformed reply stream: {exc}", partial=partial()) from exc
        if final is None or not state["complete"] or final.stop_reason is None:
            # The stream ended without its end (e.g. a proxy closed it after an upstream failure). The SDK silently
            # drops a tool input's unfinished fields, so a partial write_file would empty the file: never use it.
            raise Retryable("the reply stream ended before it was complete", partial=partial())
        return self._take(final, inputs, closed, partial())

    def _take(self, final, inputs: dict[int, list[str]], closed: set[int], partial: bool = False) -> Turn:
        u = final.usage
        read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        fresh = int(getattr(u, "input_tokens", 0) or 0)
        output = int(getattr(u, "output_tokens", 0) or 0)
        # input_tokens is the WHOLE prompt (the loop measures the context with it), of the attempt that served the reply
        usage = Usage(fresh + read + write, output, cache_read_tokens=read, cache_write_tokens=write)
        usage.cost_usd = _billed(final, self.model)
        stop = final.stop_reason
        cut = stop in ("max_tokens", "model_context_window_exceeded")
        blocks = list(final.content or [])
        kept = _kept(blocks)

        if any(_kind(blocks[i]) != "fallback" for i in kept):
            # Pass the content back unchanged (thinking blocks must survive verbatim), but for what a refusal fallback
            # left out of the turn. An empty reply is not passed back: the API refuses an empty assistant message.
            self.messages.append({"role": "assistant", "content": [blocks[i] for i in kept]})
        elif stop in ("end_turn", "tool_use") and not output:
            # Nothing at all and 0 output tokens (Claude's own empty end_turn reports a few): a provider hiccup
            raise Retryable("the reply was empty: no text, thinking or tool call, and 0 output tokens (a provider "
                            "hiccup)", partial=partial)

        if stop == "refusal":
            return Turn(text="", stop_reason="refusal", usage=usage)

        text_parts, thinking_parts, calls = [], [], []
        for i in kept:
            block = blocks[i]
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                raw = "".join(inputs.get(i, []))
                err = ""
                # A partial input (cut by the output cap or a full window): the SDK drops the unfinished field. Cut
                # right after the tool's name (the last block, no input at all), the whole input is missing.
                if i not in closed or not _whole_json(raw) or \
                        (cut and i == len(blocks) - 1 and not raw.strip() and not args):
                    err = CUT if cut else "the tool input arrived incomplete; resend the call"
                calls.append(ToolCall(id=block.id, name=block.name, arguments=args, parse_error=err))
        return Turn(
            text="".join(text_parts),
            tool_calls=calls,
            stop_reason="max_tokens" if cut else stop,  # a full context window cuts the reply like the output cap
            usage=usage,
            thinking="\n".join(t for t in thinking_parts if t),
        )


_local = threading.local()


def _attach_trace(request) -> None:
    """httpx2 request hook: the attempt running on this thread can see (and abort) its request's sockets."""
    call = getattr(_local, "call", None)
    if call is not None:
        request.extensions["trace"] = call.trace


class _Touching(httpx2.SyncByteStream):
    """A response body that reports every chunk it receives to the request's Call. The SDK swallows the server's
    `ping` events, so without this a long but healthy silence between content events (pings only) would look like a
    stall to the idle watchdog."""

    def __init__(self, inner, call: Call):
        self.inner, self.call = inner, call

    def __iter__(self):
        for chunk in self.inner:
            self.call.touch()
            yield chunk

    def close(self) -> None:
        self.inner.close()


def _watch_body(response) -> None:
    """httpx2 response hook (same thread as the attempt): the headers arrived; every body chunk keeps it alive."""
    call = getattr(_local, "call", None)
    if call is not None:
        call.touch()
        response.stream = _Touching(response.stream, call)


def _env_header_names() -> list[str]:
    """Headers the SDK adds from ANTHROPIC_CUSTOM_HEADERS (the server's own configuration)."""
    names = []
    for line in (os.environ.get("ANTHROPIC_CUSTOM_HEADERS") or "").split("\n"):
        name = line.split(":", 1)[0].strip() if ":" in line else ""
        if name:
            names.append(name)
    return names


class AnthropicProvider(Provider):
    name = "anthropic"
    policy = None

    def __init__(self, api_key: str | None = None, base_url: str | None = None, *, isolated: bool = False,
                 guard_private: bool = False):
        """api_key=None lets the SDK find the server's own credentials (environment, profile): only for the built-in
        provider. isolated (user-defined providers): never the server's key, base URL or custom headers; an empty
        key stays empty."""
        http = anthropic.DefaultHttpxClient(
            event_hooks={"request": [_attach_trace], "response": [_watch_body]},
            # a fresh connection per request, so Stop can always reach (and close) its socket
            limits=httpx2.Limits(max_connections=100, max_keepalive_connections=0),
        )
        # The read timeout is only a backstop: the first-byte / idle watchdog in _attempt acts first
        read = max(600.0, max(first_byte_timeout(), idle_timeout()) + 60)
        kwargs: dict = {"max_retries": 0, "timeout": httpx2.Timeout(read, connect=CONNECT_TIMEOUT), "http_client": http}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        elif isolated:
            kwargs["base_url"] = "https://api.anthropic.com"
        if isolated:
            headers: dict = {}
            for name in _env_header_names():  # the server's own headers (maybe credentials for its gateway)
                headers[name] = f"Bearer {api_key}" if name.lower() == "authorization" and api_key else anthropic.Omit()
            if not api_key:  # no key: send no credentials at all (an empty "Bearer " is not even a valid header)
                headers[next((n for n in headers if n.lower() == "authorization"), "Authorization")] = anthropic.Omit()
            if headers:
                kwargs["default_headers"] = headers
        self.client = anthropic.Anthropic(**kwargs)
        self.first_party = (self.client.base_url.host or "") == "api.anthropic.com"
        self.keyless = isolated and not api_key
        # (the SDK's client decides whether a request goes through a proxy: its rules decide whether to vet the peer)
        if guard_private and not proxied(str(self.client.base_url), "anthropic"):
            self.policy = lambda ip: address_problem(ip, strict=True)

    def conversation(self, model, system, tools, options: ModelOptions) -> Conversation:
        return AnthropicConversation(self.client, self.first_party, model, system, tools, options, policy=self.policy,
                                     keyless=self.keyless)

    def vision(self, model: str, prompt: str, image: bytes, mime: str, *, cancel=None, notify=None) -> tuple[str, Usage]:
        import base64

        cancel, notify = scoped(cancel, notify)  # (a tool's request: the agent's Stop and retry notices)
        opts = ModelOptions(effort="low", max_tokens=16000, cancel=cancel, notify=notify)
        conv = AnthropicConversation(self.client, self.first_party, model, "", [], opts, policy=self.policy,
                                     keyless=self.keyless)
        conv.messages.append({"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": base64.b64encode(image).decode()}},
            {"type": "text", "text": prompt},
        ]})
        turn = conv.step()
        if turn.stop_reason == "refusal":
            raise LLMError("the vision model declined this image")
        return turn.text, turn.usage
