"""Claude via the official Anthropic SDK (streaming manual tool loop)."""

from __future__ import annotations

import anthropic

from .base import (
    Conversation,
    EventSink,
    LLMError,
    ModelOptions,
    Provider,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
    Usage,
)

# $ per 1M tokens (input, output). Unknown models report cost 0.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Models that accept the server-side `fallbacks: "default"` refusal fallback.
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Older models that do not take adaptive thinking / effort.
LEGACY_THINKING = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-3")


def _cost(model: str, usage) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (usage.input_tokens * pin + usage.output_tokens * pout) / 1_000_000


class AnthropicConversation(Conversation):
    def __init__(self, client: anthropic.Anthropic, first_party: bool, model, system, tools, options):
        super().__init__(model, system, tools, options)
        self.client = client
        self.first_party = first_party
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
            "system": self.system,
            "messages": self.messages,
        }
        if self.tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in self.tools
            ]
        if not self.model.startswith(LEGACY_THINKING):
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
            if self.options.effort:
                params["output_config"] = {"effort": self.options.effort}
        betas: list[str] = []
        if self.first_party and self.model in FALLBACK_MODELS:
            betas.append(FALLBACK_BETA)
            params["fallbacks"] = "default"
        if betas:
            params["betas"] = betas
        return params

    def step(self, on_event: EventSink | None = None) -> Turn:
        params = self._params()
        try:
            with self.client.beta.messages.stream(**params) as stream:
                for event in stream:
                    if on_event is None or event.type != "content_block_delta":
                        continue
                    if event.delta.type == "text_delta":
                        on_event("text", event.delta.text)
                    elif event.delta.type == "thinking_delta":
                        on_event("thinking", event.delta.thinking)
                final = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError(f"Anthropic authentication failed: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(f"Anthropic rate limit: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc

        usage = Usage(final.usage.input_tokens, final.usage.output_tokens)
        usage.cost_usd = _cost(self.model, usage)
        stop = final.stop_reason or "end_turn"

        # Pass the content back unchanged (thinking blocks must survive verbatim).
        self.messages.append({"role": "assistant", "content": final.content})

        if stop == "refusal":
            return Turn(text="", stop_reason="refusal", usage=usage)

        text_parts, thinking_parts, calls = [], [], []
        for block in final.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                calls.append(ToolCall(id=block.id, name=block.name, arguments=args))
        if stop == "max_tokens":
            for c in calls:
                c.parse_error = "output was cut off at max_tokens; resend a smaller call"
        return Turn(
            text="".join(text_parts),
            tool_calls=calls,
            stop_reason=stop,
            usage=usage,
            thinking="\n".join(t for t in thinking_parts if t),
        )


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        kwargs: dict = {"max_retries": 4}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.first_party = not base_url

    def conversation(self, model, system, tools, options: ModelOptions) -> Conversation:
        return AnthropicConversation(self.client, self.first_party, model, system, tools, options)

    def vision(self, model: str, prompt: str, image: bytes, mime: str) -> tuple[str, Usage]:
        import base64

        conv = AnthropicConversation(self.client, self.first_party, model, "", [], ModelOptions(effort="low", max_tokens=16000))
        conv.messages.append({"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": base64.b64encode(image).decode()}},
            {"type": "text", "text": prompt},
        ]})
        turn = conv.step()
        if turn.stop_reason == "refusal":
            raise LLMError("the vision model declined this image")
        return turn.text, turn.usage
