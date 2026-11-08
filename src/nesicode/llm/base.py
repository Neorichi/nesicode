"""Provider-neutral conversation interface.

Each provider keeps its own native message list, so provider-specific state (for
example Claude thinking blocks, which must be passed back unchanged) survives intact.
The agent loop only ever talks to `Conversation`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

EventSink = Callable[[str, str], None]  # (kind, text_delta) where kind in {"text", "thinking"}


class LLMError(RuntimeError):
    """A model call failed after the provider's own retries."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str = ""


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Usage:
    input_tokens: int = 0  # the WHOLE prompt of the request (with prompt caching: uncached + cache reads + cache writes)
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_read_tokens: int = 0  # part of input_tokens served from the provider's prompt cache
    cache_write_tokens: int = 0  # part of input_tokens written to it

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens

    def to_dict(self) -> dict:
        out = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }
        if self.cache_read_tokens or self.cache_write_tokens:
            out.update(cache_read_tokens=self.cache_read_tokens, cache_write_tokens=self.cache_write_tokens)
        return out


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # end_turn | tool_use | max_tokens (cut by the output cap or a full context window) | refusal
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    thinking: str = ""


@dataclass
class ModelOptions:
    effort: str = "high"
    max_tokens: int = 64000
    role: str = "executor"  # executor | proposer | composer | selector | reflector | subagent
    cancel: Any = None  # threading.Event: set when the user presses Stop (aborts a request in flight)
    # Called before each retry with {"type": "llm_retry", "attempt": n (the one about to start), "max": m,
    # "wait": seconds, "reason": "...", "partial": bool (part of the reply had already been streamed)}.
    notify: Callable[[dict], None] | None = None
    max_attempts: int | None = None  # attempts per request (None = the provider's default)


class Conversation(ABC):
    def __init__(self, model: str, system: str, tools: list[ToolSpec], options: ModelOptions):
        self.model = model
        self.system = system
        self.tools = tools
        self.options = options

    @abstractmethod
    def add_user(self, text: str) -> None: ...

    @abstractmethod
    def add_assistant_text(self, text: str) -> None:
        """Replay an earlier assistant answer (used for chat history)."""

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None: ...

    @abstractmethod
    def step(self, on_event: EventSink | None = None) -> Turn:
        """Call the model on the current history and append its reply."""

    def mask_tool_results(self, keep_last: int, marker: Callable[[str, str], str | None]) -> int:
        """Replace the content of all but the last `keep_last` tool results with `marker(call_id, old)`
        (None = keep). Returns the number of characters freed. Default: not supported (0)."""
        return 0


class Provider(ABC):
    name: str

    @abstractmethod
    def conversation(
        self, model: str, system: str, tools: list[ToolSpec], options: ModelOptions
    ) -> Conversation: ...

    def vision(self, model: str, prompt: str, image: bytes, mime: str) -> tuple[str, Usage]:
        """Answer `prompt` about one image. Providers without vision raise LLMError."""
        raise LLMError(f"provider {self.name!r} does not support images")

    def transcribe(self, model: str, audio: bytes, filename: str, language: str = "") -> str:
        """Speech to text. Providers without it raise LLMError."""
        raise LLMError(f"provider {self.name!r} cannot transcribe audio")

    def generate_image(self, model: str, prompt: str) -> tuple[bytes, str]:
        """Create one image from a text prompt. Returns (bytes, mime type)."""
        raise LLMError(f"provider {self.name!r} cannot generate images")

    def complete(self, model: str, system: str, prompt: str, options: ModelOptions) -> tuple[str, Usage]:
        """One-shot request. `options.cancel` aborts it (the providers' step() honours it while connecting,
        streaming and waiting between retries)."""
        cancel = getattr(options, "cancel", None)
        if cancel is not None and cancel.is_set():
            raise LLMError("cancelled by the user")
        conv = self.conversation(model, system, [], options)
        conv.add_user(prompt)
        turn = conv.step()
        if turn.stop_reason == "refusal":
            raise LLMError("the model declined this request (refusal)")
        return turn.text, turn.usage
