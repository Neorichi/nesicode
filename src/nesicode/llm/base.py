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
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # end_turn | tool_use | max_tokens | refusal
    usage: Usage = field(default_factory=Usage)
    thinking: str = ""


@dataclass
class ModelOptions:
    effort: str = "high"
    max_tokens: int = 64000
    role: str = "executor"  # executor | proposer | composer | selector | reflector | subagent
    cancel: Any = None  # threading.Event: set when the user presses Stop (aborts a request in flight)


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

    def mask_tool_results(self, keep_last: int, marker: Callable[[str, str], str | None], *,
                          keep_chars: int = 0, apply: bool = True) -> int:
        """Replace the content of older tool results with `marker(call_id, old)` (None = keep) and return the number
        of characters freed. Never touched: the newest `keep_last` results, any newer result while the cumulative size
        of the newer results is <= `keep_chars`, and the results of the latest assistant turn (the round the model has
        not answered yet). apply=False only computes what would be freed (masking breaks the prompt cache, so callers
        dry-run first). Default: not supported (0)."""
        return 0

    @staticmethod
    def maskable(sizes: list[int], pending: int, keep_last: int, keep_chars: int) -> list[int]:
        """Indexes (oldest first) of the tool results that may be masked, given their sizes; the last `pending`
        results belong to the turn the model has not answered yet. Newest results are kept by count and by size."""
        keep = max(pending, keep_last)
        out, acc, budget_open = [], 0, True
        for i in range(len(sizes) - 1, -1, -1):
            if len(sizes) - i <= keep:
                acc += sizes[i]
                continue
            if budget_open and acc + sizes[i] <= keep_chars:
                acc += sizes[i]
                continue
            budget_open = False  # past the newest window: everything older may go
            out.append(i)
        return sorted(out)


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
        conv = self.conversation(model, system, [], options)
        conv.add_user(prompt)
        turn = conv.step()
        if turn.stop_reason == "refusal":
            raise LLMError("the model declined this request (refusal)")
        return turn.text, turn.usage
