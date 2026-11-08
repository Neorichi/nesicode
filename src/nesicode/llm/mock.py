"""Deterministic offline provider (model string `mock/<anything>`).

It exercises every code path of the agent and of the harnessing engine without
credentials or cost: the executor really calls tools, and harness edits really
change what the executor does (a `verify` rule makes it read its file back).
"""

from __future__ import annotations

import re

from .base import Conversation, EventSink, ModelOptions, Provider, ToolCall, ToolResult, Turn, Usage

VERIFY_RULE = "- After writing any file, read it back once to verify the content landed."
SKILL_BODY = """---
name: verify-writes
description: Re-open every file you wrote and check it against the request.
---
When you have written a deliverable file:
1. Read it back with `read_file`.
2. Check each requirement of the task against what is on disk.
3. Fix and re-read if anything is missing.
"""
MEMORY_RULE = "- Deliverables in this project are concise Markdown files at the workspace root."


def _harness_file(prompt: str, path: str) -> str | None:
    m = re.search(rf'<harness_file path="{re.escape(path)}">\n?(.*?)</harness_file>', prompt, re.S)
    return m.group(1) if m else None


def _append_line(text: str | None, line: str) -> str:
    text = (text or "").rstrip("\n")
    if line in text:
        return text + "\n"
    return (text + "\n" + line).lstrip("\n") + "\n"


def _inv_edits(prompt: str) -> str:
    guard = _append_line(_harness_file(prompt, "guardrails.md"), VERIFY_RULE)
    guard = _append_line(guard, "- When a deliverable file is written, load the `verify-writes` skill.")
    return (
        f'<file path="guardrails.md">\n{guard}</file>\n'
        f'<file path="skills/verify-writes/SKILL.md">\n{SKILL_BODY}</file>\n'
    )


def _dom_edits(prompt: str) -> str:
    mem = _append_line(_harness_file(prompt, "memory.md"), MEMORY_RULE)
    return f'<file path="memory.md">\n{mem}</file>\n'


class MockConversation(Conversation):
    def __init__(self, model, system, tools, options):
        super().__init__(model, system, tools, options)
        self.messages: list[tuple[str, str]] = []
        self.steps = 0
        self.task = ""

    def add_user(self, text: str) -> None:
        if not self.task:
            self.task = text
        self.messages.append(("user", text))

    def add_assistant_text(self, text: str) -> None:
        self.messages.append(("assistant", text))

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.messages.append(("tool", r.content, r.call_id))

    def mask_tool_results(self, keep_last: int, marker, *, keep_chars: int = 0, apply: bool = True) -> int:
        idx = [i for i, m in enumerate(self.messages) if m[0] == "tool"]
        replies = [i for i, m in enumerate(self.messages) if m[0] == "assistant"]
        pending = sum(1 for i in idx if i > replies[-1]) if replies else 0  # the round not answered yet stays
        freed = 0
        for k in self.maskable([len(self.messages[i][1]) for i in idx], pending, keep_last, keep_chars):
            _, content, call_id = self.messages[idx[k]]
            new = marker(call_id, content)
            if new is not None and len(new) < len(content):
                freed += len(content) - len(new)
                if apply:
                    self.messages[idx[k]] = ("tool", new, call_id)
        return freed

    def _reply(self, text: str, calls: list[ToolCall] | None = None, on_event=None) -> Turn:
        self.steps += 1
        if text and on_event:
            on_event("text", text)
        self.messages.append(("assistant", text))
        return Turn(
            text=text,
            tool_calls=calls or [],
            stop_reason="tool_use" if calls else "end_turn",
            usage=Usage(input_tokens=sum(len(m[1]) for m in self.messages) // 4, output_tokens=len(text) // 4 + 20),
        )

    def step(self, on_event: EventSink | None = None) -> Turn:
        role = self.options.role
        prompt = self.messages[-1][1] if self.messages else ""
        if role in ("executor", "subagent"):
            return self._executor(on_event)
        if role == "proposer":
            body = _inv_edits(prompt) if "YOUR DIMENSION: INV" in prompt else _dom_edits(prompt)
            return self._reply(body + "<notes>mock proposal grounded in base rollout trajectories</notes>", on_event=on_event)
        if role == "composer":
            body = _inv_edits(prompt) + _dom_edits(prompt)
            return self._reply(body + "<notes>kept -> inv + dom edits -> both candidates delivered</notes>", on_event=on_event)
        if role == "selector":
            ids = re.findall(r"^## Candidate ([A-Z0-9]+)", prompt, re.M)
            pick = ids[-1] if ids else "A"
            return self._reply(f"<reasoning>mock pick</reasoning>\n<choice>{pick}</choice>", on_event=on_event)
        if role == "reflector":
            return self._reply(
                "<edits>\ncomponent: guardrails.md\ncandidates: inv1\nchange: verify rule\n"
                "observed: candidate re-read its deliverable\nverdict: helped\n</edits>\n"
                "<experience>\ncue: tasks whose deliverable is a written file\n"
                "lesson: add a guardrails.md rule to read written files back once\n"
                "evidence: inv1 row (helped)\nconfidence: low, single mock rollout\n</experience>",
                on_event=on_event,
            )
        return self._reply("OK", on_event=on_event)

    def _executor(self, on_event) -> Turn:
        verify = "verify" in self.system.lower()
        first_line = (self.task.strip().splitlines() or ["task"])[0][:120]
        n = self.steps
        if n == 0:
            return self._reply("Looking at the workspace first.", [ToolCall("m0", "list_dir", {"path": "."})], on_event)
        if n == 1:
            content = f"# Result\n\nTask: {first_line}\n\nDone by nesicode mock executor.\n"
            return self._reply("Writing the deliverable.", [ToolCall("m1", "write_file", {"path": "NESICODE_RESULT.md", "content": content})], on_event)
        if n == 2 and verify:
            return self._reply("Verifying what I wrote.", [ToolCall("m2", "read_file", {"path": "NESICODE_RESULT.md"})], on_event)
        return self._reply(
            "Done: wrote NESICODE_RESULT.md" + (" and verified it." if verify else "."), on_event=on_event
        )


class MockProvider(Provider):
    name = "mock"

    def conversation(self, model, system, tools, options: ModelOptions) -> Conversation:
        return MockConversation(model, system, tools, options)

    def generate_image(self, model, prompt):
        import struct
        import zlib

        def chunk(t, d):
            c = struct.pack(">I", len(d)) + t + d
            return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

        raw = b"".join(b"\x00" + b"\x0e\x8f\x7e" * 8 for _ in range(8))
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)) + \
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
        return png, "image/png"

    def transcribe(self, model, audio, filename, language=""):
        return f"transcripción de prueba ({len(audio)} bytes)"

    def vision(self, model, prompt, image, mime):
        return f"A {mime} image of {len(image)} bytes (mock vision).", Usage(10, 10)
