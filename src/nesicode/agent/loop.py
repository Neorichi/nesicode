"""The nesicode agent loop: harness -> system prompt + tools, then model <-> tools."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from ..harness import Harness, hooks as harness_hooks, skills, sub_agents
from ..llm import LLMError, ModelOptions, ToolCall, ToolResult, Usage, get_provider, parse_limit
from .hooks import Hooks
from . import context as ctxm
from .tools import BASIC_TOOLS, GENERATE_IMAGE, SCHEMAS, SEARCH_HISTORY, SHARE_FILE, SHOW_IMAGE, VIEW_IMAGE, delegate_tool, skill_tool
from .vision import describe_image, generate_image, image_mime
from .workspace import HARNESS_DIR, Cancelled, ToolError, Workspace

Emit = Callable[[dict], None]

FINAL_NUDGE = (
    "You have used your whole step budget. Do not call any more tools. "
    "Reply now with your final summary: what you changed, what you verified, and what is left."
)


@dataclass
class AgentResult:
    final_text: str
    status: str  # done | max_steps | refusal | error | cancelled
    steps: int
    usage: Usage
    events: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def delivered(self) -> bool:
        return self.status in ("done", "max_steps") and bool(self.final_text.strip())


def build_system(h: Harness, ws: Workspace, *, base: str | None = None, extra_tools: dict | None = None) -> str:
    parts = [(base if base is not None else h.get("system_prompt.md", "")).strip()]
    if h.get("guardrails.md", "").strip():
        parts.append("# Guardrails\n\n" + h["guardrails.md"].strip())
    if h.get("memory.md", "").strip():
        parts.append("# Memory\n\n" + h["memory.md"].strip())
    if h.get("tool_guidance.md", "").strip():
        parts.append("# Tool guidance\n\n" + h["tool_guidance.md"].strip())
    sk = skills(h)
    if sk:
        parts.append(
            "# Skills\n\nLoad one with `load_skill` when its situation arises:\n"
            + "\n".join(f"- `{n}`: {s['description']}" for n, s in sorted(sk.items()))
        )
    ag = sub_agents(h)
    if ag and base is None:
        parts.append(
            "# Subagents\n\nDelegate with `delegate`:\n"
            + "\n".join(f"- `{n}`: {a['description']}" for n, a in sorted(ag.items()))
        )
    scripts = sorted(p for p in h if p.startswith("scripts/") or "/scripts/" in p)
    env = [
        "# Environment",
        "",
        "All paths are relative to the workspace root, which is the current directory for `bash`.",
        f"The workspace's absolute path is `{ws.root}`: give external tools (connectors) absolute paths inside it "
        "so the files they create stay in this conversation's folder.",
        f"Today is {time.strftime('%Y-%m-%d')}.",
    ]
    if scripts:
        env.append(
            f"Harness scripts are installed under `{HARNESS_DIR}/` (e.g. `bash {HARNESS_DIR}/{scripts[0]}` or "
            f"`python {HARNESS_DIR}/{scripts[0]}`): " + ", ".join(f"`{s}`" for s in scripts)
        )
    if any(name.startswith("chrome_") for name in (extra_tools or {})):
        env.append(
            "The chrome_* tools act on the USER'S OWN Chrome, with their logged-in sessions (only the tabs in its "
            "'Nesicode' group). Use them for anything the user asks to do in their browser; open pages with chrome_open."
        )
    parts.append("\n".join(env))
    return "\n\n".join(p for p in parts if p)


class Agent:
    def __init__(
        self,
        *,
        model: str,
        harness: Harness,
        workspace: Workspace,
        emit: Emit | None = None,
        user_keys: dict[str, str] | None = None,
        max_steps: int = 60,
        effort: str = "high",
        role: str = "executor",
        cancel: threading.Event | None = None,
        approve: Callable[[ToolCall], tuple[bool, str]] | None = None,
        depth: int = 0,
        system_base: str | None = None,
        label: str = "",
        vision_model: str = "",
        image_model: str = "",
        extra_tools: dict | None = None,
        context_limit: int | None = None,
        output_limit: int | None = None,
        on_limit: Callable[[int], None] | None = None,
        archive: str = ".nesicode",
    ):
        self.model_spec = model
        self.harness = harness
        self.ws = workspace
        self._emit = emit
        self.user_keys = user_keys
        self.max_steps = max_steps
        self.effort = effort
        self.role = role
        self.cancel = cancel or threading.Event()
        self.ws.cancel = self.cancel  # running commands are killed as soon as the user presses Stop
        # Called before a tool runs; returns (allowed, reason). None = everything is allowed (auto mode).
        self.approve = approve
        self.depth = depth
        self.system_base = system_base
        self.label = label
        self.vision_model = vision_model
        self.image_model = image_model
        # name -> (ToolSpec, fn(args, agent) -> str): e.g. MCP connector tools
        self.extra_tools = extra_tools or {}
        # Context window of the model (tokens). The conversation is compacted before it fills up.
        self.context_limit = context_limit
        self.output_limit = output_limit
        self.on_limit = on_limit  # called when the real limit is learned from a provider error
        # Per-conversation archive (transcripts, checkpoints, full outputs) so chats never see each other's.
        self.archive = archive
        self._saved: dict[str, tuple[str, str]] = {}  # call_id -> (tool, saved path)
        self._errors: set[str] = set()
        self._summary: str | None = None
        self._compactions = 0
        self._compact_mark = 0
        self._masked_at = 0
        self.events: list[dict] = []
        try:
            self.hooks = Hooks(harness_hooks(harness))
        except (ValueError, json.JSONDecodeError):
            self.hooks = Hooks({})

    def emit(self, event: dict, persist: bool = True) -> None:
        if self.depth:
            event = {**event, "depth": self.depth, "agent": self.label}
        if persist:
            self.events.append(event)
        if self._emit:
            self._emit({**event, "_persist": persist})

    def tool_specs(self):
        specs = [SCHEMAS[n] for n in BASIC_TOOLS]
        if (st := skill_tool(self.harness)) is not None:
            specs.append(st)
        if self.depth == 0 and (dt := delegate_tool(self.harness)) is not None:
            specs.append(dt)
        if self.vision_model:
            specs.append(VIEW_IMAGE)
        if self.image_model:
            specs.append(GENERATE_IMAGE)
        specs.append(SHOW_IMAGE)
        specs.append(SHARE_FILE)
        if self.depth == 0:
            specs.append(SEARCH_HISTORY)
        specs.extend(spec for spec, _ in self.extra_tools.values())
        return specs

    def _fit_history(self, history: list[tuple[str, str]], system: str) -> list[tuple[str, str]]:
        """Keep the most recent earlier turns that fit in ~30% of the context window."""
        if not self.context_limit:
            return history[-20:]
        budget = int(self.context_limit * 0.3) - len(system) // 3
        kept: list[tuple[str, str]] = []
        for u, a in reversed(history[-20:]):
            cost = (len(u) + len(a)) // 3
            if cost > budget:
                break
            budget -= cost
            kept.append((u, a))
        return list(reversed(kept))

    def _compact(self, provider, model, system, options, task, used, reason="threshold"):
        """Layer 2: structured, incremental summary + verbatim user words, identifiers, recent steps and
        fresh facts from disk. The checkpoint is saved as .nesicode/context/compact-N.md."""
        new_events = self.events[self._compact_mark:]
        budget = max(8_000, int((self.context_limit or 100_000) * 1.2))
        transcript = trajectory_text(new_events, max_chars=budget)
        n = self._compactions + 1
        tr_rel = ctxm.write_context_file(self.ws, f"transcript-{n}.md", trajectory_text(new_events, max_chars=10**9), self.archive)
        anchors = ctxm.extract_anchors(transcript)
        summary = ""
        try:
            summary, _ = provider.complete(
                model, ctxm.SUMMARY_SYSTEM, ctxm.summary_prompt(task, transcript, self._summary, anchors),
                ModelOptions(effort="low", role="summarizer", max_tokens=min(8_192, options.max_tokens or 8_192)),
            )
            summary = re.sub(r"<think>.*?</think>|<analysis>.*?</analysis>", "", summary, flags=re.S).strip()
        except LLMError:
            pass
        if not summary:  # deterministic fallback: never lose the thread
            summary = (self._summary or "") + "\n\n## Recent steps (summary unavailable)\n" + trajectory_text(new_events[-40:], 12_000)
        self._summary = summary
        self._compactions = n
        self._compact_mark = len(self.events)
        md_rel = ctxm.write_context_file(
            self.ws, f"compact-{n}.md",
            f"# Compactación {n}\n\n- Motivo: {reason}\n- Contexto: {used:,} de {self.context_limit:,} tokens\n"
            f"- Transcript completo de este tramo: {tr_rel}\n\n{summary}\n", self.archive,
        )
        tail = trajectory_text(new_events[-24:], max_chars=min(36_000, int((self.context_limit or 100_000) * 0.25)))
        parts = [
            ctxm.SUMMARY_PREFIX,
            f"## The user's request (verbatim)\n{task}",
            summary,
            ctxm.anchors_block(anchors),
            f"## Most recent steps (verbatim)\n{tail}" if tail else "",
            ctxm.recent_files(self.ws),
            ctxm.git_state(self.ws),
            ctxm.recovery_footer(self.archive),
            "Continue the task from the Next Step without asking for confirmation and without redoing completed actions.",
        ]
        conv = provider.conversation(model, system, self.tool_specs(), options)
        conv.add_user("\n\n".join(p for p in parts if p))
        self.emit({"type": "compacted", "used": used, "limit": self.context_limit, "summary": summary[:4000],
                   "path": md_rel, "n": n, "reason": reason})
        return conv

    def _other_chat_archive(self, args: dict) -> str | None:
        """Paths into another conversation's archive (.nesicode/chats/<other>) are refused."""
        for key in ("path", "command", "pattern"):
            v = str((args or {}).get(key, ""))
            for m in re.finditer(r"\.nesicode/chats/([^/\s'\"]+)", v):
                if f".nesicode/chats/{m.group(1)}" != self.archive:
                    return f".nesicode/chats/{m.group(1)}"
        return None

    def _marker(self, call_id: str, old: str):
        tool, rel = self._saved.get(call_id, ("tool", None))
        return ctxm.mask_marker(tool, rel, call_id in self._errors)(call_id, old)

    def _on_delta(self, kind: str, delta: str) -> None:
        if self.cancel.is_set():
            raise Cancelled()  # stop the model mid-reply (streaming providers)
        self.emit({"type": f"{kind}_delta", "text": delta}, persist=False)

    def _call_tool(self, call: ToolCall, usage: Usage) -> ToolResult:
        if call.parse_error:
            return ToolResult(call.id, f"Tool call rejected: {call.parse_error}", True)
        block, warnings = self.hooks.before(call.name, call.arguments)
        if block:
            return ToolResult(call.id, block, True)
        if self.approve is not None:
            allowed, reason = self.approve(call)
            if not allowed:
                return ToolResult(
                    call.id,
                    f"The user did not allow this {call.name} call ({reason}). Do not retry the same call; "
                    "choose another approach or ask the user what to do.",
                    True,
                )
        other = self._other_chat_archive(call.arguments)
        if other:
            return ToolResult(call.id, f"Error: {other} belongs to another conversation; each chat only sees its own history.", True)
        try:
            if call.name in BASIC_TOOLS:
                out = BASIC_TOOLS[call.name](self.ws, call.arguments)
            elif call.name == "load_skill":
                name = str(call.arguments.get("name", ""))
                sk = skills(self.harness).get(name)
                if not sk:
                    raise ToolError(f"unknown skill {name!r}; available: {', '.join(sorted(skills(self.harness)))}")
                out = sk["body"]
            elif call.name == "view_image" and self.vision_model:
                out = describe_image(
                    self.ws, str(call.arguments.get("path", "")), self.vision_model, self.user_keys,
                    str(call.arguments.get("question", "")),
                )
            elif call.name == "generate_image" and self.image_model:
                rel = generate_image(
                    self.ws, str(call.arguments.get("prompt", "")), str(call.arguments.get("path", "")),
                    self.image_model, self.user_keys,
                )
                self.emit({"type": "image", "path": rel, "caption": str(call.arguments.get("prompt", ""))[:300]})
                out = f"Image saved to `{rel}` and shown to the user."
            elif call.name == "show_image":
                rel = str(call.arguments.get("path", ""))
                p = self.ws.resolve(rel)
                if not p.is_file() or not image_mime(p.name):
                    raise ToolError("not an image file in the workspace (png, jpeg, gif, webp)")
                self.emit({"type": "image", "path": self.ws.rel(p), "caption": str(call.arguments.get("caption", ""))[:300]})
                out = f"Shown `{self.ws.rel(p)}` to the user."
            elif call.name in self.extra_tools:
                out = self.extra_tools[call.name][1](call.arguments, self)
            elif call.name == "search_history":
                out = ctxm.search_history(self.ws, call.arguments, self.archive)
            elif call.name == "share_file":
                p = self.ws.resolve(str(call.arguments.get("path", "")))
                if not p.is_file():
                    raise ToolError("file not found in the workspace")
                rel = self.ws.rel(p)
                self.emit({"type": "file", "path": rel, "size": p.stat().st_size,
                           "caption": str(call.arguments.get("caption", ""))[:300]})
                out = f"Shared `{rel}` with the user as a download."
            elif call.name == "delegate" and self.depth == 0:
                out = self._delegate(call.arguments, usage)
            else:
                raise ToolError(f"unknown tool {call.name!r}")
        except ToolError as exc:
            self._errors.add(call.id)
            return ToolResult(call.id, f"Error: {exc}", True)
        except Exception as exc:  # a tool bug must not kill the run
            self._errors.add(call.id)
            return ToolResult(call.id, f"Error: {type(exc).__name__}: {exc}", True)
        if warnings:
            out = "\n".join(f"[hook] {w}" for w in warnings) + "\n" + out
        if len(out) > ctxm.SAVE_OVER and self.depth == 0:
            # Layer 0: the full output lives on disk; the context keeps head + tail + path.
            try:
                rel = ctxm.save_output(self.ws, call.id, call.name, out, self.archive)
                self._saved[call.id] = (call.name, rel)
                out = ctxm.inline_output(out, rel)
            except Exception:
                pass
        return ToolResult(call.id, self.hooks.after(call.name, out))

    def _delegate(self, args: dict, usage: Usage) -> str:
        name = str(args.get("agent", ""))
        task = str(args.get("task", "")).strip()
        ag = sub_agents(self.harness).get(name)
        if not ag:
            raise ToolError(f"unknown subagent {name!r}")
        if not task:
            raise ToolError("`task` is required")
        sub = Agent(
            model=ag["meta"].get("model") or self.model_spec,
            harness=self.harness,
            workspace=self.ws,
            emit=self._emit,
            user_keys=self.user_keys,
            max_steps=25,
            effort=ag["meta"].get("effort", "medium"),
            role="subagent",
            cancel=self.cancel,
            approve=self.approve,
            depth=self.depth + 1,
            system_base=ag["body"],
            label=name,
            vision_model=self.vision_model,
            image_model=self.image_model,
            extra_tools=self.extra_tools,
        )
        res = sub.run(task)
        usage.add(res.usage)
        self.events.extend(sub.events)
        if res.status == "error":
            raise ToolError(f"subagent failed: {res.error}")
        return res.final_text or "(subagent returned nothing)"

    def run(self, task: str, history: list[tuple[str, str]] | None = None) -> AgentResult:
        usage = Usage()
        steps = 0
        final_text = ""
        try:
            provider, model = get_provider(self.model_spec, self.user_keys)
            if self.depth == 0:
                self.ws.install_harness(self.harness)
            system = build_system(self.harness, self.ws, base=self.system_base, extra_tools=self.extra_tools)
            options = ModelOptions(effort=self.effort, role=self.role, cancel=self.cancel)
            out_budget = min(options.max_tokens, self.output_limit or options.max_tokens)
            if self.context_limit:
                out_budget = min(out_budget, ctxm.OUTPUT_PER_TURN if self.context_limit <= 400_000 else out_budget)
            options.max_tokens = out_budget
            used = 0
            conv = provider.conversation(model, system, self.tool_specs(), options)
            for user_text, assistant_text in self._fit_history(history or [], system):
                conv.add_user(user_text)
                conv.add_assistant_text(assistant_text)
            conv.add_user(task)
            compactions = 0  # kept for the learned-limit path
            self.emit({"type": "start", "model": self.model_spec, "task": task[:4000]})
            nudged = False
            while True:
                if self.cancel.is_set():
                    self.emit({"type": "cancelled"})
                    return AgentResult(final_text, "cancelled", steps, usage, self.events)
                if self.context_limit:
                    # Reserve output for this turn; if there is no room left, compact first.
                    room = self.context_limit - used - ctxm.SAFETY_TOKENS
                    if used and room < ctxm.MIN_OUTPUT and self._compactions < ctxm.MAX_COMPACTIONS:
                        conv = self._compact(provider, model, system, options, task, used, reason="no room for output")
                        used = 0
                        room = self.context_limit - ctxm.SAFETY_TOKENS
                    options.max_tokens = max(1_024, min(out_budget, room))
                try:
                    turn = conv.step(on_event=self._on_delta)
                except LLMError as exc:
                    learned = parse_limit(str(exc))
                    if not learned or compactions >= ctxm.MAX_COMPACTIONS:
                        raise
                    # The model told us its real limit: remember it, compact and try again.
                    self.context_limit = learned
                    if self.on_limit:
                        self.on_limit(learned)
                    self.emit({"type": "context_learned", "limit": learned})
                    compactions += 1
                    conv = self._compact(provider, model, system, options, task, learned, reason="context exceeded")
                    used = 0
                    continue
                if turn.stop_reason == "max_tokens" and any(c.parse_error for c in turn.tool_calls) \
                        and self.context_limit and self._compactions < ctxm.MAX_COMPACTIONS:
                    # Emergency: the reply was cut (vLLM trims output near the limit). Drop the turn, compact, retry.
                    usage.add(turn.usage)
                    conv = self._compact(provider, model, system, options, task,
                                         turn.usage.input_tokens + turn.usage.output_tokens, reason="truncated reply")
                    used = 0
                    continue
                steps += 1
                usage.add(turn.usage)
                if turn.thinking:
                    self.emit({"type": "thinking", "text": turn.thinking[:8000]})
                if turn.text:
                    self.emit({"type": "assistant", "text": turn.text})
                    final_text = turn.text
                if turn.stop_reason == "refusal":
                    self.emit({"type": "error", "text": "El modelo rechazó la petición (refusal)."})
                    return AgentResult(final_text, "refusal", steps, usage, self.events)
                if not turn.tool_calls or nudged:
                    status = "max_steps" if nudged else "done"
                    ctx = {"used": turn.usage.input_tokens + turn.usage.output_tokens, "limit": self.context_limit}
                    self.emit({"type": "done", "status": status, "steps": steps, "usage": usage.to_dict(), "context": ctx})
                    return AgentResult(final_text, status, steps, usage, self.events)
                results = []
                for call in turn.tool_calls:
                    if self.cancel.is_set():
                        raise Cancelled()
                    self.emit({"type": "tool_call", "id": call.id, "name": call.name, "args": call.arguments})
                    res = self._call_tool(call, usage)
                    self.emit({
                        "type": "tool_result", "id": call.id, "name": call.name,
                        "output": res.content[:12000], "is_error": res.is_error,
                    })
                    results.append(res)
                conv.add_tool_results(results)
                used = turn.usage.input_tokens + turn.usage.output_tokens + sum(len(r.content) for r in results) // 3
                if self.context_limit:
                    lim = self.context_limit
                    # Layer 1: mask old tool outputs (no LLM), in batches, with hysteresis.
                    if used > ctxm.MASK_AT * lim and used - self._masked_at > ctxm.min_freed_tokens(lim):
                        freed = conv.mask_tool_results(ctxm.KEEP_RESULTS, self._marker)
                        if freed // 3 >= ctxm.min_freed_tokens(lim) // 2:
                            used -= freed // 3
                            self.emit({"type": "masked", "freed": freed // 3, "used": used, "limit": lim})
                        self._masked_at = used
                    # Layer 2: structured summary.
                    if used > ctxm.SUMMARY_AT * lim and self._compactions < ctxm.MAX_COMPACTIONS:
                        conv = self._compact(provider, model, system, options, task, used)
                        used = 0
                        self._masked_at = 0
                    self.emit({"type": "context", "used": used, "limit": lim}, persist=False)
                if steps >= self.max_steps:
                    conv.add_user(FINAL_NUDGE)
                    nudged = True
        except Cancelled:
            self.emit({"type": "cancelled"})
            return AgentResult(final_text, "cancelled", steps, usage, self.events)
        except LLMError as exc:
            if self.cancel.is_set():  # the provider wrapped our stop into its own error
                self.emit({"type": "cancelled"})
                return AgentResult(final_text, "cancelled", steps, usage, self.events)
            self.emit({"type": "error", "text": str(exc)})
            return AgentResult(final_text, "error", steps, usage, self.events, error=str(exc))
        except Exception as exc:
            if self.cancel.is_set():
                self.emit({"type": "cancelled"})
                return AgentResult(final_text, "cancelled", steps, usage, self.events)
            msg = f"{type(exc).__name__}: {exc}"
            self.emit({"type": "error", "text": msg})
            return AgentResult(final_text, "error", steps, usage, self.events, error=msg)


def trajectory_text(events: list[dict], max_chars: int = 30_000) -> str:
    """Readable trajectory in the [ACTION]/[OBSERVATION]/[ERROR] grammar."""
    lines: list[str] = []
    for e in events:
        who = f"({e['agent']}) " if e.get("agent") else ""
        t = e.get("type")
        if t == "assistant":
            lines.append(f"[ASSISTANT] {who}{e['text']}")
        elif t == "tool_call":
            args = json.dumps(e.get("args", {}), ensure_ascii=False)
            if len(args) > 1500:
                args = args[:1500] + "...}"
            lines.append(f"[ACTION] {who}{e['name']} {args}")
        elif t == "tool_result":
            out = e.get("output", "")
            if len(out) > 2500:
                out = out[:1200] + "\n...\n" + out[-1200:]
            tag = "[ERROR]" if e.get("is_error") else "[OBSERVATION]"
            lines.append(f"{tag} {who}{out}")
        elif t == "error":
            lines.append(f"[ERROR] {who}{e['text']}")
        elif t == "done":
            lines.append(f"[END] status={e['status']} steps={e['steps']}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        head = max_chars // 3
        text = text[:head] + "\n\n... (middle of trajectory omitted) ...\n\n" + text[-(max_chars - head):]
    return text
