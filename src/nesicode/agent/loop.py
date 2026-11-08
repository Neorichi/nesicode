"""The nesicode agent loop: harness -> system prompt + tools, then model <-> tools."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from ..harness import Harness, hooks as harness_hooks, skills, sub_agents
from ..harness.base import upgrade_hooks
from ..llm import (DEFAULT_LIMITS, LLMError, ModelOptions, ToolCall, ToolResult, Usage, get_provider, known_limits,
                   parse_limit, request_scope)
from .hooks import Hooks
from . import context as ctxm
from .tools import (BASIC_TOOLS, GENERATE_IMAGE, SCHEMAS, SEARCH_HISTORY, SHARE_FILE, SHOW_IMAGE, VIEW_IMAGE, delegate_tool,
                    read_file_spec, skill_tool)
from .vision import describe_image, generate_image, image_mime
from .workspace import HARNESS_DIR, Cancelled, ToolError, Workspace

Emit = Callable[[dict], None]

FINAL_NUDGE = (
    "You have used your whole step budget. Do not call any more tools. "
    "Reply now with your final summary: what you changed, what you verified, and what is left."
)
CONTEXT_FULL_NUDGE = (
    "The context window is full and cannot be compacted any further. Do not call any more tools. "
    "Reply now with your final summary: what you changed, what you verified, and what is left (the user can say "
    "'continue' to pick the work up again from the archived history)."
)
CONTINUE_NOTE = (
    "[harness] Your previous reply was cut off by the output limit ({cap:,} tokens per reply). Continue exactly where it "
    "stopped — no recap, no repetition. If you were about to call a tool, call it now and keep its arguments small."
)
EMPTY_CUT_NOTE = (
    "[harness] Your previous reply hit the output limit ({cap:,} tokens) before any visible answer (the reasoning used it "
    "up). Keep the reasoning short now and write the answer, or make the next tool call."
)
READ_STREAK = 20  # tool calls in a row without writing anything before the reminder (once per streak; again after a
# masking or a compaction while the streak goes on, at least READ_STREAK // 2 calls after the previous one)
READ_STREAK_NOTE = (
    "\n\n[harness] You have made {n} tool calls in a row without writing or editing any file. Write what you have now "
    "and keep results on disk as you go (findings, exact values, partial results — in the deliverable or a notes file): "
    "old tool outputs and your private reasoning may be cleared from the context, and the user can stop the turn at any "
    "time.{rules}"
)
REQUEST_QUOTE = 1_200  # chars of the user's own words the reminder quotes
MASK_NOTE = ("[harness] {what} cleared to free context (full outputs on disk; paths in the markers). Record any conclusion "
             "you have not written down yet (e.g. in the deliverable or a notes file) before re-reading.")
MASK_NOTE_REASONING = ("[harness] Your reasoning from earlier steps was cleared to free context. Record any conclusion you "
                       "have not written down yet (e.g. in the deliverable or a notes file).")
THINKING_CAP = 32_000  # chars of one reply's reasoning kept in the event log (its END: the conclusions sit there)
TRAIL_MARK = "[PREVIOUS TURN — STEPS]"  # heads the step trail of the previous turn in the history (see _trim_answer)
WRITE_TOOLS = {"write_file", "edit_file", "generate_image", "share_file"}
_WRITES_RX = re.compile(r"(?<![0-9&>])>>?\s*[\w./~$\"'-]|\btee\b|\bsed\s+-i|\b(?:mv|cp|rm|mkdir|touch|patch)\s|"
                        r"\bgit\s+(?:commit|apply|am|merge|rebase|checkout|restore)\b")


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
        "Your private reasoning may be dropped between steps, and old tool outputs get cleared to save context: put "
        "facts you will need later (exact values, findings, decisions) in your visible replies, the deliverable or a "
        "notes file as soon as you have them.",
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


def _trim_answer(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    i = text.find(TRAIL_MARK)
    if i >= 0:  # the previous turn's step trail ends where that turn stopped: its OLDEST steps go first
        head = text[: i + len(TRAIL_MARK)]
        gap = "\n[... older steps of this turn omitted here; search_history has them ...]\n"
        room = n - len(head) - len(gap)
        if room >= 600:
            tail = text[len(text) - room:]
            if 0 <= (nl := tail.find("\n")) < len(tail) // 2:
                tail = tail[nl + 1:]
            return head + gap + tail
    note = "\n[... middle of this earlier turn omitted; search_history has it ...]\n"
    keep = max(0, n - len(note))
    head = keep * 2 // 5
    return text[:head] + note + text[len(text) - (keep - head):]


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
        system_note: str = "",
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
        self._fail_streak: dict[str, tuple[int, str]] = {}
        # Called before a tool runs; returns (allowed, reason). None = everything is allowed (auto mode).
        self.approve = approve
        self.depth = depth
        self.system_base = system_base
        self.system_note = system_note
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
        self._saved: dict[str, tuple[str, str, str, str]] = {}  # call_id -> (tool, saved path, what it is, how to re-read)
        self._errors: set[str] = set()
        self._summary: str | None = None  # the last good LLM summary of this run
        self._compactions = 0
        self._compact_mark = 0  # events before this index are covered by self._summary
        self._masked_at = 0
        self._display = 0
        self.events: list[dict] = []
        # what the conversation holds, for the token estimate (see _est)
        self._history: list[tuple[str, str]] = []
        self._task = ""
        self._task_full = ""
        self._tag = ""
        self._tag_ok = False
        self._fixed_chars = 0
        self._conv_chars = 0
        self._anchor: tuple[int, int] | None = None  # (real prompt tokens of the last request, conversation chars then)
        self._cpt = ctxm.CHARS_PER_TOKEN  # chars per token seen in the provider's real counts (≤ the default)
        self._cpt_min = 3.0  # floor for that ratio; lowered when the provider proves the text is denser (_denser)
        # the densest prompt measured in this run, without that floor (JSON/HTTP traffic and minified code: 2-3
        # chars/token; base64 and CJK text: 1-1.5): sizes the summarizer's request, whose transcript is what the model
        # read. Its own floors only catch nonsense counts (1.0 from a whole prompt, 0.5 from the summarizer's own text)
        self._cpt_low = ctxm.CHARS_PER_TOKEN
        self._request_rel = ""  # where the user's full message was saved when it did not fit (written once per run)
        self._pending: list[tuple[ToolCall, ToolResult]] = []  # the tool round the model has not answered yet
        self._round_sizes: list[int] = []
        self._touched: dict[str, str] = {}  # files read/edited in this run (insertion order = recency)
        self._skills_loaded: list[str] = []
        self._steps = 0
        self._steps_at_compact = -1
        self._last_compact_after = 0
        self._no_progress = 0  # compactions in a row that did not bring the real context under the trigger
        self._judge = False
        self._compact_off = False
        self._limit_errors = 0
        self._nudged = False
        self._cut_streak = 0
        self._continuations = 0
        self._parts: list[str] = []
        self._ro_streak = 0
        self._ro_reminded = False
        self._ro_at = 0  # the streak when the last reminder was given
        self._strip_noted = False  # the model was told its older reasoning is cleared (small windows)
        self._live: dict[str, list[str]] = {}  # deltas of the reply being streamed (kept if Stop or an error cuts it)
        self._usage = Usage()
        try:  # (a project created with an earlier, quadratic spelling of the base rm rule runs the current one)
            self.hooks = Hooks(upgrade_hooks(harness_hooks(harness)))
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
        # read_file states the per-call budget this window gives it (see _inline_limit)
        specs = [read_file_spec(self._inline_limit("read_file")) if n == "read_file" else SCHEMAS[n] for n in BASIC_TOOLS]
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
        # sub-agents too: they work in the conversation's own archive, and their checkpoints point to search_history
        specs.append(SEARCH_HISTORY)
        specs.extend(spec for spec, _ in self.extra_tools.values())
        return specs

    # ------------------------------------------------------------------ history
    def _fit_history(self, history: list[tuple[str, str]], system: str = "", budget: int | None = None) -> list[tuple[str, str]]:
        """Earlier turns within `budget` tokens (default ~30% of the window minus the system prompt).

        Nothing is dropped silently: the conversation's FIRST user message always stays (bounded — it usually holds
        the main request and its rules), a note says how many turns are left out (search_history has them), then the
        most recent turns that fit. User messages are kept verbatim (a single huge one: head + tail) and take at most
        ~60% of the budget; assistant answers are trimmed first, oldest first; whole turns go last."""
        if not history:
            return []
        first, rest = history[0], list(history[1:])
        omitted = max(0, len(rest) - 19)
        rest = rest[-19:]
        if budget is None and self.context_limit:
            budget = int(self.context_limit * 0.3) - len(system) // 4
        if budget is None:
            return self._note_omitted([first] + rest, omitted)
        chars = max(0, budget) * 4
        ucap = max(1_500, min(12_000, chars // 4))
        turns = [[ctxm.clip(u, ucap, "message"), a or ""] for u, a in [first] + rest]
        while len(turns) > 1 and sum(len(u) for u, _ in turns) > chars * 0.6:
            turns.pop(1)
            omitted += 1
        if len(turns[0][0]) > max(1_500, chars):  # even alone the first message is too big: keep its head and tail
            turns[0][0] = ctxm.clip(turns[0][0], max(1_500, chars), "message")
        excess = sum(len(u) + len(a) for u, a in turns) - chars
        for t in turns:  # answers, oldest first, only as much as needed
            if excess <= 0:
                break
            new = _trim_answer(t[1], max(400, len(t[1]) - excess))
            excess -= len(t[1]) - len(new)
            t[1] = new
        while excess > 0 and len(turns) > 1:  # still too big: whole turns, oldest first (never the first one)
            u, a = turns.pop(1)
            omitted += 1
            excess -= len(u) + len(a)
        return self._note_omitted([(u, a) for u, a in turns], omitted)

    @staticmethod
    def _note_omitted(turns: list[tuple[str, str]], omitted: int) -> list[tuple[str, str]]:
        if omitted and turns:
            u, a = turns[0]
            turns[0] = (u, (a or "") + f"\n\n[{omitted} earlier turn(s) between this one and the next shown are omitted "
                                       "here — search_history finds them]")
        return turns

    def _history_note(self) -> str:
        hist = f"{self.archive}/context/history.md"
        try:
            archived = self.ws.resolve(hist).is_file()
        except Exception:
            archived = False
        return ("# Earlier turns of this conversation\n"
                "The user/assistant messages before the current request are this conversation's earlier turns (oldest "
                "first), including turns the user stopped; rules and preferences the user stated there still apply. "
                + (f"Their full steps and tool outputs are archived in `{hist}` and `{self.archive}/outputs/`. " if archived else "")
                + "If something you need is not shown, call `search_history` (plain keywords; typos are tolerated) before "
                "exploring files. Never claim to remember a finding that is not shown or in a search result.")

    # ------------------------------------------------------------------ token accounting
    def _est(self, ratio: float = ctxm.CHARS_PER_TOKEN) -> int:
        """Tokens the next request carries: the provider's real count of the last request plus an estimate of what
        was appended since — or, with no usable count (right after a rebuild, or a provider that reports none), system +
        tool schemas + every message at `ratio` chars/token."""
        if self._anchor:
            tokens, chars = self._anchor
            delta = self._conv_chars - chars
            return max(0, int(tokens + (delta / ratio if delta > 0 else delta / ctxm.CHARS_PER_TOKEN)))
        return int((self._fixed_chars + self._conv_chars) / min(ratio, self._cpt))

    def _calibrate(self, input_tokens: int) -> None:
        """The provider counted `input_tokens` for what we sent: anchor the next estimates on it, and remember how dense
        the prompt was (code is ~3.3 chars/token) so the fit check right after a rebuild does not undercount."""
        if input_tokens > 500:
            seen = (self._fixed_chars + self._conv_chars) / input_tokens
            self._cpt = max(self._cpt_min, min(ctxm.CHARS_PER_TOKEN, seen))
            self._cpt_low = max(1.0, min(self._cpt_low, seen))
        self._anchor = (input_tokens, self._conv_chars)

    def _denser(self, error: str) -> None:
        """The provider rejected a request that our estimate said would fit, so this text is denser than assumed: take
        the prompt size the error reports (else assume 15% denser) for every estimate from now on — otherwise the
        re-fit after a context error repeats the same oversized request until the run fails."""
        chars = self._fixed_chars + self._conv_chars
        sent = ctxm.prompt_tokens(error)
        cpt = chars / sent * 0.95 if sent and chars else self._cpt * 0.85
        self._cpt = max(1.5, min(self._cpt, cpt))
        self._cpt_min = min(self._cpt_min, self._cpt)  # only ever lowered: later real counts may still go below it

    def _room(self) -> int:
        lim = self.context_limit
        return lim - self._est(min(ctxm.CONSERVATIVE_CPT, self._cpt)) - ctxm.safety_tokens(lim)

    def _fits(self) -> bool:
        return self._room() >= ctxm.min_output(self.context_limit)

    def _add_user(self, conv, text: str) -> None:
        conv.add_user(text)
        self._conv_chars += len(text) + 16

    def _add_note(self, conv, text: str) -> None:
        """A harness line for the model inside the tool loop: with the newest tool results when the provider can (a
        new user turn would make interleaved-thinking templates drop the reasoning replayed so far), else a user turn."""
        add = getattr(conv, "add_note", None)
        (add if callable(add) else conv.add_user)(text)
        self._conv_chars += len(text) + 16

    def _add_assistant(self, conv, text: str) -> None:
        conv.add_assistant_text(text)
        self._conv_chars += len(text or "(no answer)") + 16

    def _add_results(self, conv, results: list[ToolResult]) -> None:
        conv.add_tool_results(results)
        self._conv_chars += sum(len(r.content or "") + 40 for r in results)

    def _ctx_event(self, reason: str = "") -> None:
        if self.context_limit and not self.depth:
            self.emit({"type": "context", "used": self._display, "limit": self.context_limit,
                       **({"reason": reason} if reason else {})}, persist=False)

    def _notify_retry(self, info: dict) -> None:
        """Providers call this before retrying a request (slow or stalled endpoint): the user sees it."""
        self._live = {}  # the retry streams the reply again from its start
        try:
            self.emit({"type": "llm_retry", **{k: v for k, v in dict(info or {}).items() if k != "type"}})
        except Exception:
            pass

    def _save_partial(self) -> None:
        """A reply cut short by Stop or an error: what had been streamed of it is persisted, marked interrupted (the UI
        shows it, and the next turn's history and digest have it) — before the cancelled/error event."""
        live, self._live = self._live, {}
        thinking, text = "".join(live.get("thinking", ())), "".join(live.get("text", ()))
        try:
            if thinking.strip():
                self.emit({"type": "thinking", "text": _keep_end(thinking, THINKING_CAP), "interrupted": True})
            if text.strip():
                self.emit({"type": "assistant", "text": text, "interrupted": True})
        except Exception:
            pass

    def _end_background(self) -> None:
        """The top-level run is over: what its commands left running in the background goes too."""
        kill = getattr(self.ws, "kill_background", None)
        if callable(kill):
            try:
                kill()
            except Exception:
                pass

    def _streak_note(self) -> str:
        """The read-only streak reminder, quoting the user's own words (the conversation's first message, where the
        standing rules usually are, and the current one) within REQUEST_QUOTE chars."""
        first = (self._history[0][0] if self._history else "").strip()
        cur = (self._task_full or self._task).strip()
        quotes = [(label, t) for label, t in (("first message of this conversation", first), ("current message", cur)) if t]
        if len(quotes) == 2 and quotes[0][1] == quotes[1][1]:
            quotes = quotes[1:]
        caps = [len(t) for _, t in quotes]
        if sum(caps) > REQUEST_QUOTE:
            if len(caps) == 1:
                caps = [REQUEST_QUOTE]
            else:  # the shorter one whole (up to half), the other gets the rest
                s = min(min(caps), REQUEST_QUOTE // 2)
                caps = [s, REQUEST_QUOTE - s] if caps[0] <= caps[1] else [REQUEST_QUOTE - s, s]
        rows = [f"- {label}: «{ctxm.clip(t, n, 'message')}»" for (label, t), n in zip(quotes, caps)]
        rules = ("\nThe user's request, verbatim (excerpt) — follow its rules:\n" + "\n".join(rows)) if rows else ""
        return READ_STREAK_NOTE.format(n=self._ro_streak, rules=rules)

    # ------------------------------------------------------------------ conversations
    def _open(self, provider, model, system, options, task, history_budget=None):
        conv = provider.conversation(model, system, self.tool_specs(), options)
        self._conv_chars, self._anchor, self._round_sizes = 0, None, []
        self._strip_noted = False  # (a new conversation: no reasoning cleared from it yet)
        _dropped(conv)  # (reasoning a server refused earlier was stripped from another conversation)
        for u, a in self._fit_history(self._history, system, history_budget):
            self._add_user(conv, u)
            self._add_assistant(conv, a)
        self._add_user(conv, task)
        return conv

    def _first_conversation(self, provider, model, system, options, task):
        """The first request, checked BEFORE it is sent: an oversized history or message is shrunk now instead of
        failing at the provider. Returns (conversation, the message as sent)."""
        lim = self.context_limit
        conv = self._open(provider, model, system, options, task)
        if not lim or self._fits():
            return conv, task
        for share in (0.15, 0.06, 0.0):
            conv = self._open(provider, model, system, options, task, int(lim * share))
            if self._fits():
                return conv, task
        if self._fixed_chars / ctxm.CONSERVATIVE_CPT < lim * 0.5 and len(task) > lim * 4 * 0.3:
            # the message itself is too big: the full text goes to the archive, the context keeps head + tail
            short = self._shrink_task(task, int(lim * 4 * 0.3))
            return self._open(provider, model, system, options, short, int(lim * 0.06)), short
        return conv, task  # nothing more to cut (system + tools fill the window): the provider decides

    def _shrink_task(self, task: str, budget: int) -> str:
        # Always cut from (and archive) the ORIGINAL message: a re-fit after a smaller limit is learned must not
        # overwrite the saved copy with an already-shortened one.
        full = self._task_full or task
        if not self._request_rel:
            try:
                self._request_rel = ctxm.write_context_file(self.ws, f"{self._archive_tag()}-request.md", full, self.archive)
            except Exception:
                return ctxm.clip(full, budget, "message")
        note = (f"\n[... the middle of this message ({len(full):,} chars in total) does not fit in the context window; "
                f"the FULL message is saved in {self._request_rel} — read the missing part with read_file (offset/limit) ...]\n")
        keep = max(0, budget - len(note))
        head = keep * 2 // 3
        return full[:head] + note + full[len(full) - (keep - head):]

    def _archive_tag(self) -> str:
        """Per-run prefix of the archive files, so later runs never overwrite earlier ones."""
        if not self._tag:
            self._tag = time.strftime("%Y%m%d-%H%M%S") + (f"-{self.label or 'sub'}" if self.depth else "")
        if not self._tag_ok:
            base, i = self._tag, 1
            while any(self._archived(f"{self._tag}-{kind}") for kind in ("transcript-1.md", "compact-1.md", "request.md")):
                i += 1
                self._tag = f"{base}-{i}"
            self._tag_ok = True
        return self._tag

    def _archived(self, name: str) -> bool:
        try:
            return self.ws.resolve(f"{self.archive}/context/{name}").exists()
        except Exception:
            return False

    # ------------------------------------------------------------------ layer 1: masking
    def _marker(self, call_id: str, old: str):
        info = self._saved.get(call_id)
        if not info:  # only outputs saved to disk are ever masked: the marker must say where they are
            return None
        tool, rel, desc, reread = info
        return ctxm.mask_marker(tool, rel, call_id in self._errors, desc, reread)(call_id, old)

    def _mask(self, conv, keep_last: int, keep_chars: int, min_tokens: int) -> int:
        """Mask old tool outputs (and the reasoning of the replies before them) if that frees at least `min_tokens` (a
        dry run first: masking breaks the prompt cache, so it only happens when it is worth it). The model is told in
        one line, and the progress reminder may come again. Returns the chars freed."""
        would = conv.mask_tool_results(keep_last, self._marker, keep_chars=keep_chars, apply=False)
        if not would or would // 4 < min_tokens:
            return 0
        cleared = [0, 0]  # tool outputs masked, chars they freed (the rest of `freed` is reasoning)

        def marker(call_id: str, old: str):
            new = self._marker(call_id, old)
            if new is not None and len(new) < len(old):
                cleared[0] += 1
                cleared[1] += len(old) - len(new)
            return new
        freed = conv.mask_tool_results(keep_last, marker, keep_chars=keep_chars, apply=True)
        self._conv_chars -= freed
        if freed:
            n = cleared[0]
            what = (f"{n} earlier tool output{'s' if n != 1 else ''}"
                    + (" and your reasoning from those steps were" if freed > cleared[1] else (" were" if n != 1 else " was")))
            self._add_note(conv, MASK_NOTE.format(what=what) if n else MASK_NOTE_REASONING)
            self._ro_reminded = False
        return freed

    # ------------------------------------------------------------------ layer 2: compaction
    def _can_compact(self) -> bool:
        """A compaction helps only if something happened since the last one (and it is not known to be useless)."""
        return not self._compact_off and self._compactions < ctxm.MAX_COMPACTIONS and self._steps > self._steps_at_compact

    def _summary_delta(self, kind: str, delta: str) -> None:
        if self.cancel.is_set():
            raise Cancelled()  # Stop works while the summary is written (streaming providers)

    def _compact(self, provider, model, system, options, task, used, reason="threshold"):
        """Layer 2: the conversation is rebuilt as system + earlier turns + a checkpoint (structured, incremental
        summary; the user's latest message verbatim; identifiers; the unanswered tool round verbatim; recent steps;
        the files and skills of this run fresh from disk). Saved as <archive>/context/<run>-compact-N.md."""
        if self.cancel.is_set():
            raise Cancelled()
        lim = self.context_limit
        n = self._compactions + 1
        new_events = self.events[self._compact_mark:]
        work = [e for e in new_events if not e.get("depth") and e.get("type") in ("assistant", "tool_call", "tool_result")]
        tag = self._archive_tag()
        tr_rel = ""
        if work:
            tr_rel = self._archive_file(
                f"{tag}-transcript-{n}.md",
                f"# Transcript {n} ({tag})\n\n## The user's request\n{self._task_full}\n\n## Steps\n"
                + ctxm.transcript(new_events, budget=None, subagents=True, args_cap=None, thinking=True) + "\n")
        anchors = self._anchors(new_events)
        fallback = False
        if work:
            summary = self._summarize(provider, model, task, new_events, anchors)
            if summary:
                self._summary = summary
                self._compact_mark = len(self.events)
            else:  # deterministic fallback; the events stay unsummarized for the next attempt
                summary, fallback = self._fallback_summary(new_events), True
        else:
            summary = self._summary or ""
        self._compactions = n
        md_rel = self._archive_file(
            f"{tag}-compact-{n}.md",
            f"# Compactación {n}\n\n- Motivo: {reason}\n- Contexto: {used:,} de {lim:,} tokens\n"
            + (f"- Transcript completo de este tramo: {tr_rel}\n" if tr_rel else "")
            + (f"\n{summary}\n" if summary else "\n(Aún no había trabajo que resumir: la conversación se reconstruyó para "
                                                "que quepa en la ventana de contexto.)\n"))
        if work or summary:
            conv = self._rebuild(provider, model, system, options, task, summary, anchors, new_events, tr_rel)
        else:  # nothing done yet in this run: the start of the conversation, re-fitted to the window (from the original)
            conv, task = self._first_conversation(provider, model, system, options, self._task_full or task)
            self._task = task
            if self._nudged:
                self._add_user(conv, FINAL_NUDGE)
        after = self._est()
        self._last_compact_after = after
        self._steps_at_compact = self._steps
        self._judge = True  # the provider's next real count says whether this compaction got under the trigger
        self._masked_at = 0
        self._ro_reminded = False  # the progress reminder may come again if the read-only streak goes on
        self.emit({"type": "compacted", "used": used, "after": after, "limit": lim,
                   "summary": ctxm.clip_summary(summary, 20_000),  # (the next message resumes from its Next Step)
                   "path": md_rel, "transcript": tr_rel, "n": n, "reason": reason, **({"fallback": True} if fallback else {})})
        self._display = after
        self._ctx_event("compacted")
        return conv

    def _archive_file(self, name: str, text: str) -> str:
        """Write <archive>/context/<name>; "" if it cannot be written (e.g. a link planted in the archive) — the
        compaction goes on without that copy."""
        try:
            return ctxm.write_context_file(self.ws, name, text, self.archive)
        except Exception:
            return ""

    def _anchors(self, events: list[dict]) -> dict:
        own = [e for e in events if not e.get("depth")]
        text = "\n".join(str(e.get("text") or e.get("output") or "") for e in own if e.get("type") in ("assistant", "tool_result"))
        a = ctxm.extract_anchors(text, exists=self._exists, errors=False)
        paths = []
        for e in own:
            if e.get("type") == "tool_call":
                p = str((e.get("args") or {}).get("path") or "")
                if p and p not in (".", "./") and p not in paths:
                    paths.append(p)
        files = [f for f in dict.fromkeys(paths + a.get("files", []))
                 if not re.sub(r"^(\./)+", "", f).startswith((".nesicode", ".attachments"))][-40:]  # not our own archive
        a.pop("files", None)
        if files:
            a["files"] = files
        if errs := ctxm.error_lines(own):
            a["errors"] = errs
        return a

    def _exists(self, rel: str) -> bool:
        try:
            return self.ws.resolve(rel).exists()
        except Exception:
            return False

    def _summarize(self, provider, model, task, new_events, anchors) -> str:
        """The summarizer sees what the model saw (outputs as they were inline; the oldest elided first if they do not
        fit, each elision said). Its request is sized with the densest chars/token measured in this run; one the window
        still rejects is sized again from the prompt size the provider reports and sent once more. Stop works during
        the call; a summary cut short, empty or stuck in reasoning is retried once with a smaller transcript; "" means
        use the fallback."""
        saved = {cid: info[1] for cid, info in self._saved.items()}
        thinking = getattr(provider, "name", "") != "anthropic"  # reasoning notes help; Claude's are not re-sent
        shrink, resized = 1, False
        for _ in range(3):
            lim = self.context_limit
            # generous (reasoning models spend it thinking) but at most a fifth of the window, so the summarizer's input
            # can hold what the model had when the 67% trigger fired
            out_tokens = max(1_024, min(self.output_limit or 32_768, 32_768, max(12_000, int(lim * 0.15)), int(lim * 0.2)))
            cpt = min(ctxm.CONSERVATIVE_CPT, self._cpt, self._cpt_low * 0.95)
            room = int((lim - out_tokens - ctxm.safety_tokens(lim)) * cpt) - len(ctxm.SUMMARY_SYSTEM)
            target = ctxm.summary_target(lim) // shrink
            earlier = ctxm.earlier_turns(self._history, min(20_000, int(lim * 4 * 0.06)))
            base = ctxm.summary_prompt(task, "", self._summary, anchors, earlier=earlier,
                                       date=time.strftime("%Y-%m-%d"), target=target)
            budget = max(2_000, (room - len(base)) // shrink)
            prompt = ctxm.summary_prompt(task, ctxm.transcript(new_events, budget=budget, saved=saved, thinking=thinking),
                                         self._summary, anchors, earlier=earlier, date=time.strftime("%Y-%m-%d"),
                                         target=target)
            opts = ModelOptions(effort="low", role="summarizer", max_tokens=out_tokens, cancel=self.cancel)
            setattr(opts, "max_attempts", 2)  # a fallback exists: retries only lengthen the wait
            setattr(opts, "notify", self._notify_retry)
            try:
                conv = provider.conversation(model, ctxm.SUMMARY_SYSTEM, [], opts)
                conv.add_user(prompt)
                turn = conv.step(on_event=self._summary_delta)
            except LLMError as exc:
                if self.cancel.is_set():
                    raise Cancelled() from exc
                self.emit({"type": "compaction_error", "text": str(exc)[:300]}, persist=False)
                if not resized and self._summary_too_big(str(exc), len(ctxm.SUMMARY_SYSTEM) + len(prompt)):
                    resized = True  # the same stretch again, sized from what the provider measured
                    continue
                return ""
            if self.cancel.is_set():
                raise Cancelled()
            self._usage.add(turn.usage)
            text, problem = ctxm.clean_summary(turn.text, turn.stop_reason, turn.thinking)
            if not problem:
                return text
            self.emit({"type": "compaction_error", "text": f"summary rejected: {problem}"}, persist=False)
            if shrink > 1:
                return ""
            shrink = 2
        return ""

    def _summary_too_big(self, error: str, sent_chars: int) -> bool:
        """Was the summarizer's request rejected as too big for the window? Then learn what its text really costs: the
        prompt size the error reports (else 15% denser than assumed) and the window, when it names a smaller one — the
        same model answers the executor, so that limit is the run's too. False for any other error."""
        learned, sent = parse_limit(error), ctxm.prompt_tokens(error)
        if not learned and not sent:
            return False
        cpt = sent_chars / sent if sent else min(self._cpt, self._cpt_low) * 0.85
        self._cpt_low = max(0.5, min(self._cpt_low, cpt))  # (the provider counted exactly this text: trust it)
        if learned and learned < self.context_limit:
            self.context_limit = learned
            if self.on_limit:
                self.on_limit(learned)
            self.emit({"type": "context_learned", "limit": learned})
        return True

    def _fallback_summary(self, new_events: list[dict]) -> str:
        """Bounded and honest: the last good summary marked as OLDER, plus what the agent stated since (verbatim)."""
        lim = self.context_limit
        parts = ["## Checkpoint (automatic — the summarizer failed, so this one is partial)"]
        if self._summary:
            parts.append("### Earlier checkpoint (OLDER than everything below — where they disagree, the newer lines win)\n"
                         + ctxm.clip(self._summary, int(lim * 4 * 0.04), "checkpoint"))
        said = [" ".join(e["text"].split()) for e in new_events
                if e.get("type") == "assistant" and e.get("text") and not e.get("depth")]
        kept, used = [], 0
        for s in reversed(said):
            s = ctxm.clip(s, 600)
            if used + len(s) > int(lim * 4 * 0.03):
                break
            kept.insert(0, f"- {s}")
            used += len(s)
        if kept:
            parts.append("### What you stated since then (verbatim, newest last)\n" + "\n".join(kept))
        parts.append("## Next Step\nContinue from the most recent steps below. Where the earlier checkpoint says something "
                     "was not done or not observed, the steps below are newer and win.")
        return "\n\n".join(parts)

    def _rebuild(self, provider, model, system, options, task, summary, anchors, new_events, tr_rel):
        """New conversation: system + earlier turns (re-fitted) + checkpoint, aiming at ~35% of a small window and ~32k
        tokens of a 128k one (ctxm.post_compact_target), so the next compaction is far away and Layer 1 has room to work
        in between: above it the parts shrink in order (tail, files, skills,
        identifiers, history); the essentials (unanswered results, summary, the user's message) only while the rebuild
        is still over half the window."""
        lim = self.context_limit
        window = lim * 4
        shares = {"history": 0.15, "task": 0.08, "pending": 0.12, "tail": 0.06, "files": 0.08, "skills": 0.03,
                  "anchors": 0.02, "summary": 0.10, "git": 0.01}
        caps = {"tail": 40_000, "files": 80_000, "skills": 12_000, "anchors": 8_000, "summary": 60_000, "git": 4_000}  # chars
        order =["tail", "files", "skills", "anchors", "history", "pending", "summary", "task"]
        scale = {k: 1.0 for k in shares}
        cache: dict = {}
        pend_ids = {c.id for c, _ in self._pending}
        tail_events = [e for e in new_events if e.get("id") not in pend_ids][-24:]
        git = ctxm.git_state(self.ws, int(window * shares["git"]))
        saved = {cid: info[1] for cid, info in self._saved.items()}
        essential = ("pending", "summary", "task")  # shrunk only while the rebuild is over POST_COMPACT_MAX
        for _ in range(30):
            b = {k: min(int(window * shares[k] * scale[k]), caps.get(k, 10**9)) for k in shares}
            hist = self._fit_history(self._history, system, b["history"] // 4)
            msg = self._checkpoint(task, summary, anchors, tail_events, b, cache, git, saved, tr_rel)
            chars = sum(len(u) + len(a) + 32 for u, a in hist) + len(msg) + 16
            est = int((self._fixed_chars + chars) / self._cpt)
            if est <= ctxm.post_compact_target(lim):
                break
            k = next((k for k in order if scale[k] > 0.125 and k not in essential), None)
            if k is None and est > lim * ctxm.POST_COMPACT_MAX:
                k = next((k for k in order if scale[k] > 0.125), None)
            if k is None:
                break
            scale[k] /= 2
        conv = provider.conversation(model, system, self.tool_specs(), options)
        self._conv_chars, self._anchor, self._round_sizes = 0, None, []
        self._strip_noted = False  # (a new conversation: no reasoning cleared from it yet)
        _dropped(conv)  # (reasoning a server refused earlier was stripped from another conversation)
        for u, a in hist:
            self._add_user(conv, u)
            self._add_assistant(conv, a)
        self._add_user(conv, msg)
        return conv

    def _checkpoint(self, task, summary, anchors, tail_events, b, cache, git, saved, tr_rel) -> str:
        lines = [ctxm.SUMMARY_PREFIX]
        if self._history:
            lines.append("(The user/assistant messages before this one are earlier turns of this conversation; rules the "
                         "user stated there still apply.)")
        shown = ctxm.clip(task, max(600, b["task"]), "message")
        where = self._request_rel or (tr_rel if shown != task else "")  # the transcript starts with the full message
        if where and where not in shown:
            shown += f"\n[The user's full message is saved in {where} — read what is not shown here with read_file.]"
        lines.append("## The user's latest message (verbatim)\n" + shown)
        if summary:
            lines.append(self._clip_summary(summary, max(1_200, b["summary"])))
        if anchors and b["anchors"] >= 300:
            lines.append(ctxm.clip(ctxm.anchors_block(anchors), b["anchors"], "list"))
        if self._pending:
            per = max(600, b["pending"] // max(1, len(self._pending)))
            rows = []
            for call, res in self._pending:
                args = ctxm.clip(json.dumps(call.arguments, ensure_ascii=False), 400, "arguments")
                body = res.content or "(empty)"
                if len(body) > per:  # re-cut from the full saved output, so the marker says exactly what is missing
                    rel = (self._saved.get(call.id) or ("", ""))[1]
                    full = cache.get(("out", rel)) if rel else None
                    if rel and full is None:
                        try:  # never through a link planted in the archive
                            full = cache[("out", rel)] = self.ws.read_bytes(self.ws.root / rel).decode("utf-8", "replace")
                        except Exception:
                            full = None
                    body = (ctxm.inline_output(full, rel, per, call.name, call.arguments) if full
                            else ctxm.clip(body, per, "output"))
                rows.append(f"### {'ERROR ' if res.is_error else ''}{call.name} {args}\n{body}")
            lines.append("## Results of your last tool calls (you have not seen them yet — act on them)\n" + "\n\n".join(rows))
        if b["tail"] >= 400:
            tail = ctxm.transcript(tail_events, budget=b["tail"], saved=saved, args_cap=600, for_agent=True)
            if tail:
                lines.append("## Most recent steps (abridged — long outputs shortened; the full ones are archived)\n" + tail)
        lines.append(ctxm.files_block(self.ws, list(reversed(list(self._touched.items()))), b["files"], cache))
        if self._skills_loaded:  # (a small budget still lists their names: the model knows to re-load them)
            bodies = {n: (s.get("body") or "") for n, s in skills(self.harness).items()}
            lines.append(ctxm.skills_block(bodies, self._skills_loaded, min(12_000, b["skills"])))
        lines.append(git)
        lines.append(ctxm.recovery_footer(self.archive, tr_rel))
        if self._parts:  # a reply cut by the output limit is being continued: pick it up, never restart it
            lines.append("## Your last reply was cut off by the output limit\nIt is part of your answer to the user (already "
                         "kept); it ended with:\n«…" + "".join(self._parts)[-1_500:] + "»\nContinue it exactly where it "
                         "stopped — no recap, no repetition.")
        elif self._nudged:
            lines.append(FINAL_NUDGE)
        else:
            lines.append("Continue the task from the Next Step without asking for confirmation and without redoing completed "
                         "actions. If something was only partly seen, read just the missing part.")
        return "\n\n".join(p for p in lines if p)

    @staticmethod
    def _clip_summary(summary: str, n: int) -> str:
        return ctxm.clip_summary(summary, n)

    # ------------------------------------------------------------------ tools
    def _other_chat_archive(self, args: dict) -> str | None:
        """Paths into another conversation's archive (.nesicode/chats/<other>) are refused."""
        for key in ("path", "command", "pattern"):
            v = str((args or {}).get(key, ""))
            for m in re.finditer(r"\.nesicode/chats/([^/\s'\"]+)", v):
                if f".nesicode/chats/{m.group(1)}" != self.archive:
                    return f".nesicode/chats/{m.group(1)}"
        return None

    def _on_delta(self, kind: str, delta: str) -> None:
        if self.cancel.is_set():
            raise Cancelled()  # stop the model mid-reply (streaming providers)
        self._live.setdefault(kind, []).append(delta)  # what the user has seen of this reply (see _save_partial)
        self.emit({"type": f"{kind}_delta", "text": delta}, persist=False)

    def _call_tool(self, call: ToolCall, usage: Usage) -> ToolResult:
        """Run a tool; after 3 failures in a row of the same tool, stop offering it for this run (no retry loops)."""
        streak, last = self._fail_streak.get(call.name, (0, ""))
        if streak >= 3:
            return ToolResult(call.id, f"Error: `{call.name}` failed {streak} times in a row with the same error and is now "
                                       "disabled for this run. Stop retrying it: explain the error to the user and how to fix it.\n"
                                       f"Last error: {last[:500]}", True)
        res = self._call_tool_inner(call, usage)
        if not res.is_error:
            self._fail_streak.pop(call.name, None)
        else:
            sig = re.sub(r"\d+", "#", res.content[:160])  # same failure, whatever the arguments
            n = streak + 1 if sig == re.sub(r"\d+", "#", last[:160]) else 1
            self._fail_streak[call.name] = (3 if "NOT RETRYABLE" in res.content else n, res.content)
        return res

    def _cut_call(self, call: ToolCall, cap: int) -> ToolResult:
        """A tool call cut by the output limit: explained, never run, and not counted as a tool failure."""
        msg = (f"Tool call rejected: your reply was cut off by the output limit ({cap:,} tokens per reply) before this "
               f"{call.name} call was complete, so it did not run. Do not resend it whole: split the work into smaller calls "
               "— for a big file, write_file the first part (a few hundred lines at most), then add the rest in further "
               "calls (edit_file at the end of the file, or bash with `cat >> FILE <<'EOF'`).")
        if self._cut_streak >= 3:  # (already counts this reply)
            msg += (f" This is reply number {self._cut_streak} in a row cut at the limit: every reply, tool-call "
                    f"arguments included, must stay well under {cap:,} tokens (about {cap * 3:,} characters).")
        return ToolResult(call.id, msg, True)

    def _track(self, call: ToolCall, res: ToolResult) -> None:
        """Files read/edited and skills loaded in this run (re-injected after a compaction), and the read-only streak."""
        if not res.is_error:
            if call.name in ("read_file", "write_file", "edit_file"):
                try:
                    rel = self.ws.rel(self.ws.resolve(str(call.arguments.get("path", ""))))
                except Exception:
                    rel = ""
                if rel and rel != ".":
                    self._touched.pop(rel, None)
                    self._touched[rel] = "read" if call.name == "read_file" else "edited"
            elif call.name == "load_skill":
                name = str(call.arguments.get("name", ""))
                if name in self._skills_loaded:
                    self._skills_loaded.remove(name)
                self._skills_loaded.append(name)
        if call.name in WRITE_TOOLS or (call.name == "bash" and _WRITES_RX.search(str(call.arguments.get("command", "")))):
            self._ro_streak, self._ro_reminded, self._ro_at = 0, False, 0
        else:
            self._ro_streak += 1

    def _call_tool_inner(self, call: ToolCall, usage: Usage) -> ToolResult:
        # Errors go through the hooks too (pre_tool warnings + post_tool rules), so a hint keyed to an error fires
        if call.parse_error:
            return ToolResult(call.id, self.hooks.finish(call.name, f"Tool call rejected: {call.parse_error}", []), True)
        block, warnings = self.hooks.before(call.name, call.arguments)
        if block:
            return ToolResult(call.id, self.hooks.finish(call.name, block, warnings), True)
        if self.approve is not None:
            allowed, reason = self.approve(call)
            if not allowed:
                return ToolResult(call.id, self.hooks.finish(
                    call.name, f"The user did not allow this {call.name} call ({reason}). Do not retry the same call; "
                    "choose another approach or ask the user what to do.", warnings), True)
        other = self._other_chat_archive(call.arguments)
        if other:
            return ToolResult(call.id, self.hooks.finish(
                call.name, f"Error: {other} belongs to another conversation; each chat only sees its own history.",
                warnings), True)
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
                with request_scope(self.cancel, self._notify_retry):  # Stop ends the vision request at once
                    out = describe_image(
                        self.ws, str(call.arguments.get("path", "")), self.vision_model, self.user_keys,
                        str(call.arguments.get("question", "")),
                    )
            elif call.name == "generate_image" and self.image_model:
                with request_scope(self.cancel, self._notify_retry):  # (and the image download)
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
            return ToolResult(call.id, self.hooks.finish(call.name, f"Error: {exc}", warnings), True)
        except Exception as exc:  # a tool bug must not kill the run
            self._errors.add(call.id)
            return ToolResult(call.id, self.hooks.finish(call.name, f"Error: {type(exc).__name__}: {exc}", warnings), True)
        if warnings:
            out = "\n".join(f"[hook] {w}" for w in warnings) + "\n" + out
        if len(out) > ctxm.SAVE_OVER:
            # Layer 0: the full output lives on disk; the context keeps what fits (and says what is missing).
            try:
                label = str(call.arguments.get("path") or call.arguments.get("command") or "")
                rel = ctxm.save_output(self.ws, call.id, call.name, out, self.archive, label)
                out = ctxm.inline_output(out, rel, self._inline_limit(call.name), call.name, call.arguments)
                # the mask marker describes what the model SAW (a cut read covers fewer lines than the file)
                desc, reread = ctxm.describe_call(call.name, call.arguments, out)
                self._saved[call.id] = (call.name, rel, desc, reread)
            except Exception:
                pass
        return ToolResult(call.id, self.hooks.after(call.name, out))

    def _inline_limit(self, tool: str) -> int:
        """~5% of the window, and below any harness hook that would cut the output again in the middle."""
        n = ctxm.inline_limit(self.context_limit)
        for rule in self.hooks.post:
            t = rule.get("tool", "*")
            if rule.get("action") == "truncate" and (t == "*" or t == tool or (isinstance(t, list) and tool in t)):
                try:
                    n = min(n, max(2_000, int(rule.get("max_chars") or 20_000) - 600))
                except (TypeError, ValueError):
                    pass
        return n

    def _delegate(self, args: dict, usage: Usage) -> str:
        name = str(args.get("agent", ""))
        task = str(args.get("task", "")).strip()
        ag = sub_agents(self.harness).get(name)
        if not ag:
            raise ToolError(f"unknown subagent {name!r}")
        if not task:
            raise ToolError("`task` is required")
        model = ag["meta"].get("model") or self.model_spec
        same = model == self.model_spec
        if same and self.context_limit:
            ctx, out = self.context_limit, self.output_limit
        else:  # its own model: its own window (never the parent's)
            ctx, out = known_limits(model.split("/", 1)[-1]) or DEFAULT_LIMITS
        sub = Agent(
            model=model,
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
            context_limit=ctx,
            output_limit=out,
            on_limit=self.on_limit if same else None,
            archive=self.archive,  # the conversation's own archive (saved outputs, history) stays readable
        )
        res = sub.run(task)
        usage.add(res.usage)
        self.events.extend(sub.events)
        # Only a finished sub-agent answered: a stopped one's text is just what it said at its last step (never its
        # findings), and a refusal is not "nothing found". The result says what happened (persisted, so the next turn's
        # trail and digest say it too).
        if res.status == "error":
            raise ToolError(f"subagent failed: {res.error}")
        if res.status == "cancelled":
            said = ctxm.clip(" ".join((res.final_text or "").split()), 400, "text")
            raise ToolError(f"subagent stopped by the user after {res.steps} step(s); it did not finish"
                            + (f" (what it said last, not an answer: «{said}»)" if said else ""))
        if res.status == "refusal":
            raise ToolError("the subagent's model refused the task (refusal / content filter): it did not do it")
        if res.status == "max_steps":  # (its step budget, or its context, ran out: both end a run as max_steps)
            return (f"[partial: the subagent stopped before finishing — its step budget or its context ran out after "
                    f"{res.steps} step(s); its answer may be incomplete]\n" + (res.final_text or "(it returned nothing)"))
        return res.final_text or "(subagent returned nothing)"

    # ------------------------------------------------------------------ the loop
    def _finish(self, status: str, final_text: str, steps: int, usage: Usage, turn=None) -> AgentResult:
        if turn is not None:
            self._display = max(self._display, turn.usage.input_tokens + _visible(turn) // 4)
        ctx = {"used": self._display, "limit": self.context_limit}
        self.emit({"type": "done", "status": status, "steps": steps, "usage": usage.to_dict(), "context": ctx})
        return AgentResult(final_text, status, steps, usage, self.events)

    def run(self, task: str, history: list[tuple[str, str]] | None = None) -> AgentResult:
        usage = self._usage = Usage()
        steps = 0
        final_text = ""
        self._history = list(history or [])
        self._task = self._task_full = task
        self._tag, self._tag_ok = "", False  # archive files of this run get their own prefix (created on first use)
        self._request_rel = ""
        self._live = {}
        try:
            provider, model = get_provider(self.model_spec, self.user_keys)
            if self.depth == 0:
                self.ws.install_harness(self.harness)
            system = build_system(self.harness, self.ws, base=self.system_base, extra_tools=self.extra_tools)
            if self.system_note.strip():
                system += "\n\n" + self.system_note.strip()
            if self._history:
                system += "\n\n" + self._history_note()
            options = ModelOptions(effort=self.effort, role=self.role, cancel=self.cancel)
            setattr(options, "notify", self._notify_retry)  # retries of a slow provider become visible events
            out_budget = min(options.max_tokens, self.output_limit or options.max_tokens)
            if self.context_limit:
                out_budget = min(out_budget, ctxm.OUTPUT_PER_TURN if self.context_limit <= 400_000 else out_budget)
            options.max_tokens = out_budget
            self._fixed_chars = len(system) + ctxm.tools_chars(self.tool_specs())
            conv, task = self._first_conversation(provider, model, system, options, task)
            self._task = task
            # Context meter: starts from what this message really sends (tool schemas included), then only grows within
            # the turn (it drops only when masking/compaction frees room, and says so).
            self._display = self._est()
            self._ctx_event("new_message")
            self.emit({"type": "start", "model": self.model_spec, "task": task[:4000]})
            while True:
                if self.cancel.is_set():
                    self.emit({"type": "cancelled"})
                    return AgentResult(final_text, "cancelled", steps, usage, self.events)
                room_limited = False
                if self.context_limit:
                    if steps and not self._fits():  # no room for the next reply
                        conv, full = self._make_room(provider, model, system, options, conv)
                        if full:
                            if self._room() < 512:  # not even a final answer fits: stop here, cleanly
                                note = ("[El contexto del modelo está lleno y no se puede compactar más: el trabajo se detuvo "
                                        "aquí. Escribe «continúa» para seguir desde el historial guardado.]")
                                final_text = (final_text + "\n\n" if final_text else "") + note
                                return self._finish("max_steps", final_text, steps, usage)
                            if not self._nudged:
                                self._add_user(conv, CONTEXT_FULL_NUDGE)
                                self._nudged = True
                    room = self._room()
                    max_tokens = min(out_budget, room)
                    room_limited = max_tokens < out_budget
                    options.max_tokens = max(256 if self._nudged else 1_024, max_tokens)
                self._live = {}
                try:
                    turn = conv.step(on_event=self._on_delta)
                except LLMError as exc:
                    if self.cancel.is_set():
                        raise Cancelled() from exc
                    learned = parse_limit(str(exc))
                    if not learned or self._limit_errors >= 3:
                        raise
                    # The model told us its real limit: remember it, re-fit everything to it and try again.
                    self._live = {}  # the reply is asked for again after the compaction
                    self._limit_errors += 1
                    self._denser(str(exc))
                    if self.context_limit != learned:
                        self.context_limit = learned
                        if self.on_limit:
                            self.on_limit(learned)
                        self.emit({"type": "context_learned", "limit": learned})
                    conv = self._compact(provider, model, system, options, self._task, self._est(), reason="context exceeded")
                    continue
                self._live = {}  # the reply is complete: its text and reasoning are recorded below
                self._conv_chars -= _dropped(conv)  # replayed reasoning the server refused (stripped from the history)
                if self._judge and self.context_limit:  # first request after a compaction: did it get under the trigger?
                    self._judge = False
                    real = turn.usage.input_tokens or self._est()
                    self._last_compact_after = real
                    self._no_progress = self._no_progress + 1 if real >= ctxm.SUMMARY_AT * self.context_limit else 0
                    if self._no_progress >= 2:  # compacting cannot bring this context under the trigger: stop trying
                        self._compact_off = True
                if turn.usage.input_tokens:
                    self._calibrate(turn.usage.input_tokens)
                self._conv_chars += _visible(turn) + 16 + (len(turn.thinking or "") if _replays(conv) else 0)
                cut = turn.stop_reason in ("max_tokens", "model_context_window_exceeded")
                if cut and self.context_limit and (room_limited or turn.stop_reason == "model_context_window_exceeded") \
                        and self._can_compact() and self._est() > self.context_limit * 0.45:
                    # The reply was cut because the context left no room: drop it, compact, retry (once per step).
                    usage.add(turn.usage)
                    conv = self._compact(provider, model, system, options, self._task,
                                         turn.usage.input_tokens + turn.usage.output_tokens, reason="truncated reply")
                    continue
                steps += 1
                self._steps = steps
                self._pending = []
                self._limit_errors = 0  # context errors only count while they repeat without progress
                usage.add(turn.usage)
                if turn.thinking:  # archived (history.md, the next turn's trail, compaction transcripts): its end matters
                    self.emit({"type": "thinking", "text": _keep_end(turn.thinking, THINKING_CAP)})
                if turn.text:
                    self.emit({"type": "assistant", "text": turn.text})
                if turn.stop_reason in ("refusal", "content_filter"):  # never a silent empty "done"
                    final_text = turn.text or final_text
                    self.emit({"type": "error", "text": "El modelo rechazó la petición (refusal)." if turn.stop_reason == "refusal"
                               else "El proveedor bloqueó la respuesta con su filtro de contenido (content_filter)."})
                    return AgentResult(final_text, "refusal", steps, usage, self.events)
                cut_calls = [c for c in turn.tool_calls if c.parse_error] if cut else []
                self._cut_streak = self._cut_streak + 1 if cut and (cut_calls or not turn.tool_calls) else 0
                if cut and not turn.tool_calls:
                    # A text reply cut by the output limit is never the final answer: ask for the rest (bounded).
                    if turn.text:
                        self._parts.append(turn.text)
                    final_text = "".join(self._parts) or final_text
                    if self._continuations < 3:
                        self._continuations += 1
                        note = (CONTINUE_NOTE if turn.text else EMPTY_CUT_NOTE).format(cap=options.max_tokens)
                        self._add_user(conv, note)
                        conv = self._after_step(conv, provider, model, system, options, turn, [])
                        continue
                    out, cap = turn.usage.output_tokens, options.max_tokens or 0
                    if out and out * 4 < cap:  # stopped far below the cap: not the model running out of room
                        msg = (f"El proveedor cortó la respuesta del modelo {self._continuations + 1} veces seguidas "
                               f"(finish_reason=length) tras solo {out:,} tokens de {cap:,} permitidos: fallo temporal "
                               "del servidor o un límite de salida menor en el proveedor. Escribe «continúa» para reintentar.")
                    else:
                        msg = (f"La respuesta del modelo se cortó en el límite de salida ({cap:,} tokens) "
                               f"{self._continuations + 1} veces seguidas.")
                    self.emit({"type": "error", "text": msg})
                    return AgentResult(final_text, "error", steps, usage, self.events, error=msg)
                if turn.text:
                    final_text = ("".join(self._parts) + turn.text) if not turn.tool_calls else turn.text
                self._parts, self._continuations = [], 0
                if not turn.tool_calls or self._nudged:
                    return self._finish("max_steps" if self._nudged else "done", final_text, steps, usage, turn)
                results = []
                for call in turn.tool_calls:
                    if self.cancel.is_set():
                        raise Cancelled()
                    self.emit({"type": "tool_call", "id": call.id, "name": call.name, "args": call.arguments})
                    res = self._cut_call(call, options.max_tokens) if call in cut_calls else self._call_tool(call, usage)
                    self._track(call, res)
                    self.emit({
                        "type": "tool_result", "id": call.id, "name": call.name,
                        "output": res.content[:30_000], "is_error": res.is_error,
                    })
                    results.append(res)
                if self._ro_streak >= READ_STREAK and not self._ro_reminded and results \
                        and self._ro_streak - self._ro_at >= READ_STREAK // 2:  # (re-armed: not right after the last)
                    last = results[-1]
                    results[-1] = ToolResult(last.call_id, last.content + self._streak_note(), last.is_error)
                    self._ro_reminded, self._ro_at = True, self._ro_streak
                self._add_results(conv, results)
                self._pending = list(zip(turn.tool_calls, results))
                self._round_sizes.append(len(results))
                if self.cancel.is_set():
                    raise Cancelled()
                conv = self._after_step(conv, provider, model, system, options, turn, results)
                if steps >= self.max_steps and not self._nudged:
                    self._add_user(conv, FINAL_NUDGE)
                    self._nudged = True
        except Cancelled:
            self._save_partial()
            self.emit({"type": "cancelled"})
            return AgentResult(final_text, "cancelled", steps, usage, self.events)
        except LLMError as exc:
            self._save_partial()
            if self.cancel.is_set():  # the provider wrapped our stop into its own error
                self.emit({"type": "cancelled"})
                return AgentResult(final_text, "cancelled", steps, usage, self.events)
            self.emit({"type": "error", "text": str(exc)})
            return AgentResult(final_text, "error", steps, usage, self.events, error=str(exc))
        except Exception as exc:
            self._save_partial()
            if self.cancel.is_set():
                self.emit({"type": "cancelled"})
                return AgentResult(final_text, "cancelled", steps, usage, self.events)
            msg = f"{type(exc).__name__}: {exc}"
            self.emit({"type": "error", "text": msg})
            return AgentResult(final_text, "error", steps, usage, self.events, error=msg)
        finally:
            if self.depth == 0:  # sub-agents share the workspace: only the top-level run ends its background processes
                self._end_background()

    def _after_step(self, conv, provider, model, system, options, turn, results):
        """Layers 1 and 2 after a step, then the context meter."""
        if not self.context_limit:
            return conv
        lim = self.context_limit
        used = self._est()
        reason = ""
        # Layer 1: mask old tool outputs (no LLM), in batches, with hysteresis.
        if used > ctxm.MASK_AT * lim and used - self._masked_at > ctxm.min_freed_tokens(lim):
            freed = self._mask(conv, sum(self._round_sizes[-2:]), int(lim * 4 * ctxm.MASK_KEEP_SHARE),
                               ctxm.min_freed_tokens(lim))
            used = self._est()
            if freed:
                reason = "masked"
                self.emit({"type": "masked", "freed": freed // 4, "used": used, "limit": lim})
                # Hysteresis only after a real masking: a dry run that found too little to clear must not postpone the
                # next try by min_freed tokens (it then came after the compaction threshold: Layer 1 never ran there).
                self._masked_at = used
        # Small windows, past MASK_AT: only the newest reply keeps its replayed reasoning, whatever tool results are kept
        # (each reply's reasoning piled up next to them: a 16k-32k reasoning model compacted every 3-4 calls). After the
        # first time each step only changes the reply before the newest, near the end: the prompt cache stays valid.
        if used > ctxm.MASK_AT * lim and lim <= ctxm.SMALL_WINDOW and _replays(conv):
            freed = _strip_reasoning(conv, 1)
            if freed:
                self._conv_chars -= freed
                if not self._strip_noted:  # (once per conversation: the system prompt already warns about it)
                    self._add_note(conv, MASK_NOTE_REASONING)
                    self._strip_noted = True
                used = self._est()
                reason = reason or "masked"
        # Layer 2: structured summary — only with real growth since the last one (no thrash).
        if used > ctxm.SUMMARY_AT * lim and used >= self._last_compact_after + ctxm.COMPACT_GROWTH * lim \
                and self._can_compact():
            return self._compact(provider, model, system, options, self._task, used)
        self._display = used if reason else max(self._display, used)
        self._ctx_event(reason)
        return conv

    def _make_room(self, provider, model, system, options, conv):
        """No room for the next reply: first clear the old tool outputs (no LLM call; the latest round stays), then
        compact if the context grew enough since the last compaction for one to pay off. Returns (conversation, full)
        — full: nothing more can be freed (the caller asks for a final answer instead of squeezing every reply)."""
        lim = self.context_limit
        freed = self._mask(conv, self._round_sizes[-1] if self._round_sizes else 0, 0, 1)
        if freed:
            self._display = self._est()
            self.emit({"type": "masked", "freed": freed // 4, "used": self._display, "limit": lim})
            self._ctx_event("masked")
            self._masked_at = self._display
            if self._fits():
                return conv, False
        if self._can_compact() and self._est() >= self._last_compact_after + ctxm.COMPACT_GROWTH * lim:
            conv = self._compact(provider, model, system, options, self._task, self._est(), reason="no room for output")
        return conv, not self._fits()


def _visible(turn) -> int:
    """Characters of a reply that go back into the conversation (text + tool calls; the reasoning is counted apart)."""
    return len(turn.text or "") + sum(len(json.dumps(c.arguments, ensure_ascii=False)) + len(c.name) for c in turn.tool_calls)


def _replays(conv) -> bool:
    """Does this conversation send the replies' reasoning back to the model (so it counts in the context)?"""
    try:
        return bool(getattr(conv, "replays_reasoning", False))
    except Exception:
        return False


def _dropped(conv) -> int:
    take = getattr(conv, "take_dropped", None)
    try:
        return int(take()) if callable(take) else 0
    except Exception:
        return 0


def _strip_reasoning(conv, keep: int) -> int:
    strip = getattr(conv, "strip_reasoning", None)
    try:
        return int(strip(keep)) if callable(strip) else 0
    except Exception:
        return 0


def _keep_end(text: str, n: int, what: str = "reasoning") -> str:
    """The last ~n chars of `text` (where a reasoning's conclusions sit), saying how much came before."""
    if len(text) <= n:
        return text
    return f"[… {len(text) - n:,} earlier chars of this {what} omitted …] " + text[len(text) - n:]


DUMP_TOOLS = ("list_dir", "search", "search_history")  # listings: a few lines tell what they were
DUMP_LINES = 6


def _shorten(out: str, n: int, dump: bool = False) -> str:
    """A tool output in about n chars: head and tail LINES, saying what is left out (a listing: its first lines)."""
    lines = out.split("\n")
    if dump and len(lines) > DUMP_LINES + 2:
        head = "\n".join(ln[:240] for ln in lines[:DUMP_LINES])
        return f"{head}\n[... {len(lines) - DUMP_LINES:,} more lines of this listing ...]"
    if len(out) <= n:
        return out
    room = max(200, n - 80)
    kept_head, size = [], 0
    for ln in lines:
        if size + len(ln) + 1 > room * 3 // 5:
            break
        kept_head.append(ln)
        size += len(ln) + 1
    kept_tail: list[str] = []
    for ln in reversed(lines[len(kept_head):]):
        if size + len(ln) + 1 > room:
            break
        kept_tail.insert(0, ln)
        size += len(ln) + 1
    hidden = len(lines) - len(kept_head) - len(kept_tail)
    if size < room // 2:  # huge lines (one-line JSON, minified code): whole lines would hide almost all of it
        h = room * 3 // 5
        return out[:h] + f" [... {len(out) - room:,} chars of this output omitted here ...] " + out[len(out) - (room - h):]
    hidden_chars = len(out) - size
    return "\n".join(kept_head + [f"[... {hidden:,} lines ({hidden_chars:,} chars) of this output omitted here ...]"]
                     + kept_tail)


def trajectory_text(events: list[dict], max_chars: int = 30_000, obs_chars: int = 2_500, args_chars: int = 1_500, *,
                    thinking_chars: int = 0, fit: str = "middle", dumps: bool = False) -> str:
    """Readable trajectory in the [ACTION]/[OBSERVATION]/[ERROR] grammar, long items shortened.

    thinking_chars > 0 adds the reasoning events as [THINKING] lines (their END, where the conclusions sit). Over
    max_chars, `fit` decides what goes: "middle" keeps the first third and the end (harnessing rollouts); "shrink" never
    drops a step — observations, arguments and reasoning get shorter instead, the budget is soft (disk archives such as
    history.md); "tail" keeps the newest steps whole and says how many older ones are not shown (the previous turn's
    trail, which must end where that turn stopped). dumps=True keeps only the first lines of list_dir/search outputs."""
    def render(obs: int, args_cap: int, think: int) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        for e in events:
            who = f"({e['agent']}) " if e.get("agent") else ""
            t = e.get("type")
            if t == "assistant":
                cut = "(interrupted mid-reply) " if e.get("interrupted") else ""
                items.append(("say", f"[ASSISTANT] {cut}{who}{e.get('text') or ''}"))
            elif t == "thinking" and think and e.get("text"):
                items.append(("think", f"[THINKING] {who}{_keep_end(str(e['text']), think)}"))
            elif t == "tool_call":
                args = json.dumps(e.get("args", {}), ensure_ascii=False)
                if len(args) > args_cap:
                    args = args[:args_cap] + "...}"
                items.append(("action", f"[ACTION] {who}{e.get('name')} {args}"))
            elif t == "tool_result":
                out = _shorten(str(e.get("output") or ""), obs, dumps and e.get("name") in DUMP_TOOLS)
                items.append(("obs", f"{'[ERROR]' if e.get('is_error') else '[OBSERVATION]'} {who}{out}"))
            elif t == "error":
                items.append(("say", f"[ERROR] {who}{e.get('text') or ''}"))
            elif t == "done":
                items.append(("say", f"[END] status={e.get('status')} steps={e.get('steps')}"))
        return items

    def size(items) -> int:
        return sum(len(x) + 1 for _, x in items)

    items = render(obs_chars, args_chars, thinking_chars)
    if fit == "shrink":
        obs, args_cap, think = obs_chars, args_chars, thinking_chars
        while size(items) > max_chars and (obs > 300 or args_cap > 300 or think > 300):
            obs, args_cap = max(300, obs // 2), max(300, args_cap // 2)
            think = max(300, think // 2) if think else 0
            items = render(obs, args_cap, think)
        return "\n".join(x for _, x in items)
    if fit == "tail" and size(items) > max_chars:
        gap = "[... the first {n} tool call(s) of this turn are not shown here — search_history finds them ...]"
        room = max_chars - len(gap) - 12
        kept: list[tuple[str, str]] = []
        used = 0
        for it in reversed(items):
            if used + len(it[1]) + 1 > room:
                break
            kept.insert(0, it)
            used += len(it[1]) + 1
        while kept and kept[0][0] == "obs":  # an output whose call is not shown
            kept.pop(0)
        if not kept and items:  # the newest item alone is too big (a long reply cut by Stop): its label and its END,
            kind, line = items[-1]  # where the turn stopped
            keep = max(200, room)
            kept = [(kind, line[:60] + " […] " + line[len(line) - (keep - 66):])]
        hidden = sum(1 for k, _ in items[: len(items) - len(kept)] if k == "action")
        return "\n".join(([gap.format(n=hidden)] if hidden else []) + [x for _, x in kept])
    text = "\n".join(x for _, x in items)
    if len(text) > max_chars:
        head = max_chars // 3
        text = text[:head] + "\n\n... (middle of trajectory omitted) ...\n\n" + text[-(max_chars - head):]
    return text
