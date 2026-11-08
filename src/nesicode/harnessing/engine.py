"""Task-level recursion (Propose-Probe-Compose), selection and reflection.

Parallel mode (width): k base rollouts -> 2k INV/DOM candidates from shared evidence, each
rolled out once -> k independent compositions, each rolled out once -> select.
Sequential mode (depth): k base rollouts -> for t in 1..k: INV_t, DOM_t edit checkpoint C_{t-1},
rolled out -> compose C_t from the whole history, rolled out -> select.
Both schedule 4k rollouts and 3k improvement stages, as in Harness^2.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..agent import Agent, Workspace, copy_workspace, diff_workspaces, trajectory_text
from ..harness import Harness, diff_summary, inventory, normalize, skills, unified_diff, validate
from ..harness.profiles import marker_problems
from ..llm import DEFAULT_LIMITS, LLMError, ModelOptions, Usage, get_provider, known_limits
from . import prompts
from .apply import apply_rollout

Emit = Callable[[dict], None]
STAGE_ATTEMPTS = 2
OBS_TRUNC = 24_000
IMPROVER_SYSTEM = "You improve the harness of an autonomous coding agent. Follow the mission exactly."
IMPROVER_MAX_TOKENS = 32_000  # full rewritten files + notes (+ thinking where the model has it)
CHARS_PER_TOKEN = 2.5  # pessimistic (diffs, JSON, hashes run below 3) so a trimmed prompt really fits
PROMPT_MARGIN = 2_000  # tokens for the system prompt, framing and tokenizer differences
TRIM_NOTE = "\n\n... [evidence trimmed to fit the model's context window] ...\n\n"

# A reply is read left to right: a <file> body is content (tags inside it are text); <notes> and inline
# reasoning blocks are skipped, whatever tags they quote ("dropped -> <delete .../>", a <think> draft).
# The last group is a closing reasoning tag met outside any block (its opener was in the prompt).
TAG_RX = re.compile(
    r'<file path="([^"]+)">|<delete path="([^"]+)"\s*/>|<(notes|think|thinking|analysis)>|</(think|thinking)>'
)
REASONING_RX = re.compile(r"<(think|thinking|analysis)>.*?</\1>", re.S)  # closed blocks, anywhere
LEADING_REASONING_RX = re.compile(r"\s*<(think|thinking|analysis)>")  # one that opens the reply may be cut
CHOICE_RX = re.compile(r"<choice>\s*([A-Z])\s*</choice>")
TRIGGER_RX = re.compile(r"load (?:the )?`([a-z0-9][a-z0-9_-]{0,47})` skill")
LABEL_PATH_RX = re.compile(r"(rollouts[\\/]+)(?:base|inv|dom|agg|c)\d+")


@dataclass
class HarnessingConfig:
    task: str
    executor_model: str
    improver_model: str
    k: int = 1
    mode: str = "parallel"  # parallel | sequential
    domain: str = "general"
    check_command: str = ""
    max_steps: int = 60
    rollout_width: int = 4
    executor_effort: str = "high"
    context_limit: int | None = None
    output_limit: int | None = None
    improver_context: int | None = None  # None: the known table for the improver model, else DEFAULT_LIMITS
    improver_output: int | None = None
    improver_effort: str = "high"


@dataclass
class Rollout:
    label: str
    status: str = "pending"
    final_text: str = ""
    steps: int = 0
    delivered: bool = False
    error: str = ""
    usage: Usage = field(default_factory=Usage)
    trajectory: str = ""
    changes: dict = field(default_factory=dict)
    check_code: int | None = None
    check_output: str = ""
    ws_path: Path | None = None
    home: Path | None = None  # the rollout's private HOME (tool caches), removed with its copy

    def summary(self) -> dict:
        return {
            "label": self.label,
            "status": self.status,
            "delivered": self.delivered,
            "steps": self.steps,
            "error": self.error,
            "final_text": self.final_text[:2000],
            "files_changed": len(self.changes.get("added", [])) + len(self.changes.get("modified", []))
            + len(self.changes.get("removed", [])),
            "check_code": self.check_code,
            "usage": self.usage.to_dict(),
        }


@dataclass
class Stage:
    label: str
    dim: str
    harness: Harness
    notes: str = ""
    error: str = ""
    kept_base: bool = False
    fell_back: bool = False
    usage: Usage = field(default_factory=Usage)

    def summary(self, base: Harness) -> dict:
        return {
            "label": self.label,
            "dim": self.dim,
            "notes": self.notes[:6000],
            "error": self.error[:1000],
            "kept_base": self.kept_base,
            "fell_back": self.fell_back,
            "changes": diff_summary(base, self.harness),
            "inventory": inventory(self.harness),
            "usage": self.usage.to_dict(),
        }


@dataclass
class Reply:
    files: list[tuple[str, str]]
    deletes: list[str]
    notes: str
    open_file: str = ""  # a <file> block that never closed: the reply was cut
    complete: bool = False  # reached a closing </notes> (the end of a well-formed reply)


ANSWER_BLOCKS = (('<file path="', "</file>"), ("<notes>", "</notes>"), ("<reasoning>", "</reasoning>"),
                 ("<edits>", "</edits>"), ("<experience>", "</experience>"))


def _after_orphan_close(text: str) -> str:
    """Some chat templates open <think> in the prompt, so the reply holds only the closing tag: what comes
    before it is reasoning, not the answer. A closing tag quoted inside an answer block ("candidate B leaked
    </think>") is text: the last one outside every block is the real end of the reasoning."""
    for tag in ("think", "thinking"):
        close = f"</{tag}>"
        if close in text and f"<{tag}>" not in text:
            i = len(text)
            while (i := text.rfind(close, 0, i)) != -1:
                head = text[:i]
                if all(head.rfind(o) <= head.rfind(c) for o, c in ANSWER_BLOCKS):
                    text = text[i + len(close):]
                    break
    return text


def strip_reasoning(text: str) -> str:
    """The answer part of a reply: no inline reasoning blocks (a <choice> drafted there is not the choice).
    As in scan_reply, a reasoning block that opens the reply and never closes holds no answer, while an opener
    quoted mid-reply that never closes ("B's answer starts with <think>") is text: it must not erase the rest."""
    text = _after_orphan_close(text)
    lead = LEADING_REASONING_RX.match(text)
    if lead and f"</{lead.group(1)}>" not in text[lead.end():]:
        return ""
    return REASONING_RX.sub("", text)


def scan_reply(reply: str) -> Reply:
    """The edits of an improver reply, in order (see TAG_RX).

    Skipped as reasoning: a closed <think>/<thinking>/<analysis> block, one that opens the reply and never
    closes (cut while thinking), and everything before a closing tag whose opener is not in the reply (chat
    templates that open <think> in the prompt; drafts written there are not the answer). Tags quoted inside a
    file body or the notes ("base1 leaked </think>") are text, and so is an opener named mid-reply that never
    closes ("its <analysis> showed..."): neither may discard the edits around it."""
    files: list[tuple[str, str]] = []
    deletes: list[str] = []
    notes: list[str] = []
    open_file, complete, pos, reasoned = "", False, 0, False
    while m := TAG_RX.search(reply, pos):
        if m.group(1) is not None:
            end = reply.find("</file>", m.end())
            if end == -1:
                open_file = m.group(1)
                break
            body = reply[m.end():end]
            files.append((m.group(1), body[1:] if body.startswith("\n") else body))
            pos = end + len("</file>")
        elif m.group(2) is not None:
            deletes.append(m.group(2))
            pos = m.end()
        elif m.group(4) is not None:  # a closing reasoning tag outside any block
            if not reasoned:  # its opener was in the prompt: all so far was reasoning, drafts included
                files, deletes, notes, complete, reasoned = [], [], [], False, True
            pos = m.end()  # (after a reasoning block of the reply's own, it is just text)
        else:
            tag = m.group(3)
            end = reply.find(f"</{tag}>", m.end())
            if tag == "notes":
                notes.append(reply[m.end():end if end != -1 else len(reply)].strip())
                complete = complete or end != -1
                if end == -1:  # unclosed notes run to the end of the reply
                    break
            else:
                if end == -1 and reply[:m.start()].strip():  # named mid-reply, never closed: prose, not a block
                    pos = m.end()
                    continue
                reasoned = True
                if end == -1:
                    break  # reasoning that opens the reply and never closes: no answer follows
            pos = end + len(tag) + 3
    return Reply(files, deletes, "\n\n".join(n for n in notes if n), open_file, complete)


def _apply_edits(r: Reply, current: Harness) -> tuple[Harness, bool]:
    new = dict(current)
    touched = False
    for path, body in r.files:
        new[path.strip().lstrip("./")] = body if body.endswith("\n") or not body else body + "\n"
        touched = True
    for path in r.deletes:
        new.pop(path.strip().lstrip("./"), None)
        touched = True
    return normalize(new), touched


def parse_edits(reply: str, current: Harness) -> tuple[Harness, bool]:
    """Apply a reply's <file>/<delete> blocks to `current`. Returns (new, touched_anything)."""
    return _apply_edits(scan_reply(reply), current)


def cut_problem(r: Reply, stop_reason: str) -> str:
    """Why a reply must be rejected as cut off at the output limit ("" if it is complete)."""
    if r.open_file:
        where = f'`<file path="{r.open_file}">` never closed'
    elif stop_reason == "max_tokens" and not r.complete:
        where = "it stopped before its closing </notes>"
    else:
        return ""
    return (f"Your reply was cut off at the output limit ({where}), so none of it was used. Write fewer or shorter "
            "files: only the files you change, each as compact as it can be, and keep the <notes> short.")


def check_proposal(old: Harness, new: Harness) -> list[str]:
    """What an edit of `old` must not do beyond validate(): break the profile markers, delete a skill that a
    trigger still names, or add a trigger for a skill that does not exist (e.g. its file was never written)."""
    errs = marker_problems(old, new)
    trig = new.get("system_prompt.md", "") + "\n" + new.get("guardrails.md", "")
    for name in sorted(set(skills(old)) - set(skills(new))):
        if f"`{name}`" in trig:
            errs.append(f"You deleted skills/{name}/SKILL.md but system_prompt.md or guardrails.md still names "
                        f"`{name}`: delete that trigger too, or keep the skill.")
    before = set(TRIGGER_RX.findall(old.get("system_prompt.md", "") + "\n" + old.get("guardrails.md", "")))
    for name in sorted(set(TRIGGER_RX.findall(trig)) - before - set(skills(new))):
        errs.append(f"A trigger tells the agent to load the `{name}` skill, but skills/{name}/SKILL.md does not "
                    "exist: write the skill or drop the trigger.")
    return errs


def remove_tree(path: Path) -> None:
    """Best-effort rmtree that also removes read-only entries (git objects on Windows, read-only dirs)."""
    path = Path(path)
    if not os.path.lexists(path):
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
        return
    root = os.path.realpath(path)

    def retry(func, p, exc):
        for q in (os.path.dirname(p), p):  # POSIX needs a writable parent; Windows a non-read-only entry
            real = os.path.realpath(q)
            if (real == root or real.startswith(root + os.sep)) and not os.path.islink(q):
                try:
                    os.chmod(q, stat.S_IRWXU if os.path.isdir(q) else stat.S_IREAD | stat.S_IWRITE)
                except OSError:
                    pass
        if func in (os.unlink, os.remove, os.rmdir):
            try:
                func(p)
            except OSError:
                pass

    for _ in range(2):  # the second pass gets into directories whose permissions the first one fixed
        shutil.rmtree(path, onexc=retry)
        if not os.path.lexists(path):
            return


def harness_block(h: Harness, title: str = "CURRENT HARNESS") -> str:
    files = "\n".join(f'<harness_file path="{p}">\n{t}</harness_file>' for p, t in sorted(h.items()))
    return f"# {title}\n\n{files}"


class HarnessingRun:
    def __init__(
        self,
        cfg: HarnessingConfig,
        *,
        base_harness: Harness,
        source_ws: Path,
        run_dir: Path,
        lessons: list[dict] | None = None,
        emit: Emit | None = None,
        user_keys: dict[str, str] | None = None,
        cancel: threading.Event | None = None,
        seed: int = 0,
        extra_files: list[tuple[Path, str]] | None = None,
    ):
        if cfg.mode not in ("parallel", "sequential"):
            raise ValueError("mode must be 'parallel' or 'sequential'")
        if not 1 <= cfg.k <= 4:
            raise ValueError("k must be between 1 and 4")
        self.cfg = cfg
        self.base = normalize(base_harness)
        self.source_ws = Path(source_ws)
        self.run_dir = Path(run_dir)
        self.lessons = lessons or []
        self._emit = emit or (lambda e: None)
        self.user_keys = user_keys
        self.cancel = cancel or threading.Event()
        self.seed = seed
        self.extra_files = extra_files or []  # (source dir, path inside the snapshot), e.g. attachments
        self.harnesses: dict[str, Harness] = {}
        self.rollouts: dict[str, Rollout] = {}
        self.stages: list[Stage] = []
        self.usage = Usage()
        self._lock = threading.Lock()
        self.snapshot = self.run_dir / "source"
        # every spelling of the run dir that can show up in output (absolute / resolved, / or \), longest first;
        # never a relative one: "run" or "." would match ordinary words and dots
        dirs = (self.run_dir.absolute(), self.run_dir.resolve())
        self._run_dirs = sorted({f for d in dirs for f in (str(d), d.as_posix())}, key=len, reverse=True)

    # ---------------------------------------------------------------- utils
    def emit(self, event: dict) -> None:
        self._emit(event)

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise RuntimeError("cancelled")

    def _check_paths(self) -> None:
        """The run dir holds copies of the project, so it must not overlap it: a copy into itself never ends,
        and `source/` or `rollouts/` landing on the project would replace (delete) it."""
        src, rd = self.source_ws.resolve(), self.run_dir.resolve()
        if src == rd or rd.is_relative_to(src) or src.is_relative_to(rd):
            raise ValueError(f"the run directory {rd} must be outside the workspace {src} (and not contain it)")

    def _add_usage(self, u: Usage) -> None:
        with self._lock:
            self.usage.add(u)

    def _parallel(self, fn: Callable, items: list, width: int) -> list:
        """map() over threads. Ctrl+C while waiting stops the workers too, instead of waiting for every agent
        and command to finish (the executor joins its threads before the interrupt can propagate)."""
        with ThreadPoolExecutor(max_workers=max(1, width)) as pool:
            try:
                return list(pool.map(fn, items))
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    self.cancel.set()
                raise

    def _blind(self, text: str) -> str:
        """Hide which rollout produced `text`: absolute paths name it (…/rollouts/<label>/ws)."""
        for rd in self._run_dirs:
            text = re.sub(re.escape(rd) + r"[\\/]+rollouts[\\/]+[\w-]+[\\/]+ws(?![\w.-])", "<workspace>", text)
            text = text.replace(rd, "<run>")
        return LABEL_PATH_RX.sub(r"\1<candidate>", text)

    # ------------------------------------------------------------- rollouts
    def _rollout(self, label: str, h: Harness) -> Rollout:
        self._check_cancel()
        self.harnesses[label] = h
        rdir = self.run_dir / "rollouts" / label
        ws_path = rdir / "ws"
        ro = Rollout(label=label, status="running", ws_path=ws_path)
        self.rollouts[label] = ro
        self.emit({"type": "rollout", "label": label, "status": "running"})
        copy_workspace(self.snapshot, ws_path)
        ws = Workspace(ws_path)
        ro.home = ws.home
        agent = Agent(
            model=self.cfg.executor_model, harness=h, workspace=ws, user_keys=self.user_keys,
            max_steps=self.cfg.max_steps, effort=self.cfg.executor_effort, cancel=self.cancel,
            context_limit=self.cfg.context_limit, output_limit=self.cfg.output_limit,
        )
        res = agent.run(self.cfg.task)
        ro.status, ro.final_text, ro.steps, ro.error = res.status, res.final_text, res.steps, res.error
        ro.usage = res.usage
        ro.delivered = res.delivered
        ro.trajectory = trajectory_text(res.events, OBS_TRUNC)
        d = diff_workspaces(self.snapshot, ws_path)
        ro.changes = d
        if self.cfg.check_command.strip() and res.status != "cancelled" and not self.cancel.is_set():
            code, out = ws.run(self.cfg.check_command, timeout=600)
            # blinded before it is cut, so no half path can give the rollout away later
            ro.check_code, ro.check_output = code, self._blind(out)[-6000:]
        self._add_usage(res.usage)
        (rdir / "trajectory.txt").write_text(ro.trajectory, encoding="utf-8")
        (rdir / "events.json").write_text(json.dumps(res.events, ensure_ascii=False), encoding="utf-8")
        (rdir / "diff.patch").write_text(d["diff"], encoding="utf-8")
        (rdir / "harness.json").write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
        (rdir / "check.txt").write_text(
            "" if ro.check_code is None else f"exit code: {ro.check_code}\n{ro.check_output}", encoding="utf-8"
        )
        self.emit({"type": "rollout", **ro.summary()})
        self._check_cancel()  # Stop during this rollout ends the run here (its record above stays)
        return ro

    def _rollout_many(self, items: list[tuple[str, Harness]]) -> None:
        if items:
            self._parallel(lambda it: self._rollout(*it), items, min(len(items), self.cfg.rollout_width))

    # ------------------------------------------------------------- evidence
    def _rollout_evidence(self, label: str, *, harness_vs: Harness | None = None, notes: str = "") -> str:
        ro = self.rollouts.get(label)
        parts = [f"## Rollout `{label}`"]
        if harness_vs is not None and label in self.harnesses:
            d = unified_diff(harness_vs, self.harnesses[label], max_chars=12_000)
            parts.append("### Harness diff vs the harness being edited\n```diff\n" + (d or "(identical)\n") + "```")
        if notes:
            parts.append(f"### Proposer notes\n{notes[:3000]}")
        if not ro:
            parts.append("(no rollout)")
            return "\n\n".join(parts)
        check = "" if ro.check_code is None else f", check exit code {ro.check_code}"
        parts.append(f"status: {ro.status}, delivered: {'yes' if ro.delivered else 'NO'}, steps: {ro.steps}{check}")
        if ro.error:
            parts.append(f"error: {ro.error}")
        parts.append("### Trajectory\n" + ro.trajectory)
        diff = ro.changes.get("diff", "")
        parts.append("### Workspace diff\n```diff\n" + (diff[:10_000] or "(no file changes)\n") + "```")
        if ro.check_code is not None:
            parts.append(f"### Check output (`{self.cfg.check_command}`)\n```\n{ro.check_output[-3000:]}\n```")
        parts.append("### Final answer\n" + (ro.final_text[:3000] or "(none)"))
        return "\n\n".join(parts)

    def _lessons_block(self) -> str:
        if not self.lessons:
            return ""
        rows = [
            f"- cue: {l.get('cue','')}\n  lesson: {l.get('lesson','')}\n  evidence: {l.get('evidence','')}"
            f"\n  confidence: {l.get('confidence','')}"
            for l in self.lessons[:12]
        ]
        return f"# LESSONS FROM EARLIER TASKS IN `{self.cfg.domain}` (hypotheses, re-verify)\n\n" + "\n".join(rows)

    def _context(self, current: Harness, base_labels: list[str], pool: list[tuple[str, str]]) -> str:
        parts = [f"# THE TASK\n\n{self.cfg.task}", harness_block(current)]
        if (lb := self._lessons_block()):
            parts.append(lb)
        parts.append("# ROLLOUTS UNDER THE UNMODIFIED HARNESS\n\n" + "\n\n".join(
            self._rollout_evidence(l) for l in base_labels
        ))
        if pool:
            parts.append("# CANDIDATES AND THEIR ROLLOUTS\n\n" + "\n\n".join(
                self._rollout_evidence(l, harness_vs=current, notes=n) for l, n in pool
            ))
        return "\n\n".join(parts)

    # --------------------------------------------------------------- stages
    def _improver_limits(self, model: str) -> tuple[int, int]:
        known = known_limits(model) or DEFAULT_LIMITS
        return self.cfg.improver_context or known[0], self.cfg.improver_output or known[1]

    def _improver(self, role: str, prompt: str) -> tuple[str, Usage, str]:
        """One improver call -> (text, usage, stop_reason), sized to the improver's own window and output cap."""
        self._check_cancel()
        provider, model = get_provider(self.cfg.improver_model, self.user_keys)
        context, output = self._improver_limits(model)
        max_tokens = max(1_024, min(IMPROVER_MAX_TOKENS, output, context // 4))
        budget = int((context - max_tokens - PROMPT_MARGIN) * CHARS_PER_TOKEN)
        if len(prompt) > budget:
            keep = max(0, budget - len(TRIM_NOTE)) // 2
            prompt = prompt[:keep] + TRIM_NOTE + prompt[len(prompt) - keep:]
        options = ModelOptions(effort=self.cfg.improver_effort, role=role, max_tokens=max_tokens, cancel=self.cancel)
        conv = provider.conversation(model, IMPROVER_SYSTEM, [], options)
        conv.add_user(prompt)
        try:
            turn = conv.step()
        finally:
            self._check_cancel()  # Stop pressed meanwhile: an error or a late reply both mean "cancelled"
        if turn.stop_reason == "refusal":
            raise LLMError("the model declined this request (refusal)")
        return turn.text, turn.usage, turn.stop_reason

    def _stage(self, label: str, dim: str, job: str, fallback: Harness, context: str) -> Stage:
        self._check_cancel()
        self.emit({"type": "stage", "label": label, "dim": dim, "status": "running"})
        role = "composer" if dim in ("AGG", "COMPOSE") else "proposer"
        stage = Stage(label=label, dim=dim, harness=dict(fallback))
        err = ""
        for attempt in range(1, STAGE_ATTEMPTS + 1):
            prompt = job + "\n\n" + context
            if err:
                prompt += f"\n\n# YOUR PREVIOUS ATTEMPT WAS REJECTED\n\n{err}\n\nFix it and reply again with full files."
            try:
                reply, usage, stop = self._improver(role, prompt)
            except LLMError as exc:
                stage.error = str(exc)
                break
            stage.usage.add(usage)
            self._add_usage(usage)
            r = scan_reply(reply)
            new, touched = _apply_edits(r, fallback)
            problems = [cut] if (cut := cut_problem(r, stop)) else []
            if not problems:
                try:
                    problems = validate(new) + check_proposal(fallback, new)
                except Exception as exc:  # a malformed proposal is a rejected attempt, never a crashed run
                    problems = [f"The proposal could not be validated ({type(exc).__name__}: {exc}); write well-formed files."]
            if problems:
                err = "\n".join(f"- {p}" for p in problems)
                stage.error = err
                continue
            stage.harness = new
            stage.notes = r.notes or strip_reasoning(reply).strip()[:4000]
            stage.kept_base = not touched or new == fallback
            stage.error = ""
            break
        else:
            stage.fell_back = True
        if stage.error:
            stage.fell_back = True
            stage.harness = dict(fallback)
        self.stages.append(stage)
        self.emit({"type": "stage", "status": "done", **stage.summary(self.base)})
        return stage

    def _stages_parallel(self, specs: list[tuple[str, str, str, Harness, str]]) -> list[Stage]:
        return self._parallel(lambda s: self._stage(*s), specs, len(specs))

    # ----------------------------------------------------------------- modes
    def _run_parallel(self, base_labels: list[str]) -> list[str]:
        k, dom = self.cfg.k, self.cfg.domain
        ctx = self._context(self.base, base_labels, [])
        self.emit({"type": "phase", "phase": "propose"})
        specs = [
            (f"{d.lower()}{i}", d, prompts.propose(d, dom, k=k, index=i, round_t=None), self.base, ctx)
            for i in range(1, k + 1) for d in ("INV", "DOM")
        ]
        cands = self._stages_parallel(specs)
        self.emit({"type": "phase", "phase": "probe"})
        self._rollout_many([(s.label, s.harness) for s in cands])
        self.emit({"type": "phase", "phase": "compose"})
        pool = [(s.label, s.notes or s.error) for s in cands]
        ctx2 = self._context(self.base, base_labels, pool)
        aggs = self._stages_parallel([
            (f"agg{i}", "AGG", prompts.compose(dom, k=k, index=i, round_t=None, final=True), self.base, ctx2)
            for i in range(1, k + 1)
        ])
        self.emit({"type": "phase", "phase": "probe_final"})
        self._rollout_many([(s.label, s.harness) for s in aggs])
        return [s.label for s in aggs]

    def _run_sequential(self, base_labels: list[str]) -> list[str]:
        k, dom = self.cfg.k, self.cfg.domain
        current = dict(self.base)
        history: list[tuple[str, str]] = []
        finals = []
        for t in range(1, k + 1):
            self.emit({"type": "phase", "phase": "propose", "round": t})
            ctx = self._context(current, base_labels, history)
            rt = t if k > 1 else None
            cands = self._stages_parallel([
                (f"inv{t}", "INV", prompts.propose("INV", dom, k=k, index=t, round_t=rt), current, ctx),
                (f"dom{t}", "DOM", prompts.propose("DOM", dom, k=k, index=t, round_t=rt), current, ctx),
            ])
            self.emit({"type": "phase", "phase": "probe", "round": t})
            self._rollout_many([(s.label, s.harness) for s in cands])
            round_pool = history + [(s.label, s.notes or s.error) for s in cands]
            self.emit({"type": "phase", "phase": "compose", "round": t})
            comp = self._stage(
                f"c{t}", "COMPOSE",
                prompts.compose(dom, k=k, index=1, round_t=t if k > 1 else None, final=t == k),
                current, self._context(current, base_labels, round_pool),
            )
            self.emit({"type": "phase", "phase": "probe_final", "round": t})
            self._rollout(comp.label, comp.harness)
            history = round_pool + [(comp.label, comp.notes or comp.error)]
            current = comp.harness
            finals.append(comp.label)
        return finals

    # ------------------------------------------------------------ selection
    def _select(self, labels: list[str]) -> tuple[str, str]:
        self._check_cancel()
        cands = [l for l in labels if l in self.rollouts]
        rng = random.Random(self.seed)
        shuffled = cands[:]
        rng.shuffle(shuffled)
        ids = {l: chr(ord("A") + i) for i, l in enumerate(shuffled)}
        blocks = []
        for l in shuffled:
            ro = self.rollouts[l]
            # Blind for real: the rollout's absolute path (…/rollouts/<label>/ws) would name it. Blinded, then cut.
            check = "" if ro.check_code is None else (
                f"\nCheck `{self.cfg.check_command}` exit code: {ro.check_code}\n```\n{self._blind(ro.check_output)[-2500:]}\n```"
            )
            diff = self._blind(ro.changes.get("diff", ""))[:12_000]
            blocks.append(
                f"## Candidate {ids[l]}\nstatus: {ro.status}, delivered: {'yes' if ro.delivered else 'NO'}{check}\n"
                f"### Changes\n```diff\n{diff or '(no file changes)'}\n```\n"
                f"### Final answer\n{self._blind(ro.final_text)[:3000] or '(none)'}"
            )
        prompt = prompts.select(len(shuffled)) + f"\n\n# THE TASK\n\n{self.cfg.task}\n\n" + "\n\n".join(blocks)
        fallback = next((l for l in labels[::-1] if self.rollouts.get(l) and self.rollouts[l].delivered), labels[0])
        try:
            reply, usage, _ = self._improver("selector", prompt)
            self._add_usage(usage)
        except LLMError as exc:
            return fallback, f"selector failed ({exc}); fell back to {fallback}"
        reply = strip_reasoning(reply)
        choices = CHOICE_RX.findall(reply)
        reason = re.search(r"<reasoning>(.*?)</reasoning>", reply, re.S)
        by_id = {v: k for k, v in ids.items()}
        if not choices or choices[-1] not in by_id:
            return fallback, "selector reply had no valid <choice>; fell back"
        return by_id[choices[-1]], (reason.group(1).strip() if reason else reply.strip()[:2000])

    # ----------------------------------------------------------- reflection
    def _reflect(self, chosen: str) -> dict:
        self._check_cancel()
        parts = [
            prompts.experience(self.cfg.domain),
            f"# THE TASK\n\n{self.cfg.task}",
            harness_block(self.base, "BASE HARNESS"),
            f"# CHOSEN RESULT: `{chosen}`",
        ]
        for st in self.stages:
            parts.append(self._rollout_evidence(st.label, harness_vs=self.base, notes=st.notes or st.error))
        for l, ro in self.rollouts.items():
            if l.startswith("base"):
                parts.append(self._rollout_evidence(l))
        try:
            reply, usage, _ = self._improver("reflector", "\n\n".join(parts))
            self._add_usage(usage)
        except LLMError as exc:
            return {"edits": "", "experience": None, "error": str(exc)}
        reply = strip_reasoning(reply)
        edits = re.search(r"<edits>(.*?)</edits>", reply, re.S)
        exp_m = re.search(r"<experience>(.*?)</experience>", reply, re.S)
        exp = None
        if exp_m:
            fields = {}
            for key in ("cue", "lesson", "evidence", "confidence"):
                fm = re.search(rf"^{key}:\s*(.+?)(?=^\w+:|\Z)", exp_m.group(1), re.S | re.M)
                if fm:
                    fields[key] = fm.group(1).strip()
            if all(fields.get(k) for k in ("cue", "lesson", "evidence")):
                exp = fields
        return {"edits": edits.group(1).strip() if edits else "", "experience": exp}

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        """The whole run. Raises RuntimeError("cancelled") when Stop was pressed at any point; then (and on any
        failure) no workspace copy is left behind. On success the snapshot and the chosen copy stay (for a later
        apply) and the other copies go; trajectories, diffs and checks always stay."""
        self._check_paths()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            copy_workspace(self.source_ws, self.snapshot)
            for src, rel in self.extra_files:
                if Path(src).is_dir():
                    shutil.copytree(src, self.snapshot / rel, dirs_exist_ok=True)
            k = self.cfg.k
            self.emit({"type": "phase", "phase": "base"})
            base_labels = [f"base{i}" for i in range(1, k + 1)]
            self._rollout_many([(l, self.base) for l in base_labels])
            finals = self._run_parallel(base_labels) if self.cfg.mode == "parallel" else self._run_sequential(base_labels)
            self.emit({"type": "phase", "phase": "select"})
            chosen, reason = self._select(base_labels + finals)
            self.emit({"type": "selected", "label": chosen, "reason": reason[:4000]})
            self.emit({"type": "phase", "phase": "reflect"})
            reflection = self._reflect(chosen)
            self._check_cancel()
            self.cleanup_workspaces(keep=chosen)  # can take minutes with big projects (node_modules, .git)...
            self._check_cancel()  # ...and a Stop pressed meanwhile still cancels: nothing applied or adopted
        except BaseException:
            self.cleanup_workspaces(snapshot=True)  # cancelled or failed: nothing will ever be applied
            raise
        adopted = self.harnesses.get(chosen, self.base)
        self.emit({"type": "phase", "phase": "done"})
        return {
            "mode": self.cfg.mode,
            "k": k,
            "finals": finals,
            "base_labels": base_labels,
            "selected": chosen,
            "selected_reason": reason,
            "harness_changed": adopted != self.base,
            "adopted_harness": adopted,
            "harness_diff": unified_diff(self.base, adopted),
            "stages": [s.summary(self.base) for s in self.stages],
            "rollouts": {l: r.summary() for l, r in self.rollouts.items()},
            "selected_changes": {k2: v for k2, v in self.rollouts[chosen].changes.items() if k2 != "diff"}
            if chosen in self.rollouts else {},
            "reflection": reflection,
            "usage": self.usage.to_dict(),
        }

    def apply_selected(self, chosen: str, target: Path) -> dict:
        """Apply the chosen rollout's file changes to `target`, three-way (see apply.py): files changed there
        since the run started are never overwritten or deleted; they come back as "conflicts"."""
        ro = self.rollouts.get(chosen)
        if not ro or not ro.ws_path or not ro.ws_path.exists():
            return {}
        return apply_rollout(self.snapshot, ro.ws_path, target)

    def cleanup_workspaces(self, keep: str | None = None, snapshot: bool = False) -> None:
        """Drop rollout workspace copies (and their private HOMEs) except `keep`'s; with snapshot=True also the
        project snapshot, after which nothing can be applied. Trajectories, diffs and checks stay."""
        run_dir = self.run_dir.resolve()
        for label, ro in list(self.rollouts.items()):
            if ro.home and ro.home.resolve().is_relative_to(run_dir):  # only a HOME this run owns
                remove_tree(ro.home)
            if label != keep and ro.ws_path:
                remove_tree(ro.ws_path)
        if snapshot:
            remove_tree(self.snapshot)
