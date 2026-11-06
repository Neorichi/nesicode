"""Task-level recursion (Propose-Probe-Compose), selection and reflection.

Parallel mode (width): k base rollouts -> 2k INV/DOM candidates from shared evidence, each
rolled out once -> k independent compositions, each rolled out once -> select.
Sequential mode (depth): k base rollouts -> for t in 1..k: INV_t, DOM_t edit checkpoint C_{t-1},
rolled out -> compose C_t from the whole history, rolled out -> select.
Both schedule 4k rollouts and 3k improvement stages, as in Harness^2.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..agent import Agent, Workspace, apply_changes, copy_workspace, diff_workspaces, trajectory_text
from ..harness import Harness, diff_summary, inventory, normalize, unified_diff, validate
from ..llm import LLMError, ModelOptions, Usage, get_provider
from . import prompts

Emit = Callable[[dict], None]
STAGE_ATTEMPTS = 2
OBS_TRUNC = 24_000

FILE_RX = re.compile(r'<file path="([^"]+)">\n?(.*?)</file>', re.S)
DELETE_RX = re.compile(r'<delete path="([^"]+)"\s*/>')
NOTES_RX = re.compile(r"<notes>(.*?)</notes>", re.S)


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
    improver_context: int | None = None
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


def parse_edits(reply: str, current: Harness) -> tuple[Harness, bool]:
    """Apply a reply's <file>/<delete> blocks to `current`. Returns (new, touched_anything)."""
    new = dict(current)
    touched = False
    for path, body in FILE_RX.findall(reply):
        new[path.strip().lstrip("./")] = body if body.endswith("\n") or not body else body + "\n"
        touched = True
    for path in DELETE_RX.findall(reply):
        new.pop(path.strip().lstrip("./"), None)
        touched = True
    return normalize(new), touched


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

    # ---------------------------------------------------------------- utils
    def emit(self, event: dict) -> None:
        self._emit(event)

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise RuntimeError("cancelled")

    def _add_usage(self, u: Usage) -> None:
        with self._lock:
            self.usage.add(u)

    # ------------------------------------------------------------- rollouts
    def _rollout(self, label: str, h: Harness) -> Rollout:
        self._check_cancel()
        self.harnesses[label] = h
        ro = Rollout(label=label, status="running")
        self.rollouts[label] = ro
        self.emit({"type": "rollout", "label": label, "status": "running"})
        rdir = self.run_dir / "rollouts" / label
        ws_path = rdir / "ws"
        copy_workspace(self.snapshot, ws_path)
        ws = Workspace(ws_path)
        agent = Agent(
            model=self.cfg.executor_model, harness=h, workspace=ws, user_keys=self.user_keys,
            max_steps=self.cfg.max_steps, effort=self.cfg.executor_effort, cancel=self.cancel,
            context_limit=self.cfg.context_limit, output_limit=self.cfg.output_limit,
        )
        res = agent.run(self.cfg.task)
        ro.status, ro.final_text, ro.steps, ro.error = res.status, res.final_text, res.steps, res.error
        ro.usage = res.usage
        ro.delivered = res.delivered
        ro.ws_path = ws_path
        ro.trajectory = trajectory_text(res.events, OBS_TRUNC)
        d = diff_workspaces(self.snapshot, ws_path)
        ro.changes = d
        if self.cfg.check_command.strip() and res.status != "cancelled":
            code, out = ws.run(self.cfg.check_command, timeout=600)
            ro.check_code, ro.check_output = code, out[-6000:]
        self._add_usage(res.usage)
        (rdir / "trajectory.txt").write_text(ro.trajectory, encoding="utf-8")
        (rdir / "events.json").write_text(json.dumps(res.events, ensure_ascii=False), encoding="utf-8")
        (rdir / "diff.patch").write_text(d["diff"], encoding="utf-8")
        (rdir / "harness.json").write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
        (rdir / "check.txt").write_text(
            "" if ro.check_code is None else f"exit code: {ro.check_code}\n{ro.check_output}", encoding="utf-8"
        )
        self.emit({"type": "rollout", **ro.summary()})
        return ro

    def _rollout_many(self, items: list[tuple[str, Harness]]) -> None:
        if not items:
            return
        width = max(1, min(len(items), self.cfg.rollout_width))
        with ThreadPoolExecutor(max_workers=width) as pool:
            list(pool.map(lambda it: self._rollout(*it), items))

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
    def _improver(self, role: str, prompt: str) -> tuple[str, Usage]:
        provider, model = get_provider(self.cfg.improver_model, self.user_keys)
        if self.cfg.improver_context:
            budget = int(self.cfg.improver_context * 3 * 0.8)  # ~3 chars per token, keep 20% for the answer
            if len(prompt) > budget:
                keep = budget // 2
                prompt = prompt[:keep] + "\n\n... [evidence trimmed to fit the model's context window] ...\n\n" + prompt[-keep:]
        system = "You improve the harness of an autonomous coding agent. Follow the mission exactly."
        return provider.complete(model, system, prompt, ModelOptions(effort=self.cfg.improver_effort, role=role, max_tokens=32000))

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
                reply, usage = self._improver(role, prompt)
            except LLMError as exc:
                stage.error = str(exc)
                break
            stage.usage.add(usage)
            self._add_usage(usage)
            notes = NOTES_RX.search(reply)
            new, touched = parse_edits(reply, fallback)
            problems = validate(new)
            if problems:
                err = "\n".join(f"- {p}" for p in problems)
                stage.error = err
                continue
            stage.harness = new
            stage.notes = notes.group(1).strip() if notes else reply.strip()[:4000]
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
        with ThreadPoolExecutor(max_workers=max(1, len(specs))) as pool:
            return list(pool.map(lambda s: self._stage(*s), specs))

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
        cands = [l for l in labels if l in self.rollouts]
        rng = random.Random(self.seed)
        shuffled = cands[:]
        rng.shuffle(shuffled)
        ids = {l: chr(ord("A") + i) for i, l in enumerate(shuffled)}
        blocks = []
        for l in shuffled:
            ro = self.rollouts[l]
            check = "" if ro.check_code is None else f"\nCheck `{self.cfg.check_command}` exit code: {ro.check_code}\n```\n{ro.check_output[-2500:]}\n```"
            blocks.append(
                f"## Candidate {ids[l]}\nstatus: {ro.status}, delivered: {'yes' if ro.delivered else 'NO'}{check}\n"
                f"### Changes\n```diff\n{ro.changes.get('diff', '')[:12_000] or '(no file changes)'}\n```\n"
                f"### Final answer\n{ro.final_text[:3000] or '(none)'}"
            )
        prompt = prompts.select(len(shuffled)) + f"\n\n# THE TASK\n\n{self.cfg.task}\n\n" + "\n\n".join(blocks)
        fallback = next((l for l in labels[::-1] if self.rollouts.get(l) and self.rollouts[l].delivered), labels[0])
        try:
            reply, usage = self._improver("selector", prompt)
            self._add_usage(usage)
        except LLMError as exc:
            return fallback, f"selector failed ({exc}); fell back to {fallback}"
        m = re.search(r"<choice>\s*([A-Z])\s*</choice>", reply)
        reason = re.search(r"<reasoning>(.*?)</reasoning>", reply, re.S)
        by_id = {v: k for k, v in ids.items()}
        if not m or m.group(1) not in by_id:
            return fallback, "selector reply had no valid <choice>; fell back"
        return by_id[m.group(1)], (reason.group(1).strip() if reason else reply.strip()[:2000])

    # ----------------------------------------------------------- reflection
    def _reflect(self, chosen: str) -> dict:
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
            reply, usage = self._improver("reflector", "\n\n".join(parts))
            self._add_usage(usage)
        except LLMError as exc:
            return {"edits": "", "experience": None, "error": str(exc)}
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
        self.run_dir.mkdir(parents=True, exist_ok=True)
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
        ro = self.rollouts.get(chosen)
        if not ro or not ro.ws_path:
            return {}
        return apply_changes(self.snapshot, ro.ws_path, target)

    def cleanup_workspaces(self) -> None:
        """Drop rollout workspace copies (trajectories and diffs stay)."""
        for ro in self.rollouts.values():
            if ro.ws_path and ro.ws_path.exists():
                shutil.rmtree(ro.ws_path, ignore_errors=True)
