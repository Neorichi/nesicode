"""Declarative tool-loop hooks from plugins/hooks.json.

    {"pre_tool":  [{"tool": "bash", "match": "<regex on the arguments>", "action": "block"|"warn", "message": "..."}],
     "post_tool": [{"tool": "*", "match": "<regex on output>", "action": "append", "message": "..."},
                   {"tool": "*", "action": "truncate", "max_chars": 20000}]}

A pre_tool `match` is tried on every string argument on its own (so `$` anchors the end of a command) and
on the JSON of all arguments (rules written against key names keep working). post_tool rules apply to a
tool's output, error messages included (`finish`), so a hint can follow a recurring error signature.
A truncate rule for every tool ("*") never cuts read_file below the per-call budget it keeps to itself
(SELF_BOUNDED); a rule that names read_file applies as written.

Hooks act inside the loop, so the model cannot skip them, and they cost nothing in
always-on prompt budget. They are data, never code, so a proposed harness cannot run
arbitrary programs on the host through them.
"""

from __future__ import annotations

import json
import re

from .context import INLINE_MAX

# Outputs already cut to a budget of their own: the loop stops a read_file at a line boundary within ~5% of the window
# (INLINE_MAX chars at most), ends it with a note on where to read on, and saves the whole output. The shipped
# 20,000-char rule for every tool cut those pages to ~19.4k chars on a 128k window instead of 24k: more round trips
# for a long file. (+600: the note.)
SELF_BOUNDED = {"read_file": INLINE_MAX + 600}


def _applies(rule: dict, tool: str) -> bool:
    t = rule.get("tool", "*")
    return t == "*" or t == tool or (isinstance(t, list) and tool in t)


def _max_chars(rule: dict, tool: str) -> int:
    """The chars a truncate rule cuts `tool`'s output to."""
    try:
        limit = max(1, int(rule.get("max_chars") or 20000))
    except (TypeError, ValueError):
        limit = 20000
    return max(limit, SELF_BOUNDED.get(tool, 0)) if rule.get("tool", "*") == "*" else limit


def truncate(output: str, limit: int) -> str:
    """`output` cut to at most `limit` chars (limit >= 1), marker included: its head and its tail, with the number of
    chars removed between them. A limit too small for the marker keeps just the first `limit` chars. Never longer
    than the original."""
    if len(output) <= limit:
        return output
    room = limit - len(f"\n... [{len(output)} chars trimmed by hook] ...\n")  # (the real count has no more digits)
    if room < 2:
        return output[:limit]
    head, tail = (room + 1) // 2, room // 2
    return (output[:head] + f"\n... [{len(output) - head - tail} chars trimmed by hook] ...\n"
            + output[len(output) - tail:])


def _strings(value, out: list[str], limit: int = 200) -> list[str]:
    """Every string inside the (JSON-like) arguments."""
    if len(out) >= limit:
        return out
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _strings(v, out, limit)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _strings(v, out, limit)
    return out


class Hooks:
    def __init__(self, config: dict):
        self.pre = [r for r in config.get("pre_tool", []) if isinstance(r, dict)]
        self.post = [r for r in config.get("post_tool", []) if isinstance(r, dict)]
        self._rx: dict[str, re.Pattern | None] = {}

    def _search(self, pattern, texts: list[str]) -> bool:
        key = str(pattern)
        if key not in self._rx:
            try:
                self._rx[key] = re.compile(key)
            except re.error:  # validate() reports it; at run time a broken rule never matches (and never crashes)
                self._rx[key] = None
        rx = self._rx[key]
        return rx is not None and any(rx.search(t) for t in texts)

    def before(self, tool: str, args: dict) -> tuple[str | None, list[str]]:
        """Returns (block_message, warnings)."""
        texts = [json.dumps(args, ensure_ascii=False), *_strings(args, [])]
        warnings: list[str] = []
        for rule in self.pre:
            if not _applies(rule, tool):
                continue
            if rule.get("match") and not self._search(rule["match"], texts):
                continue
            if rule.get("action") == "block":
                return str(rule.get("message", "Blocked by a harness hook.")), warnings
            if rule.get("action") == "warn":
                warnings.append(str(rule.get("message", "")))
        return None, warnings

    def after(self, tool: str, output: str) -> str:
        for rule in self.post:
            if not _applies(rule, tool):
                continue
            action = rule.get("action")
            if action == "append":
                if not rule.get("match") or self._search(rule["match"], [output]):
                    output = f"{output}\n[hint] {rule.get('message', '')}"
            elif action == "truncate":
                output = truncate(output, _max_chars(rule, tool))
        return output

    def truncate_limit(self, tool: str) -> int | None:
        """The chars `tool`'s output is cut to by the truncate rules (the strictest that applies), or None."""
        return min((_max_chars(r, tool) for r in self.post if r.get("action") == "truncate" and _applies(r, tool)),
                   default=None)

    def finish(self, tool: str, output: str, warnings: list[str]) -> str:
        """The text the model gets back from a tool call, success or error: pre_tool warnings first, then
        the post_tool rules. The loop must use it on error results too, or hints keyed to an error never fire."""
        if warnings:
            output = "\n".join(f"[hook] {w}" for w in warnings) + "\n" + output
        return self.after(tool, output)
