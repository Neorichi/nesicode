"""Declarative tool-loop hooks from plugins/hooks.json.

    {"pre_tool":  [{"tool": "bash", "match": "<regex on the arguments>", "action": "block"|"warn", "message": "..."}],
     "post_tool": [{"tool": "*", "match": "<regex on output>", "action": "append", "message": "..."},
                   {"tool": "*", "action": "truncate", "max_chars": 20000}]}

A pre_tool `match` is tried on every string argument on its own (so `$` anchors the end of a command) and
on the JSON of all arguments (rules written against key names keep working). post_tool rules apply to a
tool's output, error messages included (`finish`), so a hint can follow a recurring error signature.

Hooks act inside the loop, so the model cannot skip them, and they cost nothing in
always-on prompt budget. They are data, never code, so a proposed harness cannot run
arbitrary programs on the host through them.
"""

from __future__ import annotations

import json
import re


def _applies(rule: dict, tool: str) -> bool:
    t = rule.get("tool", "*")
    return t == "*" or t == tool or (isinstance(t, list) and tool in t)


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
                try:
                    limit = max(1, int(rule.get("max_chars") or 20000))
                except (TypeError, ValueError):
                    limit = 20000
                if len(output) > limit:
                    half = max(1, limit // 2)  # output[-0:] would be the whole string
                    output = (
                        output[:half]
                        + f"\n... [{len(output) - 2 * half} chars trimmed by hook] ...\n"
                        + output[-half:]
                    )
        return output

    def finish(self, tool: str, output: str, warnings: list[str]) -> str:
        """The text the model gets back from a tool call, success or error: pre_tool warnings first, then
        the post_tool rules. The loop must use it on error results too, or hints keyed to an error never fire."""
        if warnings:
            output = "\n".join(f"[hook] {w}" for w in warnings) + "\n" + output
        return self.after(tool, output)
