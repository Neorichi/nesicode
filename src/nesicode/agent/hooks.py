"""Declarative tool-loop hooks from plugins/hooks.json.

    {"pre_tool":  [{"tool": "bash", "match": "<regex on JSON args>", "action": "block"|"warn", "message": "..."}],
     "post_tool": [{"tool": "*", "match": "<regex on output>", "action": "append", "message": "..."},
                   {"tool": "*", "action": "truncate", "max_chars": 20000}]}

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


class Hooks:
    def __init__(self, config: dict):
        self.pre = [r for r in config.get("pre_tool", []) if isinstance(r, dict)]
        self.post = [r for r in config.get("post_tool", []) if isinstance(r, dict)]

    def before(self, tool: str, args: dict) -> tuple[str | None, list[str]]:
        """Returns (block_message, warnings)."""
        blob = json.dumps(args, ensure_ascii=False)
        warnings: list[str] = []
        for rule in self.pre:
            if not _applies(rule, tool):
                continue
            if rule.get("match") and not re.search(rule["match"], blob):
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
                if not rule.get("match") or re.search(rule["match"], output):
                    output = f"{output}\n[hint] {rule.get('message', '')}"
            elif action == "truncate":
                limit = int(rule.get("max_chars") or 20000)
                if len(output) > limit:
                    half = limit // 2
                    output = (
                        output[:half]
                        + f"\n... [{len(output) - limit} chars trimmed by hook] ...\n"
                        + output[-half:]
                    )
        return output
