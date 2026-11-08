"""The editable harness: a flat {relative path: text} tree with eight components.

    system_prompt.md                 role and procedure            ALWAYS ON
    guardrails.md                    standing rules                ALWAYS ON
    memory.md                        durable project/domain facts  ALWAYS ON
    tool_guidance.md                 how to use the tools          ALWAYS ON
    skills/<name>/SKILL.md           on-demand workflow            loaded via load_skill
    skills/<name>/scripts/<f>        helper carried by a skill     run via bash
    sub_agents/<name>/AGENT.md       delegated agent, own context  run via delegate
    scripts/<f>.py|.sh               deterministic helper          run via bash
    plugins/hooks.json               declarative tool-loop hooks   ALWAYS REGISTERED
"""

from __future__ import annotations

import difflib
import json
import re

Harness = dict[str, str]

ALWAYS_ON = ("system_prompt.md", "guardrails.md", "memory.md", "tool_guidance.md")
HOOKS_FILE = "plugins/hooks.json"
MAX_ALWAYS_ON_CHARS = 24_000
MAX_FILE_CHARS = 60_000

_NAME = r"[a-z0-9][a-z0-9_-]{0,47}"
_FILE = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}"
PATTERNS = [
    re.compile(r"^(system_prompt|guardrails|memory|tool_guidance)\.md$"),
    re.compile(rf"^skills/{_NAME}/SKILL\.md$"),
    re.compile(rf"^skills/{_NAME}/scripts/{_FILE}\.(py|sh)$"),
    re.compile(rf"^sub_agents/{_NAME}/AGENT\.md$"),
    re.compile(rf"^scripts/{_FILE}\.(py|sh)$"),
    re.compile(r"^plugins/hooks\.json$"),
]

COMPONENTS = [
    {"key": "system_prompt", "label": "System prompt", "path": "system_prompt.md", "always_on": True},
    {"key": "guardrails", "label": "Guardrails", "path": "guardrails.md", "always_on": True},
    {"key": "memory", "label": "Memoria", "path": "memory.md", "always_on": True},
    {"key": "tool_guidance", "label": "Guía de herramientas", "path": "tool_guidance.md", "always_on": True},
    {"key": "skills", "label": "Skills", "path": "skills/<name>/SKILL.md", "always_on": False},
    {"key": "sub_agents", "label": "Subagentes", "path": "sub_agents/<name>/AGENT.md", "always_on": False},
    {"key": "scripts", "label": "Scripts", "path": "scripts/<name>.py", "always_on": False},
    {"key": "plugins", "label": "Plugins (hooks)", "path": "plugins/hooks.json", "always_on": True},
]


def component_of(path: str) -> str:
    if path in ALWAYS_ON:
        return path[:-3]
    head = path.split("/", 1)[0]
    return {"skills": "skills", "sub_agents": "sub_agents", "scripts": "scripts", "plugins": "plugins"}.get(head, "other")


def is_editable(path: str) -> bool:
    return any(p.match(path) for p in PATTERNS)


def normalize(h: Harness) -> Harness:
    return {k: v.replace("\r\n", "\n") for k, v in sorted(h.items())}


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Tiny `key: value` frontmatter parser (no nested YAML needed here)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, text[end + 4 :].lstrip("\n")


def skills(h: Harness) -> dict[str, dict]:
    out = {}
    for path, text in h.items():
        m = re.match(rf"^skills/({_NAME})/SKILL\.md$", path)
        if m:
            meta, body = parse_frontmatter(text)
            out[m.group(1)] = {"description": meta.get("description", ""), "body": body, "meta": meta}
    return out


def sub_agents(h: Harness) -> dict[str, dict]:
    out = {}
    for path, text in h.items():
        m = re.match(rf"^sub_agents/({_NAME})/AGENT\.md$", path)
        if m:
            meta, body = parse_frontmatter(text)
            out[m.group(1)] = {"description": meta.get("description", ""), "body": body, "meta": meta}
    return out


def always_on_text(h: Harness) -> str:
    return "\n".join(h.get(p, "") for p in ALWAYS_ON)


def hooks(h: Harness) -> dict:
    """The hook rules of a harness; anything malformed reads as "no rules" (validate() reports it)."""
    empty = {"pre_tool": [], "post_tool": []}
    raw = h.get(HOOKS_FILE, "").strip()
    if not raw:
        return empty
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return empty
    if not isinstance(data, dict):
        return empty
    return {k: v if isinstance(v := data.get(k, []), list) else [] for k in ("pre_tool", "post_tool")}


def _validate_hooks(raw: str) -> list[str]:
    errs: list[str] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [f"{HOOKS_FILE} is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return [f"{HOOKS_FILE} must be a JSON object with `pre_tool` and/or `post_tool` lists"]
    allowed = {"pre_tool": {"block", "warn"}, "post_tool": {"append", "truncate"}}
    for phase, actions in allowed.items():
        rules = data.get(phase, [])
        if not isinstance(rules, list):
            errs.append(f"{HOOKS_FILE}: `{phase}` must be a list")
            continue
        for i, rule in enumerate(rules):
            where = f"{HOOKS_FILE} {phase}[{i}]"
            if not isinstance(rule, dict):
                errs.append(f"{where} must be an object")
                continue
            # Type-check before use: LLM-written JSON often has null or lists where a string belongs.
            action = rule.get("action")
            if not isinstance(action, str) or action not in actions:
                errs.append(f"{where}: action must be one of {sorted(actions)} (a string)")
                action = None
            tool = rule.get("tool", "*")
            if not (isinstance(tool, str) and tool.strip()) and not (
                isinstance(tool, list) and tool and all(isinstance(t, str) and t.strip() for t in tool)
            ):
                errs.append(f'{where}: `tool` must be a tool name, "*" or a list of tool names')
            match = rule.get("match")  # null or "" means "no condition", as at runtime
            if match is not None and not isinstance(match, str):
                errs.append(f"{where}: `match` must be a regex string (or left out)")
            elif match:
                try:
                    re.compile(match)
                except (re.error, RecursionError, OverflowError) as exc:
                    errs.append(f"{where}: bad regex in `match`: {exc}")
            msg = rule.get("message")
            if action in ("block", "warn", "append") and not (isinstance(msg, str) and msg.strip()):
                errs.append(f"{where}: `message` (a non-empty string) is required for action {action!r}")
            if action == "truncate":
                mc = rule.get("max_chars")
                if isinstance(mc, bool) or not isinstance(mc, int) or mc <= 0:
                    errs.append(f"{where}: `max_chars` (a positive integer) is required for truncate")
    return errs


def validate(h: Harness) -> list[str]:
    """Every problem, in actionable prose (it is fed back verbatim to a proposer)."""
    errs: list[str] = []
    for path, text in h.items():
        if not is_editable(path):
            errs.append(
                f"`{path}` is off the editable surface. Allowed: system_prompt.md, guardrails.md, "
                "memory.md, tool_guidance.md, skills/<name>/SKILL.md, skills/<name>/scripts/<f>.py, "
                "sub_agents/<name>/AGENT.md, scripts/<f>.py, plugins/hooks.json (names: lowercase, digits, _ or -)."
            )
        if len(text) > MAX_FILE_CHARS:
            errs.append(f"`{path}` is {len(text):,} chars; the limit per file is {MAX_FILE_CHARS:,}.")
    if not h.get("system_prompt.md", "").strip():
        errs.append("`system_prompt.md` must exist and must not be empty.")
    on = always_on_text(h)
    if len(on) > MAX_ALWAYS_ON_CHARS:
        errs.append(
            f"always-on text is {len(on):,} chars (limit {MAX_ALWAYS_ON_CHARS:,}); move material into skills."
        )
    trigger_text = h.get("system_prompt.md", "") + "\n" + h.get("guardrails.md", "")
    for name, sk in skills(h).items():
        if not sk["description"]:
            errs.append(f"skills/{name}/SKILL.md needs frontmatter with a `description:` line.")
        if sk["meta"].get("name") and sk["meta"]["name"] != name:
            errs.append(f"skills/{name}/SKILL.md: frontmatter name {sk['meta']['name']!r} must match its directory.")
        if not re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", trigger_text):
            errs.append(
                f"skill `{name}` is never named in system_prompt.md or guardrails.md. Add a trigger line such as "
                f"\"When <situation>, load the `{name}` skill.\" or the skill will never be used."
            )
    for name, ag in sub_agents(h).items():
        if not ag["description"]:
            errs.append(f"sub_agents/{name}/AGENT.md needs frontmatter with a `description:` line.")
        if not ag["body"].strip():
            errs.append(f"sub_agents/{name}/AGENT.md has no instructions.")
    if HOOKS_FILE in h and h[HOOKS_FILE].strip():
        errs += _validate_hooks(h[HOOKS_FILE])
    referenced = set(re.findall(r"(?<![\w/])(scripts/[A-Za-z0-9_.-]+\.(?:py|sh))", on))
    for name, sk in skills(h).items():
        for ref in re.findall(r"(?<![\w/])(scripts/[A-Za-z0-9_.-]+\.(?:py|sh))", sk["body"]):
            if f"skills/{name}/{ref}" not in h and ref not in h:
                errs.append(f"skill `{name}` invokes `{ref}` but neither skills/{name}/{ref} nor {ref} exists.")
    for ref in referenced:
        if ref not in h:
            errs.append(f"always-on text invokes `{ref}` but that script is not in the harness.")
    return errs


def diff_summary(old: Harness, new: Harness) -> dict[str, list[str]]:
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "modified": sorted(p for p in set(old) & set(new) if old[p] != new[p]),
    }


def unified_diff(old: Harness, new: Harness, max_chars: int = 40_000) -> str:
    out: list[str] = []
    for path in sorted(set(old) | set(new)):
        a, b = old.get(path), new.get(path)
        if a == b:
            continue
        out.extend(
            difflib.unified_diff(
                (a or "").splitlines(keepends=True),
                (b or "").splitlines(keepends=True),
                fromfile=f"a/{path}" if a is not None else "/dev/null",
                tofile=f"b/{path}" if b is not None else "/dev/null",
            )
        )
    text = "".join(line if line.endswith("\n") else line + "\n" for line in out)
    return text if len(text) <= max_chars else text[:max_chars] + "\n... (diff truncated)\n"


def inventory(h: Harness) -> str:
    n_sk, n_ag = len(skills(h)), len(sub_agents(h))
    n_sc = sum(1 for p in h if p.startswith("scripts/") or "/scripts/" in p)
    rules = hooks(h) if HOOKS_FILE in h and not _validate_hooks(h[HOOKS_FILE] or "{}") else {"pre_tool": [], "post_tool": []}
    n_hk = len(rules["pre_tool"]) + len(rules["post_tool"])
    bits = [f"{v} {k}" for k, v in (("skills", n_sk), ("subagentes", n_ag), ("scripts", n_sc), ("hooks", n_hk)) if v]
    return f"{', '.join(bits) or 'solo texto always-on'}; {len(always_on_text(h)):,} chars always-on"
