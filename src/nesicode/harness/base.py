"""The unmodified base harness every new project starts from."""

from __future__ import annotations

import json

from .tree import Harness

SYSTEM_PROMPT = """\
You are Nesicode, an autonomous software engineering agent working inside a project workspace.

Work the task end to end:
1. Orient: list and read the files that matter before changing anything.
2. Plan briefly, then make focused changes with the file tools.
3. Run the project's own checks (tests, linters, build) with bash when they exist, and fix what you broke.
4. Finish with a short summary of what changed and anything left undone.

Prefer small, correct edits over rewrites. Do not invent files, APIs or results you have not verified.
"""

GUARDRAILS = """\
- When the user asks for a chart, a presentation (pptx), a spreadsheet (xlsx) or a Word/PDF document, load the `documents` skill.
- Stay inside the workspace. Never touch files outside it.
- Never type passwords, codes or other credentials and never log in to accounts (in a browser or anywhere else). If a site asks you to sign in, stop and ask the user to do it.
- Never run destructive commands (rm -rf on broad paths, git push --force, dropping databases).
- Do not exfiltrate secrets or print credentials you find.
- If the request is ambiguous and a wrong guess is costly, say what you assumed in the final summary.
"""

MEMORY = ""

TOOL_GUIDANCE = """\
- `edit_file` needs an exact, unique `old` string: read the file first and copy the text exactly.
- Use `search` to find code before reading whole directories.
- `bash` runs from the workspace root with a timeout; avoid interactive or long-running servers.
- Your private reasoning and old tool outputs may be dropped to save context: write each conclusion to the deliverable or a notes file as soon as you reach it.
"""

# rm with any flags (-rf, -fr, -r -f, --recursive...) and among its arguments the root, the home or everything in them:
# /, /*, ~, ~/, ~/*, $HOME, ${HOME}/*, quoted or not. Not `rm -rf build/`. Its arguments end at the line (a newline, or
# `\n` in the JSON form): `rm -rf dist` + `cd ~` is fine. The argument scan never crosses an argument where another
# match could start — a standalone `rm`, or one right after a quote, `(`, `{` or a backtick (`"rm`, `x(rm`): the last
# such `rm` before the target is the one that matches. Without that, a command of many such words (`echo rm rm …`,
# `echo "rm "rm …`) took quadratic time, holding the GIL — one tool call froze the server for minutes.
RM_MATCH = (r"""(?:^|[\s;&|({`"'])rm(?:[ \t]+(?!(?:[^\s;&|\\]*[({`"'])?rm[ \t])[^\s;&|\\]+)*?[ \t]+["']?"""
            r"""(?:/\*?|~/?\*?|\$\{?HOME\}?/?\*?)["']?(?=$|[\s;&|)`"'])""")
# Earlier spellings of that rule (stored in the harnesses of projects created with them): the agent runs RM_MATCH
# instead (upgrade_hooks). Same verdicts; only the time differs.
LEGACY_RM_MATCHES = (
    r"""(?:^|[\s;&|({`"'])rm(?:\s+[^\s;&|]+)*?\s+["']?(?:/\*?|~/?\*?|\$\{?HOME\}?/?\*?)["']?(?=$|[\s;&|)`"'])""",
    r"""(?:^|[\s;&|({`"'])rm(?:[ \t]+[^\s;&|\\]+)*?[ \t]+["']?(?:/\*?|~/?\*?|\$\{?HOME\}?/?\*?)["']?(?=$|[\s;&|)`"'])""",
    r"""(?:^|[\s;&|({`"'])rm(?:[ \t]+(?!rm[ \t])[^\s;&|\\]+)*?[ \t]+["']?(?:/\*?|~/?\*?|\$\{?HOME\}?/?\*?)["']?"""
    r"""(?=$|[\s;&|)`"'])""",
)

HOOKS = {
    "pre_tool": [
        {
            "tool": "bash",
            "match": RM_MATCH,
            "action": "block",
            "message": "Blocked: refusing to delete a root or home directory.",
        }
    ],
    "post_tool": [{"tool": "*", "action": "truncate", "max_chars": 20000}],
}


def upgrade_hooks(rules: dict) -> dict:
    """Hook rules with any earlier spelling of the shipped rm rule replaced by the current one (a copy)."""
    out = dict(rules)
    for phase in ("pre_tool", "post_tool"):
        if isinstance(out.get(phase), list):
            out[phase] = [{**r, "match": RM_MATCH} if isinstance(r, dict) and r.get("match") in LEGACY_RM_MATCHES else r
                          for r in out[phase]]
    return out


DOCUMENTS_SKILL = """\
---
name: documents
description: Make charts (png), presentations (pptx), spreadsheets (xlsx) and Word/PDF documents with Python, check them and hand them to the user.
---
1. Write a small Python script in the workspace (e.g. `make_slides.py`) that builds the file. Libraries:
   charts `matplotlib`; presentations `python-pptx`; spreadsheets `openpyxl`; Word `python-docx`; PDF `matplotlib` or `reportlab`.
2. Run it with the libraries it needs, in this order of preference:
   - `uv run --no-project --with python-pptx --with matplotlib python make_slides.py` (if `command -v uv` works), or
   - `python3 -m pip install --user --quiet python-pptx matplotlib` once, then `python3 make_slides.py`.
3. Check the result before handing it over:
   - charts: `show_image` the png and look at it with `view_image` if available; fix overlapping labels or empty plots.
   - pptx: reopen it with python-pptx and print each slide's title and number of shapes; every slide must have content.
     Put charts into slides as images (render them with matplotlib first). Keep text short; 16:9 (13.333 x 7.5 in).
   - xlsx/docx: reopen and print the sheet/paragraph summary.
4. Hand every deliverable to the user with `share_file` (and `show_image` for charts), then summarise what is inside.
"""


def base_harness() -> Harness:
    return {
        "system_prompt.md": SYSTEM_PROMPT,
        "guardrails.md": GUARDRAILS,
        "memory.md": MEMORY,
        "tool_guidance.md": TOOL_GUIDANCE,
        "plugins/hooks.json": json.dumps(HOOKS, indent=2) + "\n",
        "skills/documents/SKILL.md": DOCUMENTS_SKILL,
    }
