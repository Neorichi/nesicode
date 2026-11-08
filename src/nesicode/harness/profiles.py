"""Project profiles: a starting harness tuned for a kind of work (General, Security, Web, Data, Documents).

A profile is an overlay on the base harness: a block appended to system_prompt.md / guardrails.md /
tool_guidance.md between markers, plus its own skills. It stays a normal, versioned harness, so
harnessing keeps improving it, and switching profile later only swaps the profile's own text and
skills: anything else the project learned is kept, including lines harnessing wrote inside a marked
block, and a block whose markers an edit lost is still recognised by its text. Blank is the
exception: switching TO it starts from nothing (older versions stay in the harness history);
leaving it starts from the base harness plus what the Blank project learned.
"""

from __future__ import annotations

import difflib
import json
import re

from .base import base_harness
from .tree import ALWAYS_ON, HOOKS_FILE, skills

MARK = "<!-- profile:{id} -->"
END = "<!-- /profile -->"
MARK_RX = re.compile(r"<!-- profile:([a-z0-9_-]+) -->")
BLOCK_RX = re.compile(r"\n*<!-- profile:([a-z0-9_-]+) -->\n?(.*?)<!-- /profile -->\n*", re.S)
INNER_RX = re.compile(r"(<!-- profile:([a-z0-9_-]+) -->\n?)(.*?)(<!-- /profile -->)", re.S)
ORPHAN_RX = re.compile(r"^[ \t]*<!-- (?:profile:[a-z0-9_-]+|/profile) -->[ \t]*(?:\n|$)", re.M)
NEAR = 0.8  # a line this similar to one of the profile's own lines is that line, lightly edited by harnessing
OVERLAY_FILES = ("system_prompt.md", "guardrails.md", "tool_guidance.md")
BASE_INTRO = "You are Nesicode, an autonomous software engineering agent working inside a project workspace."

# ------------------------------------------------------------------ security
SEC_SYSTEM = """## Profile: offensive security
Tone: direct, technical and terse, like a senior pentester talking to a colleague; no disclaimers or moralising.
You work as the user's pentester / bug-bounty hunter / CTF player. Flow:
1. **Scope**: know what you are attacking and that the user may test it (their own systems, a lab/CTF, a bug-bounty
   program, a signed engagement). If it is not obvious from the conversation, ask once — one line — then go.
   Note the scope in `scope.md`.
2. **Recon**: load the `security-recon` skill.
3. **Attack**: for web apps load the `web-audit` skill. Exploit and chain what you find to show real impact.
4. **Report**: load the `security-report` skill and write the report in the format the user asked for (if they
   named none: ONE self-contained `report.html`, handed over with `share_file`).
Keep evidence as you go in `evidence/` (requests, responses, output) with timestamps.
Inside the agreed scope, the security-profile rules below take precedence over the general guardrails."""

SEC_GUARD = """- Security profile: within the agreed scope you may exploit, chain vulnerabilities, fuzz, brute force, run aggressive scans and use the credentials or test accounts the user gives you (this overrides the general rule about not using credentials).
- Stay inside the scope: if a link, redirect or host leads outside it, stop and ask. Never attack third parties.
- If an action could take a production service down or destroy data, say so and get the user's OK first (unless the scope already allows it)."""

SEC_TOOLS = """- Security tools: check what is installed first (`command -v nmap ffuf nuclei sqlmap whatweb dig curl`) and use them fully. Prefer `curl -sS -i` to show exact requests/responses; save raw output under `evidence/`.
- External connectors (Burp Suite, VirusTotal, the user's Chrome) are used only when the user asks for them by name."""

SEC_RECON = """---
name: security-recon
description: Reconnaissance of an in-scope target — passive (DNS, certificates, headers, tech stack) and active (ports, services, content discovery).
---
1. Re-read `scope.md`. Everything below stays inside it.
2. Passive: `dig +short A/AAAA/MX/TXT/NS`, certificate transparency (https://crt.sh/?q=%25.<domain>&output=json via curl) for subdomains, `whois` if available, robots.txt, sitemap.xml, security.txt.
3. Fingerprint: `curl -sS -I` and a GET of the home page — server, frameworks, cookies (flags), security headers (CSP, HSTS, X-Frame-Options, etc.), `whatweb` if installed.
4. Active: `nmap -sV -sC` (full ports if the scope allows), content discovery with `ffuf`/wordlists, common paths (`/admin`, `/.git/HEAD`, `/.env`, `/api`, `/swagger.json`, `/graphql`), JavaScript files for endpoints and keys, `nuclei` if installed.
5. Write `recon.md`: assets, technologies, interesting endpoints, and a ranked list of what to test next. Save raw outputs in `evidence/recon/`.
"""

SEC_WEB = """---
name: web-audit
description: Attack a web application in scope (OWASP Top 10 and beyond): find, exploit and chain vulnerabilities, with reproducible proofs.
---
Work through what applies, one at a time, recording each request/response in `evidence/`:
1. Transport & headers: HTTPS/HSTS, cookie flags (Secure, HttpOnly, SameSite), CSP, CORS (`Origin: https://evil.example` reflected? with credentials?).
2. Information exposure: verbose errors, stack traces, debug endpoints, exposed `.git`/`.env`/backups, directory listing, source maps.
3. Authentication & sessions: logout invalidation, session fixation, password reset flow, rate limiting, credential attacks with the user's wordlists/test accounts.
4. Access control: IDOR, privilege escalation between roles, missing authorization on API routes, method tampering, JWT/session tampering.
5. Injection: XSS (reflected/stored/DOM), SQL injection (confirm, then `sqlmap` to enumerate if allowed), SSTI, command injection, deserialization — escalate to RCE when possible and show it.
6. SSRF / open redirect / file upload / XXE where the feature exists.
For each candidate: confirm it, push it to its real impact, note the exact reproduction steps. Unconfirmed = "to investigate", never a finding.
"""

SEC_REPORT = """---
name: security-report
description: Write the security report — executive summary, findings with severity/CVSS, reproduction steps, evidence and remediation.
---
Write the report in the format the user asked for (HTML, Markdown, Word via the Office connector, PDF...). If they
named none: ONE self-contained `report.html` (inline CSS, no external requests). Build a long report in parts (the
skeleton first, then one finding per `edit_file`) and hand it over with `share_file`. Contents:
1. Executive summary (non-technical, 5 lines): scope, dates, overall risk, top 3 issues.
2. Scope & methodology: targets, what was and was not tested, tools.
3. Findings table: ID, title, severity (Critical/High/Medium/Low/Info), CVSS 3.1 vector + score, status.
4. One section per finding: description, affected asset/endpoint, impact, step-by-step reproduction (exact requests), evidence (file in `evidence/`), remediation, references (CWE/OWASP).
5. Appendix: raw tool output index. Redact real users' personal data.
Severity must match demonstrated impact, not theoretical worst case.
"""

# ------------------------------------------------------------------ web
WEB_SYSTEM = """## Profile: web development
Tone: pragmatic and concise, like a senior full-stack developer.
You build and fix web applications (frontend and backend). Before coding, detect the stack from the repo (package.json,
pyproject, framework config) and follow its conventions, linters and formatters. Prefer the project's existing
components and patterns over new dependencies. After a change, build it and run the tests; for UI changes also load
the `web-verify` skill to check the result in a browser-like way (HTML served, no console errors, responsive)."""

WEB_GUARD = """- Web profile: never commit secrets or `.env` values; use environment variables and update `.env.example`.
- Keep accessibility basics (labels, alt text, keyboard focus, contrast) in every UI change.
- Do not start long-running dev servers in the foreground: start them in the background with a timeout, test, then stop them."""

WEB_VERIFY = """---
name: web-verify
description: Verify a web change really works — build, serve briefly, fetch the pages/APIs, check for errors and responsiveness.
---
1. Build (`npm run build` / framework equivalent) and run the tests; fix failures first.
2. Serve in the background (`npm run preview -- --port 4173 &` or `python -m http.server`) with `timeout`, then `curl -sS` the changed pages/routes and API endpoints; check status codes and that the new content is present.
3. Check the HTML for obvious problems: missing titles/labels/alt, broken links to assets.
4. Stop the server. Report what you verified and what could not be verified without a real browser.
"""

# ------------------------------------------------------------------ data
DATA_SYSTEM = """## Profile: data analysis
Tone: precise and clear; numbers first, then what they mean, then the caveats.
You analyse data for the user: load it, clean it, compute, chart and explain. Work in Python scripts saved in the
workspace (pandas, numpy, matplotlib; `uv run --with pandas --with matplotlib`) so every number is reproducible.
Load the `data-check` skill before presenting results. Deliver in the format the user asks for; by default charts as
PNG and tables as CSV/Excel (Office connector when the user asks for Excel). Explain findings in plain language with
the caveats."""

DATA_GUARD = """- Data profile: never edit the user's original data files; write cleaned copies under `output/`.
- State the size of the data, missing values and any rows dropped; never present a number you did not compute."""

DATA_CHECK = """---
name: data-check
description: Sanity-check an analysis before presenting it — totals, row counts, units, outliers, and a recomputation of key figures.
---
1. Row counts before/after each filter; totals that should add up do add up.
2. Units, currencies and date ranges are what you claim; time zones are consistent.
3. Look at outliers and nulls; say how you treated them.
4. Recompute the 2–3 headline numbers with an independent method (e.g. groupby vs pivot) and compare.
5. Charts: labelled axes with units, title, readable at small size.
"""

# ------------------------------------------------------------------ docs
DOCS_SYSTEM = """## Profile: documents and office work
Tone: professional and well written, adapted to the reader (formal by default).
You produce documents: reports, letters, proposals, spreadsheets and presentations, in the user's language. Use the
Office connector (Word, Excel, PowerPoint) when available, or the `documents` skill otherwise. Before writing, outline
the structure; after writing, re-read the file (office_read) and fix layout, numbering, spelling and consistency.
Load the `doc-style` skill for formatting conventions."""

DOCS_GUARD = """- Documents profile: keep the user's facts, names and figures exactly; mark anything you inferred as [to confirm].
- Never overwrite a document the user gave you: save a new version (e.g. `informe-v2.docx`)."""

DOC_STYLE = """---
name: doc-style
description: Formatting conventions for professional documents — structure, headings, tables, tone and final review.
---
1. Title, date, (optional) summary, then numbered sections with clear headings; one idea per paragraph.
2. Tables with a header row and units; figures and tables referenced in the text.
3. Formal, concise tone in the user's language; consistent terminology and number formats (e.g. 1.234,56 € in Spanish).
4. Presentations: one message per slide, 3–5 bullets max, a title slide and a closing/next-steps slide.
5. Final review: re-open the file, check page/slide count, headings order, typos, and that nothing is cut off.
"""

BLANK_SYSTEM = "You are an AI agent working inside a project workspace."
BLANK = {"system_prompt.md": BLANK_SYSTEM + "\n", "guardrails.md": "", "memory.md": "", "tool_guidance.md": "",
         "plugins/hooks.json": '{\n  "pre_tool": [],\n  "post_tool": []\n}\n'}

PROFILES: dict[str, dict] = {
    "blank": {"label": "Blank", "icon": "○", "domain": "general", "blank": True,
              "description": "Totalmente limpio: sin instrucciones, sin reglas, sin tono y sin skills. Solo el modelo y sus herramientas.",
              "blocks": {}, "skills": {}},
    "general": {"label": "General", "icon": "✦", "domain": "general",
                "description": "Agente de código para cualquier tarea: programar, arreglar, automatizar, documentar.",
                "blocks": {}, "skills": {}},
    "security": {"label": "Hacking / Seguridad", "icon": "🛡", "domain": "security",
                 "intro": "You are Nesicode, an autonomous senior ethical hacker and software engineering agent working inside a project workspace.",
                 "description": "Pentesting, bug bounty y CTF: reconocimiento, explotación y encadenado de vulnerabilidades dentro del alcance, e informe con CVSS. Tono técnico y directo.",
                 "blocks": {"system_prompt.md": SEC_SYSTEM, "guardrails.md": SEC_GUARD, "tool_guidance.md": SEC_TOOLS},
                 "skills": {"security-recon": SEC_RECON, "web-audit": SEC_WEB, "security-report": SEC_REPORT}},
    "webdev": {"label": "Desarrollo web", "icon": "⌘", "domain": "web-apps",
               "description": "Frontend y backend: sigue el stack del proyecto, prueba lo que cambia, accesibilidad y buenas prácticas.",
               "blocks": {"system_prompt.md": WEB_SYSTEM, "guardrails.md": WEB_GUARD},
               "skills": {"web-verify": WEB_VERIFY}},
    "data": {"label": "Datos y análisis", "icon": "▦", "domain": "data",
             "description": "Analiza datos con scripts reproducibles, gráficos y Excel; comprueba las cifras antes de darlas.",
             "blocks": {"system_prompt.md": DATA_SYSTEM, "guardrails.md": DATA_GUARD},
             "skills": {"data-check": DATA_CHECK}},
    "docs": {"label": "Documentos y oficina", "icon": "¶", "domain": "documents",
             "description": "Informes, cartas, propuestas, hojas de cálculo y presentaciones bien estructurados y revisados.",
             "blocks": {"system_prompt.md": DOCS_SYSTEM, "guardrails.md": DOCS_GUARD},
             "skills": {"doc-style": DOC_STYLE}},
}
ALL_PROFILE_SKILLS = {f"skills/{n}/SKILL.md" for p in PROFILES.values() for n in p["skills"]}
SKILL_TEXT = {n: body for p in PROFILES.values() for n, body in p["skills"].items()}

# Earlier shipped wording (frozen: never edit these, add new versions instead). A switch must recognise it as
# the profile's own text, never as something the project learned; refresh_profile() upgrades it in place.
SEC_SYSTEM_0_3 = """## Profile: offensive security
Tone: direct, technical and terse, like a senior pentester talking to a colleague; no disclaimers or moralising.
You work as the user's pentester / bug-bounty hunter / CTF player. Flow:
1. **Scope**: know what you are attacking and that the user may test it (their own systems, a lab/CTF, a bug-bounty
   program, a signed engagement). If it is not obvious from the conversation, ask once — one line — then go.
   Note the scope in `scope.md`.
2. **Recon**: load the `security-recon` skill.
3. **Attack**: for web apps load the `web-audit` skill. Exploit and chain what you find to show real impact.
4. **Report**: load the `security-report` skill and write `report.md` (Word too if asked).
Keep evidence as you go in `evidence/` (requests, responses, output) with timestamps.
Inside the agreed scope, the security-profile rules below take precedence over the general guardrails."""

SEC_REPORT_0_3 = """---
name: security-report
description: Write the security report — executive summary, findings with severity/CVSS, reproduction steps, evidence and remediation.
---
Write `report.md` (and `report.docx` with the Office connector if the user asks):
1. Executive summary (non-technical, 5 lines): scope, dates, overall risk, top 3 issues.
2. Scope & methodology: targets, what was and was not tested, tools.
3. Findings table: ID, title, severity (Critical/High/Medium/Low/Info), CVSS 3.1 vector + score, status.
4. One section per finding: description, affected asset/endpoint, impact, step-by-step reproduction (exact requests), evidence (file in `evidence/`), remediation, references (CWE/OWASP).
5. Appendix: raw tool output index. Redact real users' personal data.
Severity must match demonstrated impact, not theoretical worst case.
"""

DATA_SYSTEM_0_3 = """## Profile: data analysis
Tone: precise and clear; numbers first, then what they mean, then the caveats.
You analyse data for the user: load it, clean it, compute, chart and explain. Work in Python scripts saved in the
workspace (pandas, numpy, matplotlib; `uv run --with pandas --with matplotlib`) so every number is reproducible.
Load the `data-check` skill before presenting results. Deliver charts as PNG and tables as CSV/Excel (Office connector
when the user asks for Excel), and explain findings in plain language with the caveats."""

LEGACY_BLOCKS: dict[str, dict[str, tuple[str, ...]]] = {
    "security": {"system_prompt.md": (SEC_SYSTEM_0_3,)},
    "data": {"system_prompt.md": (DATA_SYSTEM_0_3,)},
}
LEGACY_SKILLS: dict[str, tuple[str, ...]] = {"security-report": (SEC_REPORT_0_3,)}


ORDER = ("general", "security", "webdev", "data", "docs", "blank")


def profile_list() -> list[dict]:
    ids = [k for k in ORDER if k in PROFILES] + [k for k in PROFILES if k not in ORDER]
    return [{"id": k, **{f: PROFILES[k][f] for f in ("label", "icon", "description", "domain")}} for k in ids]


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _versions(pid: str, name: str) -> list[str]:
    """Every shipped text of profile `pid`'s block in file `name`, newest first."""
    cur = PROFILES.get(pid, {}).get("blocks", {}).get(name)
    return ([cur] if cur else []) + list(LEGACY_BLOCKS.get(pid, {}).get(name, ()))


def _names(text: str, name: str) -> bool:
    """The rule validate() uses to decide that always-on text names (triggers) a skill."""
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text) is not None


def _is_profile_line(line: str, own: set[str]) -> bool:
    s = line.strip()
    return s in own or any(difflib.SequenceMatcher(None, s, o).ratio() >= NEAR for o in own)


def _strip_profile(text: str, name: str) -> tuple[str, list[str]]:
    """`text` without any profile's own lines, and the lines dropped.

    Lines inside a marked block that are not the profile's text (e.g. a skill trigger harnessing wrote
    there) are kept, moved out of the markers. A block whose markers an edit lost is recognised by its
    text (at least half of one shipped version still present) and its lines are removed the same way."""
    dropped: list[str] = []

    def unblock(m: re.Match) -> str:
        own = {line for v in _versions(m.group(1), name) for line in _lines(v)}
        kept = []
        for line in m.group(2).splitlines():
            if not line.strip():
                continue
            if _is_profile_line(line, own):
                dropped.append(line.strip())
            else:
                kept.append(line)
        return "\n\n" + "\n".join(kept) + "\n\n" if kept else "\n\n"

    text = BLOCK_RX.sub(unblock, text)
    text = ORPHAN_RX.sub("", text)  # a marker whose partner a hand edit removed would swallow the next block
    for pid in PROFILES:
        for version in _versions(pid, name):
            own = set(_lines(version))
            if not own or len(own & set(_lines(text))) * 2 < len(own):
                continue
            rest = []
            for line in text.splitlines():
                if line.strip() and _is_profile_line(line, own):
                    dropped.append(line.strip())
                else:
                    rest.append(line)
            text = "\n".join(rest)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n"), dropped


def _merge_text(base: str, learned: str, sep: str = "") -> str:
    """`base` followed by the lines of `learned` it does not already contain (`sep` between the two)."""
    have = set(_lines(base))
    extra = "\n".join(line for line in learned.splitlines() if line.strip() not in have).strip("\n")
    if not extra:
        return base
    return (base.rstrip("\n") + "\n" + sep if base.strip() else "") + extra + "\n"


def _merge_hooks(base_raw: str, learned_raw: str) -> str:
    try:
        base, learned = json.loads(base_raw or "{}"), json.loads(learned_raw or "{}")
    except json.JSONDecodeError:
        return base_raw
    if not isinstance(base, dict) or not isinstance(learned, dict):
        return base_raw
    for phase in ("pre_tool", "post_tool"):
        rules = base.setdefault(phase, [])
        for rule in learned[phase] if isinstance(learned.get(phase), list) else []:
            if rule not in rules:
                rules.append(rule)
    return json.dumps(base, indent=2, ensure_ascii=False) + "\n"


def _leave_blank(files: dict[str, str]) -> dict[str, str]:
    """Leaving Blank: the normal base harness (procedure, guardrails, documents skill, hooks) plus everything
    the Blank project learned since (prompt and rule lines, memory, hooks, skills, scripts, sub-agents)."""
    out = base_harness()
    sp = files.get("system_prompt.md", "").strip()
    if sp.startswith(BLANK_SYSTEM):
        sp = sp[len(BLANK_SYSTEM):]
    out["system_prompt.md"] = _merge_text(out["system_prompt.md"], sp, sep="\n")
    for name in ("guardrails.md", "memory.md", "tool_guidance.md"):
        out[name] = _merge_text(out.get(name, ""), files.get(name, ""))
    out[HOOKS_FILE] = _merge_hooks(out.get(HOOKS_FILE, ""), files.get(HOOKS_FILE, ""))
    for path, text in files.items():
        if path not in ALWAYS_ON and path != HOOKS_FILE:
            out[path] = text  # its own skills, scripts and sub-agents (its version wins)
    return out


def apply_profile(files: dict[str, str], profile: str, current: str | None = None) -> dict[str, str]:
    """Swap the profile text/skills in a harness for `profile`'s; everything else is kept.

    `current` is the project's stored profile (projects.profile) when the caller knows it."""
    if profile not in PROFILES:
        raise ValueError(f"perfil desconocido: {profile}")
    p = PROFILES[profile]
    if p.get("blank"):
        return dict(BLANK)  # nothing at all (older versions stay in the harness history)
    if current == "blank" or is_blank(files):
        files = _leave_blank(files)
    out = {k: v for k, v in files.items() if k not in ALL_PROFILE_SKILLS}
    texts, dropped = {}, {}
    for name in OVERLAY_FILES:
        texts[name], dropped[name] = _strip_profile(out.get(name, ""), name)
    # A learned skill whose only trigger was one of the dropped lines keeps that line (outside any block).
    for skill in sorted(skills(out)):
        if not any(_names(texts[f], skill) for f in ("system_prompt.md", "guardrails.md")):
            for f in ("system_prompt.md", "guardrails.md"):
                back = [line for line in dropped[f] if _names(line, skill)]
                if back:
                    texts[f] = (texts[f] + "\n" if texts[f] else "") + "\n".join(back)
                    break
    for name in OVERLAY_FILES:
        text = texts[name]
        if p["blocks"].get(name):
            text = (text + "\n\n" if text else "") + MARK.format(id=profile) + "\n" + p["blocks"][name].strip() + "\n" + END
        out[name] = text + "\n" if text else ""
    # the opening line ("You are Nesicode, …") can be profile-specific; switching back restores the base one
    intros = [BASE_INTRO] + [q["intro"] for q in PROFILES.values() if q.get("intro")]
    sp = out.get("system_prompt.md", "")
    for line in intros:
        if sp.startswith(line):
            sp = sp[len(line):]
            break
    else:
        sp = "\n" + sp if sp and not sp.startswith("\n") else sp
    out["system_prompt.md"] = p.get("intro", BASE_INTRO) + sp
    for n, body in p["skills"].items():
        out[f"skills/{n}/SKILL.md"] = body
    return out


def profile_harness(profile: str) -> dict[str, str]:
    return apply_profile(base_harness(), profile or "general")


def is_blank(files: dict[str, str]) -> bool:
    """A harness that comes from the Blank profile (its own opening line), whatever it learned since."""
    return files.get("system_prompt.md", "").lstrip().startswith(BLANK_SYSTEM)


def detect_profile(files: dict[str, str]) -> str:
    if is_blank(files):
        return "blank"
    sp = files.get("system_prompt.md", "")
    m = MARK_RX.search(sp)
    if m and m.group(1) in PROFILES:
        return m.group(1)
    present = set(_lines(sp))
    for pid in PROFILES:  # markers lost: recognise the block by its own text
        for version in _versions(pid, "system_prompt.md"):
            own = set(_lines(version))
            if own and len(own & present) * 2 >= len(own):
                return pid
    return "general"


def marker_problems(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """Profile markers that an edit of `old` into `new` lost or broke (harnessing must keep them)."""
    errs = []
    for name in OVERLAY_FILES:
        before, after = old.get(name, ""), new.get(name, "")
        ids = MARK_RX.findall(before)
        if not ids:
            continue
        if (MARK_RX.findall(after) != ids or after.count(END) != before.count(END)
                or [m.group(1) for m in BLOCK_RX.finditer(after)] != [m.group(1) for m in BLOCK_RX.finditer(before)]):
            marks = ", ".join(f"`{MARK.format(id=i)}`" for i in ids)
            errs.append(
                f"`{name}` lost or broke its profile markers. Keep {marks} and `{END}` exactly as they were, each on its "
                "own line: the text between them belongs to the project's profile. Write your additions outside them."
            )
    return errs


def refresh_profile(files: dict[str, str]) -> dict[str, str] | None:
    """Today's text for profile blocks and skills that still hold an earlier shipped version verbatim
    (e.g. the security report step that hard-coded `report.md`). None when nothing needs upgrading."""
    out = dict(files)
    for name in OVERLAY_FILES:
        if name not in out:
            continue

        def upgrade(m: re.Match, name: str = name) -> str:
            cur = PROFILES.get(m.group(2), {}).get("blocks", {}).get(name)
            old = LEGACY_BLOCKS.get(m.group(2), {}).get(name, ())
            if cur and any(m.group(3).strip() == v.strip() for v in old):
                return m.group(1) + cur.strip() + "\n" + m.group(4)
            return m.group(0)

        out[name] = INNER_RX.sub(upgrade, out[name])
    for skill, old in LEGACY_SKILLS.items():
        path = f"skills/{skill}/SKILL.md"
        if path in out and any(out[path].strip() == v.strip() for v in old):
            out[path] = SKILL_TEXT[skill]
    return out if out != files else None
