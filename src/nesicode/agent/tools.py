"""The agent's tools. Every tool returns text; failures raise ToolError."""

from __future__ import annotations

import fnmatch
import os
import re
import time
from pathlib import Path
from typing import Callable

from ..harness import Harness, skills, sub_agents
from ..llm import ToolSpec
from .workspace import (
    IGNORE_DIRS, ToolError, Workspace, _walk, codec_note, decode_text, encode_text, shell_name, shown, text_codec,
)

MAX_READ_LINES = 2000
MAX_LIST = 400
MAX_TEXT_FILE = 20_000_000  # read_file loads files up to this whole (same decoding as edit_file); bigger ones stream
MAX_STREAM_LINE = 20_000  # characters shown of one line of a huge file (a one-line dump must not fill memory)
SEARCH_MAX_FILE = 2_000_000
SEARCH_BUDGET = 20.0  # seconds; a search over a huge tree returns what it found so far
SEARCH_WINDOW, SEARCH_OVERLAP = 4_000, 400  # long (minified) lines are matched in windows: bounded backtracking


def _str(args: dict, key: str, required: bool = True, default: str = "") -> str:
    v = args.get(key, default)
    if v is None:
        v = default
    if required and (not isinstance(v, str) or (v == "" and key not in ("content", "new"))):
        raise ToolError(f"missing or invalid `{key}` (string)")
    if not isinstance(v, str):
        raise ToolError(f"`{key}` must be a string")
    return v


def _int(args: dict, key: str, default: int) -> int:
    v = args.get(key)
    if v is None or v == "":  # models often send explicit nulls for optional parameters
        return default
    if isinstance(v, bool):
        raise ToolError(f"`{key}` must be an integer")
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ToolError(f"`{key}` must be an integer") from None


def _bool(args: dict, key: str) -> bool:
    """Strict: the string "false" must not count as true."""
    v = args.get(key)
    return v is True or v == 1 or (isinstance(v, str) and v.strip().lower() in ("true", "1", "yes"))


def list_dir(ws: Workspace, args: dict) -> str:
    base = ws.resolve(_str(args, "path", False, "."))
    if not base.is_dir():
        raise ToolError(f"{args.get('path')!r} is not a directory")
    depth = max(1, min(_int(args, "depth", 2), 5))
    lines: list[str] = []

    def walk(d: Path, level: int) -> None:
        pad = "  " * level
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name))
        except OSError as exc:
            lines.append(f"{pad}[unreadable: {exc.strerror or exc}]")
            return
        for e in entries:
            if len(lines) >= MAX_LIST:
                return
            try:
                if e.is_symlink():  # shown, never followed: it may point out of the workspace
                    try:
                        lines.append(f"{pad}{e.name} -> {os.readlink(e.path)}")
                    except OSError:
                        lines.append(f"{pad}{e.name} -> ?")
                elif e.is_dir(follow_symlinks=False):
                    if e.name in IGNORE_DIRS:
                        lines.append(f"{pad}{e.name}/ (skipped)")
                        continue
                    lines.append(f"{pad}{e.name}/")
                    if level + 1 < depth and ws.contains(e.path):
                        walk(Path(e.path), level + 1)
                else:
                    lines.append(f"{pad}{e.name} ({e.stat(follow_symlinks=False).st_size} B)")
            except OSError:
                lines.append(f"{pad}{e.name} (?)")

    walk(base, 0)
    if len(lines) >= MAX_LIST:
        lines.append(f"... (truncated at {MAX_LIST} entries; list a subdirectory)")
    return "\n".join(lines) or "(empty directory)"


def _lines(text: str) -> list[str]:
    """Split like grep/sed/editors: on \\n only (a trailing \\r is dropped; form feeds and U+2028 are not breaks)."""
    out = text.split("\n")
    if out and out[-1] == "":
        out.pop()
    return [line[:-1] if line.endswith("\r") else line for line in out]


def _numbered(lines, offset: int, total: int | None, note: str = "") -> str:
    chunk = list(lines)
    if total is not None and not chunk:
        return f"(no lines at offset {offset}: the file has {total} line{'s' if total != 1 else ''})" if total else "(empty file)"
    out = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(chunk, start=offset))
    if total is not None:
        rest = total - (offset - 1 + len(chunk))
        if rest > 0:
            out += f"\n... ({rest} more lines; use offset={offset + len(chunk)})"
    elif chunk:
        out += f"\n... (more lines may follow; use offset={offset + len(chunk)})"
    return out + (f"\n({note})" if note else "")


def read_file(ws: Workspace, args: dict) -> str:
    p = ws.resolve(_str(args, "path"))
    if not p.is_file():
        raise ToolError(f"file not found: {args.get('path')}")
    offset = max(1, _int(args, "offset", 1))
    limit = max(1, min(_int(args, "limit", MAX_READ_LINES), MAX_READ_LINES))
    if p.suffix.lower() == ".pdf":
        lines = _lines(pdf_text(p))
        return _numbered(lines[offset - 1 : offset - 1 + limit], offset, len(lines))
    with ws.open_file(p) as f:
        size = os.fstat(f.fileno()).st_size
        if size <= MAX_TEXT_FILE:
            try:
                text, codec = decode_text(f.read(MAX_TEXT_FILE + 1))
            except ValueError:
                raise ToolError("binary file; not shown (images: use view_image)") from None
            lines = _lines(shown(text))
            note = codec_note(codec) if codec[1] != "utf-8" or codec[2] != "strict" else ""
            return _numbered(lines[offset - 1 : offset - 1 + limit], offset, len(lines), note)
        return _read_big(f, offset, limit, size)


def _read_big(f, offset: int, limit: int, size: int) -> str:
    """A slice of a huge file without loading it: stream lines, cap each one, then count the rest."""
    head = f.read(1 << 16)
    try:
        bom, enc, _ = text_codec(head[: head.rfind(b"\n") + 1] if b"\n" in head else head)
    except ValueError:
        raise ToolError("binary file; not shown (images: use view_image)") from None
    if enc.startswith("utf-16"):
        raise ToolError(f"UTF-16 file of {size:,} bytes: too large to show; convert it to UTF-8 with a script")
    f.seek(len(bom))
    cap = MAX_STREAM_LINE * 4
    lines: list[str] = []
    n = 0
    while True:
        raw = f.readline(cap)
        if not raw:
            break
        cut = not raw.endswith(b"\n") and len(raw) >= cap
        tail = raw
        while not tail.endswith(b"\n"):  # skip the rest of an overlong line
            tail = f.readline(1 << 20)
            if not tail:
                break
        n += 1
        if len(lines) >= limit:
            break
        if n >= offset:
            line = raw.rstrip(b"\n").rstrip(b"\r").decode(enc, "replace")
            lines.append(line[:MAX_STREAM_LINE] + " … [line truncated]" if cut else line)
    # Count what is left in big chunks (bytes.count is fast and keeps memory flat).
    rest, budget, last = 0, 512 * 1024 * 1024, b"\n"
    while budget > 0:
        chunk = f.read(1 << 20)
        if not chunk:
            break
        rest += chunk.count(b"\n")
        budget -= len(chunk)
        last = chunk[-1:]
    total = None if budget <= 0 else n + rest + (last != b"\n")
    note = f"large file: {size:,} bytes; lines longer than {MAX_STREAM_LINE:,} characters are cut"
    return _numbered(lines, offset, total, note)


def pdf_text(p) -> str:
    """Text of a PDF, page by page (scanned PDFs without a text layer come back empty)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        pages = [f"--- page {i} ---\n{(pg.extract_text() or '').strip()}" for i, pg in enumerate(reader.pages[:300], 1)]
    except Exception as exc:
        raise ToolError(f"could not read PDF: {exc}") from exc
    text = "\n".join(pages)
    if not text.replace("--- page", "").strip(" -0123456789\n"):
        raise ToolError("this PDF has no text layer (scanned); ask the user for the text or an image of the page")
    return text


_LONE_LF = re.compile(r"(?<!\r)\n")


def _crlf_dominant(text: str) -> bool:
    crlf = text.count("\r\n")
    return crlf > 0 and crlf >= text.count("\n") - crlf


def write_file(ws: Workspace, args: dict) -> str:
    p = ws.resolve(_str(args, "path"))
    if "content" not in args or args.get("content") is None:  # never empty a file because the argument was lost
        raise ToolError("missing `content` (string); pass \"\" explicitly to write an empty file")
    content = _str(args, "content", required=False)
    existed = p.exists()
    data = content.encode("utf-8")  # a new file is written exactly as given (no newline translation)
    if existed and p.is_file():
        # Overwriting: keep the file's own conventions (CRLF, BOM, legacy encoding) the model cannot see.
        try:
            raw = ws.read_bytes(p, MAX_TEXT_FILE + 1)
            if len(raw) <= MAX_TEXT_FILE:
                old, codec = decode_text(raw)
                text = _LONE_LF.sub("\r\n", content) if _crlf_dominant(old) and "\r\n" not in content else content
                try:
                    data = encode_text(text, codec)
                except UnicodeEncodeError:
                    data = text.encode("utf-8")
        except (ValueError, ToolError):
            pass
    ws.write_bytes(p, data)
    return f"{'overwrote' if existed else 'created'} {ws.rel(p)} ({len(content)} chars)"


def edit_file(ws: Workspace, args: dict) -> str:
    p = ws.resolve(_str(args, "path"))
    if not p.is_file():
        raise ToolError(f"file not found: {args.get('path')}")
    old = _str(args, "old")
    new = _str(args, "new", required=False)
    replace_all = _bool(args, "replace_all")
    data = ws.read_bytes(p, MAX_TEXT_FILE + 1)
    if len(data) > MAX_TEXT_FILE:
        raise ToolError(f"file is over {MAX_TEXT_FILE // 1_000_000} MB; edit it with a script in bash")
    try:
        text, codec = decode_text(data)  # exact round trip: other bytes, the encoding and line endings are kept
    except ValueError:
        raise ToolError("binary file; edit_file only edits text files") from None
    count = text.count(old)
    if count == 0 and "\n" in old and "\r\n" in text:  # read_file shows lines without their \r
        old, new = _LONE_LF.sub("\r\n", old), _LONE_LF.sub("\r\n", new)
        count = text.count(old)
    elif count == 0 and "\r\n" in old and "\r\n" not in text:
        old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        count = text.count(old)
    elif "\r\n" not in new and _crlf_dominant(text):  # new lines in a CRLF file are CRLF too
        new = _LONE_LF.sub("\r\n", new)
    if count == 0:
        raise ToolError("`old` was not found; read the file and copy the text exactly (whitespace included)")
    if count > 1 and not replace_all:
        raise ToolError(f"`old` occurs {count} times; add surrounding context or set replace_all=true")
    text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    try:
        out = encode_text(text, codec)
    except UnicodeEncodeError:
        raise ToolError(f"this file is {codec[1]}, which cannot store some characters of `new`; "
                        "use characters that encoding has, or convert the file with a script") from None
    ws.write_bytes(p, out)
    return f"edited {ws.rel(p)} ({count if replace_all else 1} replacement(s))"


def _expand_braces(pattern: str, limit: int = 64) -> list[str]:
    i = pattern.find("{")
    if i < 0:
        return [pattern]
    depth = 0
    for j in range(i, len(pattern)):
        depth += {"{": 1, "}": -1}.get(pattern[j], 0)
        if depth == 0:
            break
    else:
        return [pattern]  # unbalanced: literal
    alts, cur, depth = [], "", 0
    for ch in pattern[i + 1 : j]:
        if ch == "," and depth == 0:
            alts.append(cur)
            cur = ""
            continue
        depth += {"{": 1, "}": -1}.get(ch, 0)
        cur += ch
    alts.append(cur)
    out: list[str] = []
    for a in alts:  # stop at the limit: `{a,b}` x 30 would otherwise expand 2^30 patterns before cutting
        if len(out) >= limit:
            break
        out.extend(_expand_braces(pattern[:i] + a + pattern[j + 1 :], limit - len(out)))
    return out[:limit]


def _globs(glob) -> list[str]:
    """ripgrep-style globs: `*.{ts,tsx}`, `**/*.py`, or several separated by commas."""
    if not isinstance(glob, str) or not glob.strip():
        return []
    parts, cur, depth = [], "", 0
    for ch in glob:
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        depth += {"{": 1, "}": -1}.get(ch, 0)
        cur += ch
    parts.append(cur)
    return [g for p in parts if p.strip() for g in _expand_braces(p.strip())]


def _glob_ok(rel: str, globs: list[str]) -> bool:
    rel = rel.replace(os.sep, "/")
    name = rel.rsplit("/", 1)[-1]
    for g in globs:
        cands = {g}
        if g.startswith("**/"):
            cands.add(g[3:])  # `**/x` also matches x at the top level
        if "/**/" in g:
            cands.add(g.replace("/**/", "/"))
        if any(fnmatch.fnmatch(rel, c) or fnmatch.fnmatch(name, c) for c in cands):
            return True
    return False


try:  # CPython's own regex parser, to spot patterns that backtrack exponentially (best effort)
    from re import _constants as _sre, _parser as _sre_parse
except ImportError:  # pragma: no cover - other interpreters
    _sre = _sre_parse = None


def _nested_repeat(pattern: str) -> bool:
    """True for a repetition of something made only of repetitions, followed by more pattern: `(a+)+$`,
    `(\\w+\\s?)*:`, `(\\s*\\w+)+x`. One short line can then take exponential time, and Python's re holds the
    GIL while it does, which freezes the whole server. `(\\w+\\.)+` (a mandatory `.` in each round) is fine."""
    if _sre_parse is None:
        return False
    try:
        tree = _sre_parse.parse(pattern)
    except Exception:
        return False
    backtracking = (_sre.MAX_REPEAT, _sre.MIN_REPEAT)

    def children(op, av):
        if op in backtracking or op == getattr(_sre, "POSSESSIVE_REPEAT", None):
            return [av[2]]
        if op == _sre.SUBPATTERN:
            return [av[-1]]
        if op == _sre.BRANCH:
            return list(av[1])
        if op in (_sre.ASSERT, _sre.ASSERT_NOT):
            return [av[1]]
        if op == getattr(_sre, "ATOMIC_GROUP", None):
            return [av]
        if op == _sre.GROUPREF_EXISTS:
            return [s for s in av[1:] if s is not None]
        return []

    def flexible(items) -> bool:  # every element can match different lengths (or nothing): no fixed anchor per round
        for op, av in items:
            if op in backtracking:
                if av[0] == av[1]:
                    return False
            elif op == _sre.SUBPATTERN:
                if not flexible(av[-1]):
                    return False
            elif op == _sre.BRANCH:
                if not any(flexible(b) for b in av[1]):
                    return False
            elif op not in (_sre.AT, _sre.ASSERT, _sre.ASSERT_NOT):
                return False
        return True

    def unbounded(items) -> bool:
        return any((op in backtracking and av[1] == _sre.MAXREPEAT) or any(unbounded(c) for c in children(op, av))
                   for op, av in items)

    def walk(items, followed: bool) -> bool:
        items = list(items)
        for i, (op, av) in enumerate(items):
            after = followed or i + 1 < len(items)
            if (op in backtracking and (av[1] == _sre.MAXREPEAT or av[1] > 4) and after
                    and flexible(av[2]) and unbounded(av[2])):
                return True
            inside = after or op in backtracking
            if any(walk(c, inside) for c in children(op, av)):
                return True
        return False

    return walk(tree, False)


def _search_regex(pattern: str):
    """re.search already tries every start position, so a leading `.*` only adds quadratic backtracking
    (`.*TODO.*` on a minified bundle froze the server): drop it, the set of matching lines is the same."""
    m = re.match(r"(?:\.\*\??)+", pattern)
    if m and pattern[m.end() : m.end() + 1] not in ("*", "+", "?", "{"):
        try:
            return re.compile(pattern[m.end() :])
        except re.error:
            pass
    return re.compile(pattern)


def _match(rx, line: str, deadline: float):
    if len(line) <= SEARCH_WINDOW:
        return rx.search(line)
    for start in range(0, len(line), SEARCH_WINDOW - SEARCH_OVERLAP):  # bounded work per call; the GIL is free between
        stop = min(len(line), start + SEARCH_WINDOW)
        m, tries = rx.search(line, start, stop), 0
        # A match that reaches the window's end may be an artifact: there `$`, `\b` and lookaheads see a fake end of
        # line (`foo$` "matched" in the middle of a minified line). Re-check it from its start with more of the line.
        while m is not None and stop < len(line) and m.end() == stop and tries < 3:
            v = rx.match(line, m.start(), min(len(line), m.start() + 2 * SEARCH_WINDOW))
            if v is not None:
                return v
            tries += 1
            m = rx.search(line, m.start() + 1, stop) if m.start() + 1 < stop else None
        if m is not None and (m.end() < stop or stop == len(line)):
            return m
        if time.monotonic() > deadline:
            return None
    return None


def search(ws: Workspace, args: dict) -> str:
    pattern = _str(args, "pattern")
    try:
        re.compile(pattern)
        rx = _search_regex(pattern)
    except re.error as exc:
        raise ToolError(f"invalid regex: {exc}") from None
    if _nested_repeat(pattern):
        raise ToolError("this pattern repeats a group made only of repetitions (like `(a+)+$` or `(\\w+\\s?)*x`), which "
                        "can take exponential time on a single line; simplify it (e.g. `a+`, `[\\w\\s]+x`)")
    base = ws.resolve(_str(args, "path", False, "."))
    globs = _globs(args.get("glob"))
    hits: list[str] = []
    big: list[str] = []
    stopped = ""
    cancel = getattr(ws, "cancel", None)
    deadline = time.monotonic() + SEARCH_BUDGET
    files = [base] if base.is_file() else _walk(base)
    for f in files:
        if cancel is not None and cancel.is_set():
            stopped = "[search stopped by the user]"
            break
        if time.monotonic() > deadline:
            stopped = f"[search stopped after {SEARCH_BUDGET:.0f} s: results are partial; narrow `path` or `glob`]"
            break
        rel = ws.rel(f)
        if globs and not _glob_ok(rel, globs):
            continue
        try:
            with ws.open_file(f) as fh:  # regular files only, never through a link
                if os.fstat(fh.fileno()).st_size > SEARCH_MAX_FILE:
                    big.append(rel)
                    continue
                data = fh.read(SEARCH_MAX_FILE + 1)
        except ToolError:
            continue
        try:
            text = shown(decode_text(data)[0])
        except ValueError:
            continue  # binary
        for i, line in enumerate(_lines(text), 1):
            m = _match(rx, line, deadline)
            if m:
                s = line.strip()
                if len(s) > 240:  # show the match, not the start of a minified line
                    at = max(0, m.start() - 80)
                    s = ("…" if at else "") + line[at : at + 240].strip() + "…"
                hits.append(f"{rel}:{i}: {s}")
                if len(hits) >= 200:
                    return "\n".join(hits) + "\n... (truncated at 200 matches)"
            if time.monotonic() > deadline:
                break
    out = "\n".join(hits) or "no matches"
    if big:
        out += (f"\n({len(big)} file(s) over {SEARCH_MAX_FILE // 1_000_000} MB were not searched: "
                + ", ".join(big[:5]) + (", …" if len(big) > 5 else "") + "; use bash grep on them)")
    return out + (f"\n{stopped}" if stopped else "")


# When a pipeline hides a failure (`pytest | tail` reports tail's 0), say which stage failed: an EXIT trap set
# BEFORE the command reads the last pipeline's PIPESTATUS when the shell ends and leaves its exit status alone.
# (Text appended AFTER the command was swallowed by an unterminated heredoc — the trap's code ended up inside the
# agent's file — and joined to a trailing `\`.) Only for commands with a `|`; 141 is SIGPIPE from `| head`.
PIPE_TRAP = (
    '__nesi_pipes() { local __s=("${PIPESTATUS[@]}") __i=0; { set +x; } 2>/dev/null; '
    'if [ "${#__s[@]}" -gt 1 ]; then while [ "$__i" -lt "${#__s[@]}" ]; do case "${__s[$__i]}" in 0|141) ;; '
    '*) echo "[pipeline exit codes: ${__s[*]}]"; break;; esac; __i=$((__i + 1)); done; fi; }; trap __nesi_pipes EXIT; '
)


def bash(ws: Workspace, args: dict) -> str:
    command = _str(args, "command")
    if "|" in command and shell_name() == "bash":
        command = PIPE_TRAP + command  # same line: bash's "line N" in error messages stays the agent's line N
    code, out = ws.run(command, _int(args, "timeout", 120))
    out = out if len(out) <= 30_000 else out[:15_000] + "\n... (output truncated) ...\n" + out[-15_000:]
    return f"exit code: {code}\n{out}"


SCHEMAS = {
    "list_dir": ToolSpec(
        "list_dir", "List files and directories (dependency/VCS dirs are skipped; symlinks are shown, not followed).",
        {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory relative to the workspace root (default '.')."},
            "depth": {"type": "integer", "description": "Levels to descend, 1-5 (default 2)."},
        }},
    ),
    "read_file": ToolSpec(
        "read_file", "Read a text file with line numbers.",
        {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-based first line (default 1)."},
            "limit": {"type": "integer", "description": "Max lines (default 2000)."},
        }, "required": ["path"]},
    ),
    "write_file": ToolSpec(
        "write_file", "Create or overwrite a file with the given content.",
        {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
         "required": ["path", "content"]},
    ),
    "edit_file": ToolSpec(
        "edit_file", "Replace an exact string in a file. `old` must match exactly and be unique unless replace_all. "
        "The file's encoding and line endings are kept.",
        {"type": "object", "properties": {
            "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"},
            "replace_all": {"type": "boolean"},
        }, "required": ["path", "old", "new"]},
    ),
    "search": ToolSpec(
        "search", "Regex search over file contents. Returns path:line: text.",
        {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Python regular expression."},
            "path": {"type": "string", "description": "File or directory (default '.')."},
            "glob": {"type": "string", "description": "Optional filename filter, e.g. '*.py', '*.{ts,tsx}', '**/test_*.py'."},
        }, "required": ["pattern"]},
    ),
    "bash": ToolSpec(
        "bash", "Run a shell command from the workspace root. Returns exit code and combined output. "
        "Long output is trimmed automatically (start and end kept): don't pipe into head/tail just to shorten it. "
        "Processes left in the background (`cmd > log.txt 2>&1 &`) keep running until the run ends or the user "
        "presses Stop."
        + (" On this machine the shell is Windows PowerShell (not bash): use PowerShell syntax."
           if shell_name() == "powershell" else ""),
        {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "description": "Seconds, default 120, max 900."},
        }, "required": ["command"]},
    ),
}

BASIC_TOOLS: dict[str, Callable[[Workspace, dict], str]] = {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "search": search,
    "bash": bash,
}


def skill_tool(h: Harness) -> ToolSpec | None:
    names = sorted(skills(h))
    if not names:
        return None
    return ToolSpec(
        "load_skill", "Load the full instructions of a harness skill by name.",
        {"type": "object", "properties": {"name": {"type": "string", "enum": names}}, "required": ["name"]},
    )


def delegate_tool(h: Harness) -> ToolSpec | None:
    names = sorted(sub_agents(h))
    if not names:
        return None
    return ToolSpec(
        "delegate",
        "Hand a bounded sub-task to a subagent. It works in the same workspace with its own context "
        "and returns only its final answer.",
        {"type": "object", "properties": {
            "agent": {"type": "string", "enum": names},
            "task": {"type": "string", "description": "Self-contained instructions; the subagent cannot see this conversation."},
        }, "required": ["agent", "task"]},
    )


VIEW_IMAGE = ToolSpec(
    "view_image",
    "Look at an image file in the workspace (png, jpeg, gif, webp) through the vision model and get a "
    "text answer. Ask a specific question for best results.",
    {"type": "object", "properties": {
        "path": {"type": "string"},
        "question": {"type": "string", "description": "What you need to know about the image."},
    }, "required": ["path"]},
)


GENERATE_IMAGE = ToolSpec(
    "generate_image",
    "Create a new image (photo, illustration, logo, artwork) from a text description with the image "
    "model, save it in the workspace and show it to the user. For charts, diagrams or exact layouts "
    "prefer drawing with code and then `show_image`.",
    {"type": "object", "properties": {
        "prompt": {"type": "string", "description": "Detailed description of the image to create."},
        "path": {"type": "string", "description": "Optional output path without extension, e.g. 'assets/logo'."},
    }, "required": ["prompt"]},
)

SHOW_IMAGE = ToolSpec(
    "show_image",
    "Show an image file from the workspace to the user in the chat (e.g. a chart or diagram you just "
    "rendered with code).",
    {"type": "object", "properties": {
        "path": {"type": "string"},
        "caption": {"type": "string"},
    }, "required": ["path"]},
)


SHARE_FILE = ToolSpec(
    "share_file",
    "Give the user a file you created (presentation, spreadsheet, document, PDF, chart, zip...) as a "
    "download card in the chat. Use it for every deliverable file the user asked for. "
    "HTML files (.html) are shown as a live page right inside the chat: when the user asks for a report, a summary, "
    "a dashboard, a table, a landing or anything visual without naming a file format, write ONE self-contained "
    "HTML file (inline CSS and JS, data embedded; it cannot fetch anything at runtime — libraries only via <script src> "
    "from cdn.jsdelivr.net, cdnjs.cloudflare.com or unpkg.com) with a clean, readable design, then share_file it.",
    {"type": "object", "properties": {
        "path": {"type": "string"},
        "caption": {"type": "string", "description": "One line about what the file is."},
    }, "required": ["path"]},
)


SEARCH_HISTORY = ToolSpec(
    "search_history",
    "Search this conversation's archive (earlier steps, compaction checkpoints, full tool outputs saved on disk) "
    "with a regex or keywords. Use it whenever a detail was compacted away instead of guessing or redoing work.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
