"""The agent's tools. Every tool returns text; failures raise ToolError."""

from __future__ import annotations

import bisect
import difflib
import fnmatch
import itertools
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Callable

from ..harness import Harness, skills, sub_agents
from ..llm import ToolSpec
from .workspace import (
    ATTACHMENTS_DIR, IGNORE_DIRS, OUTSIDE_HINT, ToolError, Workspace, codec_note, decode_text, encode_text, link_text,
    sandbox_mode, shell_name, shown, spellings, text_codec,
)

MAX_READ_LINES = 2000
MAX_LIST = 400
MAX_TEXT_FILE = 20_000_000  # read_file loads files up to this whole (same decoding as edit_file); bigger ones stream
MAX_STREAM_LINE = 20_000  # characters shown of one line of a huge file (a one-line dump must not fill memory)
SEARCH_MAX_FILE = 2_000_000
SEARCH_BUDGET = 20.0  # seconds; a search over a huge tree returns what it found so far
SEARCH_WINDOW, SEARCH_OVERLAP = 4_000, 400  # long (minified) lines are matched in windows: bounded backtracking
SEARCH_HITS = 200


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


def _not_found(ws: Workspace, where: str, p: Path, what: str = "path not found") -> str:
    """'<what>: x', with the closest names in the nearest folder that exists (a guessed src/utils for src/util)."""
    d, missing = p, p.name
    while d != ws.root and not d.is_dir():
        d, missing = d.parent, d.name
    close: list[str] = []
    if missing and d != p:
        try:
            fd = ws.open_dir(d)  # (never listed through a link)
            try:
                with os.scandir(d if fd is None else fd) as it:
                    names = [e.name for e in itertools.islice(it, 5000)]
            finally:
                if fd is not None:
                    os.close(fd)
            close = difflib.get_close_matches(missing, names, n=3, cutoff=0.6)
        except (OSError, ToolError):
            pass
    hint = " (did you mean " + " or ".join(f"`{ws.rel(d / c)}`" for c in close) + "?)" if close else ""
    return f"{what}: {where}{hint}"


def _gitignore(ws: Workspace):
    """The project's top-level .gitignore as a test (relative path, is a folder) -> ignored, or None. Its common forms:
    `*.log`, `build/`, `/dist`, `docs/_build/`, `**/tmp`, `!keep.log` (the last rule that matches decides). Nested
    .gitignore files are not read; a folder it ignores is never entered, so nothing below it comes back."""
    try:
        raw = ws.read_bytes(ws.root / ".gitignore", 256_000)
    except ToolError:
        return None
    rules = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        line = line[1:] if negate else line
        if line.startswith("\\"):  # `\#x`, `\!x`: a name that starts with # or !
            line = line[1:]
        folders_only = line.endswith("/")
        line = line.rstrip("/")
        if not line:
            continue
        anchored = "/" in line  # a slash at the start or in the middle: relative to the root, else at any depth
        body = _gitignore_rx(line.lstrip("/"))
        try:
            rules.append((re.compile(body if anchored else "(?:.*/)?" + body), negate, folders_only))
        except re.error:
            continue
    if not rules:
        return None

    def ignored(rel: str, is_dir: bool) -> bool:
        verdict = False
        for rx, negate, folders_only in rules:
            if (is_dir or not folders_only) and rx.fullmatch(rel):
                verdict = not negate
        return verdict

    return ignored


def _gitignore_rx(glob: str) -> str:
    out, i = [], 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif glob[i] == "*":
            out.append("[^/]*")
            i += 1
        elif glob[i] == "?":
            out.append("[^/]")
            i += 1
        elif glob[i] == "[" and (j := glob.find("]", i + 2)) > 0:
            body = glob[i + 1 : j]
            out.append("[" + ("^" + body[1:] if body[:1] in "!^" else body).replace("\\", "\\\\") + "]")
            i = j + 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    return "".join(out)


def _gitignore_below(ws: Workspace, base: Path):
    """The .gitignore test for listing or searching `base`, or None when `base` is itself ignored: a folder the agent
    names on purpose is shown whole."""
    ignored = _gitignore(ws)
    if ignored is None or base == ws.root:
        return ignored
    parts = ws.rel(base).split("/")
    for k in range(1, len(parts) + 1):
        if ignored("/".join(parts[:k]), k < len(parts) or base.is_dir()):
            return None
    return ignored


def _dir_entries(ws: Workspace, d: Path, ignored, top: bool = False) -> list[tuple[str, str, str]] | str:
    """(name, kind, text) of a folder's entries, folders first, then by name; kind is "dir" for a folder to expand,
    "" for anything else. Or the reason it can't be read (`top`: a ToolError). Nothing is followed: on POSIX each folder
    is opened from the root one level at a time without following a link, so one swapped for a link mid-walk is
    refused."""
    try:
        fd = ws.open_dir(d)
    except ToolError as exc:
        if top:
            raise
        return f"[unreadable: {exc}]"
    try:
        try:
            with os.scandir(d if fd is None else fd) as it:
                entries = sorted(it, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name))
        except OSError as exc:
            return f"[unreadable: {exc.strerror or exc}]"
        out = []
        for e in entries:
            try:
                if e.is_symlink():  # shown, never followed: it may point out of the workspace
                    try:
                        target = link_text(os.readlink(e.name, dir_fd=fd) if fd is not None else os.readlink(e.path))
                    except OSError:
                        target = "?"
                    out.append((e.name, "", f"{e.name} -> {target}"))
                elif e.is_dir(follow_symlinks=False):
                    if e.name in IGNORE_DIRS:
                        out.append((e.name, "", f"{e.name}/ (skipped)"))
                    elif ignored is not None and ignored(ws.rel(d / e.name), True):
                        out.append((e.name, "", f"{e.name}/ (in .gitignore, not expanded)"))
                    else:  # (Windows, no dir fds: a junction out of the workspace is listed, never entered)
                        out.append((e.name, "dir" if fd is not None or ws.contains(e.path) else "", f"{e.name}/"))
                else:
                    out.append((e.name, "", f"{e.name} ({e.stat(follow_symlinks=False).st_size} B)"))
            except OSError:
                out.append((e.name, "", f"{e.name} (?)"))
        return out
    finally:
        if fd is not None:
            os.close(fd)


def list_dir(ws: Workspace, args: dict) -> str:
    where = _str(args, "path", False, ".")
    base = ws.resolve(where)
    if not base.is_dir():
        if base.exists():
            raise ToolError(f"{where!r} is a file, not a folder: read it with read_file")
        raise ToolError(_not_found(ws, where, base, "folder not found"))
    depth = max(1, min(_int(args, "depth", 2), 5))
    ignored = _gitignore_below(ws, base)
    # Breadth-first within MAX_LIST entries: a whole level is listed before anything below it, so one big folder that
    # sorts early (assets/ with 450 icons) can't hide the rest of the project; a folder that no longer fits is named.
    listed: dict[Path, list | str] = {}
    folded: dict[Path, int] = {}
    count, cut, queue = 0, False, deque([(base, 0)])
    while queue:
        d, level = queue.popleft()
        entries = _dir_entries(ws, d, ignored, top=d == base)
        size = 1 if isinstance(entries, str) else len(entries)
        if d != base and count + size > MAX_LIST:
            folded[d] = size
            continue
        if not isinstance(entries, str) and len(entries) > MAX_LIST:
            entries, cut = entries[:MAX_LIST], True
        listed[d] = entries
        count += size
        if level + 1 < depth and not isinstance(entries, str):
            queue.extend((d / name, level + 1) for name, kind, _ in entries if kind == "dir")
    lines: list[str] = []

    def render(d: Path, level: int) -> None:
        entries, pad = listed[d], "  " * level
        if isinstance(entries, str):
            lines.append(pad + entries)
            return
        for name, _, text in entries:
            child = d / name
            if child in folded:
                lines.append(f"{pad}{text} ({folded[child]} entries, not expanded)")
                continue
            lines.append(pad + text)
            if child in listed:
                render(child, level + 1)

    render(base, 0)
    if cut:
        lines.append(f"... (truncated at {MAX_LIST} entries; list a subdirectory)")
    if folded:
        names = [ws.rel(d) + "/" for d in folded]
        lines.append(f"... ({len(names)} folder(s) not expanded to stay within {MAX_LIST} entries: "
                     + ", ".join(names[:8]) + (", …" if len(names) > 8 else "") + "; list_dir one of them to see it)")
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


def _binary(rel: str, size: int) -> str:
    """read_file's answer for a file that is not text (view_image is only offered with a vision model)."""
    ext = os.path.splitext(rel)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        return (f"{rel} is an image ({size:,} bytes): read_file shows text only. Look at it with view_image if "
                "that tool is in your list; without it you cannot see images, so say so instead of guessing what it "
                "shows")
    if ext in (".docx", ".xlsx", ".pptx"):
        return (f"{rel} is an Office document ({size:,} bytes), not plain text: read it with the office tools if you "
                "have them, or with a script in bash (python-docx, openpyxl, python-pptx)")
    return (f"{rel} is a binary file ({size:,} bytes), not text: read_file shows text only; inspect it with a command "
            "in bash (`file`, `xxd | head`, or a script)")


def read_file(ws: Workspace, args: dict) -> str:
    where = _str(args, "path")
    p = ws.resolve(where)
    if not p.is_file():
        if p.is_dir():
            raise ToolError(f"{where!r} is a folder, not a file: list_dir shows what is in it")
        raise ToolError(_not_found(ws, where, p, "file not found"))
    offset = max(1, _int(args, "offset", 1))
    limit = max(1, min(_int(args, "limit", MAX_READ_LINES), MAX_READ_LINES))
    with ws.open_file(p) as f:  # (the PDF too: never through a link swapped in after resolve())
        st = os.fstat(f.fileno())
        _stamp(ws, p, st)
        if p.suffix.lower() == ".pdf":
            lines = _lines(pdf_text(f))
            return _numbered(lines[offset - 1 : offset - 1 + limit], offset, len(lines))
        if st.st_size <= MAX_TEXT_FILE:
            try:
                text, codec = decode_text(f.read(MAX_TEXT_FILE + 1))
            except ValueError:
                raise ToolError(_binary(ws.rel(p), st.st_size)) from None
            lines = _lines(shown(text))
            note = codec_note(codec) if codec[1] != "utf-8" or codec[2] != "strict" else ""
            return _numbered(lines[offset - 1 : offset - 1 + limit], offset, len(lines), note)
        return _read_big(f, offset, limit, st.st_size, ws.rel(p))


def _read_big(f, offset: int, limit: int, size: int, rel: str = "this file") -> str:
    """A slice of a huge file without loading it: stream lines, cap each one, then count the rest."""
    head = f.read(1 << 16)
    try:
        bom, enc, _ = text_codec(head[: head.rfind(b"\n") + 1] if b"\n" in head else head)
    except ValueError:
        raise ToolError(_binary(rel, size)) from None
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


def pdf_text(src) -> str:
    """Text of a PDF (a path, or a binary file already opened safely), page by page (scanned PDFs without a text layer
    come back empty)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(src if hasattr(src, "read") else str(src))
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


def _seen(ws) -> dict:
    """path -> (size, mtime_ns) of each file this run's tools read or wrote, as it was right after (see write_file).
    Kept on the run's Workspace, so a delegated helper shares it."""
    d = getattr(ws, "__dict__", None)
    return d.setdefault("_tool_seen", {}) if d is not None else {}


def _stamp(ws, p: Path, st: os.stat_result | None = None) -> None:
    try:
        st = st or os.stat(p, follow_symlinks=False)
    except OSError:
        return
    _seen(ws)[str(p)] = (st.st_size, st.st_mtime_ns)


def _changed_since_seen(ws, p: Path, st: os.stat_result) -> bool | None:
    """None when this run never read or wrote p, else whether it changed on disk since (a command, the user…)."""
    seen = _seen(ws).get(str(p))
    return None if seen is None else seen != (st.st_size, st.st_mtime_ns)


def write_file(ws: Workspace, args: dict) -> str:
    p = ws.resolve(_str(args, "path"))
    if "content" not in args or args.get("content") is None:  # never empty a file because the argument was lost
        raise ToolError("missing `content` (string); pass \"\" explicitly to write an empty file")
    content = _str(args, "content", required=False)
    existed = p.exists()
    data = content.encode("utf-8")  # a new file is written exactly as given (no newline translation)
    if existed and p.is_file():
        try:
            with ws.open_file(p) as f:
                st = os.fstat(f.fileno())
                raw = f.read(MAX_TEXT_FILE + 1)
        except ToolError:
            raw = b""
        old = None
        if raw and len(raw) <= MAX_TEXT_FILE:
            try:
                old, codec = decode_text(raw)
            except ValueError:
                raw = b""  # binary: there was no text the agent could have read
        if raw:
            _check_seen(ws, p, st, old)
        if old is not None:  # overwriting: keep the file's own conventions (CRLF, BOM, legacy encoding)
            text = _LONE_LF.sub("\r\n", content) if _crlf_dominant(old) and "\r\n" not in content else content
            try:
                data = encode_text(text, codec)
            except UnicodeEncodeError:
                data = text.encode("utf-8")
    ws.write_bytes(p, data)
    _stamp(ws, p)
    return f"{'overwrote' if existed else 'created'} {ws.rel(p)} ({len(content)} chars)"


def _check_seen(ws, p: Path, st: os.stat_result, old: str | None) -> None:
    """write_file replaces a whole file: like Claude Code, never one with content this run has not read (it would be
    lost unseen), nor one that changed on disk since this run last read or wrote it (what the agent knows is stale)."""
    changed = _changed_since_seen(ws, p, st)
    rel = ws.rel(p)
    if changed is None:
        size = f"{len(_lines(old)):,} lines" if old is not None else f"{st.st_size:,} bytes"
        raise ToolError(f"{rel} already exists ({size}) and you have not read it in this run: read_file it first, then "
                        "write it (write_file replaces the whole file; edit_file changes just a part)")
    if changed:
        raise ToolError(f"{rel} changed on disk since you last read or wrote it (a command, the user or another "
                        "process changed it): read_file it again before you replace it")


_EDIT_PARAMS = ("path", "old", "new", "replace_all")
_EDIT_ALIASES = {"old_string": "old", "new_string": "new"}  # Claude Code's names for the same parameters
_EDIT_USAGE = ("edit_file takes `path`, `old` (the exact text to replace), `new` (its replacement; \"\" deletes `old`) "
               "and optionally `replace_all`")


def _edit_args(args: dict) -> dict:
    """edit_file's arguments under their own names. A `new` that is missing or null must never read as "" (that deletes
    `old` and reports success), and an unknown name means the call has the tool wrong: both are refused, naming the
    right parameters."""
    out: dict = {}
    unknown = []
    for key, value in args.items():
        name = _EDIT_ALIASES.get(key, key)
        if name not in _EDIT_PARAMS:
            unknown.append(key)
        elif name in out and out[name] != value:
            raise ToolError(f"both `{name}` and `{key}` were given, with different values: {_EDIT_USAGE}")
        else:
            out[name] = value
    if unknown:
        raise ToolError(f"unknown parameter(s) {', '.join(f'`{k}`' for k in unknown)}: {_EDIT_USAGE}")
    if out.get("new") is None:
        raise ToolError(f"missing `new`: {_EDIT_USAGE}")
    return out


_NUMBERED = re.compile(r"^ *\d+\t")  # read_file's "     12<TAB>" in front of every line it shows


def _not_in_file(rel: str, text: str, old: str, stale: bool) -> str:
    """Why `old` was not found and what to do: the read_file prefix copied with it, whitespace that differs (shown with
    tabs visible), or the closest lines of the file."""
    lines = _lines(text)
    olds = _lines(old) or [old]
    written = [s for s in olds if s.strip()]
    first = written[0] if written else olds[0]
    out = [f"`old` was not found in {rel}" + (" (the file changed on disk since you last read it)" if stale else "")
           + f"; its first line: {first[:160]!r}."]
    if written and all(_NUMBERED.match(s) for s in written):
        bare = "\n".join(_NUMBERED.sub("", s, count=1) for s in olds)
        out.append("It starts with read_file's line-number prefix (spaces, the number, a tab), which is not part of "
                   "the file: leave it out of `old` and `new`"
                   + (" (without it, the text is in the file)." if bare in text else "."))
    else:
        flat = "\n".join(" ".join(s.split()) for s in lines)
        want = "\n".join(" ".join(s.split()) for s in olds).strip()
        at = flat.find(want) if want else -1
        if at >= 0:
            a = flat.count("\n", 0, at)
            b = a + want.count("\n")
            near = "\n".join(f"  {i + 1}: {lines[i][:200]!r}" for i in range(a, min(b + 1, a + 6)))
            out.append(f"Lines {a + 1}-{b + 1} match it except for whitespace (indentation, tabs, spaces at the ends); "
                       f"the file has (tabs shown as \\t):\n{near}")
        elif first.strip():
            # Score only lines with one of its rarest words that the file has (difflib on every line of a big,
            # repetitive file took seconds); a line with none of them is no "closest line" anyway.
            words = sorted((n, w) for w in set(re.findall(r"\w{3,}", first)) if (n := text.count(w)))
            rare = [w for _, w in words[:3]]
            near_lines = [i for i, s in enumerate(lines) if any(w in s for w in rare)] if rare else \
                range(len(lines) if len(lines) <= 2_000 else 0)
            sm = difflib.SequenceMatcher(None, autojunk=False)
            sm.set_seq2(first.strip())
            best = []
            for i in itertools.islice(near_lines, 2_000):
                sm.set_seq1(lines[i].strip())
                if sm.real_quick_ratio() >= 0.6 and sm.quick_ratio() >= 0.6 and sm.ratio() >= 0.6:
                    best.append((-sm.ratio(), i))
            if best:
                near = "\n".join(f"  {i + 1}: {lines[i][:200]!r}" for _, i in sorted(best)[:3])
                out.append(f"Closest lines in the file (tabs shown as \\t):\n{near}")
    out.append("Copy `old` exactly as the file has it now (read_file shows it; its \"   12<TAB>\" line-number prefix "
               "is not part of the file). If it keeps failing, use a shorter `old` (one or two distinctive lines) or, "
               "for a small file, write_file the whole new content.")
    return "\n".join(out)


def _lines_of(text: str, needle: str, limit: int = 8) -> list[int]:
    """Line numbers of the first `limit` occurrences of needle (non-overlapping, as str.count counts them)."""
    out, line, last = [], 1, 0
    at = text.find(needle)
    while at >= 0 and len(out) < limit:
        line += text.count("\n", last, at)
        last = at
        out.append(line)
        at = text.find(needle, at + max(1, len(needle)))
    return out


def edit_file(ws: Workspace, args: dict) -> str:
    args = _edit_args(args)
    where = _str(args, "path")
    p = ws.resolve(where)
    if not p.is_file():
        if p.is_dir():
            raise ToolError(f"{where!r} is a folder, not a file")
        raise ToolError(_not_found(ws, where, p, "file not found"))
    old = _str(args, "old")
    new = _str(args, "new", required=False)
    replace_all = _bool(args, "replace_all")
    with ws.open_file(p) as f:
        st = os.fstat(f.fileno())
        data = f.read(MAX_TEXT_FILE + 1)
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
    rel = ws.rel(p)
    stale = bool(_changed_since_seen(ws, p, st))
    if count == 0:
        raise ToolError(_not_in_file(rel, shown(text), old, stale))
    if count > 1 and not replace_all:
        at = _lines_of(text, old)
        raise ToolError(f"`old` occurs {count} times in {rel} (lines {', '.join(map(str, at))}"
                        f"{', …' if count > len(at) else ''}): add surrounding lines to make it unique, or set "
                        "replace_all=true")
    text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    try:
        out = encode_text(text, codec)
    except UnicodeEncodeError:
        raise ToolError(f"this file is {codec[1]}, which cannot store some characters of `new`; "
                        "use characters that encoding has, or convert the file with a script") from None
    ws.write_bytes(p, out)
    _stamp(ws, p)
    return f"edited {rel} ({count if replace_all else 1} replacement(s))" + (
        "\n(note: the file had changed on disk since you last read it — a command, the user or another process; `old` "
        "still matched, but read it again before relying on what you remember of it)" if stale else "")


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


def _glob_hit(rel: str, globs: list[str]) -> bool:
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


def _glob_ok(rel: str, globs: list[str]) -> bool:
    """ripgrep's rules: a file must match one of the globs, unless they are all exclusions; an exclusion (`!*.min.js`,
    `!vendor`, `!dist/**`) that matches the file or one of its folders always wins."""
    rel = rel.replace(os.sep, "/")
    keep = [g for g in globs if not g.startswith("!")]
    drop = [g[1:] for g in globs if g.startswith("!") and len(g) > 1]
    if drop:
        parts = rel.split("/")
        if any(_glob_hit("/".join(parts[:k]), drop) for k in range(1, len(parts) + 1)):
            return False
    return not keep or _glob_hit(rel, keep)


try:  # CPython's own regex parser, to judge what a pattern can cost before running it (best effort)
    from re import _constants as _sre, _parser as _sre_parse
except ImportError:  # pragma: no cover - other interpreters
    _sre = _sre_parse = None

# Characters to tell whether two pieces of a pattern can match the same one: all of Latin-1, a few others, and (see
# _probes) every character the pattern names
_PROBES = [chr(c) for c in range(256)] + [chr(c) for c in (0x101, 0x3B1, 0x436, 0x660, 0x2003, 0x2028, 0x4E2D, 0x1F600)]


def _children(op, av) -> list:
    if op in (_sre.MAX_REPEAT, _sre.MIN_REPEAT) or op == getattr(_sre, "POSSESSIVE_REPEAT", None):
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


def _probes(tree) -> list[str]:
    """_PROBES plus the pattern's own literals and range ends: two classes that share a character share one of these
    (`\\w+日` must not pass for `\\w+` stopping where `日` starts)."""
    named: set[str] = set()

    def walk(items) -> None:
        for op, av in items:
            if op in (_sre.LITERAL, _sre.NOT_LITERAL):
                named.add(chr(av))
            elif op == _sre.IN:
                for o, a in av:
                    named.update((chr(a),) if o == _sre.LITERAL else (chr(a[0]), chr(a[1])) if o == _sre.RANGE else ())
            for c in _children(op, av):
                walk(c)

    walk(tree)
    return _PROBES + sorted(named.difference(_PROBES))


def _char_set(op, av, flags: int, probes: list[str]) -> frozenset | None:
    """The probe characters one single-character node of a parsed pattern matches; None for any other node."""
    if op == _sre.LITERAL:
        cls = re.escape(chr(av))
    elif op == _sre.NOT_LITERAL:
        cls = "[^" + re.escape(chr(av)) + "]"
    elif op == _sre.ANY:
        cls = "."
    elif op == _sre.IN:
        names = {_sre.CATEGORY_DIGIT: r"\d", _sre.CATEGORY_NOT_DIGIT: r"\D", _sre.CATEGORY_SPACE: r"\s",
                 _sre.CATEGORY_NOT_SPACE: r"\S", _sre.CATEGORY_WORD: r"\w", _sre.CATEGORY_NOT_WORD: r"\W"}
        body = []
        for o, a in av:
            if o == _sre.NEGATE:
                body.append("^")
            elif o == _sre.LITERAL:
                body.append(re.escape(chr(a)))
            elif o == _sre.RANGE:
                body.append(re.escape(chr(a[0])) + "-" + re.escape(chr(a[1])))
            elif o == _sre.CATEGORY and a in names:
                body.append(names[a])
            else:
                return None
        cls = "[" + "".join(body) + "]"
    else:
        return None
    try:
        rx = re.compile(cls, flags & (re.IGNORECASE | re.DOTALL | re.ASCII))
    except re.error:
        return None
    return frozenset(c for c in probes if rx.fullmatch(c))


def _head(items, flags: int, probes: list[str]) -> tuple[frozenset | None, bool]:
    """(the probe characters a piece of a parsed pattern can start with, whether it can match nothing at all); the
    characters are None when that is not plain (a backreference, a conditional…). Zero-width parts are skipped."""
    chars: frozenset = frozenset()
    for op, av in items:
        if op in (_sre.AT, _sre.ASSERT, _sre.ASSERT_NOT):
            continue
        if op == _sre.SUBPATTERN:
            c, empty = _head(av[-1], flags, probes)
        elif op == _sre.BRANCH:
            heads = [_head(b, flags, probes) for b in av[1]]
            if any(h is None for h, _ in heads):
                return None, False
            c, empty = frozenset().union(*(h for h, _ in heads)), any(e for _, e in heads)
        elif op in (_sre.MAX_REPEAT, _sre.MIN_REPEAT) or op == getattr(_sre, "POSSESSIVE_REPEAT", None):
            c, empty = _head(av[2], flags, probes)
            empty = empty or av[0] == 0
        else:
            c, empty = _char_set(op, av, flags, probes), False
        if c is None:
            return None, False
        chars |= c
        if not empty:
            return chars, False
    return chars, True


def _nested_repeat(pattern: str) -> bool:
    """True for a repetition followed by more pattern whose rounds can split the same text in many ways: a group made
    only of repetitions (`(a+)+$`, `(\\w+\\s?)*:`, `(\\s*\\w+)+x`) or alternatives that can start with the same
    character (`(.|\\s)*x`, `(\\\\.|[^"])*"`). One short line can then take exponential time, and Python's re holds the
    GIL while it does, which freezes the whole server. `(\\w+\\.)+` (a mandatory `.` in each round) is fine."""
    if _sre_parse is None:
        return False
    try:
        tree = _sre_parse.parse(pattern)
    except Exception:
        return False
    backtracking = (_sre.MAX_REPEAT, _sre.MIN_REPEAT)
    flags, probes = tree.state.flags, _probes(tree)

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

    def overlapping(items) -> bool:  # a round is an alternation whose branches can start with the same character
        items = list(items)
        while len(items) == 1 and items[0][0] == _sre.SUBPATTERN:
            items = list(items[0][1][-1])
        if len(items) != 1 or items[0][0] != _sre.BRANCH:
            return False
        firsts = [h for h, _ in (_head(b, flags, probes) for b in items[0][1][1]) if h]
        return any(a & b for i, a in enumerate(firsts) for b in firsts[i + 1 :])

    def unbounded(items) -> bool:
        return any((op in backtracking and av[1] == _sre.MAXREPEAT) or any(unbounded(c) for c in _children(op, av))
                   for op, av in items)

    def walk(items, followed: bool) -> bool:
        items = list(items)
        for i, (op, av) in enumerate(items):
            after = followed or i + 1 < len(items)
            if (op in backtracking and (av[1] == _sre.MAXREPEAT or av[1] > 4) and after
                    and ((flexible(av[2]) and unbounded(av[2])) or overlapping(av[2]))):
                return True
            inside = after or op in backtracking
            if any(walk(c, inside) for c in _children(op, av)):
                return True
        return False

    return walk(tree, False)


CHEAP_WAYS = 8  # tries per start position (alternatives × optional parts) allowed next to one long repeat


def _cheap(pattern: str) -> bool:
    """True when `pattern` provably can't backtrack badly, so this process may run it (the GIL is free between lines
    and between the windows of a long line): literals, classes, anchors, alternatives and optional parts, and repeats
    of one character, of which at most one may give characters back to what follows it (`.*` in `def \\w+\\(.*\\):`);
    the others stop where what follows starts (`\\w+` before `\\(`), so a line costs at most about its length squared.
    Two such repeats (`foo.*bar.*baz` took 2 s on one 4,000-character line), a repeated group or a backreference are
    left to a child process."""
    if _sre_parse is None:
        return False
    try:
        tree = _sre_parse.parse(pattern)
    except Exception:
        return False
    flags, probes = tree.state.flags, _probes(tree)
    one = (_sre.LITERAL, _sre.NOT_LITERAL, _sre.ANY, _sre.IN)
    repeats = (_sre.MAX_REPEAT, _sre.MIN_REPEAT, getattr(_sre, "POSSESSIVE_REPEAT", None))

    def single(items) -> frozenset | None:  # the characters of a one-character body
        items = list(items)
        if len(items) != 1:
            return None
        op, av = items[0]
        return single(av[-1]) if op == _sre.SUBPATTERN else _char_set(op, av, flags, probes) if op in one else None

    def follow(rest, then):  # what can come right after an item: the start of `rest`, or `then` past its end
        chars, empty = _head(rest, flags, probes)
        return None if chars is None or (empty and then is None) else chars | then if empty else chars

    def cost(items, then, tainted: bool):
        """(tries, repeats that give characters back, long repeats) of a sequence followed by `then` (None: unknown);
        `tainted` once a repeat that gives characters back came before (a later one then multiplies its work)."""
        items = list(items)
        ways, longs, runs = 1, 0, 0
        for k, (op, av) in enumerate(items):
            if op in one or op == _sre.AT:
                continue
            after = follow(items[k + 1 :], then)
            if op == _sre.SUBPATTERN:
                c = cost(av[-1], after, tainted)
            elif op in (_sre.ASSERT, _sre.ASSERT_NOT):
                c = cost(av[1], None, tainted)
            elif op == _sre.BRANCH:
                alts = [cost(b, after, tainted) for b in av[1]]
                c = None if None in alts else (sum(a[0] for a in alts), max(a[1] for a in alts),
                                               max(a[2] for a in alts))
            elif op in repeats and (body := single(av[2])) is not None:
                if av[1] <= 8:
                    c = (av[1] - av[0] + 1, 0, 0)
                else:  # it gives nothing back when nothing that may follow can start with what it repeats
                    c = (1, 0 if not tainted and after is not None and not body & after else 1, 1)
            elif op in repeats and av[1] <= 1:  # an optional group
                c = cost(av[2], after, tainted)
                c = c and (c[0] + 1, c[1], c[2])
            else:
                return None
            if c is None:
                return None
            ways, longs, runs = ways * c[0], longs + c[1], runs + c[2]
            tainted = tainted or c[1] > 0
        return ways, longs, runs

    c = cost(tree, frozenset(), False)
    return c is not None and c[1] <= 1 and c[0] <= (CHEAP_WAYS if c[2] else 1024)


def _spans_lines(pattern: str) -> bool:
    """A pattern written to match across lines: it names a line break (`\\n`), sets DOTALL (`(?s)`), or uses a class of
    every character (`[\\s\\S]`, which `(.|\\s)` becomes): on a single line those could only ever mean `.`."""
    if _sre_parse is None:
        return "\\n" in pattern or "\n" in pattern
    try:
        tree = _sre_parse.parse(pattern)
    except Exception:
        return False
    if tree.state.flags & re.DOTALL:
        return True
    pairs = ((_sre.CATEGORY_SPACE, _sre.CATEGORY_NOT_SPACE), (_sre.CATEGORY_WORD, _sre.CATEGORY_NOT_WORD),
             (_sre.CATEGORY_DIGIT, _sre.CATEGORY_NOT_DIGIT))

    def walk(items) -> bool:
        for op, av in items:
            if op == _sre.LITERAL and av == 10:
                return True
            if op == _sre.IN and not (av and av[0][0] == _sre.NEGATE):
                cats = {a for o, a in av if o == _sre.CATEGORY}
                if (_sre.LITERAL, 10) in av or any(x in cats and y in cats for x, y in pairs):
                    return True
            if op == _sre.SUBPATTERN and av[1] & re.DOTALL:
                return True
            if any(walk(c) for c in _children(op, av)):
                return True
        return False

    return walk(tree)


_ANY_CHAR_GROUP = re.compile(r"\((\?:)?(?:\.\|\\[sn]|\\[sn]\|\.)\)")


def _any_char_groups(pattern: str) -> str:
    """`(.|\\s)` and `(.|\\n)`, "any character, line breaks too", as the class `([\\s\\S])`: the same matches and group
    numbers, but no alternation, which under a repeat backtracks exponentially when its branches overlap
    (`(.|\\s)*TODO` takes hours on one line with 30 spaces)."""
    out, i, in_class = [], 0, False
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            out.append(pattern[i : i + 2])
            i += 2
            continue
        if in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
            j = i + 1 + (pattern[i + 1 : i + 2] == "^")
            j += pattern[j : j + 1] == "]"  # a `]` right after `[` or `[^` is a literal
            out.append(pattern[i:j])
            i = j
            continue
        elif ch == "(" and (m := _ANY_CHAR_GROUP.match(pattern, i)):
            out.append("(" + (m.group(1) or "") + "[\\s\\S])")
            i = m.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out)


_INLINE_FLAGS = re.compile(r"\(\?[aiLmsux]+\)")
_ANY_RUN = re.compile(r"(?:\.|\[\\s\\S\]|\[\\S\\s\]|(\((?:\?:)?)\[\\s\\S\]\))\*\??")  # `.*`, `[\s\S]*`, `([\s\S])*?`


def _search_regex(pattern: str, flags: int = 0):
    """The model's pattern as `search` runs it, matching the same lines: `(.|\\s)`-style groups as a class (see
    _any_char_groups), and without a leading `.*` or `[\\s\\S]*` — re.search already tries every start position, so
    that only adds backtracking: quadratic on a line (`.*TODO.*` on a minified bundle froze the server), on the whole
    file once the pattern spans lines (`(.|\\s)*TODO`). A group dropped that way stays, empty, to keep the numbering."""
    rewritten = _any_char_groups(pattern)
    if rewritten != pattern:
        try:
            re.compile(rewritten, flags)
            pattern = rewritten
        except re.error:
            pass
    head = _INLINE_FLAGS.match(pattern)
    start = i = head.end() if head else 0
    groups = ""
    while m := _ANY_RUN.match(pattern, i):
        groups += "()" if m.group(1) == "(" else ""
        i = m.end()
    if i > start and pattern[i : i + 1] not in ("*", "+", "?", "{"):
        try:
            return re.compile(pattern[:start] + groups + pattern[i:], flags)
        except re.error:
            pass
    return re.compile(pattern, flags)


def _match(rx, line: str, deadline: float, cancel=None):
    """The first match of `rx` in `line`; None also once the deadline passes or `cancel` (Stop) is set between two
    windows of a long line."""
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
        if time.monotonic() > deadline or (cancel is not None and cancel.is_set()):
            return None
    return None


class _Matcher:
    """Runs one search's regex over each file within the search's deadline and the user's Stop. re holds the GIL
    while it matches, so only a pattern `_cheap` proves harmless runs in this process (line by line, long lines in
    windows); any other runs in a child process (nesicode.regex_worker) that is killed at the deadline or on Stop, so
    no pattern can freeze the server."""

    def __init__(self, rx: re.Pattern, multiline: bool, deadline: float, cancel=None):
        self.rx, self.multiline, self.deadline, self.cancel = rx, multiline, deadline, cancel
        self.local = not multiline and _cheap(rx.pattern)
        self.why = ""  # "stopped" (Stop) or "deadline" once the search must end
        self.slow = ""  # the file the child was still matching when the deadline came
        self._proc = self._conn = None

    def __enter__(self) -> _Matcher:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def over(self) -> bool:
        if not self.why:
            if self.cancel is not None and self.cancel.is_set():
                self.why = "stopped"
            elif time.monotonic() > self.deadline:
                self.why = "deadline"
        return bool(self.why)

    def find(self, lines: list[str], limit: int, rel: str = "") -> list[tuple[int, int, int, int]]:
        """(first line, last line, start in the first, end in the last) of up to `limit` matches in a file's lines
        (0-based); once `why` is set, what was found before."""
        if not lines or limit <= 0:
            return []
        if self.local:
            out = []
            for i, line in enumerate(lines):
                m = _match(self.rx, line, self.deadline, self.cancel)
                if m:
                    out.append((i, i, m.start(), m.end()))
                    if len(out) >= limit:
                        break
                if self.over():
                    break
            return out
        if not self.multiline:
            return [(i, i, a, b) for i, a, b in self._ask(("lines", lines, limit), rel)]
        starts = [0, *itertools.accumulate(len(s) + 1 for s in lines)]
        out = []
        for a, b in self._ask(("text", "\n".join(lines), limit), rel):
            first, last = bisect.bisect_right(starts, a) - 1, bisect.bisect_right(starts, max(a, b - 1)) - 1
            out.append((first, last, a - starts[first], b - starts[last]))
        return out

    def _ask(self, request: tuple, rel: str) -> list:
        if self.over():
            return []
        try:
            if self._proc is None:
                self._start()
                if self._reply("") is None:  # its "ready", or the end of the search
                    return []
            self._conn.send(request)
            reply = self._reply(rel)
        except (EOFError, OSError) as exc:
            self.close()
            raise ToolError(f"the search process ended unexpectedly ({type(exc).__name__}); try a simpler "
                            "pattern") from None
        if reply is None:
            return []
        if isinstance(reply, str):
            self.close()
            raise ToolError(f"the pattern failed on {rel}: {reply}")
        return reply

    def _reply(self, rel: str):
        """The child's next message; None once the search must end (the child is then killed)."""
        while not self._conn.poll(0.05):
            if self.over():
                self.slow = rel if self.why == "deadline" else ""
                self.close()
                return None
            if not self._proc.is_alive():
                break  # what it said last, or EOF, is read below
        return self._conn.recv()

    def _start(self) -> None:
        import multiprocessing

        from .. import regex_worker

        ctx = multiprocessing.get_context("spawn")  # (never fork a threaded server)
        self._conn, theirs = ctx.Pipe()
        try:
            self._proc = ctx.Process(target=regex_worker.serve, name="nesicode-search", daemon=True,
                                     args=(theirs, self.rx.pattern, self.rx.flags, SEARCH_BUDGET + 30))
            self._proc.start()
        except Exception as exc:
            self._proc = None
            self.close()
            raise ToolError(f"this pattern can only run in a separate process, which could not start ({exc}); "
                            "simplify it: literals, classes and one `.*` or `\\w+` run directly") from None
        finally:
            theirs.close()

    def close(self) -> None:
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        if proc is not None:
            proc.kill()
            proc.join(5)
            try:
                proc.close()
            except ValueError:  # (still running after the kill: let multiprocessing reap it)
                pass
        if conn is not None:
            conn.close()


_NOTED_SKIPS = {ATTACHMENTS_DIR, "node_modules", ".venv", "venv"}  # skipped folders a "no matches" names


def _files(ws: Workspace, base: Path, ignored, skipped: list):
    """The regular files under `base` in name order, as workspace._walk finds them (links never followed, the folders
    of IGNORE_DIRS skipped), minus what the project's .gitignore ignores. What a "no matches" should name (the user's
    attachments, dependencies, what .gitignore kept out) goes in `skipped` as (path, "" or "gitignore")."""
    stack = [base]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        subdirs = []
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    rel = ws.rel(Path(e.path))
                    if e.name in IGNORE_DIRS:
                        if e.name in _NOTED_SKIPS:
                            skipped.append((rel + "/", ""))
                    elif ignored is not None and ignored(rel, True):
                        skipped.append((rel + "/", "gitignore"))
                    else:
                        subdirs.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    p = Path(e.path)
                    if ignored is not None and ignored(ws.rel(p), False):
                        skipped.append((ws.rel(p), "gitignore"))
                    else:
                        yield p
            except OSError:
                continue
        stack.extend(reversed(subdirs))


def _skipped_note(skipped: list) -> str:
    """What a "no matches" did not look at: folders first, then files, a few of each kind named."""
    def names(paths: list[str]) -> str:
        paths = sorted(paths, key=lambda p: not p.endswith("/"))
        more = f" and {len(paths) - 6} more" if len(paths) > 6 else ""
        return ", ".join(p + (" (the files the user attached)" if p == ATTACHMENTS_DIR + "/" else "")
                         for p in paths[:6]) + more

    plain = [p for p, why in skipped if not why]
    ignored = [p for p, why in skipped if why]
    parts = ([names(plain)] if plain else []) + ([f"in .gitignore: {names(ignored)}"] if ignored else [])
    return f"\n(not searched: {'; '.join(parts)}. Pass one as `path` to search it.)" if parts else ""


def _excerpt(lines: list[str], a: int, b: int, col: int) -> str:
    """A hit as the model reads it: the line, or the stretch around the match on a long (minified) line; a match over
    several lines shows its first and last line."""
    def one(line: str, at: int) -> str:
        s = line.strip()
        if len(s) <= 240:
            return s
        at = max(0, at - 80)
        return ("…" if at else "") + line[at : at + 240].strip() + "…"

    if b == a:
        return one(lines[a], col)
    return one(lines[a], col) + (" ⏎ " if b == a + 1 else " ⏎ … ⏎ ") + one(lines[b], 0)


def search(ws: Workspace, args: dict) -> str:
    pattern = _str(args, "pattern")
    try:
        re.compile(pattern)
        rx = _search_regex(pattern)
    except re.error as exc:
        raise ToolError(f"invalid regex: {exc}") from None
    if _nested_repeat(rx.pattern):
        raise ToolError("this pattern repeats a group that can match the same text in many ways (like `(a+)+$`, "
                        "`(\\w+\\s?)*x` or `(\\\\.|[^\"])*`), which can take exponential time on a single line; "
                        "simplify it (e.g. `a+`, `[\\w\\s]+x`, `[^\"]*`)")
    where = _str(args, "path", False, ".")
    base = ws.resolve(where)
    if not base.exists():  # never a "no matches" for a place that is not there
        raise ToolError(_not_found(ws, where, base))
    multiline = _bool(args, "multiline") or _spans_lines(rx.pattern)
    if multiline:  # the whole file at once; `^` and `$` still match at every line
        rx = re.compile(rx.pattern, rx.flags | re.MULTILINE)
    globs = _globs(args.get("glob"))
    skipped: list = []
    hits: list[str] = []
    big: list[str] = []
    files = [base] if base.is_file() else _files(ws, base, _gitignore_below(ws, base), skipped)
    with _Matcher(rx, multiline, time.monotonic() + SEARCH_BUDGET, getattr(ws, "cancel", None)) as matcher:
        for f in files:
            if matcher.over():
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
                lines = _lines(shown(decode_text(data)[0]))
            except ValueError:
                continue  # binary
            for a, b, col, _ in matcher.find(lines, SEARCH_HITS - len(hits), rel):
                hits.append(f"{rel}:{a + 1}{f'-{b + 1}' if b > a else ''}: {_excerpt(lines, a, b, col)}")
            if len(hits) >= SEARCH_HITS:
                return "\n".join(hits) + f"\n... (truncated at {SEARCH_HITS} matches)"
    out = "\n".join(hits) or "no matches"
    if big:
        out += (f"\n({len(big)} file(s) over {SEARCH_MAX_FILE // 1_000_000} MB were not searched: "
                + ", ".join(big[:5]) + (", …" if len(big) > 5 else "") + "; use bash grep on them)")
    if not hits and not matcher.why:
        out += _skipped_note(skipped)
        if not multiline and "\\s" in pattern:
            out += "\n(each line is matched on its own: set multiline=true for a pattern that spans lines)"
    if matcher.why == "stopped":
        out += "\n[search stopped by the user]"
    elif matcher.why:
        out += (f"\n[search stopped after {SEARCH_BUDGET:.0f} s: results are partial; "
                + (f"the pattern is slow on {matcher.slow}: simplify it, or " if matcher.slow else "")
                + "narrow `path` or `glob`]")
    return out


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
    dirs = ws.seen_read_dirs() if hasattr(ws, "seen_read_dirs") else []
    # (whatever the exit code: `ls ~/Downloads | head; echo` ends 0)
    if _refused_by_sandbox(command, out, dirs, _own_paths(ws)):
        out += ("\n[sandbox] That is out of your reach, and no other command will get you there. " + ws.outside_hint()
                if dirs else
                "\n[sandbox] That is out of your reach: you only see this workspace, not the user's other folders or "
                "apps, and no other command will get you there. " + OUTSIDE_HINT)
    elif _refused_in_read_dirs(command, out, dirs):
        out += READ_ONLY_NOTE
    if "Operation not permitted" in out and _KILL_CMD.search(command):
        out += ("\n[sandbox] A process that an earlier command started cannot be signalled from this one: stop it "
                "with the stop_process tool (no arguments lists them).")
    return f"exit code: {code}\n{out}"


_KILL_CMD = re.compile(r"(?:^|[;&|(\s])(?:kill|pkill|killall)\b")


_HOME_REF = re.compile(r"(~|\$HOME\b|\$\{HOME\}|/Users/|/home/)")
# The user's own folders, named outright (the cage's private HOME has none of them, so "not found" there means this):
# a path into a home, or the usual folders through ~ or $HOME. Never a project's own Documents/ or apps/Desktop/.
_USER_DIRS = re.compile(r"(/Users/|/home/|(?:~|\$HOME\b|\$\{HOME\})/(?:Downloads|Documents|Desktop|Escritorio|Descargas|"
                        r"Documentos)\b)")
_APP_LAUNCH = re.compile(r"(?:^|[;&|(\s])(?:open|osascript)\s")
_APP_REFUSED = re.compile(r"lsopen|launchservices|not permitted|not authori[sz]ed|execution error|-54\b|-600\b|-1743\b", re.I)
_PATH_END = r"(?=$|[\\/\s'\";|&()<>`])[^\s'\";|&()<>`]*"  # the rest of a path that starts with a folder's own name
READ_ONLY_NOTE = ("\n[sandbox] The user's folders you may read are read-only for you, and places with keys or credentials "
                  "in them stay out of reach: copy what you need into the workspace and work on the copy. If even a plain "
                  "read is refused there, the system may not let Nesicode into that folder yet: tell the user (on a Mac: "
                  "System Settings → Privacy & Security → Files and Folders).")


def _without_read_dirs(command: str, read_dirs) -> str:
    """The command with the paths into the folders the agent may read taken out: what fails there (a file that is
    not there, one written to) is no wall of the sandbox around the user's folders."""
    for d in sorted(read_dirs, key=len, reverse=True):
        for spelled in dict.fromkeys((d, d.replace(" ", "\\ "))):
            command = re.sub(re.escape(spelled) + _PATH_END, " ", command)
    return command


def _refused_in_read_dirs(command: str, out: str, read_dirs) -> bool:
    """A command that tried to change something in a folder the agent may only read (or to read a secret in it)."""
    if not read_dirs or sandbox_mode() not in ("macos", "docker"):
        return False
    low = out.lower()
    return ("operation not permitted" in low or "read-only file system" in low) and \
        _without_read_dirs(command, read_dirs) != command


def _own_paths(ws) -> list[str]:
    """The workspace and its private HOME as a command may spell them. On the macOS app the workspace itself lives
    under /Users/<name>/Library/…: a file missing there is the agent's own business, not a wall of the sandbox."""
    out: list[str] = []
    for attr in ("root", "home"):
        try:
            out += [str(q) for q in spellings(getattr(ws, attr))]
        except Exception:  # (a stand-in workspace without them)
            continue
    return out


def _refused_by_sandbox(command: str, out: str, read_dirs=(), own=()) -> bool:
    """A command that tried to reach the user's folders or launch an app while the OS sandbox is on — judged by what
    it printed, not its exit code (a pipe or a trailing `; echo` hides the failure). Paths into the folders the user
    lets the agent read (`read_dirs`) do not count: a read there works, and a failure there is of another kind; nor do
    paths into the workspace itself (`own`)."""
    if sandbox_mode() not in ("macos", "docker"):
        return False
    command = _without_read_dirs(command, [*read_dirs, *own])
    low = out.lower()
    if "operation not permitted" in low and _HOME_REF.search(command):
        return True
    if "no such file or directory" in low and _USER_DIRS.search(command):
        return True
    return bool(_APP_LAUNCH.search(command)) and bool(_APP_REFUSED.search(out))


SCHEMAS = {
    "list_dir": ToolSpec(
        "list_dir", "List files and directories, a whole level before the next (dependency/VCS dirs and what the "
        ".gitignore ignores are not expanded; symlinks are shown, not followed).",
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
            "limit": {"type": "integer", "description": "Max lines (default 2000; fewer come back if the character "
                                                         "budget per call runs out first)."},
        }, "required": ["path"]},
    ),
    "write_file": ToolSpec(
        "write_file", "Create or overwrite a file with the given content. An existing file must have been read in this "
        "run first (to change part of a file, use edit_file).",
        {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
         "required": ["path", "content"]},
    ),
    "edit_file": ToolSpec(
        "edit_file", "Replace an exact string in a file. `old` must match the file exactly (indentation and tabs "
        "included, without read_file's line-number prefix) and be unique unless replace_all. The file's encoding and "
        "line endings are kept.",
        {"type": "object", "properties": {
            "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"},
            "replace_all": {"type": "boolean"},
        }, "required": ["path", "old", "new"]},
    ),
    "search": ToolSpec(
        "search", "Regex search over file contents, one line at a time. Returns path:line: text. Dependency/VCS dirs "
        "and what the .gitignore ignores are skipped (pass one as `path` to search it).",
        {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Python regular expression."},
            "path": {"type": "string", "description": "File or directory (default '.')."},
            "glob": {"type": "string", "description": "Optional filename filter, e.g. '*.py', '*.{ts,tsx}', "
                                                      "'**/test_*.py'; '!' excludes: '*.js,!*.min.js'."},
            "multiline": {"type": "boolean", "description": "Match across lines (on by itself when the pattern has "
                                                            "\\n or (?s))."},
        }, "required": ["pattern"]},
    ),
    "bash": ToolSpec(
        "bash", "Run a shell command from the workspace root. Returns exit code and combined output. "
        "Long output is trimmed automatically (start and end kept): don't pipe into head/tail just to shorten it. "
        "Processes left in the background (`cmd > log.txt 2>&1 &`) keep running until the run ends or the user "
        "presses Stop."
        " To list them or stop one (e.g. to restart a server after an edit) use `stop_process`: a `kill` from a later "
        "command may be refused."
        + (" On this machine the shell is Windows PowerShell (not bash): use PowerShell syntax."
           if shell_name() == "powershell" else ""),
        {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "description": "Seconds, default 120, max 900."},
        }, "required": ["command"]},
    ),
    "stop_process": ToolSpec(
        "stop_process", "List the processes your commands left running in the background (servers, watchers…), or stop "
        "one of them with everything it started, e.g. to restart a server after an edit. Only processes your own "
        "commands started can be stopped.",
        {"type": "object", "properties": {
            "pid": {"type": "integer", "description": "Process id to stop, from the list. Omit it to get the list."},
            "all": {"type": "boolean", "description": "Stop all of them."},
        }},
    ),
}


def read_file_spec(budget: int | None = None) -> ToolSpec:
    """read_file as the agent sees it: with the per-call character budget its context window allows (the loop keeps
    at most that much of one output in context, cut at a line boundary)."""
    base = SCHEMAS["read_file"]
    if not budget:
        return base
    return ToolSpec(base.name, base.description + (
        f" At most about {budget:,} characters come back per call: a longer read stops at a line boundary and ends "
        "with the exact offset to continue from (read big files in slices with offset/limit; use `search` to find the "
        "lines that matter first)."), base.parameters)


def stop_process(ws: Workspace, args: dict) -> str:
    """List what the workspace's commands left running in the background, or stop some of it (see stop_background)."""
    pid = args.get("pid")
    if _bool(args, "all") or (isinstance(pid, str) and pid.strip().lower() == "all"):
        target = None
    elif pid is None or pid == "":
        procs = ws.background_processes()
        if not procs:
            return "Nothing your commands started is running in the background."
        pids = {p for p, _, _ in procs}
        return "Running in the background (started by your commands):\n" + "\n".join(
            f"pid {p}: {_one_line(cmd)}" + (f"  (started by {parent})" if parent in pids else "") for p, parent, cmd in procs
        ) + "\nStop one with stop_process(pid=N) (what it started stops with it), or all of them with stop_process(all=true)."
    else:
        target = _int(args, "pid", 0)
        if target <= 0:
            raise ToolError("`pid` must be a process id from the list (call stop_process without arguments to get it)")
    stopped, left = ws.stop_background(target)
    if not stopped and not left:
        return "Nothing your commands started is running in the background."
    out = "Stopped: " + "; ".join(f"pid {p} ({_one_line(cmd)})" for p, cmd in stopped) if stopped else ""
    if left:
        out += ("\n" if out else "") + "Could not stop: " + "; ".join(f"pid {p} ({_one_line(cmd)})" for p, cmd in left)
    return out


def _one_line(command: str, limit: int = 200) -> str:
    command = " ".join(command.split()) or "?"
    return command if len(command) <= limit else command[: limit - 1] + "…"


BASIC_TOOLS: dict[str, Callable[[Workspace, dict], str]] = {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "search": search,
    "bash": bash,
    "stop_process": stop_process,
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
    "Search this conversation's archive (earlier turns and steps with your reasoning, compaction checkpoints, full "
    "tool outputs saved on disk) with a regex or keywords. Without an exact match it returns the lines with the most "
    "of your keywords (your own words and the user's first), with file:line to read around them. Use it whenever a "
    "detail was compacted away instead of guessing or redoing work.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
