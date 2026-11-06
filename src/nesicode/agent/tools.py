"""The agent's tools. Every tool returns text; failures raise ToolError."""

from __future__ import annotations

import fnmatch
import re
from typing import Callable

from ..harness import Harness, skills, sub_agents
from ..llm import ToolSpec
from .workspace import IGNORE_DIRS, ToolError, Workspace

MAX_READ_LINES = 2000
MAX_LIST = 400


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
    v = args.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ToolError(f"`{key}` must be an integer") from None


def list_dir(ws: Workspace, args: dict) -> str:
    base = ws.resolve(_str(args, "path", False, "."))
    if not base.is_dir():
        raise ToolError(f"{args.get('path')!r} is not a directory")
    depth = max(1, min(_int(args, "depth", 2), 5))
    lines: list[str] = []

    def walk(d, level):
        try:
            entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except OSError as exc:
            lines.append(f"{'  ' * level}[unreadable: {exc}]")
            return
        for p in entries:
            if len(lines) >= MAX_LIST:
                return
            if p.is_dir():
                if p.name in IGNORE_DIRS:
                    lines.append(f"{'  ' * level}{p.name}/ (skipped)")
                    continue
                lines.append(f"{'  ' * level}{p.name}/")
                if level + 1 < depth:
                    walk(p, level + 1)
            else:
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"{'  ' * level}{p.name} ({size} B)")

    walk(base, 0)
    if len(lines) >= MAX_LIST:
        lines.append(f"... (truncated at {MAX_LIST} entries; list a subdirectory)")
    return "\n".join(lines) or "(empty directory)"


def read_file(ws: Workspace, args: dict) -> str:
    p = ws.resolve(_str(args, "path"))
    if not p.is_file():
        raise ToolError(f"file not found: {args.get('path')}")
    if p.suffix.lower() == ".pdf":
        lines = pdf_text(p).splitlines()
    else:
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            raise ToolError("binary file; not shown (images: use view_image)")
        lines = data.decode("utf-8", "replace").splitlines()
    offset = max(1, _int(args, "offset", 1))
    limit = max(1, min(_int(args, "limit", MAX_READ_LINES), MAX_READ_LINES))
    chunk = lines[offset - 1 : offset - 1 + limit]
    out = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(chunk, start=offset))
    rest = len(lines) - (offset - 1 + len(chunk))
    if rest > 0:
        out += f"\n... ({rest} more lines; use offset={offset + len(chunk)})"
    return out or "(empty file)"


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


def write_file(ws: Workspace, args: dict) -> str:
    p = ws.resolve(_str(args, "path"))
    content = _str(args, "content", required=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content, encoding="utf-8")
    return f"{'overwrote' if existed else 'created'} {ws.rel(p)} ({len(content)} chars)"


def edit_file(ws: Workspace, args: dict) -> str:
    p = ws.resolve(_str(args, "path"))
    if not p.is_file():
        raise ToolError(f"file not found: {args.get('path')}")
    old = _str(args, "old")
    new = _str(args, "new", required=False)
    text = p.read_text(encoding="utf-8", errors="replace")
    count = text.count(old)
    if count == 0:
        raise ToolError("`old` was not found; read the file and copy the text exactly (whitespace included)")
    if count > 1 and not args.get("replace_all"):
        raise ToolError(f"`old` occurs {count} times; add surrounding context or set replace_all=true")
    text = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
    p.write_text(text, encoding="utf-8")
    return f"edited {ws.rel(p)} ({count if args.get('replace_all') else 1} replacement(s))"


def search(ws: Workspace, args: dict) -> str:
    pattern = _str(args, "pattern")
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise ToolError(f"invalid regex: {exc}") from None
    base = ws.resolve(_str(args, "path", False, "."))
    glob = args.get("glob") or ""
    hits: list[str] = []
    files = [base] if base.is_file() else None
    if files is None:
        from .workspace import _walk

        files = _walk(base)
    for f in files:
        rel = ws.rel(f)
        if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(f.name, glob)):
            continue
        try:
            if f.stat().st_size > 2_000_000:
                continue
            data = f.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        for i, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:240]}")
                if len(hits) >= 200:
                    return "\n".join(hits) + "\n... (truncated at 200 matches)"
    return "\n".join(hits) or "no matches"


def bash(ws: Workspace, args: dict) -> str:
    command = _str(args, "command")
    code, out = ws.run(command, _int(args, "timeout", 120))
    out = out if len(out) <= 30_000 else out[:15_000] + "\n... (output truncated) ...\n" + out[-15_000:]
    return f"exit code: {code}\n{out}"


SCHEMAS = {
    "list_dir": ToolSpec(
        "list_dir", "List files and directories (dependency/VCS dirs are skipped).",
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
        "edit_file", "Replace an exact string in a file. `old` must match exactly and be unique unless replace_all.",
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
            "glob": {"type": "string", "description": "Optional filename filter, e.g. '*.py'."},
        }, "required": ["pattern"]},
    ),
    "bash": ToolSpec(
        "bash", "Run a shell command from the workspace root. Returns exit code and combined output."
        + (" On this machine the shell is Windows PowerShell (not bash): use PowerShell syntax."
           if __import__("nesicode.agent.workspace", fromlist=["shell_name"]).shell_name() == "powershell" else ""),
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
    "download card in the chat. Use it for every deliverable file the user asked for.",
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
