"""Git history in a conversation's copy of the project.

«Aplicar» copies the working files only (.git is never listed, diffed or applied): commits, branches and stashes made
in the copy stay there. The copy's local refs are recorded when it is made (beside it, outside the agent's folder)
and compared later, so the user is told what «Aplicar» leaves behind.

Everything is read straight from the copy's .git, never through a link and never by running git on a repository the
agent wrote (its config can make git run commands). So only what git keeps as plain files is understood: loose and
packed refs, and loose commit objects — the commits made in the copy, until a `git gc` there packs them."""

from __future__ import annotations

import json
import os
import re
import stat
import time
import zlib
from pathlib import Path

from ..agent.workspace import _read_beneath, _scan

START_FILE = "git-start.json"  # next to the cage (<thread dir>/ws): {"at": time, "refs": {...} | null}
_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")  # SHA-1, or SHA-256 repositories
MAX_REF_FILE = 4096
MAX_PACKED_REFS = 20 << 20
MAX_REFS = 2_000  # local branches followed
MAX_WALK = 2_000  # commits looked at when counting
SKEW = 120  # s: a commit stamped up to this long before the copy was made still counts as made in it


def _read(ws: Path, rel: str, limit: int) -> bytes:
    try:
        return _read_beneath(ws, rel, limit)
    except (OSError, ValueError):
        return b""


def _sha(data: bytes) -> str:
    text = data.decode("ascii", "replace").strip()
    return text if _SHA.fullmatch(text) else ""


def local_refs(ws: Path) -> dict[str, str] | None:
    """{'refs/heads/<branch>': sha, 'refs/stash': sha, 'HEAD': sha} of ws/.git; {} when there is no repository, None
    when there is one this can't read (a worktree's or submodule's .git file, the reftable format)."""
    ws = Path(ws)
    try:
        st = os.lstat(ws / ".git")
    except FileNotFoundError:
        return {}
    except OSError:
        return None
    if not stat.S_ISDIR(st.st_mode) or os.path.lexists(ws / ".git" / "reftable"):
        return None
    refs: dict[str, str] = {}
    for line in _read(ws, ".git/packed-refs", MAX_PACKED_REFS).decode("utf-8", "replace").splitlines():
        sha, _, name = line.partition(" ")
        if len(refs) < MAX_REFS and _SHA.fullmatch(sha) and (name.startswith("refs/heads/") or name == "refs/stash"):
            refs[name] = sha
    heads = ws / ".git" / "refs" / "heads"
    for p, _e in _scan(heads):  # (each file is read beneath the cage: never through a link)
        if len(refs) >= MAX_REFS:
            break
        name = "refs/heads/" + p.relative_to(heads).as_posix()
        if sha := _sha(_read(ws, ".git/" + name, MAX_REF_FILE)):
            refs[name] = sha
    if sha := _sha(_read(ws, ".git/refs/stash", MAX_REF_FILE)):
        refs["refs/stash"] = sha
    head = _read(ws, ".git/HEAD", MAX_REF_FILE).decode("utf-8", "replace").strip()
    sha = refs.get(head[5:].strip(), "") if head.startswith("ref: ") else _sha(head.encode())
    if sha:
        refs["HEAD"] = sha
    return refs


def record(ws: Path) -> None:
    """Note the refs of a fresh cage (right after the project was copied into it). Never fails the cage."""
    try:
        refs = local_refs(ws)
        (Path(ws).parent / START_FILE).write_text(json.dumps({"at": time.time(), "refs": refs}), encoding="utf-8")
    except Exception as exc:  # the cage works without it; the Changes panel then just says nothing about git
        print("nesicode: could not record the copy's git refs:", exc)


def _loose_commit(ws: Path, sha: str) -> tuple[list[str], int] | None:
    """(parents, committer time) of the commit `sha` kept as a loose object; None when it is not one (packed, missing,
    another kind of object)."""
    data = _read(ws, f".git/objects/{sha[:2]}/{sha[2:]}", 1 << 20)
    try:
        raw = zlib.decompressobj().decompress(data, 64 << 10) if data else b""
    except zlib.error:
        return None
    kind, _, body = raw.partition(b"\0")
    if not kind.startswith(b"commit "):
        return None
    parents, when = [], 0
    for line in body.split(b"\n"):
        if not line:
            break  # the message follows
        if line.startswith(b"parent "):
            parents.append(line[7:].strip().decode("ascii", "replace"))
        elif line.startswith(b"committer ") and (m := re.search(rb" (\d+) [-+]\d{4}$", line)):
            when = int(m.group(1))
    return parents, when


def _new_commits(ws: Path, tip: str, known: set, since: float, seen: set) -> set[str]:
    """Commits reachable from `tip` made in the copy: loose, stamped after it was made, not reachable from a ref it
    started with (the walk stops at those, at older commits and at packed ones)."""
    out, stack = set(), [tip]
    while stack and len(seen) < MAX_WALK:
        sha = stack.pop()
        if sha in seen or sha in known or not _SHA.fullmatch(sha):
            continue
        seen.add(sha)
        commit = _loose_commit(ws, sha)
        if commit is None or commit[1] < since:
            continue
        out.add(sha)
        stack.extend(commit[0])
    return out


def report(ws: Path) -> dict | None:
    """What git did in the copy that «Aplicar» leaves behind: {"commits": n, "branches": [names with such commits],
    "stash": bool, "uncounted": bool}. "uncounted": a branch moved to a commit git keeps packed (a `git gc` in the copy
    packs the new ones too), so its commits could not be counted. None when nothing, or nothing known (a conversation
    older than this record, an unreadable repository)."""
    try:
        return _report(ws)
    except Exception as exc:  # an odd repository never breaks the Changes panel that shows this
        print("nesicode: could not read the git history of a conversation's copy:", exc)
        return None


def _report(ws: Path) -> dict | None:
    try:
        start = json.loads((Path(ws).parent / START_FILE).read_text(encoding="utf-8"))
        before, since = start["refs"], float(start["at"]) - SKEW
    except (OSError, ValueError, KeyError, TypeError):
        return None
    now = local_refs(ws)
    if not isinstance(before, dict) or not now:
        return None
    moved = {k: v for k, v in now.items() if before.get(k) != v}
    known, seen, new, branches, uncounted = set(before.values()), set(), set(), [], False
    for name in sorted(k for k in moved if k.startswith("refs/heads/")) + (["HEAD"] if "HEAD" in moved else []):
        tip = moved[name]
        found = _new_commits(ws, tip, known, since, seen)
        packed = not found and tip not in new and tip not in known and _loose_commit(ws, tip) is None
        if name != "HEAD" and (found or tip in new or packed):
            branches.append(name[len("refs/heads/"):])
        new |= found
        uncounted = uncounted or packed
    stash = "refs/stash" in moved
    if not new and not stash and not uncounted:
        return None
    return {"commits": len(new), "branches": branches, "stash": stash, "uncounted": uncounted}
