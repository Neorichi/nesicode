"""Apply a rollout's file changes to the project without clobbering what changed there since.

A harnessing run snapshots the project when it starts; the result may be applied minutes or days
later, while the user (or a chat) keeps editing the project. So the apply is three-way: a change
lands only where the project still matches the snapshot (or already has the change). Anything else
is a conflict: the project's file is left alone and, for a changed or new file, the rollout's version
is written next to it as `<name>.nesicode-new`, so nothing is lost and nothing is silently overwritten.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

from ..agent import diff_workspaces
# fd-based helpers: every folder on the way is opened without following a link (a link swapped in after a check is
# refused, never written through) and nothing but a regular file is opened (a FIFO would block the apply)
from ..agent.workspace import _copy_entry, _open_beneath, _unlink_entry

SIDE_SUFFIX = ".nesicode-new"
CHUNK = 1 << 20

# conflict reasons (the UI/CLI word them)
CHANGED = "changed"  # the project's file changed (or was deleted) after the snapshot
EXISTS = "exists"  # the rollout added a file that the project now has with other content
UNSAFE = "unsafe"  # the path leads out of the project through a symlink
ERROR = "error"  # it could not be written (something in the way, permissions, disk...)


def _reason(target: Path, rel: str, exc: BaseException) -> str:
    """Why writing or removing target/rel failed: UNSAFE for a link on the way (or a copied link that would point out
    of the project), ERROR for anything else (a file where a folder should be, permissions, a full disk…)."""
    no = getattr(exc, "errno", None)
    if no in (errno.ELOOP, errno.EMLINK, errno.EPERM):
        return UNSAFE
    if no == errno.ENOTDIR:  # a link or a plain file where a folder should be
        cur = Path(target)
        for name in Path(rel).parts[:-1]:
            cur = cur / name
            if os.path.islink(cur):
                return UNSAFE
    return ERROR


def _same(ra: Path, rb: Path, rel: str) -> bool:
    """ra/rel and rb/rel hold the same bytes (or are the same symlink), compared without following any link and
    without opening anything but regular files. Missing entries are never the same."""
    a, b = Path(ra, rel), Path(rb, rel)
    try:
        sa, sb = os.lstat(a), os.lstat(b)
        if stat.S_ISLNK(sa.st_mode) or stat.S_ISLNK(sb.st_mode):
            return stat.S_ISLNK(sa.st_mode) and stat.S_ISLNK(sb.st_mode) and os.readlink(a) == os.readlink(b)
        if not (stat.S_ISREG(sa.st_mode) and stat.S_ISREG(sb.st_mode)) or sa.st_size != sb.st_size:
            return False
        with os.fdopen(_open_beneath(Path(ra), rel), "rb") as fa, os.fdopen(_open_beneath(Path(rb), rel), "rb") as fb:
            while True:
                ca, cb = fa.read(CHUNK), fb.read(CHUNK)
                if ca != cb:
                    return False
                if not ca:
                    return True
    except (OSError, ValueError):
        return False


def _inside(root: str, path: Path) -> bool:
    """`path`'s directory, symlinks resolved (from its deepest existing ancestor), is inside `root`."""
    d = path.parent
    while not os.path.lexists(d) and d != d.parent:
        d = d.parent
    real = os.path.realpath(d)
    return real == root or real.startswith(root.rstrip(os.sep) + os.sep)


def apply_rollout(before: Path, after: Path, target: Path) -> dict:
    """Copy what changed between `before` (the snapshot) and `after` (the rollout) onto `target`, three-way.

    Returns {"added", "removed", "modified": [paths applied], "conflicts": [{"path", "reason", "side_file"}]}.
    """
    before, after, target = Path(before), Path(after), Path(target)
    for p in (before, after):
        if not p.is_dir():  # without both trees there is nothing to compare against: never guess
            raise FileNotFoundError(f"{p} is no longer available")
    d = diff_workspaces(before, after, max_chars=0)
    root = os.path.realpath(target)
    done: dict = {"added": [], "removed": [], "modified": [], "conflicts": []}

    def conflict(rel: str, reason: str, side: bool = True) -> None:
        side_file = None
        src = after / rel
        if side and not src.is_symlink() and src.is_file() and _inside(root, target / (rel + SIDE_SUFFIX)):
            try:
                _copy_entry(after, target, rel, dst_rel=rel + SIDE_SUFFIX)
                side_file = rel + SIDE_SUFFIX
            except (OSError, ValueError):
                pass
        done["conflicts"].append({"path": rel, "reason": reason, "side_file": side_file})

    # Removals first: a case-only rename (button.tsx -> Button.tsx) on a case-insensitive disk is one file,
    # and removing after copying would delete the renamed file.
    for rel in d["removed"]:
        dst = target / rel
        if not os.path.lexists(dst):
            done["removed"].append(rel)  # already gone
        elif not _inside(root, dst):
            conflict(rel, UNSAFE, side=False)
        elif not _same(target, before, rel):
            conflict(rel, CHANGED, side=False)  # the user changed it since: keep it
        else:
            try:
                _unlink_entry(target, rel)
                done["removed"].append(rel)
            except (OSError, ValueError) as exc:
                conflict(rel, _reason(target, rel, exc), side=False)
    for kind in ("added", "modified"):
        for rel in d[kind]:
            dst = target / rel
            if not _inside(root, dst):
                conflict(rel, UNSAFE, side=False)
                continue
            if _same(target, after, rel):
                done[kind].append(rel)  # already there
                continue
            if kind == "added" and os.path.lexists(dst) and not (dst.is_dir() and not dst.is_symlink()):
                conflict(rel, EXISTS)
                continue
            if kind == "modified" and not _same(target, before, rel):
                conflict(rel, CHANGED)
                continue
            try:
                _copy_entry(after, target, rel)
                done[kind].append(rel)
            except (OSError, ValueError) as exc:  # a link on the way or out of the project; a full folder in the way…
                why = _reason(target, rel, exc)
                conflict(rel, why, side=why != UNSAFE)
    return done
