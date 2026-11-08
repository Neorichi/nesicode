"""Apply a rollout's file changes to the project without clobbering what changed there since.

A harnessing run snapshots the project when it starts; the result may be applied minutes or days
later, while the user (or a chat) keeps editing the project. So the apply is three-way: a change
lands only where the project still matches the snapshot (or already has the change). Anything else
is a conflict: the project's file is left alone and, for a changed or new file, the rollout's version
is written next to it as `<name>.nesicode-new`, so nothing is lost and nothing is silently overwritten.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..agent import diff_workspaces

SIDE_SUFFIX = ".nesicode-new"
CHUNK = 1 << 20

# conflict reasons (the UI/CLI word them)
CHANGED = "changed"  # the project's file changed (or was deleted) after the snapshot
EXISTS = "exists"  # the rollout added a file that the project now has with other content
UNSAFE = "unsafe"  # the path leads out of the project through a symlink
ERROR = "error"  # it could not be written (something in the way, permissions, disk...)


def _same(a: Path, b: Path) -> bool:
    """Same bytes (or the same symlink), without following symlinks. Missing files are never the same."""
    try:
        if a.is_symlink() or b.is_symlink():
            return a.is_symlink() and b.is_symlink() and os.readlink(a) == os.readlink(b)
        if not (a.is_file() and b.is_file()) or a.stat().st_size != b.stat().st_size:
            return False
        with a.open("rb") as fa, b.open("rb") as fb:
            while True:
                ca, cb = fa.read(CHUNK), fb.read(CHUNK)
                if ca != cb:
                    return False
                if not ca:
                    return True
    except OSError:
        return False


def _inside(root: str, path: Path) -> bool:
    """`path`'s directory, symlinks resolved (from its deepest existing ancestor), is inside `root`."""
    d = path.parent
    while not os.path.lexists(d) and d != d.parent:
        d = d.parent
    real = os.path.realpath(d)
    return real == root or real.startswith(root.rstrip(os.sep) + os.sep)


def _place(src: Path, dst: Path) -> None:
    """Copy `src` (a file or a symlink, never followed) to `dst`, replacing a symlink or an empty directory."""
    if dst.is_symlink():
        dst.unlink()
    elif dst.is_dir():
        dst.rmdir()  # only an empty one (left by a removal); a full directory raises and is reported
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst, follow_symlinks=False)


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
        src, dst = after / rel, target / (rel + SIDE_SUFFIX)
        if side and not src.is_symlink() and src.is_file() and _inside(root, dst):
            try:
                _place(src, dst)
                side_file = rel + SIDE_SUFFIX
            except OSError:
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
        elif not _same(dst, before / rel):
            conflict(rel, CHANGED, side=False)  # the user changed it since: keep it
        else:
            try:
                dst.unlink()
                done["removed"].append(rel)
            except OSError:
                conflict(rel, ERROR, side=False)
    for kind in ("added", "modified"):
        for rel in d[kind]:
            src, dst = after / rel, target / rel
            if not _inside(root, dst):
                conflict(rel, UNSAFE, side=False)
                continue
            if _same(dst, src):
                done[kind].append(rel)  # already there
                continue
            if kind == "added" and os.path.lexists(dst) and not (dst.is_dir() and not dst.is_symlink()):
                conflict(rel, EXISTS)
                continue
            if kind == "modified" and not _same(dst, before / rel):
                conflict(rel, CHANGED)
                continue
            try:
                _place(src, dst)
                done[kind].append(rel)
            except OSError:  # e.g. a non-empty directory or a file in the way, permissions
                conflict(rel, ERROR)
    return done
