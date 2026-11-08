"""Final review, runtime area — checker's regression tests: link chains split over several «Aplicar» (and absolute
links into the project, which must still apply), the caller's context in calls that Stop can abandon (the model
layer's request scope: the agent's Stop and retry notices for a tool's vision request), plain names in store_new_file.
Offline: temp dirs, fake providers."""

import contextvars
import os
import shutil
import threading
from pathlib import Path

import pytest

from nesicode.agent import Workspace
from nesicode.agent.workspace import ToolError, _up_after_down, apply_cage, interruptible, manifest, store_new_file
from nesicode.harnessing.apply import apply_rollout
from nesicode.llm.base import Provider, Usage

POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX only (symlinks)")


def _project(tmp_path: Path, name: str) -> Path:
    (tmp_path / "secret.txt").write_text("SECRET OUTSIDE THE PROJECT\n")
    proj = tmp_path / "home" / name
    proj.mkdir(parents=True)
    (proj / "README.md").write_text("hi\n")
    (tmp_path / "home" / "secret.txt").write_text("SECRET OUTSIDE THE PROJECT (home)\n")
    return proj


def _links_inside(root: Path) -> list[str]:
    """Every link under root that resolves out of it (normalised: a loop is judged by its letters)."""
    top, out = os.path.realpath(root), []
    for dp, dns, fns in os.walk(root):
        for n in dns + fns:
            p = Path(dp, n)
            if p.is_symlink():
                real = os.path.normpath(os.path.realpath(p))
                if not (real == top or real.startswith(top + os.sep)):
                    out.append(f"{p.relative_to(root)} -> {os.readlink(p)}")
    return out


def _cage_apply(cage: Path, proj: Path, change) -> dict:
    """One «Aplicar» of a conversation: the agent changes its cage, then the change goes to the project."""
    base = manifest(cage)
    change(cage)
    return apply_cage(base, cage, proj)


# ====================================================================== security#4 over several applies
@POSIX
def test_a_link_chain_split_over_two_applies_never_leaves_the_project(tmp_path):
    """`a -> y/z/../../secret.txt` alone looks inside (y and z do not exist); a later «Aplicar» of the same
    conversation adds `y -> .` and `z -> .`, each inside too, and every check of that apply only sees its own links."""
    proj = _project(tmp_path, "p1")
    cage = tmp_path / "cage1"
    shutil.copytree(proj, cage, symlinks=True)
    first = _cage_apply(cage, proj, lambda c: os.symlink("y/z/../../secret.txt", c / "a"))
    # (worded like every refused link: the web UI translates "<path>: symlink to <target> would point outside…")
    assert first["added"] == []
    assert first["skipped"] == ["a: symlink to y/z/../../secret.txt would point outside the project"]
    second = _cage_apply(cage, proj, lambda c: (os.symlink(".", c / "y"), os.symlink(".", c / "z")))
    assert sorted(second["added"]) == ["y", "z"]
    assert not os.path.lexists(proj / "a") and _links_inside(proj) == []


@POSIX
def test_a_folder_a_later_apply_turns_into_a_link_never_takes_a_link_out(tmp_path):
    """`a -> y/../secret.txt` is inside while y is a folder — whatever check is made when it is written — and leaves
    the project once a later apply replaces the (emptied) folder y with `y -> .`."""
    proj = _project(tmp_path, "p2")
    cage = tmp_path / "cage2"
    shutil.copytree(proj, cage, symlinks=True)

    def first_change(c: Path) -> None:
        (c / "y").mkdir()
        (c / "y" / "keep.txt").write_text("x")
        os.symlink("y/../secret.txt", c / "a")

    def second_change(c: Path) -> None:
        shutil.rmtree(c / "y")
        os.symlink(".", c / "y")

    first = _cage_apply(cage, proj, first_change)
    assert first["added"] == ["y/keep.txt"] and [s.split(":")[0] for s in first["skipped"]] == ["a"]
    second = _cage_apply(cage, proj, second_change)
    assert second["added"] == ["y"] and os.readlink(proj / "y") == "."
    assert not os.path.lexists(proj / "a") and _links_inside(proj) == []


@POSIX
def test_a_link_chain_split_over_two_harnessing_applies_never_leaves_the_project(tmp_path):
    proj = _project(tmp_path, "p3")

    def rollout(k: int, change) -> dict:
        before, after = tmp_path / f"before{k}", tmp_path / f"after{k}"
        shutil.copytree(proj, before, symlinks=True)
        shutil.copytree(proj, after, symlinks=True)
        change(after)
        return apply_rollout(before, after, proj)

    first = rollout(1, lambda a: os.symlink("y/z/../../secret.txt", a / "a"))
    assert first["added"] == [] and [(c["path"], c["reason"]) for c in first["conflicts"]] == [("a", "unsafe")]
    second = rollout(2, lambda a: (os.symlink(".", a / "y"), os.symlink(".", a / "z")))
    assert sorted(second["added"]) == ["y", "z"] and second["conflicts"] == []
    assert not os.path.lexists(proj / "a") and _links_inside(proj) == []


@POSIX
def test_absolute_links_into_the_project_still_apply(tmp_path):
    """An absolute link to a file of the project (and a relative one through it) points inside the project: it is
    applied, as before — judged in the conversation's copy, it pointed out of the copy and was refused."""
    proj = _project(tmp_path, "p4")
    (proj / "shared").mkdir()
    (proj / "shared" / "config.json").write_text("{}\n")
    absolute = str(proj.resolve() / "shared")
    cage = tmp_path / "cage4"
    shutil.copytree(proj, cage, symlinks=True)
    out = _cage_apply(cage, proj, lambda c: (os.symlink(absolute, c / "linked"),
                                             os.symlink("linked/config.json", c / "config.json")))
    assert sorted(out["added"]) == ["config.json", "linked"] and out["skipped"] == []
    assert (proj / "config.json").read_text() == "{}\n"
    proj2 = _project(tmp_path, "p5")
    (proj2 / "shared").mkdir()
    before, after = tmp_path / "before5", tmp_path / "after5"
    shutil.copytree(proj2, before, symlinks=True)
    shutil.copytree(proj2, after, symlinks=True)
    os.symlink(str(proj2.resolve() / "shared"), after / "linked")
    res = apply_rollout(before, after, proj2)
    assert res["added"] == ["linked"] and res["conflicts"] == []
    # one going down and back up is refused, absolute or not
    proj3 = _project(tmp_path, "p6")
    cage3 = tmp_path / "cage6"
    shutil.copytree(proj3, cage3, symlinks=True)
    out = _cage_apply(cage3, proj3, lambda c: os.symlink(str(proj3.resolve() / "README.md") + "/../README.md",
                                                         c / "readme"))
    assert out["added"] == [] and [s.split(":")[0] for s in out["skipped"]] == ["readme"]


@POSIX
def test_a_written_link_through_a_users_link_that_the_apply_takes_out_is_removed(tmp_path):
    """The user's own `w -> q/..` is inside while q is a folder; the apply turns q into `q -> .` (inside) and adds
    `a -> w/x` (inside when written): `a` leads out of the project once all of it is there, and is taken out again."""
    proj = _project(tmp_path, "p7")
    (proj / "q").mkdir()
    (proj / "q" / "keep.txt").write_text("k")
    os.symlink("q/..", proj / "w")
    cage = tmp_path / "cage7"
    shutil.copytree(proj, cage, symlinks=True)

    def change(c: Path) -> None:
        os.symlink("w/secret.txt", c / "a")
        shutil.rmtree(c / "q")
        os.symlink(".", c / "q")

    out = _cage_apply(cage, proj, change)
    assert out["added"] == ["q"] and [s.split(":")[0] for s in out["skipped"]] == ["a"]
    assert not os.path.lexists(proj / "a")


def test_link_targets_that_go_down_and_back_up():
    for target in ("y/../x", "y/z/../../x", "./y/../x", "y/./../x", "/abs/y/../x", "y/..", "../a/../b"):
        assert _up_after_down(target), target
    for target in (".", "..", "../..", "x", "y/x", "../pkg/bin/cli.js", "../../.pnpm/b@2/node_modules/b", "./x",
                   "./../x", "/abs/proj/x", "y/", "", "..."):
        assert not _up_after_down(target), target


# ====================================================================== runtime#2 x the model layer's request scope
_SCOPE: contextvars.ContextVar = contextvars.ContextVar("test_scope", default=("default",))


def test_interruptible_runs_the_call_in_the_callers_context():
    cancel = threading.Event()
    token = _SCOPE.set(("the run's stop", "the run's retry notices"))
    try:
        assert interruptible(cancel, _SCOPE.get) == ("the run's stop", "the run's retry notices")
        assert interruptible(None, _SCOPE.get) == ("the run's stop", "the run's retry notices")

        def boom():
            raise ValueError(_SCOPE.get()[0])

        with pytest.raises(ValueError, match="the run's stop"):
            interruptible(cancel, boom)
    finally:
        _SCOPE.reset(token)
    assert interruptible(cancel, _SCOPE.get) == ("default",)


# like the model layer's request_scope: (cancel, notify) for requests whose caller does not pass them
_REQUEST: contextvars.ContextVar = contextvars.ContextVar("test_request_scope", default=(None, None))


class _ScopedVision(Provider):
    """A vision provider that, like the model layer's, takes the agent's Stop and retry listener from the enclosing
    request scope (a context variable) when the caller does not pass them."""

    name = "scoped"

    def conversation(self, *a):
        raise NotImplementedError

    def vision(self, model, prompt, image, mime, *, cancel=None, notify=None):
        scope_cancel, scope_notify = _REQUEST.get()
        self.seen = {"cancel": cancel if cancel is not None else scope_cancel, "notify": notify or scope_notify}
        if callable(self.seen["notify"]):
            self.seen["notify"]({"type": "llm_retry", "attempt": 2})  # a retried request: the user is told
        return "una captura", Usage()


def test_a_vision_request_keeps_the_agents_request_scope(tmp_path, monkeypatch):
    """The agent wraps view_image in the model layer's request scope (its Stop and its retry notices, shown to the
    user); the request now runs in a helper thread Stop can abandon, and must still see that scope."""
    from nesicode.agent import vision

    prov = _ScopedVision()
    monkeypatch.setattr(vision, "get_provider", lambda spec, keys=None: (prov, "v"))
    ws = Workspace(tmp_path / "ws")
    (ws.root / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    ws.cancel = threading.Event()
    heard = []
    token = _REQUEST.set((ws.cancel, heard.append))
    try:
        assert vision.describe_image(ws, "a.png", "scoped/v", None, "") == "una captura"
    finally:
        _REQUEST.reset(token)
    assert prov.seen["cancel"] is ws.cancel
    assert heard == [{"type": "llm_retry", "attempt": 2}]


# ====================================================================== store_new_file
@POSIX
def test_store_new_file_takes_plain_names_only(tmp_path):
    root = tmp_path / "cage"
    (root / ".attachments").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, root / ".attachments" / "sub")  # planted
    for bad in ("sub/x.txt", "..", ".", ""):
        with pytest.raises(ToolError):
            store_new_file(root, ".attachments", bad, b"x")
    assert list(outside.iterdir()) == []
    assert store_new_file(root, ".attachments", "x.txt", b"x") == "x.txt"
