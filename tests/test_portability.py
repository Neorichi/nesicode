r"""Portability guards: what made the product behave differently on Windows than on macOS, checked on any OS.

Text: Python reads and writes text in the locale's code page unless told otherwise (cp1252 on a Windows machine set up
in Spanish or English, UTF-8 on macOS and Linux), so a file written without naming its encoding, or a subprocess's
output decoded with `text=True`, turns «ó» into «Ã³» or fails with UnicodeEncodeError there. Every text read or write in
src/ names its encoding: the AST guard below fails on the first one that does not, wherever it runs.

Paths: workspace paths are spelled with '/' for the agent, the API and the UI on every OS, and a link's target is shown
as it was written (Windows reads an absolute one back as \\?\C:\…)."""

import ast
import os
import re
from pathlib import Path

import pytest

from nesicode.agent import Workspace
from nesicode.agent.tools import edit_file, list_dir, search, write_file
from nesicode.agent.workspace import (
    _changes, apply_cage, diff_workspaces, link_text, manifest, manifest_paths, readlink,
)

SRC = Path(__file__).resolve().parents[1] / "src"
MODE = re.compile(r"[rwxabt+U]{1,4}")  # a string argument that is an open() mode, not a file name
SUBPROCESS = {"run", "Popen", "check_output", "check_call", "call"}


def _kw(call: ast.Call, name: str):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _const(node) -> object:
    return node.value if isinstance(node, ast.Constant) else None


def _text_mode(mode, default: str) -> bool:
    """True when the open() mode is (or may be) a text mode."""
    if mode is None:
        return "b" not in default
    value = _const(mode)
    return not (isinstance(value, str) and "b" in value)


def _problem(call: ast.Call) -> str | None:
    f = call.func
    name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
    owner = f.value.id if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else ""
    has_encoding = _kw(call, "encoding") is not None
    args = call.args
    if any(isinstance(a, ast.Starred) for a in args) or any(k.arg is None for k in call.keywords):
        return None  # *args / **kwargs: cannot tell
    if name == "read_text" and isinstance(f, ast.Attribute):
        return None if has_encoding or len(args) >= 1 else "read_text() without encoding="
    if name == "write_text" and isinstance(f, ast.Attribute):
        return None if has_encoding or len(args) >= 2 else "write_text() without encoding="
    if (name == "open" and isinstance(f, ast.Name)) or (name == "open" and owner == "io") or name == "fdopen":
        mode = args[1] if len(args) > 1 else _kw(call, "mode")
        if _text_mode(mode, "r") and not has_encoding and len(args) < 4:
            return f"{owner + '.' if owner else ''}{name}() in text mode without encoding="
        return None
    if name == "open" and isinstance(f, ast.Attribute) and owner not in ("os", "webbrowser"):  # Path.open() & co.
        mode = _kw(call, "mode")
        if mode is None and args and isinstance(_const(args[0]), str) and MODE.fullmatch(_const(args[0])):
            mode = args[0]
        if (mode is not None or not args) and _text_mode(mode, "r") and not has_encoding:
            return ".open() in text mode without encoding="
        return None
    if name == "TextIOWrapper":
        return None if has_encoding or len(args) >= 2 else "TextIOWrapper() without encoding="
    if name in ("NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"):
        mode = args[0] if args else _kw(call, "mode")
        return "a text-mode temporary file without encoding=" if _text_mode(mode, "w+b") and not has_encoding else None
    if owner == "codecs" and name == "open":
        return None if has_encoding or len(args) >= 3 else "codecs.open() without encoding="
    if (owner == "subprocess" and name in SUBPROCESS) or name in ("create_subprocess_exec", "create_subprocess_shell"):
        text = any(_const(_kw(call, k)) is True for k in ("text", "universal_newlines"))
        return "a subprocess decoded with text=True (the locale's code page) instead of encoding=" \
            if text and not has_encoding else None
    if owner == "subprocess" and name in ("getoutput", "getstatusoutput"):
        return f"subprocess.{name}() (decodes with the locale's code page)"
    if owner == "os" and name == "popen":
        return "os.popen() (decodes with the locale's code page)"
    if owner == "logging" and name == "FileHandler":
        return None if has_encoding or len(args) >= 3 else "logging.FileHandler() without encoding="
    return None


def text_io_without_encoding(root: Path = SRC) -> list[str]:
    found = []
    for f in sorted(root.rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                why = _problem(node)
                if why:
                    found.append(f"{f.relative_to(root.parent).as_posix()}:{node.lineno}: {why}")
    return found


def test_every_text_read_and_write_in_src_names_its_encoding():
    assert text_io_without_encoding() == []


def test_the_guard_catches_what_it_is_for(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "bad.py").write_text(
        "import io, os, subprocess, tempfile\n"
        "from pathlib import Path\n"
        "Path('x').read_text()\n"
        "Path('x').write_text('ó')\n"
        "open('x')\n"
        "open('x', 'w')\n"
        "os.fdopen(3, 'w')\n"
        "Path('x').open()\n"
        "Path('x').open('a')\n"
        "subprocess.run(['git'], capture_output=True, text=True)\n"
        "subprocess.check_output(['git'], universal_newlines=True)\n"
        "tempfile.NamedTemporaryFile('w')\n"
        "io.TextIOWrapper(open('x', 'rb'))\n"
        "os.popen('ls')\n", encoding="utf-8")
    (tmp_path / "pkg" / "good.py").write_text(
        "import os, subprocess, webbrowser, zipfile\n"
        "from pathlib import Path\n"
        "Path('x').read_text(encoding='utf-8')\n"
        "Path('x').read_text('utf-8')\n"
        "Path('x').write_text('ó', encoding='utf-8')\n"
        "Path('x').read_bytes()\n"
        "open('x', 'rb')\n"
        "open('x', 'w', encoding='utf-8')\n"
        "os.fdopen(3, 'wb')\n"
        "os.open('x', os.O_RDONLY)\n"
        "Path('x').open('rb')\n"
        "zipfile.ZipFile('x.zip').open('a.txt')\n"
        "webbrowser.open('https://example.com')\n"
        "subprocess.run(['git'], capture_output=True)\n"
        "subprocess.run(['git'], capture_output=True, encoding='utf-8', errors='replace')\n"
        "b'x'.decode()\n", encoding="utf-8")
    found = text_io_without_encoding(tmp_path / "pkg")
    assert all(x.startswith("pkg/bad.py:") for x in found), found
    assert [int(x.split(":")[1]) for x in found] == list(range(3, 15))


# ---------------------------------------------------------------- paths: '/' everywhere, links as they were written
def _symlink(target, link: Path) -> None:
    try:
        os.symlink(target, link)
    except OSError as exc:  # Windows without the right (Developer Mode or admin) to create links
        pytest.skip(f"cannot create symbolic links here: {exc}")


def test_workspace_paths_in_tool_outputs_use_slashes(tmp_path):
    """The agent, the API and the UI read and match these paths: 'sub\\dir\\f.txt' on Windows was another file to
    them (the UI's links, «Aplicar»'s conflicts, search_history's file:line…)."""
    ws = Workspace(tmp_path / "ws")
    assert ws.rel(ws.root / "sub" / "dir" / "f.txt") == "sub/dir/f.txt" and ws.rel(ws.root) == "."
    assert write_file(ws, {"path": "sub/dir/f.txt", "content": "hola\n"}) == "created sub/dir/f.txt (5 chars)"
    edited = edit_file(ws, {"path": "sub/dir/f.txt", "old": "hola", "new": "adiós"})
    assert edited.startswith("edited sub/dir/f.txt (1 replacement(s))")
    assert search(ws, {"pattern": "adiós"}) == "sub/dir/f.txt:1:adiós"
    assert search(ws, {"pattern": "adi", "glob": "sub/**/*.txt"}).startswith("sub/dir/f.txt:1:")
    assert "sub/dir/f.txt" in manifest(ws.root)
    before = tmp_path / "before"
    before.mkdir()
    d = diff_workspaces(before, ws.root)
    assert d["added"] == ["sub/dir/f.txt"] and "+++ b/sub/dir/f.txt" in d["diff"]


def test_link_targets_are_shown_as_they_were_written():
    nt = {"windows": True}
    assert link_text("\\\\?\\C:\\Users\\ana\\x.txt", **nt) == "C:\\Users\\ana\\x.txt"
    assert link_text("\\\\?\\UNC\\srv\\share\\x", **nt) == "\\\\srv\\share\\x"
    assert link_text("\\\\?\\Volume{0b5e}\\x", **nt) == "\\\\?\\Volume{0b5e}\\x"  # no shorter spelling: as it is
    assert link_text("..\\x", **nt) == "..\\x" and link_text("docs/guide.md", **nt) == "docs/guide.md"
    assert link_text("\\\\?\\C:\\x", windows=False) == "\\\\?\\C:\\x"  # a POSIX name may hold anything


def test_absolute_links_are_reported_without_the_nt_prefix(tmp_path):
    """Windows reads an absolute link back as \\\\?\\C:\\…: list_dir, the diff and «Aplicar»'s reasons show the path
    the link was made with, and it is judged against the project's real path (which has no such prefix)."""
    outside = tmp_path / "outside.txt"
    outside.write_text("fuera\n", encoding="utf-8")
    ws = Workspace(tmp_path / "ws")
    _symlink(outside, ws.root / "lnk")
    assert readlink(ws.root / "lnk") == str(outside)
    assert f"lnk -> {outside}" in list_dir(ws, {})
    before = tmp_path / "before"
    before.mkdir()
    assert f"Symlink changed: lnk: missing -> {outside}" in diff_workspaces(before, ws.root)["diff"]
    project = tmp_path / "project"
    project.mkdir()
    out = apply_cage({"": "2"}, ws.root, project)
    assert out["skipped"] == [f"lnk: symlink to {outside} would point outside the project"]
    assert not os.path.lexists(project / "lnk")


def test_old_windows_manifests_are_read_with_slashes():
    old = {"": "2", "src\\app.py": "h1", "README.md": "h2"}
    assert manifest_paths(old, windows=True) == {"": "2", "src/app.py": "h1", "README.md": "h2"}
    assert manifest_paths(old, windows=False) is old  # on POSIX a '\\' is part of the name


@pytest.mark.skipif(os.name != "nt", reason="a '\\' in a stored path is a folder separator only on Windows")
def test_a_conversation_started_before_the_upgrade_shows_and_applies_only_its_own_changes(tmp_path):
    """Manifests stored by earlier versions on Windows spell 'src\\app.py': read as they are, every file in a folder
    looked added, and «Aplicar» copied the conversation's old copy over the project's newer one."""
    cage, project = tmp_path / "cage", tmp_path / "project"
    for root in (cage, project):
        (root / "src").mkdir(parents=True)
        (root / "src" / "app.py").write_bytes(b"v1\n")
    base = {k.replace("/", "\\"): v for k, v in manifest(cage).items()}
    assert "src\\app.py" in base
    (project / "src" / "app.py").write_bytes(b"v2, edited in the project meanwhile\n")
    (cage / "src" / "new.py").write_bytes(b"new\n")
    assert _changes(base, cage) == (["src/new.py"], [], [])
    assert apply_cage(base, cage, project)["added"] == ["src/new.py"]
    assert (project / "src" / "app.py").read_bytes() == b"v2, edited in the project meanwhile\n"


def test_another_conversations_archive_is_refused_however_its_path_is_spelled(tmp_path):
    """Each chat sees only its own archive (.nesicode/chats/<id>), also in a project folder they share: on Windows a
    path spelled with '\\' — or, there and on macOS, in other case — is the same folder and must be refused too."""
    from nesicode.agent import Agent
    from nesicode.harness import base_harness
    from nesicode.llm import ToolCall, Usage

    ws = Workspace(tmp_path / "ws")
    (ws.root / ".nesicode" / "chats" / "5").mkdir(parents=True)
    (ws.root / ".nesicode" / "chats" / "5" / "history.md").write_text("secreto\n", encoding="utf-8")
    agent = Agent(model="mock/demo", harness=base_harness(), workspace=ws, archive=".nesicode/chats/4")
    other = [".nesicode/chats/5/history.md", ".NESICODE/Chats/5/history.md"]
    if os.name == "nt":
        other += [".nesicode\\chats\\5\\history.md", ".nesicode/chats\\5/history.md", ".Nesicode\\\\Chats\\5"]
    for path in other:
        assert agent._other_chat_archive({"path": path}) == ".nesicode/chats/5", path
    assert agent._other_chat_archive({"command": "cat .nesicode/chats/*/history.md"}) == ".nesicode/chats/*"
    for own in (".nesicode/chats/4/context/history.md", ".nesicode/context/x.md", "src/chats/5.py"):
        assert agent._other_chat_archive({"path": own}) is None, own
    path = ".nesicode\\chats\\5\\history.md" if os.name == "nt" else ".NESICODE/Chats/5/history.md"
    r = agent._call_tool(ToolCall("x", "read_file", {"path": path}), Usage())
    assert r.is_error and "another conversation" in r.content and "secreto" not in r.content
