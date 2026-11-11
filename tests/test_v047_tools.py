"""v0.4.7 review of the agent's file tools — offline.

search: no regex the model writes can freeze the server (re holds the GIL, and agents are threads of the server), and
it never answers "no matches" for what it did not search. list_dir lists a whole level before going deeper. read_file
reads a PDF without following a link swapped in. edit_file never deletes because `new` was lost, and a miss says what
went wrong and where. write_file never overwrites what the run has not read. The sandbox note never fires on the
agent's own files. A regex that would freeze the interpreter runs in a child process with a timeout, so a regression
fails instead of hanging the suite."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nesicode.agent import ToolError, Workspace
from nesicode.agent import tools
from nesicode.agent.tools import _refused_by_sandbox, bash, edit_file, list_dir, read_file, search, write_file

posix_only = pytest.mark.skipif(os.name == "nt", reason="symbolic links")
macos_only = pytest.mark.skipif(not (sys.platform == "darwin" and tools.sandbox_mode() == "macos"),
                                reason="macOS sandbox-exec cage")
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

CHILD = """import json, sys, threading, time
from nesicode.agent import Workspace, tools
args, opts = json.loads(sys.argv[2]), json.loads(sys.argv[3])
ws = Workspace(sys.argv[1])
tools.SEARCH_BUDGET = opts.get("budget", tools.SEARCH_BUDGET)
if "stop" in opts:
    ws.cancel = threading.Event()
    threading.Timer(opts["stop"], ws.cancel.set).start()
ticks, done = [0], threading.Event()
def tick():
    while not done.is_set():
        ticks[0] += 1
        time.sleep(0.01)
threading.Thread(target=tick, daemon=True).start()
t0 = time.monotonic()
out = tools.search(ws, args)
took = time.monotonic() - t0
done.set()
print(json.dumps([took, out, ticks[0]]))
"""


def _search_in_child(root, args: dict, opts: dict | None = None, timeout: float = 30.0):
    """search in a child process: (seconds it took, its output, ticks of another thread meanwhile); None if it was
    still running after `timeout` (the server would be frozen)."""
    import nesicode

    src = os.path.dirname(os.path.dirname(nesicode.__file__))  # the child imports the very code under test
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, str(root), json.dumps(args), json.dumps(opts or {})],
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def _files_hit(out: str) -> list[str]:
    return sorted({line.split(":", 1)[0] for line in out.splitlines() if re.match(r"[^\s(\[]+:\d", line)})


def _pdf(text: str) -> bytes:
    """A one-page PDF with a text layer."""
    stream = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


# ====================================================================== tools-2: no regex can freeze the server
@pytest.mark.parametrize("pattern", [r"(.|\s)*TODO", r"(\s|.)*?TODO", r"<div(.|\s)*?>x"])
def test_the_any_character_idiom_never_freezes_search(tmp_path, pattern):
    """`(.|\\s)*` (JavaScript's "anything, line breaks too") passed the guard: 2^N backtracking on a line with N spaces,
    with the GIL held — hours on an ordinary 30-word line, and every conversation, the HTTP loop and Stop frozen."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / "README.md").write_text("# Notas\n<div" + " " * 30 + "\n" + " ".join(["palabra"] * 32) + "\n")
    res = _search_in_child(root, {"pattern": pattern})
    assert res is not None, f"search({pattern!r}) was still running after 30 s: the whole server would be frozen"
    took, out, _ = res
    assert took < 5 and out.startswith("no matches"), (took, out)


def test_stop_and_the_deadline_end_a_pattern_that_backtracks_forever_and_other_threads_keep_running(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("a" * 60 + "!\n")  # `(a|aa)+$` here takes longer than the age of the universe
    res = _search_in_child(root, {"pattern": r"(a|aa)+$"}, {"stop": 0.5})
    assert res is not None, "Stop did not end the search"
    took, out, ticks = res
    assert took < 4 and out.endswith("[search stopped by the user]"), (took, out)
    assert ticks > took / 0.01 / 4, f"other threads ran {ticks} times in {took:.1f} s: the GIL was held"
    res = _search_in_child(root, {"pattern": r"(a|aa)+$"}, {"budget": 1.0})
    assert res is not None, "the search budget did not end the search"
    took, out, _ = res
    assert took < 5 and "[search stopped after 1 s" in out and "the pattern is slow on a.txt" in out, (took, out)


def test_patterns_run_in_this_process_only_when_provably_cheap():
    from nesicode.agent.tools import _cheap

    for pat in ("TODO", r"def \w+\(", r"def \w+\(.*\):", r"foo.*bar", r"function\s+\w+", r"\w+\s*=\s*\d+",
                "useState|useEffect", r"https?://\S+", r"(get|set)_\w+", r"^\s*import ", r"(?<=def )\w+"):
        assert _cheap(pat), pat
    for pat in (r"foo.*bar.*baz", r"(ab)+x", r"(a|aa)+$", r"(\w)\1", r"[a-z ]*\s+=", r"(?=.*foo).*bar",
                r"\w+日\w+本x"):  # (`\w` matches 日: the first `\w+` gives characters back to what follows)
        assert not _cheap(pat), pat


def test_overlapping_alternations_are_rewritten_or_refused(tmp_path):
    from nesicode.agent.tools import _nested_repeat, _search_regex

    assert _search_regex(r"(.|\s)*TODO").pattern == "()TODO"  # a leading run of anything adds only backtracking
    assert _search_regex(r"foo(?:.|\s)*?bar").pattern == r"foo(?:[\s\S])*?bar"
    assert _search_regex(r"[(.|\s)]x").pattern == r"[(.|\s)]x"  # (inside a class: not a group)
    assert _search_regex(r"([\s\S])*(foo)\2").pattern == r"()(foo)\2"  # group numbers kept
    for pat in (r"(.|\s)*x", r"(\\.|[^\"])*\"", r"(<[^>]+>|.)*x"):  # (search_history and hook rules use this guard)
        assert _nested_repeat(pat), pat
    for pat in (r"(foo|bar)+x", r"(?:[^\"\\]|\\.)*\"", r"(.|\n)*x", r"(\w|-)+x"):
        assert not _nested_repeat(pat), pat
    ws = Workspace(tmp_path / "ws")
    with pytest.raises(ToolError, match="exponential"):
        search(ws, {"pattern": r'"(\\.|[^"])*"x'})


# ====================================================================== tools-4: never "no matches" for what was not searched
def test_search_never_says_no_matches_for_what_it_did_not_search(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "src" / "util").mkdir(parents=True)
    (ws.root / "src" / "util" / "dates.py").write_text("def parse_date(s):\n    return s\n")
    with pytest.raises(ToolError, match=r"path not found: src/utils \(did you mean `src/util`\?\)"):
        search(ws, {"pattern": "def parse_date", "path": "src/utils"})
    # ripgrep's exclusions, alone or in a list, for files and folders
    (ws.root / "vendor").mkdir()
    for name in ("app.js", "vendor.min.js", "vendor/lib.js"):
        (ws.root / name).write_text("const apiKey = 1;\n")
    assert _files_hit(search(ws, {"pattern": "apiKey", "glob": "!*.min.js"})) == ["app.js", "vendor/lib.js"]
    assert _files_hit(search(ws, {"pattern": "apiKey", "glob": "*.js,!*.min.js,!vendor"})) == ["app.js"]
    # patterns that span lines search the whole file
    (ws.root / "h.py").write_text("x = 1\ndef handler(event):\n    return None\n")
    for args in ({"pattern": r"def handler\(event\):\n\s+return None"}, {"pattern": r"(?s)def handler.*?return None"},
                 {"pattern": r"handler\(event\):\s+return", "multiline": True}):
        assert search(ws, args) == "h.py:2:def handler(event):\nh.py:3:    return None", args
    out = search(ws, {"pattern": r"handler\(event\):\s+return"})
    assert out.startswith("no matches") and "multiline=true" in out


def test_a_no_matches_names_the_attachments_and_what_gitignore_skipped(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / ".attachments").mkdir()
    (ws.root / ".attachments" / "spec.txt").write_text("FRASE_CLAVE\n")
    (ws.root / "dist").mkdir()
    (ws.root / "dist" / "bundle.js").write_text("FRASE_CLAVE\n")
    (ws.root / "server.log").write_text("FRASE_CLAVE\n")
    (ws.root / "app.py").write_text("print(1)\n")
    (ws.root / ".gitignore").write_text("# build output\ndist/\n*.log\n")
    out = search(ws, {"pattern": "FRASE_CLAVE"})
    assert out.startswith("no matches"), out
    assert ".attachments/ (the files the user attached)" in out and "in .gitignore: dist/, server.log" in out, out
    assert search(ws, {"pattern": "FRASE_CLAVE", "path": ".attachments"}).startswith(".attachments/spec.txt:1:")
    assert search(ws, {"pattern": "FRASE_CLAVE", "path": "dist"}).startswith("dist/bundle.js:1:")  # named: searched
    assert search(ws, {"pattern": "FRASE_CLAVE", "path": "server.log"}).startswith("server.log:1:")
    listing = list_dir(ws, {})
    assert "dist/ (in .gitignore, not expanded)" in listing and "server.log" in listing
    assert "  bundle.js" not in listing


def test_gitignore_rules():
    from nesicode.agent.tools import _gitignore

    class _Ws:
        root = Path("/w")

        @staticmethod
        def read_bytes(p, limit=None):
            return b"*.log\n!keep.log\nbuild/\n/dist\ndocs/_build/\n**/tmp\n\\#notes\n# a comment\n*.py[co]\n"

    ignored = _gitignore(_Ws())
    for rel, is_dir in (("a.log", False), ("x/y/a.log", False), ("build", True), ("pkg/build", True), ("dist", True),
                        ("docs/_build", True), ("tmp", True), ("a/b/tmp", True), ("#notes", False), ("m/x.pyc", False)):
        assert ignored(rel, is_dir), rel
    for rel, is_dir in (("keep.log", False), ("build", False), ("src/dist", True), ("x/docs/_build", True),
                        ("a.py", False), ("notes", False)):
        assert not ignored(rel, is_dir), rel


def test_search_says_it_stopped_when_the_deadline_ends_inside_the_last_file(tmp_path, monkeypatch):
    import threading

    ws = Workspace(tmp_path / "ws")
    (ws.root / "big.txt").write_text(("ab" * 2000 + "\n") * 450)  # one file, a few ms per line
    monkeypatch.setattr(tools, "SEARCH_BUDGET", 0.1)
    out = search(ws, {"pattern": "a.*xy"})
    assert out.startswith("no matches") and "[search stopped after" in out, out  # was a plain "no matches"
    ws.cancel = threading.Event()
    ws.cancel.set()
    t0 = time.monotonic()
    assert search(ws, {"pattern": "a.*b.*xy"}).endswith("[search stopped by the user]")  # (and no process started)
    assert time.monotonic() - t0 < 2


# ====================================================================== tools-6: list_dir is breadth-first
def test_list_dir_lists_a_whole_level_before_going_deeper(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "assets").mkdir()
    for i in range(450):
        (ws.root / "assets" / f"icon{i:03}.svg").write_text("<svg/>")
    for d in ("src", "tests"):
        (ws.root / d).mkdir()
        (ws.root / d / "main.py").write_bytes(b"x = 1\n")  # (bytes: Windows text mode adds \r)
    (ws.root / "README.md").write_bytes(b"# hi\n")
    (ws.root / "package.json").write_text("{}")
    out = list_dir(ws, {})
    for line in ("assets/ (450 entries, not expanded)", "src/", "  main.py (6 B)", "tests/", "README.md (5 B)",
                 "package.json (2 B)"):
        assert line in out.splitlines(), (line, out[-600:])
    assert out.splitlines()[-1].startswith("... (1 folder(s) not expanded") and "assets/" in out.splitlines()[-1]
    # an app/ of 20 routes x 20 files at depth 3: still the whole top level, and most routes
    ws = Workspace(tmp_path / "ws2")
    for r in range(20):
        (ws.root / "app" / f"route{r:02}").mkdir(parents=True)
        for i in range(20):
            (ws.root / "app" / f"route{r:02}" / f"page{i:02}.tsx").write_text("x")
    (ws.root / "components").mkdir()
    (ws.root / "components" / "Button.tsx").write_text("x")
    (ws.root / "package.json").write_text("{}")
    lines = list_dir(ws, {"depth": 3}).splitlines()
    assert "package.json (2 B)" in lines and "components/" in lines and "  Button.tsx (1 B)" in lines
    assert len(lines) <= 402 and any("entries, not expanded" in line for line in lines)


# ====================================================================== tools-8: a PDF is read without following links
@posix_only
def test_read_file_never_reads_a_pdf_through_a_link_swapped_in_after_resolve(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "ws")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "tax-return.pdf").write_bytes(_pdf("TOP-SECRET-OUTSIDE"))
    (ws.root / "doc.pdf").write_bytes(_pdf("public text"))
    assert "public text" in read_file(ws, {"path": "doc.pdf"})
    real = Workspace.resolve

    def swapped(self, rel):  # a background process wins the race right after the check
        p = real(self, rel)
        if p.name == "doc.pdf":
            p.unlink()
            os.symlink(outside / "tax-return.pdf", p)
        return p

    monkeypatch.setattr(Workspace, "resolve", swapped)
    with pytest.raises(ToolError, match="symbolic link"):
        read_file(ws, {"path": "doc.pdf"})


# ====================================================================== tools-9: the sandbox note and the agent's own files
def test_the_sandbox_note_never_fires_for_the_agents_own_files(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "sandbox_mode", lambda: "macos")
    cage = "/Users/me/Library/Application Support/nesicode/data/workspaces/1/2/threads/3/ws"
    enoent = "FileNotFoundError: [Errno 2] No such file or directory: 'data/input.csv'"
    assert not _refused_by_sandbox(f'cd "{cage}" && python3 scripts/build.py', enoent, [], [cage])
    assert not _refused_by_sandbox(f"cat {cage.replace(' ', chr(92) + ' ')}/data/input.csv", enoent, [], [cage])
    assert not _refused_by_sandbox("python3 Documents/report.py", enoent)  # the project's own Documents/ folder
    assert not _refused_by_sandbox("ls apps/Desktop/src/missing", "ls: apps/Desktop/src/missing: No such file or directory")
    # the user's own folders still get it
    assert _refused_by_sandbox("ls ~/Documents/report.pdf", "ls: report.pdf: No such file or directory", [], [cage])
    assert _refused_by_sandbox("cat /Users/me/Desktop/notes.txt", "cat: No such file or directory", [], [cage])
    assert _refused_by_sandbox("ls $HOME/Downloads", "ls: Downloads: Operation not permitted", [], [cage])
    ws = Workspace(tmp_path / "ws")
    assert str(ws.root) in tools._own_paths(ws)


@macos_only
def test_a_missing_file_in_a_project_folder_named_documents_is_just_missing(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "Documents").mkdir()
    out = bash(ws, {"command": "python3 Documents/report.py"})
    assert "No such file or directory" in out and "[sandbox]" not in out, out


# ====================================================================== tools-10: a lost `new` never deletes
def test_edit_file_never_deletes_because_new_was_lost(tmp_path):
    ws = Workspace(tmp_path / "ws")
    f = ws.root / "f.py"
    f.write_text("def a():\n    return compute_total(items)\n")
    line = "    return compute_total(items)\n"
    for args in ({"path": "f.py", "old": line}, {"path": "f.py", "old": line, "new": None},
                 {"path": "f.py", "old": line, "replacement": "    return 0\n"},
                 {"file_path": "f.py", "old": line, "new": "    return 0\n"}):
        with pytest.raises(ToolError, match=r"missing `new`|unknown parameter") as e:
            edit_file(ws, args)
        assert "edit_file takes `path`, `old`" in str(e.value), args
    assert f.read_text() == "def a():\n    return compute_total(items)\n"
    # Claude Code's names are the same parameters
    assert edit_file(ws, {"path": "f.py", "old_string": "compute_total(items)", "new_string": "total(items)"}) \
        .startswith("edited f.py (1 replacement(s))")
    assert edit_file(ws, {"path": "f.py", "old": "total(", "new_string": "sum(", "replace_all": True}).startswith("edited")
    assert f.read_text() == "def a():\n    return sum(items)\n"
    with pytest.raises(ToolError, match="both `new` and `new_string`"):
        edit_file(ws, {"path": "f.py", "old": "sum", "new": "max", "new_string": "min"})


# ====================================================================== harness-1: an edit_file miss says what to do
def test_edit_file_misses_say_what_went_wrong_and_where(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "a.py").write_text("def a():\n\treturn 1\n")
    (ws.root / "b.py").write_text("def b():\n    return compute_total(items)\n")
    (ws.root / "c.py").write_text("x = 1\ny = 2\nx = 1\n")
    copied = read_file(ws, {"path": "a.py"}).splitlines()[1]  # '     2\t\treturn 1', prefix and all
    misses = []
    with pytest.raises(ToolError) as e:
        edit_file(ws, {"path": "a.py", "old": copied, "new": "\treturn 2"})
    misses.append(str(e.value))
    assert "in a.py" in misses[-1] and "line-number prefix" in misses[-1] and "without it, the text is in the file" in \
        misses[-1]
    with pytest.raises(ToolError) as e:  # spaces typed for a tab
        edit_file(ws, {"path": "a.py", "old": "    return 1", "new": "    return 2"})
    assert "Lines 2-2 match it except for whitespace" in str(e.value) and "2: '\\treturn 1'" in str(e.value)
    with pytest.raises(ToolError) as e:  # one stale character
        edit_file(ws, {"path": "b.py", "old": "    return compute_totl(items)", "new": "x"})
    misses.append(str(e.value))
    assert "Closest lines" in misses[-1] and "2: '    return compute_total(items)'" in misses[-1]
    assert "shorter `old`" in misses[-1] and "write_file" in misses[-1]
    with pytest.raises(ToolError) as e:
        edit_file(ws, {"path": "c.py", "old": "z = 3", "new": "z = 4"})
    misses.append(str(e.value))
    # misses on different files are different errors (the loop stops offering a tool after 3 identical ones in a row)
    assert len({re.sub(r"\d+", "#", m[:160]) for m in misses}) == 3, misses
    with pytest.raises(ToolError, match=r"`old` occurs 2 times in c\.py \(lines 1, 3\)"):
        edit_file(ws, {"path": "c.py", "old": "x = 1", "new": "x = 3"})
    assert (ws.root / "a.py").read_text() == "def a():\n\treturn 1\n"


# ====================================================================== read before write
def test_write_file_never_overwrites_what_this_run_has_not_read(tmp_path):
    ws = Workspace(tmp_path / "ws")
    f = ws.root / "config.py"
    f.write_text("A = 1\nB = 2\n")
    with pytest.raises(ToolError, match=r"config\.py already exists \(2 lines\) and you have not read it in this run"):
        write_file(ws, {"path": "config.py", "content": "A = 1\n"})
    assert f.read_text() == "A = 1\nB = 2\n"
    read_file(ws, {"path": "config.py", "limit": 1})  # (any read counts)
    assert write_file(ws, {"path": "config.py", "content": "A = 1\nB = 3\n"}).startswith("overwrote")
    assert write_file(ws, {"path": "config.py", "content": "A = 2\nB = 3\n"}).startswith("overwrote")  # its own write
    f.write_text("A = 2\nB = 3\nC = 4\n")  # a formatter, the user, another process…
    with pytest.raises(ToolError, match="changed on disk since you last read or wrote it"):
        write_file(ws, {"path": "config.py", "content": "A = 9\n"})
    out = edit_file(ws, {"path": "config.py", "old": "C = 4", "new": "C = 5"})  # `old` anchors an edit: done, and said
    assert out.startswith("edited config.py (1 replacement(s))") and "changed on disk since you last read it" in out
    assert write_file(ws, {"path": "config.py", "content": "A = 3\n"}).startswith("overwrote")
    # new files, empty files and binaries need no read
    assert write_file(ws, {"path": "new.py", "content": "x\n"}).startswith("created")
    (ws.root / "empty.txt").write_text("")
    assert write_file(ws, {"path": "empty.txt", "content": "y\n"}).startswith("overwrote")
    (ws.root / "blob.bin").write_bytes(b"\0\1\2")
    assert write_file(ws, {"path": "blob.bin", "content": "text now\n"}).startswith("overwrote")
    # the next run (a new Workspace) has read nothing yet
    with pytest.raises(ToolError, match="not read it in this run"):
        write_file(Workspace(ws.root), {"path": "new.py", "content": "z\n"})


# ====================================================================== small messages
def test_read_file_says_what_the_path_is(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "src" / "util").mkdir(parents=True)
    (ws.root / "src" / "util" / "dates.py").write_text("x = 1\n")
    (ws.root / "logo.png").write_bytes(PNG)
    with pytest.raises(ToolError, match="'src' is a folder, not a file: list_dir"):
        read_file(ws, {"path": "src"})
    with pytest.raises(ToolError, match=r"file not found: src/util/date\.py \(did you mean `src/util/dates\.py`\?\)"):
        read_file(ws, {"path": "src/util/date.py"})
    with pytest.raises(ToolError, match="view_image if that tool is in your list"):
        read_file(ws, {"path": "logo.png"})
    with pytest.raises(ToolError, match="is a file, not a folder"):
        list_dir(ws, {"path": "logo.png"})
