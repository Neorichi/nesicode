"""v0.4.7 review of live runs (nesi3, 131k window) — the file tools, offline.

edit_file shows the edited code, numbered as read_file shows it, so an indentation slip is seen at once. read_file's
continuation hint asks for what fits: the whole rest of the file when it fits in one call. The base harness's 20k
truncate hook for every tool no longer shrinks read_file pages below their own budget (~24k chars on a 128k window).
search prints lines as ripgrep does (`path:line:content`, indentation kept): the model copies them into `old`."""

from __future__ import annotations

import re

from nesicode.agent import Agent, Workspace
from nesicode.agent import context as ctxm
from nesicode.agent import tools
from nesicode.agent.hooks import Hooks
from nesicode.agent.tools import edit_file, read_file, search
from nesicode.harness import base_harness
from nesicode.llm.base import ToolCall, Usage


def _view(out: str) -> list[str]:
    head, *rows = out.split("\n")
    assert head.endswith(". The changed part now reads:"), out
    return rows


# ====================================================================== edit_file shows the result
def test_edit_file_shows_the_edited_lines_as_read_file_numbers_them(tmp_path):
    # the bugfix run: a loop rewritten with its blank-line guard after the unpack, never looked at again
    ws = Workspace(tmp_path / "ws")
    src = ["import csv", "", "", "def parse(lines):", "    out = []", "    for raw in lines:", "        if not raw.strip():",
           "            continue", "        name, qty = raw.split(',')", "        out.append((name, int(qty)))",
           "    return out", "", "", "def total(rows):", "    return sum(q for _, q in rows)"]
    (ws.root / "parser.py").write_text("\n".join(src) + "\n")
    out = edit_file(ws, {"path": "parser.py",
                         "old": "        if not raw.strip():\n            continue\n        name, qty = raw.split(',')\n",
                         "new": "        name, qty = raw.split(',')\n        if not raw.strip():\n            continue\n"})
    assert out.startswith("edited parser.py (1 replacement(s)). The changed part now reads:\n")
    # lines 7-9 changed: the 4 lines before and the 4 after, exactly as read_file shows them
    page = read_file(ws, {"path": "parser.py", "offset": 3, "limit": 11})
    assert _view(out) == page.split("\n")[:11] and _view(out)[4] == "     7\t        name, qty = raw.split(',')"
    # at the start and at the end of the file
    (ws.root / "abc.txt").write_text("a\nb\nc\n")
    assert _view(edit_file(ws, {"path": "abc.txt", "old": "a\n", "new": "A\n"})) == ["     1\tA", "     2\tb", "     3\tc"]
    assert _view(edit_file(ws, {"path": "abc.txt", "old": "c\n", "new": ""})) == ["     1\tA", "     2\tb"]
    assert _view(edit_file(ws, {"path": "abc.txt", "old": "A\nb\n", "new": ""})) == ["(the file is now empty)"]
    # CRLF files: lines without their \r, as read_file shows them
    (ws.root / "win.txt").write_bytes(b"uno\r\ndos\r\ntres\r\n")
    assert _view(edit_file(ws, {"path": "win.txt", "old": "dos", "new": "DOS"})) == ["     1\tuno", "     2\tDOS", "     3\ttres"]
    assert (ws.root / "win.txt").read_bytes() == b"uno\r\nDOS\r\ntres\r\n"


def test_edit_file_view_is_short_and_keeps_both_ends_of_a_long_change(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "run.py").write_text("def run(data):\n    pass\n    return data\n\n\nAFTER = 1\n")
    body = "".join(f"    step_{i} = compute_{i}(data)\n" for i in range(80))
    out = edit_file(ws, {"path": "run.py", "old": "    pass\n", "new": body})
    view = "\n".join(_view(out))
    assert len(view) <= tools.EDIT_VIEW
    assert view.startswith("     1\tdef run(data):\n     2\t    step_0 = compute_0(data)")  # where the change starts…
    assert view.endswith("    81\t    step_79 = compute_79(data)\n    82\t    return data\n    83\t\n    84\t\n"
                         "    85\tAFTER = 1")  # …and where it meets the code after it
    assert re.search(r"\n\.\.\. \(lines \d+–\d+ not shown\)\n", view)
    # replace_all: the first places, each with its own lines around it, and how many more
    (ws.root / "cfg.py").write_text("".join(f"PORT_{i} = 8765\n" + "x = 0\n" * 12 for i in range(5)))
    out = edit_file(ws, {"path": "cfg.py", "old": "8765", "new": "9876", "replace_all": True})
    rows = _view(out)
    assert out.startswith("edited cfg.py (5 replacement(s))") and rows[-1] == "... (2 more replacement(s) not shown)"
    assert [r for r in rows if "PORT_" in r] == ["     1\tPORT_0 = 9876", "    14\tPORT_1 = 9876", "    27\tPORT_2 = 9876"]
    assert rows.count("...") == 2 and len("\n".join(rows)) <= tools.EDIT_VIEW
    # a change deep in a minified line: the stretch around it, not the line's head
    (ws.root / "app.min.js").write_text("var a=1;" * 2_000 + "oldName()" + ";var b=2" * 2_000 + "\n")
    rows = _view(edit_file(ws, {"path": "app.min.js", "old": "oldName()", "new": "newName()"}))
    assert len(rows) == 1 and rows[0].startswith("     1\t…") and "newName()" in rows[0]
    assert len(rows[0]) <= 8 + tools.EDIT_LINE + 2


# ====================================================================== read_file asks for what fits
def _log(ws: Workspace, rel: str, sizes: list[int]) -> None:
    (ws.root / rel).parent.mkdir(parents=True, exist_ok=True)
    (ws.root / rel).write_text("".join(f"2026-10-16T08:00:{i % 60:02d}Z INFO line {i} " + "k" * n + "\n"
                                       for i, n in enumerate(sizes)))


def test_read_file_hint_reads_the_rest_in_one_call_when_it_fits(tmp_path):
    # the needle run: 397 lines read in pages of 198 (as hinted), then a 2-line call for the last 109-char line
    ws = Workspace(tmp_path / "ws")
    _log(ws, "logs/search-indexer.log", [70] * 397)
    args = {"path": "logs/search-indexer.log"}
    full = read_file(ws, args)
    limit = len(full) * 6 // 10
    cut = ctxm.inline_output(full, ".nesicode/outputs/x.txt", limit, "read_file", args)
    first = int(re.search(r"lines (\d+)–397 are NOT shown", cut).group(1))
    assert ("The rest fits in one call: read_file(path=\"logs/search-indexer.log\", "
            f"offset={first}) reads to the end; full output saved to .nesicode/outputs/x.txt]") in cut
    rest = read_file(ws, {**args, "offset": first})
    assert ctxm.inline_output(rest, ".nesicode/outputs/y.txt", limit, "read_file", args) == rest  # whole, not cut
    assert rest.endswith("k" * 70) and rest.split("\n")[-1].startswith("   397\t")


def test_read_file_hint_asks_for_the_lines_that_fit_not_the_last_page_size(tmp_path):
    # short lines first, long ones after: a next page "the size of this one" would be cut again
    ws = Workspace(tmp_path / "ws")
    _log(ws, "app.log", [5] * 170 + [300] * 400)
    args = {"path": "app.log"}
    full = read_file(ws, args)
    cut = ctxm.inline_output(full, ".nesicode/outputs/x.txt", 8_000, "read_file", args)
    m = re.search(r"showing lines 1–(\d+) of 570; lines (\d+)–570 are NOT shown\. Continue with "
                  r"read_file\(path=\"app\.log\", offset=(\d+), limit=(\d+)\); full output saved to", cut)
    assert m, cut[-400:]
    shown, offset, limit = int(m.group(1)), int(m.group(3)), int(m.group(4))
    assert offset == shown + 1 and limit < shown
    page = read_file(ws, {**args, "offset": offset, "limit": limit})
    assert len(page) <= 8_000 and ctxm.inline_output(page, "o.txt", 8_000, "read_file", args) == page
    assert len(read_file(ws, {**args, "offset": offset, "limit": limit + 1})) > 8_000 - 400  # (and no smaller than that)


def test_the_truncate_hook_for_every_tool_never_shrinks_read_file_pages(tmp_path):
    # the base harness's {"tool": "*", "action": "truncate", "max_chars": 20000} made each page 19.4k chars at 131k
    ws = Workspace(tmp_path / "ws")
    _log(ws, "logs/search-indexer.log", [70] * 397)  # ~43k chars: two calls at 24k
    a = Agent(model="mock/demo", harness=base_harness(), workspace=ws, context_limit=131_072)
    assert a._inline_limit("read_file") == ctxm.inline_limit(131_072) == 24_000
    assert a._inline_limit("bash") == 19_400  # other tools: the hook as written
    assert "about 24,000 characters" in next(s.description for s in a.tool_specs() if s.name == "read_file")
    r = a._call_tool(ToolCall("c1", "read_file", {"path": "logs/search-indexer.log"}), Usage())
    assert 22_000 < len(r.content) <= 24_000 and "trimmed by hook" not in r.content
    first = int(re.search(r"lines (\d+)–397 are NOT shown\. The rest fits in one call", r.content).group(1))
    r = a._call_tool(ToolCall("c2", "read_file", {"path": "logs/search-indexer.log", "offset": first}), Usage())
    assert "NOT shown" not in r.content and r.content.split("\n")[-1].startswith("   397\t")
    # the hooks themselves: a page within read_file's own budget passes, any other output is cut as written
    hooks = Hooks({"post_tool": [{"tool": "*", "action": "truncate", "max_chars": 20_000}]})
    page = "x" * 24_000
    assert hooks.after("read_file", page) == page and hooks.truncate_limit("read_file") == 24_600
    assert "trimmed by hook" in hooks.after("bash", page) and hooks.truncate_limit("bash") == 20_000
    assert "trimmed by hook" in hooks.after("read_file", "x" * 30_000)  # (never unbounded)
    named = Hooks({"post_tool": [{"tool": ["read_file"], "action": "truncate", "max_chars": 5_000}]})
    assert named.truncate_limit("read_file") == 5_000 and len(named.after("read_file", page)) <= 5_000  # as written
    assert Hooks({}).truncate_limit("read_file") is None


# ====================================================================== search prints lines as they are
def test_search_prints_lines_as_ripgrep_does_and_they_round_trip_into_edit_file(tmp_path):
    # the multiturn run: "notas.md:5: - Puerto…" (indentation stripped, a space added) was copied into `old`: not found
    ws = Workspace(tmp_path / "ws")
    (ws.root / "notas.md").write_text("# Notas\n\n- Puerto por defecto 8765 (regla 1)\n   - Prueba de humo en el 8765\n")
    (ws.root / "app.py").write_text("class A:\n    def run(self):\n\t\treturn 8765  \n")
    out = search(ws, {"pattern": "8765"})
    assert out.split("\n") == ["app.py:3:\t\treturn 8765  ", "notas.md:3:- Puerto por defecto 8765 (regla 1)",
                               "notas.md:4:   - Prueba de humo en el 8765"]
    for hit in out.split("\n"):
        rel, _, line = hit.split(":", 2)
        assert edit_file(ws, {"path": rel, "old": line, "new": line.replace("8765", "9876")}).startswith(f"edited {rel} ")
    assert search(ws, {"pattern": "8765"}) == "no matches"
    # a long line of prose is shown whole; only a minified one is clipped, around the match
    para = "Lorem ipsum dolor " * 20 + "NEEDLE" + " sit amet" * 10
    (ws.root / "doc.md").write_text(para + "\n")
    assert search(ws, {"pattern": "NEEDLE", "path": "doc.md"}) == f"doc.md:1:{para}"
    (ws.root / "min.js").write_text("var a=1;" * 2_000 + "TARGET_FN()" + ";var b=2" * 2_000 + "\n")
    hit = search(ws, {"pattern": "TARGET_FN", "path": "min.js"})
    assert hit.startswith("min.js:1:…") and hit.endswith("…") and "TARGET_FN()" in hit
    assert len(hit) <= len("min.js:1:") + tools.SEARCH_LINE + 2


def test_a_multi_line_match_prints_each_of_its_lines(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "h.py").write_text("x = 1\ndef handler(event):\n    if event:\n        log(event)\n    return None\n")
    assert search(ws, {"pattern": r"(?s)def handler.*?return None"}).split("\n") == [
        "h.py:2:def handler(event):", "h.py:3:    if event:", "h.py:4:        log(event)", "h.py:5:    return None"]
    (ws.root / "h.py").write_text("BEGIN\n" + "".join(f"    row {i}\n" for i in range(10)) + "END\n")
    assert search(ws, {"pattern": r"BEGIN\n(.*\n)*?END"}).split("\n") == [
        "h.py:1:BEGIN", "h.py:2:    row 0", "h.py:3:    row 1", "... (lines 4–11 of this match not shown)", "h.py:12:END"]
