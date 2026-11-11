"""v0.4.7 round 2, group loopctx — what the harness tells the model about its context, and when.

Live review on nesi3 (131k window, no step cap): the «write your findings down» reminder arrived in the same round as the
masking that cleared what it was about (the model then rebuilt its findings from memory, inventing code), the markers
kept nothing of the code they replaced (every masking was followed by whole-file re-reads), the notes offered «the
deliverable or a notes file» as equals although the user had named REVIEW.md, and neither those notes nor a reply cut in
its reasoning at the output cap left a trace in the run log.

Offline: the real OpenAI-compatible conversation (its masking, its notes riding with the newest tool result, reasoning
replayed as nesi3 needs) answered by a scripted policy; temp workspaces."""

from __future__ import annotations

import json
import random

import pytest

from nesicode.agent import Agent, Workspace
from nesicode.agent import context as ctxm
from nesicode.agent import loop as loopmod
from nesicode.agent.tools import read_file
from nesicode.harness import base_harness
from nesicode.llm.base import ToolCall, ToolResult
from nesicode.llm.openai_compat import OpenAICompatProvider

WINDOW = 131_072
TASK = ("Haz una revisión de src/. Reglas: no modifiques nada salvo REVIEW.md; añade cada hallazgo a REVIEW.md en cuanto "
        "lo encuentres.")
WARNING = "[harness] The context is filling up"
REMINDER = "tool calls in a row without writing"
NAMED = "in `REVIEW.md`, the file the user asked for, rather than in your notes file"


# ====================================================================== nesi3, offline
class Offline(OpenAICompatProvider):
    """The real OpenAI-compatible conversation; chat() answered by `policy(rec)`: prompt tokens = the JSON of messages +
    tools / cpt (a request over the window fails the test)."""

    def __init__(self, policy, window=WINDOW, cpt=3.4):
        super().__init__("nesilabs", "http://127.0.0.1:9/v1")
        self.policy, self.window, self.cpt, self.log = policy, window, cpt, []

    def chat(self, body, options, on_event=None):
        chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in body["messages"]) + len(json.dumps(body.get("tools")))
        rec = {"role": options.role, "input": int(chars / self.cpt), "max_tokens": body.get("max_tokens"),
               "messages": json.loads(json.dumps(body["messages"]))}
        assert rec["input"] + (rec["max_tokens"] or 0) <= self.window, "a request over the window"
        self.log.append(rec)
        msg, finish, out = self.policy(rec)
        return {"choices": [{"message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": rec["input"], "completion_tokens": out}}, False

    def requests(self):
        return [r for r in self.log if r["role"] == "executor"]


@pytest.fixture()
def offline(monkeypatch):
    def install(policy, **kw):
        prov = Offline(policy, **kw)
        monkeypatch.setattr(loopmod, "get_provider", lambda spec, keys=None: (prov, "nesi3"))
        return prov
    return install


def reply(text="", calls=(), thinking="Sigo con la revisión.", finish=None, out=60):
    msg = {"role": "assistant", "content": text}
    if thinking:
        msg["reasoning_content"] = thinking
    if calls:
        msg["tool_calls"] = [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                             for cid, name, args in calls]
    return msg, finish or ("tool_calls" if calls else "stop"), out


def tools(rec):
    return [m for m in rec["messages"] if m["role"] == "tool"]


def newest(rec) -> str:
    """The newest tool result as the model got it (the harness's notes ride with it)."""
    t = tools(rec)
    return t[-1]["content"] if t else ""


def masked(rec) -> list[str]:
    return [m["tool_call_id"] for m in tools(rec) if m["content"].startswith("[old ")]


def agent(ws, events, **kw):
    kw.setdefault("context_limit", WINDOW)
    kw.setdefault("output_limit", 16_384)
    kw.setdefault("max_steps", 0)
    return Agent(model="nesilabs/nesi3", harness=base_harness(), workspace=ws, emit=events.append, **kw)


def module(rnd, classes, methods=5, body=7) -> str:
    """Python source: `classes` classes of `methods` methods of `body` lines (~73 chars a line as read_file shows it)."""
    out = []
    for k in range(classes):
        out.append(f"class Service{k}(Base):")
        for m in range(methods):
            out.append(f"    def handle_{k}_{m}(self, value: int) -> int:")
            out += [f"        value = self.cache.get(value, {rnd.randint(1000, 9999)}) + step(value * {rnd.randint(10, 99)})"
                    for _ in range(body)]
            out.append("        return value")
        out.append("")
    return "\n".join(out) + "\n"


def review_repo(tmp_path, files=12, classes=6):
    rnd = random.Random(7)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "src").mkdir()
    for i in range(files):
        (ws.root / "src" / f"m{i}.py").write_text(module(rnd, classes))
    return ws


# ====================================================================== 1. the reminder before the masking (major)
def test_a_masking_that_would_clear_what_was_read_since_the_last_write_waits_a_round_after_a_warning(tmp_path, offline):
    """Big reads cross the masking threshold long before the 20-call reminder: the model is told what is about to go
    while it still sees it, writes it down, and only then are those outputs cleared."""
    def policy(rec):
        n = len(prov.requests()) - 1
        if n == 0:
            return reply("Creo REVIEW.md.", [("w0", "write_file", {"path": "REVIEW.md", "content": "# Revisión\n"})])
        if WARNING in newest(rec) or REMINDER in newest(rec):
            return reply("Anoto lo que he visto.", [(f"w{n}", "edit_file", {
                "path": "REVIEW.md", "old": "# Revisión\n", "new": f"# Revisión\n- NR-{n} | src/m1.py:3 | media | …\n"})])
        if n >= 16:
            return reply("Revisión terminada: hallazgos en REVIEW.md.")
        return reply("Leo el siguiente módulo.", [(f"r{n}", "read_file", {"path": f"src/m{n % 12}.py"})])

    prov = offline(policy)
    events = []
    res = agent(review_repo(tmp_path), events).run(TASK)
    assert res.status == "done", res.error
    reqs = prov.requests()
    warned = next(i for i, r in enumerate(reqs) if WARNING in newest(r))
    assert REMINDER not in json.dumps(reqs[:warned + 1])  # (well before the 20-call reminder)
    # the warning comes while everything it is about is still in view, and names it and the deliverable
    assert not masked(reqs[warned])
    note = newest(reqs[warned])[newest(reqs[warned]).index(WARNING):]
    assert "src/m1.py lines 1–" in note and "that you read since you last wrote anything" in note and NAMED in note
    # the model wrote its findings down in its reply; the masking came after that, a round late
    after = reqs[warned + 1]
    assert masked(after) and set(masked(after)) <= {f"r{i}" for i in range(1, warned)}
    order = [(e["type"], e.get("id")) for e in events]
    assert order.index(("tool_call", f"w{warned}")) < next(i for i, (t, _) in enumerate(order) if t == "masked")
    # the 'masked' event says which outputs it cleared, and that it had waited
    m = next(e for e in events if e["type"] == "masked")
    assert m["deferred"] is True and [c["id"] for c in m["cleared"]] == masked(after)
    assert all(c["tool"] == "read_file" and c["what"].startswith("src/m") for c in m["cleared"])
    # the note after the masking names the deliverable too; both notes are on record as the model got them
    assert NAMED in newest(after) and "Record any conclusion you have not written down yet" in newest(after)
    notes = [e for e in events if e["type"] == "note"]
    assert [(e["kind"], e.get("trigger")) for e in notes[:2]] == [("read_streak", "context"), ("mask", None)]
    assert notes[0]["text"] == note and notes[1]["text"] in newest(after)


def test_the_read_streak_reminder_never_comes_with_the_masking_it_warns_about(tmp_path, offline):
    """The live case: the 20th read-only call and the masking threshold in the same round (calls #27-28 of the review
    run). The reminder went out with that round's results and the masking cleared the files it was about at once."""
    ws = review_repo(tmp_path, classes=10)

    def lines_for(path, chars):  # the longest read of `path` from line 1 within `chars`, as read_file shows it
        lo, hi = 1, 470
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if len(read_file(ws, {"path": path, "limit": mid})) <= chars else (lo, mid - 1)
        return lo

    def policy(rec):
        n = len(prov.requests()) - 1
        if n == 0:
            return reply("Creo REVIEW.md.", [("w0", "write_file", {"path": "REVIEW.md", "content": "# Revisión\n"})])
        if REMINDER in newest(rec):
            return reply("Anoto lo que he visto.", [(f"w{n}", "edit_file", {
                "path": "REVIEW.md", "old": "# Revisión\n", "new": "# Revisión\n- NR-1 | src/m2.py:12 | alta | …\n"})])
        if n <= 14:  # 14 medium reads: the streak grows, the context stays under the masking threshold
            assert rec["input"] < ctxm.MASK_AT * WINDOW
            return reply("Leo.", [(f"r{n}", "read_file", {"path": f"src/m{n % 12}.py", "offset": 1 + 150 * (n // 12),
                                                             "limit": 150})])
        if n == 15:  # six reads at once: the 20th read-only call, and they take the context to ~53% of the window
            each = int((0.53 * WINDOW - rec["input"]) * ctxm.CHARS_PER_TOKEN / 6) - 100
            assert each < ctxm.inline_limit(WINDOW)  # (whole: none of them is cut)
            return reply("Leo seis más.", [(f"b{k}", "read_file", {"path": f"src/m{k}.py",
                                                                   "limit": lines_for(f"src/m{k}.py", each)})
                                           for k in range(6)])
        if n >= 18:
            return reply("Revisión terminada.")
        return reply("Sigo.", [(f"r{n}", "read_file", {"path": f"src/m{n % 12}.py", "offset": 301, "limit": 50})])

    prov = offline(policy)
    events = []
    res = agent(ws, events).run(TASK)
    assert res.status == "done", res.error
    reqs = prov.requests()
    reminded = next(i for i, r in enumerate(reqs) if REMINDER in newest(r))
    assert reminded == 16
    assert not masked(reqs[reminded])  # before: the reminder arrived with the outputs already cleared
    assert WARNING not in json.dumps(reqs[reminded])  # one note is enough
    assert masked(reqs[reminded + 1])
    m = next(e for e in events if e["type"] == "masked")
    assert m["deferred"] is True and {c["id"] for c in m["cleared"]} <= {f"r{i}" for i in range(1, 15)}
    order = [(e["type"], e.get("id")) for e in events]
    assert order.index(("tool_call", f"w{reminded}")) < order.index(("masked", None))
    note = next(e for e in events if e["type"] == "note")
    assert (note["kind"], note["trigger"]) == ("read_streak", "calls") and NAMED in note["text"]


def test_a_masking_never_waits_twice_nor_near_the_compaction_threshold(tmp_path, offline):
    """A model that reads on past the warning still gets its context cleared the next round; and a run already close to
    the compaction trigger masks at once (a summary of everything would otherwise come instead)."""
    def policy(rec):
        n = len(prov.requests()) - 1
        if n >= 14:
            return reply("Fin.")
        return reply("Leo.", [(f"r{n}", "read_file", {"path": f"src/m{n % 12}.py"})])

    prov = offline(policy)
    events = []
    agent(review_repo(tmp_path), events).run(TASK)
    reqs = prov.requests()
    warned = next(i for i, r in enumerate(reqs) if WARNING in newest(r))
    assert not masked(reqs[warned]) and masked(reqs[warned + 1])  # read on: cleared anyway, one round later
    assert sum(1 for e in events if e["type"] == "note" and e["kind"] == "read_streak") == 1  # once per streak

    a = agent(Workspace(tmp_path / "ws2"), [])
    a._streak_ids, a._steps = {"r1"}, 3
    near = int(ctxm.SUMMARY_AT * WINDOW) - ctxm.min_freed_tokens(WINDOW) // 2
    assert a._defer_mask(None, 2, 1_000, near) is False  # (no dry run needed to decide)


# ====================================================================== 2. markers that keep an outline of the code
def test_code_outline_names_definitions_with_their_line_ranges_and_never_prose():
    def numbered(src):
        return "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(src.split("\n"), 1))

    py = numbered("import os\n\nclass Store(Base):\n    \"\"\"Keeps values: a module\n    module things for later\n"
                  "    def is a keyword\"\"\"\n    def get(self, k):\n        return self.d[k]\n\n"
                  "    async def put(self, k, v):\n        def inner():\n            pass\n        self.d[k] = v\n\n"
                  "def helper(x):\n    return x")
    assert ctxm.code_outline(py, "src/store.py") == "class Store 3–14 (get 7–9, put 10–14) · helper 15–16"
    ts = numbered("export interface Props {\n  id: string\n}\nexport type Id = string\n"
                  "export default class View extends Base {\n  render() {}\n}\n"
                  "export const useThing = async (id: Id) => {\n  return id\n}\nfunction helper() {}")
    assert ctxm.code_outline(ts, "web/view.tsx") == ("interface Props 1–3 · type Id 4–4 · class View 5–7 · "
                                                    "useThing 8–10 · helper 11–11")
    go = numbered("type Server struct {\n\tdb *DB\n}\n\nfunc (s *Server) Handle(w Writer) {\n}\nfunc main() {\n}")
    assert ctxm.code_outline(go, "main.go") == "type Server 1–4 · Handle 5–6 · main 7–8"
    rb = numbered("module Billing\n  # module things for later\n  def charge(amount)\n  end\nend")
    assert ctxm.code_outline(rb, "billing.rb") == "module Billing 1–5 (charge 3–5)"
    rs = numbered("impl Display for Money {\n    fn fmt(&self) {}\n}\npub(crate) fn parse<T>(s: &str) {}")
    assert ctxm.code_outline(rs, "money.rs") == "impl Display 1–3 (fmt 2–3) · parse 4–4"
    # not code, nothing defined in what was shown, or no room: no outline
    assert ctxm.code_outline(py, "notes.md") == "" and ctxm.code_outline(numbered("x = 1\ny = 2"), "a.py") == ""
    assert ctxm.code_outline(py, "src/store.py", 40) == ""
    many = numbered("\n".join(f"def f{i}(x):\n    return x" for i in range(60)))
    short = ctxm.code_outline(many, "m.py", 120)
    assert short.startswith("f0 1–2 · f1 3–4") and short.endswith("more") and len(short) <= 120


def test_masked_reads_of_code_keep_an_outline_on_large_windows_only_and_only_the_newest(tmp_path, offline):
    def policy(rec):
        n = len(prov.requests()) - 1
        if n >= 45:
            return reply("Fin.")
        if n % 7 == 6:  # (writes now and then: maskings come with and without a warning)
            return reply("Anoto.", [(f"w{n}", "write_file", {"path": f"notas/n{n}.md", "content": "x\n"})])
        return reply("Leo.", [(f"r{n}", "read_file", {"path": f"src/m{n % 12}.py", "offset": 1 + 120 * (n // 12),
                                                         "limit": 240})])

    prov = offline(policy)
    events = []
    agent(review_repo(tmp_path, classes=12), events).run(TASK)
    last = tools(prov.requests()[-1])
    marks = [m["content"] for m in last if m["content"].startswith("[old read_file")]
    outlined = [i for i, t in enumerate(marks) if "(outline: " in t]
    plain = [i for i, t in enumerate(marks) if "(outline: " not in t]
    assert outlined and plain and len(outlined) <= ctxm.OUTLINES_KEPT
    assert max(plain) < min(outlined)  # the newest keep theirs: older ones lost it at a later masking
    shown = marks[outlined[0]]
    assert "class Service" in shown and "if you need all of it, or just the lines you need (outline: " in shown
    # a marker that lost its outline still says what it was and how to read it again
    old = marks[plain[0]]
    assert "src/m" in old and "read_file(path=\"src/m" in old and "if you need it; full output: " in old
    assert all(e["type"] != "compacted" for e in events)

    # a small window keeps plain one-line markers (outlines there piled up into extra compactions)
    def small(rec):
        n = len(prov.requests()) - 1
        return reply("Fin.") if n >= 12 else reply("Leo.", [(f"r{n}", "read_file", {"path": f"src/m{n % 12}.py"})])
    prov = offline(small, window=32_768)
    events = []
    agent(review_repo(tmp_path / "small"), events, context_limit=32_768).run(TASK)
    marks = [m["content"] for m in tools(prov.requests()[-1]) if m["content"].startswith("[old read_file")]
    assert marks and not any("(outline: " in t for t in marks)


def test_drop_outline_keeps_the_rest_of_the_marker():
    src = ["class A:"] + [line for i in range(60) for line in (f"    def f{i}(self):", "        pass")]
    out = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(src, 1))
    marker = ctxm.mask_marker("read_file", ".nesicode/outputs/x.txt", False, "a.py lines 1–121",
                              'read_file(path="a.py", offset=1, limit=121)', "a.py", 600)("c1", out)
    plain = ctxm.mask_marker("read_file", ".nesicode/outputs/x.txt", False, "a.py lines 1–121",
                             'read_file(path="a.py", offset=1, limit=121)', "a.py")("c1", out)
    assert "(outline: class A 1–121 (60 inside)); full output: " in marker and "(outline: " not in plain
    assert ctxm.drop_outline(marker) == plain and ctxm.drop_outline(plain) is None
    assert 'read it again with read_file(path="a.py", offset=1, limit=121) if you need it; full output: ' in plain


# ====================================================================== 3. the deliverable, named
def test_the_notes_name_the_file_the_user_asked_for_and_fall_back_to_the_notes_file(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "README.md").write_text("léeme\n")
    a = Agent(model="mock/demo", harness=base_harness(), workspace=ws, context_limit=WINDOW)
    a._task = a._task_full = ("Sigue las pautas de README.md y deja los hallazgos en docs/REVIEW.md (no crees otros "
                              "archivos; tampoco .nesicode/chats/1/context/notes.md ni /tmp/fuera.txt).")
    # what exists and was only read is no deliverable; what is to be created is
    assert a._where() == "`docs/REVIEW.md`, the file the user asked for, rather than in "
    a._ro_streak = 20
    assert ("in `docs/REVIEW.md`, the file the user asked for, rather than in your notes file" in a._streak_note())
    # a file the user named that this run wrote is one too
    a._track(ToolCall("w1", "write_file", {"path": "README.md", "content": "x\n"}), ToolResult("w1", "ok"))
    assert a._where() == "`README.md`, `docs/REVIEW.md`, the files the user asked for, rather than in "
    # named in the conversation's first message (the standing rules), the current one says only «continúa»
    a = Agent(model="mock/demo", harness=base_harness(), workspace=ws, context_limit=WINDOW)
    a._history, a._task = [(TASK, "Empiezo.")], "continúa"
    assert a._where().startswith("`REVIEW.md`, the file the user asked for")
    # no output file named: the file the user asked for first, the notes file only as the fallback
    a._history, a._task = [], "Arregla el bug del login."
    assert a._where() == loopmod.WHERE_NOTES
    text = loopmod.MASK_NOTE.format(what="2 earlier tool outputs were", where=a._where())
    assert "(in the file the user asked for if they named one, else in your notes file)" in text
    assert "the deliverable or" not in text


# ====================================================================== 4/5. the run log: notes and cut replies
def test_every_note_the_model_gets_and_every_reply_cut_at_the_cap_are_on_record(tmp_path, offline):
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "a.txt").write_text("dato\n" * 50)
    thinking = "Escribo el cliente entero en mi razonamiento... " * 120

    def policy(rec):
        n = len(prov.requests()) - 1
        if n == 0:  # cut in its reasoning: no text, no call
            return reply("", thinking=thinking, finish="length", out=rec["max_tokens"])
        if n == 1:  # empty: asked for again
            return reply("", thinking="")
        if n in (2, 3, 4):  # the same call, the same result, three times
            return reply("Leo.", [(f"r{n}", "read_file", {"path": "a.txt"})])
        return reply("Resumen final.")

    prov = offline(policy)
    events = []
    res = agent(Workspace(tmp_path / "ws"), events, max_steps=5).run("revisa a.txt")
    assert res.status == "max_steps" and res.final_text.startswith("Resumen final.")
    cut = [e for e in events if e["type"] == "output_cut"]
    assert len(cut) == 1
    assert (cut[0]["visible"], cut[0]["calls"], cut[0]["then"], cut[0]["continuation"]) == (False, 0, "continue", 1)
    assert cut[0]["tokens"] == cut[0]["cap"] == prov.requests()[0]["max_tokens"] and cut[0]["reasoning"] == len(thinking)
    kinds = [(e["type"], e.get("kind")) for e in events if e["type"] in ("output_cut", "note")]
    assert kinds == [("output_cut", None), ("note", "continue"), ("note", "empty_reply"), ("note", "repeat"),
                     ("note", "final_answer")]
    first = [e["type"] for e in events if e["type"] in ("thinking", "output_cut")][:2]
    assert first == ["thinking", "output_cut"]  # (the cut reply's reasoning, then why it ends there)
    notes = {e["kind"]: e for e in events if e["type"] == "note"}
    assert notes["repeat"]["call_id"] == "r4" and notes["final_answer"]["reason"] == "step_cap"
    # each note is on record exactly as the model got it
    reqs = prov.requests()
    assert notes["continue"]["text"] == loopmod.EMPTY_CUT_NOTE.format(cap=cut[0]["cap"])
    assert {"role": "user", "content": notes["continue"]["text"]} in reqs[1]["messages"]
    assert {"role": "user", "content": notes["empty_reply"]["text"]} in reqs[2]["messages"]
    assert notes["repeat"]["text"] in newest(reqs[5]) and newest(reqs[5]).endswith(notes["final_answer"]["text"])


def test_a_reply_cut_in_the_middle_of_a_tool_call_is_on_record_too(tmp_path, offline):
    def policy(rec):
        n = len(prov.requests()) - 1
        if n == 0:
            msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "w1", "type": "function", "function": {
                "name": "write_file", "arguments": '{"path": "client.py", "content": "import httpx\\nclass Cli'}}]}
            return msg, "length", rec["max_tokens"]
        return reply("Lo divido en partes.")

    prov = offline(policy)
    events = []
    agent(Workspace(tmp_path / "ws"), events).run("escribe el cliente")
    cut = next(e for e in events if e["type"] == "output_cut")
    assert (cut["then"], cut["calls"], cut["visible"], cut["continuation"]) == ("reject_calls", 1, False, 0)
    order = [e["type"] for e in events]
    assert order.index("output_cut") < order.index("tool_call")  # before the call it explains is answered
