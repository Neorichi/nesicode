"""Web UI, final review (area "web"): the UI side of contracts the backend already keeps, run offline with node.

Same approach as tests/test_p2_web.py: each test bundles a small TypeScript entry with the esbuild that Vite ships and
runs it with node, rendering components to static HTML where it matters. The inputs come from the real backend
(the HTTP API through TestClient, the agent loop wired to the jobs emitter, apply_rollout), so the UI is checked
against what the server really sends. Skipped when node or web/node_modules is missing.
"""

import io
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nesicode.server.app import app, db

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SRC = WEB / "src"
NODE = shutil.which("node")
ESBUILD = WEB / "node_modules" / "esbuild"

pytestmark = pytest.mark.skipif(not NODE or not ESBUILD.is_dir(), reason="needs node and web/node_modules (esbuild)")

STUB_MARKDOWN = "export default function Markdown({ text }: { text: string }) { return <div className=\"md\">{text}</div> }\n"


def run_ts(tmp_path: Path, body: str, head: str = ""):
    """Bundle `body` for node and run it; return what it printed last as JSON (`out`). In scope: `L` (web/src/logic.ts),
    `setLang`, and `html(node)` = the node rendered to static HTML inside a router."""
    entry = tmp_path / "entry.tsx"
    entry.write_text(
        f"{head}\n"
        f"import * as L from {json.dumps(str(SRC / 'logic.ts'))}\n"
        f"import {{ setLang }} from {json.dumps(str(SRC / 'i18n.ts'))}\n"
        "import { renderToStaticMarkup } from 'react-dom/server'\n"
        "import { MemoryRouter, Route, Routes } from 'react-router-dom'\n"
        "const html = (node: unknown) => renderToStaticMarkup(<MemoryRouter initialEntries={['/p/1/c/2']}><Routes>"
        "<Route path='/p/:pid/c/:tid' element={node as any} /></Routes></MemoryRouter>)\n"
        "void html; void setLang\n"
        "const out: Record<string, unknown> = {}\n"
        f"{body}\nconsole.log(JSON.stringify(out))\n", encoding="utf-8")
    stub = tmp_path / "stub_markdown.tsx"
    stub.write_text(STUB_MARKDOWN, encoding="utf-8")
    bundle = tmp_path / "bundle.cjs"
    build = tmp_path / "build.cjs"
    build.write_text(
        f"require({json.dumps(str(ESBUILD))}).build({{entryPoints: [{json.dumps(str(entry))}], bundle: true, "
        f"platform: 'node', format: 'cjs', outfile: {json.dumps(str(bundle))}, logLevel: 'error', jsx: 'automatic', "
        f"nodePaths: [{json.dumps(str(WEB / 'node_modules'))}], plugins: [{{name: 'stub-markdown', setup(b) {{ "
        f"b.onResolve({{filter: /[\\\\/]Markdown$/}}, () => ({{path: {json.dumps(str(stub))}}})) }}}}]}})"
        ".catch((e) => { console.error(e); process.exit(1) })\n", encoding="utf-8")
    for args in ([NODE, str(build)], [NODE, str(bundle)]):
        p = subprocess.run(args, capture_output=True, text=True, timeout=90, cwd=tmp_path)
        assert p.returncode == 0, p.stderr[-4000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def _user(prefix: str) -> TestClient:
    c = TestClient(app)
    r = c.post("/api/auth/register", json={"email": f"{prefix}{time.time_ns()}@example.com", "password": "secret123"})
    assert r.status_code == 200, r.text
    return c


def _project(c: TestClient) -> dict:
    r = c.post("/api/projects", json={"name": "P", "executor_model": "mock/demo", "improver_model": "mock/demo"})
    assert r.status_code == 200, r.text
    p = r.json()
    Path(p["workspace"]).mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------------------------- web#1 apply retry
def _harness_run(project: dict, before: dict[str, str], after: dict[str, str]) -> tuple[int, Path]:
    """A finished harnessing run as jobs leaves it for a later «Aplicar» (apply_changes off): its snapshot of the
    project and the chosen rollout's workspace."""
    from nesicode.server.jobs import run_dir

    rid = db.insert("INSERT INTO runs (project_id, kind, task, status, result, created_at) VALUES (?, 'harness', 't', 'done', ?, ?)",
                    (project["id"], json.dumps({"selected": "base1"}), time.time()))
    rd = run_dir(db.one("SELECT * FROM projects WHERE id=?", (project["id"],)), rid)
    for root, files in ((rd / "source", before), (rd / "rollouts" / "base1" / "ws", after)):
        root.mkdir(parents=True)
        for rel, text in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
    return rid, rd


def test_an_apply_that_could_not_write_can_be_tried_again_from_the_page(tmp_path):
    """web#1: «Aplicar» while the project's folder is not there (an external disk, a renamed folder) writes nothing:
    every change comes back as 'error' and the server keeps the run's copies for another try. The page offers
    «Reintentar» from that stored result (also after a reload), the retry lands the change, and then nothing more is
    offered; once the server says the copies are gone (410), no button that could only fail again."""
    a = _user("retry")
    p = _project(a)
    ws = Path(p["workspace"])
    (ws / "README.md").write_text("# w\n")
    rid, rd = _harness_run(p, {"README.md": "# w\n"}, {"README.md": "# w\n", "NESICODE_RESULT.md": "hecho\n"})

    ws.rename(ws.with_name("project.gone"))  # the folder becomes unavailable
    r = a.post(f"/api/runs/{rid}/apply")
    assert r.status_code == 200, r.text
    assert r.json()["changes"]["conflicts"] == [{"path": "NESICODE_RESULT.md", "reason": "error", "side_file": None}]
    failed = a.get(f"/api/runs/{rid}").json()["result"]["applied"]  # what a reload of the page shows
    assert (rd / "source").is_dir()  # (the server kept its copies for another «Aplicar»)

    ws.with_name("project.gone").rename(ws)  # the folder is back
    r = a.post(f"/api/runs/{rid}/apply")
    assert r.status_code == 200 and r.json()["changes"]["added"] == ["NESICODE_RESULT.md"]
    assert (ws / "NESICODE_RESULT.md").read_text() == "hecho\n"
    applied = a.get(f"/api/runs/{rid}").json()["result"]["applied"]
    gone = a.post(f"/api/runs/{rid}/apply")  # all of it is in the project: its copies went
    assert gone.status_code == 410

    out = run_ts(tmp_path, f"""
      import {{ ApplyButton }} from {json.dumps(str(SRC / 'pages' / 'Harnessing.tsx'))}
      const button = (applied: unknown, gone = false) =>
        html(<ApplyButton applied={{applied as any}} gone={{gone}} busy={{false}} onApply={{() => {{}}}} />)
      out.fresh = button(null)
      out.failed = button({json.dumps(failed)})
      out.applied = button({json.dumps(applied)})
      out.gone = [button({json.dumps(failed)}, true), button(null, true)]
      setLang('en'); out.en = button({json.dumps(failed)}); setLang('es')
    """)
    assert ">Aplicar los cambios de código al proyecto</button>" in out["fresh"]
    assert ">Reintentar aplicar los cambios de código</button>" in out["failed"]  # (it used to be gone for good)
    assert out["applied"] == "" and out["gone"] == ["", ""]
    assert ">Retry applying the code changes</button>" in out["en"]


def test_the_retry_is_offered_whenever_the_server_keeps_the_copies(tmp_path):
    """web#1: what a real three-way apply leaves out, case by case, against jobs._fully_applied (whether the server
    keeps the run's copies for another «Aplicar»): the page offers the retry in every case the server keeps them, and
    in none where the user's file plus the proposed version beside it is the final result. The one case it can't tell
    apart (a file the result deletes but the user changed: kept on purpose, no side file) offers it, and the server's
    410 then says why."""
    from nesicode.harnessing import apply_rollout
    from nesicode.server.jobs import _fully_applied

    def case(name: str, files: dict, setup=None) -> tuple[dict, bool]:
        base = tmp_path / name
        before, after, target = (base / n for n in ("source", "rollouts/sel/ws", "target"))
        for d in (before, after, target):
            d.mkdir(parents=True)
        for rel, contents in files.items():  # (snapshot, result, project now)
            for root, text in zip((before, after, target), contents):
                if text is not None:
                    (root / rel).parent.mkdir(parents=True, exist_ok=True)
                    (root / rel).write_text(text)
        if setup:
            setup(base, target)
        try:
            result = apply_rollout(before, after, target)
        finally:
            if (target / "ro").is_dir():
                (target / "ro").chmod(0o755)  # (so tmp_path can be removed)
        return result, not _fully_applied(base, "sel", result)

    def missing(base, target):
        target.rename(base / "target.gone")

    def read_only(base, target):
        (target / "ro").mkdir()
        (target / "ro").chmod(stat.S_IRUSR | stat.S_IXUSR)

    def link_out(base, target):
        (base / "outside").mkdir()
        os.symlink(base / "outside", target / "fuera")

    cases = {
        "missing": case("missing", {"a.txt": ("1", "2", "1"), "n.txt": (None, "n", None)}, missing),
        "clean": case("clean", {"a.txt": ("1", "2", "1"), "n.txt": (None, "n", None), "v.txt": ("v", None, "v")}),
        "changed": case("changed", {"b.txt": ("1", "2", "mio")}),  # the user's file kept, the proposal beside it
        "exists": case("exists", {"c.txt": (None, "nuevo", "mio")}),
        "read_only": case("read_only", {"ro/x.txt": (None, "x", None), "ok.txt": (None, "ok", None)}, read_only),
        "unsafe": case("unsafe", {"fuera/z.txt": (None, "z", None)}, link_out),
        "removed_but_changed": case("removed_but_changed", {"d.txt": ("1", None, "editado")}),
    }
    reasons = {k: sorted({c["reason"] for c in r["conflicts"]}) for k, (r, _) in cases.items()}
    assert reasons == {"missing": ["error"], "clean": [], "changed": ["changed"], "exists": ["exists"], "read_only": ["error"],
                       "unsafe": ["unsafe"], "removed_but_changed": ["changed"]}  # (the setup works)
    keeps = {k: kept for k, (_, kept) in cases.items()}
    assert keeps == {"missing": True, "clean": False, "changed": False, "exists": False, "read_only": True,
                     "unsafe": True, "removed_but_changed": False}
    out = run_ts(tmp_path, f"""
      const cases = {json.dumps({k: r for k, (r, _) in cases.items()})}
      out.retry = Object.fromEntries(Object.entries(cases).map(([k, r]) => [k, L.canRetryApply(r as any)]))
      out.odd = [L.canRetryApply(null), L.canRetryApply({{}}), L.canRetryApply({{ conflicts: 'x' }} as any),
                 L.canRetryApply({{ conflicts: [null, {{ path: 'a', reason: 'error' }}] }} as any)]
    """)
    expected = {k: kept for k, kept in keeps.items()} | {"removed_but_changed": True}  # (the documented exception)
    assert out["retry"] == expected
    assert out["odd"] == [False, False, False, True]


# ---------------------------------------------------------------------------------------------- web#4 live text
def _truncated_reply_stream(tmp_path: Path, monkeypatch) -> dict:
    """The real loop on a small window, wired to the real jobs emitter: a tool step, then a reply streamed and cut
    because the context left no room ('model_context_window_exceeded' above 45%): the loop drops it, compacts (reason
    'truncated reply') and asks again. Returns everything a client is sent, in order (persisted events and live
    items), and the live items a client connecting while the new reply streams is replayed."""
    from types import SimpleNamespace

    import nesicode.agent.loop as loop
    from nesicode.agent import Agent, Workspace
    from nesicode.harness import base_harness
    from nesicode.llm.base import Conversation, LLMError, Provider, ToolCall, Turn, Usage
    from nesicode.server.jobs import Jobs

    calls: list[int] = []

    class Conv(Conversation):
        def __init__(self, model, system, tools, options):
            super().__init__(model, system, tools, options)
            self.chars = len(system)

        def add_user(self, text):
            self.chars += len(text)

        def add_assistant_text(self, text):
            self.chars += len(text or "")

        def add_tool_results(self, results):
            self.chars += sum(len(r.content or "") + 40 for r in results)

        def mask_tool_results(self, keep_last, marker, *, keep_chars=0, apply=True):
            return 0

        def ntokens(self):
            return self.chars // 4

        def step(self, on_event=None):
            if self.options.role == "summarizer":
                raise LLMError("read timeout")  # (the fallback summary is used: no matter here)
            calls.append(1)
            # the provider's count of the prompt: 55% of the window (over the 45% that makes a cut reply droppable,
            # under the 67% that compacts before the call), whatever the size of the system prompt
            if len(calls) == 1:
                return Turn("", [ToolCall("w1", "write_file", {"path": "p.txt", "content": "y"})], "tool_use", Usage(4_400, 40))
            word, stop, used = ("CORTADA", "model_context_window_exceeded", 4_400) if len(calls) == 2 else ("NUEVA", "end_turn", 1_500)
            text = ""
            for i in range(6):
                text += f"{word}-{i} "
                on_event("text", f"{word}-{i} ")
            return Turn(text, [], stop, Usage(used, 50))

    class Prov(Provider):
        name = "fake"

        def conversation(self, model, system, tools, options):
            return Conv(model, system, tools, options)

    monkeypatch.setattr(loop, "get_provider", lambda spec, keys=None: (Prov(), "m"))
    server = SimpleNamespace(db=SimpleNamespace(add_event=lambda run_id, event: None), live={})
    server_emit = Jobs.emitter(server, 1)
    seen: dict = {"sent": [], "replay": None}

    def emit(event: dict) -> None:
        server_emit(event)
        persist = event.get("_persist", True)
        clean = {k: v for k, v in event.items() if k not in ("_persist", "summary")}
        seen["sent"].append(["event" if persist else "live", clean])
        if event.get("type") == "text_delta" and event.get("text") == "NUEVA-3 ":
            seen["replay"] = list(server.live[1].items)  # a client connecting now gets these from the start

    res = Agent(model="fake/m", harness=base_harness(), workspace=Workspace(tmp_path / "ws"), emit=emit,
                context_limit=8_000, output_limit=2_000).run("tarea")
    assert res.status == "done" and len(calls) == 3, (res.status, len(calls))
    return seen


def test_a_reply_dropped_for_a_compaction_never_shows_glued_to_the_next_one(tmp_path, monkeypatch):
    """web#4: the cut reply was streamed, then dropped for the compaction ('truncated reply') with no 'assistant' event
    of its own; the next reply streams into the same bubble. It must show alone — while it streams (however the
    stream batches events and live items), and when a client connecting meanwhile is replayed the whole reply's live
    items, the dropped part included."""
    seen = _truncated_reply_stream(tmp_path, monkeypatch)
    events = [e for kind, e in seen["sent"] if kind == "event"]
    assert [e["type"] for e in events if e["type"] in ("compacted", "assistant")] == ["compacted", "assistant"]
    assert events[[e["type"] for e in events].index("compacted")]["reason"] == "truncated reply"
    replay_text = "".join(i.get("text", "") for i in seen["replay"] if i.get("type") == "text_delta")
    assert replay_text.startswith("CORTADA-0") and "NUEVA-3" in replay_text  # (the server still replays the dropped text)
    out = run_ts(tmp_path, f"""
      const sent = {json.dumps(seen["sent"])} as [string, any][]
      // a stream poll sends the events it found first, then the live items: try every batch size
      out.bubbles = [1, 2, 3, 5, 8, 13, 1000].map((size) => {{
        let text = ''
        const shown: string[] = []
        for (let i = 0; i < sent.length; i += size) {{
          const poll = sent.slice(i, i + size)
          for (const [kind, e] of poll) if (kind === 'event' && L.endsLiveText(e)) text = ''
          text = L.liveText(text, poll.filter(([kind]) => kind === 'live').map(([, e]) => e))
          shown.push(text)
        }}
        return shown
      }})
      out.replay = L.liveText('', {json.dumps(seen["replay"])})
      out.odd = [L.liveText('a', [null, 3, {{ type: 'text_delta', text: 5 }}, {{ type: 'text_delta', text: 'b', depth: 1 }},
                                 {{ type: 'context', reason: 'compacted', depth: 1 }}, {{ type: 'text_delta', text: 'c' }}] as any),
                 L.liveText('a', undefined as any)]
      out.ends = [{{ type: 'compacted' }}, {{ type: 'compacted', depth: 1 }}, {{ type: 'llm_retry' }}, {{ type: 'llm_retry', depth: 2 }},
                  {{ type: 'assistant' }}, {{ type: 'tool_call' }}, {{ type: 'context_learned' }}].map((e) => L.endsLiveText(e as any))
    """)
    for bubbles in out["bubbles"]:
        assert any("NUEVA-5" in b for b in bubbles)  # (the new reply was shown, whole)
        assert not [b for b in bubbles if "CORTADA" in b and "NUEVA" in b], bubbles
    assert out["replay"] == "NUEVA-0 NUEVA-1 NUEVA-2 NUEVA-3 "
    assert out["odd"] == ["ac", "a"]
    assert out["ends"] == [True, False, True, False, True, False, False]


# ---------------------------------------------------------------------------------------------- web#5 dropped fields
def test_a_provider_added_without_models_says_why_and_what_to_do(tmp_path):
    """web#5 (a): an endpoint that can't list its models still gets the provider created, with no models and the
    server's `warning`: the page says so (in the user's language) instead of a silent «0 modelos»."""
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    a = _user("prov")
    r = a.post("/api/providers", json={"name": "muerto", "label": "Caído", "kind": "openai", "base_url": f"http://127.0.0.1:{port}/v1"})
    assert r.status_code == 200 and r.json()["models"] == [] and r.json()["warning"].startswith("No se pudieron obtener los modelos: ")
    ok = a.post("/api/providers", json={"name": "vivo", "kind": "openai", "base_url": f"http://127.0.0.1:{port}/v1", "models": ["m1"]})
    assert ok.status_code == 200 and "warning" not in ok.json()
    out = run_ts(tmp_path, f"""
      out.es = L.providerAddedNote('muerto', {json.dumps(r.json())})
      out.none = [L.providerAddedNote('vivo', {json.dumps(ok.json())}), L.providerAddedNote('x', null), L.providerAddedNote('x', {{ warning: 3 }} as any)]
      setLang('en'); out.en = L.providerAddedNote('muerto', {json.dumps(r.json())}); setLang('es')
    """)
    assert out["es"].startswith("«muerto» se ha añadido sin modelos. No se pudieron obtener los modelos: no se pudo conectar")
    assert out["es"].endswith("Cuando el endpoint responda pulsa «Actualizar modelos», o escribe sus modelos en «Editar».")
    assert ".." not in out["es"] and out["none"] == ["", "", ""]
    assert out["en"].startswith("“muerto” was added with no models. Couldn't fetch the models: couldn't connect")
    assert out["en"].endswith("press “Refresh models”, or type its models in “Edit”.")


def test_zip_entries_left_out_of_an_upload_are_listed(tmp_path):
    """web#5 (b): a ZIP with a symbolic link uploads its files, never the link, and the server says which entry it
    skipped: the page lists it with the reason (it used to say only «2 fichero(s) añadidos»)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("config/base.env", "A=1\n")
        z.writestr("README.md", "# hola\n")
        link = zipfile.ZipInfo("config/actual")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(link, "base.env")
    a = _user("zip")
    p = _project(a)
    r = a.post(f"/api/projects/{p['id']}/upload", files={"file": ("conenlace.zip", buf.getvalue())})
    assert r.status_code == 200 and not os.path.lexists(Path(p["workspace"]) / "config" / "actual")
    plain = a.post(f"/api/projects/{p['id']}/upload", files={"file": ("nota.txt", b"x")})
    out = run_ts(tmp_path, f"""
      import {{ UploadSkipped }} from {json.dumps(str(SRC / 'pages' / 'ProjectSettings.tsx'))}
      const r = L.uploadReport({json.dumps(r.json())})
      out.r = r
      out.plain = L.uploadReport({json.dumps(plain.json())})
      out.odd = [L.uploadReport(null), L.uploadReport({{ files: 'x', skipped: [3, '', 'a: b'] }} as any)]
      out.page = html(<UploadSkipped skipped={{r.skipped}} />)
      out.none = html(<UploadSkipped skipped={{[]}} />)
      setLang('en'); out.en = html(<UploadSkipped skipped={{r.skipped}} />); setLang('es')
    """)
    assert out["r"] == {"files": 2, "skipped": ["config/actual: enlace simbólico, no se extrae"]}
    assert out["plain"] == {"files": 1, "skipped": []}
    assert out["odd"] == [{"files": 0, "skipped": []}, {"files": 0, "skipped": ["a: b"]}]
    assert "1 entrada(s) del .zip no se extrajeron:" in out["page"]
    assert "<code>config/actual</code> — enlace simbólico, no se extrae" in out["page"]
    assert out["none"] == ""
    assert "<code>config/actual</code> — symbolic link, not extracted" in out["en"]


def test_links_in_a_conversations_folder_are_marked_and_one_that_cannot_open_says_why(tmp_path):
    """web#5 (c): the «Archivos» tab lists links (link: true) as links. One that leads to a file of the copy opens;
    one that leads out of it (absolute, or relative and dangling in the copy) answers 404, and the page says why —
    instead of an unhandled error that left the previous file on screen under the new name."""
    shared = tmp_path / "compartido"
    shared.mkdir()
    (shared / "comun.md").write_text("fuera\n")
    a = _user("links")
    p = _project(a)
    ws = Path(p["workspace"])
    (ws / "notas.txt").write_text("notas\n")
    os.symlink("notas.txt", ws / "a_dentro")
    os.symlink("../compartido/comun.md", ws / "b_fuera")
    os.symlink(shared / "comun.md", ws / "c_absoluto")
    (ws / "docs").mkdir()
    t = a.post(f"/api/projects/{p['id']}/threads", json={}).json()  # its own copy of the project, links as links
    tree = a.get(f"/api/threads/{t['id']}/tree").json()
    assert {e["path"]: bool(e.get("link")) for e in tree} == {"a_dentro": True, "b_fuera": True, "c_absoluto": True,
                                                              "docs": False, "notas.txt": False}
    opened = {}
    for e in tree:
        if not e["dir"]:
            r = a.get(f"/api/threads/{t['id']}/file", params={"path": e["path"]})
            opened[e["path"]] = (r.status_code, r.json().get("detail") or r.json().get("content"))
    assert opened == {"a_dentro": (200, "notas\n"), "notas.txt": (200, "notas\n"), "b_fuera": (404, "Fichero no encontrado"),
                      "c_absoluto": (404, "Fichero no encontrado")}
    out = run_ts(tmp_path, f"""
      import {{ FileTree }} from {json.dumps(str(SRC / 'pages' / 'Chat.tsx'))}
      const tree = {json.dumps(tree)}
      const opened = {json.dumps(opened)} as Record<string, [number, string]>
      out.page = html(<FileTree tree={{tree}} active="notas.txt" onOpen={{() => {{}}}} />)
      out.errors = Object.fromEntries(tree.filter((e: any) => opened[e.path]?.[0] === 404)
        .map((e: any) => [e.path, L.openFileError(e, 404, opened[e.path][1])]))
      out.other = [L.openFileError({{ path: 'x.txt' }}, 404, 'Fichero no encontrado'), L.openFileError({{ path: 'l', link: true }}, 500, 'Error interno')]
      setLang('en'); out.en = L.openFileError(tree.find((e: any) => e.path === 'c_absoluto'), 404, 'x'); setLang('es')
    """)
    items = re.findall(r'<button class="tree-item ?(\w*)"[^>]*?( disabled="")?( title="[^"]*")?>(.*?)</button>', out["page"])
    assert [(text, bool(disabled), bool(title)) for _, disabled, title, text in items] == [
        ("↪ a_dentro", False, True), ("↪ b_fuera", False, True), ("↪ c_absoluto", False, True),
        ("▸ docs", True, False), ("notas.txt", False, False)]
    assert [cls for cls, *_ in items].count("active") == 1
    assert out["errors"] == {p: f"«{p}» es un enlace simbólico que apunta fuera de la carpeta de esta conversación (o a algo "
                                 "que no existe): no se puede abrir aquí." for p in ("b_fuera", "c_absoluto")}
    assert out["other"] == ["Fichero no encontrado", "Error interno"]
    assert out["en"].startswith("“c_absoluto” is a symbolic link that points outside this conversation's folder")


# ---------------------------------------------------------------------------------------------- web#6 deleted
def test_a_deleted_conversation_never_shows_its_reply_running(tmp_path):
    """web#6: a conversation deleted elsewhere while its reply streams: the server stops the reply and drops its
    queue, the stream ends 'deleted' and its page answers 404. What the page then shows of its runs: none running or
    queued (no Stop, no typing dots), the text that was streaming kept as an interrupted reply."""
    a = _user("gone")
    p = _project(a)
    t = a.post(f"/api/projects/{p['id']}/threads", json={}).json()
    rid = db.insert("INSERT INTO runs (project_id, thread_id, kind, task, status, created_at) VALUES (?, ?, 'chat', 'hola', 'running', ?)",
                    (p["id"], t["id"], time.time()))
    runs = a.get(f"/api/threads/{t['id']}").json()["runs"]  # what the page has while the reply streams
    assert [r["status"] for r in runs] == ["running"]
    db.execute("UPDATE runs SET status='cancelled' WHERE id=?", (rid,))  # (the delete stops it: halt_runs)
    assert a.delete(f"/api/threads/{t['id']}").status_code == 200
    # what the page then gets: its reload answers 404, and so does Stop («Ejecución no encontrada»)
    assert a.get(f"/api/threads/{t['id']}").status_code == 404 and a.post(f"/api/runs/{rid}/cancel").status_code == 404
    out = run_ts(tmp_path, f"""
      import Transcript from {json.dumps(str(SRC / 'components' / 'Transcript.tsx'))}
      const runs = [...{json.dumps(runs)}, {{ id: 90, status: 'waiting', events: [] }}, {{ id: 91, status: 'done', events: [] }},
                    {{ id: 92, status: 'queued', events: [] }}] as any[]
      const ended = L.endedRuns(runs)
      out.status = ended.map((r: any) => r.status)
      out.same = ended[2] === runs[2]
      const r = ended[0]
      out.page = html(<Transcript events={{[{{ type: 'start' }}] as any}} status={{r.status}} interrupted="Texto a medias" runId={{r.id}} />)
      out.before = html(<Transcript events={{[{{ type: 'start' }}] as any}} status={{runs[0].status}} runId={{r.id}} />)
    """)
    assert out["status"] == ["cancelled", "cancelled", "done", "cancelled"] and out["same"] is True
    assert 'class="typing"' in out["before"]  # (what it showed for good before)
    page = out["page"]
    assert 'class="typing"' not in page and "Texto a medias" in page and "Cancelado" in page
