"""Web UI, phase 2 (group S): the UI side of the phase-1 backend changes, run offline with node.

Same approach as tests/test_fix_web_ui.py: each test bundles a small TypeScript entry with the esbuild that Vite ships
and runs it with node. Here the entry may also render components to HTML with react-dom/server (Markdown is stubbed:
DOMPurify needs a browser). The inputs come from the real backend functions wherever they exist (apply_rollout's
conflicts, Apply's skipped paths, connectors.trigger_words, config_io's import preview, clean_summary's problems), so
the UI is checked against what the server really sends. Skipped when node or web/node_modules is missing.
"""

import errno
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SRC = WEB / "src"
NODE = shutil.which("node")
ESBUILD = WEB / "node_modules" / "esbuild"

if os.environ.get("CI") and (not NODE or not ESBUILD.is_dir()):  # a release build runs them, never skips them
    pytest.fail("the web UI tests need node and web/node_modules (esbuild): run `npm ci` in web/ first", pytrace=False)
pytestmark = pytest.mark.skipif(not NODE or not ESBUILD.is_dir(), reason="needs node and web/node_modules (esbuild)")

STUB_MARKDOWN = "export default function Markdown({ text }: { text: string }) { return <div className=\"md\">{text}</div> }\n"


def run_ts(tmp_path: Path, body: str, head: str = ""):
    """Bundle `body` for node and run it; return what it printed as JSON (`out`). In scope: `L` (web/src/logic.ts),
    `setLang`, and `html(node)` = the node rendered to static HTML inside a router (for components)."""
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


# ---------------------------------------------------------------------------------------------------- S1 conflicts
def _rollout_with_conflicts(tmp_path: Path) -> dict:
    """A real harnessing apply (apply_rollout) that applies three changes and leaves four out."""
    from nesicode.harnessing import apply_rollout

    before, after, target, outside = (tmp_path / n for n in ("before", "after", "target", "outside"))
    for d in (before, after, target, outside):
        d.mkdir()
    files = {  # path: (snapshot, rollout, project now)
        "a.txt": ("1", "2", "1"),            # modified cleanly
        "nuevo.txt": (None, "n", None),      # added cleanly
        "viejo.txt": ("o", None, "o"),       # removed cleanly
        "b.txt": ("1", "2", "editado"),      # the user changed it after the snapshot → changed (+ side file)
        "c.txt": (None, "rollout", "mio"),   # the project got its own c.txt meanwhile → exists (+ side file)
        "d.txt": ("1", None, "editado"),     # removed by the rollout, changed by the user → changed (kept)
        "fuera/z.txt": (None, "z", None),    # fuera/ is a link out of the project → unsafe
    }
    for rel, contents in files.items():
        for root, text in zip((before, after, target), contents):
            if text is not None:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(text)
    os.symlink(outside, target / "fuera")
    return apply_rollout(before, after, target)


def test_harnessing_conflicts_are_listed_in_the_cli_words_and_never_counted(tmp_path):
    """S1: an auto-apply/«Aplicar» result with conflicts lists each one (path, reason as the CLI words it, the file
    with the proposed version) and the counts only hold what really reached the project."""
    from nesicode.cli import CONFLICT_WHY

    result = _rollout_with_conflicts(tmp_path)
    assert {c["reason"] for c in result["conflicts"]} == {"changed", "exists", "unsafe"}  # (the setup works)
    dup = {**result, "modified": result["modified"] + ["b.txt"]}  # a left-out path listed as a change, too
    out = run_ts(tmp_path, f"""
      import ApplyReport from {json.dumps(str(SRC / 'components' / 'ApplyReport.tsx'))}
      const r = {json.dumps(result)}
      out.counts = L.applyCounts(r)
      out.dup = L.applyCounts({json.dumps(dup)})
      out.why = Object.fromEntries(['changed', 'exists', 'unsafe', 'error', 'nuevo'].map((c) => [c, L.conflictWhy(c)]))
      out.summary = L.applySummary(r)
      out.clean = L.applySummary({{ added: ['x'], modified: [], removed: [] }})
      out.html = html(<ApplyReport result={{r}} />)
      out.none = html(<ApplyReport result={{{{ added: ['x'] }}}} />)
      out.odd = html(<ApplyReport result={{{{ conflicts: 'no-list', skipped: null }} as any}} />)
      setLang('en'); out.en = L.conflictWhy('exists'); setLang('es')
    """)
    assert out["counts"] == {"added": 1, "modified": 1, "removed": 1, "left": 4}
    assert out["dup"]["modified"] == 1  # never counted, whatever list also holds it
    assert {k: out["why"][k] for k in CONFLICT_WHY} == CONFLICT_WHY  # the CLI's words, code by code
    assert out["why"]["nuevo"] == "nuevo"  # an unknown code is shown as it came
    assert out["summary"] == "Aplicados: +1 ~1 −1 · 4 sin aplicar (detalle abajo)"
    assert out["clean"] == "Aplicados: +1 ~0 −0"
    page = out["html"]
    for c in result["conflicts"]:
        assert f"<code>{c['path']}</code>" in page and CONFLICT_WHY[c["reason"]] in page
    assert "<code>b.txt.nesicode-new</code>" in page and "<code>c.txt.nesicode-new</code>" in page
    assert "la versión propuesta está en" in page and "4 cambio(s) sin aplicar" in page
    assert out["none"] == "" and out["odd"] == ""  # nothing left out (or nothing readable): nothing drawn
    assert out["en"] == "already exists in the project with other content"


# ---------------------------------------------------------------------------------------------------- S2 skipped
def test_paths_skipped_by_apply_are_listed_with_their_reason_in_spanish(tmp_path):
    """S2: a conversation's Apply returns skipped: ["path: reason"] (reason in English, from the server). Each one is
    listed with its path intact and the reason in the UI's language; unknown reasons are shown as they came."""
    from nesicode.agent.workspace import _why, apply_cage, manifest

    cage, project = tmp_path / "cage", tmp_path / "project"
    cage.mkdir()
    project.mkdir()
    (cage / "ok.txt").write_text("x")
    base = manifest(cage)
    (cage / "nuevo.txt").write_text("n")
    os.symlink("/etc/hosts", cage / "malo")  # a link the agent planted: it would point out of the project
    real = apply_cage(base, cage, project)
    assert real["added"] == ["nuevo.txt"] and real["skipped"] == ["malo: symlink to /etc/hosts would point outside the project"]
    entries = real["skipped"] + [f"src/x{n}.py: {_why(OSError(code, os.strerror(code)))}" for n, code in enumerate(
        (errno.ELOOP, errno.ENOTDIR, errno.ENOENT, errno.EINVAL, errno.EACCES, errno.ENOTEMPTY))]
    entries += [f"y.txt: {_why(OSError(errno.EISDIR, 'it is a directory now'))}", "carpeta/a: b.txt: not found",
                "z.bin: algo nuevo del servidor", "sin-motivo"]
    out = run_ts(tmp_path, f"""
      import ApplyReport from {json.dumps(str(SRC / 'components' / 'ApplyReport.tsx'))}
      const entries = {json.dumps(entries)}
      out.parts = entries.map(L.skippedPath)
      out.counts = L.applyCounts({{ ...{json.dumps(real)}, added: ['nuevo.txt', 'malo'] }})
      out.summary = L.applySummary({json.dumps(real)})
      out.html = html(<ApplyReport result={{{{ skipped: entries }}}} />)
      setLang('en'); out.en = L.skippedPath(entries[0]).reason; setLang('es')
    """)
    parts = out["parts"]
    assert parts[0] == {"path": "malo", "reason": "es un enlace que apuntaría fuera del proyecto (/etc/hosts)"}
    assert [p["path"] for p in parts[1:7]] == [f"src/x{n}.py" for n in range(6)]
    assert [p["reason"] for p in parts[1:7]] == [
        "es un enlace simbólico (no se siguen enlaces: usa la ruta a la que apunta)",
        "una carpeta de su ruta es un enlace simbólico o un archivo", "no se encontró", "no es un archivo normal",
        "sin permiso para escribirlo", "hay una carpeta con contenido en su lugar"]
    assert parts[7] == {"path": "y.txt", "reason": "ahora es una carpeta"}
    assert parts[8] == {"path": "carpeta/a: b.txt", "reason": "no se encontró"}  # a ': ' in the name stays in the path
    assert parts[9] == {"path": "z.bin", "reason": "algo nuevo del servidor"} and parts[10] == {"path": "sin-motivo", "reason": ""}
    assert out["counts"] == {"added": 1, "modified": 0, "removed": 0, "left": 1}
    assert out["summary"] == "Aplicados: +1 ~0 −0 · 1 sin aplicar (detalle abajo)"
    page = out["html"]
    assert "11 ruta(s) sin aplicar" in page and "<code>malo</code>" in page and "<code>carpeta/a: b.txt</code>" in page
    assert "would point outside" not in page and "not found" not in page  # no English reason left for known ones
    assert out["en"] == "it is a link that would point outside the project (/etc/hosts)"


# ---------------------------------------------------------------------------------------------------- S3 trigger words
def test_connectors_in_ask_mode_say_which_words_switch_them_on(tmp_path):
    """S3: only 'Solo cuando lo pida' connectors show their trigger words; a connector no word can switch on says so
    ([]), and an item without the field (older server) or in another mode shows nothing (null)."""
    from nesicode.connectors import trigger_words

    items = [
        {"name": "burp", "label": "Burp Suite", "template": "burp", "usage": "mention",
         "trigger_words": trigger_words("burp", "Burp Suite", "burp")},
        {"name": "navegacion", "label": "Navegación", "template": "", "usage": "mention",
         "trigger_words": trigger_words("navegacion", "Navegación", "")},
        {"name": "mcp", "label": "MCP", "template": "", "usage": "mention", "trigger_words": trigger_words("mcp", "MCP", "")},
        {"name": "github", "label": "GitHub", "template": "github", "usage": "auto", "trigger_words": ["github"]},
        {"name": "viejo", "label": "Viejo", "template": ""},  # usage defaults to 'mention'; no trigger_words field
        {"name": "raro", "label": "Raro", "usage": "mention", "trigger_words": ["raro", "", "  raro ", 3]},
    ]
    out = run_ts(tmp_path, f"out.words = {json.dumps(items)}.map((c: any) => L.triggerWords(c))")
    assert out["words"] == [["burp"], ["navegacion"], [], None, None, ["raro"]]


# ---------------------------------------------------------------------------------------------------- S5 learned limits
def test_a_model_limit_learned_meanwhile_shows_and_forgetting_it_updates_every_picker(tmp_path):
    """S5 (review): the pickers' limits are cached per page, but asked again whenever a picker shows the model — a
    chat may have learned its real window since (context_learned), and «aprendido» + «Olvidar» must show without a
    reload. «Olvidar» updates every picker of that model (only those), also when DELETE answers no limits, and a
    failure is the server's message with nothing changed."""
    out = run_ts(tmp_path, f"""
      import {{ cachedLimits, forgetLearned, watchLimits }} from {json.dumps(str(SRC / 'limits.ts'))}
      type Lim = {{ context: number; output: number | null; source: string }}
      const fresh = (): Lim => ({{ context: 128000, output: 16000, source: 'default' }})
      const server: Record<string, Lim> = {{ 'p/m': fresh(), 'p/otro': {{ context: 64000, output: 8000, source: 'table' }} }}
      let del = 'fresh'
      const calls: string[] = []
      const json = (status: number, body: unknown) => new Response(JSON.stringify(body), {{ status, headers: {{ 'Content-Type': 'application/json' }} }})
      ;(globalThis as any).fetch = async (url: string, init: {{ method: string }}) => {{
        const spec = decodeURIComponent(String(url).split('spec=')[1] ?? '')
        calls.push(`${{init.method}} ${{spec}}`)
        if (init.method === 'DELETE') {{
          if (del === 'fail') return json(500, {{ detail: 'No se pudo olvidar' }})
          server[spec] = fresh()
          return del === 'ok' ? json(200, {{ ok: true }}) : json(200, {{ spec, ...server[spec] }})
        }}
        return json(200, {{ spec, ...server[spec] }})
      }}
      const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
      const rec = (xs: string[]) => (l: Lim) => xs.push(`${{l.context}}:${{l.source}}`)
      void (async () => {{
        const a: string[] = []
        let stop = watchLimits('p/m', rec(a), 20)
        out.first = [...a]
        await sleep(80)
        out.loaded = [...a]
        stop()
        server['p/m'] = {{ context: 12000, output: 16000, source: 'learned' }} // a chat learned the real window
        const b: string[] = []
        stop = watchLimits('p/m', rec(b), 20) // the settings page shown again (no reload)
        out.atOnce = [...b]
        await sleep(50)
        out.again = [...b]
        const c: string[] = [], d: string[] = []
        const stopC = watchLimits('p/m', rec(c), 20) // Chat and Mejora on the same model, Visión on another
        const stopD = watchLimits('p/otro', rec(d), 20)
        await sleep(80)
        const n = [b.length, c.length, d.length]
        out.forgotten = (await forgetLearned('p/m')).source
        out.heard = [b.slice(n[0]), c.slice(n[1]), d.slice(n[2])]
        server['p/m'] = {{ context: 12000, output: 16000, source: 'learned' }}
        del = 'ok'
        calls.length = 0
        out.okAnswer = (await forgetLearned('p/m')).source
        out.okCalls = [...calls]
        del = 'fail'
        try {{ await forgetLearned('p/m'); out.fail = 'no error' }} catch (e) {{ out.fail = (e as Error).message }}
        out.cached = cachedLimits('p/m')
        stop(); stopC(); stopD()
        const m = [b.length, c.length]
        server['p/m'] = {{ context: 32000, output: 16000, source: 'learned' }}
        const stopE = watchLimits('p/m', () => {{}}, 20)
        await sleep(50)
        stopE()
        out.afterStop = [b.length - m[0], c.length - m[1]]
        console.log(JSON.stringify(out)) // (the last line is what the test reads)
      }})()
    """)
    assert out["first"] == [] and out["loaded"] == ["128000:default"]
    assert out["atOnce"] == ["128000:default"] and out["again"] == ["128000:default", "12000:learned"]
    assert out["forgotten"] == "default"
    assert out["heard"] == [["128000:default"], ["128000:default"], []]
    assert out["okAnswer"] == "default" and out["okCalls"] == ["DELETE p/m", "GET p/m"]
    assert out["fail"] == "No se pudo olvidar" and out["cached"] == {"context": 128000, "output": 16000, "source": "default", "spec": "p/m"}
    assert out["afterStop"] == [0, 0]


# ---------------------------------------------------------------------------------------------------- S6 events
EMIT_RX = re.compile(r'emit(?:ter\([^)]*\))?\(\s*\{\s*"type":\s*(f?)"([^"]+)"')


def backend_event_types() -> set[str]:
    """Every event type the backend emits (loop, jobs, harnessing, bots…): the literal "type" of each emit({...})."""
    found = set()
    for f in (ROOT / "src" / "nesicode").rglob("*.py"):
        for m in EMIT_RX.finditer(f.read_text(encoding="utf-8")):
            found.add("*_delta" if m.group(1) and m.group(2).endswith("_delta") else m.group(2))
    return found


# What the backend emits after phase 1 (loop, jobs, harnessing, bots), each with its place in the UI: the transcript
# draws it, it is paired with another event (tool_result, approval_result), live-only (context: the meter;
# compaction_error: a notice; *_delta: the streaming text), bookkeeping (start, done) or harnessing's own page.
PHASE1_EVENTS = {
    "start", "assistant", "thinking", "tool_call", "tool_result", "error", "cancelled", "done", "llm_retry", "context",
    "context_learned", "masked", "compacted", "compaction_error", "image", "file", "notify", "bot_created",
    "connector_error", "auto_harness", "auto_harness_skipped", "auto_harness_error", "approval_request",
    "approval_result", "phase", "stage", "rollout", "selected", "applied", "adopted", "adopt_skipped", "lesson",
}


def test_every_event_type_the_backend_emits_has_a_place_in_the_ui(tmp_path):
    """S6: each event type the backend emits after phase 1 is known to the UI (drawn by the transcript or the
    harnessing page, or left out on purpose). A type added later (a newer server, another group's branch) never
    breaks anything: it becomes one compact line with its own words, or nothing when it carries none."""
    types = backend_event_types()
    assert {"start", "assistant", "tool_call", "llm_retry", "compacted", "compaction_error", "adopt_skipped", "adopted",
            "applied", "auto_harness_error", "notify", "*_delta"} <= types  # (the scan works)
    out = run_ts(tmp_path, f"""
      out.known = [...L.KNOWN_EVENTS]
      out.later = {json.dumps(sorted(types - PHASE1_EVENTS - {"*_delta"}))}
        .filter((t: string) => !L.KNOWN_EVENTS.has(t) && !t.endsWith('_delta')) // (live text: the streaming reply)
        .map((t: string) => L.unknownEventNote({{ type: t, text: 'algo' }}))
      out.notes = [
        {{ type: 'connector_skipped', connector: 'Burp', text: 'no arrancó hace 1 min; se reintenta en 4 min' }},
        {{ type: 'brand_new', value: 3 }},
        {{ type: 'other_new', reason: 'Espera a que termine la respuesta en curso' }},
        {{ type: 'thinking_delta', text: 'x' }}, {{ type: 'assistant', text: 'x' }}, {{ type: 3 }}, {{}},
      ].map((e) => L.unknownEventNote(e as any))
      setLang('en'); out.en = L.unknownEventNote({{ type: 'other_new', reason: 'Espera a que termine la respuesta en curso' }}); setLang('es')
    """)
    assert PHASE1_EVENTS <= set(out["known"]), sorted(PHASE1_EVENTS - set(out["known"]))
    assert all(n == "algo" for n in out["later"])  # newer types still show
    assert out["notes"] == ["Burp: no arrancó hace 1 min; se reintenta en 4 min", "",
                            "Espera a que termine la respuesta en curso", "", "", "", ""]
    assert out["en"] == "Wait for the current answer to finish"


def test_the_transcript_draws_every_event_and_never_breaks_on_odd_ones(tmp_path):
    """S6: one of each event a chat run can hold (phase-1 ones included: fallback compactions, interrupted replies,
    cancelled approvals, auto-harness errors), plus events missing fields and types this version doesn't know: the
    transcript renders all of it, with no 'NaN', no raw codes and no crash. The live-only compaction_error shows while
    the run goes on."""
    events = [
        {"type": "start", "model": "mock/demo", "task": "t"},
        {"type": "thinking", "text": "pienso"},
        {"type": "assistant", "text": "Hola mundo"},
        {"type": "tool_call", "id": "a", "name": "bash", "args": {"command": "ls -la"}},
        {"type": "tool_result", "id": "a", "name": "bash", "output": "x", "is_error": False},
        {"type": "tool_call", "id": "b", "name": "write_file"},  # no args
        {"type": "tool_call", "id": "c"},  # no name either
        {"type": "llm_retry", "attempt": 2, "max": 5, "wait": 3, "reason": "read timeout"},
        {"type": "context_learned", "limit": 32000},
        {"type": "masked", "freed": 1200, "used": 9000, "limit": 32000},
        {"type": "compacted", "used": 29000, "after": 9000, "limit": 32000, "summary": "## Goal", "path": "ctx/c-1.md",
         "n": 1, "reason": "threshold", "fallback": True},
        {"type": "compacted", "reason": "no room for output"},  # no numbers
        {"type": "compaction_error", "text": "summary rejected: empty"},
        {"type": "image", "path": "a.png", "caption": "uno"}, {"type": "image", "path": "a.png"},
        {"type": "file", "path": "informe.pdf", "size": 2048},
        {"type": "notify", "text": "aviso enviado", "sent": True},
        {"type": "bot_created", "bot_id": 3, "name": "Vigía"},
        {"type": "connector_error", "connector": "Burp", "text": "Connection closed"},
        {"type": "auto_harness", "run_id": 9, "reason": "necesitó 30 pasos"},
        {"type": "auto_harness_skipped", "reason": "necesitó 30 pasos", "why": "ya hay un harnessing en curso en este proyecto"},
        {"type": "auto_harness_error", "text": "RuntimeError: boom"},
        {"type": "approval_request", "id": "n1", "call_id": "c1", "name": "mcp__burp__scan"},  # no args
        {"type": "approval_result", "id": "n1", "call_id": "c1", "decision": "cancelled"},
        {"type": "connector_skipped", "connector": "Burp", "text": "no arrancó hace 1 min"},  # a newer server's type
        {"type": "brand_new", "value": 3},  # unknown and wordless: nothing
        {"type": "phase", "phase": "probe"}, {"type": "applied", "changes": {}},  # harnessing's own: not in a chat
        {"type": "assistant", "text": "Respuesta a medias", "interrupted": True},
        {"type": "cancelled"},
        {"type": "done", "status": "cancelled", "steps": 3, "usage": {"cost_usd": 0.0123}},
    ]
    out = run_ts(tmp_path, f"""
      import Transcript from {json.dumps(str(SRC / 'components' / 'Transcript.tsx'))}
      out.page = html(<Transcript events={{{json.dumps(events)} as any}} status="cancelled" runId={{7}} />)
      out.live = html(<Transcript events={{[{{ type: 'start' }}] as any}} status="running" notice="read timeout (stalled stream)" />)
      out.ended = html(<Transcript events={{[{{ type: 'start' }}] as any}} status="done" notice="read timeout" />)
      out.junk = html(<Transcript events={{[null, 3, {{ type: 'assistant', text: 'sigue' }}] as any}} status="done" />)
    """)
    page = out["page"]
    assert re.search(r"límite real: 32\D?000 tokens", page) and re.search(r"Contexto aligerado: 1\D?200 tokens", page)
    for text in ("Razonamiento", "Hola mundo", "ls -la", "La llamada al modelo se reintentó 1 vez (read timeout)",
                 "resumen de respaldo: falló el resumen del modelo", "Contexto compactado (? de ? tokens)",
                 "no quedaba sitio para la respuesta", "No se pudo resumir el contexto (el resumen salió vacío)",
                 "informe.pdf", "aviso enviado", "Vigía", "El conector Burp no arrancó", "harnessing #9",
                 "Auto-harnessing no lanzado", "No se pudo lanzar el auto-harnessing: RuntimeError: boom",
                 "cancelado (se detuvo la respuesta)", "Burp: no arrancó hace 1 min", "Respuesta a medias",
                 "Respuesta interrumpida: se detuvo antes de terminar.", "Cancelado.", "3 pasos"):
        assert text in page, text
    assert page.count('src="/api/threads/') == 1  # the same picture once
    for absent in ("NaN", "undefined", "threshold", "brand_new", "probe"):
        assert absent not in page, absent
    assert "No se pudo resumir el contexto (read timeout (stalled stream))" in out["live"]
    assert "No se pudo resumir" not in out["ended"]  # a live notice only matters while the run goes on
    assert "sigue" in out["junk"]  # non-object entries are dropped, the rest is drawn


def test_a_failed_context_summary_is_explained(tmp_path):
    """S6: the live 'compaction_error' says what happened, with the summarizer's own problem codes in words."""
    from nesicode.agent.context import clean_summary

    problems = [clean_summary("", "end_turn")[1], clean_summary("## Goal\nx", "max_tokens")[1],
                clean_summary("<think>sigo pensando", "end_turn")[1]]
    assert problems == ["empty", "cut at max_tokens before its last section", "unclosed reasoning block"]
    out = run_ts(tmp_path, f"""
      out.t = [...{json.dumps(problems)}.map((p) => 'summary rejected: ' + p), 'summary rejected: otra cosa',
        'read timeout (stalled stream)', ''].map(L.compactionErrorText)
    """)
    lead = "No se pudo resumir el contexto ({}): se reintenta o se sigue con un resumen de respaldo."
    assert out["t"] == [lead.format(w) for w in (
        "el resumen salió vacío", "el resumen salió cortado", "el modelo se quedó razonando sin escribir el resumen",
        "otra cosa", "read timeout (stalled stream)", "?")]


def _failed_summary_stream(tmp_path: Path, monkeypatch) -> dict:
    """The real loop, whose summarizer call fails, wired to the real jobs emitter: the live items a client gets
    replayed when it connects during that compaction, and right after it; and the batch a poll taken just before the
    failure delivers (the fallback compaction follows within ms, so usually in the same batch)."""
    from types import SimpleNamespace

    import nesicode.agent.loop as loop
    from nesicode.agent import Agent, Workspace
    from nesicode.harness import base_harness
    from nesicode.llm.base import Conversation, LLMError, Provider, ToolCall, Turn, Usage
    from nesicode.server.jobs import Jobs

    steps: list[int] = []

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
                raise LLMError("read timeout (stalled stream)")
            steps.append(1)
            if len(steps) >= 8:
                return Turn("fin", [], "end_turn", Usage(self.ntokens(), 10))
            call = ToolCall(f"w{len(steps)}", "write_file", {"path": f"p{len(steps)}.txt", "content": "y" * 6_000})
            self.chars += 6_100
            return Turn("", [call], "tool_use", Usage(self.ntokens(), 40))

    class Prov(Provider):
        name = "fake"

        def conversation(self, model, system, tools, options):
            return Conv(model, system, tools, options)

    monkeypatch.setattr(loop, "get_provider", lambda spec, keys=None: (Prov(), "m"))
    server = SimpleNamespace(db=SimpleNamespace(add_event=lambda run_id, event: None), live={})
    server_emit = Jobs.emitter(server, 1)  # what the stream replays: server.live[1].items (this generation's)
    seen: dict = {"persisted": []}

    def emit(event: dict) -> None:
        server_emit(event)
        live = server.live[1].items
        if not event.get("_persist", True) and event.get("type") == "compaction_error" and "during" not in seen:
            seen["during"], seen["mark"] = list(live), len(live) - 1
        elif event.get("_persist", True) and "during" in seen and "after" not in seen:
            seen["persisted"].append(event.get("type"))
        elif event.get("type") == "context" and event.get("reason") == "compacted" and "during" in seen and "after" not in seen:
            seen["after"], seen["batch"] = list(live), live[seen["mark"]:]

    res = Agent(model="fake/m", harness=base_harness(), workspace=Workspace(tmp_path / "ws"), emit=emit,
                context_limit=8_000, output_limit=4_000).run("tarea")
    assert res.status == "done" and "after" in seen, seen.keys()
    return seen


def test_the_failed_summary_notice_ends_with_its_compaction(tmp_path, monkeypatch):
    """S6 (review): the live 'compaction_error' notice shows while that compaction goes on and is gone once it ended,
    also when the fallback compaction comes in the same batch as the failure (the usual case: within ms) and when a
    reconnect replays the generation's live items — never a stale «se reintenta…» during the next model call."""
    seen = _failed_summary_stream(tmp_path, monkeypatch)
    assert seen["persisted"][:1] == ["compacted"]  # (the setup: the fallback compaction follows the failure)
    err = lambda text, depth=0: {"type": "compaction_error", "text": text, "depth": depth}  # noqa: E731
    ctx = lambda reason=None, depth=0: {"type": "context", "used": 9000, "limit": 32000, "reason": reason, "depth": depth}  # noqa: E731
    out = run_ts(tmp_path, f"""
      const seen = {json.dumps({k: seen[k] for k in ("during", "after", "batch")})}
      out.real = [seen.during, seen.after, seen.batch].map((items) => L.compactionNotice(items))
      out.n = {json.dumps([
          [err("read timeout")],  # the summary failed: shown
          [err("read timeout"), {"type": "text_delta", "text": "x"}, ctx("compacted")],  # its compaction ended: gone
          [ctx("compacted"), err("summary rejected: empty")],  # an earlier compaction, then this one failing: shown
          [{"type": "text_delta", "text": "x"}, ctx("masked"), ctx()],  # nothing about it: keep what is shown
          [err("x", 1), ctx("compacted", 1)],  # a sub-agent's: not this run's notice
          [err("")],
      ])}.map((items: unknown[]) => L.compactionNotice(items))
      out.odd = [L.compactionNotice([null, 3, 'x']), L.compactionNotice(undefined as any)]
    """)
    assert out["real"] == ["read timeout (stalled stream)", "", ""]
    assert out["n"] == ["read timeout", "", "summary rejected: empty", None, None, "?"]
    assert out["odd"] == [None, None]


def test_an_odd_done_or_cost_never_blanks_the_chat(tmp_path):
    """S6 (review): the run's last line ('done': steps, cost) is drawn outside the per-event boundary, so an odd value
    there must not throw (it used to: a non-number cost crashed the whole chat page)."""
    out = run_ts(tmp_path, f"""
      import Transcript from {json.dumps(str(SRC / 'components' / 'Transcript.tsx'))}
      import {{ money }} from {json.dumps(str(SRC / 'api.ts'))}
      out.page = html(<Transcript events={{[{{ type: 'assistant', text: 'hola' }}, {{ type: 'done', status: 'done', usage: {{ cost_usd: '0.5' }} }}] as any}} status="done" />)
      out.ok = html(<Transcript events={{[{{ type: 'done', status: 'done', steps: 3, usage: {{ cost_usd: 0.0123 }} }}] as any}} status="done" />)
      out.money = [0.5, 0.001, undefined, null, '0.5', NaN, Infinity].map((x) => money(x as any))
    """)
    assert "hola" in out["page"] and "? pasos · —" in out["page"] and "undefined" not in out["page"]
    assert "3 pasos · $0.01" in out["ok"]
    assert out["money"] == ["$0.50", "$0.0010", "—", "—", "—", "—", "—"]


# ---------------------------------------------------------------------------------------------------- S7 409 answers
def test_busy_project_answers_show_the_servers_words_in_both_languages(tmp_path):
    """S7: a 409 shows the server's detail (Spanish), translated for English users."""
    busy = "El proyecto está ocupado por otra conversación; inténtalo cuando termine."
    stopping = ("Hay una respuesta de este proyecto que aún no se ha detenido (p. ej. esperando a un conector). "
                "Vuelve a intentarlo en unos segundos.")
    assert stopping in re.sub(r'"\s*\n\s*"', "", (ROOT / "src/nesicode/server/app.py").read_text(encoding="utf-8"))
    out = run_ts(tmp_path, f"""
      const msgs = {json.dumps([busy, stopping])}
      out.es = msgs.map((d) => L.errorMessage(409, {{ detail: d }}))
      setLang('en'); out.en = msgs.map((d) => L.errorMessage(409, {{ detail: d }})); setLang('es')
    """)
    assert out["es"] == [busy, stopping]
    assert out["en"] == ["The project is busy with another conversation; try again when it finishes.",
                         "A reply in this project hasn't stopped yet (e.g. it is waiting for a connector). Try again in a few seconds."]


# ---------------------------------------------------------------------------------------------------- S8 import preview
def _import_preview() -> dict:
    """What POST /api/config/import with preview=1 answers for a file with a bit of everything (the real validation
    and summary; the database-dependent notes are left out)."""
    from pydantic import BaseModel

    from nesicode.server.config_io import summary, validate_import

    class Prefs(BaseModel):
        language: str | None = None
        exec_mode: str | None = None
        share_workspace: bool | None = None
        browser_usage: str | None = None

    doc = {
        "providers": [{"name": "equipo", "label": "Equipo", "kind": "openai", "base_url": "https://llm.example.test/v1",
                       "models": ["m1"], "api_key": "sk-test"}],
        "connectors": [
            {"name": "notas", "label": "Notas", "kind": "stdio", "template": "custom", "command": "npx",
             "args": ["-y", "mcp-notas"], "enabled": 1, "usage": "auto"},
            {"name": "remoto", "label": "Remoto", "kind": "http", "template": "custom", "url": "https://mcp.example.test/mcp"},
            {"name": "MAL NOMBRE", "label": "Roto", "kind": "stdio"},
        ],
        "settings": {"exec_mode": "ask_all", "share_workspace": True, "browser_usage": "ask", "language": "en",
                     "hidden_providers": ["mock"]},
        "role_models": {"executor_model": "equipo/m1", "from": "Portátil"},
        "project_template": {"from": "Proyecto X", "check_command": "uvx pytest -q", "harness": {"a.md": "x"}},
        "bots": [{"name": "Resumen", "instructions": "resume", "triggers": [{"kind": "cron", "spec": "0 8 * * *"}]}],
        "telegram": {"token": "123456:" + "A" * 30},
    }
    clean, report = validate_import(doc, auth_enabled=False, check_url=lambda u: str(u).strip().rstrip("/"),
                                    settings_model=Prefs, current_settings={"exec_mode": "auto"})
    return {**summary(clean), **report}


def test_the_import_preview_shows_what_the_file_would_change_before_anything_is_applied(tmp_path):
    """S8: the confirm step lists where each provider points, what each connector runs (and that a command from a file
    arrives switched off), the preferences' values, the models by role (applied to every project), the template, the
    bots and the Telegram bot, plus the server's notes and what will not be imported."""
    preview = _import_preview()
    assert preview["notes"] and preview["skipped"]  # (the file has some of each)
    out = run_ts(tmp_path, f"""
      import {{ ImportPreviewView }} from {json.dumps(str(SRC / 'components' / 'ConfigTransfer.tsx'))}
      const p = {json.dumps(preview)}
      out.sections = L.previewSections(p)
      out.page = html(<ImportPreviewView p={{p}} />)
      out.empty = html(<ImportPreviewView p={{{{}}}} />)
      out.odd = L.previewSections({{ details: {{ providers: 'x', connectors: null }}, bots: 3 }} as any)
    """)
    sec = {s["title"]: s["items"] for s in out["sections"]}
    assert list(sec) == ["Proveedores (con sus API keys)", "Conectores", "Modelos por función", "Preferencias",
                         "Plantilla de proyecto", "Bots", "Telegram"]
    assert sec["Proveedores (con sus API keys)"] == [{"label": "equipo", "code": "https://llm.example.test/v1",
                                                      "note": "Compatible con OpenAI"}]
    notas, remoto = sec["Conectores"]
    assert notas["label"] == "Notas (notas)" and notas["code"] == "npx -y mcp-notas"
    assert notas["note"] == "ejecuta un comando en este equipo · llega desactivado" and not notas["warn"]
    assert remoto["code"] == "https://mcp.example.test/mcp" and remoto["note"] == "servidor remoto · Solo cuando lo pida"
    assert sec["Modelos por función"] == [{"label": "Chat", "code": "equipo/m1", "note": "se aplica a todos tus proyectos donde funcione"}]
    prefs = {i["label"]: i["note"] for i in sec["Preferencias"]}
    assert prefs == {"Permisos de ejecución": "Pedir para todo", "Compartir la carpeta del proyecto entre conversaciones": "sí",
                     "Cuándo puede usar tu Chrome": "Preguntarme siempre", "Idioma": "English", "Proveedores ocultos": "mock"}
    assert sec["Plantilla de proyecto"][0]["label"] == "«Proyecto X»" and sec["Plantilla de proyecto"][0]["code"] == "uvx pytest -q"
    assert sec["Bots"] == [{"label": "Resumen"}]
    assert sec["Telegram"][0]["warn"] is True
    page = out["page"]
    for text in ("https://llm.example.test/v1", "npx -y mcp-notas", "llega desactivado", "Ten en cuenta:",
                 "importado DESACTIVADO", "No se importará:", "datos no válidos", "sustituye al bot de Telegram"):
        assert text in page, text
    assert "123456:" not in page and "sk-test" not in page  # secrets never reach the preview
    assert "El archivo no trae nada que importar." in out["empty"]
    assert out["odd"] == []
