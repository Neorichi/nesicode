"""Web UI audit fixes (group I) — the pure logic behind them, run offline with node.

The repo has no JS test runner, so each test bundles a tiny TypeScript entry that imports `web/src/logic.ts` with the
esbuild that Vite already ships, runs it with node and checks the JSON it prints. Skipped when node or
`web/node_modules` is missing. The rest of the fixes (keyed pages, stale-response guards, the SSE stream, the
microphone, …) are checked by `npm run build` (tsc) and the browser checks described in the fix report.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
NODE = shutil.which("node")
ESBUILD = WEB / "node_modules" / "esbuild"

pytestmark = pytest.mark.skipif(not NODE or not ESBUILD.is_dir(), reason="needs node and web/node_modules (esbuild)")


def run_ts(tmp_path: Path, body: str, head: str = ""):
    """Bundle `body` (TypeScript, with `L` = web/src/logic.ts) for node, run it, return what it printed as JSON.
    `head`: imports that must run first (e.g. browser fakes)."""
    entry = tmp_path / "entry.ts"
    entry.write_text(f"{head}\nimport * as L from {json.dumps(str(WEB / 'src' / 'logic.ts'))}\n"
                     f"const out: Record<string, unknown> = {{}}\n{body}\nconsole.log(JSON.stringify(out))\n")
    bundle = tmp_path / "bundle.cjs"
    build = tmp_path / "build.cjs"
    build.write_text(
        f"require({json.dumps(str(ESBUILD))}).buildSync({{entryPoints: [{json.dumps(str(entry))}], bundle: true, "
        f"platform: 'node', format: 'cjs', outfile: {json.dumps(str(bundle))}, logLevel: 'error', "
        f"nodePaths: [{json.dumps(str(WEB / 'node_modules'))}]}})\n")
    for args in ([NODE, str(build)], [NODE, str(bundle)]):
        p = subprocess.run(args, capture_output=True, text=True, timeout=60, cwd=tmp_path)
        assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_api_errors_never_show_a_json_syntax_error(tmp_path):
    """web-ui#15: non-JSON error bodies (Starlette 500, proxy 502 HTML) read as an error status, not a SyntaxError."""
    out = run_ts(tmp_path, """
      out.html = L.parseBody('<html><body>Bad gateway</body></html>') === undefined
      out.plain = L.parseBody('Internal Server Error') === undefined
      out.empty = L.parseBody('')
      out.json = L.parseBody('{"a": 1}')
      out.e500 = L.errorMessage(500, undefined)
      out.e502 = L.errorMessage(502, undefined)
      out.e404 = L.errorMessage(404, null)
      out.detail = L.errorMessage(422, { detail: 'El endpoint debe ser una URL http(s)://' })
      out.list = L.errorMessage(422, { detail: [{ msg: 'a' }, { msg: 'b' }] })
      out.obj = L.errorMessage(400, { detail: { msg: 'raro' } })
      out.harness = L.errorMessage(422, { errors: ['x'] })
    """)
    assert out["html"] and out["plain"] and out["empty"] is None and out["json"] == {"a": 1}
    assert out["e500"] == "Error interno del servidor (error 500)."
    assert "no responde" in out["e502"] and "502" in out["e502"]
    assert out["e404"] == "Error 404"
    assert out["detail"] == "El endpoint debe ser una URL http(s)://"
    assert out["list"] == "a; b" and out["obj"] == "raro" and out["harness"] == "El harness no es válido"
    assert not any("JSON" in str(v) for v in out.values())


def test_markdown_images_only_load_by_themselves_from_this_app(tmp_path):
    """web-ui#6: remote image URLs (exfiltration beacons) are not loaded without a click; local/inline ones are."""
    out = run_ts(tmp_path, """
      const base = 'http://127.0.0.1:8420/p/1/c/2'
      const srcs = ['https://attacker.example/x.png?d=SECRET', 'http://127.0.0.1:9999/leak', '//evil.example/a.png',
        '/\\\\evil.example/a.png', 'http://localhost:8420/api/x', 'javascript:alert(1)',
        '/api/threads/2/file?path=a.png', 'chart.png', 'http://127.0.0.1:8420/api/health',
        'data:image/png;base64,iVBORw0KGgo=', 'data:text/html,<b>x</b>']
      out.allowed = srcs.filter((s) => L.allowedImageSrc(s, base))
    """)
    assert out["allowed"] == ["/api/threads/2/file?path=a.png", "chart.png", "http://127.0.0.1:8420/api/health",
                              "data:image/png;base64,iVBORw0KGgo="]


def test_paste_keeps_office_text_and_still_uploads_files(tmp_path):
    """web-ui#14: Excel/Word text that carries a picture of itself is pasted as text; screenshots/files upload."""
    out = run_ts(tmp_path, """
      const png = (name: string) => ({ name, type: 'image/png' })
      out.office = L.pasteUploads('Nombre\\tImporte\\nAna\\t10', [png('image.png')])
      out.officeTiff = L.pasteUploads('Hola', [{ name: 'image.tiff', type: 'image/tiff' }])
      out.screenshot = L.pasteUploads('', [png('image.png')])
      out.finderImage = L.pasteUploads('foto.png', [png('foto.png')])
      out.finderPdf = L.pasteUploads('informe.pdf', [{ name: 'informe.pdf', type: 'application/pdf' }])
      out.textOnly = L.pasteUploads('solo texto', [])
    """)
    assert out == {"office": False, "officeTiff": False, "screenshot": True, "finderImage": True,
                   "finderPdf": True, "textOnly": False}


def test_llm_retry_events_collapse_into_one_line(tmp_path):
    """New event: retries of one model call show as one status line (the latest attempt); unknown events are fine."""
    out = run_ts(tmp_path, """
      const r = (attempt: number, depth = 0) => ({ type: 'llm_retry', attempt, max: 5, wait: 7.6, reason: 'read timeout', depth })
      const events = [{ type: 'start' }, r(2), r(3), { type: 'brand_new_event' }, { type: 'assistant', text: 'x' },
        r(2), { type: 'tool_call' }, r(2, 1), r(3, 1), r(2)]
      out.groups = [...L.retryGroups(events as any).entries()]
      out.active = L.retryText(r(3) as any, 2, true)
      out.past = L.retryText(r(3) as any, 2, false)
      out.once = L.retryText(r(2) as any, 1, false)
      out.bare = L.retryText({ type: 'llm_retry' } as any, 1, true)
    """)
    assert out["groups"] == [[2, 2], [5, 1], [8, 2], [9, 1]]
    assert out["active"] == "Reintentando la llamada al modelo (3/5) en 8 s: read timeout"
    assert out["past"] == "La llamada al modelo se reintentó 2 veces (read timeout)"
    assert out["once"] == "La llamada al modelo se reintentó 1 vez (read timeout)"
    assert out["bare"] == "Reintentando la llamada al modelo"


def test_context_drop_reasons_are_named(tmp_path):
    """The meter and the compaction note say why the context dropped (backend sends English codes)."""
    out = run_ts(tmp_path, """
      out.r = ['compacted', 'masked', 'no room for output', 'context exceeded', 'truncated reply', 'nuevo', '', undefined]
        .map((x) => L.compactReason(x as any))
    """)
    assert out["r"] == ["compactado", "aligerado", "no quedaba sitio para la respuesta",
                        "el modelo avisó de que se superaba su contexto", "la respuesta salió cortada", "nuevo", "", ""]


def test_runs_are_never_duplicated_and_events_dedupe(tmp_path):
    """web-ui#16: a late POST /messages response doesn't add a run a reload already shows; events append once."""
    out = run_ts(tmp_path, """
      const run = (id: number, status = 'queued') => ({ id, status, task: 't' + id, events: [] }) as any
      let runs = L.withRun([run(1, 'done')], run(2))
      runs = L.withRun(runs, run(2))
      out.ids = runs.map((r: any) => r.id)
      runs = L.withEvent(runs, 2, { seq: 1, type: 'start' })
      runs = L.withEvent(runs, 2, { seq: 1, type: 'start' })
      runs = L.withEvent(runs, 2, { seq: 2, type: 'assistant', text: 'hola' })
      out.events = runs[1].events.map((e: any) => e.seq)
      out.status = runs[1].status
    """)
    assert out == {"ids": [1, 2], "events": [1, 2], "status": "running"}


def test_a_reload_never_loses_local_changes_or_streamed_events(tmp_path):
    """web-ui#16/#18: a reload that started before a send / a queue removal / streamed events can't undo them."""
    out = run_ts(tmp_path, """
      const ev = (seq: number) => ({ seq, type: 'x' })
      const prev = [
        { id: 1, status: 'running', task: 'a', events: [ev(1), ev(2), ev(3)] },   // the stream got 2 and 3
        { id: 2, status: 'waiting', task: 'b', events: [] },                        // taken out of the queue here
        { id: 3, status: 'queued', task: 'c', events: [] },                         // just sent from this page
      ] as any
      const snapshot = [
        { id: 1, status: 'queued', task: 'a', events: [ev(1)] },
        { id: 2, status: 'waiting', task: 'b', events: [] },
      ] as any
      const merged = L.mergeLoaded(prev, snapshot, new Set([3]), new Set([2]))
      out.ids = merged.map((r: any) => r.id)
      out.events = merged[0].events.map((e: any) => e.seq)
      out.status = merged[0].status
      const later = L.mergeLoaded(merged, [{ id: 1, status: 'done', events: [ev(1), ev(2), ev(3), ev(4)] },
        { id: 2, status: 'cancelled', events: [] }, { id: 3, status: 'running', events: [ev(1)] }] as any, new Set(), new Set([2]))
      out.later = later.map((r: any) => `${r.id}:${r.status}:${r.events.length}`)
      out.plain = L.mergeLoaded([], snapshot, new Set(), new Set()).map((r: any) => r.id)
    """)
    assert out["ids"] == [1, 3] and out["events"] == [1, 2, 3] and out["status"] == "running"
    assert out["later"] == ["1:done:4", "2:cancelled:0", "3:running:1"]
    assert out["plain"] == [1, 2]


def test_interrupted_text_hides_once_the_server_has_it(tmp_path):
    """web-ui#11: the frozen partial reply stays until an assistant event (flagged or with that text) arrives."""
    out = run_ts(tmp_path, """
      const text = 'Paso uno: miro el espacio de trabajo. palabra0 palabra1'
      out.none = L.interruptedSaved([{ type: 'start' }, { type: 'cancelled' }] as any, text)
      out.flag = L.interruptedSaved([{ type: 'assistant', text: 'otra', interrupted: true }] as any, text)
      out.same = L.interruptedSaved([{ type: 'assistant', text: text + ' palabra2' }] as any, text)
      out.sub = L.interruptedSaved([{ type: 'assistant', text, depth: 1 }] as any, text)
    """)
    assert out == {"none": False, "flag": True, "same": True, "sub": False}


def test_number_fields_and_intervals_keep_what_the_user_types(tmp_path):
    """web-ui#9/#13: an empty or half-typed number never snaps back to a default nor saves junk (20→50 gave 205)."""
    out = run_ts(tmp_path, """
      out.clamp = [L.clampInt('50', 3, 300), L.clampInt('2', 3, 300), L.clampInt('999', 3, 300), L.clampInt('', 3, 300),
        L.clampInt('5a', 3, 300)]
      out.parse = [L.parseInterval('60'), L.parseInterval('2h'), L.parseInterval(' 1d '), L.parseInterval(''), L.parseInterval('0 8 * * *')]
      out.spec = [L.intervalSpec('15', 'm'), L.intervalSpec('3', 'h'), L.intervalSpec('', 'h'), L.intervalSpec('0', 'm'),
        L.intervalSpec('015', 'd')]
      out.editable = ['cron', 'interval', 'url', 'once', 'webhook'].filter((k) => L.EDITABLE_TRIGGERS.includes(k))
    """)
    assert out["clamp"] == [50, 3, 300, None, None]
    assert out["parse"] == [{"n": "60", "u": "m"}, {"n": "2", "u": "h"}, {"n": "1", "u": "d"}, None, None]
    assert out["spec"] == ["15", "3h", None, None, "15d"]
    assert out["editable"] == ["cron", "interval", "url"]  # web-ui#7: 'once'/'webhook' are never sent back


def test_provider_endpoint_without_scheme_is_completed(tmp_path):
    """web-ui#8: 'localhost:11434/v1' (common for Ollama/LM Studio) gets http://; public hosts get https://."""
    out = run_ts(tmp_path, """
      out.u = ['localhost:11434/v1', '127.0.0.1:1234/v1', '192.168.1.20:8000', 'api.example.com/v1',
        'https://api.example.com/v1', 'http://x', '', '[::1]:8080/v1', 'mac.local:11434'].map(L.normalizeBaseUrl)
    """)
    assert out["u"] == ["http://localhost:11434/v1", "http://127.0.0.1:1234/v1", "http://192.168.1.20:8000",
                        "https://api.example.com/v1", "https://api.example.com/v1", "http://x", "",
                        "http://[::1]:8080/v1", "http://mac.local:11434"]


# Browser stand-ins for streamRun (web/src/api.ts): a manual clock behind window.setTimeout/clearTimeout, a document
# whose visibility the test flips, and an EventSource the test opens, feeds and breaks.
STREAM_FAKES = r"""
const g = globalThis as any
let now = 0
let next = 1_000_000
const timers: { at: number; id: number; fn: () => void }[] = []
const realClear = g.clearTimeout
g.clearTimeout = (id: unknown) => {
  if (typeof id === 'number' && id >= 1_000_000) {
    const i = timers.findIndex((t) => t.id === id)
    if (i >= 0) timers.splice(i, 1)
  } else if (id !== undefined) realClear(id)
}
g.window = { setTimeout: (fn: () => void, ms: number) => { const id = next++; timers.push({ at: now + ms, id, fn }); return id } }
export function advance(ms: number) {
  const end = now + ms
  for (;;) {
    timers.sort((a, b) => a.at - b.at || a.id - b.id)
    const t = timers[0]
    if (!t || t.at > end) break
    timers.shift()
    now = t.at
    t.fn()
  }
  now = end
}
const vis = new Set<() => void>()
g.document = {
  visibilityState: 'visible',
  documentElement: {},
  addEventListener: (type: string, fn: () => void) => { if (type === 'visibilitychange') vis.add(fn) },
  removeEventListener: (type: string, fn: () => void) => { if (type === 'visibilitychange') vis.delete(fn) },
}
export function setVisible(v: boolean) {
  g.document.visibilityState = v ? 'visible' : 'hidden'
  vis.forEach((fn) => fn())
}
export const listeners = () => vis.size
export class FakeES {
  static CLOSED = 2
  static all: FakeES[] = []
  url: string
  readyState = 0
  closed = false
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  on: Record<string, ((m: { data: string }) => void)[]> = {}
  constructor(url: string) { this.url = url; FakeES.all.push(this) }
  addEventListener(type: string, fn: (m: { data: string }) => void) { (this.on[type] ??= []).push(fn) }
  close() { this.closed = true; this.readyState = 2 }
  open() { this.readyState = 1; this.onopen?.() }
  send(type: string, data: unknown) { (this.on[type] ?? []).forEach((fn) => fn({ data: JSON.stringify(data) })) }
  fail() { this.readyState = 0; this.onerror?.() }
}
g.EventSource = FakeES
"""


def test_run_stream_resumes_after_drops_and_hidden_tabs_and_is_followed_again_after_giving_up(tmp_path):
    """web-ui#12/#19 (+ review): a dropped stream reconnects after the last event it got and says so (the page drops
    its partial live text); a hidden tab gives its connection back and resumes when visible; a server that stays
    unreachable ends in one 'stream-lost', and the page then follows the run again unless a reload says it is over
    (before the review fix a failed reload left the chat 'running' forever once the server came back)."""
    fakes = tmp_path / "fakes.ts"
    fakes.write_text(STREAM_FAKES)
    head = (f"import * as F from {json.dumps(str(fakes))}\n"
            f"import * as A from {json.dumps(str(WEB / 'src' / 'api.ts'))}")
    out = run_ts(tmp_path, """
      const got = { open: 0, seqs: [] as number[], ends: [] as string[] }
      const close = A.streamRun(7, 10, { onOpen: () => got.open++, onEvent: (e) => got.seqs.push(e.seq!), onEnd: (s) => got.ends.push(s) })
      const s1 = F.FakeES.all.at(-1)!
      out.url1 = s1.url
      s1.open(); s1.send('event', { seq: 11, type: 'x' }); s1.send('event', { seq: 12, type: 'x' })
      s1.fail()
      out.closedOnError = s1.closed
      F.advance(999); out.beforeRetry = F.FakeES.all.length
      F.advance(2)
      const s2 = F.FakeES.all.at(-1)!
      out.url2 = s2.url
      s2.open(); out.opens = got.open
      F.setVisible(false); F.advance(4999); out.openWhileBrieflyHidden = !s2.closed
      F.advance(2); out.closedWhenHidden = s2.closed; out.connsWhileHidden = F.FakeES.all.length
      F.setVisible(true)
      const s3 = F.FakeES.all.at(-1)!
      out.url3 = s3.url
      s3.open(); s3.send('event', { seq: 13, type: 'done' }); s3.send('end', { status: 'done' })
      out.closedAtEnd = s3.closed; out.ends = got.ends; out.seqs = got.seqs
      F.advance(60000); F.setVisible(false); F.setVisible(true)
      out.connsAfterEnd = F.FakeES.all.length; out.listenersAfterEnd = F.listeners()
      close()
      const lost: string[] = []
      const before = F.FakeES.all.length
      A.streamRun(8, 0, { onEvent: () => {}, onEnd: (s) => lost.push(s) })
      for (let i = 0; i < 12; i++) { F.FakeES.all.at(-1)!.fail(); F.advance(16000) }
      out.attempts = F.FakeES.all.length - before
      out.lost = lost
      const quiet = F.FakeES.all.length
      const stop = A.streamRun(9, 0, { onEvent: () => {}, onEnd: () => lost.push('x') })
      F.FakeES.all.at(-1)!.fail(); stop(); F.advance(60000)
      out.afterClose = F.FakeES.all.length - quiet
      out.lostAfterClose = lost.length
      out.follow = [L.followAgainIn(null, 7), L.followAgainIn([{ id: 7, status: 'running' }], 7),
        L.followAgainIn([{ id: 7, status: 'queued' }], 7), L.followAgainIn([{ id: 7, status: 'done' }], 7),
        L.followAgainIn([{ id: 6, status: 'running' }], 7), L.followAgainIn([], 7)]
    """, head=head)
    assert out["url1"] == "/api/runs/7/stream?after=10"
    assert out["closedOnError"] and out["beforeRetry"] == 1
    assert out["url2"] == "/api/runs/7/stream?after=12" and out["opens"] == 2  # resumes after the last event received
    assert out["openWhileBrieflyHidden"] and out["closedWhenHidden"] and out["connsWhileHidden"] == 2
    assert out["url3"] == "/api/runs/7/stream?after=12"
    assert out["closedAtEnd"] and out["ends"] == ["done"] and out["seqs"] == [11, 12, 13]
    assert out["connsAfterEnd"] == 3 and out["listenersAfterEnd"] == 0
    assert out["attempts"] == 7 and out["lost"] == ["stream-lost"]  # gives up once, with a status that isn't the run's
    assert out["afterClose"] == 1 and out["lostAfterClose"] == 1  # the caller closing it stops every retry
    assert out["follow"] == [5000, 0, 0, None, None, None]  # unreachable: try again soon; over: never


def test_interrupted_text_never_shows_on_a_new_run_that_reuses_a_deleted_runs_id(tmp_path):
    """Review of web-ui#11: the partial text kept for a stopped run is keyed by id AND creation time. SQLite reuses
    the id of a deleted run (stop a reply, delete the conversation, start another one: the new reply got id 27 like
    the stopped one and showed its text as 'interrumpida')."""
    out = run_ts(tmp_path, """
      const stopped = { id: 27, created_at: 1762600000.25 }
      const reused = { id: 27, created_at: 1762600090.5 }
      out.same = L.runKey(stopped) === L.runKey({ ...stopped })
      out.reused = L.runKey(stopped) === L.runKey(reused)
    """)
    assert out == {"same": True, "reused": False}
