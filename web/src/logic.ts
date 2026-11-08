/**
 * Pure helpers (no DOM, no React state) behind several UI decisions. Kept apart so they can be tested with node
 * (tests/test_fix_web_ui.py bundles this file with esbuild).
 */
import type { Run, RunEvent } from './api'
import { t, translateError } from './i18n'

// ------------------------------------------------------------------ API responses
/** The message to show for a failed API call (FastAPI detail string/list/object, or a generic one). */
export function errorMessage(status: number, data: unknown): string {
  const d = (data as { detail?: unknown } | null)?.detail
  if (typeof d === 'string' && d) return translateError(d)
  if (Array.isArray(d) && d.length)
    return d.map((x) => translateError(typeof x === 'string' ? x : String((x as { msg?: unknown })?.msg ?? JSON.stringify(x)))).join('; ')
  if (d && typeof d === 'object') {
    const o = d as { msg?: unknown; message?: unknown }
    return translateError(String(o.msg ?? o.message ?? JSON.stringify(d)))
  }
  if ((data as { errors?: unknown } | null)?.errors) return t('El harness no es válido')
  // no JSON body (a proxy's HTML page, Starlette's plain "Internal Server Error"…)
  if (status === 502 || status === 503 || status === 504) return t('El servidor no responde (error {n}). Inténtalo de nuevo en un momento.', { n: status })
  if (status >= 500) return t('Error interno del servidor (error {n}).', { n: status })
  return `Error ${status}`
}

/** Parse a response body: JSON when it is JSON, otherwise `undefined` (never throws). */
export function parseBody(text: string): unknown {
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return undefined
  }
}

// ------------------------------------------------------------------ chat runs
/** Add the run returned by POST /messages, unless the list already has it (a reload may have shown it first). */
export function withRun(runs: Run[], run: Run): Run[] {
  return runs.some((r) => r.id === run.id) ? runs : [...runs, { ...run, events: run.events ?? [] }]
}

/**
 * A reload's runs, reconciled with what the page already shows: events that arrived by the stream after the
 * snapshot are kept; runs this page just created (`created`) that the snapshot predates are kept; runs the user
 * just took out of the queue (`removed`) stay hidden while the snapshot still shows them waiting.
 */
export function mergeLoaded(prev: Run[], fresh: Run[], created: Set<number>, removed: Set<number>): Run[] {
  const old = new Map(prev.map((r) => [r.id, r]))
  const out = fresh
    .filter((r) => !(removed.has(r.id) && r.status === 'waiting'))
    .map((r) => {
      const top = Math.max(0, ...(r.events ?? []).map((e) => e.seq ?? 0))
      const extra = (old.get(r.id)?.events ?? []).filter((e) => (e.seq ?? 0) > top)
      return extra.length ? { ...r, status: r.status === 'queued' ? 'running' : r.status, events: [...(r.events ?? []), ...extra] } : r
    })
  for (const r of prev) if (created.has(r.id) && !fresh.some((f) => f.id === r.id)) out.push(r)
  return out
}

/** Append streamed events to their run, once each (by seq). */
export function withEvent(runs: Run[], runId: number, e: RunEvent): Run[] {
  return runs.map((r) =>
    r.id === runId && !(r.events ?? []).some((x) => x.seq === e.seq)
      ? { ...r, status: r.status === 'queued' ? 'running' : r.status, events: [...(r.events ?? []), e] }
      : r,
  )
}

/** How long to wait before following a run again when the server couldn't even be asked whether it still runs. */
export const FOLLOW_RETRY_MS = 5000

/**
 * A run's stream gave up (STREAM_LOST: the server was unreachable for a while). When to follow the run again: now if
 * the reload says it is still going, in a few seconds if the reload failed too (the server may come back, as an
 * EventSource would wait for it), never if it is over. `runs` = the reloaded runs, or null when the reload failed.
 */
export function followAgainIn(runs: { id: number; status: string }[] | null, runId: number): number | null {
  if (!runs) return FOLLOW_RETRY_MS
  return runs.some((r) => r.id === runId && (r.status === 'queued' || r.status === 'running')) ? 0 : null
}

/**
 * Key for what the page keeps about a run for the session (its interrupted text): the id alone is not enough, since
 * SQLite gives a new run the id of a deleted one (delete a conversation, start another), and the new reply must
 * never show the old one's text.
 */
export const runKey = (r: { id: number; created_at: number }) => `${r.id}:${r.created_at}`

/** Was the text of a stopped/failed reply saved by the server after all (so the local copy can go)? */
export function interruptedSaved(events: RunEvent[], text: string): boolean {
  const head = text.trim().slice(0, 200)
  return events.some((e) => e.type === 'assistant' && !e.depth && (e.interrupted || (!!head && String(e.text ?? '').includes(head))))
}

// ------------------------------------------------------------------ model-call retries
/**
 * llm_retry events come one per retry of the same model call. Each run of consecutive ones (same depth) is shown
 * as a single line at its last event: returns that event's index → how many retries the run of events holds.
 */
export function retryGroups(events: RunEvent[]): Map<number, number> {
  const out = new Map<number, number>()
  let count = 0
  events.forEach((e, i) => {
    if (e.type !== 'llm_retry') {
      count = 0
      return
    }
    count += 1
    const next = events[i + 1]
    if (!next || next.type !== 'llm_retry' || (next.depth ?? 0) !== (e.depth ?? 0)) {
      out.set(i, count)
      count = 0
    }
  })
  return out
}

/** "Reintentando la llamada al modelo (2/5) en 8 s: reason" while it happens; a short past note afterwards. */
export function retryText(e: RunEvent, count: number, active: boolean): string {
  const reason = String(e.reason ?? '').trim().slice(0, 300)
  if (active) {
    const n = Number(e.attempt)
    const m = Number(e.max)
    const wait = Number(e.wait)
    return t('Reintentando la llamada al modelo') + (n && m ? ` (${n}/${m})` : '')
      + (wait > 0 ? ` ${t('en {s} s', { s: Math.max(1, Math.round(wait)) })}` : '') + (reason ? `: ${reason}` : '')
  }
  return (count === 1 ? t('La llamada al modelo se reintentó 1 vez') : t('La llamada al modelo se reintentó {n} veces', { n: count }))
    + (reason ? ` (${reason})` : '')
}

/** Why the context was compacted, in words (the agent sends English codes). */
export function compactReason(reason?: string): string {
  if (!reason) return ''
  const known: Record<string, string> = {
    'no room for output': t('no quedaba sitio para la respuesta'),
    'context exceeded': t('el modelo avisó de que se superaba su contexto'),
    'truncated reply': t('la respuesta salió cortada'),
    compacted: t('compactado'),
    masked: t('aligerado'),
  }
  return known[reason] ?? reason
}

// ------------------------------------------------------------------ markdown images
/** Images the answer may load by itself: this app's own URLs and inline data. Anything else needs a click. */
export function allowedImageSrc(src: string, base: string): boolean {
  const s = src.trim()
  if (/^data:image\//i.test(s)) return true
  try {
    const here = new URL(base)
    const u = new URL(s, base)
    if (u.protocol === 'blob:') return u.origin === here.origin
    return (u.protocol === 'http:' || u.protocol === 'https:') && u.origin === here.origin
  } catch {
    return false
  }
}

// ------------------------------------------------------------------ composer paste
const CLIPBOARD_IMAGE = /^(image|pasted[ _-]?graphic(?:[ _-]?\d+)?|unknown|blob)?\.(png|jpe?g|gif|tiff?|bmp|webp)$/i

/**
 * Should a paste upload its files? Office/iWork put the copied text AND a picture of it on the clipboard: then the
 * text is what the user wants. Screenshots (no text) and real files (copied in Finder/Explorer, own names) upload.
 */
export function pasteUploads(text: string, files: { name: string; type: string }[]): boolean {
  if (!files.length) return false
  const txt = text.trim()
  if (!txt) return true
  const lines = txt.split(/\r?\n/).map((l) => l.trim())
  if (files.some((f) => f.name && lines.includes(f.name))) return true // Finder/Explorer: the text is the file name
  return !files.every((f) => f.type.startsWith('image/') && (!f.name || CLIPBOARD_IMAGE.test(f.name)))
}

// ------------------------------------------------------------------ forms
/** Whole number typed in a field, clamped to [min, max]; null when it isn't a number. */
export function clampInt(text: string, min: number, max: number): number | null {
  if (!/^\s*\d+\s*$/.test(text)) return null
  return Math.min(max, Math.max(min, Number(text)))
}

export type IntervalUnit = 'm' | 'h' | 'd'

/** "90" → 90 minutes, "2h", "1d"; null when the spec isn't an interval. */
export function parseInterval(spec: string): { n: string; u: IntervalUnit } | null {
  const m = /^\s*(\d+)\s*([mhd]?)\s*$/.exec(spec)
  return m ? { n: m[1], u: (m[2] || 'm') as IntervalUnit } : null
}

/** The spec for a typed number and unit, or null while the number is empty/invalid (keep the last good one). */
export function intervalSpec(n: string, u: IntervalUnit): string | null {
  if (!/^\s*\d+\s*$/.test(n) || Number(n) < 1) return null
  return `${Number(n)}${u === 'm' ? '' : u}`
}

/** Bot triggers the editor can change. 'once' follow-ups (scheduled by the bot itself) and webhooks are kept apart. */
export const EDITABLE_TRIGGERS = ['cron', 'interval', 'url']

/** Endpoint typed without a scheme ("localhost:11434/v1") → http for local hosts, https otherwise. */
export function normalizeBaseUrl(raw: string): string {
  const s = raw.trim()
  if (!s || /^[a-z][a-z0-9+.-]*:\/\//i.test(s)) return s
  const host = s.startsWith('[') ? s.slice(1, s.indexOf(']')) : s.split(/[/:?#]/)[0].toLowerCase()
  const local = host === 'localhost' || host.endsWith('.local') || host === '::1' || host === '0.0.0.0'
    || /^(127|10)\.|^192\.168\.|^172\.(1[6-9]|2\d|3[01])\.|^169\.254\./.test(host)
  return `${local ? 'http' : 'https'}://${s}`
}
