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

/**
 * Messages waiting in the queue with nothing running: a moment's state, since the server starts the next one as soon
 * as a reply ends. Seeing it means the last reload caught that moment (and a reply that crashed doesn't bump the
 * conversation's updated_at, which the idle page watches), so the page should look again shortly.
 */
export function queueStalled(runs: { status: string }[]): boolean {
  return runs.some((r) => r.status === 'waiting') && !runs.some((r) => r.status === 'queued' || r.status === 'running')
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

/** Why the context was compacted, in words (the agent sends English codes; the usual 67% trigger needs none). */
export function compactReason(reason?: string): string {
  if (!reason || reason === 'threshold') return ''
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

const list = <T>(x: unknown): T[] => (Array.isArray(x) ? (x as T[]) : [])

// ------------------------------------------------------------------ applying changes to the project
/** A change Apply left out: the project's file was not touched (harnessing's three-way apply). */
export type Conflict = { path: string; reason: string; side_file?: string | null }
/** What an Apply returns: the paths it changed, plus what it did not touch and why. */
export type ApplyResult = { added?: string[]; modified?: string[]; removed?: string[]; skipped?: string[]; conflicts?: Conflict[] }

/** Why a rollout's change was left out (apply_rollout's codes, in the CLI's words: cli.CONFLICT_WHY). */
export function conflictWhy(reason: string): string {
  const known: Record<string, string> = {
    changed: t('cambió en el proyecto después de empezar la ejecución'),
    exists: t('ya existe en el proyecto con otro contenido'),
    unsafe: t('sale del proyecto por un enlace simbólico'),
    error: t('no se pudo escribir'),
  }
  return known[reason] ?? String(reason ?? '')
}

// Apply's skipped paths come as "path: reason", the reason in English (workspace._why, or the system's message).
const SKIP_WHY: [string, string][] = [
  ['it is a symbolic link (links are not followed; use the path it points to)', 'es un enlace simbólico (no se siguen enlaces: usa la ruta a la que apunta)'],
  ['a parent folder is a symbolic link or a file', 'una carpeta de su ruta es un enlace simbólico o un archivo'],
  ['not found', 'no se encontró'],
  ['not a regular file', 'no es un archivo normal'],
  ['it is a directory now', 'ahora es una carpeta'],
  ['Directory not empty', 'hay una carpeta con contenido en su lugar'],
  ['Is a directory', 'es una carpeta'],
  ['Permission denied', 'sin permiso para escribirlo'],
  ['Operation not permitted', 'operación no permitida'],
  ['No space left on device', 'no queda espacio en el disco'],
  ['Read-only file system', 'el disco es de solo lectura'],
  ['File name too long', 'el nombre es demasiado largo'],
]

/** "path: reason" → the path and the reason in the UI's language (an unknown reason is shown as it came). */
export function skippedPath(entry: string): { path: string; reason: string } {
  const s = String(entry ?? '')
  for (const [en, es] of SKIP_WHY) if (s.endsWith(`: ${en}`)) return { path: s.slice(0, -(en.length + 2)), reason: t(es) }
  const link = /^(.*): symlink to (.*) would point outside the project$/s.exec(s)
  if (link) return { path: link[1], reason: t('es un enlace que apuntaría fuera del proyecto ({to})', { to: link[2] }) }
  const i = s.indexOf(': ')
  return i > 0 ? { path: s.slice(0, i), reason: s.slice(i + 2) } : { path: s, reason: '' }
}

/** What an Apply really changed: a path it left out (conflict or skipped) is never counted, whatever list holds it. */
export function applyCounts(r: ApplyResult | null | undefined): { added: number; modified: number; removed: number; left: number } {
  const conflicts = list<Conflict>(r?.conflicts)
  const skipped = list<string>(r?.skipped)
  const out = new Set([...conflicts.map((c) => c?.path), ...skipped.map((s) => skippedPath(s).path)])
  const n = (xs: unknown) => list<string>(xs).filter((p) => !out.has(p)).length
  return { added: n(r?.added), modified: n(r?.modified), removed: n(r?.removed), left: conflicts.length + skipped.length }
}

/** One line for an Apply result: "Aplicados: +2 ~1 −0 · 1 sin aplicar (detalle abajo)". */
export function applySummary(r: ApplyResult | null | undefined): string {
  const c = applyCounts(r)
  const head = t('Aplicados: +{a} ~{m} −{r}', { a: c.added, m: c.modified, r: c.removed })
  return c.left ? `${head} · ${t('{n} sin aplicar (detalle abajo)', { n: c.left })}` : head
}

// ------------------------------------------------------------------ connectors
/**
 * The words that switch on a connector in "Solo cuando lo pida" mode (GET /api/connectors' trigger_words), for its
 * row: null when it is in another mode or the server doesn't say; [] when no word can switch it on.
 */
export function triggerWords(c: { usage?: string; trigger_words?: unknown }): string[] | null {
  if ((c.usage || 'mention') !== 'mention' || !Array.isArray(c.trigger_words)) return null
  return [...new Set(c.trigger_words.filter((w): w is string => typeof w === 'string' && !!w.trim()).map((w) => w.trim()))]
}

// ------------------------------------------------------------------ run events
/**
 * Event types the transcript draws with their own component, or leaves out on purpose: paired with another event
 * (tool_result, approval_result), live-only (context, *_delta: the meter and the streaming text), the run's own
 * bookkeeping (start, done) and harnessing's pipeline (shown by its own page).
 */
export const KNOWN_EVENTS = new Set([
  'start', 'assistant', 'thinking', 'tool_call', 'tool_result', 'error', 'cancelled', 'done', 'llm_retry', 'context',
  'context_learned', 'masked', 'compacted', 'compaction_error', 'image', 'file', 'notify', 'bot_created',
  'connector_error', 'auto_harness', 'auto_harness_skipped', 'auto_harness_error', 'approval_request',
  'approval_result', 'phase', 'stage', 'rollout', 'selected', 'applied', 'adopted', 'adopt_skipped', 'lesson',
])

/** The live 'compaction_error' (the summarizer's call failed, or its summary was rejected and is retried) in words. */
export function compactionErrorText(text: string): string {
  const s = String(text ?? '').trim()
  const rejected = /^summary rejected: (.*)$/s.exec(s)
  const known: Record<string, string> = {
    empty: t('el resumen salió vacío'),
    'cut at max_tokens before its last section': t('el resumen salió cortado'),
    'unclosed reasoning block': t('el modelo se quedó razonando sin escribir el resumen'),
  }
  const why = rejected ? known[rejected[1].trim()] ?? rejected[1].trim() : s.slice(0, 300)
  return t('No se pudo resumir el contexto ({why}): se reintenta o se sigue con un resumen de respaldo.', { why: why || '?' })
}

/**
 * A compact line for an event this version doesn't know (a newer server's): its own words if it has some, else ''
 * (nothing to say). Known types and live deltas give ''.
 */
export function unknownEventNote(e: { type?: unknown; [k: string]: unknown }): string {
  const type = typeof e?.type === 'string' ? e.type : ''
  if (!type || KNOWN_EVENTS.has(type) || type.endsWith('_delta')) return ''
  const pick = (...xs: unknown[]) => xs.find((x): x is string => typeof x === 'string' && !!x.trim())?.trim() ?? ''
  const words = pick(e.text, e.message, e.reason, e.why, e.note, e.detail)
  if (!words) return ''
  const who = pick(e.connector, e.label)
  return `${who ? `${who}: ` : ''}${translateError(words)}`.slice(0, 400)
}

// ------------------------------------------------------------------ config import preview
type PreviewProvider = { name: string; kind?: string; base_url?: string }
type PreviewConnector = { name: string; kind?: string; enabled?: boolean; command?: string; url?: string; usage?: string }
/** POST /api/config/import with preview=1: what the file would import here, what it can't and why. */
export type ImportPreview = {
  providers?: string[]; connectors?: string[]; settings?: string[]; project_template?: string | null; telegram?: boolean
  role_models?: Record<string, string>; bots?: string[]; notes?: string[]; skipped?: string[]
  details?: { providers?: PreviewProvider[]; connectors?: PreviewConnector[]; settings?: Record<string, unknown>; check_command?: string }
}
export type PreviewItem = { label: string; code?: string; note?: string; warn?: boolean }

const ROLE_NAME: Record<string, string> = {
  executor_model: 'Chat', improver_model: 'Mejora', vision_model: 'Visión', image_model: 'Crear imágenes', transcribe_model: 'Transcripción',
}
const usageText = (u?: string) => (u === 'auto' ? t('Cuando le sea útil') : u === 'ask' ? t('Preguntarme siempre') : t('Solo cuando lo pida'))

function settingItem(key: string, value: unknown): PreviewItem {
  const label: Record<string, string> = {
    language: t('Idioma'), exec_mode: t('Permisos de ejecución'), share_workspace: t('Compartir la carpeta del proyecto entre conversaciones'),
    auto_harness: t('Auto-harnessing'), auto_harness_daily: t('Auto-harnessing: máximo al día'),
    auto_harness_min_steps: t('Auto-harnessing: a partir de cuántos pasos'), auto_harness_min_errors: t('Auto-harnessing: a partir de cuántos errores'),
    browser_usage: t('Cuándo puede usar tu Chrome'), hidden_providers: t('Proveedores ocultos'),
  }
  const mode: Record<string, string> = { auto: t('Auto'), ask_bash: t('Pedir para comandos'), ask_all: t('Pedir para todo') }
  const shown = typeof value === 'boolean' ? (value ? t('sí') : t('no'))
    : key === 'exec_mode' ? mode[String(value)] ?? String(value)
    : key === 'browser_usage' ? usageText(String(value))
    : key === 'language' ? (value === 'en' ? 'English' : 'Español')
    : Array.isArray(value) ? value.join(', ') || '—'
    : value && typeof value === 'object' ? JSON.stringify(value) : String(value ?? '')
  return { label: label[key] ?? key, note: shown }
}

/**
 * The preview as sections of lines for the confirm step: where each provider points, what each connector runs
 * (and whether it arrives switched off), the preferences' values, the models by role (they replace the ones of every
 * project), the project template, bots and the Telegram bot. Only non-empty sections.
 */
export function previewSections(p: ImportPreview): { title: string; items: PreviewItem[] }[] {
  const d = p.details ?? {}
  const providers = list<PreviewProvider>(d.providers).map((x) => ({
    label: x.name, code: x.base_url || '', note: x.kind === 'anthropic' ? 'Anthropic' : t('Compatible con OpenAI'),
  }))
  const labels = list<string>(p.connectors)
  const connectors = list<PreviewConnector>(d.connectors).map((c, i) => {
    const local = c.kind === 'stdio'
    return {
      label: labels[i] && labels[i] !== c.name ? `${labels[i]} (${c.name})` : c.name,
      code: (local ? c.command : c.url) || '',
      note: [local ? t('ejecuta un comando en este equipo') : c.kind === 'builtin' ? t('integrado') : t('servidor remoto'),
        c.enabled === false ? t('llega desactivado') : usageText(c.usage)].join(' · '),
      warn: local && c.enabled !== false,
    }
  })
  const settings = Object.entries(d.settings ?? {}).map(([k, v]) => settingItem(k, v))
  const roles = Object.entries(p.role_models ?? {}).filter(([, v]) => v).map(([k, v]) => ({
    label: t(ROLE_NAME[k] ?? k), code: String(v), note: t('se aplica a todos tus proyectos donde funcione'),
  }))
  const template: PreviewItem[] = p.project_template
    ? [{ label: `«${p.project_template}»`, note: t('modelos, harness y comando de verificación para proyectos nuevos'), code: d.check_command || undefined }]
    : []
  const bots = list<string>(p.bots).map((name) => ({ label: name }))
  const telegram: PreviewItem[] = p.telegram ? [{ label: t('Bot de Telegram'), note: t('sustituye al bot de Telegram de este equipo'), warn: true }] : []
  return [
    { title: t('Proveedores (con sus API keys)'), items: providers },
    { title: t('Conectores'), items: connectors },
    { title: t('Modelos por función'), items: roles },
    { title: t('Preferencias'), items: settings },
    { title: t('Plantilla de proyecto'), items: template },
    { title: t('Bots'), items: bots },
    { title: 'Telegram', items: telegram },
  ].filter((s) => s.items.length)
}
