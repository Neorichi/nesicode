import { getLang, setLang, t, type Lang } from './i18n'
import { errorMessage, parseBody } from './logic'

export type User = { id: number; email: string; name: string; auth?: boolean }

export type Project = {
  id: number
  name: string
  domain: string
  executor_model: string
  improver_model: string
  check_command: string
  max_steps: number
  harness_version: number
  created_at: number
  workspace: string
  vision_model: string
  image_model: string
  transcribe_model?: string
  profile?: string
  /** folders of the user's the agent may READ in this project (canonical paths; [] = none) */
  read_dirs?: string[]
  /** what keeps the agent to them: a sandbox ('macos' | 'docker'), nothing ('none') or not available ('off': a shared server) */
  read_dirs_mode?: 'macos' | 'docker' | 'none' | 'off'
}

export type Thread = { id: number; project_id: number; title: string; isolated: number; exec_mode?: string; effective_mode?: string; telegram?: { linked: boolean; attached: boolean }; bot?: any; created_at: number; updated_at: number }

export type Attachment = { name: string; path: string; kind: 'image' | 'text' | 'binary'; size: number }

export type RunEvent = {
  seq?: number
  type: string
  ts?: number
  [key: string]: any
}

export type Run = {
  id: number
  project_id: number
  thread_id: number | null
  kind: 'chat' | 'harness'
  task: string
  status: string
  final_text: string
  error: string
  harness_version: number | null
  created_at: number
  finished_at: number | null
  params: Record<string, any>
  attachments?: Attachment[]
  bot_id?: number | null
  trigger?: string
  summary?: Record<string, any>
  result?: Record<string, any>
  events?: RunEvent[]
}

export type HarnessFiles = Record<string, string>

export type Provider = {
  name: string
  label: string
  needs_key: boolean
  key_env: string
  available: boolean
  user_key: boolean
  examples: string[]
}

export type Lesson = {
  id: number
  domain: string
  cue: string
  lesson: string
  evidence: string
  confidence: string
  run_id: number | null
  project_id: number | null
  created_at: number
}

export class ApiError extends Error {
  status: number
  detail: any
  constructor(status: number, message: string, detail?: any) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function request<T>(method: string, url: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const init: RequestInit = { method, credentials: 'same-origin', headers: {}, signal }
  if (body instanceof FormData) {
    init.body = body
  } else if (body !== undefined) {
    init.body = JSON.stringify(body)
    ;(init.headers as Record<string, string>)['Content-Type'] = 'application/json'
  }
  let res: Response
  try {
    res = await fetch(url, init)
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e
    throw new ApiError(0, t('No se pudo conectar con Nesicode. Comprueba que la app sigue abierta.'))
  }
  const text = await res.text()
  // an error page from a proxy, or Starlette's plain-text 500, is not JSON: never show a JSON SyntaxError
  const data = parseBody(text)
  if (!res.ok) throw new ApiError(res.status, errorMessage(res.status, data ?? null), data)
  if (data === undefined) throw new ApiError(res.status, t('El servidor devolvió una respuesta que no se entiende.'))
  return data as T
}

export const api = {
  get: <T>(url: string, opts?: { signal?: AbortSignal }) => request<T>('GET', url, undefined, opts?.signal),
  post: <T>(url: string, body?: unknown) => request<T>('POST', url, body ?? {}),
  put: <T>(url: string, body?: unknown) => request<T>('PUT', url, body ?? {}),
  patch: <T>(url: string, body?: unknown) => request<T>('PATCH', url, body ?? {}),
  del: <T>(url: string) => request<T>('DELETE', url),
}

/** True when a request was cancelled on purpose (component gone, newer request issued). */
export const isAbort = (e: unknown) => (e as Error)?.name === 'AbortError'

export function relTime(ts: number): string {
  const s = Date.now() / 1000 - ts
  if (s < 60) return t('ahora')
  if (s < 3600) return t('hace {n} min', { n: Math.floor(s / 60) })
  if (s < 86400) return t('hace {n} h', { n: Math.floor(s / 3600) })
  return new Date(ts * 1000).toLocaleDateString(getLang() === 'en' ? 'en-US' : 'es-ES', { day: 'numeric', month: 'short' })
}

export function money(usd?: number): string {
  if (typeof usd !== 'number' || !Number.isFinite(usd)) return '—' // (also an odd value from an event: never a crash)
  return usd < 0.01 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(2)}`
}

/** " · $0.05" after a run's steps — nothing when there is no cost to show (free or local models, odd values). */
export function costSuffix(usd?: number): string {
  return typeof usd === 'number' && Number.isFinite(usd) && usd > 0 ? ` · ${money(usd)}` : ''
}

const HIDDEN_GRACE_MS = 5000 // a tab hidden this long gives its stream connection back (browsers allow ~6 per host)
const MAX_STREAM_FAILURES = 6
/** onEnd status when the stream gave up reconnecting (server unreachable), as opposed to the run's final status. */
export const STREAM_LOST = 'stream-lost'

/**
 * Subscribe to a run's event stream. Returns a close function.
 *
 * Every (re)connection asks only for the events after the last one received, and the server then replays the
 * current reply's live items from the start: `onOpen` fires first so the caller can drop its partial live text
 * (otherwise it would be shown twice). A tab hidden for a few seconds closes its stream and resumes when visible,
 * so many open tabs can't starve the browser's connections (Stop, sending, loading other pages).
 */
export function streamRun(
  runId: number,
  after: number,
  handlers: { onEvent: (e: RunEvent) => void; onLive?: (items: RunEvent[]) => void; onEnd?: (status: string) => void; onOpen?: () => void },
): () => void {
  let last = after
  let es: EventSource | null = null
  let done = false
  let failures = 0
  let retry: number | undefined
  let hide: number | undefined
  const hidden = () => typeof document !== 'undefined' && document.visibilityState === 'hidden'
  const drop = () => {
    es?.close()
    es = null
  }
  const stop = () => {
    done = true
    drop()
    clearTimeout(retry)
    clearTimeout(hide)
    document.removeEventListener('visibilitychange', onVisibility)
  }
  const open = () => {
    if (done || es || hidden()) return
    const src = new EventSource(`/api/runs/${runId}/stream?after=${last}`)
    es = src
    src.onopen = () => {
      failures = 0
      handlers.onOpen?.()
    }
    src.addEventListener('event', (m) => {
      const e = JSON.parse((m as MessageEvent).data) as RunEvent
      if (typeof e.seq === 'number' && e.seq > last) last = e.seq
      handlers.onEvent(e)
    })
    src.addEventListener('live', (m) => handlers.onLive?.(JSON.parse((m as MessageEvent).data)))
    src.addEventListener('end', (m) => {
      stop()
      handlers.onEnd?.(JSON.parse((m as MessageEvent).data).status)
    })
    src.onerror = () => {
      if (es !== src || done) return
      // Reconnect ourselves (the browser would reuse the first URL and replay everything since `after`).
      drop()
      if (++failures > MAX_STREAM_FAILURES) {
        stop()
        handlers.onEnd?.(STREAM_LOST) // not the run's status: the run may well go on
        return
      }
      clearTimeout(retry)
      retry = window.setTimeout(open, Math.min(15000, 500 * 2 ** failures))
    }
  }
  const onVisibility = () => {
    clearTimeout(hide)
    if (hidden()) hide = window.setTimeout(drop, HIDDEN_GRACE_MS)
    else if (!es && !done) {
      clearTimeout(retry)
      open()
    }
  }
  document.addEventListener('visibilitychange', onVisibility)
  open()
  return stop
}

export const ACTIVE = new Set(['queued', 'running'])

/** A run's status in words. `stopReason` (the run's summary): why a 'max_steps' end stopped — the step cap, or else
 *  (a project with no step cap never ends on one) a full context, a loop with no progress, empty replies. */
export function statusLabel(status: string, stopReason?: string): string {
  const stopped: Record<string, string> = {
    context_full: t('Contexto lleno'),
    stuck: t('Detenido: se repetía sin avanzar'),
    empty_replies: t('Detenido: el modelo no respondía nada'),
  }
  if (status === 'max_steps' && stopReason && stopped[stopReason]) return stopped[stopReason]
  const labels: Record<string, string> = {
    queued: t('En cola'),
    waiting: t('Esperando turno'),
    running: t('En curso'),
    done: t('Completado'),
    max_steps: t('Límite de pasos'),
    error: t('Error'),
    cancelled: t('Cancelado'),
    interrupted: t('Interrumpido'),
    refusal: t('Rechazado'),
  }
  return labels[status] ?? status
}

/** Switch the UI language and remember it on the server too (best effort). */
export function switchLang(l: Lang) {
  setLang(l)
  api.put('/api/me/settings', { language: l }).catch(() => {})
}

/** Microphone: send a WAV to the project's transcriber, get the text back. */
export async function transcribeAudio(projectId: number | string, wav: Blob): Promise<string> {
  const fd = new FormData()
  fd.append('audio', new File([wav], 'dictado.wav', { type: 'audio/wav' }))
  fd.append('lang', getLang())
  const r = await api.post<{ text: string }>(`/api/projects/${projectId}/transcribe`, fd)
  return r.text
}

/** The computer this page runs on (labels like Finder / Explorer). */
export const OS: 'mac' | 'win' | 'linux' = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)
  ? 'mac' : /Win/.test(navigator.platform || navigator.userAgent) ? 'win' : 'linux'

export function fileManager(): string {
  return OS === 'mac' ? 'Finder' : OS === 'win' ? t('el Explorador de archivos') : t('el gestor de archivos')
}

export type Profile = { id: string; label: string; icon: string; description: string; domain: string }
