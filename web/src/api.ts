import { getLang, setLang, t, translateError, type Lang } from './i18n'

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

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, credentials: 'same-origin', headers: {} }
  if (body instanceof FormData) {
    init.body = body
  } else if (body !== undefined) {
    init.body = JSON.stringify(body)
    ;(init.headers as Record<string, string>)['Content-Type'] = 'application/json'
  }
  const res = await fetch(url, init)
  const text = await res.text()
  const data = text ? JSON.parse(text) : null
  if (!res.ok) {
    let msg = `Error ${res.status}`
    if (data?.detail) {
      msg = typeof data.detail === 'string' ? data.detail : data.detail.map((d: any) => d.msg).join('; ')
    } else if (data?.errors) {
      msg = t('El harness no es válido')
    }
    throw new ApiError(res.status, translateError(msg), data)
  }
  return data as T
}

export const api = {
  get: <T>(url: string) => request<T>('GET', url),
  post: <T>(url: string, body?: unknown) => request<T>('POST', url, body ?? {}),
  put: <T>(url: string, body?: unknown) => request<T>('PUT', url, body ?? {}),
  patch: <T>(url: string, body?: unknown) => request<T>('PATCH', url, body ?? {}),
  del: <T>(url: string) => request<T>('DELETE', url),
}

export function relTime(ts: number): string {
  const s = Date.now() / 1000 - ts
  if (s < 60) return t('ahora')
  if (s < 3600) return t('hace {n} min', { n: Math.floor(s / 60) })
  if (s < 86400) return t('hace {n} h', { n: Math.floor(s / 3600) })
  return new Date(ts * 1000).toLocaleDateString(getLang() === 'en' ? 'en-US' : 'es-ES', { day: 'numeric', month: 'short' })
}

export function money(usd?: number): string {
  if (usd === undefined || usd === null) return '—'
  return usd < 0.01 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(2)}`
}

/** Subscribe to a run's event stream. Returns a close function. */
export function streamRun(
  runId: number,
  after: number,
  handlers: { onEvent: (e: RunEvent) => void; onLive?: (items: RunEvent[]) => void; onEnd?: (status: string) => void },
): () => void {
  const es = new EventSource(`/api/runs/${runId}/stream?after=${after}`)
  es.addEventListener('event', (m) => handlers.onEvent(JSON.parse((m as MessageEvent).data)))
  es.addEventListener('live', (m) => handlers.onLive?.(JSON.parse((m as MessageEvent).data)))
  es.addEventListener('end', (m) => {
    handlers.onEnd?.(JSON.parse((m as MessageEvent).data).status)
    es.close()
  })
  es.onerror = () => {
    // The browser retries automatically; close if the run is gone.
    if (es.readyState === EventSource.CLOSED) handlers.onEnd?.('error')
  }
  return () => es.close()
}

export const ACTIVE = new Set(['queued', 'running'])

export function statusLabel(status: string): string {
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
