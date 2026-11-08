import { api } from './api'

/** A model's context window and output cap, and where they come from (learned | provider | table | default). */
export type Limits = { context: number; output: number | null; source: string }

// One copy per page for every picker showing a model (Chat and Mejora often use the same one).
const cache = new Map<string, Limits>()
const listeners = new Set<(spec: string, l: Limits) => void>()
const url = (spec: string) => `/api/models/limits?spec=${encodeURIComponent(spec)}`

function publish(spec: string, l: Limits): Limits {
  cache.set(spec, l)
  listeners.forEach((fn) => fn(spec, l))
  return l
}

/** What a picker can show at once for `spec` (null until the server answered once). */
export const cachedLimits = (spec: string): Limits | null => cache.get(spec) ?? null

/**
 * Show `spec`'s limits: the cached ones at once, then the server's — asked again every time a picker shows the model,
 * since a chat may have learned its real window meanwhile (context_learned: «aprendido», with «Olvidar») — and every
 * later change (another picker's «Olvidar»). `delay` debounces a model name being typed. Returns the unsubscribe.
 */
export function watchLimits(spec: string, show: (l: Limits) => void, delay = 400): () => void {
  const hear = (s: string, l: Limits) => {
    if (s === spec) show(l)
  }
  listeners.add(hear)
  const cached = cache.get(spec)
  if (cached) show(cached)
  const id = setTimeout(() => {
    api.get<Limits>(url(spec)).then((l) => publish(spec, l), () => {})
  }, cached ? 0 : delay)
  return () => {
    clearTimeout(id)
    listeners.delete(hear)
  }
}

/**
 * Forget the limit learned for `spec` (DELETE answers the fresh {context, output, source}; asked again if it doesn't):
 * every picker showing that model updates. Rejects with the server's message.
 */
export async function forgetLearned(spec: string): Promise<Limits> {
  const r = await api.del<Partial<Limits> | null>(url(spec))
  return publish(spec, typeof r?.context === 'number' ? (r as Limits) : await api.get<Limits>(url(spec)))
}
