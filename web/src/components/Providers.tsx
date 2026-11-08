import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import { t, translateError, useLang } from '../i18n'
import { normalizeBaseUrl } from '../logic'

type Assigned = { project: string; executor_model?: string; improver_model?: string; vision_model?: string }

let assignedListener: ((msg: string) => void) | null = null
function alertAssigned(msg: string) {
  assignedListener?.(msg)
}

export function assignedText(a?: Assigned[]) {
  if (!a || a.length === 0) return ''
  const parts = a.map((x) => `${x.project}: ` + [
    x.executor_model && `${t('Chat')} → ${x.executor_model}`,
    x.vision_model && `${t('Visión')} → ${x.vision_model}`,
    x.improver_model && `${t('Mejora')} → ${x.improver_model}`,
  ].filter(Boolean).join(', '))
  return ` · ${t('Asignado automáticamente')}: ${parts.join(' | ')}`
}

export type ProviderRow = {
  name: string
  label: string
  kind: 'openai' | 'anthropic'
  base_url: string
  models: string[]
  key_hint: string | null
  has_key: boolean
  source: 'user' | 'server'
  available: boolean
}
type Template = { name: string; label: string; kind: 'openai' | 'anthropic'; base_url: string }
type ProvidersResp = { providers: ProviderRow[]; hidden: { name: string; label: string }[]; can_adopt: boolean; templates: Template[]; defaults: { executor: string; improver: string; vision: string } }

let cache: ProvidersResp | null = null
const listeners = new Set<(d: ProvidersResp) => void>()

export async function reloadProviders() {
  cache = await api.get<ProvidersResp>('/api/providers')
  listeners.forEach((l) => l(cache!))
  return cache
}

export function useProviders() {
  const [data, setData] = useState<ProvidersResp | null>(cache)
  useEffect(() => {
    listeners.add(setData)
    if (!cache) reloadProviders()
    return () => void listeners.delete(setData)
  }, [])
  return data
}

/** Pick a provider, then one of its models, for one role (chat, mejora, visión…). */
export function RolePicker({
  label,
  hint,
  value,
  onChange,
  optional,
}: {
  label: string
  hint?: string
  value: string
  onChange: (v: string) => void
  optional?: boolean
}) {
  const data = useProviders()
  useLang()
  const [prov, ...rest] = value ? value.split('/') : ['', '']
  const model = rest.join('/')
  const providers = data?.providers ?? []
  const p = providers.find((x) => x.name === prov)
  const [custom, setCustom] = useState(false)
  const inList = !!p && p.models.includes(model)

  return (
    <div className="role">
      <div className="role-label">
        <b>{label}</b>
        {hint && <span className="muted small"> · {hint}</span>}
      </div>
      <div className="role-row">
        <select
          value={prov}
          onChange={(e) => {
            const np = providers.find((x) => x.name === e.target.value)
            onChange(e.target.value ? `${e.target.value}/${np?.models[0] ?? ''}` : '')
            setCustom(false)
          }}
        >
          {optional && <option value="">{t('— ninguno —')}</option>}
          {!optional && !prov && <option value="">{t('Elige proveedor…')}</option>}
          {prov && !p && <option value={prov}>{t('{name} (no configurado)', { name: prov })}</option>}
          {providers.map((x) => (
            <option key={x.name} value={x.name}>
              {x.label}
            </option>
          ))}
        </select>
        {prov &&
          (custom || !p || !p.models.length || (!inList && model) ? (
            <input value={model} placeholder={t('nombre del modelo')} spellCheck={false} onChange={(e) => onChange(`${prov}/${e.target.value}`)} />
          ) : (
            <select
              value={model}
              onChange={(e) => {
                if (e.target.value === '__custom__') setCustom(true)
                else onChange(`${prov}/${e.target.value}`)
              }}
            >
              {p.models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
              <option value="__custom__">{t('Otro…')}</option>
            </select>
          ))}
      </div>
      {prov && model && <ModelLimits spec={`${prov}/${model}`} />}
      {prov && !p && <div className="warn small">{t('El proveedor «{name}» no existe: añádelo abajo en Proveedores.', { name: prov })}</div>}
      {p && !p.has_key && p.source === 'user' && <div className="warn small">{t('Este proveedor no tiene API key guardada.')}</div>}
    </div>
  )
}

type Limits = { context: number; output: number | null; source: string }
const limitsCache = new Map<string, Limits>()
// every ModelLimits showing a spec (Chat and Mejora may use the same model) hears when its limits change
const limitsListeners = new Set<(spec: string, l: Limits) => void>()

function ModelLimits({ spec }: { spec: string }) {
  const [l, setL] = useState(limitsCache.get(spec) ?? null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  useEffect(() => {
    const hear = (s: string, v: Limits) => s === spec && setL(v)
    limitsListeners.add(hear)
    return () => void limitsListeners.delete(hear)
  }, [spec])
  useEffect(() => {
    let alive = true
    setErr('')
    if (limitsCache.has(spec)) return setL(limitsCache.get(spec)!)
    const id = setTimeout(async () => {
      try {
        const r = await api.get<Limits>(`/api/models/limits?spec=${encodeURIComponent(spec)}`)
        limitsCache.set(spec, r)
        if (alive) setL(r)
      } catch {
        /* ignore */
      }
    }, 400)
    return () => {
      alive = false
      clearTimeout(id)
    }
  }, [spec])
  if (!l) return null
  // A limit the model itself reported (its server rejected a bigger request) is kept until the user forgets it, e.g.
  // after the server got a bigger window: then the provider's / known value applies again.
  const forget = async () => {
    setBusy(true)
    setErr('')
    try {
      const url = `/api/models/limits?spec=${encodeURIComponent(spec)}`
      let r = await api.del<Limits>(url)
      if (typeof r?.context !== 'number') r = await api.get<Limits>(url)
      limitsCache.set(spec, r)
      limitsListeners.forEach((fn) => fn(spec, r))
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const k = (n: number) => (n >= 1_000_000 ? `${n / 1_000_000}M` : `${Math.round(n / 1000)}k`)
  const learned = l.source === 'learned'
  const src = l.source === 'provider' ? t('según el proveedor') : learned ? t('aprendido') : l.source === 'table' ? t('valor conocido') : t('estimado')
  return (
    <div className="muted small limits-line">
      <span>
        {t('Contexto')}: <b>{k(l.context)}</b> {t('tokens')}{l.output ? ` · ${t('salida')} ${k(l.output)}` : ''}{' '}
        <span title={learned ? t('El modelo avisó de que su contexto real es menor y Nesicode lo recuerda.') : undefined}>({src})</span>
      </span>
      {learned && (
        <button type="button" className="link small" disabled={busy} onClick={forget}
          title={t('Vuelve al valor del proveedor o de la tabla de modelos conocidos (p. ej. si su servidor ya admite más contexto).')}>
          {busy ? '…' : t('Olvidar')}
        </button>
      )}
      {err && <span className="error-text">{err}</span>}
    </div>
  )
}

function ProviderForm({ initial, templates, onDone }: { initial?: ProviderRow; templates: Template[]; onDone: () => void }) {
  const editing = !!initial
  const [f, setF] = useState({
    name: initial?.name ?? '',
    label: initial?.label ?? '',
    kind: initial?.kind ?? ('openai' as 'openai' | 'anthropic'),
    base_url: initial?.base_url ?? '',
    api_key: '',
    models: (initial?.models ?? []).join('\n'),
  })
  const [msg, setMsg] = useState<{ ok?: string; err?: string }>({})
  const [busy, setBusy] = useState(false)
  const set = (k: keyof typeof f) => (v: string) => setF((x) => ({ ...x, [k]: v }))

  const useTemplate = (t: Template) =>
    setF((x) => ({ ...x, name: t.name === 'custom' ? '' : t.name, label: t.name === 'custom' ? '' : t.label, kind: t.kind, base_url: t.base_url }))

  // "localhost:11434/v1" (no scheme) is a common way to type a local endpoint: complete it visibly
  const baseUrl = () => {
    const u = normalizeBaseUrl(f.base_url)
    if (u !== f.base_url) set('base_url')(u)
    return u
  }

  const test = async (discover: boolean) => {
    setBusy(true)
    setMsg({})
    try {
      const r = await api.post<{ ok: boolean; models?: string[]; error?: string }>('/api/providers/test', {
        kind: f.kind, base_url: baseUrl(), api_key: f.api_key || undefined, name: editing ? f.name : undefined,
      })
      if (!r.ok) return setMsg({ err: r.error ? translateError(r.error) : t('No se pudo conectar con el proveedor.') })
      const models = r.models ?? []
      setMsg({ ok: t('Conexión correcta · {n} modelos disponibles', { n: models.length }) })
      if (discover) set('models')(models.join('\n'))
    } catch (e) {
      // a 422 for a bad URL, a 500, the server restarting…: say so and keep the form usable
      setMsg({ err: (e as Error).message })
    } finally {
      setBusy(false)
    }
  }

  const save = async () => {
    setBusy(true)
    setMsg({})
    const models = f.models.split(/[\n,]/).map((m) => m.trim()).filter(Boolean)
    try {
      const base_url = baseUrl()
      if (editing) {
        await api.patch(`/api/providers/${initial!.name}`, {
          label: f.label, kind: f.kind, base_url, models, ...(f.api_key ? { api_key: f.api_key } : {}),
        })
      } else {
        const r = await api.post<{ assigned?: Assigned[] }>('/api/providers', { name: f.name, label: f.label, kind: f.kind, base_url, api_key: f.api_key || undefined, models })
        if (r.assigned?.length) {
          alertAssigned(assignedText(r.assigned))
        }
      }
      await reloadProviders()
      onDone()
    } catch (e) {
      setMsg({ err: (e as Error).message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="prov-form">
      {!editing && (
        <div className="templates">
          {templates.map((t) => (
            <button key={t.name} type="button" className="idea" onClick={() => useTemplate(t)}>
              {t.label}
            </button>
          ))}
        </div>
      )}
      <div className="grid2">
        <label>
          {t('Nombre')}
          <input value={f.name} disabled={editing} onChange={(e) => set('name')(e.target.value.toLowerCase())} placeholder="nesilabs" />
          <span className="muted small">{t('Identificador corto; los modelos se eligen como {name}/modelo.', { name: f.name || t('nombre') })}</span>
        </label>
        <label>
          {t('Etiqueta')}
          <input value={f.label} onChange={(e) => set('label')(e.target.value)} placeholder="Nesi Labs" />
        </label>
        <label>
          {t('Tipo de API')}
          <select value={f.kind} onChange={(e) => set('kind')(e.target.value)}>
            <option value="openai">{t('Compatible con OpenAI')}</option>
            <option value="anthropic">Anthropic</option>
          </select>
        </label>
        <label>
          {t('Endpoint (base URL)')}
          <input value={f.base_url} onChange={(e) => set('base_url')(e.target.value)} onBlur={() => baseUrl()} placeholder={t('https://api.ejemplo.com/v1')} spellCheck={false} />
        </label>
      </div>
      <label>
        API key {editing && initial?.key_hint && <span className="muted small">· {t('guardada {hint} (déjalo vacío para mantenerla)', { hint: initial.key_hint })}</span>}
        <input type="password" autoComplete="off" value={f.api_key} onChange={(e) => set('api_key')(e.target.value)} placeholder={editing ? t('Nueva key (opcional)') : 'sk-…'} />
        <span className="muted small">{t('Se guarda cifrada en la bóveda; nunca se vuelve a mostrar.')}</span>
      </label>
      <label>
        {t('Modelos (uno por línea)')}
        <textarea rows={4} value={f.models} onChange={(e) => set('models')(e.target.value)} spellCheck={false} />
      </label>
      {msg.ok && <div className="ok-box">{msg.ok}</div>}
      {msg.err && <div className="error-box">{msg.err}</div>}
      <div className="row wrap">
        <button type="button" className="btn" disabled={busy || !f.base_url} onClick={() => test(false)}>{t('Probar conexión')}</button>
        <button type="button" className="btn" disabled={busy || !f.base_url} onClick={() => test(true)}>{t('Descubrir modelos')}</button>
        <span style={{ flex: 1 }} />
        <button type="button" className="btn ghost" onClick={onDone}>{t('Cancelar')}</button>
        <button type="button" className="btn primary" disabled={busy || !f.name || !f.base_url} onClick={save}>{editing ? t('Guardar') : t('Añadir proveedor')}</button>
      </div>
    </div>
  )
}

export default function ProvidersSection({ onAssigned }: { onAssigned?: () => void } = {}) {
  const data = useProviders()
  useLang()
  const [editing, setEditing] = useState<string | null>(null)
  const [msg, setMsg] = useState('')
  useEffect(() => {
    assignedListener = (m) => {
      setMsg(m.replace(/^ · /, ''))
      onAssigned?.()
    }
    return () => {
      assignedListener = null
    }
  }, [onAssigned])

  const discover = useCallback(async (name: string) => {
    setMsg('')
    try {
      const r = await api.post<{ models: string[]; assigned?: Assigned[] }>(`/api/providers/${name}/discover`)
      await reloadProviders()
      setMsg(`${name}: ${t('{n} modelos', { n: r.models.length })}${assignedText(r.assigned)}`)
      if (r.assigned?.length) onAssigned?.()
    } catch (e) {
      setMsg((e as Error).message)
    }
  }, [])

  // hide / show / delete: on failure say why instead of failing silently
  const act = async (call: () => Promise<unknown>) => {
    setMsg('')
    try {
      await call()
    } catch (e) {
      setMsg((e as Error).message)
    }
    reloadProviders().catch(() => {})
  }

  if (!data) return <div className="muted">{t('Cargando…')}</div>
  return (
    <div className="providers">
      {msg && <div className="ok-box">{msg}</div>}
      {data.providers.map((p) =>
        editing === p.name ? (
          <ProviderForm key={p.name} initial={p} templates={data.templates} onDone={() => setEditing(null)} />
        ) : (
          <div key={p.name} className="prov">
            <div className="prov-main">
              <div>
                <b>{p.label}</b> <code className="muted small">{p.name}/…</code>
                {p.source === 'server' && <span className="pill">{t('servidor')}</span>}
              </div>
              <div className="small muted clamp1">
                {p.kind === 'anthropic' ? 'Anthropic' : 'OpenAI-compatible'} · {p.base_url || '—'} · key: {p.key_hint ?? t('sin key')} · {t('{n} modelos', { n: p.models.length })}
              </div>
            </div>
            <div className="row">
              {p.name !== 'mock' && <button className="btn small" onClick={() => discover(p.name)}>{t('Actualizar modelos')}</button>}
              {p.source === 'server' && (
                <>
                  {data.can_adopt && p.name !== 'mock' && (
                    <button
                      className="btn small"
                      title={t('Lo convierte en un proveedor tuyo (copia la key del servidor a tu bóveda) para poder editarlo o borrarlo')}
                      onClick={async () => {
                        try {
                          await api.post(`/api/providers/${p.name}/adopt`)
                          await reloadProviders()
                          setEditing(p.name)
                        } catch (e) {
                          setMsg((e as Error).message)
                        }
                      }}
                    >
                      {t('Editar')}
                    </button>
                  )}
                  <button className="btn small ghost" onClick={() => act(() => api.post(`/api/providers/${p.name}/hide`))}>
                    {t('Ocultar')}
                  </button>
                </>
              )}
              {p.source === 'user' && (
                <>
                  <button className="btn small" onClick={() => setEditing(p.name)}>{t('Editar')}</button>
                  <button
                    className="btn small ghost"
                    onClick={() => {
                      if (!confirm(t('¿Eliminar el proveedor {name} y su API key de la bóveda?', { name: p.name }))) return
                      act(() => api.del(`/api/providers/${p.name}`))
                    }}
                  >
                    {t('Eliminar')}
                  </button>
                </>
              )}
            </div>
          </div>
        ),
      )}
      {data.hidden.length > 0 && (
        <div className="small muted">
          {t('Ocultos:')}{' '}
          {data.hidden.map((h) => (
            <button key={h.name} className="link small" onClick={() => act(() => api.post(`/api/providers/${h.name}/unhide`))}>
              {h.label} ({t('mostrar')})
            </button>
          ))}
        </div>
      )}
      {editing === '__new__' ? (
        <ProviderForm templates={data.templates} onDone={() => setEditing(null)} />
      ) : (
        <button className="btn" onClick={() => setEditing('__new__')}>{t('+ Añadir proveedor')}</button>
      )}
    </div>
  )
}
