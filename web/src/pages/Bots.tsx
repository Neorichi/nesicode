import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, ApiError, OS } from '../api'
import { useApp } from '../App'
import { useVisibleInterval } from '../hooks'
import { t, useLang } from '../i18n'
import { EDITABLE_TRIGGERS, intervalSpec, parseInterval, type IntervalUnit } from '../logic'

// ------------------------------------------------------------------ types
// 'once' = a follow-up the bot scheduled itself (bot_schedule_once); 'webhook' is managed on the bot's page.
export type Trigger = { id?: number; kind: 'cron' | 'interval' | 'url' | 'once' | 'webhook'; spec: string; tz?: string; prompt?: string; config?: { every?: number }; next_at?: number | null }
type RunRow = { id: number; status: string; trigger: string; created_at: number; finished_at?: number; summary?: string; usage?: { cost_usd?: number } }
export type Bot = {
  id: number; project_id: number; project: string; name: string; description: string; instructions: string; model: string
  policy: 'ask' | 'readonly' | 'auto'; max_steps: number; daily_cap: number; notify: 'telegram' | 'none'; thread_id: number | null
  enabled: boolean; paused_reason: string; fail_streak: number; next_at: number | null; last_run_at: number | null
  completed_at?: number | null // it switched itself off: its goal was reached (paused_reason says what was achieved)
  last_run?: { id: number; status: string; trigger: string } | null; webhook: boolean; triggers: Trigger[]
  memory?: { key: string; value: string; updated_at: number }[]; runs?: RunRow[]; windows?: boolean
}
type Draft = Partial<Bot> & { triggers: Trigger[] }

const DAYS = [['1', 'L'], ['2', 'M'], ['3', 'X'], ['4', 'J'], ['5', 'V'], ['6', 'S'], ['0', 'D']] as const

const TEMPLATES: { title: string; text: string }[] = [
  { title: 'Resumen diario de noticias', text: 'Cada día laborable a las 8:00, mira las portadas de Hacker News y de un par de medios tecnológicos y mándame un resumen de 5 puntos con enlaces. No repitas lo que ya me contaste.' },
  { title: 'Vigilar una web', text: 'Vigila https://example.com/precios cada 30 minutos y avísame solo cuando cambie algo importante (precio, disponibilidad), diciéndome qué ha cambiado.' },
  { title: 'Informe semanal en Word', text: 'Cada lunes a las 9:00, revisa los cambios de la semana en el proyecto (git log) y crea un informe en Word con un resumen y la lista de cambios.' },
  { title: 'Revisar errores de un servicio', text: 'Cada hora comprueba que https://mi-servicio.com/health responde bien. Si falla, avísame con el detalle; si va bien, no digas nada.' },
  { title: 'Asistente por Telegram', text: 'Un asistente al que le escribo por Telegram (/bot asistente …) para buscar cosas en la web y resumírmelas en pocas líneas.' },
]

function when(ts?: number | null): string {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  const now = new Date()
  const sameDay = d.toDateString() === now.toDateString()
  const tomorrow = new Date(now.getTime() + 86400000).toDateString() === d.toDateString()
  const hm = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  if (sameDay) return `${t('hoy')} ${hm}`
  if (tomorrow) return `${t('mañana')} ${hm}`
  return d.toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short' }) + ` ${hm}`
}

function describeTrigger(tr: Trigger): string {
  if (tr.kind === 'once') return `${t('Seguimiento')} · ${when(tr.next_at)}`
  if (tr.kind === 'webhook') return 'webhook'
  if (tr.kind === 'interval') {
    const m = parseInterval(tr.spec ?? '')
    if (m) return t('Cada {n} {u}', { n: m.n, u: m.u === 'h' ? t('horas') : m.u === 'd' ? t('días') : t('minutos') })
  }
  if (tr.kind === 'url') return t('Vigila {url} cada {n} min', { url: tr.spec.replace(/^https?:\/\//, '').slice(0, 40), n: tr.config?.every ?? 30 })
  const d = parseDaily(tr.spec)
  if (d) {
    const days = d.days.length === 7 ? t('todos los días') : d.days.join(',') === '1,2,3,4,5' ? t('de lunes a viernes')
      : DAYS.filter(([v]) => d.days.includes(v)).map(([, l]) => l).join(' ')
    return `${days} · ${d.time}`
  }
  const every = /^\*\/(\d+) \* \* \* \*$/.exec(tr.spec.trim())
  if (every) return t('Cada {n} {u}', { n: every[1], u: t('minutos') })
  const hourly = /^(\d{1,2}) \* \* \* \*$/.exec(tr.spec.trim())
  if (hourly) return t('Cada hora (a los {m} min)', { m: hourly[1] })
  return `cron ${tr.spec}`
}

function parseDaily(spec: string): { time: string; days: string[] } | null {
  const m = /^(\d{1,2}) (\d{1,2}) \* \* (\*|[0-7](?:[-,][0-7])*)$/.exec(spec.trim())
  if (!m) return null
  let days: string[] = []
  if (m[3] === '*') days = ['0', '1', '2', '3', '4', '5', '6']
  else for (const part of m[3].split(',')) {
    const [a, b] = part.split('-').map(Number)
    for (let x = a; x <= (b ?? a); x++) days.push(String(x % 7))
  }
  return { time: `${m[2].padStart(2, '0')}:${m[1].padStart(2, '0')}`, days: [...new Set(days)] }
}

function dailySpec(time: string, days: string[]): string {
  const [h, m] = time.split(':').map((x) => String(Number(x)))
  const all = days.length === 7 || days.length === 0
  return `${m} ${h} * * ${all ? '*' : [...days].sort().join(',')}`
}

// ------------------------------------------------------------------ list page
export default function BotsPage() {
  useLang()
  const { projects } = useApp()
  const navigate = useNavigate()
  const [bots, setBots] = useState<Bot[] | null>(null)
  const [telegram, setTelegram] = useState(false)
  const [tz, setTz] = useState('')
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')
  const load = () => api.get<{ bots: Bot[]; telegram: boolean; tz: string }>('/api/bots').then((r) => {
    setBots(r.bots); setTelegram(r.telegram); setTz(r.tz)
  }).catch((e) => setError(e.message))
  useEffect(() => { load() }, [])
  useVisibleInterval(load, 10000)

  const toggle = async (b: Bot) => { await api.patch(`/api/bots/${b.id}`, { enabled: !b.enabled }); load() }
  const runNow = async (b: Bot) => {
    try { await api.post(`/api/bots/${b.id}/run`, { text: '' }); load() } catch (e) { setError((e as Error).message) }
  }

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{t('Bots')}</h1>
          <p className="muted">{t('Tareas que Nesicode hace solo: a una hora, cada cierto tiempo, cuando cambia una web, desde Telegram o por webhook. Funcionan mientras la app está abierta; si estaba cerrada a la hora prevista, se ejecutan una vez al volver.')}</p>
        </div>
        {projects.length > 0 && <button className="btn primary" onClick={() => setCreating(true)}>+ {t('Nuevo bot')}</button>}
      </header>
      {error && <div className="error-box">{error}</div>}
      {!telegram && bots && bots.length > 0 && (
        <div className="hint-box">{t('Vincula Telegram (Integraciones → Telegram) para recibir los resultados y aprobar acciones desde el móvil. Sin él, los resultados quedan en la conversación de cada bot.')}</div>
      )}
      {OS === 'win' && <div className="hint-box">{t('En Windows los bots no ejecutan comandos de terminal (no hay jaula del sistema); el resto de herramientas sí.')}</div>}
      {projects.length === 0 && <div className="card">{t('Crea primero un proyecto: cada bot trabaja dentro de uno.')}</div>}

      {creating && <NewBot tz={tz} onCancel={() => setCreating(false)} onCreated={(b) => { setCreating(false); navigate(`/bots/${b.id}`) }} />}

      {bots && bots.length === 0 && !creating && projects.length > 0 && (
        <div className="card bots-empty">
          <h3>{t('Aún no tienes bots')}</h3>
          <p className="muted">{t('Pulsa «Nuevo bot» y descríbelo con tus palabras, o pídeselo al agente en cualquier conversación: «hazme un bot que cada mañana…».')}</p>
        </div>
      )}
      <div className="bot-list">
        {bots?.map((b) => (
          <div key={b.id} className={`card bot-card ${b.enabled ? '' : 'off'}`}>
            <div className="bot-card-head">
              <Link to={`/bots/${b.id}`} className="bot-name">🤖 {b.name}</Link>
              <label className="switch" title={b.enabled ? t('Pausar') : t('Activar')}>
                <input type="checkbox" checked={b.enabled} onChange={() => toggle(b)} /><span />
              </label>
            </div>
            {b.description && <div className="muted small">{b.description}</div>}
            <div className="bot-meta small">
              <span>📁 {b.project}</span>
              {b.triggers.map((tr, i) => <span key={i}>⏰ {describeTrigger(tr)}</span>)}
              {b.webhook && <span>🔗 webhook</span>}
              <span>{b.notify === 'telegram' ? '✈ Telegram' : t('sin avisos')}</span>
            </div>
            <div className="bot-meta small">
              {b.enabled ? <span>{t('Próxima')}: <b>{when(b.next_at)}</b></span> : <PausedNote b={b} />}
              {b.last_run && <span>{t('Última')}: <StatusDot s={b.last_run.status} /> {when(b.last_run_at)}</span>}
            </div>
            <div className="row wrap">
              <button className="btn small" onClick={() => runNow(b)}>▶ {t('Ejecutar ahora')}</button>
              {b.thread_id && <Link className="btn small" to={`/p/${b.project_id}/c/${b.thread_id}`}>💬 {t('Hablar')}</Link>}
              <Link className="btn small ghost" to={`/bots/${b.id}`}>{t('Tareas y ajustes')}</Link>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/** Why a bot is off: its task is done (it switched itself off), or it is paused (by the user, or after failures). */
export function PausedNote({ b }: { b: Pick<Bot, 'paused_reason' | 'completed_at'> }) {
  if (b.completed_at) return <span className="done">✅ {t('Tarea cumplida')}{b.paused_reason ? `: ${b.paused_reason}` : ''}</span>
  return <span className="warn">⏸ {b.paused_reason || t('En pausa')}</span>
}

function StatusDot({ s }: { s: string }) {
  const color = s === 'done' ? 'ok' : ['running', 'queued', 'waiting'].includes(s) ? 'run' : s === 'cancelled' ? 'muted' : 'err'
  return <span className={`status-dot ${color}`} title={s} />
}

// ------------------------------------------------------------------ create
function NewBot({ tz, onCancel, onCreated }: { tz: string; onCancel: () => void; onCreated: (b: Bot) => void }) {
  const { projects } = useApp()
  const [text, setText] = useState('')
  const [pid, setPid] = useState(projects[0]?.id ?? 0)
  const [busy, setBusy] = useState(false)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [error, setError] = useState('')
  const generate = async () => {
    setBusy(true); setError('')
    try { setDraft(await api.post<Draft>('/api/bots/draft', { project_id: pid, description: text })) } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  if (draft) return <BotEditor initial={{ ...draft, project_id: draft.project_id ?? pid }} tz={tz} onCancel={onCancel} onSaved={onCreated} />
  return (
    <div className="card bot-new">
      <h3>{t('¿Qué quieres que haga tu bot?')}</h3>
      <textarea rows={4} value={text} onChange={(e) => setText(e.target.value)} autoFocus
        placeholder={t('Ej.: cada día laborable a las 8:00 resúmeme las novedades de mi repositorio y mándamelo por Telegram')} />
      <div className="bot-templates">
        {TEMPLATES.map((tp) => <button key={tp.title} className="chip" onClick={() => setText(t(tp.text))}>{t(tp.title)}</button>)}
      </div>
      <div className="row wrap">
        <label className="inline">{t('Proyecto')}
          <select value={pid} onChange={(e) => setPid(Number(e.target.value))}>{projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
        </label>
        <button className="btn primary" disabled={busy || text.trim().length < 5} onClick={generate}>{busy ? t('Preparando…') : `✨ ${t('Preparar el bot')}`}</button>
        <button className="btn ghost" onClick={() => setDraft({ project_id: pid, name: '', instructions: '', triggers: [], policy: 'ask', notify: 'telegram' })}>{t('Empezar en blanco')}</button>
        <button className="btn ghost" onClick={onCancel}>{t('Cancelar')}</button>
      </div>
      {error && <div className="error-box">{error}</div>}
      <p className="muted small">{t('La IA rellena nombre, instrucciones y horario; lo revisas antes de crearlo.')}</p>
    </div>
  )
}

// ------------------------------------------------------------------ editor
export function BotEditor({ initial, tz, onCancel, onSaved }: { initial: Draft; tz: string; onCancel: () => void; onSaved: (b: Bot) => void }) {
  useLang()
  const { projects } = useApp()
  // Only schedules the user owns are edited and sent; pending follow-ups the bot scheduled itself ('once') are
  // shown read-only and kept by the server on save (sending them would be rejected).
  const [f, setF] = useState<Draft>(() => ({
    policy: 'ask', notify: 'telegram', max_steps: 30, daily_cap: 24, ...initial,
    triggers: (initial.triggers ?? []).filter((tr) => EDITABLE_TRIGGERS.includes(tr.kind)),
  }))
  const followUps = (initial.triggers ?? []).filter((tr) => tr.kind === 'once')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const set = <K extends keyof Draft>(k: K, v: Draft[K]) => setF((x) => ({ ...x, [k]: v }))
  const save = async () => {
    setBusy(true); setError('')
    const body = {
      project_id: f.project_id, name: f.name, description: f.description ?? '', instructions: f.instructions, model: f.model ?? '',
      policy: f.policy, notify: f.notify, max_steps: f.max_steps, daily_cap: f.daily_cap,
      triggers: f.triggers.filter((tr) => EDITABLE_TRIGGERS.includes(tr.kind))
        .map((tr) => ({ kind: tr.kind, spec: tr.spec, tz: tr.tz ?? '', prompt: tr.prompt ?? '', config: tr.config ?? {} })),
    }
    try {
      onSaved(f.id ? await api.patch<Bot>(`/api/bots/${f.id}`, body) : await api.post<Bot>('/api/bots', body))
    } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  return (
    <div className="card form bot-editor">
      <h3>{f.id ? t('Editar bot') : t('Nuevo bot')}</h3>
      <div className="grid2">
        <label>{t('Nombre')}<input value={f.name ?? ''} onChange={(e) => set('name', e.target.value)} maxLength={60} /></label>
        <label>{t('Proyecto')}
          <select value={f.project_id} onChange={(e) => set('project_id', Number(e.target.value))}>{projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
        </label>
      </div>
      <label>{t('Descripción corta')}<input value={f.description ?? ''} onChange={(e) => set('description', e.target.value)} maxLength={300} /></label>
      <label>{t('Instrucciones')}
        <textarea rows={7} value={f.instructions ?? ''} onChange={(e) => set('instructions', e.target.value)}
          placeholder={t('Qué tiene que hacer, cómo informarte y cuándo no decir nada (si no hay novedades responde NOTHING_NEW y no te avisa).')} />
      </label>

      <h4>{t('Cuándo se ejecuta')}</h4>
      <TriggerList triggers={f.triggers} tz={tz} onChange={(tr) => set('triggers', tr)} />
      {followUps.length > 0 && (
        <div className="muted small">
          {t('Seguimientos pendientes que programó el propio bot (se conservan al guardar):')}
          {followUps.map((tr, i) => (
            <div key={tr.id ?? i}>⏰ {describeTrigger(tr)}{tr.prompt ? ` — ${tr.prompt.slice(0, 120)}` : ''}</div>
          ))}
        </div>
      )}
      <p className="muted small">{t('Además siempre puedes lanzarlo a mano, desde Telegram (/bot nombre) o por webhook (se activa en la ficha del bot).')}</p>

      <h4>{t('Permisos cuando no estás')}</h4>
      <div className="policy-options">
        {([['ask', t('Preguntarme'), t('Lee, busca y consulta webs libremente; antes de escribir archivos, ejecutar comandos o usar conectores te pide permiso (por Telegram si está vinculado).')],
          ['readonly', t('Solo lectura'), t('Solo lee, busca y consulta webs. Nunca escribe ni ejecuta nada. Lo más seguro para resúmenes y vigilancia.')],
          ['auto', t('Autónomo'), t('Hace todo sin preguntar dentro de su jaula (salvo lo que siempre se pregunta, como escaneos o crear bots).')]] as const).map(([v, l, d]) => (
          <label key={v} className={`policy ${f.policy === v ? 'on' : ''}`}>
            <input type="radio" name="policy" checked={f.policy === v} onChange={() => set('policy', v)} />
            <div><b>{l}</b><div className="muted small">{d}</div></div>
          </label>
        ))}
      </div>

      <div className="grid3">
        <label className="check"><input type="checkbox" checked={f.notify === 'telegram'} onChange={(e) => set('notify', e.target.checked ? 'telegram' : 'none')} /> {t('Avisarme por Telegram')}</label>
        <label>{t('Máx. pasos por ejecución')}<input type="number" min={3} max={200} value={f.max_steps ?? 30} onChange={(e) => set('max_steps', Number(e.target.value))} /></label>
        <label>{t('Máx. ejecuciones al día')}<input type="number" min={1} max={500} value={f.daily_cap ?? 24} onChange={(e) => set('daily_cap', Number(e.target.value))} /></label>
      </div>
      <label>{t('Modelo (opcional)')}<input value={f.model ?? ''} onChange={(e) => set('model', e.target.value)} placeholder={t('el modelo de Chat del proyecto')} /></label>

      {error && <div className="error-box">{error}</div>}
      <div className="row">
        <button className="btn primary" disabled={busy || !f.name?.trim() || !f.instructions?.trim()} onClick={save}>{busy ? t('Guardando…') : f.id ? t('Guardar') : t('Crear bot')}</button>
        <button className="btn ghost" onClick={onCancel}>{t('Cancelar')}</button>
      </div>
    </div>
  )
}

function TriggerList({ triggers, tz, onChange }: { triggers: Trigger[]; tz: string; onChange: (t: Trigger[]) => void }) {
  const add = (tr: Trigger) => onChange([...triggers, tr])
  return (
    <div className="trigger-list">
      {triggers.map((tr, i) => (
        <TriggerRow key={i} tr={tr} tz={tz} onChange={(n) => onChange(triggers.map((x, j) => (j === i ? n : x)))} onRemove={() => onChange(triggers.filter((_, j) => j !== i))} />
      ))}
      <div className="row wrap">
        <button className="chip" onClick={() => add({ kind: 'cron', spec: '0 8 * * 1,2,3,4,5', tz })}>+ {t('A una hora')}</button>
        <button className="chip" onClick={() => add({ kind: 'interval', spec: '60' })}>+ {t('Cada cierto tiempo')}</button>
        <button className="chip" onClick={() => add({ kind: 'url', spec: 'https://', config: { every: 30 } })}>+ {t('Cuando cambie una web')}</button>
      </div>
    </div>
  )
}

function TriggerRow({ tr, tz, onChange, onRemove }: { tr: Trigger; tz: string; onChange: (t: Trigger) => void; onRemove: () => void }) {
  const daily = tr.kind === 'cron' ? parseDaily(tr.spec) : null
  const [advanced, setAdvanced] = useState(tr.kind === 'cron' && !daily)
  const [next, setNext] = useState<number[]>([])
  const [err, setErr] = useState('')
  useEffect(() => {
    if (tr.kind !== 'cron' && tr.kind !== 'interval') { setNext([]); setErr(''); return } // only schedules have a preview
    let alive = true
    const id = setTimeout(() => {
      api.post<{ next: number[] }>('/api/bots/preview', { kind: tr.kind, spec: tr.spec, tz: tr.tz ?? tz })
        .then((r) => { if (alive) { setNext(r.next); setErr('') } }).catch((e) => { if (alive) { setNext([]); setErr(e.message) } })
    }, 300)
    return () => { alive = false; clearTimeout(id) }
  }, [tr.kind, tr.spec, tr.tz, tz])
  // The interval field keeps what the user types (it may be empty for a moment): the spec changes only when the
  // number is valid, so clearing "60" to type "15" never snaps back to 60 (which gave 6015).
  const parsed = parseInterval(tr.spec ?? '')
  const [num, setNum] = useState(parsed?.n ?? '60')
  const [unit, setUnit] = useState<IntervalUnit>(parsed?.u ?? 'm')
  useEffect(() => {
    const p = parseInterval(tr.spec ?? '')
    if (p && intervalSpec(num, unit) !== intervalSpec(p.n, p.u)) {
      setNum(p.n)
      setUnit(p.u)
    }
  }, [tr.spec]) // eslint-disable-line react-hooks/exhaustive-deps
  const setInterval_ = (n: string, u: IntervalUnit) => {
    setNum(n)
    setUnit(u)
    const spec = intervalSpec(n, u)
    if (spec) onChange({ ...tr, spec })
  }

  return (
    <div className="trigger-row">
      <div className="trigger-body">
        {tr.kind === 'cron' && !advanced && daily && (
          <div className="row wrap">
            <div className="day-chips">
              {DAYS.map(([v, l]) => (
                <button key={v} className={`day ${daily.days.includes(v) ? 'on' : ''}`}
                  onClick={() => {
                    const days = daily.days.includes(v) ? daily.days.filter((d) => d !== v) : [...daily.days, v]
                    if (days.length) onChange({ ...tr, spec: dailySpec(daily.time, days) })
                  }}>{t(l)}</button>
              ))}
            </div>
            <input type="time" value={daily.time} onChange={(e) => e.target.value && onChange({ ...tr, spec: dailySpec(e.target.value, daily.days) })} />
            <button className="link small" onClick={() => setAdvanced(true)}>{t('cron avanzado')}</button>
          </div>
        )}
        {tr.kind === 'cron' && (advanced || !daily) && (
          <div className="row wrap">
            <input className="mono" value={tr.spec} onChange={(e) => onChange({ ...tr, spec: e.target.value })} placeholder="0 8 * * 1-5" />
            <input value={tr.tz ?? tz} onChange={(e) => onChange({ ...tr, tz: e.target.value })} placeholder="Europe/Madrid" style={{ maxWidth: 180 }} />
            {parseDaily(tr.spec) && <button className="link small" onClick={() => setAdvanced(false)}>{t('modo sencillo')}</button>}
          </div>
        )}
        {tr.kind === 'interval' && (
          <div className="row wrap">
            {t('Cada')}
            <input type="number" min={1} value={num} style={{ width: 90 }} onChange={(e) => setInterval_(e.target.value, unit)}
              onBlur={() => { if (!intervalSpec(num, unit) && parsed) { setNum(parsed.n); setUnit(parsed.u) } }} />
            <select value={unit} onChange={(e) => setInterval_(num, e.target.value as IntervalUnit)}>
              <option value="m">{t('minutos')}</option><option value="h">{t('horas')}</option><option value="d">{t('días')}</option>
            </select>
          </div>
        )}
        {tr.kind === 'url' && (
          <div className="row wrap">
            <input value={tr.spec} onChange={(e) => onChange({ ...tr, spec: e.target.value })} placeholder="https://…" style={{ flex: 1, minWidth: 220 }} />
            {t('revisar cada')}
            <input type="number" min={5} max={1440} value={tr.config?.every ?? 30} style={{ width: 80 }} onChange={(e) => onChange({ ...tr, config: { every: Number(e.target.value) } })} /> min
          </div>
        )}
        <input className="trigger-prompt" value={tr.prompt ?? ''} onChange={(e) => onChange({ ...tr, prompt: e.target.value })}
          placeholder={t('Qué hacer en este momento (opcional; si no, sigue sus instrucciones)')} />
        {next.length > 0 && <div className="muted small">{t('Próximas')}: {next.map((n) => when(n)).join(' · ')}</div>}
        {err && <div className="error-text small">{err}</div>}
      </div>
      <button className="queued-x" title={t('Quitar')} onClick={onRemove}>×</button>
    </div>
  )
}

// ------------------------------------------------------------------ detail
export function BotDetail() {
  useLang()
  const { bid } = useParams()
  const navigate = useNavigate()
  const [b, setB] = useState<Bot | null>(null)
  const [tz, setTz] = useState('')
  const [editing, setEditing] = useState(false)
  const [hook, setHook] = useState<{ url: string; token: string; url_with_token: string } | null>(null)
  const [error, setError] = useState('')
  const [runText, setRunText] = useState('')
  const alive = useRef(true)
  // only this page's bot may paint (App keys the page by bid; this also covers late answers after leaving)
  const load = () => api.get<Bot>(`/api/bots/${bid}`).then((r) => { if (alive.current && String(r.id) === bid) setB(r) }).catch((e) => alive.current && setError(e.message))
  useEffect(() => {
    alive.current = true
    api.get<{ tz: string }>('/api/bots').then((r) => alive.current && setTz(r.tz)).catch(() => {})
    return () => { alive.current = false }
  }, [bid])
  useVisibleInterval(load, 8000)
  useEffect(() => { load() }, [bid]) // eslint-disable-line react-hooks/exhaustive-deps
  if (!b) return <div className="page">{error ? <div className="error-box">{error}</div> : '…'}</div>
  if (editing) return <div className="page narrow"><BotEditor initial={b} tz={tz} onCancel={() => setEditing(false)} onSaved={(n) => { setB(n); setEditing(false) }} /></div>

  const run = async () => {
    try { await api.post(`/api/bots/${b.id}/run`, { text: runText }); setRunText(''); load() } catch (e) { setError((e as Error).message) }
  }
  const remove = async () => {
    if (!window.confirm(t('¿Borrar el bot «{name}»? Su conversación se conserva.', { name: b.name }))) return
    await api.del(`/api/bots/${b.id}`)
    navigate('/bots')
  }
  const enableHook = async () => setHook(await api.post(`/api/bots/${b.id}/webhook`))
  const disableHook = async () => { await api.del(`/api/bots/${b.id}/webhook`); setHook(null); load() }
  // a follow-up the bot scheduled itself (bot_schedule_once) that the user doesn't want any more
  const removeFollowUp = async (tr: Trigger) => {
    if (!window.confirm(t('¿Quitar el seguimiento pendiente ({when})? El bot no lo hará.', { when: describeTrigger(tr) }))) return
    setError('')
    try {
      await api.del(`/api/bots/${b.id}/triggers/${tr.id}`)
    } catch (e) {
      // 404: it already ran (a follow-up fires once and goes) or was removed elsewhere: the reload shows that
      if (!(e instanceof ApiError && e.status === 404)) setError((e as Error).message)
    }
    load()
  }

  return (
    <div className="page narrow">
      <header className="page-head">
        <div>
          <Link to="/bots" className="muted small">← {t('Bots')}</Link>
          <h1>🤖 {b.name}</h1>
          <p className="muted">{b.description}</p>
        </div>
        <div className="row">
          <button className="btn" onClick={() => setEditing(true)}>{t('Editar')}</button>
          <button className="btn" onClick={async () => { await api.patch(`/api/bots/${b.id}`, { enabled: !b.enabled }); load() }}>{b.enabled ? `⏸ ${t('Pausar')}` : `▶ ${t('Activar')}`}</button>
        </div>
      </header>
      {error && <div className="error-box">{error}</div>}
      {!b.enabled && <div className="hint-box"><PausedNote b={b} /></div>}

      <div className="card">
        <div className="bot-meta">
          <span>📁 {b.project}</span>
          <span>{b.policy === 'ask' ? t('Pregunta antes de actuar') : b.policy === 'readonly' ? t('Solo lectura') : t('Autónomo')}</span>
          <span>{b.notify === 'telegram' ? '✈ Telegram' : t('sin avisos')}</span>
          <span>{t('Próxima')}: <b>{when(b.next_at)}</b></span>
          {b.thread_id && <Link to={`/p/${b.project_id}/c/${b.thread_id}`}>{t('Ver su conversación')} →</Link>}
        </div>
        <div className="bot-meta small">
          {b.triggers.length ? b.triggers.map((tr, i) => tr.kind === 'once' && tr.id ? (
            <span key={tr.id} className="follow-up" title={tr.prompt || undefined}>
              ⏰ {describeTrigger(tr)}
              <button className="queued-x" title={t('Quitar este seguimiento')} aria-label={t('Quitar este seguimiento')} onClick={() => removeFollowUp(tr)}>×</button>
            </span>
          ) : <span key={tr.id ?? i}>⏰ {describeTrigger(tr)}</span>) : <span className="muted">{t('Sin horario: solo a mano, Telegram o webhook')}</span>}
        </div>
        <div className="row wrap" style={{ marginTop: 10 }}>
          <input value={runText} onChange={(e) => setRunText(e.target.value)} placeholder={t('Instrucción para esta vez (opcional)')} style={{ flex: 1, minWidth: 220 }}
            onKeyDown={(e) => e.key === 'Enter' && run()} />
          <button className="btn primary" onClick={run}>▶ {t('Ejecutar ahora')}</button>
        </div>
      </div>

      <div className="card">
        <h3>{t('Ejecuciones')}</h3>
        {!b.runs?.length && <p className="muted small">{t('Todavía no se ha ejecutado.')}</p>}
        <div className="bot-runs">
          {b.runs?.map((r) => (
            <Link key={r.id} className="bot-run" to={`/p/${b.project_id}/c/${b.thread_id}`}>
              <StatusDot s={r.status} />
              <span className="bot-run-when">{when(r.created_at)}</span>
              <span className="pill">{r.trigger.split('#')[0] || 'manual'}</span>
              <span className="bot-run-sum">{(r.summary || '').trim().startsWith('NOTHING_NEW') ? t('sin novedades') : r.summary || r.status}</span>
              {r.usage?.cost_usd ? <span className="muted small">${r.usage.cost_usd.toFixed(3)}</span> : null}
            </Link>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>{t('Memoria')}</h3>
        <p className="muted small">{t('Lo que el bot recuerda entre ejecuciones (lo guarda él mismo).')}</p>
        {!b.memory?.length && <p className="muted small">{t('Vacía.')}</p>}
        {b.memory?.map((m) => (
          <div key={m.key} className="mem-row">
            <code>{m.key}</code><span className="mem-val">{m.value}</span>
            <button className="queued-x" title={t('Olvidar')} onClick={async () => { await api.del(`/api/bots/${b.id}/memory/${encodeURIComponent(m.key)}`); load() }}>×</button>
          </div>
        ))}
      </div>

      <div className="card">
        <h3>Webhook</h3>
        <p className="muted small">{t('Para que otro programa de este equipo (un script, un atajo, n8n…) lance el bot con un POST. El cuerpo se le pasa como datos.')}</p>
        {hook ? (
          <div className="stack">
            <div className="ok-box small">{t('Copia el token ahora: no se volverá a mostrar.')}</div>
            <code className="copyable">{hook.url}</code>
            <code className="copyable">X-Nesicode-Token: {hook.token}</code>
            <pre className="code small">{`curl -X POST ${hook.url} \\\n  -H 'X-Nesicode-Token: ${hook.token}' \\\n  -H 'Content-Type: application/json' -d '{"texto": "hola"}'`}</pre>
          </div>
        ) : b.webhook ? <p className="small">✓ {t('Activado.')}</p> : null}
        <div className="row">
          <button className="btn small" onClick={enableHook}>{b.webhook || hook ? t('Generar token nuevo') : t('Activar webhook')}</button>
          {(b.webhook || hook) && <button className="btn small ghost" onClick={disableHook}>{t('Desactivar')}</button>}
        </div>
      </div>

      <div className="card danger">
        <button className="btn small ghost" onClick={remove}>{t('Borrar bot')}</button>
      </div>
    </div>
  )
}
