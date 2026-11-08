import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ACTIVE, api, ApiError, isAbort, money, relTime, statusLabel, STREAM_LOST, streamRun, type Run, type RunEvent } from '../api'
import { useApp, useProject } from '../App'
import ApplyReport from '../components/ApplyReport'
import Diff from '../components/Diff'
import Transcript from '../components/Transcript'
import { useVisibleInterval } from '../hooks'
import { t, translateError, useLang } from '../i18n'
import { applyCounts, canRetryApply, followAgainIn, type ApplyResult } from '../logic'

type Node = {
  label: string
  dim?: string
  stage?: RunEvent // stage "done" event
  stageRunning?: boolean
  rollout?: RunEvent // rollout summary event
  rolloutRunning?: boolean
}

const PHASES = (): [string, string][] => [
  ['base', t('Base')],
  ['propose', 'Propose'],
  ['probe', 'Probe'],
  ['compose', 'Compose'],
  ['probe_final', 'Probe final'],
  ['select', t('Selección')],
  ['reflect', t('Reflexión')],
  ['done', t('Hecho')],
]

const DIM_LABEL = (): Record<string, string> => ({ INV: t('INV · cómo trabaja'), DOM: t('DOM · qué sabe'), AGG: t('Composición'), COMPOSE: 'Checkpoint' })

function buildNodes(events: RunEvent[]): Map<string, Node> {
  const nodes = new Map<string, Node>()
  const get = (label: string) => {
    if (!nodes.has(label)) nodes.set(label, { label })
    return nodes.get(label)!
  }
  for (const e of events) {
    if (e.type === 'stage') {
      const n = get(e.label)
      n.dim = e.dim ?? n.dim
      if (e.status === 'running') n.stageRunning = true
      else {
        n.stage = e
        n.stageRunning = false
      }
    } else if (e.type === 'rollout') {
      const n = get(e.label)
      if (e.status === 'running') n.rolloutRunning = true
      else {
        n.rollout = e
        n.rolloutRunning = false
      }
    }
  }
  return nodes
}

function NodeCard({ node, selected, chosen, onClick }: { node: Node; selected: boolean; chosen: boolean; onClick: () => void }) {
  const st = node.stage
  const ro = node.rollout
  const running = node.stageRunning || node.rolloutRunning
  const ch = { added: [], modified: [], removed: [], ...st?.changes }
  return (
    <button className={`node ${selected ? 'sel' : ''} ${chosen ? 'chosen' : ''} ${running ? 'running' : ''}`} onClick={onClick}>
      <div className="node-top">
        <span className="node-label">{node.label}</span>
        {node.dim && <span className={`dim dim-${node.dim}`}>{node.dim}</span>}
        {chosen && <span className="badge-chosen">{t('elegido')}</span>}
      </div>
      {node.stageRunning && <div className="node-line muted">{t('proponiendo…')}</div>}
      {st && (
        <div className="node-line">
          {st.fell_back ? (
            <span className="warn">fallback</span>
          ) : st.kept_base ? (
            <span className="muted">KEEP_BASE</span>
          ) : (
            <span>
              <span className="add">+{ch.added.length}</span> <span className="mod">~{ch.modified.length}</span>{' '}
              <span className="del">−{ch.removed.length}</span> {t('componentes')}
            </span>
          )}
        </div>
      )}
      {node.rolloutRunning && <div className="node-line muted">{t('ejecutando agente…')}</div>}
      {ro && (
        <div className="node-line">
          <span className={ro.delivered ? 'ok' : 'warn'}>{ro.delivered ? `✓ ${t('entregado')}` : `✗ ${t('sin entrega')}`}</span>
          {ro.check_code !== null && ro.check_code !== undefined && (
            <span className={ro.check_code === 0 ? 'ok' : 'warn'}> · check {ro.check_code === 0 ? 'OK' : ro.check_code}</span>
          )}
          <span className="muted"> · {t('{n} pasos', { n: ro.steps })} · {t('{n} fich.', { n: ro.files_changed })}</span>
        </div>
      )}
    </button>
  )
}

function NodeDetail({ runId, node, onClose }: { runId: number; node: Node; onClose: () => void }) {
  const [ro, setRo] = useState<any>(null)
  const [tab, setTab] = useState<'notes' | 'trajectory' | 'diff' | 'check'>(node.stage ? 'notes' : 'trajectory')
  useEffect(() => {
    setRo(null)
    if (!node.rollout) return
    let alive = true // another node clicked meanwhile: drop this answer
    api.get(`/api/runs/${runId}/rollouts/${node.label}`).then((r) => alive && setRo(r)).catch(() => alive && setRo(null))
    return () => {
      alive = false
    }
  }, [runId, node.label, node.rollout])
  return (
    <div className="drawer">
      <div className="drawer-head">
        <div>
          <b>{node.label}</b> {node.dim && <span className={`dim dim-${node.dim}`}>{DIM_LABEL()[node.dim] ?? node.dim}</span>}
        </div>
        <button className="icon-btn" onClick={onClose} aria-label={t('Cerrar')}>✕</button>
      </div>
      <div className="tabs">
        {node.stage && <button className={tab === 'notes' ? 'on' : ''} onClick={() => setTab('notes')}>{t('Propuesta')}</button>}
        <button className={tab === 'trajectory' ? 'on' : ''} onClick={() => setTab('trajectory')}>{t('Trayectoria')}</button>
        <button className={tab === 'diff' ? 'on' : ''} onClick={() => setTab('diff')}>{t('Cambios en código')}</button>
        <button className={tab === 'check' ? 'on' : ''} onClick={() => setTab('check')}>{t('Check')}</button>
      </div>
      <div className="drawer-body">
        {tab === 'notes' && node.stage && (
          <>
            {node.stage.error && <div className="error-box"><pre className="wrap">{node.stage.error}</pre></div>}
            <h4>{t('Notas del proponente')}</h4>
            <pre className="wrap notes">{node.stage.notes || '—'}</pre>
            <h4>{t('Componentes')}</h4>
            <div className="small">{node.stage.inventory}</div>
            {ro?.harness && <HarnessDiffAgainstBase runId={runId} harness={ro.harness} />}
          </>
        )}
        {tab === 'trajectory' && (ro ? <Transcript events={ro.events} status="done" /> : <div className="muted">{t('Sin rollout todavía.')}</div>)}
        {tab === 'diff' && <Diff text={ro?.diff ?? ''} empty={t('El agente no cambió ficheros')} />}
        {tab === 'check' && <pre className="code">{ro?.check || t('No hay comando de verificación configurado.')}</pre>}
      </div>
    </div>
  )
}

function HarnessDiffAgainstBase({ runId, harness }: { runId: number; harness: Record<string, string> }) {
  const [base, setBase] = useState<Record<string, string> | null>(null)
  useEffect(() => {
    let alive = true
    api.get<any>(`/api/runs/${runId}/rollouts/base1`).then((r) => alive && setBase(r.harness)).catch(() => alive && setBase(null))
    return () => {
      alive = false
    }
  }, [runId])
  const text = useMemo(() => {
    if (!base) return ''
    const out: string[] = []
    for (const p of [...new Set([...Object.keys(base), ...Object.keys(harness)])].sort()) {
      if (base[p] === harness[p]) continue
      out.push(`--- a/${p}`, `+++ b/${p}`)
      const a = (base[p] ?? '').split('\n')
      const b = (harness[p] ?? '').split('\n')
      const sa = new Set(a)
      const sb = new Set(b)
      a.forEach((l) => !sb.has(l) && out.push('-' + l))
      b.forEach((l) => !sa.has(l) && out.push('+' + l))
    }
    return out.join('\n')
  }, [base, harness])
  return (
    <>
      <h4>{t('Harness frente al base')}</h4>
      <Diff text={text} empty={t('Idéntico al harness base')} />
    </>
  )
}

/** An apply result as stored in the run (result.applied) or sent with the 'applied' event; null when there is none. */
const asApplied = (x: unknown): ApplyResult | null => (x && typeof x === 'object' && !Array.isArray(x) ? (x as ApplyResult) : null)

/** What reached the project: the counts (left-out paths never counted) and, path by path, what was left out and why. */
function AppliedBox({ applied }: { applied: ApplyResult }) {
  const c = applyCounts(applied)
  return (
    <>
      <div className="ok-box">
        {t('Cambios de código aplicados al proyecto')}: <span className="add">+{c.added}</span> <span className="mod">~{c.modified}</span>{' '}
        <span className="del">−{c.removed}</span>
        {c.left > 0 && <span className="warn"> · {t('{n} sin aplicar', { n: c.left })}</span>}
      </div>
      <ApplyReport result={applied} />
    </>
  )
}

/**
 * «Aplicar» for a finished run: offered until something was applied, and again («Reintentar») while the last apply
 * left out changes the server keeps the run's copies for (a folder that was not there, a file that could not be
 * written…: see canRetryApply). Never once the server said those copies are gone (410): it could only fail again.
 */
export function ApplyButton({ applied, gone, busy, onApply }: { applied: ApplyResult | null; gone: boolean; busy: boolean; onApply: () => void }) {
  useLang()
  if (gone || (applied && !canRetryApply(applied))) return null
  return applied ? (
    <button className="btn" disabled={busy} onClick={onApply}
      title={t('Vuelve a aplicar lo que quedó sin aplicar (p. ej. cuando la carpeta del proyecto vuelva a estar disponible). Lo que cambiaste en el proyecto no se sobrescribe.')}>
      {t('Reintentar aplicar los cambios de código')}
    </button>
  ) : (
    <button className="btn" disabled={busy} onClick={onApply}>{t('Aplicar los cambios de código al proyecto')}</button>
  )
}

// Rendered with key={runId}: one instance per run, so Adopt/Apply always act on the run on screen.
function RunDetail({ runId }: { runId: number }) {
  const { refreshProjects } = useApp()
  const [run, setRun] = useState<Run | null>(null)
  const [sel, setSel] = useState<string | null>(null)
  const [msg, setMsg] = useState('')
  const [err, setErr] = useState('') // a failed action (e.g. 409: the project is busy) — the buttons stay usable
  const [busy, setBusy] = useState(false)
  const [lastApplied, setLastApplied] = useState<ApplyResult | null>(null) // the «Aplicar» answer until the reload has it
  const [copiesGone, setCopiesGone] = useState(false) // «Aplicar» answered 410: the run's copies of the project are gone
  const [merged, setMerged] = useState(false) // the manual adopt merged the run's edits onto a newer version
  const loadSeq = useRef(0)
  const loadCtl = useRef<AbortController | null>(null)
  const [streamKey, setStreamKey] = useState(0) // bumped to follow the run again after its stream gave up
  useLang()

  // only the newest request may paint, and only with this run's data (a big run's download stops when superseded)
  const load = useCallback(async (): Promise<Run | null> => {
    const n = ++loadSeq.current
    loadCtl.current?.abort()
    const ctl = new AbortController()
    loadCtl.current = ctl
    try {
      const r = await api.get<Run>(`/api/runs/${runId}`, { signal: ctl.signal })
      if (n !== loadSeq.current || r.id !== runId) return null
      setRun(r)
      return r
    } catch (e) {
      if (!isAbort(e)) throw e
      return null
    }
  }, [runId])
  useEffect(() => {
    setRun(null)
    setSel(null)
    load().catch((e) => setErr((e as Error).message))
    return () => {
      loadSeq.current += 1
      loadCtl.current?.abort()
    }
  }, [load])

  const active = run && ACTIVE.has(run.status)
  useEffect(() => {
    if (!run || !ACTIVE.has(run.status)) return
    const after = Math.max(0, ...(run.events ?? []).map((e) => e.seq ?? 0))
    let alive = true
    let again: number | undefined
    const close = streamRun(runId, after, {
      onEvent: (e) =>
        setRun((r) => (r && !(r.events ?? []).some((x) => x.seq === e.seq) ? { ...r, status: 'running', events: [...(r.events ?? []), e] } : r)),
      onEnd: async (status) => {
        const r = await load().catch(() => null)
        refreshProjects().catch(() => {})
        // the stream gave up (server unreachable for a while) but the run may go on: follow it again (see Chat)
        const wait = alive && status === STREAM_LOST ? followAgainIn(r ? [r] : null, runId) : null
        if (wait !== null) again = window.setTimeout(() => setStreamKey((k) => k + 1), wait)
      },
    })
    return () => {
      alive = false
      clearTimeout(again)
      close()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, !!active, streamKey])

  if (!run) return err ? <div className="error-box">{err}</div> : <div className="muted">{t('Cargando…')}</div>
  const events = run.events ?? []
  const nodes = buildNodes(events)
  const phases = events.filter((e) => e.type === 'phase').map((e) => e.phase)
  const current = phases[phases.length - 1]
  const reached = new Set(phases)
  const selected = events.find((e) => e.type === 'selected')
  const res = run.result ?? {}
  const sorted = [...nodes.values()].sort((a, b) => a.label.localeCompare(b.label, 'en', { numeric: true }))
  const col = (f: (n: Node) => boolean) => sorted.filter(f)
  const columns: [string, Node[]][] = [
    [t('Base'), col((n) => n.label.startsWith('base'))],
    [t('Candidatos (INV / DOM)'), col((n) => /^(inv|dom)\d/.test(n.label))],
    [run.params.mode === 'sequential' ? 'Checkpoints' : t('Composiciones'), col((n) => /^(agg|c)\d/.test(n.label))],
  ]
  const lesson = events.find((e) => e.type === 'lesson')
  const failure = events.find((e) => e.type === 'error')
  // the auto-apply's result (conflicts included), kept in the run's result and also sent as an 'applied' event
  const applied = asApplied(res.applied) ?? lastApplied ?? asApplied([...events].reverse().find((e) => e.type === 'applied')?.changes)
  const adoptedEv = [...events].reverse().find((e) => e.type === 'adopted')
  const adoptMerged = merged || (!!adoptedEv?.merged && adoptedEv.version === res.adopted_version)
  // why the auto-adopt didn't happen (the harness changed meanwhile in the same files, …): the manual button stays
  const adoptSkipped: string = res.adopt_skipped || events.find((e) => e.type === 'adopt_skipped')?.reason || ''

  // Adopt/Apply may be refused (409: another conversation is working on the project): say why, keep the buttons
  const act = async (call: () => Promise<void>) => {
    setBusy(true)
    setErr('')
    setMsg('')
    try {
      await call()
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const adopt = () => act(async () => {
    const r = await api.post<{ version: number; merged?: boolean }>(`/api/runs/${runId}/adopt`)
    setMerged(!!r.merged)
    refreshProjects().catch(() => {})
    const fresh = await load().catch(() => null)
    if (!fresh?.result?.adopted_version) // the reload doesn't show it (yet): say it here
      setMsg(r.merged ? t('Harness adoptado como v{n}, combinado con los cambios que se hicieron en el harness mientras trabajaba', { n: r.version })
        : t('Harness adoptado como v{n}', { n: r.version }))
  })
  const apply = () => act(async () => {
    const r = await api.post<{ changes: ApplyResult }>(`/api/runs/${runId}/apply`).catch((e) => {
      if (e instanceof ApiError && e.status === 410) setCopiesGone(true) // (its message says why: shown below)
      throw e
    })
    const changes = asApplied(r.changes) ?? {}
    setLastApplied(changes)
    // a retry's answer replaces the stored one of the apply before it at once (not only once a reload brings it: a
    // reload that fails must not leave the old «sin aplicar» and its «Reintentar» on screen)
    setRun((cur) => (cur && cur.id === runId ? { ...cur, result: { ...(cur.result ?? {}), applied: changes } } : cur))
    await load().catch(() => {})
  })

  return (
    <div className="run-detail">
      <div className="run-task">
        <div className="muted small">
          #{run.id} · {run.params.mode === 'sequential' ? t('secuencial') : t('paralelo')} · k={run.params.k} · harness v{run.harness_version} ·{' '}
          <b>{statusLabel(run.status)}</b>
          {res.usage && ` · ${money(res.usage.cost_usd)}`}
        </div>
        <div className="task-text">{run.task}</div>
        {run.params.source && (
          <div className="small muted">
            {run.params.source.auto ? t('Lanzado automáticamente ({reason}) desde', { reason: translateError(run.params.source.reason) }) : t('Lanzado desde')}{' '}
            <Link to={`/p/${run.project_id}/c/${run.params.source.thread_id}`}>{t('la conversación')}</Link>
          </div>
        )}
        {active && <button className="btn small" onClick={() => api.post(`/api/runs/${runId}/cancel`).catch((e) => setErr((e as Error).message))}>{t('Cancelar')}</button>}
        {err && run.status !== 'done' && <div className="error-box">{err}</div>}
      </div>

      <div className="phases">
        {PHASES().map(([key, label]) => (
          <div key={key} className={`phase ${reached.has(key) ? 'reached' : ''} ${current === key && active ? 'now' : ''}`}>
            {label}
          </div>
        ))}
      </div>

      {failure && <div className="error-box">{translateError(failure.text)}</div>}

      <div className="pipeline">
        {columns.map(([title, list], i) => (
          <div className="col" key={title}>
            <div className="col-title">
              <span className="col-n">{i + 1}</span> {title}
            </div>
            {list.length === 0 && <div className="node ghost">—</div>}
            {list.map((n) => (
              <NodeCard key={n.label} node={n} selected={sel === n.label} chosen={selected?.label === n.label} onClick={() => setSel(n.label)} />
            ))}
          </div>
        ))}
        <div className="col">
          <div className="col-title"><span className="col-n">4</span> {t('Selección')}</div>
          {selected ? (
            <div className="select-card">
              <div>
                {t('Elegido:')} <b>{selected.label}</b>
              </div>
              <div className="small muted clamp">{selected.reason}</div>
            </div>
          ) : (
            <div className="node ghost">{active ? t('pendiente…') : '—'}</div>
          )}
          {lesson && (
            <div className="lesson-card">
              <div className="small muted">{t('Lección aprendida')}</div>
              <div className="small"><b>{t('Cuándo:')}</b> {lesson.cue}</div>
              <div className="small"><b>{t('Qué hacer:')}</b> {lesson.lesson}</div>
            </div>
          )}
        </div>
      </div>

      {sel && nodes.get(sel) && <NodeDetail runId={runId} node={nodes.get(sel)!} onClose={() => setSel(null)} />}

      {run.status !== 'done' && applied && (
        // stopped after its changes were applied: what reached the project still matters
        <div className="result">
          <h3>{t('Resultado')}</h3>
          <AppliedBox applied={applied} />
        </div>
      )}

      {run.status === 'done' && (
        <div className="result">
          <h3>{t('Resultado')}</h3>
          <div className="row wrap">
            {res.harness_changed ? (
              res.adopted_version ? (
                <span className="ok-box inline">
                  {adoptMerged
                    ? t('Harness adoptado como v{n}, combinado con los cambios que se hicieron en el harness mientras trabajaba', { n: res.adopted_version })
                    : t('Harness adoptado como v{n}', { n: res.adopted_version })}
                </span>
              ) : (
                <button className="btn primary" disabled={busy} onClick={adopt}>{t('Adoptar el harness elegido')}</button>
              )
            ) : (
              <span className="muted">{t('El resultado elegido usa el harness base: no hay cambios que adoptar.')}</span>
            )}
            <ApplyButton applied={applied} gone={copiesGone} busy={busy} onApply={apply} />
          </div>
          {res.harness_changed && !res.adopted_version && adoptSkipped && (
            <div className="muted small">{t('No se adoptó automáticamente: {why}.', { why: translateError(adoptSkipped) })}</div>
          )}
          {applied && <AppliedBox applied={applied} />}
          {msg && <div className="ok-box">{msg}</div>}
          {err && <div className="error-box">{err}</div>}
          <h4>{t('Cambios en el harness')}</h4>
          <Diff text={res.harness_diff ?? ''} empty={t('Sin cambios en el harness')} />
          {res.reflection?.edits && (
            <>
              <h4>{t('Qué hizo cada edición')}</h4>
              <pre className="wrap notes">{res.reflection.edits}</pre>
            </>
          )}
        </div>
      )}
    </div>
  )
}

function NewRunForm({ pid, onStarted }: { pid: number; onStarted: (r: Run) => void }) {
  const project = useProject(String(pid))
  const [task, setTask] = useState('')
  const [k, setK] = useState(1)
  const [mode, setMode] = useState<'parallel' | 'sequential'>('parallel')
  const [apply, setApply] = useState(false)
  const [adopt, setAdopt] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  useLang()

  const start = async () => {
    setBusy(true)
    setError('')
    try {
      const r = await api.post<Run>(`/api/projects/${pid}/harnessing`, { task, k, mode, apply_changes: apply, adopt_harness: adopt })
      setTask('')
      onStarted(r)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card new-run">
      <h3>{t('Nueva ejecución de Harness²')}</h3>
      <textarea rows={4} placeholder={t('Describe la tarea. El agente la resolverá varias veces con distintas versiones de su harness.')} value={task} onChange={(e) => setTask(e.target.value)} />
      <div className="form-row">
        <label>
          {t('Presupuesto k')}
          <select value={k} onChange={(e) => setK(Number(e.target.value))}>
            {[1, 2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <div className="seg">
          <button className={mode === 'parallel' ? 'on' : ''} onClick={() => setMode('parallel')}>{t('Paralelo · explora en anchura')}</button>
          <button className={mode === 'sequential' ? 'on' : ''} onClick={() => setMode('sequential')}>{t('Secuencial · refina en profundidad')}</button>
        </div>
      </div>
      <label className="check"><input type="checkbox" checked={adopt} onChange={(e) => setAdopt(e.target.checked)} /> {t('Adoptar automáticamente el harness elegido')}</label>
      <label className="check"><input type="checkbox" checked={apply} onChange={(e) => setApply(e.target.checked)} /> {t('Aplicar al proyecto los cambios de código del resultado elegido')}</label>
      <div className="muted small">
        {t('Coste: {runs} ejecuciones del agente ({executor}) y {calls} llamadas al mejorador ({improver}).', { runs: 4 * k, executor: project?.executor_model ?? '', calls: 3 * k, improver: project?.improver_model ?? '' })}
        {!project?.check_command && <> {t('Consejo: define un')} <Link to={`/p/${pid}/settings`}>{t('comando de verificación')}</Link> {t('(tests) para dar mejor evidencia.')}</>}
      </div>
      {error && <div className="error-box">{error}</div>}
      <button className="btn primary" disabled={!task.trim() || busy} onClick={start}>{t('Lanzar')}</button>
    </div>
  )
}

export default function Harnessing() {
  const { pid, rid } = useParams()
  const project = useProject(pid)
  const navigate = useNavigate()
  const [runs, setRuns] = useState<Run[]>([])
  const listSeq = useRef(0)
  useLang()
  // newest request wins (switching projects must not show the previous project's runs)
  const load = useCallback(async () => {
    const n = ++listSeq.current
    try {
      const rs = await api.get<Run[]>(`/api/projects/${pid}/runs?kind=harness`)
      if (n === listSeq.current) setRuns(rs)
    } catch {
      /* keep the list; the next poll retries */
    }
  }, [pid])
  useEffect(() => {
    setRuns([])
    load()
  }, [load])
  useVisibleInterval(() => void load(), 5000)

  if (!project) return null
  return (
    <div className="page wide">
      <header className="page-head">
        <div>
          <h1>Harnessing</h1>
          <p className="muted">
            {t('Propose → Probe → Compose sobre una tarea: el agente prueba variantes de su propio harness, se queda con la mejor y guarda la lección para el dominio')} <b>{project.domain}</b>.
          </p>
        </div>
      </header>
      <div className="hn-grid">
        <div className="hn-list">
          <button className={`hn-item ${!rid ? 'active' : ''}`} onClick={() => navigate(`/p/${pid}/harnessing`)}>{t('+ Nueva ejecución')}</button>
          {runs.map((r) => (
            <button key={r.id} className={`hn-item ${Number(rid) === r.id ? 'active' : ''}`} onClick={() => navigate(`/p/${pid}/harnessing/${r.id}`)}>
              <div className="clamp1">
                {r.params?.source?.auto ? <span className="pill">auto</span> : r.params?.source ? <span className="pill">chat</span> : null} {r.task}
              </div>
              <div className="small muted">
                #{r.id} · {statusLabel(r.status)} · {relTime(r.created_at)}
                {r.summary?.adopted_version ? ` · → v${r.summary.adopted_version}` : ''}
              </div>
            </button>
          ))}
        </div>
        <div className="hn-main">
          {rid ? <RunDetail key={rid} runId={Number(rid)} /> : <NewRunForm pid={project.id} onStarted={(r) => { load(); navigate(`/p/${pid}/harnessing/${r.id}`) }} />}
        </div>
      </div>
    </div>
  )
}
