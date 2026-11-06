import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ACTIVE, api, money, relTime, statusLabel, streamRun, type Run, type RunEvent } from '../api'
import { useApp, useProject } from '../App'
import Diff from '../components/Diff'
import Transcript from '../components/Transcript'
import { t, translateError, useLang } from '../i18n'

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
  const ch = st?.changes
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
    if (!node.rollout) return
    api.get(`/api/runs/${runId}/rollouts/${node.label}`).then(setRo).catch(() => setRo(null))
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
    api.get<any>(`/api/runs/${runId}/rollouts/base1`).then((r) => setBase(r.harness)).catch(() => setBase(null))
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

function RunDetail({ runId }: { runId: number }) {
  const { refreshProjects } = useApp()
  const [run, setRun] = useState<Run | null>(null)
  const [sel, setSel] = useState<string | null>(null)
  const [msg, setMsg] = useState('')
  useLang()

  const load = useCallback(async () => setRun(await api.get<Run>(`/api/runs/${runId}`)), [runId])
  useEffect(() => {
    setRun(null)
    setSel(null)
    load()
  }, [load])

  const active = run && ACTIVE.has(run.status)
  useEffect(() => {
    if (!run || !ACTIVE.has(run.status)) return
    const after = Math.max(0, ...(run.events ?? []).map((e) => e.seq ?? 0))
    return streamRun(runId, after, {
      onEvent: (e) =>
        setRun((r) => (r && !(r.events ?? []).some((x) => x.seq === e.seq) ? { ...r, status: 'running', events: [...(r.events ?? []), e] } : r)),
      onEnd: () => {
        load()
        refreshProjects()
      },
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, !!active])

  if (!run) return <div className="muted">{t('Cargando…')}</div>
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
  const err = events.find((e) => e.type === 'error')

  const adopt = async () => {
    const r = await api.post<{ version: number }>(`/api/runs/${runId}/adopt`)
    setMsg(t('Harness adoptado como v{n}', { n: r.version }))
    await Promise.all([load(), refreshProjects()])
  }
  const apply = async () => {
    try {
      const r = await api.post<{ changes: any }>(`/api/runs/${runId}/apply`)
      setMsg(t('Aplicados {n} cambios al proyecto', { n: r.changes.added.length + r.changes.modified.length + r.changes.removed.length }))
      load()
    } catch (e) {
      setMsg((e as Error).message)
    }
  }

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
        {active && <button className="btn small" onClick={() => api.post(`/api/runs/${runId}/cancel`)}>{t('Cancelar')}</button>}
      </div>

      <div className="phases">
        {PHASES().map(([key, label]) => (
          <div key={key} className={`phase ${reached.has(key) ? 'reached' : ''} ${current === key && active ? 'now' : ''}`}>
            {label}
          </div>
        ))}
      </div>

      {err && <div className="error-box">{translateError(err.text)}</div>}

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

      {run.status === 'done' && (
        <div className="result">
          <h3>{t('Resultado')}</h3>
          <div className="row wrap">
            {res.harness_changed ? (
              res.adopted_version ? (
                <span className="ok-box inline">{t('Harness adoptado como v{n}', { n: res.adopted_version })}</span>
              ) : (
                <button className="btn primary" onClick={adopt}>{t('Adoptar el harness elegido')}</button>
              )
            ) : (
              <span className="muted">{t('El resultado elegido usa el harness base: no hay cambios que adoptar.')}</span>
            )}
            {res.applied ? (
              <span className="ok-box inline">{t('Cambios de código aplicados al proyecto')}</span>
            ) : (
              <button className="btn" onClick={apply}>{t('Aplicar los cambios de código al proyecto')}</button>
            )}
          </div>
          {msg && <div className="ok-box">{msg}</div>}
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
  useLang()
  const load = useCallback(async () => setRuns(await api.get<Run[]>(`/api/projects/${pid}/runs?kind=harness`)), [pid])
  useEffect(() => {
    load()
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
  }, [load])

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
          {rid ? <RunDetail runId={Number(rid)} /> : <NewRunForm pid={project.id} onStarted={(r) => { load(); navigate(`/p/${pid}/harnessing/${r.id}`) }} />}
        </div>
      </div>
    </div>
  )
}
