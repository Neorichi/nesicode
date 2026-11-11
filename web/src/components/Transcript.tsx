import { Component, useState, type ReactNode } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, costSuffix, statusLabel, type RunEvent } from '../api'
import { t, translateError, useLang } from '../i18n'
import { compactionErrorText, compactReason, interruptedSaved, retryGroups, retryText, splitExit, unknownEventNote, unshownFinal } from '../logic'
import Markdown from './Markdown'
import { downloadFile, showImage } from './ImageViewer'

const TOOL_LABEL = (): Record<string, string> => ({
  list_dir: t('Listar'),
  read_file: t('Leer'),
  write_file: t('Escribir'),
  edit_file: t('Editar'),
  search: t('Buscar'),
  bash: t('Terminal'),
  load_skill: 'Skill',
  delegate: t('Subagente'),
  view_image: t('Visión'),
  share_file: t('Compartir archivo'),
  chrome_tabs: 'Chrome · tabs',
  chrome_open: 'Chrome · open',
  chrome_read: 'Chrome · read',
  chrome_click: 'Chrome · click',
  chrome_type: 'Chrome · type',
  chrome_scroll: 'Chrome · scroll',
  chrome_back: 'Chrome · back',
  chrome_screenshot: 'Chrome · screenshot',
  chrome_close: 'Chrome · close',
  generate_image: t('Crear imagen'),
  create_bot: t('Crear bot'),
  http_fetch: t('Consultar web'),
  bot_memory_get: t('Leer memoria'),
  bot_memory_set: t('Guardar en memoria'),
  bot_memory_list: t('Ver memoria'),
  bot_memory_delete: t('Olvidar'),
  bot_notify: t('Avisar'),
  bot_schedule_once: t('Programar seguimiento'),
  bot_finish: t('Terminar tarea'),
  show_image: t('Mostrar imagen'),
})

function mcpLabel(name: string): string | null {
  const m = name.match(/^mcp__([^_]+(?:-[^_]+)*)__(.+)$/)
  return m ? `${m[1]} · ${m[2].replace(/_/g, ' ')}` : null
}

function argSummary(name: string, args: Record<string, any>): string {
  if (!args) return ''
  if (name === 'chrome_click' || name === 'chrome_type') return `${args.ref ?? ''}${args.text ? ` · ${args.text}` : ''}`
  if (name.startsWith('chrome_')) return args.url ?? ''
  if (name.startsWith('mcp__')) return args.url ?? args.text ?? args.selector ?? args.query ?? JSON.stringify(args).slice(0, 120)
  if (name === 'bash') return args.command ?? ''
  if (name === 'search') return `${args.pattern ?? ''}${args.path ? `  ${t('en {path}', { path: args.path })}` : ''}`
  if (name === 'load_skill') return args.name ?? ''
  if (name === 'generate_image') return String(args.prompt ?? '').slice(0, 140)
  if (name === 'show_image') return args.caption || args.path || ''
  if (name === 'view_image') return `${args.path ?? ''}${args.question ? ` · ${args.question}` : ''}`
  if (name === 'delegate') return `${args.agent}: ${String(args.task ?? '').slice(0, 120)}`
  return args.path ?? JSON.stringify(args).slice(0, 120)
}

function ToolBlock({ call, result }: { call: RunEvent; result?: RunEvent }) {
  const [open, setOpen] = useState(false)
  const pending = !result
  const err = result?.is_error
  const name = String(call.name ?? '')
  const args = call.args ?? {}
  // (a terminal result's "exit code: 0" is for the model; the user sees the code only when the command failed)
  const out = result ? splitExit(name, result.output) : null
  const failed = out?.code != null && out.code !== 0
  return (
    <div className={`tool ${err ? 'tool-err' : ''}`}>
      <button className="tool-head" onClick={() => setOpen((v) => !v)}>
        <span className={`tool-dot ${pending ? 'pending' : err ? 'err' : 'ok'}`} />
        <span className="tool-name">{TOOL_LABEL()[name] ?? mcpLabel(name) ?? name}</span>
        <code className="tool-arg">{argSummary(name, args)}</code>
        {failed && <span className="pill warn" title={t('El comando terminó con un código distinto de 0')}>{t('código {n}', { n: out!.code! })}</span>}
        <span className="chev">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div className="tool-body">
          {name === 'write_file' || name === 'edit_file' ? (
            <pre className="code">{name === 'write_file' ? args.content : `- ${args.old}\n+ ${args.new}`}</pre>
          ) : (
            name !== 'bash' && <pre className="code dim">{JSON.stringify(args, null, 2)}</pre>
          )}
          {result && (out!.body.trim()
            ? <pre className={`code ${err ? 'err' : ''}`}>{out!.body}</pre>
            : <div className="muted small">{t('(sin salida)')}</div>)}
        </div>
      )}
    </div>
  )
}

function AutoNote({ runId, reason }: { runId: number; reason: string }) {
  const { pid } = useParams()
  return (
    <div className="auto-note">
      ⟳ {t('Esta respuesta {reason}: se lanzó automáticamente un', { reason: translateError(reason) })}{' '}
      <Link to={`/p/${pid}/harnessing/${runId}`}>harnessing #{runId}</Link> {t('para mejorar el harness.')}
    </div>
  )
}

function ApprovalCard({ req, runId, decision, live }: { req: RunEvent; runId?: number; decision?: string; live: boolean }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const decide = async (d: 'allow' | 'always' | 'deny') => {
    if (!runId) return
    setBusy(true)
    setError('')
    try {
      await api.post(`/api/runs/${runId}/approve`, { call_id: req.id, decision: d })
    } catch (e) {
      setError((e as Error).message)
      setBusy(false)
    }
  }
  const what = req.name === 'bash' ? String(req.args?.command ?? '') : req.name === 'write_file' || req.name === 'edit_file' ? String(req.args?.path ?? '') : JSON.stringify(req.args ?? {})
  if (decision) {
    const ok = decision === 'allow' || decision === 'always'
    // Stop while the card was waiting: nobody answered it, so it must not read as "denied"
    const stopped = decision === 'cancelled'
    return (
      <div className={`approval done ${ok ? 'ok' : stopped ? 'stopped' : 'no'}`}>
        {ok ? '✓' : stopped ? '■' : '✗'} {TOOL_LABEL()[req.name] ?? mcpLabel(req.name) ?? req.name} <code>{what.slice(0, 120)}</code> ·{' '}
        {decision === 'always' ? t('permitido siempre en esta conversación') : ok ? t('permitido') : stopped ? t('cancelado (se detuvo la respuesta)')
          : decision === 'timeout' ? t('sin respuesta') : t('denegado')}
      </div>
    )
  }
  return (
    <div className="approval">
      <div className="approval-head">
        <span className="tool-dot pending" /> {t('Nesicode quiere ejecutar')} <b>{TOOL_LABEL()[req.name] ?? mcpLabel(req.name) ?? req.name}</b>
      </div>
      {req.always === false && <div className="muted small">{t('Esta acción se aprueba cada vez.')}</div>}
      <pre className="code">{req.name === 'write_file' ? `${what}\n\n${String(req.args?.content ?? '').slice(0, 2000)}` : req.name === 'edit_file' ? `${what}\n- ${req.args?.old}\n+ ${req.args?.new}` : what}</pre>
      {live ? (
        <div className="row wrap">
          <button className="btn primary small" disabled={busy} onClick={() => decide('allow')}>{t('Permitir')}</button>
          {req.always !== false && <button className="btn small" disabled={busy} onClick={() => decide('always')}>{t('Permitir siempre en esta conversación')}</button>}
          <button className="btn small ghost" disabled={busy} onClick={() => decide('deny')}>{t('Denegar')}</button>
        </div>
      ) : (
        <div className="muted small">{t('La ejecución terminó sin respuesta.')}</div>
      )}
      {error && <div className="error-box">{translateError(error)}</div>}
    </div>
  )
}

const FILE_ICON: Record<string, string> = { pptx: '▤', ppt: '▤', key: '▤', xlsx: '▦', xls: '▦', csv: '▦', docx: '¶', doc: '¶', pdf: '◧', zip: '◫', png: '▣', jpg: '▣', jpeg: '▣', svg: '▣' }

function HtmlPreview({ path, name, src }: { path: string; name: string; src: string }) {
  const { tid } = useParams()
  const [big, setBig] = useState(false)
  const [key, setKey] = useState(0)
  const view = `/api/threads/${tid}/file?path=${encodeURIComponent(path)}&view=1`
  const frame = (cls: string) => (
    // sandboxed: the generated page cannot read the app, its cookies or its API
    <iframe key={key} className={cls} src={view} title={name} sandbox="allow-scripts allow-popups allow-modals allow-downloads" />
  )
  return (
    <div className="html-preview">
      <div className="html-preview-bar">
        <span className="file-icon ext-html">{'</>'}</span>
        <b className="html-preview-name">{name}</b>
        <span className="row">
          <button className="btn small ghost" title={t('Recargar')} onClick={() => setKey((k) => k + 1)}>↻</button>
          <button className="btn small" onClick={() => setBig(true)}>⤢ {t('Abrir en grande')}</button>
          <button className="btn small" onClick={() => downloadFile(src, name)}>{t('Descargar')}</button>
        </span>
      </div>
      {frame('html-frame')}
      {big && (
        <div className="html-modal" onClick={() => setBig(false)}>
          <div className="html-modal-inner" onClick={(e) => e.stopPropagation()}>
            <div className="html-preview-bar">
              <b className="html-preview-name">{name}</b>
              <button className="btn small" onClick={() => setBig(false)}>✕ {t('Cerrar')}</button>
            </div>
            {frame('html-frame big')}
          </div>
        </div>
      )}
    </div>
  )
}

function FileCard({ path, size, caption }: { path: string; size?: number; caption?: string }) {
  const { tid } = useParams()
  const src = `/api/threads/${tid}/file?path=${encodeURIComponent(path)}&download=1`
  const name = path.split('/').pop() || path
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  const kb = size ? (size < 1024 * 1024 ? `${Math.max(1, Math.round(size / 1024))} KB` : `${(size / 1024 / 1024).toFixed(1)} MB`) : ''
  if (ext === 'html' || ext === 'htm') return <HtmlPreview path={path} name={name} src={src} />
  return (
    <div className="file-card">
      <span className={`file-icon ext-${ext}`}>{FILE_ICON[ext] ?? '◆'}</span>
      <div className="file-meta">
        <div className="file-name">{name}</div>
        <div className="muted small">{[ext.toUpperCase(), kb, caption].filter(Boolean).join(' · ')}</div>
      </div>
      <button className="btn small" onClick={() => downloadFile(src, name)}>{t('Descargar')}</button>
    </div>
  )
}

/** A number from an event, or '?' when the event doesn't carry it (never "NaN"). */
const num = (x: unknown) => (typeof x === 'number' || (typeof x === 'string' && x.trim()) ? (Number.isFinite(Number(x)) ? Number(x).toLocaleString() : '?') : '?')

function Compacted({ e }: { e: RunEvent }) {
  const [open, setOpen] = useState(false)
  const why = compactReason(e.reason)
  return (
    <div className="auto-note muted">
      🗜 {t('Contexto compactado ({u} de {l} tokens): el agente sigue con un resumen de lo hecho.', { u: num(e.used), l: num(e.limit) })}{' '}
      {why && <span className="small">({why}) </span>}
      {e.fallback && <span className="small">({t('resumen de respaldo: falló el resumen del modelo')}) </span>}
      {e.path && <code className="small">{e.path}</code>}{' '}
      <button className="link small" onClick={() => setOpen((v) => !v)}>{open ? t('Ocultar resumen') : t('Ver resumen')}</button>
      {open && <pre className="wrap notes">{e.summary}</pre>}
    </div>
  )
}

function ChatImage({ path, caption }: { path: string; caption?: string }) {
  const { tid } = useParams()
  const src = `/api/threads/${tid}/file?path=${encodeURIComponent(path)}`
  const name = path.split('/').pop() || 'image'
  return (
    <figure className="chat-image">
      <button className="img-btn" onClick={() => showImage(src, name)} title={t('Ampliar')}>
        <img src={src} alt={caption || path} loading="lazy" />
      </button>
      <figcaption>
        <code>{path}</code>
        <button className="link small" onClick={() => downloadFile(src, name)}>{t('Descargar')}</button>
      </figcaption>
    </figure>
  )
}

function Thinking({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="thinking">
      <button onClick={() => setOpen((v) => !v)}>{open ? '▾' : '▸'} {t('Razonamiento')}</button>
      {open && <div className="thinking-body">{text}</div>}
    </div>
  )
}

/** One event that can't be drawn (an odd or newer payload) becomes a short note instead of a blank chat. */
class EventBoundary extends Component<{ type: string; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    if (!this.state.failed) return this.props.children
    return <div className="auto-note muted small">{t('No se pudo mostrar un evento ({type}).', { type: this.props.type || '?' })}</div>
  }
}

type RowProps = { e: RunEvent; result?: RunEvent; decision?: string; retries?: number; current: boolean; runId?: number; status: string }

/** One event of the run, or nothing (events drawn with another one, live-only or bookkeeping). */
function EventRow({ e, result, decision, retries, current, runId, status }: RowProps) {
  const sub = e.depth ? <span className="sub-tag">{e.agent}</span> : null
  switch (e.type) {
    case 'llm_retry': // one line per retried model call (the latest attempt); while it is still retrying it reads as live status
      return (
        <div className={`retry-note ${current ? 'now' : 'muted'} ${e.depth ? 'nested' : ''}`}>
          {sub}↻ {retryText(e, retries ?? 1, current)}
        </div>
      )
    case 'assistant':
      return (
        <div className={`answer ${e.depth ? 'nested' : ''} ${e.interrupted ? 'interrupted' : ''}`}>
          {sub}
          <Markdown text={e.text} />
          {e.interrupted && <div className="muted small">{t('Respuesta interrumpida: se detuvo antes de terminar.')}</div>}
        </div>
      )
    case 'tool_call':
      return (
        <div className={e.depth ? 'nested' : ''}>
          {sub}
          <ToolBlock call={e} result={result} />
        </div>
      )
    case 'thinking':
      return <Thinking text={e.text} />
    case 'error':
      return <div className="error-box">{translateError(e.text)}</div>
    case 'context_learned':
      return <div className="auto-note muted">📏 {t('El modelo indicó su límite real: {n} tokens. Guardado; compacto y sigo.', { n: num(e.limit) })}</div>
    case 'masked':
      return <div className="auto-note muted">🧹 {t('Contexto aligerado: {n} tokens de salidas antiguas apartados (siguen guardados en disco).', { n: num(e.freed) })}</div>
    case 'compacted':
      return <Compacted e={e} />
    case 'compaction_error': // live-only today (Chat shows it as a notice); drawn the same way if a server keeps it
      return <div className="retry-note muted">⚠ {compactionErrorText(e.text)}</div>
    case 'notify':
      return <div className="auto-note">📣 {e.sent ? t('Aviso enviado por Telegram') : t('Aviso')}: {e.text}</div>
    case 'bot_finished':
      return <div className="auto-note">✅ {t('Tarea cumplida: el bot se ha desactivado')} — {e.text}</div>
    case 'bot_created':
      return <div className="auto-note">🤖 {t('Bot creado')}: <Link to={`/bots/${e.bot_id}`}><b>{e.name}</b></Link> — {t('gestiónalo en la página Bots')}</div>
    case 'connector_error':
      return <div className="auto-note muted">{t('El conector {c} no arrancó', { c: e.connector })}: {e.text}</div>
    case 'file':
      return <FileCard path={e.path} size={e.size} caption={e.caption} />
    case 'image':
      return <ChatImage path={e.path} caption={e.caption} />
    case 'auto_harness':
      return <AutoNote runId={e.run_id} reason={e.reason} />
    case 'auto_harness_skipped':
      return <div className="auto-note muted">{t('Auto-harnessing no lanzado ({reason}): {why}.', { reason: translateError(e.reason), why: translateError(e.why) })}</div>
    case 'auto_harness_error':
      return <div className="auto-note muted">⚠ {t('No se pudo lanzar el auto-harnessing')}: {e.text}</div>
    case 'approval_request':
      return <ApprovalCard req={e} runId={runId} decision={decision} live={status === 'running'} />
    case 'cancelled':
      return <div className="muted small">{t('Cancelado.')}</div>
    case 'stopped': // why the run ended before its work was done (its context filled up, it was stuck): its own note
      return e.depth ? null : <div className="auto-note">⏹ {translateError(String(e.text ?? ''))}</div>
    case 'omitted': // (added by the server between the start and the end of a very long run)
      return <div className="auto-note muted">⋯ {t('{n} entradas intermedias de esta respuesta no se muestran aquí: es muy larga.', { n: num(e.count) })}</div>
  }
  // a type this version doesn't know (a newer server): one compact line with its own words, if it has any
  const note = unknownEventNote(e)
  return note ? <div className="auto-note muted small">{note}</div> : null
}

/**
 * Renders one agent run's events like a chat answer (text, tool calls, thinking). Event types it doesn't know are a
 * compact line (or nothing when they say nothing), and an event that fails to draw never blanks the rest.
 */
export default function Transcript({
  events,
  live,
  status,
  runId,
  interrupted,
  notice,
  finalText,
  stopReason,
}: {
  events: RunEvent[]
  live?: string
  status: string
  runId?: number
  /** Text that was streaming when the reply was stopped or failed (the server doesn't keep it). */
  interrupted?: string
  /** A live-only problem of the step in progress (the context summary failed: 'compaction_error'). */
  notice?: string
  /** The run's final answer as the server keeps it (shown when no event of the page carries it). */
  finalText?: string
  /** Why a 'max_steps' run stopped (the run's summary: 'step_cap' | 'context_full'). */
  stopReason?: string
}) {
  useLang()
  events = events.filter((e) => e && typeof e === 'object')
  const results = new Map<string, RunEvent>()
  for (const e of events) if (e.type === 'tool_result') results.set(`${e.depth ?? 0}:${e.id}`, e)
  const done = events.find((e) => e.type === 'done' && !e.depth)
  const out: ReactNode[] = []
  const firstImage = new Map<string, number>() // the same picture is shown once
  events.forEach((e, i) => e.type === 'image' && !firstImage.has(e.path) && firstImage.set(e.path, i))
  const decided = new Map<string, string>()
  for (const e of events) if (e.type === 'approval_result') decided.set(e.id, e.decision)
  const running = status === 'running' || status === 'queued'
  const retries = retryGroups(events)
  // the partial reply goes right before the "cancelled"/error line that ended it (or at the end)
  const cut = interrupted && !interruptedSaved(events, interrupted) ? interrupted : ''
  const cutAt = cut ? events.findIndex((e) => !e.depth && (e.type === 'cancelled' || e.type === 'error')) : -1
  const cutNode = cut ? (
    <div key="interrupted" className="answer interrupted">
      <Markdown text={cut} />
      <div className="muted small">{t('Respuesta interrumpida: este texto se quedó a medias y no se guardó en la conversación.')}</div>
    </div>
  ) : null
  events.forEach((e, i) => {
    if (i === cutAt) out.push(cutNode)
    if (e.type === 'llm_retry' && !retries.get(i)) return // folded into the run's last retry
    if (e.type === 'image' && firstImage.get(e.path) !== i) return
    out.push(
      <EventBoundary key={i} type={String(e.type ?? '')}>
        <EventRow e={e} result={e.type === 'tool_call' ? results.get(`${e.depth ?? 0}:${e.id}`) : undefined}
          decision={e.type === 'approval_request' ? decided.get(e.id) : undefined} retries={retries.get(i)}
          current={e.type === 'llm_retry' && running && !live && i === events.length - 1} runId={runId} status={status} />
      </EventBoundary>,
    )
  })
  if (cut && cutAt < 0) out.push(cutNode)
  const unshown = !running && (status === 'done' || status === 'max_steps') ? unshownFinal(events, finalText) : ''
  if (unshown)
    out.push(
      <div key="final" className="answer">
        <Markdown text={unshown} />
      </div>,
    )
  return (
    <div className="transcript">
      {out}
      {notice && running && <div className="retry-note now">⚠ {compactionErrorText(notice)}</div>}
      {live && (
        <div className="answer streaming">
          <Markdown text={live} />
        </div>
      )}
      {running && !live && (
        <div className="typing">
          <span />
          <span />
          <span />
        </div>
      )}
      {!running && (
        <div className="run-meta">
          {statusLabel(status, stopReason || done?.stop_reason)}
          {done && ` · ${t('{n} pasos', { n: num(done.steps) })}${costSuffix(done.usage?.cost_usd)}`}
        </div>
      )}
    </div>
  )
}
