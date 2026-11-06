import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, money, statusLabel, type RunEvent } from '../api'
import { t, translateError, useLang } from '../i18n'
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
  return (
    <div className={`tool ${err ? 'tool-err' : ''}`}>
      <button className="tool-head" onClick={() => setOpen((v) => !v)}>
        <span className={`tool-dot ${pending ? 'pending' : err ? 'err' : 'ok'}`} />
        <span className="tool-name">{TOOL_LABEL()[call.name] ?? mcpLabel(call.name) ?? call.name}</span>
        <code className="tool-arg">{argSummary(call.name, call.args)}</code>
        <span className="chev">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div className="tool-body">
          {call.name === 'write_file' || call.name === 'edit_file' ? (
            <pre className="code">{call.name === 'write_file' ? call.args.content : `- ${call.args.old}\n+ ${call.args.new}`}</pre>
          ) : (
            call.name !== 'bash' && <pre className="code dim">{JSON.stringify(call.args, null, 2)}</pre>
          )}
          {result && <pre className={`code ${err ? 'err' : ''}`}>{result.output}</pre>}
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
  const what = req.name === 'bash' ? String(req.args?.command ?? '') : req.name === 'write_file' || req.name === 'edit_file' ? String(req.args?.path ?? '') : JSON.stringify(req.args)
  if (decision) {
    const ok = decision === 'allow' || decision === 'always'
    return (
      <div className={`approval done ${ok ? 'ok' : 'no'}`}>
        {ok ? '✓' : '✗'} {TOOL_LABEL()[req.name] ?? mcpLabel(req.name) ?? req.name} <code>{what.slice(0, 120)}</code> ·{' '}
        {decision === 'always' ? t('permitido siempre en esta conversación') : ok ? t('permitido') : decision === 'timeout' ? t('sin respuesta') : t('denegado')}
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

function FileCard({ path, size, caption }: { path: string; size?: number; caption?: string }) {
  const { tid } = useParams()
  const src = `/api/threads/${tid}/file?path=${encodeURIComponent(path)}&download=1`
  const name = path.split('/').pop() || path
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  const kb = size ? (size < 1024 * 1024 ? `${Math.max(1, Math.round(size / 1024))} KB` : `${(size / 1024 / 1024).toFixed(1)} MB`) : ''
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

function Compacted({ e }: { e: RunEvent }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="auto-note muted">
      🗜 {t('Contexto compactado ({u} de {l} tokens): el agente sigue con un resumen de lo hecho.', { u: Number(e.used).toLocaleString(), l: Number(e.limit).toLocaleString() })}{' '}
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

/** Renders one agent run's events like a chat answer (text, tool calls, thinking). */
export default function Transcript({
  events,
  live,
  status,
  runId,
}: {
  events: RunEvent[]
  live?: string
  status: string
  runId?: number
}) {
  useLang()
  const results = new Map<string, RunEvent>()
  for (const e of events) if (e.type === 'tool_result') results.set(`${e.depth ?? 0}:${e.id}`, e)
  const done = events.find((e) => e.type === 'done' && !e.depth)
  const out: React.ReactNode[] = []
  const shownImages = new Set<string>()
  const decided = new Map<string, string>()
  for (const e of events) if (e.type === 'approval_result') decided.set(e.id, e.decision)
  events.forEach((e, i) => {
    const sub = e.depth ? <span className="sub-tag">{e.agent}</span> : null
    if (e.type === 'assistant') {
      out.push(
        <div key={i} className={`answer ${e.depth ? 'nested' : ''}`}>
          {sub}
          <Markdown text={e.text} />
        </div>,
      )
    } else if (e.type === 'tool_call') {
      out.push(
        <div key={i} className={e.depth ? 'nested' : ''}>
          {sub}
          <ToolBlock call={e} result={results.get(`${e.depth ?? 0}:${e.id}`)} />
        </div>,
      )
    } else if (e.type === 'thinking') {
      out.push(<Thinking key={i} text={e.text} />)
    } else if (e.type === 'error') {
      out.push(<div key={i} className="error-box">{translateError(e.text)}</div>)
    } else if (e.type === 'context_learned') {
      out.push(<div key={i} className="auto-note muted">📏 {t('El modelo indicó su límite real: {n} tokens. Guardado; compacto y sigo.', { n: Number(e.limit).toLocaleString() })}</div>)
    } else if (e.type === 'masked') {
      out.push(<div key={i} className="auto-note muted">🧹 {t('Contexto aligerado: {n} tokens de salidas antiguas apartados (siguen guardados en disco).', { n: Number(e.freed).toLocaleString() })}</div>)
    } else if (e.type === 'compacted') {
      out.push(<Compacted key={i} e={e} />)
    } else if (e.type === 'connector_error') {
      out.push(<div key={i} className="auto-note muted">{t('El conector {c} no arrancó', { c: e.connector })}: {e.text}</div>)
    } else if (e.type === 'file') {
      out.push(<FileCard key={i} path={e.path} size={e.size} caption={e.caption} />)
    } else if (e.type === 'image') {
      if (shownImages.has(e.path)) return
      shownImages.add(e.path)
      out.push(<ChatImage key={i} path={e.path} caption={e.caption} />)
    } else if (e.type === 'auto_harness') {
      out.push(<AutoNote key={i} runId={e.run_id} reason={e.reason} />)
    } else if (e.type === 'approval_request') {
      out.push(<ApprovalCard key={i} req={e} runId={runId} decision={decided.get(e.id)} live={status === 'running'} />)
    } else if (e.type === 'auto_harness_skipped') {
      out.push(<div key={i} className="auto-note muted">{t('Auto-harnessing no lanzado ({reason}): {why}.', { reason: translateError(e.reason), why: translateError(e.why) })}</div>)
    } else if (e.type === 'cancelled') {
      out.push(<div key={i} className="muted small">{t('Cancelado.')}</div>)
    }
  })
  const running = status === 'running' || status === 'queued'
  return (
    <div className="transcript">
      {out}
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
          {statusLabel(status)}
          {done && ` · ${t('{n} pasos', { n: done.steps })} · ${money(done.usage?.cost_usd)}`}
        </div>
      )}
    </div>
  )
}
