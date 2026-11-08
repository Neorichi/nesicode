import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ACTIVE, api, isAbort, STREAM_LOST, streamRun, transcribeAudio, type Attachment, type Run, type RunEvent, type Thread } from '../api'
import { useApp, useProject } from '../App'
import Composer, { AttachmentChip } from '../components/Composer'
import Diff from '../components/Diff'
import Transcript from '../components/Transcript'
import { t, useLang } from '../i18n'
import { compactReason, followAgainIn, mergeLoaded, withEvent, withRun } from '../logic'

type Ctx = { used: number; limit: number; reason?: string }

// The text of a reply cut short by Stop or an error exists only in the browser (the server keeps finished replies).
// Keep it for the rest of the session, per run, so it doesn't vanish while the user is reading it.
const interruptedText = new Map<number, string>()

// One instance per conversation (App keys it by tid): nothing here can belong to another conversation.
export default function Chat() {
  const { pid, tid } = useParams()
  const project = useProject(pid)
  const { refreshThreads, threads, threadsPid } = useApp()
  const navigate = useNavigate()
  const [thread, setThread] = useState<Thread | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [live, setLive] = useState('')
  const liveRef = useRef('')
  const [ctxLive, setCtxLive] = useState<Ctx | null>(null)
  const [error, setError] = useState('')
  const [, setFrozen] = useState(0)
  const bottom = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  const loadSeq = useRef(0) // only the newest load may apply its response
  const seenUpdated = useRef<number | null>(null) // updated_at of the conversation on screen
  useLang()

  const appendLive = (txt: string) => {
    liveRef.current += txt
    setLive(liveRef.current)
  }
  const resetLive = () => {
    liveRef.current = ''
    setLive('')
  }
  const loadCtl = useRef<AbortController | null>(null)
  // local changes a reload that started earlier doesn't know about yet (see mergeLoaded)
  const created = useRef(new Set<number>())
  const removed = useRef(new Set<number>())

  const load = useCallback(async () => {
    const n = ++loadSeq.current
    loadCtl.current?.abort() // superseded: stop downloading it
    const ctl = new AbortController()
    loadCtl.current = ctl
    try {
      const data = await api.get<{ thread: Thread; runs: Run[] }>(`/api/threads/${tid}`, { signal: ctl.signal })
      if (n !== loadSeq.current) return null
      setThread(data.thread)
      setRuns((prev) => mergeLoaded(prev, data.runs, created.current, removed.current))
      for (const r of data.runs) {
        created.current.delete(r.id)
        if (r.status !== 'waiting') removed.current.delete(r.id)
      }
      seenUpdated.current = data.thread.updated_at
      return data
    } catch (e) {
      if (isAbort(e)) return null
      throw e
    }
  }, [tid])
  const [streamKey, setStreamKey] = useState(0) // bumped to subscribe again after the stream gave up

  useEffect(() => {
    load().catch((e) => setError(e.message))
    return () => {
      // gone: ignore (and stop downloading) anything still in flight
      loadSeq.current += 1
      loadCtl.current?.abort()
    }
  }, [load])

  // /p/<other project>/c/<id>: show it under its own project (sidebar, header, model)
  useEffect(() => {
    if (thread && pid && thread.project_id !== Number(pid)) navigate(`/p/${thread.project_id}/c/${thread.id}`, { replace: true })
  }, [thread, pid, navigate])

  const active = runs.find((r) => ACTIVE.has(r.status))
  const queue = runs.filter((r) => r.status === 'waiting')
  // queued messages show above the composer; ones removed from the queue never ran: hide them
  const shown = runs.filter((r) => r.status !== 'waiting' && !(r.status === 'cancelled' && !(r.events ?? []).length))
  const activeId = active?.id

  // Messages can arrive from Telegram or a bot. Instead of downloading the whole conversation every few seconds,
  // watch this conversation's row in the sidebar list (refreshed every 5 s): reload only when it changed.
  const row = threadsPid === Number(pid) ? threads.find((x) => x.id === Number(tid)) : undefined
  useEffect(() => {
    if (!row) return
    setThread((th) => (th && th.title !== row.title ? { ...th, title: row.title } : th)) // e.g. the auto-title
    if (activeId || seenUpdated.current === null) return // a live reply reloads by itself when it ends
    if (row.updated_at > seenUpdated.current) load().catch(() => {})
  }, [row?.updated_at, row?.title, activeId, load]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { setCtxLive(null) }, [activeId]) // a new message rebuilds the context: never show the previous run's value
  useEffect(() => {
    if (!activeId) return
    const run = runs.find((r) => r.id === activeId)!
    const after = Math.max(0, ...(run.events ?? []).map((e) => e.seq ?? 0))
    let alive = true
    let again: number | undefined
    const freeze = () => {
      if (!liveRef.current.trim()) return
      interruptedText.set(activeId, liveRef.current)
      setFrozen((n) => n + 1)
    }
    const close = streamRun(activeId, after, {
      // every (re)connection replays the reply's live text from its start: drop ours so it isn't shown twice
      onOpen: resetLive,
      onEvent: (e: RunEvent) => {
        setRuns((rs) => withEvent(rs, activeId, e))
        if (!e.depth && (e.type === 'cancelled' || e.type === 'error')) freeze()
        // a retried model call starts its reply again from the beginning
        if (e.type === 'assistant' || e.type === 'done' || e.type === 'error' || e.type === 'cancelled' || (e.type === 'llm_retry' && !e.depth)) resetLive()
      },
      onLive: (items) => {
        const c = [...items].reverse().find((i) => i.type === 'context' && !i.depth)
        if (c) setCtxLive({ used: c.used, limit: c.limit, reason: c.reason })
        const txt = items.filter((i) => i.type === 'text_delta' && !i.depth).map((i) => i.text).join('')
        if (txt) appendLive(txt)
      },
      onEnd: async (status) => {
        if (status !== 'done' && status !== 'max_steps' && status !== STREAM_LOST) freeze()
        resetLive()
        const data = await load().catch(() => null)
        refreshThreads()
        // The connection gave up (server unreachable for a while) but the reply may go on: follow it again — now if
        // the server says it still runs, or in a few seconds if it still can't be reached (else the page would show
        // "running" forever once the server is back).
        const wait = alive && status === STREAM_LOST ? followAgainIn(data?.runs ?? null, activeId) : null
        if (wait !== null) again = window.setTimeout(() => setStreamKey((k) => k + 1), wait)
      },
    })
    return () => {
      alive = false
      clearTimeout(again)
      close()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId, streamKey])

  useEffect(() => {
    if (stick.current) bottom.current?.scrollIntoView({ block: 'end' })
  }, [runs, live])

  const send = async (text: string, attachments: Attachment[] = [], opts?: { now?: boolean }) => {
    setError('')
    try {
      const run = await api.post<Run>(`/api/threads/${tid}/messages`, { text, attachments: attachments.map((a) => a.path) })
      created.current.add(run.id) // a reload started before this message must not hide it
      if (opts?.now && run.status === 'waiting') {
        if (activeId) setStoppingId(activeId)
        api.post(`/api/runs/${run.id}/send-now`).catch((e) => setError((e as Error).message))
      }
      stick.current = true
      setRuns((rs) => withRun(rs, run)) // a reload may already show it: never twice
      refreshThreads()
      return true
    } catch (e) {
      setError((e as Error).message)
      return false
    }
  }

  const upload = async (files: File[]) => {
    const fd = new FormData()
    files.forEach((f) => fd.append('files', f))
    return { attachments: await api.post<Attachment[]>(`/api/threads/${tid}/attachments`, fd), threadId: Number(tid) }
  }

  const sendNow = async (id: number) => {
    if (activeId) setStoppingId(activeId)
    try {
      await api.post(`/api/runs/${id}/send-now`)
    } catch (e) {
      setError((e as Error).message)
      setStoppingId(null)
    }
  }

  const unqueue = async (id: number) => {
    removed.current.add(id) // a reload started before this must not bring it back
    setRuns((rs) => rs.filter((r) => r.id !== id))
    try {
      await api.post(`/api/runs/${id}/cancel`)
    } catch (e) {
      removed.current.delete(id)
      setError((e as Error).message)
      load().catch(() => {})
    }
  }

  const [stoppingId, setStoppingId] = useState<number | null>(null)
  const stop = async () => {
    if (!activeId) return
    setStoppingId(activeId)
    try {
      await api.post(`/api/runs/${activeId}/cancel`)
    } catch (e) {
      setError((e as Error).message)
      setStoppingId(null)
    }
  }

  return (
    <div className="chat">
      <header className="chat-head">
        <ThreadTitle thread={thread} onSaved={(th) => { setThread(th); refreshThreads() }} />
        <div className="row">
          {thread && (
            <span className={`cage ${thread.isolated ? 'on' : 'off'}`} title={thread.isolated ? t('Esta conversación trabaja en su propia carpeta aislada; no ve el proyecto ni otras conversaciones.') : t('Esta conversación trabaja directamente sobre la carpeta del proyecto (configuración global).')}>
              {thread.isolated ? `🔒 ${t('Aislada')}` : t('Compartida')}
            </span>
          )}
          {thread?.isolated ? <ChangesButton tid={Number(tid)} busy={!!active} /> : null}
          {thread?.telegram?.linked && (
            <button
              className={`btn small ${thread.telegram.attached ? 'primary' : ''}`}
              title={thread.telegram.attached ? t('Anclada en Telegram: lo que escribas allí va aquí y ves allí lo que pasa. Pulsa para soltarla.') : t('Anclar esta conversación en Telegram')}
              onClick={async () => {
                const attach = !thread.telegram!.attached
                try {
                  await api.post(`/api/threads/${tid}/telegram`, { attach })
                  setThread((th) => (th ? { ...th, telegram: { linked: true, attached: attach } } : th))
                } catch (e) {
                  setError((e as Error).message)
                }
              }}
            >
              📌 {thread.telegram.attached ? t('En Telegram') : 'Telegram'}
            </button>
          )}
          {thread && (
            <select
              className="mode-select"
              value={thread?.exec_mode ?? ''}
              title={t('Qué necesita tu permiso en esta conversación')}
              onChange={async (e) => {
                try {
                  const r = await api.patch<Thread>(`/api/threads/${tid}`, { exec_mode: e.target.value })
                  setThread((th) => (th ? { ...th, ...r } : th))
                } catch (err) {
                  setError((err as Error).message)
                }
              }}
            >
              <option value="">{t('Modo: por defecto')} ({modeLabel(thread?.effective_mode)})</option>
              <option value="auto">{t('Modo: auto')}</option>
              <option value="ask_bash">{t('Modo: pedir para comandos')}</option>
              <option value="ask_all">{t('Modo: pedir para todo')}</option>
            </select>
          )}
          <ContextMeter runs={runs} live={activeId ? ctxLive : null} />
          {project && (
            <div className="chat-model" title={t('Modelo ejecutor · versión del harness')}>
              {project.executor_model} · v{project.harness_version}
            </div>
          )}
        </div>
      </header>
      {thread?.bot && (
        <BotBar bot={thread.bot} onRun={async () => {
          try {
            await api.post(`/api/bots/${thread.bot.id}/run`, { text: '' })
            await load()
          } catch (e) {
            setError((e as Error).message)
          }
        }} />
      )}
      <div
        className="chat-scroll"
        onScroll={(e) => {
          const el = e.currentTarget
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
        }}
      >
        <div className="chat-col">
          {shown.map((r) => (
            <div key={r.id} className="exchange">
              {r.bot_id && r.trigger && r.trigger !== 'chat' ? (
                <div className="bot-trigger">
                  <span className="bot-trigger-chip">{triggerLabel(r.trigger)}</span>
                  <span className="muted small">{new Date(r.created_at * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}</span>
                  <details className="bot-trigger-prompt"><summary className="muted small">{t('instrucción')}</summary><div className="small">{r.task}</div></details>
                </div>
              ) : (
              <div className="user-msg">
                {r.attachments && r.attachments.length > 0 && (
                  <div className="att-row in-msg">
                    {r.attachments.map((a) => <AttachmentChip key={a.path} a={a} threadId={Number(tid)} />)}
                  </div>
                )}
                {r.task}
              </div>
              )}
              <Transcript
                events={r.events ?? []}
                status={r.status}
                live={r.id === activeId ? live : undefined}
                interrupted={r.id === activeId ? undefined : interruptedText.get(r.id)}
                runId={r.id}
              />
              {r.status !== 'running' && r.status !== 'queued' && <ImproveButton run={r} pid={Number(pid)} />}
            </div>
          ))}
          <div ref={bottom} />
        </div>
      </div>
      <div className="chat-foot">
        {error && <div className="error-box">{error}</div>}
        {queue.length > 0 && (
          <div className="msg-queue">
            <div className="muted small">{t('En cola · se envían solos cuando termine la respuesta en curso')}</div>
            {queue.map((r) => (
              <div key={r.id} className="queued-msg">
                <span className="queued-text" title={r.task}>{r.task}</span>
                {r.attachments && r.attachments.length > 0 && <span className="muted small">📎 {r.attachments.length}</span>}
                <button className="queued-now" title={t('Parar lo que está haciendo y enviar este mensaje ya')} onClick={() => sendNow(r.id)}>⚡ {t('Enviar ya')}</button>
                <button className="queued-x" title={t('Quitar de la cola')} onClick={() => unqueue(r.id)}>×</button>
              </div>
            ))}
          </div>
        )}
        <Composer draftKey={`chat:${tid}`} history={runs.map((r) => r.task)} onSend={send} onStop={stop} onUpload={upload} busy={!!active} stopping={!!active && stoppingId === activeId} queueWhileBusy transcribe={(wav) => transcribeAudio(pid!, wav)} placeholder={active ? t('Escribe y se pondrá en cola…') : t('Responde o pide otra cosa…')} autoFocus />
        <div className="foot-note">{t('Nesicode puede ejecutar comandos y modificar archivos del proyecto.')}</div>
      </div>
    </div>
  )
}

type Changes = { added: string[]; modified: string[]; removed: string[]; diff: string }

function ChangesButton({ tid, busy }: { tid: number; busy: boolean }) {
  const { refreshProjects } = useApp()
  const [open, setOpen] = useState(false)
  const [ch, setCh] = useState<Changes | null>(null)
  const [msg, setMsg] = useState('')
  const [tab, setTab] = useState<'changes' | 'files'>('changes')
  const [tree, setTree] = useState<{ path: string; dir: boolean; size: number }[]>([])
  const [file, setFile] = useState<{ path: string; content: string; binary: boolean; image?: boolean } | null>(null)
  const load = useCallback(async () => setCh(await api.get<Changes>(`/api/threads/${tid}/changes`)), [tid])
  const loadTree = useCallback(async () => setTree(await api.get(`/api/threads/${tid}/tree`)), [tid])
  const openFile = async (path: string) => {
    if (/\.(png|jpe?g|gif|webp)$/i.test(path)) return setFile({ path, content: '', binary: false, image: true })
    setFile(await api.get(`/api/threads/${tid}/file?path=${encodeURIComponent(path)}`))
  }
  useEffect(() => {
    if (!busy) load()
  }, [busy, load])
  const n = ch ? ch.added.length + ch.modified.length + ch.removed.length : 0
  const apply = async () => {
    if (!confirm(t('¿Copiar {n} cambio(s) de esta conversación a la carpeta del proyecto?', { n }))) return
    try {
      const r = await api.post<Changes>(`/api/threads/${tid}/apply`)
      setMsg(t('Aplicados: +{a} ~{m} −{r}', { a: r.added.length, m: r.modified.length, r: r.removed.length }))
      await Promise.all([load(), refreshProjects()])
    } catch (e) {
      setMsg((e as Error).message)
    }
  }
  return (
    <>
      <button className={`btn small ${n ? 'primary' : ''}`} onClick={() => { setOpen(true); setMsg(''); load() }}>
        {t('Archivos')}{n ? ` · ${t('{n} cambios', { n })}` : ''}
      </button>
      {open && (
        <div className="modal-bg" onClick={() => setOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-head">
              <div className="tabs flat">
                <button className={tab === 'changes' ? 'on' : ''} onClick={() => setTab('changes')}>{t('Cambios')}{n ? ` · ${n}` : ''}</button>
                <button className={tab === 'files' ? 'on' : ''} onClick={() => { setTab('files'); loadTree() }}>{t('Archivos')}</button>
              </div>
              <button className="icon-btn" onClick={() => setOpen(false)} aria-label={t('Cerrar')}>✕</button>
            </div>
            {tab === 'files' ? (
              <div className="files-grid modal-files">
                <div className="tree">
                  {tree.length === 0 && <div className="muted small pad">{t('Carpeta vacía.')}</div>}
                  {tree.map((e) => (
                    <button key={e.path} className={`tree-item ${file?.path === e.path ? 'active' : ''}`} style={{ paddingLeft: 10 + (e.path.split('/').length - 1) * 14 }} disabled={e.dir} onClick={() => openFile(e.path)}>
                      {e.dir ? '▸ ' : ''}{e.path.split('/').pop()}
                    </button>
                  ))}
                </div>
                <div className="editor">
                  {file ? (
                    <>
                      <div className="editor-head"><code>{file.path}</code></div>
                      {file.image ? <img className="viewer-img" src={`/api/threads/${tid}/file?path=${encodeURIComponent(file.path)}`} alt={file.path} /> : <pre className="code viewer">{file.binary ? t('(binario)') : file.content}</pre>}
                    </>
                  ) : (
                    <div className="empty">{t('Elige un fichero.')}</div>
                  )}
                </div>
              </div>
            ) : (
            <div className="drawer-body">
              <p className="muted small">
                {t('La conversación trabaja en su propia jaula. Estos son los ficheros que ha cambiado desde que empezó (o desde la última vez que aplicaste). Solo llegan al proyecto si los aplicas.')}
              </p>
              {ch && (
                <div className="small">
                  <span className="add">+{ch.added.length}</span> <span className="mod">~{ch.modified.length}</span> <span className="del">−{ch.removed.length}</span>
                </div>
              )}
              <Diff text={ch?.diff ?? ''} empty={t('Sin cambios todavía')} />
              {msg && <div className="ok-box">{msg}</div>}
            </div>
            )}
            <div className="editor-foot">
              <button className="btn primary" disabled={!n || busy} onClick={apply}>{t('Aplicar al proyecto')}</button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

function ImproveButton({ run, pid }: { run: Run; pid: number }) {
  const [state, setState] = useState<{ id?: number; error?: string; busy?: boolean }>({})
  const auto = (run.events ?? []).find((e) => e.type === 'auto_harness')
  if (auto) return null
  if (state.id)
    return (
      <div className="improve">
        <Link to={`/p/${pid}/harnessing/${state.id}`}>{t('Harnessing #{n} en marcha →', { n: state.id })}</Link>
      </div>
    )
  return (
    <div className="improve">
      <button
        className="link small"
        disabled={state.busy}
        title={t('Lanza Harness² con esta misma petición para que el agente aprenda a hacerla mejor (cuesta unas 4 ejecuciones).')}
        onClick={async () => {
          setState({ busy: true })
          try {
            const r = await api.post<Run>(`/api/runs/${run.id}/improve`)
            setState({ id: r.id })
          } catch (e) {
            setState({ error: (e as Error).message })
          }
        }}
      >
        ⟳ {t('Mejorar harness con esta tarea')}
      </button>
      {state.error && <span className="warn small"> {state.error}</span>}
    </div>
  )
}

function modeLabel(m?: string) {
  return m === 'ask_all' ? t('pedir para todo') : m === 'ask_bash' ? t('pedir para comandos') : t('auto')
}

function fmtK(n: number) {
  return n >= 1_000_000 ? `${(n / 1_000_000).toFixed(n % 1_000_000 ? 1 : 0)}M` : `${Math.round(n / 1000)}k`
}

function ContextMeter({ runs, live }: { runs: Run[]; live: Ctx | null }) {
  let c: Ctx | null = live
  if (!c) {
    for (const r of [...runs].reverse()) {
      const d = [...(r.events ?? [])].reverse().find((e) => e.type === 'done' && !e.depth && e.context?.limit)
      if (d) {
        c = d.context
        break
      }
    }
  }
  if (!c || !c.limit) return null
  const pct = Math.min(100, Math.round((c.used / c.limit) * 100))
  // the step that just compacted/lightened the context says so: the drop has a visible reason
  const dropped = c.reason === 'compacted' || c.reason === 'masked'
  const why = c.reason === 'compacted'
    ? ' ' + t('Se acaba de compactar: el agente sigue con un resumen de lo hecho.')
    : c.reason === 'masked' ? ' ' + t('Se acaban de apartar salidas antiguas de herramientas (siguen en disco).') : ''
  return (
    <span className={`ctx-meter ${pct >= 67 ? 'hi' : pct >= 49 ? 'mid' : ''}`} title={t('Contexto: {u} de {l} tokens. Al 49% se aligeran las salidas antiguas de herramientas y al 67% se resume la conversación (siempre queda todo en el archivo de la conversación).', { u: c.used.toLocaleString(), l: c.limit.toLocaleString() }) + why}>
      <span className="ctx-bar"><span style={{ width: `${pct}%` }} /></span>
      {fmtK(c.used)} / {fmtK(c.limit)}
      {dropped && <span className="ctx-reason">↓ {compactReason(c.reason)}</span>}
    </span>
  )
}

function ThreadTitle({ thread, onSaved }: { thread: Thread | null; onSaved: (t: Thread) => void }) {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState('')
  useLang()
  if (!thread) return <div className="chat-title">…</div>
  const save = async () => {
    const title = value.trim()
    setEditing(false)
    if (!title || title === thread.title) return
    onSaved({ ...thread, ...(await api.patch<Thread>(`/api/threads/${thread.id}`, { title })) })
  }
  return editing ? (
    <input
      className="chat-title-input"
      value={value}
      autoFocus
      maxLength={120}
      onChange={(e) => setValue(e.target.value)}
      onBlur={save}
      onKeyDown={(e) => {
        if (e.key === 'Enter') { e.preventDefault(); save() }
        if (e.key === 'Escape') setEditing(false)
      }}
    />
  ) : (
    <button className="chat-title editable" title={t('Cambiar el título')} onClick={() => { setValue(thread.title); setEditing(true) }}>
      {thread.title} <span className="edit-pencil">✎</span>
    </button>
  )
}

function triggerLabel(trigger: string): string {
  const kind = trigger.split('#')[0]
  return ({ cron: `⏰ ${t('Tarea programada')}`, interval: `⏰ ${t('Tarea periódica')}`, once: `⏰ ${t('Seguimiento')}`,
    url: `🌐 ${t('Cambio en una web')}`, webhook: `🔗 Webhook`, telegram: `✈ Telegram`, manual: `▶ ${t('Lanzada a mano')}` } as Record<string, string>)[kind] ?? kind
}

function BotBar({ bot, onRun }: { bot: any; onRun: () => void }) {
  useLang()
  const next = bot.next_at ? new Date(bot.next_at * 1000).toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' }) : null
  return (
    <div className="bot-bar">
      <div className="bot-bar-main">
        <b>🤖 {bot.name}</b>
        <span className="muted small">{bot.description || t('Bot')}</span>
      </div>
      <div className="bot-bar-meta small">
        {bot.enabled ? (next ? <span>{t('Próxima tarea')}: <b>{next}</b></span> : <span>{t('Sin horario')}</span>) : <span className="warn">⏸ {t('En pausa')}</span>}
        <button className="btn small" onClick={onRun}>▶ {t('Ejecutar tarea ahora')}</button>
        <Link className="btn small ghost" to={`/bots/${bot.id}`}>{t('Tareas y ajustes')}</Link>
      </div>
      <div className="muted small bot-bar-hint">{t('Lo que escribas aquí lo recibe el bot, con sus instrucciones y su memoria. Puedes darle instrucciones nuevas («a partir de ahora…») y las recordará.')}</div>
    </div>
  )
}
