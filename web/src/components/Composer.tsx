import { useEffect, useRef, useState } from 'react'
import type { Attachment } from '../api'
import { showImage } from './ImageViewer'
import { drafts } from '../drafts'
import { t, useLang } from '../i18n'
import { pasteUploads } from '../logic'
import { startRecording, type Recording } from './recorder'

const HISTORY_KEY = 'nesi:history'

function globalHistory(): string[] {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
  } catch {
    return []
  }
}

function pushGlobalHistory(text: string) {
  try {
    const h = globalHistory().filter((x) => x !== text)
    h.push(text)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(-100)))
  } catch {
    /* storage unavailable: history just isn't kept */
  }
}

export type Uploader = (files: File[]) => Promise<{ attachments: Attachment[]; threadId: number }>

function fmtSize(n: number) {
  return n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`
}

export function AttachmentChip({ a, threadId, onRemove }: { a: Attachment; threadId?: number; onRemove?: () => void }) {
  const src = threadId && a.kind === 'image' ? `/api/threads/${threadId}/file?path=${encodeURIComponent(a.path)}` : ''
  return (
    <div className={`att ${a.kind}`} title={a.path}>
      {src ? (
        <button className="img-btn" onClick={() => showImage(src, a.name)} title={a.name}>
          <img src={src} alt={a.name} />
        </button>
      ) : (
        <span className="att-icon">{a.kind === 'text' ? '¶' : '◆'}</span>
      )}
      <div className="att-meta">
        <div className="att-name">{a.name}</div>
        <div className="att-size">{fmtSize(a.size)}</div>
      </div>
      {onRemove && (
        <button className="att-x" onClick={onRemove} aria-label={t('Quitar {name}', { name: a.name })}>
          ✕
        </button>
      )}
    </div>
  )
}

export default function Composer({
  onSend,
  onStop,
  onUpload,
  busy,
  placeholder,
  autoFocus,
  history,
  queueWhileBusy,
  stopping,
  transcribe,
  draftKey,
}: {
  /** Return false if sending failed: the text and attachments are put back. */
  onSend: (text: string, attachments: Attachment[], opts?: { now?: boolean }) => void | boolean | Promise<void | boolean>
  onStop?: () => void
  onUpload?: Uploader
  busy?: boolean
  placeholder?: string
  autoFocus?: boolean
  /** Earlier messages, oldest first, recalled with ↑ / ↓. Defaults to the global recent history. */
  history?: string[]
  /** While busy, sending is still allowed: the message waits its turn (shown by the parent). */
  queueWhileBusy?: boolean
  /** Stop was pressed and the run is winding down. */
  stopping?: boolean
  /** Microphone button: turn a WAV recording into text (it lands in the box to review before sending). */
  transcribe?: (wav: Blob) => Promise<string>
  /** Keep the unsent draft under this key (one per conversation) while the user looks elsewhere. */
  draftKey?: string
}) {
  const saved = draftKey ? drafts.get(draftKey) : undefined
  const [text, setText] = useState(saved?.text ?? '')
  const [atts, setAtts] = useState<Attachment[]>(saved?.atts ?? [])
  const [threadId, setThreadId] = useState<number | undefined>(saved?.threadId)
  useEffect(() => {
    if (!draftKey) return
    if (text.trim() || atts.length) drafts.set(draftKey, { text, atts, threadId })
    else drafts.delete(draftKey)
  }, [draftKey, text, atts, threadId])
  const [uploading, setUploading] = useState(0)
  const [drag, setDrag] = useState(false)
  const [error, setError] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)
  // ↑/↓ history: index into `past` (past.length = the live draft)
  const past = history && history.length ? history : globalHistory()
  const [hIdx, setHIdx] = useState<number | null>(null)
  const draft = useRef('')
  const recall = (dir: -1 | 1) => {
    if (!past.length) return false
    const cur = hIdx ?? past.length
    const next = Math.min(past.length, Math.max(0, cur + dir))
    if (next === cur) return false
    if (hIdx === null) draft.current = text
    setHIdx(next === past.length ? null : next)
    setText(next === past.length ? draft.current : past[next])
    requestAnimationFrame(() => {
      const el = ref.current
      if (el) el.selectionStart = el.selectionEnd = dir < 0 ? 0 : el.value.length
    })
    return true
  }
  const fileRef = useRef<HTMLInputElement>(null)
  useLang()

  // ---- microphone
  const rec = useRef<Recording | null>(null)
  const alive = useRef(true)
  const starting = useRef(false) // getUserMedia pending (permission prompt, device start): ignore more clicks
  const [mic, setMic] = useState<'idle' | 'starting' | 'rec' | 'busy'>('idle')
  const [secs, setSecs] = useState(0)
  const [level, setLevel] = useState(0)
  useEffect(() => {
    if (mic !== 'rec') return
    const t0 = Date.now()
    const id = setInterval(() => {
      setSecs(Math.floor((Date.now() - t0) / 1000))
      setLevel(rec.current?.level() ?? 0)
    }, 120)
    return () => clearInterval(id)
  }, [mic])
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
      rec.current?.cancel()
      rec.current = null
    }
  }, [])
  const toggleMic = async () => {
    if (!transcribe) return
    setError('')
    if (mic === 'idle') {
      if (starting.current || rec.current) return
      starting.current = true
      setMic('starting')
      try {
        const r = await startRecording()
        // the page changed (or another recording exists) while the prompt was open: never leave the mic on
        if (!alive.current || rec.current) {
          r.cancel()
          return
        }
        rec.current = r
        setSecs(0)
        setMic('rec')
      } catch (e) {
        if (!alive.current) return
        setMic('idle')
        const name = (e as Error).name
        setError(name === 'NotAllowedError' ? t('Sin permiso para el micrófono: actívalo en Ajustes del Sistema → Privacidad y seguridad → Micrófono.')
          : (e as Error).message === 'mic-unsupported' ? t('Este navegador no permite grabar audio.') : t('No se pudo usar el micrófono: ') + (e as Error).message)
      } finally {
        starting.current = false
      }
      return
    }
    if (mic === 'rec' && rec.current) {
      setMic('busy')
      try {
        const wav = await rec.current.stop()
        rec.current = null
        const said = (await transcribe(wav)).trim()
        if (said) {
          setText((prev) => (prev.trim() ? prev.replace(/\s*$/, ' ') : '') + said)
          requestAnimationFrame(() => ref.current?.focus())
        } else setError(t('No he oído nada. Prueba otra vez más cerca del micrófono.'))
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setMic('idle')
      }
    }
  }
  const cancelMic = () => {
    rec.current?.cancel()
    rec.current = null
    setMic('idle')
  }

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 280) + 'px'
  }, [text])

  const upload = async (files: File[]) => {
    if (!onUpload || files.length === 0) return
    setError('')
    setUploading((n) => n + files.length)
    try {
      const r = await onUpload(files)
      setThreadId(r.threadId)
      setAtts((a) => [...a, ...r.attachments])
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setUploading((n) => n - files.length)
    }
  }

  const send = async (now = false) => {
    const msg = text.trim()
    if ((!msg && atts.length === 0) || (busy && !queueWhileBusy) || uploading) return
    const sent = atts
    const typed = text
    setText('')
    setAtts([])
    setHIdx(null)
    draft.current = ''
    if (msg) pushGlobalHistory(msg)
    const ok = await onSend(msg || t('Mira los archivos adjuntos.'), sent, now ? { now: true } : undefined)
    if (ok === false) {
      // keep nothing from being lost when the server is busy or restarting
      setText(typed)
      setAtts(sent)
    }
  }

  return (
    <div
      className={`composer-wrap ${drag ? 'drag' : ''}`}
      onDragOver={(e) => {
        if (!onUpload || !e.dataTransfer.types.includes('Files')) return
        e.preventDefault()
        setDrag(true)
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        if (!onUpload) return
        e.preventDefault()
        setDrag(false)
        upload(Array.from(e.dataTransfer.files))
      }}
    >
      {(atts.length > 0 || uploading > 0) && (
        <div className="att-row">
          {atts.map((a) => (
            <AttachmentChip key={a.path} a={a} threadId={threadId} onRemove={() => setAtts((x) => x.filter((y) => y.path !== a.path))} />
          ))}
          {uploading > 0 && <div className="att uploading">{t('Subiendo {n}…', { n: uploading })}</div>}
        </div>
      )}
      <div className="composer">
        {onUpload && (
          <>
            <button className="attach" onClick={() => fileRef.current?.click()} title={t('Adjuntar archivos (también puedes pegar o arrastrar)')}>
              <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M15.5 9.5l-5.8 5.8a3.5 3.5 0 0 1-5-5l6.4-6.4a2.3 2.3 0 0 1 3.3 3.3l-6.4 6.4a1.2 1.2 0 0 1-1.7-1.7l5.8-5.8" /></svg>
            </button>
            <input
              ref={fileRef}
              type="file"
              multiple
              hidden
              onChange={(e) => {
                upload(Array.from(e.target.files ?? []))
                e.target.value = ''
              }}
            />
          </>
        )}
        <textarea
          ref={ref}
          rows={1}
          value={text}
          autoFocus={autoFocus}
          placeholder={placeholder ?? t('Pide algo a Nesicode…')}
          onChange={(e) => {
            setText(e.target.value)
            setHIdx(null)
          }}
          onPaste={(e) => {
            const files = Array.from(e.clipboardData.files)
            // text copied from Excel/Word/Numbers also brings a picture of itself: paste the text, don't upload
            if (onUpload && pasteUploads(e.clipboardData.getData('text/plain'), files)) {
              e.preventDefault()
              upload(files.map((f, i) => (f.name && f.name !== 'image.png' ? f : new File([f], `pegado-${Date.now()}-${i}.${(f.type.split('/')[1] || 'bin').replace('jpeg', 'jpg')}`, { type: f.type }))))
            }
          }}
          onKeyDown={(e) => {
            const el = e.currentTarget
            if (e.key === 'ArrowUp' && !e.shiftKey && !e.altKey && !e.metaKey && el.selectionStart === 0 && el.selectionEnd === 0) {
              if (recall(-1)) e.preventDefault()
              return
            }
            if (e.key === 'ArrowDown' && hIdx !== null && el.selectionStart === el.value.length) {
              if (recall(1)) e.preventDefault()
              return
            }
            if (e.key === 'Escape' && mic === 'rec') {
              e.preventDefault()
              cancelMic()
              return
            }
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              send(busy && queueWhileBusy && (e.metaKey || e.ctrlKey)) // ⌘/Ctrl+Enter: interrupt and send now
            }
          }}
        />
        {transcribe && (
          mic === 'rec' ? (
            <div className="mic-rec">
              <button className="mic-cancel" onClick={cancelMic} title={t('Descartar grabación (Esc)')}>×</button>
              <span className="mic-dot" style={{ transform: `scale(${1 + Math.min(1, level * 3) * 0.6})` }} />
              <span className="mic-time">{Math.floor(secs / 60)}:{String(secs % 60).padStart(2, '0')}</span>
              <button className="send mic-stop" onClick={toggleMic} title={t('Terminar y transcribir')}>
                <svg width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 10.5l4 4 8-9" /></svg>
              </button>
            </div>
          ) : (
            <button className={`mic ${mic === 'busy' || mic === 'starting' ? 'busy' : ''}`} onClick={toggleMic} disabled={mic === 'busy' || mic === 'starting'}
              title={mic === 'busy' ? t('Transcribiendo…') : mic === 'starting' ? t('Abriendo el micrófono…') : t('Dictar un mensaje')}>
              {mic === 'busy' || mic === 'starting' ? <span className="spinner" /> : (
                <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="7" y="2.5" width="6" height="10" rx="3" /><path d="M4 9.5a6 6 0 0 0 12 0M10 15.5V18" /></svg>
              )}
            </button>
          )
        )}
        {busy && onStop && !(queueWhileBusy && (text.trim() || atts.length > 0)) ? (
          <button className={`send stop ${stopping ? 'stopping' : ''}`} onClick={onStop} disabled={stopping} title={stopping ? t('Deteniendo…') : t('Detener')}>
            <svg width="14" height="14" viewBox="0 0 14 14"><rect width="14" height="14" rx="2" fill="currentColor" /></svg>
          </button>
        ) : (
          <button className="send" onClick={() => send()} disabled={(!text.trim() && atts.length === 0) || (busy && !queueWhileBusy) || uploading > 0} title={busy ? t('Poner en cola (Enter) · ⌘/Ctrl+Enter: parar y enviar ya') : t('Enviar (Enter)')}>
            <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10 16V4M5 9l5-5 5 5" /></svg>
          </button>
        )}
      </div>
      {drag && <div className="drop-hint">{t('Suelta los archivos para adjuntarlos')}</div>}
      {error && <div className="error-box">{error}</div>}
    </div>
  )
}
