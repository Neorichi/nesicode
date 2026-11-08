import { useRef, useState } from 'react'
import { Link, Navigate, useNavigate, useParams } from 'react-router-dom'
import { api, transcribeAudio, type Attachment, type Thread } from '../api'
import { useApp, useProject } from '../App'
import Composer from '../components/Composer'
import { t, useLang } from '../i18n'

const IDEAS = () => [
  t('Explícame la arquitectura de este proyecto'),
  t('Añade tests para el módulo principal y haz que pasen'),
  t('Busca bugs evidentes y corrígelos'),
  t('Crea un README con instrucciones de instalación'),
]

// One instance per project (App keys it by pid): a conversation created here for an attachment always belongs to
// the project on screen.
export default function NewChat() {
  const { pid } = useParams()
  const project = useProject(pid)
  const { user, refreshThreads, projects } = useApp()
  const navigate = useNavigate()
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [uploading, setUploading] = useState(0)
  // the conversation being created (one request shared by an upload and a send that race each other)
  const pending = useRef<{ project: number; thread: Promise<Thread> } | null>(null)
  useLang()

  if (!project) return projects.length ? <Navigate to="/" replace /> : <Navigate to="/new" replace />

  const thread = () => {
    // The conversation (and its cage) is created on the first attachment or message.
    if (!pending.current || pending.current.project !== project.id) {
      const p = api.post<Thread>(`/api/projects/${project.id}/threads`, {})
      const entry = { project: project.id, thread: p }
      pending.current = entry
      p.catch(() => {
        if (pending.current === entry) pending.current = null // failed: the next try creates it again
      })
    }
    return pending.current.thread
  }

  const upload = async (files: File[]) => {
    setUploading((n) => n + 1)
    try {
      const t = await thread()
      const fd = new FormData()
      files.forEach((f) => fd.append('files', f))
      return { attachments: await api.post<Attachment[]>(`/api/threads/${t.id}/attachments`, fd), threadId: t.id }
    } finally {
      setUploading((n) => n - 1)
    }
  }

  const start = async (text: string, attachments: Attachment[] = []) => {
    setBusy(true)
    setError('')
    try {
      const t = await thread()
      await api.post(`/api/threads/${t.id}/messages`, { text, attachments: attachments.map((a) => a.path) })
      await refreshThreads()
      navigate(`/p/${project.id}/c/${t.id}`)
      return true
    } catch (e) {
      setError((e as Error).message)
      setBusy(false)
      return false
    }
  }

  const hour = new Date().getHours()
  const hello = hour < 13 ? t('Buenos días') : hour < 20 ? t('Buenas tardes') : t('Buenas noches')

  return (
    <div className="welcome">
      <div className="welcome-inner">
        <h1>
          {hello}, {user.name.split(' ')[0]}
        </h1>
        <p className="muted">
          {t('Proyecto')} <b>{project.name}</b> · harness v{project.harness_version} · {project.executor_model}
        </p>
        <Composer onSend={start} onUpload={upload} busy={busy} transcribe={(wav) => transcribeAudio(pid!, wav)} placeholder={t('¿En qué trabajamos hoy?')} autoFocus />
        {error && <div className="error-box">{error}</div>}
        <div className="ideas">
          {IDEAS().map((i) => (
            <button key={i} className="idea" disabled={busy || uploading > 0} onClick={() => start(i)}>
              {i}
            </button>
          ))}
        </div>
        <div className="welcome-tip">
          {t('¿Quieres que el agente mejore su forma de trabajar en una tarea concreta?')}{' '}
          <Link to={`/p/${project.id}/harnessing`}>{t('Lanza un harnessing')}</Link>.
        </div>
      </div>
    </div>
  )
}
