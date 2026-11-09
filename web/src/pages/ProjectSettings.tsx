import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api, type Profile, type Project, type Thread } from '../api'
import { useApp, useProject } from '../App'
import { forgetDraft } from '../drafts'
import ProvidersSection, { RolePicker } from '../components/Providers'
import ConnectorsSection from '../components/Connectors'
import { t, useLang } from '../i18n'
import { skippedPath, uploadReport } from '../logic'

export default function ProjectSettings() {
  const { pid } = useParams()
  const project = useProject(pid)
  const { refreshProjects, refreshThreads } = useApp()
  const navigate = useNavigate()
  const [form, setForm] = useState({ name: '', domain: '', executor_model: '', improver_model: '', vision_model: '', image_model: '', transcribe_model: '', check_command: '', max_steps: 60 })
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [upMsg, setUpMsg] = useState('')
  const [upSkipped, setUpSkipped] = useState<string[]>([]) // the uploaded ZIP's entries not extracted ("path: reason")
  const [delError, setDelError] = useState('') // e.g. 409: a reply of this project hasn't stopped yet
  const [deleting, setDeleting] = useState(false)
  useLang()

  useEffect(() => {
    if (project)
      setForm({
        name: project.name, domain: project.domain, executor_model: project.executor_model, improver_model: project.improver_model,
        vision_model: project.vision_model ?? '', image_model: project.image_model ?? '', transcribe_model: project.transcribe_model ?? '', check_command: project.check_command, max_steps: project.max_steps,
      })
  }, [project])

  if (!project) return null
  const set = (k: keyof typeof form) => (v: any) => setForm((f) => ({ ...f, [k]: v }))

  const save = async (e?: React.FormEvent) => {
    e?.preventDefault()
    setMsg('')
    setError('')
    try {
      await api.patch(`/api/projects/${project.id}`, form)
      await refreshProjects()
      setMsg(t('Guardado'))
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const upload = async (f: File) => {
    const fd = new FormData()
    fd.append('file', f)
    setUpMsg(t('Subiendo…'))
    setUpSkipped([])
    try {
      const r = uploadReport(await api.post<{ files: number; skipped?: string[] }>(`/api/projects/${project.id}/upload`, fd))
      setUpMsg(t('{n} fichero(s) añadidos al proyecto. Las conversaciones nuevas los verán.', { n: r.files }))
      setUpSkipped(r.skipped)
    } catch (e) {
      setUpMsg((e as Error).message)
    }
  }

  const remove = async () => {
    if (!confirm(t('¿Eliminar "{name}" con su carpeta, harness y conversaciones? No se puede deshacer.', { name: project.name }))) return
    setDelError('')
    setDeleting(true)
    try {
      // its conversations go too: forget their unsent drafts (a new conversation may get one of their ids)
      const gone = await api.get<Thread[]>(`/api/projects/${project.id}/threads`).catch(() => [] as Thread[])
      await api.del(`/api/projects/${project.id}`)
      gone.forEach((th) => forgetDraft(`chat:${th.id}`))
    } catch (e) {
      // the server said why (a reply still stopping, …): the project is still there and the page stays usable
      setDelError((e as Error).message)
      setDeleting(false)
      return
    }
    await Promise.all([refreshProjects(), refreshThreads()]).catch(() => {})
    navigate('/')
  }

  return (
    <div className="page narrow">
      <header className="page-head">
        <div>
          <h1>{t('Ajustes del proyecto')}</h1>
          <p className="muted small">{t('Carpeta base:')} <code>{project.workspace}</code></p>
        </div>
      </header>

      <form className="card form" onSubmit={save}>
        <h3>{t('General')}</h3>
        <label>{t('Nombre')}<input value={form.name} onChange={(e) => set('name')(e.target.value)} /></label>
        <label>
          {t('Dominio')}
          <input value={form.domain} onChange={(e) => set('domain')(e.target.value)} />
          <span className="small muted">{t('Los proyectos del mismo dominio comparten lecciones.')}</span>
        </label>
        <label>{t('Comando de verificación')}<input value={form.check_command} onChange={(e) => set('check_command')(e.target.value)} placeholder="uvx --quiet pytest -q" /></label>
        <label>{t('Máximo de pasos por respuesta')}<input type="number" min={5} max={300} value={form.max_steps} onChange={(e) => set('max_steps')(Number(e.target.value))} /></label>
        {error && <div className="error-box">{error}</div>}
        {msg && <div className="ok-box">{msg}</div>}
        <button className="btn primary">{t('Guardar')}</button>
      </form>

      <ProfileCard projectId={project.id} current={project.profile || 'general'} onChanged={() => refreshProjects()} />

      <div className="card form">
        <h3>{t('Modelos por función')}</h3>
        <RolePicker label={t('Chat')} hint={t('el agente que trabaja en las conversaciones')} value={form.executor_model} onChange={set('executor_model')} />
        <RolePicker label={t('Mejora')} hint={t('propone, combina y elige harnesses (usa el más listo que tengas)')} value={form.improver_model} onChange={set('improver_model')} />
        <RolePicker label={t('Visión')} hint={t('lee las imágenes adjuntas si el modelo de chat no puede')} value={form.vision_model} onChange={set('vision_model')} optional />
        <RolePicker label={t('Crear imágenes')} hint={t('genera imágenes nuevas a partir de una descripción (p. ej. Grok Imagine, gpt-image)')} value={form.image_model} onChange={set('image_model')} optional />
        <RolePicker label={t('Transcripción')} hint={t('convierte en texto lo que dictas con el micrófono (p. ej. openai/whisper-1). Vacío = el reconocimiento de voz del Mac')} value={form.transcribe_model} onChange={set('transcribe_model')} optional />
        <button className="btn primary" onClick={() => save()}>{t('Guardar modelos')}</button>
      </div>

      <div className="card">
        <h3>{t('Proveedores')}</h3>
        <p className="muted small">
          {t('Tus proveedores de modelos (endpoint + API key). Son de tu cuenta y sirven para todos tus proyectos. Las API keys se guardan en la bóveda cifrada y nunca se muestran.')}
        </p>
        <ProvidersSection onAssigned={() => refreshProjects()} />
      </div>

      <div className="card mcp-wide">
        <h3>{t('Conectores (MCP)')}</h3>
        <p className="muted small">{t('Herramientas externas que el agente puede usar en todas tus conversaciones: tu Chrome, GitHub, Word… Los locales se ejecutan en tu equipo, fuera de la jaula; en los modos «pedir» cada uso te pide permiso.')}</p>
        <ConnectorsSection />
      </div>

      <div className="card">
        <h3>{t('Archivos del proyecto')}</h3>
        <p className="muted small">{t('Sube un fichero o un .zip a la carpeta base. Cada conversación nueva empieza con una copia de esta carpeta.')}</p>
        <label className="btn">
          {t('Subir fichero o .zip')}
          <input type="file" hidden onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
        </label>
        {upMsg && <div className="ok-box">{upMsg}</div>}
        <UploadSkipped skipped={upSkipped} />
      </div>

      <ReadDirsCard project={project} onSaved={() => refreshProjects()} />

      <div className="card danger">
        <h3>{t('Zona peligrosa')}</h3>
        {delError && <div className="error-box">{delError}</div>}
        <button className="btn danger" disabled={deleting} onClick={remove}>{deleting ? t('Eliminando…') : t('Eliminar proyecto')}</button>
      </div>
    </div>
  )
}

/** The entries of an uploaded .zip that were not extracted ("path: reason"), and why: a symbolic link never is. */
export function UploadSkipped({ skipped }: { skipped: string[] }) {
  useLang()
  if (!skipped.length) return null
  return (
    <div className="apply-report">
      <div className="warn small">{t('{n} entrada(s) del .zip no se extrajeron:', { n: skipped.length })}</div>
      <ul className="apply-list small">
        {skipped.map(skippedPath).map((s, i) => <li key={`${s.path}:${i}`}><code>{s.path}</code>{s.reason && <> — {s.reason}</>}</li>)}
      </ul>
    </div>
  )
}

/**
 * Folders of the user's the agent may READ (never change) in this project: the list, «Quitar», a path to add (or the
 * desktop app's folder picker) and «Añadir Descargas». Every change is saved at once; the server checks each folder and
 * stores it as the OS spells it (shown back), or says why not (its `detail`, shown as it comes).
 */
export function ReadDirsCard({ project, onSaved }: { project: Project; onSaved: () => void }) {
  useLang()
  const [dirs, setDirs] = useState<string[]>(project.read_dirs ?? [])
  const [path, setPath] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  useEffect(() => setDirs(project.read_dirs ?? []), [project.read_dirs])
  const mode = project.read_dirs_mode ?? 'off'
  const native = typeof window !== 'undefined' && !!window.pywebview?.api?.pick_folder
  const hasDownloads = dirs.some((d) => /[\\/]Downloads$/i.test(d))

  const save = async (next: string[]) => {
    setBusy(true)
    setError('')
    try {
      const p = await api.patch<Project>(`/api/projects/${project.id}`, { read_dirs: next })
      setDirs(p.read_dirs ?? [])
      setPath('')
      onSaved()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const add = (p: string) => {
    if (p.trim() && !busy) save([...dirs, p.trim()])
  }
  const pick = async () => {
    const p = await window.pywebview?.api?.pick_folder?.()
    if (p) add(p)
  }

  return (
    <div className="card form">
      <h3>{t('Carpetas que el agente puede leer')}</h3>
      <p className="warn small">{t('El agente puede leer estas carpetas (nunca cambiarlas) y lo que lea se envía al modelo.')}</p>
      {mode === 'off' ? (
        <p className="muted small">{t('Solo en la app de escritorio / modo local')}</p>
      ) : (
        <>
          {dirs.length ? (
            <ul className="read-dirs">
              {dirs.map((d) => (
                <li key={d}>
                  <code>{d}</code>
                  <button type="button" className="btn small" disabled={busy} onClick={() => save(dirs.filter((x) => x !== d))}>{t('Quitar')}</button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted small">{t('Ninguna: el agente solo ve la carpeta de la conversación.')}</p>
          )}
          <div className="folder-pick">
            {native && <button type="button" className="btn" disabled={busy} onClick={pick}>{t('Elegir carpeta…')}</button>}
            <input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  add(path)
                }
              }}
              placeholder={t('/ruta/a/la/carpeta')}
              spellCheck={false}
            />
            <button type="button" className="btn" disabled={busy || !path.trim()} onClick={() => add(path)}>{t('Añadir')}</button>
          </div>
          {!hasDownloads && (
            <div className="row">
              <button type="button" className="btn small" disabled={busy} onClick={() => add('~/Downloads')}>{t('Añadir Descargas')}</button>
            </div>
          )}
          {mode === 'none' && (
            <p className="muted small">{t('En este equipo los comandos del agente no van en una jaula: esta lista le dice qué carpetas puede leer, pero no le impide leer otras.')}</p>
          )}
          {error && <div className="error-box">{error}</div>}
        </>
      )}
    </div>
  )
}

function ProfileCard({ projectId, current, onChanged }: { projectId: number; current: string; onChanged: () => void }) {
  useLang()
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [pick, setPick] = useState(current)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  useEffect(() => { api.get<Profile[]>('/api/profiles').then(setProfiles).catch(() => {}) }, [])
  useEffect(() => setPick(current), [current])
  const apply = async () => {
    setBusy(true); setMsg('')
    try {
      const r = await api.post<{ harness_version: number }>(`/api/projects/${projectId}/profile`, { profile: pick })
      setMsg(t('Perfil aplicado: nueva versión del harness v{v} (lo que el proyecto ya había aprendido se conserva).', { v: r.harness_version }))
      onChanged()
    } catch (e) { setMsg((e as Error).message) } finally { setBusy(false) }
  }
  return (
    <div className="card form">
      <h3>{t('Perfil')}</h3>
      <p className="muted small">{t('Define el tono, las reglas y las skills del agente en este proyecto.')}</p>
      <div className="profile-grid">
        {profiles.map((p) => (
          <button type="button" key={p.id} className={`profile-card ${pick === p.id ? 'on' : ''}`} onClick={() => setPick(p.id)}>
            <span className="profile-icon">{p.icon}</span>
            <b>{t(p.label)}{p.id === current ? ` · ${t('actual')}` : ''}</b>
            <span className="muted small">{t(p.description)}</span>
          </button>
        ))}
      </div>
      <div className="row">
        <button className="btn primary" disabled={busy || pick === current} onClick={apply}>{busy ? '…' : t('Cambiar perfil')}</button>
      </div>
      {msg && <div className="ok-box small">{msg}</div>}
    </div>
  )
}
