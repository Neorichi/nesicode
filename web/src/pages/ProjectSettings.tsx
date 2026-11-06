import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { useApp, useProject } from '../App'
import ProvidersSection, { RolePicker } from '../components/Providers'
import ConnectorsSection from '../components/Connectors'
import { t, useLang } from '../i18n'

export default function ProjectSettings() {
  const { pid } = useParams()
  const project = useProject(pid)
  const { refreshProjects, refreshThreads } = useApp()
  const navigate = useNavigate()
  const [form, setForm] = useState({ name: '', domain: '', executor_model: '', improver_model: '', vision_model: '', image_model: '', transcribe_model: '', check_command: '', max_steps: 60 })
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [upMsg, setUpMsg] = useState('')
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
    try {
      const r = await api.post<{ files: number }>(`/api/projects/${project.id}/upload`, fd)
      setUpMsg(t('{n} fichero(s) añadidos al proyecto. Las conversaciones nuevas los verán.', { n: r.files }))
    } catch (e) {
      setUpMsg((e as Error).message)
    }
  }

  const remove = async () => {
    if (!confirm(t('¿Eliminar "{name}" con su carpeta, harness y conversaciones? No se puede deshacer.', { name: project.name }))) return
    await api.del(`/api/projects/${project.id}`)
    await Promise.all([refreshProjects(), refreshThreads()])
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
      </div>

      <div className="card danger">
        <h3>{t('Zona peligrosa')}</h3>
        <button className="btn danger" onClick={remove}>{t('Eliminar proyecto')}</button>
      </div>
    </div>
  )
}
