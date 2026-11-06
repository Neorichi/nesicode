import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type Profile, type Project } from '../api'
import { useApp } from '../App'
import { t, useLang } from '../i18n'
import '../components/ImageViewer'
import { ConfigImport, useProjectTemplate } from '../components/ConfigTransfer'

const baseName = (p: string) => p.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || ''

export default function NewProject() {
  const { refreshProjects, projects } = useApp()
  const navigate = useNavigate()
  useLang()
  const [folder, setFolder] = useState('')
  const [name, setName] = useState('')
  const [nameTouched, setNameTouched] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const native = !!window.pywebview?.api?.pick_folder
  const [tpl, reloadTpl] = useProjectTemplate()
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [profile, setProfile] = useState('general')
  useEffect(() => { api.get<Profile[]>('/api/profiles').then(setProfiles).catch(() => {}) }, [])
  const [useTpl, setUseTpl] = useState(true)
  const [showImport, setShowImport] = useState(false)

  const choose = async () => {
    const p = await window.pywebview?.api?.pick_folder?.()
    if (p) {
      setFolder(p)
      if (!nameTouched) setName(baseName(p))
    }
  }

  const create = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError('')
    try {
      const p = await api.post<Project>('/api/projects', { name: name.trim() || baseName(folder) || t('Proyecto'), folder, use_template: !!tpl && useTpl, profile })
      await refreshProjects()
      navigate(`/p/${p.id}`)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="welcome">
      <form className="new-project" onSubmit={create}>
        <h1>{projects.length ? t('Nuevo proyecto') : t('Crea tu primer proyecto')}</h1>
        <p className="muted">
          {t('Elige la carpeta principal del proyecto. Cada conversación trabaja en su propia copia y los cambios solo llegan a tu carpeta cuando los aplicas.')}
        </p>

        <div className="folder-pick">
          {native ? (
            <button type="button" className="btn primary" onClick={choose}>
              {folder ? t('Cambiar carpeta…') : t('Elegir carpeta…')}
            </button>
          ) : null}
          <input
            value={folder}
            onChange={(e) => {
              setFolder(e.target.value)
              if (!nameTouched) setName(baseName(e.target.value))
            }}
            placeholder={native ? t('o escribe la ruta') : t('/ruta/a/tu/proyecto')}
            spellCheck={false}
          />
        </div>

        <label>
          {t('Nombre')}
          <input
            value={name}
            onChange={(e) => {
              setName(e.target.value)
              setNameTouched(true)
            }}
            placeholder={baseName(folder) || t('mi-proyecto')}
          />
        </label>

        {profiles.length > 0 && !(tpl && useTpl) && (
          <div className="profile-pick">
            <div className="profile-pick-label">{t('Perfil')}</div>
            <div className="profile-grid">
              {profiles.map((p) => (
                <button type="button" key={p.id} className={`profile-card ${profile === p.id ? 'on' : ''}`} onClick={() => setProfile(p.id)}>
                  <span className="profile-icon">{p.icon}</span>
                  <b>{t(p.label)}</b>
                  <span className="muted small">{t(p.description)}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {tpl && (
          <label className="check">
            <input type="checkbox" checked={useTpl} onChange={(e) => setUseTpl(e.target.checked)} />
            {t('Usar la configuración importada de')} «{tpl.from}» {t('(modelos, verificación y harness)')}
          </label>
        )}

        {error && <div className="error-box">{error}</div>}
        <div className="row wrap">
          <button className="btn primary" disabled={busy || (!folder.trim() && !name.trim())}>
            {busy ? t('Creando…') : t('Crear proyecto')}
          </button>
          {!folder && <span className="muted small">{t('Sin carpeta: Nesicode crea una vacía para el proyecto.')}</span>}
        </div>
        <p className="muted small">{t('Modelos, dominio y comando de verificación se configuran después en Ajustes del proyecto.')}</p>
      </form>
      <div className="new-project import-box">
        {showImport ? (
          <>
            <h3>{t('Importar configuración')}</h3>
            <p className="muted small">{t('Carga el archivo .nesicode que exportaste en tu otro equipo (Cuenta → Exportar configuración): proveedores con sus keys, conectores, preferencias y la plantilla de proyecto.')}</p>
            <ConfigImport compact onDone={() => { reloadTpl(); refreshProjects() }} />
          </>
        ) : (
          <button type="button" className="link" onClick={() => setShowImport(true)}>{t('¿Vienes de otro equipo? Importa tu configuración')}</button>
        )}
      </div>
    </div>
  )
}
