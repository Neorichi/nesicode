import { useState } from 'react'
import { Link, NavLink, useLocation, useNavigate } from 'react-router-dom'
import { api, relTime, switchLang } from '../api'
import { t, useLang } from '../i18n'
import { useApp } from '../App'

const I = {
  plug: <path d="M7 3v4M13 3v4M5 7h10v3a5 5 0 0 1-10 0zM10 15v3" />,
  key: <path d="M12.5 3.5a4 4 0 1 1-3.2 6.4L4 15.2V17h2v-1.5h1.5V14H9l1.3-1.3a4 4 0 0 1 2.2-9.2zM14 6.5h.01" />,
  plus: <path d="M10 4v12M4 10h12" />,
  chat: <path d="M4 5h12v8H8l-4 3z" />,
  harness: <path d="M4 4h5v5H4zM11 4h5v5h-5zM4 11h5v5H4zM13.5 11v5M11 13.5h5" />,
  loop: <path d="M5 8a5 5 0 0 1 9-2l1.5 1.5M15 12a5 5 0 0 1-9 2L4.5 12.5M15.5 4v3.5H12M4.5 16v-3.5H8" />,
  book: <path d="M4 4.5A1.5 1.5 0 0 1 5.5 3H16v12H5.5A1.5 1.5 0 0 0 4 16.5zM4 16.5A1.5 1.5 0 0 0 5.5 18H16" />,
  folder: <path d="M3 6a1 1 0 0 1 1-1h4l2 2h6a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z" />,
  gear: <path d="M10 7a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM10 2v2.5M10 15.5V18M2 10h2.5M15.5 10H18M4.3 4.3l1.8 1.8M13.9 13.9l1.8 1.8M4.3 15.7l1.8-1.8M13.9 6.1l1.8-1.8" />,
  trash: <path d="M5 6h10M8 6V4h4v2M6.5 6l.7 10h5.6l.7-10" />,
}

export function Icon({ name, size = 18 }: { name: keyof typeof I; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {I[name]}
    </svg>
  )
}

export default function Sidebar({ onNavigate }: { onNavigate: () => void }) {
  const { user, projects, threads, refreshThreads, logout } = useApp()
  const navigate = useNavigate()
  const loc = useLocation()
  const m = loc.pathname.match(/^\/p\/(\d+)/)
  const pid = m ? Number(m[1]) : Number(localStorage.getItem('nesi:project')) || projects[0]?.id
  const project = projects.find((p) => p.id === pid)
  const own = threads.filter((t) => t.project_id === pid)
  const [userMenu, setUserMenu] = useState(false)
  const [lang] = useLang()

  const removeThread = async (id: number) => {
    await api.del(`/api/threads/${id}`)
    await refreshThreads()
    if (loc.pathname.endsWith(`/c/${id}`)) navigate(`/p/${pid}`)
  }

  return (
    <aside className="sidebar" onClick={(e) => (e.target as HTMLElement).closest('a') && onNavigate()}>
      <div className="brand">
        <Link to="/" className="logo">
          Nesi<span>code</span>
        </Link>
      </div>

      <div className="project-switch">
        <select
          value={project?.id ?? ''}
          onChange={(e) => {
            if (e.target.value === 'new') navigate('/new')
            else navigate(`/p/${e.target.value}`)
            onNavigate()
          }}
          aria-label={t('Proyecto')}
        >
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
          <option value="new">{t('+ Nuevo proyecto…')}</option>
        </select>
      </div>

      {project && (
        <>
          <Link to={`/p/${project.id}`} className="new-chat">
            <Icon name="plus" /> {t('Nueva conversación')}
          </Link>

          <nav className="nav">
            <NavLink to={`/p/${project.id}/harness`}><Icon name="harness" /> Harness <span className="pill">v{project.harness_version}</span></NavLink>
            <NavLink to={`/p/${project.id}/harnessing`}><Icon name="loop" /> Harnessing</NavLink>
            <NavLink to={`/p/${project.id}/lessons`}><Icon name="book" /> {t('Lecciones')}</NavLink>
            <NavLink to={`/p/${project.id}/settings`}><Icon name="gear" /> {t('Ajustes del proyecto')}</NavLink>
          </nav>

          <div className="threads-head">{t('Conversaciones')}</div>
          <div className="threads">
            {own.length === 0 && <div className="muted small pad">{t('Aún no hay conversaciones.')}</div>}
            {own.map((th) => (
              <div key={th.id} className="thread-row">
                <NavLink to={`/p/${project.id}/c/${th.id}`} title={th.title}>
                  <span className="t-title">{th.title}</span>
                  <span className="t-time">{relTime(th.updated_at)}</span>
                </NavLink>
                <button className="icon-btn ghost" title={t('Eliminar')} onClick={() => removeThread(th.id)}>
                  <Icon name="trash" size={15} />
                </button>
              </div>
            ))}
          </div>
        </>
      )}

      <nav className="nav nav-global">
        <NavLink to="/connectors"><Icon name="plug" /> {t('Conectores (MCP)')}</NavLink>
        <NavLink to="/providers"><Icon name="key" /> {t('Proveedores')}</NavLink>
      </nav>

      <div className="user">
        <button className="user-btn" onClick={() => setUserMenu((v) => !v)}>
          <span className="avatar">{user.name.slice(0, 1).toUpperCase()}</span>
          <span className="user-name">{user.name}</span>
        </button>
        {userMenu && (
          <div className="user-menu" onMouseLeave={() => setUserMenu(false)}>
            <div className="muted small">{user.email}</div>
            <Link to="/account" onClick={() => setUserMenu(false)}>{t('Cuenta y claves de API')}</Link>
            <button style={{ color: 'inherit' }} title={t('Idioma')} onClick={() => switchLang(lang === 'es' ? 'en' : 'es')}>{lang === 'es' ? 'English' : 'Español'}</button>
            {user.auth !== false && <button onClick={logout}>{t('Cerrar sesión')}</button>}
          </div>
        )}
      </div>
    </aside>
  )
}
