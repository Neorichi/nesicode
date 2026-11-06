import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { Navigate, Route, Routes, useNavigate } from 'react-router-dom'
import { api, ApiError, type Project, type Thread, type User } from './api'
import { getLang, setLang, t, useLang } from './i18n'
import Sidebar from './components/Sidebar'
import ImageViewer from './components/ImageViewer'
import Login from './pages/Login'
import NewChat from './pages/NewChat'
import Chat from './pages/Chat'
import HarnessPage from './pages/Harness'
import Harnessing from './pages/Harnessing'
import Lessons from './pages/Lessons'
import ProjectSettings from './pages/ProjectSettings'
import Account from './pages/Account'
import NewProject from './pages/NewProject'
import { ConnectorsPage, ProvidersPage } from './pages/Integrations'
import BotsPage, { BotDetail } from './pages/Bots'

type Ctx = {
  user: User
  projects: Project[]
  threads: Thread[]
  refreshProjects: () => Promise<Project[]>
  refreshThreads: () => Promise<void>
  logout: () => Promise<void>
}

const AppCtx = createContext<Ctx | null>(null)

/** Apply the language saved in the user's server-side settings, if any. */
function loadServerLang() {
  api
    .get<Record<string, any>>('/api/me/settings')
    .then((s) => {
      const l = s?.language
      if ((l === 'es' || l === 'en') && l !== getLang()) setLang(l)
    })
    .catch(() => {})
}
export const useApp = () => useContext(AppCtx)!

function Home() {
  const { projects } = useApp()
  if (projects.length === 0) return <Navigate to="/new" replace />
  const last = Number(localStorage.getItem('nesi:project')) || projects[0].id
  const p = projects.find((x) => x.id === last) ?? projects[0]
  return <Navigate to={`/p/${p.id}`} replace />
}

export default function App() {
  const [user, setUser] = useState<User | null | undefined>(undefined)
  const [projects, setProjects] = useState<Project[]>([])
  const [threads, setThreads] = useState<Thread[]>([])
  const [menuOpen, setMenuOpen] = useState(false)
  const navigate = useNavigate()
  useLang()

  const refreshProjects = useCallback(async () => {
    const ps = await api.get<Project[]>('/api/projects')
    setProjects(ps)
    return ps
  }, [])
  const refreshThreads = useCallback(async () => {
    setThreads(await api.get<Thread[]>('/api/threads'))
  }, [])

  useEffect(() => {
    api
      .get<User>('/api/auth/me')
      .then(async (u) => {
        loadServerLang()
        await Promise.all([refreshProjects(), refreshThreads()])
        setUser(u)
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) setUser(null)
        else setUser(null)
      })
  }, [refreshProjects, refreshThreads])

  useEffect(() => {
    if (!user) return
    // Conversations can also be created from Telegram: keep the sidebar fresh.
    const id = setInterval(() => refreshThreads().catch(() => {}), 5000)
    return () => clearInterval(id)
  }, [user, refreshThreads])

  const onLogin = async (u: User) => {
    loadServerLang()
    await Promise.all([refreshProjects(), refreshThreads()])
    setUser(u)
    navigate('/')
  }

  const logout = async () => {
    await api.post('/api/auth/logout')
    setUser(null)
    setProjects([])
    setThreads([])
    navigate('/')
  }

  if (user === undefined) return <div className="boot">Nesicode</div>
  if (user === null) return <Login onLogin={onLogin} />

  return (
    <AppCtx.Provider value={{ user, projects, threads, refreshProjects, refreshThreads, logout }}>
      <div className={`shell ${menuOpen ? 'menu-open' : ''}`}>
        <Sidebar onNavigate={() => setMenuOpen(false)} />
        <div className="scrim" onClick={() => setMenuOpen(false)} />
        <ImageViewer />
        <main className="main">
          <button className="menu-btn" aria-label={t('Abrir menú')} onClick={() => setMenuOpen(true)}>
            <svg width="20" height="20" viewBox="0 0 20 20"><path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
          </button>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/new" element={<NewProject />} />
            <Route path="/account" element={<Account />} />
            <Route path="/bots" element={<BotsPage />} />
            <Route path="/bots/:bid" element={<BotDetail />} />
            <Route path="/connectors" element={<ConnectorsPage />} />
            <Route path="/providers" element={<ProvidersPage />} />
            <Route path="/p/:pid" element={<NewChat />} />
            <Route path="/p/:pid/c/:tid" element={<Chat />} />
            <Route path="/p/:pid/harness" element={<HarnessPage />} />
            <Route path="/p/:pid/harnessing" element={<Harnessing />} />
            <Route path="/p/:pid/harnessing/:rid" element={<Harnessing />} />
            <Route path="/p/:pid/lessons" element={<Lessons />} />
            <Route path="/p/:pid/settings" element={<ProjectSettings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </AppCtx.Provider>
  )
}

export function useProject(pid?: string): Project | undefined {
  const { projects } = useApp()
  const id = Number(pid)
  useEffect(() => {
    if (id) localStorage.setItem('nesi:project', String(id))
  }, [id])
  return projects.find((p) => p.id === id)
}
