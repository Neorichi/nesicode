import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { Navigate, Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom'
import { api, ApiError, type Project, type Thread, type User } from './api'
import { useVisibleInterval } from './hooks'
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
  /** The project the sidebar shows (the one in the URL, else the last one opened). */
  pid: number | undefined
  /** That project's conversations (all of them), and the project they were loaded for. */
  threads: Thread[]
  threadsPid: number | undefined
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

/** The project the sidebar is about: the one in the URL, else the last one opened, else the first. */
function sidebarPid(pathname: string, projects: Project[]): number | undefined {
  const m = pathname.match(/^\/p\/(\d+)/)
  if (m) return Number(m[1])
  let last = 0
  try {
    last = Number(localStorage.getItem('nesi:project')) || 0
  } catch {
    /* storage unavailable */
  }
  return last || projects[0]?.id
}

// One element per conversation / project: a new id remounts the page, so nothing (drafts, attachments, a late
// response, a pending timer, a recording) can leak from the previous one into it.
function ChatRoute() {
  const { tid } = useParams()
  return <Chat key={tid} />
}

function NewChatRoute() {
  const { pid } = useParams()
  return <NewChat key={pid} />
}

function BotRoute() {
  const { bid } = useParams()
  return <BotDetail key={bid} />
}

export default function App() {
  const [user, setUser] = useState<User | null | undefined>(undefined)
  const [projects, setProjects] = useState<Project[]>([])
  const [threads, setThreads] = useState<{ pid: number | undefined; list: Thread[] }>({ pid: undefined, list: [] })
  const [menuOpen, setMenuOpen] = useState(false)
  const navigate = useNavigate()
  const loc = useLocation()
  useLang()
  const pid = sidebarPid(loc.pathname, projects)
  const pidRef = useRef(pid)
  pidRef.current = pid
  const threadsReq = useRef(0)

  const refreshProjects = useCallback(async () => {
    const ps = await api.get<Project[]>('/api/projects')
    setProjects(ps)
    return ps
  }, [])
  // The sidebar lists the current project's conversations (all of them: the global list is capped at 200 and a
  // busy project could hide every other one). Only the newest request may update it.
  const refreshThreads = useCallback(async () => {
    const p = pidRef.current
    const n = ++threadsReq.current
    if (!p) {
      setThreads({ pid: undefined, list: [] })
      return
    }
    try {
      const list = await api.get<Thread[]>(`/api/projects/${p}/threads`)
      if (n === threadsReq.current) setThreads({ pid: p, list })
    } catch (e) {
      if (n === threadsReq.current && e instanceof ApiError && e.status === 404) setThreads({ pid: p, list: [] })
    }
  }, [])

  useEffect(() => {
    api
      .get<User>('/api/auth/me')
      .then(async (u) => {
        loadServerLang()
        await refreshProjects()
        setUser(u)
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) setUser(null)
        else setUser(null)
      })
  }, [refreshProjects])

  // (re)load when the shown project changes
  useEffect(() => {
    if (user) refreshThreads()
  }, [user, pid, refreshThreads])

  // Conversations can also be created from Telegram: keep the sidebar fresh (not while the tab is hidden).
  useVisibleInterval(() => void refreshThreads(), 5000, !!user)

  const onLogin = async (u: User) => {
    loadServerLang()
    await refreshProjects()
    setUser(u)
    navigate('/')
  }

  const logout = async () => {
    await api.post('/api/auth/logout')
    setUser(null)
    setProjects([])
    setThreads({ pid: undefined, list: [] })
    navigate('/')
  }

  if (user === undefined) return <div className="boot">Nesicode</div>
  if (user === null) return <Login onLogin={onLogin} />

  return (
    <AppCtx.Provider value={{ user, projects, pid, threads: threads.list, threadsPid: threads.pid, refreshProjects, refreshThreads, logout }}>
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
            <Route path="/bots/:bid" element={<BotRoute />} />
            <Route path="/connectors" element={<ConnectorsPage />} />
            <Route path="/providers" element={<ProvidersPage />} />
            <Route path="/p/:pid" element={<NewChatRoute />} />
            <Route path="/p/:pid/c/:tid" element={<ChatRoute />} />
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
