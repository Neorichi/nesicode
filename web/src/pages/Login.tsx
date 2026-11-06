import { useState } from 'react'
import { api, type User } from '../api'
import { t, useLang } from '../i18n'

export default function Login({ onLogin }: { onLogin: (u: User) => void }) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  useLang()

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError('')
    try {
      const u = await api.post<User>(`/api/auth/${mode}`, { email, password, name })
      onLogin(u)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth">
      <div className="auth-side">
        <div className="logo big">
          Nesi<span>code</span>
        </div>
        <p className="auth-claim">
          {t('Un agente de código que')} <em>{t('mejora su propio harness')}</em> {t('en cada tarea: propone cambios, los prueba, combina lo que funciona y recuerda la lección.')}
        </p>
        <ol className="auth-steps">
          <li><b>Propose</b> {t('ediciones al prompt, skills, hooks…')}</li>
          <li><b>Probe</b> {t('cada variante sobre la misma tarea')}</li>
          <li><b>Compose</b> {t('el harness final a partir de las diferencias')}</li>
        </ol>
      </div>
      <form className="auth-card" onSubmit={submit}>
        <h1>{mode === 'login' ? t('Inicia sesión') : t('Crea tu cuenta')}</h1>
        {mode === 'register' && (
          <label>
            {t('Nombre')}
            <input value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" />
          </label>
        )}
        <label>
          {t('Email')}
          <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" />
        </label>
        <label>
          {t('Contraseña')}
          <input
            type="password" required minLength={8} value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
          />
        </label>
        {error && <div className="error-box">{error}</div>}
        <button className="btn primary wide" disabled={busy}>
          {busy ? '…' : mode === 'login' ? t('Entrar') : t('Crear cuenta')}
        </button>
        <button type="button" className="link" onClick={() => setMode(mode === 'login' ? 'register' : 'login')}>
          {mode === 'login' ? t('¿No tienes cuenta? Regístrate') : t('¿Ya tienes cuenta? Inicia sesión')}
        </button>
      </form>
    </div>
  )
}
