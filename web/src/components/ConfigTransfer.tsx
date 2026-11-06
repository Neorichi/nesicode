import { useEffect, useRef, useState } from 'react'
import { api, type Project } from '../api'
import { t, useLang } from '../i18n'
import { downloadFile } from './ImageViewer'

type Summary = {
  providers: string[]; connectors: string[]; settings: string[]; project_template: string | null; telegram: boolean
  role_models?: Record<string, string>; projects_updated?: string[]
}

const ROLE_LABEL: Record<string, string> = {
  executor_model: 'Chat', improver_model: 'Mejora', vision_model: 'Visión', image_model: 'Crear imágenes', transcribe_model: 'Transcripción',
}

function describe(s: Summary): string {
  const parts = [`${s.providers.length} ${t('proveedores')}`, `${s.connectors.length} ${t('conectores')}`, t('preferencias')]
  const roles = Object.entries(s.role_models ?? {})
  if (roles.length) parts.push(`${t('modelos por función')} (${roles.map(([k, v]) => `${t(ROLE_LABEL[k] ?? k)}: ${v}`).join(', ')})`)
  if (s.projects_updated?.length) parts.push(`${t('aplicados a')} ${s.projects_updated.join(', ')}`)
  if (s.project_template) parts.push(`${t('plantilla de proyecto')} «${s.project_template}»`)
  if (s.telegram) parts.push(t('bot de Telegram'))
  return parts.join(' · ')
}

/** Export: encrypt providers + keys, connectors, preferences (and optionally a project template / Telegram bot). */
export function ConfigExport({ projects }: { projects: Project[] }) {
  useLang()
  const [password, setPassword] = useState('')
  const [repeat, setRepeat] = useState('')
  const [projectId, setProjectId] = useState<string>('')
  const [telegram, setTelegram] = useState(false)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null)

  const run = async () => {
    setMsg(null)
    if (password.length < 8) return setMsg({ ok: false, text: t('La contraseña debe tener al menos 8 caracteres.') })
    if (password !== repeat) return setMsg({ ok: false, text: t('Las contraseñas no coinciden.') })
    setBusy(true)
    try {
      const r = await api.post<{ url: string; filename: string; summary: Summary }>('/api/config/export', {
        password, project_id: projectId ? Number(projectId) : null, include_telegram: telegram,
      })
      await downloadFile(r.url, r.filename)
      setMsg({ ok: true, text: `${t('Exportado')}: ${describe(r.summary)}` })
      setPassword('')
      setRepeat('')
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="stack">
      <div className="row wrap">
        <input type="password" placeholder={t('Contraseña (mín. 8)')} value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="new-password" />
        <input type="password" placeholder={t('Repite la contraseña')} value={repeat} onChange={(e) => setRepeat(e.target.value)} autoComplete="new-password" />
      </div>
      <p className="muted small">{t('Los modelos por función (Chat, Mejora, Visión, Crear imágenes, Transcripción) se copian siempre: del proyecto que elijas abajo o, si no eliges, del último que usaste.')}</p>
      <label>
        {t('Incluir plantilla de proyecto (modelos por rol, comando de verificación y harness)')}
        <select value={projectId} onChange={(e) => setProjectId(e.target.value)}>
          <option value="">{t('No incluir')}</option>
          {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
      </label>
      <label className="check">
        <input type="checkbox" checked={telegram} onChange={(e) => setTelegram(e.target.checked)} />
        {t('Incluir el bot de Telegram (solo puede estar activo en un equipo a la vez)')}
      </label>
      <div className="row wrap">
        <button className="btn primary" disabled={busy || !password} onClick={run}>{busy ? t('Cifrando…') : t('Exportar configuración')}</button>
      </div>
      {msg && <div className={msg.ok ? 'ok-box' : 'error-box'}>{msg.text}</div>}
    </div>
  )
}

/** Import a .nesicode file (from another computer). Calls onDone after a successful import. */
export function ConfigImport({ onDone, compact }: { onDone?: (s: Summary) => void; compact?: boolean }) {
  useLang()
  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null)

  const run = async () => {
    if (!file) return
    setBusy(true)
    setMsg(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('password', password)
      const r = await api.post<{ imported: Summary }>('/api/config/import', fd)
      setMsg({ ok: true, text: `${t('Importado')}: ${describe(r.imported)}` })
      setPassword('')
      onDone?.(r.imported)
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="stack">
      <input ref={fileRef} type="file" accept=".nesicode,application/json,application/octet-stream" hidden
        onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      <div className="row wrap">
        <button type="button" className="btn" onClick={() => fileRef.current?.click()}>{file ? file.name : t('Elegir archivo .nesicode…')}</button>
        <input type="password" placeholder={t('Contraseña del archivo')} value={password} onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); run() } }} />
        <button type="button" className="btn primary" disabled={busy || !file || !password} onClick={run}>{busy ? t('Importando…') : t('Importar')}</button>
      </div>
      {!compact && <p className="muted small">{t('Los proveedores y conectores con el mismo nombre se sobrescriben; el resto se conserva. Las keys se vuelven a guardar en la bóveda cifrada de este equipo. Los modelos por función se aplican a tus proyectos y a los nuevos.')}</p>}
      {msg && <div className={msg.ok ? 'ok-box' : 'error-box'}>{msg.text}</div>}
    </div>
  )
}

/** Is there an imported project template to start new projects from? */
export function useProjectTemplate(): [{ from: string } | null, () => void] {
  const [tpl, setTpl] = useState<{ from: string } | null>(null)
  const load = () => { api.get<{ template: { from: string } | null }>('/api/config/template').then((r) => setTpl(r.template)).catch(() => {}) }
  useEffect(load, [])
  return [tpl, load]
}
