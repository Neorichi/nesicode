import { useEffect, useRef, useState } from 'react'
import { api, type Project } from '../api'
import { t, useLang } from '../i18n'
import { previewSections, type ImportPreview } from '../logic'
import { downloadFile } from './ImageViewer'

type Summary = {
  providers: string[]; connectors: string[]; settings: string[]; project_template: string | null; telegram: boolean
  role_models?: Record<string, string>; projects_updated?: string[]; bots?: string[]; bots_created?: string[]; notes?: string[]
  skipped?: string[]
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
  const bots = s.bots_created?.length ? s.bots_created : s.bots ?? [] // (an import creates them once there is a project)
  if (bots.length) parts.push(t('bots: {names}', { names: bots.join(', ') }))
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

/** What the file brought that needs a look: notes (switched off, kept stricter, keys dropped…) and what was left out. */
function Warnings({ notes, skipped }: { notes?: string[]; skipped?: string[] }) {
  return (
    <>
      {!!notes?.length && (
        <div className="small">
          <b>{t('Ten en cuenta:')}</b>
          <ul className="warn-list">{notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
        </div>
      )}
      {!!skipped?.length && (
        <div className="small">
          <b>{t('No se importará:')}</b>
          <ul className="warn-list">{skipped.map((n, i) => <li key={i}>{n}</li>)}</ul>
        </div>
      )}
    </>
  )
}

/** The confirm step: everything the file would import here (where providers point, what connectors run…). */
export function ImportPreviewView({ p }: { p: ImportPreview }) {
  const sections = previewSections(p)
  return (
    <>
      {sections.length === 0 && <div className="muted small">{t('El archivo no trae nada que importar.')}</div>}
      {sections.map((sec) => (
        <div key={sec.title}>
          <h5>{sec.title}</h5>
          <ul>
            {sec.items.map((it, i) => (
              <li key={i} className={it.warn ? 'warn' : undefined}>
                <b>{it.label}</b>
                {it.code && <> <code>{it.code}</code></>}
                {it.note && <span className="muted"> · {it.note}</span>}
              </li>
            ))}
          </ul>
        </div>
      ))}
      <Warnings notes={p.notes} skipped={p.skipped} />
    </>
  )
}

/**
 * Import a .nesicode file (from another computer). Whoever made the file chose its contents, so nothing is applied
 * before the user has seen it: «Revisar» asks the server for a preview (form field preview=1, nothing is written)
 * and «Importar» applies it. Calls onDone after a successful import.
 */
export function ConfigImport({ onDone, compact }: { onDone?: (s: Summary) => void; compact?: boolean }) {
  useLang()
  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [preview, setPreview] = useState<ImportPreview | null>(null)
  const [msg, setMsg] = useState<{ ok: boolean; text: string; notes?: string[]; skipped?: string[] } | null>(null)

  const send = async (dry: boolean) => {
    const fd = new FormData()
    fd.append('file', file!)
    fd.append('password', password)
    if (dry) fd.append('preview', '1')
    return api.post<{ preview?: ImportPreview; imported?: Summary }>('/api/config/import', fd)
  }
  const imported = (s: Summary) => {
    setPreview(null)
    setPassword('')
    setMsg({ ok: true, text: `${t('Importado')}: ${describe(s)}`, notes: s.notes, skipped: s.skipped })
    onDone?.(s)
  }
  const review = async () => {
    if (!file || !password) return
    setBusy(true)
    setMsg(null)
    setPreview(null)
    try {
      const r = await send(true)
      if (r.preview) setPreview(r.preview)
      else if (r.imported) imported(r.imported) // a server without the preview step imported it already: say what it did
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message })
    } finally {
      setBusy(false)
    }
  }
  const confirmImport = async () => {
    if (!file || !preview) return
    setBusy(true)
    setMsg(null)
    try {
      const r = await send(false)
      if (r.imported) imported(r.imported)
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message })
    } finally {
      setBusy(false)
    }
  }
  // another file or password: whatever was previewed is no longer what would be imported
  const reset = () => {
    setPreview(null)
    setMsg(null)
  }

  return (
    <div className="stack">
      <input ref={fileRef} type="file" accept=".nesicode,application/json,application/octet-stream" hidden
        onChange={(e) => { setFile(e.target.files?.[0] ?? null); reset(); e.target.value = '' }} />
      <div className="row wrap">
        <button type="button" className="btn" disabled={busy} onClick={() => fileRef.current?.click()}>{file ? file.name : t('Elegir archivo .nesicode…')}</button>
        <input type="password" placeholder={t('Contraseña del archivo')} value={password} onChange={(e) => { setPassword(e.target.value); if (preview) reset() }}
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); if (!preview) review() } }} />
        {!preview && (
          <button type="button" className="btn primary" disabled={busy || !file || !password} onClick={review}>{busy ? t('Leyendo…') : t('Revisar')}</button>
        )}
      </div>
      {!compact && !preview && <p className="muted small">{t('Antes de importar verás lo que trae el archivo. Los proveedores y conectores con el mismo nombre se sobrescriben; el resto se conserva. Las keys se vuelven a guardar en la bóveda cifrada de este equipo. Los modelos por función se aplican a tus proyectos y a los nuevos.')}</p>}
      {preview && (
        <div className="import-preview">
          <b>{t('Esto es lo que se importará desde «{file}»:', { file: file?.name ?? '' })}</b>
          <ImportPreviewView p={preview} />
          <div className="muted small">{t('Lo que tenga el mismo nombre aquí se sobrescribe; el resto se conserva.')}</div>
          <div className="row wrap">
            <button type="button" className="btn primary" disabled={busy} onClick={confirmImport}>{busy ? t('Importando…') : t('Importar')}</button>
            <button type="button" className="btn ghost" disabled={busy} onClick={reset}>{t('Cancelar')}</button>
          </div>
        </div>
      )}
      {msg && (
        <div className={msg.ok ? 'ok-box' : 'error-box'}>
          {msg.text}
          <Warnings notes={msg.notes} skipped={msg.skipped} />
        </div>
      )}
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
