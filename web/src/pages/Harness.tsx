import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, ApiError, relTime, type HarnessFiles } from '../api'
import { useApp, useProject } from '../App'
import Diff from '../components/Diff'
import { t, translateError, useLang } from '../i18n'

type Version = { version: number; source: string; note: string; run_id: number | null; created_at: number; inventory: string }
type HarnessResp = { version: number; files: HarnessFiles; inventory: string; errors: string[] }

const GROUPS = (): { key: string; label: string; hint: string; match: (p: string) => boolean }[] => [
  { key: 'always', label: t('Siempre activo'), hint: t('Se envía en cada turno'), match: (p) => !p.includes('/') },
  { key: 'skills', label: 'Skills', hint: t('Se cargan bajo demanda'), match: (p) => p.startsWith('skills/') },
  { key: 'sub_agents', label: t('Subagentes'), hint: t('Contexto propio'), match: (p) => p.startsWith('sub_agents/') },
  { key: 'scripts', label: 'Scripts', hint: t('Ejecutables vía bash'), match: (p) => p.startsWith('scripts/') },
  { key: 'plugins', label: 'Plugins', hint: t('Hooks dentro del bucle'), match: (p) => p.startsWith('plugins/') },
]

const SOURCE_LABEL = (): Record<string, string> => ({
  base: 'base',
  manual: t('manual'),
  harnessing: 'Harness²',
  restore: t('restaurada'),
})

const TEMPLATES: Record<string, (name: string) => [string, string]> = {
  skill: (n) => [
    `skills/${n}/SKILL.md`,
    `---\nname: ${n}\ndescription: ${t('Qué hace este skill y cuándo usarlo.')}\n---\n1. ${t('Paso uno')}\n2. ${t('Paso dos')}\n`,
  ],
  subagent: (n) => [
    `sub_agents/${n}/AGENT.md`,
    `---\ndescription: ${t('Para qué sirve este subagente.')}\n---\nYou are a focused subagent. Do the task you are given and return a short answer.\n`,
  ],
  script: (n) => [`scripts/${n}.py`, `#!/usr/bin/env python3\n"""Describe what this helper does."""\n\nprint("ok")\n`],
}

export default function HarnessPage() {
  const { pid } = useParams()
  const project = useProject(pid)
  const { refreshProjects } = useApp()
  const [current, setCurrent] = useState<HarnessResp | null>(null)
  const [draft, setDraft] = useState<HarnessFiles>({})
  const [selected, setSelected] = useState('system_prompt.md')
  const [versions, setVersions] = useState<Version[]>([])
  const [viewing, setViewing] = useState<number | null>(null)
  const [viewData, setViewData] = useState<{ files: HarnessFiles; diff: string; note: string; source: string } | null>(null)
  const [errors, setErrors] = useState<string[]>([])
  const [note, setNote] = useState('')
  const [msg, setMsg] = useState('')
  const [showDiff, setShowDiff] = useState(false)
  useLang()

  const load = useCallback(async () => {
    const [h, v] = await Promise.all([
      api.get<HarnessResp>(`/api/projects/${pid}/harness`),
      api.get<Version[]>(`/api/projects/${pid}/harness/versions`),
    ])
    setCurrent(h)
    setDraft(h.files)
    setVersions(v)
    setErrors(h.errors)
  }, [pid])

  useEffect(() => {
    load()
    setViewing(null)
  }, [load])

  useEffect(() => {
    if (viewing === null) return setViewData(null)
    api.get<any>(`/api/projects/${pid}/harness/versions/${viewing}`).then(setViewData)
  }, [viewing, pid])

  const files = viewData ? viewData.files : draft
  const readOnly = viewData !== null
  const dirty = useMemo(() => current && JSON.stringify(current.files) !== JSON.stringify(draft), [current, draft])
  const paths = Object.keys(files).sort()

  const addFile = (kind: keyof typeof TEMPLATES) => {
    const name = prompt(t('Nombre (minúsculas, dígitos, - o _):'))?.trim()
    if (!name) return
    if (!/^[a-z0-9][a-z0-9_-]{0,47}$/.test(name)) return alert(t('Nombre no válido'))
    const [path, body] = TEMPLATES[kind](name)
    setDraft((d) => ({ ...d, [path]: body }))
    setSelected(path)
  }

  const removeFile = (path: string) => {
    if (path === 'system_prompt.md') return
    if (!confirm(t('¿Eliminar {path}?', { path }))) return
    setDraft((d) => {
      const n = { ...d }
      delete n[path]
      return n
    })
    setSelected('system_prompt.md')
  }

  const validateDraft = async () => {
    const r = await api.post<{ errors: string[] }>(`/api/projects/${pid}/harness/validate`, { files: draft })
    setErrors(r.errors)
    setMsg(r.errors.length ? '' : t('El harness es válido'))
  }

  const save = async () => {
    setMsg('')
    try {
      const r = await api.post<{ version: number; unchanged?: boolean }>(`/api/projects/${pid}/harness`, { files: draft, note })
      setNote('')
      setMsg(r.unchanged ? t('Sin cambios') : t('Guardada la versión {n}', { n: r.version }))
      await Promise.all([load(), refreshProjects()])
    } catch (e) {
      if (e instanceof ApiError && e.detail?.errors) setErrors(e.detail.errors)
      else setMsg((e as Error).message)
    }
  }

  const restore = async (v: number) => {
    await api.post(`/api/projects/${pid}/harness/restore/${v}`)
    setViewing(null)
    await Promise.all([load(), refreshProjects()])
    setMsg(t('Restaurada la versión {n}', { n: v }))
  }

  if (!project || !current) return <div className="page"><div className="muted">{t('Cargando…')}</div></div>

  return (
    <div className="page wide harness-page">
      <header className="page-head">
        <div>
          <h1>Harness</h1>
          <p className="muted">
            {t('Todo lo que rodea al modelo: prompt, reglas, memoria, skills, subagentes, scripts y hooks. Se versiona en cada cambio, manual o de')} <Link to={`/p/${pid}/harnessing`}>Harness²</Link>.
          </p>
        </div>
        <div className="head-actions">
          <select value={viewing ?? ''} onChange={(e) => setViewing(e.target.value ? Number(e.target.value) : null)}>
            <option value="">{t('Editar versión actual (v{n})', { n: current.version })}</option>
            {versions.map((v) => (
              <option key={v.version} value={v.version}>
                v{v.version} · {SOURCE_LABEL()[v.source] ?? v.source} · {relTime(v.created_at)}
              </option>
            ))}
          </select>
        </div>
      </header>

      {readOnly && viewData && (
        <div className="banner">
          <div>
            {t('Viendo')} <b>v{viewing}</b> ({SOURCE_LABEL()[viewData.source] ?? viewData.source}) — {translateError(viewData.note)}
          </div>
          <div className="row">
            <button className="btn" onClick={() => setShowDiff(true)}>{t('Ver cambios')}</button>
            {viewing !== current.version && (
              <button className="btn primary" onClick={() => restore(viewing!)}>{t('Restaurar esta versión')}</button>
            )}
          </div>
        </div>
      )}
      {readOnly && showDiff && viewData && (
        <div className="modal-bg" onClick={() => setShowDiff(false)}>
          <div className="modal wide-modal" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-head">
              <div>
                <b>{t('Cambios de la v{v}', { v: viewing ?? '' })}</b>{' '}
                <span className="muted small">{t('respecto a la versión anterior')}</span>
              </div>
              <button className="icon-btn" onClick={() => setShowDiff(false)} aria-label={t('Cerrar')}>✕</button>
            </div>
            <div className="drawer-body">
              <Diff text={viewData.diff} empty={t('Primera versión: no hay versión anterior con la que comparar.')} />
            </div>
          </div>
        </div>
      )}

      <div className="harness-grid">
        <div className="tree">
          {GROUPS().map((g) => {
            const items = paths.filter(g.match)
            return (
              <div key={g.key} className="tree-group">
                <div className="tree-label">
                  {g.label} <span className="muted small">· {g.hint}</span>
                </div>
                {items.map((p) => (
                  <button key={p} className={`tree-item ${selected === p ? 'active' : ''}`} onClick={() => setSelected(p)}>
                    <span>{p.replace(/^(skills|sub_agents)\//, '')}</span>
                    {!readOnly && current.files[p] !== draft[p] && <span className="dot-mod" title={t('Modificado')} />}
                  </button>
                ))}
                {!readOnly && g.key === 'skills' && <button className="tree-add" onClick={() => addFile('skill')}>+ skill</button>}
                {!readOnly && g.key === 'sub_agents' && <button className="tree-add" onClick={() => addFile('subagent')}>{t('+ subagente')}</button>}
                {!readOnly && g.key === 'scripts' && <button className="tree-add" onClick={() => addFile('script')}>+ script</button>}
                {!readOnly && g.key === 'plugins' && !draft['plugins/hooks.json'] && (
                  <button className="tree-add" onClick={() => { setDraft((d) => ({ ...d, 'plugins/hooks.json': '{\n  "pre_tool": [],\n  "post_tool": []\n}\n' })); setSelected('plugins/hooks.json') }}>+ hooks.json</button>
                )}
              </div>
            )
          })}
          <div className="muted small pad">{current.inventory}</div>
        </div>

        <div className="editor">
          <div className="editor-head">
            <code>{selected}</code>
            {!readOnly && selected !== 'system_prompt.md' && files[selected] !== undefined && (
              <button className="btn ghost small" onClick={() => removeFile(selected)}>{t('Eliminar')}</button>
            )}
          </div>
          <textarea
            className="editor-area"
            spellCheck={false}
            readOnly={readOnly}
            value={files[selected] ?? ''}
            placeholder={files[selected] === undefined ? t('Este fichero no existe en esta versión.') : t('(vacío)')}
            onChange={(e) => setDraft((d) => ({ ...d, [selected]: e.target.value }))}
          />
          {!readOnly && (
            <div className="editor-foot">
              <input placeholder={t('Nota de la versión (opcional)')} value={note} onChange={(e) => setNote(e.target.value)} />
              <button className="btn" onClick={validateDraft}>{t('Validar')}</button>
              <button className="btn" disabled={!dirty} onClick={() => { setDraft(current.files); setErrors(current.errors) }}>{t('Descartar')}</button>
              <button className="btn primary" disabled={!dirty} onClick={save}>{t('Guardar versión')}</button>
            </div>
          )}
          {msg && <div className="ok-box">{msg}</div>}
          {errors.length > 0 && (
            <div className="error-box">
              <b>{t('Problemas de validación')}</b>
              <ul>{errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
