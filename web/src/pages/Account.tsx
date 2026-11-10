import { useEffect, useState } from 'react'
import { api, switchLang } from '../api'
import { t, useLang } from '../i18n'
import { useApp } from '../App'
import { ConfigExport, ConfigImport } from '../components/ConfigTransfer'
import { clampInt } from '../logic'

/**
 * A number setting: what the user types stays in the field (an empty or half-typed value is fine) and is saved on
 * blur or Enter, clamped to [min, max]. Saving on every keystroke turned "20 → 50" into 205.
 */
function NumberSetting({ value, min, max, onCommit }: { value: number; min: number; max: number; onCommit: (n: number) => Promise<boolean> }) {
  const [text, setText] = useState(String(value ?? ''))
  useEffect(() => setText(String(value ?? '')), [value])
  const commit = async () => {
    const n = clampInt(text, min, max)
    if (n === null || n === value) return setText(String(value ?? ''))
    setText(String(n))
    if (!(await onCommit(n))) setText(String(value ?? ''))
  }
  return (
    <input type="number" min={min} max={max} value={text} onChange={(e) => setText(e.target.value)} onBlur={commit}
      onKeyDown={(e) => e.key === 'Enter' && (e.currentTarget as HTMLInputElement).blur()} />
  )
}

export default function Account() {
  const { user, projects } = useApp()
  const [lang] = useLang()
  const [prefs, setPrefs] = useState<Record<string, any> | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    api.get<Record<string, any>>('/api/me/settings').then(setPrefs).catch((e) => setError(e.message))
  }, [])
  const update = async (patch: Record<string, any>) => {
    setError('')
    try {
      setPrefs(await api.put<Record<string, any>>('/api/me/settings', patch))
      return true
    } catch (e) {
      setError((e as Error).message)
      return false
    }
  }
  const share = prefs ? !!prefs.share_workspace : null
  const toggleShare = (v: boolean) => update({ share_workspace: v })

  return (
    <div className="page narrow">
      <header className="page-head">
        <div>
          <h1>{t('Cuenta')}</h1>
          <p className="muted">{user.name} · {user.email}</p>
        </div>
      </header>
      {error && <div className="error-box">{error}</div>}
      <div className="card">
        <h3>{t('Idioma')}</h3>
        <div className="seg">
          <button className={lang === 'es' ? 'on' : ''} onClick={() => switchLang('es')}>Español</button>
          <button className={lang === 'en' ? 'on' : ''} onClick={() => switchLang('en')}>English</button>
        </div>
      </div>
      <div className="card">
        <h3>{t('Permisos de ejecución')}</h3>
        <p className="muted small">{t('Qué tiene que pedirte el agente antes de hacerlo. Es el modo por defecto de las conversaciones; cada una puede cambiarlo desde su cabecera. El harnessing siempre va en auto (en sus propias jaulas).')}</p>
        <div className="exec-modes">
          {[
            ['auto', t('Auto'), t('Ejecuta comandos y edita archivos sin preguntar (dentro de su jaula).')],
            ['ask_bash', t('Pedir para comandos'), t('Te pide permiso antes de cada comando de terminal.')],
            ['ask_all', t('Pedir para todo'), t('Te pide permiso antes de comandos, de escribir o editar archivos y de crear imágenes.')],
          ].map(([v, label, desc]) => (
            <label key={v} className={`exec-mode ${prefs?.exec_mode === v ? 'on' : ''}`}>
              <input type="radio" name="exec_mode" checked={prefs?.exec_mode === v} disabled={!prefs} onChange={() => update({ exec_mode: v })} />
              <div>
                <b>{label}</b>
                <div className="muted small">{desc}</div>
              </div>
            </label>
          ))}
        </div>
      </div>
      <div className="card">
        <h3>{t('Aislamiento')}</h3>
        <div className="layers small">
          <div><b>{t('1 · Usuario')}</b> — {t('siempre activo. Tus proyectos, conversaciones y claves son solo tuyos.')}</div>
          <div><b>{t('2 · Proyecto')}</b> — {t('cada proyecto tiene su carpeta base, su harness y su historial.')}</div>
          <div><b>{t('3 · Conversación')}</b> — {t('cada conversación trabaja en su propia jaula (una copia del proyecto). No ve otras conversaciones ni el resto del disco, y sus cambios solo llegan al proyecto cuando pulsas «Aplicar».')}</div>
        </div>
        <label className="check" style={{ marginTop: 12 }}>
          <input type="checkbox" checked={!!share} disabled={share === null} onChange={(e) => toggleShare(e.target.checked)} />
          {t('Compartir la carpeta del proyecto entre conversaciones (desactiva la capa 3 en las conversaciones nuevas)')}
        </label>
        {share && <div className="error-box small">{t('Las conversaciones nuevas escribirán directamente en el proyecto y verán lo que hayan hecho las demás.')}</div>}
      </div>
      <div className="card">
        <h3>{t('Auto-harnessing')}</h3>
        <p className="muted small">
          {t('Si una respuesta del chat sale mal, Nesicode lanza solo un harnessing (k=1) con esa misma tarea para que el agente aprenda. Gasta unas 4 ejecuciones del agente cada vez.')}
        </p>
        <label className="check">
          <input type="checkbox" checked={!!prefs?.auto_harness} disabled={!prefs} onChange={(e) => update({ auto_harness: e.target.checked })} />
          {t('Activar auto-harnessing')}
        </label>
        {prefs?.auto_harness && (
          <div className="auto-grid">
            <label>
              {t('Se lanza si usa al menos')}
              <NumberSetting min={3} max={300} value={prefs.auto_harness_min_steps} onCommit={(n) => update({ auto_harness_min_steps: n })} />
              {t('pasos')}
            </label>
            <label>
              {t('…o tiene al menos')}
              <NumberSetting min={1} max={50} value={prefs.auto_harness_min_errors} onCommit={(n) => update({ auto_harness_min_errors: n })} />
              {t('errores de herramientas')}
            </label>
            <label>
              {t('Máximo al día')}
              <NumberSetting min={0} max={50} value={prefs.auto_harness_daily} onCommit={(n) => update({ auto_harness_daily: n })} />
              {t('ejecuciones')}
            </label>
            <div className="muted small">{t('También se lanza si agota el límite de pasos. Nunca dos a la vez en el mismo proyecto.')} {t('0 = sin límite diario.')}</div>
          </div>
        )}
      </div>
      <div className="card">
        <h3>{t('Proveedores y API keys')}</h3>
        <p className="muted small">{t('Se gestionan en')} <b>{t('Ajustes del proyecto')} → {t('Proveedores')}</b>. {t('Son de tu cuenta y valen para todos tus proyectos; las keys van a la bóveda cifrada.')}</p>
      </div>
      <div className="card">
        <h3>{t('Exportar configuración')}</h3>
        <p className="muted small">{t('Guarda en un archivo tus proveedores (con sus API keys), conectores, preferencias y, si quieres, la plantilla de un proyecto, para llevarlos a otro equipo. El archivo va cifrado con la contraseña que elijas: sin ella no se puede leer.')}</p>
        <ConfigExport projects={projects} />
      </div>
      <div className="card">
        <h3>{t('Importar configuración')}</h3>
        <p className="muted small">{t('Carga un archivo .nesicode exportado desde otro equipo.')}</p>
        <ConfigImport onDone={() => api.get<Record<string, any>>('/api/me/settings').then(setPrefs)} />
      </div>
    </div>
  )
}
