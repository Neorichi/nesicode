import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import { t, useLang } from '../i18n'
import { downloadFile } from './ImageViewer'
import { fileManager } from '../api'

type Field = { key: string; label: string; secret?: boolean; help?: string; default?: string }
type CatalogItem = {
  id: string
  label: string
  icon: string
  description: string
  kind: 'stdio' | 'http' | 'builtin'
  local_only?: boolean
  url?: string
  fields: Field[]
  missing: string[]
  allowed: boolean
  added: boolean
}
type Connector = {
  name: string
  label: string
  template: string
  kind: 'stdio' | 'http' | 'builtin'
  local_only?: boolean
  command: string
  url: string
  enabled: boolean
  tools: string[]
  status: string
}
type Resp = { catalog: CatalogItem[]; connectors: Connector[]; local_allowed: boolean }

const ICON: Record<string, string> = { chrome: '◉', browser: '◎', github: '⌥', word: 'W', office: 'O', microsoft: '⊞', burp: 'B', custom: '⚙' }

function Card({ item, onAdded }: { item: CatalogItem; onAdded: (msg: string) => void }) {
  const [open, setOpen] = useState(false)
  const [values, setValues] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const add = async () => {
    setBusy(true)
    setError('')
    try {
      const r = await api.post<{ tools: string[] }>('/api/connectors', { template: item.id, values: Object.fromEntries(item.fields.map((f) => [f.key, values[f.key] ?? f.default ?? ''])) })
      setOpen(false)
      onAdded(t('{label} conectado · {n} herramientas', { label: item.label, n: r.tools.length }))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const blocked = !item.allowed ? t('Solo en la app de escritorio / modo local') : item.missing.length ? t('Falta instalar: {x}', { x: item.missing.join(', ') }) : ''

  return (
    <div className={`mcp-card ${item.added ? 'added' : ''}`}>
      <div className="mcp-card-top">
        <span className={`mcp-icon icon-${item.icon}`}>{ICON[item.icon] ?? '⚙'}</span>
        <div>
          <b>{item.label}</b>
          <div className="muted small">{item.kind === 'stdio' || item.kind === 'builtin' || item.local_only ? t('Local') : t('Remoto')}</div>
        </div>
      </div>
      <p className="small">{t(item.description)}</p>
      {item.added ? (
        <div className="ok small">✓ {t('Añadido')}</div>
      ) : blocked ? (
        <div className="warn small">{blocked}</div>
      ) : open ? (
        <div className="mcp-form">
          {item.fields.map((f) => (
            <label key={f.key}>
              {t(f.label)}
              <input
                type={f.secret ? 'password' : 'text'}
                autoComplete="off"
                value={values[f.key] ?? f.default ?? ''}
                onChange={(e) => setValues((v) => ({ ...v, [f.key]: e.target.value }))}
              />
              {f.help && <span className="muted small">{t(f.help)}</span>}
            </label>
          ))}
          {error && <div className="error-box">{error}</div>}
          <div className="row">
            <button className="btn primary small" disabled={busy} onClick={add}>
              {busy ? t('Conectando… (la primera vez puede tardar)') : t('Conectar')}
            </button>
            <button className="btn ghost small" onClick={() => setOpen(false)}>{t('Cancelar')}</button>
          </div>
        </div>
      ) : (
        <>
          {error && <div className="error-box">{error}</div>}
          <button className="btn primary small" disabled={busy} onClick={() => (item.fields.length ? setOpen(true) : add())}>
            {busy ? t('Conectando… (la primera vez puede tardar)') : `+ ${t('Añadir')}`}
          </button>
        </>
      )}
    </div>
  )
}

function CustomForm({ localAllowed, onAdded, onCancel }: { localAllowed: boolean; onAdded: (m: string) => void; onCancel: () => void }) {
  const [kind, setKind] = useState<'stdio' | 'http'>(localAllowed ? 'stdio' : 'http')
  const [label, setLabel] = useState('')
  const [command, setCommand] = useState('')
  const [url, setUrl] = useState('')
  const [token, setToken] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const add = async () => {
    setBusy(true)
    setError('')
    try {
      const r = await api.post<{ tools: string[] }>('/api/connectors/custom', { label, kind, command, url, token })
      onAdded(t('{label} conectado · {n} herramientas', { label, n: r.tools.length }))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="prov-form">
      <div className="seg">
        <button className={kind === 'stdio' ? 'on' : ''} disabled={!localAllowed} onClick={() => setKind('stdio')}>{t('Local (comando)')}</button>
        <button className={kind === 'http' ? 'on' : ''} onClick={() => setKind('http')}>{t('Remoto (URL)')}</button>
      </div>
      <label>{t('Nombre')}<input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Mi servidor MCP" /></label>
      {kind === 'stdio' ? (
        <label>
          {t('Comando')}
          <input value={command} onChange={(e) => setCommand(e.target.value)} placeholder="npx -y @modelcontextprotocol/server-everything" spellCheck={false} />
        </label>
      ) : (
        <>
          <label>URL<input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://ejemplo.com/mcp" spellCheck={false} /></label>
          <label>{t('Token (opcional)')}<input type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} /></label>
        </>
      )}
      {error && <div className="error-box">{error}</div>}
      <div className="row">
        <button className="btn primary small" disabled={busy || !label || (kind === 'stdio' ? !command : !url)} onClick={add}>
          {busy ? t('Conectando… (la primera vez puede tardar)') : t('Conectar')}
        </button>
        <button className="btn ghost small" onClick={onCancel}>{t('Cancelar')}</button>
      </div>
    </div>
  )
}

type BrowserStatus = { connected: boolean; paired: boolean; version?: string; browser?: string; extension_dir: string; local_mode: boolean }

function YourChrome() {
  const [st, setSt] = useState<BrowserStatus | null>(null)
  const [code, setCode] = useState('')
  const [path, setPath] = useState('')
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState('')
  const load = useCallback(async () => setSt(await api.get<BrowserStatus>('/api/browser')), [])
  useEffect(() => {
    load()
    const id = setInterval(load, 3000)
    return () => clearInterval(id)
  }, [load])
  if (!st) return null
  const install = async () => {
    setError('')
    try {
      const p = (await api.post<{ path: string }>('/api/browser/extension/install')).path
      setPath(p)
      await window.pywebview?.api?.reveal?.(p)
    } catch (e) {
      setError((e as Error).message)
    }
  }
  const newCode = async () => {
    setCopied(false)
    setCode((await api.post<{ code: string }>('/api/browser/pairing')).code)
  }
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
    } catch {
      /* select it by hand */
    }
  }
  return (
    <div className="your-chrome">
      <div className="mcp-card-top">
        <span className="mcp-icon icon-chrome">◉</span>
        <div style={{ flex: 1 }}>
          <b>{t('Tu Chrome (extensión)')}</b>
          <div className="muted small">{t('Usa tu Chrome de siempre, con tus sesiones abiertas: solo las pestañas que abre Nesicode (grupo «Nesicode») o las que compartas tú.')}</div>
        </div>
        <span className={`chrome-state ${st.connected ? 'on' : ''}`}>
          <span className={`mcp-dot ${st.connected ? 'running' : 'off'}`} /> {st.connected ? t('Conectado') : t('No conectado')}
        </span>
      </div>
      {!st.connected && (
        <ol className="steps">
          <li>
            <div className="row wrap">
              <button className="btn small primary" onClick={() => downloadFile('/api/browser/extension.zip', 'nesicode-chrome.zip')}>
                ⬇ {t('Descargar la extensión (.zip)')}
              </button>
              {st.local_mode && (
                <button className="btn small" onClick={install}>{t('Instalar en Documentos y mostrar en {fm}', { fm: fileManager() })}</button>
              )}
            </div>
            {path ? (
              <div className="small">{t('Lista en')} <code>{path}</code></div>
            ) : (
              <div className="muted small">{t('Si descargas el .zip, ábrelo para descomprimirlo: Chrome necesita la carpeta.')}</div>
            )}
          </li>
          <li>
            {t('En Chrome abre')} <code>chrome://extensions</code>, {t('activa «Modo de desarrollador» (arriba a la derecha) y pulsa «Cargar descomprimida»; elige esa carpeta.')}
          </li>
          <li>
            <button className="btn small primary" onClick={newCode}>{t('Generar código de conexión')}</button>{' '}
            <span className="muted small">{t('y pégalo en la extensión (icono de Nesicode en la barra de Chrome).')}</span>
            {code && (
              <div className="pair-code">
                <code>{code}</code>
                <button className="btn small" onClick={copy}>{copied ? t('Copiado') : t('Copiar')}</button>
              </div>
            )}
          </li>
        </ol>
      )}
      {st.connected && (
        <div className="small muted">
          {t('El agente ya puede usar tu Chrome. Para que vea una pestaña tuya, pulsa el icono de Nesicode en Chrome → «Compartir».')}{' '}
          <button className="link small" onClick={newCode}>{t('Generar código nuevo')}</button>{' · '}
          <button className="link small" onClick={() => downloadFile('/api/browser/extension.zip', 'nesicode-chrome.zip')}>{t('Descargar la extensión (.zip)')}</button>{' · '}
          {st.local_mode && <button className="link small" onClick={install}>{t('Actualizar la carpeta de la extensión')}</button>}
          {code && (
            <div className="pair-code">
              <code>{code}</code>
              <button className="btn small" onClick={copy}>{copied ? t('Copiado') : t('Copiar')}</button>
            </div>
          )}
        </div>
      )}
      {error && <div className="error-box">{error}</div>}
    </div>
  )
}

type TgStatus = { configured: boolean; bot: string; running: boolean; error: string; linked: boolean; chat_name: string }

function Telegram() {
  const [st, setSt] = useState<TgStatus | null>(null)
  const [token, setToken] = useState('')
  const [pair, setPair] = useState<{ link: string; command: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const load = useCallback(async () => setSt(await api.get<TgStatus>('/api/telegram')), [])
  useEffect(() => {
    load()
    const id = setInterval(load, 3000)
    return () => clearInterval(id)
  }, [load])
  if (!st) return null
  const save = async () => {
    setBusy(true)
    setError('')
    try {
      setSt(await api.put<TgStatus>('/api/telegram', { token }))
      setToken('')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const newPair = async () => {
    setError('')
    try {
      setPair(await api.post<{ link: string; command: string }>('/api/telegram/pair'))
    } catch (e) {
      setError((e as Error).message)
    }
  }
  const openLink = (url: string) => {
    const ext = window.pywebview?.api?.open_external
    if (ext) ext(url)
    else window.open(url, '_blank')
  }
  return (
    <div className="your-chrome telegram">
      <div className="mcp-card-top">
        <span className="mcp-icon icon-telegram">✈</span>
        <div style={{ flex: 1 }}>
          <b>Telegram</b>
          <div className="muted small">{t('Gestiona tus conversaciones desde Telegram: ver, anclarte, crear nuevas, recibir respuestas y aprobar permisos.')}</div>
        </div>
        <span className={`chrome-state ${st.linked ? 'on' : ''}`}>
          <span className={`mcp-dot ${st.linked && st.running ? 'running' : st.configured ? 'error' : 'off'}`} />{' '}
          {st.linked ? `${t('Vinculado')} · @${st.bot}` : st.configured ? `@${st.bot}` : t('No configurado')}
        </span>
      </div>
      {!st.configured && (
        <ol className="steps">
          <li>
            {t('En Telegram abre')}{' '}
            <button className="link" onClick={() => openLink('https://t.me/BotFather')}>@BotFather</button>
            {t(', envía /newbot y elige un nombre. Te dará un token (123456:ABC…).')}
          </li>
          <li>
            <div className="pair-code">
              <input type="password" autoComplete="off" placeholder={t('Pega aquí el token del bot')} value={token} onChange={(e) => setToken(e.target.value)} />
              <button className="btn small primary" disabled={busy || token.length < 20} onClick={save}>{busy ? '…' : t('Guardar')}</button>
            </div>
          </li>
        </ol>
      )}
      {st.configured && !st.linked && (
        <ol className="steps">
          <li>
            <button className="btn small primary" onClick={newPair}>{t('Vincular mi Telegram')}</button>
            {pair && (
              <div className="small" style={{ marginTop: 8 }}>
                <button className="btn small" onClick={() => openLink(pair.link)}>{t('Abrir en Telegram')}</button>{' '}
                {t('o envía al bot:')} <code>{pair.command}</code> <span className="muted">({t('caduca en 15 min')})</span>
              </div>
            )}
          </li>
        </ol>
      )}
      {st.linked && (
        <div className="small muted">
          {t('Vinculado con {name}. En Telegram usa /chats para ver tus conversaciones y anclarte a una, /new para empezar otra y /help para el resto. En cada conversación de la app tienes un botón 📌 Telegram para anclarla desde aquí.', { name: st.chat_name })}{' '}
          <button className="link small" onClick={async () => { await api.post('/api/telegram/unlink'); load() }}>{t('Desvincular')}</button>
        </div>
      )}
      {st.configured && (
        <div className="small">
          <button className="link small" onClick={async () => { if (confirm(t('¿Quitar el bot de Telegram?'))) { await api.del('/api/telegram'); load() } }}>{t('Quitar el bot')}</button>
          {st.error && <span className="warn"> · {st.error}</span>}
        </div>
      )}
      {error && <div className="error-box">{error}</div>}
    </div>
  )
}

export default function ConnectorsSection() {
  useLang()
  const [data, setData] = useState<Resp | null>(null)
  const [msg, setMsg] = useState('')
  const [custom, setCustom] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const load = useCallback(async () => setData(await api.get<Resp>('/api/connectors')), [])
  useEffect(() => {
    load()
  }, [load])

  const done = (m: string) => {
    setMsg(m)
    setCustom(false)
    load()
  }

  if (!data) return <div className="muted">{t('Cargando…')}</div>
  return (
    <div className="mcp">
      <YourChrome />
      <Telegram />
      {msg && <div className="ok-box">{msg}</div>}

      {data.connectors.length > 0 && (
        <div className="mcp-mine">
          {data.connectors.map((c) => (
            <div key={c.name} className="prov">
              <div className="prov-main">
                <div>
                  <span className={`mcp-dot ${c.status}`} /> <b>{c.label}</b>{' '}
                  <span className="muted small">· {t('{n} herramientas', { n: c.tools.length })}</span>
                </div>
                <div className="small muted clamp1">{c.kind === 'stdio' ? c.command : c.url}</div>
              </div>
              <div className="row">
                <label className="check small">
                  <input
                    type="checkbox"
                    checked={c.enabled}
                    onChange={async (e) => {
                      await api.patch(`/api/connectors/${c.name}`, { enabled: e.target.checked })
                      load()
                    }}
                  />{' '}
                  {t('Activo')}
                </label>
                <button
                  className="btn small"
                  disabled={busy === c.name}
                  onClick={async () => {
                    setBusy(c.name)
                    setMsg('')
                    try {
                      const r = await api.post<{ tools: string[] }>(`/api/connectors/${c.name}/test`)
                      setMsg(t('{label} conectado · {n} herramientas', { label: c.label, n: r.tools.length }))
                    } catch (e) {
                      setMsg((e as Error).message)
                    } finally {
                      setBusy(null)
                      load()
                    }
                  }}
                >
                  {busy === c.name ? '…' : t('Probar')}
                </button>
                <button
                  className="btn small ghost"
                  onClick={async () => {
                    if (!confirm(t('¿Quitar el conector {label}?', { label: c.label }))) return
                    await api.del(`/api/connectors/${c.name}`)
                    load()
                  }}
                >
                  {t('Quitar')}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mcp-grid">
        {data.catalog.map((item) => (
          <Card key={item.id} item={item} onAdded={done} />
        ))}
        <div className="mcp-card">
          <div className="mcp-card-top">
            <span className="mcp-icon icon-custom">{ICON.custom}</span>
            <div>
              <b>{t('Otro servidor MCP')}</b>
              <div className="muted small">{t('Local o remoto')}</div>
            </div>
          </div>
          <p className="small">{t('Cualquier servidor MCP: pega su comando o su URL.')}</p>
          {!custom && <button className="btn small" onClick={() => setCustom(true)}>+ {t('Añadir')}</button>}
        </div>
      </div>
      {custom && <CustomForm localAllowed={data.local_allowed} onAdded={done} onCancel={() => setCustom(false)} />}
    </div>
  )
}
