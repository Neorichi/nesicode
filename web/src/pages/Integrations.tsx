import ConnectorsSection from '../components/Connectors'
import ProvidersSection from '../components/Providers'
import { useApp } from '../App'
import { t, useLang } from '../i18n'

/** Account-wide pages, reachable from the sidebar even before any project exists. */
export function ConnectorsPage() {
  useLang()
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{t('Conectores (MCP)')}</h1>
          <p className="muted">
            {t('Herramientas externas que el agente puede usar en todas tus conversaciones: tu Chrome, GitHub, Word… Los locales se ejecutan en tu equipo, fuera de la jaula; en los modos «pedir» cada uso te pide permiso.')}
          </p>
        </div>
      </header>
      <ConnectorsSection />
    </div>
  )
}

export function ProvidersPage() {
  useLang()
  const { refreshProjects } = useApp()
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{t('Proveedores')}</h1>
          <p className="muted">
            {t('Tus proveedores de modelos (endpoint + API key). Son de tu cuenta y sirven para todos tus proyectos. Las API keys se guardan en la bóveda cifrada y nunca se muestran.')}
          </p>
        </div>
      </header>
      <ProvidersSection onAssigned={() => refreshProjects()} />
    </div>
  )
}
