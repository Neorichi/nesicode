import { t, useLang } from '../i18n'
import { conflictWhy, skippedPath, type ApplyResult, type Conflict } from '../logic'

/**
 * What an Apply left out, path by path: harnessing conflicts (the project's file was kept; the proposed version may
 * be next to it) and paths a conversation's Apply skipped. Nothing when everything was applied.
 */
export default function ApplyReport({ result, title }: { result: ApplyResult | null | undefined; title?: string }) {
  useLang()
  const conflicts: Conflict[] = Array.isArray(result?.conflicts) ? result!.conflicts! : []
  const skipped = (Array.isArray(result?.skipped) ? result!.skipped! : []).map(skippedPath)
  if (!conflicts.length && !skipped.length) return null
  return (
    <div className="apply-report">
      {title && <div className="small"><b>{title}</b></div>}
      {conflicts.length > 0 && (
        <>
          <div className="warn small">
            {t('{n} cambio(s) sin aplicar: el archivo del proyecto se dejó como estaba.', { n: conflicts.length })}
          </div>
          <ul className="apply-list small">
            {conflicts.map((c, i) => (
              <li key={`${c?.path}:${i}`}>
                <code>{c?.path}</code> — {conflictWhy(c?.reason)}
                {c?.side_file && <> · {t('la versión propuesta está en')} <code>{c.side_file}</code></>}
              </li>
            ))}
          </ul>
        </>
      )}
      {skipped.length > 0 && (
        <>
          <div className="warn small">{t('{n} ruta(s) sin aplicar:', { n: skipped.length })}</div>
          <ul className="apply-list small">
            {skipped.map((s, i) => (
              <li key={`${s.path}:${i}`}>
                <code>{s.path}</code>{s.reason && <> — {s.reason}</>}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  )
}
