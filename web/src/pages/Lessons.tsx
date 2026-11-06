import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, relTime, type Lesson } from '../api'
import { useProject } from '../App'
import { t, useLang } from '../i18n'

export default function Lessons() {
  const { pid } = useParams()
  const project = useProject(pid)
  const [lessons, setLessons] = useState<Lesson[]>([])
  const [all, setAll] = useState(false)
  useLang()

  const load = async () => {
    if (!project) return
    setLessons(await api.get<Lesson[]>(all ? '/api/lessons' : `/api/lessons?domain=${encodeURIComponent(project.domain)}`))
  }
  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project?.domain, all])

  if (!project) return null
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{t('Lecciones')}</h1>
          <p className="muted">
            {t('La biblioteca de experiencia: cada harnessing deja como mucho una lección transferible. Las del dominio')}{' '}
            <b>{project.domain}</b> {t('se dan como hipótesis a los proponentes de las siguientes tareas.')}
          </p>
        </div>
        <label className="check">
          <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} /> {t('Todos los dominios')}
        </label>
      </header>
      {lessons.length === 0 && <div className="empty">{t('Todavía no hay lecciones. Aparecen tras una ejecución de')} <Link to={`/p/${pid}/harnessing`}>harnessing</Link>.</div>}
      <div className="lessons">
        {lessons.map((l) => (
          <div key={l.id} className="card lesson">
            <div className="lesson-head">
              <span className="pill">{l.domain}</span>
              <span className="muted small">
                {relTime(l.created_at)}
                {l.run_id && l.project_id && <> · <Link to={`/p/${l.project_id}/harnessing/${l.run_id}`}>{t('ejecución #{n}', { n: l.run_id })}</Link></>}
              </span>
              <button className="btn ghost small" onClick={async () => { await api.del(`/api/lessons/${l.id}`); load() }}>{t('Olvidar')}</button>
            </div>
            <div><b>{t('Cuándo:')}</b> {l.cue}</div>
            <div><b>{t('Qué hacer:')}</b> {l.lesson}</div>
            <div className="muted small"><b>{t('Evidencia:')}</b> {l.evidence}</div>
            {l.confidence && <div className="muted small"><b>{t('Confianza:')}</b> {l.confidence}</div>}
          </div>
        ))}
      </div>
    </div>
  )
}
