import { t } from '../i18n'

export default function Diff({ text, empty }: { text: string; empty?: string }) {
  if (!text?.trim()) return <div className="muted small">{empty ?? t('Sin cambios')}</div>
  return (
    <pre className="diff">
      {text.split('\n').map((line, i) => {
        let cls = ''
        if (line.startsWith('+++') || line.startsWith('---')) cls = 'd-file'
        else if (line.startsWith('@@')) cls = 'd-hunk'
        else if (line.startsWith('+')) cls = 'd-add'
        else if (line.startsWith('-')) cls = 'd-del'
        return (
          <span key={i} className={cls}>
            {line}
            {'\n'}
          </span>
        )
      })}
    </pre>
  )
}
