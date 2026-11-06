import { useEffect, useState } from 'react'
import { t } from '../i18n'

declare global {
  interface Window {
    pywebview?: { api?: { pick_folder?: () => Promise<string | null>; save_file?: (path: string, name: string) => Promise<boolean>; open_external?: (url: string) => Promise<boolean>; reveal?: (path: string) => Promise<boolean> } }
  }
}

/** In the desktop app, links to other sites open in the system browser instead of replacing the app. */
export function installExternalLinks() {
  document.addEventListener('click', (e) => {
    const open = window.pywebview?.api?.open_external
    const a = (e.target as HTMLElement)?.closest?.('a') as HTMLAnchorElement | null
    if (!open || !a || !a.href) return
    const url = new URL(a.href, location.href)
    if (url.origin !== location.origin && /^https?:$/.test(url.protocol)) {
      e.preventDefault()
      open(url.href)
    }
  })
}

/** Download that works both in the browser and inside the desktop app (native "Save as…"). */
export async function downloadFile(url: string, name: string) {
  const save = window.pywebview?.api?.save_file
  if (save) {
    await save(url, name)
    return
  }
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
}

let openViewer: ((src: string, name: string) => void) | null = null

export function showImage(src: string, name: string) {
  openViewer?.(src, name)
}

/** One global lightbox: images open over the chat instead of in a new tab (which breaks the desktop app). */
export default function ImageViewer() {
  const [img, setImg] = useState<{ src: string; name: string } | null>(null)
  useEffect(() => {
    openViewer = (src, name) => setImg({ src, name })
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setImg(null)
    window.addEventListener('keydown', onKey)
    return () => {
      openViewer = null
      window.removeEventListener('keydown', onKey)
    }
  }, [])
  if (!img) return null
  return (
    <div className="lightbox" onClick={() => setImg(null)}>
      <img src={img.src} alt={img.name} onClick={(e) => e.stopPropagation()} />
      <div className="lightbox-bar" onClick={(e) => e.stopPropagation()}>
        <code>{img.name}</code>
        <button className="btn small" onClick={() => downloadFile(img.src, img.name)}>{t('Descargar')}</button>
        <button className="btn small ghost" onClick={() => setImg(null)}>{t('Cerrar')}</button>
      </div>
    </div>
  )
}
