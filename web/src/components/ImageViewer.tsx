import { useEffect, useState } from 'react'
import { t } from '../i18n'
import { linkAction, linkClick } from '../logic'

declare global {
  interface Window {
    pywebview?: { api?: { pick_folder?: () => Promise<string | null>; save_file?: (path: string, name: string) => Promise<boolean>; open_external?: (url: string) => Promise<boolean>; reveal?: (path: string) => Promise<boolean> } }
  }
}

/** A link to another site (or an e-mail link): a new browser tab, or in the desktop app the user's browser or mail
 * app — never the app's own window. */
export function openExternal(url: string) {
  const native = window.pywebview?.api?.open_external
  if (native) {
    void native(url)
    return
  }
  if (!/^https?:/i.test(url)) {
    location.href = url // mailto: the browser hands it to the mail app and the page stays
    return
  }
  // a new-tab link: a browser opens a tab; a desktop window whose bridge is not ready yet sends it to the browser
  const a = document.createElement('a')
  a.href = url
  a.target = '_blank'
  a.rel = 'noopener noreferrer'
  a.click()
}

/**
 * Nothing replaces the app's page (the desktop window has no Back button to return): links to other sites open in a
 * new tab, or in the system browser in the desktop app (see linkClick). It listens in the capture phase, before any
 * handler that stops the click (the modals do), and leaves alone a button inside a link (the «load this image»
 * placeholder), which handles its own click. The desktop window also turns away any navigation that gets past this
 * (desktop.py).
 */
export function installExternalLinks() {
  document.addEventListener(
    'click',
    (e) => {
      const target = e.target instanceof Element ? e.target : null
      const a = target?.closest('a[href], area[href]')
      if (!(a instanceof HTMLAnchorElement || a instanceof HTMLAreaElement)) return
      const button = target?.closest('button')
      if (button && a.contains(button)) return
      const what = linkClick(linkAction(a.getAttribute('href') ?? '', location.href), {
        desktop: !!window.pywebview,
        download: a.hasAttribute('download'),
        modified: e.metaKey || e.ctrlKey || e.shiftKey || e.altKey,
      })
      if (what === 'follow') return
      e.preventDefault()
      if (what === 'external') openExternal(a.href)
    },
    true,
  )
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
