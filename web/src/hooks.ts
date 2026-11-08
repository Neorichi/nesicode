import { useEffect, useRef } from 'react'

/**
 * Call `fn` every `ms` while the tab is visible; pause while it is hidden and call it once when it comes back.
 * (A hidden tab polling the server costs requests and connections for nothing.)
 */
export function useVisibleInterval(fn: () => void, ms: number, enabled = true) {
  const cb = useRef(fn)
  cb.current = fn
  useEffect(() => {
    if (!enabled) return
    let id: number | undefined
    const start = () => {
      if (id === undefined) id = window.setInterval(() => cb.current(), ms)
    }
    const stop = () => {
      clearInterval(id)
      id = undefined
    }
    const onVisibility = () => {
      if (document.visibilityState === 'hidden') stop()
      else {
        cb.current()
        start()
      }
    }
    if (document.visibilityState !== 'hidden') start()
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      stop()
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [ms, enabled])
}
