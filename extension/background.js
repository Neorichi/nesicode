// Nesicode for Chrome — background service worker.
// Keeps a WebSocket to the Nesicode server and runs the agent's commands on the tabs of the
// "Nesicode" tab group only: everything in the group is usable by the agent, nothing outside it.

const GROUP_TITLE = 'Nesicode'
let ws = null
let state = 'disconnected' // disconnected | connecting | connected | error
let lastError = ''
let retry = 1000
let pingTimer = null

// ---------------------------------------------------------------- storage helpers
async function config() {
  const c = await chrome.storage.local.get(['server', 'token', 'shared', 'agentTabs'])
  return { server: c.server || '', token: c.token || '', shared: c.shared || [], agentTabs: c.agentTabs || [] }
}
async function setList(key, list) {
  await chrome.storage.local.set({ [key]: [...new Set(list)] })
}
// The "Nesicode" tab group IS the permission: the agent sees exactly the tabs inside it.
async function nesiGroupIds() {
  const groups = await chrome.tabGroups.query({ title: GROUP_TITLE })
  return new Set(groups.map((g) => g.id))
}
async function allowedTabs() {
  const ids = await nesiGroupIds()
  if (!ids.size) return []
  const tabs = await chrome.tabs.query({})
  return tabs.filter((t) => ids.has(t.groupId))
}
async function assertAllowed(tabId) {
  const ok = (await allowedTabs()).some((t) => t.id === tabId)
  if (!ok) throw new Error('That tab is not in the “Nesicode” group: the user must drag it into the group so Nesicode can use it')
}

// ---------------------------------------------------------------- connection
function setState(s, err = '') {
  state = s
  lastError = err
  chrome.action.setBadgeText({ text: s === 'connected' ? '' : s === 'connecting' ? '…' : '!' })
  if (s === 'connected') updateBadge()
  chrome.action.setBadgeBackgroundColor({ color: s === 'error' ? '#b42318' : '#8b877f' })
  chrome.runtime.sendMessage({ type: 'state', state, error: lastError }).catch(() => {})
}

async function connect() {
  const { server, token } = await config()
  if (!server || !token) return setState('disconnected', 'No connection code')
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return
  setState('connecting')
  // Probe first: a refused fetch is quiet, a refused WebSocket fills chrome://extensions with errors.
  try {
    await fetch(server + '/api/health', { cache: 'no-store' })
  } catch {
    return scheduleReconnect('Nesicode is not running')
  }
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return
  const url = server.replace(/^http/, 'ws') + '/api/browser/ws?token=' + encodeURIComponent(token)
  try {
    ws = new WebSocket(url)
  } catch (e) {
    return scheduleReconnect(String(e))
  }
  ws.onopen = () => {
    retry = 1000
    ws.send(JSON.stringify({ type: 'hello', version: chrome.runtime.getManifest().version, browser: navigator.userAgent }))
    setState('connected')
    clearInterval(pingTimer)
    pingTimer = setInterval(() => ws && ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify({ type: 'ping' })), 20000)
  }
  ws.onmessage = async (ev) => {
    let msg
    try {
      msg = JSON.parse(ev.data)
    } catch {
      return
    }
    if (!msg.cmd) return
    try {
      const result = await run(msg.cmd, msg.args || {})
      ws.send(JSON.stringify({ id: msg.id, ok: true, result }))
    } catch (e) {
      ws.send(JSON.stringify({ id: msg.id, ok: false, error: String(e && e.message ? e.message : e) }))
    }
  }
  ws.onclose = (ev) => {
    clearInterval(pingTimer)
    ws = null
    if (ev.code === 4401) return setState('error', 'Invalid connection code: generate a new one in Nesicode')
    scheduleReconnect('Nesicode is not running')
  }
  ws.onerror = () => {}
}

async function updateBadge() {
  if (state !== 'connected') return
  const n = (await allowedTabs()).length
  chrome.action.setBadgeText({ text: n ? String(n) : '' })
  chrome.action.setBadgeBackgroundColor({ color: '#0e8f7e' })
}
chrome.tabs.onUpdated.addListener((_id, info) => info.groupId !== undefined && updateBadge())
chrome.tabGroups.onUpdated.addListener(() => updateBadge())
chrome.tabs.onRemoved.addListener(() => setTimeout(updateBadge, 100))

function scheduleReconnect(err) {
  setState('error', err)
  setTimeout(connect, retry)
  retry = Math.min(retry * 2, 10000)
}

chrome.runtime.onStartup.addListener(connect)
chrome.runtime.onInstalled.addListener(connect)
chrome.alarms.create('keepalive', { periodInMinutes: 0.5 })
chrome.alarms.onAlarm.addListener(() => connect())
chrome.tabs.onRemoved.addListener(async (tabId) => {
  const c = await config()
  await setList('agentTabs', c.agentTabs.filter((id) => id !== tabId))
})

// ---------------------------------------------------------------- popup API
chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  ;(async () => {
    if (msg.type === 'status') {
      const c = await config()
      const tabs = await allowedTabs()
      reply({ state, error: lastError, server: c.server, tabs: tabs.map((t) => ({ id: t.id, title: t.title, url: t.url, agent: c.agentTabs.includes(t.id) })) })
      return
    } else if (msg.type === 'pair') {
      const raw = String(msg.code || '').trim().replace(/^NESI-/, '')
      const data = JSON.parse(atob(raw.replace(/-/g, '+').replace(/_/g, '/')))
      if (!data.server || !data.token) throw new Error('Invalid code')
      await chrome.storage.local.set({ server: data.server, token: data.token })
      if (ws) ws.close()
      ws = null
      await connect()
      reply({ ok: true })
    } else if (msg.type === 'share') {
      if (msg.on) await addToGroup(msg.tabId)
      else await chrome.tabs.ungroup(msg.tabId)
      reply({ ok: true })
    } else if (msg.type === 'reconnect') {
      retry = 1000
      await connect()
      reply({ ok: true })
    } else if (msg.type === 'unpair') {
      await chrome.storage.local.remove(['server', 'token'])
      if (ws) ws.close()
      setState('disconnected')
      reply({ ok: true })
    }
  })().catch((e) => reply({ ok: false, error: String(e.message || e) }))
  return true
})

// ---------------------------------------------------------------- commands
async function waitLoaded(tabId, timeout = 30000) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeout) {
    const tab = await chrome.tabs.get(tabId)
    if (tab.status === 'complete') return tab
    await new Promise((r) => setTimeout(r, 250))
  }
  return chrome.tabs.get(tabId)
}

async function addToGroup(tabId) {
  const tab = await chrome.tabs.get(tabId)
  const groups = await chrome.tabGroups.query({ title: GROUP_TITLE, windowId: tab.windowId })
  const groupId = await chrome.tabs.group({ tabIds: tabId, ...(groups[0] ? { groupId: groups[0].id } : {}) })
  if (!groups[0]) await chrome.tabGroups.update(groupId, { title: GROUP_TITLE, color: 'cyan' })
}

async function pickTab(args) {
  if (args.tab_id != null) {
    await assertAllowed(args.tab_id)
    return args.tab_id
  }
  const tabs = await allowedTabs()
  if (!tabs.length) throw new Error('No tabs: open one with chrome_open')
  return (tabs.find((t) => t.active) || tabs[tabs.length - 1]).id
}

function tabInfo(tab) {
  return { tab_id: tab.id, title: tab.title, url: tab.url }
}

async function inPage(tabId, func, args = []) {
  const [res] = await chrome.scripting.executeScript({ target: { tabId }, func, args })
  if (res && res.result && res.result.__error) throw new Error(res.result.__error)
  return res ? res.result : null
}

// --- functions injected into the page (must be self-contained)
function pageSnapshot() {
  const visible = (el) => {
    const r = el.getBoundingClientRect()
    const s = getComputedStyle(el)
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'
  }
  const sel = 'a[href], button, input:not([type=hidden]), textarea, select, [role=button], [role=link], [role=tab], [role=menuitem], [role=checkbox], [contenteditable=true], summary'
  const els = [...document.querySelectorAll(sel)].filter(visible).slice(0, 300)
  let n = 0
  const elements = els.map((el) => {
    const ref = 'e' + ++n
    el.setAttribute('data-nesi-ref', ref)
    const label = (el.getAttribute('aria-label') || el.innerText || el.value || el.placeholder || el.title || el.alt || '').trim().replace(/\s+/g, ' ').slice(0, 120)
    return {
      ref,
      tag: el.tagName.toLowerCase(),
      type: el.getAttribute('type') || el.getAttribute('role') || undefined,
      label,
      href: el.tagName === 'A' ? el.href : undefined,
      value: ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName) && el.type !== 'password' ? String(el.value).slice(0, 200) : undefined,
    }
  })
  document.querySelectorAll('[data-nesi-ref]').forEach((el) => { if (!els.includes(el)) el.removeAttribute('data-nesi-ref') })
  const text = (document.body ? document.body.innerText : '').replace(/\n{3,}/g, '\n\n').slice(0, 15000)
  return { title: document.title, url: location.href, text, elements }
}

function pageClick(ref) {
  const el = document.querySelector(`[data-nesi-ref="${ref}"]`)
  if (!el) return { __error: `Cannot find ${ref}: read the page again (chrome_read)` }
  el.scrollIntoView({ block: 'center' })
  el.click()
  return { clicked: ref }
}

function pageType(ref, text, submit) {
  const el = document.querySelector(`[data-nesi-ref="${ref}"]`)
  if (!el) return { __error: `Cannot find ${ref}: read the page again (chrome_read)` }
  const kind = (el.getAttribute('type') || '').toLowerCase()
  const ac = (el.getAttribute('autocomplete') || '').toLowerCase()
  if (kind === 'password' || /password|one-time-code/.test(ac)) {
    return { __error: 'Nesicode does not type passwords or codes: ask the user to sign in themselves.' }
  }
  el.scrollIntoView({ block: 'center' })
  el.focus()
  if (el.isContentEditable) {
    document.execCommand('selectAll', false)
    document.execCommand('insertText', false, text)
  } else {
    const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set
    setter.call(el, text)
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  }
  if (submit) {
    const opts = { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }
    el.dispatchEvent(new KeyboardEvent('keydown', opts))
    el.dispatchEvent(new KeyboardEvent('keyup', opts))
    if (el.form) el.form.requestSubmit ? el.form.requestSubmit() : el.form.submit()
  }
  return { typed: ref }
}

function pageScroll(direction) {
  window.scrollBy({ top: (direction === 'up' ? -1 : 1) * window.innerHeight * 0.85 })
  return { scrollY: Math.round(window.scrollY) }
}

async function run(cmd, args) {
  if (cmd === 'tabs') {
    const c = await config()
    return { tabs: (await allowedTabs()).map((t) => ({ ...tabInfo(t), active: t.active, opened_by_agent: c.agentTabs.includes(t.id) })) }
  }
  if (cmd === 'open') {
    if (!/^https?:\/\//i.test(args.url || '')) throw new Error('Solo se pueden abrir URLs http(s)')
    let tab
    if (args.tab_id != null) {
      await assertAllowed(args.tab_id)
      tab = await chrome.tabs.update(args.tab_id, { url: args.url, active: true })
    } else {
      tab = await chrome.tabs.create({ url: args.url, active: true })
      const c = await config()
      await setList('agentTabs', [...c.agentTabs, tab.id])
      await addToGroup(tab.id)
    }
    return tabInfo(await waitLoaded(tab.id))
  }
  const tabId = await pickTab(args)
  if (cmd === 'read') return { tab_id: tabId, ...(await inPage(tabId, pageSnapshot)) }
  if (cmd === 'click') {
    const r = await inPage(tabId, pageClick, [String(args.ref)])
    await new Promise((res) => setTimeout(res, 600))
    return { tab_id: tabId, ...r, ...tabInfo(await waitLoaded(tabId, 10000)) }
  }
  if (cmd === 'type') {
    const r = await inPage(tabId, pageType, [String(args.ref), String(args.text ?? ''), !!args.submit])
    if (args.submit) await new Promise((res) => setTimeout(res, 800))
    return { tab_id: tabId, ...r, ...tabInfo(await waitLoaded(tabId, 10000)) }
  }
  if (cmd === 'scroll') return { tab_id: tabId, ...(await inPage(tabId, pageScroll, [args.direction || 'down'])) }
  if (cmd === 'back') {
    await chrome.tabs.goBack(tabId)
    return tabInfo(await waitLoaded(tabId))
  }
  if (cmd === 'close') {
    await chrome.tabs.remove(tabId)
    return { closed: tabId }
  }
  if (cmd === 'screenshot') {
    const tab = await chrome.tabs.update(tabId, { active: true })
    await chrome.windows.update(tab.windowId, { focused: true })
    await new Promise((r) => setTimeout(r, 300))
    const image = await chrome.tabs.captureVisibleTab(tab.windowId, { format: 'jpeg', quality: 70 })
    return { ...tabInfo(tab), image }
  }
  throw new Error('Comando desconocido: ' + cmd)
}

connect()
