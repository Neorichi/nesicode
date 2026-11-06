const $ = (id) => document.getElementById(id)
const send = (msg) => chrome.runtime.sendMessage(msg)
const LABEL = { connected: 'Connected to Nesicode', connecting: 'Connecting…', disconnected: 'Not connected', error: 'Not connected' }

async function refresh() {
  const s = await send({ type: 'status' })
  const [current] = await chrome.tabs.query({ active: true, currentWindow: true })
  $('dot').className = 'dot ' + s.state
  $('status').innerHTML = `<b>${LABEL[s.state] || s.state}</b>` + (s.error && s.state !== 'connected' ? `<div class="err">${s.error}</div>` : '')
  const paired = !!s.server
  $('pairBox').hidden = paired
  $('repair').hidden = !paired
  $('reconnect').hidden = !paired || s.state === 'connected'
  $('hint').hidden = !paired
  $('shareBox').hidden = !paired || !current
  $('tabsBox').hidden = !paired || s.tabs.length === 0
  if (current) {
    const shared = s.tabs.find((t) => t.id === current.id)
    $('shareInfo').textContent = shared ? 'In the “Nesicode” group: Nesicode can use it' : 'Outside the group: Nesicode cannot see it'
    $('share').textContent = shared ? 'Remove from group' : 'Add to group'
    $('share').className = shared ? '' : 'primary'
    $('share').onclick = async () => {
      await send({ type: 'share', tabId: current.id, on: !shared })
      setTimeout(refresh, 400)
    }
  }
  $('tabs').innerHTML = ''
  for (const t of s.tabs) {
    const div = document.createElement('div')
    div.className = 'tab'
    div.textContent = '● ' + (t.title || t.url)
    div.title = t.url
    $('tabs').appendChild(div)
  }
}

$('pair').onclick = async () => {
  const r = await send({ type: 'pair', code: $('code').value })
  if (r && r.ok === false) $('status').innerHTML = `<div class="err">${r.error}</div>`
  setTimeout(refresh, 600)
}
$('reconnect').onclick = async () => {
  $('status').innerHTML = '<b>Connecting…</b>'
  await send({ type: 'reconnect' })
  setTimeout(refresh, 800)
}
$('repair').onclick = async () => {
  await send({ type: 'unpair' })
  setTimeout(refresh, 400)
}
chrome.runtime.onMessage.addListener((m) => m.type === 'state' && refresh())
send({ type: 'reconnect' }).finally(() => setTimeout(refresh, 400))
refresh()
