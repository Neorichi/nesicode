"""End-to-end check of the desktop window's guards in the REAL window code (nesicode.desktop.main) on this machine's
web view: WebKit on macOS, WebKitGTK on Linux (CI runs it under Xvfb). Links clicked or forced in the app page must
open in the browser or nowhere, never in the window; the native bridge must answer the app page and nothing inside
it (an agent-written preview). Uses a scratch HOME and never opens a browser (webbrowser.open is recorded instead).
Prints the results and exits 1 when a case does not do what it must.

Usage: python scripts/desktop_window_check.py [out.json]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import webbrowser

OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
sys.argv[0] = os.path.abspath(sys.argv[0])  # (pywebview resolves its base folder from it; the cwd changes below)
HOME = tempfile.mkdtemp(prefix="nesi-window-")
for _k in [k for k in os.environ if k.startswith("NESICODE_")]:  # nothing of a real setup (its data dir, keys…)
    del os.environ[_k]
os.environ.update(HOME=HOME, XDG_DATA_HOME=os.path.join(HOME, ".local", "share"), NESICODE_NO_DOTENV="1",
                  NESICODE_NO_TELEGRAM="1", NESICODE_NO_SCHEDULER="1")
os.chdir(HOME)  # (config reads ./.env too)
opened: list[str] = []


def _record(url, *a, **k):
    opened.append(str(url))
    return True


webbrowser.open = webbrowser.open_new = webbrowser.open_new_tab = _record


def _watchdog():
    time.sleep(240)
    print("window check: watchdog fired (the window or a case hung)", file=sys.stderr)
    os._exit(3)


threading.Thread(target=_watchdog, daemon=True).start()

import webview  # noqa: E402

from nesicode import desktop  # noqa: E402

ROOT = "(document.getElementById('root') || document.body)"


def _click(href: str, target: str = "", box: bool = False, download: str = "") -> str:
    """A link added to the page and clicked (optionally inside a box that stops clicks, as the app's modals do)."""
    extra = (f"a.target = {json.dumps(target)};" if target else "") + (f"a.download = {json.dumps(download)};" if download else "")
    holder = ("const d = document.createElement('div'); d.addEventListener('click', e => e.stopPropagation()); "
              f"{ROOT}.appendChild(d);") if box else f"const d = {ROOT};"
    return (f"(() => {{ {holder} const a = document.createElement('a'); a.href = {json.dumps(href)}; {extra} "
            "a.textContent = 'x'; d.appendChild(a); a.click(); return 1 })()")


def _go(href: str) -> str:
    return f"(() => {{ location.href = {json.dumps(href)}; return 1 }})()"


# name: (script, what the browser must have been asked to open, whether the page must stay the same document)
CASES = {
    "A_plain_link_click": (_click("https://example.com/a"), ["https://example.com/a"], True),
    "B_link_in_stopPropagation_box": (_click("https://example.com/b", box=True), ["https://example.com/b"], True),
    "C_direct_navigation": (_go("https://example.com/c"), ["https://example.com/c"], True),
    "C2_direct_navigation_unknown_host": (_go("https://nesicode-check.invalid/c2"), ["https://nesicode-check.invalid/c2"], True),
    "D_file_url": (_go("file:///etc/hosts"), [], True),
    "E_app_route": (_go("/settings"), [], False),
    "F_api_url_as_page": (_go("/api/health"), [], True),
    "G_target_blank_link": (_click("https://example.com/g", target="_blank"), ["https://example.com/g"], True),
    "H_button_inside_link": ("(() => { const a = document.createElement('a'); a.href = 'https://example.com/h'; "
                             "const b = document.createElement('button'); b.type = 'button'; b.className = 'md-img-blocked'; "
                             f"b.textContent = 'img'; a.appendChild(b); {ROOT}.appendChild(a); b.click(); return 1 }})()",
                             ["https://example.com/h"], True),
    "J_same_origin_link_click": (_click("/projects"), [], False),
    # the page's fallback before the bridge exists: a detached target=_blank link it clicks. Inside the user's click it
    # opens the browser everywhere; run from a script, as here, WebKitGTK (which tells a user's gesture from a script)
    # opens nothing, as it must for a preview's own a.click()
    "K_no_bridge_yet_fallback": ("(() => { const saved = window.pywebview; window.pywebview = undefined; "
                                 "const a = document.createElement('a'); a.href = 'https://example.com/k'; a.textContent = 'K'; "
                                 f"{ROOT}.appendChild(a); a.click(); window.pywebview = saved; return 1 }})()",
                                 [] if sys.platform.startswith("linux") else ["https://example.com/k"], True),
    "M_encoded_api_link_click": (_click("/%61pi/health"), [], True),
    "M2_encoded_api_direct": (_go("/api%2Fhealth"), [], True),
    "N_assets_direct": (_go("/assets/missing.js"), [], True),
    "N2_docs_link_in_stopPropagation_box": (_click("/docs", box=True), [], True),
    "N3_openapi_direct": (_go("/openapi.json"), [], True),
    "O_download_anchor": (_click("/api/health", download="h.json"), [], True),
    "P_blank_smb_link": (_click("smb://example.com/share", target="_blank"), [], True),
    "Q_window_open_by_script": ("(() => { window.open('https://example.com/q'); return 1 })()", [], True),
    "R_data_url_direct": (_go("data:text/html,<p>away</p>"), [], True),
    "S_bridge_from_app_page": ("(() => { window.pywebview.api.open_external('https://example.com/s'); return 1 })()",
                               ["https://example.com/s"], True),
    "I_mailto_link_click": (_click("mailto:someone@example.com"), ["mailto:someone@example.com"], True),
}

# An agent-written page in the preview (the app's own sandboxed iframe, served by the API with its sandbox policy)
# tries every way to reach the bridge. None may open anything.
EVIL = r"""
function t(f) { try { f() } catch (e) {} }
t(function () { window.webkit.messageHandlers.jsBridge.postMessage(JSON.stringify(
  {funcName: 'open_external', params: ['https://evil.example/plain'], id: 'e1'})) });
t(function () { window.webkit.messageHandlers.jsBridge.postMessage(JSON.stringify(
  {funcName: 'open_external', params: ['https://evil.example/forged'], id: 'e2', nesiKey: 'guess'})) });
t(function () { window.webkit.messageHandlers.jsBridge.postMessage(JSON.stringify(
  {funcName: '_pywebviewAlert', params: 'evil', id: 'alert'})) });
t(function () { window.chrome.webview.postMessage(['open_external', JSON.stringify(['https://evil.example/edge']), 'e3']) });
t(function () { parent.pywebview.api.open_external('https://evil.example/parent') });
t(function () { top.location.href = 'https://evil.example/top' });
t(function () { window.open('https://evil.example/popup') });
"""
PREVIEW = """(() => {
  window.__e2ePreview = 'pending';
  const html = '<!doctype html><meta charset="utf-8"><p>preview</p><script>' + %s + '<\\/script>';
  const json = r => r.ok ? r.json() : Promise.reject(new Error(r.status + ' ' + r.url));
  const post = (url, body) => fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}).then(json);
  (async () => {
    try {
      const p = await post('/api/projects', {name: 'window check', profile: 'general', executor_model: 'mock/demo', improver_model: 'mock/demo'});
      const t = await post('/api/projects/' + p.id + '/threads', {});
      const fd = new FormData();
      fd.append('files', new Blob([html], {type: 'text/html'}), 'evil.html');
      const up = await fetch('/api/threads/' + t.id + '/attachments', {method: 'POST', body: fd}).then(json);
      const f = document.createElement('iframe');
      f.setAttribute('sandbox', 'allow-scripts allow-popups allow-modals allow-downloads');
      f.onload = () => { window.__e2ePreview = 'loaded' };
      f.src = '/api/threads/' + t.id + '/file?path=' + encodeURIComponent(up[0].path) + '&view=1';
      document.body.appendChild(f);
    } catch (e) { window.__e2ePreview = 'error: ' + e }
  })();
  return 1;
})()""" % json.dumps(EVIL)

def _descendants(root: int) -> list[str]:
    """The names of the processes this one started, and theirs (Linux /proc)."""
    parent: dict[int, int] = {}
    names: dict[int, str] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            raw = open(f"/proc/{d}/stat", "rb").read()
        except OSError:
            continue
        names[int(d)] = raw[raw.index(b"(") + 1:raw.rindex(b")")].decode(errors="replace")
        parent[int(d)] = int(raw[raw.rindex(b")") + 2:].split()[1])
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        for c, pp in parent.items():
            if pp == p:
                out.append(names[c])
                todo.append(c)
    return sorted(out)


holder: dict = {}
real_create, real_start = webview.create_window, webview.start


def create_window(*a, **k):
    k["hidden"] = sys.platform == "darwin"  # (no window popping up on a Mac someone is using; Xvfb on Linux CI)
    holder["url"] = a[1] if len(a) > 1 else k.get("url")
    holder["w"] = real_create(*a, **k)
    return holder["w"]


def steps():
    w, app_url = holder["w"], holder["url"]
    base = app_url.split("/?", 1)[0]
    res: dict = {"base": base, "platform": sys.platform}

    def js(expr):
        try:
            return w.evaluate_js(expr)
        except Exception as exc:  # noqa: BLE001
            return f"ERR {exc!r}"

    def url():
        """The address of the document the window shows (WebKitGTK's own 'uri' is a pending navigation's meanwhile)."""
        here = js("location.href")
        return here if isinstance(here, str) and not here.startswith("ERR") else (w.get_current_url() or "")

    def api_ready():
        return js("!!(window.pywebview && window.pywebview.api && window.pywebview.api.open_external)") is True

    def wait(cond, limit):
        t0 = time.time()
        while time.time() - t0 < limit:
            if cond():
                return True
            time.sleep(0.3)
        return False

    def home():
        if not url().startswith(base + "/") or "/api/" in url():
            w.load_url(app_url)
        wait(api_ready, 30)
        time.sleep(0.5)

    res["api_ready"] = wait(api_ready, 60)
    res["start_url_is_app"] = url().startswith(base)
    if sys.platform.startswith("linux"):
        from webview.platforms import gtk

        views = list(gtk.BrowserView.instances.values())
        res["gtk_context_menu_kept"] = bool(views and getattr(views[0], "_nesi_menu", 0))
        res["gtk_clipboard_reads_off"] = bool(views) and views[0].webview.get_settings().props.javascript_can_access_clipboard is False
        res["web_sandbox"] = desktop._web_sandbox.get("on")
        res["web_processes"] = _descendants(os.getpid())
    for name, (script, want, _) in CASES.items():
        home()
        opened.clear()
        before = url()
        js("window.pywebview.__e2eProbe = 1")
        res[name] = {"js": js(script)}
        # what must open may take a while on a busy machine; then a moment more for anything that must not
        wait(lambda: len(opened) >= len(want), 10 if want else 3)
        time.sleep(1.0)
        after = url()
        res[name].update({"url_after": after.replace(base, "<app>").split("?token=")[0],
                          "stayed_in_app": after.startswith(base + "/") and "/api/" not in after,
                          "same_url": after == before, "opened": list(opened),
                          "same_page": js("!!(window.pywebview && window.pywebview.__e2eProbe === 1)") is True})
    home()
    opened.clear()
    js(PREVIEW)
    wait(lambda: js("window.__e2ePreview") != "pending", 20)
    time.sleep(3)
    res["preview_bridge"] = {"state": js("window.__e2ePreview"), "opened": list(opened),
                             "stayed_in_app": url().startswith(base + "/") and "/api/" not in url()}
    holder["res"] = res
    w.destroy()


def start(*a, **k):
    k["func"] = steps
    return real_start(*a, **k)


webview.create_window = create_window
webview.start = start
desktop.main()

res = holder.get("res") or {}
fails = []
linux_keys = ("gtk_context_menu_kept", "gtk_clipboard_reads_off")
if sys.platform.startswith("linux") and os.environ.get("WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS", "0") == "0":
    linux_keys += ("web_sandbox",)  # (a real machine: its pages must run in WebKit's bubblewrap sandbox)
    if "bwrap" not in (res.get("web_processes") or []):
        fails.append(f"no bubblewrap among the window's processes: {res.get('web_processes')}")
for key in ("api_ready", "start_url_is_app") + (linux_keys if sys.platform.startswith("linux") else ()):
    if res.get(key) is not True:
        fails.append(f"{key}: {res.get(key)!r}")
for name, (_, want, same) in CASES.items():
    r = res.get(name) or {}
    if not r.get("stayed_in_app"):
        fails.append(f"{name}: the window left the app ({r.get('url_after')})")
    if r.get("opened") != want:
        fails.append(f"{name}: opened {r.get('opened')} instead of {want}")
    if same and not (r.get("same_url") and r.get("same_page")):
        fails.append(f"{name}: the app page was replaced or reloaded (url {r.get('url_after')}, same page {r.get('same_page')})")
pb = res.get("preview_bridge") or {}
if pb.get("state") != "loaded" or pb.get("opened") != [] or not pb.get("stayed_in_app"):
    fails.append(f"preview_bridge: {pb}")
print(json.dumps(res, indent=1))
if OUT:
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
print("\n".join(["window check FAILED:", *fails]) if fails else f"window check passed: {len(CASES)} link cases + preview bridge")
sys.exit(1 if fails else 0)
