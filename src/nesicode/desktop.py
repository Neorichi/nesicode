"""nesicode desktop: starts the server on a private port and shows it in a native window.

Used by the macOS .app, the Windows installer and the Linux .deb (PyInstaller bundles), and by
`nesicode desktop` from a source checkout.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path


PREFERRED_PORT = 47821  # stable, so the Chrome extension's connection code keeps working across launches


# ------------------------------------------------------------- the native bridge (window.pywebview.api)
# WebKit gives `window.webkit.messageHandlers.jsBridge` to EVERY frame of the window, including the
# sandboxed iframes that show HTML files the agent wrote or downloaded, and pywebview neither checks the
# sender nor escapes the call id it pastes into JS it then runs in the app's own page. A preview could
# thus run code as the app (its cookie, its whole API: connectors that run commands, exports with keys).
# Two locks: only the app's own page may call the bridge (macOS: checked natively per message), and a
# call's name and id must be plain identifiers (all platforms), so nothing can be smuggled into that JS.
_BRIDGE_FUNC = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}(\.[A-Za-z_][A-Za-z0-9_]{0,63}){0,3}$")
_BRIDGE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_guarded_bridge: dict[str, object] = {}


def bridge_call_ok(func_name, value_id) -> bool:
    """pywebview evaluates `…_returnValuesCallbacks["<func_name>"]["<value_id>"](…)` in the main frame."""
    if isinstance(value_id, int) and not isinstance(value_id, bool):
        value_id = str(value_id)
    return (isinstance(func_name, str) and bool(_BRIDGE_FUNC.match(func_name))
            and isinstance(value_id, str) and bool(_BRIDGE_ID.match(value_id)))


def trusted_frame(message, port: int) -> bool:
    """Was this WKScriptMessage posted by the app's own page: the main frame, origin http://127.0.0.1:<port>,
    and not a file served under /api/ (an HTML file opened as the whole page is still a sandboxed preview)?"""
    try:
        info = message.frameInfo()
        if info is None or not info.isMainFrame():
            return False
        origin = info.securityOrigin()
        if origin is None or (str(origin.protocol()), str(origin.host()), int(origin.port())) != ("http", "127.0.0.1", port):
            return False
        req = info.request()
        url = req.URL() if req is not None else None
        return url is not None and not str(url.path() or "").startswith("/api/")
    except Exception:  # anything unexpected: refuse
        return False


_blocked_calls = {"n": 0}


def _note_blocked(why: str) -> None:
    """Log a refused bridge call: the first ones, then one line per 100 (a page posting in a loop must not fill the
    log)."""
    _blocked_calls["n"] += 1
    n = _blocked_calls["n"]
    if n <= 5 or n % 100 == 0:
        print(f"nesicode: blocked a desktop bridge call {why} ({n} so far)", file=sys.stderr)


def _guarded_js_bridge_call(original):
    def js_bridge_call(window, func_name, param, value_id):
        if not bridge_call_ok(func_name, value_id):
            _note_blocked("with an invalid name or id")
            return None
        return original(window, func_name, param, value_id)

    js_bridge_call._nesi_original = original
    return js_bridge_call


def guard_js_bridge(port: int) -> list[str]:
    """Lock pywebview's bridge to the app page (see above). Returns the backends it guarded."""
    import importlib

    native = ("cocoa",) if sys.platform == "darwin" else ("edgechromium", "mshtml", "cef") if os.name == "nt" else ("gtk", "qt", "cef")
    done = []
    for name in ("cocoa", "gtk", "qt", "edgechromium", "mshtml", "cef"):  # the backends that call js_bridge_call
        modname = f"webview.platforms.{name}"
        if modname not in sys.modules and name not in native:
            continue  # only the backends this platform can load (importing the others fails or is slow)
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        fn = getattr(mod, "js_bridge_call", None)
        if fn is None:
            continue
        if not hasattr(fn, "_nesi_original"):
            mod.js_bridge_call = _guarded_js_bridge_call(fn)
        if name in ("cocoa", "gtk") and name not in _guarded_bridge:
            try:  # the per-frame lock: WebKit, on macOS and on Linux, hands the bridge to every frame
                _guarded_bridge[name] = _guard_cocoa_bridge(mod, port) if name == "cocoa" else _guard_gtk_bridge(mod, port=port)
            except Exception as exc:
                print(f"nesicode: could not guard the {name} bridge:", exc, file=sys.stderr)
                continue
        done.append(name)
    return done


def bridge_is_safe(guarded: list[str], platform: str | None = None) -> bool:
    """May the window get the native API? WebKit (macOS and Linux) gives the bridge to every frame, the agent's
    previews included, so there it needs its per-frame lock; WebView2 gives it to the main frame only."""
    platform = platform or sys.platform
    if platform == "darwin":
        return "cocoa" in guarded
    if platform == "win32":
        return bool(guarded)
    return "gtk" in guarded


def _guard_cocoa_bridge(cocoa, port: int):
    """WKWebView tells who sent each message (WKScriptMessage.frameInfo): drop all but the app page's."""
    import objc

    base = cocoa.BrowserView.JSBridge

    class NesiJSBridge(base):
        def userContentController_didReceiveScriptMessage_(self, controller, message):
            if trusted_frame(message, port):
                objc.super(NesiJSBridge, self).userContentController_didReceiveScriptMessage_(controller, message)
            else:
                _note_blocked("from a frame that is not the app")

    cocoa.BrowserView.JSBridge = NesiJSBridge  # pywebview creates BrowserView.JSBridge for every window
    return NesiJSBridge


# WebKitGTK does not say which frame posted a message, so the app's page signs its calls instead: pywebview's script,
# which the window runs in the main frame only, gets a key made at launch and adds it to every call. The previews are
# sandboxed in an opaque origin (no allow-same-origin), so they can never read the app page's variables.
_GTK_KEYED_CALLS = """
;(function (key, origin) {
  var p = window.pywebview;
  // only the app's own page: a page that is not (were one ever to load in the window) never sees the key
  if (!p || p.platform !== 'gtkwebkit2' || window.location.origin !== origin) return;
  function post(funcName, params, id) {
    return window.webkit.messageHandlers.jsBridge.postMessage(
      p.stringify({ funcName: funcName, params: params, id: id, nesiKey: key }));
  }
  p._jsApiCallback = post;
  window.alert = function (message) { post('_pywebviewAlert', message, 'alert'); };
})(%s, %s);
"""


def gtk_bridge_message_ok(message, key: str) -> bool:
    """Does this WebKitGTK script message carry the launch's key (so it came from the app's own page)?"""
    try:
        value = message.get_js_value() if hasattr(message, "get_js_value") else message
        body = json.loads(value.to_string())
        given = body.get("nesiKey") if isinstance(body, dict) else None
        return isinstance(given, str) and hmac.compare_digest(given.encode(), key.encode())
    except Exception:  # anything unexpected: refuse
        return False


def _guard_gtk_bridge(gtk, util=None, port: int | None = None):
    if util is None:
        import webview.util as util
    origin = json.dumps(f"http://127.0.0.1:{port if port is not None else _guarded_nav.get('port')}")

    key = secrets.token_urlsafe(32)
    view_cls = gtk.BrowserView
    original = view_cls.on_js_bridge_call
    original_js = util.load_js_files

    def on_js_bridge_call(self, manager, message):
        if not gtk_bridge_message_ok(message, key):
            _note_blocked("from a frame that is not the app")
            return None
        return original(self, manager, message)

    def load_js_files(window, platform):
        js_code, finish_script = original_js(window, platform)
        if platform == "gtkwebkit2":  # the script pywebview runs in the main frame on every load
            js_code += _GTK_KEYED_CALLS % (json.dumps(key), origin)
        return js_code, finish_script

    view_cls.on_js_bridge_call = on_js_bridge_call  # connected to the window's script messages when pywebview builds it
    util.load_js_files = load_js_files
    return on_js_bridge_call


# ------------------------------------------------------------- the window never leaves the app
# The window has no address bar and no Back button: a link that replaced the app's page (another website, a local
# file, a raw file of the API) left the user stuck on it until they quit the app. The page itself sends links to other
# sites to the browser, but a navigation can reach the window in ways the page never sees (a click inside a modal that
# stops it, the context menu's «Abrir enlace», a script), so the window turns such navigations away natively too.
_guarded_nav: dict[str, object] = {}
# what the server answers with something other than the app's page: FastAPI's API docs (and /assets, /api below)
_RAW_PAGES = ("/assets", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json")


def app_page(path: str) -> bool:
    """Is this path of the app's own origin one of its pages, rather than the API, the page's scripts and styles or the
    API docs? Judged the way the server routes it: uvicorn decodes the path once, so /%61pi/… and /api%2F… are the API
    (web/src/logic.ts `appPagePath` is the page's twin)."""
    p = urllib.parse.unquote(path)
    return not (p.startswith(("/api/", "/assets/")) or p.rstrip("/") in _RAW_PAGES)


def window_navigation(url: str, port: int) -> str:
    """What the app window does with a navigation of its main frame to `url`: "allow" (the app's own pages), "browser"
    (another website or an e-mail link: the user's browser or mail app opens it instead) or "block" (anything else:
    the API's raw output, local files, other schemes)."""
    try:
        text = str(url).strip()
        if text == "about:blank":
            return "allow"  # what a web view may load before the app
        parts = urllib.parse.urlsplit(text)
        scheme = parts.scheme.lower()
        if scheme == "mailto":
            return "browser"
        if scheme in ("http", "https") and parts.hostname:
            if scheme == "http" and parts.hostname == "127.0.0.1" and parts.port == port:
                return "allow" if app_page(parts.path) else "block"
            return "browser"
    except ValueError:  # e.g. a port that is not a number
        pass
    return "block"


def is_app_url(url: str, port: int) -> bool:
    """Is `url` one of the app's own pages (not about:blank, not the API's raw output)?"""
    try:
        parts = urllib.parse.urlsplit(str(url).strip())
        return (parts.scheme.lower() == "http" and parts.hostname == "127.0.0.1" and parts.port == port
                and app_page(parts.path))
    except ValueError:
        return False


def navigation_decision(action, port: int) -> tuple[str, str]:
    """`window_navigation` for a WKNavigationAction of the main frame, as (decision, url). New windows (target=_blank,
    which pywebview already sends to the browser) and the frames inside the page (the sandboxed previews) are left as
    they are, and so is anything unexpected."""
    try:
        frame = action.targetFrame()
        if frame is None or not frame.isMainFrame():
            return "allow", ""
        url = str(action.request().URL().absoluteString())
    except Exception:
        return "allow", ""
    return window_navigation(url, port), url


def new_window_decision(action, port: int, link_activated: int) -> tuple[str, str]:
    """For a request to open a new window (a target=_blank link, window.open): ("browser", url) only for a link the
    user activated to another website or an e-mail address, else ("none", url). pywebview hands whatever such a link
    holds to the system, any scheme: a target=_blank smb: or file: link in an agent-written page (the previews allow
    popups) would reach Finder / Explorer, which on Windows connects to that share with the user's credentials."""
    try:
        if action.navigationType() != link_activated:
            return "none", ""
        url = str(action.request().URL().absoluteString())
    except Exception:
        return "none", ""
    return ("browser" if window_navigation(url, port) == "browser" else "none"), url


def _turned_away(decision: str, url: str) -> None:
    if decision == "browser":  # off the UI thread: the macOS browser hand-off waits for osascript
        threading.Thread(target=webbrowser.open, args=(url,), name="nesicode-open-link", daemon=True).start()
    else:
        kind = url.split(":", 1)[0] if ":" in url else "bare"
        print(f"nesicode: kept the window on the app (a {kind[:20]} link)", file=sys.stderr)


def _guard_cocoa_navigation(cocoa):
    import objc

    base = cocoa.BrowserView.BrowserDelegate
    cancel = getattr(cocoa.WebKit, "WKNavigationActionPolicyCancel", 0)
    link_activated = getattr(cocoa.WebKit, "WKNavigationTypeLinkActivated", 0)

    class NesiBrowserDelegate(base):
        def webView_decidePolicyForNavigationAction_decisionHandler_(self, webview, action, handler):
            decision, url = navigation_decision(action, _guarded_nav["port"])
            if decision == "allow":
                objc.super(NesiBrowserDelegate, self).webView_decidePolicyForNavigationAction_decisionHandler_(
                    webview, action, handler)
                return
            handler(cancel)
            _turned_away(decision, url)

        def webView_createWebViewWithConfiguration_forNavigationAction_windowFeatures_(
                self, webview, config, action, features):
            decision, url = new_window_decision(action, _guarded_nav["port"], link_activated)
            if decision == "browser":
                _turned_away(decision, url)
            return None  # never a second web view

    cocoa.BrowserView.BrowserDelegate = NesiBrowserDelegate  # pywebview creates one per window from this attribute
    return NesiBrowserDelegate


def _guard_edge_navigation(edgechromium):
    """WebView2's NavigationStarting is raised for the main frame only (frames have their own event);
    NewWindowRequested for target=_blank links and window.open, from any frame (see new_window_decision)."""
    original = edgechromium.EdgeChrome.on_navigation_start
    original_new_window = edgechromium.EdgeChrome.on_new_window_request

    def on_navigation_start(self, sender, args):
        try:
            url = str(args.Uri)
            decision = window_navigation(url, _guarded_nav["port"])
        except Exception:
            decision, url = "allow", ""
        if decision != "allow":
            args.Cancel = True
            _turned_away(decision, url)
            return None
        return original(self, sender, args)

    def on_new_window_request(self, sender, args):
        try:
            url = str(args.get_Uri())
            decision = window_navigation(url, _guarded_nav["port"])
        except Exception:
            return original_new_window(self, sender, args)
        args.set_Handled(True)  # never a WebView2 popup window
        if decision == "browser":
            _turned_away(decision, url)
        return None

    # pywebview binds both to WebView2's events when it creates the window
    edgechromium.EdgeChrome.on_navigation_start = on_navigation_start
    edgechromium.EdgeChrome.on_new_window_request = on_new_window_request
    return on_navigation_start


# Linux (WebKitGTK). Its policy signal for a navigation does not say whether the main frame or a frame inside the page
# (an agent-written preview) is navigating; its response signal does (WebKitGTK 2.40+, which pywebview 6 needs anyway).
# So the main frame is turned away there, before the new page replaces the app (the browser opens then), or when its
# load fails before any response (an unknown host, a mailto: link: the browser or the mail app opens then, and WebKit's
# error page never replaces the app). Turning away from the policy signal of the navigation itself would cut the
# frames' navigations too. The address the web view reports when a load starts is still the app's (checked on
# WebKitGTK 2.50 and 2.52), so it cannot be used. WebKit stops the page's own loads as soon as it lets a main-frame
# navigation start: the replies' stream then reconnects where it was (api.ts streamRun), and the context menu has no
# «Open Link» here (the page sends clicked links to the browser itself, without any navigation).
def gtk_new_window_decision(action, port: int, link_clicked) -> tuple[str, str]:
    """`new_window_decision` for a WebKitGTK navigation action, which also says whether the user's own click (or key)
    caused it: a script's a.click() in an agent-written preview counts as a clicked link too, and opens nothing here."""
    try:
        if action.get_navigation_type() != link_clicked or not action.is_user_gesture():
            return "none", ""
        url = str(action.get_request().get_uri())
    except Exception:
        return "none", ""
    return ("browser" if window_navigation(url, port) == "browser" else "none"), url


def _gtk_frame_name(decision) -> str:
    """The target of a navigation ('_blank' for a target=_blank link); WebKitGTK before 2.40 kept it on the decision."""
    for get in (lambda: decision.get_navigation_action().get_frame_name(), lambda: decision.get_frame_name()):
        try:
            return str(get() or "")
        except Exception:
            continue
    return ""


def _gtk_after_init(gtk, hook) -> None:
    """Run hook(view) once pywebview has built each GTK BrowserView (its WebKitWebView is view.webview)."""
    view_cls = gtk.BrowserView
    hooks = view_cls.__dict__.get("_nesi_after_init")
    if hooks is None:
        hooks = []
        original = view_cls.__init__

        def __init__(self, *args, **kwargs):
            original(self, *args, **kwargs)
            for fn in list(hooks):
                try:
                    fn(self)
                except Exception as exc:  # a missing nicety must not cost the window
                    print("nesicode: a window setup step failed:", exc, file=sys.stderr)

        view_cls.__init__ = __init__
        view_cls._nesi_after_init = hooks
    if hook not in hooks:
        hooks.append(hook)


def _guard_gtk_navigation(gtk):
    webkit = gtk.webkit
    view_cls = gtk.BrowserView
    new_window, response = webkit.PolicyDecisionType.NEW_WINDOW_ACTION, webkit.PolicyDecisionType.RESPONSE
    link_clicked = webkit.NavigationType.LINK_CLICKED
    started, committed, finished = webkit.LoadEvent.STARTED, webkit.LoadEvent.COMMITTED, webkit.LoadEvent.FINISHED
    original_policy, original_load = view_cls.on_navigation, view_cls.on_load_finish

    def turn_away_once(view, decision: str, url: str) -> None:
        if not getattr(view, "_nesi_away", False):  # once per navigation (its response, then the failure it ends in)
            view._nesi_away = True
            _turned_away(decision, url)

    def on_navigation(self, webview, decision, decision_type):
        port = _guarded_nav["port"]
        if decision_type == new_window or (decision_type != response and _gtk_frame_name(decision) == "_blank"):
            try:
                kind, url = gtk_new_window_decision(decision.get_navigation_action(), port, link_clicked)
            except Exception:
                kind, url = "none", ""
            try:
                decision.ignore()  # never a second window (pywebview handed any scheme to the system)
            except Exception:
                pass
            if kind == "browser":
                _turned_away(kind, url)
            return True
        if decision_type == response:
            try:
                main = bool(decision.is_main_frame_main_resource())
                url = str(decision.get_request().get_uri()) if main else ""
            except Exception:  # WebKitGTK < 2.40 cannot tell: the page still sends its links to the browser
                main, url = False, ""
            if main:
                kind = window_navigation(url, port)
                if kind != "allow":
                    decision.ignore()
                    turn_away_once(self, kind, url)
                    return True
            try:
                displayable = bool(decision.is_mime_type_supported())
            except Exception:
                displayable = True
            if not displayable and not _gtk_downloads_allowed(gtk):
                decision.ignore()  # a download: refused, as on macOS and Windows (pywebview's GTK saved it silently)
                return True
        return original_policy(self, webview, decision, decision_type)

    def on_load_finish(self, webview, status):  # (load-changed: the main frame's loads only)
        try:
            if status == started:
                self._nesi_away, self._nesi_committed = False, False
            elif status == committed:
                self._nesi_committed = True
            elif status == finished:
                self._nesi_away = False
                if getattr(self, "_nesi_committed", True) is False:
                    # A navigation that never replaced the page (turned away, or failed): the app's page, with its
                    # bridge, is still there. pywebview would inject the bridge again, dropping pending calls.
                    return None
        except Exception as exc:
            print("nesicode: window navigation check failed:", exc, file=sys.stderr)
        return original_load(self, webview, status)

    def watch_failures(view):
        def on_load_failed(webview, load_event, url, error):
            try:
                kind = window_navigation(str(url), _guarded_nav["port"])
            except Exception:
                return False
            if kind == "allow":
                return False  # the app's own page did not load: WebKit's error page says why
            turn_away_once(view, kind, str(url))  # (unless its response already did)
            return True  # no error page in place of the app

        view.webview.connect("load-failed", on_load_failed)

    # pywebview connects these to each window's web view when it builds it
    view_cls.on_navigation, view_cls.on_load_finish = on_navigation, on_load_finish
    _gtk_after_init(gtk, watch_failures)
    return on_navigation


def _gtk_downloads_allowed(gtk) -> bool:
    return bool((getattr(gtk, "settings", None) or {}).get("ALLOW_DOWNLOADS"))


def _sandbox_available() -> bool:
    """Can WebKitGTK's sandbox (bubblewrap, with a user namespace) start here? Not in most containers, nor where an
    administrator turned unprivileged user namespaces off."""
    flag = os.environ.get("WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS")
    if flag and flag != "0":
        return False
    import shutil
    import subprocess

    bwrap = "/usr/bin/bwrap" if os.path.exists("/usr/bin/bwrap") else shutil.which("bwrap")
    if not bwrap:
        return False
    try:
        probe = [bwrap, "--unshare-user", "--unshare-pid", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "true"]
        return subprocess.run(probe, capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def _gtk_web_content(gtk, sandbox: bool | None = None) -> bool:
    """What the pages in the Linux window may do, as on macOS and Windows: their web process sandboxed (WebKitGTK's
    4.1 API leaves it off unless asked; WKWebView and WebView2 always have one), no reading the clipboard from a script
    (pywebview turns it on, and a preview of an agent-written page could read what the user copied), no downloads
    saved unasked. Returns whether the sandbox is on."""
    webkit = gtk.webkit
    if sandbox is None:
        sandbox = _sandbox_available()
    if sandbox:  # before the window's web process starts: pywebview makes the context and loads the page in one go
        cls = webkit.WebContext
        for name in ("new_with_website_data_manager", "new_ephemeral", "new"):
            original = getattr(cls, name, None)
            if original is None or getattr(original, "_nesi_sandboxed", False):
                continue

            def sandboxed(*args, _original=original):
                context = _original(*args)
                context.set_sandbox_enabled(True)
                return context

            sandboxed._nesi_sandboxed = True
            setattr(cls, name, staticmethod(sandboxed))
    else:
        print("nesicode: WebKit's sandbox cannot start here (no user namespaces): the window's pages run without it",
              file=sys.stderr)

    def harden(view):
        view.webview.get_settings().props.javascript_can_access_clipboard = False
        if not _gtk_downloads_allowed(gtk):
            view.webview.get_context().connect("download-started", lambda context, download: download.cancel())

    if not gtk.BrowserView.__dict__.get("_nesi_web_content"):  # (once per process)
        gtk.BrowserView._nesi_web_content = True
        _gtk_after_init(gtk, harden)
    return sandbox


_web_sandbox: dict[str, bool] = {}


def _harden_web_content() -> None:
    if not sys.platform.startswith("linux"):
        return  # (WKWebView and WebView2 already do all of it)
    try:
        from webview.platforms import gtk

        _web_sandbox["on"] = _gtk_web_content(gtk)
    except Exception as exc:
        print("nesicode: could not restrict the window's pages:", exc, file=sys.stderr)


_GUARDS = {"cocoa": "_guard_cocoa_navigation", "edgechromium": "_guard_edge_navigation", "gtk": "_guard_gtk_navigation"}


def guard_navigation(port: int) -> list[str]:
    """Keep the app window on the app (see above). Call it before the window exists. Returns the backends guarded."""
    import importlib

    _guarded_nav["port"] = port
    name = "cocoa" if sys.platform == "darwin" else "edgechromium" if os.name == "nt" else "gtk"
    if name not in _guarded_nav:
        try:
            mod = importlib.import_module(f"webview.platforms.{name}")
            _guarded_nav[name] = globals()[_GUARDS[name]](mod)
        except Exception as exc:  # the page still sends its links to the browser
            print("nesicode: could not keep the window on the app:", exc, file=sys.stderr)
            return []
    return [name]


def _free_port() -> int:
    for port in (PREFERRED_PORT, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if os.name != "nt":
                # As the server's own socket does (asyncio on Unix): the connections of an app just quit linger in
                # TIME_WAIT for ~30 s and must not push a quick reopen off the port the Chrome extension knows; a
                # running app still holds it (Windows' SO_REUSEADDR would let a second app take it, so not there)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return s.getsockname()[1]
    raise RuntimeError("no free port")


def _wait(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except Exception as exc:  # 403 from the token guard means the server is up
            if getattr(exc, "code", None) == 403:
                return True
            time.sleep(0.2)
    return False


def _adopt_login_path() -> None:
    """Apps started from the Dock/Finder get a bare PATH; take the user's shell PATH so tools like
    npx (nvm), uv or brew installs are found by connectors and by the agent."""
    if os.name == "nt":
        return
    import subprocess

    shell = os.environ.get("SHELL") or ("/bin/zsh" if sys.platform == "darwin" else "/bin/bash")
    try:
        out = subprocess.run([shell, "-ilc", "printf '__NESI_PATH__%s' \"$PATH\""], capture_output=True,
                             encoding="utf-8", errors="surrogateescape", timeout=10).stdout
    except Exception:
        return
    if "__NESI_PATH__" in out:
        extra = out.rsplit("__NESI_PATH__", 1)[1].strip()
        parts = [p for p in (extra + os.pathsep + os.environ.get("PATH", "")).split(os.pathsep) if p]
        os.environ["PATH"] = os.pathsep.join(dict.fromkeys(parts))


# WebKit context-menu items that make no sense inside the app window.
_MENU_BLOCK = {
    "WKMenuItemIdentifierReload", "WKMenuItemIdentifierGoBack", "WKMenuItemIdentifierGoForward",
    "WKMenuItemIdentifierInspectElement", "WKMenuItemIdentifierOpenLinkInNewWindow",
    "WKMenuItemIdentifierDownloadLinkedFile", "WKMenuItemIdentifierOpenImageInNewWindow",
    "WKMenuItemIdentifierDownloadImage", "WKMenuItemIdentifierOpenFrameInNewWindow",
    "WKMenuItemIdentifierOpenMediaInNewWindow", "WKMenuItemIdentifierDownloadMedia",
}


def _enable_native_context_menu(webview) -> None:
    """pywebview empties the right-click menu unless in debug mode. Keep the native one (cut, copy,
    paste, select all, spelling, look up…) and drop only the browser-ish items."""
    if sys.platform == "darwin":
        try:
            import objc
            from webview.platforms import cocoa

            def willOpenMenu_withEvent_(self, menu, event):
                for item in list(menu.itemArray()):
                    if str(item.identifier() or "") in _MENU_BLOCK:
                        menu.removeItem_(item)

            objc.classAddMethods(cocoa.BrowserView.WebKitHost, [
                objc.selector(willOpenMenu_withEvent_, selector=b"willOpenMenu:withEvent:", signature=b"v@:@@"),
            ])
        except Exception as exc:  # never block the app over a context menu
            print("nesicode: could not enable the context menu:", exc)
    elif os.name == "nt":
        try:
            from webview.platforms import edgechromium

            original = edgechromium.EdgeChrome.on_webview_ready

            def on_ready(self, sender, args):
                original(self, sender, args)
                try:
                    sender.CoreWebView2.Settings.AreDefaultContextMenusEnabled = True
                except Exception:
                    pass

            edgechromium.EdgeChrome.on_webview_ready = on_ready
        except Exception as exc:
            print("nesicode: could not enable the context menu:", exc)
    else:
        try:
            from webview.platforms import gtk

            _gtk_context_menu(gtk)
        except Exception as exc:
            print("nesicode: could not enable the context menu:", exc)


# WebKitGTK's items that make no sense inside the app window (as _MENU_BLOCK on macOS), and «Open Link», which would
# navigate the window (see the navigation guard): a click on the link opens it in the browser
_GTK_MENU_BLOCK = (
    "OPEN_LINK", "OPEN_LINK_IN_NEW_WINDOW", "DOWNLOAD_LINK_TO_DISK", "OPEN_IMAGE_IN_NEW_WINDOW", "DOWNLOAD_IMAGE_TO_DISK",
    "OPEN_FRAME_IN_NEW_WINDOW", "GO_BACK", "GO_FORWARD", "STOP", "RELOAD", "INSPECT_ELEMENT",
    "OPEN_VIDEO_IN_NEW_WINDOW", "OPEN_AUDIO_IN_NEW_WINDOW", "DOWNLOAD_VIDEO_TO_DISK", "DOWNLOAD_AUDIO_TO_DISK",
)


def gtk_menu_filter(menu, blocked) -> bool:
    """Drop the browser-ish items of a WebKitGTK context menu and the separators they leave stranded. Returns True
    when nothing is left (the signal's 'handled': no menu at all)."""
    for item in list(menu.get_items()):
        if item.get_stock_action() in blocked:
            menu.remove(item)
    last_separator = True  # no separator first, none twice in a row, none last
    for item in list(menu.get_items()):
        separator = bool(item.is_separator())
        if separator and last_separator:
            menu.remove(item)
        last_separator = separator
    items = list(menu.get_items())
    if items and items[-1].is_separator():
        menu.remove(items[-1])
    return not list(menu.get_items())


def _gtk_context_menu(gtk) -> None:
    """pywebview empties the menu (outside debug mode) with a handler of its own: swap it for one that keeps cut, copy,
    paste, select all, spelling… and drops only the browser-ish items."""
    from gi.repository import GObject

    webkit = gtk.webkit
    actions = webkit.ContextMenuAction
    blocked = {getattr(actions, name) for name in _GTK_MENU_BLOCK if hasattr(actions, name)}

    def on_menu(webview, menu, *rest):
        try:
            return gtk_menu_filter(menu, blocked)
        except Exception:
            return True  # never a menu that could not be cleaned

    def hook(view):
        signal = GObject.signal_lookup("context-menu", webkit.WebView)
        found = GObject.signal_handler_find(view.webview, GObject.SignalMatchType.ID, signal, 0, None, None, None)
        if found:  # pywebview's (none in debug mode: then WebKit's whole menu stays)
            view.webview.disconnect(found)
            view._nesi_menu = view.webview.connect("context-menu", on_menu)

    _gtk_after_init(gtk, hook)


def gtk_permission_decision(request, page_url: str, port: int, webkit) -> str | None:
    """What the Linux window answers a WebKitGTK permission request: "allow" the microphone alone to the app's own
    page, "deny" any other capture (camera, screen, or a page that is not the app), None (WebKit's own answer: refused)
    for everything else."""
    if not isinstance(request, webkit.UserMediaPermissionRequest):
        return None
    audio, video = bool(request.props.is_for_audio_device), bool(request.props.is_for_video_device)
    is_display = getattr(webkit, "user_media_permission_is_for_display_device", None)
    display = bool(is_display(request)) if is_display is not None else False
    return "allow" if audio and not video and not display and is_app_url(page_url, port) else "deny"


def _gtk_microphone(gtk, port: int) -> None:
    """WebKitGTK refuses every capture unless the window answers its permission requests (the mic button)."""
    webkit = gtk.webkit

    def on_permission(webview, request):
        try:
            answer = gtk_permission_decision(request, str(webview.get_uri() or ""), port, webkit)
        except Exception:
            answer = "deny"
        if answer is None:
            return False
        (request.allow if answer == "allow" else request.deny)()
        return True

    _gtk_after_init(gtk, lambda view: view.webview.connect("permission-request", on_permission))


def _allow_microphone(port: int) -> None:
    """Let the app's own page use the microphone (mic button) without WebKit asking on every recording.
    macOS still asks the user once for the app (NSMicrophoneUsageDescription in Info.plist)."""
    if os.name == "nt":
        return  # WebView2 on Windows shows its own permission prompt
    if sys.platform != "darwin":
        try:
            from webview.platforms import gtk

            _gtk_microphone(gtk, port)
        except Exception as exc:  # the mic button then says the microphone is not allowed
            print("nesicode: could not allow the microphone:", exc)
        return
    try:
        import objc
        from webview.platforms import cocoa

        sel = b"webView:requestMediaCapturePermissionForOrigin:initiatedByFrame:type:decisionHandler:"
        objc.registerMetaDataForSelector(b"NSObject", sel, {
            "arguments": {6: {"callable": {"retval": {"type": b"v"},
                                           "arguments": {0: {"type": b"^v"}, 1: {"type": b"q"}}}}},
        })

        def grant(self, webview, origin, frame, kind, decision_handler):
            try:
                ours = (origin is not None and str(origin.host() or "") == "127.0.0.1" and int(origin.port()) == port
                        and (frame is None or frame.isMainFrame()))
            except Exception:
                ours = False
            # 1 = WKPermissionDecisionGrant, 2 = Deny: only our own page (never a preview iframe)
            decision_handler(1 if ours else 2)

        objc.classAddMethods(cocoa.BrowserView.BrowserDelegate, [
            objc.selector(grant, selector=sel, signature=b"v@:@@@q@?"),
        ])
    except Exception as exc:  # the mic button then falls back to WebKit's own prompt
        print("nesicode: could not pre-authorize the microphone:", exc)


def _restore_child_env() -> None:
    """The Linux bundle's launcher puts the bundle's own library folder first in LD_LIBRARY_PATH (keeping the user's
    value in LD_LIBRARY_PATH_ORIG). Everything this app starts (the agent's commands, git, the browser, WebKit's own
    processes) must load the system's libraries, as when the user runs them: give them back the user's value. This
    process is unaffected: the dynamic linker read the variable when it started."""
    if not (getattr(sys, "frozen", False) and sys.platform.startswith("linux")):
        return
    if "LD_LIBRARY_PATH_ORIG" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = os.environ.pop("LD_LIBRARY_PATH_ORIG")
        return
    bundle = os.path.realpath(getattr(sys, "_MEIPASS", "") or os.path.dirname(sys.executable))
    current = os.environ.get("LD_LIBRARY_PATH")
    if current is None:
        return
    rest = [p for p in current.split(os.pathsep) if p and os.path.realpath(p) != bundle]
    if rest:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(rest)
    else:
        del os.environ["LD_LIBRARY_PATH"]


def _prepare_gtk() -> None:
    """Linux, before GTK starts. The window gets the name the menu entry (nesicode.desktop, StartupWMClass=Nesicode)
    knows, so the dock shows Nesicode's icon. WebKitGTK's DMA-BUF renderer leaves the window blank on some GPUs
    (NVIDIA's own driver) and virtual machines: draw through shared memory unless the user says otherwise."""
    os.environ.setdefault("WEBKIT_DISABLE_DMABUF_RENDERER", "1")
    try:
        from gi.repository import GLib

        GLib.set_prgname("nesicode")
        GLib.set_application_name("Nesicode")
    except Exception as exc:
        print("nesicode: GTK is not available:", exc, file=sys.stderr)


def _window_icon() -> str | None:
    """The Linux window's icon (X11; Wayland takes the menu entry's): the installed 256 px one, else the bundle's or
    the source checkout's (1024 px)."""
    here = Path(__file__).resolve()
    for p in (Path("/usr/share/icons/hicolor/256x256/apps/nesicode.png"),
              Path(getattr(sys, "_MEIPASS", "") or "/nonexistent") / "nesicode.png",
              here.parents[2] / "packaging" / "icons" / "nesicode.png"):
        if p.is_file():
            return str(p)
    return None


class _HelperThread(threading.Thread):
    """A pywebview helper thread that cannot keep the app alive once its window is gone."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("daemon", True)
        super().__init__(*args, **kwargs)


def _daemon_bridge_threads(util=None) -> None:
    """pywebview injects its script and runs each bridge call in a thread of its own that waits, with no time limit,
    for the window to answer. Closed meanwhile (right after it opened, while the page loaded), the window never
    answers: the app quit its server but its process stayed alive, invisible (seen on Ubuntu). Those threads must not
    keep the process alive after the window is gone."""
    if util is None:
        try:
            import webview.util as util
        except Exception:
            return
    if getattr(util, "Thread", None) is threading.Thread:
        util.Thread = _HelperThread


def _browser_fallback(url: str, server, thread: threading.Thread, stop_runtime) -> None:
    webbrowser.open(url)
    try:
        while thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    _close(server, thread, stop_runtime)


def _close(server, thread: threading.Thread, stop_runtime) -> None:
    """The window (or the browser fallback) is gone: stop every run and MCP server first, then the server. Done here
    rather than left to the server's own shutdown, which a slow connection delays and a quitting interpreter can cut
    short (the server thread is a daemon)."""
    stop_runtime()
    server.should_exit = True
    thread.join(timeout=5)


def main() -> None:
    # Desktop defaults, before anything imports nesicode.config / the server.
    _restore_child_env()  # first: the login shell below is the first program this app starts
    _adopt_login_path()
    os.environ["NESICODE_DESKTOP"] = "1"
    os.environ.setdefault("NESICODE_AUTH", "false")
    token = secrets.token_urlsafe(32)
    os.environ["NESICODE_DESKTOP_TOKEN"] = token

    from .config import app_data_dir, settings

    app_data_dir().mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    log = open(app_data_dir() / "nesicode.log", "a", buffering=1, encoding="utf-8")
    if getattr(sys, "frozen", False):  # no console in the bundles: keep a log instead
        sys.stdout = sys.stderr = log

    import uvicorn

    from .server.app import app, stop_runtime

    port = _free_port()
    # (an open response stream must not keep a closing app waiting for long)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False,
                                           timeout_graceful_shutdown=3))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    if not _wait(base + "/api/health"):
        print("nesicode: the server did not start; see", app_data_dir() / "nesicode.log", file=log)
        sys.exit(1)
    url = f"{base}/?token={token}"

    try:
        import webview
    except Exception:  # no native webview available: fall back to the default browser
        _browser_fallback(url, server, thread, stop_runtime)
        return
    linux = sys.platform.startswith("linux")
    if linux:
        _prepare_gtk()  # before anything imports GTK (pywebview's backend starts it on import)

    class Api:
        """Exposed to the page as window.pywebview.api (native dialogs the browser cannot do)."""

        def pick_folder(self):
            kind = getattr(getattr(webview, "FileDialog", None), "FOLDER", None) or getattr(webview, "FOLDER_DIALOG", 20)
            result = webview.windows[0].create_file_dialog(kind)
            if not result:
                return None
            return result[0] if isinstance(result, (list, tuple)) else str(result)

        def reveal(self, path: str):
            """Show a folder or file in Finder / Explorer."""
            import subprocess

            if not os.path.exists(path):
                return False
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path])
            elif os.name == "nt":
                subprocess.Popen(["explorer", "/select,", path])
            else:
                subprocess.Popen(["xdg-open", path if os.path.isdir(path) else os.path.dirname(path)])
            return True

        def open_external(self, url: str):
            """Links to other sites (and e-mail links) open in the user's browser or mail app, never in the window."""
            if isinstance(url, str) and window_navigation(url, port) == "browser":
                webbrowser.open(url)
                return True
            return False

        def save_file(self, path: str, suggested_name: str):
            """Native 'Save as…' for a file served by this app (e.g. /api/threads/3/file?path=...)."""
            if not path.startswith("/api/"):
                return False
            kind = getattr(getattr(webview, "FileDialog", None), "SAVE", None) or getattr(webview, "SAVE_DIALOG", 30)
            target = webview.windows[0].create_file_dialog(kind, save_filename=suggested_name or "file")
            if not target:
                return False
            target = target[0] if isinstance(target, (list, tuple)) else str(target)
            req = urllib.request.Request(base + path, headers={"Cookie": f"nesi_desktop={token}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(target, "wb") as f:
                f.write(r.read())
            return True

    _daemon_bridge_threads()
    _enable_native_context_menu(webview)
    _allow_microphone(port)
    _harden_web_content()
    kept = guard_navigation(port)  # these two before the window exists: pywebview builds its delegate and bridge with it
    guarded = guard_js_bridge(port)
    # WebKit (macOS, Linux) must have the per-frame lock: it hands the bridge to every frame. Without it, no native
    # API at all: folder picking, the file manager and "Save as…" fall back to what the page can do itself. On Linux
    # the lock is a key given to the page in the window, so the window must also be kept on the app's pages.
    safe = bridge_is_safe(guarded) and (not linux or "gtk" in kept)
    if not safe:
        print("nesicode: the desktop bridge could not be secured; native dialogs are disabled", file=sys.stderr)
    window = webview.create_window("Nesicode", url, width=1360, height=880, min_size=(900, 600), text_select=True,
                                   js_api=Api() if safe else None)
    options: dict = {"private_mode": False, "storage_path": str(app_data_dir() / "webview")}
    if linux:  # the guards above are WebKitGTK's (pywebview would try Qt first on KDE)
        options.update(gui="gtk", icon=_window_icon())
    try:
        webview.start(**options)
    except Exception as exc:
        shown = getattr(getattr(window, "events", None), "shown", None)
        if not linux or (shown is not None and shown.is_set()):
            raise
        # GTK or WebKitGTK missing or broken (a source checkout without PyGObject…): the app in the browser instead
        print("nesicode: the window could not open (GTK / WebKitGTK):", exc, file=sys.stderr)
        _browser_fallback(url, server, thread, stop_runtime)
        return
    _close(server, thread, stop_runtime)


if __name__ == "__main__":
    main()
