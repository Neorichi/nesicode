"""nesicode desktop: starts the server on a private port and shows it in a native window.

Used by the macOS .app and the Windows installer (PyInstaller bundles), and by
`nesicode desktop` from a source checkout.
"""

from __future__ import annotations

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


def _guarded_js_bridge_call(original):
    def js_bridge_call(window, func_name, param, value_id):
        if not bridge_call_ok(func_name, value_id):
            print("nesicode: blocked a desktop bridge call with an invalid name or id", file=sys.stderr)
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
        if name == "cocoa" and "cocoa" not in _guarded_bridge:
            try:
                _guarded_bridge["cocoa"] = _guard_cocoa_bridge(mod, port)
            except Exception as exc:
                print("nesicode: could not guard the macOS bridge:", exc, file=sys.stderr)
                continue
        done.append(name)
    return done


def _guard_cocoa_bridge(cocoa, port: int):
    """WKWebView tells who sent each message (WKScriptMessage.frameInfo): drop all but the app page's."""
    import objc

    base = cocoa.BrowserView.JSBridge

    class NesiJSBridge(base):
        def userContentController_didReceiveScriptMessage_(self, controller, message):
            if trusted_frame(message, port):
                objc.super(NesiJSBridge, self).userContentController_didReceiveScriptMessage_(controller, message)
            else:
                print("nesicode: blocked a desktop bridge call from a frame that is not the app", file=sys.stderr)

    cocoa.BrowserView.JSBridge = NesiJSBridge  # pywebview creates BrowserView.JSBridge for every window
    return NesiJSBridge


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


def guard_navigation(port: int) -> list[str]:
    """Keep the app window on the app (see above). Call it before the window exists. Returns the backends guarded."""
    import importlib

    _guarded_nav["port"] = port
    name = "cocoa" if sys.platform == "darwin" else "edgechromium" if os.name == "nt" else None
    if name is None:
        return []
    if name not in _guarded_nav:
        try:
            mod = importlib.import_module(f"webview.platforms.{name}")
            _guarded_nav[name] = _guard_cocoa_navigation(mod) if name == "cocoa" else _guard_edge_navigation(mod)
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


def _allow_microphone(port: int) -> None:
    """Let the app's own page use the microphone (mic button) without WebKit asking on every recording.
    macOS still asks the user once for the app (NSMicrophoneUsageDescription in Info.plist)."""
    if sys.platform != "darwin":
        return  # WebView2 on Windows shows its own permission prompt
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


def _close(server, thread: threading.Thread, stop_runtime) -> None:
    """The window (or the browser fallback) is gone: stop every run and MCP server first, then the server. Done here
    rather than left to the server's own shutdown, which a slow connection delays and a quitting interpreter can cut
    short (the server thread is a daemon)."""
    stop_runtime()
    server.should_exit = True
    thread.join(timeout=5)


def main() -> None:
    # Desktop defaults, before anything imports nesicode.config / the server.
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
        webbrowser.open(url)
        try:
            while thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        _close(server, thread, stop_runtime)
        return

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

    _enable_native_context_menu(webview)
    _allow_microphone(port)
    guard_navigation(port)  # these two before the window exists: pywebview builds its delegate and bridge with it
    guarded = guard_js_bridge(port)
    # macOS must have the per-frame check (WebKit hands the bridge to every frame); without any guard,
    # no native API at all: folder picking, Finder and "Save as…" fall back to what the page can do itself
    safe = "cocoa" in guarded if sys.platform == "darwin" else bool(guarded)
    if not safe:
        print("nesicode: the desktop bridge could not be secured; native dialogs are disabled", file=sys.stderr)
    webview.create_window("Nesicode", url, width=1360, height=880, min_size=(900, 600), text_select=True,
                          js_api=Api() if safe else None)
    webview.start(private_mode=False, storage_path=str(app_data_dir() / "webview"))
    _close(server, thread, stop_runtime)


if __name__ == "__main__":
    main()
