"""nesicode desktop: starts the server on a private port and shows it in a native window.

Used by the macOS .app and the Windows installer (PyInstaller bundles), and by
`nesicode desktop` from a source checkout.
"""

from __future__ import annotations

import os
import secrets
import socket
import sys
import threading
import time
import urllib.request
import webbrowser


PREFERRED_PORT = 47821  # stable, so the Chrome extension's connection code keeps working across launches


def _free_port() -> int:
    for port in (PREFERRED_PORT, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
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
                             text=True, timeout=10).stdout
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


def _allow_microphone() -> None:
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
            host = str(origin.host() or "") if origin is not None else ""
            # 1 = WKPermissionDecisionGrant, 2 = Deny: only our own local page
            decision_handler(1 if host in ("127.0.0.1", "localhost") else 2)

        objc.classAddMethods(cocoa.BrowserView.BrowserDelegate, [
            objc.selector(grant, selector=sel, signature=b"v@:@@@q@?"),
        ])
    except Exception as exc:  # the mic button then falls back to WebKit's own prompt
        print("nesicode: could not pre-authorize the microphone:", exc)


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

    from .server.app import app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False))
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
            """External links open in the user's browser, never inside the app window."""
            if url.startswith(("http://", "https://")) and not url.startswith(base):
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
    _allow_microphone()
    webview.create_window("Nesicode", url, width=1360, height=880, min_size=(900, 600), text_select=True, js_api=Api())
    webview.start(private_mode=False, storage_path=str(app_data_dir() / "webview"))
    server.should_exit = True
    thread.join(timeout=5)


if __name__ == "__main__":
    main()
