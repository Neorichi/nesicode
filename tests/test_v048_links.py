"""A link must never replace the app's page: the desktop window has no address bar and no Back button.

A user clicked a link in an answer and the whole window turned into that website, with no way back but quitting the
app. The page's own link handler missed clicks inside modals (they stop the click before it reaches `document`) and
never sees the context menu's «Abrir enlace» or a script; the window now also turns such navigations away natively.
The real window was checked end to end with a hidden pywebview window (links in and out of modals, a direct
navigation, file: and /api/ pages, target=_blank, mailto:); these tests pin the logic and the glue offline.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from nesicode import desktop

PORT = 47821
APP = f"http://127.0.0.1:{PORT}"


# ------------------------------------------------------------------ the desktop window (desktop.py)
def test_the_window_stays_on_the_app_and_other_sites_go_to_the_browser():
    nav = desktop.window_navigation
    # the app's pages, including paths the server hands to the page (GET /hooks/… is the page: the hooks are POST)
    for url in (f"{APP}/?token=abc", f"{APP}/p/3/c/9", APP, f"{APP}/new#x", "about:blank", f"{APP}/api",
                f"{APP}/api%252Fhealth", f"{APP}/hooks/3", f"{APP}/docsx", f"{APP}//api/health"):
        assert nav(url, PORT) == "allow", url
    for url in ("https://tmpfiles.org/wwA5XFP25KYY/photo.png", "http://example.com/", "HTTPS://EXAMPLE.COM/x",
                "mailto:someone@example.com", "http://127.0.0.1:8420/", "http://localhost:47821/",
                f"https://127.0.0.1:{PORT}/", f"http://127.0.0.1:{PORT}@evil.example/"):
        assert nav(url, PORT) == "browser", url
    # what the server answers with raw output (uvicorn decodes the path once before routing), local files, schemes
    for url in (f"{APP}/api/health", f"{APP}/api/threads/3/file?path=a.html", f"{APP}/%61pi/health",
                f"{APP}/api%2Fhealth", f"{APP}/assets/index-abc.js", f"{APP}/assets", f"{APP}/docs", f"{APP}/docs/",
                f"{APP}/docs/oauth2-redirect", f"{APP}/redoc", f"{APP}/openapi.json", "file:///etc/hosts",
                "javascript:alert(1)", "data:text/html,hi", f"blob:{APP}/0f1e", "about:srcdoc", "",
                "vscode://file/x", "http:///no-host", "http://127.0.0.1:99999/", "http://[::1"):
        assert nav(url, PORT) == "block", url


class _URL:
    def __init__(self, text):
        self.text = text

    def absoluteString(self):  # noqa: N802
        return self.text


class _Request:
    def __init__(self, url):
        self.url = url

    def URL(self):  # noqa: N802
        return _URL(self.url) if self.url is not None else None


class _TargetFrame:
    def __init__(self, main):
        self.main = main

    def isMainFrame(self):  # noqa: N802
        return self.main


class _Action:
    def __init__(self, frame, url, kind=0):
        self.frame, self.url, self.kind = frame, url, kind

    def targetFrame(self):  # noqa: N802
        return self.frame

    def navigationType(self):  # noqa: N802
        return self.kind

    def request(self):
        return _Request(self.url)


LINK, OTHER = 0, -1  # WKNavigationTypeLinkActivated, WKNavigationTypeOther


def test_new_windows_open_only_websites_and_mail_links_the_user_clicked():
    """target=_blank links and window.open (the previews allow popups): pywebview handed any scheme to the system."""
    decide = desktop.new_window_decision
    for url in ("https://example.com/x", "mailto:a@b.example"):
        assert decide(_Action(None, url, LINK), PORT, LINK) == ("browser", url)
    for url in ("smb://evil.example/share", "file://evil.example/share/x", "file:///etc/hosts", "vscode://x",
                "tel:+34600000000", f"{APP}/api/threads/3/file?path=b.html", f"{APP}/p/3"):
        assert decide(_Action(None, url, LINK), PORT, LINK) == ("none", url), url
    assert decide(_Action(None, "https://example.com/x", OTHER), PORT, LINK) == ("none", "")  # window.open by a script
    assert decide(_Broken(), PORT, LINK) == ("none", "")


class _Broken:
    def targetFrame(self):  # noqa: N802
        raise RuntimeError("unexpected")

    def navigationType(self):  # noqa: N802
        raise RuntimeError("unexpected")


def test_only_the_main_frame_is_judged():
    """Frames inside the page (sandboxed previews) and new windows (target=_blank, which pywebview already sends to
    the browser) keep their own behaviour, and anything unexpected falls back to pywebview's."""
    decide = desktop.navigation_decision
    out = "https://example.com/x"
    assert decide(_Action(_TargetFrame(True), out), PORT) == ("browser", out)
    assert decide(_Action(_TargetFrame(True), f"{APP}/api/health"), PORT) == ("block", f"{APP}/api/health")
    assert decide(_Action(_TargetFrame(True), f"{APP}/p/1"), PORT) == ("allow", f"{APP}/p/1")
    assert decide(_Action(_TargetFrame(False), out), PORT) == ("allow", "")
    assert decide(_Action(None, out), PORT) == ("allow", "")
    assert decide(_Action(_TargetFrame(True), None), PORT) == ("allow", "")
    assert decide(_Broken(), PORT) == ("allow", "")


def _wait_for(items, n=1, limit=3.0):
    t0 = time.monotonic()
    while len(items) < n and time.monotonic() - t0 < limit:
        time.sleep(0.02)
    return items


def test_windows_window_turns_away_other_pages(monkeypatch):
    """WebView2 (Windows): NavigationStarting is cancelled and other sites go to the browser; the app's own pages
    reach pywebview's handler as before."""
    calls, new_windows, opened = [], [], []

    class EdgeChrome:
        def on_navigation_start(self, sender, args):
            calls.append(args)

        def on_new_window_request(self, sender, args):
            new_windows.append(args)

    class Args:
        def __init__(self, uri):
            self.Uri, self.Cancel = uri, False

    class NoUri:
        Cancel = False

        @property
        def Uri(self):  # noqa: N802
            raise RuntimeError("no uri")

    class NewWindowArgs:  # CoreWebView2NewWindowRequestedEventArgs, as pywebview calls it
        def __init__(self, uri):
            self.uri, self.handled = uri, False

        def get_Uri(self):  # noqa: N802
            if self.uri is None:
                raise RuntimeError("no uri")
            return self.uri

        def set_Handled(self, value):  # noqa: N802
            self.handled = value

    monkeypatch.setitem(desktop._guarded_nav, "port", PORT)
    monkeypatch.setattr(desktop.webbrowser, "open", lambda url, *a, **k: opened.append(url))
    fake = types.SimpleNamespace(EdgeChrome=EdgeChrome)
    desktop._guard_edge_navigation(fake)
    view = EdgeChrome()

    home = Args(f"{APP}/?token=abc")
    view.on_navigation_start(None, home)
    assert calls == [home] and home.Cancel is False

    away = Args("https://tmpfiles.org/x/photo.png")
    view.on_navigation_start(None, away)
    assert away.Cancel is True and _wait_for(opened) == ["https://tmpfiles.org/x/photo.png"]

    raw = Args(f"{APP}/api/health")
    view.on_navigation_start(None, raw)
    assert raw.Cancel is True and calls == [home]
    time.sleep(0.1)
    assert opened == ["https://tmpfiles.org/x/photo.png"]  # an API file is never sent anywhere

    odd = NoUri()
    view.on_navigation_start(None, odd)
    assert odd.Cancel is False and calls == [home, odd]  # unexpected: pywebview's own handling

    # new windows (target=_blank, window.open): only websites and mail links reach the system, never a popup window
    site, share = NewWindowArgs("https://example.org/n"), NewWindowArgs("file://evil.example/share/x")
    view.on_new_window_request(None, site)
    view.on_new_window_request(None, share)
    assert site.handled is True and share.handled is True and new_windows == []
    assert _wait_for(opened, 2) == ["https://tmpfiles.org/x/photo.png", "https://example.org/n"]
    time.sleep(0.1)
    assert len(opened) == 2  # the share was never handed to Explorer
    broken = NewWindowArgs(None)
    view.on_new_window_request(None, broken)
    assert new_windows == [broken]  # unexpected: pywebview's own handling


@pytest.mark.skipif(sys.platform != "darwin", reason="WKWebView navigation delegate (macOS)")
def test_macos_window_cancels_navigations_away_from_the_app(monkeypatch):
    cocoa = pytest.importorskip("webview.platforms.cocoa")
    opened, decisions = [], []
    monkeypatch.setattr(desktop.webbrowser, "open", lambda url, *a, **k: opened.append(url))

    assert desktop.guard_navigation(PORT) == ["cocoa"]
    assert cocoa.BrowserView.BrowserDelegate.__name__ == "NesiBrowserDelegate"
    assert desktop.guard_navigation(PORT) == ["cocoa"]  # twice in one process: the same class, no error
    delegate = cocoa.BrowserView.BrowserDelegate.alloc().init()
    cancel = getattr(cocoa.WebKit, "WKNavigationActionPolicyCancel", 0)

    def handler(decision):
        decisions.append(decision)

    delegate.webView_decidePolicyForNavigationAction_decisionHandler_(
        None, _Action(_TargetFrame(True), "https://tmpfiles.org/x/photo.png"), handler)
    assert decisions == [cancel] and _wait_for(opened) == ["https://tmpfiles.org/x/photo.png"]

    delegate.webView_decidePolicyForNavigationAction_decisionHandler_(
        None, _Action(_TargetFrame(True), "file:///etc/hosts"), handler)
    delegate.webView_decidePolicyForNavigationAction_decisionHandler_(
        None, _Action(_TargetFrame(True), f"{APP}/%61pi/health"), handler)
    assert decisions == [cancel, cancel, cancel]
    time.sleep(0.1)
    assert opened == ["https://tmpfiles.org/x/photo.png"]

    # new windows: a clicked target=_blank link to a website goes to the browser; a share, or a script's
    # window.open, goes nowhere; and no second web view is ever created
    link = getattr(cocoa.WebKit, "WKNavigationTypeLinkActivated", 0)
    other = getattr(cocoa.WebKit, "WKNavigationTypeOther", -1)
    new = delegate.webView_createWebViewWithConfiguration_forNavigationAction_windowFeatures_
    assert new(None, None, _Action(None, "https://example.org/n", link), None) is None
    assert _wait_for(opened, 2)[-1] == "https://example.org/n"
    assert new(None, None, _Action(None, "smb://evil.example/share", link), None) is None
    assert new(None, None, _Action(None, "https://example.org/popup", other), None) is None
    time.sleep(0.1)
    assert opened == ["https://tmpfiles.org/x/photo.png", "https://example.org/n"]


# ------------------------------------------------------------------ the page (web/src/logic.ts), run with node
WEB = Path(__file__).resolve().parents[1] / "web"
NODE = shutil.which("node")
ESBUILD = WEB / "node_modules" / "esbuild"


def run_ts(tmp_path: Path, body: str):
    """Bundle `body` (TypeScript, with `L` = web/src/logic.ts) for node with the esbuild Vite ships, run it, return
    the JSON it printed (`out`). UTF-8 both ways."""
    if not NODE or not ESBUILD.is_dir():
        if os.environ.get("CI"):  # a release build runs it, never skips it
            pytest.fail("needs node and web/node_modules (esbuild): run `npm ci` in web/ first", pytrace=False)
        pytest.skip("needs node and web/node_modules (esbuild)")
    entry = tmp_path / "entry.ts"
    entry.write_text(f"import * as L from {json.dumps(str(WEB / 'src' / 'logic.ts'))}\n"
                     f"const out: Record<string, unknown> = {{}}\n{body}\nconsole.log(JSON.stringify(out))\n",
                     encoding="utf-8")
    bundle, build = tmp_path / "bundle.cjs", tmp_path / "build.cjs"
    build.write_text(
        f"require({json.dumps(str(ESBUILD))}).buildSync({{entryPoints: [{json.dumps(str(entry))}], bundle: true, "
        f"platform: 'node', format: 'cjs', outfile: {json.dumps(str(bundle))}, logLevel: 'error', "
        f"nodePaths: [{json.dumps(str(WEB / 'node_modules'))}]}})\n", encoding="utf-8")
    for args in ([NODE, str(build)], [NODE, str(bundle)]):
        p = subprocess.run(args, capture_output=True, encoding="utf-8", timeout=60, cwd=tmp_path)
        assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_the_page_sorts_links_before_following_them(tmp_path):
    out = run_ts(tmp_path, """
      const page = 'http://127.0.0.1:47821/p/3/c/9?x=1'
      const sort = (hrefs: string[]) => hrefs.map((h) => L.linkAction(h, page))
      out.app = sort(['/p/3', '#fn1', 'http://127.0.0.1:47821/new', '', 'other', '/api', '/api%252Fhealth', '/hooks/3',
                      '/docsx'])
      out.external = sort(['https://tmpfiles.org/wwA5XFP25KYY/photo.png', 'mailto:a@b.example', '//evil.example/x',
                           'http://127.0.0.1:8420/', 'http://localhost:47821/', ' https://example.com/ '])
      out.file = sort(['/api/threads/3/file?path=a.html', 'http://127.0.0.1:47821/api/health', '/%61pi/health',
                       '/api%2Fhealth', '/assets/index-abc.js', '/assets', '/docs', '/docs/', '/redoc', '/openapi.json'])
      out.none = sort(['file:///etc/hosts', 'javascript:alert(1)', 'vscode://x', 'data:text/html,x', 'http://[bad',
                       'blob:http://127.0.0.1:47821/0f1e', 'tel:+34600000000'])
    """)
    assert out["app"] == ["app"] * 9
    assert out["external"] == ["external"] * 6
    assert out["file"] == ["file"] * 10
    assert out["none"] == ["none"] * 7


def test_the_page_follows_only_what_cannot_replace_it(tmp_path):
    """The desktop window follows only the app's pages; a browser keeps downloads (the «Descargar» buttons click a
    same-origin /api/ link with `download`), modifier clicks (new tab or window) and other apps' links (tel:)."""
    out = run_ts(tmp_path, """
      const click = (action: L.LinkAction, desktop: boolean, download = false, modified = false) =>
        L.linkClick(action, { desktop, download, modified })
      out.desktop = [click('app', true), click('external', true), click('file', true), click('file', true, true),
                     click('none', true), click('external', true, false, true), click('none', true, false, true)]
      out.browser = [click('app', false), click('external', false), click('file', false), click('file', false, true),
                     click('none', false), click('external', false, false, true), click('file', false, false, true)]
    """)
    assert out["desktop"] == ["follow", "external", "stop", "stop", "stop", "external", "stop"]
    assert out["browser"] == ["follow", "external", "external", "follow", "follow", "follow", "follow"]
