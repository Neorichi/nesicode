"""The Linux desktop app (0.4.9): the .deb for Ubuntu and Debian and its window, pywebview's WebKitGTK backend.

The window gets the same guards as on macOS and Windows, adapted to what WebKitGTK tells: its navigation policy does
not say which frame navigates, so the main frame is turned away when its response arrives (and the browser opens
when its load starts); its bridge does not say which frame sent a message, so the app page signs each call with a
key made at launch. The real window was checked end to end on Ubuntu (scripts/desktop_window_check.py, CI under
Xvfb) and the installed .deb on Ubuntu 22.04/24.04/26.04 and Debian 12/13 (packaging/linux/smoke.sh); these tests
pin the logic and the glue offline, with stand-ins for WebKitGTK.
"""

import importlib.util
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
REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wait_for(items, n=1, limit=3.0):
    t0 = time.monotonic()
    while len(items) < n and time.monotonic() - t0 < limit:
        time.sleep(0.02)
    return items


# ------------------------------------------------------------------ stand-ins for pywebview's GTK backend
class _Enum:
    def __init__(self, **values):
        self.__dict__.update(values)


class _UserMedia:
    def __init__(self, audio=True, video=False):
        self.props = types.SimpleNamespace(is_for_audio_device=audio, is_for_video_device=video)
        self.answer = None

    def allow(self):
        self.answer = "allow"

    def deny(self):
        self.answer = "deny"


class _Display(_UserMedia):
    pass


def _webkit():
    return types.SimpleNamespace(
        PolicyDecisionType=_Enum(NAVIGATION_ACTION=0, NEW_WINDOW_ACTION=1, RESPONSE=2),
        NavigationType=_Enum(LINK_CLICKED=0, OTHER=5),
        LoadEvent=_Enum(STARTED=0, REDIRECTED=1, COMMITTED=2, FINISHED=3),
        UserMediaPermissionRequest=_UserMedia,
        user_media_permission_is_for_display_device=lambda r: isinstance(r, _Display),
    )


class _WebView:
    def __init__(self):
        self.uri, self.handlers = "", {}

    def get_uri(self):
        return self.uri

    def connect(self, signal, fn):
        self.handlers.setdefault(signal, []).append(fn)
        return len(self.handlers[signal])


def _gtk():
    calls = {"policy": [], "load": [], "bridge": []}

    class BrowserView:
        def __init__(self, window=None):
            self.webview = _WebView()

        def on_navigation(self, webview, decision, decision_type):
            calls["policy"].append((decision, decision_type))

        def on_load_finish(self, webview, status):
            calls["load"].append(status)

        def on_js_bridge_call(self, manager, message):
            calls["bridge"].append(message)

    return types.SimpleNamespace(webkit=_webkit(), BrowserView=BrowserView), calls


class _Request:
    def __init__(self, uri):
        self.uri = uri

    def get_uri(self):
        if self.uri is None:
            raise RuntimeError("no uri")
        return self.uri


class _Action:
    def __init__(self, uri, nav_type=0, frame=""):
        self.uri, self.nav_type, self.frame = uri, nav_type, frame

    def get_navigation_type(self):
        return self.nav_type

    def get_request(self):
        return _Request(self.uri)

    def get_frame_name(self):
        return self.frame


class _Decision:
    def __init__(self, uri=None, nav_type=0, frame="", main=None):
        self.action, self.main, self.ignored = _Action(uri, nav_type, frame), main, False
        self.uri = uri

    def get_navigation_action(self):
        return self.action

    def get_request(self):
        return _Request(self.uri)

    def is_main_frame_main_resource(self):
        if self.main is None:
            raise AttributeError("WebKitGTK < 2.40")
        return self.main

    def ignore(self):
        self.ignored = True


@pytest.fixture
def opened(monkeypatch):
    out = []
    monkeypatch.setitem(desktop._guarded_nav, "port", PORT)
    monkeypatch.setattr(desktop.webbrowser, "open", lambda url, *a, **k: out.append(url))
    return out


# ------------------------------------------------------------------ the window never leaves the app
def test_linux_new_windows_open_only_websites_and_mail_links_the_user_clicked(opened):
    gtk, calls = _gtk()
    desktop._guard_gtk_navigation(gtk)
    view = gtk.BrowserView()
    NEW, NAV = 1, 0

    site = _Decision("https://example.org/n")
    assert view.on_navigation(view.webview, site, NEW) is True and site.ignored
    assert _wait_for(opened) == ["https://example.org/n"]
    for d in (_Decision("file:///etc/passwd"), _Decision("smb://evil.example/share"),
              _Decision("https://example.org/popup", nav_type=5),  # window.open by a script
              _Decision(f"{APP}/api/health"), _Decision(None)):  # an API file; something unexpected
        assert view.on_navigation(view.webview, d, NEW) is True and d.ignored
    blank = _Decision("mailto:someone@example.com", frame="_blank")  # pywebview also took '_blank' navigations
    assert view.on_navigation(view.webview, blank, NAV) is True and blank.ignored
    assert _wait_for(opened, 2) == ["https://example.org/n", "mailto:someone@example.com"]
    time.sleep(0.1)
    assert len(opened) == 2 and calls["policy"] == []  # pywebview's handler (any scheme to the system) never ran


def test_linux_main_frame_is_turned_away_at_its_response_and_frames_are_left_alone(opened):
    gtk, calls = _gtk()
    desktop._guard_gtk_navigation(gtk)
    view = gtk.BrowserView()
    NAV, RESPONSE = 0, 2

    # a navigation's own policy does not say which frame: always WebKit's (pywebview's) answer
    for uri in ("https://example.org/x", f"{APP}/api/health", "file:///etc/hosts"):
        d = _Decision(uri)
        assert view.on_navigation(view.webview, d, NAV) is None and not d.ignored
    # responses of frames inside the page (the previews) too
    frame = _Decision("https://example.org/embed", main=False)
    assert view.on_navigation(view.webview, frame, RESPONSE) is None and not frame.ignored
    # the main frame: the app's pages go on, anything else is cancelled (other sites to the browser)
    page = _Decision(f"{APP}/projects", main=True)
    assert view.on_navigation(view.webview, page, RESPONSE) is None and not page.ignored
    away = _Decision("https://tmpfiles.org/x/photo.png", main=True)
    assert view.on_navigation(view.webview, away, RESPONSE) is True and away.ignored
    assert _wait_for(opened) == ["https://tmpfiles.org/x/photo.png"]
    view._nesi_away = False
    raw = _Decision(f"{APP}/%61pi/threads/1/file?path=x.html", main=True)
    assert view.on_navigation(view.webview, raw, RESPONSE) is True and raw.ignored
    old = _Decision("https://example.org/old", main=None)  # WebKitGTK < 2.40: cannot tell, left to WebKit
    assert view.on_navigation(view.webview, old, RESPONSE) is None and not old.ignored
    time.sleep(0.1)
    assert opened == ["https://tmpfiles.org/x/photo.png"]
    passed = [d for d, _ in calls["policy"]]  # what pywebview's own handler got: all but the two turned away
    assert len(passed) == 6 and away not in passed and raw not in passed


def test_linux_the_browser_opens_when_the_load_starts_once_and_the_page_keeps_its_bridge(opened):
    gtk, calls = _gtk()
    desktop._guard_gtk_navigation(gtk)
    view = gtk.BrowserView()
    STARTED, REDIRECTED, COMMITTED, FINISHED, RESPONSE = 0, 1, 2, 3, 2
    web = view.webview

    def load(uri, event):
        web.uri = uri
        return view.on_load_finish(web, event)

    # the app itself loads: pywebview sees every event and injects its bridge at the end
    for event in (STARTED, REDIRECTED, COMMITTED, FINISHED):
        load(f"{APP}/?token=t" if event == STARTED else f"{APP}/", event)
    assert calls["load"] == [STARTED, REDIRECTED, COMMITTED, FINISHED] and opened == []

    # a link to another site: the browser opens as soon as the main frame starts, once (not again at its response)
    calls["load"].clear()
    load("https://example.org/a", STARTED)
    assert _wait_for(opened) == ["https://example.org/a"]
    resp = _Decision("https://example.org/a", main=True)
    assert view.on_navigation(web, resp, RESPONSE) is True and resp.ignored
    web.uri = f"{APP}/"  # (WebKit restores the page's address when the navigation fails)
    load(f"{APP}/", FINISHED)
    assert calls["load"] == [STARTED]  # no FINISHED for pywebview: it would inject its bridge again
    time.sleep(0.1)
    assert opened == ["https://example.org/a"]

    # the same link again opens again; a redirect of it does not open a second time
    load("http://example.org/b", STARTED)
    load("https://example.org/b", REDIRECTED)
    load(f"{APP}/", FINISHED)
    assert _wait_for(opened, 2) == ["https://example.org/a", "http://example.org/b"]
    load("https://example.org/a", STARTED)
    load(f"{APP}/", FINISHED)
    assert _wait_for(opened, 3)[-1] == "https://example.org/a"

    # an app route that redirects to another site: turned away at the redirect
    load(f"{APP}/somewhere", STARTED)
    load("https://elsewhere.example/", REDIRECTED)
    assert _wait_for(opened, 4)[-1] == "https://elsewhere.example/"
    load(f"{APP}/", FINISHED)
    # an API file: never sent anywhere, the page stays
    calls["load"].clear()
    load(f"{APP}/api/health", STARTED)
    load(f"{APP}/", FINISHED)
    assert calls["load"] == [STARTED]
    time.sleep(0.1)
    assert len(opened) == 4


def test_linux_no_webkit_error_page_replaces_the_app(opened):
    gtk, _ = _gtk()
    desktop._guard_gtk_navigation(gtk)
    view = gtk.BrowserView()
    (failed,) = view.webview.handlers["load-failed"]
    assert failed(view.webview, 0, "https://nesicode-check.invalid/x", None) is True  # unknown host: no error page
    assert failed(view.webview, 0, "mailto:someone@example.com", None) is True
    assert failed(view.webview, 0, f"{APP}/api/health", None) is True
    assert failed(view.webview, 0, f"{APP}/", None) is False  # the app's own page failing: WebKit's page says why


def test_guard_navigation_picks_the_platform_backend(monkeypatch):
    seen = []
    monkeypatch.setattr(desktop, "_guarded_nav", {})
    monkeypatch.setattr(desktop.sys, "platform", "linux")
    monkeypatch.setattr(desktop, "_guard_gtk_navigation", lambda mod: seen.append(mod.__name__) or "guard")
    monkeypatch.setitem(sys.modules, "webview.platforms.gtk", types.ModuleType("webview.platforms.gtk"))
    assert desktop.guard_navigation(PORT) == ["gtk"] and seen == ["webview.platforms.gtk"]
    assert desktop.guard_navigation(PORT) == ["gtk"] and len(seen) == 1  # once per process


def test_window_setup_steps_run_after_each_window_and_never_cost_it():
    gtk, _ = _gtk()
    seen = []
    desktop._gtk_after_init(gtk, lambda view: seen.append(("a", view)))
    desktop._gtk_after_init(gtk, lambda view: 1 / 0)
    desktop._gtk_after_init(gtk, lambda view: seen.append(("b", view)))
    view = gtk.BrowserView()
    assert seen == [("a", view), ("b", view)] and isinstance(view.webview, _WebView)


# ------------------------------------------------------------------ the bridge answers the app page only
class _JSValue:
    def __init__(self, text):
        self.text = text

    def to_string(self):
        return self.text


class _Message:
    def __init__(self, body):
        self.value = _JSValue(body if isinstance(body, str) else json.dumps(body))

    def get_js_value(self):
        return self.value


def test_linux_bridge_answers_only_calls_signed_with_the_launch_key():
    gtk, calls = _gtk()
    util = types.SimpleNamespace(load_js_files=lambda window, platform: ("/*pywebview*/", "/*finish*/"))
    desktop._guard_gtk_bridge(gtk, util)
    js, finish = util.load_js_files(None, "gtkwebkit2")
    assert finish == "/*finish*/" and js.startswith("/*pywebview*/") and "nesiKey" in js
    key = json.loads(js.rsplit("})(", 1)[1].rsplit(");", 1)[0])
    assert len(key) >= 40
    assert util.load_js_files(None, "cocoa") == ("/*pywebview*/", "/*finish*/")  # other web views: untouched

    view = gtk.BrowserView()
    ok = _Message({"funcName": "open_external", "params": ["https://example.org"], "id": "1", "nesiKey": key})
    view.on_js_bridge_call(None, ok)
    assert calls["bridge"] == [ok]
    for bad in ({"funcName": "open_external", "params": [], "id": "2"},  # a preview: it cannot know the key
                {"funcName": "open_external", "params": [], "id": "3", "nesiKey": "guess"},
                {"funcName": "_pywebviewAlert", "params": "x", "id": "alert", "nesiKey": key[:-1]},
                {"funcName": "f", "params": [], "id": "4", "nesiKey": 7}, ["nesiKey", key], "not json"):
        view.on_js_bridge_call(None, _Message(bad))
    assert calls["bridge"] == [ok]
    assert desktop.gtk_bridge_message_ok(_JSValue(json.dumps({"nesiKey": key})), key)  # (a bare JS value too)


@pytest.mark.skipif(not shutil.which("node"), reason="node")
def test_linux_app_page_signs_every_bridge_call(tmp_path):
    """The script added to pywebview's: every API call and alert() carries the key; nothing else changes."""
    script = desktop._GTK_KEYED_CALLS % json.dumps("K3y")
    page = tmp_path / "page.js"
    page.write_text(
        "const posted = [];\n"
        "globalThis.window = globalThis;\n"
        "window.webkit = {messageHandlers: {jsBridge: {postMessage: m => posted.push(JSON.parse(m))}}};\n"
        "window.pywebview = {platform: 'gtkwebkit2', stringify: JSON.stringify,\n"
        "  _jsApiCallback: () => { throw new Error('unsigned') }};\n"
        + script +
        "window.pywebview._jsApiCallback('open_external', ['https://example.org'], '12');\n"
        "window.pywebview._jsApiCallback('pywebviewStateUpdate', {key: 'a', value: 1}, '13');\n"
        "window.alert('hi');\n"
        "console.log(JSON.stringify(posted));\n", encoding="utf-8")
    out = subprocess.run(["node", str(page)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    posted = json.loads(out.stdout)
    assert posted == [
        {"funcName": "open_external", "params": ["https://example.org"], "id": "12", "nesiKey": "K3y"},
        {"funcName": "pywebviewStateUpdate", "params": {"key": "a", "value": 1}, "id": "13", "nesiKey": "K3y"},
        {"funcName": "_pywebviewAlert", "params": "hi", "id": "alert", "nesiKey": "K3y"},
    ]


def test_the_native_api_needs_the_per_frame_lock_on_webkit():
    safe = desktop.bridge_is_safe
    assert safe(["gtk"], "linux") and not safe([], "linux") and not safe(["qt", "cef"], "linux")
    assert safe(["cocoa"], "darwin") and not safe(["gtk"], "darwin")
    assert safe(["edgechromium"], "win32") and not safe([], "win32")


# ------------------------------------------------------------------ microphone, context menu
def test_linux_microphone_only_for_the_app_page():
    webkit = _webkit()
    decide = desktop.gtk_permission_decision
    assert decide(_UserMedia(), f"{APP}/t/3", PORT, webkit) == "allow"
    assert decide(_UserMedia(audio=True, video=True), f"{APP}/t/3", PORT, webkit) == "deny"  # camera
    assert decide(_UserMedia(audio=False, video=True), f"{APP}/", PORT, webkit) == "deny"
    assert decide(_Display(), f"{APP}/", PORT, webkit) == "deny"  # the screen
    for page in ("about:blank", f"{APP}/api/threads/3/file?path=x.html", "https://example.org/", f"http://127.0.0.1:1/"):
        assert decide(_UserMedia(), page, PORT, webkit) == "deny"
    assert decide(object(), f"{APP}/", PORT, webkit) is None  # notifications, location…: WebKit's own (refused)


class _Item:
    def __init__(self, action=None):
        self.action = action

    def get_stock_action(self):
        return self.action

    def is_separator(self):
        return self.action is None


class _Menu:
    def __init__(self, items):
        self.items = list(items)

    def get_items(self):
        return list(self.items)

    def remove(self, item):
        self.items.remove(item)


def test_linux_context_menu_keeps_editing_and_drops_browser_items():
    copy, paste, reload_, back, inspect, open_new = 1, 2, 3, 4, 5, 6
    blocked = {reload_, back, inspect, open_new}
    menu = _Menu([_Item(back), _Item(reload_), _Item(), _Item(copy), _Item(), _Item(), _Item(paste), _Item(),
                  _Item(inspect)])
    assert desktop.gtk_menu_filter(menu, blocked) is False
    assert [i.action for i in menu.items] == [copy, None, paste]
    only_browser = _Menu([_Item(back), _Item(), _Item(open_new), _Item(inspect)])
    assert desktop.gtk_menu_filter(only_browser, blocked) is True and only_browser.items == []  # no menu at all


# ------------------------------------------------------------------ the bundle
def test_what_the_app_starts_gets_the_users_library_path_back(monkeypatch):
    monkeypatch.setattr(desktop.sys, "frozen", True, raising=False)
    monkeypatch.setattr(desktop.sys, "_MEIPASS", "/opt/nesicode/_internal", raising=False)
    monkeypatch.setattr(desktop.sys, "platform", "linux")

    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/nesicode/_internal:/home/u/lib")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/home/u/lib")
    desktop._restore_child_env()
    assert os.environ["LD_LIBRARY_PATH"] == "/home/u/lib" and "LD_LIBRARY_PATH_ORIG" not in os.environ

    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/nesicode/_internal")  # the user had none
    desktop._restore_child_env()
    assert "LD_LIBRARY_PATH" not in os.environ

    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/nesicode/_internal:/opt/cuda/lib64")
    desktop._restore_child_env()
    assert os.environ["LD_LIBRARY_PATH"] == "/opt/cuda/lib64"

    monkeypatch.setattr(desktop.sys, "platform", "darwin")  # only the Linux bundle
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/nesicode/_internal")
    desktop._restore_child_env()
    assert os.environ["LD_LIBRARY_PATH"] == "/opt/nesicode/_internal"


def test_the_linux_bundle_leaves_the_gtk_stack_to_the_system():
    bundle = _load("nesicode_linux_bundle", REPO / "packaging" / "linux" / "bundle.py")
    keep = [("libpython3.12.so.1.0", "/home/runner/.local/share/uv/python/cpython-3.12/lib/libpython3.12.so.1.0", "BINARY"),
            ("libpython3.12.so.1.0", "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0", "BINARY"),
            ("gi/_gi.cpython-312-x86_64-linux-gnu.so", "/w/.venv/lib/python3.12/site-packages/gi/_gi.cpython-312-x86_64-linux-gnu.so", "EXTENSION"),
            ("cffi.libs/libffi-1a2b3c.so.8.1.0", "/w/.venv/lib/python3.12/site-packages/cffi.libs/libffi-1a2b3c.so.8.1.0", "BINARY")]
    drop = [("libgtk-3.so.0", "/usr/lib/x86_64-linux-gnu/libgtk-3.so.0", "BINARY"),
            ("libwebkit2gtk-4.1.so.0", "/lib/x86_64-linux-gnu/libwebkit2gtk-4.1.so.0", "BINARY"),
            ("libstdc++.so.6", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6", "BINARY"),
            ("gio_modules/libgiognutls.so", "/usr/lib/x86_64-linux-gnu/gio/modules/libgiognutls.so", "BINARY")]
    assert [b for b in keep + drop if bundle.keep_binary(b)] == keep
    assert not bundle.keep_data(("gi_typelibs/WebKit2-4.1.typelib", "/usr/lib/x86_64-linux-gnu/girepository-1.0/WebKit2-4.1.typelib", "DATA"))
    assert not bundle.keep_data(("share/glib-2.0/schemas/gschemas.compiled", "/usr/share/glib-2.0/schemas/gschemas.compiled", "DATA"))
    assert bundle.keep_data(("web_dist/index.html", "/w/web/dist/index.html", "DATA"))
    assert not bundle.keep_script(("pyi_rth_gdkpixbuf", "/x/pyi_rth_gdkpixbuf.py", "PYSOURCE"))
    assert bundle.keep_script(("pyi_rth_multiprocessing", "/x/pyi_rth_multiprocessing.py", "PYSOURCE"))
    assert bundle.keep_script(("launcher", "/w/packaging/launcher.py", "PYSOURCE"))
    assert bundle.forbidden(["_internal/libglib-2.0.so.0", "_internal/cffi.libs/libffi-1a2b3c.so.8", "_internal/libz.so.1",
                             "_internal/libpython3.12.so.1.0"]) == ["_internal/libglib-2.0.so.0", "_internal/libz.so.1"]


def test_the_spec_filters_the_linux_bundle():
    spec = (REPO / "packaging" / "nesicode.spec").read_text(encoding="utf-8")
    assert "bundle.keep_binary" in spec and "bundle.keep_data" in spec and "bundle.keep_script" in spec
    assert 'name="nesicode"' in spec


@pytest.mark.skipif(os.name == "nt", reason="the .deb is built on Linux (links, modes)")
def test_the_deb_package_layout(tmp_path):
    pytest.importorskip("PIL")
    make_deb = _load("nesicode_make_deb", REPO / "packaging" / "linux" / "make_deb.py")
    fake = tmp_path / "nesicode"
    (fake / "_internal").mkdir(parents=True)
    (fake / "nesicode").write_bytes(b"\x7fELF fake")
    os.chmod(fake / "nesicode", 0o700)
    (fake / "_internal" / "web.txt").write_text("x", encoding="utf-8")
    os.chmod(fake / "_internal" / "web.txt", 0o600)
    tree = make_deb.build_tree(fake, "0.4.9", "amd64", tmp_path / "work")

    control = (tree / "DEBIAN" / "control").read_text(encoding="utf-8")
    fields = dict(line.split(": ", 1) for line in control.splitlines() if ": " in line and not line.startswith(" "))
    assert fields["Package"] == "nesicode" and fields["Version"] == "0.4.9" and fields["Architecture"] == "amd64"
    assert "gir1.2-webkit2-4.1" in fields["Depends"] and "libgirepository-1.0-1" in fields["Depends"]
    assert int(fields["Installed-Size"]) > 0 and control.endswith("\n")
    assert all(line.startswith(" ") for line in control.split("Description: ", 1)[1].splitlines()[1:])
    assert os.readlink(tree / "usr" / "bin" / "nesicode") == "/opt/nesicode/nesicode"
    assert os.stat(tree / "opt" / "nesicode" / "nesicode").st_mode & 0o777 == 0o755
    assert os.stat(tree / "opt" / "nesicode" / "_internal" / "web.txt").st_mode & 0o777 == 0o644
    entry = (tree / "usr" / "share" / "applications" / "nesicode.desktop").read_text(encoding="utf-8")
    assert "Exec=nesicode" in entry and "Icon=nesicode" in entry and "StartupWMClass=Nesicode" in entry
    from PIL import Image

    for size in make_deb.ICON_SIZES:
        with Image.open(tree / "usr" / "share" / "icons" / "hicolor" / f"{size}x{size}" / "apps" / "nesicode.png") as im:
            assert im.size == (size, size)
    assert "Nesi Labs" in (tree / "usr" / "share" / "doc" / "nesicode" / "copyright").read_text(encoding="utf-8")
    sums = (tree / "DEBIAN" / "md5sums").read_text(encoding="utf-8")
    assert "  opt/nesicode/nesicode\n" in sums and "usr/bin/nesicode" not in sums  # (a link has no sum)
