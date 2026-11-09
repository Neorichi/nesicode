"""Record the "providers and models" how-to video (Playwright → webm).

1. Start a clean, throwaway instance (no .env, its own data dir and vault, English UI, neutral user), as for
   record_demo.py:
     NESICODE_NO_DOTENV=1 NESICODE_DATA_DIR=/tmp/nesicode-demo NESICODE_LOCAL_USER=demo@nesicode.local \
     NESICODE_VAULT_KEY=$(python3 -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())") \
     uv run nesicode serve --port 8431
     curl -X PUT localhost:8431/api/me/settings -H 'Content-Type: application/json' -d '{"language":"en"}'
2. DEMO_KEY=<a Nesi Labs API key> uv run --with playwright==1.56.0 python docs/demo/record_providers.py out/
   The key is typed into a password field and any key hint (••••abcd) is masked on screen.
3. Export MP4 + GIF with ffmpeg (docs/demo/providers.mp4, providers.gif).
"""
import json
import os
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8431"
KEY = os.environ["DEMO_KEY"]
OUT = sys.argv[1]
# Model-name prefixes the video may show (e.g. "nesi"): the provider's other models never reach the screen.
SHOW = [p for p in os.environ.get("DEMO_SHOW_MODELS", "").split(",") if p]

INIT = r"""
localStorage.setItem('nesi:lang', 'en');
(() => {
  // never show even the last characters of a key
  const mask = (root) => {
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let n; while ((n = w.nextNode())) {
      if (/••••[A-Za-z0-9_-]{2,}/.test(n.nodeValue)) n.nodeValue = n.nodeValue.replace(/••••[A-Za-z0-9_-]+/g, '••••••••');
      if (/\/(private|tmp)\/[^\s]*demo[^\s]*/.test(n.nodeValue)) n.nodeValue = n.nodeValue.replace(/\/(private|tmp)\/[^\s]*demo[^\s]*/g, '~/Nesicode/demo-app');
    }
  };
  new MutationObserver(() => mask(document.body || document.documentElement)).observe(document, { subtree: true, childList: true, characterData: true });
  // visible cursor + captions
  window.addEventListener('DOMContentLoaded', () => {
    const c = document.createElement('div');
    c.id = '__cursor';
    c.style.cssText = 'position:fixed;z-index:2147483647;width:18px;height:18px;border-radius:50%;background:rgba(47,191,168,.85);border:2px solid #fff;box-shadow:0 0 0 4px rgba(47,191,168,.25);pointer-events:none;left:-40px;top:-40px;transform:translate(-50%,-50%);transition:transform .08s';
    document.body.appendChild(c);
    document.addEventListener('mousemove', (e) => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    document.addEventListener('mousedown', () => { c.style.transform = 'translate(-50%,-50%) scale(.7)'; }, true);
    document.addEventListener('mouseup', () => { c.style.transform = 'translate(-50%,-50%)'; }, true);
    const cap = document.createElement('div');
    cap.id = '__cap';
    cap.style.cssText = 'position:fixed;z-index:2147483646;left:50%;bottom:26px;transform:translateX(-50%);max-width:880px;padding:12px 22px;border-radius:14px;background:rgba(10,12,12,.88);color:#fff;font:600 19px/1.35 -apple-system,Segoe UI,system-ui,sans-serif;text-align:center;box-shadow:0 8px 30px rgba(0,0,0,.35);opacity:0;transition:opacity .35s;pointer-events:none';
    document.body.appendChild(cap);
    window.__cap = (t) => { cap.textContent = t; cap.style.opacity = t ? 1 : 0; };
    if (window.__pendingCap) window.__cap(window.__pendingCap);
  });
})();
"""


def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


# The project the video starts in (made off camera: this video is about providers).
project = api("POST", "/api/projects", {"name": "demo-app", "profile": "general"})

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={"width": 1280, "height": 800}, color_scheme="dark",
                              record_video_dir=OUT, record_video_size={"width": 1280, "height": 800})
    ctx.add_init_script(INIT)
    if SHOW:
        def only_shown(route):
            resp = route.fetch()
            body = resp.json()
            if isinstance(body.get("models"), list):
                body["models"] = [m for m in body["models"] if any(str(m).startswith(p) for p in SHOW)]
            route.fulfill(response=resp, json=body)

        ctx.route("**/api/providers/test", only_shown)  # "Test connection" and "Discover models"
    page = ctx.new_page()
    page.set_default_timeout(20000)
    T0 = time.monotonic()

    def cap(text, wait=0.0):
        print("STEP:", round(time.monotonic() - T0, 2), text, flush=True)
        page.evaluate("t => { window.__pendingCap = t; window.__cap && window.__cap(t) }", text)
        time.sleep(wait)

    def move(loc, pause=0.25):
        loc.scroll_into_view_if_needed()
        b = loc.bounding_box()
        page.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, steps=20)
        time.sleep(pause)

    def click(loc):
        move(loc)
        page.mouse.down()
        page.mouse.up()
        time.sleep(0.6)

    def type_in(loc, text, delay=45):
        click(loc)
        loc.press_sequentially(text, delay=delay)
        time.sleep(0.4)

    page.goto(f"{BASE}/p/{project['id']}")
    page.wait_for_load_state("networkidle")
    cap("Providers: connect Nesicode to the models you use", 2.8)

    # ---------------------------------------------------------------- the providers page + templates
    click(page.get_by_role("link", name="Providers"))
    cap("Add as many as you like — each API key is kept in an encrypted vault", 1.2)
    click(page.get_by_role("button", name="+ Add provider"))
    cap("Start from a template: Nesi Labs, Claude, OpenAI, OpenRouter, Gemini, GLM, Grok, DeepSeek…", 0.3)
    tpl = page.locator(".templates button")
    for i in range(tpl.count()):
        move(tpl.nth(i), 0.3)
    time.sleep(0.6)
    click(tpl.filter(has_text="Ollama (local)"))
    cap("Local models (Ollama, LM Studio, vLLM…): just the address, no key needed", 0.3)
    move(page.get_by_label("Endpoint (base URL)"), 2.6)
    click(tpl.filter(has_text="Other (OpenAI-compatible)"))
    cap("Any other OpenAI-compatible API: its address, its key and its models", 2.8)

    # ---------------------------------------------------------------- Nesi Labs: key, test, discover, add
    click(tpl.filter(has_text="Nesi Labs"))
    cap("Pick yours and paste the API key — it is stored encrypted and never shown again", 0.3)
    type_in(page.locator("input[type=password]"), KEY, delay=8)
    time.sleep(1.2)
    cap("Test the connection…", 0.2)
    click(page.get_by_role("button", name="Test connection"))
    page.locator(".ok-box").wait_for()
    time.sleep(2.0)
    cap("…and discover its models in one click", 0.2)
    click(page.get_by_role("button", name="Discover models"))
    page.wait_for_function("() => (document.querySelector('.prov-form textarea')||{}).value?.length > 0")
    move(page.locator(".prov-form textarea"), 2.2)
    click(page.get_by_role("button", name="Add provider", exact=True))
    page.locator(".prov").filter(has_text="Nesi Labs").first.wait_for()
    time.sleep(1.0)
    cap("Done: its models are assigned to each role automatically", 3.0)
    row = page.locator(".prov").filter(has_text="Nesi Labs").first
    cap("Refresh its models, edit it or remove it at any time", 0.3)
    move(row.get_by_role("button", name="Refresh models"), 0.8)
    move(row.get_by_role("button", name="Edit"), 0.8)
    move(row.get_by_role("button", name="Delete"), 1.4)

    # ---------------------------------------------------------------- models by role
    click(page.get_by_role("link", name="Project settings"))
    time.sleep(1.0)
    roles = page.get_by_text("Models by role").first
    roles.scroll_into_view_if_needed()
    move(roles, 0.4)
    cap("Then choose a model for each role: Chat, Improvement, Vision, Image generation, Transcription", 0.4)
    selects = page.locator(".card.form select")
    for i in range(min(selects.count(), 6)):
        move(selects.nth(i), 0.55)
    lim = page.locator(".limits-line").first
    if lim.count():
        move(lim, 0.3)
        cap("Each model shows its context window: Nesicode keeps long tasks within it", 3.0)
    click(page.get_by_role("button", name="Save models"))
    cap("Saved — new chats in this project use these models", 2.2)

    # ---------------------------------------------------------------- moving to another computer
    click(page.locator(".user-btn"))
    click(page.get_by_role("link", name="Account and API keys"))
    time.sleep(1.0)
    exp = page.get_by_role("button", name="Export configuration")
    exp.scroll_into_view_if_needed()
    cap("Another computer? Export your providers, keys and settings in a password-encrypted file…", 0.3)
    move(exp, 2.8)
    imp = page.get_by_text("Import configuration").first
    if imp.count():
        imp.scroll_into_view_if_needed()
        move(imp, 0.3)
    cap("…and import it there: you review what it brings before anything changes", 3.0)
    cap("Nesicode · desktop app for macOS & Windows · github.com/Neorichi/nesicode", 3.2)

    page.close()
    ctx.close()
    browser.close()
print("ok")
