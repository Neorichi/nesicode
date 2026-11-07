"""Record the Nesicode quick-start video (Playwright → webm).

1. Start a clean, throwaway instance (no .env, its own data dir and vault, English UI, neutral user):
     NESICODE_NO_DOTENV=1 NESICODE_DATA_DIR=/tmp/nesicode-demo NESICODE_LOCAL_USER=demo@nesicode.local \
     NESICODE_VAULT_KEY=$(python3 -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())") \
     uv run nesicode serve --port 8431
     curl -X PUT localhost:8431/api/me/settings -H 'Content-Type: application/json' -d '{"language":"en"}'
2. DEMO_KEY=<a Nesi Labs API key> uv run --with playwright==1.56.0 python docs/demo/record_demo.py out/
   The key is typed into a password field and any key hint (••••abcd) is masked on screen.
3. Trim / speed up the wait for the answer with ffmpeg and export MP4 + GIF (see git history of docs/demo).
"""
import os
import sys
import time
import json
import urllib.request

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8431"
KEY = os.environ["DEMO_KEY"]
OUT = sys.argv[1]

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


def api(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read())


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={"width": 1280, "height": 800}, color_scheme="dark",
                              record_video_dir=OUT, record_video_size={"width": 1280, "height": 800})
    ctx.add_init_script(INIT)
    page = ctx.new_page()
    page.set_default_timeout(15000)
    T0 = time.monotonic()
    mouse = {"x": 640, "y": 400}

    def cap(text, wait=0.0):
        print("STEP:", text, flush=True)
        page.evaluate("t => { window.__pendingCap = t; window.__cap && window.__cap(t) }", text)
        time.sleep(wait)

    def move(loc):
        loc.scroll_into_view_if_needed()
        b = loc.bounding_box()
        x, y = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
        page.mouse.move(x, y, steps=22)
        mouse.update(x=x, y=y)
        time.sleep(0.25)

    def click(loc):
        move(loc)
        page.mouse.down()
        page.mouse.up()
        time.sleep(0.6)

    def type_in(loc, text, delay=45):
        click(loc)
        loc.press_sequentially(text, delay=delay)
        time.sleep(0.4)

    # ---------------------------------------------------------------- 0. intro + project
    page.goto(BASE + "/")
    page.wait_for_load_state("networkidle")
    cap("Nesicode — a coding agent that improves its own harness. Set up in one minute.", 3.2)
    cap("Create a project (any folder — every chat works in its own isolated copy)", 0.6)
    form = page.locator("form.new-project")
    type_in(form.locator("input").nth(1), "demo-app")
    cap("Pick a profile: General, Security, Web, Data, Documents — or Blank", 0.3)
    cards = page.locator(".profile-card")
    for i in range(cards.count()):
        move(cards.nth(i))
        time.sleep(0.35)
    click(cards.filter(has_text="General").first)
    click(page.get_by_role("button", name="Create project"))
    page.wait_for_url("**/p/*")
    time.sleep(1)

    # ---------------------------------------------------------------- 1. provider + key
    cap("1 · Add your model provider and API key", 0.4)
    click(page.get_by_role("link", name="Providers"))
    time.sleep(0.8)
    click(page.get_by_role("button", name="+ Add provider"))
    click(page.locator(".templates button", has_text="Nesi Labs"))
    type_in(page.locator("input[type=password]"), KEY, delay=8)
    cap("Keys go to an encrypted vault — the app never shows them again", 0.3)
    click(page.get_by_role("button", name="Add provider", exact=True))
    page.wait_for_timeout(4000)
    cap("Models are discovered and assigned to each role automatically", 3.0)

    # ---------------------------------------------------------------- 2. models by role
    cap("2 · Pick a model for each role: Chat, Improvement, Vision, Images, Transcription", 0.4)
    click(page.get_by_role("link", name="Project settings"))
    time.sleep(1)
    roles = page.get_by_text("Models by role")
    roles.scroll_into_view_if_needed()
    move(roles)
    time.sleep(2.0)
    selects = page.locator(".card.form select")
    for i in range(min(selects.count(), 4)):
        move(selects.nth(i))
        time.sleep(0.5)
    click(page.get_by_role("button", name="Save models"))
    time.sleep(1.2)

    # ---------------------------------------------------------------- 3. chat
    cap("3 · Ask for something — it plans, edits files and runs commands in its sandbox", 0.4)
    click(page.get_by_role("link", name="New chat"))
    time.sleep(0.8)
    box = page.locator("textarea").first
    type_in(box, "Write a Python script that prints the first 20 prime numbers, run it and show me the output.", delay=28)
    page.keyboard.press("Enter")
    print("MARK chat_start", round(time.monotonic() - T0, 2), flush=True)
    page.wait_for_url("**/c/*", timeout=20000)
    tid = page.url.rsplit("/", 1)[-1]
    deadline = time.time() + 240
    while time.time() < deadline:
        runs = api(f"/api/threads/{tid}")["runs"]
        if runs and runs[-1]["status"] not in ("queued", "running", "waiting"):
            break
        time.sleep(1)
    print("MARK chat_end", round(time.monotonic() - T0, 2), flush=True)
    time.sleep(1.5)
    page.mouse.wheel(0, 3000)
    cap("Done — tools, outputs and the final answer, all in the chat", 3.5)

    # ---------------------------------------------------------------- 4. connectors
    cap("4 · Connectors: Office, Burp Suite, VirusTotal, GitHub, your own Chrome…", 0.4)
    click(page.get_by_role("link", name="Connectors (MCP)"))
    time.sleep(1.5)
    page.mouse.wheel(0, 500)
    time.sleep(1.2)
    cap("…used only when you ask for them (or whenever useful — your choice)", 2.8)

    # ---------------------------------------------------------------- 5. bots
    cap("5 · Bots: tasks that run on their own — on a schedule, on Telegram or by webhook", 0.4)
    click(page.get_by_role("link", name="Bots"))
    time.sleep(1)
    click(page.get_by_role("button", name="+ New bot"))
    click(page.locator(".bot-templates .chip").first)
    time.sleep(2.2)
    cap("Nesicode · desktop app for macOS & Windows · github.com/Neorichi/nesicode", 3.5)

    page.close()
    ctx.close()
    browser.close()
print("ok")
