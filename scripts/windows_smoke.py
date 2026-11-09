"""End-to-end smoke test of a real Nesicode server (CI, any OS): project → conversation → a chat with the offline
mock/demo model, which really writes and reads files through the agent's tools. Prints what happened and exits 1
if the run did not end 'done'. Usage: python scripts/windows_smoke.py [port]"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8432
BASE = f"http://127.0.0.1:{PORT}"


def call(method, path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


data_dir = tempfile.mkdtemp(prefix="nesi-smoke-")
env = {**os.environ, "NESICODE_NO_DOTENV": "1", "NESICODE_DATA_DIR": data_dir, "NESICODE_AUTH": "false",
       "NESICODE_NO_TELEGRAM": "1", "NESICODE_NO_SCHEDULER": "1", "NESICODE_LOCAL_USER": "smoke@nesicode.local",
       "NESICODE_VAULT_KEY": base64.b64encode(os.urandom(32)).decode()}
log = open(os.path.join(data_dir, "serve.log"), "wb")
srv = subprocess.Popen([sys.executable, "-c", "from nesicode.cli import main; main()", "serve", "--host", "127.0.0.1",
                        "--port", str(PORT)], env=env, stdout=log, stderr=subprocess.STDOUT)
ok = False
try:
    for _ in range(120):
        try:
            call("GET", "/api/auth/me", timeout=3)
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise SystemExit("server did not start")
    print("server up on", BASE)
    page = urllib.request.urlopen(BASE + "/", timeout=10)
    print("SPA:", page.status, page.headers.get("content-security-policy", "")[:60])
    p = call("POST", "/api/projects", {"name": "smoke", "profile": "general", "executor_model": "mock/demo",
                                       "improver_model": "mock/demo"})
    t = call("POST", f"/api/projects/{p['id']}/threads", {})
    rid = call("POST", f"/api/threads/{t['id']}/messages", {"text": "Write notes.md with three bullet points about Windows, "
                                                                     "then read it back."})["id"]
    run = {}
    for _ in range(360):
        run = call("GET", f"/api/runs/{rid}", timeout=30)
        if run["status"] not in ("queued", "running", "waiting"):
            break
        time.sleep(0.5)
    ev = run.get("events", [])
    print("run status:", run.get("status"), "| events:", len(ev))
    for e in ev:
        if e.get("type") in ("tool_call", "tool_result", "error", "done", "assistant"):
            print("  ", e.get("type"), e.get("name", ""), json.dumps(e.get("args") or e.get("output") or e.get("text") or "",
                                                                   ensure_ascii=False)[:160])
    ok = run.get("status") == "done" and not any(e.get("type") == "error" for e in ev)
finally:
    srv.terminate()
    try:
        srv.wait(15)
    except Exception:
        srv.kill()
    log.close()
    tail = open(os.path.join(data_dir, "serve.log"), encoding="utf-8", errors="replace").read()[-3000:]
    print("---- server log (tail) ----\n" + tail)
sys.exit(0 if ok else 1)
