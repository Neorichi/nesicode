"""Cross-turn continuity and the context meter — offline.

Scenario (fictional): an authorized test of the deliberately vulnerable demo app "vulnbank.org". Every tool output and
model reply below is scripted; nothing touches the network.
"""

import time

from fastapi.testclient import TestClient

from nesicode.agent import Agent, Workspace
from nesicode.harness import base_harness
from nesicode.llm.base import Conversation, Provider, ToolCall, Turn, Usage
from nesicode.server.app import app, db

UPLOAD_403 = ("HTTP/1.1 403 Forbidden\nContent-Type: text/html\n\n<h1>CSRF verification failed.</h1> "
              "Request aborted. POST https://vulnbank.org/api/upload")
COOKIE = "HTTP/1.1 200 OK\nSet-Cookie: csrftoken=AbC123secretTOKEN; Path=/\n\n<html>VulnBank</html>"


def _client():
    c = TestClient(app)
    assert c.post("/api/auth/register", json={"email": f"t{time.time_ns()}@x.com", "password": "secret123"}).status_code == 200
    p = c.post("/api/projects", json={"name": "vulnbank", "profile": "security",
                                      "executor_model": "mock/demo", "improver_model": "mock/demo"}).json()
    t = c.post(f"/api/projects/{p['id']}/threads", json={}).json()
    return c, p, t


def _run(p, t, task, status, final="", events=(), at=None):
    rid = db.insert("INSERT INTO runs (project_id, thread_id, kind, task, status, final_text, created_at) "
                    "VALUES (?, ?, 'chat', ?, ?, ?, ?)", (p["id"], t["id"], task, status, final, at or time.time()))
    for e in events:
        db.add_event(rid, e)
    return db.one("SELECT * FROM runs WHERE id=?", (rid,))


def _call(i, name, **args):
    return {"type": "tool_call", "id": f"c{i}", "name": name, "args": args}


def _res(i, out, err=False):
    return {"type": "tool_result", "id": f"c{i}", "output": out, "is_error": err}


def test_stopped_turn_reaches_next_message():
    from nesicode.server.jobs import continuity_note, conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "Haz recon de vulnbank.org", "done", "Encontré /api/upload y /login.", [
        _call(1, "bash", command="curl -sS -I https://vulnbank.org/"), _res(1, "HTTP/1.1 200 OK\nServer: nginx"),
        {"type": "assistant", "text": "Encontré /api/upload y /login."}], at=now - 30)
    _run(p, t, "prueba la subida de ficheros", "cancelled", "", [
        _call(1, "bash", command="curl -sS -i -F file=@t.txt https://vulnbank.org/api/upload"), _res(1, UPLOAD_403),
        _call(2, "bash", command="curl -sS -i -c cj.txt https://vulnbank.org/"), _res(2, COOKIE),
        {"type": "assistant", "text": "Pruebo con la cabecera X-CSRFToken."},
        _call(3, "bash", command="curl -sS -i -b cj.txt -H 'X-CSRFToken: AbC123secretTOKEN' -F file=@t.txt https://vulnbank.org/api/upload"),
        {"type": "cancelled"}], at=now - 10)
    nxt = _run(p, t, "continúa", "waiting", at=now)
    history, interrupted = conversation_history(db, nxt)
    assert [h[0] for h in history] == ["Haz recon de vulnbank.org", "prueba la subida de ficheros"]
    stopped = history[-1][1]
    assert "cancelled at step 3" in stopped and "did NOT finish" in stopped
    assert "https://vulnbank.org/api/upload" in stopped and "403" in stopped and "X-CSRFToken" in stopped
    assert "AbC123secretTOKEN" not in stopped  # session/CSRF values are never carried between turns
    assert "[PREVIOUS TURN — STEPS]" in stopped
    assert interrupted == {"id": history and interrupted["id"], "status": "cancelled", "steps": 3}
    note = continuity_note(t["id"], interrupted)
    assert "history.md" in note and "search_history" in note and "do not start over" in note


def test_cancel_without_work_is_one_line():
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    # it started and was stopped at once (a message removed from the queue never started: no events, not shown at all)
    _run(p, t, "algo", "cancelled", events=[{"type": "start", "task": "algo"}, {"type": "cancelled"}], at=time.time() - 5)
    history, interrupted = conversation_history(db, _run(p, t, "sigue", "waiting"))
    assert history == [("algo", "[this turn was cancelled before doing any work]")]


def test_search_history_tolerates_typos(tmp_path):
    from nesicode.agent.context import search_history

    ws = Workspace(tmp_path / "ws")
    hist = ws.root / ".nesicode/chats/9/context/history.md"
    hist.parent.mkdir(parents=True)
    hist.write_text("[ACTION] bash curl https://vulnbank.org/ckeditor5/image_upload/\n[OBSERVATION] HTTP/1.1 403 Forbidden\n"
                    "[ACTION] bash curl -F file=@t.txt https://vulnbank.org/api/upload\n")
    out = search_history(ws, {"query": "ckeifor ckeifor la idea del"}, ".nesicode/chats/9")
    assert "ckeditor5/image_upload" in out and "similar words" in out
    out = search_history(ws, {"query": "vulnbnk upload idea"}, ".nesicode/chats/9")
    assert "api/upload" in out
    assert "no matches" in search_history(ws, {"query": "zzzz qqqq"}, ".nesicode/chats/9")


# ------------------------------------------------------------------ meter
class Scripted(Conversation):
    def __init__(self, turns, *a):
        super().__init__(*a)
        self.turns, self.sent = list(turns), []

    def add_user(self, text):
        self.sent.append(("user", text))

    def add_assistant_text(self, text):
        self.sent.append(("assistant", text))

    def add_tool_results(self, results):
        self.sent.extend(("tool", r.content) for r in results)

    def step(self, on_event=None):
        return self.turns.pop(0)


class ScriptedProvider(Provider):
    name = "scripted"

    def __init__(self, turns):
        self.turns, self.conv = turns, None

    def conversation(self, model, system, tools, options):
        self.conv = Scripted(self.turns, model, system, tools, options)
        return self.conv


def test_context_meter_never_drops_without_a_reason(tmp_path, monkeypatch):
    import nesicode.agent.loop as loop

    turns = [
        # step 1: big reasoning (not sent back) + a tool call whose output is 8,000 chars of mocked upload page
        Turn("Miro la página de subida.", [ToolCall("t1", "page", {"url": "https://vulnbank.org/upload"})], "tool_use",
             Usage(13_000, 2_200), thinking="x" * 7_000),
        Turn("Hecho: el formulario exige CSRF.", [], "end_turn", Usage(13_300, 150)),
    ]
    prov = ScriptedProvider(turns)
    monkeypatch.setattr(loop, "get_provider", lambda spec, keys=None: (prov, "m"))
    events = []
    agent = Agent(model="scripted/m", harness=base_harness(), workspace=Workspace(tmp_path / "ws"),
                  emit=events.append, context_limit=131_072)
    agent.extra_tools = {"page": (None, lambda args, ag: "<form>" + "a" * 8_000 + "</form>")}
    res = agent.run("revisa la subida de vulnbank.org", history=[("hola", "hola, ¿qué probamos?")])
    assert res.status == "done"
    ctx = [e for e in events if e["type"] == "context"]
    assert ctx[0]["reason"] == "new_message"
    used = [e["used"] for e in ctx]
    assert used == sorted(used), used  # monotonic within the turn (no masking/compaction happened)
    done = next(e for e in events if e["type"] == "done")
    assert done["context"]["used"] == used[-1]
    assert 13_000 < used[-1] < 17_000  # prompt + visible reply + ~8k chars of output at ~4 chars/token


def test_fit_history_trims_a_huge_turn_instead_of_dropping_older_ones(tmp_path):
    agent = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(tmp_path), context_limit=40_000)
    history = [("t1", "a1"), ("t2", "x" * 60_000), ("t3", "a3")]
    kept = agent._fit_history(history, "system")
    assert [k[0] for k in kept] == ["t1", "t2", "t3"] and "middle of this earlier turn omitted" in kept[1][1]


def test_connector_stays_available_once_used():
    from nesicode.connectors import mentioned
    from nesicode.server.jobs import connector_context, conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "usa virustotal para ver vulnbank.org", "done", "VirusTotal: 0 motores lo marcan.", [
        _call(1, "mcp__virustotal__vt_domain_report", domain="vulnbank.org"), _res(1, "Verdict: clean")], at=now - 20)
    nxt = _run(p, t, "busca más subdominios", "waiting", at=now)
    history, _ = conversation_history(db, nxt)
    ctx = connector_context(db, nxt, history)
    assert mentioned(ctx, name="virustotal", label="VirusTotal", template="virustotal")
    assert not mentioned("busca más subdominios", name="virustotal", label="VirusTotal", template="virustotal")


def test_virustotal_subdomains(tmp_path):
    import httpx

    from nesicode import virustotal as vt
    from nesicode.connectors import ConnectorConfig

    def handler(req):
        assert req.url.path == "/api/v3/domains/vulnbank.org/subdomains" and req.url.params["limit"] == "40"
        return httpx.Response(200, json={"data": [{"id": "api.vulnbank.org", "attributes": {}},
                                                  {"id": "dev.vulnbank.org", "attributes": {"last_analysis_stats": {"malicious": 2}}}],
                                         "meta": {"cursor": "NEXT"}})

    vt.transport = httpx.MockTransport(handler)
    try:
        cfg = ConnectorConfig(name="virustotal", kind="builtin", command="virustotal", env={"VT_API_KEY": "K"})
        out, _ = vt.vt_relations({"target": "https://vulnbank.org/x", "relation": "subdomains"}, Workspace(tmp_path / "w"), cfg)
    finally:
        vt.transport = None
    assert "api.vulnbank.org" in out and "dev.vulnbank.org · 2 malicious" in out and "cursor: NEXT" in out


def test_redact_common_credential_shapes():
    from nesicode.agent.context import redact

    cases = ["Authorization: Basic dXNlcjpwYXNzd29yZA==", "Authorization: Token 9944b09199c62bcf9418ad846dd0e4bb",
             "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY", "https://admin:S3cretPass@vulnbank.org/x",
             "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ2dWxuYmFuayJ9.c2lnbmF0dXJlLXZhbHVl",
             "Authorization: Bearer abcdefghijklmnop123"]
    for c, secret in zip(cases, ["dXNlcjpw", "9944b0", "wJalrX", "S3cretPass", "c2lnbmF0", "abcdefgh"]):
        out = redact(c)
        assert secret not in out, out
    assert "Authorization: Basic <redacted>" == redact(cases[0])
    assert "https://admin:<redacted>@vulnbank.org/x" == redact(cases[3])
    assert "HTTP/1.1 403 Forbidden" == redact("HTTP/1.1 403 Forbidden")
