"""A bot whose goal has an end («tell me when X happens») switches itself off once it is reached, and says so
(the user asked: «he completado mi tarea, me desactivo»). Switching it on again is the usual toggle."""

import json
import time
from types import SimpleNamespace

import pytest

from nesicode.agent.workspace import ToolError
from nesicode.server.app import bots, db
from test_fix_bots import FakeTG, alice, make_bot, project, wait_for  # noqa: F401 (fixtures)
from test_p2_web import SRC, run_ts


def finish_tool(bot_id, run_id=0):
    return bots.agent_tools(bots.bot(bot_id), run_id)["bot_finish"][1]


def quiet():
    return SimpleNamespace(emit=lambda e: None)


def test_a_bot_that_reached_its_goal_switches_itself_off_and_says_so(alice, project, monkeypatch):
    fake = FakeTG()
    monkeypatch.setattr(bots, "telegram", fake)
    b = make_bot(alice, project, name="Vigilancia", notify="telegram", triggers=[{"kind": "interval", "spec": "30"}])
    db.execute("INSERT INTO bot_triggers (bot_id, kind, prompt, next_at, created_at) VALUES (?, 'once', 'mira otra vez', ?, ?)",
               (b["id"], time.time() + 600, time.time()))
    events = []
    out = finish_tool(b["id"])({"reason": "  Trazvia ya está\n en Google Play "}, SimpleNamespace(emit=events.append))
    assert "desactivado" in out
    v = alice.get(f"/api/bots/{b['id']}").json()
    assert v["enabled"] is False and v["completed_at"] and v["paused_reason"] == "Trazvia ya está en Google Play"
    assert v["next_at"] is None
    # its schedule stays (for when it is switched on again); its pending follow-up was a step towards the goal: gone
    assert [r["kind"] for r in db.all("SELECT kind FROM bot_triggers WHERE bot_id=?", (b["id"],))] == ["interval"]
    assert events == [{"type": "bot_finished", "text": "Trazvia ya está en Google Play"}]
    wait_for(lambda: fake.sent, 5, "the notice")
    text, locked, thread = fake.sent[0]
    assert "«Vigilancia» ha cumplido su tarea y se ha desactivado: Trazvia ya está en Google Play" in text
    assert not locked and thread == "nesi-bots-notice"  # (sent in the background, never under the scheduler's lock)
    assert bots.fire(b["id"], "interval") is None  # off: its schedule no longer runs it


def test_switching_it_on_again_clears_the_finish(alice, project):
    b = make_bot(alice, project, name="Vigilancia dos", triggers=[{"kind": "interval", "spec": "30"}])
    finish_tool(b["id"])({"reason": "hecho"}, quiet())
    r = alice.patch(f"/api/bots/{b['id']}", json={"enabled": True})
    assert r.status_code == 200, r.text
    v = alice.get(f"/api/bots/{b['id']}").json()
    assert v["enabled"] is True and v["completed_at"] is None and v["paused_reason"] == "" and v["next_at"]


def test_finishing_needs_a_reason_and_sends_no_notice_into_its_own_chat(alice, project, monkeypatch):
    fake = FakeTG()
    monkeypatch.setattr(bots, "telegram", fake)
    b = make_bot(alice, project, name="Charla", notify="telegram")
    with pytest.raises(ToolError):
        finish_tool(b["id"])({"reason": "  "}, quiet())
    assert bots.bot(b["id"])["enabled"] and not bots.bot(b["id"])["completed_at"]
    # the user is talking to the bot in its own conversation («ya está, desactívate»): they see it there
    run_id = db.insert("INSERT INTO runs (project_id, thread_id, kind, task, status, created_at, bot_id, trigger) "
                       "VALUES (?, ?, 'chat', 'ya está, desactívate', 'running', ?, ?, 'chat')",
                       (project["id"], bots.bot(b["id"])["thread_id"], time.time(), b["id"]))
    try:
        finish_tool(b["id"], run_id)({"reason": "el usuario lo da por cumplido"}, quiet())
        time.sleep(0.3)  # (time for a notice, if one went out)
        assert not bots.bot(b["id"])["enabled"] and fake.sent == []
    finally:
        db.execute("UPDATE runs SET status='cancelled', finished_at=? WHERE id=?", (time.time(), run_id))


def test_the_bot_is_told_when_to_finish_and_when_not(alice, project):
    b = bots.bot(make_bot(alice, project, name="Guía")["id"])
    section = bots.system_section(b)
    assert "bot_finish" in section and "recurring duties" in section and "payload asks you to" in section
    assert "bot_finish" in bots.agent_tools(b, 0)


def test_the_app_shows_a_finished_bot_as_done_not_as_paused(tmp_path):
    out = run_ts(tmp_path, f"""
      import {{ PausedNote }} from {json.dumps(str(SRC / 'pages' / 'Bots.tsx'))}
      import Transcript from {json.dumps(str(SRC / 'components' / 'Transcript.tsx'))}
      out.done = html(<PausedNote b={{{{ completed_at: 1, paused_reason: 'Trazvia ya está en Google Play' }}}} />)
      out.paused = html(<PausedNote b={{{{ completed_at: null, paused_reason: '' }}}} />)
      out.failed = html(<PausedNote b={{{{ completed_at: null, paused_reason: '3 fallos seguidos' }}}} />)
      out.chat = html(<Transcript events={{[{{ type: 'bot_finished', text: 'Trazvia ya está en Google Play' }}] as any}} status="done" />)
      out.known = L.KNOWN_EVENTS.has('bot_finished')
    """)
    assert "✅ Tarea cumplida: Trazvia ya está en Google Play" in out["done"] and 'class="done"' in out["done"]
    assert "⏸ En pausa" in out["paused"] and "⏸ 3 fallos seguidos" in out["failed"]
    assert "Tarea cumplida: el bot se ha desactivado" in out["chat"] and "Trazvia ya está en Google Play" in out["chat"]
    assert out["known"] is True
