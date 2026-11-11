"""v0.4.7, conversation history and the compaction rebuild — offline: the test database, temp workspaces and a fake
OpenAI-format provider at the user's real 131,072-token window (nesi3).

1. Stop → a quick question → «continúa»: the stopped turn keeps its step trail and the "did not finish" note.
2. Earlier user messages are never dropped by a fixed turn count, nor cut at a fixed size: the token budget decides,
   answers give way first and a clipped message says where its full text is.
3. Checkpoints are cumulative across turns: a run's first summary updates the newest earlier checkpoint, and the
   conversation's newest checkpoint always travels in the history.
4. The rebuild after a compaction shrinks the earlier turns before this run's recent steps and files, by exactly
   what is over the target."""

from __future__ import annotations

import re
import time

import pytest

from nesicode.agent import Agent, Workspace
from nesicode.agent import context as ctxm
from nesicode.agent.loop import ANSWER_LEFT_OUT, RECORD_CLOSE, RECORD_OPEN, TRAIL_MARK
from nesicode.harness import base_harness
from nesicode.llm.base import ToolCall, Turn, Usage

from test_fix_context import RULE, _agent, _client, _has, _post, _run, fake  # noqa: F401  (fake is a fixture)

WINDOW = 131_072
CHECKPOINT = ("## Goal\nMigrar la base de datos y portar la API.\n## Completed Actions\n"
              "1. APPLIED db/migrations/0042_users.sql with psql — NEVER run it again [tool: bash]\n"
              "2. EDITED src/api/users.py — ported to FastAPI [tool: edit_file]\n"
              "## Next Step\nSeguir con 0043_orders.sql")


def _tool_steps(n, prefix="e"):
    events = [{"type": "start"}]
    for i in range(n):
        events += [{"type": "thinking", "text": f"DECISION-{i}: el router r{i} pasa a FastAPI"},
                   {"type": "tool_call", "id": f"{prefix}{i}", "name": "edit_file",
                    "args": {"path": f"src/api/r{i}.py", "old": "flask", "new": "fastapi"}},
                   {"type": "tool_result", "id": f"{prefix}{i}", "output": f"MIGRADO-R{i}", "is_error": False}]
    return events


def _reply(text):
    return [{"type": "start"}, {"type": "assistant", "text": text}, {"type": "done", "status": "done", "steps": 1}]


# ====================================================================== 1. a question between Stop and «continúa»
@pytest.mark.parametrize("status,question", [("cancelled", "¿cuánto te falta?"), ("error", "¿qué ha pasado?")])
def test_a_question_between_stop_and_continue_keeps_the_trail_and_the_unfinished_note(status, question):
    from nesicode.server.app import db
    from nesicode.server.jobs import continuity_note, conversation_history

    _, p, t = _client()
    now = time.time()
    end = {"type": "cancelled"} if status == "cancelled" else {"type": "error", "text": "HTTP 503"}
    work = _run(p, t, "Porta toda la API a FastAPI", status, "", _tool_steps(45) + [end], at=now - 30)
    _run(p, t, question, "done", "Me faltan 3 routers.", _reply("Me faltan 3 routers."), at=now - 20)
    history, interrupted = conversation_history(db, _run(p, t, "vale, continúa", "running", at=now), window=WINDOW)
    stopped = history[0][1]
    assert TRAIL_MARK in stopped and "MIGRADO-R44" in stopped and "DECISION-44" in stopped  # it ends where it stopped
    assert TRAIL_MARK not in history[1][1] and history[1][1].startswith("Me faltan 3 routers.")
    assert interrupted == {"id": work["id"], "status": status, "steps": 45, "later_replied": 1}
    note = continuity_note(t["id"], interrupted)
    assert "at step 45" in note and "do not start over" in note and "only answered, with no tool work" in note


def test_the_trail_stays_with_the_last_turn_that_used_tools():
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "porta users.py", "done", "Hecho: users.py portado.",
         _tool_steps(6) + [{"type": "assistant", "text": "Hecho: users.py portado."}], at=now - 20)
    _run(p, t, "gracias", "done", "De nada.", _reply("De nada."), at=now - 10)
    history, interrupted = conversation_history(db, _run(p, t, "ahora orders.py", "running", at=now), window=WINDOW)
    assert TRAIL_MARK in history[0][1] and "MIGRADO-R5" in history[0][1] and interrupted is None
    assert history[0][1].count("Hecho: users.py portado.") == 1  # (a finished turn's answer is not repeated)


def test_a_reply_cut_by_stop_after_stopped_tool_work_does_not_take_its_place():
    from nesicode.server.app import db
    from nesicode.server.jobs import continuity_note, conversation_history

    _, p, t = _client()
    now = time.time()
    work = _run(p, t, "Porta toda la API a FastAPI", "cancelled", "", _tool_steps(8) + [{"type": "cancelled"}],
                at=now - 30)
    _run(p, t, "¿cuánto te falta?", "cancelled", "", [
        {"type": "start"}, {"type": "assistant", "text": "Me faltan los routers de", "interrupted": True},
        {"type": "cancelled"}], at=now - 20)
    _run(p, t, "¿y?", "cancelled", "", [{"type": "start"}, {"type": "cancelled"}], at=now - 10)  # stopped at once
    history, interrupted = conversation_history(db, _run(p, t, "continúa", "running", at=now), window=WINDOW)
    assert TRAIL_MARK in history[0][1] and "MIGRADO-R7" in history[0][1]
    assert "Me faltan los routers de" in history[1][1] and "did NOT finish" in history[1][1]
    assert interrupted == {"id": work["id"], "status": "cancelled", "steps": 8, "later_replied": 1, "later_empty": 1}
    assert "(or stopped before doing any)" in continuity_note(t["id"], interrupted)


# ====================================================================== 2. the budget decides, user messages first
def test_a_rule_from_an_early_turn_stays_in_context_after_many_short_turns(tmp_path, fake):
    prov = fake(lambda conv, rec: Turn("Hecho.", [], "end_turn", Usage(0, 5)), window=WINDOW)
    history = [(f"pregunta corta {i}", f"respuesta corta {i}") for i in range(60)]
    history[2] = ("REGLA PERMANENTE: nunca modifiques la carpeta legacy/ y responde siempre en inglés", "OK.")
    res = _agent(Workspace(tmp_path / "ws"), [], context_limit=WINDOW).run("ahora limpia el código viejo", history)
    assert res.status == "done"
    sent = prov.executor()[0]["messages"]
    users = [m["content"] for m in sent if m["role"] == "user"]
    assert len(users) == 61 and any(u.startswith("REGLA PERMANENTE") for u in users)
    assert not any("earlier turn(s) between" in (m["content"] or "") for m in sent)


def test_a_long_first_message_is_sent_whole_when_it_fits(tmp_path, fake):
    prov = fake(lambda conv, rec: Turn("Hecho.", [], "end_turn", Usage(0, 5)), window=WINDOW)
    lines = [f"Requisito {i}: el endpoint /v1/recurso{i} devuelve JSON paginado y valida la entrada."
             for i in range(200)]
    lines.insert(100, "REGLA-CENTRAL: no cambies la API pública de src/api/")
    spec = "Especificación del proyecto.\n" + "\n".join(lines) + "\nFIN-DE-LA-ESPECIFICACION"
    assert len(spec) > 15_000 and spec.index("REGLA-CENTRAL") > 6_000
    history = [(spec, "Entendido."), ("paso 1", "Hecho.")]
    _agent(Workspace(tmp_path / "ws"), [], context_limit=WINDOW).run("sigue", history)
    users = [m["content"] for m in prov.executor()[0]["messages"] if m["role"] == "user"]
    assert users[0] == spec  # verbatim: ~4k of a 131k window


def test_when_turns_must_go_the_answers_go_before_the_users_messages(tmp_path):
    a = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(tmp_path), context_limit=WINDOW)
    history = [(f"REGLA-{i}: en el paso {i} usa siempre la rama release-{i} y nunca hagas push --force.",
                f"Respuesta {i}: " + "detalle del trabajo hecho. " * 80) for i in range(30)]
    kept = a._fit_history(history, "s", budget=3_000)
    assert [u for u, _ in kept] == [u for u, _ in history]  # every user message, verbatim
    assert kept[0][1] == ANSWER_LEFT_OUT and kept[-1][1].startswith("Respuesta 29:")  # the oldest answers go first
    assert sum(len(u) + len(x) for u, x in kept) <= 3_000 * 4


def test_a_clipped_message_says_where_its_full_text_is(tmp_path):
    ws = Workspace(tmp_path / "ws")
    archive = ".nesicode/chats/7"
    (ws.root / archive / "context").mkdir(parents=True)
    (ws.root / archive / "context" / "history.md").write_text("# Historial\n")
    a = Agent(model="mock/demo", harness=base_harness(), workspace=ws, context_limit=16_384, archive=archive)
    big = "INICIO " + "texto pegado por el usuario " * 2_000 + "FINAL"
    kept = a._fit_history([("primero", "ok"), (big, "visto"), ("tercero", "ok")], "s", budget=1_500)
    clipped = kept[1][0]
    assert clipped.startswith("INICIO") and clipped.endswith("FINAL") and len(clipped) < len(big)
    assert f"`{archive}/context/history.md`" in clipped and "search_history" in clipped


# ====================================================================== 3. cumulative checkpoints
def test_the_newest_checkpoint_always_travels_in_the_history():
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "prepara el esquema", "done", "Hecho.", _tool_steps(3, "a") + [
        {"type": "compacted", "summary": "## Goal\nEsquema.\n## Next Step\nOLD-CHECKPOINT", "n": 1}], at=now - 50)
    compacted = {"type": "compacted", "summary": CHECKPOINT, "n": 1}
    _run(p, t, "aplica las migraciones", "error", "",
         _tool_steps(30, "b") + [compacted] + _tool_steps(2, "c") + [{"type": "error", "text": "503"}], at=now - 40)
    for k, status in enumerate(("cancelled", "error", "cancelled")):  # three «continúa» that did not compact
        _run(p, t, "continúa", status, "", _tool_steps(2, f"d{k}") + [{"type": status}], at=now - 30 + k)
    history, _ = conversation_history(db, _run(p, t, "continúa", "running", at=now), window=WINDOW)
    assert ctxm.CHECKPOINT_MARK in history[1][1] and "NEVER run it again" in history[1][1]
    assert "OLD-CHECKPOINT" not in "".join(a for _, a in history)  # (the newest one holds what is still true)


def _compacting_run(tmp_path, fake, summary_text):
    """A «continúa» at 131,072 whose long replies force one compaction; returns (provider, events)."""
    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return Turn(summary_text, [], "end_turn", Usage(0, 500))
        if _has(rec, ctxm.SUMMARY_PREFIX):
            return Turn("Hecho.", [], "end_turn", Usage(0, 10))
        n = len(prov.executor())
        return Turn(f"Paso {n}: " + "reviso la migración y anoto el estado. " * 1_500,
                    # (a different file each step: the same call with the same result is, by design, a stuck run)
                    [ToolCall(f"r{n}", "read_file", {"path": f"a{n}.txt"})], "tool_use", Usage(0, 4_000))
    prov = fake(policy, window=WINDOW)
    ws = Workspace(tmp_path / "ws")
    for i in range(80):
        (ws.root / f"a{i}.txt").write_text(f"hola {i}\n")
    trail = ("[ACTION] read_file {\"path\": \"notas.md\"}\n[OBSERVATION] " + ctxm.CHECKPOINT_MARK
             + "\n## Next Step\nUNA COPIA VIEJA DENTRO DE UNA SALIDA")  # (a tool output quoting the mark: not it)
    answer = ("[This turn was error at step 31 and did NOT finish. What it had done so far:]\nSteps run: 31 tool calls."
              f"\n\n{ctxm.CHECKPOINT_MARK}\n{CHECKPOINT}\n\n{TRAIL_MARK}\n{trail}")
    events = []
    res = _agent(ws, events, context_limit=WINDOW).run("continúa", [("aplica las migraciones", answer)])
    assert res.status == "done", res.error
    assert [e for e in events if e["type"] == "compacted"], "the scenario must compact"
    return prov, events


def test_a_runs_first_summary_updates_the_earlier_checkpoint(tmp_path, fake):
    prov, _ = _compacting_run(tmp_path, fake, "## Goal\nMigrar.\n## Next Step\nSeguir con 0043_orders.sql")
    prompt = next(r for r in prov.log if r["role"] == "summarizer")["messages"][-1]["content"]
    assert "# PREVIOUS SUMMARY" in prompt
    previous = prompt.split("# PREVIOUS SUMMARY", 1)[1].split("\n# ", 1)[0]
    assert "NEVER run it again" in previous and "an earlier turn of this conversation" in previous
    assert "UNA COPIA VIEJA" not in previous


def test_a_failed_summary_keeps_the_earlier_checkpoint_too(tmp_path, fake):
    _, events = _compacting_run(tmp_path, fake, "")  # the summarizer returns nothing: the deterministic fallback
    comp = next(e for e in events if e["type"] == "compacted")
    assert comp.get("fallback") and "Earlier checkpoint" in comp["summary"] and "NEVER run it again" in comp["summary"]


# ====================================================================== 4. the rebuild after a compaction
def _module(n, seed):
    lines, i = [], 0
    while sum(len(x) + 1 for x in lines) < n:
        lines.append(f"    async function step_{seed}_{i}(ctx: Ctx): Promise<void> {{ await ctx.fence({i}); }}")
        i += 1
    return "\n".join(lines)


def test_a_continue_rebuild_trims_the_earlier_trail_before_this_runs_steps_and_files(tmp_path, fake):
    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return Turn("## Goal\nRevisar el repo.\n## Next Step\nSeguir con el siguiente módulo.", [], "end_turn",
                        Usage(0, 300))
        n = len(prov.executor())
        if _has(rec, ctxm.SUMMARY_PREFIX):
            return Turn("Hecho.", [], "end_turn", Usage(0, 10))
        text = f"Análisis {n}: " + "el orden de los candados y los reintentos parece correcto en este tramo. " * 20
        if n % 6 == 5:
            call = ToolCall(f"w{n}", "write_file", {"path": f"notes/n{n}.md", "content": "hallazgo " * 300})
            return Turn(text, [call], "tool_use", Usage(0, 900))
        return Turn(text, [ToolCall(f"r{n}", "read_file", {"path": f"src/mod{n % 12}.py"})], "tool_use", Usage(0, 600))

    prov = fake(policy, window=WINDOW)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "src").mkdir(parents=True)
    for i in range(12):
        (ws.root / "src" / f"mod{i}.py").write_text(_module(6_000, i))
    trail = "\n".join(f"[ACTION] read_file {{\"path\": \"src/mod{i % 12}.py\"}}\n[OBSERVATION] " + "x" * 300
                      for i in range(250))
    history = [(f"Revisa el repo. Regla: {RULE}.",
                f"[This turn was cancelled at step 250 and did NOT finish.]\n\n{TRAIL_MARK}\n{trail}\n[ÚLTIMO-PASO]")]
    events = []
    res = _agent(ws, events, context_limit=WINDOW, max_steps=0).run("continúa", history)
    assert res.status == "done", res.error
    comp = [e for e in events if e["type"] == "compacted"]
    assert len(comp) == 1
    target = ctxm.post_compact_target(WINDOW)
    assert 0.9 * target <= comp[0]["after"] <= target  # exactly what was over: no overshoot from halving whole parts
    post = _post(prov)[0]["messages"]
    checkpoint = post[-1]["content"]
    steps = re.search(r"## Most recent steps.*?(?=\n## )", checkpoint, re.S).group(0)
    assert len(steps) > 20_000  # this run's recent steps stay (v0.4.6: crushed to ~4k while the old trail stayed whole)
    assert "shown COMPLETE" in checkpoint  # and the files it was reading
    assert post[2] == {"role": "assistant", "content": "(no answer)"}  # (that stopped turn wrote no answer)
    # Nesicode's record of the previous turn opens the next user message: its trail gave way first, keeping its newest
    # steps
    earlier = post[3]["content"].split(RECORD_CLOSE, 1)[0]
    assert earlier.startswith(RECORD_OPEN) and len(earlier) < 60_000 and earlier.rstrip().endswith("[ÚLTIMO-PASO]")
    assert RULE in post[1]["content"]  # the user's message (the rule) stays whole
