"""v0.4.7, round 2 — how earlier turns are replayed (live judges, nesi3 at its real 131,072-token window). Offline: the
test database, temp workspaces, the real OpenAI-compatible conversation with a scripted server, a fake provider.

1. Nesicode's record of an earlier turn (digest, step trail, checkpoint, how it ended) never comes back as the model's
   own words: the answers are replayed as written, the record opens the next user message, labelled, and the roles
   alternate strictly. (A model that saw the footer under its earlier answers copied it, with the previous turn's
   reasoning, into a reply.)
2. The previous turn's trail carries the files it wrote whole when the budget allows; a cut argument says so, with its
   real size and where to read the file.
3. A stopped turn's last words are never headed or replayed as its answer (history.md, the history, harnessing).
4. A short coding turn's trail carries what it read (budget-driven, not a fixed 4,000 chars) to the next turn, from the
   newest turn that used tools."""

from __future__ import annotations

import json
import time

from nesicode.agent import Workspace, trajectory_text
from nesicode.agent import context as ctxm
from nesicode.agent import loop as loopmod
from nesicode.agent.loop import DIGEST_MARK, RECORD_CLOSE, RECORD_OPEN, STOP_NOTES, TRAIL_MARK, split_record
from nesicode.llm.base import Turn, Usage
from nesicode.llm.openai_compat import OpenAICompatProvider

from test_fix_context import _agent, _client, _post, _run, fake  # noqa: F401  (fake is a fixture)

WINDOW = 131_072
HEADS = (DIGEST_MARK, TRAIL_MARK, ctxm.CHECKPOINT_MARK, "[THINKING]", "[ACTION]", "[OBSERVATION]", "[END] status=",
         "did NOT finish", "[harness]")


class _Server(OpenAICompatProvider):
    """The real OpenAI-compatible conversation (what nesi3 gets through LiteLLM); its server is `reply(messages, n)`."""

    def __init__(self, reply):
        super().__init__("nesilabs", "http://127.0.0.1:9", "k")
        self.reply, self.bodies = reply, []

    def chat(self, body, options, on_event=None):
        self.bodies.append(json.loads(json.dumps(body["messages"])))
        msg = self.reply(self.bodies[-1], len(self.bodies))
        finish = "tool_calls" if msg.get("tool_calls") else "stop"
        usage = {"prompt_tokens": sum(len(json.dumps(m)) for m in body["messages"]) // 4, "completion_tokens": 40}
        return {"choices": [{"message": {"role": "assistant", **msg}, "finish_reason": finish}], "usage": usage}, False


def _server(monkeypatch, reply) -> _Server:
    prov = _Server(reply)
    monkeypatch.setattr(loopmod, "get_provider", lambda spec, keys=None: (prov, "nesi3"))
    return prov


def _strictly_alternating(messages) -> bool:
    roles = [m["role"] for m in messages]
    return roles[0] == "system" and roles[-1] == "user" and all(
        r == ("user" if k % 2 == 0 else "assistant") for k, r in enumerate(roles[1:]))


def _record(user_message: str) -> str:
    """Nesicode's record that opens a user message ('' when there is none)."""
    if not user_message.startswith(RECORD_OPEN):
        return ""
    return user_message[len(RECORD_OPEN): user_message.index(RECORD_CLOSE)]


def _needle_turns(p, t, now):
    """Two finished turns with tools, as in the live needle run: the second re-read a line to answer."""
    _run(p, t, "Lee config/app.ini y dime el release_code", "done", "El release_code es ORCA-7731 (línea 83).", [
        {"type": "start"}, {"type": "thinking", "text": "Leo el fichero de configuración."},
        {"type": "tool_call", "id": "a1", "name": "read_file", "args": {"path": "config/app.ini"}},
        {"type": "tool_result", "id": "a1", "name": "read_file", "output": "    83\trelease_code = ORCA-7731"},
        {"type": "assistant", "text": "El release_code es ORCA-7731 (línea 83)."},
        {"type": "done", "status": "done", "steps": 2}], at=now - 30)
    _run(p, t, "¿En qué línea exactamente?", "done", "En la línea 83: `release_code = ORCA-7731`.", [
        {"type": "start"},
        {"type": "thinking", "text": "The user asks on which line. I have it from history: line 83. RAZON-PRIVADA-T2"},
        {"type": "tool_call", "id": "b1", "name": "read_file",
         "args": {"path": "config/app.ini", "offset": 79, "limit": 10}},
        {"type": "tool_result", "id": "b1", "name": "read_file",
         "output": "    79\t[release]\n    83\trelease_code = ORCA-7731"},
        {"type": "assistant", "text": "En la línea 83: `release_code = ORCA-7731`."},
        {"type": "done", "status": "done", "steps": 2}], at=now - 20)


# ====================================================================== 1. the record never in the model's voice
def test_earlier_answers_are_replayed_as_written_and_the_record_opens_the_next_user_message(tmp_path, monkeypatch):
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    _needle_turns(p, t, now)
    task = "¿Qué contenía el tercer log que leíste? Dame 3 líneas literales."
    history, _ = conversation_history(db, _run(p, t, task, "running", at=now), window=WINDOW)
    prov = _server(monkeypatch, lambda messages, n: {"content": "El tercer log fue cache-node.log."})
    res = _agent(Workspace(tmp_path / "ws"), [], context_limit=WINDOW).run(task, history)
    assert res.status == "done", res.error
    sent = prov.bodies[0]
    assert _strictly_alternating(sent)  # a strict server rejects two user messages in a row
    assert [m["content"] for m in sent if m["role"] == "assistant"] == [
        "El release_code es ORCA-7731 (línea 83).", "En la línea 83: `release_code = ORCA-7731`."]  # as written
    first, second, last = (m["content"] for m in sent if m["role"] == "user")
    assert first == "Lee config/app.ini y dime el release_code"
    assert DIGEST_MARK in _record(second) and second.endswith("\n\n¿En qué línea exactamente?")
    trail = _record(last)  # the previous turn's record, right before the new message
    assert TRAIL_MARK in trail and "RAZON-PRIVADA-T2" in trail
    assert '[ACTION] read_file {"path": "config/app.ini"' in trail
    assert last.endswith(RECORD_CLOSE + "\n\n" + task)
    system = sent[0]["content"]
    assert "never copy it, or its format, into a reply" in system and "If something you need is not shown" in system


def test_a_model_that_echoes_footers_finds_none_under_its_own_answers(tmp_path, monkeypatch):
    # the live failure: the model ended its answer with the footer its earlier answers carried ([Key steps of this
    # turn] + the previous turn's trail and reasoning). This server echoes any such footer of an earlier answer.
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    _needle_turns(p, t, now)
    task = "¿Qué contenía el tercer log?"
    history, _ = conversation_history(db, _run(p, t, task, "running", at=now), window=WINDOW)

    def echo(messages, n):
        footer = next((m["content"][m["content"].index(DIGEST_MARK):] for m in messages
                       if m["role"] == "assistant" and DIGEST_MARK in (m["content"] or "")), "")
        return {"content": "El tercer log fue cache-node.log." + (f"\n\n{footer}" if footer else "")}
    _server(monkeypatch, echo)
    res = _agent(Workspace(tmp_path / "ws"), [], context_limit=WINDOW).run(task, history)
    assert res.final_text == "El tercer log fue cache-node.log."


def test_an_answer_stored_with_an_echoed_footer_comes_back_clean(tmp_path, fake):
    # v0.4.6 stored such answers: the next turns must not replay the copied trail, as the model's words or as a record
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    polluted = ("El tercer log fue cache-node.log.\n\n" + DIGEST_MARK + "\nSteps run: 2 tool calls.\n" + TRAIL_MARK
                + "\n[THINKING] RAZONAMIENTO-COPIADO-DE-OTRO-TURNO\n[ACTION] read_file {\"path\": \"config/app.ini\"}\n"
                "[END] status=done steps=2")
    _run(p, t, "¿Qué contenía el tercer log?", "done", polluted, [
        {"type": "start"},
        {"type": "tool_call", "id": "c1", "name": "read_file", "args": {"path": "logs/cache-node.log"}},
        {"type": "tool_result", "id": "c1", "name": "read_file", "output": "1\tDEBUG [cache-node] db: SELECT"},
        {"type": "assistant", "text": polluted}, {"type": "done", "status": "done", "steps": 2}], at=now - 10)
    history, _ = conversation_history(db, _run(p, t, "gracias", "running", at=now), window=WINDOW)
    prov = fake(lambda conv, rec: Turn("De nada.", [], "end_turn", Usage(0, 5)), window=WINDOW)
    _agent(Workspace(tmp_path / "ws"), [], context_limit=WINDOW).run("gracias", history)
    sent = prov.executor()[0]["messages"]
    assert sent[2] == {"role": "assistant", "content": "El tercer log fue cache-node.log."}
    assert "RAZONAMIENTO-COPIADO" not in json.dumps(sent, ensure_ascii=False)  # (its real trail is built from events)
    assert "logs/cache-node.log" in _record(sent[3]["content"])


def test_stop_question_continue_keeps_the_record_in_the_user_role_and_the_reasoning_replay(tmp_path, monkeypatch):
    from nesicode.server.app import db
    from nesicode.server.jobs import continuity_note, conversation_history

    _, p, t = _client()
    now = time.time()
    events = [{"type": "start"}]
    for i in range(12):
        events += [{"type": "thinking", "text": f"DECISION-{i}: el router r{i} pasa a FastAPI"},
                   {"type": "tool_call", "id": f"e{i}", "name": "edit_file",
                    "args": {"path": f"src/api/r{i}.py", "old": "flask", "new": "fastapi"}},
                   {"type": "tool_result", "id": f"e{i}", "name": "edit_file", "output": f"MIGRADO-R{i}"}]
    events += [{"type": "assistant", "text": "Sigo con r12.", "interrupted": True}, {"type": "cancelled"}]
    _run(p, t, "Porta toda la API a FastAPI", "cancelled", "Sigo con r12.", events, at=now - 30)
    _run(p, t, "¿cuánto te falta?", "done", "Me faltan 3 routers.", [
        {"type": "start"}, {"type": "assistant", "text": "Me faltan 3 routers."}, {"type": "done", "status": "done"}],
         at=now - 20)
    history, interrupted = conversation_history(db, _run(p, t, "vale, continúa", "running", at=now), window=WINDOW)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "src" / "api").mkdir(parents=True)
    (ws.root / "src" / "api" / "r12.py").write_text("import flask\n")

    def server(messages, n):
        if n == 1:
            call = {"id": "k1", "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": "src/api/r12.py"})}}
            return {"content": "", "reasoning_content": "RAZ-PASO-1: r12 es el siguiente", "tool_calls": [call]}
        return {"content": "r12 migrado; faltan r13 y r14."}
    prov = _server(monkeypatch, server)
    res = _agent(ws, [], context_limit=WINDOW, max_steps=0,
                 system_note=continuity_note(t["id"], interrupted)).run("vale, continúa", history)
    assert res.status == "done", res.error
    first = prov.bodies[0]
    assert _strictly_alternating(first)
    assert first[2] == {"role": "assistant", "content": "(no answer)"}  # its last words are not an answer
    record = _record(first[3]["content"])  # its record opens the next user message (the question)
    assert "did NOT finish" in record and "DECISION-11" in record and record.rstrip().endswith("Sigo con r12.")
    assert first[3]["content"].endswith("¿cuánto te falta?") and first[4]["content"] == "Me faltan 3 routers."
    assert first[5]["content"] == "vale, continúa"  # (the question's turn has no record)
    assert "in the user message right after it" in first[0]["content"]
    second = prov.bodies[1]  # the tool loop still replays its reasoning after the last user message
    assert second[-2]["role"] == "assistant" and second[-2]["reasoning_content"].startswith("RAZ-PASO-1")
    assert second[-1]["role"] == "tool" and _strictly_alternating(second[:-2])


def test_the_rebuild_after_a_compaction_keeps_the_record_in_the_user_role(tmp_path, fake):
    from test_v047_history import _compacting_run

    prov, _ = _compacting_run(tmp_path, fake, "## Goal\nMigrar.\n## Next Step\nSeguir con 0043_orders.sql")
    post = _post(prov)[0]["messages"]
    assert _strictly_alternating(post) and post[2] == {"role": "assistant", "content": "(no answer)"}
    record = _record(post[3]["content"])  # the stopped turn's record opens the checkpoint message
    assert ctxm.CHECKPOINT_MARK in record and TRAIL_MARK in record and "did NOT finish" in record
    assert post[3]["content"].split(RECORD_CLOSE, 1)[1].lstrip().startswith(ctxm.SUMMARY_PREFIX)


def test_a_note_the_harness_added_for_the_user_is_not_replayed_as_the_models_words(tmp_path, fake):
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    final = "Resumen: migré 3 de 5 routers; faltan orders y users.\n\n" + STOP_NOTES["context_full"]
    _run(p, t, "Porta la API", "max_steps", final, [
        {"type": "start"},
        {"type": "tool_call", "id": "a", "name": "edit_file", "args": {"path": "r1.py", "old": "a", "new": "b"}},
        {"type": "tool_result", "id": "a", "name": "edit_file", "output": "edited r1.py"},
        {"type": "assistant", "text": "Resumen: migré 3 de 5 routers; faltan orders y users."},
        {"type": "stopped", "reason": "context_full", "text": STOP_NOTES["context_full"]},
        {"type": "done", "status": "max_steps", "steps": 2, "stop_reason": "context_full"}], at=now - 10)
    history, _ = conversation_history(db, _run(p, t, "continúa", "running", at=now), window=WINDOW)
    prov = fake(lambda conv, rec: Turn("Sigo.", [], "end_turn", Usage(0, 5)), window=WINDOW)
    _agent(Workspace(tmp_path / "ws"), [], context_limit=WINDOW).run("continúa", history)
    sent = prov.executor()[0]["messages"]
    assert sent[2]["content"] == "Resumen: migré 3 de 5 routers; faltan orders y users."
    assert "[Nesicode ended this turn early: the model's context window was full" in _record(sent[3]["content"])


def test_trimming_an_earlier_turn_trims_its_answer_and_its_record_apart(tmp_path, fake):
    steps = "\n".join(f"[ACTION] read_file {{\"path\": \"src/m{i}.py\"}}\n[OBSERVATION] " + "x" * 200
                      for i in range(400))
    answer = ("INFORME " + "hallazgo detallado. " * 2_000 + "FIN-INFORME\n\n" + DIGEST_MARK + "\nSteps run: 401 tool "
              "calls.\n\n" + TRAIL_MARK + "\n" + steps + "\n[ACTION] read_file {\"path\": \"ULTIMO.py\"}")
    prov = fake(lambda conv, rec: Turn("Sigo.", [], "end_turn", Usage(0, 5)), window=32_768)
    _agent(Workspace(tmp_path / "ws"), [], context_limit=32_768).run("sigue", [("revisa todo el repo", answer)])
    sent = prov.executor()[0]["messages"]
    assert _strictly_alternating(sent) and len(json.dumps(sent)) < len(answer)  # (it had to be trimmed)
    words = sent[2]["content"]
    assert words.startswith("INFORME") and not any(h in words for h in HEADS)
    record = _record(sent[3]["content"])
    assert record.lstrip().startswith(DIGEST_MARK) and record.rstrip().endswith('"ULTIMO.py"}')  # newest steps kept
    assert "older steps of this turn omitted" in record
    combined = loopmod._trim_answer(answer, 20_000)  # (the string form keeps the two parts apart too)
    assert len(combined) <= 20_000 and split_record(combined)[1].startswith(DIGEST_MARK)


def test_split_record_knows_every_record_head_and_nothing_else():
    for head in ("[This turn was cancelled at step 3 and did NOT finish. What it had done so far:]",
                 "[this turn was error before doing any work]",
                 "[this turn was cancelled while still reasoning, before doing any work; the end of its reasoning:]",
                 "[this turn ended as max_steps without an answer]", ctxm.CHECKPOINT_MARK, DIGEST_MARK, TRAIL_MARK,
                 "[Nesicode ended this turn early: the model's context window was full; the work stopped there]",
                 "[3 earlier turn(s) between this one and the next shown are omitted here — search_history finds them]",
                 RECORD_OPEN):
        assert split_record(f"Hecho.\n\n{head}\nmás") == ("Hecho.", f"{head}\nmás")
    prose = ("[This turn of events](https://x.org) lo cambia todo.\n[Key steps] abajo.\nThis turn was fine.\n"
             "Cito el marcador [PREVIOUS TURN — STEPS] en medio de una línea.")
    assert split_record(prose) == (prose, "")


# ====================================================================== 2. the files a stopped turn wrote, whole
SAY = "El proyecto es casi vacío (solo un README). Creo el paquete con los cuatro módulos y los tests."


def _stopped_coding_turn():
    files = {"geometria/__init__.py": "from .puntos import Punto\n" * 14,
             "geometria/figuras.py": "".join(f"class Figura{i}:\n    def area(self):\n        raise ValueError('no')\n"
                                             for i in range(36)),
             "geometria/cli.py": "".join(f"    sub.add_parser('figura{i}', help='calcula el área')\n"
                                         for i in range(42))}
    assert all(1_900 < len(files[f]) < 2_500 for f in ("geometria/figuras.py", "geometria/cli.py"))  # (as live)
    events = [{"type": "start"},
              {"type": "tool_call", "id": "l", "name": "list_dir", "args": {"depth": 2}},
              {"type": "tool_result", "id": "l", "name": "list_dir", "output": "README.md (104 B)"},
              {"type": "thinking", "text": "Write all files."},
              {"type": "assistant", "text": SAY}]
    for i, (path, content) in enumerate(files.items()):
        events += [{"type": "tool_call", "id": f"w{i}", "name": "write_file",
                    "args": {"path": path, "content": content}},
                   {"type": "tool_result", "id": f"w{i}", "name": "write_file",
                    "output": f"created {path} ({len(content)} chars)"}]
    return files, events + [{"type": "cancelled"}]


def test_the_previous_turns_trail_carries_the_files_it_wrote_whole(tmp_path):
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    files, events = _stopped_coding_turn()
    _run(p, t, "Crea el paquete geometria/ con sus tests", "cancelled", SAY, events, at=now - 10)
    history, interrupted = conversation_history(db, _run(p, t, "continúa", "running", at=now), window=WINDOW)
    words, record = split_record(history[0][1])
    assert words == "" and interrupted["steps"] == 4  # its interim words are not an answer
    for content in files.values():
        assert json.dumps(content, ensure_ascii=False)[1:-1] in record  # whole (v0.4.6: cut at 1,500 chars)
    assert "...}" not in record and "cut here" not in record
    assert ("Files it wrote or edited (calls that succeeded): geometria/__init__.py, geometria/figuras.py, "
            "geometria/cli.py") in record


def test_a_cut_argument_says_its_real_size_and_where_to_read_the_file():
    files, events = _stopped_coding_turn()
    trail = trajectory_text(events, 3_000, obs_chars=4_000, args_chars=1_500, fit="tail", grow=16)
    assert len(trail) <= 3_000 and "...}" not in trail
    n = len(files["geometria/cli.py"])
    assert f"…[cut here: {n:,} chars in all; read_file geometria/cli.py to see the file now]" in trail
    long = {"command": "python - <<'EOF'\n" + "print('x')\n" * 400 + "EOF", "timeout": 60}
    shown = loopmod._args_shown("bash", long, 1_500)
    assert len(shown) < 1_700 and f"…[cut here: {len(long['command']):,} chars in all]" in shown
    assert json.loads(shown)["timeout"] == 60  # (still JSON: only the long value was cut)


# ====================================================================== 3. last words are not an answer
def test_history_md_never_heads_a_stopped_turns_last_words_as_its_answer(tmp_path):
    from nesicode.server.app import db
    from nesicode.server.jobs import _write_thread_history, follow_up_task

    _, p, t = _client()
    now = time.time()
    _, events = _stopped_coding_turn()
    _run(p, t, "Crea el paquete geometria/ con sus tests", "cancelled", SAY, events, at=now - 20)
    _run(p, t, "¿qué falta?", "done", "Faltan los tests.", [
        {"type": "start"}, {"type": "assistant", "text": "Faltan los tests."}, {"type": "done", "status": "done"}],
         at=now - 10)
    cur = _run(p, t, "continúa", "running", at=now)
    ws = Workspace(tmp_path / "ws")
    _write_thread_history(db, ws, cur, ".nesicode/chats/3")
    md = (ws.root / ".nesicode/chats/3/context/history.md").read_text()
    stopped, answered = md.split("## Turno #")[1:]
    assert "### Respuesta final" not in stopped
    assert f"### Sin respuesta final (turno cancelled, sin terminar); sus últimas palabras:\n{SAY}" in stopped
    assert answered.rstrip().endswith("### Respuesta final\nFaltan los tests.")
    assert "[this turn ended cancelled without an answer; its last words: El proyecto es casi vacío" in \
        follow_up_task(db, cur)  # (harnessing a follow-up replays the conversation too)


def test_an_old_stopped_turn_brings_its_last_words_as_such():
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "Crea el paquete y sus tests", "cancelled", "VOY-A-CREAR-LOS-TESTS", [
        {"type": "start"}, {"type": "assistant", "text": "VOY-A-CREAR-LOS-TESTS"}, {"type": "cancelled"}], at=now - 100)
    for i in range(22):  # past the newest 20 turns, older ones bring only their answer
        _run(p, t, f"pregunta {i}", "done", f"respuesta {i}", [{"type": "assistant", "text": f"respuesta {i}"}],
             at=now - 90 + i)
    history, _ = conversation_history(db, _run(p, t, "sigue", "running", at=now), window=WINDOW)
    words, record = split_record(history[0][1])
    assert history[0][0] == "Crea el paquete y sus tests" and words == ""
    assert record == ("[this turn ended as cancelled without an answer]\n"
                      "Its last words (it did not finish): VOY-A-CREAR-LOS-TESTS")


# ====================================================================== 4. a short coding turn's reads reach the next
def _reading_turn(docs: int = 5, size: int = 9_000):
    events = [{"type": "start"}]
    for k in range(docs):
        half = "\n".join(f"{n:>6}\tdoc{k} línea {n}: requisito de la especificación" for n in range(size // 100))
        body = half + f"\n  9999\tMITAD-DOC-{k}: el límite es {k * 7 + 3} peticiones/s\n" + half
        events += [{"type": "thinking", "text": f"Leo el documento {k}."},
                   {"type": "tool_call", "id": f"d{k}", "name": "read_file", "args": {"path": f"docs/0{k}.md"}},
                   {"type": "tool_result", "id": f"d{k}", "name": "read_file", "output": body}]
    return events + [{"type": "assistant", "text": "Resumen de los 5 documentos: listo."},
                     {"type": "done", "status": "done", "steps": docs + 1}]


def test_a_short_coding_turns_reads_reach_the_next_turn_whole_when_the_window_has_room():
    from nesicode.server.app import db
    from nesicode.server.jobs import conversation_history

    _, p, t = _client()
    now = time.time()
    _run(p, t, "Lee los 5 documentos de docs/", "done", "Resumen de los 5 documentos: listo.", _reading_turn(),
         at=now - 20)
    _run(p, t, "gracias", "done", "De nada.", [{"type": "assistant", "text": "De nada."}], at=now - 10)
    cur = _run(p, t, "ahora implementa el límite del documento 3", "running", at=now)
    history, _ = conversation_history(db, cur, window=WINDOW)
    record = split_record(history[0][1])[1]  # the newest turn that used tools has the trail (not the reply after it)
    trail = record[record.index(TRAIL_MARK):]
    assert all(f"MITAD-DOC-{k}" in trail for k in range(5))  # v0.4.6: each output cut to 4,000 chars
    assert len(trail) <= WINDOW * 0.2 * 3.5 + 100 and TRAIL_MARK not in history[1][1]
    small, _ = conversation_history(db, cur, window=16_384)  # little room: the budget decides, newest steps first
    trail = small[0][1][small[0][1].index(TRAIL_MARK):]
    assert len(trail) <= 16_384 * 0.2 * 3.5 + 100 and '"docs/04.md"' in trail and '"docs/00.md"' not in trail
    assert "of this turn are not shown here" in trail
