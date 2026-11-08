import json

import pytest

from nesicode.agent import Agent, ToolError, Workspace
from nesicode.agent.hooks import Hooks
from nesicode.agent.tools import edit_file, read_file, write_file
from nesicode.harness import base_harness, validate
from nesicode.harnessing import HarnessingConfig, HarnessingRun, parse_edits


def test_base_harness_is_valid():
    assert validate(base_harness()) == []


def test_skill_without_trigger_is_rejected():
    h = base_harness()
    h["skills/foo/SKILL.md"] = "---\nname: foo\ndescription: does foo\n---\nbody\n"
    errs = validate(h)
    assert any("never named" in e for e in errs)
    h["guardrails.md"] += "- When foo happens, load the `foo` skill.\n"
    assert validate(h) == []


def test_off_surface_and_missing_script():
    h = base_harness()
    h["evil/../x.md"] = "x"
    h["guardrails.md"] += "- run scripts/check.py first\n"
    errs = validate(h)
    assert any("off the editable surface" in e for e in errs)
    assert any("scripts/check.py" in e for e in errs)


def test_bad_hooks_rejected():
    h = base_harness()
    h["plugins/hooks.json"] = json.dumps({"pre_tool": [{"tool": "bash", "action": "explode"}]})
    assert any("action must be" in e for e in validate(h))


def test_parse_edits_replace_and_delete():
    cur = {"system_prompt.md": "a\n", "memory.md": "m\n"}
    new, touched = parse_edits('<file path="memory.md">\nnew\n</file>\n<delete path="system_prompt.md"/>', cur)
    assert touched and new == {"memory.md": "new\n"}
    same, touched = parse_edits("<notes>KEEP_BASE</notes>", cur)
    assert not touched and same == cur


def test_workspace_confinement(tmp_path):
    ws = Workspace(tmp_path / "ws")
    with pytest.raises(ToolError):
        ws.resolve("../outside.txt")
    with pytest.raises(ToolError):
        write_file(ws, {"path": "/etc/passwd", "content": "x"})
    write_file(ws, {"path": "a/b.txt", "content": "hello world\n"})
    assert "hello" in read_file(ws, {"path": "a/b.txt"})
    edit_file(ws, {"path": "a/b.txt", "old": "world", "new": "nesi"})
    assert (tmp_path / "ws/a/b.txt").read_text() == "hello nesi\n"
    with pytest.raises(ToolError):
        edit_file(ws, {"path": "a/b.txt", "old": "missing", "new": "x"})


def test_hooks_block_append_truncate():
    hooks = Hooks({
        "pre_tool": [{"tool": "bash", "match": "rm -rf /", "action": "block", "message": "no"}],
        "post_tool": [
            {"tool": "*", "match": "ModuleNotFoundError", "action": "append", "message": "install deps"},
            {"tool": "*", "action": "truncate", "max_chars": 50},
        ],
    })
    assert hooks.before("bash", {"command": "rm -rf / "})[0] == "no"
    assert hooks.before("bash", {"command": "ls"})[0] is None
    assert "[hint] install deps" in hooks.after("bash", "ModuleNotFoundError: x")
    assert "trimmed by hook" in hooks.after("read_file", "x" * 500)


def test_agent_blocked_command_is_reported(tmp_path):
    h = base_harness()
    agent = Agent(model="mock/demo", harness=h, workspace=Workspace(tmp_path))
    from nesicode.llm import ToolCall, Usage

    res = agent._call_tool(ToolCall("1", "bash", {"command": "rm -rf / "}), Usage())
    assert res.is_error and "Blocked" in res.content


def test_harness_changes_executor_behavior(tmp_path):
    """The verify rule proposed by the mock makes the executor re-read its file."""
    h = base_harness()
    plain = Agent(model="mock/demo", harness=h, workspace=Workspace(tmp_path / "a")).run("task")
    h2 = dict(h, **{"guardrails.md": h["guardrails.md"] + "- verify written files\n"})
    verified = Agent(model="mock/demo", harness=h2, workspace=Workspace(tmp_path / "b")).run("task")
    assert verified.steps == plain.steps + 1


@pytest.mark.parametrize("mode,k,rollouts,stages", [("parallel", 1, 4, 3), ("sequential", 2, 8, 6)])
def test_engine_schedule(tmp_path, mode, k, rollouts, stages):
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("print(1)\n")
    cfg = HarnessingConfig(task="t", executor_model="mock/demo", improver_model="mock/demo", k=k, mode=mode,
                           check_command="test -f NESICODE_RESULT.md")
    hr = HarnessingRun(cfg, base_harness=base_harness(), source_ws=src, run_dir=tmp_path / "run")
    r = hr.run()
    assert len(r["rollouts"]) == rollouts and len(r["stages"]) == stages  # 4k rollouts, 3k stages
    assert all(ro["check_code"] == 0 for ro in r["rollouts"].values())
    assert r["selected"] in r["rollouts"]
    assert validate(r["adopted_harness"]) == []
    assert r["reflection"]["experience"]["lesson"]
    assert not (src / "NESICODE_RESULT.md").exists()  # source untouched until applied
    hr.apply_selected(r["selected"], src)
    assert (src / "NESICODE_RESULT.md").exists()


@pytest.mark.skipif(__import__("sys").platform != "darwin", reason="macOS sandbox-exec cage")
def test_os_sandbox_cage(tmp_path):
    from nesicode.config import settings

    root = settings.data_dir / "workspaces" / "_cage_test"
    mine = Workspace(root / "a" / "ws")
    other = Workspace(root / "b" / "ws")
    (other.root / "secret.txt").write_text("other")
    (mine.root / "own.txt").write_text("mine")
    try:
        assert mine.run("cat own.txt")[1].strip() == "mine"
        assert "not permitted" in mine.run(f"cat {other.root}/secret.txt")[1]
        assert "not permitted" in mine.run(f"ls {__import__('os').path.expanduser('~')}")[1]
        assert "not permitted" in mine.run(f"echo x > {root}/escape.txt")[1]
        assert not (root / "escape.txt").exists()
        assert mine.run("python3 -c 'print(6*7)'")[1].strip() == "42"
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


def test_suggest_roles_picks_newest_nesi_and_vision():
    from nesicode.server.app import suggest_roles

    assert suggest_roles(["nesi", "nesi2", "nesi3", "redarrow"]) == {"executor_model": "nesi3"}
    assert suggest_roles(["nesi3", "nesi10", "nesi4", "nesi-vision", "claude-haiku-4-5"]) == {
        "executor_model": "nesi10", "vision_model": "nesi-vision"}
    assert suggest_roles(["gpt-5", "grok-4.7"]) == {}


def test_share_file_tool(tmp_path):
    from nesicode.llm import ToolCall, Usage

    events = []
    ws = Workspace(tmp_path / "ws")
    (ws.root / "deck.pptx").write_bytes(b"PK\x03\x04fake")
    a = Agent(model="mock/demo", harness=base_harness(), workspace=ws, emit=events.append)
    assert "share_file" in {s.name for s in a.tool_specs()}
    r = a._call_tool(ToolCall("1", "share_file", {"path": "deck.pptx", "caption": "Ventas Q3"}), Usage())
    assert not r.is_error
    ev = [e for e in events if e["type"] == "file"][0]
    assert ev["path"] == "deck.pptx" and ev["size"] == 8 and ev["caption"] == "Ventas Q3"
    assert a._call_tool(ToolCall("2", "share_file", {"path": "../x"}), Usage()).is_error


def test_base_harness_has_documents_skill():
    h = base_harness()
    assert "skills/documents/SKILL.md" in h and "`documents`" in h["guardrails.md"]
    assert validate(h) == []


def test_credentials_guardrail_in_base():
    assert "Never type passwords" in base_harness()["guardrails.md"]
    assert validate(base_harness()) == []


def test_absolutize_mcp_paths(tmp_path):
    from nesicode.server.jobs import absolutize_paths

    ws = Workspace(tmp_path / "ws")
    out = absolutize_paths({"filename": "informe.docx", "url": "https://x.y", "text": "hola.docx",
                            "output_path": "/abs/x.pptx", "filePath": "sub/a.png", "count": 3}, ws)
    assert out["filename"] == str(ws.root / "informe.docx")
    assert out["filePath"] == str(ws.root / "sub" / "a.png")
    assert out["url"] == "https://x.y" and out["text"] == "hola.docx"  # not path-like keys
    assert out["output_path"] == "/abs/x.pptx" and out["count"] == 3


def test_context_compaction(tmp_path):
    events = []
    a = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(tmp_path / "ws"),
              emit=events.append, context_limit=400, output_limit=1000)
    res = a.run("write the report " + "x" * 2000)
    assert res.status in ("done", "max_steps")
    comp = [e for e in events if e["type"] == "compacted"]
    assert comp and comp[0]["limit"] == 400 and comp[0]["used"] > 300
    assert len(comp) <= 8  # bounded
    done = [e for e in events if e["type"] == "done"][-1]
    assert done["context"]["limit"] == 400


def test_history_fits_context(tmp_path):
    a = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(tmp_path / "ws"), context_limit=10_000)
    hist = [("q" * 3000, "a" * 3000)] * 10  # each turn ~2000 tokens; only ~1 fits in 30% of 10k
    kept = a._fit_history(hist, "system")
    assert 0 < len(kept) < 3


def test_known_limits():
    from nesicode.llm import known_limits

    assert known_limits("claude-opus-5-5") == (1_000_000, 128_000)
    assert known_limits("nesi3")[0] == 131_072 and known_limits("nesi2")[0] == 245_760
    assert known_limits("grok-4.7")[0] == 256_000
    assert known_limits("misterio") is None


def test_parse_limit_from_provider_errors():
    from nesicode.llm import parse_limit

    assert parse_limit("This model's maximum context length is 131072 tokens. However, you requested 600010") == 131072
    assert parse_limit("request (600010 tokens) exceeds the available context size (245760 tokens), try increasing it") == 245760
    assert parse_limit("nothing useful") is None


def test_agent_learns_limit_from_context_error(tmp_path, monkeypatch):
    from nesicode.llm import LLMError
    from nesicode.llm.mock import MockConversation

    calls = {"n": 0}
    orig = MockConversation.step

    def step(self, on_event=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMError("HTTP 400: This model's maximum context length is 131072 tokens. However, you requested 600010")
        return orig(self, on_event)

    monkeypatch.setattr(MockConversation, "step", step)
    learned, events = [], []
    a = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(tmp_path / "ws"), emit=events.append,
              context_limit=245_000, on_limit=learned.append)
    res = a.run("write it")
    assert res.status == "done" and learned == [131072] and a.context_limit == 131072
    assert any(e["type"] == "context_learned" for e in events) and any(e["type"] == "compacted" for e in events)


def test_layer0_big_outputs_saved_to_disk(tmp_path):
    from nesicode.llm import ToolCall, Usage

    ws = Workspace(tmp_path / "ws")
    (ws.root / "big.txt").write_text("\n".join(f"line {i} " + "x" * 80 for i in range(2000)))
    a = Agent(model="mock/demo", harness=base_harness(), workspace=ws)
    r = a._call_tool(ToolCall("c1", "read_file", {"path": "big.txt"}), Usage())
    assert "full output saved to .nesicode/outputs/" in r.content and len(r.content) < 10_000
    saved = a._saved["c1"][1]
    assert (ws.root / saved).read_text().count("\n") > 1500  # nothing lost


def test_layer1_masking_keeps_recent_and_errors():
    from nesicode.agent.context import mask_marker
    from nesicode.llm import ModelOptions, ToolResult
    from nesicode.llm.mock import MockConversation

    conv = MockConversation("m", "", [], ModelOptions())
    conv.add_tool_results([ToolResult(f"c{i}", ("ERROR boom\n" if i == 0 else "") + "y" * 2000) for i in range(12)])
    markers = {"c0": mask_marker("bash", ".nesicode/outputs/c0.txt", True)}
    freed = conv.mask_tool_results(8, lambda cid, old: (markers.get(cid) or mask_marker("read_file", f".nesicode/outputs/{cid}.txt", False))(cid, old))
    contents = [m[1] for m in conv.messages]
    assert freed > 6000
    assert contents[0].startswith("[old bash error kept short: ERROR boom")
    assert contents[1].startswith("[old read_file output cleared") and ".nesicode/outputs/c1.txt" in contents[1]
    assert all(c.startswith("y") for c in contents[4:])  # the last 8 are intact


def test_layer2_checkpoint_md_and_search(tmp_path):
    events = []
    ws = Workspace(tmp_path / "ws")
    a = Agent(model="mock/demo", harness=base_harness(), workspace=ws, emit=events.append, context_limit=600)
    res = a.run("write the report about src/app.py " + "z" * 1500)
    assert res.status in ("done", "max_steps")
    comp = [e for e in events if e["type"] == "compacted"]
    # archive files are named per run (<YYYYmmdd-HHMMSS>-compact-N.md) so a later run never overwrites them
    import re
    assert comp and re.fullmatch(r"\.nesicode/context/\d{8}-\d{6}(-\d+)?-compact-1\.md", comp[0]["path"])
    md = (ws.root / comp[0]["path"]).read_text()
    assert md.startswith("# Compactación 1") and "transcript-1.md" in md
    from nesicode.agent.context import search_history
    assert "transcript-1.md" in search_history(ws, {"query": "ACTION"})
    # the archive survives the next run's harness install
    Agent(model="mock/demo", harness=base_harness(), workspace=ws).run("otra")
    assert (ws.root / comp[0]["path"]).exists()


def test_extract_anchors():
    from nesicode.agent.context import extract_anchors

    a = extract_anchors("edited src/app.py and tests/test_x.py, see https://x.com/a, commit 3f2a9bc1, PR #42\nValueError: bad thing")
    assert "src/app.py" in a["files"] and "https://x.com/a" in a["urls"] and "3f2a9bc1" in a["commits"]
    assert "42" in a["issues"] and any("ValueError" in e for e in a["errors"])


def test_office_connector_roundtrip(tmp_path):
    import struct
    import zlib

    from nesicode import office

    ws = Workspace(tmp_path / "ws")
    call = lambda tool, **a: office.TOOLS[tool][1](a, ws)  # noqa: E731
    # a tiny PNG for image blocks / slides
    raw = b"".join(b"\x00" + b"\x10\x80\xf0" * 4 for _ in range(4))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    (ws.root / "dot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
                                      + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

    text, rel = call("word_write", path="docs/informe.docx", title="Informe", blocks=[
        {"type": "heading", "text": "Resumen", "level": 1}, "Hola **{{nombre}}**, esto es *importante*.",
        {"type": "bullets", "items": ["uno", "dos"]}, {"type": "table", "rows": [["Mes", "Ventas"], ["Ene", 10]]},
        {"type": "image", "path": "dot.png", "width_cm": 2}])
    assert rel == "docs/informe.docx"
    call("word_replace", path="docs/informe.docx", replacements=[{"find": "{{nombre}}", "replace": "Ana"}])
    read, _ = call("office_read", path="docs/informe.docx")
    assert "# Resumen" in read and "Hola Ana" in read and "Ene | 10" in read

    call("excel_write", path="ventas.xlsx", sheet="Datos", header=True,
         rows=[["Mes", "Ventas"], ["Ene", 10], ["Feb", 20], ["Total", "=SUM(B2:B3)"]], number_format={"B2:B4": "#,##0"})
    call("excel_write", path="ventas.xlsx", sheet="Datos", append=True, rows=[["Mar", 5]])
    call("excel_chart", path="ventas.xlsx", sheet="Datos", type="column", data="B1:B3", categories="A2:A3", title="Ventas")
    read, _ = call("office_read", path="ventas.xlsx")
    assert "=SUM(B2:B3)" in read and "5: Mar | 5" in read
    from openpyxl import load_workbook
    assert len(load_workbook(ws.root / "ventas.xlsx")["Datos"]._charts) == 1

    call("pptx_write", path="deck.pptx", slides=[
        {"layout": "title", "title": "Q1", "subtitle": "Resultados"},
        {"layout": "content", "title": "Puntos", "bullets": ["a", {"text": "b", "level": 1}], "notes": "decir esto"},
        {"layout": "table", "title": "Tabla", "rows": [["x", "y"], [1, 2]]},
        {"layout": "image", "title": "Imagen", "image": "dot.png"}])
    read, _ = call("office_read", path="deck.pptx")
    assert "## Slide 4" in read and "Resultados" in read and "(notes) decir esto" in read and "x | y" in read

    with pytest.raises(ToolError):
        call("word_write", path="../fuera.docx", blocks=["x"])  # confined to the cage
    with pytest.raises(ToolError):
        call("excel_write", path="mal.docx", rows=[[1]])


def test_burp_rest_connector(tmp_path, monkeypatch):
    import httpx

    from nesicode import burp_rest
    from nesicode.connectors import ConnectorConfig
    from nesicode.server.jobs import needs_approval

    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        if not req.url.path.startswith("/KEY/v0.1"):
            return httpx.Response(401)
        path = req.url.path.removeprefix("/KEY/v0.1")
        if path == "/":
            return httpx.Response(200)
        if path == "/scan" and req.method == "POST":
            body = json.loads(req.content)
            assert body["urls"] == ["https://app.test/"] and body["scan_configurations"][0]["type"] == "NamedConfiguration"
            return httpx.Response(201, headers={"Location": "/KEY/v0.1/scan/7"})
        if path == "/scan/7":
            return httpx.Response(200, json={"task_id": "7", "scan_status": "auditing",
                                             "scan_metrics": {"crawl_and_audit_progress": 40},
                                             "issue_events": [
                {"type": "issue_found", "issue": {"name": "Cookie without HttpOnly", "severity": "low", "confidence": "firm",
                                                  "origin": "https://app.test", "path": "/", "serial_number": "1"}},
                {"type": "issue_found", "issue": {"name": "SQL injection", "severity": "high", "confidence": "certain",
                                                  "origin": "https://app.test", "path": "/login", "serial_number": "2",
                                                  "description": "<p>The <b>user</b> parameter</p>"}}]})
        if path == "/knowledge_base/issue_definitions":
            return httpx.Response(200, json=[{"name": "SQL injection", "issue_type_id": "1049088", "typical_severity": "high",
                                              "description": "SQL injection vulnerabilities arise..."}])
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(burp_rest.httpx, "request", lambda method, url, **kw: client.request(method, url, **{k: v for k, v in kw.items() if k != "timeout"}))
    cfg = ConnectorConfig(name="burp-rest", kind="builtin", command="burp-rest", url="http://127.0.0.1:1337",
                          env={"BURP_API_KEY": "KEY"})
    ws = Workspace(tmp_path / "ws")
    burp_rest.ping(cfg)
    text, _ = burp_rest.burp_scan({"urls": ["https://app.test/"], "scan_configurations": ["Crawl strategy - fastest"]}, ws, cfg)
    assert "task_id=7" in text
    text, rel = burp_rest.burp_scan_status({"task_id": "7", "details": True, "save_json": True}, ws, cfg)
    assert text.index("SQL injection") < text.index("Cookie without HttpOnly")  # sorted by severity
    assert "The user parameter" in text and rel == "burp/scan-7.json" and (ws.root / rel).is_file()
    text, _ = burp_rest.burp_issue_definitions({"query": "sql"}, ws, cfg)
    assert "1049088" in text
    with pytest.raises(ToolError, match="API key"):
        burp_rest.ping(ConnectorConfig(name="b", kind="builtin", url="http://127.0.0.1:1337", env={"BURP_API_KEY": "BAD"}))
    with pytest.raises(ToolError):
        burp_rest.burp_scan({"urls": ["ftp://x"]}, ws, cfg)
    # active scans always need the user's OK, even in auto mode; reading status does not
    assert needs_approval("auto", "mcp__burp-rest__burp_scan")
    assert not needs_approval("auto", "mcp__burp-rest__burp_scan_status")


def test_stop_kills_running_command_and_hung_request(tmp_path):
    import socket
    import threading
    import time as _t

    from nesicode.llm import LLMError
    from nesicode.llm.openai_compat import OpenAICompatProvider

    ws = Workspace(tmp_path / "ws")
    ws.cancel = threading.Event()
    threading.Timer(0.5, ws.cancel.set).start()
    t0 = _t.monotonic()
    code, out = ws.run("sleep 30", timeout=60)
    assert code == 130 and "stopped by the user" in out and _t.monotonic() - t0 < 5

    # a model endpoint that accepts the connection and never answers (like a hung backend)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    conns = []
    threading.Thread(target=lambda: conns.append(srv.accept()), daemon=True).start()
    prov = OpenAICompatProvider.__new__(OpenAICompatProvider)
    prov.name, prov.base_url, prov.headers = "hung", f"http://127.0.0.1:{srv.getsockname()[1]}", {}
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    t0 = _t.monotonic()
    with pytest.raises(LLMError, match="cancelled"):
        prov.post("/chat/completions", {"model": "x"}, cancel=cancel)
    assert _t.monotonic() - t0 < 5
    srv.close()


def test_bot_cron_parser_and_schedule():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from nesicode.server.bots import CronError, cron_next, interval_seconds, next_runs

    mad = ZoneInfo("Europe/Madrid")
    start = datetime(2026, 11, 6, 9, 30, tzinfo=mad).timestamp()  # a Friday
    nxt = datetime.fromtimestamp(cron_next("0 8 * * 1-5", "Europe/Madrid", start), mad)
    assert (nxt.weekday(), nxt.hour, nxt.minute) == (0, 8, 0)  # next weekday 08:00 is Monday
    nxt = datetime.fromtimestamp(cron_next("*/15 * * * *", "Europe/Madrid", start), mad)
    assert (nxt.hour, nxt.minute) == (9, 45)
    nxt = datetime.fromtimestamp(cron_next("30 7 1 * *", "Europe/Madrid", start), mad)
    assert (nxt.month, nxt.day, nxt.hour, nxt.minute) == (12, 1, 7, 30)
    nxt = datetime.fromtimestamp(cron_next("0 9 * * sun", "Europe/Madrid", start), mad)
    assert nxt.weekday() == 6
    # DST change (last Sunday of March): 08:00 local still means 08:00 local
    nxt = datetime.fromtimestamp(cron_next("0 8 * * *", "Europe/Madrid", datetime(2027, 3, 28, 1, 0, tzinfo=mad).timestamp()), mad)
    assert (nxt.day, nxt.hour) == (28, 8)
    assert interval_seconds("2h") == 7200 and len(next_runs("interval", "30", "", 3)) == 3
    for bad in ("0 8 * *", "61 * * * *", "0 25 * * *", "x * * * *"):
        with pytest.raises(CronError):
            cron_next(bad, "Europe/Madrid", start)
    with pytest.raises(CronError):
        interval_seconds("1")  # below the 5-minute minimum


def test_bot_fetch_blocks_local_network():
    import httpx

    from nesicode.server.bots import check_public_url, page_text, safe_fetch

    for url in ("http://127.0.0.1:47821/api/projects", "http://localhost/", "http://192.168.1.1/", "http://169.254.169.254/",
                "file:///etc/passwd", "http://[::1]/"):
        with pytest.raises(ToolError):
            check_public_url(url)
    hops = []

    def handler(req):
        hops.append(str(req.url))
        if req.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/end"})
        return httpx.Response(200, text="<html><script>x()</script><h1>Hola</h1><p>mundo &amp; más</p></html>",
                              headers={"content-type": "text/html"})

    status, ctype, body = safe_fetch("https://example.test/start", transport=httpx.MockTransport(handler))
    assert status == 200 and len(hops) == 2
    assert page_text(ctype, body) == "Hola \n mundo & más" or "Hola" in page_text(ctype, body) and "x()" not in page_text(ctype, body)


def test_bot_unattended_rules(monkeypatch):
    from nesicode.server import bots as botmod

    assert botmod.bot_denies("readonly", "bash") and botmod.bot_denies("readonly", "write_file")
    assert botmod.bot_denies("readonly", "mcp__office__word_write")
    assert not botmod.bot_denies("readonly", "read_file") and not botmod.bot_denies("readonly", "http_fetch")
    assert not botmod.bot_denies("ask", "bash")
    monkeypatch.setattr(botmod.os, "name", "nt")
    assert botmod.bot_denies("auto", "bash")  # Windows: no sandbox → no commands for bots


def test_same_tool_error_three_times_stops_the_loop(tmp_path):
    from nesicode.llm.base import ToolCall, Usage

    agent = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(tmp_path))
    agent.extra_tools = {"flaky": (None, lambda args, ag: (_ for _ in ()).throw(ToolError("backend says 400 for request 17")))}
    outs = [agent._call_tool(ToolCall(f"c{i}", "flaky", {"n": i}), Usage()) for i in range(4)]
    assert all(o.is_error for o in outs) and "disabled for this run" in outs[3].content
    # different errors don't count as a loop
    agent2 = Agent(model="mock/demo", harness=base_harness(), workspace=Workspace(tmp_path / "b"))
    outs = [agent2._call_tool(ToolCall(f"r{i}", "read_file", {"path": f"nope{i}-{'x' * i}.txt"}), Usage()) for i in range(4)]
    assert "disabled" not in outs[3].content


def test_virustotal_builtin(tmp_path):
    import httpx

    from nesicode import virustotal as vt
    from nesicode.connectors import ConnectorConfig

    seen = []

    def handler(req):
        seen.append((req.method, req.url.path, req.headers.get("x-apikey")))
        if req.headers.get("x-apikey") != "K":
            return httpx.Response(401)
        if req.url.path.endswith("/files/" + "a" * 64):
            return httpx.Response(200, json={"data": {"attributes": {
                "meaningful_name": "evil.exe", "type_description": "Win32 EXE", "size": 1234, "sha256": "a" * 64,
                "last_analysis_stats": {"malicious": 40, "suspicious": 1, "undetected": 20},
                "last_analysis_results": {"Kaspersky": {"category": "malicious", "result": "Trojan.Win32.X"}},
                "popular_threat_classification": {"suggested_threat_label": "trojan.x"}}}})
        if "/domains/" in req.url.path:
            return httpx.Response(200, json={"data": {"attributes": {"last_analysis_stats": {"harmless": 70}, "registrar": "R",
                                                                       "categories": {"a": "search engines"}}}})
        return httpx.Response(404)

    vt.transport = httpx.MockTransport(handler)
    try:
        cfg = ConnectorConfig(name="virustotal", kind="builtin", command="virustotal", env={"VT_API_KEY": "K"})
        ws = Workspace(tmp_path / "ws")
        text, _ = vt.vt_file_report({"hash": "A" * 64}, ws, cfg)
        assert "MALICIOUS" in text and "Trojan.Win32.X" in text and "trojan.x" in text
        text, _ = vt.vt_domain_report({"domain": "https://google.com/x"}, ws, cfg)
        assert "clean" in text and seen[-1][1].endswith("/domains/google.com")
        assert "never seen" in vt.vt_file_report({"hash": "b" * 64}, ws, cfg)[0]
        with pytest.raises(ToolError):
            vt.vt_file_report({"hash": "not-a-hash"}, ws, cfg)
        with pytest.raises(ToolError, match="API key"):
            vt.ping(ConnectorConfig(name="v", kind="builtin", env={"VT_API_KEY": "bad"}))
    finally:
        vt.transport = None
    from nesicode.server.jobs import needs_approval
    assert needs_approval("auto", "mcp__virustotal__vt_scan_file") and needs_approval("auto", "mcp__virustotal__vt_scan_url")
    assert not needs_approval("auto", "mcp__virustotal__vt_file_report")


def test_connector_only_when_asked():
    from nesicode.connectors import mentioned

    burp = dict(name="burp", label="Burp Suite", template="burp")
    assert not mentioned("audita la web example.com y dime qué falla", **burp)
    assert mentioned("usa Burp para auditar example.com", **burp)
    assert mentioned("hazme un Excel con las ventas", name="office", label="Office (Word, Excel, PowerPoint)", template="office")
    assert mentioned("mira este hash en virus total", name="virustotal", label="VirusTotal", template="virustotal")
    assert not mentioned("busca información en la web", template="browser")
    assert mentioned("ábrelo en mi navegador", template="browser")
