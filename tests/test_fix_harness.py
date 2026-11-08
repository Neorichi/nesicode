"""Regression tests for the harness / profiles / harnessing engine / CLI audit fixes (group H).

Offline: the mock provider or a scripted fake improver, temp dirs, no network.
"""

import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from nesicode.agent import copy_workspace
from nesicode.harness import base_harness, hooks, inventory, validate
from nesicode.harness import profiles as prof
from nesicode.harnessing import HarnessingConfig, HarnessingRun, apply_rollout, engine, parse_edits, prompts
from nesicode.llm import DEFAULT_LIMITS, LLMError, ToolCall, Turn, Usage, known_limits
from nesicode.llm.base import Conversation, Provider

SRC = Path(__file__).resolve().parents[1] / "src"
LABEL_PATH = re.compile(r"rollouts[\\/]+(base|inv|dom|agg|c)\d")


# ---------------------------------------------------------------- helpers
class FakeImprover(Provider):
    """Scripted improver: `script(role, prompt, options)` returns a Turn or text; every call is recorded."""

    name = "fake"

    def __init__(self, script):
        self.script, self.calls = script, []

    def conversation(self, model, system, tools, options):
        prov = self

        class C(Conversation):
            def add_user(self, text):
                self.prompt = text

            def add_assistant_text(self, text):
                pass

            def add_tool_results(self, results):
                pass

            def step(self, on_event=None):
                prov.calls.append((options.role, self.prompt, options))
                out = prov.script(options.role, self.prompt, options)
                return out if isinstance(out, Turn) else Turn(text=out, usage=Usage(1, 1))

        return C(model, system, tools, options)


def use_improver(monkeypatch, script, model="fake-model"):
    fake = FakeImprover(script)
    monkeypatch.setattr(engine, "get_provider", lambda spec, keys=None: (fake, model))
    return fake


def make_run(tmp_path, cancel=None, **cfg):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "app.py").write_text("print(1)\n")
    c = {"task": "Write a report", "executor_model": "mock/demo", "improver_model": "mock/demo", **cfg}
    return HarnessingRun(HarnessingConfig(**c), base_harness=base_harness(), source_ws=src, run_dir=tmp_path / "run",
                         cancel=cancel)


def tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def strip_markers(text: str) -> str:
    return re.sub(r"<!-- /?profile[^>]*-->\n", "", text)


# ------------------------------------------- #2 apply: never clobber the user's later edits
def test_apply_does_not_clobber_files_changed_after_the_snapshot(tmp_path):
    project = tmp_path / "project"
    tree(project, {"app.py": "v1\n", "old_notes.md": "notes v1\n", "lib.py": "lib v1\n", "gone.md": "x\n"})
    snap = tmp_path / "run/source"
    copy_workspace(project, snap)
    ws = tmp_path / "run/rollouts/agg1/ws"
    copy_workspace(snap, ws)
    (ws / "app.py").write_text("v2 rollout\n")
    (ws / "lib.py").write_text("lib v2\n")
    (ws / "old_notes.md").unlink()
    (ws / "gone.md").unlink()
    tree(ws, {"new.md": "new\n", "clash.md": "rollout clash\n", "same.md": "same\n"})
    # meanwhile the user keeps working in the project folder
    (project / "app.py").write_text("v1\nUSER FIX\n")
    (project / "old_notes.md").write_text("notes v1\nIMPORTANT\n")
    tree(project, {"clash.md": "user clash\n", "same.md": "same\n"})

    res = apply_rollout(snap, ws, project)
    assert (project / "app.py").read_text() == "v1\nUSER FIX\n"  # the user's fix survives
    assert (project / "app.py.nesicode-new").read_text() == "v2 rollout\n"  # the rollout's version is kept aside
    assert (project / "old_notes.md").read_text() == "notes v1\nIMPORTANT\n"  # not deleted
    assert (project / "clash.md").read_text() == "user clash\n"
    assert (project / "lib.py").read_text() == "lib v2\n"  # files the user did not touch get the change
    assert not (project / "gone.md").exists() and (project / "new.md").read_text() == "new\n"
    assert sorted(c["path"] for c in res["conflicts"]) == ["app.py", "clash.md", "old_notes.md"]
    assert "lib.py" in res["modified"] and "gone.md" in res["removed"] and "new.md" in res["added"]
    assert "app.py" not in res["modified"] and "old_notes.md" not in res["removed"]


def test_apply_case_only_rename_keeps_the_file(tmp_path):
    project = tmp_path / "project"
    tree(project, {"src/button.tsx": "export const B = 1\n"})
    snap = tmp_path / "source"
    copy_workspace(project, snap)
    ws = tmp_path / "ws"
    copy_workspace(snap, ws)
    (ws / "src/button.tsx").rename(ws / "src/tmp")
    (ws / "src/tmp").rename(ws / "src/Button.tsx")  # `git mv button.tsx Button.tsx`
    apply_rollout(snap, ws, project)
    assert os.listdir(project / "src") == ["Button.tsx"]
    assert (project / "src/Button.tsx").read_text() == "export const B = 1\n"


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_apply_never_writes_through_a_symlink_out_of_the_project(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "project"
    tree(project, {"a.txt": "a\n"})
    snap = tmp_path / "source"
    copy_workspace(project, snap)
    ws = tmp_path / "ws"
    copy_workspace(snap, ws)
    tree(ws, {"link/evil.txt": "pwned\n"})  # the rollout made a real directory `link`...
    os.symlink(outside, project / "link")  # ...while in the project `link` points outside it
    res = apply_rollout(snap, ws, project)
    assert not (outside / "evil.txt").exists() and not list(outside.iterdir())
    assert [c["path"] for c in res["conflicts"]] == ["link/evil.txt"]


def test_engine_apply_selected_reports_conflicts(tmp_path):
    hr = make_run(tmp_path)
    r = hr.run()
    (tmp_path / "src/NESICODE_RESULT.md").write_text("the user's own file\n")  # created after the snapshot
    res = hr.apply_selected(r["selected"], tmp_path / "src")
    assert (tmp_path / "src/NESICODE_RESULT.md").read_text() == "the user's own file\n"
    assert [c["path"] for c in res["conflicts"]] == ["NESICODE_RESULT.md"]


# ------------------------------------------------------------- #4 Stop is honoured to the end
@pytest.mark.parametrize("mode,k,final", [("parallel", 1, "agg1"), ("sequential", 2, "c2")])
def test_stop_during_the_final_rollout_cancels_the_run(tmp_path, monkeypatch, mode, k, final):
    from nesicode.agent import loop
    from nesicode.llm import mock

    cancel = threading.Event()
    after_stop = []
    orig_step, orig_run = mock.MockConversation.step, loop.Agent.run

    def step(self, on_event=None):
        if cancel.is_set() and self.options.role not in ("executor", "subagent"):
            after_stop.append(self.options.role)
        return orig_step(self, on_event)

    def run(self, task, history=None):
        if f"/rollouts/{final}/" in Path(self.ws.root).as_posix() + "/":
            cancel.set()  # the user presses Stop while the final rollout runs
        return orig_run(self, task, history)

    monkeypatch.setattr(mock.MockConversation, "step", step)
    monkeypatch.setattr(loop.Agent, "run", run)
    hr = make_run(tmp_path, cancel=cancel, k=k, mode=mode, check_command="test -f NESICODE_RESULT.md")
    with pytest.raises(RuntimeError, match="cancelled"):
        hr.run()
    assert after_stop == []  # no selector / reflector call once Stop was pressed
    rd = tmp_path / "run"
    assert not (rd / "source").exists() and not list(rd.glob("rollouts/*/ws"))  # #11: no copies left behind
    assert (rd / "rollouts/base1/trajectory.txt").exists()  # the record of what ran stays


def test_improver_calls_carry_stop_and_errors_after_stop_mean_cancelled(tmp_path, monkeypatch):
    cancel = threading.Event()

    def script(role, prompt, options):
        assert options.cancel is cancel
        cancel.set()  # Stop pressed while the improver request is in flight
        raise LLMError("connection error: aborted")

    use_improver(monkeypatch, script)
    hr = make_run(tmp_path, cancel=cancel)
    with pytest.raises(RuntimeError, match="cancelled"):
        hr._stage("inv1", "INV", "job", base_harness(), "ctx")
    assert not hr.stages  # not recorded as a fallback stage
    with pytest.raises(RuntimeError, match="cancelled"):
        hr._select(["base1"])
    with pytest.raises(RuntimeError, match="cancelled"):
        hr._reflect("base1")


def test_reply_received_after_stop_is_discarded(tmp_path, monkeypatch):
    cancel = threading.Event()

    def script(role, prompt, options):
        cancel.set()  # a provider that does not abort: the reply still arrives
        return '<file path="memory.md">\n- fact\n</file>\n<notes>n</notes>'

    use_improver(monkeypatch, script)
    with pytest.raises(RuntimeError, match="cancelled"):
        make_run(tmp_path, cancel=cancel)._stage("inv1", "INV", "job", base_harness(), "ctx")


# ------------------------------------------------- #5 hooks.json types never crash validation
@pytest.mark.parametrize("phase,rule,ok", [
    ("post_tool", {"tool": "*", "match": None, "action": "truncate", "max_chars": 8000}, True),  # null = no match
    ("pre_tool", {"tool": ["bash", "write_file"], "match": "x", "action": "warn", "message": "m"}, True),
    ("pre_tool", {"tool": "bash", "match": 5, "action": "warn", "message": "m"}, False),
    ("pre_tool", {"tool": "bash", "match": "x", "action": ["block"], "message": "m"}, False),
    ("pre_tool", {"tool": 3, "action": "warn", "message": "m"}, False),
    ("pre_tool", {"tool": "bash", "action": "block", "message": ["no"]}, False),
    ("post_tool", {"tool": "*", "action": "truncate", "max_chars": True}, False),
    ("post_tool", {"tool": "*", "action": "truncate", "max_chars": 0}, False),
    ("pre_tool", {"tool": "bash", "match": "(", "action": "warn", "message": "m"}, False),
])
def test_hook_rules_are_type_checked(phase, rule, ok):
    h = base_harness()
    h["plugins/hooks.json"] = json.dumps({phase: [rule]})
    errs = validate(h)  # never raises
    assert (errs == []) == ok, errs
    assert "chars always-on" in inventory(h)  # never raises either
    h["plugins/hooks.json"] = "[1, 2]"
    assert validate(h) and hooks(h) == {"pre_tool": [], "post_tool": []} and inventory(h)


def test_unexpected_validation_error_rejects_the_attempt(tmp_path, monkeypatch):
    use_improver(monkeypatch, lambda role, prompt, o: '<file path="memory.md">\n- x\n</file>\n<notes>n</notes>')

    def boom(h):
        raise TypeError("first argument must be string or compiled pattern")

    monkeypatch.setattr(engine, "validate", boom)
    st = make_run(tmp_path)._stage("dom1", "DOM", "job", base_harness(), "ctx")
    assert st.fell_back and st.harness == base_harness() and "could not be validated" in st.error


# --------------------------------------------- #6 improver requests fit the improver's window
@pytest.mark.parametrize("model,ctx,out", [
    ("unknown-model", 128_000, None),  # DEFAULT_LIMITS: compat, ollama, anything unknown
    ("nesi3", 131_072, 32_768),
    ("deepseek-chat", 128_000, None),
    ("unknown-model", None, None),  # the CLI passes no limits
    ("small-local", 32_000, 8_000),
])
def test_improver_request_fits_the_window(tmp_path, monkeypatch, model, ctx, out):
    fake = use_improver(monkeypatch, lambda role, prompt, o: "KEEP_BASE <notes>x</notes>", model=model)
    hr = make_run(tmp_path, improver_context=ctx, improver_output=out)
    hr._improver("composer", "E" * 600_000)  # evidence bigger than any budget (k>=2 compose / reflect)
    _, prompt, opts = fake.calls[0]
    window = ctx or (known_limits(model) or DEFAULT_LIMITS)[0]
    cap = out or (known_limits(model) or DEFAULT_LIMITS)[1]
    assert opts.max_tokens <= cap
    assert len(prompt) / 2.5 + opts.max_tokens + 1_000 <= window  # fits even at a pessimistic 2.5 chars/token
    assert "trimmed" in prompt and opts.cancel is hr.cancel


# ------------------------------------------------------------------- #7 leaving Blank
def test_leaving_blank_restores_the_base_and_keeps_what_was_learned():
    from nesicode.agent.hooks import Hooks

    b = prof.profile_harness("blank")
    b["guardrails.md"] = "- Run `make test` before finishing.\n"
    b["memory.md"] = "- Tests run with `uv run pytest -q`.\n"
    b["skills/deploy/SKILL.md"] = "---\nname: deploy\ndescription: Deploy the app.\n---\n1. go\n"
    b["system_prompt.md"] += "When deploying, load the `deploy` skill.\n"
    b["plugins/hooks.json"] = json.dumps({"pre_tool": [], "post_tool": [{"tool": "bash", "action": "truncate", "max_chars": 9000}]})
    assert validate(b) == [] and prof.detect_profile(b) == "blank"
    for target in ("general", "security"):
        for current in (None, "blank"):
            out = prof.apply_profile(b, target, current=current)
            sp = out["system_prompt.md"]
            assert validate(out) == [], (target, current)
            assert sum(line.startswith("You are") for line in sp.splitlines()) == 1
            assert "Work the task end to end" in sp and "load the `deploy` skill" in sp
            assert "Never run destructive commands" in out["guardrails.md"] and "make test" in out["guardrails.md"]
            assert "uv run pytest" in out["memory.md"]
            assert "skills/documents/SKILL.md" in out and "skills/deploy/SKILL.md" in out
            rules = hooks(out)
            assert Hooks(rules).before("bash", {"command": "rm -rf / "})[0]  # the base rm -rf guard is back
            assert {"tool": "bash", "action": "truncate", "max_chars": 9000} in rules["post_tool"]
    # a Blank project that learned only memory (guardrails still empty) keeps it too
    b2 = prof.profile_harness("blank")
    b2["memory.md"] = "- a learned fact\n"
    out = prof.apply_profile(b2, "general")
    assert "- a learned fact" in out["memory.md"] and "Work the task end to end" in out["system_prompt.md"]
    # the stored profile wins over the content heuristic when it is passed
    g = prof.profile_harness("general")
    assert prof.apply_profile(g, "general", current="general") == prof.apply_profile(g, "general")


# ------------------------------------------- #8 learned trigger written inside a profile block
def test_switch_keeps_a_learned_trigger_written_inside_the_profile_block():
    h = prof.profile_harness("security")
    anchor = "3. **Attack**: for web apps load the `web-audit` skill."
    assert anchor in h["system_prompt.md"]
    h["system_prompt.md"] = h["system_prompt.md"].replace(anchor, anchor + "\n   For login forms, load the `auth-bruteforce` skill.")
    h["skills/auth-bruteforce/SKILL.md"] = "---\nname: auth-bruteforce\ndescription: Test login rate limits.\n---\n1. ...\n"
    assert validate(h) == []
    for target in ("webdev", "general", "security"):
        out = prof.apply_profile(h, target, current="security")
        sp = out["system_prompt.md"]
        assert validate(out) == [], target
        assert "load the `auth-bruteforce` skill" in sp and "skills/auth-bruteforce/SKILL.md" in out
        assert sp.count("## Profile: offensive security") == (target == "security")
        for block in re.findall(r"<!-- profile:[\w-]+ -->.*?<!-- /profile -->", sp, re.S):
            assert "auth-bruteforce" not in block  # the learned line now lives outside the markers


# --------------------------------------------- #15 markers lost: never mix two profiles
def test_switch_without_markers_does_not_mix_profiles():
    h = prof.profile_harness("security")
    for f in prof.OVERLAY_FILES:
        h[f] = strip_markers(h[f])
    h["guardrails.md"] += "- Regla aprendida.\n"
    assert prof.detect_profile(h) == "security"
    out = prof.apply_profile(h, "webdev")
    text = "\n".join(out[f] for f in prof.OVERLAY_FILES)
    assert validate(out) == []
    assert "offensive security" not in text and "Security profile:" not in text and "Security tools:" not in text
    assert not any(f"`{n}`" in text for n in ("security-recon", "web-audit", "security-report"))
    assert "Profile: web development" in out["system_prompt.md"] and "- Regla aprendida." in out["guardrails.md"]
    assert out["system_prompt.md"].startswith(prof.BASE_INTRO)


def test_switch_with_a_half_deleted_marker_pair():
    h = prof.profile_harness("security")
    h["system_prompt.md"] = h["system_prompt.md"].replace("<!-- /profile -->\n", "")  # a hand edit lost the end marker
    out = prof.apply_profile(h, "webdev", current="security")
    sp = out["system_prompt.md"]
    assert validate(out) == [] and prof.detect_profile(out) == "webdev"
    assert "offensive security" not in sp and sp.count("<!-- profile:") == 1 and sp.count("<!-- /profile -->") == 1
    assert prof.apply_profile(out, "webdev", current="webdev") == out


def test_proposals_must_keep_the_profile_markers(tmp_path, monkeypatch):
    h = prof.profile_harness("security")
    lost = strip_markers(h["system_prompt.md"]) + "Always write tests.\n"
    kept = h["system_prompt.md"] + "Always write tests.\n"
    replies = iter([f'<file path="system_prompt.md">\n{lost}</file>\n<notes>x</notes>',
                    f'<file path="system_prompt.md">\n{kept}</file>\n<notes>y</notes>'])
    fake = use_improver(monkeypatch, lambda role, prompt, o: next(replies))
    st = make_run(tmp_path)._stage("inv1", "INV", "job", h, "ctx")
    assert not st.fell_back and st.harness["system_prompt.md"] == kept
    assert "<!-- profile:security -->" in fake.calls[1][1].split("YOUR PREVIOUS ATTEMPT WAS REJECTED")[1]
    assert "<!-- profile:" in prompts.SURFACE  # improvers are told what the markers mean


# ------------------------------------------------------------ #9 blind selection stays blind
def test_selector_never_sees_rollout_labels_or_paths(tmp_path, monkeypatch):
    from nesicode.llm import mock

    seen = {}
    orig = mock.MockConversation.step

    def step(self, on_event=None):
        if self.options.role == "selector":
            seen["prompt"] = self.messages[-1][1]
        turn = orig(self, on_event)
        if self.options.role == "executor" and not turn.tool_calls:  # the final answer cites the absolute path
            root = re.search(r"absolute path is `([^`]+)`", self.system).group(1)
            turn.text += f" ({root}/NESICODE_RESULT.md)"
        return turn

    monkeypatch.setattr(mock.MockConversation, "step", step)
    hr = make_run(tmp_path, check_command="pwd; exit 1")  # a failing check prints the absolute path
    hr.run()
    p = seen["prompt"]
    assert "<workspace>" in p
    assert not LABEL_PATH.search(p)
    assert str(tmp_path) not in p and os.path.realpath(tmp_path) not in p


# ------------------------------------------------ #10 CLI on a non-UTF-8 console / locale
def _cli_env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY") and k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    env.update(PYTHONPATH=str(SRC), NESICODE_NO_DOTENV="1", NESICODE_DATA_DIR=str(tmp_path / "data"),
               NESICODE_SANDBOX="local", NESICODE_NO_TELEGRAM="1", NESICODE_NO_SCHEDULER="1")
    env.update(extra)
    return env


def test_cli_run_survives_a_non_utf8_console(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    r = subprocess.run(
        [sys.executable, "-m", "nesicode.cli", "run", "Escribe un informe →", "--workspace", str(ws), "--model", "mock/demo"],
        env=_cli_env(tmp_path, PYTHONIOENCODING="cp1252"), capture_output=True, timeout=90,
    )
    out = r.stdout.decode("cp1252", "replace") + r.stderr.decode("cp1252", "replace")
    assert r.returncode == 0 and "[done]" in out, out


def test_cli_harness_files_are_utf8_whatever_the_locale(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("# demo\n")
    h = base_harness()
    h["memory.md"] = "- Guía: usa → siempre ✓\n"
    (tmp_path / "h.json").write_text(json.dumps(h, ensure_ascii=False), encoding="utf-8")
    # an ASCII locale (no UTF-8 mode): the default text encoding cannot hold "í", "→" or "✓", like cp1252 on Windows
    env = _cli_env(tmp_path, LC_ALL="C", LANG="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    r = subprocess.run(
        [sys.executable, "-m", "nesicode.cli", "harness", "Write a report", "--workspace", str(ws), "--model", "mock/demo",
         "--harness", str(tmp_path / "h.json"), "--out", str(tmp_path / "out")],
        env=env, capture_output=True, timeout=120,
    )
    assert r.returncode == 0, (r.stdout + r.stderr).decode("latin-1")
    adopted = json.loads((tmp_path / "out/adopted_harness.json").read_text(encoding="utf-8"))
    assert "- Guía: usa → siempre ✓" in adopted["memory.md"]
    assert json.loads((tmp_path / "out/result.json").read_text(encoding="utf-8"))["selected"]


# --------------------------------------------------- #11 rollout copies are cleaned up
def test_successful_run_keeps_only_the_snapshot_and_the_chosen_copy(tmp_path):
    hr = make_run(tmp_path)
    r = hr.run()
    left = sorted(p.parent.name for p in (tmp_path / "run").glob("rollouts/*/ws"))
    assert left == [r["selected"]] and (tmp_path / "run/source").is_dir()
    assert len(list((tmp_path / "run").glob("rollouts/*/trajectory.txt"))) == 4  # records stay


def test_crashed_run_leaves_no_copies(tmp_path, monkeypatch):
    def boom(self, *a, **kw):
        raise ValueError("selector exploded")

    monkeypatch.setattr(HarnessingRun, "_select", boom)
    with pytest.raises(ValueError):
        make_run(tmp_path).run()
    assert not (tmp_path / "run/source").exists() and not list((tmp_path / "run").glob("rollouts/*/ws"))


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="POSIX permissions")
def test_remove_tree_handles_read_only_entries(tmp_path):
    d = tmp_path / "ws/.git/objects/ab"
    d.mkdir(parents=True)
    (d / "obj").write_text("x")
    os.chmod(d / "obj", 0o444)
    os.chmod(d, 0o555)  # like git's read-only object store (Windows: read-only files)
    engine.remove_tree(tmp_path / "ws")
    assert not (tmp_path / "ws").exists()


def test_cli_keeps_only_the_chosen_result(tmp_path):
    from nesicode import cli

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("x\n")
    cli.main(["harness", "t", "--workspace", str(ws), "--model", "mock/demo", "--out", str(tmp_path / "out")])
    result = json.loads((tmp_path / "out/result.json").read_text(encoding="utf-8"))
    assert sorted(p.parent.name for p in (tmp_path / "out").glob("rollouts/*/ws")) == [result["selected"]]
    assert not (tmp_path / "out/source").exists()


# -------------------------------------------------------- #13 tags inside notes / reasoning
def test_parse_edits_ignores_tags_inside_notes_and_reasoning():
    h = base_harness()
    reply = ('<think>Draft:\n<file path="memory.md">\n- DRAFT idea\n</file>\nNo.</think>\n'
             '<file path="guardrails.md">\n' + h["guardrails.md"] + '- Re-read files after writing.\n</file>\n'
             '<file path="skills/fmt/SKILL.md">\n---\nname: fmt\ndescription: d\n---\nEnd with a <notes> block.\n</file>\n'
             '<notes>\ndropped -> <delete path="skills/documents/SKILL.md"/> (dom1) -> UNSAFE\n</notes>')
    new, touched = parse_edits(reply, h)
    assert touched and "skills/documents/SKILL.md" in new  # the dropped delete is not applied
    assert "DRAFT" not in new.get("memory.md", "")  # nor the draft inside <think>
    assert "- Re-read files after writing." in new["guardrails.md"]
    assert "End with a <notes> block." in new["skills/fmt/SKILL.md"]  # a file body is content, tags and all


def test_reasoning_whose_opening_tag_was_in_the_prompt_is_not_parsed():
    h = base_harness()
    # templates that open <think> in the prompt: the reply carries only the closing tag
    reply = ('I could rewrite <file path="memory.md">\n- DRAFT\n</file> but no.</think>\n'
             '<file path="guardrails.md">\n' + h["guardrails.md"] + '- New rule.\n</file>\n<notes>n</notes>')
    new, _ = parse_edits(reply, h)
    assert "DRAFT" not in new["memory.md"] and "- New rule." in new["guardrails.md"]
    # ...but a closing tag inside a file body is content
    body = "---\nname: tags\ndescription: d\n---\nCut the reply after </think> first.\n"
    new, _ = parse_edits(f'<file path="skills/tags/SKILL.md">\n{body}</file>\n<notes>n</notes>', h)
    assert new["skills/tags/SKILL.md"] == body


def test_selector_choice_inside_reasoning_is_ignored(tmp_path, monkeypatch):
    use_improver(monkeypatch, lambda role, prompt, o: "<think>first guess <choice>A</choice></think>\n"
                                                    "<reasoning>B is better</reasoning>\n<choice>B</choice>")
    hr = make_run(tmp_path)
    hr.rollouts = {l: engine.Rollout(l, status="done", delivered=True) for l in ("base1", "agg1")}
    order = ["base1", "agg1"]
    random.Random(hr.seed).shuffle(order)
    chosen, reason = hr._select(["base1", "agg1"])
    assert chosen == order[1] and reason == "B is better"


# ---------------------------------------------------------- #14 cut improver replies
def test_cut_improver_reply_is_rejected_and_retried(tmp_path, monkeypatch):
    h = base_harness()
    cut = ('<file path="system_prompt.md">\n' + h["system_prompt.md"] + "Before changing an endpoint, load the "
           "`api-consistency` skill.\n</file>\n<file path=\"skills/api-consistency/SKILL.md\">\n---\nname: api-consistency\n"
           "description: Keep API schema, client and tests")
    good = '<file path="memory.md">\n- fact\n</file>\n<notes>n</notes>'
    replies = iter([Turn(text=cut, stop_reason="max_tokens"), Turn(text=good)])
    fake = use_improver(monkeypatch, lambda role, prompt, o: next(replies))
    st = make_run(tmp_path)._stage("dom1", "DOM", "job", h, "ctx")
    assert not st.fell_back and st.harness["memory.md"] == "- fact\n"
    assert "api-consistency" not in st.harness["system_prompt.md"]
    assert "cut off" in fake.calls[1][1].split("YOUR PREVIOUS ATTEMPT WAS REJECTED")[1]
    # an unterminated <file> is rejected even when the provider did not flag the cut; twice -> the base is kept
    use_improver(monkeypatch, lambda role, prompt, o: Turn(text=cut))
    st = make_run(tmp_path)._stage("dom1", "DOM", "job", h, "ctx")
    assert st.fell_back and st.harness == h and "cut off" in st.error
    # max_tokens after a complete reply (closed notes) is fine
    use_improver(monkeypatch, lambda role, prompt, o: Turn(text=good, stop_reason="max_tokens"))
    assert not make_run(tmp_path)._stage("dom1", "DOM", "job", h, "ctx").fell_back


def test_proposal_deleting_a_skill_its_trigger_still_names_is_rejected(tmp_path, monkeypatch):
    h = base_harness()
    replies = iter(['<delete path="skills/documents/SKILL.md"/>\n<notes>x</notes>', "KEEP_BASE\n<notes>ok</notes>"])
    fake = use_improver(monkeypatch, lambda role, prompt, o: next(replies))
    st = make_run(tmp_path)._stage("inv1", "INV", "job", h, "ctx")
    assert "skills/documents/SKILL.md" in st.harness and st.kept_base and not st.fell_back
    assert "`documents`" in fake.calls[1][1].split("YOUR PREVIOUS ATTEMPT WAS REJECTED")[1]


# ---------------------------------------------------------------- #16 --out inside workspace
def test_out_inside_the_workspace_is_refused(tmp_path):
    from nesicode import cli

    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "README.md").write_text("x\n")
    with pytest.raises(SystemExit) as e:
        cli.main(["harness", "t", "--workspace", str(ws), "--model", "mock/demo", "--out", str(ws / "runs" / "1")])
    assert e.value.code == 2 and not (ws / "runs").exists()
    cfg = HarnessingConfig(task="t", executor_model="mock/demo", improver_model="mock/demo")
    with pytest.raises(ValueError):
        HarnessingRun(cfg, base_harness=base_harness(), source_ws=ws, run_dir=ws / "runs" / "1").run()
    assert not (ws / "runs").exists()
    # a run dir whose snapshot folder IS the project would delete it: refused before anything is touched
    other = tmp_path / "x" / "source"
    other.mkdir(parents=True)
    (other / "keep.txt").write_text("k")
    with pytest.raises(ValueError):
        HarnessingRun(cfg, base_harness=base_harness(), source_ws=other, run_dir=tmp_path / "x").run()
    assert (other / "keep.txt").read_text() == "k"


# ------------------------------------------------------------------------ #17 Ctrl+C
_timers: list[threading.Timer] = []


def _ctrl_c_in(delay):
    t = threading.Timer(delay, lambda: os.kill(os.getpid(), signal.SIGINT))
    _timers.append(t)
    t.start()


@pytest.fixture
def sigint_guard():
    """Outside the CLI's own handler a stray SIGINT is only recorded: a CLI that installs none fails the test
    instead of aborting the whole pytest session with KeyboardInterrupt."""
    stray = []
    previous = signal.signal(signal.SIGINT, lambda *a: stray.append(1))
    try:
        yield stray
    finally:
        while _timers:  # nothing may fire once the default handler is back
            t = _timers.pop()
            t.cancel()
            t.join()
        signal.signal(signal.SIGINT, previous)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_ctrl_c_stops_nesicode_run_and_its_command(tmp_path, monkeypatch, sigint_guard):
    from nesicode import cli
    from nesicode.llm import mock

    def executor(self, on_event):
        if self.steps == 0:
            _ctrl_c_in(0.2)  # fired once the CLI is running (its handler is installed)
            return self._reply("Running it.", [ToolCall("b0", "bash", {"command": "sleep 2 && touch still_ran.txt"})],
                               on_event)
        return self._reply("done", on_event=on_event)

    monkeypatch.setattr(mock.MockConversation, "_executor", executor)
    ws = tmp_path / "ws"
    ws.mkdir()
    before = signal.getsignal(signal.SIGINT)
    t0 = time.monotonic()
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "t", "--workspace", str(ws), "--model", "mock/demo"])
    assert e.value.code == 130 and time.monotonic() - t0 < 1.8 and not sigint_guard  # not after the command's 2 s
    assert signal.getsignal(signal.SIGINT) is before  # the previous handler is restored
    time.sleep(max(0.0, t0 + 2.5 - time.monotonic()))  # past the command's own end
    assert not (ws / "still_ran.txt").exists()  # the command was killed, not orphaned


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_ctrl_c_stops_nesicode_harness_promptly(tmp_path, monkeypatch, sigint_guard):
    from nesicode import cli
    from nesicode.agent import loop

    orig, fired = loop.Agent.run, []

    def run(self, task, history=None):
        if not fired:
            fired.append(1)
            _ctrl_c_in(0.5)
        return orig(self, task, history)

    monkeypatch.setattr(loop.Agent, "run", run)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("x\n")
    out = tmp_path / "out"
    t0 = time.monotonic()
    with pytest.raises(SystemExit) as e:
        cli.main(["harness", "t", "--workspace", str(ws), "--model", "mock/demo", "--check", "sleep 3", "--out", str(out)])
    assert e.value.code == 130 and time.monotonic() - t0 < 2.5 and not sigint_guard  # the 3 s check was killed
    assert not list(out.glob("rollouts/*/ws")) and not (out / "source").exists()


# -------------------------------------------------------- #18 sequential proposer framing
def test_sequential_proposer_is_not_told_it_is_an_independent_candidate():
    seq = prompts.propose("INV", "general", k=3, index=2, round_t=2)
    assert "candidate 2 of 3" not in seq and "This is round 2" in seq
    assert "candidate 2 of 3" in prompts.propose("INV", "general", k=3, index=2, round_t=None)


# ------------------------------------- live security-audit#7: the user's format wins over defaults
def test_security_profile_report_follows_the_users_format():
    h = prof.profile_harness("security")
    sp, skill = h["system_prompt.md"], h["skills/security-report/SKILL.md"]
    assert "write `report.md`" not in sp and "Write `report.md`" not in skill
    for text in (sp, skill):
        assert "format the user asked for" in text and "report.html" in text and "share_file" in text
    assert validate(h) == []
    assert "format the user asks for" in prof.profile_harness("data")["system_prompt.md"]


def test_old_profile_wording_is_upgraded_not_kept_as_learned_text():
    new = prof.profile_harness("security")
    old = dict(new)
    legacy_block = prof.LEGACY_BLOCKS["security"]["system_prompt.md"][0]
    old["system_prompt.md"] = new["system_prompt.md"].replace(prof.SEC_SYSTEM.strip(), legacy_block.strip())
    old["skills/security-report/SKILL.md"] = prof.LEGACY_SKILLS["security-report"][0]
    assert "write `report.md`" in old["system_prompt.md"]
    again = prof.apply_profile(old, "security", current="security")
    assert again == new  # re-applying the profile gives today's text, nothing left over as "learned"
    assert prof.refresh_profile(old) == new and prof.refresh_profile(new) is None
    learned = dict(old, **{"memory.md": "- fact\n"})
    assert prof.refresh_profile(learned) == dict(new, **{"memory.md": "- fact\n"})


# ------------------------------------------------------------------- review follow-ups
def test_quoted_reasoning_tags_never_discard_a_proposal():
    h = base_harness()
    rule = h["guardrails.md"] + "- New rule.\n"
    answer = f'<file path="guardrails.md">\n{rule}</file>\n'
    replies = {
        "notes quote a leaked closing tag": answer + "<notes>\nkept -> rule -> base1 leaked </think> in its answer\n</notes>",
        "prose names an opener that never closes": "My <analysis> of the runs is in the notes.\n" + answer + "<notes>n</notes>",
        "template reasoning, then notes quoting the tag": '<file path="memory.md">\n- DRAFT\n</file> no.</think>\n'
                                                          + answer + "<notes>base1 printed </think></notes>",
    }
    for why, reply in replies.items():
        new, touched = parse_edits(reply, h)
        assert touched and "- New rule." in new["guardrails.md"], why
        assert "DRAFT" not in new.get("memory.md", ""), why
    # reasoning that opens the reply and never closes (cut while thinking) holds no answer
    r = engine.scan_reply("<think>\nI will write:\n" + answer + "<notes>draft</notes>\nhmm")
    assert r.files == [] and not r.complete


def test_reflection_keeps_edits_that_quote_a_closing_think_tag(tmp_path, monkeypatch):
    reply = ("<edits>\ncomponent: guardrails.md\nobserved: base1 printed </think> in its final answer\n</edits>\n"
             "<experience>\ncue: reasoning models\nlesson: strip think tags\nevidence: base1\nconfidence: low\n</experience>")
    use_improver(monkeypatch, lambda role, prompt, o: reply)
    out = make_run(tmp_path)._reflect("base1")
    assert "printed </think>" in out["edits"] and out["experience"]["lesson"] == "strip think tags"


def test_stop_during_the_final_cleanup_cancels_the_run(tmp_path):
    cancel = threading.Event()
    hr = make_run(tmp_path, cancel=cancel)
    orig = hr.cleanup_workspaces

    def cleanup(keep=None, snapshot=False):
        if keep is not None:
            cancel.set()  # Stop pressed while the other copies are deleted (minutes with a big project)
        orig(keep=keep, snapshot=snapshot)

    hr.cleanup_workspaces = cleanup
    with pytest.raises(RuntimeError, match="cancelled"):
        hr.run()  # so jobs neither applies nor adopts
    rd = tmp_path / "run"
    assert not (rd / "source").exists() and not list(rd.glob("rollouts/*/ws"))


def test_blind_never_rewrites_ordinary_text_for_a_relative_run_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    cfg = HarnessingConfig(task="t", executor_model="mock/demo", improver_model="mock/demo")
    hr = HarnessingRun(cfg, base_harness=base_harness(), source_ws=Path("src"), run_dir=Path("r"))
    ws = (tmp_path / "r/rollouts/base1/ws").resolve()
    out = hr._blind(f"error: result was wrong ({ws}/t.py, rollouts/agg1/ws/t.py)")
    assert out == "error: result was wrong (<workspace>/t.py, rollouts/<candidate>/ws/t.py)"


def test_an_unclosed_tag_quoted_mid_reply_does_not_erase_the_choice_or_the_lesson(tmp_path, monkeypatch):
    sel = ("<reasoning>B's final answer starts with <think> and never closes it.</reasoning>\n<choice>B</choice>")
    use_improver(monkeypatch, lambda role, prompt, o: sel)
    hr = make_run(tmp_path)
    hr.rollouts = {l: engine.Rollout(l, status="done", delivered=True) for l in ("base1", "agg1")}
    order = ["base1", "agg1"]
    random.Random(hr.seed).shuffle(order)
    chosen, reason = hr._select(["base1", "agg1"])
    assert chosen == order[1] and "never closes it" in reason  # the selector's own pick, not the fallback
    ref = ("<edits>\nobserved: base1 answered with an unclosed <analysis> tag\n</edits>\n"
           "<experience>\ncue: c\nlesson: l\nevidence: e\n</experience>")
    use_improver(monkeypatch, lambda role, prompt, o: ref)
    out = make_run(tmp_path)._reflect("base1")
    assert "unclosed <analysis> tag" in out["edits"] and out["experience"]["lesson"] == "l"
    # a reasoning block that opens the reply and never closes is still no answer
    assert engine.strip_reasoning("<think>\nmaybe <choice>A</choice>") == ""
