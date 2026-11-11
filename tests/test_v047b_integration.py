"""v0.4.7, round 2: where the groups' work joins (offline)."""

from pathlib import Path

from nesicode.agent import Workspace
from nesicode.agent.loop import NO_DRAFTS, OWN_FOLDER, ROOMY_WINDOW, build_system
from nesicode.harness import base_harness

WEB = Path(__file__).resolve().parents[1] / "web" / "src"


def test_the_optional_prompt_lines_only_go_where_tokens_are_cheap(tmp_path):
    """On a small window every token of the system prompt is paid on every request and moves the compactions earlier
    (tests/test_final_ctx.py::test_small_windows_with_a_reasoning_model_compact_rarely): the two optional lines go
    only on roomy (or unknown) windows, such as the user's 131,072."""
    ws = Workspace(tmp_path / "ws")
    for window, sent in ((16_384, False), (32_768, False), (ROOMY_WINDOW, True), (131_072, True), (None, True)):
        system = build_system(base_harness(), ws, context_limit=window)
        assert (NO_DRAFTS in system) is sent and (OWN_FOLDER in system) is sent, window


def test_harness_notes_and_cut_replies_have_their_own_lines_in_the_transcript():
    logic = (WEB / "logic.ts").read_text(encoding="utf-8")
    tsx = (WEB / "components" / "Transcript.tsx").read_text(encoding="utf-8")
    i18n = (WEB / "i18n.ts").read_text(encoding="utf-8")
    assert "'stopped', 'note', 'output_cut'," in logic
    assert "case 'note':" in tsx and "case 'output_cut':" in tsx
    for kind in ("read_streak", "final_answer", "repeat"):  # the telling ones; the rest already have a line or none
        assert f"  {kind}: " in tsx
    for key in ("Se recordó al agente que anote lo que ha encontrado", "Se pidió al agente su respuesta final",
                "Aviso al agente: estaba repitiendo la misma llamada",
                "Respuesta cortada en el límite de salida ({n} tokens)",
                "Respuesta cortada en el límite de salida ({n} tokens, solo razonamiento)", "se le pide que siga ({k}/3)"):
        assert f"t('{key}'" in tsx and f"'{key}': '" in i18n, key
