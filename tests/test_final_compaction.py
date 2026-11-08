"""After a compaction on a 128k window, Layer 1 must still have room to work before the next compaction.

Seen live (2026-11-08, nesi3, real 131,072-token window): a long code review stopped and continued with «continúa»
compacted 9 times in an hour, every 13-27 tool calls, while the same review run without the Stop never compacted
(masking kept it under the threshold). Two causes: the rebuilt conversation was ~44k tokens (35% of the window), so
every output read after it was still in the newest-kept share when the 88k threshold came; and a masking dry run that
found too little to clear still moved the hysteresis mark, postponing the next try past the compaction threshold."""

from types import SimpleNamespace

from nesicode.agent import Workspace
from nesicode.agent import context as ctxm
from nesicode.llm import ToolCall, Turn, Usage

from test_fix_context import RULE, _agent, _has, _post, fake  # noqa: F401  (fake is a fixture)

WINDOW = 131_072
SUMMARY = ("## Goal\nReview the repository and add findings to REVIEW.md as they are found.\n"
           "## Constraints & Preferences\n- " + RULE + "\n- Spanish.\n"
           "## Completed Actions\n" + "".join(f"{i}. Read src/mod{i % 12}.py lines 1-400: checked the lock order.\n"
                                                  for i in range(60)) +
           "## Active State\nReading the delivery code.\n## Next Step\nContinue with the next module.")


def _code(n, seed):
    lines, i = [], 0
    while sum(len(x) + 1 for x in lines) < n:
        lines.append(f"    async function step_{seed}_{i}(ctx: Ctx, attempt: number): Promise<void> {{ await ctx.fence({i}); }}")
        i += 1
    return "\n".join(lines)


def test_post_compact_target_scales_with_the_window():
    assert ctxm.post_compact_target(16_384) == int(0.35 * 16_384)  # small windows keep their 35% (P11)
    assert ctxm.post_compact_target(32_768) == int(0.35 * 32_768)
    assert ctxm.post_compact_target(131_072) == 32_000  # ~24% of the user's real window, not 46k
    assert ctxm.post_compact_target(1_000_000) == 200_000  # never under 20% of huge windows


def test_the_rebuild_on_a_128k_window_is_about_32k_and_keeps_the_rules(tmp_path, fake):
    # Long analysis texts (never masked) push the context over the threshold; the files read this run and a long
    # stopped earlier turn would fill a 35% rebuild.
    def policy(conv, rec):
        if rec["role"] == "summarizer":
            return Turn(SUMMARY, [], "end_turn", Usage(0, 3_000))
        n = len(prov.executor())
        if n >= 14:
            return Turn("Hecho.", [], "end_turn", Usage(0, 10))
        text = f"Análisis {n}: " + "el orden de los candados y los reintentos parece correcto en este tramo. " * 300
        return Turn(text, [ToolCall(f"r{n}", "read_file", {"path": f"src/mod{n % 12}.py"})], "tool_use", Usage(0, 6_000))

    prov = fake(policy, window=WINDOW)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "src").mkdir(parents=True)
    for i in range(12):
        (ws.root / "src" / f"mod{i}.py").write_text(_code(23_000, i))
    events = []
    history = [(f"Revisa el repo. Regla: {RULE}.", "Interrumpido. " + ("paso: leí un módulo y anoté hallazgos. " * 3_000))]
    res = _agent(ws, events, context_limit=WINDOW, max_steps=200).run("continúa", history)
    assert res.status == "done", res.error
    comp = [e for e in events if e["type"] == "compacted"]
    assert comp, "the scenario must compact"
    assert all(e["after"] <= ctxm.post_compact_target(WINDOW) * 1.1 for e in comp), [e["after"] for e in comp]
    assert all(_has(r, RULE) for r in _post(prov))  # the earlier turn's rules survive the smaller rebuild


def test_a_dry_run_that_frees_too_little_does_not_postpone_masking(tmp_path):
    agent = _agent(Workspace(tmp_path), [], context_limit=WINDOW)
    lim = WINDOW
    levels = iter([int(0.60 * lim), int(0.62 * lim), int(0.62 * lim), int(0.55 * lim), int(0.55 * lim)])
    tried = []

    def mask(conv, keep_last, keep_chars, min_tokens):
        tried.append(min_tokens)
        return 0 if len(tried) == 1 else 40_000  # first: too little to clear; then enough

    agent._est = lambda: next(levels)
    agent._mask = mask
    agent._ctx_event = lambda reason: None
    agent._can_compact = lambda: False
    conv = SimpleNamespace()
    agent._after_step(conv, None, "m", "sys", None, None, [])  # dry run: nothing cleared
    assert agent._masked_at == 0  # the hysteresis mark did not move
    # 2% of the window later (less than min_freed tokens) the next try happens and clears
    agent._after_step(conv, None, "m", "sys", None, None, [])
    assert len(tried) == 2 and agent._masked_at == int(0.55 * lim)
