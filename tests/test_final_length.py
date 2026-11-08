"""A reply the server stops as 'length' after a handful of tokens is a provider hiccup, not the output limit.

Seen live (2026-11-08, nesi3 behind LiteLLM): for about two minutes three runs got replies cut as finish_reason=length
after 1-5 tokens although 16,384 were allowed; the loop took them for real cuts, "continued" them four times and
ended the runs with "se cortó en el límite de salida (16,384 tokens) 4 veces seguidas". Offline: the scripted SSE
server of the transport tests plays the provider."""

import time

import pytest

from nesicode.agent import Agent, Workspace
from nesicode.llm import LLMError, ModelOptions, Turn, Usage
from nesicode.llm import transport
from nesicode.llm.openai_compat import OpenAICompatProvider

from test_fix_llm import DONE, TOOLS, chunk, completion, respond, srv, stream  # noqa: F401  (srv is a fixture)


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(transport, "BACKOFF_BASE", 0.01)


def conv_for(server, max_tokens=16384, **opts):
    prov = OpenAICompatProvider("nesilabs", server.url, "sk-test")
    conv = prov.conversation("nesi3", "sys", TOOLS, ModelOptions(max_tokens=max_tokens, **opts))
    conv.add_user("¿Cuál era el release_code?")
    return prov, conv


def cut_after(text, tokens=1):
    return stream(chunk({"reasoning_content": text}), chunk(finish="length"),
                  chunk(choices=False, usage={"prompt_tokens": 41308, "completion_tokens": tokens}), DONE)


def good():
    return stream(chunk({"reasoning_content": "Está en el config."}), chunk({"content": "ORCA-7731, línea 83."}),
                  chunk(finish="stop"), chunk(choices=False, usage={"prompt_tokens": 41308, "completion_tokens": 40}),
                  DONE)


def test_a_reply_cut_as_length_after_a_few_tokens_is_retried_not_taken_for_a_cut(srv):
    s = srv(cut_after("Need"), cut_after("Just"), good())
    notes, deltas = [], []
    _, conv = conv_for(s, notify=notes.append)
    turn = conv.step(lambda kind, text: deltas.append((kind, text)))
    assert turn.text == "ORCA-7731, línea 83." and turn.stop_reason == "end_turn"
    assert len(s.requests) == 3 and len(notes) == 2
    assert all("'length' after 1 tokens" in n["reason"] and n["partial"] is True for n in notes)
    # the history holds the good reply only (the hiccups were never appended)
    assert [m["role"] for m in conv.messages if m["role"] != "system"] == ["user", "assistant"]


def test_plain_json_replies_are_checked_too(srv):
    s = srv(respond(200, completion("", finish="length", usage=(41308, 2))),
            respond(200, completion("ORCA-7731", usage=(41308, 9))))
    prov, conv = conv_for(s)
    prov.stream = False
    assert conv.step().text == "ORCA-7731" and len(s.requests) == 2


def test_a_real_cut_is_not_retried(srv):
    # The model used the whole budget (usage says so): a genuine cut, handled by the loop as before.
    s = srv(stream(chunk({"content": "x" * 400}), chunk(finish="length"),
                   chunk(choices=False, usage={"prompt_tokens": 10, "completion_tokens": 16384}), DONE))
    _, conv = conv_for(s)
    turn = conv.step()
    assert turn.stop_reason == "max_tokens" and len(s.requests) == 1


def test_long_reasoning_left_out_of_completion_tokens_is_not_a_hiccup(srv):
    # Some servers count only the visible answer in completion_tokens: what came back decides as well.
    s = srv(stream(chunk({"reasoning_content": "Thinking about the lock order. " * 40}), chunk(finish="length"),
                   chunk(choices=False, usage={"prompt_tokens": 10, "completion_tokens": 0}), DONE))
    _, conv = conv_for(s)
    assert conv.step().stop_reason == "max_tokens" and len(s.requests) == 1


def test_small_budgets_are_left_alone(srv):
    # A deliberately tiny max_tokens (e.g. a probe) can legitimately stop after a few tokens.
    s = srv(stream(chunk({"content": "ok"}), chunk(finish="length"), DONE))
    _, conv = conv_for(s, max_tokens=8)
    assert conv.step().stop_reason == "max_tokens" and len(s.requests) == 1


def test_a_hiccup_that_lasts_ends_with_an_honest_error(srv):
    s = srv(default=cut_after("Keep"))
    _, conv = conv_for(s, max_attempts=3)
    with pytest.raises(LLMError, match=r"stopped as 'length' after 1 tokens although 16384 were allowed.*after 3 attempts"):
        conv.step()
    assert len(s.requests) == 3


def test_retries_stop_at_the_time_budget(monkeypatch):
    calls = []

    def attempt(n):
        calls.append(n)
        raise transport.Retryable("first byte never came")

    clock = iter(range(0, 100_000, 400))  # every look at the clock: 400 s later (a slow failure per attempt)
    monkeypatch.setattr(transport.time, "monotonic", lambda: next(clock))
    with pytest.raises(LLMError, match="first byte never came"):
        transport.with_retries(attempt, label="x", attempts=8)
    assert len(calls) < 8  # the 900-s budget ended it before the attempt count did


def test_default_budget_survives_a_two_minute_incident(monkeypatch):
    # 8 attempts with the default backoff wait ~3 minutes in all (jitter ±25%): longer than the incident seen live.
    monkeypatch.setattr(transport, "BACKOFF_BASE", 2.0)  # the real default (the autouse fixture speeds it up)
    waits = [min(transport.BACKOFF_CAP, transport.BACKOFF_BASE * 2 ** (n - 1)) for n in range(1, transport.DEFAULT_ATTEMPTS)]
    assert 0.75 * sum(waits) > 120 and sum(waits) * 1.25 < transport.RETRY_BUDGET


class _CutProvider:
    """Every reply comes back cut as max_tokens after one token (what the loop sees if the hiccup outlasts the
    provider's retries on a server whose stream the provider cannot judge)."""

    name = "fake"

    def conversation(self, model, system, tools, options):
        prov = self

        class Conv:
            messages = []

            def add_user(self, text):
                self.messages.append({"role": "user", "content": text})

            def add_assistant_text(self, text):
                self.messages.append({"role": "assistant", "content": text})

            def add_tool_results(self, results):
                pass

            def mask_tool_results(self, keep_last, marker, *, keep_chars=0, apply=True):
                return 0

            def step(self, on_event=None):
                return Turn(text="", tool_calls=[], stop_reason="max_tokens", usage=Usage(1000, 1), thinking="Just")

        return Conv()


def test_the_loops_message_says_how_little_came_back(tmp_path, monkeypatch):
    from nesicode.agent import loop
    from nesicode.harness import base_harness

    monkeypatch.setattr(loop, "get_provider", lambda spec, keys=None: (_CutProvider(), "m"))
    agent = Agent(model="fake/m", harness=base_harness(), workspace=Workspace(tmp_path), context_limit=131_072,
                  output_limit=16_384)
    res = agent.run("¿Cuál era el release_code?")
    assert res.status == "error"
    assert "tras solo 1 tokens de 16,384" in res.error and "continúa" in res.error
