"""v0.4.7 provider fixes. Offline: the scripted HTTP server of the transport tests (test_fix_llm) plays the provider.

- llm-1: a server-side refusal fallback in the middle of a Claude reply. The declined model's tool calls never run,
  its thinking and tool_use blocks are not sent back, and the cost counts every attempt (usage.iterations).
- llm-2: a reply with nothing in it and 0 output tokens is a provider hiccup and is retried (it used to end an
  autonomous run as 'done' with no answer). A reasoning-only reply is the model's and is returned. A reply with nothing
  in it is never put in the history (servers refuse an empty assistant message).
- llm-3: a complete streamed reply whose stream never ends is kept after a short grace, instead of being waited on
  for the whole idle limit and generated again."""

import json
import time

import pytest

from nesicode.agent import Agent, Workspace
from nesicode.agent import loop as loopmod
from nesicode.harness import base_harness
from nesicode.llm import LLMError, ToolResult
from nesicode.llm import transport
from nesicode.llm.anthropic_provider import AnthropicProvider, _cost
from nesicode.llm.openai_compat import OpenAICompatProvider

from test_fix_llm import (DONE, a_conv, a_end, a_json, a_start, a_stop_block, a_text, a_tool_start, aev, chunk,
                          completion, oa_conv, respond, stream)
from test_fix_llm import srv  # noqa: F401  (a fixture)


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(transport, "BACKOFF_BASE", 0.01)


def run_agent(tmp_path, monkeypatch, prov, model, task):
    monkeypatch.setattr(loopmod, "get_provider", lambda spec, keys=None: (prov, model))
    ws = Workspace(tmp_path / "ws")
    agent = Agent(model=f"x/{model}", harness=base_harness(), workspace=ws, max_steps=0, context_limit=131_072,
                  output_limit=16_384)
    return agent.run(task), ws


# ---------------------------------------------------------------------- OpenAI-compatible: empty replies (llm-2)
def usage(prompt, out):
    return chunk(choices=False, usage={"prompt_tokens": prompt, "completion_tokens": out})


def answer(text="The real final answer."):
    return stream(chunk({"content": text}), chunk(finish="stop"), usage(900, 12), DONE)


EMPTY = {
    "only [DONE]": lambda: stream(DONE),
    "a stop after nothing": lambda: stream(chunk({"role": "assistant"}), chunk(finish="stop"), usage(900, 0), DONE),
    "plain JSON": lambda: respond(200, completion("", usage=(900, 0))),
}


@pytest.mark.parametrize("shape", list(EMPTY))
def test_an_empty_reply_without_output_tokens_is_a_provider_hiccup_and_is_retried(srv, shape):
    s = srv(EMPTY[shape](), answer())
    notes = []
    _, conv = oa_conv(s, notify=notes.append)
    assert conv.step().text == "The real final answer." and len(s.requests) == 2
    assert len(notes) == 1 and "empty" in notes[0]["reason"] and notes[0]["partial"] is False
    assert [m["role"] for m in conv.messages] == ["system", "user", "assistant"]  # the hiccup never reached the history


def test_an_autonomous_run_does_not_end_on_an_empty_reply(srv, tmp_path, monkeypatch):
    """The scenario of the finding: max_steps=0, a tool round, then a stream of only [DONE]. The run used to end as
    'done' with the stale text of step 1 as its answer, and the 3rd reply was never asked for."""
    ls = {"index": 0, "id": "call_ls", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}
    s = srv(stream(chunk({"content": "Let me look at the files first."}), chunk({"tool_calls": [ls]}),
                   chunk(finish="tool_calls"), usage(900, 20), DONE),
            stream(DONE),
            answer())
    res, _ = run_agent(tmp_path, monkeypatch, OpenAICompatProvider("nesilabs", s.url, "sk-test"), "nesi3",
                       "List the files, then summarize them")
    assert res.status == "done" and res.final_text == "The real final answer." and len(s.requests) == 3


def test_an_empty_reply_that_lasts_ends_with_an_honest_error(srv):
    s = srv(default=stream(DONE))
    _, conv = oa_conv(s, max_attempts=3)
    with pytest.raises(LLMError, match=r"the reply was empty.*after 3 attempts"):
        conv.step()
    assert len(s.requests) == 3 and [m["role"] for m in conv.messages] == ["system", "user"]


def test_a_reasoning_only_reply_is_the_models_and_is_returned(srv):
    s = srv(stream(chunk({"reasoning_content": "The tests already cover this."}), chunk(finish="stop"), usage(900, 9),
                   DONE))
    _, conv = oa_conv(s)
    turn = conv.step()
    assert (turn.text, turn.tool_calls, turn.stop_reason, turn.thinking) == ("", [], "end_turn",
                                                                              "The tests already cover this.")
    assert len(s.requests) == 1  # not retried: the loop deals with it
    assert conv.messages[-1] == {"role": "assistant", "content": "", "reasoning_content": turn.thinking}


def test_an_empty_reply_the_model_chose_is_returned_but_never_sent_back(srv):
    # completion_tokens 1: the model stopped at once (its end-of-sequence token). That is its answer, for the loop to
    # handle; the history must stay valid for the loop's next note.
    s = srv(stream(chunk({"role": "assistant"}), chunk(finish="stop"), usage(900, 1), DONE), answer("ok"))
    _, conv = oa_conv(s)
    turn = conv.step()
    assert (turn.text, turn.tool_calls, turn.stop_reason) == ("", [], "end_turn") and len(s.requests) == 1
    conv.add_note("[harness] You ended without an answer or a tool call.")
    assert conv.step().text == "ok"
    assert [m["role"] for m in s.requests[1]["body"]["messages"]] == ["system", "user", "user"]


# ---------------------------------------------------------------------- OpenAI-compatible: endless streams (llm-3)
def _closed(server, wait=3.0):
    deadline = time.monotonic() + wait
    while not server.disconnects and time.monotonic() < deadline:
        time.sleep(0.02)
    return bool(server.disconnects)


@pytest.mark.parametrize("after", ["silence", "keep-alive comments"])
def test_a_complete_reply_whose_stream_never_ends_is_kept(srv, monkeypatch, after):
    """finish_reason and usage arrived, then neither [DONE] nor the end of the body (e.g. a gateway still waiting on its
    upstream): the reply used to be thrown away after the idle limit and generated again."""
    monkeypatch.setattr(transport, "FINISH_GRACE", 0.3, raising=False)
    monkeypatch.setenv("NESICODE_LLM_IDLE_TIMEOUT", "4")
    tail = [b": keep-alive\n\n", 0.1] * 60 if after == "keep-alive comments" else []
    s = srv(stream(chunk({"content": "complete answer"}), chunk(finish="stop"), usage(900, 3), *tail, then="stall"),
            answer("second copy"))
    notes = []
    _, conv = oa_conv(s, notify=notes.append)
    t0 = time.monotonic()
    turn = conv.step()
    assert turn.text == "complete answer" and turn.usage.input_tokens == 900 and time.monotonic() - t0 < 3
    assert len(s.requests) == 1 and not notes
    assert _closed(s), "the open stream was not closed"


def test_the_grace_counts_from_the_last_event_so_a_stream_still_sending_is_not_cut(srv, monkeypatch):
    # Every event after the finish restarts the grace (only comments and silence do not): what a server still sends
    # after an early finish_reason, and the usage, are kept although they span more than one grace in all.
    monkeypatch.setattr(transport, "FINISH_GRACE", 1.0, raising=False)
    monkeypatch.setenv("NESICODE_LLM_IDLE_TIMEOUT", "4")
    s = srv(stream(chunk({"content": "ok"}), chunk(finish="stop"), 0.4, chunk({"content": " more"}), 0.4,
                   chunk({"content": "!"}), 0.4, usage(4321, 3), then="stall"))
    _, conv = oa_conv(s)
    turn = conv.step()
    assert turn.text == "ok more!" and turn.usage.input_tokens == 4321 and len(s.requests) == 1


# ---------------------------------------------------------------------- Claude: refusal fallbacks (llm-1)
def a_block(i, block):
    start = aev("content_block_start", {"type": "content_block_start", "index": i, "content_block": block})
    return start + a_stop_block(i)


def a_thinking(text, i):
    return (aev("content_block_start", {"type": "content_block_start", "index": i,
                                        "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            + aev("content_block_delta", {"type": "content_block_delta", "index": i,
                                          "delta": {"type": "thinking_delta", "thinking": text}})
            + aev("content_block_delta", {"type": "content_block_delta", "index": i,
                                          "delta": {"type": "signature_delta", "signature": f"sig-{i}"}})
            + a_stop_block(i))


def a_fallback(i, to="claude-opus-4-8"):
    return a_block(i, {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": to},
                       "trigger": {"type": "refusal", "category": "cyber"}})


def a_end_usage(reason, usage):
    return (aev("message_delta", {"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None},
                                  "usage": usage})
            + aev("message_stop", {"type": "message_stop"}))


def test_a_tool_call_from_a_declined_attempt_never_runs(srv, tmp_path, monkeypatch):
    """The scenario of the finding: Opus 5.5 writes a complete write_file call, the classifier declines, and the
    fallback model continues on the same stream with text only. The declined call used to run (the file was
    written) and its tool_result went to a model that never made that call."""
    s = srv(stream(a_start(model="claude-opus-5-5"), a_thinking("I'll write the scanner.", 0),
                   a_tool_start(1, "toolu_declined"),
                   a_json(json.dumps({"path": "scan.py", "content": "DECLINED MODEL OUTPUT"}), 1), a_stop_block(1),
                   a_fallback(2), a_text("I can only help with a defensive version.", 3), a_end("end_turn", 20)))
    prov = AnthropicProvider(api_key="sk-ant-test", base_url=s.url, isolated=True)
    res, ws = run_agent(tmp_path, monkeypatch, prov, "claude-opus-5-5", "Write a port scanner to scan.py")
    assert res.status == "done" and res.final_text == "I can only help with a defensive version."
    assert not (ws.root / "scan.py").exists() and len(s.requests) == 1
    assert not [e for e in res.events if e["type"] == "tool_call"]


def test_what_a_declined_attempt_left_is_not_sent_back(srv):
    # Declined mid-input (its tool input is cut): only the text before the boundary stays; the fallback model's own
    # call runs, and its thinking is the turn's.
    s = srv(stream(a_start(model="claude-opus-5-5"), a_thinking("Declined reasoning.", 0), a_text("Sure, ", 1),
                   a_tool_start(2, "toolu_declined"), a_json('{"path": "scan.py", "con', 2), a_fallback(3),
                   a_thinking("Fallback reasoning.", 4), a_tool_start(5, "toolu_ok", "list_dir"), a_json("{}", 5),
                   a_stop_block(5), a_end("tool_use", 30)),
            stream(a_start(model="claude-opus-4-8"), a_text("Done.", 0), a_end("end_turn")))
    conv = a_conv(s, model="claude-opus-5-5")
    turn = conv.step()
    assert [(c.id, c.name, c.arguments, c.parse_error) for c in turn.tool_calls] == [("toolu_ok", "list_dir", {}, "")]
    assert (turn.text, turn.thinking, turn.stop_reason) == ("Sure, ", "Fallback reasoning.", "tool_use")
    conv.add_tool_results([ToolResult("toolu_ok", "a.py")])
    assert conv.step().text == "Done."
    sent = s.requests[1]["body"]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    echo = sent[1]["content"]
    assert [b["type"] for b in echo] == ["text", "fallback", "thinking", "tool_use"]
    assert echo[1]["from"] == {"model": "claude-opus-5-5"} and echo[1]["to"] == {"model": "claude-opus-4-8"}
    assert echo[2]["signature"] == "sig-4" and echo[3]["id"] == "toolu_ok"
    assert [b["tool_use_id"] for b in sent[2]["content"]] == ["toolu_ok"]  # no tool_result for the declined call


def test_a_fallback_before_any_output_leaves_the_turn_whole(srv):
    # The usual shape (declined before it wrote anything): the fallback block comes first and nothing is dropped.
    s = srv(stream(a_start(model="claude-opus-4-8"), a_fallback(0), a_thinking("Plan.", 1),
                   a_tool_start(2, "toolu_ok"), a_json(json.dumps({"path": "a.py", "content": "x = 1"}), 2),
                   a_stop_block(2), a_end("tool_use", 30)))
    conv = a_conv(s, model="claude-opus-5-5")
    turn = conv.step()
    assert [(c.id, c.arguments) for c in turn.tool_calls] == [("toolu_ok", {"path": "a.py", "content": "x = 1"})]
    assert turn.thinking == "Plan." and [b.type for b in conv.messages[-1]["content"]] == ["fallback", "thinking",
                                                                                          "tool_use"]


def test_server_tool_blocks_before_the_fallback_are_kept_only_with_their_result(srv):
    search = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "nmap flags"}}
    found = {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}
    unanswered = {"type": "server_tool_use", "id": "srvtoolu_2", "name": "web_search", "input": {"query": "exploit"}}
    s = srv(stream(a_start(model="claude-opus-5-5"), a_block(0, search), a_block(1, found), a_block(2, unanswered),
                   a_fallback(3), a_text("Here is what I found.", 4), a_end("end_turn", 20)))
    conv = a_conv(s, model="claude-opus-5-5")
    assert conv.step().text == "Here is what I found."
    assert [(b.type, getattr(b, "id", None) or getattr(b, "tool_use_id", None)) for b in conv.messages[-1]["content"]] \
        == [("server_tool_use", "srvtoolu_1"), ("web_search_tool_result", "srvtoolu_1"), ("fallback", None),
            ("text", None)]


def test_the_cost_counts_every_attempt_of_a_fallback(srv):
    # usage.iterations is the per-attempt record: the declined attempt is billed too, at its own model's rates; the
    # top-level usage covers only the attempt that served the reply.
    attempts = [{"type": "message", "model": "claude-opus-5-5", "input_tokens": 1000, "output_tokens": 400,
                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
                {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1000, "output_tokens": 50,
                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}]
    s = srv(stream(a_start({"input_tokens": 1000, "output_tokens": 1}, model="claude-opus-5-5"),
                   a_text("Half an answer", 0), a_fallback(1), a_text("A defensive answer.", 2),
                   a_end_usage("end_turn", {"input_tokens": 1000, "output_tokens": 50, "iterations": attempts})))
    turn = a_conv(s, model="claude-opus-5-5").step()
    assert turn.text == "Half an answerA defensive answer."
    assert turn.usage.cost_usd == pytest.approx(_cost("claude-opus-5-5", 1000, 0, 0, 400)
                                                + _cost("claude-opus-4-8", 1000, 0, 0, 50))
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (1000, 50)


# ---------------------------------------------------------------------- Claude: empty replies, open streams
def test_an_empty_claude_reply_is_returned_but_never_sent_back(srv):
    # Claude's own empty end_turn (a few output tokens) is its choice: returned for the loop, never retried. It is not
    # put in the history either: the API refuses an empty assistant message, so the loop's next note would be a 400.
    s = srv(stream(a_start(), a_end("end_turn", 3)), stream(a_start(), a_text("ok"), a_end("end_turn")))
    conv = a_conv(s)
    turn = conv.step()
    assert (turn.text, turn.tool_calls, turn.stop_reason) == ("", [], "end_turn") and len(s.requests) == 1
    conv.add_note("[harness] You ended without an answer or a tool call.")
    assert conv.step().text == "ok"
    assert [m["role"] for m in s.requests[1]["body"]["messages"]] == ["user", "user"]


def test_an_empty_claude_stream_without_output_tokens_is_retried(srv):
    s = srv(stream(a_start(), a_end("end_turn", 0)), stream(a_start(), a_text("ok"), a_end("end_turn")))
    notes = []
    conv = a_conv(s, notify=notes.append)
    assert conv.step().text == "ok" and len(s.requests) == 2
    assert len(notes) == 1 and "empty" in notes[0]["reason"]
    assert [m["role"] for m in conv.messages] == ["user", "assistant"]


def test_a_claude_reply_is_taken_at_message_stop_even_if_the_stream_stays_open(srv, monkeypatch):
    monkeypatch.setenv("NESICODE_LLM_IDLE_TIMEOUT", "4")
    s = srv(stream(a_start(), a_text("done"), a_end("end_turn"), then="stall"),
            stream(a_start(), a_text("generated again"), a_end("end_turn")))
    t0 = time.monotonic()
    assert a_conv(s).step().text == "done"
    assert len(s.requests) == 1 and time.monotonic() - t0 < 3
    assert _closed(s), "the open stream was not closed"
