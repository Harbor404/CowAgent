# encoding:utf-8
"""Provider-reported usage is persisted and shown per assistant reply.

The context donut answers "how full is the window right now?" from the latest
request. These tests pin the separate, per-message view: a reply's complete
provider usage is persisted with the reply, survives a history reload, and is
exposed to the web console.
"""

import tempfile
from pathlib import Path

from agent.memory.conversation_store import ConversationStore
from agent.protocol.agent_stream import AgentStreamExecutor


def _user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant(text, usage=None):
    message = {"role": "assistant", "content": [{"type": "text", "text": text}]}
    if usage is not None:
        message["extras"] = {"usage": usage}
    return message


def test_history_turn_exposes_persisted_assistant_usage():
    usage = {
        "prompt_tokens": 1200,
        "completion_tokens": 240,
        "total_tokens": 1440,
        "prompt_cache_hit_tokens": 800,
    }
    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(Path(tmp) / "index.db")
        store.append_messages("s", [_user("question"), _assistant("answer", usage)])

        page = store.load_history_page("s")

        assistant = [m for m in page["messages"] if m["role"] == "assistant"][0]
        assert assistant["usage"] == usage


def test_latest_pair_seqs_returns_latest_assistant_usage_for_live_done():
    usage = {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "prompt_cache_hit_tokens": 12,
    }
    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(Path(tmp) / "index.db")
        store.append_messages("s", [_user("question"), _assistant("answer", usage)])

        latest = store.get_latest_pair_seqs("s")

        assert latest["usage"] == usage


def test_stream_usage_is_normalized_and_accumulated_across_model_calls():
    first = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
             "prompt_tokens_details": {"cached_tokens": 600}}
    second = {"prompt_tokens": 1400, "completion_tokens": 200, "total_tokens": 1600,
              "prompt_cache_hit_tokens": 1200}

    normalized = AgentStreamExecutor._normalize_stream_usage(first)
    combined = AgentStreamExecutor._merge_usage(
        normalized, AgentStreamExecutor._normalize_stream_usage(second)
    )

    assert normalized["prompt_cache_miss_tokens"] == 400
    assert normalized["calls"] == 1
    assert combined == {
        "prompt_tokens": 2400,
        "completion_tokens": 300,
        "total_tokens": 2700,
        "prompt_cache_hit_tokens": 1800,
        "prompt_cache_miss_tokens": 600,
        "calls": 2,
    }


def test_run_usage_is_attached_to_the_final_assistant_message():
    executor = object.__new__(AgentStreamExecutor)
    executor.messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "tool step"}]},
        _user("tool result"),
        _assistant("final answer"),
    ]
    executor.run_usage = {
        "prompt_tokens": 300,
        "completion_tokens": 40,
        "total_tokens": 340,
        "prompt_cache_hit_tokens": 100,
        "prompt_cache_miss_tokens": 200,
        "calls": 2,
    }

    executor._attach_run_usage_to_last_assistant()

    assert executor.messages[-1]["extras"]["usage"] == executor.run_usage


def test_web_console_wires_usage_into_history_and_live_done():
    render_js = Path("channel/web/static/js/chat/render.js").read_text(encoding="utf-8")
    send_js = Path("channel/web/static/js/chat/send.js").read_text(encoding="utf-8")
    channel_py = Path("channel/web/core/channel.py").read_text(encoding="utf-8")

    assert "renderMessageUsageHtml(msg && msg.usage)" in render_js
    assert "attachMessageUsage(targetBotEl, item.usage)" in send_js
    assert channel_py.count('"usage": seqs.get("usage")') == 2
