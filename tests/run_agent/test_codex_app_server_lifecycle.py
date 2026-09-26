"""Codex app-server session lifecycle on hard agent teardown (#65260).

The Codex runtime drops ``agent._codex_session`` on turn crash and on
retirement (agent/codex_runtime.py), but ``AIAgent.close()`` — the hard
teardown for /new, /reset, and session expiry — had no owner for it, so
the app-server child process survived until interpreter exit.
"""

import threading
from types import SimpleNamespace

import pytest

from agent import codex_runtime
from agent.transports.codex_app_server_session import TurnResult
from run_agent import AIAgent


class _FakeCodexSession:
    def __init__(self, raises: bool = False):
        self.close_calls = 0
        self._raises = raises

    def close(self):
        self.close_calls += 1
        if self._raises:
            raise RuntimeError("app-server already dead")


def _bare_agent(session_id: str) -> AIAgent:
    """Minimal agent shell exercising close() without a real build."""
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent.client = None
    agent._active_children_lock = threading.Lock()
    agent._active_children = set()
    agent._end_session_on_close = False
    agent._session_messages = ["retained"]
    return agent


def test_agent_close_releases_codex_app_server_session(monkeypatch):
    agent = _bare_agent("test-codex-lifecycle")
    codex_session = _FakeCodexSession()
    agent._codex_session = codex_session

    monkeypatch.setattr("run_agent.cleanup_vm", lambda _task_id: None)
    monkeypatch.setattr("run_agent.cleanup_browser", lambda _task_id: None)

    agent.close()
    agent.close()

    # Idempotent: the second close must not re-close a released session.
    assert codex_session.close_calls == 1
    assert agent._codex_session is None
    assert agent._session_messages == []


def test_close_clears_reference_even_when_session_close_raises(monkeypatch):
    """A wedged app-server must not strand a stale session reference.

    The attribute is cleared BEFORE close() precisely so a raising close
    can't leave a dead session attached to the agent.
    """
    agent = _bare_agent("test-codex-lifecycle-raises")
    codex_session = _FakeCodexSession(raises=True)
    agent._codex_session = codex_session

    monkeypatch.setattr("run_agent.cleanup_vm", lambda _task_id: None)
    monkeypatch.setattr("run_agent.cleanup_browser", lambda _task_id: None)

    agent.close()

    assert codex_session.close_calls == 1
    assert agent._codex_session is None


def test_close_without_codex_session_is_a_noop(monkeypatch):
    """Non-Codex sessions (the common case) must be unaffected."""
    agent = _bare_agent("test-no-codex")

    monkeypatch.setattr("run_agent.cleanup_vm", lambda _task_id: None)
    monkeypatch.setattr("run_agent.cleanup_browser", lambda _task_id: None)

    agent.close()

    assert getattr(agent, "_codex_session", None) is None


def test_cache_eviction_releases_codex_writer_without_tearing_down_session_tools():
    agent = _bare_agent("test-codex-cache-eviction")
    codex_session = _FakeCodexSession()
    agent._codex_session = codex_session

    agent.release_clients()

    assert codex_session.close_calls == 1
    assert agent._codex_session is None
    assert agent._session_messages == ["retained"]


@pytest.mark.parametrize("bridge_owned", [True, False])
def test_completed_bridge_turn_releases_writer_but_other_codex_turns_reuse_it(
    monkeypatch, bridge_owned,
):
    turn = TurnResult(
        final_text="reply", turn_id="turn-1", thread_id="thread-1",
        submitted_user_text="hello", turn_status="completed", turn_status_confirmed=True,
    )
    closed = []
    session = SimpleNamespace(
        run_turn=lambda **_kwargs: turn, close=lambda: closed.append(True),
    )
    agent = SimpleNamespace(
        _codex_session=session, _gateway_codex_full_access=bridge_owned,
        _interim_text_was_delivered=lambda _text: False,
    )
    monkeypatch.setattr(codex_runtime, "_ensure_codex_session", lambda _agent: None)
    monkeypatch.setattr(codex_runtime, "_persist_projected_messages", lambda *_args: None)
    monkeypatch.setattr(codex_runtime, "_finish_codex_turn", lambda *_args, **_kwargs: {})

    result = codex_runtime.run_codex_app_server_turn(
        agent, user_message="hello", original_user_message="hello",
        messages=[{"role": "user", "content": "hello"}], effective_task_id="task-1",
    )

    assert result["completed"] is True
    assert result["codex_thread_id"] == "thread-1"
    assert bool(closed) is bridge_owned
    assert (agent._codex_session is None) is bridge_owned


def test_bridge_releases_writer_when_post_turn_processing_fails(monkeypatch):
    turn = TurnResult(final_text="reply", turn_id="turn-1", thread_id="thread-1")
    closed = []
    session = SimpleNamespace(
        run_turn=lambda **_kwargs: turn, close=lambda: closed.append(True),
    )
    agent = SimpleNamespace(_codex_session=session, _gateway_codex_full_access=True)
    monkeypatch.setattr(codex_runtime, "_ensure_codex_session", lambda _agent: None)
    monkeypatch.setattr(codex_runtime, "_persist_projected_messages", lambda *_args: None)

    def fail_finish(*_args, **_kwargs):
        raise RuntimeError("session persistence failed")

    monkeypatch.setattr(codex_runtime, "_finish_codex_turn", fail_finish)

    with pytest.raises(RuntimeError, match="session persistence failed"):
        codex_runtime.run_codex_app_server_turn(
            agent, user_message="hello", original_user_message="hello",
            messages=[{"role": "user", "content": "hello"}], effective_task_id="task-1",
        )

    assert closed == [True]
    assert agent._codex_session is None
