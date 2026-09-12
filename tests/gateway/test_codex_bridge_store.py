"""Durable ownership contracts for the Telegram ↔ Codex bridge."""

import sqlite3

from gateway.codex_bridge.store import CodexBridgeStore
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, build_session_key


def _source(**kwargs) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="1001",
        user_id="42",
        chat_type="dm",
        **kwargs,
    )


def _event(source: SessionSource, message_id: str = "77") -> MessageEvent:
    return MessageEvent(text="continue the work", source=source, message_id=message_id)


def test_binding_generation_fences_a_to_b_to_a(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()

    first = store.bind("control", source, thread_id="thread-a", cwd="/a")
    second = store.bind("control", source, thread_id="thread-b", cwd="/b")
    third = store.bind("control", source, thread_id="thread-a", cwd="/a")

    assert (first.generation, second.generation, third.generation) == (1, 2, 3)
    assert third.thread_id == "thread-a"


def test_telegram_topics_keep_independent_codex_bindings(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    first_source = _source(thread_id="topic-10")
    second_source = _source(thread_id="topic-20")
    first_key = build_session_key(first_source)
    second_key = build_session_key(second_source)

    store.bind(first_key, first_source, thread_id="codex-a")
    store.bind(second_key, second_source, thread_id="codex-b")

    assert first_key != second_key
    assert {
        (row.source.thread_id, row.thread_id) for row in store.list_bindings()
    } == {("topic-10", "codex-a"), ("topic-20", "codex-b")}


def test_input_lifecycle_and_restart_recovery(tmp_path, monkeypatch):
    store = CodexBridgeStore(tmp_path / "state.db")
    binding = store.bind("control", _source(), thread_id="thread-a")
    input_id, state, inserted = store.enqueue_input(binding, "lane-a", _event(binding.source))

    assert state == "routed"
    assert inserted is True
    assert store.mark_executing(input_id)
    assert store.mark_executed(input_id, "finished")

    monkeypatch.setattr(store, "_owner_alive", lambda _pid, _started: False)
    store.recover_after_restart()
    recovered = store.recoverable_outputs()
    assert [(row.input_id, row.final_text) for row in recovered] == [(input_id, "finished")]
    assert (recovered[0].delivery_owner, recovered[0].turn_outcome) == ("ledger", "completed")

    store.mark_completed(input_id)
    assert store.input_state(input_id) == "completed"
    with sqlite3.connect(store.path) as conn:
        assert conn.execute(
            "SELECT delivery_owner, turn_outcome FROM codex_bridge_inputs WHERE input_id=?",
            (input_id,),
        ).fetchone() == ("none", "completed")


def test_restart_does_not_steal_live_gateway_input(tmp_path, monkeypatch):
    store = CodexBridgeStore(tmp_path / "state.db")
    binding = store.bind("control", _source(), thread_id="thread-a")
    input_id, _, _ = store.enqueue_input(binding, "lane-a", _event(binding.source))
    assert store.mark_executing(input_id)

    monkeypatch.setattr(store, "_owner_alive", lambda _pid, _started: True)
    store.recover_after_restart()
    assert store.input_state(input_id) == "executing"
    assert store.recoverable_inputs() == []


def test_restart_never_replays_a_possibly_submitted_turn(tmp_path, monkeypatch):
    store = CodexBridgeStore(tmp_path / "state.db")
    binding = store.bind("control", _source(), thread_id="thread-a")
    input_id, _, _ = store.enqueue_input(binding, "lane-a", _event(binding.source))
    assert store.mark_executing(input_id)
    assert store.mark_submitting(input_id)
    assert store.mark_running(input_id, "turn-123")

    monkeypatch.setattr(store, "_owner_alive", lambda _pid, _started: False)
    store.recover_after_restart()

    assert store.input_state(input_id) == "uncertain"
    assert store.recoverable_inputs() == []
    assert store.capture_recovery_output(input_id, "recovered final")
    outputs = store.recoverable_outputs()
    assert [(row.input_id, row.final_text, row.codex_turn_id) for row in outputs] == [
        (input_id, "recovered final", "turn-123"),
    ]


def test_pending_new_promotion_updates_owned_inputs_without_rotating_grant(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    binding = store.bind(
        "control", _source(), thread_id="pending_ns_token", cwd="/project", pending_new=True,
    )
    input_id, _, _ = store.enqueue_input(binding, "lane-a", _event(binding.source))
    assert store.mark_executing(input_id)

    promoted = store.promote_pending(binding, "real-thread-id")
    assert promoted is not None
    assert promoted.generation == binding.generation
    assert promoted.thread_id == "real-thread-id"
    assert promoted.pending_new is False

    with sqlite3.connect(store.path) as conn:
        row = conn.execute(
            "SELECT thread_id, generation, state FROM codex_bridge_inputs WHERE input_id=?",
            (input_id,),
        ).fetchone()
    assert row == ("real-thread-id", binding.generation, "executing")


def test_queue_capacity_rejection_can_cancel_routed_input(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    binding = store.bind("control", _source(), thread_id="thread-a")
    input_id, _, _ = store.enqueue_input(binding, "lane-a", _event(binding.source))
    store.cancel_input(input_id, "busy queue at capacity")
    assert store.input_state(input_id) == "cancelled"
