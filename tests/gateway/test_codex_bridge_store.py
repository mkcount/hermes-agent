"""Durable ownership contracts for the Telegram ↔ Codex bridge."""

import sqlite3

import pytest

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


def test_binding_schema_migrates_model_and_reasoning_columns(tmp_path):
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE codex_bridge_bindings (
                control_session_key TEXT PRIMARY KEY,
                thread_id TEXT,
                cwd TEXT NOT NULL DEFAULT '',
                generation INTEGER NOT NULL,
                source_json TEXT NOT NULL,
                rollout_path TEXT,
                cursor_device INTEGER,
                cursor_inode INTEGER,
                cursor_offset INTEGER NOT NULL DEFAULT 0,
                last_event_id TEXT,
                pending_new INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )

    CodexBridgeStore(path)

    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(codex_bridge_bindings)")}
    assert {"codex_model", "reasoning_effort"} <= columns


def test_inference_pair_update_is_atomic_and_generation_fenced(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    first = store.bind(
        "control", _source(), thread_id="thread-a",
        codex_model="gpt-5.6-sol", reasoning_effort="xhigh",
    )

    updated = store.set_inference(
        first, codex_model="gpt-6-astra", reasoning_effort="high",
    )
    assert (updated.codex_model, updated.reasoning_effort) == ("gpt-6-astra", "high")

    store.bind("control", _source(), thread_id="thread-b")
    assert store.set_inference(
        first, codex_model="gpt-5.6-terra", reasoning_effort="medium",
    ) is None
    current = store.get_binding("control")
    assert current.thread_id == "thread-b"
    assert current.codex_model is None
    assert current.reasoning_effort is None


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
    assert store.has_lane_session("lane-a") is True
    assert store.has_lane_session("lane-missing") is False
    assert store.mark_executing(input_id)
    assert store.mark_executed(input_id, "finished")

    monkeypatch.setattr(store, "_owner_alive", lambda _pid, _started: False)
    store.recover_after_restart()
    recovered = store.recoverable_outputs()
    assert [(row.input_id, row.final_text) for row in recovered] == [(input_id, "finished")]
    assert (recovered[0].delivery_owner, recovered[0].turn_outcome) == ("ledger", "completed")

    obligation_id = recovered[0].delivery_obligation_id
    assert obligation_id
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE delivery_obligations SET state='delivered' WHERE obligation_id=?",
            (obligation_id,),
        )
    assert store.mark_completed(input_id)
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
        codex_model="gpt-5.6-sol", reasoning_effort="xhigh",
    )
    input_id, _, _ = store.enqueue_input(binding, "lane-a", _event(binding.source))
    assert store.mark_executing(input_id)

    promoted = store.promote_pending(binding, "real-thread-id")
    assert promoted is not None
    assert promoted.generation == binding.generation
    assert promoted.thread_id == "real-thread-id"
    assert promoted.pending_new is False
    assert promoted.codex_model == "gpt-5.6-sol"
    assert promoted.reasoning_effort == "xhigh"

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


def test_terminal_output_and_send_intent_commit_atomically_and_stay_immutable(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    binding = store.bind("control", _source(), thread_id="thread-a")
    input_id, _, _ = store.enqueue_input(binding, "lane-a", _event(binding.source))
    assert store.mark_executing(input_id)

    obligation_id = store.stage_terminal_output(
        input_id, "the final answer", physical_turn_status="completed",
        output_kind="final_answer",
    )

    assert obligation_id
    with sqlite3.connect(store.path) as conn:
        assert conn.execute(
            "SELECT state, content FROM delivery_obligations WHERE obligation_id=?",
            (obligation_id,),
        ).fetchone() == ("pending", "the final answer")
    assert store.mark_completed(input_id) is False
    with pytest.raises(RuntimeError, match="identity collision"):
        store.stage_terminal_output(
            input_id, "a conflicting late answer", physical_turn_status="completed",
            output_kind="final_answer", recovery=True,
        )


def test_failed_work_and_completed_delivery_are_independent_dimensions(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    binding = store.bind("control", _source(), thread_id="thread-a")
    input_id, _, _ = store.enqueue_input(binding, "lane-a", _event(binding.source))
    assert store.mark_executing(input_id)
    obligation_id = store.stage_terminal_output(
        input_id, "the turn failed", physical_turn_status="failed",
        output_kind="terminal_notice",
    )
    assert obligation_id
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE delivery_obligations SET state='delivered' WHERE obligation_id=?",
            (obligation_id,),
        )
    assert store.mark_completed(input_id)
    with sqlite3.connect(store.path) as conn:
        assert conn.execute(
            """SELECT state, physical_turn_status, logical_input_status,
                      output_kind, delivery_status
               FROM codex_bridge_inputs WHERE input_id=?""",
            (input_id,),
        ).fetchone() == (
            "completed", "failed", "failed", "terminal_notice", "delivered",
        )


def test_finalized_progress_is_not_reopened_by_late_commentary(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    binding = store.bind("control", _source(), thread_id="thread-a")
    first = store.upsert_progress(binding, "logical-1", ["working"])
    assert store.mark_progress_delivered(first, "message-1")
    store.complete_progress(binding, "logical-1")

    late = store.upsert_progress(binding, "logical-1", ["late commentary"])

    assert late.state == "finalized"
    assert "late commentary" not in late.content


def test_prune_never_removes_uncertain_or_continuation_work(tmp_path, monkeypatch):
    store = CodexBridgeStore(tmp_path / "state.db")
    binding = store.bind("control", _source(), thread_id="thread-a")
    uncertain_id, _, _ = store.enqueue_input(
        binding, "lane-a", _event(binding.source, "uncertain"),
    )
    assert store.mark_executing(uncertain_id)
    assert store.mark_submitting(uncertain_id)
    assert store.mark_uncertain(uncertain_id)
    continuation_id, _, _ = store.enqueue_input(
        binding, "lane-a", _event(binding.source, "continuation"),
    )
    assert store.mark_executing(continuation_id)
    assert store.mark_continuation_pending(continuation_id)
    monkeypatch.setattr("gateway.codex_bridge.store.time.time", lambda: 1_000_000_000_000.0)

    store.prune(retention_seconds=1)

    assert store.input_state(uncertain_id) == "uncertain"
    assert store.input_state(continuation_id) == "continuation_pending"
