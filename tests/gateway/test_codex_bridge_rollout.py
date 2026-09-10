"""Real rollout-shape tests for exact ownership and incremental mirroring."""

import json
import os
import sqlite3
from pathlib import Path

import gateway.codex_bridge.rollout as rollout_mod
from gateway.codex_bridge.rollout import RolloutTail, inspect_rollout, resolve_rollout_path


THREAD = "01a0896d-46b8-7422-9a9c-e16635c18eae"
OTHER = "01a08986-bc2d-7b11-9f0d-1bdb2d3d49fd"


def _line(record: dict) -> bytes:
    return (json.dumps(record, ensure_ascii=False) + "\n").encode()


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_line(record) for record in records))


def _meta(thread_id: str) -> dict:
    return {"timestamp": "2026-09-10T00:00:00Z", "type": "session_meta", "payload": {"id": thread_id}}


def test_path_uses_session_meta_not_ambiguous_filename(tmp_path):
    root = tmp_path / "sessions" / "2026" / "09" / "10"
    original = root / f"rollout-old-{THREAD}.jsonl"
    continuation = root / f"rollout-next-{THREAD}_{OTHER}.jsonl"
    unrelated = root / f"rollout-child-{THREAD}_{OTHER}-newer.jsonl"
    _write(original, [_meta(THREAD)])
    _write(continuation, [_meta(THREAD)])
    _write(unrelated, [_meta(OTHER)])
    os.utime(original, (1, 1))
    os.utime(continuation, (2, 2))
    os.utime(unrelated, (3, 3))

    assert resolve_rollout_path(THREAD, codex_home=str(tmp_path)) == continuation.resolve()
    assert resolve_rollout_path(
        THREAD, hinted_path=str(unrelated), codex_home=str(tmp_path),
    ) == continuation.resolve()


def test_path_follows_state_index_when_a_valid_hint_is_an_old_incarnation(tmp_path):
    root = tmp_path / "sessions" / "2026" / "09" / "10"
    old = root / f"rollout-old-{THREAD}.jsonl"
    current = root / f"rollout-current-{THREAD}_{OTHER}.jsonl"
    _write(old, [_meta(THREAD)])
    _write(current, [_meta(THREAD)])
    with sqlite3.connect(tmp_path / "state_5.sqlite") as connection:
        connection.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO threads (id, rollout_path) VALUES (?, ?)",
            (THREAD, str(current)),
        )

    assert resolve_rollout_path(
        THREAD, hinted_path=str(old), codex_home=str(tmp_path),
    ) == current.resolve()


def test_explicit_aborted_turn_continues_provenance_without_new_user_item(tmp_path):
    path = tmp_path / "sessions" / f"rollout-{THREAD}.jsonl"
    records = [
        _meta(THREAD),
        {"timestamp": "2026-09-10T00:00:01Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "turn-one"}},
        {"timestamp": "2026-09-10T00:00:02Z", "type": "response_item",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "do all fixes"}],
                     "internal_chat_message_metadata_passthrough": {"turn_id": "turn-one"}}},
        {"timestamp": "2026-09-10T00:00:02Z", "type": "event_msg",
         "payload": {"type": "item_completed", "turn_id": "turn-one",
                     "item": {"type": "UserMessage", "client_id": "desktop-client"}}},
        {"timestamp": "2026-09-10T00:00:03Z", "type": "event_msg",
         "payload": {"type": "turn_aborted", "turn_id": "turn-one"}},
        {"timestamp": "2026-09-10T00:00:04Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "turn-two"}},
        {"timestamp": "2026-09-10T00:00:05Z", "type": "response_item",
         "payload": {"type": "message", "role": "assistant", "phase": "commentary",
                     "content": [{"type": "output_text", "text": "still working"}],
                     "internal_chat_message_metadata_passthrough": {"turn_id": "turn-two"}}},
        {"timestamp": "2026-09-10T00:00:06Z", "type": "event_msg",
         "payload": {"type": "task_complete", "turn_id": "turn-two",
                     "last_agent_message": "all done"}},
    ]
    _write(path, records)

    events, _, _ = RolloutTail(THREAD, path).scan()
    assert [(event.kind, event.text, event.client_id) for event in events] == [
        ("user", "do all fixes", None),
        ("commentary", "still working", "desktop-client"),
        ("final", "all done", "desktop-client"),
    ]


def test_stale_unrelated_turn_does_not_inherit_aborted_user(tmp_path):
    path = tmp_path / "sessions" / f"rollout-{THREAD}.jsonl"
    records = [
        _meta(THREAD),
        {"timestamp": "2026-09-10T00:00:01Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "turn-one"}},
        {"timestamp": "2026-09-10T00:00:02Z", "type": "event_msg",
         "payload": {"type": "user_message", "turn_id": "turn-one", "message": "old work"}},
        {"timestamp": "2026-09-10T00:00:03Z", "type": "event_msg",
         "payload": {"type": "turn_aborted", "turn_id": "turn-one"}},
        {"timestamp": "2026-09-10T00:01:03Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "unrelated"}},
        {"timestamp": "2026-09-10T00:01:04Z", "type": "event_msg",
         "payload": {"type": "task_complete", "turn_id": "unrelated", "last_agent_message": "secret"}},
    ]
    _write(path, records)

    events, _, _ = RolloutTail(THREAD, path).scan()
    assert [(event.kind, event.text) for event in events] == [
        ("user", "old work"),
        ("error", "Codex 작업이 중단되었습니다."),
    ]


def test_structured_environment_context_never_authorizes_output_mirroring(tmp_path):
    path = tmp_path / "sessions" / f"rollout-{THREAD}.jsonl"
    records = [
        _meta(THREAD),
        {"timestamp": "2026-09-10T00:00:01Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "environment-only"}},
        {"timestamp": "2026-09-10T00:00:02Z", "type": "response_item",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "looks like ordinary text"}],
                     "internal_chat_message_metadata_passthrough": {
                         "turn_id": "environment-only",
                         "content_item_kinds": ["environments.environment_context"],
                     }}},
        {"timestamp": "2026-09-10T00:00:03Z", "type": "event_msg",
         "payload": {"type": "task_complete", "turn_id": "environment-only",
                     "last_agent_message": "must stay private"}},
    ]
    _write(path, records)

    events, _, _ = RolloutTail(THREAD, path).scan()

    assert events == []
    snapshot = inspect_rollout(THREAD, hinted_path=str(path), codex_home=str(tmp_path))
    assert snapshot is not None
    assert snapshot.latest_final_text == ""


def test_partial_line_retries_and_malformed_line_is_quarantined_after_three_scans(tmp_path):
    path = tmp_path / "sessions" / f"rollout-{THREAD}.jsonl"
    prefix = [_meta(THREAD), {
        "timestamp": "2026-09-10T00:00:01Z", "type": "event_msg",
        "payload": {"type": "task_started", "turn_id": "turn-one"},
    }, {
        "timestamp": "2026-09-10T00:00:02Z", "type": "event_msg",
        "payload": {"type": "user_message", "turn_id": "turn-one", "message": "hello"},
    }]
    _write(path, prefix)
    with path.open("ab") as handle:
        handle.write(b'{"broken":\n')
        handle.write(_line({
            "timestamp": "2026-09-10T00:00:03Z", "type": "event_msg",
            "payload": {"type": "task_complete", "turn_id": "turn-one", "last_agent_message": "done"},
        }))

    tail = RolloutTail(THREAD, path)
    first, offset, _ = tail.scan()
    tail.offset = offset
    assert [event.kind for event in first] == ["user"]
    second, offset, _ = tail.scan()
    tail.offset = offset
    assert second == []
    third, offset, _ = tail.scan()
    assert [(event.kind, event.text) for event in third] == [
        ("error", "Codex 기록 한 줄이 손상되어 건너뛰었습니다."),
        ("final", "done"),
    ]
    assert offset == path.stat().st_size


def test_inspect_reports_active_continuation_start(tmp_path):
    path = tmp_path / "sessions" / f"rollout-{THREAD}_{OTHER}.jsonl"
    records = [
        _meta(THREAD),
        {"timestamp": "2026-09-10T00:00:01Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "active-turn"}},
        {"timestamp": "2026-09-10T00:00:02Z", "type": "event_msg",
         "payload": {"type": "user_message", "turn_id": "active-turn", "message": "go"}},
    ]
    _write(path, records)

    snapshot = inspect_rollout(THREAD, hinted_path=str(path), codex_home=str(tmp_path))
    assert snapshot is not None
    assert snapshot.path == str(path.resolve())
    assert snapshot.active_turn_id == "active-turn"
    assert snapshot.active_start_offset is not None


def test_initial_inspection_streams_beyond_bootstrap_window(tmp_path, monkeypatch):
    path = tmp_path / "sessions" / f"rollout-{THREAD}.jsonl"
    records = [
        _meta(THREAD),
        {"timestamp": "2026-09-10T00:00:01Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "finished"}},
        {"timestamp": "2026-09-10T00:00:02Z", "type": "event_msg",
         "payload": {"type": "user_message", "turn_id": "finished", "message": "old"}},
        {"timestamp": "2026-09-10T00:00:03Z", "type": "event_msg",
         "payload": {"type": "task_complete", "turn_id": "finished",
                     "last_agent_message": "last complete answer"}},
        {"timestamp": "2026-09-10T00:00:04Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "active-turn"}},
        {"timestamp": "2026-09-10T00:00:05Z", "type": "event_msg",
         "payload": {"type": "user_message", "turn_id": "active-turn", "message": "current"}},
        *[
            {"timestamp": "2026-09-10T00:00:06Z", "type": "event_msg",
             "payload": {"type": "agent_message", "turn_id": "active-turn",
                         "phase": "commentary", "message": f"progress {index}"}}
            for index in range(20)
        ],
    ]
    _write(path, records)
    monkeypatch.setattr(rollout_mod, "_BOOTSTRAP_BYTES", 64)

    snapshot = inspect_rollout(THREAD, hinted_path=str(path), codex_home=str(tmp_path))

    assert snapshot is not None
    assert snapshot.latest_final_text == "last complete answer"
    assert snapshot.active_turn_id == "active-turn"
    assert snapshot.active_start_offset is not None
