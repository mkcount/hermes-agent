"""Pre-guard routing tests for bound Telegram Codex lanes."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.codex_bridge import handoff as handoff_mod
from gateway.codex_bridge import rollout as rollout_mod
from gateway.codex_bridge.catalog import CodexProjectSummary, CodexThreadSummary
from gateway.codex_bridge.mixin import GatewayCodexBridgeMixin
from gateway.codex_bridge.rollout import RolloutEvent
from gateway.codex_bridge.store import CodexBridgeStore
from gateway.config import Platform
from gateway.platforms.base import SessionRouteRejected
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, build_session_key


class _Adapter:
    @staticmethod
    def _event_session_key(event):
        return build_session_key(event.source)

    @staticmethod
    def _is_sender_authorized(*_args, **_kwargs):
        return True


class _SessionStore:
    @staticmethod
    def lookup_by_session_key(_key):
        return None


class _Bridge(GatewayCodexBridgeMixin):
    def __init__(self, store, profile_home=None):
        self.store = store
        self.profile_home = profile_home
        self.session_store = _SessionStore()
        self.async_session_store = SimpleNamespace(
            get_or_create_session=AsyncMock(),
            get_or_create_isolated_session=AsyncMock(),
        )
        self._codex_bridge_binding_locks = {}
        self._codex_bridge_tails = {}

    @staticmethod
    def _session_key_for_source(source):
        return build_session_key(source)

    def _codex_bridge_store_for_source(self, _source):
        return self.store

    def _resolve_profile_home_for_source(self, _source):
        return self.profile_home

    @staticmethod
    def _thread_metadata_for_source(source):
        return {"message_thread_id": source.thread_id} if source.thread_id else None

    @staticmethod
    def _peek_session_state(_session_key):
        return None

    @staticmethod
    def _evict_cached_agent(_session_key):
        return None

    @staticmethod
    def _is_session_running(_session_key):
        return False


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="1001", user_id="42", chat_type="dm",
    )


@pytest.mark.asyncio
async def test_bound_message_is_persisted_but_not_executable_before_runner_gates(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    store.bind(build_session_key(source), source, thread_id="thread-123")
    bridge = _Bridge(store)
    event = MessageEvent(text="continue", source=source, message_id="77")

    routed = await bridge._resolve_codex_bridge_route(_Adapter(), event)

    assert routed.source.trusted_local_lane
    assert routed.metadata["codex_bridge_thread_id"] == "thread-123"
    assert store.input_state(routed.metadata["codex_bridge_input_id"]) == "routed"
    assert build_session_key(routed.source) != build_session_key(source)
    bridge.async_session_store.get_or_create_isolated_session.assert_awaited_once_with(
        routed.source,
    )


@pytest.mark.asyncio
async def test_terminal_route_rejection_cannot_be_released_for_retry(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    store.bind(build_session_key(source), source, thread_id="thread-123")
    bridge = _Bridge(store)
    routed = await bridge._resolve_codex_bridge_route(
        _Adapter(), MessageEvent(text="continue", source=source, message_id="lease-timeout"),
    )
    input_id = routed.metadata["codex_bridge_input_id"]
    assert bridge._codex_bridge_begin_input(routed) is None
    assert store.input_state(input_id) == "executing"

    bridge._codex_bridge_cancel_input(routed, "turn lease timeout")
    bridge._codex_bridge_release_input(routed, "outer finally")

    assert store.input_state(input_id) == "cancelled"
    assert store.recoverable_inputs() == []


@pytest.mark.asyncio
async def test_duplicate_platform_message_fails_closed(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    store.bind(build_session_key(source), source, thread_id="thread-123")
    bridge = _Bridge(store)
    event = MessageEvent(text="continue", source=source, message_id="77")
    await bridge._resolve_codex_bridge_route(_Adapter(), event)

    with pytest.raises(SessionRouteRejected):
        await bridge._resolve_codex_bridge_route(_Adapter(), event)


@pytest.mark.asyncio
async def test_unauthorized_message_never_creates_a_durable_bridge_input(tmp_path):
    class UnauthorizedAdapter(_Adapter):
        @staticmethod
        def _is_sender_authorized(*_args, **_kwargs):
            return False

    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    store.bind(build_session_key(source), source, thread_id="thread-123")
    bridge = _Bridge(store)

    routed = await bridge._resolve_codex_bridge_route(
        UnauthorizedAdapter(), MessageEvent(text="continue", source=source, message_id="unauthorized"),
    )

    assert routed.source.trusted_local_lane is None
    assert store.recoverable_inputs() == []
    assert store.input_state(store.input_id(build_session_key(source), routed)) is None


@pytest.mark.asyncio
async def test_selection_command_stays_on_control_lane(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    store.bind(build_session_key(source), source, thread_id="thread-123")
    bridge = _Bridge(store)

    routed = await bridge._resolve_codex_bridge_route(
        _Adapter(), MessageEvent(text="/코덱스세션", source=source, message_id="78"),
    )
    assert routed.source.trusted_local_lane is None
    assert "codex_bridge_input_id" not in routed.metadata

    model_routed = await bridge._resolve_codex_bridge_route(
        _Adapter(), MessageEvent(text="/codex_model", source=source, message_id="79"),
    )
    assert model_routed.source.trusted_local_lane is None
    assert "codex_bridge_input_id" not in model_routed.metadata


@pytest.mark.asyncio
async def test_codex_model_picker_commits_model_and_reasoning_together(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    binding = store.bind(
        build_session_key(source), source, thread_id="thread-123",
        codex_model="gpt-5.6-sol", reasoning_effort="xhigh",
    )
    bridge = _Bridge(store)
    pickers = []

    async def _send_picker(event, session_key, title, choices, callback):
        pickers.append({
            "event": event, "session_key": session_key, "title": title,
            "choices": choices, "callback": callback,
        })
        return True

    bridge._try_send_choice_picker = _send_picker
    answer = await bridge._handle_codex_model_command(
        MessageEvent(text="/codex_model", source=source, message_id="model-1"),
    )

    assert answer is None
    assert [choice["label"] for choice in pickers[0]["choices"]] == [
        "5.6 S", "5.6 T", "5.6 L", "6.0 A",
    ]
    assert "현재 설정: *5.6 S · Sol / XHigh (텔레그램 지정)*" in pickers[0]["title"]
    first_result = await pickers[0]["callback"]("1001", "gpt-5.6-terra")
    assert "5.6 T" in first_result
    # Model selection alone is not a partial commit.
    assert store.get_binding(binding.control_session_key).codex_model == "gpt-5.6-sol"
    assert [choice["value"] for choice in pickers[1]["choices"]] == [
        "medium", "high", "xhigh",
    ]
    assert "현재 설정: *5.6 S · Sol / XHigh (텔레그램 지정)*" in pickers[1]["title"]
    assert "새 모델: *5.6 T · Terra*" in pickers[1]["title"]

    final_result = await pickers[1]["callback"]("1001", "high")
    updated = store.get_binding(binding.control_session_key)
    assert "Codex 설정 완료" in final_result
    assert updated.codex_model == "gpt-5.6-terra"
    assert updated.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_codex_model_picker_shows_inherited_desktop_settings(tmp_path, monkeypatch):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    store.bind(build_session_key(source), source, thread_id="thread-123")
    bridge = _Bridge(store)
    pickers = []
    monkeypatch.setattr(
        "gateway.codex_bridge.mixin.inspect_rollout",
        lambda *_args, **_kwargs: SimpleNamespace(
            latest_model="gpt-5.6-sol", latest_reasoning_effort="xhigh",
        ),
    )

    async def _send_picker(_event, _session_key, title, choices, callback):
        pickers.append({"title": title, "choices": choices, "callback": callback})
        return True

    bridge._try_send_choice_picker = _send_picker
    answer = await bridge._handle_codex_model_command(
        MessageEvent(text="/codex_model", source=source, message_id="model-inherited"),
    )

    assert answer is None
    assert "현재 설정: *5.6 S · Sol / XHigh (데스크톱 세션 계승)*" in pickers[0]["title"]
    assert next(
        choice for choice in pickers[0]["choices"] if choice["value"] == "gpt-5.6-sol"
    )["is_current"] is True

    await pickers[0]["callback"]("1001", "gpt-5.6-sol")
    assert next(
        choice for choice in pickers[1]["choices"] if choice["value"] == "xhigh"
    )["is_current"] is True


@pytest.mark.asyncio
async def test_codex_model_picker_rejects_selection_after_binding_changes(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    old = store.bind(build_session_key(source), source, thread_id="thread-old")
    bridge = _Bridge(store)
    pickers = []

    async def _send_picker(_event, _session_key, _title, _choices, callback):
        pickers.append(callback)
        return True

    bridge._try_send_choice_picker = _send_picker
    await bridge._handle_codex_model_command(
        MessageEvent(text="/codex_model", source=source, message_id="model-stale"),
    )
    await pickers[0]("1001", "gpt-6-astra")
    store.bind(build_session_key(source), source, thread_id="thread-new")

    result = await pickers[1]("1001", "xhigh")
    current = store.get_binding(old.control_session_key)
    assert "세션이 바뀌었습니다" in result
    assert current.thread_id == "thread-new"
    assert current.codex_model is None
    assert current.reasoning_effort is None


@pytest.mark.asyncio
async def test_ns_reserves_explicit_sol_xhigh_defaults(tmp_path, monkeypatch):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    bridge = _Bridge(store)
    monkeypatch.setattr(
        "gateway.codex_bridge.mixin.list_recent_projects",
        lambda **_kwargs: [CodexProjectSummary(cwd="/project", name="Project", updated_at=1)],
    )

    answer = await bridge._handle_ns_command(
        MessageEvent(text="/ns 1", source=source, message_id="ns-default"),
    )

    binding = store.get_binding(build_session_key(source))
    assert "5.6 S · Sol / XHigh" in answer
    assert binding.pending_new is True
    assert binding.codex_model == "gpt-5.6-sol"
    assert binding.reasoning_effort == "xhigh"


@pytest.mark.asyncio
async def test_codex_session_status_reports_active_turn_and_pending_progress(tmp_path, monkeypatch):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    binding = store.bind(
        build_session_key(source), source, thread_id="thread-123", cwd="/project",
        rollout_path="/rollout.jsonl", cursor_device=1, cursor_inode=2, cursor_offset=75,
    )
    progress = store.upsert_progress(binding, "turn-1", ["working"])
    store.mark_progress_failed(progress, "network unavailable")
    bridge = _Bridge(store)
    monkeypatch.setattr(
        "gateway.codex_bridge.mixin.inspect_rollout",
        lambda *_args, **_kwargs: SimpleNamespace(
            path="/rollout.jsonl", device=1, inode=2, size=100,
            active_turn_id="turn-1", active_start_offset=20,
        ),
    )
    monkeypatch.setattr(
        "gateway.codex_bridge.mixin.list_recent_threads",
        lambda **_kwargs: pytest.fail("status must not require the app-server thread list"),
    )

    answer = await bridge._handle_codex_session_command(
        MessageEvent(text="/codex_session status", source=source, message_id="status-1"),
    )

    assert "thread-123" in answer
    assert "턴 진행 중" in answer
    assert "25 bytes" in answer
    assert "대기 1" in answer
    assert "network unavailable" in answer


def test_surviving_bound_prompt_without_durable_id_fails_closed(tmp_path):
    bridge = _Bridge(CodexBridgeStore(tmp_path / "state.db"))
    event = MessageEvent(
        text="rewritten prompt",
        source=_source(),
        metadata={"codex_bridge_control_key": "control"},
    )

    error = bridge._codex_bridge_begin_input(event)

    assert "내구 실행 ID" in error
    assert "실행하지 않았습니다" in error


@pytest.mark.asyncio
async def test_reselecting_current_thread_does_not_rotate_or_cancel_work(tmp_path, monkeypatch):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    current = store.bind(build_session_key(source), source, thread_id="thread-123", cwd="/project")
    input_id, _, _ = store.enqueue_input(
        current, "lane", MessageEvent(text="continue", source=source, message_id="current-work"),
    )
    old_progress = store.upsert_progress(current, "old-turn", ["old location"])
    store.mark_progress_delivered(old_progress, "old-message")
    bridge = _Bridge(store)
    monkeypatch.setattr(
        "gateway.codex_bridge.mixin.inspect_rollout",
        lambda *_args, **_kwargs: SimpleNamespace(
            path="/rollout.jsonl", device=1, inode=2, size=99,
            active_turn_id=None, active_start_offset=None,
            latest_final_text="repeat this final",
        ),
    )

    answer = await bridge._codex_bridge_store_binding(
        source,
        CodexThreadSummary(
            thread_id="thread-123", title="Current", cwd="/project",
            updated_at=1, status="idle",
        ),
    )

    assert "다시 연결됨" in answer
    assert "repeat this final" in answer
    refreshed = store.get_binding(build_session_key(source))
    assert refreshed.generation == current.generation
    assert refreshed.cursor_offset == 99
    assert store.list_progress(refreshed) == []
    assert store.input_state(input_id) == "routed"


@pytest.mark.asyncio
async def test_selecting_another_thread_interrupts_the_running_previous_lane(
    tmp_path, monkeypatch,
):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    previous = store.bind(
        build_session_key(source), source, thread_id="thread-old", cwd="/old-project",
    )
    input_id, _, _ = store.enqueue_input(
        previous, "old-lane", MessageEvent(text="keep working", source=source, message_id="old"),
    )
    assert store.mark_executing(input_id)
    bridge = _Bridge(store)
    previous_lane_key = bridge._codex_bridge_lane_key(source, previous)
    bridge._is_session_running = lambda key: key == previous_lane_key
    bridge._interrupt_and_clear_session = AsyncMock()
    monkeypatch.setattr("gateway.codex_bridge.mixin.inspect_rollout", lambda *_args, **_kwargs: None)

    answer = await bridge._codex_bridge_store_binding(
        source,
        CodexThreadSummary(
            thread_id="thread-new", title="New target", cwd="/new-project",
            updated_at=2, status="idle",
        ),
    )

    assert "이전 연결" in answer
    assert store.input_state(input_id) == "cancelled"
    bridge._interrupt_and_clear_session.assert_awaited_once()
    call = bridge._interrupt_and_clear_session.await_args
    assert call.args[0] == previous_lane_key
    assert call.args[1].trusted_local_lane.endswith(f":{previous.generation}")
    assert call.kwargs == {
        "interrupt_reason": "Stop requested",
        "invalidation_reason": "codex_binding_changed",
    }


@pytest.mark.asyncio
async def test_selecting_active_thread_replays_all_commentary_as_fresh_rows(tmp_path, monkeypatch):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    current = store.bind(build_session_key(source), source, thread_id="thread-123")
    bridge = _Bridge(store)
    snapshot = SimpleNamespace(
        path="/rollout.jsonl", device=1, inode=2, size=120,
        active_turn_id="active-turn", active_start_offset=40,
        latest_final_text="previous final",
    )
    repeated_id = "same-persisted-event"
    commentary = [
        RolloutEvent(
            event_id=repeated_id, turn_id="active-turn", kind="commentary",
            text="same report", offset=70,
        ),
        RolloutEvent(
            event_id=repeated_id, turn_id="active-turn", kind="commentary",
            text="same report", offset=70,
        ),
        RolloutEvent(
            event_id="later-event", turn_id="active-turn", kind="commentary",
            text="later report", offset=100,
        ),
    ]
    monkeypatch.setattr("gateway.codex_bridge.mixin.inspect_rollout", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(
        "gateway.codex_bridge.mixin.collect_active_commentary",
        lambda *_args, **_kwargs: commentary,
    )

    answer = await bridge._codex_bridge_store_binding(
        source,
        CodexThreadSummary(
            thread_id="thread-123", title="Current", cwd="/project",
            updated_at=1, status="active",
        ),
    )

    refreshed = store.get_binding(build_session_key(source))
    progress = store.list_progress(refreshed, pending_only=True)
    assert "중간보고 3개" in answer
    assert "previous final" in answer
    assert refreshed.generation == current.generation
    assert refreshed.cursor_offset == snapshot.size
    assert len(progress) == 3
    assert [row.segments for row in progress].count(("same report",)) == 2
    assert [row.segments for row in progress].count(("later report",)) == 1
    assert all(row.message_id is None for row in progress)


@pytest.mark.asyncio
async def test_rollout_incarnation_change_starts_with_a_file_local_cursor(tmp_path, monkeypatch):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    binding = store.bind(
        build_session_key(source),
        source,
        thread_id="thread-123",
        rollout_path="/old.jsonl",
        cursor_device=11,
        cursor_inode=22,
        cursor_offset=33,
    )
    bridge = _Bridge(store)
    bridge._adapter_for_source = lambda _source: object()
    current = tmp_path / "current.jsonl"
    current.write_text("", encoding="utf-8")
    captured = {}

    class Tail:
        def __init__(self, thread_id, path, **kwargs):
            captured.update(thread_id=thread_id, path=path, **kwargs)
            self.path = Path(path)
            self.offset = kwargs["offset"]

        def scan(self):
            return [], 0, current.stat()

    monkeypatch.setattr("gateway.codex_bridge.mixin.resolve_rollout_path", lambda *_args, **_kwargs: current)
    monkeypatch.setattr("gateway.codex_bridge.mixin.RolloutTail", Tail)

    await bridge._codex_bridge_poll_binding_locked(store, binding)

    assert captured == {
        "thread_id": "thread-123",
        "path": current,
        "device": None,
        "inode": None,
        "offset": 0,
        "handoff_graph": handoff_mod.CodexHandoffGraph({}, {}),
    }


@pytest.mark.asyncio
async def test_ambiguous_turn_submission_is_not_replayed(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    store.bind(build_session_key(source), source, thread_id="thread-123")
    bridge = _Bridge(store)
    routed = await bridge._resolve_codex_bridge_route(
        _Adapter(), MessageEvent(text="continue", source=source, message_id="79"),
    )
    input_id = routed.metadata["codex_bridge_input_id"]
    assert bridge._codex_bridge_begin_input(routed) is None
    assert store.mark_submitting(input_id)
    routed._codex_bridge_agent_result = {
        "completed": False,
        "codex_should_retire": True,
        "error": "turn/start timed out",
    }

    result = await bridge._codex_bridge_finalize_input(routed, "lane", "timeout", 1)
    bridge._codex_bridge_release_input(routed, "finally")

    assert "자동 재실행하지 않습니다" in result
    assert store.input_state(input_id) == "uncertain"
    assert store.recoverable_inputs() == []


@pytest.mark.asyncio
async def test_transport_rejected_turn_start_is_requeued_on_a_fresh_connection(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    store.bind(build_session_key(source), source, thread_id="thread-123")
    bridge = _Bridge(store)
    routed = await bridge._resolve_codex_bridge_route(
        _Adapter(), MessageEvent(text="continue", source=source, message_id="safe-retry"),
    )
    input_id = routed.metadata["codex_bridge_input_id"]
    assert bridge._codex_bridge_begin_input(routed) is None
    assert store.mark_submitting(input_id)
    routed._codex_bridge_agent_result = {
        "completed": False,
        "codex_should_retire": True,
        "codex_submission_not_admitted": True,
        "error": "turn/start frame was not admitted",
    }

    result = await bridge._codex_bridge_finalize_input(routed, "lane", "", 1)
    bridge._codex_bridge_release_input(routed, "finally")

    assert "자동으로 다시 시도합니다" in result
    assert store.input_state(input_id) == "pending"
    assert [item.input_id for item in store.recoverable_inputs()] == [input_id]


@pytest.mark.asyncio
async def test_second_transport_rejection_stops_retrying_and_requests_resend(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    store.bind(build_session_key(source), source, thread_id="thread-123")
    bridge = _Bridge(store)
    bridge._adapter_for_source = lambda _source: SimpleNamespace(
        _owner_profile=None,
        register_post_delivery_callback=lambda *_args, **_kwargs: None,
    )
    routed = await bridge._resolve_codex_bridge_route(
        _Adapter(), MessageEvent(text="continue", source=source, message_id="bounded-retry"),
    )
    input_id = routed.metadata["codex_bridge_input_id"]
    assert bridge._codex_bridge_begin_input(routed) is None
    assert store.mark_submitting(input_id)
    assert store.retry_unaccepted_submission(input_id, "first rejection")
    assert store.mark_executing(input_id)
    assert store.mark_submitting(input_id)
    routed._codex_bridge_agent_result = {
        "completed": False,
        "codex_should_retire": True,
        "codex_submission_not_admitted": True,
        "error": "second turn/start frame was not admitted",
    }

    result = await bridge._codex_bridge_finalize_input(routed, "lane", "", 2)

    assert "같은 메시지를 다시 보내 주세요" in result
    assert store.input_state(input_id) == "executed"
    assert store.recoverable_inputs() == []


@pytest.mark.asyncio
async def test_unconfirmed_handoff_intent_does_not_abandon_an_uncertain_turn(
    tmp_path, monkeypatch,
):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    store.bind(build_session_key(source), source, thread_id="thread-123")
    bridge = _Bridge(store)
    routed = await bridge._resolve_codex_bridge_route(
        _Adapter(), MessageEvent(text="continue", source=source, message_id="handoff-race"),
    )
    input_id = routed.metadata["codex_bridge_input_id"]
    assert bridge._codex_bridge_begin_input(routed) is None
    assert store.mark_submitting(input_id)
    assert store.mark_running(input_id, "turn-old")
    routed._codex_bridge_agent_result = {
        "completed": False,
        "interrupted": False,
        "codex_should_retire": True,
        "error": "app-server disconnected before an interrupt was observed",
        "codex_thread_id": "thread-123",
        "codex_turn_id": "turn-old",
    }
    handoff_path = tmp_path / "profile-monitor-state.json"
    handoff_path.write_text(json.dumps({
        "codex_turn_handoffs": {
            "thread-123:turn-old": {
                "thread_id": "thread-123",
                "interrupted_turn_id": "turn-old",
                "continued_turn_id": "",
                "status": "interrupting",
                "reason": "",
            },
        },
    }), encoding="utf-8")
    monkeypatch.setattr(handoff_mod, "_STATE_PATH", handoff_path)

    result = await bridge._codex_bridge_finalize_input(routed, "lane", "failure", 1)

    assert "자동 재실행하지 않습니다" in result
    assert store.input_state(input_id) == "uncertain"


@pytest.mark.asyncio
async def test_relogin_handoff_streams_resumed_progress_and_captures_one_final(
    tmp_path, monkeypatch,
):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    current = tmp_path / "rollout.jsonl"
    current.write_text("", encoding="utf-8")
    binding = store.bind(
        build_session_key(source), source, thread_id="thread-123", rollout_path=str(current),
    )
    bridge = _Bridge(store)
    routed = await bridge._resolve_codex_bridge_route(
        _Adapter(), MessageEvent(text="continue", source=source, message_id="handoff-input"),
    )
    input_id = routed.metadata["codex_bridge_input_id"]
    assert bridge._codex_bridge_begin_input(routed) is None
    assert store.mark_submitting(input_id)
    assert store.mark_running(input_id, "turn-old")
    routed._codex_bridge_agent_result = {
        "completed": False,
        "error": "Codex 작업이 외부 요인으로 중단되어 완료되지 않았습니다.",
        "codex_thread_id": "thread-123",
        "codex_turn_id": "turn-old",
    }
    handoff_path = tmp_path / "profile-monitor-state.json"
    handoff_path.write_text(json.dumps({
        "codex_turn_handoffs": {
            "thread-123:turn-old": {
                "thread_id": "thread-123",
                "interrupted_turn_id": "turn-old",
                "continued_turn_id": "turn-new",
                "status": "continued",
                "reason": "continued",
            },
        },
    }), encoding="utf-8")
    monkeypatch.setattr(handoff_mod, "_STATE_PATH", handoff_path)

    assert await bridge._codex_bridge_finalize_input(routed, "lane", "failure", 1) is None
    assert store.input_state(input_id) == "continuation_pending"

    events = [
        RolloutEvent(
            event_id="unrelated-commentary", turn_id="turn-other", kind="commentary",
            text="must stay local", offset=5, client_id=input_id,
        ),
        RolloutEvent(
            event_id="continued-commentary", turn_id="turn-new", kind="commentary",
            text="still working", offset=10, client_id=input_id,
        ),
        RolloutEvent(
            event_id="continued-final", turn_id="turn-new", kind="final",
            text="all done", offset=20, client_id=input_id,
        ),
    ]

    class Tail:
        def __init__(self, _thread_id, path, **kwargs):
            self.path = Path(path)
            self.offset = kwargs["offset"]

        def scan(self):
            return events, 20, current.stat()

    adapter = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="501")),
        edit_message=AsyncMock(),
    )
    bridge._adapter_for_source = lambda _source: adapter
    monkeypatch.setattr(
        "gateway.codex_bridge.mixin.resolve_rollout_path",
        lambda *_args, **_kwargs: current,
    )
    monkeypatch.setattr("gateway.codex_bridge.mixin.RolloutTail", Tail)

    await bridge._codex_bridge_poll_binding_locked(store, binding)

    assert adapter.send.await_count == 1, adapter.send.await_args_list
    assert "still working" in adapter.send.await_args.args[1]
    outputs = store.recoverable_outputs()
    assert [(item.input_id, item.final_text) for item in outputs] == [(input_id, "all done")]


@pytest.mark.asyncio
async def test_completed_interruption_transfers_delivery_to_a_resumed_physical_turn(
    tmp_path, monkeypatch,
):
    """Delivering an interruption notice completes that delivery attempt, not
    the logical work. A later Codex turn inheriting its client id is watcher-owned.
    """
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    thread_id = "01a00000-0000-7000-8000-000000000001"
    current = (
        tmp_path / "sessions" / "2026" / "09" / "12"
        / f"rollout-{thread_id}.jsonl"
    )
    current.parent.mkdir(parents=True)
    current.write_text("", encoding="utf-8")
    binding = store.bind(
        build_session_key(source), source, thread_id=thread_id, rollout_path=str(current),
    )
    bridge = _Bridge(store)
    routed = await bridge._resolve_codex_bridge_route(
        _Adapter(), MessageEvent(text="inspect everything", source=source, message_id="watchdog-input"),
    )
    input_id = routed.metadata["codex_bridge_input_id"]
    assert bridge._codex_bridge_begin_input(routed) is None
    assert store.mark_submitting(input_id)
    assert store.mark_running(input_id, "turn-interrupted")
    assert store.mark_executed(
        input_id, "Turn aborted by liveness watchdog", turn_outcome="interrupted",
    )
    records = [
        {"timestamp": "2026-09-12T05:16:00Z", "type": "session_meta",
         "payload": {"id": thread_id}},
        {"timestamp": "2026-09-12T05:16:01Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "turn-interrupted"}},
        {"timestamp": "2026-09-12T05:16:02Z", "type": "response_item",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "inspect everything"}],
                     "internal_chat_message_metadata_passthrough": {
                         "turn_id": "turn-interrupted",
                     }}},
        {"timestamp": "2026-09-12T05:16:03Z", "type": "event_msg",
         "payload": {"type": "item_completed", "turn_id": "turn-interrupted",
                     "item": {"type": "UserMessage", "client_id": input_id}}},
        {"timestamp": "2026-09-12T05:26:16Z", "type": "event_msg",
         "payload": {"type": "turn_aborted", "turn_id": "turn-interrupted",
                     "reason": "interrupted"}},
        {"timestamp": "2026-09-12T05:33:03Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "turn-resumed"}},
        {"timestamp": "2026-09-12T05:35:41Z", "type": "response_item",
         "payload": {"type": "message", "role": "assistant", "phase": "commentary",
                     "content": [{"type": "output_text", "text": "working again"}],
                     "internal_chat_message_metadata_passthrough": {
                         "turn_id": "turn-resumed",
                     }}},
        {"timestamp": "2026-09-12T05:38:26Z", "type": "event_msg",
         "payload": {"type": "task_complete", "turn_id": "turn-resumed",
                     "last_agent_message": "finished after resume"}},
    ]
    current.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )

    adapter = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="502")),
        edit_message=AsyncMock(),
    )
    bridge._adapter_for_source = lambda _source: adapter
    monkeypatch.setattr(
        "gateway.codex_bridge.mixin.resolve_rollout_path",
        lambda thread, hinted_path=None: rollout_mod.resolve_rollout_path(
            thread, hinted_path=hinted_path, codex_home=str(tmp_path),
        ),
    )

    await bridge._codex_bridge_poll_binding_locked(store, binding)

    # The old turn's interruption notice is still ledger-owned. The watcher
    # must wait without acknowledging the replacement turn's rollout bytes.
    waiting = store.get_binding(binding.control_session_key)
    assert waiting is not None
    assert waiting.cursor_offset < current.stat().st_size
    adapter.send.assert_not_awaited()
    assert store.recoverable_outputs() == []

    with sqlite3.connect(store.path) as conn:
        obligation_id = conn.execute(
            "SELECT delivery_obligation_id FROM codex_bridge_inputs WHERE input_id=?",
            (input_id,),
        ).fetchone()[0]
        conn.execute(
            "UPDATE delivery_obligations SET state='delivered' WHERE obligation_id=?",
            (obligation_id,),
        )
    assert store.mark_completed(input_id)
    resumed = store.get_binding(binding.control_session_key)
    assert resumed is not None
    await bridge._codex_bridge_poll_binding_locked(store, resumed)

    assert adapter.send.await_count == 1, adapter.send.await_args_list
    assert "working again" in adapter.send.await_args.args[1]
    outputs = store.recoverable_outputs()
    assert [(item.input_id, item.final_text) for item in outputs] == [
        (input_id, "finished after resume"),
    ]


@pytest.mark.asyncio
async def test_runner_configures_full_access_and_durable_turn_callbacks(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    binding = store.bind(
        build_session_key(source), source, thread_id="thread-123", cwd="/project",
        codex_model="gpt-6-astra", reasoning_effort="high",
    )
    bridge = _Bridge(store)
    routed = await bridge._resolve_codex_bridge_route(
        _Adapter(), MessageEvent(text="continue", source=source, message_id="80"),
    )
    assert bridge._codex_bridge_begin_input(routed) is None
    input_id = routed.metadata["codex_bridge_input_id"]
    agent = SimpleNamespace()
    ctx = SimpleNamespace(
        source=routed.source,
        session_key=build_session_key(routed.source),
        codex_bridge_control_key=binding.control_session_key,
        codex_bridge_generation=binding.generation,
        codex_bridge_input_id=input_id,
    )

    bridge._configure_codex_bridge_agent(agent, ctx)
    agent._codex_turn_starting_callback("thread-123", input_id)
    agent._codex_turn_started_callback("thread-123", "turn-456")

    assert agent.api_mode == "codex_app_server"
    assert agent._gateway_codex_full_access is True
    assert agent._codex_resume_thread_id == "thread-123"
    assert agent._codex_resume_active_turn_mode == "queue"
    assert agent._codex_model_override == "gpt-6-astra"
    assert agent._codex_reasoning_effort_override == "high"
    assert agent.session_cwd == "/project"
    assert store.input_state(input_id) == "running"


@pytest.mark.asyncio
async def test_background_ledger_write_uses_binding_profile(tmp_path):
    from gateway.delivery_ledger import ensure_obligation

    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "work"
    default_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)
    bridge = _Bridge(CodexBridgeStore(default_home / "state.db"), profile_home=profile_home)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="1001", user_id="42", chat_type="dm", profile="work",
    )

    inserted = await bridge._codex_bridge_ledger_call(
        source,
        ensure_obligation,
        obligation_id="profile-owned-output",
        session_key="lane",
        platform="telegram",
        chat_id="1001",
        thread_id=None,
        content="done",
        adapter_profile="work",
    )

    assert inserted is True
    with sqlite3.connect(profile_home / "state.db") as conn:
        row = conn.execute(
            "SELECT content, adapter_profile FROM delivery_obligations WHERE obligation_id=?",
            ("profile-owned-output",),
        ).fetchone()
    assert row == ("done", "work")


@pytest.mark.asyncio
async def test_progress_is_durable_across_restart_and_each_report_gets_a_new_message(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "state.db"
    store = CodexBridgeStore(db_path)
    bridge = _Bridge(store)
    binding = store.bind("control", _source(), thread_id="thread-123")
    events = [
        RolloutEvent(
            event_id=f"event-{index}", turn_id="turn-1", kind="commentary",
            text=f"progress {index}", offset=index,
        )
        for index in range(10)
    ]
    failed_adapter = SimpleNamespace(
        supports_draft_streaming=lambda **_kwargs: True,
        send_draft=AsyncMock(),
        send=AsyncMock(return_value=SimpleNamespace(success=False, error="temporary")),
        edit_message=AsyncMock(),
    )
    progress_rows = bridge._codex_bridge_stage_commentary(store, binding, events)

    with pytest.raises(RuntimeError, match="temporary"):
        await bridge._codex_bridge_deliver_progress(
            store, binding, progress_rows[0], failed_adapter,
        )

    pending = store.list_progress(binding, pending_only=True)
    assert len(pending) == len(events)
    failed_adapter.send_draft.assert_not_awaited()
    retry_at = max(progress.next_attempt_at for progress in pending) + 1
    monkeypatch.setattr("gateway.codex_bridge.store.time.time", lambda: retry_at)

    restarted_store = CodexBridgeStore(db_path)
    restarted_bridge = _Bridge(restarted_store)
    restarted_binding = restarted_store.get_binding("control")
    assert restarted_binding is not None
    adapter = SimpleNamespace(
        supports_draft_streaming=lambda **_kwargs: True,
        send_draft=AsyncMock(),
        send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="900")),
        edit_message=AsyncMock(return_value=SimpleNamespace(success=True, message_id="900")),
    )

    await restarted_bridge._codex_bridge_retry_progress(
        restarted_store, restarted_binding, adapter,
    )

    assert adapter.send.await_count == len(events)
    sent_text = "\n".join(call.args[1] for call in adapter.send.await_args_list)
    assert all(event.text in sent_text for event in events)
    assert all(
        call.kwargs["metadata"]["_interim_send"] is True
        for call in adapter.send.await_args_list
    )
    adapter.send_draft.assert_not_awaited()

    update = RolloutEvent(
        event_id="event-new", turn_id="turn-1", kind="commentary",
        text="progress new", offset=11,
    )
    revised = restarted_bridge._codex_bridge_stage_commentary(
        restarted_store, restarted_binding, [update],
    )
    await restarted_bridge._codex_bridge_deliver_progress(
        restarted_store, restarted_binding, revised[0], adapter,
    )

    assert adapter.send.await_count == len(events) + 1
    assert "progress new" in adapter.send.await_args.args[1]
    adapter.edit_message.assert_not_awaited()
