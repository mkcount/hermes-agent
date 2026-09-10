"""Pre-guard routing tests for bound Telegram Codex lanes."""

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.codex_bridge.catalog import CodexThreadSummary
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
        self.async_session_store = SimpleNamespace(get_or_create_session=AsyncMock())
        self._codex_bridge_binding_locks = {}
        self._codex_bridge_tails = {}
        self._codex_bridge_mirror_ui = {}

    @staticmethod
    def _session_key_for_source(source):
        return build_session_key(source)

    def _codex_bridge_store_for_source(self, _source):
        return self.store

    def _resolve_profile_home_for_source(self, _source):
        return self.profile_home

    @staticmethod
    def _thread_metadata_for_source(_source):
        return {}

    @staticmethod
    def _peek_session_state(_session_key):
        return None

    @staticmethod
    def _evict_cached_agent(_session_key):
        return None


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
    bridge = _Bridge(store)
    monkeypatch.setattr("gateway.codex_bridge.mixin.inspect_rollout", lambda *_args, **_kwargs: None)

    answer = await bridge._codex_bridge_store_binding(
        source,
        CodexThreadSummary(
            thread_id="thread-123", title="Current", cwd="/project",
            updated_at=1, status="idle",
        ),
    )

    assert "이미 연결된" in answer
    assert store.get_binding(build_session_key(source)).generation == current.generation
    assert store.input_state(input_id) == "routed"


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
async def test_runner_configures_full_access_and_durable_turn_callbacks(tmp_path):
    store = CodexBridgeStore(tmp_path / "state.db")
    source = _source()
    binding = store.bind(build_session_key(source), source, thread_id="thread-123", cwd="/project")
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
async def test_progress_burst_is_coalesced_into_one_draft_update(tmp_path):
    bridge = _Bridge(CodexBridgeStore(tmp_path / "state.db"))
    bridge._codex_bridge_mirror_ui = {}
    binding = bridge.store.bind("control", _source(), thread_id="thread-123")
    adapter = SimpleNamespace(
        supports_draft_streaming=lambda **_kwargs: True,
        send_draft=AsyncMock(return_value=SimpleNamespace(success=True)),
    )
    events = [
        RolloutEvent(
            event_id=f"event-{index}", turn_id="turn-1", kind="commentary",
            text=f"progress {index}", offset=index,
        )
        for index in range(3)
    ]

    await bridge._codex_bridge_mirror_commentary_batch(binding, events, adapter)

    adapter.send_draft.assert_awaited_once()
    sent_text = adapter.send_draft.await_args.args[2]
    assert all(event.text in sent_text for event in events)
