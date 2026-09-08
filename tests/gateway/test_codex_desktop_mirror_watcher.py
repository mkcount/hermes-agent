import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, _codex_result_is_successful_completion
from gateway.session import SessionSource


def _event(payload):
    return json.dumps({"type": "event_msg", "payload": payload}) + "\n"


def _response_message(turn_id, role, text, *, phase=None):
    payload = {
        "type": "message",
        "role": role,
        "content": [
            {
                "type": "input_text" if role == "user" else "output_text",
                "text": text,
            }
        ],
        "internal_chat_message_metadata_passthrough": {"turn_id": turn_id},
    }
    if phase is not None:
        payload["phase"] = phase
    return json.dumps({"type": "response_item", "payload": payload}) + "\n"


def _completed_user_item(turn_id, text, client_id):
    return _event(
        {
            "type": "item_completed",
            "turn_id": turn_id,
            "item": {
                "type": "UserMessage",
                "client_id": client_id,
                "content": [{"type": "text", "text": text}],
            },
        }
    )


def _turn(
    turn_id,
    *,
    client_id=None,
    final_text="done",
    commentary=None,
):
    user = {"type": "user_message", "message": "hello"}
    if client_id is not None:
        user["client_id"] = client_id
    return (
        _event({"type": "task_started", "turn_id": turn_id})
        + _event(user)
        + (
            _event(
                {
                    "type": "agent_message",
                    "phase": "commentary",
                    "message": commentary,
                }
            )
            if commentary
            else ""
        )
        + _event(
            {
                "type": "task_complete",
                "turn_id": turn_id,
                "last_agent_message": final_text,
            }
        )
    )


class _Store:
    def __init__(self, binding):
        self.binding = dict(binding)
        self.lane_binding = None
        self._store = self

    async def get_session_metadata(self, _session_key, metadata_key, *_args):
        if metadata_key == "codex_execution_thread":
            return (
                dict(self.lane_binding)
                if self.lane_binding is not None
                else None
            )
        return dict(self.binding) if self.binding is not None else None

    async def update_session_metadata_dict_if_matches(
        self,
        _session_key,
        _key,
        *,
        expected,
        updates,
    ):
        if self.binding is None:
            return False
        if any(self.binding.get(k) != v for k, v in expected.items()):
            return False
        self.binding.update(updates)
        return True

    def append_session_metadata_list_if_matches(
        self,
        _session_key,
        metadata_key,
        *,
        expected,
        list_key,
        value,
        max_items,
    ):
        target = (
            self.lane_binding
            if metadata_key == "codex_execution_thread"
            else self.binding
        )
        if target is None:
            return False
        if any(target.get(k) != v for k, v in expected.items()):
            return False
        values = list(target.get(list_key) or [])
        if value not in values:
            values.append(value)
        target[list_key] = values[-max_items:]
        return True


def _runner(binding):
    runner = object.__new__(GatewayRunner)
    runner._codex_desktop_mirror_states = {}
    runner._running_agents = {}
    store = _Store(binding)
    runner.session_store = store
    runner._async_session_store = store
    adapter = MagicMock()
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(
            success=True, error=None, message_id="message-1"
        )
    )
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(
            success=True, error=None, message_id="message-1"
        )
    )
    adapter.send_image_file = AsyncMock(
        return_value=SimpleNamespace(
            success=True, error=None, message_id="image-1"
        )
    )
    adapter.delete_message = AsyncMock(return_value=True)
    runner._adapter_for_source = MagicMock(return_value=adapter)
    runner._thread_metadata_for_target = MagicMock(return_value=None)
    runner._session_key_for_source = lambda source: (
        "execution-lane" if source.session_lane else "telegram-session"
    )
    return runner, adapter


@pytest.mark.asyncio
async def test_desktop_completion_is_delivered_but_telegram_completion_is_not(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(_turn("baseline", final_text="old"), encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )
    adapter.send.assert_not_awaited()

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _turn(
                "desktop-new",
                client_id="desktop-client",
                final_text="from desktop",
            )
        )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )
    adapter.send.assert_awaited_once()
    assert "from desktop" in adapter.send.await_args.args[1]
    assert "👤 질문" in adapter.send.await_args.args[1]
    assert "hello" in adapter.send.await_args.args[1]
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "desktop-new"
    )

    adapter.send.reset_mock()
    runner.async_session_store.binding[
        "mirror_hermes_turn_ids"
    ] = ["telegram-new"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _turn(
                "telegram-new",
                client_id="hermes-client",
                final_text="from telegram",
            )
        )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )
    adapter.send.assert_not_awaited()
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "telegram-new"
    )


@pytest.mark.asyncio
async def test_pre_turn_client_claim_suppresses_modern_live_mirror(
    tmp_path,
    monkeypatch,
):
    """Hermes owns the rollout before turn/start returns the turn id."""
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    turn_id = "hermes-pre-response"
    client_id = "hermes-client-pre-response"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _event({"type": "task_started", "turn_id": turn_id})
        + _response_message(turn_id, "user", "Telegram question")
        + _completed_user_item(turn_id, "Telegram question", client_id),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    authority = runner._get_codex_delivery_authority()
    authority.observe_binding("telegram-session", thread_id)
    grant = authority.issue_grant("telegram-session", thread_id)
    assert grant is not None
    grant_metadata = grant.to_metadata()
    assert authority.acquire_direct_client(grant_metadata, client_id)

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session",
        source,
        binding,
    )

    adapter.send.assert_not_awaited()
    pending = runner._codex_desktop_mirror_states[
        "telegram-session"
    ]["pending_updates"]
    assert pending[turn_id].client_id == client_id

    # turn/start returned: exact turn ownership replaces the temporary client
    # claim without leaving a mirror-visible interval.
    assert authority.acquire_direct_turn(grant_metadata, turn_id)
    runner.async_session_store.binding["mirror_hermes_turn_ids"] = [turn_id]
    authority.release_direct_client(grant_metadata, client_id)

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session",
        source,
        runner.async_session_store.binding,
    )

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_late_direct_claim_deletes_already_sent_mirror_without_new_event(
    tmp_path,
    monkeypatch,
):
    """A late turn/start response heals a mirror bubble on the next poll."""
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    turn_id = "late-direct-owner"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _event({"type": "task_started", "turn_id": turn_id})
        + _event(
            {
                "type": "user_message",
                "message": "Telegram question",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session",
        source,
        binding,
    )
    adapter.send.assert_awaited_once()
    assert turn_id in runner._codex_desktop_mirror_states[
        "telegram-session"
    ]["live_messages"]

    authority = runner._get_codex_delivery_authority()
    grant = authority.issue_grant("telegram-session", thread_id)
    assert grant is not None
    grant_metadata = grant.to_metadata()
    assert authority.acquire_direct_turn(grant_metadata, turn_id)
    runner.async_session_store.binding["mirror_hermes_turn_ids"] = [turn_id]

    # No additional rollout line is appended here. Reconciliation itself must
    # notice the exact direct claim and remove the duplicate bubble.
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session",
        source,
        runner.async_session_store.binding,
    )

    adapter.delete_message.assert_awaited_once_with(
        "8775784529",
        "message-1",
    )
    assert turn_id not in runner._codex_desktop_mirror_states[
        "telegram-session"
    ]["live_messages"]


@pytest.mark.asyncio
async def test_mirror_aba_switch_revokes_inflight_old_generation(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(_turn("baseline", final_text="old"), encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    session_key = "telegram-session"
    await runner._poll_codex_desktop_mirror_binding(
        session_key,
        source,
        binding,
    )

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _turn(
                "desktop-after-switch",
                client_id="desktop-client",
                final_text="must stay hidden",
            )
        )

    original_render = runner._render_codex_desktop_mirror_turn

    def switch_away_and_back(turn):
        authority = runner._get_codex_delivery_authority()
        generation = authority.begin_transition(session_key)
        assert authority.commit_transition(
            session_key,
            generation,
            "thread-b",
        )
        generation = authority.begin_transition(session_key)
        assert authority.commit_transition(
            session_key,
            generation,
            thread_id,
        )
        return original_render(turn)

    runner._render_codex_desktop_mirror_turn = switch_away_and_back

    await runner._poll_codex_desktop_mirror_binding(
        session_key,
        source,
        runner.async_session_store.binding,
    )

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_reselected_thread_does_not_echo_hermes_turn_from_lane_metadata(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(_turn("baseline", final_text="old"), encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session",
        source,
        binding,
    )

    runner.async_session_store.lane_binding = {
        "thread_id": thread_id,
        "mirror_hermes_turn_ids": ["hermes-finished-after-reselect"],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _turn(
                "hermes-finished-after-reselect",
                # Use a desktop-shaped rollout event so the test proves the
                # durable lane turn-id guard, independent of client-id
                # heuristics in the rollout parser.
                client_id="desktop-client",
                final_text="must not be mirrored",
            )
        )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session",
        source,
        runner.async_session_store.binding,
    )

    adapter.send.assert_not_awaited()
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "hermes-finished-after-reselect"
    )


@pytest.mark.asyncio
async def test_desktop_task_error_is_delivered_and_advances_cursor(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(_turn("baseline", final_text="old"), encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
        "mirror_hermes_turn_ids": ["capacity-error"],
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _event({"type": "task_started", "turn_id": "capacity-error"})
            + _event(
                {
                    "type": "user_message",
                    "message": "Unity 상태를 확인해줘",
                }
            )
            + _event(
                {
                    "type": "task_complete",
                    "turn_id": "capacity-error",
                    "last_agent_message": None,
                    "error": {
                        "message": (
                            "Selected model is at capacity. "
                            "Please try a different model."
                        ),
                        "codex_error_info": "server_overloaded",
                    },
                }
            )
        )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )

    adapter.send.assert_awaited_once()
    content = adapter.send.await_args.args[1]
    assert "⚠️ 오류" in content
    assert "Selected model is at capacity" in content
    assert "🤖 답변" not in content
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "capacity-error"
    )


@pytest.mark.asyncio
async def test_aborted_hermes_turn_preserves_progress_and_advances_cursor(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    turn_id = "hermes-timeout-abort"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _event({"type": "task_started", "turn_id": turn_id})
        + _event(
            {
                "type": "user_message",
                "message": "진행해",
                "client_id": "hermes-client",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "보존되어야 하는 중간보고",
            }
        )
        + _event(
            {
                "type": "turn_aborted",
                "turn_id": turn_id,
                "reason": "interrupted",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
        "mirror_hermes_turn_ids": [turn_id],
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )

    adapter.send.assert_awaited_once()
    content = adapter.send.await_args.args[1]
    assert "보존되어야 하는 중간보고" in content
    assert "⚠️ 오류" in content
    assert "완료되기 전에 중단" in content
    assert "⏳ 작업 중" not in content
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == turn_id
    )


@pytest.mark.asyncio
async def test_desktop_progress_edits_one_message_until_completion(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(_turn("baseline", final_text="old"), encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _event({"type": "task_started", "turn_id": "desktop-live"})
            + _event({"type": "user_message", "message": "live question"})
            + _event(
                    {
                        "type": "agent_message",
                        "phase": "commentary",
                        "message": "첫 번째 진행 상황",
                }
            )
        )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )
    adapter.send.assert_awaited_once()
    first_content = adapter.send.await_args.args[1]
    assert "live question" in first_content
    assert "첫 번째 진행 상황" in first_content
    assert "⏳ 작업 중" in first_content

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _event(
                    {
                        "type": "agent_message",
                        "phase": "commentary",
                        "message": "두 번째 진행 상황",
                }
            )
            + _event(
                {
                    "type": "agent_message",
                    "phase": "final_answer",
                    "message": "live answer",
                }
            )
            + _event(
                {
                    "type": "task_complete",
                    "turn_id": "desktop-live",
                    "last_agent_message": "live answer",
                }
            )
        )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )

    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_awaited_once()
    edit_args = adapter.edit_message.await_args.args
    assert edit_args[1] == "message-1"
    assert "첫 번째 진행 상황" in edit_args[2]
    assert "두 번째 진행 상황" in edit_args[2]
    assert "live answer" in edit_args[2]
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "desktop-live"
    )


@pytest.mark.asyncio
async def test_new_turn_finalizes_live_message_for_superseded_turn(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(_turn("baseline", final_text="old"), encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _event({"type": "task_started", "turn_id": "old-live"})
            + _event({"type": "user_message", "message": "중단될 질문"})
        )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )
    adapter.send.assert_awaited_once()

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _event({"type": "task_started", "turn_id": "replacement"})
            + _event({"type": "user_message", "message": "새 질문"})
        )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )

    assert adapter.edit_message.await_count == 1
    assert "이전 작업이 중단" in adapter.edit_message.await_args.args[2]
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "old-live"
    )


@pytest.mark.asyncio
async def test_attach_mid_turn_immediately_restores_question_and_progress(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _event({"type": "task_started", "turn_id": "already-running"})
        + _event(
            {
                "type": "user_message",
                "message": "연결 전에 내린 작업 지시",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "연결 전에 쌓인 첫 보고",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "연결 전에 쌓인 둘째 보고",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )

    adapter.send.assert_awaited_once()
    initial_content = adapter.send.await_args.args[1]
    assert "연결 전에 내린 작업 지시" in initial_content
    assert "연결 전에 쌓인 첫 보고" in initial_content
    assert "연결 전에 쌓인 둘째 보고" in initial_content
    assert "⏳ 작업 중" in initial_content

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _event(
                {
                    "type": "agent_message",
                    "phase": "commentary",
                    "message": "연결 후 실시간 보고",
                }
            )
        )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )

    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_awaited_once()
    edited_content = adapter.edit_message.await_args.args[2]
    assert "연결 전에 쌓인 첫 보고" in edited_content
    assert "연결 후 실시간 보고" in edited_content


@pytest.mark.asyncio
async def test_reattach_does_not_resurrect_superseded_unfinished_turn(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _event({"type": "task_started", "turn_id": "orphan-turn"})
        + _event({"type": "user_message", "message": "중단된 옛 질문"})
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "중단된 옛 작업 보고",
            }
        )
        + _turn("newer-complete", final_text="이미 전달된 최신 답변"),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "newer-complete",
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )

    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()
    state = runner._codex_desktop_mirror_states["telegram-session"]
    assert state["pending"] == []
    assert state["pending_updates"] == {}


@pytest.mark.asyncio
async def test_attach_completion_boundary_sends_only_completed_turn(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _turn(
            "completed-during-attach",
            final_text="경계에서 완료된 답변",
            commentary="경계 직전 보고",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )

    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_not_awaited()
    content = adapter.send.await_args.args[1]
    assert "경계에서 완료된 답변" in content
    assert "⏳ 작업 중" not in content
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "completed-during-attach"
    )


@pytest.mark.asyncio
async def test_attach_does_not_echo_active_hermes_originated_turn(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _event({"type": "task_started", "turn_id": "hermes-active"})
        + _event(
            {
                "type": "user_message",
                "message": "Telegram에서 시작된 질문",
                "client_id": "hermes-client",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "중복되면 안 되는 보고",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
        "mirror_hermes_turn_ids": ["hermes-active"],
    }
    runner, adapter = _runner(binding)
    runner._running_agents["execution-lane"] = object()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )

    adapter.send.assert_not_awaited()
    assert "hermes-active" in runner._codex_desktop_mirror_states[
        "telegram-session"
    ]["pending_updates"]


@pytest.mark.asyncio
async def test_reselect_hands_stale_direct_turn_to_mirror_through_completion(
    tmp_path,
    monkeypatch,
):
    """A revoked A turn must be mirrored after an A -> B -> A selection."""
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    turn_id = "hermes-active-before-reselect"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _event({"type": "task_started", "turn_id": turn_id})
        + _event(
            {
                "type": "user_message",
                "message": "전환 전에 Telegram에서 시작된 질문",
                "client_id": "hermes-client",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "전환 뒤 복원되어야 하는 중간보고",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
        "mirror_hermes_turn_ids": [turn_id],
    }
    runner, adapter = _runner(binding)
    runner.async_session_store.lane_binding = {
        "thread_id": thread_id,
        "mirror_hermes_turn_ids": [turn_id],
    }
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    authority = runner._get_codex_delivery_authority()
    authority.observe_binding("telegram-session", thread_id)
    old_grant = authority.issue_grant("telegram-session", thread_id)
    assert old_grant is not None
    old_grant_metadata = old_grant.to_metadata()
    assert authority.acquire_direct_turn(old_grant_metadata, turn_id)
    runner._running_agents["execution-lane"] = SimpleNamespace(
        _gateway_codex_delivery_grant=old_grant_metadata,
        _codex_session=SimpleNamespace(
            is_directly_streaming_turn=lambda candidate: candidate == turn_id
        ),
    )

    generation = authority.begin_transition("telegram-session")
    assert authority.commit_transition(
        "telegram-session", generation, "thread-b"
    )
    generation = authority.begin_transition("telegram-session")
    assert authority.commit_transition(
        "telegram-session", generation, thread_id
    )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )

    adapter.send.assert_awaited_once()
    assert "전환 뒤 복원되어야 하는 중간보고" in adapter.send.await_args.args[1]
    state = runner._codex_desktop_mirror_states["telegram-session"]
    assert turn_id in state["mirror_owned_turn_ids"]

    # The revoked direct run can finish close to the transition and leave its
    # durable marker behind. Once the mirror has visibly taken ownership, that
    # late marker must not retract the live bubble or suppress its final edit.
    runner.async_session_store.binding[
        "mirror_hermes_completed_turn_ids"
    ] = [turn_id]
    runner.async_session_store.lane_binding[
        "mirror_hermes_completed_turn_ids"
    ] = [turn_id]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _event(
                {
                    "type": "task_complete",
                    "turn_id": turn_id,
                    "last_agent_message": "전환 뒤 미러가 전달할 최종 답변",
                }
            )
        )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )

    adapter.delete_message.assert_not_awaited()
    adapter.edit_message.assert_awaited_once()
    final_content = adapter.edit_message.await_args.args[2]
    assert "전환 뒤 미러가 전달할 최종 답변" in final_content
    assert "⏳ 작업 중" not in final_content
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == turn_id
    )


def test_completed_turn_marker_is_persisted_to_control_and_execution_lane():
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    binding = {"thread_id": thread_id}
    runner, _adapter = _runner(binding)
    runner.async_session_store.lane_binding = {"thread_id": thread_id}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
        session_lane=thread_id,
    )

    runner._record_codex_mirror_turn_marker(
        source=source,
        session_key="execution-lane",
        thread_id=thread_id,
        turn_id="hermes-completed",
        list_key="mirror_hermes_completed_turn_ids",
    )

    assert runner.async_session_store.binding[
        "mirror_hermes_completed_turn_ids"
    ] == ["hermes-completed"]
    assert runner.async_session_store.lane_binding[
        "mirror_hermes_completed_turn_ids"
    ] == ["hermes-completed"]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {
                "completed": True,
                "partial": False,
                "interrupted": False,
                "error": None,
            },
            True,
        ),
        (
            {
                "completed": False,
                "partial": True,
                # App-server timeouts are not user interrupts, so this field
                # is False even though the underlying Codex turn was aborted.
                "interrupted": False,
                "error": "turn timed out after 600.0s",
            },
            False,
        ),
        ({"completed": True, "partial": True, "error": None}, False),
        ({"completed": True, "partial": False, "error": "failed"}, False),
        ({"partial": False, "error": None}, False),
    ],
)
def test_codex_result_successful_completion_requires_explicit_terminal_success(
    result,
    expected,
):
    assert _codex_result_is_successful_completion(result) is expected


@pytest.mark.asyncio
async def test_released_direct_lane_does_not_echo_completed_hermes_turn(
    tmp_path,
    monkeypatch,
):
    """Reproduce the completion-poll race that emitted a second summary."""
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    turn_id = "hermes-direct-completed"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _event({"type": "task_started", "turn_id": turn_id})
        + _event(
            {
                "type": "user_message",
                "message": "Telegram에서 시작된 질문",
                "client_id": "hermes-client",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "이미 직접 전달된 중간보고",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
        "mirror_hermes_turn_ids": [turn_id],
    }
    runner, adapter = _runner(binding)
    runner._running_agents["execution-lane"] = SimpleNamespace(
        _codex_session=SimpleNamespace(
            is_directly_streaming_turn=lambda candidate: candidate == turn_id
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )

    # The live snapshot is queued while Hermes still owns direct streaming.
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )
    adapter.send.assert_not_awaited()
    assert turn_id in runner._codex_desktop_mirror_states[
        "telegram-session"
    ]["pending_updates"]

    # Hermes receives the successful completion and persists ownership before
    # its execution lane is released. The next mirror scan sees both the old
    # pending snapshot and task_complete, exactly matching the production race.
    runner.async_session_store.binding[
        "mirror_hermes_completed_turn_ids"
    ] = [turn_id]
    runner._running_agents.pop("execution-lane")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _event(
                {
                    "type": "task_complete",
                    "turn_id": turn_id,
                    "last_agent_message": "이미 직접 전달된 최종 답변",
                }
            )
        )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )

    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == turn_id
    )
    assert turn_id not in runner._codex_desktop_mirror_states[
        "telegram-session"
    ]["pending_updates"]


@pytest.mark.asyncio
async def test_direct_turn_lease_closes_task_complete_delivery_race(
    tmp_path,
    monkeypatch,
):
    """task_complete must wait for direct delivery to persist its outcome."""
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    turn_id = "hermes-direct-unwinding"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _event({"type": "task_started", "turn_id": turn_id})
        + _event(
            {
                "type": "user_message",
                "message": "Telegram에서 시작된 질문",
                "client_id": "hermes-client",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "직접 전달 중인 보고",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
        "mirror_hermes_turn_ids": [turn_id],
    }
    runner, adapter = _runner(binding)
    direct_active = {"value": True}
    runner._running_agents["execution-lane"] = SimpleNamespace(
        _codex_session=SimpleNamespace(
            is_directly_streaming_turn=lambda candidate: (
                candidate == turn_id and direct_active["value"]
            )
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    authority = runner._get_codex_delivery_authority()
    authority.observe_binding("telegram-session", thread_id)
    grant = authority.issue_grant("telegram-session", thread_id)
    assert grant is not None
    grant_metadata = grant.to_metadata()
    assert authority.acquire_direct_turn(grant_metadata, turn_id)

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )
    adapter.send.assert_not_awaited()

    # App-server clears _active_turn_id immediately after task_complete, but
    # the gateway still owns final delivery until it persists the success
    # marker. The generation-scoped lease must cover this exact gap.
    direct_active["value"] = False
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _event(
                {
                    "type": "task_complete",
                    "turn_id": turn_id,
                    "last_agent_message": "직접 전달된 최종 답변",
                }
            )
        )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )

    adapter.send.assert_not_awaited()
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "baseline"
    )
    assert runner._codex_desktop_mirror_states["telegram-session"]["pending"]

    # Success persistence happens-before lease release. The following poll
    # re-reads the marker after observing no owner, suppresses the mirror send,
    # and only then advances the durable cursor.
    runner.async_session_store.binding[
        "mirror_hermes_completed_turn_ids"
    ] = [turn_id]
    authority.release_direct_turn(grant_metadata, turn_id)

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )

    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == turn_id
    )


@pytest.mark.asyncio
async def test_direct_turn_lease_hands_failed_delivery_to_mirror(
    tmp_path,
    monkeypatch,
):
    """A released lease without a success marker must fail over to the mirror."""
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    turn_id = "hermes-direct-delivery-failed"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _event({"type": "task_started", "turn_id": turn_id})
        + _event(
            {
                "type": "user_message",
                "message": "Telegram에서 시작된 질문",
                "client_id": "hermes-client",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "직접 전달 실패 전 중간보고",
            }
        )
        + _event(
            {
                "type": "task_complete",
                "turn_id": turn_id,
                "last_agent_message": "미러가 대신 보낼 최종 답변",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
        "mirror_hermes_turn_ids": [turn_id],
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    authority = runner._get_codex_delivery_authority()
    authority.observe_binding("telegram-session", thread_id)
    grant = authority.issue_grant("telegram-session", thread_id)
    assert grant is not None
    grant_metadata = grant.to_metadata()
    assert authority.acquire_direct_turn(grant_metadata, turn_id)

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )

    adapter.send.assert_not_awaited()
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "baseline"
    )

    # Direct final delivery failed, so no durable completion marker exists.
    # Releasing ownership must let the mirror deliver the queued completion.
    authority.release_direct_turn(grant_metadata, turn_id)
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )

    adapter.send.assert_awaited_once()
    content = adapter.send.await_args.args[1]
    assert "직접 전달 실패 전 중간보고" in content
    assert "미러가 대신 보낼 최종 답변" in content
    assert "⏳ 작업 중" not in content
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == turn_id
    )


@pytest.mark.asyncio
async def test_restart_queue_wait_hands_live_hermes_turn_to_mirror(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _event({"type": "task_started", "turn_id": "restart-active"})
        + _event(
            {
                "type": "user_message",
                "message": "재시작 전 Telegram 질문",
                "client_id": "hermes-client",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "재시작 뒤에도 보여야 하는 보고",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
        "mirror_hermes_turn_ids": ["restart-active"],
    }
    runner, adapter = _runner(binding)
    waiting_session = SimpleNamespace(
        is_directly_streaming_turn=lambda turn_id: False
    )
    runner._running_agents["execution-lane"] = SimpleNamespace(
        _codex_session=waiting_session
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )

    adapter.send.assert_awaited_once()
    content = adapter.send.await_args.args[1]
    assert "재시작 전 Telegram 질문" in content
    assert "재시작 뒤에도 보여야 하는 보고" in content
    assert "⏳ 작업 중" in content


@pytest.mark.asyncio
async def test_inactive_direct_lane_hands_live_hermes_turn_to_mirror(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        _turn("baseline", final_text="old")
        + _event({"type": "task_started", "turn_id": "hermes-timeout"})
        + _event(
            {
                "type": "user_message",
                "message": "Telegram에서 시작한 장시간 작업",
                "client_id": "hermes-client",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "직접 중계 종료 뒤에도 보이는 보고",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
        "mirror_hermes_turn_ids": ["hermes-timeout"],
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )

    adapter.send.assert_awaited_once()
    content = adapter.send.await_args.args[1]
    assert "Telegram에서 시작한 장시간 작업" in content
    assert "직접 중계 종료 뒤에도 보이는 보고" in content
    assert "⏳ 작업 중" in content

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _event(
                {
                    "type": "agent_message",
                    "phase": "commentary",
                    "message": "인계 뒤 새 중간보고",
                }
            )
            + _event(
                {
                    "type": "task_complete",
                    "turn_id": "hermes-timeout",
                    "last_agent_message": "장시간 작업 최종 답변",
                }
            )
        )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )

    adapter.edit_message.assert_awaited_once()
    final_content = adapter.edit_message.await_args.args[2]
    assert "인계 뒤 새 중간보고" in final_content
    assert "장시간 작업 최종 답변" in final_content
    assert "⏳ 작업 중" not in final_content
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "hermes-timeout"
    )


def test_long_progress_is_preserved_and_split_without_omission():
    turn = SimpleNamespace(
        user_text="긴 질문 " + "나" * 900,
        progress=(
            SimpleNamespace(kind="commentary", text="오래된 보고 " + "가" * 1400),
            SimpleNamespace(
                kind="commentary",
                text="가장 최신 진행 상황 " + "다" * 1400,
            ),
        ),
        final_text="긴 최종 답변 " + "라" * 1800,
        error_text="",
        completed=True,
    )

    rendered = GatewayRunner._render_codex_desktop_mirror_turn(turn)
    chunks = GatewayRunner._split_codex_desktop_mirror_content(rendered)

    assert "…(일부 생략)" not in rendered
    assert "…(이전 과정 일부 생략)" not in rendered
    assert "긴 질문 " + "나" * 900 in rendered
    assert "오래된 보고 " + "가" * 1400 in rendered
    assert "가장 최신 진행 상황" in rendered
    assert "긴 최종 답변 " + "라" * 1800 in rendered
    assert len(chunks) >= 2
    assert "".join(chunks) == rendered
    assert all(
        len(chunk.encode("utf-16-le")) // 2 <= 1800
        for chunk in chunks
    )


@pytest.mark.asyncio
async def test_long_desktop_turn_uses_multiple_linked_messages():
    runner, adapter = _runner({})
    adapter.send.side_effect = [
        SimpleNamespace(
            success=True,
            error=None,
            message_id=f"message-{index}",
        )
        for index in range(1, 10)
    ]
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    turn = SimpleNamespace(
        turn_id="long-turn",
        user_text="질문 " + "가" * 2500,
        progress=(
            SimpleNamespace(
                kind="commentary",
                text="중간보고 " + "나" * 2500,
            ),
        ),
        final_text="최종답변 " + "다" * 2500,
        error_text="",
        completed=True,
    )
    authority = runner._get_codex_delivery_authority()
    authority.observe_binding("telegram-session", "desktop-thread")
    grant = authority.issue_grant(
        "telegram-session",
        "desktop-thread",
    ).to_metadata()

    delivered = await runner._upsert_codex_desktop_mirror_message(
        {},
        source,
        turn,
        grant,
    )

    assert delivered is True
    expected_chunks = GatewayRunner._split_codex_desktop_mirror_content(
        GatewayRunner._render_codex_desktop_mirror_turn(turn)
    )
    assert adapter.send.await_count == len(expected_chunks)
    sent_chunks = [call.args[1] for call in adapter.send.await_args_list]
    rendered = GatewayRunner._render_codex_desktop_mirror_turn(turn)
    assert "".join(sent_chunks) == rendered
    assert all("생략)" not in chunk for chunk in sent_chunks)
    for index, call in enumerate(adapter.send.await_args_list[1:], start=1):
        assert call.kwargs["reply_to"] == f"message-{index}"


@pytest.mark.asyncio
async def test_detached_binding_stops_pending_delivery(tmp_path, monkeypatch):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(_turn("baseline"), encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_turn("desktop-new", final_text="should not send"))
    runner.async_session_store.binding = None

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )

    adapter.send.assert_not_awaited()
    assert "telegram-session" not in runner._codex_desktop_mirror_states


@pytest.mark.asyncio
async def test_completed_desktop_turn_uploads_local_markdown_image(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(_turn("baseline", final_text="old"), encoding="utf-8")
    image_path = tmp_path / "screen shot.png"
    image_path.write_bytes(b"fake-png-for-upload-mock")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
    }
    runner, adapter = _runner(binding)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )

    final_text = (
        "캡처했습니다.\n\n"
        f"![현재 화면](<{image_path}>)"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_turn("desktop-image", final_text=final_text))
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )

    text_content = adapter.send.await_args.args[1]
    assert "캡처했습니다." in text_content
    assert str(image_path) not in text_content
    assert "![" not in text_content
    adapter.send_image_file.assert_awaited_once()
    image_kwargs = adapter.send_image_file.await_args.kwargs
    assert image_kwargs["image_path"] == str(image_path.resolve())
    assert image_kwargs["caption"] == "현재 화면"
    assert image_kwargs["reply_to"] == "message-1"
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "desktop-image"
    )


def test_local_image_extractor_drops_missing_host_path_without_leaking_it():
    cleaned, images = GatewayRunner._extract_codex_desktop_local_images(
        "before\n![secret](/tmp/does-not-exist.png)\nafter"
    )

    assert images == []
    assert "/tmp/does-not-exist.png" not in cleaned
    assert cleaned == "before\n\nafter"


@pytest.mark.asyncio
async def test_image_upload_failure_retries_without_duplicate_text(
    tmp_path,
    monkeypatch,
):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(_turn("baseline", final_text="old"), encoding="utf-8")
    image_path = tmp_path / "retry.png"
    image_path.write_bytes(b"retry-image")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    binding = {
        "thread_id": thread_id,
        "rollout_path": str(path),
        "mirror_cursor_turn_id": "baseline",
    }
    runner, adapter = _runner(binding)
    adapter.send_image_file.side_effect = [
        SimpleNamespace(success=False, error="temporary"),
        SimpleNamespace(success=True, error=None, message_id="image-1"),
    ]
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8775784529",
        user_id="8775784529",
        chat_type="dm",
    )
    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, binding
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            _turn(
                "desktop-retry",
                final_text=f"answer\n![image]({image_path})",
            )
        )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "baseline"
    )

    await runner._poll_codex_desktop_mirror_binding(
        "telegram-session", source, runner.async_session_store.binding
    )
    adapter.send.assert_awaited_once()
    assert adapter.send_image_file.await_count == 2
    assert (
        runner.async_session_store.binding["mirror_cursor_turn_id"]
        == "desktop-retry"
    )
