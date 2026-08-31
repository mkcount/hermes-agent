"""Behavior tests for the gateway's Codex desktop-session picker."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from agent.transports.codex_app_server_session import CodexThreadSummary
from agent.transports.codex_desktop_mirror import CodexDesktopCompletion
from gateway.codex_delivery import CODEX_DELIVERY_GRANT_METADATA_KEY
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, build_session_key


class _SessionStore:
    def __init__(self):
        self._store = self
        self.metadata = {}
        self.promotions = []

    async def get_or_create_session(self, source):
        return object()

    async def get_session_metadata(self, session_key, key, default=None):
        return self.metadata.get((session_key, key), default)

    async def set_session_metadata(self, session_key, key, value):
        self.metadata[(session_key, key)] = value
        return True

    async def update_session_metadata_dict_if_matches(
        self,
        session_key,
        key,
        *,
        expected,
        updates,
    ):
        current = self.metadata.get((session_key, key))
        if not isinstance(current, dict):
            return False
        if any(current.get(name) != value for name, value in expected.items()):
            return False
        merged = dict(current)
        merged.update(updates)
        self.metadata[(session_key, key)] = merged
        return True

    async def promote_session_lane_and_set_metadata(
        self,
        control_session_key,
        metadata_key,
        *,
        expected,
        value,
        old_session_key,
        new_source,
    ):
        current = self.metadata.get((control_session_key, metadata_key))
        if not isinstance(current, dict):
            return False
        if any(current.get(name) != item for name, item in expected.items()):
            return False
        self.metadata[(control_session_key, metadata_key)] = value
        self.promotions.append((old_session_key, new_source.session_lane))
        return True


class _PostDeliveryAdapter:
    def __init__(self):
        self.callback = None

    def register_post_delivery_callback(
        self,
        session_key,
        callback,
        *,
        generation=None,
    ):
        self.callback = callback


def _event(text="/세션"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="123",
            chat_id="456",
            user_name="tester",
        ),
    )


def _runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    store = _SessionStore()
    runner.session_store = store
    runner._async_session_store = store
    runner._normalize_source_for_session_key = lambda source: source
    runner._session_key_for_source = lambda source: "telegram:456:123"
    runner._evict_cached_agent = MagicMock()
    runner._try_send_choice_picker = AsyncMock(return_value=True)
    return runner


def test_korean_session_command_is_parsed_and_resolved():
    from hermes_cli.commands import resolve_command

    event = _event("/세션 새로고침")
    assert event.get_command() == "세션"
    assert event.get_command_args() == "새로고침"
    assert resolve_command(event.get_command()).name == "codex-session"


@pytest.mark.parametrize("mode", ["native", "off", "invalid-value", None])
def test_gateway_hygiene_leaves_native_codex_context_in_place(mode):
    assert gateway_run._gateway_hygiene_delegates_to_codex(
        "codex_app_server",
        mode,
    )


def test_gateway_hygiene_can_use_explicit_hermes_compaction():
    assert not gateway_run._gateway_hygiene_delegates_to_codex(
        "codex_app_server",
        "hermes",
    )
    assert not gateway_run._gateway_hygiene_delegates_to_codex(
        "chat_completions",
        "native",
    )


def test_gateway_result_preserves_codex_thread_and_turn_identity():
    assert gateway_run._codex_result_identity({
        "codex_thread_id": "thread-real",
        "codex_turn_id": "turn-real",
        "final_response": "done",
    }) == {
        "codex_thread_id": "thread-real",
        "codex_turn_id": "turn-real",
    }
    assert gateway_run._codex_result_identity(None) == {
        "codex_thread_id": None,
        "codex_turn_id": None,
    }


@pytest.mark.asyncio
async def test_selected_codex_thread_resolves_to_independent_execution_lane():
    runner = _runner()
    runner._session_key_for_source = build_session_key
    event = _event("첫 번째 세션에서 작업해")
    control_key = build_session_key(event.source)
    binding = {
        "thread_id": "desktop-thread-a",
        "title": "Desktop A",
    }
    runner.async_session_store.metadata[
        (control_key, "codex_desktop_thread")
    ] = binding

    resolved = await runner._resolve_codex_execution_source(event)

    assert resolved.session_lane == "desktop-thread-a"
    assert resolved.chat_id == event.source.chat_id
    assert resolved.thread_id == event.source.thread_id
    assert build_session_key(resolved) != control_key
    assert event.metadata["codex_desktop_binding_snapshot"] == binding
    assert CODEX_DELIVERY_GRANT_METADATA_KEY in event.metadata
    assert runner._codex_delivery_authorized_for_event(
        MessageEvent(
            text=event.text,
            source=resolved,
            metadata=event.metadata,
        )
    )


@pytest.mark.asyncio
async def test_session_picker_command_always_uses_control_lane():
    runner = _runner()
    runner._session_key_for_source = build_session_key
    event = _event("/세션")
    event.source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="123",
        chat_id="456",
        user_name="tester",
        session_lane="desktop-thread-a",
    )
    event.metadata["codex_desktop_binding_snapshot"] = {
        "thread_id": "desktop-thread-a"
    }
    event.metadata[CODEX_DELIVERY_GRANT_METADATA_KEY] = {
        "control_session_key": "telegram:456:123",
        "generation": 1,
        "thread_id": "desktop-thread-a",
    }

    resolved = await runner._resolve_codex_execution_source(event)

    assert resolved.session_lane is None
    assert "codex_desktop_binding_snapshot" not in event.metadata
    assert CODEX_DELIVERY_GRANT_METADATA_KEY not in event.metadata


@pytest.mark.asyncio
async def test_switch_revokes_old_lane_and_aba_does_not_reauthorize_it(
    monkeypatch,
):
    threads = [
        CodexThreadSummary(
            thread_id="thread-a",
            title="Desktop A",
            cwd="/work/a",
            updated_at=2,
            status="active",
        ),
        CodexThreadSummary(
            thread_id="thread-b",
            title="Desktop B",
            cwd="/work/b",
            updated_at=1,
            status="notLoaded",
        ),
    ]
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"model": {"openai_runtime": "codex_app_server"}},
    )
    import agent.transports.codex_app_server_session as session_module

    monkeypatch.setattr(
        session_module,
        "list_recent_codex_desktop_threads",
        lambda *, limit: threads[:limit],
    )
    import agent.transports.codex_desktop_mirror as mirror_module

    monkeypatch.setattr(
        mirror_module,
        "snapshot_codex_rollout_latest",
        lambda thread_id, *, hinted_path=None: (None, None),
    )

    runner = _runner()
    runner._session_key_for_source = build_session_key
    control_source = _event("control").source
    control_key = build_session_key(control_source)
    runner.async_session_store.metadata[
        (control_key, "codex_desktop_thread")
    ] = {"thread_id": "thread-a", "title": "Desktop A"}

    old_event = _event("keep working in A")
    old_event.source = await runner._resolve_codex_execution_source(old_event)
    assert runner._codex_delivery_authorized_for_event(old_event)

    assert "연결됨" in await runner._handle_codex_session_command(
        _event("/세션 2")
    )
    assert not runner._codex_delivery_authorized_for_event(old_event)

    assert "연결됨" in await runner._handle_codex_session_command(
        _event("/세션 1")
    )
    assert not runner._codex_delivery_authorized_for_event(old_event)

    new_event = _event("new work in A")
    new_event.source = await runner._resolve_codex_execution_source(new_event)
    assert runner._codex_delivery_authorized_for_event(new_event)
    assert (
        new_event.metadata[CODEX_DELIVERY_GRANT_METADATA_KEY]["generation"]
        > old_event.metadata[CODEX_DELIVERY_GRANT_METADATA_KEY]["generation"]
    )


@pytest.mark.asyncio
async def test_picker_offers_exactly_five_recent_threads_and_persists_selection(
    monkeypatch,
):
    threads = [
        CodexThreadSummary(
            thread_id=f"thread-{index}",
            title=f"Desktop conversation {index}",
            cwd=f"/work/project-{index}",
            updated_at=100 - index,
            status="active" if index == 1 else "notLoaded",
        )
        for index in range(1, 6)
    ]
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"model": {"openai_runtime": "codex_app_server"}},
    )
    import agent.transports.codex_app_server_session as session_module

    monkeypatch.setattr(
        session_module,
        "list_recent_codex_desktop_threads",
        lambda *, limit: threads[:limit],
    )
    import agent.transports.codex_desktop_mirror as mirror_module

    monkeypatch.setattr(
        mirror_module,
        "snapshot_codex_rollout_latest",
        lambda thread_id, *, hinted_path=None: (
            f"/tmp/rollout-{thread_id}.jsonl",
            CodexDesktopCompletion(
                turn_id=f"latest-{thread_id}",
                final_text="previous desktop answer " + "끝" * 3000,
                client_id=None,
            ),
        ),
    )
    runner = _runner()

    result = await runner._handle_codex_session_command(_event())

    assert result is None
    picker_kwargs = runner._try_send_choice_picker.await_args.kwargs
    choices = picker_kwargs["choices"]
    assert [choice["value"] for choice in choices] == [
        "thread-1",
        "thread-2",
        "thread-3",
        "thread-4",
        "thread-5",
    ]
    assert all(choice["full_width"] for choice in choices)

    confirmation = await picker_kwargs["on_choice_selected"]("456", "thread-2")
    binding = runner.async_session_store.metadata[
        ("telegram:456:123", "codex_desktop_thread")
    ]
    assert binding["thread_id"] == "thread-2"
    assert binding["cwd"] == "/work/project-2"
    assert binding["mirror_cursor_turn_id"] == "latest-thread-2"
    assert "연결됨" in confirmation
    assert "🧾 마지막 답변" in confirmation
    assert "previous desktop answer" in confirmation
    assert confirmation.endswith("끝" * 3000)
    assert "일부 생략" not in confirmation
    runner._evict_cached_agent.assert_called_once_with("telegram:456:123")


@pytest.mark.asyncio
async def test_typed_off_clears_persisted_binding(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"model": {"openai_runtime": "codex_app_server"}},
    )
    runner = _runner()
    key = ("telegram:456:123", "codex_desktop_thread")
    runner.async_session_store.metadata[key] = {"thread_id": "thread-1"}

    result = await runner._handle_codex_session_command(_event("/세션 해제"))

    assert runner.async_session_store.metadata[key] is None
    assert "해제" in result
    runner._evict_cached_agent.assert_called_once_with("telegram:456:123")


@pytest.mark.asyncio
async def test_run_agent_forwards_codex_binding_without_session_entry_lookup():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = MagicMock(multiplex_profiles=False)
    runner._run_agent_inner = AsyncMock(return_value={"final_response": "ok"})
    source = _event("hello").source

    result = await gateway_run.GatewayRunner._run_agent(
        runner,
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="session-1",
        session_key="telegram:456:123",
        codex_resume_thread_id="desktop-thread-1",
    )

    assert result == {"final_response": "ok"}
    assert (
        runner._run_agent_inner.await_args.kwargs["codex_resume_thread_id"]
        == "desktop-thread-1"
    )


@pytest.mark.asyncio
async def test_new_session_promotes_without_rollout_index(monkeypatch):
    import agent.transports.codex_desktop_mirror as mirror_module

    monkeypatch.setattr(
        mirror_module,
        "resolve_codex_rollout_path",
        lambda _thread_id: None,
    )
    runner = _runner()
    runner._session_key_for_source = build_session_key
    adapter = _PostDeliveryAdapter()
    runner._adapter_for_source = lambda _source: adapter
    runner._migrate_codex_execution_overrides = AsyncMock()
    runner._reset_codex_desktop_mirror_state = MagicMock()
    source = _event("first prompt").source
    control_key = build_session_key(source)
    placeholder = "pending_ns_test"
    binding = {
        "thread_id": placeholder,
        "pending_new": True,
        "title": "새 대화",
    }
    runner.async_session_store.metadata[
        (control_key, "codex_desktop_thread")
    ] = binding

    await runner._stage_codex_pending_thread_promotion(
        source=source,
        execution_session_key=build_session_key(
            SessionSource(
                platform=source.platform,
                user_id=source.user_id,
                chat_id=source.chat_id,
                chat_type=source.chat_type,
                session_lane=placeholder,
            )
        ),
        pending_binding=binding,
        actual_thread_id="codex-thread-real",
        actual_turn_id="codex-turn-first",
        run_generation=7,
    )

    staged = runner.async_session_store.metadata[
        (control_key, "codex_desktop_thread")
    ]
    assert staged["created_thread_id"] == "codex-thread-real"
    assert staged["created_turn_id"] == "codex-turn-first"
    assert "created_rollout_path" not in staged
    assert adapter.callback is not None

    await adapter.callback()

    promoted = runner.async_session_store.metadata[
        (control_key, "codex_desktop_thread")
    ]
    assert promoted["thread_id"] == "codex-thread-real"
    assert "pending_new" not in promoted
    assert runner.async_session_store.promotions == [
        (
            build_session_key(
                SessionSource(
                    platform=source.platform,
                    user_id=source.user_id,
                    chat_id=source.chat_id,
                    chat_type=source.chat_type,
                    session_lane=placeholder,
                )
            ),
            "codex-thread-real",
        )
    ]
