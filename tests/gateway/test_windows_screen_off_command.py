from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.commands import resolve_command, telegram_menu_commands


def _event(text: str = "/sc", *, chat_type: str = "dm") -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="8775784529",
            chat_type=chat_type,
            user_id="8775784529",
        ),
    )


def test_screen_off_shortcut_is_exact_owner_dm_shape():
    assert GatewayRunner._is_windows_screen_off_request(_event())
    assert GatewayRunner._is_windows_screen_off_request(_event("  /SC  "))
    assert not GatewayRunner._is_windows_screen_off_request(_event("/sc now"))
    assert not GatewayRunner._is_windows_screen_off_request(
        _event("/sc", chat_type="group")
    )


def test_screen_off_is_registered_as_visible_telegram_command():
    command = resolve_command("sc")
    assert command is not None
    assert command.gateway_only is True
    menu, _hidden = telegram_menu_commands(max_commands=60)
    assert ("sc", command.description) in menu


@pytest.mark.asyncio
async def test_screen_off_executor_runs_interactive_windows_task(monkeypatch):
    captured = {}

    class _Process:
        returncode = 0

        async def communicate(self):
            return b"SUCCESS", b""

    async def _create(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Process()

    monkeypatch.setattr(
        "gateway.run.asyncio.create_subprocess_exec",
        _create,
    )
    runner = object.__new__(GatewayRunner)

    result = await runner._run_windows_screen_off_command()

    assert "절전 모드" in result
    assert captured["args"][-5:] == (
        "windows-local",
        "schtasks.exe",
        "/Run",
        "/TN",
        "HermesScreenOff",
    )


@pytest.mark.asyncio
async def test_screen_off_shortcut_bypasses_busy_agent_queue():
    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(_send_with_retry=AsyncMock())
    runner._is_user_authorized = lambda _source: True
    runner._adapter_for_source = lambda _source: adapter
    runner._run_windows_screen_off_command = AsyncMock(
        return_value="🖥️ 두 모니터를 절전 모드로 전환했습니다."
    )
    runner._reply_anchor_for_event = lambda _event: "message-1"
    runner._thread_metadata_for_source = lambda _source, _reply: None

    handled = await runner._handle_active_session_busy_message(
        _event(),
        "telegram-session",
    )

    assert handled is True
    runner._run_windows_screen_off_command.assert_awaited_once()
    adapter._send_with_retry.assert_awaited_once()
    assert adapter._send_with_retry.await_args.kwargs["reply_to"] == "message-1"
