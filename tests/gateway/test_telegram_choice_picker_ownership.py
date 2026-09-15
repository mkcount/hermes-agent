"""Choice picker ownership is per message and atomically single-use."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult, utf16_len
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter():
    adapter = object.__new__(TelegramAdapter)
    adapter._choice_picker_state = {}
    adapter._callback_authorized = AsyncMock(return_value=True)
    adapter._callback_ctx = lambda _query: None
    adapter._edit_result_text = AsyncMock(return_value=True)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="11"))
    return adapter


@pytest.mark.asyncio
async def test_two_picker_messages_in_one_chat_do_not_overwrite_each_other():
    adapter = _adapter()
    first = AsyncMock(return_value="first selected")
    second = AsyncMock(return_value="second selected")
    expires = time.monotonic() + 60
    adapter._choice_picker_state = {
        "chat:10": {"choices": [{"value": "a"}], "on_choice_selected": first, "expires_at": expires},
        "chat:11": {"choices": [{"value": "b"}], "on_choice_selected": second, "expires_at": expires},
    }
    query = SimpleNamespace(message=SimpleNamespace(message_id=10), answer=AsyncMock())

    await adapter._handle_choice_picker_callback(query, "cp:0", "chat")

    first.assert_awaited_once_with("chat", "a")
    second.assert_not_awaited()
    assert "chat:10" not in adapter._choice_picker_state
    assert "chat:11" in adapter._choice_picker_state


@pytest.mark.asyncio
async def test_double_tap_is_claimed_before_callback_runs():
    adapter = _adapter()
    callback = AsyncMock(return_value="selected")
    adapter._choice_picker_state["chat:10"] = {
        "choices": [{"value": "a"}],
        "on_choice_selected": callback,
        "expires_at": time.monotonic() + 60,
    }
    first_query = SimpleNamespace(message=SimpleNamespace(message_id=10), answer=AsyncMock())
    second_query = SimpleNamespace(message=SimpleNamespace(message_id=10), answer=AsyncMock())

    await adapter._handle_choice_picker_callback(first_query, "cp:0", "chat")
    await adapter._handle_choice_picker_callback(second_query, "cp:0", "chat")

    callback.assert_awaited_once()
    second_query.answer.assert_awaited_once_with(text="Picker expired — run the command again.")


@pytest.mark.asyncio
async def test_simultaneous_taps_crossing_authorization_apply_once():
    adapter = _adapter()
    both_authorizing = asyncio.Event()
    authorization_count = 0

    async def authorize(*_args, **_kwargs):
        nonlocal authorization_count
        authorization_count += 1
        if authorization_count == 2:
            both_authorizing.set()
        await both_authorizing.wait()
        return True

    adapter._callback_authorized = authorize
    callback = AsyncMock(return_value="selected")
    adapter._choice_picker_state["chat:10"] = {
        "choices": [{"value": "a"}],
        "on_choice_selected": callback,
        "expires_at": time.monotonic() + 60,
    }
    first_query = SimpleNamespace(message=SimpleNamespace(message_id=10), answer=AsyncMock())
    second_query = SimpleNamespace(message=SimpleNamespace(message_id=10), answer=AsyncMock())

    await asyncio.gather(
        adapter._handle_choice_picker_callback(first_query, "cp:0", "chat"),
        adapter._handle_choice_picker_callback(second_query, "cp:0", "chat"),
    )

    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_long_picker_result_is_acknowledged_and_sent_via_chunking_path():
    adapter = _adapter()
    result_text = "x" * (adapter.MAX_MESSAGE_LENGTH + 1)
    callback = AsyncMock(return_value=result_text)
    adapter._choice_picker_state["chat:10"] = {
        "choices": [{"value": "a"}],
        "on_choice_selected": callback,
        "expires_at": time.monotonic() + 60,
    }
    query = SimpleNamespace(
        message=SimpleNamespace(message_id=10, message_thread_id=77),
        answer=AsyncMock(),
    )

    await adapter._handle_choice_picker_callback(query, "cp:0", "chat")

    edited_text = adapter._edit_result_text.await_args.args[1]
    assert utf16_len(edited_text) <= adapter.MAX_MESSAGE_LENGTH
    adapter.send.assert_awaited_once_with(
        "chat", result_text, metadata={"thread_id": "77"},
    )
    query.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_failed_picker_edit_falls_back_to_a_new_message():
    adapter = _adapter()
    adapter._edit_result_text.return_value = False
    callback = AsyncMock(return_value="selected")
    adapter._choice_picker_state["chat:10"] = {
        "choices": [{"value": "a"}],
        "on_choice_selected": callback,
        "expires_at": time.monotonic() + 60,
    }
    query = SimpleNamespace(
        message=SimpleNamespace(message_id=10, message_thread_id=None),
        answer=AsyncMock(),
    )

    await adapter._handle_choice_picker_callback(query, "cp:0", "chat")

    adapter.send.assert_awaited_once_with("chat", "selected", metadata=None)
    query.answer.assert_awaited_once_with()
