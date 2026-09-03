"""Tests for the gateway interactive choice picker (/reasoning, /fast).

The picker mirrors the /model picker architecture: the gateway gates on the
adapter *type* exposing ``send_choice_picker``, sends a flat choice list, and
falls back to the text status card when the platform has no picker or the
send fails. Selection flows through the same application path as the typed
command, so picker and typed arguments can never diverge.
"""

import asyncio
import dataclasses
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionSource


def _make_event(text="/reasoning", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


class _PickerAdapter:
    """Adapter whose *type* exposes ``send_choice_picker`` (the gate the
    handler checks via ``getattr(type(adapter), 'send_choice_picker', None)``)."""

    def __init__(self, success=True):
        self.calls = []
        self._success = success

    async def send_choice_picker(self, **kwargs):
        self.calls.append(kwargs)
        return SendResult(success=self._success, message_id="m1")


class _NoPickerAdapter:
    """Adapter with no choice-picker capability."""


def _make_runner(adapter=None):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_reasoning_overrides = {}
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._adapter_for_source = lambda source: adapter
    runner._thread_metadata_for_source = lambda source, anchor=None: {}
    runner._reply_anchor_for_event = lambda event: None
    return runner


class TestReasoningChoicePicker:
    @pytest.mark.asyncio
    async def test_bare_reasoning_sends_picker_when_adapter_supports_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)

        result = await runner._handle_reasoning_command(_make_event("/reasoning"))

        assert result is None  # picker sent — adapter owns the response
        assert len(adapter.calls) == 1
        call = adapter.calls[0]
        values = [c["value"] for c in call["choices"]]
        # Full canonical ladder + none + subcommands, in order
        from hermes_constants import VALID_REASONING_EFFORTS
        assert values[0] == "none"
        assert values[1:1 + len(VALID_REASONING_EFFORTS)] == list(VALID_REASONING_EFFORTS)
        assert values[-3:] == ["reset", "show", "hide"]

    @pytest.mark.asyncio
    async def test_bare_reasoning_falls_back_to_text_without_picker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner(_NoPickerAdapter())

        result = await runner._handle_reasoning_command(_make_event("/reasoning"))

        assert isinstance(result, str)
        assert "/reasoning" in result  # text status card

    @pytest.mark.asyncio
    async def test_bare_reasoning_falls_back_to_text_when_picker_send_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        adapter = _PickerAdapter(success=False)
        runner = _make_runner(adapter)

        result = await runner._handle_reasoning_command(_make_event("/reasoning"))

        assert isinstance(result, str)
        assert len(adapter.calls) == 1  # attempted, then fell back

    @pytest.mark.asyncio
    async def test_typed_argument_never_sends_picker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)

        result = await runner._handle_reasoning_command(_make_event("/reasoning high"))

        assert isinstance(result, str)
        assert adapter.calls == []

    @pytest.mark.asyncio
    async def test_picker_selection_applies_same_as_typed(self, tmp_path, monkeypatch):
        """The picker's on_choice_selected must produce the identical state
        change as typing the argument (single application path)."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)
        event = _make_event("/reasoning")
        session_key = runner._session_key_for_source(event.source)

        await runner._handle_reasoning_command(event)
        on_choice = adapter.calls[0]["on_choice_selected"]

        reply = await on_choice(event.source.chat_id, "ultra")

        assert "ultra" in reply
        override = runner._session_reasoning_overrides.get(session_key)
        assert override == {"enabled": True, "effort": "ultra"}

    @pytest.mark.asyncio
    async def test_picker_selection_of_current_level_marks_is_current(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"agent": {"reasoning_effort": "xhigh"}}), encoding="utf-8"
        )
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)

        await runner._handle_reasoning_command(_make_event("/reasoning"))

        current = [c["value"] for c in adapter.calls[0]["choices"] if c.get("is_current")]
        assert current == ["xhigh"]

    @pytest.mark.asyncio
    async def test_bound_codex_picker_uses_rollout_model_and_effort(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": {
                        "default": "gpt-config-default",
                        "openai_runtime": "codex_app_server",
                    },
                    "agent": {"reasoning_effort": "medium"},
                }
            ),
            encoding="utf-8",
        )
        thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
        rollout = (
            tmp_path
            / "sessions"
            / "2026"
            / "09"
            / "03"
            / f"rollout-{thread_id}.jsonl"
        )
        rollout.parent.mkdir(parents=True)
        rollout.write_text(
            json.dumps(
                {
                    "type": "turn_context",
                    "payload": {
                        "model": "gpt-desktop-current",
                        "effort": "xhigh",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))
        monkeypatch.setattr(
            "agent.transports.codex_app_server_session."
            "list_codex_model_reasoning_efforts",
            lambda model: ["low", "medium", "high", "xhigh"],
        )
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)
        event = _make_event("/reasoning")
        event.source = dataclasses.replace(
            event.source, session_lane=thread_id
        )
        event.metadata["codex_desktop_binding_snapshot"] = {
            "thread_id": thread_id,
            "rollout_path": str(rollout),
        }

        result = await runner._handle_reasoning_command(event)

        assert result is None
        current = [
            choice["value"]
            for choice in adapter.calls[0]["choices"]
            if choice.get("is_current")
        ]
        assert current == ["xhigh"]
        assert "xhigh" in adapter.calls[0]["title"]
        assert "session" in adapter.calls[0]["title"].lower()

    @pytest.mark.asyncio
    async def test_picker_show_choice_toggles_display(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)
        event = _make_event("/reasoning")

        await runner._handle_reasoning_command(event)
        on_choice = adapter.calls[0]["on_choice_selected"]
        await on_choice(event.source.chat_id, "show")

        assert runner._show_reasoning is True
        saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        assert saved["display"]["platforms"]["telegram"]["show_reasoning"] is True


class TestFastChoicePicker:
    def _patch_fast_support(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda cfg: "gpt-5.6")
        import hermes_cli.models as models_mod
        monkeypatch.setattr(models_mod, "model_supports_fast_mode", lambda m: True)

    @pytest.mark.asyncio
    async def test_bare_fast_sends_picker_when_adapter_supports_it(self, tmp_path, monkeypatch):
        self._patch_fast_support(monkeypatch, tmp_path)
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)

        result = await runner._handle_fast_command(_make_event("/fast"))

        assert result is None
        values = [c["value"] for c in adapter.calls[0]["choices"]]
        assert values == ["fast", "normal"]

    @pytest.mark.asyncio
    async def test_fast_picker_selection_is_session_scoped(self, tmp_path, monkeypatch):
        """A bare /fast picker tap applies a session override, not a config write."""
        self._patch_fast_support(monkeypatch, tmp_path)
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)
        event = _make_event("/fast")

        await runner._handle_fast_command(event)
        on_choice = adapter.calls[0]["on_choice_selected"]
        await on_choice(event.source.chat_id, "fast")

        assert runner._service_tier == "priority"
        assert runner._session_service_tier_overrides
        assert not (tmp_path / "config.yaml").exists()

    @pytest.mark.asyncio
    async def test_fast_picker_global_flag_persists_service_tier(self, tmp_path, monkeypatch):
        """A /fast --global picker tap persists agent.service_tier to config."""
        self._patch_fast_support(monkeypatch, tmp_path)
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)
        event = _make_event("/fast --global")

        await runner._handle_fast_command(event)
        on_choice = adapter.calls[0]["on_choice_selected"]
        await on_choice(event.source.chat_id, "fast")

        assert runner._service_tier == "priority"
        saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        assert saved["agent"]["service_tier"] == "fast"

    @pytest.mark.asyncio
    async def test_bare_fast_falls_back_to_text_without_picker(self, tmp_path, monkeypatch):
        self._patch_fast_support(monkeypatch, tmp_path)
        runner = _make_runner(_NoPickerAdapter())

        result = await runner._handle_fast_command(_make_event("/fast"))

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_typed_fast_argument_never_sends_picker(self, tmp_path, monkeypatch):
        self._patch_fast_support(monkeypatch, tmp_path)
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)

        result = await runner._handle_fast_command(_make_event("/fast normal"))

        assert isinstance(result, str)
        assert adapter.calls == []
