"""Gateway-facing orchestration for the Telegram ↔ Codex bridge.

The generic gateway only sees a trusted lane and ordinary MessageEvents.  All
Codex binding, input ownership, rollout cursors, and delivery arbitration stay
behind this mixin so neither Telegram nor the main inbound pipeline becomes a
second Codex implementation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

from gateway.codex_bridge.catalog import (
    CodexProjectSummary,
    CodexThreadSummary,
    list_recent_projects,
    list_recent_threads,
)
from gateway.codex_bridge.handoff import read_turn_handoff
from gateway.codex_bridge.rollout import RolloutEvent, RolloutTail, inspect_rollout, resolve_rollout_path
from gateway.codex_bridge.store import (
    CodexBridgeBinding,
    CodexBridgeStore,
    DurableCodexInput,
    DurableCodexProgress,
)
from gateway.config import Platform
from gateway.platforms.base import SessionRouteRejected
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")

_CONTROL_KEY = "codex_bridge_control_key"
_THREAD_KEY = "codex_bridge_thread_id"
_GENERATION_KEY = "codex_bridge_generation"
_INPUT_KEY = "codex_bridge_input_id"
_LANE_KEY = "codex_bridge_lane_key"
_SELECTION_COMMANDS = frozenset({"codex-session", "ns"})
_MIRROR_PROGRESS_MAX_SEGMENTS = 8
# Leave ample room for Telegram's Markdown escaping while keeping one editable
# message below its 4,096 UTF-16-unit ceiling.
_MIRROR_PROGRESS_MAX_CHARS = 1_800


class GatewayCodexBridgeMixin:
    """Independent Codex session/control plane used by ``GatewayRunner``."""

    def _init_codex_bridge(self) -> None:
        self._codex_bridge_stores: dict[str, CodexBridgeStore] = {}
        self._codex_bridge_tails: dict[tuple[str, int], RolloutTail] = {}
        self._codex_bridge_binding_locks: dict[str, asyncio.Lock] = {}
        # Create/recover the primary authority before any adapter can accept an
        # event. Secondary profile stores are recovered lazily on first use.
        self._codex_bridge_store_for_source(None)

    def _codex_bridge_binding_lock(self, control_key: str) -> asyncio.Lock:
        locks = getattr(self, "_codex_bridge_binding_locks", None)
        if locks is None:
            locks = self._codex_bridge_binding_locks = {}
        return locks.setdefault(control_key, asyncio.Lock())

    def _codex_bridge_home_for_source(self, source: Optional[SessionSource]) -> Path:
        if source is not None and getattr(source, "profile", None) not in {None, "", "default"}:
            with suppress(Exception):
                return Path(self._resolve_profile_home_for_source(source)).expanduser().resolve()
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()).expanduser().resolve()

    def _codex_bridge_store_for_source(self, source: Optional[SessionSource]) -> CodexBridgeStore:
        stores = getattr(self, "_codex_bridge_stores", None)
        if stores is None:
            stores = self._codex_bridge_stores = {}
        home = self._codex_bridge_home_for_source(source)
        key = str(home)
        store = stores.get(key)
        if store is None:
            store = CodexBridgeStore(home / "state.db")
            store.recover_after_restart()
            stores[key] = store
        return store

    async def _codex_bridge_ledger_call(self, source: SessionSource, function: Any, /, *args, **kwargs):
        """Run one delivery-ledger operation against the source profile's state DB."""
        home = self._codex_bridge_home_for_source(source)

        def _call():
            from gateway.run import _profile_runtime_scope

            with _profile_runtime_scope(home, hydrate_secrets=False):
                return function(*args, **kwargs)

        return await asyncio.to_thread(_call)

    @staticmethod
    def _codex_bridge_control_source(source: SessionSource) -> SessionSource:
        return dataclasses.replace(source, trusted_local_lane=None)

    def _codex_bridge_control_key(self, source: SessionSource) -> str:
        return self._session_key_for_source(self._codex_bridge_control_source(source))

    @staticmethod
    def _codex_bridge_canonical_command(event: MessageEvent) -> Optional[str]:
        command = event.get_command()
        if not command:
            return None
        with suppress(Exception):
            from hermes_cli.commands import resolve_command

            definition = resolve_command(command)
            if definition is not None:
                return definition.name
        return command

    async def _codex_bridge_migrate_legacy_binding(
        self, store: CodexBridgeStore, control_key: str, source: SessionSource,
    ) -> Optional[CodexBridgeBinding]:
        """One-way import of the old SessionStore metadata representation."""
        session_store = getattr(self, "session_store", None)
        lookup = getattr(session_store, "lookup_by_session_key", None)
        if not callable(lookup):
            return None
        entry = lookup(control_key)
        legacy = entry.metadata.get("codex_desktop_thread") if entry is not None else None
        if not isinstance(legacy, dict):
            return None
        thread_id = str(legacy.get("thread_id") or "").strip()
        if not thread_id:
            return None
        pending = bool(legacy.get("pending_new"))
        snapshot = None if pending else await asyncio.to_thread(
            inspect_rollout,
            thread_id,
            hinted_path=str(legacy.get("rollout_path") or "") or None,
        )
        cursor = (
            snapshot.active_start_offset
            if snapshot is not None and snapshot.active_start_offset is not None
            else snapshot.size if snapshot is not None else 0
        )
        binding = store.bind(
            control_key, self._codex_bridge_control_source(source), thread_id=thread_id,
            cwd=str(legacy.get("cwd") or ""), pending_new=pending,
            rollout_path=snapshot.path if snapshot else None,
            cursor_device=snapshot.device if snapshot else None,
            cursor_inode=snapshot.inode if snapshot else None,
            cursor_offset=cursor,
        )
        logger.info("Migrated legacy Codex binding for %s to generation %s", control_key, binding.generation)
        return binding

    async def _resolve_codex_bridge_route(self, adapter: Any, event: MessageEvent) -> MessageEvent:
        """Resolve a Telegram DM before the adapter claims its serialization key.

        Any authority/storage failure raises to BasePlatformAdapter, whose
        trusted-router boundary fails closed instead of executing in Hermes.
        """
        source = event.source
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return event
        control_source = self._codex_bridge_control_source(source)
        # This seam necessarily runs before the adapter's serialization guard,
        # while the runner's normal authorization gate runs inside the spawned
        # task. Never create even a pre-admission durable row for a sender the
        # adapter cannot positively authorize; the runner still owns pairing.
        authorized = adapter._is_sender_authorized(
            source.user_id,
            source.chat_type,
            source.chat_id,
            is_bot=bool(source.is_bot),
            thread_id=source.thread_id,
        )
        if authorized is not True:
            recovered_input_id = str((event.metadata or {}).get(_INPUT_KEY) or "").strip()
            if recovered_input_id:
                self._codex_bridge_store_for_source(control_source).cancel_input(
                    recovered_input_id, "sender is no longer authorized",
                )
            return dataclasses.replace(event, source=control_source)
        control_key = self._session_key_for_source(control_source)
        canonical = self._codex_bridge_canonical_command(event)
        if canonical in _SELECTION_COMMANDS:
            return dataclasses.replace(event, source=control_source)

        store = self._codex_bridge_store_for_source(control_source)
        async with self._codex_bridge_binding_lock(control_key):
            binding = store.get_binding(control_key)
            if binding is None:
                binding = await self._codex_bridge_migrate_legacy_binding(
                    store, control_key, control_source,
                )
        if binding is None or not binding.active:
            return dataclasses.replace(event, source=control_source)

        # The generation, rather than the thread id, is the lane identity. A
        # /ns placeholder can promote to its real id without splitting the
        # first conversation, while A→B→A still receives three distinct lanes.
        lane_source = dataclasses.replace(
            control_source,
            trusted_local_lane=f"codex-binding:{control_key}:{binding.generation}",
        )
        lane_key = adapter._event_session_key(dataclasses.replace(event, source=lane_source))
        metadata = dict(event.metadata or {})
        metadata.update({
            _CONTROL_KEY: control_key,
            _THREAD_KEY: binding.thread_id,
            _GENERATION_KEY: binding.generation,
            _LANE_KEY: lane_key,
        })

        # Slash commands manipulate the selected lane but are not Codex inputs.
        # Ordinary messages are persisted as non-executable routing records.
        # The runner atomically claims execution only after its authorization,
        # pause, plugin, and command gates have passed.
        if canonical is None:
            routed = dataclasses.replace(event, source=lane_source, metadata=metadata)
            replay_id = str(metadata.get(_INPUT_KEY) or "").strip()
            input_id, state, inserted = store.enqueue_input(binding, lane_key, routed)
            # A queued/startup-replayed event carries the bridge id we assigned
            # previously. A raw duplicate platform update does not, so reject
            # it without racing a second handler for the same durable input.
            if state not in {"routed", "pending"} or (not inserted and replay_id != input_id):
                raise SessionRouteRejected("")
            metadata[_INPUT_KEY] = input_id
        return dataclasses.replace(event, source=lane_source, metadata=metadata)

    def _codex_bridge_begin_input(self, event: MessageEvent) -> Optional[str]:
        metadata = event.metadata or {}
        input_id = str(metadata.get(_INPUT_KEY) or "").strip()
        if not input_id:
            if metadata.get(_CONTROL_KEY):
                return (
                    "Codex 연결 경로에서 이 메시지의 내구 실행 ID를 확인하지 못해 "
                    "실행하지 않았습니다. 일반 메시지로 다시 보내 주세요."
                )
            return None
        store = self._codex_bridge_store_for_source(event.source)
        binding = store.get_binding(str(metadata.get(_CONTROL_KEY) or ""))
        if (
            binding is None
            or binding.generation != int(metadata.get(_GENERATION_KEY) or 0)
            or not binding.active
        ):
            store.cancel_lane(str(metadata.get(_LANE_KEY) or ""))
            return "선택된 Codex 세션이 바뀌어 이 메시지는 실행하지 않았습니다. 다시 보내 주세요."
        if not store.mark_executing(input_id):
            return "이 Codex 메시지는 이미 처리되었거나 취소되었습니다."
        return None

    def _configure_codex_bridge_agent(self, agent: Any, ctx: Any) -> None:
        """Apply the explicit Telegram grant to one lane-scoped AIAgent."""
        control_key = str(getattr(ctx, "codex_bridge_control_key", "") or "")
        if not control_key:
            return
        store = self._codex_bridge_store_for_source(ctx.source)
        binding = store.get_binding(control_key)
        generation = int(getattr(ctx, "codex_bridge_generation", 0) or 0)
        if binding is None or binding.generation != generation or not binding.active:
            raise RuntimeError("Codex bridge binding changed before turn start")

        agent.api_mode = "codex_app_server"
        agent._gateway_codex_full_access = True
        agent._codex_resume_thread_id = None if binding.pending_new else binding.thread_id
        agent._codex_resume_active_turn_mode = "queue"
        agent._codex_client_user_message_id = getattr(ctx, "codex_bridge_input_id", None)
        input_id = str(agent._codex_client_user_message_id or "").strip()
        if binding.cwd:
            agent.session_cwd = binding.cwd

        state = self._peek_session_state(ctx.session_key)
        model_override = state.conversation.model_override if state is not None else None
        reasoning_override = state.conversation.reasoning_override if state is not None else None
        agent._codex_model_override = (
            str(model_override.get("model") or "").strip()
            if isinstance(model_override, dict) else None
        ) or None
        agent._codex_reasoning_effort_override = (
            str(reasoning_override.get("effort") or "").strip().lower()
            if isinstance(reasoning_override, dict) and reasoning_override.get("enabled", True)
            else None
        ) or None

        def _starting(_thread_id: str, _client_message_id: str) -> None:
            if input_id and not store.mark_submitting(input_id):
                raise RuntimeError("Codex bridge input lost ownership before turn/start")

        def _started(thread_id: str, turn_id: str) -> None:
            if binding.pending_new:
                promoted = store.promote_pending(binding, thread_id)
                if promoted is not None:
                    # The event metadata is shared with the loop thread. Only
                    # the generation authorizes delivery; this update improves
                    # diagnostics and subsequent callback state.
                    agent._codex_resume_thread_id = promoted.thread_id
            if input_id:
                store.mark_running(input_id, turn_id)

        agent._codex_turn_started_callback = _started
        agent._codex_turn_starting_callback = _starting

    async def _codex_bridge_finalize_input(
        self, event: MessageEvent, lane_key: str, result: Any, run_generation: int,
    ) -> Any:
        metadata = event.metadata or {}
        input_id = str(metadata.get(_INPUT_KEY) or "").strip()
        if not input_id:
            return result
        store = self._codex_bridge_store_for_source(event.source)
        binding = store.get_binding(str(metadata.get(_CONTROL_KEY) or ""))
        if binding is None or binding.generation != int(metadata.get(_GENERATION_KEY) or 0):
            store.mark_completed(input_id)
            return None

        transport = getattr(event, "_codex_bridge_agent_result", {})
        transport = transport if isinstance(transport, dict) else {}
        delivered_stream = str(getattr(event, "_streamed_final_response", "") or "").strip()
        current_state = store.input_state(input_id)
        codex_turn_id = str(
            transport.get("codex_turn_id") or store.input_turn_id(input_id) or ""
        ).strip()
        codex_thread_id = str(
            transport.get("codex_thread_id") or binding.thread_id or ""
        ).strip()
        handoff = read_turn_handoff(codex_thread_id, codex_turn_id)
        handoff_owns_continuation = handoff is not None and handoff.continues and (
            handoff.status != "interrupting" or bool(transport.get("interrupted"))
        )
        if (
            current_state in {"submitting", "running"}
            and not bool(transport.get("completed"))
            and not delivered_stream
            and handoff_owns_continuation
        ):
            assert handoff is not None
            # ReloginTool published continuation ownership before interrupting
            # this exact turn. The live request ends here, while the rollout
            # watcher remains the sole owner of its resumed progress and final.
            store.mark_continuation_pending(
                input_id,
                str(transport.get("error") or "Codex profile switch continuation pending"),
                continuation_turn_id=handoff.continued_turn_id,
            )
            return None
        if (
            current_state in {"submitting", "running"}
            and not bool(transport.get("completed"))
            and not delivered_stream
            and (not transport or bool(transport.get("codex_should_retire")))
        ):
            error = str(transport.get("error") or "Codex submission result unknown")
            store.mark_uncertain(input_id, error)
            return (
                "⚠️ Codex가 이 입력을 접수했는지 확인 중입니다. 같은 지시를 다시 보내면 "
                "중복 실행될 수 있어 자동 재실행하지 않습니다. 확인되는 최종 결과는 이 채팅으로 전달됩니다."
            )

        final_text = str(result or "").strip() if isinstance(result, str) else delivered_stream
        if not final_text:
            final_text = "Codex 작업이 답변을 만들기 전에 종료되었습니다. 같은 메시지를 다시 보내 주세요."
            result = final_text
        turn_outcome = (
            "completed" if bool(transport.get("completed"))
            else "interrupted" if bool(transport.get("interrupted"))
            else "failed" if transport.get("error")
            else "unknown"
        )
        if not store.mark_executed(input_id, final_text, turn_outcome=turn_outcome):
            return None

        # For ordinary (non-streamed) final delivery, persist the exact output
        # before returning to BasePlatformAdapter. Base records/attempts the
        # same stable id; this closes the handler-return → ledger-write gap.
        if not delivered_stream:
            with suppress(Exception):
                from gateway.delivery_ledger import compute_obligation_id, ensure_obligation

                message_ref = event.ledger_message_id or event.message_id or input_id
                obligation_id = compute_obligation_id(lane_key, str(message_ref), final_text)
                await self._codex_bridge_ledger_call(
                    event.source,
                    ensure_obligation,
                    obligation_id=obligation_id, session_key=lane_key,
                    platform=event.source.platform.value, chat_id=event.source.chat_id,
                    thread_id=event.source.thread_id, content=final_text,
                    adapter_profile=getattr(self._adapter_for_source(event.source), "_owner_profile", None),
                )

        adapter = self._adapter_for_source(event.source)
        if adapter is not None and hasattr(adapter, "register_post_delivery_callback"):
            adapter.register_post_delivery_callback(
                lane_key, lambda: store.mark_completed(input_id), generation=run_generation,
            )
        elif delivered_stream:
            store.mark_completed(input_id)
        return result

    def _codex_bridge_release_input(self, event: MessageEvent, error: str = "") -> None:
        input_id = str((event.metadata or {}).get(_INPUT_KEY) or "").strip()
        if input_id:
            self._codex_bridge_store_for_source(event.source).release_for_retry(input_id, error)

    def _codex_bridge_cancel_input(self, event: MessageEvent, error: str = "") -> None:
        input_id = str((event.metadata or {}).get(_INPUT_KEY) or "").strip()
        if input_id:
            self._codex_bridge_store_for_source(event.source).cancel_input(input_id, error)

    def _codex_bridge_cancel_lane(self, lane_key: str, source: SessionSource) -> int:
        return self._codex_bridge_store_for_source(source).cancel_lane(lane_key)

    def _codex_bridge_forget_binding_runtime(self, binding: CodexBridgeBinding) -> None:
        """Drop the process-local tail when a durable grant rotates."""
        key = (binding.control_session_key, binding.generation)
        self._codex_bridge_tails.pop(key, None)

    async def _codex_bridge_store_binding(
        self, source: SessionSource, summary: Optional[CodexThreadSummary],
    ) -> str:
        control_source = self._codex_bridge_control_source(source)
        control_key = self._session_key_for_source(control_source)
        await self.async_session_store.get_or_create_session(control_source)
        store = self._codex_bridge_store_for_source(control_source)
        snapshot = None
        if summary is not None:
            snapshot = await asyncio.to_thread(
                inspect_rollout, summary.thread_id, hinted_path=summary.rollout_path or None,
            )
        async with self._codex_bridge_binding_lock(control_key):
            previous = store.get_binding(control_key)
            if summary is None:
                store.unbind(control_key, control_source)
                if previous is not None:
                    self._codex_bridge_forget_binding_runtime(previous)
                    self._evict_cached_agent(self._codex_bridge_lane_key(control_source, previous))
                return "Codex 세션 연결을 해제했습니다. 이제 이 채팅은 일반 Hermes 세션을 사용합니다."

            # Re-selecting the active row in the picker must be a no-op.  A
            # gratuitous generation rotation would otherwise cancel queued
            # inputs even though the user did not actually change sessions.
            unchanged = bool(
                previous is not None
                and previous.active
                and not previous.pending_new
                and previous.thread_id == summary.thread_id
            )
            rotated = bool(previous is not None and previous.active and not unchanged)
            if unchanged:
                binding = previous
            else:
                cursor = (
                    snapshot.active_start_offset
                    if snapshot is not None and snapshot.active_start_offset is not None
                    else snapshot.size if snapshot is not None else 0
                )
                binding = store.bind(
                    control_key, control_source, thread_id=summary.thread_id, cwd=summary.cwd,
                    rollout_path=snapshot.path if snapshot else summary.rollout_path or None,
                    cursor_device=snapshot.device if snapshot else None,
                    cursor_inode=snapshot.inode if snapshot else None,
                    cursor_offset=cursor,
                )
                if previous is not None:
                    self._codex_bridge_forget_binding_runtime(previous)
                    self._evict_cached_agent(self._codex_bridge_lane_key(control_source, previous))
        answer = (
            f"{'✅ 이미 연결된' if unchanged else '✅ Codex 세션 연결됨:'} {summary.title[:100]}\n"
            "이 채팅의 일반 메시지는 승인 요청 없이 전체 액세스로 같은 Codex 세션에 이어집니다. "
            "데스크톱에서 시작한 진행 보고는 Telegram에 남는 메시지로 전달되고 새 보고 때 갱신되며, "
            "최종 답변은 별도 메시지로 전달됩니다. "
            "데스크톱 작업이 쓰기 권한을 사용 중이면 Telegram 입력은 그 작업이 끝날 때까지 순서대로 기다립니다."
        )
        answer += (
            "\n현재 Codex 턴이 진행 중이므로 이후 진행 보고부터 전달합니다."
            if snapshot is not None and snapshot.active_turn_id
            else "\n현재 실행 중인 Codex 턴은 없습니다. 새 턴이 시작되기 전에는 추가 보고가 없습니다."
        )
        answer += (
            "\n이 Telegram 토픽에는 한 세션만 연결되며, 다른 토픽은 별도로 연결할 수 있습니다."
            if source.thread_id
            else "\n이 Telegram 대화에는 한 세션만 연결됩니다. 새 세션을 고르면 기존 연결을 교체합니다."
        )
        if rotated:
            answer += "\n\n⚠️ 이전 연결에서 아직 실행·전송되지 않은 Telegram 작업은 취소됐습니다."
        if not unchanged and snapshot is not None and snapshot.latest_final_text:
            answer += f"\n\n🧾 마지막 답변\n\n{snapshot.latest_final_text}"
        logger.info("Bound Codex thread %s to %s generation=%s", summary.thread_id, control_key, binding.generation)
        return answer

    async def _codex_bridge_status(
        self, store: CodexBridgeStore, control_key: str, source: SessionSource,
    ) -> str:
        binding = store.get_binding(control_key)
        scope = "이 Telegram 토픽" if source.thread_id else "이 Telegram 대화"
        if binding is None or not binding.active:
            return (
                f"📱 Codex 연결 상태\n\n{scope}: 연결 없음\n"
                "/codex_session으로 최근 세션을 선택할 수 있습니다."
            )
        if binding.pending_new:
            return (
                f"📱 Codex 연결 상태\n\n{scope}: 새 세션의 첫 입력 대기 중\n"
                f"프로젝트: {binding.cwd or '-'}\n세대: {binding.generation}\n"
                "다음 일반 메시지가 새 Codex 세션의 첫 지시가 됩니다."
            )
        try:
            snapshot = await asyncio.to_thread(
                inspect_rollout, binding.thread_id, hinted_path=binding.rollout_path,
            )
        except Exception:
            logger.warning("Codex rollout status inspection failed", exc_info=True)
            snapshot = None
        progress = store.list_progress(binding)
        pending = [row for row in progress if row.state == "pending"]
        if snapshot is None:
            state = "rollout 파일을 찾지 못함"
            lag = "확인 불가"
        else:
            state = "턴 진행 중" if snapshot.active_turn_id else "대기/완료"
            same_incarnation = (
                binding.rollout_path == snapshot.path
                and binding.cursor_device == snapshot.device
                and binding.cursor_inode == snapshot.inode
            )
            cursor = binding.cursor_offset if same_incarnation else 0
            lag = f"{max(0, snapshot.size - cursor):,} bytes"
        lines = [
            "📱 Codex 연결 상태",
            "",
            f"범위: {scope}",
            f"세션: {binding.thread_id}",
            f"상태: {state}",
            f"프로젝트: {binding.cwd or '-'}",
            f"세대: {binding.generation}",
            f"rollout 미처리량: {lag}",
            f"진행 보고 outbox: 대기 {len(pending)} / 전체 {len(progress)}",
        ]
        if pending and pending[-1].last_error:
            lines.append(f"최근 전송 오류: {pending[-1].last_error[:300]}")
        lines.append(
            "연결 규칙: 현재 대화/토픽당 1개 세션이며, Telegram 토픽은 각각 독립적으로 연결됩니다."
        )
        return "\n".join(lines)

    def _codex_bridge_lane_key(self, source: SessionSource, binding: CodexBridgeBinding) -> str:
        lane = dataclasses.replace(
            self._codex_bridge_control_source(source),
            trusted_local_lane=f"codex-binding:{binding.control_session_key}:{binding.generation}",
        )
        return self._session_key_for_source(lane)

    async def _handle_codex_session_command(self, event: MessageEvent) -> Optional[str]:
        if event.source.platform != Platform.TELEGRAM or event.source.chat_type != "dm":
            return "이 명령은 Telegram 개인 대화에서만 사용할 수 있습니다."
        source = self._codex_bridge_control_source(event.source)
        store = self._codex_bridge_store_for_source(source)
        control_key = self._session_key_for_source(source)
        raw = event.get_command_args().strip()
        arg = raw.lower()
        if arg in {"off", "detach", "disconnect", "해제"}:
            return await self._codex_bridge_store_binding(source, None)
        if arg in {"status", "상태"}:
            return await self._codex_bridge_status(store, control_key, source)
        try:
            threads = await asyncio.to_thread(list_recent_threads, limit=8)
        except Exception:
            logger.warning("Codex thread listing failed", exc_info=True)
            return "최근 Codex 세션을 불러오지 못했습니다. Codex 로그인과 app-server 상태를 확인해 주세요."
        if not threads:
            return "연결 가능한 최근 Codex 세션이 없습니다."
        by_id = {item.thread_id: item for item in threads}
        if arg and arg not in {"refresh", "새로고침"}:
            selected = threads[int(arg) - 1] if arg.isdigit() and 1 <= int(arg) <= len(threads) else by_id.get(raw)
            if selected is None:
                return "사용법: /codex-session, /codex-session 1, /codex-session status, 또는 /codex-session off"
            return await self._codex_bridge_store_binding(source, selected)

        current = store.get_binding(control_key)
        current_id = current.thread_id if current and current.active else ""
        choices = []
        for index, summary in enumerate(threads, 1):
            folder = os.path.basename(summary.cwd.rstrip(os.sep)) if summary.cwd else ""
            active = str(summary.status).lower() in {"active", "inprogress", "in_progress"}
            label = f"{index}. {'● ' if active else ''}{summary.title}{' · ' + folder if folder else ''}"
            choices.append({
                "value": summary.thread_id, "label": label[:52],
                "is_current": summary.thread_id == current_id, "full_width": True,
            })
        if current_id:
            choices.append({"value": "off", "label": "✕ 연결 해제", "is_current": False, "full_width": True})

        async def _selected(_chat_id: str, value: str) -> str:
            if value == "off":
                return await self._codex_bridge_store_binding(source, None)
            summary = by_id.get(value)
            if summary is None:
                return "선택 항목이 만료됐습니다. /codex-session을 다시 실행해 주세요."
            return await self._codex_bridge_store_binding(source, summary)

        if await self._try_send_choice_picker(
            event, control_key,
            "📱 *Codex 세션 연결*\n\n이어갈 최근 세션을 선택하세요. ●는 진행 중인 세션입니다.",
            choices, _selected,
        ):
            return None
        lines = ["최근 Codex 세션:"] + [f"{i}. {row.title[:100]}" for i, row in enumerate(threads, 1)]
        lines.append("\n`/codex-session 1`처럼 번호로 선택할 수 있습니다.")
        return "\n".join(lines)

    async def _handle_ns_command(self, event: MessageEvent) -> Optional[str]:
        if event.source.platform != Platform.TELEGRAM or event.source.chat_type != "dm":
            return "이 명령은 Telegram 개인 대화에서만 사용할 수 있습니다."
        source = self._codex_bridge_control_source(event.source)
        raw = event.get_command_args().strip()
        arg = raw.lower()
        try:
            projects = await asyncio.to_thread(list_recent_projects, limit=10)
        except Exception:
            logger.warning("Codex project listing failed", exc_info=True)
            return "Codex 프로젝트 목록을 불러오지 못했습니다. Codex 로그인 상태를 확인해 주세요."
        if not projects:
            return "새 세션을 시작할 Codex 프로젝트가 없습니다. 먼저 Codex에서 프로젝트를 한 번 여세요."
        by_cwd = {item.cwd: item for item in projects}

        async def _reserve(project: CodexProjectSummary) -> str:
            control_key = self._session_key_for_source(source)
            await self.async_session_store.get_or_create_session(source)
            store = self._codex_bridge_store_for_source(source)
            async with self._codex_bridge_binding_lock(control_key):
                old = store.get_binding(control_key)
                placeholder = f"pending_ns_{uuid.uuid4().hex}"
                store.bind(
                    control_key, source, thread_id=placeholder, cwd=project.cwd, pending_new=True,
                )
                if old is not None:
                    self._codex_bridge_forget_binding_runtime(old)
                    self._evict_cached_agent(self._codex_bridge_lane_key(source, old))
            return (
                f"✅ 새 Codex 세션 준비됨: {project.name}\n"
                "다음 일반 메시지가 새 세션의 첫 지시가 됩니다. 그 턴도 승인 없이 전체 액세스로 실행됩니다."
                + (
                    "\n\n⚠️ 이전 연결에서 아직 실행·전송되지 않은 Telegram 작업은 취소됐습니다."
                    if old is not None and old.active else ""
                )
            )

        if arg and arg not in {"refresh", "새로고침"}:
            project = projects[int(arg) - 1] if arg.isdigit() and 1 <= int(arg) <= len(projects) else by_cwd.get(raw)
            return await _reserve(project) if project else "사용법: /ns 또는 /ns 1"
        choices = [
            {"value": item.cwd, "label": f"{index}. {item.name} · {item.cwd}"[:52],
             "is_current": False, "full_width": True}
            for index, item in enumerate(projects, 1)
        ]

        async def _selected(_chat_id: str, value: str) -> str:
            project = by_cwd.get(value)
            return await _reserve(project) if project else "선택 항목이 만료됐습니다. /ns를 다시 실행해 주세요."

        if await self._try_send_choice_picker(
            event, self._session_key_for_source(source),
            "🆕 *새 Codex 세션*\n\n새 대화를 시작할 프로젝트를 선택하세요.", choices, _selected,
        ):
            return None
        lines = ["새 Codex 세션을 시작할 프로젝트:"] + [
            f"{index}. {item.name} — {item.cwd}" for index, item in enumerate(projects, 1)
        ]
        lines.append("\n`/ns 1`처럼 번호로 선택할 수 있습니다.")
        return "\n".join(lines)

    async def _codex_bridge_recover_input(self, store: CodexBridgeStore, item: DurableCodexInput) -> None:
        binding = store.get_binding(item.control_session_key)
        if binding is None or binding.generation != item.generation or binding.thread_id != item.thread_id:
            store.cancel_lane(item.lane_session_key)
            return
        adapter = self._adapter_for_source(binding.source)
        if adapter is None:
            return
        metadata = dict(item.event.metadata or {})
        metadata[_INPUT_KEY] = item.input_id
        await adapter.handle_message(dataclasses.replace(item.event, metadata=metadata))

    async def _codex_bridge_recover_output(self, store: CodexBridgeStore, item: DurableCodexInput) -> None:
        adapter = self._adapter_for_source(item.event.source)
        if adapter is None or not item.final_text:
            return
        from gateway.delivery_ledger import (
            compute_obligation_id,
            ensure_obligation,
            mark_attempting,
            mark_delivered,
            mark_failed,
        )

        message_ref = item.event.ledger_message_id or item.event.message_id or item.input_id
        obligation_id = compute_obligation_id(item.lane_session_key, str(message_ref), item.final_text)
        # Startup redelivery may already have claimed or delivered this row.
        # INSERT OR IGNORE preserves that state; the bridge never becomes a
        # second sender for an output the shared ledger already owns.
        inserted = await self._codex_bridge_ledger_call(
            item.event.source,
            ensure_obligation,
            obligation_id=obligation_id, session_key=item.lane_session_key,
            platform=item.event.source.platform.value, chat_id=item.event.source.chat_id,
            thread_id=item.event.source.thread_id, content=item.final_text,
            adapter_profile=getattr(adapter, "_owner_profile", None),
        )
        # A row first reconstructed here did not exist during the startup
        # sweep, and no prior send was attempted. Deliver it once under the
        # same ledger id; pre-existing rows remain exclusively ledger-owned.
        if inserted:
            await self._codex_bridge_ledger_call(
                item.event.source, mark_attempting, obligation_id,
            )
            try:
                result = await asyncio.wait_for(
                    adapter.send(
                        item.event.source.chat_id, item.final_text,
                        metadata=self._thread_metadata_for_source(item.event.source),
                    ),
                    timeout=20.0,
                )
            except Exception as exc:
                await self._codex_bridge_ledger_call(
                    item.event.source, mark_failed, obligation_id, str(exc)[:500],
                )
            else:
                if getattr(result, "success", False):
                    await self._codex_bridge_ledger_call(
                        item.event.source, mark_delivered, obligation_id,
                    )
                else:
                    await self._codex_bridge_ledger_call(
                        item.event.source, mark_failed, obligation_id,
                        str(getattr(result, "error", ""))[:500],
                    )
        store.mark_completed(item.input_id)
        binding = store.get_binding(item.control_session_key)
        if (
            binding is not None
            and binding.generation == item.generation
            and binding.thread_id == item.thread_id
        ):
            store.complete_progress(binding, item.input_id)

    @staticmethod
    def _codex_bridge_mirror_identity(event: RolloutEvent) -> str:
        # Automatic Codex continuation turns inherit the originating client id
        # from the explicit turn_aborted→task_started boundary. Keep one
        # persistent progress message across that transition.
        return event.client_id or event.turn_id

    @staticmethod
    def _codex_bridge_stage_commentary(
        store: CodexBridgeStore, binding: CodexBridgeBinding, events: list[RolloutEvent],
    ) -> DurableCodexProgress:
        """Commit commentary to the outbox before acknowledging rollout bytes."""
        if not events:
            raise ValueError("events are required")
        event = events[-1]
        return store.upsert_progress(
            binding,
            GatewayCodexBridgeMixin._codex_bridge_mirror_identity(event),
            [candidate.text for candidate in events],
            max_segments=_MIRROR_PROGRESS_MAX_SEGMENTS,
            max_chars=_MIRROR_PROGRESS_MAX_CHARS,
        )

    async def _codex_bridge_deliver_progress(
        self, store: CodexBridgeStore, binding: CodexBridgeBinding,
        progress: DurableCodexProgress, adapter: Any,
    ) -> None:
        """Send once, then edit; failures stay pending in SQLite for restart recovery."""
        metadata = dict(self._thread_metadata_for_source(binding.source) or {})
        metadata["_interim_send"] = True
        try:
            if progress.message_id:
                result = await asyncio.wait_for(
                    adapter.edit_message(
                        binding.source.chat_id, progress.message_id,
                        progress.content, metadata=metadata,
                    ),
                    timeout=20.0,
                )
            else:
                result = await asyncio.wait_for(
                    adapter.send(
                        binding.source.chat_id, progress.content, metadata=metadata,
                    ),
                    timeout=20.0,
                )
        except Exception as exc:
            store.mark_progress_failed(progress, str(exc))
            raise
        if not getattr(result, "success", False):
            error = str(getattr(result, "error", "") or "progress delivery failed")
            if progress.message_id and getattr(result, "error_kind", None) == "not_found":
                store.clear_progress_message(progress, error)
            else:
                store.mark_progress_failed(progress, error)
            raise RuntimeError(f"Codex progress delivery failed: {error}")
        message_id = getattr(result, "message_id", None) or progress.message_id
        if not store.mark_progress_delivered(progress, message_id):
            raise RuntimeError("Codex progress changed while delivery was in flight")

    async def _codex_bridge_retry_progress(
        self, store: CodexBridgeStore, binding: CodexBridgeBinding, adapter: Any,
    ) -> None:
        for progress in store.list_progress(binding, pending_only=True, ready_only=True):
            try:
                await self._codex_bridge_deliver_progress(store, binding, progress, adapter)
            except Exception:
                logger.warning(
                    "Codex progress retry failed for %s turn=%s",
                    binding.control_session_key, progress.logical_turn_id,
                    exc_info=True,
                )

    async def _codex_bridge_mirror_final(
        self, store: CodexBridgeStore, binding: CodexBridgeBinding,
        event: RolloutEvent, adapter: Any,
    ) -> None:
        from gateway.delivery_ledger import (
            compute_obligation_id,
            ensure_obligation,
            mark_attempting,
            mark_delivered,
            mark_failed,
        )

        label = "💻 Codex 오류" if event.kind == "error" else "💻 Codex 답변"
        content = f"{label}\n\n{event.text}"
        obligation_id = compute_obligation_id(
            self._codex_bridge_lane_key(binding.source, binding), event.event_id, content,
        )
        inserted = await self._codex_bridge_ledger_call(
            binding.source,
            ensure_obligation,
            obligation_id=obligation_id,
            session_key=self._codex_bridge_lane_key(binding.source, binding),
            platform=binding.source.platform.value, chat_id=binding.source.chat_id,
            thread_id=binding.source.thread_id, content=content,
            adapter_profile=getattr(adapter, "_owner_profile", None),
        )
        if not inserted:
            store.complete_progress(binding, self._codex_bridge_mirror_identity(event))
            return
        await self._codex_bridge_ledger_call(
            binding.source, mark_attempting, obligation_id,
        )
        try:
            result = await asyncio.wait_for(
                adapter.send(
                    binding.source.chat_id, content,
                    metadata=self._thread_metadata_for_source(binding.source),
                ), timeout=20.0,
            )
        except Exception as exc:
            await self._codex_bridge_ledger_call(
                binding.source, mark_failed, obligation_id, str(exc)[:500],
            )
        else:
            if getattr(result, "success", False):
                await self._codex_bridge_ledger_call(
                    binding.source, mark_delivered, obligation_id,
                )
            else:
                await self._codex_bridge_ledger_call(
                    binding.source, mark_failed, obligation_id,
                    str(getattr(result, "error", ""))[:500],
                )
        store.complete_progress(binding, self._codex_bridge_mirror_identity(event))

    async def _codex_bridge_poll_binding(self, store: CodexBridgeStore, binding: CodexBridgeBinding) -> None:
        async with self._codex_bridge_binding_lock(binding.control_session_key):
            await self._codex_bridge_poll_binding_locked(store, binding)

    @staticmethod
    def _codex_bridge_reconcile_continuation(store: CodexBridgeStore, item: DurableCodexInput) -> None:
        """Close a handoff that ReloginTool proved needed no replacement turn."""
        handoff = read_turn_handoff(item.thread_id, item.codex_turn_id or "")
        if handoff is not None and handoff.status == "terminal":
            store.mark_completed(item.input_id)

    async def _codex_bridge_poll_binding_locked(
        self, store: CodexBridgeStore, binding: CodexBridgeBinding,
    ) -> None:
        if binding.pending_new:
            return
        adapter = self._adapter_for_source(binding.source)
        if adapter is None:
            return
        current = store.get_binding(binding.control_session_key)
        if (
            current is None
            or current.generation != binding.generation
            or current.thread_id != binding.thread_id
        ):
            self._codex_bridge_tails.pop(
                (binding.control_session_key, binding.generation), None,
            )
            return
        await self._codex_bridge_retry_progress(store, binding, adapter)
        path = await asyncio.to_thread(
            resolve_rollout_path, binding.thread_id, hinted_path=binding.rollout_path,
        )
        if path is None:
            return
        key = (binding.control_session_key, binding.generation)
        tail = self._codex_bridge_tails.get(key)
        if tail is None or tail.path != path:
            same_incarnation = binding.rollout_path == str(path)
            tail = RolloutTail(
                binding.thread_id,
                path,
                device=binding.cursor_device if same_incarnation else None,
                inode=binding.cursor_inode if same_incarnation else None,
                offset=binding.cursor_offset if same_incarnation else 0,
            )
            self._codex_bridge_tails[key] = tail
        events, next_offset, stat = await asyncio.to_thread(tail.scan)
        last_event_id = binding.last_event_id
        committed_offset = tail.offset

        async def _commit(event: RolloutEvent) -> None:
            nonlocal last_event_id, committed_offset
            last_event_id, committed_offset = event.event_id, event.offset
            if store.update_cursor(
                binding, rollout_path=str(path), device=stat.st_dev, inode=stat.st_ino,
                offset=committed_offset, last_event_id=last_event_id,
            ):
                tail.offset = committed_offset
            else:
                raise RuntimeError("Codex binding changed while committing its rollout cursor")

        pending_commentary: dict[str, list[RolloutEvent]] = {}
        deferred = False

        async def _flush_commentary() -> None:
            if not pending_commentary:
                return
            last: Optional[RolloutEvent] = None
            for grouped in pending_commentary.values():
                progress = self._codex_bridge_stage_commentary(store, binding, grouped)
                try:
                    if progress.state == "pending":
                        await self._codex_bridge_deliver_progress(
                            store, binding, progress, adapter,
                        )
                except Exception:
                    # The outbox is committed already, so a transient failure
                    # cannot erase progress even though the rollout cursor
                    # advances to keep a durable final from starving.
                    logger.warning(
                        "Codex progress delivery failed for %s",
                        binding.control_session_key,
                        exc_info=True,
                    )
                if last is None or grouped[-1].offset > last.offset:
                    last = grouped[-1]
            pending_commentary.clear()
            if last is not None:
                await _commit(last)

        for event in events:
            # The live app-server runner owns its physical turn. The rollout
            # watcher may atomically take over only for crash recovery or a
            # later physical turn that continues an interrupted logical input.
            disposition, transferred = (
                store.claim_rollout_event(event.client_id, event.turn_id)
                if event.client_id else ("unmanaged", False)
            )
            if transferred:
                logger.info(
                    "Codex rollout delivery ownership transferred: input=%s turn=%s binding=%s",
                    event.client_id, event.turn_id, binding.control_session_key,
                )
            if disposition == "wait":
                await _flush_commentary()
                logger.debug(
                    "Deferring Codex rollout event until live delivery ownership settles: "
                    "input=%s turn=%s kind=%s",
                    event.client_id, event.turn_id, event.kind,
                )
                deferred = True
                break
            watcher_owned = disposition in {"unmanaged", "rollout"}
            if event.kind != "commentary" or not watcher_owned:
                await _flush_commentary()
            if disposition == "rollout" and event.kind == "commentary":
                identity = self._codex_bridge_mirror_identity(event)
                pending_commentary.setdefault(identity, []).append(event)
                continue
            if disposition == "rollout" and event.kind in {"final", "error"}:
                captured = store.capture_recovery_output(
                    event.client_id,
                    event.text or "Codex 작업이 오류로 종료되었습니다.",
                    turn_outcome="failed" if event.kind == "error" else "completed",
                )
                if not captured:
                    raise RuntimeError(
                        f"Codex rollout final lost delivery ownership: input={event.client_id}"
                    )
            elif disposition == "unmanaged" and event.kind == "commentary":
                identity = self._codex_bridge_mirror_identity(event)
                pending_commentary.setdefault(identity, []).append(event)
                continue
            elif disposition == "unmanaged" and event.kind in {"final", "error"}:
                await self._codex_bridge_mirror_final(store, binding, event, adapter)
            elif disposition in {"live", "terminal"}:
                log_suppression = (
                    logger.info if event.kind in {"final", "error"} else logger.debug
                )
                log_suppression(
                    "Suppressing Codex rollout event owned by %s delivery: "
                    "input=%s turn=%s kind=%s",
                    disposition, event.client_id, event.turn_id, event.kind,
                )
            await _commit(event)
        await _flush_commentary()
        if deferred:
            # ``scan`` projected through the whole read window before delivery
            # ownership was known. Rebuild that projection from the durable
            # cursor next time; neither the blocked event nor later records may
            # be acknowledged by the end-of-window cursor catch-up below.
            self._codex_bridge_tails.pop(key, None)
        elif next_offset > committed_offset and store.update_cursor(
            binding, rollout_path=str(path), device=stat.st_dev, inode=stat.st_ino,
            offset=next_offset, last_event_id=last_event_id,
        ):
            tail.offset = next_offset

    async def _codex_bridge_watcher(self) -> None:
        """Forever-retrying input recovery and desktop rollout mirror."""
        while getattr(self, "_running", False):
            stores = list(getattr(self, "_codex_bridge_stores", {}).values())
            for store in stores:
                try:
                    for item in await asyncio.to_thread(store.recoverable_outputs):
                        await self._codex_bridge_recover_output(store, item)
                    for item in await asyncio.to_thread(store.recoverable_inputs):
                        await self._codex_bridge_recover_input(store, item)
                    bindings = await asyncio.to_thread(store.list_bindings)
                    results = await asyncio.gather(
                        *(self._codex_bridge_poll_binding(store, binding) for binding in bindings),
                        return_exceptions=True,
                    )
                    for binding, result in zip(bindings, results):
                        if isinstance(result, BaseException):
                            logger.warning(
                                "Codex bridge watcher failed for %s: %s",
                                binding.control_session_key, result,
                            )
                    for item in await asyncio.to_thread(store.continuation_pending_inputs):
                        await asyncio.to_thread(
                            self._codex_bridge_reconcile_continuation, store, item,
                        )
                    await asyncio.to_thread(store.prune)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("Codex bridge watcher iteration failed", exc_info=True)
            await asyncio.sleep(0.75)
