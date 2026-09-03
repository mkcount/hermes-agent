"""Linearizable delivery authority for Telegram-bound Codex sessions.

Execution lanes and outbound chat routing deliberately have different
lifetimes: a Codex turn may keep running after the user selects another
desktop thread, but its late progress and final response must no longer be
delivered into the shared Telegram chat.

This module owns that control-plane boundary.  Every selected binding gets a
monotonic generation, and every execution turn receives an immutable grant.
A grant is valid only while all three fields still match:

* control-plane session key
* binding generation
* selected Codex thread id

The generation check is essential for A -> B -> A switches: comparing only
the thread id would incorrectly re-authorize the old A turn.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import threading
from typing import Any, Dict, Optional


CODEX_DELIVERY_GRANT_METADATA_KEY = "codex_delivery_grant"


@dataclass(frozen=True)
class CodexBindingState:
    generation: int
    thread_id: Optional[str]


@dataclass(frozen=True)
class CodexDeliveryGrant:
    control_session_key: str
    generation: int
    thread_id: str

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "control_session_key": self.control_session_key,
            "generation": self.generation,
            "thread_id": self.thread_id,
        }

    @classmethod
    def from_metadata(cls, value: Any) -> Optional["CodexDeliveryGrant"]:
        if not isinstance(value, dict):
            return None
        control_session_key = str(value.get("control_session_key") or "").strip()
        thread_id = str(value.get("thread_id") or "").strip()
        try:
            generation = int(value.get("generation"))
        except (TypeError, ValueError):
            return None
        if not control_session_key or not thread_id or generation <= 0:
            return None
        return cls(
            control_session_key=control_session_key,
            generation=generation,
            thread_id=thread_id,
        )


class CodexDeliveryAuthority:
    """Own current Codex binding generations and per-chat transition locks."""

    def __init__(self) -> None:
        self._states: Dict[str, CodexBindingState] = {}
        self._transition_locks: Dict[str, asyncio.Lock] = {}
        self._direct_turn_claims: set[tuple[str, int, str, str]] = set()
        # A new Codex turn is persisted before ``turn/start`` returns its
        # server-assigned turn id.  Track the caller-supplied
        # ``clientUserMessageId`` during that gap so the rollout mirror cannot
        # briefly claim the same Telegram-originated request.
        self._direct_client_claims: set[tuple[str, int, str, str]] = set()
        self._state_lock = threading.RLock()

    def transition_lock(self, control_session_key: str) -> asyncio.Lock:
        """Return the event-loop lock serializing one chat's binding changes."""
        lock = self._transition_locks.get(control_session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._transition_locks[control_session_key] = lock
        return lock

    @staticmethod
    def _normalize_thread_id(thread_id: Any) -> Optional[str]:
        normalized = str(thread_id or "").strip()
        return normalized or None

    @staticmethod
    def _normalize_turn_id(turn_id: Any) -> Optional[str]:
        normalized = str(turn_id or "").strip()
        return normalized or None

    @staticmethod
    def _normalize_client_id(client_id: Any) -> Optional[str]:
        normalized = str(client_id or "").strip()
        return normalized or None

    @staticmethod
    def _direct_turn_claim_key(
        grant: CodexDeliveryGrant,
        turn_id: str,
    ) -> tuple[str, int, str, str]:
        return (
            grant.control_session_key,
            grant.generation,
            grant.thread_id,
            turn_id,
        )

    @staticmethod
    def _direct_client_claim_key(
        grant: CodexDeliveryGrant,
        client_id: str,
    ) -> tuple[str, int, str, str]:
        return (
            grant.control_session_key,
            grant.generation,
            grant.thread_id,
            client_id,
        )

    def _allows_grant_locked(self, grant: CodexDeliveryGrant) -> bool:
        current = self._states.get(grant.control_session_key)
        return bool(
            current is not None
            and current.generation == grant.generation
            and current.thread_id == grant.thread_id
        )

    def _discard_direct_turn_claims_locked(
        self,
        control_session_key: str,
    ) -> None:
        self._direct_turn_claims = {
            claim
            for claim in self._direct_turn_claims
            if claim[0] != control_session_key
        }
        self._direct_client_claims = {
            claim
            for claim in self._direct_client_claims
            if claim[0] != control_session_key
        }

    def observe_binding(
        self,
        control_session_key: str,
        thread_id: Any,
    ) -> CodexBindingState:
        """Synchronize in-memory authority with a persisted binding.

        Re-reading an unchanged binding keeps the generation stable.  A
        different persisted value rotates it, invalidating every earlier
        grant even when the new value later returns to the same thread id.
        """
        selected = self._normalize_thread_id(thread_id)
        with self._state_lock:
            current = self._states.get(control_session_key)
            if current is not None and current.thread_id == selected:
                return current
            self._discard_direct_turn_claims_locked(control_session_key)
            state = CodexBindingState(
                generation=(current.generation if current else 0) + 1,
                thread_id=selected,
            )
            self._states[control_session_key] = state
            return state

    def begin_transition(self, control_session_key: str) -> int:
        """Revoke the current binding before persistent selection changes."""
        with self._state_lock:
            current = self._states.get(control_session_key)
            generation = (current.generation if current else 0) + 1
            self._discard_direct_turn_claims_locked(control_session_key)
            self._states[control_session_key] = CodexBindingState(
                generation=generation,
                thread_id=None,
            )
            return generation

    def commit_transition(
        self,
        control_session_key: str,
        generation: int,
        thread_id: Any,
    ) -> bool:
        """Publish a persisted selection if this transition still owns it."""
        selected = self._normalize_thread_id(thread_id)
        with self._state_lock:
            current = self._states.get(control_session_key)
            if current is None or current.generation != generation:
                return False
            self._states[control_session_key] = CodexBindingState(
                generation=generation,
                thread_id=selected,
            )
            return True

    def issue_grant(
        self,
        control_session_key: str,
        execution_thread_id: Any,
    ) -> Optional[CodexDeliveryGrant]:
        thread_id = self._normalize_thread_id(execution_thread_id)
        if thread_id is None:
            return None
        with self._state_lock:
            state = self._states.get(control_session_key)
            if state is None:
                return None
            return CodexDeliveryGrant(
                control_session_key=control_session_key,
                generation=state.generation,
                thread_id=thread_id,
            )

    def allows(self, metadata: Any) -> bool:
        grant = CodexDeliveryGrant.from_metadata(metadata)
        if grant is None:
            return False
        with self._state_lock:
            return self._allows_grant_locked(grant)

    def acquire_direct_turn(self, metadata: Any, turn_id: Any) -> bool:
        """Claim one Codex turn for the live gateway delivery path.

        The immutable binding generation makes the claim ABA-safe. A binding
        transition revokes every earlier claim, while a gateway process crash
        naturally drops the process-local claim so the rollout mirror can
        recover an unfinished turn.
        """
        grant = CodexDeliveryGrant.from_metadata(metadata)
        normalized_turn_id = self._normalize_turn_id(turn_id)
        if grant is None or normalized_turn_id is None:
            return False
        with self._state_lock:
            if not self._allows_grant_locked(grant):
                return False
            self._direct_turn_claims.add(
                self._direct_turn_claim_key(grant, normalized_turn_id)
            )
            return True

    def owns_direct_turn(self, metadata: Any, turn_id: Any) -> bool:
        """Return whether the current generation still owns direct delivery."""
        grant = CodexDeliveryGrant.from_metadata(metadata)
        normalized_turn_id = self._normalize_turn_id(turn_id)
        if grant is None or normalized_turn_id is None:
            return False
        with self._state_lock:
            return bool(
                self._allows_grant_locked(grant)
                and self._direct_turn_claim_key(grant, normalized_turn_id)
                in self._direct_turn_claims
            )

    def release_direct_turn(self, metadata: Any, turn_id: Any) -> None:
        """Release the exact generation-scoped direct-delivery claim."""
        grant = CodexDeliveryGrant.from_metadata(metadata)
        normalized_turn_id = self._normalize_turn_id(turn_id)
        if grant is None or normalized_turn_id is None:
            return
        with self._state_lock:
            self._direct_turn_claims.discard(
                self._direct_turn_claim_key(grant, normalized_turn_id)
            )

    def acquire_direct_client(self, metadata: Any, client_id: Any) -> bool:
        """Claim a submitted request before Codex assigns its turn id.

        This lease is deliberately process-local. If the gateway dies before
        learning the turn id, the rollout mirror is allowed to recover the
        still-running turn after restart instead of suppressing it forever.
        """
        grant = CodexDeliveryGrant.from_metadata(metadata)
        normalized_client_id = self._normalize_client_id(client_id)
        if grant is None or normalized_client_id is None:
            return False
        with self._state_lock:
            if not self._allows_grant_locked(grant):
                return False
            self._direct_client_claims.add(
                self._direct_client_claim_key(grant, normalized_client_id)
            )
            return True

    def owns_direct_client(self, metadata: Any, client_id: Any) -> bool:
        """Return whether direct delivery owns this pre-turn client id."""
        grant = CodexDeliveryGrant.from_metadata(metadata)
        normalized_client_id = self._normalize_client_id(client_id)
        if grant is None or normalized_client_id is None:
            return False
        with self._state_lock:
            return bool(
                self._allows_grant_locked(grant)
                and self._direct_client_claim_key(
                    grant,
                    normalized_client_id,
                )
                in self._direct_client_claims
            )

    def release_direct_client(self, metadata: Any, client_id: Any) -> None:
        """Release the request claim after turn-id ownership is established."""
        grant = CodexDeliveryGrant.from_metadata(metadata)
        normalized_client_id = self._normalize_client_id(client_id)
        if grant is None or normalized_client_id is None:
            return
        with self._state_lock:
            self._direct_client_claims.discard(
                self._direct_client_claim_key(grant, normalized_client_id)
            )

    def state_for(self, control_session_key: str) -> Optional[CodexBindingState]:
        """Return an immutable snapshot for diagnostics and behavior tests."""
        with self._state_lock:
            return self._states.get(control_session_key)
