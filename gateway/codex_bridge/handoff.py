"""Read ReloginTool's durable App Server turn-continuation graph."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_STATE_PATH = Path.home() / ".local" / "state" / "relogin-tool" / "profile-monitor-state.json"
_PENDING_STATUSES = frozenset({
    "prepared", "interrupting", "interrupt_confirmed", "pending", "reconciling",
})
_BOUND_STATUSES = frozenset({"successor_bound", "continued"})
_NOOP_STATUSES = frozenset({"noop_original_completed", "terminal"})
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, tuple[int, int, dict[str, dict[str, Any]]]] = {}


@dataclass(frozen=True)
class CodexTurnHandoff:
    thread_id: str
    interrupted_turn_id: str
    continued_turn_id: str
    continuation_turn_ids: tuple[str, ...]
    status: str
    reason: str
    operation_id: str = ""

    @property
    def continues(self) -> bool:
        return self.status in _PENDING_STATUSES | _BOUND_STATUSES

    @property
    def no_successor_required(self) -> bool:
        return self.status in _NOOP_STATUSES

    def owns_continuation(self, physical_turn_status: str) -> bool:
        """Whether Hermes must keep the logical input open at this boundary."""
        physical = str(physical_turn_status or "unknown").strip().lower()
        if physical == "completed":
            return (
                self.status in _BOUND_STATUSES
                or self.reason == "usage_limit_continuation"
            )
        if self.status in _BOUND_STATUSES:
            return True
        if self.status in {"interrupt_confirmed", "pending", "reconciling"}:
            return True
        # prepare is only intent. It becomes authoritative when Hermes itself
        # observed the old physical turn end as interrupted/failed.
        return self.status in {"prepared", "interrupting"} and physical in {"interrupted", "failed"}


@dataclass(frozen=True)
class CodexHandoffGraph:
    successors: dict[str, str]
    statuses: dict[str, str]

    def successor_for(self, turn_id: str) -> str:
        return str(self.successors.get(str(turn_id)) or "")

    def status_for(self, turn_id: str) -> str:
        return str(self.statuses.get(str(turn_id)) or "")

    def is_pending(self, turn_id: str) -> bool:
        return self.status_for(turn_id) in _PENDING_STATUSES

    def is_managed(self, turn_id: str) -> bool:
        return str(turn_id) in self.statuses


def _read_handoffs(path: Path) -> dict[str, dict[str, Any]]:
    try:
        stat = path.stat()
    except OSError:
        return {}
    cache_key = str(path)
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    handoffs = raw.get("codex_turn_handoffs") if isinstance(raw, dict) else None
    if not isinstance(handoffs, dict):
        return {}
    parsed = {
        str(key): value for key, value in handoffs.items()
        if isinstance(value, dict)
    }
    with _CACHE_LOCK:
        _CACHE[cache_key] = (stat.st_mtime_ns, stat.st_size, parsed)
    return parsed


def read_thread_handoff_graph(
    thread_id: str, *, state_path: Optional[Path] = None,
) -> CodexHandoffGraph:
    """Return exact, formally recorded predecessor→successor edges."""
    cleaned_thread = str(thread_id or "").strip()
    path = Path(state_path) if state_path is not None else _STATE_PATH
    successors: dict[str, str] = {}
    statuses: dict[str, str] = {}
    for value in _read_handoffs(path).values():
        if str(value.get("thread_id") or "") != cleaned_thread:
            continue
        predecessor = str(value.get("interrupted_turn_id") or "").strip()
        status = str(value.get("status") or "").strip().lower()
        if not predecessor or not status:
            continue
        statuses[predecessor] = status
        successor = str(value.get("continued_turn_id") or "").strip()
        if successor and status in _BOUND_STATUSES:
            successors[predecessor] = successor
    return CodexHandoffGraph(successors=successors, statuses=statuses)


def read_turn_handoff(
    thread_id: str,
    turn_id: str,
    *,
    state_path: Optional[Path] = None,
) -> Optional[CodexTurnHandoff]:
    """Return one handoff plus its append-only A→B→C successor chain."""
    cleaned_thread = str(thread_id or "").strip()
    cleaned_turn = str(turn_id or "").strip()
    if not cleaned_thread or not cleaned_turn:
        return None
    path = Path(state_path) if state_path is not None else _STATE_PATH
    handoffs = _read_handoffs(path)
    value = handoffs.get(f"{cleaned_thread}:{cleaned_turn}")
    if not isinstance(value, dict):
        return None
    if (
        str(value.get("thread_id") or "") != cleaned_thread
        or str(value.get("interrupted_turn_id") or "") != cleaned_turn
    ):
        return None
    status = str(value.get("status") or "").strip().lower()
    if not status:
        return None
    immediate = str(value.get("continued_turn_id") or "").strip()
    chain: list[str] = []
    seen = {cleaned_turn}
    successor = immediate
    while successor and successor not in seen:
        chain.append(successor)
        seen.add(successor)
        next_value = handoffs.get(f"{cleaned_thread}:{successor}")
        if not isinstance(next_value, dict):
            break
        if str(next_value.get("status") or "").strip().lower() not in _BOUND_STATUSES:
            break
        successor = str(next_value.get("continued_turn_id") or "").strip()
    return CodexTurnHandoff(
        thread_id=cleaned_thread,
        interrupted_turn_id=cleaned_turn,
        continued_turn_id=immediate,
        continuation_turn_ids=tuple(chain),
        status=status,
        reason=str(value.get("reason") or "").strip(),
        operation_id=str(value.get("operation_id") or "").strip(),
    )
