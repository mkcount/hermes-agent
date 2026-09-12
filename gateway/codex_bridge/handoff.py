"""Read ReloginTool's durable App Server turn-continuation handoff."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_STATE_PATH = Path.home() / ".local" / "state" / "relogin-tool" / "profile-monitor-state.json"
_CONTINUING_STATUSES = frozenset({"interrupting", "pending", "continued"})


@dataclass(frozen=True)
class CodexTurnHandoff:
    thread_id: str
    interrupted_turn_id: str
    continued_turn_id: str
    status: str
    reason: str

    @property
    def continues(self) -> bool:
        return self.status in _CONTINUING_STATUSES


def read_turn_handoff(
    thread_id: str,
    turn_id: str,
    *,
    state_path: Optional[Path] = None,
) -> Optional[CodexTurnHandoff]:
    """Return an exact handoff record; malformed or unavailable state is no claim."""
    cleaned_thread = str(thread_id or "").strip()
    cleaned_turn = str(turn_id or "").strip()
    if not cleaned_thread or not cleaned_turn:
        return None
    path = Path(state_path) if state_path is not None else _STATE_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    handoffs = raw.get("codex_turn_handoffs") if isinstance(raw, dict) else None
    if not isinstance(handoffs, dict):
        return None
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
    return CodexTurnHandoff(
        thread_id=cleaned_thread,
        interrupted_turn_id=cleaned_turn,
        continued_turn_id=str(value.get("continued_turn_id") or "").strip(),
        status=status,
        reason=str(value.get("reason") or "").strip(),
    )
