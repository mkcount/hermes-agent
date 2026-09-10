"""Exact, incremental ingestion of public Codex rollout events."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_RUNTIME_PREFIXES = (
    "<recommended_plugins>", "# agents.md instructions", "<environment_context>",
    "<permissions instructions>", "<collaboration_mode>", "<apps_instructions>",
    "<plugins_instructions>",
)
_BOOTSTRAP_BYTES = 16 * 1024 * 1024
_SCAN_BYTES = 4 * 1024 * 1024
_SESSION_META_MAX_BYTES = 1024 * 1024
_CONTINUATION_WINDOW_SECONDS = 30.0


@dataclass(frozen=True)
class RolloutEvent:
    event_id: str
    turn_id: str
    kind: str  # user | commentary | final | error
    text: str
    offset: int
    client_id: Optional[str] = None


@dataclass(frozen=True)
class RolloutSnapshot:
    path: str
    device: int
    inode: int
    size: int
    active_turn_id: Optional[str]
    active_start_offset: Optional[int]
    latest_final_text: str = ""


def _codex_home(value: Optional[str]) -> Path:
    return Path(value or os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()


def resolve_rollout_path(
    thread_id: str, *, hinted_path: Optional[str] = None, codex_home: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a rollout whose internal session id exactly matches ``thread_id``.

    Codex continuation files may contain both a logical thread id and a file
    incarnation id in their filename. Conversely, child files can contain a
    parent's id as a filename substring. The first ``session_meta`` record is
    therefore the authority; the filename is only a bounded candidate index.
    """
    cleaned = str(thread_id or "").strip()
    if not _THREAD_ID_RE.fullmatch(cleaned):
        return None
    root = (_codex_home(codex_home) / "sessions").resolve()
    def accepted(candidate: Path) -> Optional[Path]:
        try:
            resolved = candidate.expanduser().resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            return None
        if not resolved.is_file():
            return None
        try:
            with resolved.open("rb") as handle:
                raw = handle.readline(_SESSION_META_MAX_BYTES + 1)
            if len(raw) > _SESSION_META_MAX_BYTES or not raw.endswith(b"\n"):
                return None
            record = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        payload = record.get("payload") if isinstance(record, dict) else None
        if record.get("type") != "session_meta" or not isinstance(payload, dict):
            return None
        return resolved if str(payload.get("id") or "") == cleaned else None

    if hinted_path and (hit := accepted(Path(hinted_path))) is not None:
        return hit
    try:
        matches = [
            hit for candidate in root.rglob(f"*{cleaned}*.jsonl")
            if (hit := accepted(candidate)) is not None
        ]
    except OSError:
        return None
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _aligned_tail_start(path: Path, size: int, window: int = _BOOTSTRAP_BYTES) -> int:
    start = max(0, size - max(1, int(window)))
    if start == 0:
        return 0
    with path.open("rb") as handle:
        handle.seek(start - 1)
        if handle.read(1) != b"\n":
            handle.readline()
        return handle.tell()


def _content(payload: dict, wanted: str) -> str:
    parts = payload.get("content")
    if not isinstance(parts, list):
        return ""
    return "\n".join(
        str(part.get("text") or "").strip() for part in parts
        if isinstance(part, dict) and part.get("type") == wanted and str(part.get("text") or "").strip()
    ).strip()


def _client_id(payload: dict) -> Optional[str]:
    metadata = payload.get("internal_chat_message_metadata_passthrough")
    value = payload.get("client_id") or payload.get("clientId")
    if not value and isinstance(metadata, dict):
        value = metadata.get("client_id") or metadata.get("clientId")
    return str(value).strip() if value else None


def _is_user_authored(payload: dict, text: str) -> bool:
    """Use Codex's structured provenance, with a legacy text fallback."""
    metadata = payload.get("internal_chat_message_metadata_passthrough")
    kinds = metadata.get("content_item_kinds") if isinstance(metadata, dict) else None
    if isinstance(kinds, list) and kinds:
        return any(str(kind).strip().lower().startswith("user.") for kind in kinds)
    return not all(
        part.lstrip().lower().startswith(_RUNTIME_PREFIXES)
        for part in text.split("\n\n")
    )


class RolloutTail:
    """Partial-line safe tailer whose cursor is committed by the caller after delivery."""

    def __init__(
        self, thread_id: str, path: Path, *, device: Optional[int] = None,
        inode: Optional[int] = None, offset: int = 0,
    ) -> None:
        self.thread_id, self.path = str(thread_id), Path(path)
        self.device, self.inode, self.offset = device, inode, max(0, int(offset))
        self.active_turn_id: Optional[str] = None
        self.active_start_offset: Optional[int] = None
        self.turns: dict[str, dict[str, Any]] = {}
        self.latest_final_text = ""
        self._continuation_seed: Optional[tuple[dict[str, Any], float, str, int]] = None
        self._bad_line: Optional[tuple[int, str]] = None
        self._bad_line_attempts = 0
        if self.offset:
            self._prime(self.offset)

    def _reset_for_replacement(self, stat: os.stat_result) -> None:
        self.device, self.inode = stat.st_dev, stat.st_ino
        self.offset = _aligned_tail_start(self.path, stat.st_size)
        self.active_turn_id = self.active_start_offset = None
        self.turns.clear()
        self.latest_final_text = ""
        self._continuation_seed = None
        self._prime(self.offset)

    @staticmethod
    def _record_time(record: dict) -> Optional[float]:
        value = record.get("timestamp")
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None

    def _prime(self, end: int) -> None:
        """Rebuild turn state from a bounded pre-cursor window without emitting."""
        self._prime_range(_aligned_tail_start(self.path, end), end)

    def prime_full(self, end: int) -> None:
        """Stream the initial history once to recover exact active/final state."""
        self.active_turn_id = self.active_start_offset = None
        self.turns.clear()
        self.latest_final_text = ""
        self._continuation_seed = None
        self._prime_range(0, end)

    def _prime_range(self, start: int, end: int) -> None:
        try:
            with self.path.open("rb") as handle:
                handle.seek(start)
                while handle.tell() < end:
                    line_start = handle.tell()
                    raw = handle.readline()
                    if not raw.endswith(b"\n") or handle.tell() > end:
                        break
                    try:
                        record = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    self._consume(record, line_start, handle.tell(), None)
        except OSError:
            return

    def scan(self) -> tuple[list[RolloutEvent], int, os.stat_result]:
        stat = self.path.stat()
        if self.device is None or self.inode is None:
            self.device, self.inode = stat.st_dev, stat.st_ino
        elif (self.device, self.inode) != (stat.st_dev, stat.st_ino) or self.offset > stat.st_size:
            self._reset_for_replacement(stat)
        events: list[RolloutEvent] = []
        next_offset = self.offset
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            while True:
                line_start = handle.tell()
                raw = handle.readline()
                if not raw or not raw.endswith(b"\n"):
                    break
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    fingerprint = hashlib.sha256(raw).hexdigest()[:16]
                    marker = (line_start, fingerprint)
                    self._bad_line_attempts = self._bad_line_attempts + 1 if marker == self._bad_line else 1
                    self._bad_line = marker
                    # Retry twice across scans. On the third observation emit a
                    # visible diagnostic and quarantine exactly this complete
                    # malformed line so later finals are not starved forever.
                    if self._bad_line_attempts < 3:
                        break
                    next_offset = handle.tell()
                    events.append(self._event("rollout", "error", "Codex 기록 한 줄이 손상되어 건너뛰었습니다.", next_offset))
                    continue
                self._bad_line = None
                self._bad_line_attempts = 0
                next_offset = handle.tell()
                self._consume(record, line_start, next_offset, events)
                if next_offset - self.offset >= _SCAN_BYTES:
                    break
        self._expire_continuation(events, time.time())
        return events, next_offset, stat

    def _event(
        self, turn_id: str, kind: str, text: str, offset: int, client_id: Optional[str] = None,
    ) -> RolloutEvent:
        identity = f"{self.thread_id}\0{turn_id}\0{kind}\0{offset}\0{text}"
        return RolloutEvent(
            event_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(), turn_id=turn_id,
            kind=kind, text=text, offset=offset, client_id=client_id,
        )

    def _turn(self, turn_id: str) -> dict[str, Any]:
        return self.turns.setdefault(turn_id, {
            "user_seen": False, "user_text": "", "client_id": None, "final": "", "start": None,
            "pending_commentary": [],
        })

    def _expire_continuation(
        self, events: Optional[list[RolloutEvent]], observed_at: Optional[float],
    ) -> None:
        seed_info = self._continuation_seed
        if seed_info is None or observed_at is None:
            return
        seed, aborted_at, turn_id, offset = seed_info
        if observed_at - aborted_at < _CONTINUATION_WINDOW_SECONDS:
            return
        self._continuation_seed = None
        if events is not None and seed.get("user_seen"):
            events.append(self._event(
                turn_id,
                "error",
                "Codex 작업이 중단되었습니다.",
                max(offset, self.offset),
                seed.get("client_id"),
            ))

    def _flush_pending_commentary(
        self, events: Optional[list[RolloutEvent]], turn_id: str, offset: int,
    ) -> None:
        turn = self._turn(turn_id)
        pending = turn.get("pending_commentary") or []
        turn["pending_commentary"] = []
        for text in pending:
            self._emit(events, turn_id, "commentary", str(text), offset)

    def _commentary(
        self, events: Optional[list[RolloutEvent]], turn_id: str, text: str, offset: int,
    ) -> None:
        if not text.strip():
            return
        if events is None:
            return
        turn = self._turn(turn_id)
        # Some rollout versions attach clientUserMessageId in a later
        # item_completed record. Hold commentary until that correlation point
        # so a Telegram-originated turn cannot leak one duplicate frame.
        if turn.get("client_id"):
            self._emit(events, turn_id, "commentary", text, offset)
        else:
            turn.setdefault("pending_commentary", []).append(text)

    def _turn_id_for_item(self, payload: dict) -> Optional[str]:
        meta = payload.get("internal_chat_message_metadata_passthrough")
        value = None
        if isinstance(meta, dict):
            value = meta.get("turn_id") or meta.get("turnId")
        value = value or payload.get("turn_id") or payload.get("turnId") or self.active_turn_id
        return str(value).strip() if value else None

    def _emit(
        self, events: Optional[list[RolloutEvent]], turn_id: str, kind: str,
        text: str, offset: int,
    ) -> None:
        if events is None or not text.strip():
            return
        turn = self._turn(turn_id)
        if not turn["user_seen"] and kind != "error":
            return
        events.append(self._event(turn_id, kind, text.strip(), offset, turn.get("client_id")))

    def _consume(
        self, record: object, line_start: int, offset: int,
        events: Optional[list[RolloutEvent]],
    ) -> None:
        if not isinstance(record, dict) or not isinstance(record.get("payload"), dict):
            return
        payload = record["payload"]
        if record.get("type") == "response_item" and payload.get("type") == "message":
            turn_id = self._turn_id_for_item(payload)
            if not turn_id:
                return
            turn = self._turn(turn_id)
            role = str(payload.get("role") or "").lower()
            if role == "user":
                text = _content(payload, "input_text")
                if text and _is_user_authored(payload, text):
                    turn.update(user_seen=True, user_text=text, client_id=_client_id(payload) or turn.get("client_id"))
                    self._emit(events, turn_id, "user", text, offset)
            elif role == "assistant":
                text = _content(payload, "output_text")
                phase = str(payload.get("phase") or "").strip().lower()
                if phase == "commentary":
                    self._commentary(events, turn_id, text, offset)
                elif phase == "final_answer":
                    turn["final"] = text
            return
        if record.get("type") != "event_msg":
            return
        event_type = payload.get("type")
        if event_type == "task_started":
            turn_id = str(payload.get("turn_id") or "").strip()
            if turn_id:
                self.active_turn_id, self.active_start_offset = turn_id, line_start
                inherited: Optional[dict[str, Any]] = None
                if self._continuation_seed is not None:
                    seed, aborted_at, _aborted_turn_id, _aborted_offset = self._continuation_seed
                    started_at = self._record_time(record)
                    if (
                        started_at is not None
                        and 0 <= started_at - aborted_at <= _CONTINUATION_WINDOW_SECONDS
                    ):
                        inherited = seed
                    else:
                        self._expire_continuation(events, started_at or float("inf"))
                self._continuation_seed = None
                self.turns[turn_id] = {
                    "user_seen": bool(inherited and inherited.get("user_seen")),
                    "user_text": str(inherited.get("user_text") or "") if inherited else "",
                    "client_id": inherited.get("client_id") if inherited else None,
                    "final": "", "start": line_start, "pending_commentary": [],
                }
            return
        turn_id = str(payload.get("turn_id") or self.active_turn_id or "").strip()
        if not turn_id:
            return
        turn = self._turn(turn_id)
        if event_type == "item_completed":
            item = payload.get("item")
            if isinstance(item, dict) and str(item.get("type") or "").replace("_", "").lower() == "usermessage":
                turn["client_id"] = _client_id(item) or turn.get("client_id")
                if turn.get("client_id"):
                    self._flush_pending_commentary(events, turn_id, offset)
            return
        if event_type == "user_message":
            text = str(payload.get("message") or "").strip()
            turn.update(user_seen=True, user_text=text, client_id=_client_id(payload) or turn.get("client_id"))
            self._emit(events, turn_id, "user", text, offset)
        elif event_type == "agent_message":
            text = str(payload.get("message") or "").strip()
            phase = str(payload.get("phase") or "").strip().lower()
            if phase == "commentary":
                self._commentary(events, turn_id, text, offset)
            elif phase == "final_answer":
                turn["final"] = text
        elif event_type in {"task_complete", "turn_aborted"}:
            self._flush_pending_commentary(events, turn_id, offset)
            if event_type == "turn_aborted":
                # Codex host compaction/model transitions explicitly emit
                # turn_aborted immediately followed by a replacement
                # task_started with no repeated user item. Carry only this
                # scoped turn's provenance across a short timestamp window;
                # never infer ownership from language or arbitrary later work.
                aborted_at = self._record_time(record)
                if turn.get("user_seen") and aborted_at is not None:
                    self._continuation_seed = (dict(turn), aborted_at, turn_id, offset)
            else:
                final = str(payload.get("last_agent_message") or turn.get("final") or "").strip()
                if final and turn.get("user_seen"):
                    self.latest_final_text = final
                self._emit(events, turn_id, "final", final, offset)
                if not final and payload.get("error"):
                    error = payload["error"]
                    text = str(error.get("message") if isinstance(error, dict) else error).strip()
                    self._emit(events, turn_id, "error", text, offset)
            if self.active_turn_id == turn_id:
                self.active_turn_id = self.active_start_offset = None
            self.turns.pop(turn_id, None)


def inspect_rollout(
    thread_id: str, *, hinted_path: Optional[str] = None, codex_home: Optional[str] = None,
) -> Optional[RolloutSnapshot]:
    path = resolve_rollout_path(thread_id, hinted_path=hinted_path, codex_home=codex_home)
    if path is None:
        return None
    stat = path.stat()
    tail = RolloutTail(thread_id, path, device=stat.st_dev, inode=stat.st_ino)
    # Initial binding is rare and correctness-sensitive. Scan linearly with
    # bounded memory so an active turn older than the rolling bootstrap window
    # and the true last final are still discovered exactly.
    tail.prime_full(stat.st_size)
    return RolloutSnapshot(
        path=str(path), device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
        active_turn_id=tail.active_turn_id, active_start_offset=tail.active_start_offset,
        latest_final_text=tail.latest_final_text,
    )
