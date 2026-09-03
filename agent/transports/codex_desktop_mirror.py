"""Incremental reader for persisted Codex desktop turn activity.

Codex app-server clients and the desktop app append to the same rollout JSONL.
This reader extracts the user prompt, public progress summaries/commentary, and
durable completion. The gateway separately registers the exact turn IDs it
starts through Hermes and excludes only those IDs, avoiding unreliable
client-origin heuristics.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_GOAL_CONTEXT_RE = re.compile(
    r"<codex_internal_context\b[^>]*\bsource\s*=\s*"
    r"(?:\"goal\"|'goal')[^>]*>",
    re.IGNORECASE,
)
_GOAL_OBJECTIVE_RE = re.compile(
    r"<objective>\s*(?P<objective>.*?)\s*</objective>",
    re.IGNORECASE | re.DOTALL,
)
_RUNTIME_CONTEXT_PREFIXES = (
    "<recommended_plugins>",
    "# agents.md instructions",
    "<environment_context>",
    "<permissions instructions>",
    "<collaboration_mode>",
    "<apps_instructions>",
    "<plugins_instructions>",
)
_SUPERSEDED_TURN_ERROR = (
    "새 작업이 시작되어 이전 작업이 중단된 것으로 처리했습니다."
)


@dataclass(frozen=True)
class CodexDesktopProgress:
    kind: str
    text: str


@dataclass(frozen=True)
class CodexDesktopCompletion:
    turn_id: str
    final_text: str
    client_id: Optional[str]
    user_text: str = ""
    progress: tuple[CodexDesktopProgress, ...] = ()
    error_text: str = ""

    @property
    def is_desktop_originated(self) -> bool:
        return not self.client_id


@dataclass(frozen=True)
class CodexDesktopTurnUpdate:
    turn_id: str
    user_text: str
    progress: tuple[CodexDesktopProgress, ...]
    final_text: str
    client_id: Optional[str]
    completed: bool
    error_text: str = ""


@dataclass(frozen=True)
class CodexThreadRuntimeState:
    model: str
    reasoning_effort: str


def resolve_codex_rollout_path(
    thread_id: str,
    *,
    hinted_path: Optional[str] = None,
    codex_home: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a thread's rollout under ``CODEX_HOME/sessions`` safely."""
    cleaned_thread_id = str(thread_id or "").strip()
    if not _THREAD_ID_RE.fullmatch(cleaned_thread_id):
        return None

    home = Path(
        codex_home
        or os.environ.get("CODEX_HOME")
        or Path.home() / ".codex"
    ).expanduser()
    sessions_root = (home / "sessions").resolve()

    def _accepted(candidate: Path) -> Optional[Path]:
        try:
            resolved = candidate.expanduser().resolve()
            resolved.relative_to(sessions_root)
        except (OSError, ValueError):
            return None
        if (
            resolved.suffix != ".jsonl"
            or cleaned_thread_id not in resolved.name
            or not resolved.is_file()
        ):
            return None
        return resolved

    if hinted_path:
        accepted = _accepted(Path(hinted_path))
        if accepted is not None:
            return accepted

    try:
        matches = [
            accepted
            for candidate in sessions_root.rglob(
                f"*{cleaned_thread_id}*.jsonl"
            )
            if (accepted := _accepted(candidate)) is not None
        ]
    except OSError:
        return None
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def read_codex_rollout_runtime_state(
    thread_id: str,
    *,
    hinted_path: Optional[str] = None,
    codex_home: Optional[str] = None,
    max_scan_bytes: int = 16 * 1024 * 1024,
) -> Optional[CodexThreadRuntimeState]:
    """Read the latest persisted model/effort without scanning a rollout.

    ``turn_context`` is written once per turn. Reading backward from EOF keeps
    the common path proportional to the latest turn's tail rather than the
    entire long-lived desktop transcript. The bounded scan deliberately falls
    back to caller defaults when an unusually large turn pushes its context
    beyond the window.
    """
    path = resolve_codex_rollout_path(
        thread_id,
        hinted_path=hinted_path,
        codex_home=codex_home,
    )
    if path is None:
        return None
    try:
        size = path.stat().st_size
        scan_size = min(size, max(1, int(max_scan_bytes)))
        start = size - scan_size
        with path.open("rb") as handle:
            begins_at_record = start == 0
            if start > 0:
                handle.seek(start - 1)
                begins_at_record = handle.read(1) == b"\n"
            handle.seek(start)
            raw = handle.read(scan_size)
    except (OSError, ValueError):
        return None

    lines = raw.splitlines()
    if not begins_at_record and lines:
        # The window normally begins inside a JSONL record.
        lines = lines[1:]
    for line in reversed(lines):
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("type") != "turn_context":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        model = str(payload.get("model") or "").strip()
        effort = str(payload.get("effort") or "").strip().lower()
        if model or effort:
            return CodexThreadRuntimeState(
                model=model,
                reasoning_effort=effort,
            )
    return None


class CodexDesktopRolloutTail:
    """Stateful, partial-line-safe tailer for one Codex rollout JSONL."""

    def __init__(self, thread_id: str, path: Path):
        self.thread_id = str(thread_id)
        self.path = Path(path)
        self.offset = 0
        self._active_turn_id: Optional[str] = None
        self._turn_origins: dict[str, tuple[bool, Optional[str]]] = {}
        self._turn_details: dict[str, dict] = {}

    @classmethod
    def open(
        cls,
        thread_id: str,
        *,
        hinted_path: Optional[str] = None,
        codex_home: Optional[str] = None,
    ) -> Optional["CodexDesktopRolloutTail"]:
        path = resolve_codex_rollout_path(
            thread_id,
            hinted_path=hinted_path,
            codex_home=codex_home,
        )
        return cls(thread_id, path) if path is not None else None

    def scan(self) -> list[CodexDesktopCompletion]:
        """Read newly completed records without consuming partial JSONL lines."""
        completions, _updates = self._scan_new_records()
        return completions

    def scan_with_updates(
        self,
    ) -> tuple[list[CodexDesktopCompletion], list[CodexDesktopTurnUpdate]]:
        """Read new records and return completions plus live turn snapshots."""
        return self._scan_new_records()

    def _scan_new_records(
        self,
    ) -> tuple[list[CodexDesktopCompletion], list[CodexDesktopTurnUpdate]]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return [], []
        if self.offset > size:
            self.offset = 0
            self._active_turn_id = None
            self._turn_origins.clear()
            self._turn_details.clear()

        completions: list[CodexDesktopCompletion] = []
        updates: list[CodexDesktopTurnUpdate] = []
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                while True:
                    line_start = handle.tell()
                    raw = handle.readline()
                    if not raw:
                        break
                    if not raw.endswith(b"\n"):
                        handle.seek(line_start)
                        break
                    self.offset = handle.tell()
                    try:
                        record = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    self._consume_record(record, completions, updates)
        except OSError:
            return [], []
        return completions, updates

    def _consume_record(
        self,
        record: object,
        completions: list[CodexDesktopCompletion],
        updates: list[CodexDesktopTurnUpdate],
    ) -> None:
        if not isinstance(record, dict):
            return
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return

        # Recent Codex versions persist normal user/assistant messages as
        # ``response_item`` records rather than ``event_msg.user_message`` /
        # ``event_msg.agent_message``.  Keep the legacy event path below for
        # older rollouts, but process the durable response-item form first.
        if record.get("type") == "response_item":
            self._consume_response_item(payload, updates)
            return
        if record.get("type") != "event_msg":
            return
        event_type = payload.get("type")

        if event_type == "task_started":
            turn_id = str(payload.get("turn_id") or "").strip()
            if turn_id:
                previous_turn_id = self._active_turn_id
                if previous_turn_id and previous_turn_id != turn_id:
                    # A Codex thread can execute only one turn at a time. Some
                    # client/profile transitions omit ``turn_aborted`` before
                    # appending the next ``task_started``. Close that orphan
                    # here so a later Telegram attach cannot resurrect it as
                    # the current live turn.
                    self._finish_turn(
                        previous_turn_id,
                        final_text="",
                        error_text=_SUPERSEDED_TURN_ERROR,
                        completions=completions,
                        updates=updates,
                    )
                self._active_turn_id = turn_id
                self._turn_origins[turn_id] = (False, None)
                self._turn_details[turn_id] = {
                    "user_text": "",
                    "progress": [],
                    "final_text": "",
                }
            return

        if event_type == "user_message":
            turn_id = self._active_turn_id
            if turn_id:
                message = payload.get("message")
                self._record_user_message(
                    turn_id,
                    message if isinstance(message, str) else "",
                    self._client_id_from_payload(payload),
                    updates,
                )
            return

        if event_type == "agent_reasoning":
            # Codex emits terse internal status summaries here, typically in
            # English ("Planning...", "Inspecting..."). The Telegram mirror
            # intentionally shows only user-facing Korean commentary.
            return

        if event_type == "agent_message":
            turn_id = self._active_turn_id
            if not turn_id:
                return
            text = payload.get("message")
            self._record_agent_message(
                turn_id,
                text if isinstance(text, str) else "",
                str(payload.get("phase") or "").strip(),
                updates,
            )
            return

        if event_type == "item_completed":
            self._consume_completed_user_item(payload, updates)
            return

        if event_type not in {"task_complete", "turn_aborted"}:
            return
        turn_id = str(payload.get("turn_id") or "").strip()
        if not turn_id:
            return
        if event_type == "turn_aborted":
            # Codex persists an explicit terminal event when an interrupt,
            # timeout, or external teardown stops a turn. Treat it as a
            # completed error snapshot so the mirror preserves the accumulated
            # commentary, removes the misleading "working" state, and advances
            # its cursor instead of resurrecting the turn forever on reattach.
            final_text = ""
            error_text = "Codex 작업이 완료되기 전에 중단되었습니다."
        else:
            final_text = payload.get("last_agent_message")
            final_text = (
                final_text.strip() if isinstance(final_text, str) else ""
            )
            error_text = self._task_error_text(payload.get("error"))
        self._finish_turn(
            turn_id,
            final_text=final_text,
            error_text=error_text,
            completions=completions,
            updates=updates,
        )

    def _finish_turn(
        self,
        turn_id: str,
        *,
        final_text: str,
        error_text: str,
        completions: list[CodexDesktopCompletion],
        updates: list[CodexDesktopTurnUpdate],
    ) -> None:
        """Close one turn and publish a terminal snapshot when it was visible."""
        user_seen, client_id = self._turn_origins.pop(
            turn_id, (False, None)
        )
        if self._active_turn_id == turn_id:
            self._active_turn_id = None
        details = self._turn_details.pop(
            turn_id,
            {"user_text": "", "progress": [], "final_text": ""},
        )
        if user_seen and (final_text or error_text):
            details["final_text"] = final_text
            details["error_text"] = error_text
            completion = CodexDesktopCompletion(
                turn_id=turn_id,
                final_text=details["final_text"],
                client_id=client_id,
                user_text=str(details["user_text"]),
                progress=tuple(details["progress"]),
                error_text=error_text,
            )
            completions.append(completion)
            updates.append(
                CodexDesktopTurnUpdate(
                    turn_id=completion.turn_id,
                    user_text=completion.user_text,
                    progress=completion.progress,
                    final_text=completion.final_text,
                    client_id=completion.client_id,
                    completed=True,
                    error_text=completion.error_text,
                )
            )

    def _consume_response_item(
        self,
        payload: dict,
        updates: list[CodexDesktopTurnUpdate],
    ) -> None:
        """Project modern persisted Codex message items into a live turn."""
        if payload.get("type") != "message":
            return
        turn_id = self._response_item_turn_id(payload)
        if not turn_id:
            return

        role = str(payload.get("role") or "").strip().lower()
        if role == "user":
            # Goal continuations are stored as a private user-role message.
            # They must use the objective-only projection below rather than
            # exposing Codex's internal continuation contract to Telegram.
            if self._consume_goal_user_item(payload, updates, turn_id):
                return
            content = self._message_content(payload, "input_text")
            if self._is_runtime_context_message(payload):
                return
            self._record_user_message(
                turn_id,
                content,
                self._client_id_from_payload(payload),
                updates,
            )
            return

        if role == "assistant":
            self._record_agent_message(
                turn_id,
                self._message_content(payload, "output_text"),
                str(payload.get("phase") or "").strip(),
                updates,
            )

    def _consume_completed_user_item(
        self,
        payload: dict,
        updates: list[CodexDesktopTurnUpdate],
    ) -> None:
        """Attach Codex's client id to the preceding persisted user item.

        Current Codex rollouts put the visible text and turn id in a
        ``response_item`` first, then persist ``clientUserMessageId`` only in
        the following ``event_msg.item_completed.item.client_id``. Emitting a
        corrected snapshot here lets the gateway arbitrate ownership before
        it sends the first live mirror message.
        """
        item = payload.get("item")
        if not isinstance(item, dict):
            return
        item_type = str(item.get("type") or "").replace("_", "").lower()
        if item_type != "usermessage":
            return
        turn_id = str(
            payload.get("turn_id")
            or payload.get("turnId")
            or self._active_turn_id
            or ""
        ).strip()
        if not turn_id or turn_id not in self._turn_origins:
            return
        user_seen, existing_client_id = self._turn_origins.get(
            turn_id,
            (False, None),
        )
        client_id = self._client_id_from_payload(item)
        if not user_seen or not client_id or client_id == existing_client_id:
            return
        self._turn_origins[turn_id] = (True, client_id)
        details = self._turn_details.setdefault(
            turn_id,
            {"user_text": "", "progress": [], "final_text": ""},
        )
        updates.append(
            self._turn_update(
                turn_id,
                details,
                client_id,
                completed=False,
            )
        )

    def _response_item_turn_id(self, payload: dict) -> Optional[str]:
        """Resolve a response item to its persisted or currently active turn."""
        metadata = payload.get("internal_chat_message_metadata_passthrough")
        candidate = ""
        if isinstance(metadata, dict):
            candidate = str(
                metadata.get("turn_id") or metadata.get("turnId") or ""
            ).strip()
        if not candidate:
            candidate = str(
                payload.get("turn_id") or payload.get("turnId") or ""
            ).strip()
        if candidate:
            return candidate if candidate in self._turn_origins else None
        return self._active_turn_id

    @staticmethod
    def _message_content(payload: dict, content_type: str) -> str:
        return "\n".join(
            CodexDesktopRolloutTail._message_content_parts(
                payload, content_type
            )
        ).strip()

    @staticmethod
    def _message_content_parts(payload: dict, content_type: str) -> list[str]:
        content = payload.get("content")
        if not isinstance(content, list):
            return []
        return [
            str(item.get("text") or "").strip()
            for item in content
            if isinstance(item, dict)
            and item.get("type") == content_type
            and str(item.get("text") or "").strip()
        ]

    @staticmethod
    def _client_id_from_payload(payload: dict) -> Optional[str]:
        metadata = payload.get("internal_chat_message_metadata_passthrough")
        candidate = payload.get("client_id") or payload.get("clientId")
        if not candidate and isinstance(metadata, dict):
            candidate = metadata.get("client_id") or metadata.get("clientId")
        cleaned = str(candidate or "").strip()
        return cleaned or None

    @staticmethod
    def _is_runtime_context_message(payload: dict) -> bool:
        """Ignore Codex desktop's injected runtime context as a user prompt."""
        parts = [
            part.lower()
            for part in CodexDesktopRolloutTail._message_content_parts(
                payload, "input_text"
            )
        ]
        return bool(parts) and all(
            part.startswith(_RUNTIME_CONTEXT_PREFIXES) for part in parts
        )

    def _record_user_message(
        self,
        turn_id: str,
        message: str,
        client_id: Optional[str],
        updates: list[CodexDesktopTurnUpdate],
    ) -> None:
        """Record the visible user instruction and emit an active-turn update."""
        self._turn_origins[turn_id] = (True, client_id)
        details = self._turn_details.setdefault(
            turn_id,
            {"user_text": "", "progress": [], "final_text": ""},
        )
        details["user_text"] = message.strip()
        # Emit an immediate live snapshot even before the first commentary.
        # This lets a client that attaches mid-turn reconstruct the request.
        updates.append(
            self._turn_update(turn_id, details, client_id, completed=False)
        )

    def _record_agent_message(
        self,
        turn_id: str,
        text: str,
        phase: str,
        updates: list[CodexDesktopTurnUpdate],
    ) -> None:
        """Record durable public commentary or a final answer for one turn."""
        user_seen, client_id = self._turn_origins.get(turn_id, (False, None))
        if not user_seen:
            return
        details = self._turn_details.setdefault(
            turn_id,
            {"user_text": "", "progress": [], "final_text": ""},
        )
        cleaned = text.strip()
        if phase == "final_answer":
            if cleaned:
                details["final_text"] = cleaned
            updates.append(
                self._turn_update(turn_id, details, client_id, completed=False)
            )
            return
        if cleaned and re.search(r"[가-힣]", cleaned):
            item = CodexDesktopProgress(kind="commentary", text=cleaned)
            progress = details["progress"]
            if not progress or progress[-1] != item:
                progress.append(item)
            updates.append(
                self._turn_update(turn_id, details, client_id, completed=False)
            )

    def _consume_goal_user_item(
        self,
        payload: dict,
        updates: list[CodexDesktopTurnUpdate],
        turn_id: str,
    ) -> bool:
        """Recognize the private prompt that starts an automatic goal turn."""
        if (
            not turn_id
            or payload.get("type") != "message"
            or payload.get("role") != "user"
        ):
            return False
        user_seen, _client_id = self._turn_origins.get(
            turn_id, (False, None)
        )
        raw_text = self._message_content(payload, "input_text")
        if not _GOAL_CONTEXT_RE.search(raw_text):
            return False
        if user_seen:
            return True

        objective_match = _GOAL_OBJECTIVE_RE.search(raw_text)
        objective = (
            objective_match.group("objective").strip()
            if objective_match is not None
            else "활성 목표 계속 진행"
        )
        details = self._turn_details.setdefault(
            turn_id,
            {"user_text": "", "progress": [], "final_text": ""},
        )
        details["user_text"] = objective or "활성 목표 계속 진행"
        self._turn_origins[turn_id] = (True, None)
        updates.append(
            self._turn_update(
                turn_id,
                details,
                None,
                completed=False,
            )
        )
        return True

    @staticmethod
    def _task_error_text(error: object) -> str:
        if isinstance(error, str):
            return error.strip()
        if not isinstance(error, dict):
            return ""
        message = error.get("message")
        return message.strip() if isinstance(message, str) else ""

    @staticmethod
    def _turn_update(
        turn_id: str,
        details: dict,
        client_id: Optional[str],
        *,
        completed: bool,
    ) -> CodexDesktopTurnUpdate:
        return CodexDesktopTurnUpdate(
            turn_id=turn_id,
            user_text=str(details.get("user_text") or ""),
            progress=tuple(details.get("progress") or ()),
            final_text=str(details.get("final_text") or ""),
            client_id=client_id,
            completed=completed,
            error_text=str(details.get("error_text") or ""),
        )


def snapshot_codex_rollout_latest(
    thread_id: str,
    *,
    hinted_path: Optional[str] = None,
    codex_home: Optional[str] = None,
) -> tuple[Optional[str], Optional[CodexDesktopCompletion]]:
    """Return the rollout path and latest completed answer at attach time."""
    tail = CodexDesktopRolloutTail.open(
        thread_id,
        hinted_path=hinted_path,
        codex_home=codex_home,
    )
    if tail is None:
        return None, None
    completions = tail.scan()
    return str(tail.path), completions[-1] if completions else None


def snapshot_codex_rollout(
    thread_id: str,
    *,
    hinted_path: Optional[str] = None,
    codex_home: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(rollout_path, latest_completed_turn_id)`` at attach time."""
    rollout_path, latest = snapshot_codex_rollout_latest(
        thread_id,
        hinted_path=hinted_path,
        codex_home=codex_home,
    )
    return rollout_path, latest.turn_id if latest is not None else None
