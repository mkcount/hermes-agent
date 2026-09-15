"""Durable ownership store for Telegram-bound Codex threads."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from hermes_constants import get_hermes_home


@dataclass(frozen=True)
class CodexBridgeBinding:
    control_session_key: str
    thread_id: Optional[str]
    cwd: str
    generation: int
    source: SessionSource
    rollout_path: Optional[str] = None
    cursor_device: Optional[int] = None
    cursor_inode: Optional[int] = None
    cursor_offset: int = 0
    last_event_id: Optional[str] = None
    pending_new: bool = False
    codex_model: Optional[str] = None
    reasoning_effort: Optional[str] = None

    @property
    def active(self) -> bool:
        return bool(self.thread_id)


@dataclass(frozen=True)
class DurableCodexInput:
    input_id: str
    control_session_key: str
    lane_session_key: str
    thread_id: str
    generation: int
    event: MessageEvent
    state: str
    final_text: Optional[str] = None
    codex_turn_id: Optional[str] = None
    delivery_owner: str = "none"
    turn_outcome: str = "unknown"
    continuation_turn_id: Optional[str] = None
    continuation_turn_ids: tuple[str, ...] = ()
    physical_turn_status: str = "unknown"
    logical_input_status: str = "open"
    output_kind: str = "none"
    delivery_status: str = "none"
    delivery_obligation_id: Optional[str] = None
    legacy_recovery_required: bool = False


@dataclass(frozen=True)
class DurableCodexProgress:
    control_session_key: str
    generation: int
    logical_turn_id: str
    segments: tuple[str, ...]
    content: str
    message_id: Optional[str]
    state: str
    last_error: Optional[str]
    attempt_count: int
    next_attempt_at: float
    updated_at: float


class CodexBridgeStore:
    """Small SQLite authority colocated with Hermes' profile state database."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else get_hermes_home() / "state.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._transaction() as conn:
            self._initialize(conn)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            from hermes_state_wal import apply_wal_with_fallback

            apply_wal_with_fallback(conn, db_label="state.db (codex_bridge)")
            with closing(conn), conn:
                yield conn
        except BaseException:
            conn.close()
            raise

    @staticmethod
    def _initialize(conn: sqlite3.Connection) -> None:
        from gateway.delivery_ledger import initialize_delivery_schema

        initialize_delivery_schema(conn)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS codex_bridge_bindings (
                control_session_key TEXT PRIMARY KEY,
                thread_id TEXT,
                cwd TEXT NOT NULL DEFAULT '',
                generation INTEGER NOT NULL,
                source_json TEXT NOT NULL,
                rollout_path TEXT,
                cursor_device INTEGER,
                cursor_inode INTEGER,
                cursor_offset INTEGER NOT NULL DEFAULT 0,
                last_event_id TEXT,
                pending_new INTEGER NOT NULL DEFAULT 0,
                codex_model TEXT,
                reasoning_effort TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS codex_bridge_inputs (
                input_id TEXT PRIMARY KEY,
                control_session_key TEXT NOT NULL,
                lane_session_key TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                state TEXT NOT NULL,
                owner_pid INTEGER,
                owner_started_at INTEGER,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                final_text TEXT,
                codex_turn_id TEXT,
                delivery_owner TEXT NOT NULL DEFAULT 'none',
                turn_outcome TEXT NOT NULL DEFAULT 'unknown',
                continuation_turn_id TEXT,
                continuation_turn_ids_json TEXT NOT NULL DEFAULT '[]',
                physical_turn_status TEXT NOT NULL DEFAULT 'unknown',
                logical_input_status TEXT NOT NULL DEFAULT 'open',
                output_kind TEXT NOT NULL DEFAULT 'none',
                delivery_status TEXT NOT NULL DEFAULT 'none',
                delivery_obligation_id TEXT,
                finalized_at REAL,
                legacy_recovery_required INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_codex_bridge_inputs_state ON codex_bridge_inputs(state, created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_codex_bridge_inputs_lane ON codex_bridge_inputs(lane_session_key)"
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS codex_bridge_progress (
                control_session_key TEXT NOT NULL,
                generation INTEGER NOT NULL,
                logical_turn_id TEXT NOT NULL,
                segments_json TEXT NOT NULL,
                content TEXT NOT NULL,
                message_id TEXT,
                state TEXT NOT NULL,
                last_error TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (control_session_key, generation, logical_turn_id)
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_codex_bridge_progress_state "
            "ON codex_bridge_progress(state, updated_at)"
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(codex_bridge_bindings)")}
        if "pending_new" not in columns:
            conn.execute(
                "ALTER TABLE codex_bridge_bindings ADD COLUMN pending_new INTEGER NOT NULL DEFAULT 0"
            )
        if "codex_model" not in columns:
            conn.execute("ALTER TABLE codex_bridge_bindings ADD COLUMN codex_model TEXT")
        if "reasoning_effort" not in columns:
            conn.execute("ALTER TABLE codex_bridge_bindings ADD COLUMN reasoning_effort TEXT")
        input_columns = {row[1] for row in conn.execute("PRAGMA table_info(codex_bridge_inputs)")}
        if "owner_started_at" not in input_columns:
            conn.execute("ALTER TABLE codex_bridge_inputs ADD COLUMN owner_started_at INTEGER")
        if "codex_turn_id" not in input_columns:
            conn.execute("ALTER TABLE codex_bridge_inputs ADD COLUMN codex_turn_id TEXT")
        if "delivery_owner" not in input_columns:
            conn.execute(
                "ALTER TABLE codex_bridge_inputs ADD COLUMN delivery_owner TEXT NOT NULL DEFAULT 'none'"
            )
            conn.execute(
                """UPDATE codex_bridge_inputs
                   SET delivery_owner=CASE
                       WHEN state IN ('routed','pending','admitted','executing','submitting','running')
                           THEN 'runner'
                       WHEN state IN ('uncertain','continuation_pending') THEN 'rollout'
                       WHEN state IN ('executed','recovery_output') THEN 'ledger'
                       ELSE 'none'
                   END"""
            )
        if "turn_outcome" not in input_columns:
            conn.execute(
                "ALTER TABLE codex_bridge_inputs ADD COLUMN turn_outcome TEXT NOT NULL DEFAULT 'unknown'"
            )
        if "continuation_turn_id" not in input_columns:
            conn.execute("ALTER TABLE codex_bridge_inputs ADD COLUMN continuation_turn_id TEXT")
        semantic_columns = {
            "continuation_turn_ids_json": "TEXT NOT NULL DEFAULT '[]'",
            "physical_turn_status": "TEXT NOT NULL DEFAULT 'unknown'",
            "logical_input_status": "TEXT NOT NULL DEFAULT 'open'",
            "output_kind": "TEXT NOT NULL DEFAULT 'none'",
            "delivery_status": "TEXT NOT NULL DEFAULT 'none'",
            "delivery_obligation_id": "TEXT",
            "finalized_at": "REAL",
            "legacy_recovery_required": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, declaration in semantic_columns.items():
            if name not in input_columns:
                conn.execute(f"ALTER TABLE codex_bridge_inputs ADD COLUMN {name} {declaration}")
                if name == "legacy_recovery_required":
                    conn.execute(
                        """UPDATE codex_bridge_inputs SET legacy_recovery_required=1
                           WHERE state='continuation_pending'"""
                    )
        # Translate legacy overloaded states once.  These columns are the
        # durable contract going forward; ``state`` remains the scheduler's
        # compact index and is no longer asked to encode every dimension.
        conn.execute(
            """UPDATE codex_bridge_inputs SET
                   physical_turn_status=CASE
                       WHEN physical_turn_status='unknown' AND turn_outcome IN
                            ('running','completed','interrupted','failed','cancelled')
                           THEN turn_outcome ELSE physical_turn_status END,
                   logical_input_status=CASE
                       WHEN logical_input_status!='open' THEN logical_input_status
                       WHEN state='continuation_pending' THEN 'awaiting_successor'
                       WHEN state='uncertain' THEN 'reconciling'
                       WHEN state='cancelled' THEN 'cancelled'
                       WHEN state='completed' AND turn_outcome='completed' THEN 'completed'
                       WHEN state='completed' THEN 'failed'
                       WHEN state IN ('executed','recovery_output') THEN
                           CASE WHEN turn_outcome='completed' THEN 'completed' ELSE 'failed' END
                       ELSE 'open' END,
                   output_kind=CASE
                       WHEN output_kind!='none' THEN output_kind
                       WHEN final_text IS NULL THEN 'none'
                       WHEN turn_outcome='completed' THEN 'final_answer'
                       ELSE 'terminal_notice' END,
                   delivery_status=CASE
                       WHEN delivery_status!='none' THEN delivery_status
                       WHEN state='completed' THEN 'delivered'
                       WHEN state IN ('executed','recovery_output') THEN 'pending'
                       ELSE 'none' END,
                   finalized_at=CASE
                       WHEN finalized_at IS NULL AND state='completed' THEN updated_at
                       ELSE finalized_at END"""
        )
        progress_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(codex_bridge_progress)")
        }
        if "attempt_count" not in progress_columns:
            conn.execute(
                "ALTER TABLE codex_bridge_progress "
                "ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0"
            )
        if "next_attempt_at" not in progress_columns:
            conn.execute(
                "ALTER TABLE codex_bridge_progress "
                "ADD COLUMN next_attempt_at REAL NOT NULL DEFAULT 0"
            )

    @staticmethod
    def _owner_stamp() -> tuple[int, Optional[int]]:
        from gateway.status import get_process_start_time

        pid = os.getpid()
        try:
            return pid, get_process_start_time(pid)
        except Exception:
            return pid, None

    @staticmethod
    def _owner_alive(pid: Any, started_at: Any) -> bool:
        if not pid:
            return False
        try:
            from gateway.status import _pid_exists, get_process_start_time

            pid_int = int(pid)
            current = get_process_start_time(pid_int)
            if current is None:
                return bool(_pid_exists(pid_int))
            return started_at is None or int(current) == int(started_at)
        except Exception:
            return False

    @staticmethod
    def _binding_from_row(row: tuple[Any, ...]) -> CodexBridgeBinding:
        source = SessionSource.from_dict(json.loads(row[4]))
        return CodexBridgeBinding(
            control_session_key=row[0], thread_id=row[1], cwd=row[2] or "",
            generation=int(row[3]), source=source, rollout_path=row[5],
            cursor_device=row[6], cursor_inode=row[7], cursor_offset=int(row[8] or 0),
            last_event_id=row[9], pending_new=bool(row[10]),
            codex_model=str(row[11]) if row[11] else None,
            reasoning_effort=str(row[12]).lower() if row[12] else None,
        )

    def get_binding(self, control_session_key: str) -> Optional[CodexBridgeBinding]:
        with self._lock, self._transaction() as conn:
            row = conn.execute(
                """SELECT control_session_key, thread_id, cwd, generation, source_json,
                          rollout_path, cursor_device, cursor_inode, cursor_offset, last_event_id,
                          pending_new, codex_model, reasoning_effort
                   FROM codex_bridge_bindings WHERE control_session_key=?""",
                (control_session_key,),
            ).fetchone()
        return self._binding_from_row(row) if row else None

    def list_bindings(self) -> list[CodexBridgeBinding]:
        with self._lock, self._transaction() as conn:
            rows = conn.execute(
                """SELECT control_session_key, thread_id, cwd, generation, source_json,
                          rollout_path, cursor_device, cursor_inode, cursor_offset, last_event_id,
                          pending_new, codex_model, reasoning_effort
                   FROM codex_bridge_bindings WHERE thread_id IS NOT NULL ORDER BY updated_at DESC"""
            ).fetchall()
        return [self._binding_from_row(row) for row in rows]

    @staticmethod
    def _progress_from_row(row: tuple[Any, ...]) -> DurableCodexProgress:
        try:
            segments = tuple(str(value) for value in json.loads(row[3]) if str(value).strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            segments = ()
        return DurableCodexProgress(
            control_session_key=str(row[0]), generation=int(row[1]),
            logical_turn_id=str(row[2]), segments=segments, content=str(row[4]),
            message_id=str(row[5]) if row[5] is not None else None,
            state=str(row[6]), last_error=str(row[7]) if row[7] is not None else None,
            attempt_count=int(row[8] or 0), next_attempt_at=float(row[9] or 0),
            updated_at=float(row[10]),
        )

    @staticmethod
    def _truncate_progress(value: str, max_units: int) -> str:
        """Keep one progress bubble within a UTF-16 based transport limit."""
        limit = max(1, int(max_units))
        if len(value.encode("utf-16-le")) // 2 <= limit:
            return value
        kept: list[str] = []
        used = 0
        for character in value:
            width = 2 if ord(character) > 0xFFFF else 1
            if used + width >= limit:
                break
            kept.append(character)
            used += width
        return "".join(kept).rstrip() + "…"

    def upsert_progress(
        self, binding: CodexBridgeBinding, logical_turn_id: str, segments: list[str],
        *, max_segments: int = 8, max_chars: int = 12_000,
    ) -> DurableCodexProgress:
        """Durably stage rolling commentary before its rollout cursor is acknowledged."""
        identity = str(logical_turn_id or "").strip()
        if not identity:
            raise ValueError("logical_turn_id is required")
        candidates = [str(value).strip() for value in segments if value and str(value).strip()]
        if not candidates:
            raise ValueError("at least one progress segment is required")
        now = time.time()
        with self._lock, self._transaction() as conn:
            current = conn.execute(
                "SELECT thread_id, generation FROM codex_bridge_bindings WHERE control_session_key=?",
                (binding.control_session_key,),
            ).fetchone()
            if (
                current is None
                or str(current[0] or "") != str(binding.thread_id or "")
                or int(current[1]) != binding.generation
            ):
                raise RuntimeError("Codex binding changed while staging progress")
            previous = conn.execute(
                """SELECT segments_json, message_id, state, last_error,
                          attempt_count, next_attempt_at
                   FROM codex_bridge_progress
                   WHERE control_session_key=? AND generation=? AND logical_turn_id=?""",
                (binding.control_session_key, binding.generation, identity),
            ).fetchone()
            if previous is not None and str(previous[2]) == "finalized":
                row = conn.execute(
                    """SELECT control_session_key, generation, logical_turn_id, segments_json,
                              content, message_id, state, last_error, attempt_count,
                              next_attempt_at, updated_at
                       FROM codex_bridge_progress
                       WHERE control_session_key=? AND generation=? AND logical_turn_id=?""",
                    (binding.control_session_key, binding.generation, identity),
                ).fetchone()
                return self._progress_from_row(row)
            try:
                accumulated = list(json.loads(previous[0])) if previous is not None else []
            except (TypeError, ValueError, json.JSONDecodeError):
                accumulated = []
            changed = False
            for candidate in candidates:
                if candidate not in accumulated:
                    accumulated.append(candidate)
                    changed = True
            while len(accumulated) > max(1, int(max_segments)):
                accumulated.pop(0)
                changed = True
            while (
                len("\n\n".join(accumulated).encode("utf-16-le")) // 2
                > max(1, int(max_chars))
                and len(accumulated) > 1
            ):
                accumulated.pop(0)
                changed = True
            if accumulated:
                trimmed = self._truncate_progress(accumulated[0], max_chars)
                if trimmed != accumulated[0]:
                    accumulated[0] = trimmed
                    changed = True
            content = "💻 Codex 진행\n\n" + "\n\n".join(accumulated)
            state = "pending" if previous is None or changed else str(previous[2])
            last_error = None if previous is None or changed else previous[3]
            attempt_count = 0 if previous is None or changed else int(previous[4] or 0)
            next_attempt_at = 0.0 if previous is None or changed else float(previous[5] or 0)
            conn.execute(
                """INSERT INTO codex_bridge_progress (
                       control_session_key, generation, logical_turn_id, segments_json,
                       content, message_id, state, last_error, attempt_count,
                       next_attempt_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(control_session_key, generation, logical_turn_id) DO UPDATE SET
                       segments_json=excluded.segments_json, content=excluded.content,
                       state=excluded.state, last_error=excluded.last_error,
                       attempt_count=excluded.attempt_count,
                       next_attempt_at=excluded.next_attempt_at,
                       updated_at=excluded.updated_at""",
                (
                    binding.control_session_key, binding.generation, identity,
                    json.dumps(accumulated, ensure_ascii=False, separators=(",", ":")),
                    content, previous[1] if previous is not None else None,
                    state, last_error, attempt_count, next_attempt_at, now, now,
                ),
            )
            row = conn.execute(
                """SELECT control_session_key, generation, logical_turn_id, segments_json,
                          content, message_id, state, last_error, attempt_count,
                          next_attempt_at, updated_at
                   FROM codex_bridge_progress
                   WHERE control_session_key=? AND generation=? AND logical_turn_id=?""",
                (binding.control_session_key, binding.generation, identity),
            ).fetchone()
        return self._progress_from_row(row)

    def list_progress(
        self, binding: CodexBridgeBinding, *, pending_only: bool = False,
        ready_only: bool = False,
    ) -> list[DurableCodexProgress]:
        state_filter = " AND state='pending'" if pending_only else ""
        ready_filter = " AND next_attempt_at<=?" if ready_only else ""
        params: tuple[Any, ...] = (binding.control_session_key, binding.generation)
        if ready_only:
            params += (time.time(),)
        with self._lock, self._transaction() as conn:
            rows = conn.execute(
                """SELECT control_session_key, generation, logical_turn_id, segments_json,
                          content, message_id, state, last_error, attempt_count,
                          next_attempt_at, updated_at
                   FROM codex_bridge_progress
                   WHERE control_session_key=? AND generation=?""" + state_filter + ready_filter +
                " ORDER BY updated_at",
                params,
            ).fetchall()
        return [self._progress_from_row(row) for row in rows]

    def mark_progress_delivered(
        self, progress: DurableCodexProgress, message_id: Optional[str],
    ) -> bool:
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_progress
                   SET message_id=?, state='delivered', last_error=NULL,
                       attempt_count=0, next_attempt_at=0, updated_at=?
                   WHERE control_session_key=? AND generation=? AND logical_turn_id=?
                     AND content=?""",
                (
                    str(message_id) if message_id is not None else progress.message_id,
                    time.time(), progress.control_session_key, progress.generation,
                    progress.logical_turn_id, progress.content,
                ),
            )
        return bool(cur.rowcount)

    def mark_progress_failed(self, progress: DurableCodexProgress, error: str) -> None:
        attempt_count = progress.attempt_count + 1
        next_attempt_at = time.time() + min(300.0, 1.5 * (2 ** min(progress.attempt_count, 8)))
        with self._lock, self._transaction() as conn:
            conn.execute(
                """UPDATE codex_bridge_progress
                   SET state='pending', last_error=?, attempt_count=?,
                       next_attempt_at=?, updated_at=?
                   WHERE control_session_key=? AND generation=? AND logical_turn_id=?
                     AND content=?""",
                (
                    str(error or "progress delivery failed")[:500], attempt_count,
                    next_attempt_at, time.time(),
                    progress.control_session_key, progress.generation,
                    progress.logical_turn_id, progress.content,
                ),
            )

    def clear_progress_message(self, progress: DurableCodexProgress, error: str) -> None:
        attempt_count = progress.attempt_count + 1
        next_attempt_at = time.time() + min(30.0, 1.5 * (2 ** min(progress.attempt_count, 4)))
        with self._lock, self._transaction() as conn:
            conn.execute(
                """UPDATE codex_bridge_progress
                   SET message_id=NULL, state='pending', last_error=?, attempt_count=?,
                       next_attempt_at=?, updated_at=?
                   WHERE control_session_key=? AND generation=? AND logical_turn_id=?
                     AND content=?""",
                (
                    str(error or "progress message is no longer editable")[:500],
                    attempt_count, next_attempt_at, time.time(),
                    progress.control_session_key, progress.generation,
                    progress.logical_turn_id, progress.content,
                ),
            )

    def complete_progress(self, binding: CodexBridgeBinding, logical_turn_id: str) -> None:
        """Freeze a preview once a terminal output intent exists.

        Keeping a compact tombstone prevents late JSONL commentary from
        recreating or editing the progress message after the final answer.
        """
        with self._lock, self._transaction() as conn:
            conn.execute(
                """UPDATE codex_bridge_progress
                   SET state='finalized', next_attempt_at=0, updated_at=?
                   WHERE control_session_key=? AND generation=? AND logical_turn_id=?
                     AND state!='finalized'""",
                (time.time(), binding.control_session_key, binding.generation, logical_turn_id),
            )

    def bind(
        self, control_session_key: str, source: SessionSource, *, thread_id: str, cwd: str = "",
        rollout_path: Optional[str] = None, cursor_device: Optional[int] = None,
        cursor_inode: Optional[int] = None, cursor_offset: int = 0,
        last_event_id: Optional[str] = None,
        pending_new: bool = False,
        codex_model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> CodexBridgeBinding:
        cleaned = str(thread_id or "").strip()
        if not cleaned:
            raise ValueError("thread_id is required")
        now = time.time()
        selected_model = str(codex_model or "").strip() or None
        selected_effort = str(reasoning_effort or "").strip().lower() or None
        source_json = json.dumps(source.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._transaction() as conn:
            current = conn.execute(
                "SELECT generation FROM codex_bridge_bindings WHERE control_session_key=?",
                (control_session_key,),
            ).fetchone()
            generation = int(current[0] if current else 0) + 1
            conn.execute(
                """INSERT INTO codex_bridge_bindings (
                       control_session_key, thread_id, cwd, generation, source_json,
                       rollout_path, cursor_device, cursor_inode, cursor_offset, last_event_id,
                       pending_new, codex_model, reasoning_effort, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(control_session_key) DO UPDATE SET
                       thread_id=excluded.thread_id, cwd=excluded.cwd,
                       generation=excluded.generation, source_json=excluded.source_json,
                       rollout_path=excluded.rollout_path, cursor_device=excluded.cursor_device,
                       cursor_inode=excluded.cursor_inode, cursor_offset=excluded.cursor_offset,
                       last_event_id=excluded.last_event_id, pending_new=excluded.pending_new,
                       codex_model=excluded.codex_model,
                       reasoning_effort=excluded.reasoning_effort,
                       updated_at=excluded.updated_at""",
                (control_session_key, cleaned, cwd or "", generation, source_json,
                 rollout_path, cursor_device, cursor_inode, max(0, int(cursor_offset)),
                 last_event_id, int(bool(pending_new)), selected_model, selected_effort, now, now),
            )
            conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='cancelled', delivery_owner='none', turn_outcome='cancelled',
                       owner_pid=NULL, owner_started_at=NULL, updated_at=?,
                       last_error='binding changed'
                   WHERE control_session_key=? AND generation<>?
                     AND state IN (
                         'routed','pending','admitted','executing','submitting','running','uncertain',
                         'continuation_pending'
                     )""",
                (now, control_session_key, generation),
            )
            conn.execute(
                "DELETE FROM codex_bridge_progress WHERE control_session_key=? AND generation<>?",
                (control_session_key, generation),
            )
        return self.get_binding(control_session_key)  # type: ignore[return-value]

    def unbind(self, control_session_key: str, source: SessionSource) -> int:
        now = time.time()
        source_json = json.dumps(source.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._transaction() as conn:
            current = conn.execute(
                "SELECT generation FROM codex_bridge_bindings WHERE control_session_key=?",
                (control_session_key,),
            ).fetchone()
            generation = int(current[0] if current else 0) + 1
            conn.execute(
                """INSERT INTO codex_bridge_bindings
                       (control_session_key, thread_id, cwd, generation, source_json,
                        cursor_offset, created_at, updated_at)
                   VALUES (?, NULL, '', ?, ?, 0, ?, ?)
                   ON CONFLICT(control_session_key) DO UPDATE SET
                       thread_id=NULL, cwd='', generation=excluded.generation,
                       source_json=excluded.source_json, rollout_path=NULL,
                       cursor_device=NULL, cursor_inode=NULL, cursor_offset=0,
                       last_event_id=NULL, pending_new=0, codex_model=NULL,
                       reasoning_effort=NULL, updated_at=excluded.updated_at""",
                (control_session_key, generation, source_json, now, now),
            )
            conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='cancelled', delivery_owner='none', turn_outcome='cancelled',
                       owner_pid=NULL, owner_started_at=NULL, updated_at=?,
                       last_error='binding changed'
                   WHERE control_session_key=? AND state IN (
                       'routed','pending','admitted','executing','submitting','running','uncertain',
                       'continuation_pending'
                   )""",
                (now, control_session_key),
            )
            conn.execute(
                "DELETE FROM codex_bridge_progress WHERE control_session_key=?",
                (control_session_key,),
            )
        return generation

    def promote_pending(self, binding: CodexBridgeBinding, thread_id: str) -> Optional[CodexBridgeBinding]:
        """CAS a /ns reservation to the real app-server thread without rotating its grant."""
        cleaned = str(thread_id or "").strip()
        if not cleaned:
            return None
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_bindings
                   SET thread_id=?, pending_new=0, rollout_path=NULL,
                       cursor_device=NULL, cursor_inode=NULL, cursor_offset=0,
                       last_event_id=NULL, updated_at=?
                   WHERE control_session_key=? AND generation=? AND thread_id=? AND pending_new=1""",
                (cleaned, time.time(), binding.control_session_key, binding.generation, binding.thread_id),
            )
            if cur.rowcount:
                conn.execute(
                    """UPDATE codex_bridge_inputs SET thread_id=?, updated_at=?
                       WHERE control_session_key=? AND generation=? AND thread_id=?
                         AND state IN (
                             'routed','pending','admitted','executing','submitting','running','uncertain',
                             'continuation_pending'
                         )""",
                    (cleaned, time.time(), binding.control_session_key, binding.generation, binding.thread_id),
                )
        return self.get_binding(binding.control_session_key) if cur.rowcount else None

    def set_inference(
        self, binding: CodexBridgeBinding, *, codex_model: str, reasoning_effort: str,
    ) -> Optional[CodexBridgeBinding]:
        """Atomically update the model/effort pair while the selected binding generation is current."""
        model = str(codex_model or "").strip()
        effort = str(reasoning_effort or "").strip().lower()
        if not model or not effort:
            raise ValueError("codex_model and reasoning_effort are required")
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_bindings
                   SET codex_model=?, reasoning_effort=?, updated_at=?
                   WHERE control_session_key=? AND generation=? AND thread_id=?""",
                (model, effort, time.time(), binding.control_session_key,
                 binding.generation, binding.thread_id),
            )
        return self.get_binding(binding.control_session_key) if cur.rowcount else None

    def update_cursor(
        self, binding: CodexBridgeBinding, *, rollout_path: str, device: int, inode: int,
        offset: int, last_event_id: Optional[str],
    ) -> bool:
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_bindings SET rollout_path=?, cursor_device=?, cursor_inode=?,
                          cursor_offset=?, last_event_id=?, updated_at=?
                   WHERE control_session_key=? AND generation=? AND thread_id=?""",
                (rollout_path, int(device), int(inode), max(0, int(offset)), last_event_id,
                 time.time(), binding.control_session_key, binding.generation, binding.thread_id),
            )
        return bool(cur.rowcount)

    @staticmethod
    def input_id(control_session_key: str, event: MessageEvent) -> str:
        existing = str((event.metadata or {}).get("codex_bridge_input_id") or "").strip()
        if existing:
            return existing
        if event.message_id:
            raw = "\0".join((control_session_key, event.source.platform.value, str(event.message_id)))
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return uuid.uuid4().hex

    @staticmethod
    def _event_to_json(event: MessageEvent) -> str:
        data = {
            "text": event.text, "message_type": event.message_type.value,
            "source": event.source.to_dict(), "message_id": event.message_id,
            "ledger_message_id": event.ledger_message_id,
            "platform_update_id": event.platform_update_id,
            "media_urls": list(event.media_urls or []), "media_types": list(event.media_types or []),
            "media_text_inlined": list(event.media_text_inlined or []),
            "reply_to_message_id": event.reply_to_message_id, "reply_to_text": event.reply_to_text,
            "reply_to_author_id": event.reply_to_author_id,
            "reply_to_author_name": event.reply_to_author_name,
            "reply_to_is_own_message": bool(event.reply_to_is_own_message),
            "auto_skill": event.auto_skill, "channel_prompt": event.channel_prompt,
            "channel_context": event.channel_context, "internal": bool(event.internal),
            "metadata": dict(event.metadata or {}), "timestamp": event.timestamp.isoformat(),
            "allow_gateway_control": bool(event.allow_gateway_control),
        }
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _event_from_json(raw: str) -> MessageEvent:
        data = json.loads(raw)
        metadata = dict(data.get("metadata") or {})
        metadata["codex_bridge_recovered"] = True
        return MessageEvent(
            text=str(data.get("text") or ""),
            message_type=MessageType(data.get("message_type") or "text"),
            source=SessionSource.from_dict(data["source"]), message_id=data.get("message_id"),
            ledger_message_id=data.get("ledger_message_id"),
            platform_update_id=data.get("platform_update_id"),
            media_urls=list(data.get("media_urls") or []), media_types=list(data.get("media_types") or []),
            media_text_inlined=list(data.get("media_text_inlined") or []),
            reply_to_message_id=data.get("reply_to_message_id"), reply_to_text=data.get("reply_to_text"),
            reply_to_author_id=data.get("reply_to_author_id"),
            reply_to_author_name=data.get("reply_to_author_name"),
            reply_to_is_own_message=bool(data.get("reply_to_is_own_message")),
            auto_skill=data.get("auto_skill"), channel_prompt=data.get("channel_prompt"),
            channel_context=data.get("channel_context"), internal=bool(data.get("internal")),
            metadata=metadata,
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            allow_gateway_control=bool(data.get("allow_gateway_control", True)),
        )

    def enqueue_input(
        self, binding: CodexBridgeBinding, lane_session_key: str, event: MessageEvent,
    ) -> tuple[str, str, bool]:
        """Persist pre-guard routing; return ``(id, state, inserted)``.

        ``routed`` rows are deliberately not restart-recoverable: authorization,
        pause, and plugin gates have not accepted them yet. The runner claims
        one directly as ``executing`` only after all those gates pass.
        """
        input_id = self.input_id(binding.control_session_key, event)
        now = time.time()
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO codex_bridge_inputs
                       (input_id, control_session_key, lane_session_key, thread_id, generation,
                        event_json, state, delivery_owner, turn_outcome,
                        physical_turn_status, logical_input_status, output_kind,
                        delivery_status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'routed', 'runner', 'pending',
                           'pending', 'open', 'none', 'none', ?, ?)""",
                (input_id, binding.control_session_key, lane_session_key, binding.thread_id,
                 binding.generation, self._event_to_json(event), now, now),
            )
            inserted = bool(cur.rowcount)
            row = conn.execute(
                "SELECT state FROM codex_bridge_inputs WHERE input_id=?", (input_id,)
            ).fetchone()
        return input_id, str(row[0]), inserted

    def mark_executing(self, input_id: str) -> bool:
        """Atomically claim one gate-approved routed/recovered input for execution."""
        owner_pid, owner_started_at = self._owner_stamp()
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='executing', owner_pid=?, owner_started_at=?,
                       delivery_owner='runner', turn_outcome='pending',
                       physical_turn_status='pending', logical_input_status='running',
                       output_kind='none', delivery_status='none', finalized_at=NULL,
                       updated_at=?, last_error=NULL
                   WHERE input_id=? AND state IN ('routed','pending')""",
                (owner_pid, owner_started_at, time.time(), input_id),
            )
        return bool(cur.rowcount)

    def mark_submitting(self, input_id: str) -> bool:
        """Fence the request immediately before ``turn/start`` is written."""
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='submitting', physical_turn_status='submitting', updated_at=?
                   WHERE input_id=? AND state='executing'""",
                (time.time(), input_id),
            )
        return bool(cur.rowcount)

    def mark_running(self, input_id: str, turn_id: str) -> bool:
        """Record the server-assigned turn after ``turn/start`` is acknowledged."""
        cleaned = str(turn_id or "").strip()
        if not cleaned:
            return False
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='running', codex_turn_id=?, continuation_turn_id=NULL,
                       continuation_turn_ids_json='[]', delivery_owner='runner',
                       turn_outcome='running', physical_turn_status='running',
                       logical_input_status='running', updated_at=?
                   WHERE input_id=? AND state IN ('submitting','executing')""",
                (cleaned, time.time(), input_id),
            )
        return bool(cur.rowcount)

    def mark_uncertain(self, input_id: str, error: str = "") -> bool:
        """Preserve an accepted-or-possibly-accepted request without replaying it."""
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='uncertain', delivery_owner='rollout', turn_outcome='unknown',
                       physical_turn_status='unknown', logical_input_status='reconciling',
                       delivery_status='none', output_kind='none',
                       owner_pid=NULL, owner_started_at=NULL, updated_at=?, last_error=?
                   WHERE input_id=? AND state IN ('submitting','running')""",
                (time.time(), str(error or "submission result unknown")[:500], input_id),
            )
        return bool(cur.rowcount)

    def mark_continuation_pending(
        self, input_id: str, error: str = "", *, continuation_turn_id: str = "",
        continuation_turn_ids: tuple[str, ...] = (), physical_turn_status: str = "interrupted",
    ) -> bool:
        """Release the live runner while an external owner continues the same Codex turn."""
        continued = str(continuation_turn_id or "").strip() or None
        chain = tuple(dict.fromkeys(
            str(value).strip() for value in continuation_turn_ids if str(value).strip()
        ))
        if continued and continued not in chain:
            chain += (continued,)
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='continuation_pending', owner_pid=NULL, owner_started_at=NULL,
                       delivery_owner='rollout', turn_outcome='continuing', final_text=NULL,
                       continuation_turn_id=?, continuation_turn_ids_json=?,
                       physical_turn_status=?, logical_input_status='awaiting_successor',
                       output_kind='none', delivery_status='none', finalized_at=NULL,
                       updated_at=?, last_error=?
                   WHERE input_id=? AND state IN ('executing','submitting','running')""",
                (
                    continued, json.dumps(chain, separators=(",", ":")),
                    str(physical_turn_status or "unknown"), time.time(),
                    str(error or "external continuation pending")[:500], input_id,
                ),
            )
        return bool(cur.rowcount)

    def mark_executed(
        self, input_id: str, final_text: str, *, turn_outcome: str = "completed",
        output_kind: Optional[str] = None,
    ) -> bool:
        physical = str(turn_outcome or "unknown").strip().lower()
        kind = str(output_kind or (
            "final_answer" if physical == "completed" else "terminal_notice"
        )).strip().lower()
        return self.stage_terminal_output(
            input_id, final_text, physical_turn_status=physical, output_kind=kind,
        ) is not None

    def stage_terminal_output(
        self, input_id: str, final_text: str, *, physical_turn_status: str,
        output_kind: str, adapter_profile: Optional[str] = None,
        recovery: bool = False,
    ) -> Optional[str]:
        """Atomically reduce a logical input and persist its canonical send intent."""
        cleaned = str(final_text or "").strip()
        physical = str(physical_turn_status or "unknown").strip().lower()
        kind = str(output_kind or "").strip().lower()
        if not cleaned:
            return None
        if kind == "final_answer" and physical != "completed":
            raise ValueError("final_answer requires a completed physical turn")
        if kind not in {"final_answer", "terminal_notice"}:
            raise ValueError("unsupported Codex terminal output kind")
        logical = "completed" if kind == "final_answer" else "failed"
        target_state = "recovery_output" if recovery else "executed"
        accepted = (
            "'submitting','running','uncertain','continuation_pending','executed','recovery_output'"
            if recovery else "'executing','submitting','running','uncertain'"
        )
        with self._lock, self._transaction() as conn:
            row = conn.execute(
                """SELECT lane_session_key, event_json
                   FROM codex_bridge_inputs WHERE input_id=?""", (input_id,),
            ).fetchone()
            if row is None:
                return None
            event = self._event_from_json(str(row[1]))
            logical_key = f"codex-input:{input_id}"
            from gateway.delivery_ledger import (
                compute_semantic_obligation_id, stage_obligation_in_transaction,
            )
            obligation_id = compute_semantic_obligation_id(str(row[0]), logical_key, kind)
            cur = conn.execute(
                f"""UPDATE codex_bridge_inputs
                    SET state=?, final_text=?, delivery_owner='ledger', turn_outcome=?,
                        physical_turn_status=?, logical_input_status=?, output_kind=?,
                        delivery_status='pending', delivery_obligation_id=?,
                        owner_pid=NULL, owner_started_at=NULL, updated_at=?, last_error=NULL
                    WHERE input_id=? AND state IN ({accepted})""",
                (
                    target_state, cleaned, physical, physical, logical, kind,
                    obligation_id, time.time(), input_id,
                ),
            )
            if not cur.rowcount:
                existing = conn.execute(
                    "SELECT delivery_obligation_id FROM codex_bridge_inputs WHERE input_id=?",
                    (input_id,),
                ).fetchone()
                return str(existing[0]) if existing and existing[0] else None
            stage_obligation_in_transaction(
                conn, obligation_id=obligation_id, session_key=str(row[0]),
                platform=event.source.platform.value, chat_id=event.source.chat_id,
                thread_id=event.source.thread_id, content=cleaned,
                adapter_profile=adapter_profile, logical_key=logical_key,
                output_kind=kind, delivery_sequence=1,
            )
        return obligation_id

    def capture_recovery_output(
        self, input_id: str, final_text: str, *, turn_outcome: str = "completed",
    ) -> bool:
        """Resolve an uncertain request from its exact rollout client id."""
        kind = "final_answer" if str(turn_outcome) == "completed" else "terminal_notice"
        return self.stage_terminal_output(
            input_id, final_text, physical_turn_status=turn_outcome,
            output_kind=kind, recovery=True,
        ) is not None

    def mark_completed(self, input_id: str) -> bool:
        """Close a logical input only after its exact outbox row is ACKed."""
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='completed', delivery_owner='none',
                       delivery_status='delivered', finalized_at=?,
                       owner_pid=NULL, owner_started_at=NULL, updated_at=?
                   WHERE input_id=? AND state IN (
                       'executing','executed','recovery_output','continuation_pending'
                   ) AND delivery_obligation_id IS NOT NULL
                     AND EXISTS (
                         SELECT 1 FROM delivery_obligations
                         WHERE obligation_id=delivery_obligation_id AND state='delivered'
                     )""",
                (time.time(), time.time(), input_id),
            )
        return bool(cur.rowcount)

    def mark_delivery_pending(self, input_id: str, error: str = "") -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='recovery_output', delivery_owner='ledger',
                       delivery_status='pending', updated_at=?, last_error=?
                   WHERE input_id=? AND state IN ('executed','recovery_output')""",
                (time.time(), str(error or "delivery pending")[:500], input_id),
            )

    def update_continuation_chain(
        self, input_id: str, turn_ids: tuple[str, ...], *, reason: str = "",
    ) -> bool:
        chain = tuple(dict.fromkeys(
            str(value).strip() for value in turn_ids if str(value).strip()
        ))
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_inputs
                   SET continuation_turn_id=?, continuation_turn_ids_json=?, updated_at=?,
                       last_error=COALESCE(NULLIF(?, ''), last_error)
                   WHERE input_id=? AND state='continuation_pending'""",
                (
                    chain[-1] if chain else None,
                    json.dumps(chain, separators=(",", ":")), time.time(),
                    str(reason or "")[:500], input_id,
                ),
            )
        return bool(cur.rowcount)

    def reopen_for_rollout_reconciliation(self, input_id: str, reason: str = "") -> bool:
        """A prepared handoff became a no-op; recover the old turn's final."""
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='uncertain', delivery_owner='rollout', turn_outcome='unknown',
                       logical_input_status='reconciling', output_kind='none',
                       delivery_status='none', updated_at=?, last_error=?
                   WHERE input_id=? AND state='continuation_pending'""",
                (time.time(), str(reason or "original turn completed during handoff")[:500], input_id),
            )
        return bool(cur.rowcount)

    def cancel_active_input(self, input_id: str, error: str = "") -> bool:
        """Terminally cancel work whose binding was revoked before delivery."""
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='cancelled', delivery_owner='none', turn_outcome='cancelled',
                       physical_turn_status='cancelled', logical_input_status='cancelled',
                       output_kind='none', delivery_status='none',
                       owner_pid=NULL, owner_started_at=NULL, updated_at=?, last_error=?
                   WHERE input_id=? AND state IN (
                       'routed','pending','admitted','executing','submitting','running','uncertain',
                       'continuation_pending'
                   )""",
                (time.time(), str(error or "binding changed")[:500], input_id),
            )
        return bool(cur.rowcount)

    def release_for_retry(self, input_id: str, error: str = "") -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='pending', owner_pid=NULL, owner_started_at=NULL,
                       delivery_owner='runner', turn_outcome='pending',
                       updated_at=?, last_error=?
                   WHERE input_id=? AND state='executing'""",
                (time.time(), str(error or "")[:500] or None, input_id),
            )

    def cancel_input(self, input_id: str, error: str = "") -> None:
        """Terminally reject an input that cannot enter the bounded FIFO."""
        with self._lock, self._transaction() as conn:
            conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='cancelled', owner_pid=NULL, owner_started_at=NULL,
                       delivery_owner='none', turn_outcome='cancelled',
                       updated_at=?, last_error=?
                   WHERE input_id=? AND state IN ('routed','pending','admitted')""",
                (time.time(), str(error or "cancelled")[:500], input_id),
            )

    def cancel_lane(self, lane_session_key: str) -> int:
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='cancelled', delivery_owner='none', turn_outcome='cancelled',
                       owner_pid=NULL, owner_started_at=NULL,
                       updated_at=?, last_error='cancelled by /stop'
                   WHERE lane_session_key=? AND state IN (
                       'routed','pending','admitted','executing','submitting','running','uncertain',
                       'continuation_pending'
                   )""",
                (time.time(), lane_session_key),
            )
        return int(cur.rowcount)

    def has_lane_session(self, lane_session_key: str) -> bool:
        """Whether this store has ever owned execution for a gateway lane."""
        cleaned = str(lane_session_key or "").strip()
        if not cleaned:
            return False
        with self._lock, self._transaction() as conn:
            row = conn.execute(
                "SELECT 1 FROM codex_bridge_inputs WHERE lane_session_key=? LIMIT 1",
                (cleaned,),
            ).fetchone()
        return row is not None

    def recover_after_restart(self) -> None:
        """Make crash-owned rows recoverable before any gateway task can claim them.

        Ownership includes process start time, so PID reuse is safe and an
        overlapping live gateway is not robbed of work. Work that died before
        submission is replayed; a possibly accepted request becomes uncertain
        and is never blindly submitted again; produced output is only delivered.
        """
        now = time.time()
        with self._lock, self._transaction() as conn:
            rows = conn.execute(
                """SELECT input_id, state, owner_pid, owner_started_at, turn_outcome
                   FROM codex_bridge_inputs
                   WHERE state IN ('admitted','executing','submitting','running','executed')"""
            ).fetchall()
            for input_id, state, owner_pid, owner_started_at, turn_outcome in rows:
                if self._owner_alive(owner_pid, owner_started_at):
                    continue
                recovered_state = (
                    "recovery_output" if state == "executed"
                    else "uncertain" if state in {"submitting", "running"}
                    else "pending"
                )
                delivery_owner = {
                    "recovery_output": "ledger",
                    "uncertain": "rollout",
                    "pending": "runner",
                }[recovered_state]
                recovered_outcome = (
                    str(turn_outcome or "unknown")
                    if recovered_state == "recovery_output"
                    else "unknown" if recovered_state == "uncertain"
                    else "pending"
                )
                conn.execute(
                    """UPDATE codex_bridge_inputs
                       SET state=?, delivery_owner=?, turn_outcome=?,
                           owner_pid=NULL, owner_started_at=NULL, updated_at=?
                       WHERE input_id=? AND state=? AND owner_pid IS ? AND owner_started_at IS ?""",
                    (
                        recovered_state, delivery_owner, recovered_outcome, now,
                        input_id, state, owner_pid, owner_started_at,
                    ),
                )

    def recoverable_inputs(self) -> list[DurableCodexInput]:
        """Return unclaimed persisted inputs in FIFO order."""
        with self._lock, self._transaction() as conn:
            rows = conn.execute(
                """SELECT input_id, control_session_key, lane_session_key, thread_id,
                          generation, event_json, state
                   FROM codex_bridge_inputs WHERE state='pending' ORDER BY created_at"""
            ).fetchall()
        return [
            DurableCodexInput(
                input_id=row[0], control_session_key=row[1], lane_session_key=row[2],
                thread_id=row[3], generation=int(row[4]), event=self._event_from_json(row[5]), state=row[6],
            )
            for row in rows
        ]

    def recoverable_outputs(self) -> list[DurableCodexInput]:
        """Return completed turns whose final delivery was not durably acknowledged."""
        with self._lock, self._transaction() as conn:
            rows = conn.execute(
                """SELECT input_id, control_session_key, lane_session_key, thread_id,
                          generation, event_json, state, final_text, codex_turn_id,
                          delivery_owner, turn_outcome, continuation_turn_id,
                          continuation_turn_ids_json, physical_turn_status,
                          logical_input_status, output_kind, delivery_status,
                          delivery_obligation_id, legacy_recovery_required
                   FROM codex_bridge_inputs
                   WHERE state='recovery_output' AND final_text IS NOT NULL
                   ORDER BY created_at"""
            ).fetchall()
        return [self._durable_input_from_full_row(row) for row in rows]

    def continuation_pending_inputs(self) -> list[DurableCodexInput]:
        """Return externally continued inputs awaiting their authoritative rollout final."""
        with self._lock, self._transaction() as conn:
            rows = conn.execute(
                """SELECT input_id, control_session_key, lane_session_key, thread_id,
                          generation, event_json, state, final_text, codex_turn_id,
                          delivery_owner, turn_outcome, continuation_turn_id,
                          continuation_turn_ids_json, physical_turn_status,
                          logical_input_status, output_kind, delivery_status,
                          delivery_obligation_id, legacy_recovery_required
                   FROM codex_bridge_inputs
                   WHERE state='continuation_pending' ORDER BY created_at"""
            ).fetchall()
        return [self._durable_input_from_full_row(row) for row in rows]

    def legacy_completed_handoff_candidates(self) -> list[DurableCodexInput]:
        """Legacy rows where a delivered interruption notice may hide a successor."""
        with self._lock, self._transaction() as conn:
            rows = conn.execute(
                """SELECT input_id, control_session_key, lane_session_key, thread_id,
                          generation, event_json, state, final_text, codex_turn_id,
                          delivery_owner, turn_outcome, continuation_turn_id,
                          continuation_turn_ids_json, physical_turn_status,
                          logical_input_status, output_kind, delivery_status,
                          delivery_obligation_id, legacy_recovery_required
                   FROM codex_bridge_inputs
                   WHERE state='completed' AND turn_outcome IN ('interrupted','failed')
                     AND codex_turn_id IS NOT NULL
                   ORDER BY created_at"""
            ).fetchall()
        return [self._durable_input_from_full_row(row) for row in rows]

    def pending_legacy_rollout_recovery(self) -> list[DurableCodexInput]:
        with self._lock, self._transaction() as conn:
            rows = conn.execute(
                """SELECT input_id, control_session_key, lane_session_key, thread_id,
                          generation, event_json, state, final_text, codex_turn_id,
                          delivery_owner, turn_outcome, continuation_turn_id,
                          continuation_turn_ids_json, physical_turn_status,
                          logical_input_status, output_kind, delivery_status,
                          delivery_obligation_id, legacy_recovery_required
                   FROM codex_bridge_inputs
                   WHERE state='continuation_pending' AND legacy_recovery_required=1
                   ORDER BY created_at"""
            ).fetchall()
        return [self._durable_input_from_full_row(row) for row in rows]

    def reopen_legacy_handoff(
        self, input_id: str, continuation_turn_ids: tuple[str, ...],
    ) -> bool:
        chain = tuple(dict.fromkeys(
            str(value).strip() for value in continuation_turn_ids if str(value).strip()
        ))
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='continuation_pending', delivery_owner='rollout',
                       turn_outcome='continuing', continuation_turn_id=?,
                       continuation_turn_ids_json=?, final_text=NULL,
                       logical_input_status='awaiting_successor', output_kind='none',
                       delivery_status='none', delivery_obligation_id=NULL,
                       finalized_at=NULL, legacy_recovery_required=1, updated_at=?,
                       last_error='legacy interrupted input reopened from formal handoff'
                   WHERE input_id=? AND state='completed'
                     AND turn_outcome IN ('interrupted','failed')""",
                (
                    chain[-1] if chain else None,
                    json.dumps(chain, separators=(",", ":")), time.time(), input_id,
                ),
            )
        return bool(cur.rowcount)

    def finish_legacy_recovery_scan(self, input_id: str) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """UPDATE codex_bridge_inputs SET legacy_recovery_required=0, updated_at=?
                   WHERE input_id=? AND state IN ('continuation_pending','recovery_output')""",
                (time.time(), input_id),
            )

    @classmethod
    def _durable_input_from_full_row(cls, row: tuple[Any, ...]) -> DurableCodexInput:
        try:
            chain = tuple(
                str(value).strip() for value in json.loads(row[12] or "[]")
                if str(value).strip()
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            chain = ()
        return DurableCodexInput(
            input_id=row[0], control_session_key=row[1], lane_session_key=row[2],
            thread_id=row[3], generation=int(row[4]), event=cls._event_from_json(row[5]),
            state=row[6], final_text=row[7], codex_turn_id=row[8],
            delivery_owner=row[9], turn_outcome=row[10], continuation_turn_id=row[11],
            continuation_turn_ids=chain, physical_turn_status=str(row[13] or "unknown"),
            logical_input_status=str(row[14] or "open"), output_kind=str(row[15] or "none"),
            delivery_status=str(row[16] or "none"),
            delivery_obligation_id=str(row[17]) if row[17] is not None else None,
            legacy_recovery_required=bool(row[18]),
        )

    def claim_rollout_event(self, input_id: str, turn_id: str) -> tuple[str, bool]:
        """Resolve one rollout event's current delivery owner atomically.

        Returns ``(disposition, transferred)`` where disposition is one of
        ``unmanaged``, ``live``, ``rollout``, ``terminal``, or ``wait``.
        Provenance inheritance alone never steals a live/ledger-owned event;
        a replacement physical turn can take ownership after the interrupted
        delivery attempt has settled.
        """
        cleaned_input = str(input_id or "").strip()
        cleaned_turn = str(turn_id or "").strip()
        if not cleaned_input or not cleaned_turn:
            return "unmanaged", False
        with self._lock, self._transaction() as conn:
            row = conn.execute(
                """SELECT state, codex_turn_id, delivery_owner, turn_outcome,
                          continuation_turn_id, continuation_turn_ids_json
                   FROM codex_bridge_inputs WHERE input_id=?""",
                (cleaned_input,),
            ).fetchone()
            if row is None:
                return "unmanaged", False
            state = str(row[0] or "")
            original_turn = str(row[1] or "").strip()
            delivery_owner = str(row[2] or "none")
            turn_outcome = str(row[3] or "unknown")
            continuation_turn = str(row[4] or "").strip()
            try:
                continuation_chain = tuple(
                    str(value).strip() for value in json.loads(row[5] or "[]")
                    if str(value).strip()
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continuation_chain = ()
            if state == "cancelled":
                return "terminal", False
            if not original_turn:
                if delivery_owner != "rollout":
                    return ("live" if delivery_owner == "runner" else "terminal"), False
                cur = conn.execute(
                    """UPDATE codex_bridge_inputs SET codex_turn_id=?, updated_at=?
                       WHERE input_id=? AND codex_turn_id IS NULL AND delivery_owner='rollout'""",
                    (cleaned_turn, time.time(), cleaned_input),
                )
                return ("rollout", False) if cur.rowcount else ("wait", False)
            if cleaned_turn == original_turn:
                if delivery_owner == "runner":
                    return "live", False
                if delivery_owner == "rollout" and state == "uncertain":
                    return "rollout", False
                return "terminal", False
            if cleaned_turn == continuation_turn or cleaned_turn in continuation_chain:
                return ("rollout" if delivery_owner == "rollout" else "terminal"), False
            if turn_outcome not in {"interrupted", "unknown", "continuing"}:
                return "terminal", False
            if delivery_owner in {"runner", "ledger"}:
                return "wait", False
            if state not in {"completed", "uncertain", "continuation_pending"}:
                return "terminal", False
            cur = conn.execute(
                """UPDATE codex_bridge_inputs
                   SET state='continuation_pending', delivery_owner='rollout',
                       turn_outcome='continuing', continuation_turn_id=?,
                       continuation_turn_ids_json=?, final_text=NULL,
                       physical_turn_status='interrupted',
                       logical_input_status='awaiting_successor',
                       owner_pid=NULL, owner_started_at=NULL, updated_at=?,
                   last_error='resumed Codex turn claimed by rollout watcher'
                   WHERE input_id=? AND state=? AND delivery_owner=? AND turn_outcome=?""",
                (
                    cleaned_turn,
                    json.dumps(tuple(dict.fromkeys((*continuation_chain, cleaned_turn))), separators=(",", ":")),
                    time.time(), cleaned_input,
                    state, delivery_owner, turn_outcome,
                ),
            )
        return ("rollout", True) if cur.rowcount else ("wait", False)

    def prune(self, *, retention_seconds: float = 7 * 24 * 60 * 60) -> int:
        """Bound terminal bridge history without touching recoverable work."""
        cutoff = time.time() - max(0.0, float(retention_seconds))
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """DELETE FROM codex_bridge_inputs
                   WHERE state IN ('routed','completed','cancelled') AND updated_at < ?""",
                (cutoff,),
            )
            progress_cur = conn.execute(
                "DELETE FROM codex_bridge_progress WHERE state IN ('delivered','finalized') AND updated_at < ?",
                (cutoff,),
            )
        return int(cur.rowcount) + int(progress_cur.rowcount)

    def input_state(self, input_id: str) -> Optional[str]:
        with self._lock, self._transaction() as conn:
            row = conn.execute(
                "SELECT state FROM codex_bridge_inputs WHERE input_id=?", (input_id,)
            ).fetchone()
        return str(row[0]) if row else None

    def input_turn_id(self, input_id: str) -> Optional[str]:
        with self._lock, self._transaction() as conn:
            row = conn.execute(
                "SELECT codex_turn_id FROM codex_bridge_inputs WHERE input_id=?", (input_id,)
            ).fetchone()
        value = str(row[0] or "").strip() if row else ""
        return value or None
