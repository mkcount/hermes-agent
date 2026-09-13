"""Durable delivery-obligation ledger for gateway final responses.

``pending`` is known-unsent, ``attempting`` is ACK-ambiguous after a crash,
``failed`` is a platform rejection, and only ``delivered`` is retention-prunable.
Exhausted obligations become ``manual_review`` and are retained: unresolved
user output must never disappear merely because a timer or row cap elapsed.
Generic gateway callers remain best-effort; correctness-sensitive producers
such as the Codex bridge pre-stage their row transactionally and fail closed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
_DB_LOCK = threading.Lock()

# Redelivery policy knobs (deliberately not config — the ledger is gated by
# ``gateway.delivery_ledger`` and these only matter in the rare recovery path).
MAX_ATTEMPTS = 3
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 500

# Visible prefixes for redeliveries that might duplicate an already-received message (crash mid-send /
# post-rejection retry) — honest at-least-once. Runtime recovery uses a distinct marker: no restart
# occurred, but a network rejection's acknowledgement can still have been lost independently.
RECOVERED_MARKER = "♻️ Recovered reply — the gateway restarted during delivery, so this may be a duplicate:\n\n"
RECONNECTED_MARKER = ("♻️ Recovered reply — the messaging platform reconnected after the original "
                      "delivery failed, so this may be a duplicate:\n\n")
# A reply refused by flood control may have gone out as several requests (the adapter chunks long replies,
# and MarkdownV2 escaping alone can push a reply that fits one message into two), and the platform may have
# accepted the first chunk(s) before refusing the rest; the send result does not say which landed. The raw
# length of the stored text says nothing about that, so every flood redelivery carries a marker, and neither
# of the markers above tells the truth here (no restart, no reconnect): the rate limit gets its own.
FLOOD_MARKER = ("♻️ Recovered reply — the messaging platform's rate limit refused the original, so part of "
                "it may already have arrived above:\n\n")

# Runtime replay is fail-closed: only errors whose send contract proves they are transient reconnect
# failures. Permanent rejects (blocked bot, bad auth, missing chat) must not be retried on reconnect.
_RUNTIME_RETRYABLE_ERRORS = frozenset({"send_path_degraded"})

# A final send the platform refused with flood control is the other transient case: a 429 means the
# refused request was never accepted, and the platform said how long to wait. Adapters fail such sends
# closed as ``flood_control:<seconds>`` on purpose (#91969) so that this ledger owns the wait instead of
# the send coroutine sleeping through it. Before this the row simply sat in ``failed`` until the next
# restart's sweep, which then redelivered it hours late under the "gateway restarted during delivery"
# marker.
FLOOD_ERROR_PREFIX = "flood_control:"
FLOOD_RETRY_DEFAULT_SECONDS = 60.0
FLOOD_RETRY_CAP_SECONDS = 15 * 60.0
FLOOD_RETRY_SLACK_SECONDS = 2.0

# The canonical prefix above is what the adapters produce for a flood they decide not to sleep. It is
# not the only shape that reaches this ledger. PTB raises ``RetryAfter``, whose own text reads
# "Flood control exceeded. Retry in 185 seconds", and that text is what lands in ``last_error``
# whenever a send fails on a path that has not been normalized, or was written by an older build
# before it was. Such a row has to be recognised too: unrecognised, it is treated as an ordinary
# failure, so no redelivery timer is armed for it and a boot sweep claims it immediately instead of
# adopting it until its deadline, spending the one attempt inside the penalty that caused it.
# Matching requires the "flood control" wording as well as the delay, so an unrelated error that
# merely suggests retrying is never mistaken for a flood.
_RAW_FLOOD_RE = re.compile(r"flood control exceeded.*?retry in\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


def _raw_flood_wait(text: str) -> Optional[float]:
    """Seconds asked for by a flood error still carrying the platform's own wording, else ``None``."""
    match = _RAW_FLOOD_RE.search(text or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def is_flood_error(error: Any) -> bool:
    """True for a flood refusal: the adapters' fail-closed ``flood_control:<seconds>`` result, or a
    row still carrying the platform's own flood wording (see ``_RAW_FLOOD_RE``)."""
    text = str(error or "").strip().lower()
    return text.startswith(FLOOD_ERROR_PREFIX) or _raw_flood_wait(text) is not None


def flood_wait_seconds(error: Any, default: float = FLOOD_RETRY_DEFAULT_SECONDS) -> float:
    """The wait the platform asked for, read from ``flood_control:<seconds>``; ``default`` when unreadable."""
    text = str(error or "").strip().lower()
    wait = default
    if text.startswith(FLOOD_ERROR_PREFIX):
        try:
            wait = float(text[len(FLOOD_ERROR_PREFIX):].strip())
        except ValueError:
            wait = default
    else:
        # A row still carrying the platform's own flood wording states its delay just as precisely,
        # and the deadline must use it rather than the generic default.
        raw = _raw_flood_wait(text)
        if raw is not None:
            wait = raw
    return wait if wait > 0 else default


def flood_retry_delay(seconds: Any) -> float:
    """How long a redelivery timer sleeps for a wait of ``seconds``: capped so a huge or bogus value cannot
    park the timer for hours (the row's own deadline, not the timer, decides eligibility when it fires),
    plus a little slack so the timer lands after the deadline rather than on it."""
    try:
        wait = float(seconds)
    except (TypeError, ValueError):
        wait = FLOOD_RETRY_DEFAULT_SECONDS
    return min(max(wait, 0.0), FLOOD_RETRY_CAP_SECONDS) + FLOOD_RETRY_SLACK_SECONDS


def flood_not_before(updated_at: Any, last_error: Any) -> float:
    """Earliest moment a flood-refused row may be resent: the refusal's timestamp (``mark_failed`` sets
    ``updated_at``) plus the platform's wait. Enforced by the sweeps so neither an early timer nor a
    reconnect sweep spends a redelivery attempt inside the penalty window."""
    try:
        stamp = float(updated_at or 0.0)
    except (TypeError, ValueError):
        stamp = 0.0
    return stamp + flood_wait_seconds(last_error)


def _runtime_retryable(last_error: Any) -> bool:
    text = str(last_error or "").strip().lower()
    return text in _RUNTIME_RETRYABLE_ERRORS or is_flood_error(text)


def _db_path(path: Optional[Path | str] = None) -> Path:
    return Path(path) if path is not None else get_hermes_home() / "state.db"


def _connect(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    # Preserve the zero-argument seam used by existing embedders/tests while
    # allowing correctness-sensitive producers to select an exact profile DB.
    path = _db_path() if db_path is None else Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        conn.close()  # a PRAGMA/DDL failure after connect() must not leak the connection
        raise
    return conn


def initialize_delivery_schema(conn: sqlite3.Connection) -> None:
    """Create/migrate the shared outbox schema on an existing transaction."""
    from hermes_state_wal import apply_wal_with_fallback
    apply_wal_with_fallback(conn, db_label="state.db (delivery_ledger)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_obligations (
            obligation_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            content TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT,
            adapter_profile TEXT,
            logical_key TEXT,
            output_kind TEXT NOT NULL DEFAULT 'final',
            delivery_sequence INTEGER NOT NULL DEFAULT 0
        )"""
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(delivery_obligations)")}
    additions = {
        "adapter_profile": "TEXT",
        "logical_key": "TEXT",
        "output_kind": "TEXT NOT NULL DEFAULT 'final'",
        "delivery_sequence": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, declaration in additions.items():
        if name in columns:
            continue
        try:
            conn.execute(f"ALTER TABLE delivery_obligations ADD COLUMN {name} {declaration}")
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
    # One logical input may legitimately emit an interruption notice and,
    # after a bound successor, one final answer. Path duplicates converge on
    # the same kind while those two semantic outputs remain distinct.
    # Multiple gateway processes can perform first-use migration together.
    # Make the obsolete-index removal itself idempotent instead of using a
    # check-then-drop race across SQLite connections.
    conn.execute("DROP INDEX IF EXISTS idx_delivery_obligations_logical")
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_delivery_obligations_semantic
           ON delivery_obligations(session_key, logical_key, output_kind)
           WHERE logical_key IS NOT NULL"""
    )


def _initialize_schema(conn: sqlite3.Connection) -> None:
    initialize_delivery_schema(conn)


@contextmanager
def _transaction(db_path: Optional[Path | str] = None) -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it: ``sqlite3.Connection`` as a
    context manager only commits/rolls back, so ``with _connect()`` alone leaks a connection (and its
    WAL/SHM fds) per call — ``record_obligation`` runs on every final response; exhausts RLIMIT_NOFILE.

    On a long-running gateway that exhausts ``RLIMIT_NOFILE`` (the cron-ledger sibling of this bug was
    #69567 / PR #69594). ``record_obligation`` runs on every outbound final response, so this ledger is the
    highest-frequency leaker.
    """
    conn = _connect(db_path)
    with closing(conn), conn:
        yield conn


def _start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time  # lazy: tests monkeypatch gateway.status
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    return pid, _start_time(pid)


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """True when the recorded owning process still exists (pid + start time)."""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    current_start = _start_time(pid)
    if current_start is None:
        # Start time unreadable: alive iff the pid exists. Route through the cross-platform probe — on Windows
        # ``os.kill(pid, 0)`` is NOT a no-op (bpo-14484: it maps to ``GenerateConsoleCtrlEvent(0, pid)`` and could
        # Ctrl+C the gateway's own console group). ``_pid_exists`` keeps EPERM-means-alive (pid owned by another user).
        try:
            from gateway.status import _pid_exists
        except Exception:
            if os.name == "nt":
                return False  # never fall back to a raw sig-0 probe on Windows
            try:
                os.kill(pid, 0)  # windows-footgun: ok — POSIX-only fallback branch
                return True
            except OSError as exc:  # incl. ProcessLookupError; EPERM means the pid exists
                return isinstance(exc, PermissionError)
        try:
            return bool(_pid_exists(pid))
        except Exception:
            return False
    try:
        return started_at is None or int(current_start) == int(started_at)
    except (TypeError, ValueError):
        return True


def compute_obligation_id(session_key: str, message_ref: str, content: str) -> str:
    """Stable id: same turn + same content re-records idempotently, while distinct threads/topics on one
    chat never collide (session_key carries platform/chat/thread; ``message_ref`` = inbound message id)."""
    return hashlib.sha256(f"{session_key}|{message_ref}|{content}".encode("utf-8", "replace")).hexdigest()[:24]


def compute_semantic_obligation_id(session_key: str, logical_key: str, output_kind: str) -> str:
    """Path-independent identity for one logical output."""
    raw = f"semantic\0{session_key}\0{logical_key}\0{output_kind}"
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:24]


def stage_obligation_in_transaction(
    conn: sqlite3.Connection, *, obligation_id: str, session_key: str, platform: str,
    chat_id: str, thread_id: Optional[str], content: str,
    adapter_profile: Optional[str] = None, logical_key: Optional[str] = None,
    output_kind: str = "final", delivery_sequence: int = 0,
) -> bool:
    """Insert one immutable send intent using the caller's SQLite transaction."""
    now, (pid, started) = time.time(), _owner_stamp()
    cursor = conn.execute(
        """INSERT OR IGNORE INTO delivery_obligations
           (obligation_id, session_key, platform, chat_id, thread_id,
            content, state, attempts, created_at, updated_at,
            owner_pid, owner_started_at, adapter_profile, logical_key,
            output_kind, delivery_sequence)
           VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            obligation_id, session_key, platform, str(chat_id),
            str(thread_id) if thread_id else None, content, now, now, pid, started,
            str(adapter_profile).strip() if adapter_profile else "default",
            str(logical_key) if logical_key else None, str(output_kind or "final"),
            max(0, int(delivery_sequence)),
        ),
    )
    row = conn.execute(
        """SELECT session_key, platform, chat_id, thread_id, content,
                  COALESCE(logical_key, ''), output_kind, delivery_sequence
           FROM delivery_obligations WHERE obligation_id=?""",
        (obligation_id,),
    ).fetchone()
    expected = (
        session_key, platform, str(chat_id), str(thread_id) if thread_id else None,
        content, str(logical_key or ""), str(output_kind or "final"),
        max(0, int(delivery_sequence)),
    )
    if row is None or tuple(row) != expected:
        raise RuntimeError("delivery obligation identity collision")
    return bool(cursor.rowcount)


def record_obligation(*, obligation_id: str, session_key: str, platform: str, chat_id: str,
                      thread_id: Optional[str], content: str, adapter_profile: Optional[str] = None,
                      logical_key: Optional[str] = None, output_kind: str = "final",
                      delivery_sequence: int = 0,
                      db_path: Optional[Path | str] = None) -> None:
    """Record a final response as owed to the platform (state='pending')."""
    with _DB_LOCK, _transaction(db_path) as conn:
        stage_obligation_in_transaction(
            conn, obligation_id=obligation_id, session_key=session_key,
            platform=platform, chat_id=chat_id, thread_id=thread_id, content=content,
            adapter_profile=adapter_profile, logical_key=logical_key,
            output_kind=output_kind, delivery_sequence=delivery_sequence,
        )
    _prune(db_path=db_path)


def ensure_obligation(*, obligation_id: str, session_key: str, platform: str, chat_id: str,
                      thread_id: Optional[str], content: str,
                      adapter_profile: Optional[str] = None,
                      logical_key: Optional[str] = None, output_kind: str = "final",
                      delivery_sequence: int = 0,
                      db_path: Optional[Path | str] = None) -> bool:
    """Create an obligation only when its stable id has no existing state.

    Recovery paths use this instead of ``record_obligation`` so observing an
    already-delivered or concurrently-attempting row can never reset it to
    pending and duplicate a reply. Returns whether this call inserted it.
    """
    with _DB_LOCK, _transaction(db_path) as conn:
        inserted = stage_obligation_in_transaction(
            conn, obligation_id=obligation_id, session_key=session_key,
            platform=platform, chat_id=chat_id, thread_id=thread_id, content=content,
            adapter_profile=adapter_profile, logical_key=logical_key,
            output_kind=output_kind, delivery_sequence=delivery_sequence,
        )
    _prune(db_path=db_path)
    return inserted


def mark_attempting(obligation_id: str, *, db_path: Optional[Path | str] = None) -> bool:
    return _update_state(
        obligation_id, "attempting", from_states=("pending", "failed"), db_path=db_path,
    )


def mark_delivered(obligation_id: str, *, db_path: Optional[Path | str] = None) -> bool:
    return _update_state(
        obligation_id, "delivered", from_states=("pending", "attempting"), db_path=db_path,
    )


def mark_failed(
    obligation_id: str, error: str = "", *, db_path: Optional[Path | str] = None,
) -> bool:
    return _update_state(
        obligation_id, "failed", error=error, from_states=("pending", "attempting", "failed"),
        db_path=db_path,
    )


def release_runtime_claim(obligation_id: str, error: str = "") -> bool:
    """Return an unsent runtime claim to ``failed`` without spending an attempt.

    Runtime recovery claims before clearing ``resume_pending`` so two reconnect paths cannot send the
    same row; if the flag cannot be cleared no send was attempted and the claim must not consume the
    redelivery budget. Fail-closed to the exact current process instance and ``attempting`` state."""
    pid, started = _owner_stamp()
    if started is None:
        return False
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET state='failed', attempts=CASE
                       WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                   updated_at=?, last_error=?
               WHERE obligation_id=? AND state='attempting'
                 AND owner_pid IS ? AND owner_started_at IS ?""",
            (time.time(), error[:500] if error else None, obligation_id, pid, started))
    return bool(cursor.rowcount)


def _update_state(
    obligation_id: str, state: str, error: str = "", *, from_states: tuple[str, ...] = (),
    db_path: Optional[Path | str] = None,
) -> bool:
    with _DB_LOCK, _transaction(db_path) as conn:
        where = " AND state IN ({})".format(",".join("?" for _ in from_states)) if from_states else ""
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET state=?, updated_at=?, last_error=?
               WHERE obligation_id=?""" + where,
            (state, time.time(), error[:500] if error else None, obligation_id, *from_states))
    return bool(cursor.rowcount)


def obligation_state(
    obligation_id: str, *, db_path: Optional[Path | str] = None,
) -> Optional[str]:
    with _DB_LOCK, _transaction(db_path) as conn:
        row = conn.execute(
            "SELECT state FROM delivery_obligations WHERE obligation_id=?", (obligation_id,),
        ).fetchone()
    return str(row[0]) if row else None


def _claimed_row(oid, session_key, platform, chat_id, thread_id, content, attempts, profile, *,
                 needs_marker: bool, runtime: bool = False, flood: bool = False,
                 last_error: Optional[str] = None) -> Dict[str, Any]:
    """Claimed-row dict handed back for redelivery. A marked row names its own cause: ``flood`` (a reply
    the rate limit refused, possibly after accepting part of it) gets FLOOD_MARKER at boot or at runtime, a
    ``runtime`` reconnect replay gets RECONNECTED_MARKER, and a boot-recovered crash keeps the runner's
    restart marker default. ``last_error`` is the row's pre-claim error, carried so a runtime claim that is
    released unsent goes back to ``failed`` with the same error and keeps its retry eligibility."""
    marker = FLOOD_MARKER if flood else (RECONNECTED_MARKER if runtime else None)
    return {"obligation_id": oid, "session_key": session_key, "platform": platform, "chat_id": chat_id,
            "thread_id": thread_id, "content": content, "needs_marker": needs_marker,
            **({"marker": marker} if needs_marker and marker else {}), "profile": profile,
            **({"runtime_recovery": True} if runtime else {}),
            **({"last_error": last_error} if last_error else {}), "attempts": attempts + 1}


def sweep_recoverable(now: Optional[float] = None, *, deliverable_platforms: Optional[set] = None,
                      deliverable_targets: Optional[set] = None) -> List[Dict[str, Any]]:
    """Claim undelivered rows owned by dead processes; return them for redelivery.

    Claiming atomically re-stamps the owner to THIS process and increments ``attempts`` (the UPDATE is
    guarded on the previous owner stamp, so a second gateway racing the same sweep cannot double-claim).
    Rows over the attempts cap become retained ``manual_review`` records. ``deliverable_platforms`` restricts
    claiming to platforms the caller can send on this boot: ``attempts`` is the redelivery budget and
    must only be spent on a real send, else a platform that failed to connect burns one attempt per boot
    and hits the cap having never been sent once.
    ``deliverable_targets`` further scopes multiplexed gateways by exact ``(platform, adapter_profile)``
    so one connected bot cannot spend another disconnected bot's retry budget.

    A flood-refused row still inside its wait is adopted (owner re-stamped, no attempt spent) and
    returned flagged ``adopted`` with its ``not_before``: the caller clears its session's resume flag
    like any other claimed row, since the answer is in the ledger, but must not send it; the flood
    timer does once the wait has passed. A legacy row without ``adapter_profile`` is normalised to
    ``'default'`` on claim or adoption (the caller only accepts such rows when it is not multiplexed),
    because the runtime sweep matches profiles exactly and could otherwise never claim it."""
    now, (pid, started) = now if now is not None else time.time(), _owner_stamp()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, created_at,
                      owner_pid, owner_started_at, adapter_profile, last_error, updated_at
               FROM delivery_obligations
               WHERE state IN ('pending', 'attempting', 'failed')"""
        ).fetchall()
        for (oid, session_key, platform, chat_id, thread_id, content, state, attempts, created_at,
             owner_pid, owner_started_at, adapter_profile, last_error, updated_at) in rows:
            if _owner_alive(owner_pid, owner_started_at):
                continue  # a live gateway still owns this row
            if attempts >= MAX_ATTEMPTS:
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='manual_review', updated_at=?,
                           last_error=COALESCE(last_error, 'delivery attempts exhausted')
                       WHERE obligation_id=?""", (now, oid))
                continue
            if ((deliverable_platforms is not None and platform not in deliverable_platforms)
                    or (deliverable_targets is not None and (platform, adapter_profile) not in deliverable_targets)):
                continue  # no adapter this boot — claiming would spend an attempt on a no-op
            flood_row = state == "failed" and is_flood_error(last_error)
            if flood_row and now < flood_not_before(updated_at, last_error):
                # Still inside the platform's wait: adopt the dead owner's row without spending an attempt
                # (state and error kept) so this process's flood timer can claim it once the wait passes.
                cursor = conn.execute(
                    """UPDATE delivery_obligations
                       SET owner_pid=?, owner_started_at=?,
                           adapter_profile=COALESCE(adapter_profile, 'default')
                       WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
                    (pid, started, oid, owner_pid, owner_pid))
                if cursor.rowcount:
                    claimed.append({
                        "obligation_id": oid, "session_key": session_key, "platform": platform,
                        "chat_id": chat_id, "thread_id": thread_id, "content": content,
                        "profile": adapter_profile or "default", "attempts": attempts,
                        "adopted": True, "not_before": flood_not_before(updated_at, last_error)})
                continue
            # A claimed flood row is resent as a fresh attempt: clear the stale refusal so an interrupted
            # resend is seen as 'attempting' with no error by the next boot and gets the marker.
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET owner_pid=?, owner_started_at=?, attempts=attempts+1, updated_at=?,
                       adapter_profile=COALESCE(adapter_profile, 'default'),
                       state='attempting', last_error=NULL
                   WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
                (pid, started, now, oid, owner_pid, owner_pid))
            if cursor.rowcount:
                # pending = never started, redeliver plainly; anything else (crashed mid-await, other
                # rejection, a flood refusal whose earlier chunks the platform may have accepted) carries
                # the marker.
                claimed.append(_claimed_row(oid, session_key, platform, chat_id, thread_id, content, attempts,
                                            adapter_profile or "default", needs_marker=state != "pending",
                                            flood=flood_row))
    return claimed


def sweep_failed_for_runtime(platform: str, now: Optional[float] = None, *,
                             profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """Claim this process's reconnect-retryable failed rows for one adapter.

    ``profile`` scopes multiplexed gateways to the bot identity that owned the failed send (``None`` =
    primary/default adapter); unowned rows and rows owned by another process are left for the
    startup/dead-owner sweep. Startup recovery ignores rows owned by a live gateway, so a response
    rejected with ``send_path_degraded`` would stay stranded when only the adapter reconnects; this closes
    that gap without weakening ownership: only rows stamped to this exact process instance, only
    allowlisted transient errors, the same attempts bound, every update guarded by the prior owner
    stamp and ``failed`` state. Claimed rows always carry a marker (the failed send's ack is not safe to
    infer): the reconnect one, or the rate-limit one for a flood-refused row."""
    now, (pid, started) = now if now is not None else time.time(), _owner_stamp()
    if started is None:  # PID alone cannot distinguish this process from a stale row left after PID
        return []        # reuse; runtime replay is optional, so fail closed (startup recovery remains).
    expected_profile = "default" if not profile or profile == "default" else str(profile)
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, attempts, created_at, owner_pid,
                      owner_started_at, last_error, adapter_profile, updated_at
               FROM delivery_obligations
               WHERE state='failed' AND platform=?""", (platform,)).fetchall()
        for (oid, session_key, row_platform, chat_id, thread_id, content, attempts, created_at,
             owner_pid, owner_started_at, last_error, adapter_profile, updated_at) in rows:
            # Exact process-start matching prevents PID reuse from stealing work.
            if (adapter_profile != expected_profile or owner_pid != pid or owner_started_at != started
                    or not _runtime_retryable(last_error)):
                continue
            owner_guard = (now, oid, owner_pid, owner_started_at)
            if attempts >= MAX_ATTEMPTS:
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='manual_review', updated_at=?,
                           last_error=COALESCE(last_error, 'delivery attempts exhausted')
                       WHERE obligation_id=? AND state='failed'
                         AND owner_pid IS ? AND owner_started_at IS ?""", owner_guard)
                continue
            if is_flood_error(last_error) and now < flood_not_before(updated_at, last_error):
                continue  # the platform's wait has not passed; the flood timer comes back for it
            # The claim clears the stale error: this is a fresh attempt, and if it is interrupted the next
            # boot must see 'attempting' with no proof of non-delivery, hence the marker.
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET state='attempting', attempts=attempts+1, updated_at=?, last_error=NULL
                   WHERE obligation_id=? AND state='failed'
                     AND owner_pid IS ? AND owner_started_at IS ?""", owner_guard)
            if cursor.rowcount:
                # The failed send's ack may have been lost (reconnect) or its earlier chunks accepted
                # (flood): every runtime redelivery carries a marker. The pre-claim error rides along so a
                # claim released unsent keeps its flood retry eligibility.
                claimed.append(_claimed_row(oid, session_key, row_platform, chat_id, thread_id, content,
                                            attempts, adapter_profile, needs_marker=True, runtime=True,
                                            flood=is_flood_error(last_error), last_error=last_error))
    return claimed


def pending_flood_retries(now: Optional[float] = None) -> List[Dict[str, Any]]:
    """This process's flood-refused rows that still await redelivery, one entry per adapter identity
    with the earliest deadline (``not_before``). The runner arms one redelivery timer per entry, so a
    row adopted at boot, skipped because its wait had not passed, or refused again is never stranded.
    Rows past the attempts cap are retained by the sweeps for manual review."""
    now, (pid, started) = now if now is not None else time.time(), _owner_stamp()
    if started is None:
        return []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT platform, adapter_profile, updated_at, last_error, attempts, created_at
               FROM delivery_obligations
               WHERE state='failed' AND owner_pid IS ? AND owner_started_at IS ?""", (pid, started)).fetchall()
    earliest: Dict[tuple, float] = {}
    for platform, adapter_profile, updated_at, last_error, attempts, created_at in rows:
        if not is_flood_error(last_error) or attempts >= MAX_ATTEMPTS:
            continue
        due = flood_not_before(updated_at, last_error)
        key = (platform, adapter_profile or "default")
        if key not in earliest or due < earliest[key]:
            earliest[key] = due
    return [{"platform": platform, "profile": profile, "not_before": due}
            for (platform, profile), due in sorted(earliest.items())]


def _prune(
    now: Optional[float] = None, *, db_path: Optional[Path | str] = None,
) -> None:
    now = now if now is not None else time.time()
    try:
        with _transaction(db_path) as conn:
            conn.execute(
                """DELETE FROM delivery_obligations
                   WHERE state IN ('delivered', 'abandoned') AND updated_at < ?""", (now - _RETENTION_SECONDS,))
            total = conn.execute("SELECT COUNT(*) FROM delivery_obligations").fetchone()[0]
            if total > _MAX_ROWS:
                conn.execute(
                    """DELETE FROM delivery_obligations WHERE obligation_id IN (
                         SELECT obligation_id FROM delivery_obligations
                         WHERE state IN ('delivered', 'abandoned')
                         ORDER BY updated_at ASC
                         LIMIT ?)""", (total - _MAX_ROWS,))
    except Exception:
        logger.debug("delivery ledger prune failed", exc_info=True)


def ledger_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Read the ``gateway.delivery_ledger`` config gate (default on)."""
    try:
        if config is None:
            from hermes_cli.config import load_config
            config = load_config()
        value = (config.get("gateway") or {}).get("delivery_ledger", True)
        return value.strip().lower() not in {"false", "0", "no", "off"} if isinstance(value, str) else bool(value)
    except Exception:
        return True


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
import json  # noqa: F401,E402

def debug_rows(limit: int = 20) -> str:
    """Human-readable dump for ad-hoc inspection (sqlite3-free path)."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, state, attempts,
                      created_at, updated_at, last_error
               FROM delivery_obligations
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return json.dumps(
        [
            {
                "id": r[0], "session": r[1], "state": r[2], "attempts": r[3],
                "created_at": r[4], "updated_at": r[5], "last_error": r[6],
            }
            for r in rows
        ],
        indent=2,
    )
# ---- END PLUGIN-COMPAT ----
