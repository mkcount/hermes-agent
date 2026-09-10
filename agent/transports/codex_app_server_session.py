"""Session adapter for codex app-server runtime.

Owns one Codex thread per Hermes session: drives ``turn/start``, consumes
streaming notifications via CodexEventProjector, bridges server-initiated
approval requests, translates cancellation, and returns a TurnResult that
AIAgent.run_conversation() splices into ``messages``. Synchronous: the client's
reader threads feed queues that this adapter polls, like the chat_completions loop.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.codex_responses_adapter import _format_responses_error
from agent.redact import redact_sensitive_text
from agent.transports.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
    find_codex_control_socket,
)
from agent.transports.codex_event_projector import CodexEventProjector, ProjectionResult

logger = logging.getLogger(__name__)


_STDERR_TAIL_LINES = 12  # stderr tail on generic errors: legible, yet enough for a config/auth diagnostic

# Hermes' tools.terminal.security_mode -> Codex permissions profile id.
# Missing config -> workspace-write (Codex's own default).
_HERMES_TO_CODEX_PERMISSION_PROFILE = {
    "auto": "workspace-write", "approval-required": "read-only-with-approval",
    "unrestricted": "full-access", "yolo": "full-access",  # yolo: backstop alias used by some skills/tests
}


@dataclass
class TurnResult:
    """Result of one user→assistant→tool turn through the codex app-server."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None  # non-recoverable turn error
    turn_id: Optional[str] = None
    thread_id: Optional[str] = None
    # Exact turn/start text distinguishes the input echo from a new user event.
    submitted_user_text: Optional[str] = None
    token_usage_last: Optional[dict[str, Any]] = None
    model_context_window: Optional[int] = None
    compacted: bool = False
    # Codex likely wedged (turn timeout, watchdog, token refresh failure): caller respawns next turn.
    should_retire: bool = False


# Some codex versions stream ``<turn_aborted>`` as raw agentMessage text when an
# interrupt/upstream error tears the turn down without emitting turn/completed.
_TURN_ABORTED_MARKERS = ("<turn_aborted>", "<turn_aborted/>")


def _first_scope_id(*lookups: tuple[Any, str, str]) -> Any:
    """``src.get(a) or src.get(b)`` over successive dict sources until one is not None."""
    for src, primary, fallback in lookups:
        if isinstance(src, dict):
            observed = src.get(primary) or src.get(fallback)
            if observed is not None:
                return observed
    return None


def _notification_scope_ids(note: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract the thread/turn identity carried by a notification (top-level, then turn/item)."""
    params = (note.get("params") or {}) if isinstance(note, dict) else None
    if not isinstance(params, dict):
        return None, None
    turn, item = params.get("turn") or {}, params.get("item") or {}
    return (
        _first_scope_id((params, "threadId", "thread_id"), (turn, "threadId", "thread_id"), (item, "threadId", "thread_id")),
        _first_scope_id((params, "turnId", "turn_id"), (turn, "id", "turnId"), (item, "turnId", "turn_id")),
    )


def _notification_belongs_to_turn(note: dict, *, thread_id: Optional[str], turn_id: Optional[str]) -> bool:
    """Whether a multiplexed notification belongs to this turn.

    One connection can carry parent and hosted subagent threads; an explicitly
    foreign thread/turn event must not mutate this transcript. Unscoped
    notifications remain accepted for protocol compatibility.
    """
    if not isinstance(note, dict):
        return False
    observed = _notification_scope_ids(note)
    return not any(
        expected is not None and seen is not None and str(seen) != str(expected)
        for expected, seen in zip((thread_id, turn_id), observed)
    )


def _coerce_turn_input_text(user_input: Any) -> str:
    """Collapse rich content parts into app-server text (``turn/start`` is text-only; images become a marker)."""
    if isinstance(user_input, str):
        return user_input
    if not isinstance(user_input, list):
        return "" if user_input is None else str(user_input)
    parts: list[str] = []
    for item in user_input:
        if not isinstance(item, dict):
            if item.strip() if isinstance(item, str) else item is not None:
                parts.append(str(item))
        elif item.get("type") in {"text", "input_text"}:
            parts.append(str(item.get("text") or item.get("content") or ""))
        elif item.get("type") in {"image", "image_url", "input_image"}:
            parts.append("[image attached]")
    return "\n\n".join(p for p in parts if p).strip() or "What do you see in this image?"


# Substrings in codex stderr / JSON-RPC errors signalling expired OAuth creds.
# Conservative: only redirect to `codex login` on a strong signal.
_OAUTH_REFRESH_FAILURE_HINTS = (
    "invalid_grant", "invalid grant", "refresh token", "refresh_token", "token refresh", "token_refresh",
    "token has expired", "expired_token", "expired token", "not authenticated", "unauthenticated", "unauthorized",
    "401 unauthorized", "re-authenticate", "reauthenticate", "please log in", "please login", "auth profile",
    "no auth profile", "oauth",
)

_OAUTH_REAUTH_HINT = (
    "Codex authentication failed — your ChatGPT/Codex login looks expired or invalid. Run `codex login` to refresh, "
    "then retry. (Fall back to default runtime with `/codex-runtime auto` if the issue persists.)"
)


def _classify_oauth_failure(*parts: str) -> Optional[str]:
    """Re-auth hint if any part looks like a codex OAuth/token-refresh failure, else None."""
    haystack = " ".join(p for p in parts if p).lower()
    return _OAUTH_REAUTH_HINT if any(needle in haystack for needle in _OAUTH_REFRESH_FAILURE_HINTS) else None


@dataclass
class _ServerRequestRouting:
    """Default approval policies when no interactive approval_callback is wired in (tests, cron)."""

    auto_approve_exec: bool = False
    auto_approve_apply_patch: bool = False


class CodexAppServerSession:
    """One Codex thread per Hermes session, lifetime owned by AIAgent. Not thread-safe: one caller at a time."""

    def __init__(
        self, *, cwd: Optional[str] = None, codex_bin: str = "codex",
        codex_home: Optional[str] = None, permission_profile: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        on_turn_starting: Optional[Callable[[str, str], None]] = None,
        on_turn_started: Optional[Callable[[str, str], None]] = None,
        request_routing: Optional[_ServerRequestRouting] = None,
        client_factory: Optional[Callable[..., CodexAppServerClient]] = None,
        resume_thread_id: Optional[str] = None,
        prefer_desktop_control_socket: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        resume_active_turn_mode: str = "steer",
        approval_policy: Optional[str] = None,
        sandbox_policy: Optional[dict[str, Any]] = None,
        client_user_message_id: Optional[str] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._codex_bin = codex_bin
        self._codex_home = codex_home
        self._permission_profile = permission_profile or _HERMES_TO_CODEX_PERMISSION_PROFILE.get(
            os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto"), "workspace-write"
        )
        self._approval_callback = approval_callback
        self._on_event = on_event  # Display hook (kawaii spinner ticks etc.)
        self._on_turn_starting = on_turn_starting
        self._on_turn_started = on_turn_started
        self._routing = request_routing or _ServerRequestRouting()
        self._client_factory = client_factory or CodexAppServerClient
        self._resume_thread_id = str(resume_thread_id or "").strip() or None
        self._prefer_desktop_control_socket = bool(prefer_desktop_control_socket)
        self._model = str(model or "").strip() or None
        self._reasoning_effort = str(reasoning_effort or "").strip().lower() or None
        self._resume_active_turn_mode = (
            "queue" if str(resume_active_turn_mode or "").strip().lower() == "queue" else "steer"
        )
        self._approval_policy = str(approval_policy or "").strip() or None
        self._sandbox_policy = dict(sandbox_policy) if isinstance(sandbox_policy, dict) else None
        self._client_user_message_id = str(client_user_message_id or "").strip() or None

        self._client: Optional[CodexAppServerClient] = None
        self._using_desktop_control = False
        self._thread_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        self._active_turn_id: Optional[str] = None
        self._resumed_active_turn_id: Optional[str] = None
        self._waiting_for_resumed_turn_id: Optional[str] = None
        self._active_turn_lock = threading.Lock()
        # In-progress fileChange items by id (item/started -> item/completed):
        # approval params don't carry the changeset, so this feeds the prompt summary.
        self._pending_file_changes: dict[str, str] = {}
        self._closed = False

    def set_turn_callbacks(
        self, *, on_turn_starting: Optional[Callable[[str, str], None]],
        on_turn_started: Optional[Callable[[str, str], None]],
        client_user_message_id: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """Refresh request-scoped ownership and policy inputs on a cached session."""
        self._on_turn_starting = on_turn_starting
        self._on_turn_started = on_turn_started
        self._client_user_message_id = str(client_user_message_id or "").strip() or None
        self._model = str(model or "").strip() or None
        self._reasoning_effort = str(reasoning_effort or "").strip().lower() or None

    def _open_client(self, control_socket: Optional[str] = None) -> CodexAppServerClient:
        kwargs: dict[str, Any] = {
            "codex_bin": self._codex_bin,
            "codex_home": self._codex_home,
        }
        if control_socket:
            kwargs["control_socket_path"] = control_socket
        client = self._client_factory(**kwargs)
        try:
            client.initialize(
                client_name="hermes", client_title="Hermes Agent",
                client_version=_get_hermes_version(),
            )
        except BaseException:
            with contextlib.suppress(Exception):
                client.close()
            raise
        return client

    def ensure_started(self) -> str:
        """Spawn, handshake, then start or resume a thread; idempotent."""
        if self._thread_id is not None:
            return self._thread_id
        if (
            self._resume_thread_id
            and self._resume_active_turn_mode == "queue"
            and self._interrupt_event.is_set()
        ):
            raise InterruptedError("Codex resume cancelled before acquiring the thread writer")
        if self._client is None:
            control_socket = (
                find_codex_control_socket(self._codex_home)
                if self._prefer_desktop_control_socket and self._resume_thread_id else None
            )
            try:
                self._client = self._open_client(control_socket)
                self._using_desktop_control = bool(control_socket)
            except Exception:
                if not control_socket:
                    raise
                logger.info(
                    "Codex desktop control unavailable; falling back to an isolated app-server",
                    exc_info=True,
                )
                if self._client is not None:
                    with contextlib.suppress(Exception):
                        self._client.close()
                self._client = self._open_client()
                self._using_desktop_control = False
        if self._resume_thread_id:
            method, params = "thread/resume", {"threadId": self._resume_thread_id}
        else:
            method, params = "thread/start", {"cwd": self._cwd}
            if self._model:
                params["model"] = self._model
        if self._resume_thread_id and self._using_desktop_control:
            # Observe the existing writer before subscribing with
            # thread/resume. This keeps a long desktop turn's notifications
            # and approval requests on its owning client rather than filling
            # this Telegram connection's queues while it waits.
            self._thread_id = self._resume_thread_id
            wait_error = self._wait_for_resumed_turn_boundary(
                "desktop-writer", timeout=None, poll_timeout=0.25,
            )
            self._thread_id = None
            if wait_error:
                if self._interrupt_event.is_set():
                    raise InterruptedError(wait_error)
                raise RuntimeError(wait_error)
        while True:
            try:
                result = self._client.request(method, params, timeout=15)
            except CodexAppServerError as exc:
                message = str(exc.message or "").lower()
                if method == "thread/resume" and "archived" in message:
                    self._client.request("thread/unarchive", {"threadId": self._resume_thread_id}, timeout=15)
                    continue
                if (
                    method == "thread/resume"
                    and self._resume_active_turn_mode == "queue"
                    and exc.code == -32600
                    and "active writer" in message
                ):
                    # Codex enforces one writer per durable thread. A desktop
                    # turn can therefore make thread/resume temporarily fail
                    # before this app-server has enough state to poll it. Keep
                    # the Telegram input durable and retry acquisition; never
                    # fork, interrupt, or silently run it in another thread.
                    if self._interrupt_event.wait(0.5) or self._closed:
                        raise InterruptedError(
                            "Codex resume cancelled while waiting for the desktop writer"
                        ) from exc
                    continue
                raise
            thread_obj = result.get("thread") or {}
            raced_turn_id = (
                self._in_progress_turn_id(thread_obj)
                or ("desktop-writer" if self._thread_is_active(thread_obj) else None)
                if method == "thread/resume"
                and self._using_desktop_control
                and self._resume_active_turn_mode == "queue"
                else None
            )
            if raced_turn_id:
                # Desktop started after the pre-resume idle read. Detach this
                # newly subscribed client immediately, then observe from a
                # fresh unresumed connection so its notifications/approvals
                # remain exclusively with the desktop owner.
                self._client.close()
                control_socket = find_codex_control_socket(self._codex_home)
                if not control_socket:
                    raise RuntimeError("Codex desktop control disappeared while waiting for its active turn")
                self._client = self._open_client(control_socket)
                self._thread_id = self._resume_thread_id
                wait_error = self._wait_for_resumed_turn_boundary(
                    raced_turn_id, timeout=None, poll_timeout=0.25,
                )
                self._thread_id = None
                if wait_error:
                    if self._interrupt_event.is_set():
                        raise InterruptedError(wait_error)
                    raise RuntimeError(wait_error)
                continue
            break
        # Different codex versions serialize the id under thread.id / sessionId / threadId.
        thread_obj = result.get("thread") or {}
        thread_id = thread_obj.get("id") or thread_obj.get("sessionId") or result.get("sessionId") or result.get("threadId")
        if not thread_id:
            raise CodexAppServerError(
                code=-32603, message=f"codex thread/start returned no thread id (payload keys: {sorted(result.keys())})",
            )
        self._thread_id = thread_id
        if self._resume_thread_id:
            self._resumed_active_turn_id = self._in_progress_turn_id(thread_obj)
        logger.info(
            "codex app-server thread %s: id=%s profile=%s cwd=%s",
            "resumed" if self._resume_thread_id else "started", thread_id[:8], self._permission_profile, self._cwd,
        )
        return thread_id

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._active_turn_lock:
            self._active_turn_id = None
            self._waiting_for_resumed_turn_id = None
        if self._client is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - best-effort cleanup
                self._client.close()
        self._client = None
        self._thread_id = None
        self._resumed_active_turn_id = None
        self._using_desktop_control = False

    def request_interrupt(self) -> None:
        """Idempotent: signal the active turn loop to issue turn/interrupt and unwind."""
        self._interrupt_event.set()
        # A queue-mode wait observes a desktop-owned turn but does not own it.
        # /stop cancels only the Telegram input and must never interrupt that
        # foreign desktop turn.
        with self._active_turn_lock:
            turn_id = self._active_turn_id
            waiting_for = self._waiting_for_resumed_turn_id
        if turn_id and turn_id != waiting_for:
            self._issue_interrupt(turn_id)

    def request_steer(self, text: str) -> bool:
        """Append user guidance to the active Codex turn via ``turn/steer``."""
        cleaned = str(text or "").strip()
        if not cleaned:
            return False
        with self._active_turn_lock:
            turn_id, thread_id, client = self._active_turn_id, self._thread_id, self._client
        if not turn_id or not thread_id or client is None:
            return False
        try:
            response = client.request(
                "turn/steer",
                {"threadId": thread_id, "input": [{"type": "text", "text": cleaned}], "expectedTurnId": turn_id}, timeout=10,
            )
        except (CodexAppServerError, TimeoutError):
            logger.debug("turn/steer rejected for active Codex turn", exc_info=True)
            return False
        accepted_turn_id = response.get("turnId") if isinstance(response, dict) else None
        return accepted_turn_id in {None, turn_id}

    def _format_error_with_stderr(self, prefix: str, exc: Any = "", *, tail_lines: int = _STDERR_TAIL_LINES) -> str:
        """User-facing error string plus the force-redacted stderr tail (keeps secrets out of chat output)."""
        exc_str = "" if exc is None else str(exc)
        base = f"{prefix}: {exc_str}" if exc_str else prefix
        try:
            tail = self._client.stderr_tail(tail_lines) if self._client is not None else []
        except Exception:  # pragma: no cover - diagnostic best-effort
            return base
        joined = "\n".join(line.rstrip() for line in tail if line)
        if not joined.strip():
            return base
        return f"{base}\ncodex stderr (last {len(tail)} lines):\n{redact_sensitive_text(joined, force=True)}"

    def _stderr_blob(self, n: int) -> str:
        return "\n".join(self._client.stderr_tail(n))

    @staticmethod
    def _retire(result: TurnResult, error: str) -> None:
        """Record a terminal error and flag the session for respawn on the next turn."""
        result.error = error
        result.should_retire = True

    def _set_classified_error(self, result: TurnResult, prefix: str, classify_text: str, detail: Any) -> None:
        """OAuth failures -> re-auth hint AND retire (token store broken though JSON-RPC is fine); else stderr tail."""
        hint = _classify_oauth_failure(classify_text, self._stderr_blob(40))
        if hint is not None:
            self._retire(result, hint)
        else:
            result.error = self._format_error_with_stderr(prefix, detail)

    def _start_for(self, result: TurnResult) -> bool:
        """ensure_started(); startup failures become a retiring TurnResult.error instead of raw exceptions."""
        try:
            self.ensure_started()
        except InterruptedError:
            result.interrupted = True
            return False
        except (CodexAppServerError, TimeoutError, RuntimeError, OSError) as exc:
            self._retire(result, self._format_error_with_stderr("codex app-server startup failed", exc))
            return False
        assert self._client is not None and self._thread_id is not None
        result.thread_id = self._thread_id
        return True

    def _request_for(self, result: TurnResult, method: str, params: dict, label: str) -> Optional[dict]:
        """Issue ``method``; on failure fill ``result.error`` and return None. A timeout always retires."""
        try:
            return self._client.request(method, params, timeout=10)
        # ``TimeoutError`` is an ``OSError`` subclass on CPython, so it must
        # be classified before transport failures. Timeouts retire the
        # subprocess; an ordinary JSON-RPC error does not.
        except TimeoutError as exc:
            hint = _classify_oauth_failure(self._stderr_blob(40))
            self._retire(result, hint or self._format_error_with_stderr(f"{label} timed out", exc))
        except (CodexAppServerError, RuntimeError, OSError) as exc:
            if not isinstance(exc, CodexAppServerError):
                self._retire(result, self._format_error_with_stderr(f"{label} transport failed", exc))
                return None
            self._set_classified_error(result, f"{label} failed", exc.message, exc)
        return None

    def _subprocess_died(self, result: TurnResult) -> bool:
        """Bail out early (rather than waiting on the deadline) when codex exited."""
        if self._client.is_alive():
            return False
        hint = _classify_oauth_failure(self._stderr_blob(60))
        self._retire(result, hint or self._format_error_with_stderr("codex app-server subprocess exited unexpectedly", tail_lines=20))
        return True

    def _absorb_notification(
        self, result: TurnResult, projector: CodexEventProjector, note: dict
    ) -> tuple[ProjectionResult, bool]:
        """Fan one in-scope notification out to display, accounting, file-change tracking and the projector.

        Returns (projection, aborted); aborted = agent text carried a terminal ``<turn_aborted>`` marker.
        """
        if self._on_event is not None:
            try:
                self._on_event(note)
            except Exception:  # pragma: no cover - display callback
                logger.debug("on_event callback raised", exc_info=True)
        _apply_accounting_notification(result, note)
        self._track_pending_file_change(note)
        projection = projector.project(note)
        if projection.messages:
            result.projected_messages.extend(projection.messages)
        if projection.is_tool_iteration:
            result.tool_iterations += 1
        aborted = False
        if projection.final_text is not None:
            # Multiple agentMessage items per turn: the last one is canonical.
            result.final_text = projection.final_text
            aborted = _has_turn_aborted_marker(projection.final_text)
            if aborted:
                result.interrupted = True
                result.error = result.error or "codex reported turn_aborted"
        return projection, aborted

    def _turn_start_params(self, user_text: str, client_message_id: str) -> dict[str, Any]:
        """Build the stable per-turn app-server policy and idempotency payload."""
        params: dict[str, Any] = {
            "threadId": self._thread_id,
            "input": [{"type": "text", "text": user_text}],
            "clientUserMessageId": client_message_id,
        }
        if self._model:
            params["model"] = self._model
        if self._reasoning_effort:
            params["effort"] = self._reasoning_effort
        if self._approval_policy:
            params["approvalPolicy"] = self._approval_policy
        if self._sandbox_policy:
            params["sandboxPolicy"] = dict(self._sandbox_policy)
        return params

    @staticmethod
    def _in_progress_turn_id(thread_obj: Any) -> Optional[str]:
        if not isinstance(thread_obj, dict):
            return None
        for turn in reversed(thread_obj.get("turns") or []):
            if not isinstance(turn, dict):
                continue
            raw_status = turn.get("status")
            status = raw_status.get("type") if isinstance(raw_status, dict) else raw_status
            if str(status or "").replace("_", "").lower() == "inprogress":
                return str(turn.get("id") or "").strip() or None
        return None

    @staticmethod
    def _thread_is_active(thread_obj: Any) -> bool:
        if not isinstance(thread_obj, dict):
            return False
        raw_status = thread_obj.get("status")
        status = raw_status.get("type") if isinstance(raw_status, dict) else raw_status
        return str(status or "").replace("_", "").lower() in {"active", "inprogress"}

    def _wait_for_resumed_turn_boundary(
        self, turn_id: str, *, timeout: Optional[float], poll_timeout: float,
    ) -> Optional[str]:
        """Wait for a desktop-owned turn without steering or interrupting it."""
        assert self._client is not None and self._thread_id is not None
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._active_turn_lock:
            self._active_turn_id = turn_id
            self._waiting_for_resumed_turn_id = turn_id
        try:
            while deadline is None or time.monotonic() < deadline:
                if self._interrupt_event.is_set():
                    return "queued Telegram turn cancelled while waiting for the desktop turn"
                if not self._client.is_alive():
                    return self._format_error_with_stderr(
                        "codex app-server subprocess exited while waiting for the desktop turn",
                        tail_lines=20,
                    )
                # A resumed app-server reads the durable thread state. Polling
                # thread/read works even though the desktop turn's live events
                # belong to another process and are mirrored from its rollout.
                try:
                    state = self._client.request(
                        # Runtime status is sufficient here. Omitting turn
                        # bodies avoids reloading multi-megabyte histories on
                        # every desktop-writer poll.
                        "thread/read", {"threadId": self._thread_id, "includeTurns": False},
                        timeout=max(2.0, min(10.0, poll_timeout * 4)),
                    )
                except (CodexAppServerError, TimeoutError, RuntimeError, OSError) as exc:
                    return f"could not observe active desktop turn: {exc}"
                thread_state = state.get("thread") or state
                current = self._in_progress_turn_id(thread_state)
                if current is not None:
                    still_active = current == turn_id
                else:
                    still_active = self._thread_is_active(thread_state)
                if not still_active:
                    return None
                time.sleep(max(0.1, min(2.0, poll_timeout)))
            return "timed out waiting for the active desktop turn before starting the Telegram turn"
        finally:
            with self._active_turn_lock:
                if self._active_turn_id == turn_id:
                    self._active_turn_id = None
                if self._waiting_for_resumed_turn_id == turn_id:
                    self._waiting_for_resumed_turn_id = None

    def run_turn(
        self, user_input: Any, *, turn_timeout: float = 600.0,
        notification_poll_timeout: float = 0.25,
    ) -> TurnResult:
        """Send a user message and block until turn/completed, bridging approvals and projecting items.

        ``turn_timeout`` is an inactivity deadline renewed by every in-scope app-server
        notification or server request. Explicit interrupts, transport/process failure,
        and the app-server's terminal ``turn/completed`` event remain authoritative.
        """
        result = TurnResult()
        if self._start_for(result):
            # Do not clear first: a hard stop arriving during ensure_started() must
            # be honored before launching a Codex turn.
            if self._interrupt_event.is_set():
                result.interrupted = True
            else:
                result.submitted_user_text = _coerce_turn_input_text(user_input)
                client_message_id = self._client_user_message_id or str(uuid.uuid4())
                if self._resumed_active_turn_id and self._resume_active_turn_mode == "queue":
                    wait_error = self._wait_for_resumed_turn_boundary(
                        # A foreign desktop turn has no Hermes inactivity
                        # deadline. The durable Telegram input waits until the
                        # boundary or an explicit /stop/shutdown signal.
                        self._resumed_active_turn_id, timeout=None,
                        poll_timeout=notification_poll_timeout,
                    )
                    self._resumed_active_turn_id = None
                    if wait_error:
                        result.error = wait_error
                        result.interrupted = self._interrupt_event.is_set()
                        result.should_retire = not result.interrupted
                        self._interrupt_event.clear()
                        return result
                # This callback is the durable ambiguity fence. It must run
                # immediately before the request write, after every wait that
                # can still cancel without submitting anything to Codex.
                if self._on_turn_starting is not None:
                    self._on_turn_starting(str(self._thread_id), client_message_id)
                ts = self._request_for(
                    result, "turn/start", self._turn_start_params(result.submitted_user_text, client_message_id),
                    "turn/start",
                )
                if ts is not None:
                    self._run_started_turn(result, ts, turn_timeout, notification_poll_timeout)
        self._interrupt_event.clear()
        return result

    def _run_started_turn(
        self, result: TurnResult, ts: dict, turn_timeout: float, notification_poll_timeout: float,
    ) -> None:
        """Drive an accepted ``turn/start`` to completion: approvals and event projection."""
        projector = CodexEventProjector()
        result.turn_id = (ts.get("turn") or {}).get("id")
        with self._active_turn_lock:
            self._active_turn_id = result.turn_id
            self._waiting_for_resumed_turn_id = None
        if self._on_turn_started is not None and result.turn_id is not None:
            with contextlib.suppress(Exception):
                self._on_turn_started(str(self._thread_id), str(result.turn_id))

        def on_server_request(sreq: dict) -> bool:
            # Drain pending notifications first (bounded) so _pending_file_changes is
            # current for the approval decision and display events still reach on_event.
            turn_complete = False
            for _ in range(8):
                pending = self._client.take_notification(timeout=0)
                if pending is None:
                    break
                if not _notification_belongs_to_turn(pending, thread_id=self._thread_id, turn_id=result.turn_id):
                    logger.debug("ignoring foreign codex notification while draining server request: method=%s", pending.get("method"))
                    continue
                _, aborted = self._absorb_notification(result, projector, pending)
                turn_complete = turn_complete or aborted
            self._handle_server_request(sreq)
            return turn_complete

        def on_note(note: dict, method: str) -> bool:
            _, aborted = self._absorb_notification(result, projector, note)
            if method != "turn/completed":
                return aborted
            turn_obj = (note.get("params") or {}).get("turn") or {}
            turn_status = turn_obj.get("status")
            if turn_status == "interrupted":
                result.interrupted = True
                if not self._interrupt_event.is_set():
                    result.error = result.error or "Codex 작업이 외부 요인으로 중단되어 완료되지 않았습니다."
            elif turn_status and turn_status != "completed" and turn_obj.get("error"):
                err_msg = _format_responses_error(turn_obj["error"], str(turn_status))
                self._set_classified_error(result, f"turn ended status={turn_status}", err_msg, err_msg)
            return True

        self._drive_turn(
            result, turn_timeout=turn_timeout, notification_poll_timeout=notification_poll_timeout,
            timeout_label="turn", on_server_request=on_server_request,
            on_note=on_note, accept_final_text_at_deadline=True,
        )
        with self._active_turn_lock:
            self._active_turn_id = None

    def _drive_turn(
        self, result: TurnResult, *, turn_timeout: float, notification_poll_timeout: float,
        timeout_label: str, on_server_request: Callable[[dict], bool],
        on_note: Callable[[dict, str], bool],
        pre_scope_filter: Optional[Callable[[dict, str], bool]] = None,
        accept_final_text_at_deadline: bool = False,
    ) -> None:
        """Shared poll loop for run_turn / compact_thread until turn/completed or deadline.

        Per iteration: interrupt -> subprocess death -> server requests (answered first
        so codex isn't blocked) -> one notification, filtered by ``pre_scope_filter``
        then turn scope, handed to ``on_note``. In-scope requests and notifications
        renew the inactivity deadline. Hooks return True to complete the turn. Deadline
        without completion interrupts and retires the session.
        """
        deadline = time.monotonic() + turn_timeout
        turn_complete = False
        while time.monotonic() < deadline and not turn_complete:
            if self._interrupt_event.is_set():
                self._issue_interrupt(result.turn_id)
                result.interrupted = True
                break
            if self._subprocess_died(result):
                break
            sreq = self._client.take_server_request(timeout=0)
            if sreq is not None:
                turn_complete = on_server_request(sreq)
                deadline = time.monotonic() + turn_timeout
                continue
            note = self._client.take_notification(timeout=notification_poll_timeout)
            if note is None:
                continue
            method = note.get("method", "")
            if pre_scope_filter is not None and not pre_scope_filter(note, method):
                continue
            if not _notification_belongs_to_turn(note, thread_id=self._thread_id, turn_id=result.turn_id):
                logger.debug("ignoring foreign codex notification: method=%s", method)
                continue
            # This is an inactivity deadline, not a wall-clock cap. Healthy
            # multi-hour turns keep their lease while scoped events arrive.
            deadline = time.monotonic() + turn_timeout
            turn_complete = on_note(note, method)

        if accept_final_text_at_deadline and not turn_complete and not result.interrupted and result.final_text and result.error is None:
            logger.warning(
                "codex app-server turn reached deadline after a completed assistant message but before "
                "turn/completed; accepting the assistant text as the terminal response"
            )
            turn_complete = True

        if not turn_complete and not result.interrupted:
            self._issue_interrupt(result.turn_id)
            result.interrupted = True
            if not result.error:
                result.error = self._format_error_with_stderr(f"{timeout_label} timed out after {turn_timeout}s")
            result.should_retire = True

    def compact_thread(
        self, *, turn_timeout: float = 600.0, notification_poll_timeout: float = 0.25
    ) -> TurnResult:
        """Trigger Codex-native history compaction for the current thread.

        ``thread/compact/start`` returns immediately with no turn id; progress streams
        as normal turn/item notifications, so wait for the matching ``turn/completed``.
        """
        result = TurnResult()
        if not self._start_for(result):
            return result
        self._interrupt_event.clear()
        projector = CodexEventProjector()

        if self._request_for(result, "thread/compact/start", {"threadId": self._thread_id}, "thread/compact/start") is None:
            return result

        def pre_scope_filter(note: dict, method: str) -> bool:
            if result.turn_id is not None:
                return True
            observed_thread_id, observed_turn_id = _notification_scope_ids(note)
            if method == "turn/started":
                if observed_thread_id is not None and str(observed_thread_id) != str(self._thread_id):
                    logger.debug("ignoring foreign compact turn/started: thread=%s", observed_thread_id)
                    return False
                if observed_turn_id is None:
                    logger.debug("ignoring compact turn/started without a turn id")
                    return False
                result.turn_id = str(observed_turn_id)
            elif observed_turn_id is not None or method in {"item/completed", "turn/completed"}:
                # Before the new turn/started, terminal/projectable events are stale or unattributable.
                logger.debug("ignoring codex notification before compact turn start: method=%s", method)
                return False
            return True

        def on_note(note: dict, method: str) -> bool:
            _, aborted = self._absorb_notification(result, projector, note)
            if method not in {"turn/started", "turn/completed"}:
                return aborted
            turn_obj = (note.get("params") or {}).get("turn") or {}
            result.turn_id = turn_obj.get("id") or result.turn_id
            if method == "turn/started":
                return aborted
            turn_status = turn_obj.get("status")
            if turn_status == "interrupted":
                result.interrupted = True
                result.error = result.error or "compact turn interrupted"
            elif turn_status and turn_status != "completed":
                err_msg = _format_responses_error(turn_obj.get("error"), str(turn_status))
                self._set_classified_error(result, f"compact turn ended status={turn_status}", err_msg, err_msg)
            return True

        def on_server_request(sreq: dict) -> bool:
            self._handle_server_request(sreq)
            return False

        self._drive_turn(
            result, turn_timeout=turn_timeout, notification_poll_timeout=notification_poll_timeout,
            timeout_label="compact turn", on_server_request=on_server_request, on_note=on_note,
            pre_scope_filter=pre_scope_filter,
        )
        return result

    def _issue_interrupt(self, turn_id: Optional[str]) -> None:
        if self._client is None or self._thread_id is None or turn_id is None:
            return
        try:
            self._client.request("turn/interrupt", {"threadId": self._thread_id, "turnId": turn_id}, timeout=5)
        except (CodexAppServerError, RuntimeError, OSError) as exc:
            # "no active turn to interrupt" is fine — already done.
            logger.debug("turn/interrupt non-fatal: %s", exc)
        except TimeoutError:
            logger.warning("turn/interrupt timed out")

    def _handle_server_request(self, req: dict) -> None:
        """Answer a codex server request (approval / elicitation) via Hermes' approval flow.

        Permission escalations are always declined (the user chose their profile in
        ~/.codex/config.toml); unknown methods get a JSON-RPC error so codex doesn't hang.
        """
        if self._client is None:
            return
        method = req.get("method", "")
        rid = req.get("id")
        params = req.get("params") or {}
        observed_thread, observed_turn = _notification_scope_ids({"params": params})
        with self._active_turn_lock:
            active_turn = self._active_turn_id
        approval_request = method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "item/permissions/requestApproval",
        }
        if (
            (approval_request and (observed_thread is None or observed_turn is None or active_turn is None))
            or (observed_thread is not None and str(observed_thread) != str(self._thread_id))
            or (observed_turn is not None and active_turn is not None and str(observed_turn) != str(active_turn))
        ):
            logger.warning(
                "Rejecting foreign codex server request: method=%s thread=%s turn=%s",
                method, observed_thread, observed_turn,
            )
            self._client.respond_error(rid, code=-32602, message="Request is outside this Hermes turn")
            return
        handler = self._SERVER_REQUEST_HANDLERS.get(method)
        if handler is None:
            logger.warning("Unknown codex server request: %s", method)
            self._client.respond_error(rid, code=-32601, message=f"Unsupported method: {method}")
            return
        self._client.respond(rid, handler(self, params))

    def _respond_elicitation(self, params: dict) -> dict:
        """MCP elicitation: auto-accept our own hermes-tools server (opted in by enabling the runtime;
        exposes nothing codex's shell can't do); decline others so the user opts in via codex's own flow."""
        action = "accept" if (params.get("serverName") or "") == "hermes-tools" else "decline"
        return {"action": action, "content": None, "_meta": None}

    _SERVER_REQUEST_HANDLERS: dict[str, Callable[..., dict]] = {
        "item/commandExecution/requestApproval": lambda self, p: {"decision": self._decide_exec_approval(p)},
        "item/fileChange/requestApproval": lambda self, p: {"decision": self._decide_apply_patch_approval(p)},
        "item/permissions/requestApproval": lambda self, p: {
            "decision": "accept" if self._approval_policy == "never" else "decline"
        },
        "mcpServer/elicitation/request": _respond_elicitation,
    }

    def _run_approval_callback(self, auto_approve: bool, prompt: Callable[[], tuple[str, str]], log_label: str) -> str:
        """Protocol routing only: auto-approve, fail-closed without a callback, else ask via ``prompt()``.

        Approval mode/timeout resolution lives upstream (codex_runtime.py derives the
        auto flags; the callback runs the shared gate). Do not re-read config here.
        """
        if auto_approve:
            return "accept"
        if self._approval_callback is None:
            return "decline"
        command, description = prompt()
        try:
            choice = self._approval_callback(command, description, allow_permanent=False)
            return _approval_choice_to_codex_decision(choice)
        except Exception:
            logger.exception("approval_callback raised on %s", log_label)
            return "decline"

    def _decide_exec_approval(self, params: dict) -> str:
        def prompt() -> tuple[str, str]:
            # ``cwd`` is Optional on codex's side; fall back so the prompt is never empty.
            description = f"Codex requests exec in {params.get('cwd') or self._cwd or '<unknown>'}"
            if params.get("reason"):
                description += f" — {params['reason']}"
            return params.get("command") or "", description

        return self._run_approval_callback(self._routing.auto_approve_exec, prompt, "exec request")

    def _decide_apply_patch_approval(self, params: dict) -> str:
        def prompt() -> tuple[str, str]:
            # Params carry reason + grantRoot only; the changeset comes from _track_pending_file_change.
            reason, grant_root = params.get("reason"), params.get("grantRoot")
            change_summary = self._pending_file_changes.get(params.get("itemId") or "") or None
            parts = [p for p in (reason, change_summary, grant_root and f"grants write to {grant_root}") if p]
            detail = change_summary or reason
            return (
                f"apply_patch: {detail}" if detail else "apply_patch",
                "; ".join(parts) if parts else "Codex requests to apply a patch",
            )

        return self._run_approval_callback(self._routing.auto_approve_apply_patch, prompt, "apply_patch")

    def _track_pending_file_change(self, note: dict) -> None:
        """Track fileChange items (item/started -> item/completed) so the apply_patch prompt can show the changeset."""
        method = note.get("method", "")
        item = (note.get("params") or {}).get("item") or {}
        item_id = item.get("id") or ""
        if item.get("type") != "fileChange" or not item_id:
            return
        if method == "item/completed":
            self._pending_file_changes.pop(item_id, None)
        elif method == "item/started":
            self._pending_file_changes[item_id] = _summarize_file_changes(item.get("changes") or [])


def _summarize_file_changes(raw_changes: list) -> str:
    """One-line ``"<n> add, <m> update: a.py, b.py, +k more"`` summary of a fileChange item's changes."""
    if not raw_changes:
        return "1 change pending"
    changes = [ch for ch in raw_changes if isinstance(ch, dict)]
    kinds: dict[str, int] = {}
    for ch in changes:
        kind = (ch.get("kind") or {}).get("type") or "update"
        kinds[kind] = kinds.get(kind, 0) + 1
    paths: list[str] = [ch["path"] for ch in changes if ch.get("path")]
    counts = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
    preview = ", ".join(paths[:3])
    if len(paths) > 3:
        preview += f", +{len(paths) - 3} more"
    return f"{counts}: {preview}" if preview else counts


def _apply_accounting_notification(result: TurnResult, note: dict) -> None:
    """Capture token usage (thread/tokenUsage/updated, not turn/completed) and compaction
    boundaries (a contextCompaction item on recent builds, deprecated thread/compacted on older)."""
    if not isinstance(note, dict):
        return
    method = note.get("method") or ""
    params = note.get("params") or {}
    if not isinstance(params, dict):
        return
    if method == "thread/tokenUsage/updated":
        token_usage = params.get("tokenUsage") or {}
        if isinstance(token_usage, dict):
            last, window = token_usage.get("last"), token_usage.get("modelContextWindow")
            if isinstance(last, dict):
                result.token_usage_last = dict(last)
            if isinstance(window, int) and window > 0:
                result.model_context_window = window
        return
    item = params.get("item") if method in {"item/started", "item/completed"} else None
    if method == "thread/compacted" or (isinstance(item, dict) and item.get("type") == "contextCompaction"):
        result.compacted = True
        result.thread_id = params.get("threadId") or result.thread_id
        result.turn_id = params.get("turnId") or result.turn_id


# Hermes approval choice -> codex decision (app-server-protocol v2). "deny" and
# "timeout" both decline — codex has no "prompt expired" wire value.
_APPROVAL_CHOICE_TO_DECISION = {"once": "accept", "session": "acceptForSession", "always": "acceptForSession"}


def _approval_choice_to_codex_decision(choice: str) -> str:
    """Map a Hermes approval choice onto codex's approval decision wire value."""
    return _APPROVAL_CHOICE_TO_DECISION.get(choice, "decline")


def _has_turn_aborted_marker(text: str) -> bool:
    """True if ``text`` carries a raw ``<turn_aborted>`` marker (terminal without turn/completed)."""
    return bool(text) and any(marker in text for marker in _TURN_ABORTED_MARKERS)


def _get_hermes_version() -> str:
    """Best-effort Hermes version string for codex's userAgent line."""
    try:
        from importlib.metadata import version

        return version("hermes-agent")
    except Exception:  # pragma: no cover
        return "0.0.0"
