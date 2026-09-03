"""Session adapter for codex app-server runtime.

Owns one Codex thread per Hermes session. Drives `turn/start`, consumes
streaming notifications via CodexEventProjector, handles server-initiated
approval requests (apply_patch, exec command), translates cancellation,
and returns a clean turn result that AIAgent.run_conversation() can splice
into its `messages` list.

Lifecycle:
    session = CodexAppServerSession(cwd="/home/x/proj")
    session.ensure_started()                              # spawns + handshake + thread/start
    result = session.run_turn(user_input="hello")         # blocks until turn/completed
    # result.final_text          → assistant text returned to caller
    # result.projected_messages  → list of {role, content, ...} for messages list
    # result.tool_iterations     → how many tool-shaped items completed (skill nudge counter)
    # result.interrupted         → True if Ctrl+C / interrupt_requested fired mid-turn
    session.close()                                       # tears down subprocess

Threading model: the adapter is single-threaded from the caller's perspective.
The underlying CodexAppServerClient owns its own reader threads but exposes
blocking-with-timeout queues that this adapter polls in a loop, so the run_turn
call is synchronous and behaves like AIAgent's existing chat_completions loop.
"""

from __future__ import annotations

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
)
from agent.transports.codex_event_projector import CodexEventProjector

logger = logging.getLogger(__name__)


# How many tailing stderr lines from the codex subprocess to attach to a
# user-facing error when we don't have a more specific classification (OAuth,
# wedge watchdog, etc.). Small enough to keep error messages legible, large
# enough to surface a config/provider/auth diagnostic.
_STDERR_TAIL_LINES = 12


# Permission profile mapping mirrors the docstring in PR proposal:
# Hermes' tools.terminal.security_mode → Codex's permissions profile id.
# Defaults if config is missing → workspace-write (matches Codex's own default).
_HERMES_TO_CODEX_PERMISSION_PROFILE = {
    "auto": "workspace-write",
    "approval-required": "read-only-with-approval",
    "unrestricted": "full-access",
    # Backstop alias used by some skills/tests.
    "yolo": "full-access",
}


@dataclass(frozen=True)
class CodexThreadSummary:
    """Small, UI-safe projection of a Codex desktop thread."""

    thread_id: str
    title: str
    cwd: str
    updated_at: int
    status: str
    path: str = ""


@dataclass(frozen=True)
class CodexProjectSummary:
    """One local project inferred from the desktop thread index."""

    cwd: str
    name: str
    updated_at: int


def _control_socket_path(codex_home: Optional[str] = None) -> Optional[str]:
    """Return the live Codex desktop control socket, when available."""
    home = codex_home or os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    path = os.path.join(home, "app-server-control", "app-server-control.sock")
    try:
        import stat

        return path if stat.S_ISSOCK(os.stat(path).st_mode) else None
    except OSError:
        return None


def list_recent_codex_desktop_threads(
    *,
    limit: int = 5,
    codex_bin: str = "codex",
    codex_home: Optional[str] = None,
    client_factory: Optional[Callable[..., CodexAppServerClient]] = None,
) -> list[CodexThreadSummary]:
    """List recently-active desktop threads without loading their turns.

    Prefer the already-running desktop app-server so live status is accurate.
    If the desktop isn't running, a short-lived app-server reads the same
    persisted CODEX_HOME index, which keeps the mobile picker useful offline.
    """
    factory = client_factory or CodexAppServerClient
    socket_path = _control_socket_path(codex_home)
    client: Optional[CodexAppServerClient] = None
    try:
        kwargs: dict[str, Any] = {
            "codex_bin": codex_bin,
            "codex_home": codex_home,
        }
        if socket_path:
            kwargs["control_socket_path"] = socket_path
        try:
            client = factory(**kwargs)
        except Exception:
            if not socket_path:
                raise
            logger.info(
                "Codex desktop control socket unavailable; falling back to "
                "persisted thread index",
                exc_info=True,
            )
            kwargs.pop("control_socket_path", None)
            client = factory(**kwargs)

        client.initialize(
            client_name="hermes-session-picker",
            client_title="Hermes Session Picker",
            client_version=_get_hermes_version(),
        )
        result = client.request(
            "thread/list",
            {
                "limit": max(1, min(int(limit), 100)),
                "sourceKinds": ["vscode"],
                "archived": False,
                "sortKey": "recency_at",
                "sortDirection": "desc",
                "useStateDbOnly": True,
            },
            timeout=15,
        )
        summaries: list[CodexThreadSummary] = []
        for thread in result.get("data") or []:
            if not isinstance(thread, dict) or not thread.get("id"):
                continue
            title = str(thread.get("name") or thread.get("preview") or "Untitled")
            title = " ".join(title.split())
            status_obj = thread.get("status") or {}
            status = (
                str(status_obj.get("type") or "unknown")
                if isinstance(status_obj, dict)
                else str(status_obj)
            )
            summaries.append(
                CodexThreadSummary(
                    thread_id=str(thread["id"]),
                    title=title,
                    cwd=str(thread.get("cwd") or ""),
                    updated_at=int(
                        thread.get("recencyAt")
                        or thread.get("updatedAt")
                        or 0
                    ),
                    status=status,
                    path=str(thread.get("path") or ""),
                )
            )
        return summaries[:limit]
    finally:
        if client is not None:
            client.close()


def list_recent_codex_desktop_projects(
    *,
    limit: int = 10,
    codex_bin: str = "codex",
    codex_home: Optional[str] = None,
    client_factory: Optional[Callable[..., CodexAppServerClient]] = None,
) -> list[CodexProjectSummary]:
    """List distinct, locally reachable projects from the desktop index.

    App-server does not expose a separate project-list method.  The persisted
    thread index is the authoritative cross-client source that carries each
    conversation's cwd, so collapse its newest entries by normalized path.
    """
    threads = list_recent_codex_desktop_threads(
        limit=100,
        codex_bin=codex_bin,
        codex_home=codex_home,
        client_factory=client_factory,
    )
    projects: list[CodexProjectSummary] = []
    seen: set[str] = set()
    for thread in threads:
        cwd = os.path.abspath(os.path.expanduser(str(thread.cwd or "")))
        if not thread.cwd or not os.path.isdir(cwd):
            continue
        identity = os.path.normcase(os.path.normpath(cwd))
        if identity in seen:
            continue
        seen.add(identity)
        projects.append(
            CodexProjectSummary(
                cwd=cwd,
                name=os.path.basename(cwd.rstrip(os.sep)) or cwd,
                updated_at=thread.updated_at,
            )
        )
        if len(projects) >= max(1, int(limit)):
            break
    return projects


def list_codex_model_reasoning_efforts(
    model: str,
    *,
    codex_bin: str = "codex",
    codex_home: Optional[str] = None,
    client_factory: Optional[Callable[..., CodexAppServerClient]] = None,
) -> list[str]:
    """Return effort values advertised for ``model`` by this Codex account."""
    model = str(model or "").strip()
    if not model:
        return []
    factory = client_factory or CodexAppServerClient
    client: Optional[CodexAppServerClient] = None
    try:
        client = factory(codex_bin=codex_bin, codex_home=codex_home)
        client.initialize(
            client_name="hermes-model-picker",
            client_title="Hermes Model Picker",
            client_version=_get_hermes_version(),
        )
        result = client.request("model/list", {}, timeout=15)
        for item in result.get("data") or []:
            if not isinstance(item, dict) or str(item.get("id") or "") != model:
                continue
            efforts: list[str] = []
            for option in item.get("supportedReasoningEfforts") or []:
                value = (
                    option.get("reasoningEffort")
                    if isinstance(option, dict)
                    else option
                )
                value = str(value or "").strip().lower()
                if value and value not in efforts:
                    efforts.append(value)
            return efforts
        return []
    finally:
        if client is not None:
            client.close()


def list_codex_models(
    *,
    codex_bin: str = "codex",
    codex_home: Optional[str] = None,
    client_factory: Optional[Callable[..., CodexAppServerClient]] = None,
) -> list[str]:
    """Return model IDs advertised by the authenticated Codex app-server."""
    factory = client_factory or CodexAppServerClient
    client: Optional[CodexAppServerClient] = None
    try:
        client = factory(codex_bin=codex_bin, codex_home=codex_home)
        client.initialize(
            client_name="hermes-model-picker",
            client_title="Hermes Model Picker",
            client_version=_get_hermes_version(),
        )
        result = client.request("model/list", {}, timeout=15)
        models: list[str] = []
        for item in result.get("data") or []:
            model_id = (
                str(item.get("id") or "").strip()
                if isinstance(item, dict)
                else ""
            )
            if model_id and model_id not in models:
                models.append(model_id)
        return models
    finally:
        if client is not None:
            client.close()


@dataclass
class TurnResult:
    """Result of one user→assistant→tool turn through the codex app-server."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None  # Set if turn ended in a non-recoverable error
    turn_id: Optional[str] = None
    thread_id: Optional[str] = None
    token_usage_last: Optional[dict[str, Any]] = None
    token_usage_total: Optional[dict[str, Any]] = None
    model_context_window: Optional[int] = None
    compacted: bool = False
    # Hint to the caller that the underlying codex subprocess is likely
    # wedged (turn-level timeout fired, post-tool watchdog tripped, or
    # token-refresh failure killed the child). The caller should retire
    # the session so the next turn respawns codex from scratch instead
    # of riding a CPU-spinning or auth-broken process. Mirrors openclaw
    # beta.8's "retire timed-out app-server clients" fix.
    should_retire: bool = False


# Markers we accept as terminal even when codex never emits turn/completed.
# Some codex versions stream `<turn_aborted>` as raw text in agentMessage
# items when an interrupt or upstream error tears the turn down before the
# normal completion path fires. Mirrors openclaw beta.8 fix.
_TURN_ABORTED_MARKERS = ("<turn_aborted>", "<turn_aborted/>")


def _notification_scope_ids(
    note: dict,
) -> tuple[Optional[str], Optional[str]]:
    """Extract the thread/turn identity carried by a notification."""
    if not isinstance(note, dict):
        return None, None
    params = note.get("params") or {}
    if not isinstance(params, dict):
        return None, None

    nested_turn = params.get("turn") or {}
    nested_item = params.get("item") or {}

    observed_thread_id = params.get("threadId") or params.get("thread_id")
    if observed_thread_id is None and isinstance(nested_turn, dict):
        observed_thread_id = (
            nested_turn.get("threadId")
            or nested_turn.get("thread_id")
        )
    if observed_thread_id is None and isinstance(nested_item, dict):
        observed_thread_id = (
            nested_item.get("threadId")
            or nested_item.get("thread_id")
        )

    observed_turn_id = params.get("turnId") or params.get("turn_id")
    if observed_turn_id is None and isinstance(nested_turn, dict):
        observed_turn_id = nested_turn.get("id") or nested_turn.get("turnId")
    if observed_turn_id is None and isinstance(nested_item, dict):
        observed_turn_id = (
            nested_item.get("turnId")
            or nested_item.get("turn_id")
        )

    return observed_thread_id, observed_turn_id


def _notification_belongs_to_turn(
    note: dict,
    *,
    thread_id: Optional[str],
    turn_id: Optional[str],
) -> bool:
    """Return whether a multiplexed notification belongs to this turn.

    Codex app-server can carry parent and hosted subagent threads over one
    JSON-RPC connection.  An explicitly foreign child or
    stale-turn event must not mutate the active parent's transcript or mark
    its turn complete.  Unscoped notifications remain accepted for protocol
    compatibility.
    """
    if not isinstance(note, dict):
        return False

    observed_thread_id, observed_turn_id = _notification_scope_ids(note)

    if (
        thread_id is not None
        and observed_thread_id is not None
        and str(observed_thread_id) != str(thread_id)
    ):
        return False

    if (
        turn_id is not None
        and observed_turn_id is not None
        and str(observed_turn_id) != str(turn_id)
    ):
        return False

    return True


def _coerce_turn_input_text(user_input: Any) -> str:
    """Collapse Hermes/OpenAI rich content into app-server text input.

    The current `turn/start` path sends text items only. TUI image attachment
    can hand us OpenAI-style content parts, so keep the text/path hints and
    replace opaque image payloads with a small marker instead of putting a
    Python list into the `text` field.
    """
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        parts: list[str] = []
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                if item is not None:
                    parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item_type in {"image", "image_url", "input_image"}:
                parts.append("[image attached]")
        text = "\n\n".join(p for p in parts if p).strip()
        return text or "What do you see in this image?"
    return "" if user_input is None else str(user_input)


# Substrings in codex stderr / JSON-RPC error messages that signal the
# subprocess died because its OAuth credentials are no longer valid.
# Kept conservative: we only redirect users to `codex login` when we're
# reasonably sure that's the actual failure, otherwise we surface the
# original error verbatim. Mirrors openclaw beta.8's auth-refresh
# classification.
_OAUTH_REFRESH_FAILURE_HINTS = (
    "invalid_grant",
    "invalid grant",
    "refresh token",
    "refresh_token",
    "token refresh",
    "token_refresh",
    "token has expired",
    "expired_token",
    "expired token",
    "not authenticated",
    "unauthenticated",
    "unauthorized",
    "401 unauthorized",
    "re-authenticate",
    "reauthenticate",
    "please log in",
    "please login",
    "auth profile",
    "no auth profile",
    "oauth",
)


def _classify_oauth_failure(*parts: str) -> Optional[str]:
    """Return a user-friendly re-auth hint if any of the provided strings
    look like a codex OAuth/token-refresh failure; otherwise None.

    Used for both `turn/start` JSON-RPC errors and post-mortem stderr
    inspection when the subprocess exits unexpectedly. Conservative on
    purpose — we only redirect users to `codex login` when the signal
    is strong, so unrelated runtime failures still surface verbatim.
    """
    haystack = " ".join(p for p in parts if p).lower()
    if not haystack:
        return None
    for needle in _OAUTH_REFRESH_FAILURE_HINTS:
        if needle in haystack:
            return (
                "Codex authentication failed — your ChatGPT/Codex login "
                "looks expired or invalid. Run `codex login` to refresh, "
                "then retry. (Fall back to default runtime with "
                "`/codex-runtime auto` if the issue persists.)"
            )
    return None


@dataclass
class _ServerRequestRouting:
    """Default policies for codex-side approval requests when no interactive
    callback is wired in. These are only used by tests + cron / non-interactive
    contexts; the live CLI path passes an approval_callback that defers to
    tools.approval.prompt_dangerous_approval()."""

    auto_approve_exec: bool = False
    auto_approve_apply_patch: bool = False


class CodexAppServerSession:
    """One Codex thread per Hermes session, lifetime owned by AIAgent.

    Not thread-safe — one caller drives it at a time, matching how AIAgent's
    run_conversation() loop is structured today. The codex client itself can
    handle interleaved reads/writes via its own threads, but the adapter's
    state (projector, thread_id, turn counter) is owned by the caller thread.
    """

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        codex_bin: str = "codex",
        codex_home: Optional[str] = None,
        permission_profile: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        on_turn_started: Optional[Callable[[str, str], None]] = None,
        request_routing: Optional[_ServerRequestRouting] = None,
        client_factory: Optional[Callable[..., CodexAppServerClient]] = None,
        resume_thread_id: Optional[str] = None,
        prefer_desktop_control_socket: bool = True,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        resume_active_turn_mode: str = "steer",
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._codex_bin = codex_bin
        self._codex_home = codex_home
        self._permission_profile = (
            permission_profile or _HERMES_TO_CODEX_PERMISSION_PROFILE.get(
                os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto"),
                "workspace-write",
            )
        )
        self._approval_callback = approval_callback
        self._on_event = on_event  # Display hook (kawaii spinner ticks etc.)
        self._on_turn_started = on_turn_started
        self._routing = request_routing or _ServerRequestRouting()
        self._client_factory = client_factory or CodexAppServerClient
        self._resume_thread_id = str(resume_thread_id or "").strip() or None
        self._prefer_desktop_control_socket = prefer_desktop_control_socket
        # App-server configuration overrides must travel on the protocol
        # request. Merely setting AIAgent.model/reasoning_config changes
        # Hermes' bookkeeping but does not change the Codex thread.
        self._model = str(model or "").strip() or None
        self._reasoning_effort = (
            str(reasoning_effort or "").strip().lower() or None
        )
        normalized_resume_mode = str(
            resume_active_turn_mode or "steer"
        ).strip().lower()
        self._resume_active_turn_mode = (
            "queue" if normalized_resume_mode == "queue" else "steer"
        )

        self._client: Optional[CodexAppServerClient] = None
        self._thread_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        self._active_turn_id: Optional[str] = None
        self._resumed_active_turn_id: Optional[str] = None
        self._waiting_for_resumed_turn_id: Optional[str] = None
        self._active_turn_lock = threading.Lock()
        # Pending file-change items, keyed by item id. Populated on
        # item/started for fileChange items; consumed by the approval
        # bridge when codex sends item/fileChange/requestApproval. The
        # approval params don't carry the changeset, so we cache here
        # to surface a real summary in the approval prompt (quirk #4).
        self._pending_file_changes: dict[str, str] = {}
        self._closed = False

    # ---------- lifecycle ----------

    def ensure_started(self) -> str:
        """Spawn the subprocess, do the initialize handshake, and start a
        thread. Returns the codex thread id. Idempotent — repeated calls
        return the same thread id."""
        if self._thread_id is not None:
            return self._thread_id
        if self._client is None:
            client_kwargs: dict[str, Any] = {
                "codex_bin": self._codex_bin,
                "codex_home": self._codex_home,
            }
            if self._prefer_desktop_control_socket:
                socket_path = _control_socket_path(self._codex_home)
                if socket_path:
                    client_kwargs["control_socket_path"] = socket_path
            try:
                self._client = self._client_factory(**client_kwargs)
            except Exception:
                # The desktop may close between routing and the next Telegram
                # message. Retry against a short-lived app-server, which can
                # either resume the persisted thread or start the requested
                # new one.
                if "control_socket_path" not in client_kwargs:
                    raise
                logger.info(
                    "Codex desktop control socket disappeared; falling back "
                    "to a short-lived app-server",
                    exc_info=True,
                )
                client_kwargs.pop("control_socket_path", None)
                self._client = self._client_factory(**client_kwargs)
        self._client.initialize(
            client_name="hermes",
            client_title="Hermes Agent",
            client_version=_get_hermes_version(),
        )
        # Permission selection is intentionally NOT sent on thread/start.
        # Two reasons (live-tested against codex 0.130.0):
        #   1. `thread/start.permissions` is gated behind the experimentalApi
        #      capability on this codex version — we'd have to opt in during
        #      initialize and accept the unstable surface.
        #   2. Even with experimentalApi declared and the correct shape
        #      (`{"type": "profile", "id": "..."}`, not `{"profileId": ...}`),
        #      codex requires a matching `[permissions]` table in
        #      ~/.codex/config.toml or it fails the request with
        #      'default_permissions requires a [permissions] table'.
        # Letting codex pick its default (`:read-only` unless the user has
        # configured otherwise in their codex config.toml) is the standard
        # codex CLI workflow and avoids fighting codex's own validation.
        # Users who want a write-capable profile configure it in their
        # ~/.codex/config.toml the same way they would for any codex usage.
        if self._resume_thread_id:
            method = "thread/resume"
            params: dict[str, Any] = {"threadId": self._resume_thread_id}
        else:
            method = "thread/start"
            params = {"cwd": self._cwd}
            if self._model:
                params["model"] = self._model
        try:
            result = self._client.request(method, params, timeout=15)
        except CodexAppServerError as exc:
            # Archived threads are still durable and can be resumed after the
            # documented thread/unarchive transition. Do this in place instead
            # of silently starting a replacement thread.
            if (
                method != "thread/resume"
                or "archived" not in str(exc.message or "").lower()
            ):
                raise
            self._client.request(
                "thread/unarchive",
                {"threadId": self._resume_thread_id},
                timeout=15,
            )
            result = self._client.request(method, params, timeout=15)
        # Cross-fill thread.id/sessionId — different codex versions have
        # serialized this under either key. Mirrors openclaw beta.8's
        # tolerance fix so future codex drops/renames don't KeyError us
        # at handshake time.
        thread_obj = result.get("thread") or {}
        thread_id = (
            thread_obj.get("id")
            or thread_obj.get("sessionId")
            or result.get("sessionId")
            or result.get("threadId")
        )
        if not thread_id:
            raise CodexAppServerError(
                code=-32603,
                message=(
                    f"codex {method} returned no thread id "
                    f"(payload keys: {sorted(result.keys())})"
                ),
            )
        self._thread_id = thread_id
        if self._resume_thread_id:
            for turn in reversed(thread_obj.get("turns") or []):
                if isinstance(turn, dict) and turn.get("status") == "inProgress":
                    self._resumed_active_turn_id = str(turn.get("id") or "") or None
                    break
        logger.info(
            "codex app-server thread %s: id=%s profile=%s cwd=%s",
            "resumed" if self._resume_thread_id else "started",
            self._thread_id[:8],
            self._permission_profile,
            self._cwd,
        )
        return self._thread_id

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._active_turn_lock:
            self._active_turn_id = None
            self._waiting_for_resumed_turn_id = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            self._client = None
        self._thread_id = None
        self._resumed_active_turn_id = None

    def __enter__(self) -> "CodexAppServerSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def request_interrupt(self) -> None:
        """Idempotent: signal the active turn loop to issue turn/interrupt
        and unwind. Called by AIAgent's _interrupt_requested path."""
        self._interrupt_event.set()
        # /stop is invoked from the gateway thread while run_turn is blocked on
        # its notification queue. Send the protocol interrupt immediately so
        # releasing the gateway's per-session lock cannot leave the underlying
        # Codex turn running for another poll cycle (or through a reconnect).
        with self._active_turn_lock:
            turn_id = self._active_turn_id
        if turn_id is not None:
            self._issue_interrupt(turn_id)

    def request_steer(self, text: str) -> bool:
        """Append user guidance to the active Codex turn via ``turn/steer``."""
        cleaned = str(text or "").strip()
        if not cleaned:
            return False
        with self._active_turn_lock:
            turn_id = self._active_turn_id
            thread_id = self._thread_id
            client = self._client
        if not turn_id or not thread_id or client is None:
            return False
        try:
            response = client.request(
                "turn/steer",
                {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": cleaned}],
                    "expectedTurnId": turn_id,
                },
                timeout=10,
            )
        except (CodexAppServerError, TimeoutError):
            logger.debug("turn/steer rejected for active Codex turn", exc_info=True)
            return False
        accepted_turn_id = response.get("turnId") if isinstance(response, dict) else None
        return accepted_turn_id in {None, turn_id}

    def is_directly_streaming_turn(self, turn_id: str) -> bool:
        """Return whether this session owns live delivery for ``turn_id``.

        A queue-mode resume temporarily tracks the pre-existing Codex turn as
        active so ``/stop`` can interrupt it, but intentionally does not
        forward its events: the rollout mirror owns that delivery path.
        """
        cleaned = str(turn_id or "").strip()
        if not cleaned:
            return False
        with self._active_turn_lock:
            return (
                self._active_turn_id == cleaned
                and self._waiting_for_resumed_turn_id != cleaned
            )

    # ---------- diagnostics ----------

    def _format_error_with_stderr(
        self,
        prefix: str,
        exc: Any = "",
        *,
        tail_lines: int = _STDERR_TAIL_LINES,
    ) -> str:
        """Build a user-facing error string for codex failures.

        Appends the last few lines of codex's stderr buffer when available,
        passed through agent.redact with force=True so secrets in provider
        error responses (auth headers, query-string tokens, sk-* keys) never
        leak into chat output or trajectories. The codex CLI's own error
        text ('Internal error', 'turn/start failed: ...') is otherwise
        opaque and forces users to re-run with verbose flags to diagnose
        config / provider / auth-bridge problems.

        Use this for the generic / catch-all branches. Specific
        classifications (OAuth via _classify_oauth_failure, post-tool wedge
        watchdog) already produce a clean hint and should be used instead.
        """
        exc_str = str(exc) if exc != "" and exc is not None else ""
        base = f"{prefix}: {exc_str}" if exc_str else prefix
        if self._client is None:
            return base
        try:
            tail = self._client.stderr_tail(tail_lines)
        except Exception:  # pragma: no cover - diagnostic best-effort
            return base
        if not tail:
            return base
        joined = "\n".join(line.rstrip() for line in tail if line)
        if not joined.strip():
            return base
        redacted = redact_sensitive_text(joined, force=True)
        return f"{base}\ncodex stderr (last {len(tail)} lines):\n{redacted}"

    # ---------- per-turn ----------

    def _turn_start_params(
        self,
        input_items: list[dict],
        *,
        client_message_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build one turn/start payload with durable Codex overrides.

        Codex documents model and effort on ``turn/start`` as applying to the
        current and subsequent turns. Sending both here makes Telegram's
        session-scoped /model and /reasoning selections authoritative even
        when the thread was originally created by the desktop app.
        """
        assert self._thread_id is not None
        params: dict[str, Any] = {
            "threadId": self._thread_id,
            "input": input_items,
        }
        if client_message_id:
            # Reuse this id if a broken control-socket proxy makes the first
            # turn/start response ambiguous. Codex can then deduplicate the
            # retry instead of creating the user's Telegram instruction twice.
            params["clientUserMessageId"] = client_message_id
        if self._model:
            params["model"] = self._model
        if self._reasoning_effort:
            params["effort"] = self._reasoning_effort
        return params

    @staticmethod
    def _is_retryable_transport_failure(exc: BaseException) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        if not isinstance(exc, CodexAppServerError):
            return False
        message = str(exc.message or "").lower()
        return any(
            marker in message
            for marker in (
                "broken pipe",
                "connection",
                "control socket",
                "relay",
                "socket closed",
                "unexpected eof",
            )
        )

    def _reconnect_to_thread(self, thread_id: str) -> None:
        """Replace a stale proxy client and resume the durable thread."""
        stale_client = self._client
        self._client = None
        self._thread_id = None
        self._resumed_active_turn_id = None
        self._resume_thread_id = thread_id
        if stale_client is not None:
            try:
                stale_client.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        self.ensure_started()

    def _start_turn_with_reconnect(
        self,
        input_items: list[dict],
        *,
        client_message_id: str,
    ) -> dict[str, Any]:
        """Start once, reconnecting to a replaced desktop socket if needed."""
        assert self._client is not None and self._thread_id is not None
        params = self._turn_start_params(
            input_items,
            client_message_id=client_message_id,
        )
        try:
            return self._client.request("turn/start", params, timeout=30)
        except (CodexAppServerError, TimeoutError) as exc:
            if not self._is_retryable_transport_failure(exc):
                raise
            thread_id = self._thread_id
            logger.warning(
                "turn/start lost its Codex control-socket connection; "
                "reconnecting to thread %s and retrying once",
                thread_id,
            )
            self._reconnect_to_thread(thread_id)
            assert self._client is not None
            return self._client.request(
                "turn/start",
                self._turn_start_params(
                    input_items,
                    client_message_id=client_message_id,
                ),
                timeout=30,
            )

    def _wait_for_resumed_turn_boundary(
        self,
        turn_id: str,
        *,
        timeout: float,
        notification_poll_timeout: float,
    ) -> Optional[str]:
        """Wait for a desktop-owned active turn without steering into it.

        The rollout mirror remains the sole delivery owner for this pre-existing
        desktop turn. In particular, events consumed here must not be forwarded
        through ``_on_event`` and the turn must not be reported through
        ``_on_turn_started``: either action would incorrectly mark the desktop
        turn as Hermes-originated and suppress its Telegram mirror.

        Returns an error string on failure, otherwise ``None`` after the turn
        reaches its boundary.
        """
        assert self._client is not None and self._thread_id is not None
        deadline = time.monotonic() + max(0.0, timeout)
        with self._active_turn_lock:
            self._active_turn_id = turn_id
            self._waiting_for_resumed_turn_id = turn_id

        def _release_wait_ownership() -> None:
            with self._active_turn_lock:
                if self._active_turn_id == turn_id:
                    self._active_turn_id = None
                if self._waiting_for_resumed_turn_id == turn_id:
                    self._waiting_for_resumed_turn_id = None

        while time.monotonic() < deadline:
            if self._interrupt_event.is_set():
                self._issue_interrupt(turn_id)
                _release_wait_ownership()
                return "queued turn cancelled while waiting for desktop turn"

            if not self._client.is_alive():
                _release_wait_ownership()
                return self._format_error_with_stderr(
                    "codex app-server subprocess exited while waiting for "
                    "the active desktop turn",
                    tail_lines=20,
                )

            note = self._client.take_notification(
                timeout=notification_poll_timeout
            )
            if note is None:
                continue
            if not _notification_belongs_to_turn(
                note,
                thread_id=self._thread_id,
                turn_id=turn_id,
            ):
                logger.debug(
                    "ignoring foreign codex notification while waiting for "
                    "desktop turn boundary: method=%s",
                    note.get("method"),
                )
                continue
            if note.get("method") == "turn/completed":
                _release_wait_ownership()
                return None

        _release_wait_ownership()
        return (
            "timed out waiting for the active desktop turn to finish before "
            "starting the queued Telegram turn"
        )

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: float = 600.0,
        notification_poll_timeout: float = 0.25,
        post_tool_quiet_timeout: float = 300.0,
    ) -> TurnResult:
        """Send a user message and block until turn/completed, while
        forwarding server-initiated approval requests and projecting items
        into Hermes' messages shape.

        turn_timeout: maximum seconds without a notification or approval
        request belonging to this turn. Active long-running turns extend this
        deadline whenever Codex emits liveness.

        post_tool_quiet_timeout: if codex emits a tool completion and then
        goes quiet for this many seconds without emitting another item or
        `turn/completed`, fast-fail and mark the session for retirement.
        Mirrors openclaw beta.8's post-tool completion watchdog (#81697)
        so a wedged codex doesn't burn the full turn deadline.
        """
        # Pre-create the result so startup failures (codex subprocess can't
        # spawn, initialize handshake rejects, thread/start blows up) surface
        # the same way per-turn failures do — with a TurnResult.error string
        # the caller can render — instead of bubbling raw codex exceptions
        # up to AIAgent.run_conversation.
        result = TurnResult()
        try:
            self.ensure_started()
        except (CodexAppServerError, TimeoutError) as exc:
            result.error = self._format_error_with_stderr(
                "codex app-server startup failed", exc
            )
            # Subprocess almost certainly unhealthy — retire so the next
            # turn re-spawns cleanly.
            result.should_retire = True
            self._interrupt_event.clear()
            return result
        assert self._client is not None and self._thread_id is not None
        result.thread_id = self._thread_id

        # Do not clear here: a hard stop can arrive while ensure_started() is
        # spawning/initializing the subprocess. Honor it before launching a
        # Codex turn instead of erasing the signal.
        if self._interrupt_event.is_set():
            result.interrupted = True
            self._interrupt_event.clear()
            return result
        projector = CodexEventProjector()

        user_input_text = _coerce_turn_input_text(user_input)

        # Send turn/start with the user input. In queue mode, a desktop-owned
        # active turn remains untouched and keeps reporting through the rollout
        # mirror; the Telegram input starts as a separate turn only after that
        # boundary. In steer mode, retain the protocol-native append behavior.
        input_items = [{"type": "text", "text": user_input_text}]
        client_message_id = str(uuid.uuid4())
        try:
            if (
                self._resumed_active_turn_id
                and self._resume_active_turn_mode == "queue"
            ):
                resumed_turn_id = self._resumed_active_turn_id
                logger.info(
                    "queueing Telegram input behind active desktop Codex turn %s",
                    resumed_turn_id,
                )
                wait_error = self._wait_for_resumed_turn_boundary(
                    resumed_turn_id,
                    timeout=turn_timeout,
                    notification_poll_timeout=notification_poll_timeout,
                )
                if wait_error is not None:
                    result.error = wait_error
                    result.interrupted = self._interrupt_event.is_set()
                    result.should_retire = not result.interrupted
                    self._resumed_active_turn_id = None
                    self._interrupt_event.clear()
                    return result
                self._resumed_active_turn_id = None
                ts = self._start_turn_with_reconnect(
                    input_items,
                    client_message_id=client_message_id,
                )
            elif self._resumed_active_turn_id:
                result.turn_id = self._resumed_active_turn_id
                try:
                    self._client.request(
                        "turn/steer",
                        {
                            "threadId": self._thread_id,
                            "expectedTurnId": result.turn_id,
                            "input": input_items,
                        },
                        timeout=10,
                    )
                    ts = {"turn": {"id": result.turn_id}}
                except CodexAppServerError:
                    # The desktop turn can finish in the small gap between
                    # thread/resume and this request. Start a normal new turn
                    # instead of making the user resend their message.
                    logger.info(
                        "active desktop turn ended before steer; starting a "
                        "new turn on the resumed thread"
                    )
                    self._resumed_active_turn_id = None
                    ts = self._start_turn_with_reconnect(
                        input_items,
                        client_message_id=client_message_id,
                    )
            else:
                ts = self._start_turn_with_reconnect(
                    input_items,
                    client_message_id=client_message_id,
                )
        except CodexAppServerError as exc:
            # Classify auth/refresh failures so the user gets a clear
            # `codex login` pointer instead of a raw RPC error string.
            stderr_blob = "\n".join(self._client.stderr_tail(40))
            hint = _classify_oauth_failure(exc.message, stderr_blob)
            if hint is not None:
                result.error = hint
                # Subprocess is fine on a JSON-RPC level here, but the
                # token store is broken — retire so the next turn does a
                # clean handshake (and the user has a chance to re-auth
                # via `codex login` between turns).
                result.should_retire = True
            else:
                result.error = self._format_error_with_stderr(
                    "turn/start failed", exc
                )
            self._interrupt_event.clear()
            return result
        except TimeoutError as exc:
            # turn/start hanging is a strong signal the subprocess is wedged.
            stderr_blob = "\n".join(self._client.stderr_tail(40))
            hint = _classify_oauth_failure(stderr_blob)
            result.error = hint or self._format_error_with_stderr(
                "turn/start timed out", exc
            )
            result.should_retire = True
            self._interrupt_event.clear()
            return result

        result.turn_id = (ts.get("turn") or {}).get("id")
        notified_turn_id: Optional[str] = None

        def _notify_turn_started() -> None:
            nonlocal notified_turn_id
            if (
                self._on_turn_started is None
                or self._thread_id is None
                or result.turn_id is None
                or notified_turn_id == str(result.turn_id)
            ):
                return
            try:
                self._on_turn_started(
                    str(self._thread_id),
                    str(result.turn_id),
                )
                notified_turn_id = str(result.turn_id)
            except Exception:
                logger.debug("on_turn_started callback raised", exc_info=True)

        with self._active_turn_lock:
            self._active_turn_id = result.turn_id
            self._waiting_for_resumed_turn_id = None
        _notify_turn_started()
        # ``turn_timeout`` is an inactivity deadline, not a wall-clock cap.
        # Long Codex turns may legitimately run for hours while tool, reasoning,
        # and usage events continue to arrive. A fixed 600-second deadline used
        # to abort healthy work at exactly ten minutes even when the last event
        # was only seconds old.
        activity_deadline = time.monotonic() + turn_timeout
        turn_complete = False
        # Post-tool watchdog state. last_tool_completion_at is set whenever
        # a tool-shaped item completes; if no further notification arrives
        # within post_tool_quiet_timeout and the turn hasn't completed, we
        # fast-fail and retire the session.
        last_tool_completion_at: Optional[float] = None

        while time.monotonic() < activity_deadline and not turn_complete:
            if self._interrupt_event.is_set():
                self._issue_interrupt(result.turn_id)
                result.interrupted = True
                break

            # Detect a dead subprocess between iterations. If codex exited
            # (e.g. crashed, segfaulted, or its auth refresh thread killed
            # the process), we won't get any more notifications — bail out
            # rather than waiting for the full turn deadline.
            if not self._client.is_alive():
                stderr_blob = "\n".join(self._client.stderr_tail(60))
                hint = _classify_oauth_failure(stderr_blob)
                if hint is not None:
                    result.error = hint
                else:
                    result.error = self._format_error_with_stderr(
                        "codex app-server subprocess exited unexpectedly",
                        tail_lines=20,
                    )
                result.should_retire = True
                break

            # Post-tool watchdog: if a tool completion was the most recent
            # signal and codex has been silent past the quiet timeout, give
            # up on this turn instead of waiting for the outer deadline.
            if (
                last_tool_completion_at is not None
                and (time.monotonic() - last_tool_completion_at)
                    > post_tool_quiet_timeout
            ):
                self._issue_interrupt(result.turn_id)
                result.interrupted = True
                result.error = (
                    f"codex went silent for "
                    f"{post_tool_quiet_timeout:.0f}s after a tool result; "
                    f"retiring app-server session."
                )
                result.should_retire = True
                break

            # Drain any server-initiated requests (approvals) before
            # reading notifications, so the codex side isn't blocked.
            sreq = self._client.take_server_request(timeout=0)
            if sreq is not None:
                # Drain any pending notifications first so per-turn state
                # (e.g. _pending_file_changes for fileChange approvals) is
                # up to date when we make the approval decision. Bounded
                # to avoid starving the server-request response.
                for _ in range(8):
                    pending = self._client.take_notification(timeout=0)
                    if pending is None:
                        break
                    if not _notification_belongs_to_turn(
                        pending,
                        thread_id=self._thread_id,
                        turn_id=result.turn_id,
                    ):
                        logger.debug(
                            "ignoring foreign codex notification while draining "
                            "server request: method=%s",
                            pending.get("method"),
                        )
                        continue
                    activity_deadline = time.monotonic() + turn_timeout
                    if last_tool_completion_at is not None:
                        # Token-usage, reasoning, and other non-projecting
                        # notifications are still proof that Codex is alive.
                        last_tool_completion_at = time.monotonic()
                    # Mirror the main notification-handling block below so
                    # display events surface and stay in step with projector
                    # state. Without this, item/started / item/completed
                    # events drained as part of the approval-roundtrip
                    # preamble are projected into messages but never reach
                    # the tool-progress display, silently hiding tool
                    # bubbles around approvals.
                    if self._on_event is not None:
                        try:
                            self._on_event(pending)
                        except Exception:  # pragma: no cover - display callback
                            logger.debug(
                                "on_event callback raised", exc_info=True
                            )
                    _apply_token_usage_notification(result, pending)
                    _apply_compaction_notification(result, pending)
                    self._track_pending_file_change(pending)
                    proj = projector.project(pending)
                    if proj.messages:
                        result.projected_messages.extend(proj.messages)
                    if proj.is_tool_iteration:
                        result.tool_iterations += 1
                        last_tool_completion_at = time.monotonic()
                    if proj.final_text is not None:
                        result.final_text = proj.final_text
                        if _has_turn_aborted_marker(proj.final_text):
                            turn_complete = True
                            result.interrupted = True
                            result.error = (
                                result.error
                                or "codex reported turn_aborted"
                            )
                self._handle_server_request(sreq)
                # Activity counts as live signal — reset the post-tool
                # quiet timer and the overall inactivity deadline so an
                # approval round-trip doesn't trip either watchdog.
                last_tool_completion_at = None
                activity_deadline = time.monotonic() + turn_timeout
                continue

            note = self._client.take_notification(
                timeout=notification_poll_timeout
            )
            if note is None:
                continue

            method = note.get("method", "")
            if result.turn_id is None and method == "turn/started":
                observed_thread_id, observed_turn_id = (
                    _notification_scope_ids(note)
                )
                if (
                    observed_thread_id is None
                    or str(observed_thread_id) == str(self._thread_id)
                ):
                    result.turn_id = observed_turn_id
                    _notify_turn_started()
            if not _notification_belongs_to_turn(
                note,
                thread_id=self._thread_id,
                turn_id=result.turn_id,
            ):
                logger.debug(
                    "ignoring foreign codex notification: method=%s", method
                )
                continue

            # Any notification scoped to this turn proves Codex is still
            # making progress, including reasoning/token-usage events that do
            # not project into a visible Telegram message.
            activity_deadline = time.monotonic() + turn_timeout

            if last_tool_completion_at is not None:
                # The previous implementation only noticed projected messages.
                # Long reasoning phases often emit usage/status events instead,
                # so they were incorrectly killed after 90 seconds.
                last_tool_completion_at = time.monotonic()

            if self._on_event is not None:
                try:
                    self._on_event(note)
                except Exception:  # pragma: no cover - display callback
                    logger.debug("on_event callback raised", exc_info=True)

            _apply_token_usage_notification(result, note)
            _apply_compaction_notification(result, note)

            # Track in-progress fileChange items so the approval bridge
            # can surface a real change summary when codex requests
            # approval (the approval params themselves don't carry the
            # changeset). Quirk #4 fix.
            self._track_pending_file_change(note)

            # Project into messages
            projection = projector.project(note)
            if projection.messages:
                result.projected_messages.extend(projection.messages)
            if projection.is_tool_iteration:
                result.tool_iterations += 1
                # Arm/refresh the post-tool quiet watchdog whenever a
                # tool-shaped item completes.
                last_tool_completion_at = time.monotonic()
            else:
                # Any non-tool projected activity (assistant message,
                # status update, etc.) means codex is still producing
                # output — clear the quiet timer so we don't fast-fail.
                if projection.messages or projection.final_text is not None:
                    last_tool_completion_at = None
            if projection.final_text is not None:
                # Codex can emit multiple agentMessage items in one turn
                # (e.g. partial then final). Take the last one as canonical.
                result.final_text = projection.final_text
                # Some codex builds tear a turn down by emitting a
                # `<turn_aborted>` marker in the agent message text and
                # never sending turn/completed. Treat the marker itself
                # as terminal so we don't burn the full deadline.
                if _has_turn_aborted_marker(projection.final_text):
                    turn_complete = True
                    result.interrupted = True
                    result.error = (
                        result.error or "codex reported turn_aborted"
                    )

            if method == "turn/completed":
                turn_complete = True
                turn_status = (
                    (note.get("params") or {}).get("turn") or {}
                ).get("status")
                if turn_status == "interrupted":
                    result.interrupted = True
                    if not self._interrupt_event.is_set():
                        result.error = (
                            "Codex 작업이 외부 요인으로 중단되어 완료되지 않았습니다."
                        )
                elif turn_status and turn_status != "completed":
                    err_obj = (
                        (note.get("params") or {}).get("turn") or {}
                    ).get("error")
                    if err_obj:
                        err_msg = _format_responses_error(err_obj, str(turn_status))
                        # If the turn failed for an auth/refresh reason,
                        # rewrite the error into a re-auth hint AND mark
                        # the session for retirement.
                        stderr_blob = "\n".join(
                            self._client.stderr_tail(40)
                        )
                        hint = _classify_oauth_failure(err_msg, stderr_blob)
                        if hint is not None:
                            result.error = hint
                            result.should_retire = True
                        else:
                            result.error = self._format_error_with_stderr(
                                f"turn ended status={turn_status}", err_msg
                            )

        if (
            not turn_complete
            and not result.interrupted
            and result.final_text
            and result.error is None
        ):
            logger.warning(
                "codex app-server turn reached inactivity deadline after a completed "
                "assistant message but before turn/completed; accepting "
                "the assistant text as the terminal response"
            )
            turn_complete = True

        if not turn_complete and not result.interrupted:
            # Hit the inactivity deadline. Issue interrupt to stop wasted compute, and
            # tell the caller to retire the session — a turn that never
            # finished is a strong sign codex is wedged in a way the next
            # turn shouldn't inherit.
            self._issue_interrupt(result.turn_id)
            result.interrupted = True
            if not result.error:
                result.error = self._format_error_with_stderr(
                    f"turn timed out after {turn_timeout}s without activity"
                )
            result.should_retire = True

        with self._active_turn_lock:
            self._active_turn_id = None
        self._resumed_active_turn_id = None
        self._interrupt_event.clear()
        return result

    def compact_thread(
        self,
        *,
        turn_timeout: float = 600.0,
        notification_poll_timeout: float = 0.25,
    ) -> TurnResult:
        """Trigger Codex-native history compaction for the current thread.

        `thread/compact/start` returns immediately; the actual compaction
        progress streams through the same turn/item notifications as a normal
        turn. We wait for the matching `turn/completed` so callers can treat a
        successful return as a completed compaction boundary.
        """
        result = TurnResult()
        try:
            self.ensure_started()
        except (CodexAppServerError, TimeoutError) as exc:
            result.error = self._format_error_with_stderr(
                "codex app-server startup failed", exc
            )
            result.should_retire = True
            return result

        assert self._client is not None and self._thread_id is not None
        result.thread_id = self._thread_id
        self._interrupt_event.clear()
        projector = CodexEventProjector()

        try:
            self._client.request(
                "thread/compact/start",
                {"threadId": self._thread_id},
                timeout=10,
            )
        except CodexAppServerError as exc:
            stderr_blob = "\n".join(self._client.stderr_tail(40))
            hint = _classify_oauth_failure(exc.message, stderr_blob)
            if hint is not None:
                result.error = hint
                result.should_retire = True
            else:
                result.error = self._format_error_with_stderr(
                    "thread/compact/start failed", exc
                )
            return result
        except TimeoutError as exc:
            stderr_blob = "\n".join(self._client.stderr_tail(40))
            hint = _classify_oauth_failure(stderr_blob)
            result.error = hint or self._format_error_with_stderr(
                "thread/compact/start timed out", exc
            )
            result.should_retire = True
            return result

        deadline = time.monotonic() + turn_timeout
        turn_complete = False

        while time.monotonic() < deadline and not turn_complete:
            if self._interrupt_event.is_set():
                self._issue_interrupt(result.turn_id)
                result.interrupted = True
                break

            if not self._client.is_alive():
                stderr_blob = "\n".join(self._client.stderr_tail(60))
                hint = _classify_oauth_failure(stderr_blob)
                if hint is not None:
                    result.error = hint
                else:
                    result.error = self._format_error_with_stderr(
                        "codex app-server subprocess exited unexpectedly",
                        tail_lines=20,
                    )
                result.should_retire = True
                break

            sreq = self._client.take_server_request(timeout=0)
            if sreq is not None:
                self._handle_server_request(sreq)
                continue

            note = self._client.take_notification(
                timeout=notification_poll_timeout
            )
            if note is None:
                continue

            method = note.get("method", "")
            observed_thread_id, observed_turn_id = _notification_scope_ids(note)
            if result.turn_id is None:
                if method == "turn/started":
                    if (
                        observed_thread_id is not None
                        and str(observed_thread_id) != str(self._thread_id)
                    ):
                        logger.debug(
                            "ignoring foreign compact turn/started: thread=%s",
                            observed_thread_id,
                        )
                        continue
                    if observed_turn_id is None:
                        logger.debug(
                            "ignoring compact turn/started without a turn id"
                        )
                        continue
                    result.turn_id = str(observed_turn_id)
                elif observed_turn_id is not None or method in {
                    "item/completed",
                    "turn/completed",
                }:
                    # thread/compact/start does not return a turn id. Until the
                    # new turn/started arrives, any terminal/projectable event
                    # is stale or cannot be safely attributed to this compaction.
                    logger.debug(
                        "ignoring codex notification before compact turn start: "
                        "method=%s",
                        method,
                    )
                    continue

            if not _notification_belongs_to_turn(
                note,
                thread_id=self._thread_id,
                turn_id=result.turn_id,
            ):
                logger.debug(
                    "ignoring foreign codex notification: method=%s", method
                )
                continue

            if self._on_event is not None:
                try:
                    self._on_event(note)
                except Exception:  # pragma: no cover - display callback
                    logger.debug("on_event callback raised", exc_info=True)

            _apply_token_usage_notification(result, note)
            _apply_compaction_notification(result, note)
            self._track_pending_file_change(note)

            projection = projector.project(note)
            if projection.messages:
                result.projected_messages.extend(projection.messages)
            if projection.is_tool_iteration:
                result.tool_iterations += 1
            if projection.final_text is not None:
                result.final_text = projection.final_text
                if _has_turn_aborted_marker(projection.final_text):
                    turn_complete = True
                    result.interrupted = True
                    result.error = (
                        result.error or "codex reported turn_aborted"
                    )

            if method == "turn/started":
                turn_obj = (note.get("params") or {}).get("turn") or {}
                result.turn_id = turn_obj.get("id") or result.turn_id
            elif method == "turn/completed":
                turn_complete = True
                turn_obj = (note.get("params") or {}).get("turn") or {}
                result.turn_id = turn_obj.get("id") or result.turn_id
                turn_status = turn_obj.get("status")
                if turn_status == "interrupted":
                    result.interrupted = True
                    result.error = result.error or "compact turn interrupted"
                elif turn_status and turn_status != "completed":
                    err_obj = turn_obj.get("error")
                    err_msg = _format_responses_error(err_obj, str(turn_status))
                    stderr_blob = "\n".join(self._client.stderr_tail(40))
                    hint = _classify_oauth_failure(err_msg, stderr_blob)
                    if hint is not None:
                        result.error = hint
                        result.should_retire = True
                    else:
                        result.error = self._format_error_with_stderr(
                            f"compact turn ended status={turn_status}",
                            err_msg,
                        )

        if not turn_complete and not result.interrupted:
            self._issue_interrupt(result.turn_id)
            result.interrupted = True
            if not result.error:
                result.error = self._format_error_with_stderr(
                    f"compact turn timed out after {turn_timeout}s"
                )
            result.should_retire = True

        return result

    # ---------- internals ----------

    def _issue_interrupt(self, turn_id: Optional[str]) -> bool:
        if self._client is None or self._thread_id is None or turn_id is None:
            return False
        try:
            self._client.request(
                "turn/interrupt",
                {"threadId": self._thread_id, "turnId": turn_id},
                timeout=5,
            )
            return True
        except CodexAppServerError as exc:
            # "no active turn to interrupt" is fine — already done.
            logger.debug("turn/interrupt non-fatal: %s", exc)
        except TimeoutError:
            logger.warning("turn/interrupt timed out")
        return False

    def _handle_server_request(self, req: dict) -> None:
        """Translate a codex server request (approval) into Hermes' approval
        flow, then send the response.

        Method names verified live against codex 0.130.0 (Apr 2026):
          item/commandExecution/requestApproval — exec approvals
          item/fileChange/requestApproval       — apply_patch approvals
          item/permissions/requestApproval      — permissions changes
                                                  (we decline; user controls
                                                  permission profile in
                                                  ~/.codex/config.toml).
        """
        if self._client is None:
            return
        method = req.get("method", "")
        rid = req.get("id")
        params = req.get("params") or {}

        if method == "item/commandExecution/requestApproval":
            decision = self._decide_exec_approval(params)
            self._client.respond(rid, {"decision": decision})
        elif method == "item/fileChange/requestApproval":
            decision = self._decide_apply_patch_approval(params)
            self._client.respond(rid, {"decision": decision})
        elif method == "item/permissions/requestApproval":
            # Codex sometimes asks to escalate permissions mid-turn. We
            # always decline — the user already chose their permission
            # profile in ~/.codex/config.toml and surprise escalations
            # shouldn't be silently accepted.
            self._client.respond(rid, {"decision": "decline"})
        elif method == "mcpServer/elicitation/request":
            # Codex's MCP layer asks the user for structured input on
            # behalf of an MCP server (e.g. tool-call confirmation,
            # OAuth, form data). For our own hermes-tools callback we
            # auto-accept — the user already approved Hermes' tools
            # by enabling the runtime, and we never expose anything
            # codex's built-in shell can't already do. For other MCP
            # servers we decline so the user explicitly opts in via
            # codex's own auth flow.
            server_name = params.get("serverName") or ""
            if server_name == "hermes-tools":
                self._client.respond(
                    rid,
                    {"action": "accept", "content": None, "_meta": None},
                )
            else:
                self._client.respond(
                    rid,
                    {"action": "decline", "content": None, "_meta": None},
                )
        else:
            # Unknown server request — codex can extend this surface. Reject
            # cleanly so codex doesn't hang waiting for us.
            logger.warning("Unknown codex server request: %s", method)
            self._client.respond_error(
                rid, code=-32601, message=f"Unsupported method: {method}"
            )

    def _decide_exec_approval(self, params: dict) -> str:
        if self._routing.auto_approve_exec:
            return "accept"
        command = params.get("command") or ""
        # Codex's CommandExecutionRequestApprovalParams has cwd as Optional —
        # fall back to the session's cwd when codex doesn't include it so the
        # approval prompt is never empty (quirk #10 fix).
        cwd = params.get("cwd") or self._cwd or "<unknown>"
        reason = params.get("reason")
        description = f"Codex requests exec in {cwd}"
        if reason:
            description += f" — {reason}"
        if self._approval_callback is not None:
            try:
                choice = self._approval_callback(
                    command, description, allow_permanent=False
                )
                return _approval_choice_to_codex_decision(choice)
            except Exception:
                logger.exception("approval_callback raised on exec request")
                return "decline"
        return "decline"  # fail-closed when no callback wired

    def _decide_apply_patch_approval(self, params: dict) -> str:
        if self._routing.auto_approve_apply_patch:
            return "accept"
        if self._approval_callback is not None:
            # FileChangeRequestApprovalParams gives us reason + grantRoot.
            # The actual changeset lives on the corresponding fileChange
            # item which the projector has already cached for us — look it
            # up by item_id so the user sees what's actually changing.
            reason = params.get("reason")
            grant_root = params.get("grantRoot")
            item_id = params.get("itemId") or ""
            change_summary = self._lookup_pending_file_change(item_id)
            description_parts = []
            if reason:
                description_parts.append(reason)
            if change_summary:
                description_parts.append(change_summary)
            if grant_root:
                description_parts.append(f"grants write to {grant_root}")
            description = (
                "; ".join(description_parts)
                if description_parts
                else "Codex requests to apply a patch"
            )
            command_label = (
                f"apply_patch: {change_summary}" if change_summary
                else f"apply_patch: {reason}" if reason
                else "apply_patch"
            )
            try:
                choice = self._approval_callback(
                    command_label,
                    description,
                    allow_permanent=False,
                )
                return _approval_choice_to_codex_decision(choice)
            except Exception:
                logger.exception("approval_callback raised on apply_patch")
                return "decline"
        return "decline"

    def _track_pending_file_change(self, note: dict) -> None:
        """Maintain self._pending_file_changes from item/started + item/completed
        notifications. Lets the apply_patch approval prompt show what's
        actually changing — codex's approval params don't carry the data."""
        method = note.get("method", "")
        params = note.get("params") or {}
        item = params.get("item") or {}
        if item.get("type") != "fileChange":
            return
        item_id = item.get("id") or ""
        if not item_id:
            return
        if method == "item/started":
            changes = item.get("changes") or []
            if not changes:
                self._pending_file_changes[item_id] = "1 change pending"
                return
            kinds: dict[str, int] = {}
            paths: list[str] = []
            for ch in changes:
                if not isinstance(ch, dict):
                    continue
                kind = (ch.get("kind") or {}).get("type") or "update"
                kinds[kind] = kinds.get(kind, 0) + 1
                p = ch.get("path") or ""
                if p:
                    paths.append(p)
            counts = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
            preview = ", ".join(paths[:3])
            if len(paths) > 3:
                preview += f", +{len(paths) - 3} more"
            self._pending_file_changes[item_id] = (
                f"{counts}: {preview}" if preview else counts
            )
        elif method == "item/completed":
            self._pending_file_changes.pop(item_id, None)

    def _lookup_pending_file_change(self, item_id: str) -> Optional[str]:
        """Look up an in-progress fileChange item by id and summarize its
        changes for the approval prompt. Returns None when we don't have
        the item cached (e.g. approval arrived before item/started, or
        fileChange item content not tracked yet)."""
        if not item_id:
            return None
        cached = self._pending_file_changes.get(item_id)
        if not cached:
            return None
        return cached


def _apply_token_usage_notification(result: TurnResult, note: dict) -> None:
    """Capture Codex app-server token usage updates for caller accounting.

    Codex does not put token usage on turn/completed. It emits a separate
    thread/tokenUsage/updated notification containing cumulative totals and
    the latest turn breakdown.
    """
    if not isinstance(note, dict) or note.get("method") != "thread/tokenUsage/updated":
        return
    params = note.get("params") or {}
    token_usage = params.get("tokenUsage") or {}
    if not isinstance(token_usage, dict):
        return
    last = token_usage.get("last")
    total = token_usage.get("total")
    if isinstance(last, dict):
        result.token_usage_last = dict(last)
    if isinstance(total, dict):
        result.token_usage_total = dict(total)
    window = token_usage.get("modelContextWindow")
    if isinstance(window, int) and window > 0:
        result.model_context_window = window


def _apply_compaction_notification(result: TurnResult, note: dict) -> None:
    """Capture Codex-native context compaction boundaries.

    Recent app-server builds expose compaction as a ContextCompaction item.
    Older builds also emit the deprecated thread/compacted notification. Both
    mean the underlying Codex thread history has been compacted.
    """
    if not isinstance(note, dict):
        return
    method = note.get("method") or ""
    params = note.get("params") or {}
    if not isinstance(params, dict):
        return

    if method == "thread/compacted":
        result.compacted = True
        result.thread_id = params.get("threadId") or result.thread_id
        result.turn_id = params.get("turnId") or result.turn_id
        return

    if method not in {"item/started", "item/completed"}:
        return

    item = params.get("item") or {}
    if not isinstance(item, dict) or item.get("type") != "contextCompaction":
        return

    result.compacted = True
    result.thread_id = params.get("threadId") or result.thread_id
    result.turn_id = params.get("turnId") or result.turn_id


def _approval_choice_to_codex_decision(choice: str) -> str:
    """Map Hermes approval choices onto codex's CommandExecutionApprovalDecision
    / FileChangeApprovalDecision wire values.

    Hermes returns 'once', 'session', 'always', or 'deny'.
    Codex expects 'accept', 'acceptForSession', 'decline', or 'cancel'
    (verified against codex-rs/app-server-protocol/src/protocol/v2/item.rs
    on codex 0.130.0).
    """
    if choice in {"once",}:
        return "accept"
    if choice in {"session", "always"}:
        return "acceptForSession"
    return "decline"


def _has_turn_aborted_marker(text: str) -> bool:
    """Return True if `text` contains any of the raw markers codex uses
    to signal a turn was aborted without emitting `turn/completed`.

    Codex emits `<turn_aborted>` (and sometimes `<turn_aborted/>`) as raw
    text inside agentMessage items when an interrupt or upstream error
    tears the turn down before the normal completion path fires. Mirrors
    openclaw beta.8's terminal-marker fix so we don't burn the full turn
    deadline waiting for a turn/completed that never comes.
    """
    if not text:
        return False
    for marker in _TURN_ABORTED_MARKERS:
        if marker in text:
            return True
    return False


def _get_hermes_version() -> str:
    """Best-effort Hermes version string for codex's userAgent line."""
    try:
        from importlib.metadata import version

        return version("hermes-agent")
    except Exception:  # pragma: no cover
        return "0.0.0"
