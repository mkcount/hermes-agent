"""Codex app-server JSON-RPC client (stdio or desktop control proxy, codex 0.125+).

``initialize`` handshake, then ``thread/start`` + ``turn/start`` with streaming
``item/*`` notifications until ``turn/completed``. Wire-level speaker only —
projection, approvals and transcript handling live in sibling modules.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import socket
import stat
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Optional

from tools.environments.local import hermes_subprocess_env

MIN_CODEX_VERSION = (0, 125, 0)
_NOTIFICATION_QUEUE_LIMIT = 4096
_SERVER_REQUEST_QUEUE_LIMIT = 256
_WEBSOCKET_MESSAGE_LIMIT = 64 * 1024 * 1024


def find_codex_control_socket(codex_home: Optional[str] = None) -> Optional[str]:
    """Return the live local Codex desktop control socket, if present."""
    home = codex_home or os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    path = os.path.join(home, "app-server-control", "app-server-control.sock")
    try:
        return path if stat.S_ISSOCK(os.stat(path).st_mode) else None
    except OSError:
        return None


@dataclass
class CodexAppServerError(RuntimeError):
    """Raised on JSON-RPC errors from the app-server."""

    code: int
    message: str
    data: Optional[Any] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"codex app-server error {self.code}: {self.message}"


@dataclass(frozen=True)
class _TransportFailure:
    """Queue sentinel used to wake requests when the stdio reader dies."""

    error: RuntimeError


class CodexAppServerClient:
    """Minimal synchronous JSON-RPC 2.0 client for ``codex app-server`` over stdio.

    One reader thread routes replies to pending queues and notifications / server
    requests to queues; another captures stderr. Deliberately NOT async:
    AIAgent.run_conversation() is synchronous and cancels via ``turn/interrupt``.
    """

    def __init__(
        self, codex_bin: str = "codex", codex_home: Optional[str] = None,
        extra_args: Optional[list[str]] = None, env: Optional[dict[str, str]] = None,
        control_socket_path: Optional[str] = None,
    ) -> None:
        self._codex_bin = codex_bin
        self._control_socket_path = str(control_socket_path or "").strip() or None
        # codex needs LLM provider creds but must not receive Tier-1 Hermes secrets (gateway/GitHub/infra tokens).
        # codex app-server is a model-driving CLI executor: it runs a model-chosen agentic loop that
        # executes shell commands, so it legitimately needs LLM provider credentials
        # (inherit_credentials=True) to authenticate against the model endpoint. But the previous
        # `os.environ.copy()` also handed it every Tier-1 Hermes secret — gateway bot tokens, GitHub auth,
        # Modal/Daytona infra tokens, the dashboard session token, AUXILIARY_* side-LLM keys,
        # GATEWAY_RELAY_* auth — none of which a coding subprocess has any use for. Route through the
        # centralized helper so Tier-1 + dynamic-internal secrets are always stripped while provider creds
        # still flow, matching copilot_acp_client (#29157 sibling spawn-site gap).
        # The desktop proxy only forwards bytes to an already-authenticated
        # local app-server and doesn't need provider credentials. A standalone
        # app-server does, while the centralized helper still strips Hermes'
        # Tier-1 operational secrets in both modes.
        spawn_env = hermes_subprocess_env(
            inherit_credentials=not bool(self._control_socket_path)
        )
        if env:
            spawn_env.update(env)
        if codex_home:
            spawn_env["CODEX_HOME"] = codex_home

        cmd = (
            [codex_bin, "app-server", "proxy", "--sock", self._control_socket_path]
            if self._control_socket_path
            else [codex_bin, "app-server", *(extra_args or [])]
        )
        from agent.delegation_context import (
            DELEGATED_CHILD_ENV_MARKER, KANBAN_ENV_KEYS,
            delegated_child_subprocess_env, is_dispatcher_owned_worker_context,
        )
        # Native shell children remain unowned. Only Hermes' managed MCP tool
        # endpoint acts for this worker; grant it scope via its existing per-server
        # environment, never by granting the whole executor process ownership.
        owned_task = os.environ.get("HERMES_KANBAN_TASK") and is_dispatcher_owned_worker_context()
        if owned_task and not self._control_socket_path:
            for key in (*KANBAN_ENV_KEYS, "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
                if key in os.environ:
                    cmd += ["-c", f"mcp_servers.hermes-mcp.env.{key}={json.dumps(os.environ[key])}"]
            cmd += ["-c", f'mcp_servers.hermes-mcp.env.{DELEGATED_CHILD_ENV_MARKER}=""']
        spawn_env = delegated_child_subprocess_env(spawn_env)
        # Kanban workers must write handoff/status to the board DB outside the
        # workspace: keep the sandbox on, add the Kanban root as writable.
        if owned_task and not self._control_socket_path:
            kanban_db = spawn_env.get("HERMES_KANBAN_DB")
            default_root = os.path.join(spawn_env.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "kanban")
            kanban_root = os.path.dirname(kanban_db) if kanban_db else spawn_env.get("HERMES_KANBAN_ROOT", default_root)
            cmd += [
                "-c", 'sandbox_mode="workspace-write"',
                "-c", f'sandbox_workspace_write.writable_roots=["{kanban_root}"]',
                "-c", "sandbox_workspace_write.network_access=false",
            ]
        # Codex emits tracing to stderr; default WARN keeps it quiet for users.
        spawn_env.setdefault("RUST_LOG", "warn")

        # Hide the console the codex child would otherwise flash on Windows (#56747).
        # Hide-only — stdio pipes stay intact for the app-server wire.
        # See #56747.
        from hermes_cli._subprocess_compat import windows_hide_flags

        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, env=spawn_env, creationflags=windows_hide_flags(),
        )
        self._next_id = 1
        self._pending: dict[int, queue.Queue] = {}  # request id -> single-slot reply queue
        self._pending_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._send_lock = threading.Lock()
        # A disconnected or stalled consumer must not let an app-server flood grow
        # the gateway process without bound. Overflow is fatal because dropping a
        # server request (especially an approval) would deadlock the turn.
        self._notifications: queue.Queue = queue.Queue(maxsize=_NOTIFICATION_QUEUE_LIMIT)
        self._server_requests: queue.Queue = queue.Queue(maxsize=_SERVER_REQUEST_QUEUE_LIMIT)
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._closed = False
        self._initialized = False
        self._websocket = None
        self._proxy_socket: Optional[socket.socket] = None
        self._proxy_threads: list[threading.Thread] = []

        if self._control_socket_path:
            try:
                self._connect_control_proxy()
            except BaseException:
                with contextlib.suppress(Exception):
                    self._proc.terminate()
                    self._proc.wait(timeout=1.0)
                raise

        reader_target = self._read_websocket if self._websocket is not None else self._read_stdout
        self._reader = threading.Thread(target=reader_target, daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()
        self._stderr_reader.start()

    def initialize(
        self, client_name: str = "hermes", client_title: str = "Hermes Agent",
        client_version: str = "0.1", capabilities: Optional[dict] = None, timeout: float = 10.0,
    ) -> dict:
        """Send ``initialize`` + ``initialized``; return the server's InitializeResponse."""
        if self._initialized:
            raise RuntimeError("already initialized")
        params = {
            "clientInfo": {"name": client_name, "title": client_title, "version": client_version},
            "capabilities": capabilities or {},
        }
        result = self.request("initialize", params, timeout=timeout)
        self.notify("initialized")
        self._initialized = True
        return result

    def close(self, timeout: float = 3.0) -> None:
        """Close stdin and wait for the subprocess to exit, escalating to kill."""
        if self._closed:
            return
        self._closed = True
        self._fail_pending(RuntimeError("codex app-server client closed"))
        with contextlib.suppress(Exception):
            if self._websocket is not None:
                self._websocket.close()
        with contextlib.suppress(Exception):
            if self._proxy_socket is not None:
                self._proxy_socket.shutdown(socket.SHUT_RDWR)
                self._proxy_socket.close()
        with contextlib.suppress(Exception):
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        try:
            self._proc.terminate()
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                self._proc.kill()
                self._proc.wait(timeout=1.0)

    def __enter__(self) -> "CodexAppServerClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def request(self, method: str, params: Optional[dict] = None, timeout: float = 30.0) -> dict:
        """Send a request and block for ``result``; raise CodexAppServerError on ``error``."""
        with self._id_lock:
            rid, self._next_id = self._next_id, self._next_id + 1
        q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[rid] = q
        try:
            self._send({"id": rid, "method": method, "params": params or {}})
        except BaseException:
            # The request was never admitted to the transport. Leaving its
            # single-slot queue in _pending leaks one entry per broken write.
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"codex app-server method {method!r} timed out after {timeout}s")
        if isinstance(msg, _TransportFailure):
            raise msg.error
        if "error" in msg:
            err = msg["error"]
            raise CodexAppServerError(code=err.get("code", -1), message=err.get("message", ""), data=err.get("data"))
        return msg.get("result", {})

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        self._send({"method": method, "params": params or {}})

    def respond(self, request_id: Any, result: dict) -> None:
        """Reply to a server-initiated request (e.g. approval prompts)."""
        self._send({"id": request_id, "result": result})

    def respond_error(self, request_id: Any, code: int, message: str, data: Optional[Any] = None) -> None:
        """Reply to a server-initiated request with an error."""
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._send({"id": request_id, "error": err})

    @staticmethod
    def _take(q: queue.Queue, timeout: float) -> Optional[dict]:
        try:
            return q.get_nowait() if timeout <= 0 else q.get(timeout=timeout)
        except queue.Empty:
            return None

    def take_notification(self, timeout: float = 0.0) -> Optional[dict]:
        """Pop the next streaming notification, or None on timeout (0 = non-blocking)."""
        return self._take(self._notifications, timeout)

    def take_server_request(self, timeout: float = 0.0) -> Optional[dict]:
        """Pop the next server-initiated request (e.g. exec/applyPatch approval)."""
        return self._take(self._server_requests, timeout)

    def stderr_tail(self, n: int = 20) -> list[str]:
        """Return last n lines of codex's stderr (for error reports)."""
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def _send(self, obj: dict) -> None:
        if self._closed:
            raise RuntimeError("codex app-server client is closed")
        if self._proc.stdin is None:
            raise RuntimeError("codex app-server stdin not available")
        payload = json.dumps(obj)
        try:
            # request(), notify(), interrupt and approval replies may originate
            # on different threads. Serialize full JSONL frames so writes cannot
            # interleave at the pipe boundary.
            with self._send_lock:
                if self._websocket is not None:
                    self._websocket.send(payload)
                else:
                    self._proc.stdin.write((payload + "\n").encode("utf-8"))
                    self._proc.stdin.flush()
        except Exception as exc:
            error = RuntimeError(f"codex app-server stdin closed unexpectedly: {exc}")
            self._fatal_reader_failure(str(error))
            raise error from exc

    def _connect_control_proxy(self) -> None:
        """Bridge the CLI proxy pipes into ``websockets``' maintained codec.

        ``codex app-server proxy`` intentionally exposes an opaque byte stream
        on stdio. A socketpair lets the library own WebSocket framing and the
        HTTP upgrade while two tiny pumps remain byte-for-byte transports.
        """
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("codex app-server control proxy stdio not available")
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError("the websockets package is required for Codex desktop control") from exc

        client_socket, proxy_socket = socket.socketpair()
        self._proxy_socket = proxy_socket

        def close_proxy_socket() -> None:
            with contextlib.suppress(OSError):
                proxy_socket.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                proxy_socket.close()

        def socket_to_proxy() -> None:
            try:
                while data := proxy_socket.recv(64 * 1024):
                    view = memoryview(data)
                    while view:
                        written = os.write(self._proc.stdin.fileno(), view)
                        view = view[written:]
            except (OSError, ValueError):
                pass
            finally:
                close_proxy_socket()

        def proxy_to_socket() -> None:
            try:
                while data := os.read(self._proc.stdout.fileno(), 64 * 1024):
                    proxy_socket.sendall(data)
            except (OSError, ValueError):
                pass
            finally:
                close_proxy_socket()

        self._proxy_threads = [
            threading.Thread(target=socket_to_proxy, daemon=True),
            threading.Thread(target=proxy_to_socket, daemon=True),
        ]
        for pump in self._proxy_threads:
            pump.start()
        try:
            self._websocket = connect(
                "ws://localhost/rpc", sock=client_socket,
                # ``unix=True`` tells websockets not to apply TCP socket
                # options to the local socketpair. Codex doesn't negotiate
                # permessage-deflate, so omit that extension explicitly.
                unix=True, proxy=None, compression=None,
                max_size=_WEBSOCKET_MESSAGE_LIMIT, max_queue=32,
                open_timeout=10.0, close_timeout=1.0,
            )
        except BaseException:
            client_socket.close()
            with contextlib.suppress(Exception):
                proxy_socket.close()
            raise

    def _fail_pending(self, error: RuntimeError) -> None:
        """Wake every blocked request exactly once with a transport failure."""
        failure = _TransportFailure(error)
        with self._pending_lock:
            pending, self._pending = list(self._pending.values()), {}
        for waiter in pending:
            with contextlib.suppress(queue.Full):
                waiter.put_nowait(failure)

    def _fatal_reader_failure(self, message: str) -> None:
        """Make a lossy/closed reader observable and stop further production."""
        error = RuntimeError(message)
        self._append_stderr(f"<stdout reader error> {message}")
        self._fail_pending(error)
        with contextlib.suppress(Exception):
            if self._proc.poll() is None:
                self._proc.terminate()

    def _append_stderr(self, line: str) -> None:
        with self._stderr_lock:
            self._stderr_lines.append(line)
            if len(self._stderr_lines) > 500:  # bound memory
                self._stderr_lines = self._stderr_lines[-500:]

    def _read_stdout(self) -> None:
        if self._proc.stdout is None:
            self._fail_pending(RuntimeError("codex app-server stdout not available"))
            return
        try:
            for line in iter(self._proc.stdout.readline, b""):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Non-JSON stdout is unexpected; surface it via the stderr buffer.
                    self._append_stderr(f"<non-json on stdout> {line[:200]!r}")
                    continue
                self._dispatch(msg)
        except Exception as exc:
            self._fatal_reader_failure(str(exc))
            return
        if not self._closed:
            self._fatal_reader_failure("codex app-server stdout reached EOF")

    def _read_websocket(self) -> None:
        """Decode control-proxy messages with the pinned websockets library."""
        try:
            while not self._closed:
                message = self._websocket.recv()
                if message is None:
                    break
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                try:
                    parsed = json.loads(message)
                except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
                    self._append_stderr(f"<non-json websocket message> {str(message)[:200]!r}")
                    continue
                self._dispatch(parsed)
        except Exception as exc:
            if not self._closed:
                self._fatal_reader_failure(f"codex desktop control connection failed: {exc}")
            return
        if not self._closed:
            self._fatal_reader_failure("codex desktop control connection reached EOF")

    def _dispatch(self, msg: dict) -> None:
        if "id" in msg and ("result" in msg or "error" in msg):  # reply
            with self._pending_lock:
                pending = self._pending.pop(msg["id"], None)
            if pending is not None:
                with contextlib.suppress(queue.Full):  # pragma: no cover - defensive
                    pending.put_nowait(msg)
        elif "method" in msg:  # server-initiated request (has id) or notification
            target = self._server_requests if "id" in msg else self._notifications
            try:
                target.put_nowait(msg)
            except queue.Full:
                kind = "server-request" if "id" in msg else "notification"
                self._fatal_reader_failure(
                    f"codex app-server {kind} queue overflow; terminating transport to avoid silent loss"
                )

    def _read_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        with contextlib.suppress(Exception):  # pragma: no cover
            for line in iter(self._proc.stderr.readline, b""):
                self._append_stderr(line.decode("utf-8", "replace").rstrip())


def parse_codex_version(output: str) -> Optional[tuple[int, int, int]]:
    """Parse ``codex --version`` output ("codex-cli 0.130.0 ...") into (major, minor, patch)."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output or "")
    return tuple(int(g) for g in match.groups()) if match else None


def check_codex_binary(
    codex_bin: str = "codex", min_version: tuple[int, int, int] = MIN_CODEX_VERSION
) -> tuple[bool, str]:
    """Verify codex CLI is installed and meets minimum version. Returns (ok, message)."""
    try:
        proc = subprocess.run(
            [codex_bin, "--version"], capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=10, stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, f"codex CLI not found at {codex_bin!r}. Install with: npm i -g @openai/codex"
    except subprocess.TimeoutExpired:
        return False, "codex --version timed out"
    if proc.returncode != 0:
        return False, f"codex --version exited {proc.returncode}: {proc.stderr.strip()}"
    version = parse_codex_version(proc.stdout)
    if version is None:
        return False, f"could not parse codex version from: {proc.stdout!r}"
    have = ".".join(map(str, version))
    if version < min_version:
        return False, f"codex {have} is older than required {'.'.join(map(str, min_version))}. Run: npm i -g @openai/codex"
    return True, have


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from dataclasses import field  # noqa: F401,E402
import time  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
