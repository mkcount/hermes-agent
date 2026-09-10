"""Codex thread/project discovery through the app-server protocol."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from agent.transports.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
    find_codex_control_socket,
)


@dataclass(frozen=True)
class CodexThreadSummary:
    thread_id: str
    title: str
    cwd: str
    updated_at: int
    status: str
    rollout_path: str = ""


@dataclass(frozen=True)
class CodexProjectSummary:
    cwd: str
    name: str
    updated_at: int


def _thread_summary(value: object) -> Optional[CodexThreadSummary]:
    if not isinstance(value, dict):
        return None
    thread_id = str(value.get("id") or value.get("sessionId") or "").strip()
    if not thread_id:
        return None
    status_obj = value.get("status")
    status = (
        str(status_obj.get("type") or "unknown") if isinstance(status_obj, dict)
        else str(status_obj or "unknown")
    )
    title = " ".join(str(value.get("name") or value.get("preview") or "Untitled").split())
    return CodexThreadSummary(
        thread_id=thread_id, title=title, cwd=str(value.get("cwd") or ""),
        updated_at=int(value.get("updatedAt") or value.get("recencyAt") or 0),
        status=status, rollout_path=str(value.get("path") or ""),
    )


def _unique_thread_summaries(
    values: Iterable[object], *, limit: int,
) -> list[CodexThreadSummary]:
    """Collapse recovered rollout rows into logical app-server threads.

    A default ``thread/list`` may recover more than one paginated JSONL
    incarnation for the same thread id.  Runtime fields belong to the newest
    row, while an explicit name (or otherwise the oldest preview) is the
    stable user-facing title rather than a continuation fragment's first
    prompt.
    """
    merged: dict[str, tuple[CodexThreadSummary, bool, int]] = {}
    for value in values:
        summary = _thread_summary(value)
        if summary is None:
            continue
        raw = value if isinstance(value, dict) else {}
        named = bool(str(raw.get("name") or "").strip())
        created_at = int(raw.get("createdAt") or 0)
        current = merged.get(summary.thread_id)
        if current is None:
            merged[summary.thread_id] = (summary, named, created_at)
            continue

        newest, title_named, title_created_at = current
        title = newest.title
        if summary.updated_at > newest.updated_at:
            newest = summary
        if named and not title_named:
            title, title_named, title_created_at = summary.title, True, created_at
        elif named == title_named and created_at and (
            not title_created_at or created_at < title_created_at
        ):
            title, title_created_at = summary.title, created_at
        merged[summary.thread_id] = (
            CodexThreadSummary(
                thread_id=newest.thread_id,
                title=title,
                cwd=newest.cwd,
                updated_at=newest.updated_at,
                status=newest.status,
                rollout_path=newest.rollout_path,
            ),
            title_named,
            title_created_at,
        )

    rows = [entry[0] for entry in merged.values()]
    rows.sort(key=lambda row: row.updated_at, reverse=True)
    return rows[:max(1, int(limit))]


def _list_thread_rows(client: CodexAppServerClient, *, limit: int) -> list[object]:
    requested = max(1, min(int(limit), 100))
    params = {
        "limit": requested,
        "archived": False,
        "sortKey": "updated_at",
        "sortDirection": "desc",
        # The state database has one current row per logical thread.  The
        # default JSONL recovery scan can expose each paginated incarnation as
        # a separate picker item with a fragment-local preview.
        "useStateDbOnly": True,
    }
    try:
        result = client.request("thread/list", params, timeout=15)
    except CodexAppServerError as exc:
        if exc.code not in {-32601, -32602}:
            raise
        params.pop("useStateDbOnly")
        result = client.request("thread/list", params, timeout=15)
    return list(result.get("data") or [])


def list_recent_threads(
    *, limit: int = 8, codex_home: Optional[str] = None,
    client_factory: Callable[..., CodexAppServerClient] = CodexAppServerClient,
) -> list[CodexThreadSummary]:
    """Read the thread index, preferring the live desktop control connection."""
    control_socket = find_codex_control_socket(codex_home)
    for socket_path in ((control_socket, None) if control_socket else (None,)):
        kwargs = {"codex_home": codex_home}
        if socket_path:
            kwargs["control_socket_path"] = socket_path
        client = None
        try:
            client = client_factory(**kwargs)
            client.initialize(
                client_name="hermes-codex-picker", client_title="Hermes Codex Session Picker",
                client_version="1",
            )
            rows = _list_thread_rows(client, limit=limit)
            return _unique_thread_summaries(rows, limit=limit)
        except Exception:
            if not socket_path:
                raise
        finally:
            if client is not None:
                client.close()
    return []


def list_recent_projects(
    *, limit: int = 10, codex_home: Optional[str] = None,
    client_factory: Callable[..., CodexAppServerClient] = CodexAppServerClient,
) -> list[CodexProjectSummary]:
    projects: list[CodexProjectSummary] = []
    seen: set[str] = set()
    for thread in list_recent_threads(limit=100, codex_home=codex_home, client_factory=client_factory):
        cwd = os.path.abspath(os.path.expanduser(thread.cwd)) if thread.cwd else ""
        if not cwd or not os.path.isdir(cwd):
            continue
        identity = os.path.normcase(os.path.normpath(cwd))
        if identity in seen:
            continue
        seen.add(identity)
        projects.append(CodexProjectSummary(
            cwd=cwd, name=os.path.basename(cwd.rstrip(os.sep)) or cwd, updated_at=thread.updated_at,
        ))
        if len(projects) >= max(1, int(limit)):
            break
    return projects
