"""Codex thread/project discovery through the app-server protocol."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from agent.transports.codex_app_server import CodexAppServerClient, find_codex_control_socket


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
            result = client.request(
                "thread/list",
                {"limit": max(1, min(int(limit), 100)), "archived": False,
                 "sortKey": "updated_at", "sortDirection": "desc"},
                timeout=15,
            )
            rows = [_thread_summary(item) for item in result.get("data") or []]
            return [row for row in rows if row is not None][:limit]
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
