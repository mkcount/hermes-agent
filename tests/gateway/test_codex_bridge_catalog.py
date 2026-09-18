"""Codex picker catalog contracts."""

from gateway.codex_bridge import catalog
from gateway.codex_bridge.handoff import CodexHandoffGraph


class _FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.requests = []
        self.closed = False
        self.__class__.instances.append(self)

    def initialize(self, **_kwargs):
        return {}

    def request(self, method, params, timeout):
        self.requests.append((method, params, timeout))
        return {
            "data": [
                {
                    "id": "thread-a",
                    "preview": "continuation prompt",
                    "createdAt": 20,
                    "updatedAt": 30,
                    "cwd": "/new",
                    "path": "/new.jsonl",
                    "status": {"type": "active"},
                },
                {
                    "id": "thread-a",
                    "preview": (
                        "[Note: model was just switched from old to new via OpenAI Codex. "
                        "Adjust your self-identification accordingly.]\n\noriginal task"
                    ),
                    "createdAt": 10,
                    "updatedAt": 20,
                    "cwd": "/old",
                    "path": "/old.jsonl",
                    "status": {"type": "idle"},
                },
                {
                    "id": "thread-b",
                    "name": "Named task",
                    "preview": "ignored preview",
                    "createdAt": 25,
                    "updatedAt": 25,
                    "cwd": "/other",
                    "path": "/other.jsonl",
                    "status": {"type": "idle"},
                },
            ],
        }

    def close(self):
        self.closed = True


def test_picker_uses_canonical_index_and_collapses_rollout_incarnations(monkeypatch):
    _FakeClient.instances.clear()
    monkeypatch.setattr(catalog, "find_codex_control_socket", lambda _home=None: "/control.sock")

    rows = catalog.list_recent_threads(limit=8, client_factory=_FakeClient)

    assert [(row.thread_id, row.title) for row in rows] == [
        ("thread-a", "original task"),
        ("thread-b", "Named task"),
    ]
    assert rows[0].rollout_path == "/new.jsonl"
    assert rows[0].status == "active"
    client = _FakeClient.instances[0]
    assert client.requests[0][1]["useStateDbOnly"] is True
    assert client.closed is True


def test_stored_replay_follows_handoff_chain_and_preserves_failed_commentary(monkeypatch):
    class ReplayClient(_FakeClient):
        def request(self, method, params, timeout):
            self.requests.append((method, params, timeout))
            return {
                "thread": {
                    "turns": [
                        {
                            "id": "turn-old",
                            "status": "interrupted",
                            "items": [{
                                "id": "old-report", "type": "agentMessage",
                                "phase": "commentary", "text": "first report",
                            }],
                        },
                        {
                            "id": "dying-server-turn",
                            "status": "interrupted",
                            "items": [{
                                "id": "wrong-report", "type": "agentMessage",
                                "phase": "commentary", "text": "must stay hidden",
                            }],
                        },
                        {
                            "id": "turn-successor",
                            "status": "failed",
                            "error": {"codexErrorInfo": "usageLimitExceeded"},
                            "items": [{
                                "id": "new-report", "type": "agentMessage",
                                "phase": "commentary", "text": "last report",
                            }],
                        },
                    ],
                },
            }

    ReplayClient.instances.clear()
    monkeypatch.setattr(catalog, "find_codex_control_socket", lambda _home=None: "/control.sock")

    replay = catalog.read_thread_replay(
        "thread-a",
        handoff_graph=CodexHandoffGraph(
            {"turn-old": "turn-successor"}, {"turn-old": "successor_bound"},
        ),
        client_factory=ReplayClient,
    )

    assert replay is not None
    assert replay.status == "failed"
    assert replay.error_code == "usage_limit_exceeded"
    assert [(frame.turn_id, frame.text) for frame in replay.commentary] == [
        ("turn-old", "first report"),
        ("turn-successor", "last report"),
    ]
    client = ReplayClient.instances[0]
    assert client.requests == [
        ("thread/read", {"threadId": "thread-a", "includeTurns": True}, 20),
    ]
    assert client.closed is True
