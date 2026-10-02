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
            if method == "thread/turns/list":
                return {
                    "data": [
                        {
                            "id": "turn-old",
                            "status": "interrupted",
                            "items": [],
                        },
                        {
                            "id": "dying-server-turn",
                            "status": "interrupted",
                            "items": [],
                        },
                        {
                            "id": "turn-successor",
                            "status": "failed",
                            "error": {"codexErrorInfo": "usageLimitExceeded"},
                            "items": [],
                        },
                    ],
                    "nextCursor": None,
                }
            reports = {
                "turn-old": [{
                    "id": "old-report", "type": "agentMessage",
                    "phase": "commentary", "text": "first report",
                }],
                "turn-successor": [{
                    "id": "new-report", "type": "agentMessage",
                    "phase": "commentary", "text": "last report",
                }],
            }
            return {"data": reports.get(params["turnId"], []), "nextCursor": None}

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
        ("thread/turns/list", {
            "threadId": "thread-a", "limit": 100,
            "sortDirection": "asc", "itemsView": "notLoaded",
        }, 20),
        ("thread/items/list", {
            "threadId": "thread-a", "turnId": "turn-old",
            "limit": 100, "sortDirection": "asc",
        }, 20),
        ("thread/items/list", {
            "threadId": "thread-a", "turnId": "dying-server-turn",
            "limit": 100, "sortDirection": "asc",
        }, 20),
        ("thread/items/list", {
            "threadId": "thread-a", "turnId": "turn-successor",
            "limit": 100, "sortDirection": "asc",
        }, 20),
    ]
    assert client.closed is True


def test_paged_replay_unwraps_real_item_envelopes(monkeypatch):
    class WrappedClient(_FakeClient):
        def request(self, method, params, timeout):
            self.requests.append((method, params, timeout))
            if method == "thread/turns/list":
                return {"data": [
                    {"id": f"turn-{index}", "status": "completed", "items": []}
                    for index in range(11)
                ]}
            if method == "thread/items/list":
                turn_id = params["turnId"]
                if turn_id != "turn-10":
                    return {"data": [{"turnId": turn_id, "item": {
                        "id": f"final-{turn_id}", "type": "agentMessage",
                        "phase": "final_answer", "text": f"answer {turn_id}",
                    }}]}
                return {"data": [
                    {"turnId": turn_id, "item": {
                        "id": "progress-1", "type": "agentMessage",
                        "phase": "commentary", "text": "working",
                    }},
                    {"turnId": turn_id, "item": {
                        "id": "final-1", "type": "agentMessage",
                        "phase": "final_answer", "text": "done",
                    }},
                ]}
            raise AssertionError(method)

    monkeypatch.setattr(catalog, "find_codex_control_socket", lambda _home=None: None)
    replay = catalog.read_thread_replay("thread-a", client_factory=WrappedClient)

    assert replay is not None
    assert [frame.text for frame in replay.commentary] == ["working"]
    assert replay.final_text == "done"
    assert replay.recent_turn_ids == tuple(f"turn-{index}" for index in range(1, 11))
    assert [frame.text for frame in replay.frames] == [
        *(f"answer turn-{index}" for index in range(1, 10)), "working", "done",
    ]
    assert len([method for method, _, _ in WrappedClient.instances[-1].requests if method == "thread/items/list"]) == 10


def test_invalid_paged_item_envelope_falls_back_to_full_thread_read(monkeypatch):
    class InvalidPageClient(_FakeClient):
        def request(self, method, params, timeout):
            if method == "thread/turns/list":
                return {"data": [{"id": "turn-1", "status": "completed"}]}
            if method == "thread/items/list":
                return {"data": [{"turnId": "wrong-turn", "item": {
                    "type": "agentMessage", "phase": "final_answer", "text": "bad",
                }}]}
            if method == "thread/read":
                return {"thread": {"turns": [{
                    "id": "turn-1", "status": "completed", "items": [{
                        "id": "final-1", "type": "agentMessage",
                        "phase": "final_answer", "text": "good",
                    }],
                }]}}
            raise AssertionError(method)

    monkeypatch.setattr(catalog, "find_codex_control_socket", lambda _home=None: None)
    replay = catalog.read_thread_replay("thread-a", client_factory=InvalidPageClient)

    assert replay is not None
    assert replay.final_text == "good"


def test_picker_includes_threads_created_by_app_server(monkeypatch):
    class FilteredClient(_FakeClient):
        def request(self, method, params, timeout):
            if "appServer" not in params.get("sourceKinds", []):
                return {"data": []}
            return {"data": [{"id": "telegram-thread", "source": "appServer", "updatedAt": 1}]}

    monkeypatch.setattr(catalog, "find_codex_control_socket", lambda _home=None: None)
    rows = catalog.list_recent_threads(client_factory=FilteredClient)
    assert [row.thread_id for row in rows] == ["telegram-thread"]
