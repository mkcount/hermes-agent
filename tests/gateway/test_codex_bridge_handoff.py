"""Formal ReloginTool handoff graph contracts."""

import json

from gateway.codex_bridge.handoff import read_thread_handoff_graph, read_turn_handoff


def _write(path, records):
    path.write_text(json.dumps({"codex_turn_handoffs": records}), encoding="utf-8")


def test_prepare_is_intent_until_the_physical_interrupt_is_confirmed(tmp_path):
    path = tmp_path / "state.json"
    _write(path, {
        "thread:a": {
            "thread_id": "thread", "interrupted_turn_id": "a",
            "continued_turn_id": "", "status": "prepared",
        },
    })

    handoff = read_turn_handoff("thread", "a", state_path=path)

    assert handoff is not None
    assert handoff.owns_continuation("completed") is False
    assert handoff.owns_continuation("interrupted") is True


def test_successor_chain_is_append_only_across_multiple_switches(tmp_path):
    path = tmp_path / "state.json"
    _write(path, {
        "thread:a": {
            "thread_id": "thread", "interrupted_turn_id": "a",
            "continued_turn_id": "b", "status": "successor_bound",
            "operation_id": "op-a",
        },
        "thread:b": {
            "thread_id": "thread", "interrupted_turn_id": "b",
            "continued_turn_id": "c", "status": "successor_bound",
            "operation_id": "op-b",
        },
    })

    handoff = read_turn_handoff("thread", "a", state_path=path)
    graph = read_thread_handoff_graph("thread", state_path=path)

    assert handoff is not None
    assert handoff.continuation_turn_ids == ("b", "c")
    assert graph.successors == {"a": "b", "b": "c"}
