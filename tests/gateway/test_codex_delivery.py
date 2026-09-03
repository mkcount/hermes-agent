"""Behavior contracts for Codex execution-lane delivery authority."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.codex_delivery import CodexDeliveryAuthority
from gateway.stream_consumer import GatewayStreamConsumer


def _transition(
    authority: CodexDeliveryAuthority,
    control_key: str,
    thread_id: str | None,
) -> None:
    generation = authority.begin_transition(control_key)
    assert authority.commit_transition(control_key, generation, thread_id)


def test_binding_generation_prevents_aba_reauthorization():
    authority = CodexDeliveryAuthority()
    control_key = "telegram:456:123"

    authority.observe_binding(control_key, "thread-a")
    old_a = authority.issue_grant(control_key, "thread-a")
    assert old_a is not None
    assert authority.allows(old_a.to_metadata())

    _transition(authority, control_key, "thread-b")
    grant_b = authority.issue_grant(control_key, "thread-b")
    assert grant_b is not None
    assert not authority.allows(old_a.to_metadata())
    assert authority.allows(grant_b.to_metadata())

    _transition(authority, control_key, "thread-a")
    new_a = authority.issue_grant(control_key, "thread-a")
    assert new_a is not None
    assert not authority.allows(old_a.to_metadata())
    assert not authority.allows(grant_b.to_metadata())
    assert authority.allows(new_a.to_metadata())
    assert new_a.generation > old_a.generation


def test_begin_transition_revokes_before_persistent_commit():
    authority = CodexDeliveryAuthority()
    control_key = "telegram:456:123"
    authority.observe_binding(control_key, "thread-a")
    old_grant = authority.issue_grant(control_key, "thread-a")
    assert old_grant is not None

    generation = authority.begin_transition(control_key)

    assert not authority.allows(old_grant.to_metadata())
    assert authority.state_for(control_key).thread_id is None

    assert authority.commit_transition(control_key, generation, "thread-b")
    assert authority.state_for(control_key).thread_id == "thread-b"


def test_direct_turn_claim_is_generation_scoped_and_releasable():
    authority = CodexDeliveryAuthority()
    control_key = "telegram:456:123"
    authority.observe_binding(control_key, "thread-a")
    grant = authority.issue_grant(control_key, "thread-a")
    assert grant is not None
    metadata = grant.to_metadata()

    assert authority.acquire_direct_turn(metadata, "turn-1")
    assert authority.owns_direct_turn(metadata, "turn-1")

    authority.release_direct_turn(metadata, "turn-1")
    assert not authority.owns_direct_turn(metadata, "turn-1")

    assert authority.acquire_direct_turn(metadata, "turn-1")
    _transition(authority, control_key, "thread-b")
    assert not authority.owns_direct_turn(metadata, "turn-1")

    _transition(authority, control_key, "thread-a")
    new_grant = authority.issue_grant(control_key, "thread-a")
    assert new_grant is not None
    assert new_grant.generation > grant.generation
    assert not authority.owns_direct_turn(new_grant.to_metadata(), "turn-1")


def test_invalid_grant_cannot_claim_direct_turn():
    authority = CodexDeliveryAuthority()

    assert not authority.acquire_direct_turn({}, "turn-1")
    assert not authority.acquire_direct_turn(
        {
            "control_session_key": "telegram:456:123",
            "generation": 1,
            "thread_id": "thread-a",
        },
        "turn-1",
    )


def test_direct_client_claim_covers_pre_turn_gap_and_is_generation_scoped():
    authority = CodexDeliveryAuthority()
    control_key = "telegram:456:123"
    authority.observe_binding(control_key, "thread-a")
    grant = authority.issue_grant(control_key, "thread-a")
    assert grant is not None
    metadata = grant.to_metadata()

    assert authority.acquire_direct_client(metadata, "client-1")
    assert authority.owns_direct_client(metadata, "client-1")

    authority.release_direct_client(metadata, "client-1")
    assert not authority.owns_direct_client(metadata, "client-1")

    assert authority.acquire_direct_client(metadata, "client-1")
    _transition(authority, control_key, "thread-b")
    assert not authority.owns_direct_client(metadata, "client-1")

    _transition(authority, control_key, "thread-a")
    new_grant = authority.issue_grant(control_key, "thread-a")
    assert new_grant is not None
    assert not authority.owns_direct_client(
        new_grant.to_metadata(),
        "client-1",
    )


@pytest.mark.asyncio
async def test_transition_lock_serializes_same_control_chat():
    authority = CodexDeliveryAuthority()
    control_key = "telegram:456:123"
    lock = authority.transition_lock(control_key)
    entered = []
    release_first = asyncio.Event()

    async def first():
        async with lock:
            entered.append("first")
            await release_first.wait()

    async def second():
        async with authority.transition_lock(control_key):
            entered.append("second")

    first_task = asyncio.create_task(first())
    await asyncio.sleep(0)
    second_task = asyncio.create_task(second())
    await asyncio.sleep(0)
    assert entered == ["first"]

    release_first.set()
    await first_task
    await second_task
    assert entered == ["first", "second"]


@pytest.mark.asyncio
async def test_revoked_grant_stops_late_streamed_commentary():
    authority = CodexDeliveryAuthority()
    control_key = "telegram:456:123"
    authority.observe_binding(control_key, "thread-a")
    grant = authority.issue_grant(control_key, "thread-a")
    assert grant is not None

    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock()
    adapter.edit_message = AsyncMock()
    consumer = GatewayStreamConsumer(
        adapter,
        "456",
        run_still_current=lambda: authority.allows(grant.to_metadata()),
    )

    _transition(authority, control_key, "thread-b")
    consumer.on_commentary("late report from old A")
    consumer.finish()
    await consumer.run()

    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()
