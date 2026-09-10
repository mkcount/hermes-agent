"""Telegram ↔ Codex integration boundary.

Bindings, durable input ownership, rollout ingestion, and delivery arbitration
live here. Hermes gateway and Telegram remain transport/UI shells.
"""

from gateway.codex_bridge.store import CodexBridgeBinding, CodexBridgeStore

__all__ = ["CodexBridgeBinding", "CodexBridgeStore"]
