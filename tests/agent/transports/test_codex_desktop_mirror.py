import json
from pathlib import Path

from agent.transports.codex_desktop_mirror import (
    CodexDesktopRolloutTail,
    read_codex_rollout_runtime_state,
    resolve_codex_rollout_path,
    snapshot_codex_rollout,
    snapshot_codex_rollout_latest,
)


def _event(payload):
    return json.dumps({"type": "event_msg", "payload": payload}) + "\n"


def _response_message(turn_id, role, text, *, phase=None):
    texts = text if isinstance(text, list) else [text]
    payload = {
        "type": "message",
        "role": role,
        "content": [
            {
                "type": "input_text" if role == "user" else "output_text",
                "text": item,
            }
            for item in texts
        ],
        "internal_chat_message_metadata_passthrough": {"turn_id": turn_id},
    }
    if phase is not None:
        payload["phase"] = phase
    return json.dumps({"type": "response_item", "payload": payload}) + "\n"


def _modern_turn(turn_id, *, final_text="done"):
    return (
        _event({"type": "task_started", "turn_id": turn_id})
        + _response_message(turn_id, "user", "desktop question")
        + _response_message(
            turn_id,
            "assistant",
            "관련 구조를 확인하고 있습니다.",
            phase="commentary",
        )
        + _response_message(
            turn_id,
            "assistant",
            final_text,
            phase="final_answer",
        )
        + _event(
            {
                "type": "task_complete",
                "turn_id": turn_id,
                "last_agent_message": final_text,
            }
        )
    )


def _turn(turn_id, *, client_id=None, final_text="done"):
    user = {"type": "user_message", "message": "hello"}
    if client_id is not None:
        user["client_id"] = client_id
    return (
        _event({"type": "task_started", "turn_id": turn_id})
        + _event(user)
        + _event(
            {
                "type": "task_complete",
                "turn_id": turn_id,
                "last_agent_message": final_text,
            }
        )
    )


def _turn_with_progress(turn_id):
    return (
        _event({"type": "task_started", "turn_id": turn_id})
        + _event({"type": "user_message", "message": "desktop question"})
        + _event({"type": "agent_reasoning", "text": "Inspecting files"})
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "Planning direct upload",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "관련 구조를 확인하고 있습니다.",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "final_answer",
                "message": "final answer",
            }
        )
        + _event(
            {
                "type": "task_complete",
                "turn_id": turn_id,
                "last_agent_message": "final answer",
            }
        )
    )


def _rollout(tmp_path: Path, thread_id: str) -> Path:
    path = (
        tmp_path
        / "sessions"
        / "2026"
        / "07"
        / "26"
        / f"rollout-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    return path


def test_tail_distinguishes_desktop_and_app_server_turns(tmp_path):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = _rollout(tmp_path, thread_id)
    path.write_text(
        _turn("desktop-turn", final_text="desktop answer")
        + _turn(
            "telegram-turn",
            client_id="hermes-client",
            final_text="telegram answer",
        ),
        encoding="utf-8",
    )

    tail = CodexDesktopRolloutTail(thread_id, path)
    completions = tail.scan()

    assert [item.turn_id for item in completions] == [
        "desktop-turn",
        "telegram-turn",
    ]
    assert completions[0].is_desktop_originated is True
    assert completions[1].is_desktop_originated is False
    assert completions[0].final_text == "desktop answer"
    assert tail.scan() == []


def test_tail_does_not_consume_partial_jsonl_line(tmp_path):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = _rollout(tmp_path, thread_id)
    complete = _turn("turn-1", final_text="first")
    partial = _event({"type": "task_started", "turn_id": "turn-2"}).rstrip(
        "\n"
    )
    path.write_text(complete + partial, encoding="utf-8")

    tail = CodexDesktopRolloutTail(thread_id, path)
    assert [item.turn_id for item in tail.scan()] == ["turn-1"]

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n"
            + _event({"type": "user_message", "message": "hello"})
            + _event(
                {
                    "type": "task_complete",
                    "turn_id": "turn-2",
                    "last_agent_message": "second",
                }
            )
        )
    assert [item.turn_id for item in tail.scan()] == ["turn-2"]


def test_tail_collects_question_public_progress_and_final_answer(tmp_path):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = _rollout(tmp_path, thread_id)
    path.write_text(_turn_with_progress("turn-progress"), encoding="utf-8")

    tail = CodexDesktopRolloutTail(thread_id, path)
    completions, updates = tail.scan_with_updates()

    assert len(completions) == 1
    completion = completions[0]
    assert completion.user_text == "desktop question"
    assert [item.kind for item in completion.progress] == ["commentary"]
    assert [item.text for item in completion.progress] == [
        "관련 구조를 확인하고 있습니다.",
    ]
    assert completion.final_text == "final answer"
    assert updates[-1].completed is True
    assert updates[-1].final_text == "final answer"


def test_tail_reads_modern_response_item_messages(tmp_path):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = _rollout(tmp_path, thread_id)
    path.write_text(
        _modern_turn("modern-turn", final_text="modern final answer"),
        encoding="utf-8",
    )

    completions, updates = CodexDesktopRolloutTail(
        thread_id, path
    ).scan_with_updates()

    assert len(completions) == 1
    assert completions[0].turn_id == "modern-turn"
    assert completions[0].user_text == "desktop question"
    assert [item.text for item in completions[0].progress] == [
        "관련 구조를 확인하고 있습니다."
    ]
    assert completions[0].final_text == "modern final answer"
    assert updates[-1].completed is True
    assert updates[-1].final_text == "modern final answer"


def test_tail_ignores_modern_runtime_context_before_user_prompt(tmp_path):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = _rollout(tmp_path, thread_id)
    context = [
        "<recommended_plugins>\nplugin list",
        "# AGENTS.md instructions for /workspace\nrule list",
        "<environment_context>cwd</environment_context>",
    ]
    path.write_text(
        _event({"type": "task_started", "turn_id": "context-turn"})
        + _response_message("context-turn", "user", context)
        + _response_message("context-turn", "user", "실제 사용자 요청")
        + _event(
            {
                "type": "task_complete",
                "turn_id": "context-turn",
                "last_agent_message": "완료 답변",
            }
        ),
        encoding="utf-8",
    )

    completions, updates = CodexDesktopRolloutTail(
        thread_id, path
    ).scan_with_updates()

    assert len(completions) == 1
    assert completions[0].user_text == "실제 사용자 요청"
    assert all("recommended_plugins" not in item.user_text for item in updates)


def test_initial_scan_exposes_current_incomplete_turn_snapshot(tmp_path):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = _rollout(tmp_path, thread_id)
    path.write_text(
        _event({"type": "task_started", "turn_id": "active-turn"})
        + _event(
            {
                "type": "user_message",
                "message": "진행 중인 작업 지시",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "첫 번째 누적 보고",
            }
        )
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "두 번째 누적 보고",
            }
        ),
        encoding="utf-8",
    )

    tail = CodexDesktopRolloutTail(thread_id, path)
    completions, updates = tail.scan_with_updates()

    assert completions == []
    assert updates
    assert updates[-1].turn_id == "active-turn"
    assert updates[-1].completed is False
    assert updates[-1].user_text == "진행 중인 작업 지시"
    assert [item.text for item in updates[-1].progress] == [
        "첫 번째 누적 보고",
        "두 번째 누적 보고",
    ]


def test_new_turn_implicitly_aborts_unfinished_previous_turn(tmp_path):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = _rollout(tmp_path, thread_id)
    path.write_text(
        _event({"type": "task_started", "turn_id": "orphan-turn"})
        + _event({"type": "user_message", "message": "중단된 질문"})
        + _event(
            {
                "type": "agent_message",
                "phase": "commentary",
                "message": "중단 전 진행 보고",
            }
        )
        + _turn("newer-turn", final_text="최신 완료 답변"),
        encoding="utf-8",
    )

    completions, updates = CodexDesktopRolloutTail(
        thread_id, path
    ).scan_with_updates()

    assert [item.turn_id for item in completions] == [
        "orphan-turn",
        "newer-turn",
    ]
    assert "이전 작업이 중단" in completions[0].error_text
    assert updates[-1].turn_id == "newer-turn"
    assert updates[-1].completed is True
    assert not any(
        update.turn_id == "orphan-turn" and not update.completed
        for update in updates[3:]
    )


def test_runtime_state_reads_latest_turn_context_from_rollout_tail(tmp_path):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = _rollout(tmp_path, thread_id)
    def context(model, effort):
        return json.dumps(
            {
                "type": "turn_context",
                "payload": {"model": model, "effort": effort},
            }
        ) + "\n"
    path.write_text(
        context("gpt-old", "low")
        + ("x" * 256)
        + "\n"
        + context("gpt-current", "XHIGH"),
        encoding="utf-8",
    )

    state = read_codex_rollout_runtime_state(
        thread_id,
        hinted_path=str(path),
        codex_home=str(tmp_path),
        max_scan_bytes=160,
    )

    assert state is not None
    assert state.model == "gpt-current"
    assert state.reasoning_effort == "xhigh"


def test_snapshot_and_path_resolution_stay_inside_codex_sessions(tmp_path):
    thread_id = "019fa0b8-f2d1-7f01-9749-953a39197b16"
    path = _rollout(tmp_path, thread_id)
    path.write_text(_turn("latest-turn"), encoding="utf-8")
    outside = tmp_path / f"outside-{thread_id}.jsonl"
    outside.write_text(_turn("outside-turn"), encoding="utf-8")

    assert resolve_codex_rollout_path(
        thread_id,
        hinted_path=str(outside),
        codex_home=str(tmp_path),
    ) == path.resolve()
    resolved_path, latest_turn = snapshot_codex_rollout(
        thread_id,
        codex_home=str(tmp_path),
    )
    assert resolved_path == str(path.resolve())
    assert latest_turn == "latest-turn"
    latest_path, latest_completion = snapshot_codex_rollout_latest(
        thread_id,
        codex_home=str(tmp_path),
    )
    assert latest_path == str(path.resolve())
    assert latest_completion is not None
    assert latest_completion.turn_id == "latest-turn"
    assert latest_completion.final_text == "done"
