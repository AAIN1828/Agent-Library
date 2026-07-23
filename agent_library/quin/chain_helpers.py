"""Internal helpers for driving the Quin agent chain (no imports from agent.py)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from autogen import UserProxyAgent

MAX_CHAIN_ROUNDS = 50


def parse_message_content(content: str) -> dict[str, Any]:
    cleaned = content.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_initial_message(
    initial_question: str,
    analysis_type: str | None,
    updated_question: str | None,
) -> str:
    payload: dict[str, Any] = {
        "initial_question": initial_question,
        "analysis_type": analysis_type or "SQL-based",
    }
    if updated_question is not None:
        payload["updated_question"] = updated_question
    return json.dumps(payload)


def merge_chain_state(state: dict[str, Any], speaker: str, content: str, parsed: dict[str, Any]) -> None:
    if speaker == "Sql_tool":
        state["query_results"] = content
    for key, value in parsed.items():
        if value is not None:
            state[key] = value
    if "Inference" in parsed:
        state["inference"] = parsed["Inference"]


def shape_output(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": state.get("query") or state.get("sql_query"),
        "query_answer": state.get("query_answer", ""),
        "analysis_type": state.get("analysis_type", ""),
        "query_results": state.get("query_results", ""),
        "inference": state.get("inference") or state.get("Inference", ""),
        "python_code": state.get("python_code"),
    }


def drive_chain(
    *,
    agents: dict[str, Any],
    entry_agent: str,
    exit_agents: set[str],
    route_fn: Callable[[str, dict[str, Any]], str | None],
    initial_message: str,
) -> dict[str, Any]:
    """
    Parent-orchestrator loop: one agent turn per route() step, JSON handoff
    between speakers. Matches the integration contract documented on route().
    """
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
    )

    state: dict[str, Any] = {}
    message = initial_message
    current_name = entry_agent

    for _ in range(MAX_CHAIN_ROUNDS):
        if current_name in exit_agents:
            break

        agent = agents.get(current_name)
        if agent is None:
            break

        user_proxy.initiate_chat(
            recipient=agent,
            message=message,
            clear_history=True,
            silent=True,
        )
        reply = user_proxy.last_message(agent) or {}
        content = reply.get("content", "") if isinstance(reply, dict) else str(reply)
        parsed = parse_message_content(content)
        merge_chain_state(state, current_name, content, parsed)

        next_name = route_fn(current_name, parsed)
        if next_name is None or next_name in exit_agents:
            break

        message = content
        current_name = next_name

    return shape_output(state)
