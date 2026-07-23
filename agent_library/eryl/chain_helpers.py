"""Internal helpers for driving the Eryl agent chain (no imports from agent.py)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from autogen import UserProxyAgent

MAX_CHAIN_ROUNDS = 50


def parse_message_content(content: str) -> dict[str, Any]:
    cleaned = content.replace("```json", "").replace("```", "").replace("Happy with the answer", "").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_initial_message(
    initial_question: str,
    analysis_type: str | None,
    sql_query: str | None,
    sql_answer: str | None,
    updated_question: str | None,
) -> str:
    payload: dict[str, Any] = {
        "initial_question": initial_question,
        "analysis_type": analysis_type or "Semantic-based",
    }
    if sql_query is not None:
        payload["sql_query"] = sql_query
    if sql_answer is not None:
        payload["sql_answer"] = sql_answer
    if updated_question is not None:
        payload["updated_question"] = updated_question
    elif analysis_type in (None, "Semantic-based"):
        payload["updated_question"] = initial_question
    return json.dumps(payload)


def merge_chain_state(state: dict[str, Any], speaker: str, content: str, parsed: dict[str, Any]) -> None:
    for key, value in parsed.items():
        if value is not None:
            state[key] = value
    if speaker == "llm_answer_maker" and content and not parsed:
        state["llm_answer"] = content.strip()
    if speaker == "critic_agent":
        if parsed.get("llm_answer"):
            state["llm_answer"] = parsed["llm_answer"]
        if parsed.get("Updated_question"):
            state["updated_question"] = parsed["Updated_question"]


def shape_output(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "llm_answer": state.get("llm_answer", ""),
        "analysis_type": state.get("analysis_type", ""),
        "updated_question": state.get("updated_question", ""),
    }


def drive_chain(
    *,
    agents: dict[str, Any],
    entry_agent: str,
    exit_agents: set[str],
    route_fn: Callable[[str, str, dict[str, Any]], str | None],
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

        next_name = route_fn(current_name, content, parsed)
        if next_name is None or next_name in exit_agents:
            break

        message = content
        current_name = next_name

    return shape_output(state)
