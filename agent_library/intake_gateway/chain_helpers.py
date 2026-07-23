"""Internal helpers for driving the Intake Gateway agent chain."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from autogen import UserProxyAgent

MAX_CHAIN_ROUNDS = 10


def parse_message_content(content: str) -> dict[str, Any]:
    cleaned = content.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def drive_chain(
    *,
    agents: dict[str, Any],
    entry_agent: str,
    exit_agents: set[str],
    route_fn: Callable[[str, dict[str, Any]], str | None],
    initial_message: str,
) -> dict[str, Any]:
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
        for key, value in parsed.items():
            if value is not None:
                state[key] = value

        next_name = route_fn(current_name, parsed)
        if next_name is None or next_name in exit_agents:
            break
        if next_name == current_name:
            break
        message = content if content else message
        current_name = next_name

    return state
