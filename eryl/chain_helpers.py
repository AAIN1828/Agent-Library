"""Internal helpers for driving the Eryl agent chain (no imports from agent.py)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from autogen import UserProxyAgent

MAX_CHAIN_ROUNDS = 50

# Speakers that propose AutoGen tool/function calls for a downstream executor.
# Eryl_agent has register_for_llm(extract_context); retriever has register_for_execution.
_TOOL_PROPOSER_SPEAKERS = frozenset({"Eryl_agent"})

_TOOL_RESULT_ROLES = frozenset({"tool", "function"})


def parse_message_content(content: str) -> dict[str, Any]:
    cleaned = (
        (content or "")
        .replace("```json", "")
        .replace("```", "")
        .replace("Happy with the answer", "")
        .strip()
    )
    if not cleaned:
        return {}
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_meta(msg: Any) -> dict[str, Any]:
    """Pull function_call / tool_calls fields from an AutoGen message dict."""
    if not isinstance(msg, dict):
        return {}
    meta: dict[str, Any] = {}
    if msg.get("tool_calls"):
        meta["tool_calls"] = msg["tool_calls"]
    if msg.get("function_call"):
        meta["function_call"] = msg["function_call"]
    return meta


def _message_content(msg: Any) -> str:
    if not isinstance(msg, dict):
        return str(msg or "")
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(text if isinstance(text, str) else str(text))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content or "")


def _is_tool_result_message(msg: Any) -> bool:
    """True for AutoGen tool/function execution replies with non-empty content."""
    if not isinstance(msg, dict):
        return False
    if msg.get("role") not in _TOOL_RESULT_ROLES:
        return False
    return bool(_message_content(msg).strip())


def stage_reply_usable(
    speaker: str,
    content: str,
    parsed: dict[str, Any],
    *,
    msg: Any | None = None,
) -> bool:
    """Return True when the stage produced a reply the chain can act on."""
    text = (content or "").strip()
    if speaker in _TOOL_PROPOSER_SPEAKERS and _tool_meta(msg):
        return True
    if speaker in _TOOL_PROPOSER_SPEAKERS and _is_tool_result_message(msg):
        return True
    return bool(text)


def is_termination_msg(msg: dict[str, Any] | None) -> bool:
    """Stop the AutoGen turn once a successfully-parsed JSON handoff arrives."""
    if not isinstance(msg, dict):
        return False
    return bool(parse_message_content(_message_content(msg)))


def _iter_agent_messages(user_proxy: Any, agent: Any) -> list[Any]:
    """Best-effort chat history for ``agent`` from either side of the pair."""
    messages: list[Any] = []
    for holder, key in (
        (getattr(user_proxy, "chat_messages", None), agent),
        (getattr(agent, "chat_messages", None), user_proxy),
        (getattr(user_proxy, "chat_messages", None), getattr(agent, "name", None)),
        (getattr(agent, "chat_messages", None), getattr(user_proxy, "name", None)),
    ):
        if not isinstance(holder, dict) or key is None:
            continue
        try:
            raw = holder.get(key)
        except TypeError:
            continue
        if isinstance(raw, list) and raw:
            messages = raw
            break
    return messages


def _is_agent_turn(msg: Any, agent_name: str) -> bool:
    """Return True for messages authored by ``agent_name``."""
    if not isinstance(msg, dict):
        return True
    name = msg.get("name")
    role = msg.get("role")
    if role in _TOOL_RESULT_ROLES:
        return False
    if name == agent_name:
        return True
    if name in ("user_proxy", "user"):
        return False
    if role == "assistant" and name in (None, "assistant"):
        return True
    if role == "user" and name not in ("user_proxy", "user", None):
        return name == agent_name
    if role == "user" and name is None:
        return True
    return False


def _prefer_tool_result_message(history: list[Any]) -> dict[str, Any] | None:
    """Eryl_agent selection rule (tool result over later NL paraphrase).

    Walk ``history`` newest-first. Return the most recent message whose
    ``role`` is in ``{tool, function}`` and whose content is non-empty.

    Rationale: after user_proxy executes ``extract_context``, the transcript
    contains both the tool-result JSON and a later Eryl_agent NL/JSON summary;
    we must hand off the retrieved context, not the paraphrase.
    """
    for msg in reversed(history):
        if _is_tool_result_message(msg):
            return msg if isinstance(msg, dict) else {"content": _message_content(msg)}
    return None


def extract_usable_stage_reply(
    *,
    speaker: str,
    user_proxy: Any,
    agent: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """
    Prefer the last usable agent reply in chat history (not merely last_message).

    Returns ``(content, parsed, raw_msg)`` so callers can forward
    ``tool_calls`` / ``function_call`` across stage boundaries.

    Eryl_agent selection rule
    -------------------------
    1. If any ``role in {tool, function}`` message with non-empty content exists
       in the current chat history, take the **most recent** such message
       (vector/graph context from ``extract_context``).
    2. Else fall back to the previous NL / tool_calls-preferring scan.
    """
    history = _iter_agent_messages(user_proxy, agent)
    agent_name = getattr(agent, "name", speaker)
    last = user_proxy.last_message(agent) or {}

    if speaker in _TOOL_PROPOSER_SPEAKERS:
        preferred = _prefer_tool_result_message(history)
        if preferred is not None:
            content = _message_content(preferred)
            parsed = parse_message_content(content)
            return content, parsed, preferred

    candidates: list[Any] = list(reversed(history)) if history else []
    if last:
        if not candidates or candidates[0] is not last:
            candidates.append(last)

    for msg in candidates:
        if msg is not last and history and not _is_agent_turn(msg, agent_name):
            continue
        content = _message_content(msg)
        parsed = parse_message_content(content)
        if stage_reply_usable(speaker, content, parsed, msg=msg):
            raw = msg if isinstance(msg, dict) else {"content": content}
            return content, parsed, raw

    return None


def build_handoff_message(
    content: str,
    raw_msg: dict[str, Any],
    *,
    speaker: str,
) -> str | dict[str, Any]:
    """Build the next ``initiate_chat`` message, preserving tool-call metadata.

    Option (b) fallback: when Eryl_agent emits ``tool_calls`` /
    ``function_call`` without a prior tool-result extract, forward those fields
    so retriever can still execute them.
    """
    meta = _tool_meta(raw_msg)
    if not meta:
        return content
    handoff: dict[str, Any] = {
        "role": "assistant",
        "name": speaker,
        "content": content if (content or "").strip() else None,
    }
    handoff.update(meta)
    return handoff


def execute_registered_tools(agent: Any, message: str | dict[str, Any]) -> str | None:
    """Run ``tool_calls`` / ``function_call`` on ``agent`` before empty-reply checks.

    Returns the tool result content string, or ``None`` if this message has no
    executable calls or the agent has no matching registered functions.
    """
    if not isinstance(message, dict):
        return None
    meta = _tool_meta(message)
    if not meta:
        return None
    function_map = getattr(agent, "_function_map", None) or {}
    if not function_map:
        return None

    if message.get("tool_calls") and hasattr(agent, "generate_tool_calls_reply"):
        _final, reply = agent.generate_tool_calls_reply(messages=[message])
        if isinstance(reply, dict):
            return _message_content(reply)
        if isinstance(reply, str) and reply.strip():
            return reply
        if isinstance(reply, list):
            parts = [_message_content(item) for item in reply]
            joined = "\n".join(p for p in parts if p)
            return joined or None
        return None

    if message.get("function_call") and hasattr(agent, "generate_function_call_reply"):
        _final, reply = agent.generate_function_call_reply(messages=[message])
        if isinstance(reply, dict):
            return _message_content(reply)
        if isinstance(reply, str) and reply.strip():
            return reply
        return None

    return None


def _inbound_resolved_content(message: str | dict[str, Any]) -> str:
    """Non-empty content from a stage handoff that is already a tool result string."""
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, dict):
        if _tool_meta(message) and not _message_content(message).strip():
            return ""
        return _message_content(message).strip()
    return ""


def _attach_retriever_functions_to_proxy(user_proxy: Any, agents: dict[str, Any]) -> None:
    """Register retriever's execution map on user_proxy (AutoGen propose/execute split).

    Eryl_agent has ``register_for_llm`` only; in ``user_proxy ↔ Eryl_agent``
    chats the proxy must own ``register_for_execution`` or AutoGen returns
    ``Error: Function extract_context not found.``
    """
    retriever = agents.get("retriever")
    if retriever is None:
        return
    fmap = getattr(retriever, "_function_map", None) or {}
    if not fmap:
        return
    user_proxy.register_function(dict(fmap), silent_override=True)


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
    if speaker == "retriever":
        # Prefer resolved tool-result text; never let a later empty overwrite wipe it.
        if (content or "").strip():
            state["retrieved_context"] = content
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
        feedback_query = parsed.get("feedback_query")
        if feedback_query and str(feedback_query).strip().lower() not in ("none", ""):
            state["updated_question"] = feedback_query


def shape_output(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "llm_answer": state.get("llm_answer", "") or "",
        "analysis_type": state.get("analysis_type", "") or "Semantic-based",
        "updated_question": state.get("updated_question", "") or state.get("initial_question", "") or "",
        "question": state.get("initial_question", "") or state.get("updated_question", "") or "",
    }


def drive_chain(
    *,
    agents: dict[str, Any],
    entry_agent: str,
    exit_agents: set[str],
    route_fn: Callable[[str, str, dict[str, Any]], str | None],
    initial_message: str | dict[str, Any],
) -> dict[str, Any]:
    """
    Parent-orchestrator loop: one agent turn per route() step, JSON handoff
    between speakers. Matches the integration contract documented on route().

    Option (b) for Eryl_agent → retriever: attach retriever executables on
    user_proxy, execute forwarded ``tool_calls`` / ``function_call`` on the
    current agent before treating a stage as empty, and preserve tool metadata
    via ``build_handoff_message`` when leaving a tool-proposing stage.
    """
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
        is_termination_msg=is_termination_msg,
    )
    _attach_retriever_functions_to_proxy(user_proxy, agents)

    state: dict[str, Any] = {}
    message: str | dict[str, Any] = initial_message
    current_name = entry_agent

    for _ in range(MAX_CHAIN_ROUNDS):
        if current_name in exit_agents:
            break

        agent = agents.get(current_name)
        if agent is None:
            break

        usable: tuple[str, dict[str, Any], dict[str, Any]] | None = None

        # Option (b) fallback: execute forwarded tool_calls on this agent.
        tool_content = execute_registered_tools(agent, message)
        if tool_content is not None and stage_reply_usable(
            current_name,
            tool_content,
            parse_message_content(tool_content),
        ):
            usable = (
                tool_content,
                parse_message_content(tool_content),
                {"content": tool_content, "role": "tool", "name": current_name},
            )

        # Retriever pass-through: already-resolved tool-result string from Eryl_agent.
        if usable is None and current_name == "retriever":
            resolved = _inbound_resolved_content(message)
            if resolved:
                usable = (
                    resolved,
                    parse_message_content(resolved),
                    {"content": resolved, "role": "tool", "name": "retriever"},
                )

        if usable is None:
            user_proxy.initiate_chat(
                recipient=agent,
                message=message,
                clear_history=True,
                silent=True,
            )
            usable = extract_usable_stage_reply(
                speaker=current_name,
                user_proxy=user_proxy,
                agent=agent,
            )

        if usable is None:
            reply = user_proxy.last_message(agent) or {}
            content = _message_content(reply)
            parsed = parse_message_content(content)
            raw_msg = reply if isinstance(reply, dict) else {"content": content}
        else:
            content, parsed, raw_msg = usable

        merge_chain_state(state, current_name, content, parsed)

        next_name = route_fn(current_name, content, parsed)
        if next_name is None or next_name in exit_agents:
            break

        message = build_handoff_message(content, raw_msg, speaker=current_name)
        current_name = next_name

    return shape_output(state)


# ---------------------------------------------------------------------------
# Reuse entrypoint — wraps agent.get_answer without editing agent.py
# ---------------------------------------------------------------------------
ENTRY_AGENT = "Eryl_agent"
EXIT_AGENTS = {"user_proxy"}


def build_agents(llm_config: Any | None = None) -> dict[str, Any]:
    """
    Return Eryl's live module-level agents for orchestrator / drive_chain use.

    ``llm_config`` is accepted for API compatibility with other reuse packages;
    the standalone ``agent.py`` builds agents at import time, so the argument is
    ignored.
    """
    del llm_config  # agents are constructed in agent.py at import time
    from . import agent as eryl_agent

    return {
        "Eryl_agent": eryl_agent.Eryl_agent,
        "retriever": eryl_agent.retriever,
        "llm_answer_maker": eryl_agent.llm_answer_maker,
        "critic_agent": eryl_agent.critic_agent,
    }


def route(
    last_speaker_name: str,
    last_message: str = "",
    parsed_content: dict[str, Any] | None = None,
) -> str | None:
    """Mirror ``agent.state_transition`` as a name-based next-speaker function."""
    del parsed_content  # critic pass/fail uses the raw message string
    if last_speaker_name == "Eryl_agent":
        return "retriever"
    if last_speaker_name == "retriever":
        return "llm_answer_maker"
    if last_speaker_name == "llm_answer_maker":
        return "critic_agent"
    if last_speaker_name == "critic_agent":
        if "Happy with the answer" in (last_message or ""):
            return "user_proxy"
        return "Eryl_agent"
    return None


class ErylChainRunner:
    """Public reuse entrypoint — runs the standalone Eryl GroupChat via get_answer."""

    def run(
        self,
        initial_question: str,
        analysis_type: str | None = None,
        sql_query: str | None = None,
        sql_answer: str | None = None,
        updated_question: str | None = None,
        **_extra: Any,
    ) -> dict[str, Any]:
        """
        Run Eryl end-to-end.

        Prefers ``updated_question`` when provided; otherwise uses
        ``initial_question``. Returns fields matching ``contract.json``
        ``output_schema``.
        """
        del sql_query, sql_answer, _extra  # retained in signature for pipeline compat
        from .agent import get_answer

        question = (updated_question or initial_question or "").strip()
        if not question:
            return {
                "llm_answer": "",
                "analysis_type": analysis_type or "Semantic-based",
                "updated_question": updated_question or initial_question or "",
                "question": "",
            }

        result = get_answer(question)
        return {
            "llm_answer": result.get("llm_answer", "") or "",
            "analysis_type": analysis_type or "Semantic-based",
            "updated_question": updated_question or initial_question or question,
            "question": result.get("question", question),
        }
