"""Internal helpers for driving the Quin agent chain (no imports from agent.py)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from autogen import UserProxyAgent

MAX_CHAIN_ROUNDS = 50
MAX_EMPTY_STAGE_RETRIES = 2

# Speakers whose usable reply is a non-empty JSON object (handoff payload).
_JSON_HANDOFF_SPEAKERS = frozenset(
    {
        "Sql_Generator",
        "Sql_Execution_Critic",
        "Insight_Generator",
        "evaluation_agent",
    }
)


class QuinChainStageError(RuntimeError):
    """Raised when a chain stage yields no usable reply after bounded retries."""


def parse_message_content(content: str) -> dict[str, Any]:
    cleaned = (content or "").replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return {}
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def stage_reply_usable(speaker: str, content: str, parsed: dict[str, Any]) -> bool:
    """Return True when the stage produced a reply the chain can act on.

    Usability is **only** emptiness/parseability of the reply — not field-value
    checks like ``db_name == "SQL"`` (that literal is the contract placeholder).
    """
    text = (content or "").strip()
    if speaker in _JSON_HANDOFF_SPEAKERS:
        return bool(parsed)
    # Query_Executor / Sql_tool often return tool results or raw SQL rows, not
    # a handoff dict — non-empty content is enough.
    return bool(text)


def is_termination_msg(msg: dict[str, Any] | None) -> bool:
    """Stop the AutoGen turn once a successfully-parsed JSON handoff arrives.

    Without this, ``user_proxy`` keeps auto-replying until
    ``max_consecutive_auto_reply``, and later empty assistant replies overwrite
    a valid first reply when ``last_message`` is read.
    """
    if not isinstance(msg, dict):
        return False
    return bool(parse_message_content(_message_content(msg)))


def _message_content(msg: Any) -> str:
    if not isinstance(msg, dict):
        return str(msg or "")
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal / structured AutoGen content blocks.
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(text if isinstance(text, str) else str(text))
        return "\n".join(parts)
    return str(content or "")


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
            # Unhashable agent key — try name fallback only.
            continue
        if isinstance(raw, list) and raw:
            messages = raw
            break
    return messages


def _is_agent_turn(msg: Any, agent_name: str) -> bool:
    """Return True for messages authored by ``agent_name``.

    Classic AutoGen stores the *other* party's replies in
    ``user_proxy.chat_messages[agent]`` with ``role="user"`` (received). Do
    **not** treat ``role == "user"`` alone as "skip" — that discarded valid
    Sql_Generator JSON while ``last_message`` still showed it.
    """
    if not isinstance(msg, dict):
        return True
    name = msg.get("name")
    role = msg.get("role")
    if name == agent_name:
        return True
    if name in ("user_proxy", "user"):
        return False
    if role == "assistant" and name in (None, "assistant"):
        return True
    # Received peer turn without a reliable name: keep if not clearly ours.
    if role == "user" and name not in ("user_proxy", "user", None):
        # Named peer reply stored as role=user on the proxy side.
        return name == agent_name
    if role == "user" and name is None:
        # Ambiguous — allow parse attempt (stage_reply_usable still gates).
        return True
    return False


def _describe_unusable(
    speaker: str,
    *,
    last_content: str,
    last_parsed: dict[str, Any],
    history_len: int,
) -> str:
    """Human-readable rejection reason (never a blanket 'empty {}' lie)."""
    if stage_reply_usable(speaker, last_content, last_parsed):
        return (
            f"last_message IS usable (keys={sorted(last_parsed.keys())}), but "
            f"history scan ({history_len} msg(s)) failed to select it — "
            f"extractor bug or filter mismatch."
        )
    if not (last_content or "").strip():
        return "last_message content is empty."
    # Empty dict {} from json.loads("{}") or non-dict JSON.
    if not last_parsed:
        stripped = last_content.strip()
        if stripped == "{}" or stripped == "null":
            return "last_message parsed to an empty JSON object {}."
        return (
            "last_message is non-empty but did not parse to a non-empty JSON "
            f"object (preview={last_content[:180]!r})."
        )
    return f"last_message parsed but failed stage_reply_usable for {speaker!r}."


def extract_usable_stage_reply(
    *,
    speaker: str,
    user_proxy: Any,
    agent: Any,
) -> tuple[str, dict[str, Any]] | None:
    """
    Prefer the last usable agent reply in chat history (not merely last_message).

    Always also considers ``last_message(agent)`` so a valid reply is never
    dropped when history filtering is wrong (AutoGen role=user peer storage).
    """
    history = _iter_agent_messages(user_proxy, agent)
    agent_name = getattr(agent, "name", speaker)
    last = user_proxy.last_message(agent) or {}

    # Newest-first history, then last_message as a guaranteed fallback candidate.
    candidates: list[Any] = list(reversed(history)) if history else []
    if last:
        # Avoid duplicate work if last is already the first history entry.
        if not candidates or candidates[0] is not last:
            candidates.append(last)

    for msg in candidates:
        # Only apply agent-turn filter to history entries; last_message is
        # already scoped to this agent by AutoGen.
        if msg is not last and history and not _is_agent_turn(msg, agent_name):
            continue
        content = _message_content(msg)
        parsed = parse_message_content(content)
        if stage_reply_usable(speaker, content, parsed):
            return content, parsed

    return None


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


_CONTEXT_HANDOFF_SPEAKERS = frozenset(
    {"Sql_Execution_Critic", "Insight_Generator", "evaluation_agent"}
)
_PRESERVE_NONEMPTY_KEYS = frozenset(
    {"query_results", "sql_query", "query", "sql_question", "query_question"}
)
_CONTEXT_STATE_KEYS = (
    "initial_question",
    "analysis_type",
    "sql_question",
    "query_question",
    "sql_query",
    "query",
    "query_results",
    "db_name",
    "feedback",
)


def merge_chain_state(state: dict[str, Any], speaker: str, content: str, parsed: dict[str, Any]) -> None:
    if speaker == "Sql_tool":
        if (content or "").strip():
            state["query_results"] = content
    for key, value in parsed.items():
        if value is None:
            continue
        if key in _PRESERVE_NONEMPTY_KEYS and not str(value).strip() and state.get(key):
            continue
        state[key] = value
    if "Inference" in parsed:
        state["inference"] = parsed["Inference"]
    if state.get("sql_question") and not state.get("query_question"):
        state["query_question"] = state["sql_question"]


def _looks_insufficient_answer(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return True
    return (
        "no query question or query results" in lowered
        or "no data-driven answer" in lowered
        or "were provided, so no" in lowered
    )


def shape_output(state: dict[str, Any]) -> dict[str, Any]:
    query_answer = str(state.get("query_answer", "") or "")
    query_results = str(state.get("query_results", "") or "")
    if _looks_insufficient_answer(query_answer) and query_results.strip():
        query_answer = ""
    return {
        "query": state.get("query") or state.get("sql_query"),
        "query_answer": query_answer,
        "query_question": state.get("query_question")
        or state.get("sql_question")
        or state.get("initial_question", ""),
        "analysis_type": state.get("analysis_type", ""),
        "query_results": query_results,
        "inference": state.get("inference") or state.get("Inference", ""),
        "python_code": state.get("python_code"),
    }


def inject_chain_context_into_handoff(
    message: str | dict[str, Any],
    state: dict[str, Any],
    *,
    next_speaker: str,
) -> str | dict[str, Any]:
    """Embed accumulated Sql_tool/SQL fields into clear_history stage handoffs."""
    if next_speaker not in _CONTEXT_HANDOFF_SPEAKERS:
        return message

    context = {
        key: state[key]
        for key in _CONTEXT_STATE_KEYS
        if state.get(key) not in (None, "")
    }
    if not context:
        return message

    def _merge_into_payload(payload: dict[str, Any]) -> dict[str, Any]:
        merged = dict(payload)
        for key, value in context.items():
            existing = merged.get(key)
            if existing in (None, "") or (
                key == "query_results" and _looks_insufficient_answer(str(existing))
            ):
                merged[key] = value
        if merged.get("sql_question") and not merged.get("query_question"):
            merged["query_question"] = merged["sql_question"]
        if merged.get("sql_query") and not merged.get("query"):
            merged["query"] = merged["sql_query"]
        return merged

    if isinstance(message, dict):
        content = _message_content(message)
        parsed = parse_message_content(content)
        if parsed:
            out = dict(message)
            out["content"] = json.dumps(_merge_into_payload(parsed))
            return out
        merged = _merge_into_payload({})
        if (content or "").strip() and "query_results" not in merged:
            merged["query_results"] = content
        out = dict(message)
        out["content"] = json.dumps(merged)
        return out

    text = message if isinstance(message, str) else str(message or "")
    parsed = parse_message_content(text)
    if parsed:
        return json.dumps(_merge_into_payload(parsed))

    wrapped = dict(context)
    if (text or "").strip():
        wrapped.setdefault("query_results", text)
    return json.dumps(wrapped)


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

    Does **not** advance on an empty parsed handoff for JSON stages: retries the
    same stage up to ``MAX_EMPTY_STAGE_RETRIES``, then raises
    ``QuinChainStageError`` with the **real** rejection reason.
    """
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
        is_termination_msg=is_termination_msg,
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

        usable: tuple[str, dict[str, Any]] | None = None
        for attempt in range(MAX_EMPTY_STAGE_RETRIES + 1):
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
            if usable is not None:
                break
            if attempt < MAX_EMPTY_STAGE_RETRIES:
                continue

        if usable is None:
            last = user_proxy.last_message(agent) or {}
            raw = _message_content(last)
            parsed_last = parse_message_content(raw)
            history_len = len(_iter_agent_messages(user_proxy, agent))
            reason = _describe_unusable(
                current_name,
                last_content=raw,
                last_parsed=parsed_last,
                history_len=history_len,
            )
            raise QuinChainStageError(
                f"Quin chain stage {current_name!r} produced no usable reply "
                f"after {MAX_EMPTY_STAGE_RETRIES + 1} attempt(s). "
                f"Reason: {reason} Last content={raw!r}."
            )

        content, parsed = usable
        merge_chain_state(state, current_name, content, parsed)

        next_name = route_fn(current_name, parsed)
        if next_name is None or next_name in exit_agents:
            break

        message = content
        message = inject_chain_context_into_handoff(
            message, state, next_speaker=next_name
        )
        current_name = next_name

    return shape_output(state)
