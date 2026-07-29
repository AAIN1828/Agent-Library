"""Internal helpers for driving the Quin agent chain (no imports from agent.py)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from autogen import UserProxyAgent

# Unmissable runtime marker — must appear in uvicorn logs / error text if this
# file is the one actually imported by the serving process.
QUIN_CHAIN_HELPERS_BUILD = "QUIN_HELPERS_BUILD_20260727_T1805_MARKER"
_log = logging.getLogger(__name__)

MAX_CHAIN_ROUNDS = 50
MAX_EMPTY_STAGE_RETRIES = 2

# Speakers whose usable reply is a non-empty JSON object (handoff payload).
_JSON_HANDOFF_SPEAKERS = frozenset(
    {
        "Sql_Generator",
        "Sql_Execution_Critic",
        "Insight_Generator",
    }
)

# Speakers that propose AutoGen tool/function calls for a downstream executor.
_TOOL_PROPOSER_SPEAKERS = frozenset({"Query_Executor"})

_TOOL_RESULT_ROLES = frozenset({"tool", "function"})


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
    """Return True when the stage produced a reply the chain can act on.

    Usability is **only** emptiness/parseability of the reply — not field-value
    checks like ``db_name == "SQL"`` (that literal is the contract placeholder).

    For tool-proposing stages (Query_Executor), a message that carries
    ``tool_calls`` / ``function_call`` is usable even when ``content`` is empty
    (OpenAI-style tool-only assistant turns).
    """
    text = (content or "").strip()
    if speaker in _JSON_HANDOFF_SPEAKERS:
        return bool(parsed)
    if speaker in _TOOL_PROPOSER_SPEAKERS and _tool_meta(msg):
        return True
    if speaker in _TOOL_PROPOSER_SPEAKERS and _is_tool_result_message(msg):
        return True
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
    if content is None:
        return ""
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

    Tool/function execution replies are **not** agent turns; Query_Executor
    extraction prefers them via ``_prefer_tool_result_message`` instead of
    changing this predicate (other stages still rely on it).
    """
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
    # Received peer turn without a reliable name: keep if not clearly ours.
    if role == "user" and name not in ("user_proxy", "user", None):
        # Named peer reply stored as role=user on the proxy side.
        return name == agent_name
    if role == "user" and name is None:
        # Ambiguous — allow parse attempt (stage_reply_usable still gates).
        return True
    return False


def _prefer_tool_result_message(history: list[Any]) -> dict[str, Any] | None:
    """Query_Executor selection rule (tool result over later NL paraphrase).

    Walk ``history`` newest-first. Return the most recent message whose
    ``role`` is in ``{tool, function}`` and whose content is non-empty.

    Rationale: after user_proxy executes ``db_execute_query``, the transcript
    contains both the tool-result JSON and a later Query_Executor NL summary;
    we must hand off the JSON records, not the paraphrase.
    """
    for msg in reversed(history):
        if _is_tool_result_message(msg):
            return msg if isinstance(msg, dict) else {"content": _message_content(msg)}
    return None


def _describe_unusable(
    speaker: str,
    *,
    last_content: str,
    last_parsed: dict[str, Any],
    history_len: int,
    last_msg: Any | None = None,
) -> str:
    """Human-readable rejection reason (never a blanket 'empty {}' lie)."""
    if stage_reply_usable(speaker, last_content, last_parsed, msg=last_msg):
        return (
            f"last_message IS usable (keys={sorted(last_parsed.keys())}), but "
            f"history scan ({history_len} msg(s)) failed to select it — "
            f"extractor bug or filter mismatch."
        )
    if _tool_meta(last_msg) and not (last_content or "").strip():
        return (
            "last_message has tool_calls/function_call but content is empty and "
            "this speaker is not treated as a tool proposer (or execution failed)."
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
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """
    Prefer the last usable agent reply in chat history (not merely last_message).

    Returns ``(content, parsed, raw_msg)`` so callers can forward
    ``tool_calls`` / ``function_call`` across stage boundaries.

    Query_Executor selection rule
    -----------------------------
    1. If any ``role in {tool, function}`` message with non-empty content exists
       in the current chat history, take the **most recent** such message
       (JSON records / error string from ``db_execute_query``).
    2. Else fall back to the previous NL / tool_calls-preferring scan via
       ``_is_agent_turn`` + ``stage_reply_usable`` (unchanged for other stages).

    Always also considers ``last_message(agent)`` so a valid reply is never
    dropped when history filtering is wrong (AutoGen role=user peer storage).
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

    Option (b) fallback: when Query_Executor emits ``tool_calls`` /
    ``function_call`` without a prior tool-result extract, forward those fields
    so Sql_tool can still execute them.
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

    # Prefer modern tool_calls; fall back to legacy function_call.
    if message.get("tool_calls") and hasattr(agent, "generate_tool_calls_reply"):
        _final, reply = agent.generate_tool_calls_reply(messages=[message])
        if isinstance(reply, dict):
            return _message_content(reply)
        if isinstance(reply, str) and reply.strip():
            return reply
        if isinstance(reply, list):
            # Some ag2 paths return a list of tool-role messages.
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
        # Pending tool_calls are not "already resolved" — leave for execute path.
        if _tool_meta(message) and not _message_content(message).strip():
            return ""
        return _message_content(message).strip()
    return ""


def _attach_sql_tool_functions_to_proxy(user_proxy: Any, agents: dict[str, Any]) -> None:
    """Register Sql_tool's execution map on user_proxy (AutoGen propose/execute split).

    Query_Executor has ``register_for_llm`` only; in ``user_proxy ↔ Query_Executor``
    chats the proxy must own ``register_for_execution`` or AutoGen returns
    ``Error: Function db_execute_query not found.``
    """
    sql_tool = agents.get("Sql_tool")
    if sql_tool is None:
        return
    fmap = getattr(sql_tool, "_function_map", None) or {}
    if not fmap:
        return
    user_proxy.register_function(dict(fmap), silent_override=True)


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


_CONTEXT_HANDOFF_SPEAKERS = frozenset({"Sql_Execution_Critic", "Insight_Generator"})
_PRESERVE_NONEMPTY_KEYS = frozenset(
    {"query_results", "sql_query", "query", "sql_question", "query_question", "result"}
)
_CONTEXT_STATE_KEYS = (
    "initial_question",
    "analysis_type",
    "sql_question",
    "query_question",
    "sql_query",
    "query",
    "query_results",
    "result",
    "feedback",
)


def merge_chain_state(state: dict[str, Any], speaker: str, content: str, parsed: dict[str, Any]) -> None:
    if speaker == "Sql_tool":
        # Prefer resolved tool-result text; never let a later empty overwrite wipe it.
        if (content or "").strip():
            state["query_results"] = content
    for key, value in parsed.items():
        if value is None:
            continue
        # Do not let empty strings from later stages erase Sql_tool results / SQL.
        if key in _PRESERVE_NONEMPTY_KEYS and not str(value).strip() and state.get(key):
            continue
        state[key] = value
    if "Inference" in parsed and not state.get("inference"):
        state["inference"] = parsed["Inference"]
    # Critic success payload uses "result"; mirror into query_results when empty.
    if parsed.get("result") and not state.get("query_results"):
        state["query_results"] = parsed["result"]
    # Keep sql_question mirrored for consumers that expect query_question.
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
    query_results = str(state.get("query_results") or state.get("result") or "")
    # Insight sometimes emits an insufficiency narrative when the handoff lacked
    # query_results; demote that so adapters can fall through to real rows.
    if _looks_insufficient_answer(query_answer) and query_results.strip():
        query_answer = ""
    return {
        "query": state.get("query") or state.get("sql_query") or "",
        "query_answer": query_answer,
        "analysis_type": state.get("analysis_type", "") or "SQL-based",
        "inference": state.get("inference") or state.get("Inference", ""),
        # Retained for adapters / older pipeline stages that still read rows.
        "query_results": query_results,
        "query_question": state.get("query_question")
        or state.get("sql_question")
        or state.get("initial_question", ""),
        "python_code": state.get("python_code"),
    }


def inject_chain_context_into_handoff(
    message: str | dict[str, Any],
    state: dict[str, Any],
    *,
    next_speaker: str,
) -> str | dict[str, Any]:
    """
    Each drive_chain stage uses clear_history=True, so Insight_Generator never
    sees Sql_tool's raw rows unless we embed them in the handoff JSON.

    Critic → Insight historically forwarded only the critic object (no
    query_results), which made Insight emit \"No query question or query
    results were provided...\".
    """
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
        # Mirror naming for Insight prompts that ask for query_question.
        if merged.get("sql_question") and not merged.get("query_question"):
            merged["query_question"] = merged["sql_question"]
        if merged.get("sql_query") and not merged.get("query"):
            merged["query"] = merged["sql_query"]
        return merged

    if isinstance(message, dict):
        content = _message_content(message)
        parsed = parse_message_content(content)
        if parsed:
            merged = _merge_into_payload(parsed)
            out = dict(message)
            out["content"] = json.dumps(merged)
            return out
        # Tool-call-only / non-JSON content: attach a JSON content sibling.
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

    # Raw Sql_tool JSON array / error string → wrap with chain context.
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
    initial_message: str | dict[str, Any],
) -> dict[str, Any]:
    """
    Parent-orchestrator loop: one agent turn per route() step, JSON handoff
    between speakers. Matches the integration contract documented on route().

    Does **not** advance on an empty parsed handoff for JSON stages: retries the
    same stage up to ``MAX_EMPTY_STAGE_RETRIES``, then raises
    ``QuinChainStageError`` with the **real** rejection reason.

    Sql_tool: if the inbound handoff is already a resolved tool-result string
    (JSON records / error text), pass it through without ``initiate_chat``.
    Option (b) tool_calls forwarding remains as a fallback when the inbound
    message still carries executable ``tool_calls`` / ``function_call``.
    """
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
        is_termination_msg=is_termination_msg,
    )
    _attach_sql_tool_functions_to_proxy(user_proxy, agents)

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
        for attempt in range(MAX_EMPTY_STAGE_RETRIES + 1):
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
                break

            # Sql_tool pass-through: already-resolved tool-result string from QE.
            if current_name == "Sql_tool":
                resolved = _inbound_resolved_content(message)
                if resolved:
                    usable = (
                        resolved,
                        parse_message_content(resolved),
                        {"content": resolved, "role": "tool", "name": "Sql_tool"},
                    )
                    break

            chat_message: str | dict[str, Any] = message
            user_proxy.initiate_chat(
                recipient=agent,
                message=chat_message,
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
                last_msg=last,
            )
            raise QuinChainStageError(
                f"[{QUIN_CHAIN_HELPERS_BUILD}] Quin chain stage {current_name!r} "
                f"produced no usable reply after {MAX_EMPTY_STAGE_RETRIES + 1} attempt(s). "
                f"Reason: {reason} Last content={raw!r}."
            )

        content, parsed, raw_msg = usable
        merge_chain_state(state, current_name, content, parsed)

        next_name = route_fn(current_name, parsed)
        if next_name is None or next_name in exit_agents:
            break

        message = build_handoff_message(content, raw_msg, speaker=current_name)
        message = inject_chain_context_into_handoff(
            message, state, next_speaker=next_name
        )
        current_name = next_name

    return shape_output(state)


# ---------------------------------------------------------------------------
# Reuse entrypoint — wraps agent.get_answer without editing agent.py
# ---------------------------------------------------------------------------
ENTRY_AGENT = "Sql_Generator"
EXIT_AGENTS = {"user_proxy"}


def build_agents(llm_config: Any | None = None) -> dict[str, Any]:
    """
    Return Quin's live module-level agents for orchestrator / drive_chain use.

    ``llm_config`` is accepted for API compatibility with other reuse packages;
    the standalone ``agent.py`` builds agents at import time, so the argument is
    ignored.
    """
    del llm_config  # agents are constructed in agent.py at import time
    from . import agent as quin_agent

    return {
        "Sql_Generator": quin_agent.Sql_Generator,
        "Query_Executor": quin_agent.Query_Executor,
        "Sql_tool": quin_agent.Sql_tool,
        "Sql_Execution_Critic": quin_agent.Sql_Execution_Critic,
        "Insight_Generator": quin_agent.Insight_Generator,
    }


def route(last_speaker_name: str, parsed_content: dict[str, Any]) -> str | None:
    """Mirror ``agent.state_transition`` as a name-based next-speaker function."""
    if last_speaker_name == "Sql_Generator":
        return "Query_Executor"
    if last_speaker_name == "Query_Executor":
        return "Sql_tool"
    if last_speaker_name == "Sql_tool":
        return "Sql_Execution_Critic"
    if last_speaker_name == "Sql_Execution_Critic":
        next_agent = (parsed_content or {}).get("next_agent", "")
        if next_agent == "Insight_Generator":
            return "Insight_Generator"
        return "Sql_Generator"
    if last_speaker_name == "Insight_Generator":
        return "user_proxy"
    return None


class QuinChainRunner:
    """Public reuse entrypoint — runs the standalone Quin GroupChat via get_answer."""

    def run(
        self,
        initial_question: str,
        analysis_type: str | None = None,
        updated_question: str | None = None,
        **_extra: Any,
    ) -> dict[str, Any]:
        """
        Run Quin end-to-end.

        Prefers ``updated_question`` when provided; otherwise uses
        ``initial_question``. Returns fields matching ``contract.json``
        ``output_schema`` (plus ``query_results`` for adapter compatibility).
        """
        from .agent import get_answer

        question = (updated_question or initial_question or "").strip()
        if not question:
            return {
                "query": "",
                "query_answer": "",
                "analysis_type": analysis_type or "SQL-based",
                "inference": "",
                "query_results": "",
            }

        result = get_answer(question)
        return {
            "query": result.get("query", "") or "",
            "query_answer": result.get("query_answer", "") or "",
            "analysis_type": analysis_type or "SQL-based",
            "inference": result.get("inference", "") or "",
            "query_results": result.get("query_results", "") or "",
        }
