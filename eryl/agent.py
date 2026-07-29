"""
Eryl Agent — standalone semantic / document search agent.

Flow:
    user_proxy -> Eryl_agent (forwards question) -> retriever (vector search)
    -> llm_answer_maker (answers from retrieved context) -> critic_agent
    (scores the answer; on FAIL sends a refined query back to Eryl_agent,
    on PASS returns control to user_proxy)

Everything unrelated to Eryl's own retrieve -> answer -> critique loop
(routing agent, Responsible AI gate, SQL flow, cross-flow evaluation,
SQL-driven "query_transformer" step, logging to a DB, etc.) has been
removed. This file is meant to run end to end on its own.
"""

import asyncio
import json
import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.oai.client import OpenAIWrapper

load_dotenv()

# ---------------------------------------------------------------------------
# AutoGen strips dots from Azure deployment names (gpt-5.4 -> gpt-54).
# Preserve the deployment name exactly as configured in Azure.
# ---------------------------------------------------------------------------
def _configure_azure_openai_preserve_dots(self, config, openai_config):
    openai_config["azure_deployment"] = openai_config.get("azure_deployment", config.get("model"))
    openai_config["azure_endpoint"] = openai_config.get("azure_endpoint", openai_config.pop("base_url", None))
    if openai_config.get("azure_ad_token_provider") == "DEFAULT":
        import azure.identity
        openai_config["azure_ad_token_provider"] = azure.identity.get_bearer_token_provider(
            azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )

OpenAIWrapper._configure_azure_openai = _configure_azure_openai_preserve_dots

# ---------------------------------------------------------------------------
# Configuration — fill these in via environment variables / .env
# ---------------------------------------------------------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
GPT4_LLM_MODEL_DEPLOYMENT_NAME = os.getenv("GPT4_LLM_MODEL_DEPLOYMENT_NAME")
EMBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME")

AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "dupont_email_demo")
AZURE_SEARCH_SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "my-semantic-config")

llm_config = {
    "config_list": [
        {
            "model": GPT4_LLM_MODEL_DEPLOYMENT_NAME,
            "azure_deployment": GPT4_LLM_MODEL_DEPLOYMENT_NAME,
            "api_type": "azure",
            "base_url": AZURE_OPENAI_API_BASE,
            "api_key": AZURE_OPENAI_API_KEY,
            "api_version": AZURE_OPENAI_API_VERSION,
        }
    ],
    "temperature": 0.1,
}

embedding_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_API_BASE,
)



def _read_optional_env(name: str) -> str:
    return os.getenv(name, "").strip()


AZURE_OPENAI_DEPLOYMENT = _read_optional_env("AZURE_OPENAI_DEPLOYMENT")
if AZURE_OPENAI_DEPLOYMENT and not GPT4_LLM_MODEL_DEPLOYMENT_NAME:
    GPT4_LLM_MODEL_DEPLOYMENT_NAME = AZURE_OPENAI_DEPLOYMENT


def get_vector_field_name() -> str:
    return _read_optional_env("AZURE_SEARCH_VECTOR_FIELD_NAME") or "content_vector"


def get_document_id_field_name() -> str:
    return _read_optional_env("AZURE_SEARCH_DOCUMENT_ID_FIELD_NAME") or "id"

search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)


def generate_embeddings(text: str) -> list:
    return embedding_client.embeddings.create(
        input=[text], model=EMBEDDING_MODEL_DEPLOYMENT_NAME
    ).data[0].embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_json(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def _extract_user_question(message: str) -> str:
    text = message.strip()
    if text.startswith('"question":'):
        return text.split('"question":', 1)[1].strip().strip('"').strip()
    try:
        payload = json.loads(_clean_json(text))
        if isinstance(payload, dict):
            return payload.get("question") or text
    except json.JSONDecodeError:
        pass
    return text


def _critic_passed(last_message: str) -> bool:
    return "Happy with the answer" in last_message


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
user_proxy = UserProxyAgent(
    name="user_proxy",
    system_message="You pass the user's question to Eryl_agent and receive the final answer.",
    code_execution_config=False,
    max_consecutive_auto_reply=3,
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: msg["content"],
)

Eryl_agent = AssistantAgent(
    name="Eryl_agent",
    system_message="""
You are the Eryl selector agent. Your job is to forward the current question
(or, if you are being re-invoked after a failed critique, the critic's
`feedback_query`) to the retriever agent so it can fetch relevant context.

Output strictly in JSON:
{
  "question": "the question to search for",
  "next_agent": "retriever"
}

Never return plain text outside JSON.
""",
    max_consecutive_auto_reply=3,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

retriever = AssistantAgent(
    name="retriever",
    system_message="Your role is to execute the extract_context function and pass the returned context to llm_answer_maker.",
    max_consecutive_auto_reply=3,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

llm_answer_maker = AssistantAgent(
    name="llm_answer_maker",
    system_message="""
You are the Answer Maker Agent. Generate a detailed answer to the question
using ONLY the provided retrieved context. Do not invent information.

- Include any relevant numbers or facts present in the context.
- If the context is insufficient, respond exactly with:
  "I do not have enough information to answer this question."
""",
    llm_config=llm_config,
    max_consecutive_auto_reply=4,
    description="Generates the final answer from retrieved context.",
)

critic_agent = AssistantAgent(
    name="critic_agent",
    system_message="""
You are the Critic Agent. Evaluate `llm_answer` against the question and the
context that was used to produce it.

Score these 0-10: helpfulness, relevance, groundedness, completeness, faithfulness.
composite_score = average of the five scores.
status = "PASS" if composite_score >= 7 else "FAIL".

Output strictly in JSON:
{
  "question": "<question>",
  "llm_answer": "<answer being evaluated>",
  "helpfulness": 9,
  "relevance": 9,
  "groundedness": 9,
  "completeness": 8,
  "faithfulness": 9,
  "composite_score": 8.8,
  "status": "PASS",
  "feedback_query": "None"
}

If status is "FAIL", set `feedback_query` to a focused follow-up question that
would retrieve the missing information, and explain the gap in `feedback_detail`.
If status is "PASS", set `feedback_query` to "None" and, after the JSON block,
append exactly the line: Happy with the answer
""",
    llm_config=llm_config,
    max_consecutive_auto_reply=5,
    description="Scores the semantic answer and requests another retrieval round on failure.",
    is_termination_msg=lambda msg: _critic_passed(msg.get("content", "")),
)


@retriever.register_for_execution()
@Eryl_agent.register_for_llm(description="Retrieve relevant context via vector search.")
async def extract_context(question: str = None) -> dict:
    if not question:
        return {}

    vector_query = VectorizedQuery(
        vector=generate_embeddings(question),
        k_nearest_neighbors=3,
        fields=get_vector_field_name(),
    )
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=[get_document_id_field_name(), "content"],
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name=AZURE_SEARCH_SEMANTIC_CONFIG,
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE,
        top=3,
    )
    return {"vector_context": [result["content"] for result in results]}


# ---------------------------------------------------------------------------
# Flow control
# ---------------------------------------------------------------------------
def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    last_message = messages[-1]["content"]

    if last_speaker is user_proxy:
        return Eryl_agent

    elif last_speaker is Eryl_agent:
        return retriever

    elif last_speaker is retriever:
        return llm_answer_maker

    elif last_speaker is llm_answer_maker:
        return critic_agent

    elif last_speaker is critic_agent:
        if _critic_passed(last_message):
            return user_proxy
        return Eryl_agent


groupchat = GroupChat(
    agents=[user_proxy, Eryl_agent, retriever, llm_answer_maker, critic_agent],
    messages=[],
    max_round=30,
    speaker_selection_method=state_transition,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)


def get_answer(question: str) -> dict:
    """Run the Eryl pipeline end to end and return the final answer."""
    chat_history = user_proxy.initiate_chat(
        manager,
        message=json.dumps({"question": question}),
        summary_method="reflection_with_llm",
    )

    llm_answer = ""
    for item in chat_history.chat_history:
        if item.get("name") == "llm_answer_maker":
            llm_answer = item["content"]

    return {"question": question, "llm_answer": llm_answer}


if __name__ == "__main__":
    # Standalone smoke run. For reuse/codegen, prefer ErylChainRunner.run
    # ErylChainRunner.run wraps get_answer with the orchestrator-friendly contract.
    q = "What specific global factors cloud the 2025 foreign exchange outlook? "
    result = get_answer(q)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Reuse / orchestrator (formerly chain_helpers.py)
# ---------------------------------------------------------------------------

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
    del llm_config
    return {
        "Eryl_agent": Eryl_agent,
        "retriever": retriever,
        "llm_answer_maker": llm_answer_maker,
        "critic_agent": critic_agent,
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
