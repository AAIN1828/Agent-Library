"""
Quin Agent — standalone SQL / structured-database search agent.

Flow:
    user_proxy -> Sql_Generator (writes SQL from the question)
    -> Query_Executor (sanity-checks the query) -> Sql_tool (executes it)
    -> Sql_Execution_Critic (checks the result / error; on error routes
       back to Sql_Generator with feedback) -> Insight_Generator
       (turns the raw rows into a direct answer) -> user_proxy

Everything unrelated to Quin's own generate -> execute -> validate -> answer
loop (routing agent, Responsible AI gate, Eryl flow, cross-flow evaluation,
DB logging, etc.) has been removed. This file runs end to end on its own —
no separate `metadata.py` step and no schema text file. Table schema is
introspected live from the database (via SQLAlchemy) at process start,
using only the connection details in `.env`.

Required .env keys (one of the two connection styles below):
    DATABASE_URL=mssql+pyodbc://...                # full SQLAlchemy URL, OR
    SQL_SERVER=yourserver.database.windows.net      # + the pieces below
    SQL_DATABASE=your_database
    SQL_USERNAME=your_username
    SQL_PASSWORD=your_password
    SQL_DRIVER=ODBC Driver 18 for SQL Server         # optional, this is the default
    SQL_SCHEMA=dbo                                   # optional, this is the default

    AZURE_OPENAI_API_KEY=...
    AZURE_OPENAI_API_BASE=...
    AZURE_OPENAI_API_VERSION=2024-02-15-preview       # optional, this is the default
    GPT4_LLM_MODEL_DEPLOYMENT_NAME=...

Optional override:
    SQL_METADATA=...   # if set, used verbatim instead of live DB introspection
                        # (rarely needed — mainly for schemas too large to
                        # introspect quickly, or when the DB isn't reachable
                        # from this process but you already have the schema
                        # text from elsewhere)
"""

import json
from collections.abc import Callable
from typing import Any
import logging
import os
import urllib.parse

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.oai.client import OpenAIWrapper

load_dotenv()


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing from the environment."""


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
# Azure OpenAI configuration — from .env
# ---------------------------------------------------------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
GPT4_LLM_MODEL_DEPLOYMENT_NAME = os.getenv("GPT4_LLM_MODEL_DEPLOYMENT_NAME")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
if AZURE_OPENAI_DEPLOYMENT and not GPT4_LLM_MODEL_DEPLOYMENT_NAME:
    GPT4_LLM_MODEL_DEPLOYMENT_NAME = AZURE_OPENAI_DEPLOYMENT

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, GPT4_LLM_MODEL_DEPLOYMENT_NAME]):
    raise ConfigurationError(
        "Missing Azure OpenAI configuration. Set AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_API_BASE, and GPT4_LLM_MODEL_DEPLOYMENT_NAME in your .env file."
    )

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

# ---------------------------------------------------------------------------
# Database configuration — from .env, either as a full DATABASE_URL or as
# discrete SQL_* pieces that get assembled into one here.
# ---------------------------------------------------------------------------
SQL_SCHEMA_NAME = os.getenv("SQL_SCHEMA", "dbo")


def _build_connection_string() -> str:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    server = os.getenv("SQL_SERVER")
    database = os.getenv("SQL_DATABASE")
    username = os.getenv("SQL_USERNAME")
    password = os.getenv("SQL_PASSWORD")
    driver = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")

    missing = [
        name
        for name, value in [
            ("SQL_SERVER", server),
            ("SQL_DATABASE", database),
            ("SQL_USERNAME", username),
            ("SQL_PASSWORD", password),
        ]
        if not value
    ]
    if missing:
        raise ConfigurationError(
            "No DATABASE_URL set, and the following SQL_* variables are "
            f"missing from .env: {', '.join(missing)}. Set either DATABASE_URL "
            "directly, or all of SQL_SERVER / SQL_DATABASE / SQL_USERNAME / SQL_PASSWORD."
        )

    odbc_connect = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password}"
    )
    return f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc_connect)}"


DATABASE_URL = _build_connection_string()
engine = create_engine(DATABASE_URL)


# ---------------------------------------------------------------------------
# Schema introspection — replaces the old metadata.py + quin_table_schemas.txt
# step. Runs once at import time, straight against the live database, using
# only the connection details above. No file is written or read.
# ---------------------------------------------------------------------------
def _introspect_schema(engine, schema_name: str) -> str:
    """Inspect the given schema and return a text description of all tables."""
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema=schema_name)

    schema_details = ""
    for table_name in tables:
        columns = inspector.get_columns(table_name, schema=schema_name)
        schema = (
            f'schema = "{schema_name}", '
            f'table_name = "{schema_name}.{table_name}", '
            f"{table_name}_table = Table(\n"
        )
        for column in columns:
            col_name = column["name"]
            col_type = column["type"]
            primary_key = "primary_key=True" if column.get("primary_key", False) else ""
            nullable = "nullable=False" if not column["nullable"] else ""
            schema += f'    Column("{col_name}", {col_type}, {primary_key} {nullable}),\n'
        schema = schema.rstrip(",\n")
        schema += "\n)\n\n"
        schema_details += schema

    return schema_details


def _load_sql_metadata() -> str:
    # Optional escape hatch: a caller can set SQL_METADATA directly in .env
    # to skip live introspection entirely (e.g. schema is huge, or this
    # process shouldn't touch the DB at import time).
    override = os.getenv("SQL_METADATA")
    if override:
        return override

    try:
        metadata = _introspect_schema(engine, SQL_SCHEMA_NAME)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to introspect database schema for schema '{SQL_SCHEMA_NAME}': {e}. "
            "Check DATABASE_URL / SQL_* connection settings in .env, or set SQL_METADATA "
            "directly to bypass live introspection."
        ) from e

    if not metadata:
        raise ConfigurationError(
            f"No tables found in schema '{SQL_SCHEMA_NAME}'. Check SQL_SCHEMA in .env, "
            "or set SQL_METADATA directly to bypass live introspection."
        )

    return metadata


SQL_METADATA = _load_sql_metadata()


def _clean_json(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def _parsed_last_message(last_message: str) -> dict:
    return json.loads(_clean_json(last_message))


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
user_proxy = UserProxyAgent(
    name="user_proxy",
    system_message="You pass the user's question to Sql_Generator and receive the final answer.",
    code_execution_config=False,
    max_consecutive_auto_reply=3,
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: msg["content"],
)

Sql_Generator = AssistantAgent(
    name="Sql_Generator",
    system_message=f"""
You are the Sql_Generator Agent. Use the provided metadata to turn the
question into a valid SQL query.

**Meta_Data**:
{SQL_METADATA}

Rules:
- Only use column/table names that appear in the metadata.
- Use TOP instead of LIMIT.
- If you are receiving feedback from Sql_Execution_Critic about a failed
  query, fix the query according to that feedback.
- If no valid query can be formed, set "sql_query" to null.

Output strictly in JSON:
{{
  "initial_question": "original question",
  "sql_query": "SQL query or null",
  "next_agent": "Query_Executor"
}}
""",
    max_consecutive_auto_reply=3,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

Query_Executor = AssistantAgent(
    name="Query_Executor",
    system_message="""
You are the Query_Executor Agent. Briefly confirm the SQL query from
Sql_Generator looks syntactically reasonable, then hand it to Sql_tool to run.
""",
    max_consecutive_auto_reply=3,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

Sql_tool = AssistantAgent(
    name="Sql_tool",
    system_message="You execute the given SQL query using the db_execute_query function and return its result.",
    max_consecutive_auto_reply=3,
    llm_config=None,
    human_input_mode="NEVER",
)

Sql_Execution_Critic = AssistantAgent(
    name="Sql_Execution_Critic",
    system_message=f"""
You evaluate the result (or error) coming back from Sql_tool.

**Meta_Data**:
{SQL_METADATA}

- If the query executed successfully, pass it on to Insight_Generator.
- If it errored, explain what went wrong and send it back to Sql_Generator.

Output strictly in JSON:

On success:
{{
  "sql_query": "the query that ran",
  "result": "<raw result>",
  "next_agent": "Insight_Generator",
  "feedback": "Happy with the result"
}}

On error:
{{
  "sql_query": "the query that failed",
  "next_agent": "Sql_Generator",
  "feedback": "explanation of what went wrong"
}}
""",
    max_consecutive_auto_reply=3,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

Insight_Generator = AssistantAgent(
    name="Insight_Generator",
    system_message="""
You turn SQL query results into a precise, professional answer.

- Base the answer only on the query results, no assumptions.
- Include numeric values and short summaries/counts where helpful.
- Keep it clear and concise.

Output strictly in JSON:
{
  "query": "the SQL query that was run",
  "query_answer": "the final natural-language answer",
  "inference": "one or two sentences of data-driven insight, if any"
}
""",
    max_consecutive_auto_reply=3,
    llm_config=llm_config,
    human_input_mode="NEVER",
)


@Sql_tool.register_for_execution()
@Query_Executor.register_for_llm(description="Execute a SQL query against the configured database and return the result.")
def db_execute_query(query: str = None, db_name: str = None):
    del db_name  # accepted for LLM/tool-call compatibility; connection is from .env
    if not query:
        return "No query provided."
    try:
        df = pd.read_sql(query, engine)
        return df.to_json(orient="records")
    except Exception as e:
        return f"An error occurred while executing the query: {e}"


# ---------------------------------------------------------------------------
# Flow control
# ---------------------------------------------------------------------------
def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    last_message = messages[-1]["content"]

    if last_speaker is user_proxy:
        return Sql_Generator

    elif last_speaker is Sql_Generator:
        return Query_Executor

    elif last_speaker is Query_Executor:
        return Sql_tool

    elif last_speaker is Sql_tool:
        return Sql_Execution_Critic

    elif last_speaker is Sql_Execution_Critic:
        try:
            parsed = _parsed_last_message(last_message)
        except json.JSONDecodeError:
            return Sql_Generator
        next_agent = parsed.get("next_agent", "")
        if next_agent == "Insight_Generator":
            return Insight_Generator
        return Sql_Generator

    elif last_speaker is Insight_Generator:
        return user_proxy


groupchat = GroupChat(
    agents=[
        user_proxy,
        Sql_Generator,
        Query_Executor,
        Sql_tool,
        Sql_Execution_Critic,
        Insight_Generator,
    ],
    messages=[],
    max_round=30,
    speaker_selection_method=state_transition,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)


def get_answer(question: str) -> dict:
    """Run the Quin pipeline end to end and return the final answer."""
    chat_history = user_proxy.initiate_chat(
        manager,
        message=json.dumps({"question": question}),
        summary_method="reflection_with_llm",
    )

    response = {"query": "", "query_answer": ""}
    for item in chat_history.chat_history:
        if item.get("name") == "Insight_Generator":
            try:
                parsed = _parsed_last_message(item["content"])
                response["query"] = parsed.get("query", "")
                response["query_answer"] = parsed.get("query_answer", "")
            except json.JSONDecodeError:
                pass

    return response


if __name__ == "__main__":
    # Standalone smoke run. For reuse/codegen, prefer QuinChainRunner.run
    # QuinChainRunner.run wraps get_answer with the orchestrator-friendly contract.
    q = "Calculate the average Sharpe ratio and Alpha for each fund manager. Among those managing a total combined AUM of over ₹10,000 Crore, which manager delivers the best risk-adjusted performance?"
    result = get_answer(q)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Reuse / orchestrator (formerly chain_helpers.py)
# ---------------------------------------------------------------------------

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
    del llm_config
    return {
        "Sql_Generator": Sql_Generator,
        "Query_Executor": Query_Executor,
        "Sql_tool": Sql_tool,
        "Sql_Execution_Critic": Sql_Execution_Critic,
        "Insight_Generator": Insight_Generator,
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
