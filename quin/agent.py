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
def db_execute_query(query: str = None):
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
    # (agent_library.quin.chain_helpers) which wraps get_answer with the
    # orchestrator-friendly input/output contract.
    q = "Calculate the average Sharpe ratio and Alpha for each fund manager. Among those managing a total combined AUM of over ₹10,000 Crore, which manager delivers the best risk-adjusted performance?"
    result = get_answer(q)
    print(json.dumps(result, indent=2))