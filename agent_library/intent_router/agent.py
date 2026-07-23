"""
Intent Router — SQL vs semantic vs both routing.

Split out of the unified AutoGen pipeline. Owns the routing_agent that
selects Sql_Generator, Eryl_agent, or a both-dependent/independent path.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from autogen import AssistantAgent
from autogen.oai.client import OpenAIWrapper

from .runtime_config import (
    AZURE_OPENAI_API_BASE,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    GPT4_LLM_MODEL_DEPLOYMENT_NAME,
    get_table_schemas,
)


def _configure_azure_openai_preserve_dots(self, config, openai_config):
    openai_config["azure_deployment"] = openai_config.get("azure_deployment", config.get("model"))
    openai_config["azure_endpoint"] = openai_config.get(
        "azure_endpoint", openai_config.pop("base_url", None)
    )
    if openai_config.get("azure_ad_token_provider") == "DEFAULT":
        import azure.identity

        openai_config["azure_ad_token_provider"] = azure.identity.get_bearer_token_provider(
            azure.identity.DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )


OpenAIWrapper._configure_azure_openai = _configure_azure_openai_preserve_dots

LLM_CONFIG = {
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


def build_agents(llm_config: Optional[dict] = None) -> dict:
    cfg = llm_config or LLM_CONFIG
    schemas = get_table_schemas()

    routing_agent = AssistantAgent(
        name="routing_agent",
        system_message=f"""
You are the Routing Agent. Your role is to analyze the user's question and determine whether it should be processed as a SQL query, a semantic search, or a combination of both. Use the provided metadata to guide your decision and distinguish cases where the question may require both SQL and semantic analysis, either dependently or independently.

**Decision-Making Process**:

1. **SQL-based Questions**:
- If the question directly references fields or columns from the metadata (e.g., city_name, population, country) or follows a query-like structure, classify it as SQL-based.

2. **Semantic Questions**:
- If the question is abstract, general, or does not reference specific fields or columns from the metadata, classify it as a semantic question.

3. **Both-Dependent**:
- If the second part of the question depends on information from the first part, classify it as "both-dependent."
- Example: "Which country has the highest GDP and what are the key features?" Here, the second part depends on identifying the country with the highest GDP first.

4. **Both-Independent**:
- If both parts of the question can be addressed independently, classify it as "both-independent."
- Example: "Which country has the highest GDP rate and what are the key features of Indian GDP?"

**Meta_Data**:
{schemas}

**Routing Instructions**:
- For SQL-based questions, pass them to the Sql_Generator.
- For semantic questions, pass them to the Eryl_agent.
- For both-dependent questions, start with SQL then refine with semantic search.
- For both-independent questions, execute SQL then semantic search sequentially.

**Output Format in Strictly JSON Format**:

For SQL-based analysis:
{{
"initial_question": "Original question from the user",
"analysis_type": "SQL-based",
"next_agent": "Sql_Generator",
"next_step": "Proceed with SQL-based query"
}}

For Semantic-based analysis:
{{
"initial_question": "Original question from the user",
"analysis_type": "Semantic-based",
"next_agent": "Eryl_agent",
"next_step": "Proceed with semantic search"
}}

For Both-Dependent analysis:
{{
"initial_question": "Original question from the user",
"analysis_type": "Both-dependent",
"next_agent": "Sql_Generator",
"next_step": "Start with SQL-based query and refine with semantic search based on SQL results"
}}

For Both-Independent analysis:
{{
"initial_question": "Original question from the user",
"analysis_type": "Both-independent",
"next_agent": "Sql_Generator",
"next_step": "Execute SQL, then semantic search independently (sequentially)"
}}

Always return strict JSON only.
""",
        max_consecutive_auto_reply=3,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    return {"routing_agent": routing_agent}


ENTRY_AGENT = "routing_agent"
EXIT_AGENTS = {"Sql_Generator", "Eryl_agent", "user_proxy"}


def route(last_speaker_name: str, parsed_content: dict) -> Optional[str]:
    if last_speaker_name == "routing_agent":
        next_agent = parsed_content.get("next_agent", "")
        if next_agent == "Sql_Generator":
            return "Sql_Generator"
        if next_agent == "Eryl_agent":
            return "Eryl_agent"
        return "user_proxy"
    return None


from .chain_helpers import drive_chain as _drive_chain


class IntentRouterRunner:
    """Single entrypoint: classify question into SQL / semantic / both."""

    def run(self, question: str | None = None, initial_question: str | None = None, **_: Any) -> dict:
        q = (initial_question or question or "").strip()
        agents = build_agents()
        initial = json.dumps({"initial_question": q, "question": q})
        state = _drive_chain(
            agents=agents,
            entry_agent=ENTRY_AGENT,
            exit_agents=EXIT_AGENTS,
            route_fn=route,
            initial_message=initial,
        )
        state.setdefault("initial_question", q)
        return {
            "initial_question": state.get("initial_question", q),
            "analysis_type": state.get("analysis_type", ""),
            "next_agent": state.get("next_agent", ""),
            "next_step": state.get("next_step", ""),
            "mode": state.get("analysis_type", ""),
            "reason": state.get("next_step", ""),
        }
