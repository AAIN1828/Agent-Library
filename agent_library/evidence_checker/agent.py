"""
Evidence Checker — reconcile SQL + document evidence, flag conflicts.

Derived from Evaluation_Agent / critic_agent completeness and groundedness
checks in the unified AutoGen pipeline. Compares structured (SQL) and
semantic (Eryl) evidence before answer synthesis.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from autogen import AssistantAgent
from autogen.oai.client import OpenAIWrapper

from .runtime_config import (
    AZURE_OPENAI_API_BASE,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    GPT4_LLM_MODEL_DEPLOYMENT_NAME,
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


def evaluation_timestamp_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def build_agents(llm_config: Optional[dict] = None) -> dict:
    cfg = llm_config or LLM_CONFIG

    Evidence_Checker = AssistantAgent(
        name="Evidence_Checker",
        system_message="""
You are the Evidence Conflict and Strength Checker.

## Input
You receive JSON with:
- initial_question
- analysis_type
- query_answer (SQL / structured evidence, may be empty)
- llm_answer (document / semantic evidence, may be empty)
- optional query, inference, updated_question

## Tasks
1. Reconcile structured and semantic evidence against the initial_question.
2. Flag conflicts when SQL and document answers contradict each other on material facts.
3. Score evidence_strength HIGH / MEDIUM / LOW based on coverage and agreement.
4. Decide whether synthesis can proceed or more retrieval is needed.

## Output (strict JSON only)
{
  "initial_question": "<question>",
  "analysis_type": "<unchanged>",
  "query_answer": "<echo or empty>",
  "llm_answer": "<echo or empty>",
  "conflict_detected": false,
  "conflicts": [],
  "evidence_strength": "HIGH",
  "groundedness": 9,
  "completeness": 8,
  "faithfulness": 9,
  "evaluation_timestamp": "2026-05-28T14:30:00.000Z",
  "feedback": "<brief reconciliation note>",
  "next_agent": "grounded_answer",
  "ready_for_synthesis": true
}

When conflicts are material or evidence is too weak:
- conflict_detected: true
- evidence_strength: "LOW"
- ready_for_synthesis: false
- next_agent: "Sql_Generator" if more SQL needed, "Eryl_agent" if more documents needed, else "user_proxy"
- Include blocked_reason explaining the gap

Never invent facts that are not in query_answer or llm_answer.
""",
        max_consecutive_auto_reply=3,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    return {"Evidence_Checker": Evidence_Checker}


ENTRY_AGENT = "Evidence_Checker"
EXIT_AGENTS = {"grounded_answer", "Sql_Generator", "Eryl_agent", "user_proxy"}


def route(last_speaker_name: str, parsed_content: dict) -> Optional[str]:
    if last_speaker_name == "Evidence_Checker":
        next_agent = parsed_content.get("next_agent", "grounded_answer")
        if parsed_content.get("ready_for_synthesis", True):
            return "grounded_answer"
        return next_agent if next_agent else "user_proxy"
    return None


from .chain_helpers import drive_chain as _drive_chain


class EvidenceCheckerRunner:
    """Reconcile SQL + document evidence before answer synthesis."""

    def run(
        self,
        initial_question: str = "",
        analysis_type: str = "",
        query_answer: str = "",
        llm_answer: str = "",
        query: str | None = None,
        inference: str | None = None,
        **extra: Any,
    ) -> dict:
        payload = {
            "initial_question": initial_question,
            "analysis_type": analysis_type,
            "query_answer": query_answer or "",
            "llm_answer": llm_answer or "",
            "query": query,
            "inference": inference,
            "evaluation_timestamp": evaluation_timestamp_iso(),
            **{k: v for k, v in extra.items() if v is not None},
        }
        agents = build_agents()
        state = _drive_chain(
            agents=agents,
            entry_agent=ENTRY_AGENT,
            exit_agents=EXIT_AGENTS,
            route_fn=route,
            initial_message=json.dumps(payload),
        )
        for key, value in payload.items():
            state.setdefault(key, value)
        return state
