"""
Grounded Answer — consolidate SQL + semantic evidence into one draft answer.

Derived from Insight_Generator + llm_answer_maker consolidation patterns
and final_answer assembly in the unified AutoGen pipeline.
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

    Grounded_Answer_Agent = AssistantAgent(
        name="Grounded_Answer_Agent",
        system_message="""
You are the Grounded Answer Synthesis agent.

## Input
JSON with:
- initial_question
- analysis_type
- query_answer (structured / SQL narrative; may be empty)
- llm_answer (document / RAG narrative; may be empty)
- optional: query, inference, conflict_detected, conflicts, evidence_strength, feedback from Evidence Checker

## Tasks
1. Build a single grounded draft answer using ONLY the provided evidence.
2. Prefer numerical facts from query_answer; prefer policy/procedural detail from llm_answer.
3. If conflict_detected is true, state the conflict clearly and do not silently pick a side without labeling uncertainty.
4. Keep the tone professional and concise; include numbers when present.
5. Do not invent facts outside the provided evidence. If evidence is insufficient, say so.

## Output (strict JSON only)
{
  "initial_question": "<question>",
  "analysis_type": "<unchanged>",
  "query_answer": "<echo structured portion used>",
  "llm_answer": "<echo semantic portion used>",
  "final_answer": "<single grounded draft for the user>",
  "reasoning": "<how sources were combined>",
  "Inference": "<brief data-driven inference or None>",
  "next_agent": "response_delivery"
}
""",
        max_consecutive_auto_reply=4,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    return {"Grounded_Answer_Agent": Grounded_Answer_Agent}


ENTRY_AGENT = "Grounded_Answer_Agent"
EXIT_AGENTS = {"response_delivery", "user_proxy"}


def route(last_speaker_name: str, parsed_content: dict) -> Optional[str]:
    if last_speaker_name == "Grounded_Answer_Agent":
        return parsed_content.get("next_agent") or "response_delivery"
    return None


from .chain_helpers import drive_chain as _drive_chain


class GroundedAnswerRunner:
    """Consolidate dual-source evidence into one grounded draft answer."""

    def run(
        self,
        initial_question: str = "",
        analysis_type: str = "",
        query_answer: str = "",
        llm_answer: str = "",
        **extra: Any,
    ) -> dict:
        payload = {
            "initial_question": initial_question,
            "analysis_type": analysis_type,
            "query_answer": query_answer or "",
            "llm_answer": llm_answer or "",
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
        if not state.get("final_answer"):
            parts = [p for p in (query_answer, llm_answer) if str(p).strip()]
            state["final_answer"] = "\n\n".join(parts)
        return {
            "initial_question": state.get("initial_question", initial_question),
            "analysis_type": state.get("analysis_type", analysis_type),
            "query_answer": state.get("query_answer", query_answer),
            "llm_answer": state.get("llm_answer", llm_answer),
            "final_answer": state.get("final_answer", ""),
            "reasoning": state.get("reasoning", ""),
            "Inference": state.get("Inference") or state.get("inference", ""),
            "next_agent": state.get("next_agent", "response_delivery"),
        }
