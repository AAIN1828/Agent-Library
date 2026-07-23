"""
Response Delivery — confidence-aware polish + post-generation evaluation.

Derived from critic_agent / evaluation_agent post-generation rubric and
final user-facing response assembly in the unified AutoGen pipeline.
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

POST_GENERATION_EVALUATION_RUBRIC = """
## Post-Generation Evaluation Dimensions (score each 0–10 unless noted)

| Dimension | What to assess |
|-----------|----------------|
| **helpfulness** | How useful the answer is for the user |
| **relevance** | Alignment with the question |
| **level_of_detail** | Sufficient depth without unnecessary filler |
| **groundedness** | Supported by SQL results / retrieved context |
| **completeness** | All parts of the question addressed |
| **faithfulness** | No distortion of source data |
| **content_safety** | No harmful or unsafe generated content |

Also assess generated **output**:
- **output_safety**: `pii_leak`, `toxic_output`, `score` (0–10)
- **output_governance**: `restricted_data_exposed`, `score` (0–10)
- **output_compliance**: `policy_violation`, `score` (0–10)

## Composite score & status
```
composite_score = round(
  (helpfulness + relevance + level_of_detail + groundedness + completeness
   + faithfulness + content_safety
   + output_safety.score + output_governance.score + output_compliance.score) / 10
, 1)
status = "PASS" if composite_score >= 7 else "FAIL"
```

Use differentiated scores; reserve 10 for exceptional performance.
Include `confidence` HIGH/MEDIUM/LOW. Include `blocked_reason` only on FAIL.
"""


def evaluation_timestamp_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def compute_post_generation_composite(evaluation: dict) -> tuple:
    def _score(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    numeric_scores = [
        _score(evaluation.get("helpfulness")),
        _score(evaluation.get("relevance")),
        _score(evaluation.get("level_of_detail")),
        _score(evaluation.get("groundedness")),
        _score(evaluation.get("completeness")),
        _score(evaluation.get("faithfulness")),
        _score(evaluation.get("content_safety")),
        _score((evaluation.get("output_safety") or {}).get("score")),
        _score((evaluation.get("output_governance") or {}).get("score")),
        _score((evaluation.get("output_compliance") or {}).get("score")),
    ]
    composite = round(sum(numeric_scores) / len(numeric_scores), 1)
    status = "PASS" if composite >= 7 else "FAIL"
    return composite, status


def build_agents(llm_config: Optional[dict] = None) -> dict:
    cfg = llm_config or LLM_CONFIG

    Response_Delivery_Agent = AssistantAgent(
        name="Response_Delivery_Agent",
        system_message=f"""
You are the Confidence-Aware Response Delivery agent.

## Pipeline role
1. Polish the grounded `final_answer` draft into the user-facing response.
2. Preserve conflict/confidence wording from upstream evidence checks.
3. Score the delivered answer with the post-generation rubric.
4. Hand control back to user_proxy when done.

{POST_GENERATION_EVALUATION_RUBRIC}

## Input
JSON with initial_question, analysis_type, final_answer (draft), optional
query_answer, llm_answer, evidence_strength, conflict_detected, conflicts.

## Output (strict JSON only)
{{
  "initial_question": "<question>",
  "analysis_type": "<unchanged>",
  "delivered_answer": "<polished user-facing answer>",
  "final_answer": "<same as delivered_answer>",
  "helpfulness": 9,
  "relevance": 9,
  "level_of_detail": 8,
  "groundedness": 10,
  "completeness": 8,
  "faithfulness": 9,
  "content_safety": 9,
  "output_safety": {{ "pii_leak": false, "toxic_output": false, "score": 9 }},
  "output_governance": {{ "restricted_data_exposed": false, "score": 9 }},
  "output_compliance": {{ "policy_violation": false, "score": 8 }},
  "composite_score": 8.7,
  "status": "PASS",
  "confidence": "HIGH",
  "evaluation_timestamp": "2026-05-28T14:30:00.000Z",
  "feedback": "<brief delivery note>",
  "next_agent": "user_proxy"
}}

On FAIL include blocked_reason and keep delivered_answer as the best honest answer
(or an insufficiency message). Never invent unsupported facts.
""",
        max_consecutive_auto_reply=4,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    return {"Response_Delivery_Agent": Response_Delivery_Agent}


ENTRY_AGENT = "Response_Delivery_Agent"
EXIT_AGENTS = {"user_proxy"}


def route(last_speaker_name: str, parsed_content: dict) -> Optional[str]:
    if last_speaker_name == "Response_Delivery_Agent":
        return "user_proxy"
    return None


from .chain_helpers import drive_chain as _drive_chain


class ResponseDeliveryRunner:
    """Polish grounded draft + post-generation evaluation for the user."""

    def run(
        self,
        initial_question: str = "",
        analysis_type: str = "",
        final_answer: str = "",
        **extra: Any,
    ) -> dict:
        payload = {
            "initial_question": initial_question,
            "analysis_type": analysis_type,
            "final_answer": final_answer or "",
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

        delivered = state.get("delivered_answer") or state.get("final_answer") or final_answer
        state["delivered_answer"] = delivered
        state["final_answer"] = delivered

        composite, status = compute_post_generation_composite(state)
        if "composite_score" not in state or state.get("composite_score") in (None, ""):
            state["composite_score"] = composite
        if "status" not in state or state.get("status") in (None, ""):
            state["status"] = status
        state.setdefault("next_agent", "user_proxy")
        state.setdefault("confidence", "MEDIUM")
        return state
