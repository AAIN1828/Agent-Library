"""
Intake Gateway — question intake + Responsible AI safety gate.

Split out of the unified AutoGen pipeline (Multiple DB SEQ / AURIX). Owns:

    azure_safety_check (deterministic) -> Responsible_AI_Agent

Passes only when RAI clears the question; otherwise returns a blocked response
for the parent orchestrator to surface to the user.
"""

from __future__ import annotations

import json
import re
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


def _extract_user_question(message: str) -> str:
    text = message.strip()
    if text.startswith('"question":'):
        return text.split('"question":', 1)[1].strip().strip('"').strip()
    try:
        payload = json.loads(text.replace("```json", "").replace("```", "").strip())
        if isinstance(payload, dict):
            return payload.get("question") or payload.get("initial_question") or text
    except json.JSONDecodeError:
        pass
    return text


def azure_safety_check(question: str) -> dict:
    """Deterministic Azure Content Safety-style pre-check (not an LLM agent)."""
    text = (question or "").lower()
    flags: list[str] = []

    toxicity_patterns = [
        r"\b(kill|hate|violence|attack|abuse|harass)\b",
        r"\b(bomb|weapon|murder|terror)\b",
    ]
    toxicity_detected = any(re.search(pattern, text) for pattern in toxicity_patterns)
    if toxicity_detected:
        flags.append("toxicity")

    pii_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"\b(?:ssn|social security number)\b",
    ]
    pii_detected = any(re.search(pattern, question or "", re.IGNORECASE) for pattern in pii_patterns)
    if pii_detected:
        flags.append("pii")

    injection_patterns = [
        r"ignore (all )?(previous|prior|above) instructions",
        r"disregard (the )?(system|developer) (prompt|message|instructions)",
        r"you are now (?:in )?(?:dan|developer|admin) mode",
        r"reveal (the )?(system prompt|hidden instructions|api key|secret)",
        r"jailbreak",
        r"prompt injection",
    ]
    prompt_injection = any(re.search(pattern, text) for pattern in injection_patterns)
    if prompt_injection:
        flags.append("prompt_injection")

    if not flags:
        risk_score = 10
    elif toxicity_detected or prompt_injection:
        risk_score = 2
    elif pii_detected:
        risk_score = 6
    else:
        risk_score = 5

    return {
        "toxicity_detected": toxicity_detected,
        "pii_detected": pii_detected,
        "prompt_injection": prompt_injection,
        "flags": flags,
        "risk_score": risk_score,
        "azure_safety_result": {
            "toxicity": "unsafe" if toxicity_detected else "safe",
            "jailbreak": bool(prompt_injection),
            "self_harm": False,
            "pii": "detected" if pii_detected else "none",
        },
    }


def build_agents(llm_config: Optional[dict] = None) -> dict:
    cfg = llm_config or LLM_CONFIG

    Responsible_AI_Agent = AssistantAgent(
        name="Responsible_AI_Agent",
        system_message="""
# **Responsible AI Agent - Governance & Safety Gate**

You run **after** the deterministic Azure Content Safety pre-check and **before** the routing agent.

## **Input**
You will receive:
1. The user's original question
2. A JSON blob from `azure_safety_check` with: toxicity_detected, pii_detected, prompt_injection, flags, risk_score, azure_safety_result

## **Your Tasks**
Evaluate the request for:
- **content_safety** — confirm or escalate Azure safety signals
- **data_governance** — block bulk exfiltration, unauthorized cross-tenant access, or requests for secrets/credentials
- **business_rules** — block clearly out-of-scope destructive actions

## **Pass / Fail Rules**
- Set `rai_check_passed: true` and `next_agent: "routing_agent"` only when the question is safe and in scope.
- Set `rai_check_passed: false` and `next_agent: "user_proxy"` when any high-severity issue is present.

## **risk_score** (0–10, higher = safer)
- **9–10**: safe / benign in-scope questions
- **5–8**: suspicious or policy-edge requests needing caution
- **1–4**: blocked / dangerous requests

## **Azure safety payload**
- Copy `azure_safety_result` from the pre-check input when provided.
- Never return an empty azure_safety_result object.

Always return strict JSON only. On pass include next_agent "routing_agent".
On fail include blocked_reason and next_agent "user_proxy".
""",
        max_consecutive_auto_reply=3,
        llm_config=cfg,
        human_input_mode="NEVER",
    )

    return {"Responsible_AI_Agent": Responsible_AI_Agent}


ENTRY_AGENT = "Responsible_AI_Agent"
EXIT_AGENTS = {"user_proxy", "routing_agent"}


def route(last_speaker_name: str, parsed_content: dict) -> Optional[str]:
    if last_speaker_name == "Responsible_AI_Agent":
        if parsed_content.get("rai_check_passed") and parsed_content.get("next_agent") == "routing_agent":
            return "routing_agent"
        return "user_proxy"
    return None


from .chain_helpers import drive_chain as _drive_chain


class IntakeGatewayRunner:
    """Single entrypoint: safety pre-check + Responsible AI gate."""

    def run(self, question: str, **_: Any) -> dict:
        q = _extract_user_question(str(question or ""))
        safety = azure_safety_check(q)
        agents = build_agents()
        initial = json.dumps(
            {
                "initial_question": q,
                "azure_content_safety": safety,
                "evaluation_timestamp": evaluation_timestamp_iso(),
            },
            indent=2,
        )
        state = _drive_chain(
            agents=agents,
            entry_agent=ENTRY_AGENT,
            exit_agents=EXIT_AGENTS,
            route_fn=route,
            initial_message=initial,
        )
        if not state:
            state = {
                "initial_question": q,
                "rai_check_passed": False,
                "blocked_reason": "Empty RAI response",
            }
        state.setdefault("initial_question", q)
        state["azure_content_safety"] = safety
        state["question"] = q
        return state
