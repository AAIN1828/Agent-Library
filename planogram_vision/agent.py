"""
Planogram Vision Agent — standalone multi-agent shelf-image analysis chain.

Flow:
    user_proxy -> Routing_Agent
        -> Count_Agent   (numeric / quantity queries)
        |  Generic_Agent (descriptive / visual queries)
        -> Final_Answer_Agent -> user_proxy

Everything unrelated to planogram vision (parent pipeline routing, other
domain agents, DB logging, etc.) is omitted. This file runs end to end
on its own with one or more shelf-row images + a user query.

Required .env keys:
    AZURE_OPENAI_API_KEY=...
    AZURE_OPENAI_ENDPOINT=...           # or AZURE_OPENAI_API_BASE
    AZURE_OPENAI_DEPLOYMENT_NAME=...    # or GPT4_LLM_MODEL_DEPLOYMENT_NAME
    AZURE_OPENAI_API_VERSION=2024-02-15-preview   # optional
    LLM_TEMPERATURE=0                   # optional
    AGENT_WORK_DIR=Routing_File         # optional

Usage:
    python -m planogram_vision.agent --image row1.png row2.png --query "how many total products?"
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import mimetypes
import os
import re
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.agentchat.groupchat import GroupChat as GroupChatClass
from autogen.code_utils import content_str
from autogen.oai.client import OpenAIWrapper

load_dotenv()


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing from the environment."""


# ---------------------------------------------------------------------------
# AutoGen strips dots from Azure deployment names (gpt-5.4 -> gpt-54).
# Preserve the deployment name exactly as configured in Azure.
# ---------------------------------------------------------------------------
def _configure_azure_openai_preserve_dots(self, config, openai_config):
    openai_config["azure_deployment"] = openai_config.get(
        "azure_deployment", config.get("model")
    )
    openai_config["azure_endpoint"] = openai_config.get(
        "azure_endpoint", openai_config.pop("base_url", None)
    )
    if openai_config.get("azure_ad_token_provider") == "DEFAULT":
        import azure.identity

        openai_config["azure_ad_token_provider"] = (
            azure.identity.get_bearer_token_provider(
                azure.identity.DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
        )


OpenAIWrapper._configure_azure_openai = _configure_azure_openai_preserve_dots


def _patch_groupchat_image_preservation() -> None:
    """GroupChat.append mutates multimodal messages into text placeholders."""
    if getattr(GroupChatClass, "_image_preservation_patched", False):
        return

    def append(self, message: dict[str, Any], speaker) -> None:
        stored = copy.deepcopy(message)
        if stored["role"] != "function":
            stored["name"] = speaker.name
        if not isinstance(stored["content"], str) and not isinstance(
            stored["content"], list
        ):
            stored["content"] = str(stored["content"])
        stored["content"] = content_str(stored["content"])
        self.messages.append(stored)

    GroupChatClass.append = append  # type: ignore[method-assign]
    GroupChatClass._image_preservation_patched = True


_patch_groupchat_image_preservation()


# ---------------------------------------------------------------------------
# Configuration — from .env
# ---------------------------------------------------------------------------
def load_settings() -> dict[str, str]:
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or os.getenv("GPT4_LLM_MODEL_DEPLOYMENT_NAME")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or ""
    ).strip()
    endpoint = (
        os.getenv("AZURE_OPENAI_ENDPOINT")
        or os.getenv("AZURE_OPENAI_API_BASE")
        or ""
    ).strip()
    api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    temperature = os.getenv("LLM_TEMPERATURE", "0")
    work_dir = os.getenv("AGENT_WORK_DIR", "Routing_File")

    missing = [
        name
        for name, value in [
            ("AZURE_OPENAI_DEPLOYMENT_NAME (or GPT4_LLM_MODEL_DEPLOYMENT_NAME)", deployment),
            ("AZURE_OPENAI_ENDPOINT (or AZURE_OPENAI_API_BASE)", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
        ]
        if not value
    ]
    if missing:
        raise ConfigurationError(
            "Missing Azure OpenAI configuration. Set: " + ", ".join(missing)
        )

    return {
        "AZURE_OPENAI_DEPLOYMENT_NAME": deployment,
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_API_KEY": api_key,
        "AZURE_OPENAI_API_VERSION": api_version,
        "LLM_TEMPERATURE": temperature,
        "AGENT_WORK_DIR": work_dir,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def encode_image_b64(image_path: str) -> str:
    """Base64-encodes an image file for Azure OpenAI vision input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/jpeg"


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def json_safe_loads(text: str) -> dict[str, Any]:
    text = (text or "").strip()

    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))

    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Could not parse JSON from LLM response.")


def prepare_message_single(image_paths: list[str], user_query: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"User Query:\n{user_query}\n"
                "Shelf rows follow (Row 1, Row 2, ... in order):"
            ),
        }
    ]

    for img_path in image_paths:
        encoded_img = encode_image_b64(img_path)
        mime_type = image_mime_type(img_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded_img}"},
            }
        )

    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ROUTING_PROMPT = r"""
You are a Routing Agent.

Decide whether the user query requires numeric counting
or only visual / descriptive understanding.

Choose "Count_Agent" ONLY if the user explicitly asks for numbers,
totals or quantities, such as:
- how many
- count
- total number
- number of items

Otherwise choose "Generic_Agent".

Return STRICT JSON only:

{
  "next_agent": "Count_Agent" | "Generic_Agent",
  "reason": "<short explanation>"
}

No extra text.
""".strip()

COUNT_PROMPT = r"""
You are a Deterministic Retail Shelf Counting Agent.
Behave like a rule-based visual auditor, NOT a guesser.

IMPORTANT:
- MULTIPLE shelf-row images in ONE message (each corresponds to one row)

Rules:
1) Process all row images before responding.
2) Count ONLY if 50%+ of front face is visible.
3) No guessing / no inference outside the frame.
4) If user requests total items: return counts PER BRAND and compute totals internally.
5) If row is 60-80%+ truncated/blurry: counts must be {} and explain why.

Output STRICT JSON ONLY:

{
  "row_results": [
    {
      "row": <row_number>,
      "counts": { "<brand_or_product_name>": <integer> },
      "reasoning": "<brief explanation>"
    }
  ],
  "next_agent": "Final_Answer_Agent"
}
""".strip()

GENERIC_PROMPT = r"""
You are a Generic Shelf Understanding Agent.

CRITICAL:
- The user uploads ONLY ONE shelf image.
- Any multiple images you receive are CROPPED ROWS derived internally.
- DO NOT imply the user uploaded multiple images.
- Read visible brand names and product names directly from packaging labels.
- Do not guess product categories from colors or shapes alone.

Use row numbers (Row 1, Row 2, ...) for placement.

Return STRICT JSON ONLY:

{
  "Answer": "<clear natural language answer>",
  "reasoning": "<brief visual explanation referencing row numbers and visible brand names>",
  "next_agent": "Final_Answer_Agent"
}
""".strip()

FINAL_PROMPT = r"""
You are the Final Answer Agent.

INPUT:
- Original user query
- Aggregated results from previous agent

TASK:
- If user asks total in whole image, merge row counts.
- If user asks per row, format per row.
- If comparisons, rank/compare.

Return STRICT JSON only:

{
  "user_query":"<original query>",
  "final_answer":"<clear natural language answer>",
  "reasoning":"<short explanation>"
}
""".strip()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def create_llm_orchestrator() -> tuple[Any, Any, dict[str, Any]]:
    """Build a fresh AutoGen GroupChat for one planogram vision run."""
    s = load_settings()

    llm_config = {
        "config_list": [
            {
                "model": s["AZURE_OPENAI_DEPLOYMENT_NAME"],
                "azure_deployment": s["AZURE_OPENAI_DEPLOYMENT_NAME"],
                "api_type": "azure",
                "base_url": s["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
                "api_key": s["AZURE_OPENAI_API_KEY"],
                "api_version": s["AZURE_OPENAI_API_VERSION"],
            }
        ],
        "temperature": float(s["LLM_TEMPERATURE"]),
    }

    user_proxy = UserProxyAgent(
        name="user_proxy",
        system_message=(
            "You are the User Proxy.\n"
            "You will send shelf-row images along with the user query.\n"
            "Do not reason or modify content."
        ),
        code_execution_config={"work_dir": s["AGENT_WORK_DIR"], "use_docker": False},
        max_consecutive_auto_reply=5,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: False,
    )

    routing_agent = AssistantAgent(
        name="Routing_Agent",
        system_message=ROUTING_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    count_agent = AssistantAgent(
        name="Count_Agent",
        system_message=COUNT_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    generic_agent = AssistantAgent(
        name="Generic_Agent",
        system_message=GENERIC_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    final_agent = AssistantAgent(
        name="Final_Answer_Agent",
        system_message=FINAL_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    agents = {
        "user_proxy": user_proxy,
        "Routing_Agent": routing_agent,
        "Count_Agent": count_agent,
        "Generic_Agent": generic_agent,
        "Final_Answer_Agent": final_agent,
    }

    def state_transition(last_speaker, groupchat):
        if last_speaker is user_proxy:
            return routing_agent

        if not groupchat.messages:
            return None

        last_msg = groupchat.messages[-1].get("content", "")
        try:
            parsed = json_safe_loads(last_msg)
            next_agent_name = parsed.get("next_agent")
        except Exception:
            return None

        if next_agent_name == "Count_Agent":
            return count_agent
        if next_agent_name == "Generic_Agent":
            return generic_agent
        if next_agent_name == "Final_Answer_Agent":
            return final_agent

        return None

    groupchat = GroupChat(
        agents=[user_proxy, routing_agent, count_agent, generic_agent, final_agent],
        messages=[],
        max_round=10,
        speaker_selection_method=state_transition,
    )

    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    return user_proxy, manager, agents


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def run(
    user_query: str,
    image_paths: list[str],
    clear_history: bool = True,
) -> dict[str, Any]:
    user_proxy, manager, _agents = create_llm_orchestrator()
    message = prepare_message_single(image_paths, user_query)

    user_proxy.initiate_chat(
        manager,
        message=message,
        clear_history=clear_history,
    )

    final_raw = manager.groupchat.messages[-1]["content"]
    try:
        return json_safe_loads(final_raw)
    except Exception:
        return {
            "user_query": user_query,
            "final_answer": str(final_raw or ""),
            "reasoning": "Could not parse Final_Answer_Agent JSON; returning raw content.",
        }


def get_answer(
    user_query: str,
    image_paths: list[str] | None = None,
    *,
    clear_history: bool = True,
    **_extra: Any,
) -> dict[str, Any]:
    """Library-facing alias — matches eryl/quin ``get_answer`` naming."""
    paths = list(image_paths or _extra.get("images") or _extra.get("image_paths") or [])
    query = (
        (user_query or "").strip()
        or str(_extra.get("updated_question") or "").strip()
        or str(_extra.get("initial_question") or "").strip()
        or str(_extra.get("question") or "").strip()
    )
    if not query:
        return {
            "user_query": "",
            "final_answer": "",
            "reasoning": "No user query provided.",
        }
    if not paths:
        return {
            "user_query": query,
            "final_answer": "",
            "reasoning": "No image paths provided.",
        }
    return run(query, paths, clear_history=clear_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the planogram vision 5-agent chain standalone."
    )
    parser.add_argument("--image", nargs="+", required=True, help="One or more image paths.")
    parser.add_argument("--query", required=True, help="User query.")
    args = parser.parse_args()

    result = get_answer(args.query, args.image)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Reuse / orchestrator surface (mirrors eryl / quin)
# ---------------------------------------------------------------------------

ENTRY_AGENT = "Routing_Agent"
EXIT_AGENTS = {"user_proxy"}


def build_agents(llm_config: Any | None = None) -> dict[str, Any]:
    """
    Return a fresh planogram vision agent map for orchestrator use.

    Agents are built per call (vision chain needs a clean GroupChat per request).
    ``llm_config`` is accepted for API compatibility and ignored — config comes
    from env via load_settings().
    """
    del llm_config
    _user_proxy, _manager, agents = create_llm_orchestrator()
    return {
        "Routing_Agent": agents["Routing_Agent"],
        "Count_Agent": agents["Count_Agent"],
        "Generic_Agent": agents["Generic_Agent"],
        "Final_Answer_Agent": agents["Final_Answer_Agent"],
    }


def route(
    last_speaker_name: str,
    last_message: str = "",
    parsed_content: dict[str, Any] | None = None,
) -> str | None:
    """Mirror GroupChat state_transition as a name-based next-speaker function."""
    parsed = parsed_content or {}
    if not parsed and last_message:
        try:
            parsed = json_safe_loads(last_message)
        except Exception:
            parsed = {}

    if last_speaker_name == "Routing_Agent":
        next_name = parsed.get("next_agent", "Generic_Agent")
        if next_name in ("Count_Agent", "Generic_Agent"):
            return next_name
        return "Generic_Agent"
    if last_speaker_name in ("Count_Agent", "Generic_Agent"):
        return "Final_Answer_Agent"
    if last_speaker_name == "Final_Answer_Agent":
        return "user_proxy"
    return None


def drive_chain(
    *,
    agents: dict[str, Any] | None = None,
    entry_agent: str = ENTRY_AGENT,
    exit_agents: set[str] | None = None,
    route_fn: Callable[..., str | None] | None = None,
    initial_message: str | dict[str, Any],
) -> dict[str, Any]:
    """
    Lightweight wrapper: parse the handoff and call get_answer.

    Multimodal image content cannot be staged through pure JSON string handoffs
    cleanly, so drive_chain resolves query + image_paths and runs the full chain.
    """
    del agents, entry_agent, exit_agents, route_fn

    if isinstance(initial_message, str):
        try:
            payload = json.loads(initial_message)
        except json.JSONDecodeError:
            payload = {"user_query": initial_message}
    elif isinstance(initial_message, dict):
        payload = dict(initial_message)
    else:
        payload = {}

    query = (
        payload.get("user_query")
        or payload.get("updated_question")
        or payload.get("initial_question")
        or payload.get("question")
        or payload.get("query")
        or ""
    )
    paths = (
        payload.get("image_paths")
        or payload.get("images")
        or payload.get("image_path")
        or []
    )
    if isinstance(paths, str):
        paths = [paths]
    return get_answer(str(query), list(paths))


class PlanogramVisionChainRunner:
    """Public reuse entrypoint — runs the vision GroupChat via get_answer."""

    def run(
        self,
        initial_question: str | None = None,
        user_query: str | None = None,
        image_paths: list[str] | str | None = None,
        images: list[str] | str | None = None,
        updated_question: str | None = None,
        analysis_type: str | None = None,
        **_extra: Any,
    ) -> dict[str, Any]:
        query = (
            (user_query or "").strip()
            or (updated_question or "").strip()
            or (initial_question or "").strip()
        )
        paths: list[str] | str | None = image_paths if image_paths is not None else images
        if paths is None:
            paths = _extra.get("image_path")
        if isinstance(paths, str):
            paths = [paths]
        paths = list(paths or [])

        result = get_answer(query, paths)
        return {
            "user_query": result.get("user_query", query) or query,
            "final_answer": result.get("final_answer", "") or "",
            "reasoning": result.get("reasoning", "") or "",
            "analysis_type": analysis_type or "Planogram-vision",
            "question": initial_question or query,
            "updated_question": updated_question or initial_question or query,
            # Pass through extra keys (e.g. row_results) if present on parse fallbacks.
            **{
                k: v
                for k, v in result.items()
                if k not in {"user_query", "final_answer", "reasoning"}
            },
        }
