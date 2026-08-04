"""
Patch Generation Agent — standalone document update-patch agent.

Flow:
    pages_json + instruction
    -> generate_update_patch (LLM emits targeted old/new text replacements)
    -> filter no-op updates
    -> return {updates, ...}

The agent takes page-wise document JSON and a natural-language change
instruction, and returns a list of precise text patches (old -> new) that
can be applied by a downstream document editor. No multi-agent GroupChat
is required; the public surface matches eryl/quin for orchestrator reuse
(get_answer / PatchGenerationChainRunner.run / contract.json).
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

load_dotenv()


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing from the environment."""


# ---------------------------------------------------------------------------
# Configuration — fill these in via environment variables / .env
# ---------------------------------------------------------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
GPT4_LLM_MODEL_DEPLOYMENT_NAME = os.getenv("GPT4_LLM_MODEL_DEPLOYMENT_NAME")
AZURE_OPENAI_DEPLOYMENT = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip()
if AZURE_OPENAI_DEPLOYMENT and not GPT4_LLM_MODEL_DEPLOYMENT_NAME:
    GPT4_LLM_MODEL_DEPLOYMENT_NAME = AZURE_OPENAI_DEPLOYMENT

# Optional non-Azure fallback (OPENAI_API_KEY + PATCH_GEN_MODEL).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PATCH_GEN_MODEL = os.getenv("PATCH_GEN_MODEL", GPT4_LLM_MODEL_DEPLOYMENT_NAME or "gpt-4o")


def _build_llm_client() -> tuple[Any, str]:
    """Return (client, model_name) preferring Azure OpenAI when configured."""
    if all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, GPT4_LLM_MODEL_DEPLOYMENT_NAME]):
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_API_BASE,
        )
        return client, GPT4_LLM_MODEL_DEPLOYMENT_NAME

    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY), PATCH_GEN_MODEL

    raise ConfigurationError(
        "Missing LLM configuration. Set either "
        "(AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, GPT4_LLM_MODEL_DEPLOYMENT_NAME) "
        "or OPENAI_API_KEY in your .env file."
    )


llm, DEFAULT_MODEL = _build_llm_client()

# ---------------------------------------------------------------------------
# System prompt — document patch generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are the Patch Generation Agent. You receive a multi-page document as
page-wise JSON and a natural-language change instruction. Your job is to
produce precise text replacement patches that implement the instruction.

Rules:
1. Only change text that the instruction requires; leave everything else alone.
2. Each update must copy the EXACT existing substring into "old" (character-
   accurate match against the document text). "new" is the replacement text.
3. Prefer the smallest unique "old" span that still disambiguates the edit.
4. Do not invent pages, sections, or content that are not present.
5. If the instruction cannot be applied to any passage, return an empty
   "updates" list.
6. Never emit a no-op update where "old" equals "new".
7. When multiple independent edits are needed, return one object per edit.
8. Preserve original spelling, punctuation, and whitespace outside the edited span.

Output strictly as a JSON object (no markdown fences):
{
  "updates": [
    {
      "page": <page number or id if known, else null>,
      "section": "<optional section/heading if identifiable, else null>",
      "old": "<exact existing text to replace>",
      "new": "<replacement text>",
      "reason": "<brief why this edit satisfies the instruction>"
    }
  ],
  "summary": "<one-sentence description of the patch set>"
}
"""


def _clean_json(text: str) -> str:
    return (text or "").replace("```json", "").replace("```", "").strip()


def _normalize_pages_json(pages_json: Any) -> list[dict]:
    if pages_json is None:
        return []
    if isinstance(pages_json, str):
        try:
            pages_json = json.loads(_clean_json(pages_json))
        except json.JSONDecodeError:
            return [{"page": 1, "content": pages_json}]
    if isinstance(pages_json, dict):
        # Allow a single page object or a wrapper like {"pages": [...]}
        if "pages" in pages_json and isinstance(pages_json["pages"], list):
            pages_json = pages_json["pages"]
        else:
            pages_json = [pages_json]
    if not isinstance(pages_json, list):
        return []
    out: list[dict] = []
    for item in pages_json:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            out.append({"content": item})
    return out


def generate_update_patch(
    pages_json: list[dict] | dict | str | None,
    instruction: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> list[dict]:
    """
    Ask the LLM for old/new text patches implementing ``instruction``.

    Returns a list of update dicts (no-ops filtered out).
    """
    pages = _normalize_pages_json(pages_json)
    instruction = (instruction or "").strip()
    if not instruction:
        return []
    if not pages:
        return []

    user_prompt = f"""
DOCUMENT (PAGE-WISE JSON):
{json.dumps(pages, indent=2)}

CHANGE INSTRUCTIONS:
{instruction}
"""

    response = llm.chat.completions.create(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return []

    updates = payload.get("updates", []) if isinstance(payload, dict) else []
    if not isinstance(updates, list):
        return []

    cleaned: list[dict] = []
    for item in updates:
        if not isinstance(item, dict):
            continue
        old = item.get("old", "")
        new = item.get("new", "")
        if not isinstance(old, str) or not isinstance(new, str):
            continue
        if old == new:
            continue
        if not old and not new:
            continue
        cleaned.append(item)

    return cleaned


def get_answer(
    instruction: str,
    pages_json: list[dict] | dict | str | None = None,
    **_extra: Any,
) -> dict[str, Any]:
    """
    Run patch generation end to end.

    Accepts either a plain instruction (with pages_json passed separately)
    or orchestrator-style kwargs: initial_question / updated_question used as
    the instruction when ``instruction`` is empty.
    """
    instr = (instruction or "").strip()
    if not instr:
        instr = (
            (_extra.get("updated_question") or "")
            or (_extra.get("initial_question") or "")
            or (_extra.get("question") or "")
        ).strip()

    pages = pages_json
    if pages is None:
        pages = _extra.get("pages_json") or _extra.get("document") or _extra.get("pages")

    updates = generate_update_patch(pages, instr)
    return {
        "instruction": instr,
        "updates": updates,
        "update_count": len(updates),
        "summary": (
            f"Generated {len(updates)} text patch(es)."
            if updates
            else "No applicable patches generated."
        ),
    }


if __name__ == "__main__":
    sample_pages = [
        {
            "page": 1,
            "content": (
                "Acme Corp Employment Agreement\n"
                "Effective Date: January 1, 2024\n"
                "The Employee shall receive an annual salary of $80,000."
            ),
        },
        {
            "page": 2,
            "content": (
                "Notice Period: 30 days written notice is required to terminate "
                "this agreement."
            ),
        },
    ]
    sample_instruction = (
        "Update the salary to $95,000 and change the notice period from 30 days "
        "to 60 days."
    )
    result = get_answer(sample_instruction, pages_json=sample_pages)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Reuse / orchestrator surface (mirrors eryl / quin)
# ---------------------------------------------------------------------------

ENTRY_AGENT = "Patch_Generator"
EXIT_AGENTS = {"user_proxy"}


def build_agents(llm_config: Any | None = None) -> dict[str, Any]:
    """
    Patch generation is a single LLM call, not a multi-agent GroupChat.

    Returns an empty agent map for API compatibility with other packages.
    """
    del llm_config
    return {}


def route(last_speaker_name: str, *args: Any, **kwargs: Any) -> str | None:
    """No intermediate routing — single-stage agent."""
    del args, kwargs
    if last_speaker_name == "Patch_Generator":
        return "user_proxy"
    return None


def drive_chain(
    *,
    agents: dict[str, Any] | None = None,
    entry_agent: str = ENTRY_AGENT,
    exit_agents: set[str] | None = None,
    route_fn: Any = None,
    initial_message: str | dict[str, Any],
) -> dict[str, Any]:
    """
    Lightweight chain: parse the handoff payload and call generate_update_patch.
    """
    del agents, entry_agent, exit_agents, route_fn

    if isinstance(initial_message, str):
        try:
            payload = json.loads(_clean_json(initial_message))
        except json.JSONDecodeError:
            payload = {"instruction": initial_message}
    elif isinstance(initial_message, dict):
        payload = dict(initial_message)
    else:
        payload = {}

    instruction = (
        payload.get("instruction")
        or payload.get("updated_question")
        or payload.get("initial_question")
        or payload.get("question")
        or ""
    )
    pages = (
        payload.get("pages_json")
        or payload.get("document")
        or payload.get("pages")
    )
    return get_answer(str(instruction), pages_json=pages)


class PatchGenerationChainRunner:
    """Public reuse entrypoint — runs patch generation via get_answer."""

    def run(
        self,
        initial_question: str | None = None,
        instruction: str | None = None,
        pages_json: list[dict] | dict | str | None = None,
        document: list[dict] | dict | str | None = None,
        updated_question: str | None = None,
        analysis_type: str | None = None,
        **_extra: Any,
    ) -> dict[str, Any]:
        """
        Run patch generation end-to-end.

        Instruction resolution order:
            instruction -> updated_question -> initial_question

        Document resolution order:
            pages_json -> document -> _extra["pages"]
        """
        instr = (
            (instruction or "").strip()
            or (updated_question or "").strip()
            or (initial_question or "").strip()
        )
        pages = pages_json if pages_json is not None else document
        if pages is None:
            pages = _extra.get("pages")

        if not instr:
            return {
                "instruction": "",
                "updates": [],
                "update_count": 0,
                "summary": "No instruction provided.",
                "analysis_type": analysis_type or "Patch-generation",
            }

        result = get_answer(instr, pages_json=pages)
        result["analysis_type"] = analysis_type or "Patch-generation"
        result["updated_question"] = updated_question or initial_question or instr
        result["question"] = initial_question or instr
        return result
