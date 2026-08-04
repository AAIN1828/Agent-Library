"""
Executive Brief Generator — PDF → one-page structured executive summary.

Flow:
    extract_pdf_text (plain Python / PyMuPDF)
    -> Brief_Writer (AutoGen AssistantAgent)
    -> structured JSON {title, key_points, executive_summary, word_count, ...}

No critic loop: word-count and grounding constraints live in the Brief_Writer
prompt plus post-parse validation in _normalize_brief / shape_output.

Required .env keys:
    AZURE_OPENAI_API_KEY=...
    AZURE_OPENAI_ENDPOINT=...           # or AZURE_OPENAI_API_BASE
    GPT4_LLM_MODEL_DEPLOYMENT_NAME=...  # or AZURE_OPENAI_DEPLOYMENT_NAME
    AZURE_OPENAI_API_VERSION=2024-02-15-preview   # optional
    LLM_TEMPERATURE=0.2                 # optional
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from autogen.oai.client import OpenAIWrapper

load_dotenv()


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing from the environment."""


# ---------------------------------------------------------------------------
# AutoGen strips dots from Azure deployment names (gpt-5.4 -> gpt-54).
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


_BRIEF_WRITER_PROMPT = """You are Brief_Writer, an executive-communications specialist.

You receive extracted PDF text (and an optional focus hint). Write a ONE-PAGE
executive brief grounded ONLY in that text. Do not invent facts.

Return ONLY a JSON object (no markdown fences) with this exact shape:
{
  "title": "string — concise brief title",
  "key_points": ["string", "..."]  // 3 to 7 bullets, each one clear finding,
  "executive_summary": "string — narrative brief body",
  "word_count": 0  // integer word count of executive_summary only
}

Hard constraints:
- executive_summary MUST be between 250 and 400 words (inclusive).
- key_points MUST have 3–7 items.
- Every claim must be supportable from the provided PDF text.
- If the PDF text is empty or unusable, still return the JSON shape with
  title "Unable to summarize", key_points describing the problem, and a short
  executive_summary explaining that no usable text was extracted.
"""

_TARGET_MIN_WORDS = 250
_TARGET_MAX_WORDS = 400


def extract_pdf_text(pdf_path: str) -> dict[str, Any]:
    """Extract plain text from a PDF using PyMuPDF (fitz). Not an LLM call."""
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (pymupdf) is required for PDF extraction. "
            "Install it with: pip install pymupdf"
        ) from exc

    doc = fitz.open(path)
    try:
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text("text") or "")
        text = "\n".join(pages).strip()
        return {
            "pdf_path": str(path),
            "page_count": int(doc.page_count),
            "char_count": len(text),
            "text": text,
        }
    finally:
        doc.close()


def _clean_json(text: str) -> str:
    return (text or "").replace("```json", "").replace("```", "").strip()


def _parse_brief_json(content: str) -> dict[str, Any]:
    cleaned = _clean_json(content)
    if not cleaned:
        return {}
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _normalize_brief(
    parsed: dict[str, Any],
    *,
    source_pdf: str,
    page_count: int,
) -> dict[str, Any]:
    title = str(parsed.get("title") or "").strip() or "Executive Brief"
    key_points_raw = parsed.get("key_points") or []
    if not isinstance(key_points_raw, list):
        key_points_raw = [str(key_points_raw)]
    key_points = [str(p).strip() for p in key_points_raw if str(p).strip()]
    summary = str(parsed.get("executive_summary") or "").strip()
    count = int(parsed.get("word_count") or 0)
    measured = _word_count(summary)
    if count <= 0 or abs(count - measured) > 25:
        count = measured
    return {
        "title": title,
        "key_points": key_points,
        "executive_summary": summary,
        "word_count": count,
        "page_count": page_count,
        "source_pdf": source_pdf,
        "within_length_bounds": _TARGET_MIN_WORDS <= count <= _TARGET_MAX_WORDS,
    }


def _build_llm_config() -> dict[str, Any]:
    deployment = (
        os.getenv("GPT4_LLM_MODEL_DEPLOYMENT_NAME")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
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
    temperature = os.getenv("LLM_TEMPERATURE", "0.2")

    if not all([deployment, endpoint, api_key]):
        raise ConfigurationError(
            "Missing Azure OpenAI configuration. Set AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_ENDPOINT (or AZURE_OPENAI_API_BASE), and "
            "GPT4_LLM_MODEL_DEPLOYMENT_NAME (or AZURE_OPENAI_DEPLOYMENT_NAME)."
        )

    return {
        "config_list": [
            {
                "model": deployment,
                "azure_deployment": deployment,
                "api_type": "azure",
                "base_url": endpoint.rstrip("/"),
                "api_key": api_key,
                "api_version": api_version,
            }
        ],
        "temperature": float(temperature),
    }


def _brief_complete(msg: dict[str, Any] | str | None) -> bool:
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = msg or ""
    parsed = _parse_brief_json(str(content))
    return bool(parsed.get("executive_summary") or parsed.get("title"))


def create_orchestrator() -> dict[str, Any]:
    """Build a fresh user_proxy + Brief_Writer GroupChat (lazy; needs credentials)."""
    from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent

    llm_config = _build_llm_config()

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config=False,
        is_termination_msg=_brief_complete,
    )

    brief_writer = AssistantAgent(
        name="Brief_Writer",
        system_message=_BRIEF_WRITER_PROMPT,
        llm_config=llm_config,
        max_consecutive_auto_reply=3,
        description="Writes a one-page structured executive brief from PDF text.",
        is_termination_msg=_brief_complete,
    )

    def state_transition(last_speaker, groupchat):  # noqa: ANN001
        if last_speaker is user_proxy:
            return brief_writer
        if last_speaker is brief_writer:
            return user_proxy
        return None

    groupchat = GroupChat(
        agents=[user_proxy, brief_writer],
        messages=[],
        max_round=6,
        speaker_selection_method=state_transition,
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    return {
        "user_proxy": user_proxy,
        "Brief_Writer": brief_writer,
        "manager": manager,
        "groupchat": groupchat,
    }


def get_answer(
    pdf_path: str | None = None,
    *,
    pdf_text: str | None = None,
    focus: str | None = None,
    page_count: int | None = None,
    **_extra: Any,
) -> dict[str, Any]:
    """Run PDF extraction (if needed) + Brief_Writer; return structured brief."""
    source_pdf = ""
    pages = 0
    text = (pdf_text or _extra.get("text") or "").strip()

    if not text:
        path = pdf_path or _extra.get("document_path") or _extra.get("source_pdf")
        if not path:
            raise ValueError("Provide pdf_path or pdf_text")
        extracted = extract_pdf_text(str(path))
        text = str(extracted.get("text") or "")
        source_pdf = str(extracted.get("pdf_path") or path)
        pages = int(extracted.get("page_count") or 0)
    else:
        source_pdf = str(pdf_path or _extra.get("source_pdf") or "").strip() or "inline_text"
        pages = int(page_count or (1 if text else 0))

    focus_hint = (focus or _extra.get("updated_question") or _extra.get("initial_question") or "")
    payload = {
        "pdf_path": source_pdf,
        "page_count": pages,
        "focus": str(focus_hint).strip() or None,
        "pdf_text": text,
    }

    orch = create_orchestrator()
    user_proxy = orch["user_proxy"]
    manager = orch["manager"]
    user_proxy.initiate_chat(
        manager,
        message=json.dumps(payload),
        clear_history=True,
    )

    history: list[Any] = []
    brief_agent = orch["Brief_Writer"]
    chat_map = getattr(user_proxy, "chat_messages", {}) or {}
    for key, msgs in chat_map.items():
        if key is brief_agent or getattr(key, "name", None) == "Brief_Writer":
            history = list(msgs or [])
            break
    if not history:
        history = list(orch["groupchat"].messages or [])

    parsed: dict[str, Any] = {}
    for msg in reversed(history):
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg or "")
        candidate = _parse_brief_json(str(content))
        if candidate.get("executive_summary") or candidate.get("title"):
            parsed = candidate
            break

    return _normalize_brief(parsed, source_pdf=source_pdf, page_count=pages)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m executive_brief_generator.agent <pdf_path> [focus]")
        raise SystemExit(2)
    pdf = sys.argv[1]
    focus_arg = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(get_answer(pdf, focus=focus_arg), indent=2))


# ---------------------------------------------------------------------------
# Reuse / orchestrator surface (mirrors eryl / quin)
# ---------------------------------------------------------------------------

ENTRY_AGENT = "Brief_Writer"
EXIT_AGENTS = {"user_proxy"}


def build_agents(llm_config: Any | None = None) -> dict[str, Any]:
    """Return a fresh Brief_Writer agent map for orchestrator use."""
    del llm_config
    orch = create_orchestrator()
    return {"Brief_Writer": orch["Brief_Writer"]}


def route(
    last_speaker_name: str,
    last_message: str = "",
    parsed_content: dict[str, Any] | None = None,
) -> str | None:
    del last_message, parsed_content
    if last_speaker_name == "Brief_Writer":
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
    """Parse the handoff payload and run get_answer."""
    del agents, entry_agent, exit_agents, route_fn

    if isinstance(initial_message, str):
        try:
            payload = json.loads(_clean_json(initial_message))
        except json.JSONDecodeError:
            # Treat bare path as pdf_path.
            if initial_message.strip().lower().endswith(".pdf"):
                payload = {"pdf_path": initial_message.strip()}
            else:
                payload = {"pdf_text": initial_message}
    elif isinstance(initial_message, dict):
        payload = dict(initial_message)
    else:
        payload = {}

    return get_answer(
        pdf_path=payload.get("pdf_path") or payload.get("document_path"),
        pdf_text=payload.get("pdf_text") or payload.get("text"),
        focus=payload.get("focus")
        or payload.get("updated_question")
        or payload.get("initial_question"),
        page_count=payload.get("page_count"),
    )


class ExecutiveBriefChainRunner:
    """Public reuse entrypoint — runs extract + Brief_Writer via get_answer."""

    def run(
        self,
        initial_question: str | None = None,
        pdf_path: str | None = None,
        pdf_text: str | None = None,
        focus: str | None = None,
        document_path: str | None = None,
        updated_question: str | None = None,
        analysis_type: str | None = None,
        page_count: int | None = None,
        **_extra: Any,
    ) -> dict[str, Any]:
        path = pdf_path or document_path or _extra.get("source_pdf")
        focus_hint = (
            (focus or "").strip()
            or (updated_question or "").strip()
            or (initial_question or "").strip()
        )
        result = get_answer(
            pdf_path=path,
            pdf_text=pdf_text or _extra.get("text"),
            focus=focus_hint or None,
            page_count=page_count,
        )
        result["analysis_type"] = analysis_type or "Executive-brief"
        result["question"] = initial_question or focus_hint or ""
        result["updated_question"] = updated_question or initial_question or focus_hint or ""
        return result
