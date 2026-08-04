"""Env-driven Azure OpenAI config for Document Generation (PDF in-place updater)."""

from __future__ import annotations

import os


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            "Configure it before running Document Generation "
            "(see agent_library/document_generation/runtime_config.py)."
        )
    return value


def _read_optional(name: str) -> str:
    return os.environ.get(name, "").strip()


AZURE_OPENAI_DEPLOYMENT = _read_optional("AZURE_OPENAI_DEPLOYMENT")
GPT4_LLM_MODEL_DEPLOYMENT_NAME = (
    AZURE_OPENAI_DEPLOYMENT
    or _read_optional("GPT4_LLM_MODEL_DEPLOYMENT_NAME")
    or _read_optional("AZURE_OPENAI_DEPLOYMENT_NAME")
    or _read_optional("AZURE_OPENAI_CHAT_DEPLOYMENT")
)
AZURE_OPENAI_API_BASE = (
    _read_optional("AZURE_OPENAI_API_BASE") or _read_optional("AZURE_OPENAI_ENDPOINT")
)
AZURE_OPENAI_API_KEY = _read_optional("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = (
    _read_optional("AZURE_OPENAI_API_VERSION") or "2024-02-15-preview"
)


def ensure_azure_openai_config() -> None:
    if not GPT4_LLM_MODEL_DEPLOYMENT_NAME:
        raise RuntimeError(
            "Required environment variable 'AZURE_OPENAI_DEPLOYMENT' "
            "(or GPT4_LLM_MODEL_DEPLOYMENT_NAME) is not set."
        )
    if not AZURE_OPENAI_API_BASE:
        raise RuntimeError(
            "Required environment variable 'AZURE_OPENAI_API_BASE' "
            "(or AZURE_OPENAI_ENDPOINT) is not set."
        )
    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError(
            "Required environment variable 'AZURE_OPENAI_API_KEY' is not set."
        )


__all__ = [
    "AZURE_OPENAI_DEPLOYMENT",
    "GPT4_LLM_MODEL_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_BASE",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "ensure_azure_openai_config",
    "require_env",
]
