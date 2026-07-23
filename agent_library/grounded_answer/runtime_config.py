"""Env-driven Azure OpenAI config for Grounded Answer."""

from __future__ import annotations

import os


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            "Configure it before running Grounded Answer."
        )
    return value


GPT4_LLM_MODEL_DEPLOYMENT_NAME = os.environ.get("GPT4_LLM_MODEL_DEPLOYMENT_NAME", "").strip()
AZURE_OPENAI_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE", "").strip()
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()


def ensure_azure_openai_config() -> None:
    for name in (
        "GPT4_LLM_MODEL_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_BASE",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
    ):
        if not globals()[name]:
            raise RuntimeError(f"Required environment variable {name!r} is not set.")


__all__ = [
    "GPT4_LLM_MODEL_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_BASE",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "ensure_azure_openai_config",
    "require_env",
]
