"""Env-driven Azure OpenAI + optional SQL schema metadata for routing."""

from __future__ import annotations

import os
from pathlib import Path


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            "Configure it before running Intent Router."
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


def get_table_schemas() -> str:
    path = os.environ.get("QUIN_TABLE_SCHEMAS_PATH", "").strip()
    if path:
        schema_path = Path(path)
        if not schema_path.is_file():
            raise RuntimeError(f"QUIN_TABLE_SCHEMAS_PATH={path!r} is not a file.")
        return schema_path.read_text(encoding="utf-8")
    inline = os.environ.get("QUIN_TABLE_SCHEMAS", "").strip()
    if inline:
        return inline
    return "(No table schemas configured. Set QUIN_TABLE_SCHEMAS_PATH or QUIN_TABLE_SCHEMAS.)"


__all__ = [
    "GPT4_LLM_MODEL_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_BASE",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "ensure_azure_openai_config",
    "get_table_schemas",
    "require_env",
]
