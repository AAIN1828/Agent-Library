"""Quin runtime configuration — no ambient project-root globals.

Replaces the former ``API_unified`` / ``config`` / ``metadata`` imports with
explicit env-driven values and lazy resource construction.

Symbol resolution
-----------------
* ``GPT4_LLM_MODEL_DEPLOYMENT_NAME``, ``AZURE_OPENAI_API_*``
  — env-constructed (required; clear error if missing).
* ``all_table_schemas``
  — static reference data (SQL table documentation interpolated into
    system prompts). Not dynamically queried. Loaded from
    ``QUIN_TABLE_SCHEMAS_PATH`` (file) or ``QUIN_TABLE_SCHEMAS`` (inline
    string). Required when ``build_agents()`` runs.
* ``engine``
  — live SQLAlchemy engine. Built from ``DATABASE_URL`` on first use
    (not at import time), so importing the package does not open a DB
    connection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            "Configure it before running Quin (see agent_library/quin/runtime_config.py)."
        )
    return value


def _read_optional(name: str) -> str:
    return os.environ.get(name, "").strip()


# Azure OpenAI — required for LLM_CONFIG at import when agent.py builds it.
GPT4_LLM_MODEL_DEPLOYMENT_NAME = os.environ.get("GPT4_LLM_MODEL_DEPLOYMENT_NAME", "").strip()
AZURE_OPENAI_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE", "").strip()
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()


def ensure_azure_openai_config() -> None:
    """Raise a clear error if any Azure OpenAI env var is missing."""
    for name in (
        "GPT4_LLM_MODEL_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_BASE",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
    ):
        if not globals()[name]:
            raise RuntimeError(
                f"Required environment variable {name!r} is not set. "
                "Configure it before running Quin."
            )


def get_all_table_schemas() -> str:
    """
    Static table-schema documentation for SQL prompt templates.

    Case: **bundled/injected static data** — not a live DB query. Operators
    must supply the schema text via env (file path preferred).
    """
    path = _read_optional("QUIN_TABLE_SCHEMAS_PATH")
    if path:
        schema_path = Path(path)
        if not schema_path.is_file():
            raise RuntimeError(
                f"QUIN_TABLE_SCHEMAS_PATH={path!r} does not exist or is not a file."
            )
        return schema_path.read_text(encoding="utf-8")

    inline = _read_optional("QUIN_TABLE_SCHEMAS")
    if inline:
        return inline

    raise RuntimeError(
        "Table schema metadata is not configured. Set QUIN_TABLE_SCHEMAS_PATH "
        "(path to a text/SQL-schema file) or QUIN_TABLE_SCHEMAS (inline string) "
        "before calling build_agents()."
    )


_engine: Any | None = None


def get_engine() -> Any:
    """
    Live SQLAlchemy engine constructed from ``DATABASE_URL``.

    Case: **env-constructed live resource** — not created at import time.
    """
    global _engine
    if _engine is not None:
        return _engine
    url = require_env("DATABASE_URL")
    from sqlalchemy import create_engine

    _engine = create_engine(url)
    return _engine


__all__ = [
    "GPT4_LLM_MODEL_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_BASE",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "ensure_azure_openai_config",
    "get_all_table_schemas",
    "get_engine",
    "require_env",
]
