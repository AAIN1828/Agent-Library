"""Quin runtime configuration — env-driven DB + Azure OpenAI setup.

Matches the connection / schema model used by ``agent.py``:

* Azure OpenAI via ``AZURE_OPENAI_API_*`` + ``GPT4_LLM_MODEL_DEPLOYMENT_NAME``
  (``AZURE_OPENAI_DEPLOYMENT`` accepted as a preferred alias).
* Database via ``DATABASE_URL`` **or** discrete ``SQL_*`` pieces.
* Schema text via live SQLAlchemy introspection (default), or the optional
  ``SQL_METADATA`` env override (same escape hatch as ``agent.py``).

Helpers here are lazy where possible so importing this module alone does not
open a DB connection. ``agent.py`` still introspects at its own import time.
"""

from __future__ import annotations

import os
import urllib.parse
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


# Azure OpenAI — preferred AZURE_OPENAI_DEPLOYMENT, legacy GPT4_* fallback.
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
GPT4_LLM_MODEL_DEPLOYMENT_NAME = (
    AZURE_OPENAI_DEPLOYMENT
    or os.environ.get("GPT4_LLM_MODEL_DEPLOYMENT_NAME", "").strip()
)
AZURE_OPENAI_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE", "").strip()
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.environ.get(
    "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
).strip()

SQL_SCHEMA_NAME = _read_optional("SQL_SCHEMA") or "dbo"


def ensure_azure_openai_config() -> None:
    """Raise a clear error if any required Azure OpenAI env var is missing."""
    if not GPT4_LLM_MODEL_DEPLOYMENT_NAME:
        raise RuntimeError(
            "Required environment variable 'AZURE_OPENAI_DEPLOYMENT' "
            "(or GPT4_LLM_MODEL_DEPLOYMENT_NAME) is not set. "
            "Configure it before running Quin."
        )
    for name in ("AZURE_OPENAI_API_BASE", "AZURE_OPENAI_API_KEY"):
        if not globals()[name]:
            raise RuntimeError(
                f"Required environment variable {name!r} is not set. "
                "Configure it before running Quin."
            )


def build_connection_string() -> str:
    """
    Build a SQLAlchemy URL from ``DATABASE_URL`` or discrete ``SQL_*`` pieces.

    Same rules as ``agent.py``: prefer a full ``DATABASE_URL``; otherwise
    assemble ``mssql+pyodbc:///?odbc_connect=...`` from
    ``SQL_SERVER`` / ``SQL_DATABASE`` / ``SQL_USERNAME`` / ``SQL_PASSWORD``
    (``SQL_DRIVER`` optional, default ODBC Driver 18 for SQL Server).
    """
    database_url = _read_optional("DATABASE_URL")
    if database_url:
        return database_url

    server = _read_optional("SQL_SERVER")
    database = _read_optional("SQL_DATABASE")
    username = _read_optional("SQL_USERNAME")
    password = _read_optional("SQL_PASSWORD")
    driver = _read_optional("SQL_DRIVER") or "ODBC Driver 18 for SQL Server"

    missing = [
        name
        for name, value in [
            ("SQL_SERVER", server),
            ("SQL_DATABASE", database),
            ("SQL_USERNAME", username),
            ("SQL_PASSWORD", password),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            "No DATABASE_URL set, and the following SQL_* variables are "
            f"missing from .env: {', '.join(missing)}. Set either DATABASE_URL "
            "directly, or all of SQL_SERVER / SQL_DATABASE / SQL_USERNAME / SQL_PASSWORD."
        )

    odbc_connect = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password}"
    )
    return f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc_connect)}"


_engine: Any | None = None


def get_engine() -> Any:
    """Live SQLAlchemy engine — constructed on first use, not at import time."""
    global _engine
    if _engine is not None:
        return _engine
    from sqlalchemy import create_engine

    _engine = create_engine(build_connection_string())
    return _engine


def introspect_schema(engine: Any | None = None, schema_name: str | None = None) -> str:
    """Inspect the given schema and return a text description of all tables."""
    from sqlalchemy import inspect

    eng = engine if engine is not None else get_engine()
    schema = schema_name or SQL_SCHEMA_NAME
    inspector = inspect(eng)
    tables = inspector.get_table_names(schema=schema)

    schema_details = ""
    for table_name in tables:
        columns = inspector.get_columns(table_name, schema=schema)
        block = (
            f'schema = "{schema}", '
            f'table_name = "{schema}.{table_name}", '
            f"{table_name}_table = Table(\n"
        )
        for column in columns:
            col_name = column["name"]
            col_type = column["type"]
            primary_key = "primary_key=True" if column.get("primary_key", False) else ""
            nullable = "nullable=False" if not column["nullable"] else ""
            block += f'    Column("{col_name}", {col_type}, {primary_key} {nullable}),\n'
        block = block.rstrip(",\n")
        block += "\n)\n\n"
        schema_details += block

    return schema_details


def get_sql_metadata() -> str:
    """
    Schema text for SQL prompt templates.

    Prefer ``SQL_METADATA`` when set (verbatim override). Otherwise introspect
    the live database for ``SQL_SCHEMA`` (default ``dbo``).
    """
    override = os.environ.get("SQL_METADATA")
    if override is not None and override.strip():
        return override

    try:
        metadata = introspect_schema()
    except Exception as e:
        raise RuntimeError(
            f"Failed to introspect database schema for schema '{SQL_SCHEMA_NAME}': {e}. "
            "Check DATABASE_URL / SQL_* connection settings in .env, or set SQL_METADATA "
            "directly to bypass live introspection."
        ) from e

    if not metadata:
        raise RuntimeError(
            f"No tables found in schema '{SQL_SCHEMA_NAME}'. Check SQL_SCHEMA in .env, "
            "or set SQL_METADATA directly to bypass live introspection."
        )
    return metadata


# Back-compat alias used by older callers / tests.
def get_all_table_schemas() -> str:
    return get_sql_metadata()


__all__ = [
    "AZURE_OPENAI_DEPLOYMENT",
    "GPT4_LLM_MODEL_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_BASE",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "SQL_SCHEMA_NAME",
    "ensure_azure_openai_config",
    "build_connection_string",
    "get_engine",
    "introspect_schema",
    "get_sql_metadata",
    "get_all_table_schemas",
    "require_env",
]
