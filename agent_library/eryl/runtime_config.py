"""Eryl runtime configuration — no ambient project-root globals.

Replaces the former ``API_unified`` / ``config`` / ``metadata`` / ``graph_config``
imports with explicit env-driven values and lazy resource construction.

Symbol resolution
-----------------
* ``GPT4_LLM_MODEL_DEPLOYMENT_NAME``, ``AZURE_OPENAI_API_*``, ``AZURE_SEARCH_SERVICE_ENDPOINT``
  — env-constructed (required when clients are first used).
* ``Embedding_Model``
  — env ``EMBEDDING_MODEL_DEPLOYMENT_NAME`` (deployment name string).
* ``azure_search_credential``
  — env-constructed ``AzureKeyCredential`` from ``AZURE_SEARCH_API_KEY``.
* ``Vector_Data`` / ``Eryl_Meta_data``
  — static reference data for routing/index hints. Loaded from
    ``ERYL_VECTOR_DATA_PATH`` or ``ERYL_VECTOR_DATA``. Not a live query.
* ``Localcontext_builder``
  — optional graph helper. Loaded from module named by
    ``ERYL_LOCAL_CONTEXT_MODULE`` (must export ``Localcontext_builder``).
    Clear error if graph selector is used without it.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            "Configure it before running Eryl (see agent_library/eryl/runtime_config.py)."
        )
    return value


def _read_optional(name: str) -> str:
    return os.environ.get(name, "").strip()


GPT4_LLM_MODEL_DEPLOYMENT_NAME = os.environ.get("GPT4_LLM_MODEL_DEPLOYMENT_NAME", "").strip()
AZURE_OPENAI_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE", "").strip()
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT", "").strip()


def ensure_azure_openai_config() -> None:
    for name in (
        "GPT4_LLM_MODEL_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_BASE",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
    ):
        if not globals()[name]:
            raise RuntimeError(
                f"Required environment variable {name!r} is not set. "
                "Configure it before running Eryl."
            )


def get_embedding_model() -> str:
    """Embedding deployment name — env-constructed string."""
    return require_env("EMBEDDING_MODEL_DEPLOYMENT_NAME")


def get_vector_data() -> str:
    """
    Static vector-index / routing metadata.

    Case: **bundled/injected static data** — not dynamically queried.
    """
    path = _read_optional("ERYL_VECTOR_DATA_PATH")
    if path:
        data_path = Path(path)
        if not data_path.is_file():
            raise RuntimeError(
                f"ERYL_VECTOR_DATA_PATH={path!r} does not exist or is not a file."
            )
        return data_path.read_text(encoding="utf-8")

    inline = _read_optional("ERYL_VECTOR_DATA")
    if inline:
        return inline

    raise RuntimeError(
        "Vector metadata is not configured. Set ERYL_VECTOR_DATA_PATH or "
        "ERYL_VECTOR_DATA before using Eryl_Meta_data / Vector_Data."
    )


def get_azure_search_credential() -> Any:
    """Live Azure Search credential — env-constructed from AZURE_SEARCH_API_KEY."""
    from azure.core.credentials import AzureKeyCredential

    return AzureKeyCredential(require_env("AZURE_SEARCH_API_KEY"))


def get_azure_openai_client() -> Any:
    """Live Azure OpenAI client — constructed on first use from env vars."""
    ensure_azure_openai_config()
    from openai import AzureOpenAI

    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_API_BASE,
    )


def get_search_client(*, index_name: str) -> Any:
    """Live Azure AI Search client — constructed on first use from env vars."""
    if not AZURE_SEARCH_SERVICE_ENDPOINT:
        raise RuntimeError(
            "Required environment variable 'AZURE_SEARCH_SERVICE_ENDPOINT' is not set."
        )
    from azure.search.documents import SearchClient

    return SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=index_name,
        credential=get_azure_search_credential(),
    )


def get_localcontext_builder() -> Any:
    """
    Optional graph-context builder.

    Case: **injected via module path** — set ERYL_LOCAL_CONTEXT_MODULE to a
    Python module that exports ``Localcontext_builder``. Not bundled; clear
    error if missing when graph retrieval is requested.
    """
    mod_name = _read_optional("ERYL_LOCAL_CONTEXT_MODULE")
    if not mod_name:
        raise RuntimeError(
            "Graph context requested but ERYL_LOCAL_CONTEXT_MODULE is not set. "
            "Set it to a module that exports Localcontext_builder, or use "
            "selector='vector' only."
        )
    mod = importlib.import_module(mod_name)
    builder = getattr(mod, "Localcontext_builder", None)
    if builder is None:
        raise RuntimeError(
            f"Module {mod_name!r} does not export Localcontext_builder."
        )
    return builder


# Module-level alias matching the old ``Embedding_Model`` name for LLM_CONFIG-era code.
# Resolved lazily so import succeeds before env is fully configured for embeddings.
class _EmbeddingModelProxy:
    def __str__(self) -> str:
        return get_embedding_model()

    def __repr__(self) -> str:
        return f"<EmbeddingModelProxy {get_embedding_model()!r}>"


Embedding_Model = _EmbeddingModelProxy()


__all__ = [
    "GPT4_LLM_MODEL_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_BASE",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_SEARCH_SERVICE_ENDPOINT",
    "Embedding_Model",
    "ensure_azure_openai_config",
    "get_azure_openai_client",
    "get_azure_search_credential",
    "get_embedding_model",
    "get_localcontext_builder",
    "get_search_client",
    "get_vector_data",
    "require_env",
]
