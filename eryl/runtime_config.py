"""Eryl runtime configuration — env-driven Azure OpenAI + AI Search setup.

Matches the connection model used by ``agent.py``:

* Azure OpenAI via ``AZURE_OPENAI_API_*`` + ``GPT4_LLM_MODEL_DEPLOYMENT_NAME``
  (``AZURE_OPENAI_DEPLOYMENT`` accepted as a preferred alias).
* Embeddings via ``EMBEDDING_MODEL_DEPLOYMENT_NAME``.
* Azure AI Search via ``AZURE_SEARCH_SERVICE_ENDPOINT`` (or legacy
  ``AZURE_SEARCH_ENDPOINT``), ``AZURE_SEARCH_API_KEY``, optional index /
  semantic-config overrides.

Clients are constructed lazily on first use so importing this module alone does
not require live credentials. ``agent.py`` still builds clients at import time.
"""

from __future__ import annotations

import os
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


# Azure OpenAI — preferred AZURE_OPENAI_DEPLOYMENT, legacy GPT4_* fallback.
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
GPT4_LLM_MODEL_DEPLOYMENT_NAME = (
    AZURE_OPENAI_DEPLOYMENT
    or os.environ.get("GPT4_LLM_MODEL_DEPLOYMENT_NAME", "").strip()
)
AZURE_OPENAI_API_BASE = (
    _read_optional("AZURE_OPENAI_API_BASE")
    or _read_optional("AZURE_OPENAI_ENDPOINT")
)
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.environ.get(
    "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
).strip()

AZURE_SEARCH_SERVICE_ENDPOINT = (
    _read_optional("AZURE_SEARCH_SERVICE_ENDPOINT")
    or _read_optional("AZURE_SEARCH_ENDPOINT")
)


def ensure_azure_openai_config() -> None:
    """Raise a clear error if any required Azure OpenAI env var is missing."""
    if not GPT4_LLM_MODEL_DEPLOYMENT_NAME:
        raise RuntimeError(
            "Required environment variable 'AZURE_OPENAI_DEPLOYMENT' "
            "(or GPT4_LLM_MODEL_DEPLOYMENT_NAME) is not set. "
            "Configure it before running Eryl."
        )
    for name in ("AZURE_OPENAI_API_BASE", "AZURE_OPENAI_API_KEY"):
        if not globals()[name]:
            raise RuntimeError(
                f"Required environment variable {name!r} is not set. "
                "Configure it before running Eryl."
            )


def get_embedding_model() -> str:
    """Embedding deployment name — env-constructed string."""
    return require_env("EMBEDDING_MODEL_DEPLOYMENT_NAME")


def get_index_name() -> str:
    """Azure AI Search index name — env with safe default matching agent.py."""
    return _read_optional("AZURE_SEARCH_INDEX_NAME") or "dupont_email_demo"


def get_semantic_config_name() -> str:
    """Azure AI Search semantic configuration — env with safe default."""
    return (
        _read_optional("AZURE_SEARCH_SEMANTIC_CONFIG")
        or _read_optional("AZURE_SEARCH_SEMANTIC_CONFIG_NAME")
        or "my-semantic-config"
    )


def get_vector_field_name() -> str:
    """Vector field name used by extract_context — default ``embedding``."""
    return _read_optional("AZURE_SEARCH_VECTOR_FIELD_NAME") or "embedding"


def get_document_id_field_name() -> str:
    """Document key field — default ``documentId`` (matches agent.py select)."""
    return _read_optional("AZURE_SEARCH_DOCUMENT_ID_FIELD_NAME") or "documentId"


def get_content_field_name() -> str:
    """Content field returned from search — default ``content``."""
    return _read_optional("AZURE_SEARCH_CONTENT_FIELD_NAME") or "content"


def get_azure_search_credential() -> Any:
    """Live Azure Search credential — env-constructed from AZURE_SEARCH_API_KEY."""
    from azure.core.credentials import AzureKeyCredential

    return AzureKeyCredential(require_env("AZURE_SEARCH_API_KEY"))


_embedding_client: Any | None = None
_search_client: Any | None = None


def get_azure_openai_client() -> Any:
    """Live Azure OpenAI client — constructed on first use from env vars."""
    global _embedding_client
    if _embedding_client is not None:
        return _embedding_client
    ensure_azure_openai_config()
    from openai import AzureOpenAI

    _embedding_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_API_BASE,
    )
    return _embedding_client


def get_search_client(*, index_name: str | None = None) -> Any:
    """Live Azure AI Search client — constructed on first use from env vars."""
    global _search_client
    if _search_client is not None:
        return _search_client
    if not AZURE_SEARCH_SERVICE_ENDPOINT:
        raise RuntimeError(
            "Required environment variable 'AZURE_SEARCH_SERVICE_ENDPOINT' "
            "(or AZURE_SEARCH_ENDPOINT) is not set."
        )
    from azure.search.documents import SearchClient

    _search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=index_name or get_index_name(),
        credential=get_azure_search_credential(),
    )
    return _search_client


__all__ = [
    "AZURE_OPENAI_DEPLOYMENT",
    "GPT4_LLM_MODEL_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_BASE",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_SEARCH_SERVICE_ENDPOINT",
    "ensure_azure_openai_config",
    "get_azure_openai_client",
    "get_azure_search_credential",
    "get_content_field_name",
    "get_document_id_field_name",
    "get_embedding_model",
    "get_index_name",
    "get_search_client",
    "get_semantic_config_name",
    "get_vector_field_name",
    "require_env",
]
