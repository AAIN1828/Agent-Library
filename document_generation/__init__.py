"""
Document Generation — instruction-driven in-place PDF text updater package.
"""

from __future__ import annotations

from typing import Any

from .agent import DocumentGenerationRunner

__all__ = [
    "DocumentGenerationRunner",
    "update_document",
    "extract_pdf_pages",
    "generate_update_patch",
    "apply_updates_inplace",
]

__agent_name__ = "document_generation"
__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    if name in {
        "update_document",
        "extract_pdf_pages",
        "generate_update_patch",
        "apply_updates_inplace",
    }:
        from . import chain_helpers

        return getattr(chain_helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
