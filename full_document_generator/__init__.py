"""
Full Document Generator — from-scratch PDF document generation package.
"""

from __future__ import annotations

from typing import Any

from .agent import FullDocumentGeneratorRunner

__all__ = [
    "FullDocumentGeneratorRunner",
    "create_full_document",
    "generate_document_content",
    "render_pdf_from_content",
]

__agent_name__ = "full_document_generator"
__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    if name in {
        "create_full_document",
        "generate_document_content",
        "render_pdf_from_content",
    }:
        from . import chain_helpers

        return getattr(chain_helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
