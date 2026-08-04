"""
Executive Brief Generator — PDF → one-page structured executive summary package.

``agent.py`` owns PDF extraction (``extract_pdf_text``), the Brief_Writer
GroupChat (``get_answer``), env configuration, orchestrator helpers, and
``ExecutiveBriefChainRunner``. Catalog metadata lives in ``contract.json``.

``get_answer`` / ``build_agents`` are lazy so importing the package does not
force credential checks until those symbols are used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ExecutiveBriefChainRunner",
    "get_answer",
    "extract_pdf_text",
    "build_agents",
    "route",
    "drive_chain",
    "ENTRY_AGENT",
    "EXIT_AGENTS",
]

__agent_name__ = "executive_brief_generator"
__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    if name in {
        "ExecutiveBriefChainRunner",
        "get_answer",
        "extract_pdf_text",
        "build_agents",
        "route",
        "drive_chain",
        "ENTRY_AGENT",
        "EXIT_AGENTS",
    }:
        from . import agent as executive_brief_agent

        return getattr(executive_brief_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
