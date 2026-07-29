"""
Quin — SQL / structured-data agent package.

``agent.py`` owns the standalone GroupChat pipeline (``get_answer``), env
configuration, orchestrator helpers, and ``QuinChainRunner``. Catalog metadata
lives in ``contract.json``.

``get_answer`` / ``build_agents`` are lazy so importing the package does not
force ``agent.py``'s import-time DB introspection until those symbols are used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "QuinChainRunner",
    "get_answer",
    "build_agents",
    "route",
    "drive_chain",
    "ENTRY_AGENT",
    "EXIT_AGENTS",
]

__agent_name__ = "quin"
__version__ = "1.1.0"


def __getattr__(name: str) -> Any:
    if name in {
        "QuinChainRunner",
        "get_answer",
        "build_agents",
        "route",
        "drive_chain",
        "ENTRY_AGENT",
        "EXIT_AGENTS",
    }:
        from . import agent as quin_agent

        return getattr(quin_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
