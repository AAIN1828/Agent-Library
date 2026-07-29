"""
Quin — SQL / structured-data agent package.

``agent.py`` owns the standalone GroupChat pipeline (``get_answer``).
Reuse metadata lives in ``spec.json`` / ``contract.json``; env helpers in
``runtime_config.py``; orchestrator helpers + ``QuinChainRunner`` in
``chain_helpers.py``.

``get_answer`` / ``build_agents`` are lazy so importing the package does not
force ``agent.py``'s import-time DB introspection until those symbols are used.
"""

from __future__ import annotations

from typing import Any

from .chain_helpers import (
    ENTRY_AGENT,
    EXIT_AGENTS,
    QuinChainRunner,
    drive_chain,
    route,
)

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
    if name == "get_answer":
        from .agent import get_answer

        return get_answer
    if name == "build_agents":
        from .chain_helpers import build_agents

        return build_agents
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
