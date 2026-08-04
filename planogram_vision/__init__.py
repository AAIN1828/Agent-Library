"""
Planogram Vision — multi-agent shelf-image analysis package.

``agent.py`` owns the standalone GroupChat pipeline (``get_answer`` / ``run``),
env configuration, orchestrator helpers, and ``PlanogramVisionChainRunner``.
Catalog metadata lives in ``contract.json``.

``get_answer`` / ``build_agents`` are lazy so importing the package does not
force ``agent.py``'s import-time AutoGen setup until those symbols are used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "PlanogramVisionChainRunner",
    "get_answer",
    "run",
    "build_agents",
    "route",
    "drive_chain",
    "ENTRY_AGENT",
    "EXIT_AGENTS",
]

__agent_name__ = "planogram_vision"
__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    if name in {
        "PlanogramVisionChainRunner",
        "get_answer",
        "run",
        "build_agents",
        "route",
        "drive_chain",
        "ENTRY_AGENT",
        "EXIT_AGENTS",
    }:
        from . import agent as planogram_vision_agent

        return getattr(planogram_vision_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
