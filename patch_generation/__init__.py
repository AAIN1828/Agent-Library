"""
Patch Generation — document update-patch agent package.

``agent.py`` owns the standalone pipeline (``generate_update_patch`` /
``get_answer``), env configuration, orchestrator helpers, and
``PatchGenerationChainRunner``. Catalog metadata lives in ``contract.json``.

``get_answer`` / ``build_agents`` are lazy so importing the package does not
force ``agent.py``'s import-time LLM client setup until those symbols are used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "PatchGenerationChainRunner",
    "get_answer",
    "generate_update_patch",
    "build_agents",
    "route",
    "drive_chain",
    "ENTRY_AGENT",
    "EXIT_AGENTS",
]

__agent_name__ = "patch_generation"
__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    if name in {
        "PatchGenerationChainRunner",
        "get_answer",
        "generate_update_patch",
        "build_agents",
        "route",
        "drive_chain",
        "ENTRY_AGENT",
        "EXIT_AGENTS",
    }:
        from . import agent as patch_generation_agent

        return getattr(patch_generation_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
