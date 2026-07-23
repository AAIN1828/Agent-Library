"""
Grounded Answer — dual-source answer consolidation package.
"""

from .agent import (
    ENTRY_AGENT,
    EXIT_AGENTS,
    LLM_CONFIG,
    GroundedAnswerRunner,
    build_agents,
    route,
)

__all__ = [
    "GroundedAnswerRunner",
    "build_agents",
    "route",
    "ENTRY_AGENT",
    "EXIT_AGENTS",
    "LLM_CONFIG",
]

__agent_name__ = "grounded_answer"
__version__ = "1.0.0"
