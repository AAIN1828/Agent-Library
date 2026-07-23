"""
Intent Router — SQL vs semantic vs both routing package.
"""

from .agent import (
    ENTRY_AGENT,
    EXIT_AGENTS,
    LLM_CONFIG,
    IntentRouterRunner,
    build_agents,
    route,
)

__all__ = [
    "IntentRouterRunner",
    "build_agents",
    "route",
    "ENTRY_AGENT",
    "EXIT_AGENTS",
    "LLM_CONFIG",
]

__agent_name__ = "intent_router"
__version__ = "1.0.0"
