"""
Eryl — semantic / vector-retrieval agent package.

Split out of the unified AutoGen pipeline. See agent.py for the
AssistantAgent definitions and routing logic, spec.json for the
agent's registry metadata, and contract.json for its input/output
message contracts.
"""

from .agent import (
    build_agents,
    route,
    ENTRY_AGENT,
    EXIT_AGENTS,
    LLM_CONFIG,
    POST_GENERATION_EVALUATION_RUBRIC,
    compute_post_generation_composite,
    generate_embeddings,
    search_client,
)

__all__ = [
    "build_agents",
    "route",
    "ENTRY_AGENT",
    "EXIT_AGENTS",
    "LLM_CONFIG",
    "POST_GENERATION_EVALUATION_RUBRIC",
    "compute_post_generation_composite",
    "generate_embeddings",
    "search_client",
]

__agent_name__ = "eryl"
__version__ = "1.0.0"
