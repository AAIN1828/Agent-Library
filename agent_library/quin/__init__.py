"""
Quin — SQL / structured-data agent package.

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
    evaluation_timestamp_iso,
)

__all__ = [
    "build_agents",
    "route",
    "ENTRY_AGENT",
    "EXIT_AGENTS",
    "LLM_CONFIG",
    "POST_GENERATION_EVALUATION_RUBRIC",
    "compute_post_generation_composite",
    "evaluation_timestamp_iso",
]

__agent_name__ = "quin"
__version__ = "1.0.0"
