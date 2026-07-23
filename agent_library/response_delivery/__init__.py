"""
Response Delivery — confidence-aware polish + evaluation package.
"""

from .agent import (
    ENTRY_AGENT,
    EXIT_AGENTS,
    LLM_CONFIG,
    POST_GENERATION_EVALUATION_RUBRIC,
    ResponseDeliveryRunner,
    build_agents,
    compute_post_generation_composite,
    evaluation_timestamp_iso,
    route,
)

__all__ = [
    "ResponseDeliveryRunner",
    "build_agents",
    "route",
    "ENTRY_AGENT",
    "EXIT_AGENTS",
    "LLM_CONFIG",
    "POST_GENERATION_EVALUATION_RUBRIC",
    "compute_post_generation_composite",
    "evaluation_timestamp_iso",
]

__agent_name__ = "response_delivery"
__version__ = "1.0.0"
