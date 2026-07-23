"""
Intake Gateway — question intake + Responsible AI safety gate package.
"""

from .agent import (
    ENTRY_AGENT,
    EXIT_AGENTS,
    LLM_CONFIG,
    IntakeGatewayRunner,
    azure_safety_check,
    build_agents,
    evaluation_timestamp_iso,
    route,
)

__all__ = [
    "IntakeGatewayRunner",
    "build_agents",
    "route",
    "ENTRY_AGENT",
    "EXIT_AGENTS",
    "LLM_CONFIG",
    "azure_safety_check",
    "evaluation_timestamp_iso",
]

__agent_name__ = "intake_gateway"
__version__ = "1.0.0"
