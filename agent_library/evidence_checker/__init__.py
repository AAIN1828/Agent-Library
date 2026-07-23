"""
Evidence Checker — conflict / strength reconciliation package.
"""

from .agent import (
    ENTRY_AGENT,
    EXIT_AGENTS,
    LLM_CONFIG,
    EvidenceCheckerRunner,
    build_agents,
    evaluation_timestamp_iso,
    route,
)

__all__ = [
    "EvidenceCheckerRunner",
    "build_agents",
    "route",
    "ENTRY_AGENT",
    "EXIT_AGENTS",
    "LLM_CONFIG",
    "evaluation_timestamp_iso",
]

__agent_name__ = "evidence_checker"
__version__ = "1.0.0"
