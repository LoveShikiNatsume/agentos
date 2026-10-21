"""Unified software engineering task definition for cloud benchmark.

Reuses prompts from agentos.scenarios.software_eng.prompts.
Each role gets 1 LLM call (not the full 2-3 calls in P1).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentStep:
    """One agent step in the DAG."""
    role: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)


def build_task(max_tokens: int = 512) -> list[AgentStep]:
    """Build the 8-agent DAG with 1 call per role.

    Order: architect → api_design/db_design → backend → test/security/docs → integrate
    """
    # Lazy import to avoid hard dep on agentos
    try:
        from agentos.scenarios.software_eng.prompts import (
            PROJECT_CONTEXT, REPO_SUMMARY,
            ARCHITECT_PROMPT, DB_DESIGNER_PROMPT, API_DESIGNER_PROMPT,
            BACKEND_CODER_PROMPT, TEST_WRITER_PROMPT, SECURITY_REVIEWER_PROMPT,
            DOC_WRITER_PROMPT, INTEGRATOR_PROMPT,
        )
    except ImportError:
        # Fallback if agentos not on path
        PROJECT_CONTEXT = "Project: E-commerce Wishlist feature. Python/FastAPI/PostgreSQL."
        REPO_SUMMARY = "src/api/ src/models/ src/services/ tests/ docs/"
        ARCHITECT_PROMPT = "Design the Wishlist feature: data model, API endpoints, service layer."
        DB_DESIGNER_PROMPT = "Create SQLAlchemy model for Wishlist table with foreign keys."
        API_DESIGNER_PROMPT = "Define FastAPI router with wishlist endpoints and Pydantic schemas."
        BACKEND_CODER_PROMPT = "Implement Wishlist service layer: add, remove, get, check."
        TEST_WRITER_PROMPT = "Write pytest tests for all wishlist API endpoints."
        SECURITY_REVIEWER_PROMPT = "Audit wishlist feature for auth, input validation, IDOR."
        DOC_WRITER_PROMPT = "Write API documentation for wishlist endpoints."
        INTEGRATOR_PROMPT = "Review all outputs, produce integration report."

    system_prefix = f"{PROJECT_CONTEXT}\n\n{REPO_SUMMARY}\n\n"

    return [
        AgentStep(role="architect", prompt=system_prefix + ARCHITECT_PROMPT),
        AgentStep(role="db_designer", prompt=system_prefix + DB_DESIGNER_PROMPT, depends_on=["architect"]),
        AgentStep(role="api_designer", prompt=system_prefix + API_DESIGNER_PROMPT, depends_on=["architect"]),
        AgentStep(role="backend_coder", prompt=system_prefix + BACKEND_CODER_PROMPT, depends_on=["db_designer", "api_designer"]),
        AgentStep(role="test_writer", prompt=system_prefix + TEST_WRITER_PROMPT, depends_on=["backend_coder"]),
        AgentStep(role="security_reviewer", prompt=system_prefix + SECURITY_REVIEWER_PROMPT, depends_on=["backend_coder"]),
        AgentStep(role="doc_writer", prompt=system_prefix + DOC_WRITER_PROMPT, depends_on=["backend_coder"]),
        AgentStep(role="integrator", prompt=system_prefix + INTEGRATOR_PROMPT, depends_on=["test_writer", "security_reviewer", "doc_writer"]),
    ]
