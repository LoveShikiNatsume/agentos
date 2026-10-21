"""Unified task execution API. CLI and Bench both call execute()."""
from agentos.execution.service import execute, register_harness

__all__ = ["execute", "register_harness"]
