"""Dependency detection for cloud framework adapters."""
from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class DepStatus:
    name: str
    available: bool = False
    error: str = ""


def _can_import(module: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module)
        return True, ""
    except ImportError as exc:
        return False, str(exc)


# Map: framework_key -> [(pip_package, import_module), ...]
_FRAMEWORK_DEPS: dict[str, list[tuple[str, str]]] = {
    "openai_agents": [("openai-agents", "agents")],
    "langgraph": [("langgraph", "langgraph"), ("langchain-openai", "langchain_openai")],
    "autogen": [("autogen-agentchat", "autogen_agentchat"), ("autogen-ext[openai]", "autogen_ext.models.openai")],
    "crewai": [("crewai", "crewai")],
    "google_adk": [("google-adk", "google.adk"), ("litellm", "litellm")],
}


def check_deps(framework: str) -> DepStatus:
    """Check if a framework's dependencies are importable."""
    deps = _FRAMEWORK_DEPS.get(framework, [])
    if not deps:
        return DepStatus(name=framework, available=False, error=f"unknown framework: {framework}")
    for pip_name, mod in deps:
        ok, err = _can_import(mod)
        if not ok:
            return DepStatus(name=framework, available=False, error=f"missing {pip_name}: {err[:120]}")
    return DepStatus(name=framework, available=True)


def install_deps(frameworks: list[str] | None = None, venv_python: str | None = None) -> dict[str, DepStatus]:
    """Attempt pip install for missing deps. Returns status per framework."""
    targets = frameworks or list(_FRAMEWORK_DEPS.keys())
    python = venv_python or sys.executable
    results: dict[str, DepStatus] = {}

    for fw in targets:
        status = check_deps(fw)
        if status.available:
            results[fw] = status
            continue

        packages = [pip_name for pip_name, _ in _FRAMEWORK_DEPS.get(fw, [])]
        try:
            subprocess.check_call(
                [python, "-m", "pip", "install", "--quiet", *packages],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            results[fw] = check_deps(fw)
        except Exception as exc:
            results[fw] = DepStatus(name=fw, available=False, error=f"install failed: {str(exc)[:120]}")

    return results
