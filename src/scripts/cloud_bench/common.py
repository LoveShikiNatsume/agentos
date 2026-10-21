"""Shared dataclasses and cost calculation for cloud benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# DeepSeek pricing (CNY per 1M tokens)
DEEPSEEK_INPUT_PRICE_CNY = 0.5
DEEPSEEK_CACHED_INPUT_PRICE_CNY = 0.1
DEEPSEEK_OUTPUT_PRICE_CNY = 2.0
USD_CNY = 7.2

# Unified prompt length cap — applied identically to native and runtime
MAX_PROMPT_CHARS = 4000


def normalize_prompt(prompt: str) -> tuple[str, bool]:
    """Normalize prompt to unified length cap.

    Returns (truncated_prompt, was_truncated).
    Applied identically across all adapters and modes for fairness.
    """
    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt, False
    return prompt[:MAX_PROMPT_CHARS], True


@dataclass
class RunMetrics:
    """Metrics from a single benchmark run (native or runtime)."""
    avg_latency: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    api_cost: float = 0.0
    context_reuse: float = 0.0
    recovered_faults: int = 0
    observable_events: int = 0
    status: str = "ok"
    error: str = ""
    token_source: str = "usage"       # "usage" | "estimated" | "none"
    adapter_source: str = "sdk"       # "sdk" | "direct_adapter" | "runtime_gateway" | "runtime_hook"
    note: str = ""
    # Runtime contract fields
    orchestration_source: str = ""    # "sdk_runner" | "sdk_stategraph" | "sdk_agentchat" | "sdk_crew" | "direct_loop"
    model_path: str = ""              # "deepseek_direct" | "agentos_model_gateway"
    runtime_contract: str = ""        # "framework_orchestrated_runtime" | "model_gateway_direct_runtime"
    prompt_chars_total: int = 0       # Total prompt chars across all steps
    prompt_chars_sent: int = 0        # Chars actually sent (after truncation)
    prompt_truncated: bool = False    # Whether any prompt was truncated
    framework_entrypoint_invoked: bool = False  # Framework entrypoint actually called
    model_gateway_invoked: bool = False          # ModelGateway actually called
    runtime_mode_actual: str = ""                # "framework_orchestrated" | "gateway_direct"

    def compute_cost(self) -> None:
        """Compute api_cost from token counts using DeepSeek pricing."""
        cached_input_cny = self.cache_hit_tokens * DEEPSEEK_CACHED_INPUT_PRICE_CNY / 1_000_000
        uncached_input_cny = self.cache_miss_tokens * DEEPSEEK_INPUT_PRICE_CNY / 1_000_000
        output_cny = self.completion_tokens * DEEPSEEK_OUTPUT_PRICE_CNY / 1_000_000
        total_cny = cached_input_cny + uncached_input_cny + output_cny
        self.api_cost = round(total_cny / USD_CNY, 6)

    def compute_context_reuse(self) -> None:
        """Compute context_reuse percentage."""
        if self.prompt_tokens > 0:
            self.context_reuse = round(self.cache_hit_tokens / self.prompt_tokens * 100, 1)
        else:
            self.context_reuse = 0.0

    def to_display_dict(self, include_runtime_contract: bool = False) -> dict[str, Any]:
        """Return dict matching cloud_benchmark.py schema.

        Args:
            include_runtime_contract: Only True for runtime side. Native side
                uses orchestration_source/model_path instead.
        """
        d: dict[str, Any] = {
            "avg_latency": round(self.avg_latency, 2),
            "prompt_tokens": round(self.prompt_tokens / 1000, 1),
            "api_cost": round(self.api_cost, 3),
            "context_reuse": round(self.context_reuse, 1),
            "recovered_faults": self.recovered_faults,
            "observable_events": self.observable_events,
        }
        # Provenance fields
        if self.status != "ok":
            d["status"] = self.status
        d["token_source"] = self.token_source
        if self.adapter_source != "sdk":
            d["adapter_source"] = self.adapter_source
        if self.note:
            d["note"] = self.note
        # Orchestration/model fields (both sides)
        if self.orchestration_source:
            d["orchestration_source"] = self.orchestration_source
        if self.model_path:
            d["model_path"] = self.model_path
        # runtime_contract only on runtime side
        if self.runtime_contract and include_runtime_contract:
            d["runtime_contract"] = self.runtime_contract
        if self.prompt_chars_total > 0:
            d["prompt_chars_total"] = self.prompt_chars_total
            d["prompt_chars_sent"] = self.prompt_chars_sent
            d["prompt_truncated"] = self.prompt_truncated
        # Verification fields
        if self.framework_entrypoint_invoked:
            d["framework_entrypoint_invoked"] = True
        if self.model_gateway_invoked:
            d["model_gateway_invoked"] = True
        if self.runtime_mode_actual:
            d["runtime_mode_actual"] = self.runtime_mode_actual
        return d


@dataclass
class FrameworkResult:
    """Result for one framework: native + runtime + improvement."""
    framework: str = ""
    native: RunMetrics | None = None
    runtime: RunMetrics | None = None
    improvement: dict[str, Any] = field(default_factory=dict)

    def compute_improvement(self) -> None:
        """Compute improvement percentages from native vs runtime."""
        if self.native is None or self.runtime is None:
            self.improvement = {}
            return
        n, r = self.native, self.runtime
        if n.status != "ok" or r.status != "ok":
            self.improvement = {}
            return
        if n.token_source == "estimated" or r.token_source == "estimated":
            self.improvement = {}
            return

        def pct(nv: float, rv: float) -> float:
            return round((rv - nv) / abs(nv) * 100, 1) if nv != 0 else 0.0

        self.improvement = {
            "latency": pct(n.avg_latency, r.avg_latency),
            "tokens": pct(n.prompt_tokens, r.prompt_tokens),
            "cost": pct(n.api_cost, r.api_cost),
            "context_reuse": round(r.context_reuse - n.context_reuse, 1),
            "events": round(r.observable_events / max(n.observable_events, 1), 1),
        }

    def _metrics_or_error(self, m: RunMetrics | None, is_runtime: bool = False) -> dict[str, Any] | None:
        if m is None:
            return None
        if m.status not in ("ok", "degraded"):
            d: dict[str, Any] = {"status": m.status}
            if m.error:
                d["error"] = m.error
            return d
        return m.to_display_dict(include_runtime_contract=is_runtime)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "native": self._metrics_or_error(self.native, is_runtime=False),
            "runtime": self._metrics_or_error(self.runtime, is_runtime=True),
            "improvement": self.improvement,
        }
