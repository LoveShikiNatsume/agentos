from __future__ import annotations

import asyncio
import inspect
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agentos.comm.bus import MessageBus
from agentos.context.manager import ContextManager, PreparedContext
from agentos.context.types import ContextScope
from agentos.core.agent import AgentControlBlock, agent_transition
from agentos.core.resources import ResourceQuota, ResourceUsage, ResourceUsageSummary
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.core.types import AgentState, Message
from agentos.events.log import EventLog
from agentos.events.types import (
    EventCategory,
    EventSeverity,
    RuntimeEvent,
    agent_state_changed,
    llm_request_sent,
    task_completed,
)
from agentos.fault.timeout import AgentTimeoutError, fault_timeout, run_with_timeout
from agentos.fault.recovery import (
    RecoveryAction, RepetitionError, SemanticValidationError,
    ValidationExecutionError, ValidationResult, Validator, failure_type,
    normalized_result_hash,
)
from agentos.fault.injector import FaultInjectionPlan
from agentos.isolation.bpf import MockBPFMapAdapter
from agentos.isolation.cgroup import CgroupV2Manager
from agentos.isolation.cgroup_feedback import (
    CgroupFeedbackAccumulator,
    CgroupFeedbackSnapshot,
)
from agentos.isolation.kernel_adapter import (
    KernelSchedulerAdapter,
    PressureSample,
    UserSpaceKernelAdapter,
)
from agentos.isolation.pressure import max_pressure_level
from agentos.isolation.sched_ext import KernelProbeResult, probe_sched_ext
from agentos.isolation.sched_ext_adapter import SchedExtKernelAdapter
from agentos.isolation.sched_ext_control import SchedExtControlClient
from agentos.llm.base import LLMBackend, LLMRequest, LLMResponse
from agentos.context.cache_tracker import CacheHeatTracker
from agentos.llm.gateway import ModelGateway
from agentos.scheduler.affinity import CacheAffinityAdvisor
from agentos.scheduler.dag import DAGScheduler
from agentos.scheduler.adaptive import AdaptivePolicyTuner
from agentos.scheduler.fairness import FairnessTracker
from agentos.tools.manager import (
    ToolConflictError,
    ToolError,
    ToolExecutionError,
    ToolManager,
    ToolPermissionError,
    ToolResult,
    ToolTimeoutError,
)
from agentos.worker.process import (
    SubprocessWorkerError,
    SubprocessWorkerManager,
    WorkerResult,
    WorkerSpec,
)


@dataclass
class _QueuedLLMRequest:
    request: LLMRequest
    future: asyncio.Future[LLMResponse]
    submitted_at: float = field(default_factory=time.monotonic)


class _LLMBatchPool:
    def __init__(
        self,
        *,
        backend: LLMBackend,
        emit: Callable[[RuntimeEvent], None],
        max_batch_size: int,
        wait_ms: float,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_llm_batch_size must be at least 1")
        if wait_ms < 0:
            raise ValueError("llm_batch_wait_ms must be non-negative")
        self._backend = backend
        self._emit = emit
        self._max_batch_size = max_batch_size
        self._wait_seconds = wait_ms / 1000.0
        self._buckets: dict[tuple[str, str, str], list[_QueuedLLMRequest]] = {}
        self._timers: dict[tuple[str, str, str], asyncio.TimerHandle] = {}

    async def submit(self, request: LLMRequest) -> LLMResponse:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[LLMResponse] = loop.create_future()
        key = self._key(request)
        bucket = self._buckets.setdefault(key, [])
        bucket.append(_QueuedLLMRequest(request=request, future=future))

        if len(bucket) >= self._max_batch_size:
            self._cancel_timer(key)
            queued = bucket[: self._max_batch_size]
            remaining = bucket[self._max_batch_size :]
            if remaining:
                self._buckets[key] = remaining
                self._timers[key] = loop.call_later(
                    self._wait_seconds,
                    lambda: loop.create_task(
                        self._flush_bucket(key, reason="wait_window")
                    ),
                )
            else:
                self._buckets.pop(key, None)
            loop.create_task(self._flush(queued, reason="max_batch_size"))
        elif key not in self._timers:
            self._timers[key] = loop.call_later(
                self._wait_seconds,
                lambda: loop.create_task(
                    self._flush_bucket(key, reason="wait_window")
                ),
            )

        return await future

    async def _flush_bucket(
        self,
        key: tuple[str, str, str],
        *,
        reason: str,
    ) -> None:
        self._cancel_timer(key)
        queued = self._buckets.pop(key, [])
        await self._flush(queued, reason=reason)

    async def _flush(self, queued: list[_QueuedLLMRequest], *, reason: str) -> None:
        if not queued:
            return

        requests = [item.request for item in queued]
        started = time.monotonic()
        event_data = {
            "source": "global_pool",
            "flush_reason": reason,
            "batch_size": len(requests),
            "max_batch_size": self._max_batch_size,
            "batch_wait_ms": self._wait_seconds * 1000,
            "backend": requests[0].backend,
            **_llm_batch_group_data(requests),
        }
        self._emit(
            RuntimeEvent(
                event_type="llm.batch_started",
                category=EventCategory.LLM,
                data=event_data,
            )
        )
        try:
            if hasattr(self._backend, "generate_batch"):
                responses = await self._backend.generate_batch(requests)  # type: ignore[attr-defined]
            else:
                responses = await asyncio.gather(
                    *(self._backend.generate(request) for request in requests)
                )
        except Exception as exc:
            self._emit(
                RuntimeEvent(
                    event_type="llm.batch_failed",
                    category=EventCategory.LLM,
                    severity=EventSeverity.ERROR,
                    data={**event_data, "error": str(exc)},
                )
            )
            for item in queued:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        total_tokens = sum(response.tokens_used for response in responses)
        queue_wait_ms = [
            (started - item.submitted_at) * 1000 for item in queued
        ]
        self._emit(
            RuntimeEvent(
                event_type="llm.batch_completed",
                category=EventCategory.LLM,
                data={
                    **event_data,
                    "response_count": len(responses),
                    "tokens_used": total_tokens,
                    "avg_queue_wait_ms": (
                        sum(queue_wait_ms) / len(queue_wait_ms)
                        if queue_wait_ms
                        else 0.0
                    ),
                    "max_queue_wait_ms": max(queue_wait_ms) if queue_wait_ms else 0.0,
                },
            )
        )

        if len(responses) != len(queued):
            error = RuntimeError(
                "LLM batch backend returned mismatched response count"
            )
            for item in queued:
                if not item.future.done():
                    item.future.set_exception(error)
            return

        for item, response in zip(queued, responses):
            if not item.future.done():
                item.future.set_result(response)

    def _cancel_timer(self, key: tuple[str, str, str]) -> None:
        timer = self._timers.pop(key, None)
        if timer is not None:
            timer.cancel()

    @staticmethod
    def _key(request: LLMRequest) -> tuple[str, str, str]:
        return (
            request.backend,
            request.model,
            request.prefix_id or "__no_prefix__",
        )


def _llm_batch_group_data(requests: list[LLMRequest]) -> dict[str, Any]:
    groups: dict[str, int] = {}
    for request in requests:
        prefix_id = request.prefix_id or "__no_prefix__"
        groups[prefix_id] = groups.get(prefix_id, 0) + 1
    return {
        "prefix_group_count": len(groups),
        "prefix_groups": [
            {"prefix_id": prefix_id, "size": size}
            for prefix_id, size in groups.items()
        ],
    }


class AgentHandler(ABC):
    @dataclass
    class Context:
        task: Task
        acb: AgentControlBlock
        bus: MessageBus
        context_manager: ContextManager
        _llm: LLMBackend
        _emit: Callable[[RuntimeEvent], None]
        _transition: Callable[[AgentControlBlock, AgentState], None]
        _submit_task: Callable[[Task, AgentControlBlock], None]
        _submit_tasks: Callable[[list[Task], AgentControlBlock], None]
        _cancel_task: Callable[[str, AgentControlBlock, str | None], None]
        _cancel_downstream: Callable[[str, AgentControlBlock, str | None], list[str]]
        _replace_downstream: Callable[
            [str, list[Task], AgentControlBlock, str | None],
            tuple[list[str], list[str]],
        ]
        _llm_batch_submit: Callable[[LLMRequest], Any] | None
        _tools: ToolManager
        attempt: int = 1
        recovery_feedback: list[dict[str, Any]] = field(default_factory=list)
        _tool_call_counts: dict[str, int] = field(default_factory=dict)
        _repetition_error: RepetitionError | None = None
        _context_requests: int = 0
        _context_refs_granted: int = 0

        def request_context_refs(self, refs: list[str], *, reason: str = "") -> dict[str, Any]:
            """Append allowed shared refs, with task-lifetime (not attempt) quotas.

            Names and ctx:// URIs are equivalent. Every well-typed request,
            including empty/denied requests, consumes one request slot. Invalid
            or absent policies fail closed. CONTEXT_RETRY restores initial
            declarations but never refunds either quota; ordinary RETRY retains
            additions. This synchronous operation cannot interleave with another
            request on the asyncio event loop.
            """
            if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
                raise TypeError("refs must be a list of strings")
            if not isinstance(reason, str):
                raise TypeError("reason must be a string")
            policy = self.task.metadata.get("context_request_policy")
            valid = (
                isinstance(policy, dict)
                and isinstance(policy.get("candidates"), list)
                and all(isinstance(r, str) and r.removeprefix("ctx://")
                        for r in policy["candidates"])
                and all(type(policy.get(k)) is int and policy[k] >= 0
                        for k in ("max_requests", "max_refs"))
            )
            max_requests = policy["max_requests"] if valid else 0
            max_refs = policy["max_refs"] if valid else 0
            candidates = {r.removeprefix("ctx://") for r in policy["candidates"]} if valid else set()
            exhausted = self._context_requests >= max_requests
            self._context_requests += 1
            declared = {r.removeprefix("ctx://") for r in self.acb.context_refs}
            implicit_all = not declared and not self.task.metadata.get("explicit_context_refs", False)
            granted, denied, seen = [], [], set()
            for raw in refs:
                name = raw.removeprefix("ctx://")
                ref = self.context_manager.get_ref(name)
                why = None
                if not valid:
                    why = "policy_disabled" if not policy else "invalid_policy"
                elif exhausted:
                    why = "request_limit"
                elif name in seen:
                    why = "duplicate_request"
                elif name not in candidates:
                    why = "not_candidate"
                elif ref is None:
                    why = "not_found"
                elif ref.scope != ContextScope.SHARED:
                    why = "not_shared"
                elif not ref.can_read():
                    why = "not_readable"
                elif name in declared or implicit_all:
                    why = "already_declared"
                elif self._context_refs_granted >= max_refs:
                    why = "ref_limit"
                seen.add(name)
                if why:
                    denied.append({"ref": raw, "reason": why})
                else:
                    self.acb.context_refs.append(name)
                    declared.add(name)
                    granted.append(name)
                    self._context_refs_granted += 1
            result = {
                "requested": list(refs), "granted": granted, "denied": denied,
                "remaining_requests": max(0, max_requests - self._context_requests),
                "remaining_refs": max(0, max_refs - self._context_refs_granted),
                "requests_used": self._context_requests, "refs_granted": self._context_refs_granted,
                "reason": reason, "attempt": self.attempt,
            }
            self._emit(RuntimeEvent(
                event_type="context.refs_requested", category=EventCategory.CONTEXT,
                agent_id=self.acb.agent_id, task_id=self.task.task_id, trace_id=self.acb.trace_id,
                data=deepcopy(result),
            ))
            return result

        @property
        def _detect_repetition(self) -> bool:
            return self.task.metadata.get("validator") is not None or bool(
                self.task.metadata.get("detect_repetition", False)
            )

        def submit_task(self, task: Task) -> None:
            self._submit_task(task, self.acb)

        def submit_tasks(self, tasks: list[Task]) -> None:
            self._submit_tasks(tasks, self.acb)

        def cancel_task(self, task_id: str, *, reason: str | None = None) -> None:
            self._cancel_task(task_id, self.acb, reason)

        def cancel_downstream(
            self,
            task_id: str | None = None,
            *,
            reason: str | None = None,
        ) -> list[str]:
            return self._cancel_downstream(task_id or self.task.task_id, self.acb, reason)

        def replace_downstream(
            self,
            task_id: str | None,
            replacement_tasks: list[Task],
            *,
            reason: str | None = None,
        ) -> tuple[list[str], list[str]]:
            return self._replace_downstream(
                task_id or self.task.task_id,
                replacement_tasks,
                self.acb,
                reason,
            )

        async def tool_call(self, name: str, **kwargs: Any) -> ToolResult:
            if self._detect_repetition:
                if self._repetition_error is not None:
                    raise self._repetition_error
                threshold = self.task.metadata.get("repeated_tool_call_threshold", 3)
                if type(threshold) is not int or threshold < 1:
                    raise ValueError("repeated_tool_call_threshold must be a positive integer")
                call_hash = normalized_result_hash(
                    {"tool": name, "kwargs": kwargs}, normalize_text=False
                )
                self._tool_call_counts[call_hash] = self._tool_call_counts.get(call_hash, 0) + 1
                if self._tool_call_counts[call_hash] > threshold:
                    self._repetition_error = RepetitionError(
                        f"identical tool call {name!r} exceeded threshold {threshold}"
                    )
                    raise self._repetition_error
            self._emit(
                RuntimeEvent(
                    event_type="tool.call_started",
                    agent_id=self.acb.agent_id,
                    task_id=self.task.task_id,
                    trace_id=self.acb.trace_id,
                    category=EventCategory.TOOL,
                    data={"tool": name, "kwargs": sorted(kwargs)},
                )
            )
            self._transition(self.acb, AgentState.BLOCKED_TOOL)
            started = time.monotonic()
            try:
                result = await self._tools.call(
                    name,
                    kwargs=kwargs,
                    allowed_tools=self.task.resource_requirements.allowed_tools,
                )
            except ToolError as exc:
                result = exc.result or ToolResult(
                    name=name,
                    ok=False,
                    error=str(exc),
                    duration_secs=time.monotonic() - started,
                )
                self.acb.stats.tool_calls += 1
                self.acb.stats.total_tool_wait_secs += result.duration_secs
                event_type = self._tool_error_event_type(exc)
                self._emit(
                    RuntimeEvent(
                        event_type=event_type,
                        agent_id=self.acb.agent_id,
                        task_id=self.task.task_id,
                        trace_id=self.acb.trace_id,
                        category=EventCategory.TOOL,
                        severity=(
                            EventSeverity.WARNING
                            if isinstance(exc, (ToolPermissionError, ToolConflictError))
                            else EventSeverity.ERROR
                        ),
                        data=self._tool_event_data(result),
                    )
                )
                self._transition(self.acb, AgentState.RUNNING)
                raise

            self.acb.stats.tool_calls += 1
            self.acb.stats.total_tool_wait_secs += result.duration_secs
            self._emit(
                RuntimeEvent(
                    event_type="tool.call_finished",
                    agent_id=self.acb.agent_id,
                    task_id=self.task.task_id,
                    trace_id=self.acb.trace_id,
                    category=EventCategory.TOOL,
                    data=self._tool_event_data(result),
                )
            )
            self._transition(self.acb, AgentState.RUNNING)
            return result

        @staticmethod
        def _tool_error_event_type(exc: ToolError) -> str:
            if isinstance(exc, ToolTimeoutError):
                return "tool.timeout"
            if isinstance(exc, ToolConflictError):
                return "tool.conflict"
            if isinstance(exc, ToolPermissionError):
                return "tool.permission_denied"
            if isinstance(exc, ToolExecutionError):
                return "tool.failed"
            return "tool.failed"

        @staticmethod
        def _tool_event_data(result: ToolResult) -> dict[str, Any]:
            return {
                "tool": result.name,
                "ok": result.ok,
                "output_preview": str(result.output)[:200],
                "error": result.error,
                "duration_secs": result.duration_secs,
                "timed_out": result.timed_out,
            }

        def _prepare_llm_context(
            self, messages: list[dict[str, Any]]
        ) -> PreparedContext:
            if self._repetition_error is not None:
                raise self._repetition_error
            # Existing tasks with no declared refs retain all-shared behavior.
            requested = list(self.acb.context_refs)
            if not requested and not self.task.metadata.get("explicit_context_refs", False):
                requested = None
            if self.recovery_feedback:
                messages = [*messages, dict(self.recovery_feedback[-1])]
            prepared = self.context_manager.prepare_context(
                self.acb.agent_id, messages, context_ref_names=requested
            )
            self._emit(
                RuntimeEvent(
                    event_type="context.assembled",
                    agent_id=self.acb.agent_id,
                    task_id=self.task.task_id,
                    trace_id=self.acb.trace_id,
                    category=EventCategory.CONTEXT,
                    data={
                        "selection": prepared.selection_details.get("selection", "explicit" if requested is not None else "all_shared"),
                        "requested_refs": prepared.selection_details.get("requested_refs", requested),
                        "shared_refs": list(prepared.shared_refs),
                        "prefix_id": prepared.prefix_id,
                        "context_policy": prepared.selection_details,
                    },
                )
            )
            return prepared

        def _make_llm_request(
            self,
            full_messages: list[dict[str, Any]],
            prefix_id: str,
        ) -> LLMRequest:
            return LLMRequest(
                agent_id=self.acb.agent_id,
                messages=full_messages,
                task_id=self.task.task_id,
                trace_id=self.acb.trace_id,
                context_refs=list(self.acb.context_refs),
                prefix_id=prefix_id,
            )

        @staticmethod
        def _llm_batch_group_data(
            requests: list[LLMRequest],
        ) -> dict[str, Any]:
            return _llm_batch_group_data(requests)

        async def llm_call(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> LLMResponse:
            prepared = self._prepare_llm_context(messages)
            full_messages = prepared.messages
            self._emit(
                llm_request_sent(
                    self.acb.agent_id,
                    "default",
                    sum(len(message.get("content") or "") for message in full_messages),
                    task_id=self.task.task_id,
                    trace_id=self.acb.trace_id,
                )
            )

            self._transition(self.acb, AgentState.BLOCKED_LLM)

            req = self._make_llm_request(full_messages, prepared.prefix_id)
            if prepared.selection_details:
                req.context_refs = list(prepared.shared_refs)
                req.metadata["context_selection"] = prepared.selection_details
            req.tools = list(tools or [])
            if self._llm_batch_submit is not None:
                resp = await self._llm_batch_submit(req)
            else:
                resp = await self._llm.generate(req)

            self._transition(self.acb, AgentState.RUNNING)

            self.acb.stats.llm_calls += 1
            self.acb.stats.tokens_received += resp.tokens_used
            metrics_data: dict[str, Any] = {
                "model": resp.model,
                "tokens_used": resp.tokens_used,
                "latency_secs": resp.latency_secs,
            }
            if hasattr(resp, "metrics") and resp.metrics is not None:
                m = resp.metrics
                for attr in ("cache_hit_tokens", "cache_miss_tokens", "prompt_tokens", "completion_tokens"):
                    if hasattr(m, attr):
                        metrics_data[attr] = getattr(m, attr)
            self._emit(
                RuntimeEvent(
                    event_type="llm.response_received",
                    agent_id=self.acb.agent_id,
                    task_id=self.task.task_id,
                    trace_id=self.acb.trace_id,
                    category=EventCategory.LLM,
                    data=metrics_data,
                )
            )
            return resp

        async def llm_batch_call(
            self, messages_batch: list[list[dict[str, Any]]]
        ) -> list[LLMResponse]:
            if not messages_batch:
                return []

            batch_size = len(messages_batch)
            requests: list[LLMRequest] = []
            for messages in messages_batch:
                prepared = self._prepare_llm_context(messages)
                full_messages = prepared.messages
                self._emit(
                    llm_request_sent(
                        self.acb.agent_id,
                        "default",
                        sum(
                            len(message.get("content") or "")
                            for message in full_messages
                        ),
                        task_id=self.task.task_id,
                        trace_id=self.acb.trace_id,
                    )
                )
                request = self._make_llm_request(full_messages, prepared.prefix_id)
                if prepared.selection_details:
                    request.context_refs = list(prepared.shared_refs)
                    request.metadata["context_selection"] = prepared.selection_details
                requests.append(request)

            group_data = self._llm_batch_group_data(requests)
            batch_event_data = {
                "batch_size": batch_size,
                "backend": "default",
                **group_data,
            }
            self._emit(
                RuntimeEvent(
                    event_type="llm.batch_started",
                    agent_id=self.acb.agent_id,
                    task_id=self.task.task_id,
                    trace_id=self.acb.trace_id,
                    category=EventCategory.LLM,
                    data=batch_event_data,
                )
            )
            self._transition(self.acb, AgentState.BLOCKED_LLM)
            try:
                if hasattr(self._llm, "generate_batch"):
                    responses = await self._llm.generate_batch(requests)  # type: ignore[attr-defined]
                else:
                    responses = await asyncio.gather(
                        *(self._llm.generate(request) for request in requests)
                    )
            except Exception as exc:
                self._emit(
                    RuntimeEvent(
                        event_type="llm.batch_failed",
                        agent_id=self.acb.agent_id,
                        task_id=self.task.task_id,
                        trace_id=self.acb.trace_id,
                        category=EventCategory.LLM,
                        severity=EventSeverity.ERROR,
                        data={**batch_event_data, "error": str(exc)},
                    )
                )
                raise
            self._transition(self.acb, AgentState.RUNNING)

            total_tokens = 0
            for response in responses:
                self.acb.stats.llm_calls += 1
                self.acb.stats.tokens_received += response.tokens_used
                total_tokens += response.tokens_used
                batch_metrics: dict[str, Any] = {
                    "model": response.model,
                    "tokens_used": response.tokens_used,
                    "latency_secs": response.latency_secs,
                }
                if hasattr(response, "metrics") and response.metrics is not None:
                    m = response.metrics
                    for attr in ("cache_hit_tokens", "cache_miss_tokens", "prompt_tokens", "completion_tokens"):
                        if hasattr(m, attr):
                            batch_metrics[attr] = getattr(m, attr)
                self._emit(
                    RuntimeEvent(
                        event_type="llm.response_received",
                        agent_id=self.acb.agent_id,
                        task_id=self.task.task_id,
                        trace_id=self.acb.trace_id,
                        category=EventCategory.LLM,
                        data=batch_metrics,
                    )
                )
            self._emit(
                RuntimeEvent(
                    event_type="llm.batch_completed",
                    agent_id=self.acb.agent_id,
                    task_id=self.task.task_id,
                    trace_id=self.acb.trace_id,
                    category=EventCategory.LLM,
                    data={
                        **batch_event_data,
                        "response_count": len(responses),
                        "tokens_used": total_tokens,
                    },
                )
            )
            return list(responses)

    async def accept_result(self, ctx: "AgentHandler.Context", result: Any) -> None:
        """Publish an accepted result after optional application validation.

        Adapters can defer shared publication to this hook. Runtime cannot undo
        arbitrary external side effects performed by a handler during run().
        """

    @abstractmethod
    async def run(self, ctx: "AgentHandler.Context") -> Any: ...


class WorkflowExecutionError(RuntimeError):
    """A failed workflow whose independent branches have finished executing."""

    def __init__(
        self,
        *,
        failures: dict[str, str],
        results: dict[str, Any],
        skipped_tasks: list[str],
    ) -> None:
        self.failures = dict(failures)
        self.results = dict(results)
        self.skipped_tasks = list(skipped_tasks)
        super().__init__(
            f"workflow failed in {len(failures)} task(s): {', '.join(failures)}; "
            f"{len(results)} completed, {len(skipped_tasks)} skipped"
        )


class Runtime:
    """Execute task batches and feed their wall time back to the scheduler.

    ``fairness_scale="service_time"`` opts into relative starvation limits
    and bounded cache-affinity bypass; the default ``"absolute"`` retains
    fixed thresholds. An injected fairness tracker must use the same scale.
    """

    def __init__(
        self,
        llm_backend: LLMBackend,
        handlers: dict[str, AgentHandler],
        context_manager: ContextManager | None = None,
        available_resources: ResourceQuota | None = None,
        max_batch_size: int | None = None,
        max_llm_batch_size: int | None = None,
        llm_batch_wait_ms: float | None = None,
        event_log: EventLog | str | Path | None = None,
        recovery_policy: Any | None = None,
        fault_injection: FaultInjectionPlan | dict[str, Any] | None = None,
        worker_manager: SubprocessWorkerManager | None = None,
        execution_mode: str = "coroutine",
        cgroup_manager: CgroupV2Manager | None = None,
        cleanup_cgroup_on_exit: bool = True,
        tool_manager: ToolManager | None = None,
        propagation_depth: int | None = None,
        broadcaster: Any | None = None,
        scheduler_fairness_tracker: FairnessTracker | None = None,
        kernel_probe: KernelProbeResult | None = None,
        bpf_map_adapter: MockBPFMapAdapter | None = None,
        enable_kernel_probe: bool = True,
        kernel_adapter: KernelSchedulerAdapter | None = None,
        sched_ext_mode: str = "disabled",
        sched_ext_pin_dir: str | Path = "/sys/fs/bpf/agentos/sched_ext",
        sched_ext_control: SchedExtControlClient | None = None,
        force_scheduler_policy: str | None = None,
        adaptive_tuner: AdaptivePolicyTuner | None = None,
        worker_sample_interval_seconds: float = 0.1,
        failure_policy: str = "fail_fast",
        validators: dict[str, Validator] | None = None,
        fairness_scale: str = "absolute",
    ) -> None:
        if fairness_scale not in {"absolute", "service_time"}:
            raise ValueError("fairness_scale must be 'absolute' or 'service_time'")
        if (scheduler_fairness_tracker is not None
                and scheduler_fairness_tracker.fairness_scale != fairness_scale):
            raise ValueError(
                "fairness_scale must match the supplied scheduler_fairness_tracker"
            )
        self._fairness_scale = fairness_scale
        if failure_policy not in {"fail_fast", "isolate"}:
            raise ValueError("failure_policy must be 'fail_fast' or 'isolate'")
        self._failure_policy = failure_policy
        if worker_sample_interval_seconds <= 0:
            raise ValueError(
                "worker_sample_interval_seconds must be positive"
            )
        if sched_ext_mode not in {"disabled", "auto", "strict"}:
            raise ValueError(
                "sched_ext_mode must be 'disabled', 'auto', or 'strict'"
            )
        if kernel_adapter is not None and sched_ext_mode != "disabled":
            raise ValueError(
                "kernel_adapter and sched_ext_mode cannot both configure isolation"
            )
        self._llm = llm_backend
        self._handlers = handlers
        self._context = context_manager or ContextManager()
        self._context.set_event_hook(self._emit)
        self._bus = MessageBus()
        self._available_resources = available_resources
        self._max_batch_size = max_batch_size
        self._max_llm_batch_size = max_llm_batch_size
        self._llm_batch_wait_ms = llm_batch_wait_ms
        self._active_dag: TaskDAG | None = None
        self._event_log = (
            event_log
            if isinstance(event_log, EventLog) or event_log is None
            else EventLog(event_log)
        )
        self._recovery_policy = recovery_policy
        self._validators = dict(validators or {})
        if any(not isinstance(name, str) or not callable(check)
               for name, check in self._validators.items()):
            raise ValueError("validators must map string names to callable validators")
        self._fault_injection = (
            fault_injection
            if isinstance(fault_injection, FaultInjectionPlan) or fault_injection is None
            else FaultInjectionPlan(fault_injection)
        )
        self._worker_manager = worker_manager or SubprocessWorkerManager()
        self._worker_sample_interval_seconds = worker_sample_interval_seconds
        self._execution_mode = execution_mode
        self._cgroup_manager = cgroup_manager or CgroupV2Manager()
        self._cleanup_cgroup_on_exit = cleanup_cgroup_on_exit
        self._tool_manager = tool_manager or ToolManager()
        self._propagation_depth = propagation_depth
        self._broadcaster = broadcaster
        self._resource_usage_by_role: dict[str, ResourceUsageSummary] = {}
        self._active_resource_reservations: dict[str, ResourceQuota] = {}
        self._cgroup_feedback = CgroupFeedbackAccumulator()
        self.agents: dict[str, AgentControlBlock] = {}
        self.events: list[RuntimeEvent] = []
        self._llm_batch_pool = self._build_llm_batch_pool()
        self._scheduler_fairness_tracker = scheduler_fairness_tracker
        self._last_scheduler_policy: str | None = None

        # Kernel probe and BPF map adapter
        self._kernel_probe = kernel_probe
        if self._kernel_probe is None and enable_kernel_probe:
            self._kernel_probe = probe_sched_ext(force_mock=True)
        self._bpf_map_adapter = bpf_map_adapter or MockBPFMapAdapter()
        if kernel_adapter is None:
            base_adapter = UserSpaceKernelAdapter(
                cgroup_manager=self._cgroup_manager,
                bpf_map_adapter=self._bpf_map_adapter,
                probe_result=self._kernel_probe,
            )
            if sched_ext_mode == "disabled":
                self._kernel_adapter = base_adapter
            else:
                control = sched_ext_control or SchedExtControlClient(
                    pin_dir=sched_ext_pin_dir
                )
                self._kernel_adapter = SchedExtKernelAdapter.from_mode(
                    mode=sched_ext_mode,
                    control=control,
                    base=base_adapter,
                )
                self._kernel_probe = self._kernel_adapter.probe()
        else:
            self._kernel_adapter = kernel_adapter
            self._kernel_probe = self._kernel_adapter.probe()
        self._kernel_probe_emitted = False
        self._force_scheduler_policy = force_scheduler_policy
        self._adaptive_tuner = adaptive_tuner

    @property
    def context(self) -> ContextManager:
        return self._context

    @property
    def resource_usage_by_role(self) -> dict[str, ResourceUsageSummary]:
        return self._resource_usage_by_role

    @property
    def active_resource_reservations(self) -> dict[str, ResourceQuota]:
        return dict(self._active_resource_reservations)

    def _record_resource_usage(self, task: Task, usage: ResourceUsage) -> None:
        summary = self._resource_usage_by_role.setdefault(
            task.agent_role,
            ResourceUsageSummary(),
        )
        summary.record(usage)

    def _estimate_task_resources(self, task: Task) -> ResourceQuota | None:
        summary = self._resource_usage_by_role.get(task.agent_role)
        if summary is None:
            return None
        return summary.to_quota()

    def _build_llm_batch_pool(self) -> _LLMBatchPool | None:
        if self._max_llm_batch_size is None and self._llm_batch_wait_ms is None:
            return None
        return _LLMBatchPool(
            backend=self._llm,
            emit=self._emit,
            max_batch_size=self._max_llm_batch_size or 16,
            wait_ms=self._llm_batch_wait_ms or 0.0,
        )

    def _reservation_quota_for_task(self, task: Task) -> ResourceQuota:
        return self._estimate_task_resources(task) or task.resource_requirements

    def _active_reserved_resources(self) -> ResourceQuota:
        reserved = ResourceQuota()
        for quota in self._active_resource_reservations.values():
            self._add_resource_quota(reserved, quota)
        return reserved

    def _scheduler_available_resources(self) -> ResourceQuota | None:
        if self._available_resources is None:
            return None
        available = ResourceQuota(**self._available_resources.to_dict())
        reserved = self._active_reserved_resources()
        for name in self._reservable_resource_fields():
            available_value = getattr(available, name)
            reserved_value = getattr(reserved, name)
            if available_value is None or reserved_value is None:
                continue
            setattr(available, name, max(0, available_value - reserved_value))
        return available

    def _reserve_task_resources(self, task: Task) -> None:
        quota = self._reservation_quota_for_task(task)
        if not quota.has_limits():
            return
        self._active_resource_reservations[task.task_id] = quota
        self._emit(
            RuntimeEvent(
                event_type="resource.reserved",
                task_id=task.task_id,
                category=EventCategory.RESOURCE,
                data={
                    "quota": quota.to_dict(),
                    "active_reservations": self._active_reservation_snapshot(),
                },
            )
        )

    def _release_task_resources(self, task: Task) -> None:
        quota = self._active_resource_reservations.pop(task.task_id, None)
        if quota is None:
            return
        self._emit(
            RuntimeEvent(
                event_type="resource.released",
                task_id=task.task_id,
                category=EventCategory.RESOURCE,
                data={
                    "quota": quota.to_dict(),
                    "active_reservations": self._active_reservation_snapshot(),
                },
            )
        )

    def _active_reservation_snapshot(self) -> dict[str, dict[str, object]]:
        return {
            task_id: quota.to_dict()
            for task_id, quota in sorted(self._active_resource_reservations.items())
        }

    def _add_resource_quota(self, target: ResourceQuota, quota: ResourceQuota) -> None:
        for name in self._reservable_resource_fields():
            value = getattr(quota, name)
            if value is None:
                continue
            current = getattr(target, name) or 0
            setattr(target, name, current + value)

    def _reservable_resource_fields(self) -> tuple[str, ...]:
        return (
            "cpu_max_percent",
            "memory_max_mb",
            "pids_max",
            "io_read_bps",
            "io_write_bps",
        )

    @property
    def bus(self) -> MessageBus:
        return self._bus

    def _emit(self, event: RuntimeEvent) -> None:
        self.events.append(event)
        if self._event_log is not None:
            self._event_log.append(event)
        if self._broadcaster is not None:
            self._broadcaster.publish(event)

    def _transition_agent(
        self, acb: AgentControlBlock, new_state: AgentState
    ) -> None:
        old_state = acb.state
        if old_state == new_state:
            return
        agent_transition(acb, new_state)
        self._emit(
            agent_state_changed(
                acb.agent_id,
                old_state.value,
                new_state.value,
                task_id=acb.task_id,
                trace_id=acb.trace_id,
            )
        )

    def _propagate_failure(self, failed_task_id: str) -> None:
        if self._active_dag is None:
            return

        downstream = self._active_dag.get_downstream_tasks(
            failed_task_id,
            max_depth=self._propagation_depth,
        )

        for task_id, depth in downstream:
            task = self._active_dag.tasks[task_id]
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.SKIPPED
                task.updated_at = time.time()
                self._emit(
                    RuntimeEvent(
                        event_type="fault.propagated",
                        task_id=task_id,
                        category=EventCategory.FAULT,
                        severity=EventSeverity.WARNING,
                        data={
                            "source_task_id": failed_task_id,
                            "depth": depth,
                            "reason": "dependency failed",
                        },
                    )
                )

    def _create_agent_for_task(self, task: Task) -> AgentControlBlock:
        agent_id = f"agent-{task.task_id}"
        if task.assigned_agent_id is not None and task.assigned_agent_id != agent_id:
            raise RuntimeError(
                f"Task {task.task_id} is already assigned to {task.assigned_agent_id}"
            )
        if agent_id in self.agents:
            raise RuntimeError(
                f"Agent {agent_id} already exists for task {task.task_id}"
            )
        acb = AgentControlBlock(
            agent_id=agent_id,
            name=task.name,
            role=task.agent_role,
            task_id=task.task_id,
            context_refs=list(task.context_refs),
            resource_quota=task.resource_requirements,
            metadata=(
                {"source": deepcopy(task.metadata["source"])}
                if "source" in task.metadata else {}
            ),
        )
        self.agents[acb.agent_id] = acb
        task.assigned_agent_id = acb.agent_id
        self._emit(
            RuntimeEvent(
                event_type="agent.created",
                agent_id=acb.agent_id,
                task_id=task.task_id,
                trace_id=acb.trace_id,
                category=EventCategory.AGENT,
                data={
                    "name": acb.name,
                    "role": acb.role,
                    "context_refs": list(acb.context_refs),
                    "resource_quota": acb.resource_quota.to_dict(),
                    "dependencies": list(task.dependencies),
                    "execution_mode": self._task_execution_mode(task),
                    **(
                        {"source": deepcopy(acb.metadata["source"])}
                        if "source" in acb.metadata else {}
                    ),
                },
            )
        )
        return acb

    def _submit_dynamic_task(self, task: Task, parent_acb: AgentControlBlock) -> None:
        if self._active_dag is None:
            raise RuntimeError("No active DAG is running")
        self._active_dag.add_dynamic_task(task)
        self._active_dag.compute_critical_path_scores()
        self._emit(
            RuntimeEvent(
                event_type="task.dynamic_submitted",
                agent_id=parent_acb.agent_id,
                task_id=task.task_id,
                trace_id=parent_acb.trace_id,
                category=EventCategory.TASK,
                data={
                    "name": task.name,
                    "agent_role": task.agent_role,
                    "dependencies": list(task.dependencies),
                    "parent_task_id": parent_acb.task_id,
                    "parent_agent_id": parent_acb.agent_id,
                    "context_refs": list(task.context_refs),
                    "resource_requirements": task.resource_requirements.to_dict(),
                },
            )
        )

    def _submit_dynamic_tasks(
        self,
        tasks: list[Task],
        parent_acb: AgentControlBlock,
    ) -> None:
        if self._active_dag is None:
            raise RuntimeError("No active DAG is running")
        self._active_dag.add_dynamic_tasks(tasks)
        self._active_dag.compute_critical_path_scores()
        self._emit(
            RuntimeEvent(
                event_type="task.dynamic_batch_submitted",
                agent_id=parent_acb.agent_id,
                task_id=parent_acb.task_id,
                trace_id=parent_acb.trace_id,
                category=EventCategory.TASK,
                data={
                    "parent_task_id": parent_acb.task_id,
                    "parent_agent_id": parent_acb.agent_id,
                    "batch_size": len(tasks),
                    "task_ids": [task.task_id for task in tasks],
                    "tasks": [
                        {
                            "task_id": task.task_id,
                            "name": task.name,
                            "agent_role": task.agent_role,
                            "dependencies": list(task.dependencies),
                            "context_refs": list(task.context_refs),
                            "resource_requirements": (
                                task.resource_requirements.to_dict()
                            ),
                        }
                        for task in tasks
                    ],
                },
            )
        )

    def _cancel_dynamic_task(
        self,
        task_id: str,
        parent_acb: AgentControlBlock,
        reason: str | None = None,
    ) -> None:
        if self._active_dag is None:
            raise RuntimeError("No active DAG is running")
        task = self._active_dag.cancel_task(task_id, reason=reason)
        self._active_dag.compute_critical_path_scores()
        self._emit_task_cancelled(task, parent_acb, reason)

    def _cancel_dynamic_downstream(
        self,
        task_id: str,
        parent_acb: AgentControlBlock,
        reason: str | None = None,
    ) -> list[str]:
        if self._active_dag is None:
            raise RuntimeError("No active DAG is running")
        cancelled = self._active_dag.cancel_downstream(task_id, reason=reason)
        self._active_dag.compute_critical_path_scores()
        for task in cancelled:
            self._emit_task_cancelled(task, parent_acb, reason)
        return [task.task_id for task in cancelled]

    def _replace_dynamic_downstream(
        self,
        task_id: str,
        replacement_tasks: list[Task],
        parent_acb: AgentControlBlock,
        reason: str | None = None,
    ) -> tuple[list[str], list[str]]:
        if self._active_dag is None:
            raise RuntimeError("No active DAG is running")
        cancelled, replacements = self._active_dag.replace_downstream(
            task_id,
            replacement_tasks,
            reason=reason,
        )
        for task in cancelled:
            self._emit_task_cancelled(task, parent_acb, reason)
        self._emit(
            RuntimeEvent(
                event_type="task.dynamic_branch_replaced",
                agent_id=parent_acb.agent_id,
                task_id=task_id,
                trace_id=parent_acb.trace_id,
                category=EventCategory.TASK,
                data={
                    "parent_task_id": parent_acb.task_id,
                    "parent_agent_id": parent_acb.agent_id,
                    "source_task_id": task_id,
                    "reason": reason,
                    "cancelled_task_ids": [task.task_id for task in cancelled],
                    "replacement_task_ids": [task.task_id for task in replacements],
                    "replacement_tasks": [
                        {
                            "task_id": task.task_id,
                            "name": task.name,
                            "agent_role": task.agent_role,
                            "dependencies": list(task.dependencies),
                            "context_refs": list(task.context_refs),
                            "resource_requirements": (
                                task.resource_requirements.to_dict()
                            ),
                        }
                        for task in replacements
                    ],
                },
            )
        )
        return [task.task_id for task in cancelled], [
            task.task_id for task in replacements
        ]

    def _emit_task_cancelled(
        self,
        task: Task,
        parent_acb: AgentControlBlock,
        reason: str | None,
    ) -> None:
        self._release_task_resources(task)
        self._emit(
            RuntimeEvent(
                event_type="task.cancelled",
                agent_id=parent_acb.agent_id,
                task_id=task.task_id,
                trace_id=parent_acb.trace_id,
                category=EventCategory.TASK,
                data={
                    "reason": reason,
                    "parent_task_id": parent_acb.task_id,
                    "parent_agent_id": parent_acb.agent_id,
                },
            )
        )

    def _task_execution_mode(self, task: Task) -> str:
        return str(task.metadata.get("execution", self._execution_mode))

    @staticmethod
    def _clamp_int(
        value: Any,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _clamp_float(
        value: Any,
        *,
        default: float,
        minimum: float,
        maximum: float,
    ) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    def _memory_pressure_fault_data(self, data: dict[str, Any]) -> dict[str, Any]:
        memory_mb = self._clamp_int(
            data.get("memory_mb"),
            default=8,
            minimum=1,
            maximum=256,
        )
        chunk_mb = self._clamp_int(
            data.get("chunk_mb"),
            default=1,
            minimum=1,
            maximum=memory_mb,
        )
        hold_seconds = self._clamp_float(
            data.get("hold_seconds"),
            default=0.05,
            minimum=0.0,
            maximum=5.0,
        )
        return {
            "fault_type": "memory_pressure",
            "memory_mb": memory_mb,
            "chunk_mb": chunk_mb,
            "hold_seconds": hold_seconds,
        }

    def _emit_memory_pressure_fault(
        self,
        acb: AgentControlBlock,
        task: Task,
        fault: dict[str, Any],
        *,
        execution: str,
    ) -> None:
        self._emit(
            RuntimeEvent(
                event_type="fault.memory_pressure",
                agent_id=acb.agent_id,
                task_id=task.task_id,
                trace_id=acb.trace_id,
                category=EventCategory.FAULT,
                severity=EventSeverity.WARNING,
                data={**fault, "execution": execution},
            )
        )

    async def _run_subprocess_worker(
        self,
        task: Task,
        acb: AgentControlBlock,
        *,
        attempt: int,
        kill_after_start: bool = False,
        memory_pressure_fault: dict[str, Any] | None = None,
        recovery_feedback: list[dict[str, Any]] | None = None,
    ) -> Any:
        worker_type = str(task.metadata.get("worker_type", task.agent_role))
        payload = dict(task.metadata.get("worker_payload", {}))
        payload.setdefault("task_name", task.name)
        if recovery_feedback:
            payload["recovery_feedback"] = deepcopy(recovery_feedback)
        cgroup_path: Path | None = None
        cgroup_available = False
        sched_ext_bound_pid: int | None = None
        sampling_started_at: float | None = None
        sample_index = 0
        spec = WorkerSpec(
            worker_type=worker_type,
            task_id=task.task_id,
            agent_id=acb.agent_id,
            trace_id=acb.trace_id,
            payload=payload,
            resource_quota=task.resource_requirements,
            attempt=attempt,
            fault=memory_pressure_fault,
        )

        def sample_phase(phase: str) -> None:
            nonlocal sample_index
            if (
                not cgroup_available
                or cgroup_path is None
                or sampling_started_at is None
            ):
                return
            elapsed_ms = (time.monotonic() - sampling_started_at) * 1000
            self._sample_worker_usage(
                acb,
                task,
                cgroup_path,
                phase=phase,
                sample_index=sample_index,
                elapsed_ms=elapsed_ms,
            )
            sample_index += 1

        def on_start(pid: int) -> None:
            nonlocal cgroup_available, cgroup_path, sampling_started_at
            nonlocal sched_ext_bound_pid
            acb.pid = pid
            self._emit(
                RuntimeEvent(
                    event_type="worker.started",
                    agent_id=acb.agent_id,
                    task_id=task.task_id,
                    trace_id=acb.trace_id,
                    category=EventCategory.AGENT,
                    data={
                        "pid": pid,
                        "worker_type": worker_type,
                        "execution": "subprocess",
                        "attempt": attempt,
                    },
                )
            )
            apply_result = self._kernel_adapter.apply_agent_quota(
                agent_id=acb.agent_id,
                task_id=task.task_id,
                pid=pid,
                quota=task.resource_requirements,
            )
            cgroup_path = apply_result.cgroup_path
            cgroup_available = apply_result.available
            if cgroup_path is not None:
                acb.cgroup = str(cgroup_path)

            for event in apply_result.events:
                self._emit(
                    RuntimeEvent(
                        event_type=event.event_type,
                        agent_id=acb.agent_id,
                        task_id=task.task_id,
                        trace_id=acb.trace_id,
                        category=event.category,
                        severity=event.severity,
                        data=event.data,
                    )
                )
            if cgroup_available and cgroup_path is not None:
                sampling_started_at = time.monotonic()
                sample_phase("initial")

            if self._kernel_adapter.supports_sched_ext_binding:
                self._maintain_sched_ext(trigger="worker_bind")
                if not cgroup_available:
                    reason = apply_result.reason or "worker cgroup setup is unavailable"
                    self._emit(
                        RuntimeEvent(
                            event_type="kernel.sched_ext_attach_failed",
                            agent_id=acb.agent_id,
                            task_id=task.task_id,
                            trace_id=acb.trace_id,
                            category=EventCategory.SCHEDULER,
                            severity=(
                                EventSeverity.ERROR
                                if self._kernel_adapter.strict_sched_ext
                                else EventSeverity.WARNING
                            ),
                            data={"pid": pid, "reason": reason, "step": "cgroup"},
                        )
                    )
                    if self._kernel_adapter.strict_sched_ext:
                        raise RuntimeError(f"sched_ext cgroup setup failed: {reason}")
                    return

                bind_result = self._kernel_adapter.bind_worker_policy(
                    agent_id=acb.agent_id,
                    task_id=task.task_id,
                    pid=pid,
                )
                for event in bind_result.events:
                    self._emit_resource_event(event, acb, task)
                if bind_result.attached:
                    sched_ext_bound_pid = pid
                elif self._kernel_adapter.strict_sched_ext:
                    raise RuntimeError(
                        "sched_ext worker attach failed: "
                        f"{bind_result.reason or 'unknown error'}"
                    )

        try:
            result = await self._worker_manager.run(
                spec,
                timeout_seconds=task.resource_requirements.timeout_seconds,
                on_start=on_start,
                on_sample=lambda: sample_phase("running"),
                sample_interval_seconds=self._worker_sample_interval_seconds,
                kill_after_start=kill_after_start,
            )
            if cgroup_available and cgroup_path is not None:
                sample_phase("final")
            if result.timed_out:
                self._emit_worker_event("worker.timeout", acb, task, result)
                raise AgentTimeoutError(
                    task.resource_requirements.timeout_seconds or 0.0,
                    agent_id=acb.agent_id,
                    task_id=task.task_id,
                )
            if not result.ok:
                self._emit_worker_event("worker.failed", acb, task, result)
                raise SubprocessWorkerError(result.error or "worker failed")

            self._emit_worker_event("worker.finished", acb, task, result)
            return result.result
        finally:
            maintenance_error: BaseException | None = None
            try:
                self._maintain_sched_ext(trigger="worker_cleanup")
            except BaseException as exc:
                maintenance_error = exc
            if sched_ext_bound_pid is not None:
                for event in self._kernel_adapter.remove_worker_policy(
                    agent_id=acb.agent_id,
                    task_id=task.task_id,
                    pid=sched_ext_bound_pid,
                ):
                    self._emit_resource_event(event, acb, task)
                if maintenance_error is None:
                    try:
                        self._maintain_sched_ext(trigger="worker_cleanup_complete")
                    except BaseException as exc:
                        maintenance_error = exc
            if (
                cgroup_available
                and cgroup_path is not None
                and self._cleanup_cgroup_on_exit
            ):
                self._cleanup_worker_cgroup(acb, task, cgroup_path)
            if maintenance_error is not None:
                raise maintenance_error

    def _emit_resource_event(
        self,
        event: RuntimeEvent,
        acb: AgentControlBlock,
        task: Task,
    ) -> None:
        self._emit(
            RuntimeEvent(
                event_type=event.event_type,
                agent_id=acb.agent_id,
                task_id=task.task_id,
                trace_id=acb.trace_id,
                category=event.category,
                severity=event.severity,
                data=event.data,
            )
        )

    def _sample_pressure_safely(
        self, *, cgroup_path: Path | None, scope: str,
    ) -> PressureSample:
        """Sample pressure with unified exception handling.

        Exceptions degrade to a normal PressureSample so the runtime
        never fails due to pressure sampling issues.
        """
        self._maintain_sched_ext(trigger=f"pressure:{scope}")
        try:
            return self._kernel_adapter.sample_pressure(cgroup_path=cgroup_path)
        except Exception as exc:
            return PressureSample(
                pressure_level="normal",
                reason=f"{scope} pressure sampling failed: {type(exc).__name__}: {exc}",
            )

    def _maintain_sched_ext(self, *, trigger: str) -> None:
        maintain = getattr(self._kernel_adapter, "maintain_sched_ext", None)
        if maintain is None:
            return
        result = maintain(trigger=trigger)
        for event in result.events:
            self._emit(event)
        if result.fatal:
            raise RuntimeError(
                f"sched_ext lifecycle failure: {result.reason or result.state.value}"
            )

    def _sample_system_pressure(self) -> PressureSample:
        """Sample system-level PSI pressure (no cgroup path)."""
        return self._sample_pressure_safely(cgroup_path=None, scope="system")

    def _emit_pressure_event(
        self,
        *,
        scope: str,
        sample: PressureSample,
        acb: AgentControlBlock | None,
        task: Task | None,
        cgroup_path: Path | None,
        phase: str | None = None,
        sample_index: int | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        """Emit a resource.pressure_sampled event."""
        data: dict[str, Any] = {
            "scope": scope,
            "pressure": sample.to_dict(),
        }
        if cgroup_path is not None:
            data["path"] = str(cgroup_path)
        if phase is not None:
            data.update(
                {
                    "phase": phase,
                    "sample_index": sample_index,
                    "elapsed_ms": elapsed_ms,
                }
            )
        self._emit(
            RuntimeEvent(
                event_type="resource.pressure_sampled",
                agent_id=acb.agent_id if acb is not None else None,
                task_id=task.task_id if task is not None else None,
                trace_id=acb.trace_id if acb is not None else None,
                category=EventCategory.RESOURCE,
                data=data,
            )
        )

    def _sample_worker_usage(
        self,
        acb: AgentControlBlock,
        task: Task,
        cgroup_path: Path,
        *,
        phase: str = "final",
        sample_index: int = 0,
        elapsed_ms: float = 0.0,
    ) -> None:
        sample = self._sample_pressure_safely(
            cgroup_path=cgroup_path, scope="agent_cgroup",
        )
        usage = sample.cgroup_usage if sample.cgroup_usage is not None else ResourceUsage()
        acb.resource_usage = usage
        self._record_resource_usage(task, usage)
        self._emit(
            RuntimeEvent(
                event_type="resource.usage_sampled",
                agent_id=acb.agent_id,
                task_id=task.task_id,
                trace_id=acb.trace_id,
                category=EventCategory.RESOURCE,
                data={
                    "path": str(cgroup_path),
                    "phase": phase,
                    "sample_index": sample_index,
                    "elapsed_ms": elapsed_ms,
                    "usage": usage.to_dict(),
                },
            )
        )
        # Emit agent_cgroup pressure event
        self._emit_pressure_event(
            scope="agent_cgroup",
            sample=sample,
            acb=acb,
            task=task,
            cgroup_path=cgroup_path,
            phase=phase,
            sample_index=sample_index,
            elapsed_ms=elapsed_ms,
        )
        self._cgroup_feedback.observe(
            sample,
            agent_id=acb.agent_id,
            task_id=task.task_id,
            phase=phase,
            sample_index=sample_index,
        )

    def _apply_scheduler_pressure(
        self,
        scheduler: DAGScheduler,
        system_sample: PressureSample,
        *,
        scheduler_round: int,
    ) -> None:
        local = self._cgroup_feedback.consume()
        if local is None:
            sample_data = {
                **system_sample.to_dict(),
                "pressure_source": "system",
                "scheduler_round": scheduler_round,
            }
            scheduler.set_pressure(
                pressure_level=system_sample.pressure_level,
                pressure_sample=sample_data,
            )
            return

        effective_level = max_pressure_level(
            system_sample.pressure_level,
            local.pressure_level,
        )
        combined = self._combined_pressure_sample(
            system_sample,
            local,
            effective_level=effective_level,
            scheduler_round=scheduler_round,
        )
        scheduler.set_pressure(
            pressure_level=effective_level,
            pressure_sample=combined,
        )
        self._emit(
            RuntimeEvent(
                event_type="resource.cgroup_feedback_applied",
                category=EventCategory.RESOURCE,
                data={
                    **local.to_dict(),
                    "system_pressure_level": system_sample.pressure_level,
                    "local_pressure_level": local.pressure_level,
                    "effective_pressure_level": effective_level,
                    "pressure_source": "system+cgroup",
                    "scheduler_round": scheduler_round,
                },
            )
        )

    @staticmethod
    def _combined_pressure_sample(
        system_sample: PressureSample,
        local: CgroupFeedbackSnapshot,
        *,
        effective_level: str,
        scheduler_round: int,
    ) -> dict[str, Any]:
        local_data = local.to_dict()
        return {
            "cpu_some_avg10": max(
                system_sample.cpu_some_avg10,
                local.cpu_some_avg10,
            ),
            "memory_some_avg10": max(
                system_sample.memory_some_avg10,
                local.memory_some_avg10,
            ),
            "io_some_avg10": max(
                system_sample.io_some_avg10,
                local.io_some_avg10,
            ),
            "pressure_level": effective_level,
            "effective_pressure_level": effective_level,
            "pressure_source": "system+cgroup",
            "scheduler_round": scheduler_round,
            "feedback_id": local.feedback_id,
            "system": system_sample.to_dict(),
            "cgroup_local": local_data,
        }

    def _cleanup_worker_cgroup(
        self,
        acb: AgentControlBlock,
        task: Task,
        cgroup_path: Path,
    ) -> None:
        events = self._kernel_adapter.cleanup_agent(
            agent_id=acb.agent_id,
            cgroup_path=cgroup_path,
        )
        for event in events:
            self._emit_resource_event(event, acb, task)

    def _emit_worker_event(
        self,
        event_type: str,
        acb: AgentControlBlock,
        task: Task,
        result: WorkerResult,
    ) -> None:
        self._emit(
            RuntimeEvent(
                event_type=event_type,
                agent_id=acb.agent_id,
                task_id=task.task_id,
                trace_id=acb.trace_id,
                category=EventCategory.AGENT,
                severity=(
                    EventSeverity.ERROR
                    if event_type in {"worker.failed", "worker.timeout"}
                    else EventSeverity.INFO
                ),
                data={
                    "pid": result.pid,
                    "worker_type": result.worker_type,
                    "exit_code": result.exit_code,
                    "duration_secs": result.duration_secs,
                    "timed_out": result.timed_out,
                    "killed": result.killed,
                    "error": result.error,
                    "result_preview": str(result.result)[:200],
                },
            )
        )

    async def _run_task(self, task: Task) -> Any:
        validator_name = task.metadata.get("validator")
        validator = None
        if validator_name is not None:
            if not isinstance(validator_name, str) or validator_name not in self._validators:
                raise ValueError(f"unregistered validator: {validator_name!r}")
            validator = self._validators[validator_name]
        execution_mode = self._task_execution_mode(task)
        use_subprocess = execution_mode == "subprocess"
        handler = self._handlers.get(task.agent_role)
        if handler is None and not use_subprocess:
            raise ValueError(f"No handler registered for role: {task.agent_role}")

        acb = self._create_agent_for_task(task)
        self._transition_agent(acb, AgentState.READY)
        self._transition_agent(acb, AgentState.RUNNING)
        self._emit(
            RuntimeEvent(
                event_type="task.started",
                agent_id=acb.agent_id,
                task_id=task.task_id,
                trace_id=acb.trace_id,
                category=EventCategory.TASK,
                data={"execution_mode": execution_mode},
            )
        )

        ctx = AgentHandler.Context(
            task=task,
            acb=acb,
            bus=self._bus,
            context_manager=self._context,
            _llm=self._llm,
            _emit=self._emit,
            _transition=self._transition_agent,
            _submit_task=self._submit_dynamic_task,
            _submit_tasks=self._submit_dynamic_tasks,
            _cancel_task=self._cancel_dynamic_task,
            _cancel_downstream=self._cancel_dynamic_downstream,
            _replace_downstream=self._replace_dynamic_downstream,
            _llm_batch_submit=(
                self._llm_batch_pool.submit
                if self._llm_batch_pool is not None
                else None
            ),
            _tools=self._tool_manager,
        )

        active_handler = handler
        initial_context_refs = list(acb.context_refs)
        attempt = 0
        previous_result_hash: str | None = None
        recovered_failure_type: str | None = None
        recovered_reason: str = ""
        recovered_action: str | None = None
        recovery_started_at: float | None = None
        injected_fault = (
            self._fault_injection.pop_for_task(task.task_id, agent_id=acb.agent_id)
            if self._fault_injection is not None
            else None
        )
        subprocess_kill_once = False
        subprocess_memory_pressure_once: dict[str, Any] | None = None
        if injected_fault is not None:
            self._emit(injected_fault)
            fault_type = injected_fault.data["fault_type"]
            self._emit(
                RuntimeEvent(
                    event_type="fault.triggered",
                    agent_id=acb.agent_id,
                    task_id=task.task_id,
                    trace_id=acb.trace_id,
                    category=EventCategory.FAULT,
                    severity=EventSeverity.WARNING,
                    data=dict(injected_fault.data),
                )
            )
            if fault_type == "kill" and use_subprocess:
                subprocess_kill_once = True
                injected_error: BaseException | None = None
            elif fault_type == "kill":
                injected_error: BaseException | None = RuntimeError(
                    f"injected kill fault for task {task.task_id}"
                )
            elif fault_type == "memory_pressure":
                memory_pressure_fault = self._memory_pressure_fault_data(
                    injected_fault.data
                )
                self._emit_memory_pressure_fault(
                    acb,
                    task,
                    memory_pressure_fault,
                    execution=execution_mode,
                )
                if use_subprocess:
                    subprocess_memory_pressure_once = memory_pressure_fault
                    injected_error = None
                else:
                    injected_error = MemoryError(
                        "injected memory pressure fault "
                        f"allocated {memory_pressure_fault['memory_mb']} MiB"
                    )
            else:
                injected_error = AgentTimeoutError(
                    0.0,
                    agent_id=acb.agent_id,
                    task_id=task.task_id,
                )
        else:
            injected_error = None

        context_snapshot = self._context.export_agent_context(acb.agent_id)
        context_snapshot_tokens = sum(
            len(seg.get("content", "")) // 4
            for seg in context_snapshot.get("shared", []) + context_snapshot.get("private", [])
        )
        self._emit(
            RuntimeEvent(
                event_type="fault.context_snapshot_saved",
                agent_id=acb.agent_id,
                task_id=task.task_id,
                trace_id=acb.trace_id,
                category=EventCategory.FAULT,
                data={
                    "shared_refs": len(context_snapshot.get("shared", [])),
                    "private_refs": len(context_snapshot.get("private", [])),
                    "estimated_tokens": context_snapshot_tokens,
                },
            )
        )

        while True:
            ctx.attempt = attempt + 1
            ctx._tool_call_counts.clear()
            ctx._repetition_error = None
            prior_result_hash = previous_result_hash
            previous_result_hash = None
            async def perform_attempt() -> Any:
                nonlocal injected_error, subprocess_kill_once
                nonlocal subprocess_memory_pressure_once, previous_result_hash
                if injected_error is not None:
                    error = injected_error
                    injected_error = None
                    raise error
                if use_subprocess:
                    kill_after_start = subprocess_kill_once
                    subprocess_kill_once = False
                    memory_pressure_fault = subprocess_memory_pressure_once
                    subprocess_memory_pressure_once = None
                    result = await self._run_subprocess_worker(
                        task,
                        acb,
                        attempt=attempt,
                        kill_after_start=kill_after_start,
                        memory_pressure_fault=memory_pressure_fault,
                        recovery_feedback=ctx.recovery_feedback,
                    )
                else:
                    if active_handler is None:
                        raise ValueError(
                            f"No handler registered for role: {task.agent_role}"
                        )
                    result = await active_handler.run(ctx)
                if ctx._repetition_error is not None:
                    raise ctx._repetition_error
                if validator is not None:
                    self._emit(RuntimeEvent(
                        event_type="fault.validation_started", agent_id=acb.agent_id,
                        task_id=task.task_id, trace_id=acb.trace_id,
                        category=EventCategory.FAULT,
                        data={"validator": validator_name, "attempt": ctx.attempt,
                              "failure_type": None, "reason": ""},
                    ))
                    try:
                        validation = validator(result)
                        if inspect.isawaitable(validation):
                            validation = await validation
                        if not isinstance(validation, ValidationResult):
                            raise TypeError("validator must return ValidationResult")
                    except asyncio.CancelledError:
                        self._emit(RuntimeEvent(
                            event_type="fault.validation_completed", agent_id=acb.agent_id,
                            task_id=task.task_id, trace_id=acb.trace_id,
                            category=EventCategory.FAULT, severity=EventSeverity.WARNING,
                            data={"validator": validator_name, "passed": False,
                                  "failure_type": "execution", "reason": "validation cancelled",
                                  "attempt": ctx.attempt},
                        ))
                        raise
                    except Exception as exc:
                        error = ValidationExecutionError(f"validator {validator_name!r} raised: {exc}")
                        self._emit(RuntimeEvent(
                            event_type="fault.validation_completed", agent_id=acb.agent_id,
                            task_id=task.task_id, trace_id=acb.trace_id,
                            category=EventCategory.FAULT, severity=EventSeverity.ERROR,
                            data={"validator": validator_name, "passed": False,
                                  "failure_type": "validation_error", "reason": str(error),
                                  "attempt": ctx.attempt},
                        ))
                        raise error from exc
                    self._emit(RuntimeEvent(
                        event_type="fault.validation_completed", agent_id=acb.agent_id,
                        task_id=task.task_id, trace_id=acb.trace_id,
                        category=EventCategory.FAULT,
                        severity=EventSeverity.INFO if validation.passed else EventSeverity.WARNING,
                        data={"validator": validator_name, "passed": validation.passed,
                              "failure_type": None if validation.passed else "semantic",
                              "reason": validation.reason, "attempt": ctx.attempt},
                    ))
                    if not validation.passed:
                        current_hash = normalized_result_hash(result)
                        if prior_result_hash == current_hash:
                            raise RepetitionError(
                                "consecutive rejected attempts produced the same normalized result: "
                                + validation.reason
                            )
                        previous_result_hash = current_hash
                        raise SemanticValidationError(validation.reason or "task result rejected by validator")
                accept_result = getattr(active_handler, "accept_result", None)
                if not use_subprocess and accept_result is not None:
                    await accept_result(ctx, result)
                return result

            try:
                result = await run_with_timeout(
                    perform_attempt(),
                    (None if use_subprocess and validator is None
                     else task.resource_requirements.timeout_seconds),
                    agent_id=acb.agent_id, task_id=task.task_id,
                )
                self._transition_agent(acb, AgentState.FINISHED)
                if recovered_action is not None:
                    acb.stats.recoveries += 1
                    recovery_time_secs = (
                        time.monotonic() - recovery_started_at
                        if recovery_started_at is not None
                        else 0.0
                    )
                    self._emit(
                        RuntimeEvent(
                            event_type="fault.recovery_completed",
                            agent_id=acb.agent_id,
                            task_id=task.task_id,
                            trace_id=acb.trace_id,
                            category=EventCategory.FAULT,
                            data={
                                "action": recovered_action,
                                "attempt": ctx.attempt if validator is not None else attempt,
                                "failure_type": recovered_failure_type,
                                "reason": recovered_reason,
                                "recovery_time_secs": recovery_time_secs,
                            },
                        )
                    )
                self._emit(
                    task_completed(
                        task.task_id,
                        acb.agent_id,
                        result,
                        trace_id=acb.trace_id,
                    )
                )
                return result
            except AgentTimeoutError as exc:
                self._emit(
                    fault_timeout(
                        agent_id=acb.agent_id,
                        task_id=task.task_id,
                        trace_id=acb.trace_id,
                        timeout_seconds=exc.timeout_seconds,
                    )
                )
                error: BaseException = exc
            except Exception as exc:
                error = exc

            # A handler may catch a tool-call exception; the repetition latch is
            # terminal even if it then raises something else or returns a value.
            if ctx._repetition_error is not None:
                error = ctx._repetition_error
            if acb.state != AgentState.FAILED:
                self._transition_agent(acb, AgentState.FAILED)
            acb.stats.failures += 1

            kind = failure_type(error)
            if kind == "repetition":
                self._emit(RuntimeEvent(
                    event_type="fault.repetition_detected", agent_id=acb.agent_id,
                    task_id=task.task_id, trace_id=acb.trace_id,
                    category=EventCategory.FAULT, severity=EventSeverity.ERROR,
                    data={"failure_type": kind, "reason": str(error), "attempt": ctx.attempt},
                ))
            if kind in {"repetition", "validation_error"}:
                self._emit(RuntimeEvent(
                    event_type="fault.recovery_stopped", agent_id=acb.agent_id,
                    task_id=task.task_id, trace_id=acb.trace_id,
                    category=EventCategory.FAULT, severity=EventSeverity.ERROR,
                    data={"failure_type": kind, "reason": str(error), "attempt": ctx.attempt},
                ))
                self._propagate_failure(task.task_id)
                raise error
            if self._recovery_policy is None:
                if validator is not None:
                    self._emit(RuntimeEvent(
                        event_type="fault.recovery_stopped", agent_id=acb.agent_id,
                        task_id=task.task_id, trace_id=acb.trace_id,
                        category=EventCategory.FAULT, severity=EventSeverity.ERROR,
                        data={"failure_type": kind, "reason": str(error),
                              "attempt": ctx.attempt},
                    ))
                self._propagate_failure(task.task_id)
                raise error

            attempt += 1
            recovery_started_at = time.monotonic()
            decision = self._recovery_policy.decide(error, attempt)
            recovered_failure_type = kind
            recovered_reason = str(error)
            self._emit(
                RuntimeEvent(
                    event_type="fault.recovery_started",
                    agent_id=acb.agent_id,
                    task_id=task.task_id,
                    trace_id=acb.trace_id,
                    category=EventCategory.FAULT,
                    severity=EventSeverity.WARNING,
                    data={
                        "action": decision.action.value,
                        "attempt": attempt,
                        "error": str(error),
                        "failure_type": kind,
                        "failure_reason": str(error),
                        "reason": decision.reason,
                    },
                )
            )

            if decision.action == RecoveryAction.FAIL:
                self._emit(
                    RuntimeEvent(
                        event_type="fault.recovery_failed",
                        agent_id=acb.agent_id,
                        task_id=task.task_id,
                        trace_id=acb.trace_id,
                        category=EventCategory.FAULT,
                        severity=EventSeverity.ERROR,
                        data={
                            "attempt": attempt,
                            "error": str(error),
                            "failure_type": kind,
                            "failure_reason": str(error),
                            "reason": decision.reason,
                        },
                    )
                )
                self._propagate_failure(task.task_id)
                raise error

            self._transition_agent(acb, AgentState.RECOVERING)

            if decision.action == RecoveryAction.SKIP:
                self._transition_agent(acb, AgentState.READY)
                self._transition_agent(acb, AgentState.RUNNING)
                self._transition_agent(acb, AgentState.FINISHED)
                acb.stats.recoveries += 1
                recovery_time_secs = (
                    time.monotonic() - recovery_started_at
                    if recovery_started_at is not None
                    else 0.0
                )
                self._emit(
                    RuntimeEvent(
                        event_type="fault.recovery_completed",
                        agent_id=acb.agent_id,
                        task_id=task.task_id,
                        trace_id=acb.trace_id,
                        category=EventCategory.FAULT,
                        data={
                            "action": "skip",
                            "attempt": attempt,
                            "failure_type": kind,
                            "reason": str(error),
                            "recovery_time_secs": recovery_time_secs,
                        },
                    )
                )
                self._emit(
                    task_completed(
                        task.task_id,
                        acb.agent_id,
                        None,
                        trace_id=acb.trace_id,
                    )
                )
                return None

            if decision.action == RecoveryAction.CONTEXT_RETRY:
                self._context.import_agent_context(context_snapshot, restore_shared=False)
                if ctx._context_refs_granted:
                    acb.context_refs = list(initial_context_refs)
                context_snapshot = self._context.export_agent_context(acb.agent_id)
                restored_refs = len(context_snapshot.get("private", []))
                restored_tokens = sum(
                    len(seg.get("content", "")) // 4
                    for seg in context_snapshot.get("private", [])
                )
                self._emit(
                    RuntimeEvent(
                        event_type="fault.context_restored",
                        agent_id=acb.agent_id,
                        task_id=task.task_id,
                        trace_id=acb.trace_id,
                        category=EventCategory.FAULT,
                        data={
                            "context_refs_restored": restored_refs,
                            "failure_type": kind,
                            "reason": str(error),
                            "attempt": ctx.attempt,
                            "estimated_private_tokens_restored": restored_tokens,
                            "shared_refs_available": len(context_snapshot.get("shared", [])),
                            "shared_context_rewound": False,
                        },
                    )
                )
                recovered_action = "context_retry"
            elif decision.action == RecoveryAction.FALLBACK:
                if decision.fallback_role not in self._handlers:
                    self._emit(
                        RuntimeEvent(
                            event_type="fault.recovery_failed",
                            agent_id=acb.agent_id,
                            task_id=task.task_id,
                            trace_id=acb.trace_id,
                            category=EventCategory.FAULT,
                            severity=EventSeverity.ERROR,
                            data={
                                "attempt": attempt,
                                "error": f"missing fallback role {decision.fallback_role}",
                                "failure_type": kind,
                                "reason": str(error),
                            },
                        )
                    )
                    raise error
                active_handler = self._handlers[decision.fallback_role]
                acb.role = decision.fallback_role or acb.role
                use_subprocess = False
                recovered_action = "fallback"
            else:
                recovered_action = "retry"

            feedback = getattr(decision, "feedback", "")
            if feedback:
                message = {"role": "user", "content": feedback}
                ctx.recovery_feedback.append(message)
                await self._bus.send(Message(
                    from_agent="runtime", to_agent=acb.agent_id,
                    content={"failure_type": kind, "reason": str(error),
                             "attempt": ctx.attempt, "message": message},
                    message_type="recovery_feedback", task_id=task.task_id,
                    trace_id=acb.trace_id,
                ))
            self._emit(RuntimeEvent(
                event_type="fault.retry_scheduled", agent_id=acb.agent_id,
                task_id=task.task_id, trace_id=acb.trace_id,
                category=EventCategory.FAULT, severity=EventSeverity.WARNING,
                data={"failure_type": kind, "reason": str(error),
                      "attempt": ctx.attempt, "next_attempt": ctx.attempt + 1,
                      "action": decision.action.value, "feedback": feedback},
            ))
            self._transition_agent(acb, AgentState.READY)
            self._transition_agent(acb, AgentState.RUNNING)

    def _build_affinity_advisor(self) -> CacheAffinityAdvisor | None:
        tracker: CacheHeatTracker | None = None
        if isinstance(self._llm, ModelGateway):
            tracker = self._llm.cache_tracker
        if tracker is None:
            return None
        return CacheAffinityAdvisor(self._context, tracker)

    def _emit_scheduler_decision(self, decision: dict[str, Any] | None) -> None:
        if decision is None:
            return
        has_visible_work = any(
            decision.get(key)
            for key in (
                "ready_task_ids",
                "selected_task_ids",
                "blocked_task_ids",
                "deferred_task_ids",
            )
        )
        if not has_visible_work:
            return

        # Keep the scheduler decision kernel-agnostic. Runtime adds the
        # adapter evidence only to the emitted event after publication.
        event_data = dict(decision)
        adapter_snapshot = self._kernel_adapter.snapshot()
        bpf_map = adapter_snapshot.get("bpf_map")
        if isinstance(bpf_map, dict):
            event_data["bpf_map"] = bpf_map

        # Emit scheduler.decision event
        self._emit(
            RuntimeEvent(
                event_type="scheduler.decision",
                category=EventCategory.SCHEDULER,
                severity=EventSeverity.DEBUG,
                data=event_data,
            )
        )

        # Emit scheduler.policy_selected event when policy changes
        current_policy = decision.get("policy")
        if current_policy is not None and current_policy != self._last_scheduler_policy:
            self._last_scheduler_policy = current_policy
            self._emit(
                RuntimeEvent(
                    event_type="scheduler.policy_selected",
                    category=EventCategory.SCHEDULER,
                    severity=EventSeverity.INFO,
                    data={
                        "policy": current_policy,
                        "workload_class": decision.get("workload_class"),
                        "reason": decision.get("policy_reason"),
                        "candidate_count": len(decision.get("candidate_task_ids", [])),
                        "fairness": decision.get("policy_score", {}).get("fairness", {}),
                    },
                )
            )

        # Emit scheduler.fairness_warning events
        fairness_warnings = decision.get("fairness_warnings", [])
        for warning in fairness_warnings:
            self._emit(
                RuntimeEvent(
                    event_type="scheduler.fairness_warning",
                    category=EventCategory.SCHEDULER,
                    severity=EventSeverity.WARNING,
                    data={
                        "task_id": warning.get("task_id"),
                        "wait_ms": warning.get("wait_ms"),
                        "deferred_count": warning.get("deferred_count"),
                        "blocked_count": warning.get("blocked_count"),
                        "starvation_risk": warning.get("starvation_risk"),
                        "suggested_policy": warning.get("suggested_policy"),
                    },
                )
            )

        # Emit scheduler.kernel_probe event (once per run)
        if not self._kernel_probe_emitted and self._kernel_probe is not None:
            self._kernel_probe_emitted = True
            self._emit(
                RuntimeEvent(
                    event_type="scheduler.kernel_probe",
                    category=EventCategory.SCHEDULER,
                    severity=EventSeverity.INFO,
                    data=self._kernel_probe.to_dict(),
                )
            )

        # Emit scheduler.runtime_health event
        runtime_health = decision.get("runtime_health")
        if runtime_health is not None:
            self._emit(
                RuntimeEvent(
                    event_type="scheduler.runtime_health",
                    category=EventCategory.SCHEDULER,
                    severity=EventSeverity.INFO,
                    data=runtime_health,
                )
            )

        # Emit scheduler.sla_summary event
        sla_summary = decision.get("sla")
        if sla_summary is not None:
            self._emit(
                RuntimeEvent(
                    event_type="scheduler.sla_summary",
                    category=EventCategory.SCHEDULER,
                    severity=EventSeverity.INFO,
                    data=sla_summary,
                )
            )

        # Emit scheduler.risk_warning events
        risk_summary = decision.get("risk")
        if risk_summary is not None:
            risk_warnings = risk_summary.get("risk_warnings", [])
            for warning in risk_warnings:
                self._emit(
                    RuntimeEvent(
                        event_type="scheduler.risk_warning",
                        category=EventCategory.SCHEDULER,
                        severity=EventSeverity.WARNING,
                        data={
                            "warning": warning,
                            "max_task_risk_score": risk_summary.get("max_task_risk_score", 0.0),
                            "high_risk_task_ids": risk_summary.get("high_risk_task_ids", []),
                            "policy": decision.get("policy"),
                        },
                    )
                )

        # Emit scheduler.backpressure event
        backpressure = decision.get("backpressure")
        if backpressure is not None:
            self._emit(
                RuntimeEvent(
                    event_type="scheduler.backpressure",
                    category=EventCategory.SCHEDULER,
                    severity=EventSeverity.INFO,
                    data=backpressure,
                )
            )

        # Emit scheduler.safety_violation events
        safety_summary = decision.get("safety")
        if safety_summary is not None:
            violations = safety_summary.get("safety_violations", [])
            for violation in violations:
                self._emit(
                    RuntimeEvent(
                        event_type="scheduler.safety_violation",
                        category=EventCategory.SCHEDULER,
                        severity=EventSeverity.WARNING,
                        data={
                            "violation": violation,
                            "rejected_task_ids": safety_summary.get("rejected_task_ids", []),
                            "degraded_task_ids": safety_summary.get("degraded_task_ids", []),
                            "safety_deferred_task_ids": safety_summary.get("safety_deferred_task_ids", []),
                            "backpressure_signal": backpressure.get("signal") if backpressure else "none",
                            "pressure_level": backpressure.get("pressure_level") if backpressure else "normal",
                        },
                    )
                )

        # Emit scheduler.adaptive_tuning event
        adaptive_tuning = decision.get("adaptive_tuning")
        if adaptive_tuning is not None and adaptive_tuning.get("enabled"):
            self._emit(
                RuntimeEvent(
                    event_type="scheduler.adaptive_tuning",
                    category=EventCategory.SCHEDULER,
                    severity=EventSeverity.INFO,
                    data={
                        "window_size": adaptive_tuning.get("window_size", 5),
                        "observed_decisions": adaptive_tuning.get("observed_decisions", 0),
                        "policy_bias": adaptive_tuning.get("policy_bias", {}),
                        "reasons": adaptive_tuning.get("reasons", []),
                        "selected_policy_before_bias": adaptive_tuning.get("base_policy"),
                        "selected_policy_after_bias": decision.get("policy"),
                    },
                )
            )

    def _skip_blocked_tasks(self, dag: TaskDAG) -> None:
        """Close unreachable dependency chains without stopping other branches."""
        changed = True
        terminal_dependencies = {
            TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED
        }
        while changed:
            changed = False
            for task in dag.tasks.values():
                if task.status != TaskStatus.PENDING:
                    continue
                blockers = [
                    dep for dep in task.dependencies
                    if dag.tasks[dep].status in terminal_dependencies
                ]
                if not blockers:
                    continue
                task.status = TaskStatus.SKIPPED
                task.error = f"dependency did not complete: {', '.join(blockers)}"
                task.updated_at = time.time()
                changed = True
                self._emit(
                    RuntimeEvent(
                        event_type="fault.propagated",
                        task_id=task.task_id,
                        category=EventCategory.FAULT,
                        severity=EventSeverity.WARNING,
                        data={
                            "source_task_id": blockers[0],
                            "blocked_by": blockers,
                            "depth": 1,
                            "reason": task.error,
                        },
                    )
                )

    async def run(self, dag: TaskDAG) -> dict[str, Any]:
        """Run a DAG; isolate mode reports failure after independent work finishes.

        fail_fast preserves the original exception behavior. isolate raises
        WorkflowExecutionError with successful results and failure details.
        External cancellation continues to cancel the whole run.
        """
        scheduler = DAGScheduler(
            dag,
            available_resources=self._scheduler_available_resources(),
            max_batch_size=self._max_batch_size,
            affinity_advisor=self._build_affinity_advisor(),
            resource_estimator=self._estimate_task_resources,
            fairness_tracker=self._scheduler_fairness_tracker,
            fairness_scale=self._fairness_scale,
            kernel_probe=self._kernel_probe,
            force_policy=self._force_scheduler_policy,
            adaptive_tuner=self._adaptive_tuner,
        )
        results: dict[str, Any] = {}
        failures: dict[str, str] = {}
        first_failure: BaseException | None = None
        self._active_dag = dag
        self._cgroup_feedback.clear()
        scheduler_round = 0

        try:
            while not scheduler.is_complete:
                if self._failure_policy == "isolate":
                    self._skip_blocked_tasks(dag)
                    if scheduler.is_complete:
                        break
                scheduler_round += 1
                # Pressure feedback loop: sample system pressure before scheduling
                system_sample = self._sample_system_pressure()
                self._apply_scheduler_pressure(
                    scheduler,
                    system_sample,
                    scheduler_round=scheduler_round,
                )
                self._emit_pressure_event(
                    scope="system",
                    sample=system_sample,
                    acb=None,
                    task=None,
                    cgroup_path=None,
                )

                scheduler.set_available_resources(self._scheduler_available_resources())
                batch = await scheduler.get_next_batch()
                decision = scheduler.last_decision
                if decision is not None:
                    self._maintain_sched_ext(trigger="publish_decision")
                    for event in self._kernel_adapter.publish_decision(decision):
                        self._emit(event)
                self._emit_scheduler_decision(decision)
                if not batch:
                    await asyncio.sleep(0.01)
                    continue

                for task in batch:
                    self._reserve_task_resources(task)

                batch_started_at = time.monotonic()
                batch_results = await asyncio.gather(
                    *(self._run_task(task) for task in batch),
                    return_exceptions=True,
                )
                batch_duration_ms = (time.monotonic() - batch_started_at) * 1000.0
                # Coarse or injected clocks can yield zero; retain startup
                # fallback until a usable service-time sample is observed.
                if batch_duration_ms > 0:
                    scheduler.record_batch_service_time(batch_duration_ms)

                first_error: BaseException | None = None
                control_error: BaseException | None = None
                for task, result in zip(batch, batch_results):
                    self._release_task_resources(task)
                    if isinstance(result, BaseException):
                        dag.fail_task(task.task_id, error=str(result))
                        failures[task.task_id] = str(result)
                        if first_failure is None:
                            first_failure = result
                        if not isinstance(result, Exception):
                            control_error = result
                        agent_id = task.assigned_agent_id
                        trace_id = (
                            self.agents[agent_id].trace_id
                            if agent_id is not None and agent_id in self.agents
                            else None
                        )
                        self._emit(
                            RuntimeEvent(
                                event_type="task.failed",
                                agent_id=agent_id,
                                task_id=task.task_id,
                                trace_id=trace_id,
                                category=EventCategory.TASK,
                                severity=EventSeverity.ERROR,
                                data={"error": str(result)},
                            )
                        )
                        if first_error is None:
                            first_error = result
                        continue
                    dag.complete_task(task.task_id, result=result)
                    results[task.task_id] = result
                    self._emit(
                        RuntimeEvent(
                            event_type="context.stats",
                            category=EventCategory.CONTEXT,
                            data={"stats": self._context.stats()},
                        )
                    )

                if control_error is not None:
                    raise control_error
                if first_error is not None and self._failure_policy == "fail_fast":
                    raise first_error
            if failures:
                raise WorkflowExecutionError(
                    failures=failures,
                    results=results,
                    skipped_tasks=[
                        task.task_id for task in dag.tasks.values()
                        if task.status == TaskStatus.SKIPPED
                    ],
                ) from first_failure
        finally:
            for task_id in list(self._active_resource_reservations):
                task = dag.tasks.get(task_id)
                if task is not None:
                    self._release_task_resources(task)
            self._emit(
                RuntimeEvent(
                    event_type="context.stats",
                    category=EventCategory.CONTEXT,
                    data={"stats": self._context.stats()},
                )
            )
            self._cgroup_feedback.clear()
            self._active_dag = None

        return results
