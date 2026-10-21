from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG
from agentos.core.types import Message
from agentos.events.types import EventCategory, RuntimeEvent
from agentos.runtime.engine import AgentHandler

Runner = Callable[[AgentHandler.Context], Awaitable[Any]]


@dataclass(frozen=True)
class PlannedTask:
    task_id: str
    goal: str
    runner: str
    dependencies: tuple[str, ...] = ()
    context_refs: tuple[str, ...] = ()
    output_ref: str | None = None
    allowed_tools: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class RunnerHandler(AgentHandler):
    """Runtime owns task scheduling; the supplied runner owns its internal loop."""
    def __init__(self, runner: Runner, framework: str) -> None:
        self.runner = runner
        self.framework = framework

    def _emit(self, ctx: AgentHandler.Context, kind: str, **data: Any) -> None:
        ctx._emit(RuntimeEvent(
            event_type=kind, agent_id=ctx.acb.agent_id, task_id=ctx.task.task_id,
            trace_id=ctx.acb.trace_id, category=EventCategory.AGENT,
            data={"framework": self.framework, **data},
        ))

    async def run(self, ctx: AgentHandler.Context) -> Any:
        self._emit(ctx, "adapter.runner_started")
        result = await self.runner(ctx)
        # Validation must precede shared publication and notification. Existing
        # unvalidated direct RunnerHandler.run() calls keep their old behavior.
        if ctx.task.metadata.get("validator") is None:
            await self._publish(ctx, result)
        self._emit(ctx, "adapter.runner_completed")
        return result

    async def accept_result(self, ctx: AgentHandler.Context, result: Any) -> None:
        if ctx.task.metadata.get("validator") is not None:
            await self._publish(ctx, result)

    async def _publish(self, ctx: AgentHandler.Context, result: Any) -> None:
        output_ref = ctx.task.metadata.get("output_ref")
        if output_ref is not None:
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            ref = ctx.context_manager.add_shared_ref(output_ref, text)
            self._emit(ctx, "context.result_published", ref=output_ref, content_hash=ref.content_hash)
            receivers = list(ctx.task.metadata.get("notify_agents", []))
            receiver = ctx.task.metadata.get("notify_agent")
            if receiver:
                receivers.append(receiver)
            for receiver in dict.fromkeys(receivers):
                await ctx.bus.send(Message(
                    from_agent=ctx.acb.agent_id, to_agent=receiver,
                    content={"context_ref": output_ref}, message_type="result_ref",
                    task_id=ctx.task.task_id, trace_id=ctx.acb.trace_id,
                    context_refs=[output_ref],
                ))
                self._emit(ctx, "message.reference_sent", ref=output_ref, receiver=receiver)


def adapt_plan(
    tasks: Sequence[PlannedTask], runners: dict[str, Runner], *, framework: str,
) -> tuple[TaskDAG, dict[str, AgentHandler]]:
    """Adapt an application-owned explicit plan; this function does not plan it."""
    dag = TaskDAG()
    handlers = {name: RunnerHandler(runner, framework) for name, runner in runners.items()}
    output_refs = [task.output_ref for task in tasks if task.output_ref is not None]
    if len(output_refs) != len(set(output_refs)):
        raise ValueError("multiple tasks cannot publish the same output_ref")
    for spec in tasks:
        if spec.runner not in handlers:
            raise ValueError(f"unregistered runner: {spec.runner}")
        dag.add_task(Task(
            task_id=spec.task_id, name=spec.goal, agent_role=spec.runner,
            dependencies=list(spec.dependencies), context_refs=list(spec.context_refs),
            resource_requirements=ResourceQuota(allowed_tools=list(spec.allowed_tools)),
            metadata={**spec.metadata, "output_ref": spec.output_ref},
        ))
    return dag, handlers
