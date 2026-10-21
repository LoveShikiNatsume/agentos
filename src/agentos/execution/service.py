"""Public execution service shared by CLI, Bench, and future App clients."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from agentos.context.prefix import ContextPriority
from agentos.execution.context_policy import ExecutionContextManager, ExecutionModelGateway
from agentos.execution.backend import OpenAIBackend, RecordedBackend
from agentos.execution.config import resolve_config
from agentos.execution.langgraph import LangGraphHarness, POLICY
from agentos.execution.records import EXCLUDED, Records, git_state, inventory, usage_summary
from agentos.execution.tools import make_tools
from agentos.llm.base import LLMRequest
from agentos.runtime.engine import Runtime

HARNESS_REGISTRY = {"langgraph": LangGraphHarness}


def register_harness(name, factory):
    if name in HARNESS_REGISTRY:
        raise ValueError(f"harness already registered: {name}")
    HARNESS_REGISTRY[name] = factory


class NativeContext:
    """Direct native graph calls; no Runtime, gateway, scheduler or ToolManager."""
    def __init__(self, task, session):
        self.task, self.session = task, session

    async def llm_call(self, messages, *, tools=None):
        s = self.session
        # Same context serialization as Runtime for a fair input contract; native
        # publishes/reads its own references directly and never calls Runtime.run.
        prepared = s.context.prepare_context(f"agent-{self.task.task_id}", messages,
                                             context_ref_names=self.task.context_refs)
        return await s.backend.generate(LLMRequest(agent_id=f"agent-{self.task.task_id}",
            task_id=self.task.task_id, trace_id=s.records.run_id, messages=prepared.messages,
            context_refs=list(prepared.shared_refs), prefix_id=prepared.prefix_id, tools=list(tools or []),
            metadata={"context_selection": prepared.selection_details}))

    async def tool_call(self, name, **kwargs):
        import inspect
        if name not in self.task.resource_requirements.allowed_tools:
            raise ValueError(f"tool not allowed: {name}")
        function = self.session.tools.get(name).handler
        result = function(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return SimpleNamespace(ok=True, output=result, error=None)


async def execute(task: str, *, cwd: Path | str = ".", overrides: dict | None = None,
                  backend=None, listener=None, scorer=None, provenance=None) -> dict:
    """Execute once. Backend injection is for explicit tests only, never CLI fallback.

    output is a root directory; each invocation creates a unique run directory.
    listener receives sanitized RuntimeEvent dictionaries; summary is the stable
    App/CLI read model. Scorers consume this very workspace after execution.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be nonempty natural language")
    config = resolve_config(Path(cwd), overrides)
    factory = HARNESS_REGISTRY.get(config["harness"])
    if factory is None:
        raise ValueError(f"unsupported harness {config['harness']!r}; supported: {', '.join(HARNESS_REGISTRY)}")
    if config["profile"] not in factory.profiles:
        raise ValueError(f"unsupported profile for {config['harness']}: {config['profile']}")
    # Fail before creating a run if the selected framework is not installed.
    try:
        framework_version = importlib.metadata.version(getattr(factory, "distribution", config["harness"]))
    except importlib.metadata.PackageNotFoundError:
        raise ValueError(f"install the selected harness dependency: pip install '.[{config['harness']}]'") from None
    root = Path(config["cwd"])
    output_root = Path(config.get("output") or root / ".agentos/runs").expanduser().resolve()
    if output_root == root:
        raise ValueError("output must be a dedicated records directory, not the working directory itself")
    run_id = "run-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    secret = os.environ.get(config["api_key_env"], "")
    records = Records(output, run_id, secrets=[secret], listener=listener)
    start = time.monotonic()
    before = inventory(root, output_root)
    records.write("start.json", {"files": before, "git": git_state(root), "provenance": provenance})
    records.write("config.json", config)
    records.write("task.json", {"run_id": run_id, "task": task, "framework_version": framework_version,
                 "execution_path": "Runtime.run + dynamic TaskDAG + LangGraph" if config["profile"] == "runtime" else "LangGraph.ainvoke directly",
                 "backend_injected_for_test": backend is not None})
    recorder = RecordedBackend(backend or OpenAIBackend(config), records, config)
    context = ExecutionContextManager(config["context_optimization"])
    if config["context_optimization"] == "on":
        context.add_shared_ref("policy", POLICY, priority=ContextPriority.SYSTEM_INSTRUCTION)
        context.add_shared_ref("task", task, priority=ContextPriority.TASK_BACKGROUND)
    else:
        context.add_shared("policy", POLICY)
        context.add_shared("task", task)
    session = SimpleNamespace(config=config, task=task, records=records, backend=recorder,
                              context=context, tools=make_tools(root, config, records),
                              entries=sorted(p.name for p in root.iterdir()), plan=[], answers={})
    harness = factory(session)
    dag, handlers = harness.runtime_plan()
    statuses = {}
    error = None
    runtime = None
    records.emit("execution.started", profile=config["profile"], harness=config["harness"], task=task)
    try:
        async with asyncio.timeout(config["run_timeout"]):
            if config["profile"] == "runtime":
                runtime = Runtime(ExecutionModelGateway(recorder, records, config["context_optimization"]), handlers,
                                  context_manager=context, tool_manager=session.tools, event_log=records,
                                  max_batch_size=config["max_parallel"], enable_kernel_probe=False, failure_policy="isolate")
                await runtime.run(dag)
            else:
                planner_task = dag.tasks["planner"]
                records.emit("native.task_started", "planner", dependencies=[])
                statuses["planner"] = "running"
                result = await harness.plan(NativeContext(planner_task, session))
                context.add_shared("result/planner", result)
                statuses["planner"] = "completed"
                records.emit("native.task_completed", "planner")
                await harness.native_tasks(lambda item: NativeContext(item, session), statuses)

    except Exception as exc:
        error = records.clean(f"{type(exc).__name__}: {exc}")
        for tid in statuses:
            if statuses[tid] == "running":
                statuses[tid] = "failed"
        records.emit("execution.failed", error=error)
    finally:
        if config["profile"] == "runtime":
            statuses = {tid: item.status.value for tid, item in dag.tasks.items()}
        else:
            for item in session.plan:
                statuses.setdefault(item["id"], "skipped")
    execution_seconds = time.monotonic() - start
    after = inventory(root, output_root)
    changes = [{"path": name, "before_sha256": before.get(name), "after_sha256": after.get(name)}
               for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)]
    for item in changes:
        if item["after_sha256"]:
            destination = output / "artifacts/files" / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Artifacts may include credentials in edited configuration: redact known key.
            content = (root / item["path"]).read_bytes()
            if secret:
                content = content.replace(secret.encode(), b"[REDACTED]")
            destination.write_bytes(content)
    records.write("artifacts.json", {"changes": changes, "answers": session.answers,
                  "git": git_state(root), "inventory_exclusions": sorted(EXCLUDED)})
    score = {"status": "not_requested", "passed": None}
    score_start = time.monotonic()
    if scorer:
        try:
            score = await scorer(root, output)
        except Exception as exc:
            score = {"status": "error", "passed": None, "error": f"{type(exc).__name__}: {exc}"}
        records.emit("evaluation.completed", score=score)
    communication = None
    if runtime is not None:
        communication = {"transport": "in_process_message_bus", "stats": runtime.bus.stats(),
                         "payload_bytes_scope": "UTF-8 bytes of JSON(message.content); excludes envelopes, transport framing and referenced context storage; not model tokens"}
        records.emit("communication.stats", **communication)
    summary = {"schema": "agentos.run.v1", "run_id": run_id, "task": task, "output": str(output),
               "config": config, "framework_version": framework_version,
               "arm": (provenance or {}).get("arm", {"off": "baseline", "on": "optimized"}.get(config["context_optimization"], config["profile"])),
               "context_optimization": config["context_optimization"],
               "execution_policies": {"failure_policy": "isolate", "scheduler_policy": "existing_runtime_default",
                   "max_parallel": config["max_parallel"], "api_retries": config["retries"],
                   "cache_affinity_feedback_enabled": config["context_optimization"] == "on",
                   "task_recovery_policy": "not_configured", "cgroup_enforcement": False,
                   "command_sandbox": config["command_sandbox"], "command_network": config["command_network"]},
               "response_models": sorted({c["response_model"] for c in recorder.calls if c["status"] == "completed"}),
               "backend_injected_for_test": backend is not None,
               "workflow_status": "failed" if error else "completed", "error": error,
               "task_statuses": statuses, "answers": session.answers, "artifacts": changes,
               "execution_seconds": execution_seconds,
               "evaluation_seconds": time.monotonic() - score_start if scorer else 0,
               "score": score, "usage": usage_summary(recorder.calls), "communication": communication,
               "protocol": harness.protocol_stats() if hasattr(harness, "protocol_stats") else {}}
    records.emit("execution.completed", workflow_status=summary["workflow_status"], usage=summary["usage"])
    records.write("summary.json", summary)
    return records.clean(summary)
