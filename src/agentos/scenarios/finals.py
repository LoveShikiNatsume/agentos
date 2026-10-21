"""Reproducible finals demonstration using synthetic release-review inputs.

Six bounded model calls compare actual context payloads. Fault isolation and
context retry are deterministic runtime checks, independent of model answers.
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentos.context.manager import ContextManager
from agentos.core.task import Task, TaskDAG
from agentos.dashboard import build_dashboard
from agentos.events.log import EventLog
from agentos.fault.recovery import ContextRetryPolicy
from agentos.llm.base import LLMBackend, LLMRequest, LLMResponse
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime, WorkflowExecutionError


MATERIALS = {
    "policy": (
        "Synthetic release-review exercise. Use only the supplied evidence. "
        "State one concrete finding and one next check in at most two sentences. "
        "Do not invent measurements or claim a deployment happened. "
        "The source review, test review and rollout review are separate tasks."
    ),
    "source": (
        "SOURCE_REVIEW_MATERIAL\n"
        "A fictional checkout service introduces an idempotency key for POST /orders. "
        "The key is unique per customer and stored with a request-body hash and order id. "
        "On a duplicate key, the service returns the existing order only when the body "
        "hash matches; a changed body receives HTTP 409. Database writes and key "
        "registration share a transaction. The old handler retried the payment call "
        "before recording the order, which could charge twice after a timeout. The new "
        "handler forwards the same key to the payment provider. The migration creates "
        "a unique index on customer_id plus idempotency_key; the index is built online."
    ),
    "tests": (
        "TEST_REVIEW_MATERIAL\n"
        "Synthetic test report for the checkout change: same key and same payload "
        "returns the same order; same key and different payload returns HTTP 409; "
        "two concurrent requests with the same key create one row. Simulated payment "
        "timeouts are retried with the original provider key. A database rollback test "
        "confirms no key remains registered when order creation aborts. A process "
        "termination between provider response and database commit has not yet been "
        "tested. Load tests were not run, so the report has no throughput or P99 claim."
    ),
    "rollout": (
        "ROLLOUT_REVIEW_MATERIAL\n"
        "Synthetic rollout proposal: create the index before enabling the handler; "
        "enable it for an internal customer cohort, then five percent of traffic. "
        "Watch duplicate payment alerts, HTTP 409 rates, database lock waits and "
        "payment-provider timeouts. Keep the old handler behind a feature flag. "
        "Rollback disables new traffic to the handler but retains idempotency records "
        "so already submitted requests remain deduplicated. An operator must verify "
        "the provider's key-retention window before widening the cohort. There is "
        "currently no scheduled production deployment and no observed rollout result."
    ),
}


class RecordingBackend:
    def __init__(self, backend: LLMBackend, model: str, max_tokens: int) -> None:
        self.backend = backend
        self.model = model
        self.max_tokens = max_tokens
        self.records: list[dict[str, Any]] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        bounded = replace(request, model=self.model, max_tokens=self.max_tokens, temperature=0)
        response = await self.backend.generate(bounded)
        self.records.append({
            "task_id": bounded.task_id,
            "prefix_id": bounded.prefix_id,
            "requested_refs": bounded.context_refs,
            "messages": bounded.messages,
            "prompt_characters": sum(len(m.get("content", "")) for m in bounded.messages),
            "response": response.content,
            "model": response.model,
            "metrics": asdict(response.metrics) if response.metrics is not None else None,
            "provider_usage": response.metadata.get("usage"),
        })
        return response


class Review(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        result = await ctx.llm_call([{"role": "user", "content": ctx.task.name}])
        return result.content


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_events(runtime: Runtime, output: Path, case: str) -> None:
    log = EventLog(output / "events.jsonl")
    for event in runtime.events:
        log.append(replace(event, data={**event.data, "demo_case": case}))


async def _context_case(output: Path, backend: RecordingBackend, selected: bool) -> dict[str, Any]:
    cm = ContextManager()
    for name, content in MATERIALS.items():
        cm.add_shared(name, content)
    case = "selected" if selected else "all_shared"
    dag = TaskDAG()
    for name in ("source", "tests", "rollout"):
        dag.add_task(Task(
            f"{case}/{name}", f"Review the {name} material. Give a finding and next check.",
            "review", context_refs=["policy", name] if selected else [],
        ))
    runtime = Runtime(backend, {"review": Review()}, context_manager=cm,
                      max_batch_size=1, enable_kernel_probe=False)
    try:
        results = await runtime.run(dag)
    finally:
        _append_events(runtime, output, case)
    records = [item for item in backend.records if item["task_id"].startswith(case + "/")]
    selection_checks = []
    for record in records:
        role = record["task_id"].split("/")[-1]
        system = record["messages"][0]["content"]
        expected = [MATERIALS["policy"], MATERIALS[role]] if selected else list(MATERIALS.values())
        selection_checks.append(system == "\n\n".join(expected))
    return {
        "completed": sorted(results),
        "calls": len(records),
        "prompt_characters": sum(item["prompt_characters"] for item in records),
        "payload_matches_declared_selection": len(records) == 3 and all(selection_checks),
        "prompt_tokens": sum((item["metrics"] or {}).get("prompt_tokens", 0) for item in records),
        "cache_hit_tokens": sum((item["metrics"] or {}).get("cache_hit_tokens", 0) for item in records),
        "cache_miss_tokens": sum((item["metrics"] or {}).get("cache_miss_tokens", 0) for item in records),
    }


async def _isolation_case(output: Path) -> dict[str, Any]:
    class Fail(AgentHandler):
        async def run(self, ctx):
            raise RuntimeError("injected optional telemetry failure")

    class Complete(AgentHandler):
        async def run(self, ctx):
            return {"task": ctx.task.task_id, "completed": True}

    dag = TaskDAG()
    dag.add_task(Task("isolation/optional", "Optional telemetry", "fail", priority=100))
    dag.add_task(Task("isolation/blocked", "Telemetry summary", "complete", dependencies=["isolation/optional"]))
    dag.add_task(Task("isolation/review", "Independent review", "complete"))
    dag.add_task(Task("isolation/report", "Independent report", "complete", dependencies=["isolation/review"]))
    rt = Runtime(MockLLMBackend(), {"fail": Fail(), "complete": Complete()},
                 max_batch_size=1, failure_policy="isolate", enable_kernel_probe=False)
    try:
        await rt.run(dag)
    except WorkflowExecutionError as exc:
        failures, completed, skipped = exc.failures, sorted(exc.results), exc.skipped_tasks
    else:
        raise AssertionError("injected workflow failure was not reported")
    finally:
        _append_events(rt, output, "isolation")
    statuses = {key: task.status.value for key, task in dag.tasks.items()}
    return {
        "failures": failures, "completed": completed, "skipped": skipped,
        "statuses": statuses,
        "passed": statuses == {
            "isolation/optional": "failed", "isolation/blocked": "skipped",
            "isolation/review": "completed", "isolation/report": "completed",
        },
    }


async def _retry_case(output: Path) -> dict[str, Any]:
    cm = ContextManager()
    cm.add_shared("status", "initial shared state")
    written, updated = asyncio.Event(), asyncio.Event()

    class Retry(AgentHandler):
        calls = 0

        async def run(self, ctx):
            self.calls += 1
            if self.calls == 1:
                cm.add_private(ctx.acb.agent_id, "FAILED_ATTEMPT_SCRATCH")
                written.set()
                await updated.wait()
                raise RuntimeError("injected transient failure")
            payload = str(cm.build_messages(ctx.acb.agent_id, []))
            return {
                "failed_scratch_removed": "FAILED_ATTEMPT_SCRATCH" not in payload,
                "concurrent_shared_update_kept": "HEALTHY_SHARED_UPDATE" in payload,
                "other_private_not_visible": "OTHER_PRIVATE" not in payload,
            }

    class Update(AgentHandler):
        async def run(self, ctx):
            await written.wait()
            cm.add_shared("status", "HEALTHY_SHARED_UPDATE")
            cm.add_private(ctx.acb.agent_id, "OTHER_PRIVATE")
            updated.set()
            return ctx.acb.agent_id

    dag = TaskDAG()
    dag.add_task(Task("retry/recover", "Recover task", "retry"))
    dag.add_task(Task("retry/healthy", "Update shared state", "update"))
    rt = Runtime(MockLLMBackend(), {"retry": Retry(), "update": Update()},
                 context_manager=cm, max_batch_size=2, enable_kernel_probe=False,
                 recovery_policy=ContextRetryPolicy(max_retries=1))
    try:
        results = await asyncio.wait_for(rt.run(dag), timeout=5)
    finally:
        _append_events(rt, output, "retry")
    checks = results["retry/recover"]
    checks["other_private_kept"] = "OTHER_PRIVATE" in str(cm.build_messages(results["retry/healthy"], []))
    return {"checks": checks, "passed": all(checks.values())}


async def run_finals_demo(
    output: Path, *, backend_name: str = "mock", model: str = "deepseek-chat",
    max_tokens: int = 128,
) -> dict[str, Any]:
    """Write an inspectable bundle; refuse to mix a run into an existing directory."""
    if backend_name not in {"mock", "deepseek"}:
        raise ValueError("backend must be mock or deepseek")
    if not 32 <= max_tokens <= 512:
        raise ValueError("max_tokens must be between 32 and 512")
    if backend_name == "deepseek":
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY is required for the deepseek backend")
        from agentos.llm.deepseek_backend import DeepSeekBackend
        backend = DeepSeekBackend(model=model, timeout=45)
    else:
        backend = MockLLMBackend(latency=0)
        model = "mock"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    recorder = RecordingBackend(backend, model, max_tokens)
    summary: dict[str, Any] = {
        "schema": "agentos_finals_demo.v1", "passed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name, "model": model, "synthetic_inputs": True,
        "token_source": "provider_usage" if backend_name == "deepseek" else "mock_character_estimate",
        "performance_benchmark": False, "kernel_scheduler_enabled": False,
        "environment": {"python": platform.python_version(), "kernel": platform.release()},
    }
    try:
        before = await _context_case(output, recorder, selected=False)
        after = await _context_case(output, recorder, selected=True)
        summary["context"] = {
            "all_shared": before, "selected": after,
            "prompt_character_reduction_pct": round(100 * (1 - after["prompt_characters"] / before["prompt_characters"]), 2),
        }
        summary["isolation"] = await _isolation_case(output)
        summary["retry"] = await _retry_case(output)
        summary["passed"] = (
            before["payload_matches_declared_selection"]
            and after["payload_matches_declared_selection"]
            and summary["isolation"]["passed"] and summary["retry"]["passed"]
        )
    except Exception as exc:
        summary["error_type"] = type(exc).__name__
        raise
    finally:
        _write_json(output / "summary.json", summary)
        (output / "requests.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in recorder.records),
            encoding="utf-8",
        )
    _write_report(output, summary)
    build_dashboard(log_paths=[output / "events.jsonl"], output_path=output / "dashboard.html")
    return summary


def _write_report(output: Path, summary: dict[str, Any]) -> None:
    context = summary["context"]
    before, after = context["all_shared"], context["selected"]
    lines = [
        "# AgentOS 国赛功能演示", "",
        f"- 全部检查通过：{summary['passed']}；后端：{summary['backend']}；模型：{summary['model']}。",
        "- 输入是公开可检查的虚构发布评审材料，共 6 次模型调用；故障与重试使用确定性处理器。",
        f"- Token 来源：`{summary['token_source']}`。这是一轮功能演示，不是性能基准或质量评测。", "",
        "## 1. 任务选择真正决定模型输入", "",
        f"全部共享材料：{before['prompt_characters']} 字符；按任务选择：{after['prompt_characters']} 字符。",
        f"本例减少 {context['prompt_character_reduction_pct']}%。`requests.jsonl` 保存实际消息，可逐条核对。",
        f"两组 prompt token：{before['prompt_tokens']} → {after['prompt_tokens']}；只在真实后端时表示接口 usage。",
        f"缓存命中 token：{before['cache_hit_tokens']} → {after['cache_hit_tokens']}；未命中：{before['cache_miss_tokens']} → {after['cache_miss_tokens']}。",
        "运行顺序固定为全部共享在先、选择在后；前缀长度和预热状态不同，不能据此比较缓存策略优劣。",
        "该变化来自材料选择，不能归因为 KV 缓存命中，也不能外推为整体任务成功率或普遍节省比例。", "",
        "## 2. 单分支失败后的继续执行", "",
        "可选遥测任务失败，其依赖项跳过；独立评审和下一批报告任务完成。运行时仍返回失败状态，并保留成功结果。", "",
        "## 3. 重试恢复范围", "",
        "失败尝试的私有内容清除，其他任务的共享更新和私有内容保留。此操作不回滚文件、数据库或工具副作用。", "",
        "## 查看", "",
        "- `summary.json`：逐项结果、环境及数据来源。",
        "- `requests.jsonl`：实际发送的合成输入、回复与 usage。",
        "- `events.jsonl`：带 demo_case 的完整 Runtime 事件。",
        "- `dashboard.html`：现有事件面板，可离线查看。", "",
        "本轮关闭内核探针与 sched_ext；不能用此结果证明当前机器的内核调度性能。", "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
