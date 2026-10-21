from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Sequence

from agentos.core.task import Task, TaskDAG
from agentos.events.log import EventLog
from agentos.fault.recovery import RetryPolicy
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime


class FlakyRecoveryHandler(AgentHandler):
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, ctx: AgentHandler.Context) -> str:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("injected recoverable failure")
        return "recovered"


async def run_recovery_demo(
    *,
    log_path: str | Path = "runs/latest/recovery_events.jsonl",
) -> dict[str, Any]:
    dag = TaskDAG()
    dag.add_task(Task(task_id="recover", name="recover", agent_role="flaky"))
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={"flaky": FlakyRecoveryHandler()},
        recovery_policy=RetryPolicy(max_retries=1),
        event_log=EventLog(log_path),
    )
    return await runtime.run(dag)


async def _async_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recovery-demo")
    parser.add_argument("--log", default="runs/latest/recovery_events.jsonl")
    args = parser.parse_args(argv)

    results = await run_recovery_demo(log_path=args.log)
    for task_id, result in results.items():
        print(f"{task_id}: {result}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
