from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Sequence

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG
from agentos.fault.recovery import RecoveryAction, RecoveryDecision
from agentos.fault.timeout import AgentTimeoutError
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import Runtime


class WorkerDemoRecoveryPolicy:
    def decide(self, error: BaseException, attempt: int) -> RecoveryDecision:
        if isinstance(error, AgentTimeoutError):
            return RecoveryDecision(RecoveryAction.SKIP, reason="demo skips timeout")
        if attempt == 1:
            return RecoveryDecision(RecoveryAction.RETRY, reason="demo retries once")
        return RecoveryDecision(RecoveryAction.SKIP, reason="demo skips failed worker")


def build_demo_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(
        Task(
            task_id="worker-ok",
            name="normal subprocess worker",
            agent_role="echo",
            metadata={
                "execution": "subprocess",
                "worker_type": "echo",
                "worker_payload": {"result": "worker ok"},
            },
        )
    )
    dag.add_task(
        Task(
            task_id="worker-timeout",
            name="timeout subprocess worker",
            agent_role="sleep",
            resource_requirements=ResourceQuota(timeout_seconds=0.05, pids_max=16),
            metadata={
                "execution": "subprocess",
                "worker_type": "sleep",
                "worker_payload": {"seconds": 1.0, "result": "too slow"},
            },
        )
    )
    dag.add_task(
        Task(
            task_id="worker-flaky",
            name="flaky subprocess worker",
            agent_role="flaky",
            metadata={
                "execution": "subprocess",
                "worker_type": "flaky",
                "worker_payload": {"result": "recovered"},
            },
        )
    )
    dag.add_task(
        Task(
            task_id="worker-fail",
            name="controlled subprocess failure",
            agent_role="fail",
            metadata={
                "execution": "subprocess",
                "worker_type": "fail",
                "worker_payload": {"error": "controlled failure"},
            },
        )
    )
    return dag


async def run_demo(log_path: str | Path = "runs/latest/events.jsonl") -> dict[str, object]:
    runtime = Runtime(
        llm_backend=MockLLMBackend(),
        handlers={},
        event_log=log_path,
        recovery_policy=WorkerDemoRecoveryPolicy(),
    )
    return await runtime.run(build_demo_dag())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentos.scenarios.worker_subprocess.demo")
    parser.add_argument("--log", default="runs/latest/events.jsonl")
    args = parser.parse_args(argv)

    results = asyncio.run(run_demo(args.log))
    for task_id, result in results.items():
        print(f"{task_id}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
