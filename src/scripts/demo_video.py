#!/usr/bin/env python3
"""One-key demo driver for the presentation video (对应 PPT 4.1 作品演示页).

Starts the live dashboard, then runs four acts mapped to the four highlights:

    Act 1  提交 software_eng 流水线 -> ACB 状态机实时流转     (亮点一)
    Act 2  DAG 视图: 同层并发 + 运行中动态插入 lint 节点        (亮点二, 与 Act1 同一次运行)
    Act 3  缓存命中: 多 Agent 命中同一共享前缀, 命中率抬升      (亮点三)
    Act 4  注入故障 -> 单节点 FAILED -> 局部恢复, 其余分支不停  (亮点四)

Usage:
    # 排练 (mock 后端, 无需 API key):
    .venv/bin/python scripts/demo_video.py

    # 正式录制 (真实 DeepSeek API, 与第三章跑数同一后端):
    DEEPSEEK_API_KEY=sk-xxx .venv/bin/python scripts/demo_video.py --backend deepseek

    # 全自动节拍 (无需按回车, 适合一键录屏):
    .venv/bin/python scripts/demo_video.py --auto

Options:
    --port 9410        dashboard 端口
    --backend mock     mock | cache-sim | deepseek (Act3 在 mock 模式下自动用 cache-sim)
    --auto             幕间不等待回车, 按固定秒数推进
    --no-browser       不自动打开浏览器
    --exit-after-acts  四幕结束后直接退出 (默认保持服务直到 Ctrl+C)
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentos.context.manager import ContextManager
from agentos.context.eviction import SegmentEvictor
from agentos.context.prefix import ContextPriority
from agentos.core.task import Task, TaskDAG
from agentos.dashboard.broadcast import EventBroadcaster
from agentos.dashboard.server import run_dashboard_server
from agentos.events.log import EventLog
from agentos.llm.factory import create_backend
from agentos.runtime.engine import Runtime
from agentos.scenarios.software_eng.pipeline import (
    PROJECT_CONTEXT,
    REPO_SUMMARY,
    SEGMENT_BUDGET,
    BackendCoderHandler,
    _build_handlers,
    build_software_eng_dag,
    run_software_eng,
)

DEMO_DIR = _REPO_ROOT.parent / "demo" / "replay"

BANNER = "=" * 68


def _cue(act: str, title: str, lines: list[str]) -> None:
    """Print a narration cue block for the presenter."""
    print(f"\n{BANNER}")
    print(f"  {act}  {title}")
    print(BANNER)
    for line in lines:
        print(f"  {line}")
    print(BANNER, flush=True)


async def _pause(auto: bool, seconds: float, prompt: str) -> None:
    if auto:
        for remaining in range(int(seconds), 0, -1):
            print(f"\r  ... {prompt} ({remaining}s) ", end="", flush=True)
            await asyncio.sleep(1)
        print()
    else:
        await asyncio.to_thread(input, f"  >>> {prompt} — 按回车继续 ")


class DynamicLintBackendCoder(BackendCoderHandler):
    """backend 写完代码后, 运行中动态向 DAG 插入 lint 任务 (亮点二演示点)."""

    async def run(self, ctx):  # type: ignore[override]
        result = await super().run(ctx)
        ctx.submit_task(Task(
            task_id="lint",
            name="lint",
            agent_role="linter",
            dependencies=["backend"],
            metadata={
                "execution": "subprocess",
                "worker_type": "sleep",
                "worker_payload": {
                    "seconds": 1.0,
                    "result": "lint passed: 0 errors, 12 info",
                },
            },
        ))
        return result


async def run_act_1_2(backend: str, broadcaster: EventBroadcaster, delay: float) -> None:
    """Acts 1+2: full pipeline, lint inserted dynamically at runtime."""
    log_path = DEMO_DIR / "act12_pipeline.jsonl"
    log_path.unlink(missing_ok=True)

    ctx_mgr = ContextManager(evictor=SegmentEvictor(), segment_budget=SEGMENT_BUDGET)
    ctx_mgr.add_shared_ref(
        "project-context", PROJECT_CONTEXT, priority=ContextPriority.TASK_BACKGROUND,
    )
    ctx_mgr.add_shared_ref(
        "repo-summary", REPO_SUMMARY, priority=ContextPriority.REPO_SUMMARY,
    )

    # Build the standard 11-node DAG, then remove lint from the static graph:
    # it will be re-inserted dynamically by DynamicLintBackendCoder while running.
    dag = build_software_eng_dag()
    del dag.tasks["lint"]
    dag.tasks["integrate"].dependencies = [
        d for d in dag.tasks["integrate"].dependencies if d != "lint"
    ]

    handlers = _build_handlers()
    handlers["backend_coder"] = DynamicLintBackendCoder()

    backend_kwargs = {"latency": delay} if backend == "mock" else {}
    rt = Runtime(
        llm_backend=create_backend(backend, **backend_kwargs),
        handlers=handlers,
        context_manager=ctx_mgr,
        event_log=EventLog(log_path),
        broadcaster=broadcaster,
    )
    await rt.run(dag)
    statuses = {t.task_id: t.status.value for t in dag.tasks.values()}
    print(f"  [act1+2] done: {statuses}")


async def run_act_3(backend: str, broadcaster: EventBroadcaster, delay: float) -> None:
    """Act 3: cache-hit run — agents share one canonical prefix."""
    cache_backend = backend if backend == "deepseek" else "cache-sim"
    log_path = DEMO_DIR / "act3_cache.jsonl"
    log_path.unlink(missing_ok=True)
    await run_software_eng(
        backend=cache_backend,
        log_path=log_path,
        broadcaster=broadcaster,
        delay=delay if cache_backend != "deepseek" else 0.0,
    )
    print("  [act3] done")


async def run_act_4(backend: str, broadcaster: EventBroadcaster, delay: float) -> None:
    """Act 4: kill the security agent mid-run; watch local recovery."""
    log_path = DEMO_DIR / "act4_fault.jsonl"
    log_path.unlink(missing_ok=True)
    await run_software_eng(
        backend=backend,
        log_path=log_path,
        inject_fault=True,
        broadcaster=broadcaster,
        delay=delay if backend == "mock" else 0.0,
    )
    print("  [act4] done")


async def main_async(args: argparse.Namespace) -> int:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    # Live counters (KPI / decision / DAG / cache / fault timeline) start from
    # ZERO on purpose: we do NOT preload DASHBOARD_EVENT_LOG, so every number
    # the audience sees grows in real time from the acts below.
    # The Pressure Suite matrix panel is historical real-experiment data —
    # point it at its JSON file so it renders as a clearly-labelled archive
    # instead of "loading…". The cloud benchmark page is left to
    # agentos.dashboard.cloud_benchmark's own pinned exploratory-pilot default
    # (see that module's docstring) unless CLOUD_BENCHMARK_JSON is already set
    # by the caller — it is NOT the official national-batch result and this
    # script must not silently pick "whichever pilot file is newest".
    import os
    wm = _REPO_ROOT / "runs" / "workload_matrix_deepseek_real" / "summary.json"
    if wm.exists():
        os.environ.setdefault("DASHBOARD_WORKLOAD_MATRIX", str(wm))

    broadcaster = EventBroadcaster()

    server_task = asyncio.create_task(
        run_dashboard_server(broadcaster, host=args.host, port=args.port)
    )
    await asyncio.sleep(0.5)  # let the server bind
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Dashboard: {url}")

    if not args.no_browser:
        opener = shutil.which("xdg-open") or shutil.which("open")
        if opener:
            subprocess.Popen(
                [opener, url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    _cue("准备", "开始录屏", [
        f"1. 浏览器打开 {url} , 停在【本地 LLM Runtime 观测】页, 全屏 (F11)",
        "2. 画面停在顶部 KPI 行 (Token Saved / Cache Hit / Fault Recovery ...)",
        "3. 启动录屏 (OBS / ffmpeg), 回到本终端确认开始",
    ])
    await _pause(args.auto, 12, "录屏就绪后开始 Act 1")

    _cue("Act 1+2", "提交流水线: ACB 状态机 + DAG 动态插入 (亮点一/二)", [
        "口播: 现在提交一条 11 节点的软件工程流水线。",
        "看点A 滚动到【① 进程/Agent 调度】区块: 每个 Agent 一行,",
        "        AGENT / PID / WORKER / CGROUP / 状态实时流转 —— 这就是 ACB。",
        "看点B 滚动到【② DAG 调度流】区块: db_design 与 api_design 同层并发;",
        "        backend 完成瞬间, lint 节点被『运行中动态插入』出现在图上。",
    ])
    await run_act_1_2(args.backend, broadcaster, delay=args.delay)
    await _pause(args.auto, 8, "滚动到【③ 上下文内存 / ④ 缓存亲和】区块, 进入 Act 3")

    _cue("Act 3", "缓存命中: 共享前缀只算一次 (亮点三)", [
        "口播: 同一条流水线, 打开上下文共享 —— 所有 Agent 命中同一段共享前缀。",
        "看点【③ 上下文内存】: 共享段 spec_shared 被多个 Agent 引用 (ContextRef);",
        "看点【④ 缓存亲和】: KV 命中率趋势曲线抬升, Agent x Shard 热力图变深。",
        ("后端: 真实 DeepSeek API" if args.backend == "deepseek"
         else "后端: cache-sim (排练模式; 正式录制请加 --backend deepseek)"),
    ])
    await run_act_3(args.backend, broadcaster, delay=max(args.delay * 0.4, 0.2))
    await _pause(args.auto, 8, "滚动到【② DAG 调度流 + F 故障/恢复时间线】, 进入 Act 4")

    _cue("Act 4", "故障注入: 单点失败, 局部恢复 (亮点四)", [
        "口播: 现在故意杀死 security 这个 Agent。",
        "看点【② DAG 调度流】: security 节点变红 FAILED -> RECOVERING -> 恢复;",
        "        docs / build / test 等兄弟分支全程不受影响, integrate 正常完成。",
        "看点【F 故障/恢复时间线】: fault.injected -> snapshot_saved ->",
        "        recovery_start -> context_restored 逐条滚动出现。",
        "收束口播: 单点故障止于单点, DAG 不整张重跑。",
    ])
    await run_act_4(args.backend, broadcaster, delay=args.delay)

    _cue("收尾", "演示结束", [
        "口播: 演示用的就是第三章跑数的同一套系统, 不是另做的 demo。",
        "回到页面顶部 KPI 行: EventLog 事件数 / Fault Recovery / Cache Hit 已更新;",
        "(可选) 点【云端多 Agent 框架对比】页展示四框架 A/B 数据, 停 3 秒, 停止录屏。",
        f"事件日志已落盘: {DEMO_DIR}/",
    ])

    if args.exit_after_acts:
        server_task.cancel()
        return 0

    print("\n  dashboard 继续运行中, Ctrl+C 结束 ...")
    try:
        await server_task
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AgentOS demo-video driver")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9410)
    parser.add_argument("--backend", default="mock",
                        choices=["mock", "cache-sim", "deepseek"])
    parser.add_argument("--delay", type=float, default=1.2,
                        help="mock 后端每次 LLM 调用延迟(秒), 控制画面节拍")
    parser.add_argument("--auto", action="store_true",
                        help="幕间自动倒计时, 不等待回车")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--exit-after-acts", action="store_true")
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
