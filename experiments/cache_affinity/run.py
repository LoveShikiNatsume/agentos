"""实验 B：缓存亲和调度在 Runtime.run 主链上的对照。

对照因素：Runtime 的调度策略及公平性尺度。arm=auto 不强制策略，使用默认绝对阈值；
arm=auto_service_time 不强制策略，使用观测服务时间缩放公平性阈值。其余条件相同：同一 ModelGateway（带
CacheHeatTracker）、同一批任务、同一批宽、同一模型与解码参数。每次运行在每份共享文档开头
加独立随机串，保证不同运行之间不会复用缓存。

用法：
  python run.py --base-url http://127.0.0.1:18095 --model qwen2.5-7b \
      --condition roomy --arms latency_first,cache_affinity --reps 5 --output DIR
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agentos.adapters.workflow import PlannedTask, adapt_plan  # noqa: E402
from agentos.context.cache_tracker import CacheHeatTracker  # noqa: E402
from agentos.context.manager import ContextManager  # noqa: E402
from agentos.events.log import EventLog  # noqa: E402
from agentos.execution.backend import OpenAIBackend  # noqa: E402
from agentos.llm.gateway import ModelGateway  # noqa: E402
from agentos.runtime.engine import Runtime  # noqa: E402

SOURCES = ["argparse.py", "subprocess.py", "logging/__init__.py", "http/client.py",
           "email/message.py", "tarfile.py"]
STDLIB = Path("/usr/lib/python3.11")
DOC_CHARS = 17000          # 约 5k Token（Qwen 分词器下的代码文本）
TASKS_PER_GROUP = 4
MAX_TOKENS = 512


def build_workload():
    groups = []
    for index, name in enumerate(SOURCES):
        text = (STDLIB / name).read_text(encoding="utf-8")[:DOC_CHARS]
        funcs = list(dict.fromkeys(re.findall(r"^\s*def (\w+)\(", text, re.M)))
        funcs = [f for f in funcs if not f.startswith("__")] or funcs
        picked = funcs[:: max(1, len(funcs) // TASKS_PER_GROUP)][:TASKS_PER_GROUP]
        if len(picked) < TASKS_PER_GROUP:
            raise ValueError(f"{name} 中可用函数不足")
        groups.append({"group": index, "source": name, "text": text,
                       "sha256": hashlib.sha256(text.encode()).hexdigest(), "functions": picked})
    return groups


def scrape(client, base):
    text = client.get(base + "/metrics").text
    out = {}
    for key in ("prefix_cache_queries_total", "prefix_cache_hits_total", "prompt_tokens_total",
                "generation_tokens_total", "num_preemptions_total"):
        m = re.search(rf"^vllm:{key}\{{[^}}]*\}} ([0-9.e+]+)$", text, re.M)
        out[key] = float(m.group(1)) if m else None
    return out


async def run_once(args, groups, arm, rep, out_dir):
    nonce = uuid.uuid4().hex
    context = ContextManager()
    for g in groups:
        context.add_shared_ref(f"doc/{g['group']}",
                               f"[run {nonce} source {g['source']}]\n" + g["text"])
    rows = []
    t0 = time.monotonic()

    async def runner(ctx):
        meta = ctx.task.metadata
        start = time.monotonic() - t0
        resp = await ctx.llm_call([{"role": "user", "content":
            f"In the source code above, what does the function `{meta['function']}` do? "
            "Answer in at most two sentences."}])
        end = time.monotonic() - t0
        usage = resp.metadata.get("usage") or {}
        # 服务端开着推理解析器时，非思考模型的正文会落进 reasoning_content，两段都算输出。
        reasoning = (resp.metadata.get("assistant_message") or {}).get("reasoning_content") or ""
        text = resp.content + "\x00" + reasoning
        rows.append({"task_id": ctx.task.task_id, "group": meta["group"], "function": meta["function"],
                     "start": start, "end": end, "latency": end - start,
                     "prompt_tokens": usage.get("prompt_tokens"),
                     "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                     "completion_tokens": usage.get("completion_tokens"),
                     "finish_reason": resp.metadata.get("finish_reason"),
                     "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "content": resp.content, "reasoning_content": reasoning})
        return resp.content

    # 交错到达：g0t0, g1t0, ..., g5t0, g0t1, ...
    plan = []
    for t in range(TASKS_PER_GROUP):
        for g in groups:
            plan.append(PlannedTask(f"g{g['group']}_t{t}", f"explain {g['functions'][t]}", "ask",
                                    context_refs=(f"doc/{g['group']}",),
                                    metadata={"group": g["group"], "function": g["functions"][t]}))
    dag, handlers = adapt_plan(plan, {"ask": runner}, framework="expB")
    config = {"model": args.model, "base_url": args.base_url, "timeout": 600.0, "api_key_env": "NONE",
              "temperature": 0.0, "max_tokens": MAX_TOKENS, "thinking": None,
              "reasoning_effort": None, "response_format": None}
    backend = OpenAIBackend(config)
    if args.thinking_off:
        # Qwen3.5 默认思考；vLLM 通过 chat_template_kwargs 关闭，两组相同。
        base_payload = backend.request_payload
        backend.request_payload = lambda request: {
            **base_payload(request), "chat_template_kwargs": {"enable_thinking": False}}
    gateway = ModelGateway({"vllm": backend}, default_backend="vllm",
                           cache_tracker=CacheHeatTracker())
    events = out_dir / f"events_{arm}_rep{rep}.jsonl"
    runtime = Runtime(gateway, handlers, context_manager=context, max_batch_size=args.batch,
                      event_log=EventLog(events), enable_kernel_probe=False,
                      force_scheduler_policy=None if arm in ("auto", "auto_service_time") else arm,
                      fairness_scale="service_time" if arm == "auto_service_time" else "absolute",
                      failure_policy="isolate")
    with httpx.Client(timeout=30) as client:
        before = scrape(client, args.base_url.rstrip("/"))
        start = time.monotonic()
        await runtime.run(dag)
        makespan = time.monotonic() - start
        after = scrape(client, args.base_url.rstrip("/"))
    delta = {k: (after[k] - before[k]) if after[k] is not None and before[k] is not None else None
             for k in before}
    rows.sort(key=lambda r: r["start"])
    order = [r["task_id"] for r in rows]
    prompt = sum(r["prompt_tokens"] or 0 for r in rows)
    hit = sum(r["cached_tokens"] or 0 for r in rows)
    summary = {"arm": arm, "rep": rep, "nonce": nonce, "condition": args.condition,
               "makespan": makespan, "requests": len(rows), "prompt_tokens": prompt,
               "cached_tokens": hit, "hit_ratio": hit / prompt if prompt else None,
               "miss_tokens": prompt - hit,
               "truncated": sum(1 for r in rows if r["finish_reason"] == "length"),
               "mean_latency": sum(r["latency"] for r in rows) / len(rows),
               "server_delta": delta, "dispatch_order": order}
    with (out_dir / f"requests_{arm}_rep{rep}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return summary


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True, help="KV 条件标签，如 roomy / tight")
    ap.add_argument("--arms", default="latency_first,cache_affinity",
                    help="逗号分隔的策略名；auto 使用绝对公平阈值，auto_service_time 按服务时间缩放")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--thinking-off", action="store_true", help="对思考模型关闭思考")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = build_workload()
    (args.output / "workload.json").write_text(json.dumps(
        [{k: v for k, v in g.items() if k != "text"} for g in groups], indent=2, ensure_ascii=False))
    arms = args.arms.split(",")
    summaries = []
    for rep in range(args.reps):
        # ABBA：偶数轮 A 先，奇数轮 B 先，抵消顺序与服务端状态漂移。
        order = arms if rep % 2 == 0 else list(reversed(arms))
        for arm in order:
            s = await run_once(args, groups, arm, rep, args.output)
            summaries.append(s)
            print(json.dumps({k: s[k] for k in ("arm", "rep", "makespan", "hit_ratio", "miss_tokens",
                                                "truncated")}, ensure_ascii=False), flush=True)
    (args.output / "summary.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "runs": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
