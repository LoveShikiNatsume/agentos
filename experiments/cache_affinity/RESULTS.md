# 实验 B 结果：缓存亲和调度（2026-10-19，本地 vLLM）

## 问题与设计

问题：多个 Agent 任务共享少数几份长文档时，Runtime 按缓存亲和顺序派发，能否减少未命中 Token 和完工时间？收益在什么条件下出现？

- 负载：6 份共享文档（`/usr/lib/python3.11` 的 argparse、subprocess、logging、http.client、email.message、tarfile，各取前 17,000 字符，约 3.5k–4.3k Token），每份文档 4 个任务（解释其中一个函数），共 24 个任务，同时就绪，交错到达（g0t0, g1t0, …, g5t0, g0t1, …）。
- 执行：全部经过 `adapt_plan` → `Runtime.run`，同一 `ModelGateway` 带 `CacheHeatTracker`，批宽 4，`failure_policy="isolate"`。温度 0，输出上限 512（实测最长输出远低于上限，截断 0）。
- 单因素：只改 Runtime 的调度策略（`force_scheduler_policy`）或公平性尺度（`fairness_scale`）。
- 每次运行在每份文档开头加独立随机串，不同运行之间不复用缓存。
- 重复：每个条件 6–10 轮，两组按 ABBA 交替。完工时间、命中率、未命中 Token 取每轮值，按轮配对做 bootstrap 95% 置信区间（10,000 次重采样）。
- 指标来源：命中率与未命中 Token 来自逐请求 `prompt_tokens_details.cached_tokens`，同时记录服务端 `/metrics` 增量（命中、查询、抢占）。
- KV 条件：用 `--num-gpu-blocks-override` 压 KV 容量。Qwen2.5-7B 每块 16 Token，750 块约 1.2 万 Token，1125 块约 1.8 万 Token；6 份文档合计约 2.4 万 Token。

## 结果

| 模型 | KV 条件 | 对照 | 完工时间 s | 命中率 | 完工时间相对变化 [95% CI] |
|---|---|---|---:|---:|---|
| Qwen2.5-7B | 充足（8192 上下文默认块数） | latency_first → cache_affinity | 8.10 → 8.06 | 74.7% → 74.7% | −0.4% [−2.1%, +1.2%] |
| Qwen2.5-7B | 1125 块 | latency_first → cache_affinity | 11.96 → 8.01 | 28.3% → 74.7% | −33.0% [−34.0%, −31.6%] |
| Qwen2.5-7B | 750 块 | latency_first → cache_affinity | 17.71 → 7.99 | 0.8% → 74.7% | −54.9% [−55.4%, −54.2%] |
| Qwen2.5-7B | 750 块 | latency_first → auto（绝对阈值） | 17.76 → 17.61 | 0.9% → 3.4% | −0.8% [−4.8%, +1.9%] |
| Qwen2.5-7B | 750 块 | auto → auto_service_time | 17.20 → 9.08 | 5.9% → 70.6% | −47.0% [−49.2%, −43.9%] |
| Qwen2.5-7B | 充足 | latency_first → auto_service_time | 8.17 → 8.18 | 74.7% → 74.7% | +0.1% [−1.5%, +1.8%] |
| Qwen2.5-7B | 750 块，批宽 2（6 轮） | latency_first → cache_affinity | 18.89 → 12.85 | 0.7% → 74.7% | −31.9% [−33.4%, −30.6%] |
| Qwen2.5-7B | 750 块，批宽 8（6 轮） | latency_first → cache_affinity | 15.57 → 5.48 | 9.8% → 74.7% | −64.8% [−65.5%, −64.0%] |
| Qwen3.5-9B（混合注意力） | 24 块（约 8.9k Token） | latency_first → cache_affinity | 39.98 → 19.59 | 0.0% → 70.7% | −51.0% [−51.3%, −50.7%] |
| Qwen3.5-9B | 充足 | latency_first → cache_affinity | 12.51 → 12.53 | 70.2% → 70.7% | +0.2% [−1.2%, +1.6%] |

任务完成时刻（从运行开始计，所有轮次汇总）：

| 条件 | 组 | p50 s | p95 s | 最大 s |
|---|---|---:|---:|---:|
| Qwen2.5 750 块 | latency_first | 10.47 | 17.12 | 17.93 |
| Qwen2.5 750 块 | cache_affinity | 4.92 | 7.96 | 8.29 |
| Qwen2.5 750 块 | auto | 8.79 | 16.93 | 17.92 |
| Qwen2.5 750 块 | auto_service_time | 6.06 | 8.99 | 9.33 |
| Qwen2.5 充足 | latency_first | 5.66 | 8.06 | 8.42 |
| Qwen2.5 充足 | cache_affinity | 4.98 | 8.03 | 8.28 |
| Qwen3.5 24 块 | latency_first | 21.88 | 38.58 | 40.38 |
| Qwen3.5 24 块 | cache_affinity | 10.43 | 18.74 | 19.78 |

截断：全部 0。抢占：Qwen2.5 各条件合计 0–1 次；Qwen3.5 24 块条件下 latency_first 72 次、cache_affinity 12 次（KV 小到同时容纳不了 4 个请求，两组都受影响）。

## 结论

1. 缓存亲和的收益只在 KV 容量放不下全部共享前缀时出现。容量充足时，交错顺序下前缀也一直留在缓存里，两组没有差异；容量不足时，交错顺序让前缀被逐出后重算，亲和顺序让同前缀任务相邻执行，未命中 Token 减少 64%–75%，完工时间减少 32%–65%，750 块条件下批宽 2、4、8 的完工时间降幅依次为 32%、55%、65%。
2. 亲和顺序在这个负载下没有造成饥饿：任务完成时刻的 p50 和 p95 都提前。
3. 修复前的自动策略在压力条件下没有收益：约 97% 的决策是 `fairness_balanced`（原因 `high starvation risk`），因为饥饿阈值是固定的 3000 ms，而一次 LLM 批处理约 2 s。`fairness_scale="service_time"` 按观测到的批耗时和队列位置缩放阈值，并给亲和排序加老化上限，修复后自动策略的完工时间为 9.08 s，接近强制亲和的 7.99 s；差距来自开头几批还没有热度信息，先按关键路径顺序派发。
4. 混合注意力模型（Qwen3.5，缓存块 528 Token）上结论相同。

## 限制

- 只有一种负载形状（6 份文档 × 4 任务、同时就绪）；批宽只测了 2、4、8，在 750 块条件下。
- 每轮的随机串让提示词逐轮不同，温度 0 下同一任务的输出也会随批组成变化，所以没有比较两组的回答质量；回答长度分布相近。
- 2026-10-19 之前的运行（充足、1125 块、750 块、auto 那几组）服务端开着 `--reasoning-parser qwen3`，Qwen2.5 的正文落进了 `reasoning_content` 而没有保存；生成过程不受影响，时延与缓存指标有效。

## 复现

```sh
# 启动（例：Qwen2.5-7B，750 块）
MAX_LEN=8192 REASONING_PARSER= AGENTOS_MODEL_PATH=$HOME/models/Qwen2.5-7B-Instruct SERVED_NAME=qwen2.5-7b \
  experiments/local_vllm/restart.sh --num-gpu-blocks-override 750
python experiments/cache_affinity/run.py --base-url http://127.0.0.1:18095 --model qwen2.5-7b \
  --condition tight750 --arms latency_first,cache_affinity --reps 10 --output OUT
python experiments/cache_affinity/analyze.py OUT --output OUT/report.json
```

原始记录在服务器 `~/agentos-runs/expB/`（每个条件一个目录：`summary.json`、逐请求 `requests_*.jsonl`、Runtime 事件 `events_*.jsonl`、vLLM 日志）。汇总表见同目录 `report*.md`。
