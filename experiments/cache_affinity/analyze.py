"""实验 B 分析：各条件下两组的均值、按轮次配对的差值与 bootstrap 95% 置信区间。

用法：python analyze.py DIR [DIR ...] --output report.json
每个 DIR 是 run.py 的一个 --output 目录（一个 KV 条件）。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

METRICS = ("makespan", "hit_ratio", "miss_tokens", "mean_latency")


def boot(diffs, samples=10000, seed=20261019):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.fmean(rng.choice(diffs) for _ in range(n)) for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def load_requests(path):
    return {r["task_id"]: r for r in (json.loads(l) for l in path.open())}


def analyze_dir(d: Path, base="latency_first", treat="cache_affinity"):
    runs = json.loads((d / "summary.json").read_text())["runs"]
    by = {(r["arm"], r["rep"]): r for r in runs}
    reps = sorted({r["rep"] for r in runs})
    arms = list(dict.fromkeys(r["arm"] for r in runs))
    if {base, treat} - set(arms):
        # 非默认组合（如 auto 对 auto_service_time）：按 --arms 顺序，第一组为基准。
        base, treat = arms[0], arms[1]
    out ={"condition": runs[0]["condition"], "reps": len(reps), "arms": {}, "paired": {}}
    for arm in (base, treat):
        rows = [by[(arm, k)] for k in reps]
        out["arms"][arm] = {m: {"mean": statistics.fmean(r[m] for r in rows),
                                "sd": statistics.stdev(r[m] for r in rows) if len(rows) > 1 else 0.0}
                            for m in METRICS}
        out["arms"][arm]["truncated_total"] = sum(r["truncated"] for r in rows)
        out["arms"][arm]["preemptions_total"] = sum(r["server_delta"].get("num_preemptions_total") or 0
                                                    for r in rows)
        out["arms"][arm]["requests_total"] = sum(r["requests"] for r in rows)
    for m in METRICS:
        diffs = [by[(treat, k)][m] - by[(base, k)][m] for k in reps]
        rel = [by[(treat, k)][m] / by[(base, k)][m] - 1 for k in reps if by[(base, k)][m]]
        lo, hi = boot(diffs)
        rlo, rhi = boot(rel) if rel else (None, None)
        out["paired"][m] = {"mean_diff": statistics.fmean(diffs), "ci95": [lo, hi],
                            "mean_rel": statistics.fmean(rel) if rel else None, "rel_ci95": [rlo, rhi]}
    # 同一轮内两组对同一任务的输出是否相同（温度 0）。早期运行只存了 content，
    # 而推理解析器把正文放进了 reasoning_content，content 全空；这类运行只能比较输出 Token 数。
    same = total = 0
    full_text = True
    lat = {base: [], treat: []}
    for k in reps:
        a = load_requests(d / f"requests_{base}_rep{k}.jsonl")
        b = load_requests(d / f"requests_{treat}_rep{k}.jsonl")
        lat[base] += [r["latency"] for r in a.values()]
        lat[treat] += [r["latency"] for r in b.values()]
        for tid in a:
            total += 1
            if "reasoning_content" in a[tid]:
                same += a[tid]["content_sha256"] == b[tid]["content_sha256"]
            else:
                full_text = False
                same += a[tid]["completion_tokens"] == b[tid]["completion_tokens"]
    out["identical_outputs"] = {"same": same, "total": total,
                                "basis": "text" if full_text else "completion_tokens"}
    for arm, xs in lat.items():
        xs.sort()
        out["arms"][arm]["latency_p95"] = xs[int(0.95 * (len(xs) - 1))]
        out["arms"][arm]["latency_max"] = xs[-1]
    return out


def table(results):
    lines = ["| 条件 | 组 | 完工时间 s | 命中率 | 未命中 Token | 单请求时延 s | 截断 | 抢占 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for res in results:
        for arm, v in res["arms"].items():
            lines.append(f"| {res['condition']} | {arm} | {v['makespan']['mean']:.2f} ± {v['makespan']['sd']:.2f} | "
                         f"{v['hit_ratio']['mean']:.1%} ± {v['hit_ratio']['sd']:.1%} | "
                         f"{v['miss_tokens']['mean']:,.0f} | {v['mean_latency']['mean']:.2f} | "
                         f"{v['truncated_total']}/{v['requests_total']} | {v['preemptions_total']:.0f} |")
    lines += ["", "| 条件 | 指标 | 亲和相对基准（配对均值） | 95% CI | 相对变化 | 相对变化 95% CI |",
              "|---|---|---:|---|---:|---|"]
    for res in results:
        for m, p in res["paired"].items():
            rel = "—" if p["mean_rel"] is None else f"{p['mean_rel']:+.1%}"
            rci = "—" if p["rel_ci95"][0] is None else f"[{p['rel_ci95'][0]:+.1%}, {p['rel_ci95'][1]:+.1%}]"
            lines.append(f"| {res['condition']} | {m} | {p['mean_diff']:+.4g} | "
                         f"[{p['ci95'][0]:+.4g}, {p['ci95'][1]:+.4g}] | {rel} | {rci} |")
    lines += ["", "| 条件 | 组 | 单请求时延 p95 s | 最大 s |", "|---|---|---:|---:|"]
    for res in results:
        for arm, v in res["arms"].items():
            lines.append(f"| {res['condition']} | {arm} | {v['latency_p95']:.2f} | {v['latency_max']:.2f} |")
    lines += ["", "| 条件 | 两组输出相同 | 比较依据 |", "|---|---:|---|"]
    for res in results:
        io = res["identical_outputs"]
        basis = "正文逐字" if io["basis"] == "text" else "输出 Token 数（正文未保存）"
        lines.append(f"| {res['condition']} | {io['same']}/{io['total']} | {basis} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    results = [analyze_dir(d) for d in args.dirs]
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    md = table(results)
    args.output.with_suffix(".md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
