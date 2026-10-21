"""Question-paired, intention-to-treat analysis, stratified by injected failure type."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
from statistics import mean

from .common import ARMS, PROTOCOL_VERSION, load, write_new

INVARIANT_CONFIG = ("backend", "model", "base_url", "seed", "max_tokens", "temperature",
                    "enable_thinking", "retries", "concurrency", "timeout", "api_key_env")
METRICS = ("correct", "completed", "model_calls", "prompt_tokens", "completion_tokens", "attempts")


def validate_summaries(summaries):
    if not {"dirty", "snapshot"} <= set(summaries) or set(summaries) - set(ARMS):
        raise ValueError("analysis requires dirty and snapshot; no_retry is optional")
    baseline = summaries["dirty"]
    grid = None
    assignments = None
    for arm, summary in summaries.items():
        if summary.get("schema_version") != 1:
            raise ValueError("unsupported summary schema")
        if summary["arm"] != arm or summary["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("arm/protocol mismatch")
        for key in ("protocol_version", "split", "manifest_sha256", "prompt_sha256", "token_measurement"):
            if summary[key] != baseline[key]:
                raise ValueError(f"incomparable summaries: {key}")
        for key in INVARIANT_CONFIG:
            if summary["config"].get(key) != baseline["config"].get(key):
                raise ValueError(f"incomparable model/configuration: {key}")
        comparable = lambda config: {key: value for key, value in config.items()
                                     if key not in ("max_attempts", "restore_context")}
        if comparable(summary["config"]) != comparable(baseline["config"]):
            raise ValueError("incomparable model/configuration")
        if summary["config"]["max_attempts"] != (1 if arm == "no_retry" else 2):
            raise ValueError("unexpected maximum attempts")
        if summary["config"]["restore_context"] != (arm == "snapshot"):
            raise ValueError("unexpected restoration condition")
        rows = summary["questions"]
        for row in rows:
            if row["injection_type"] not in ("execution", "semantic"):
                raise ValueError("unknown injection type")
            if not isinstance(row["question_id"], str):
                raise ValueError("question_id must be a string")
            for field in ("completed", "correct", "injection_reached", "semantic_injection_detected"):
                if not isinstance(row[field], bool):
                    raise ValueError("invalid boolean outcome: " + field)
            for field in ("model_calls", "attempts", "truncated_calls", "injection_after_private_segments"):
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("invalid outcome counter: " + field)
            for field in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                value = row.get(field)
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                    raise ValueError("invalid nullable token counter: " + field)
            if row["attempts"] > summary["config"]["max_attempts"] or row["truncated_calls"] > row["model_calls"]:
                raise ValueError("inconsistent outcome counters")
            if row["correct"] and not row["completed"]:
                raise ValueError("correct output must be accepted as completed")
        ids = [row["question_id"] for row in rows]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("empty or duplicated question grid")
        current = {row["question_id"]: (row["injection_type"], row["injection_after_private_segments"]) for row in rows}
        if grid is not None and (set(ids) != grid or current != assignments):
            raise ValueError("question grid or fault assignments differ")
        grid, assignments = set(ids), current


def _mean(values):
    return mean(values) if values else None


def aggregate(rows):
    semantic = [row for row in rows if row["injection_type"] == "semantic"]
    reached = [row for row in semantic if row["injection_reached"]]
    result = {
        "questions": len(rows), "accuracy": _mean([float(row["correct"]) for row in rows]),
        "completion_rate": _mean([float(row["completed"]) for row in rows]),
        "attempts_distribution": dict(sorted(Counter(str(row["attempts"]) for row in rows).items())),
        "semantic_injections_scheduled": len(semantic), "semantic_injections_reached": len(reached),
        "semantic_failures_detected": sum(row["semantic_injection_detected"] for row in semantic),
        "semantic_detection_rate_scheduled": _mean([float(row["semantic_injection_detected"]) for row in semantic]),
        "semantic_detection_rate_reached": _mean([float(row["semantic_injection_detected"]) for row in reached]),
        "truncated_calls": sum(row["truncated_calls"] for row in rows),
        "truncated_questions": sum(row["truncated_calls"] > 0 for row in rows),
        "injections_reached": sum(row["injection_reached"] for row in rows),
        "total_model_calls": sum(row["model_calls"] for row in rows),
    }
    for field in ("model_calls", "prompt_tokens", "completion_tokens", "cached_tokens", "attempts"):
        known = [row[field] for row in rows if row.get(field) is not None]
        result["mean_" + field] = _mean(known) if len(known) == len(rows) else None
        result["total_" + field] = sum(known) if len(known) == len(rows) else None
        result["known_total_" + field] = sum(known)
        result["missing_" + field + "_questions"] = len(rows) - len(known)
    return result


def _quantile(values, p):
    values = sorted(values)
    i = (len(values) - 1) * p
    low, high = math.floor(i), math.ceil(i)
    return values[low] + (values[high] - values[low]) * (i - low)


def paired_metric(pairs, *, samples=10000, seed=20261019):
    if not pairs:
        return {"questions": 0, "snapshot_mean": None, "dirty_mean": None, "difference": None, "ci95": None}
    rng = random.Random(seed)
    differences = [float(a) - float(b) for a, b in pairs]
    n = len(differences)
    draws = [mean(differences[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {"questions": n, "snapshot_mean": mean(float(pair[0]) for pair in pairs),
            "dirty_mean": mean(float(pair[1]) for pair in pairs), "difference": mean(differences),
            "ci95": [_quantile(draws, .025), _quantile(draws, .975)]}


def analyze(summaries, *, bootstrap_samples=10000, seed=20261019):
    validate_summaries(summaries)
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    baseline = summaries["dirty"]
    questions = {arm: {row["question_id"]: row for row in summary["questions"]}
                 for arm, summary in summaries.items()}
    report = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "split": baseline["split"],
        "manifest_sha256": baseline["manifest_sha256"], "prompt_sha256": baseline["prompt_sha256"],
        "config": {key: value for key, value in baseline["config"].items()
                   if key not in ("max_attempts", "restore_context")},
        "token_measurement": baseline["token_measurement"],
        "bootstrap": {"samples": bootstrap_samples, "seed": seed, "unit": "question_id", "confidence": .95,
                      "method": "paired percentile; snapshot minus dirty"},
        "protocol": {"population": "all scheduled questions including failed and truncated (ITT)",
            "missing_usage": "no zero imputation; token pairs require both usages and report coverage",
            "completion": "Runtime task completed after the exact-answer validator accepted it",
            "no_retry": "descriptive no-recovery baseline, not part of the single-factor paired comparison",
            "stub": "pipeline checks only; fake AST backend and char/4 estimates are not model-quality evidence"},
        "arms": {}, "paired_snapshot_minus_dirty": {},
    }
    for arm, summary in summaries.items():
        rows = summary["questions"]
        report["arms"][arm] = {
            **aggregate(rows),
            "strata": {kind: aggregate([row for row in rows if row["injection_type"] == kind])
                       for kind in ("execution", "semantic")},
            "failures": [{"question_id": row["question_id"], "error": row["error"], "task_status": row["task_status"]}
                         for row in rows if not row["completed"]],
        }
    for stratum in ("all", "execution", "semantic"):
        ids = sorted(qid for qid, row in questions["dirty"].items()
                     if stratum == "all" or row["injection_type"] == stratum)
        result = {"questions": len(ids), "question_ids": ids, "metrics": {}}
        for field in METRICS:
            paired_ids = [qid for qid in ids if questions["snapshot"][qid].get(field) is not None
                          and questions["dirty"][qid].get(field) is not None]
            pairs = [(questions["snapshot"][qid][field], questions["dirty"][qid][field]) for qid in paired_ids]
            result["metrics"][field] = {
                **paired_metric(pairs, samples=bootstrap_samples, seed=seed),
                "coverage": len(paired_ids) / len(ids) if ids else None,
                "excluded_question_ids": sorted(set(ids) - set(paired_ids)),
            }
        report["paired_snapshot_minus_dirty"][stratum] = result
    return report


def load_summaries(paths):
    summaries = {}
    for path in paths:
        path = Path(path)
        candidates = [path] if path.is_file() else sorted(path.rglob("summary.json"))
        for candidate in candidates:
            summary = load(candidate)
            if "arm" not in summary or "questions" not in summary:
                continue
            if summary["arm"] in summaries:
                raise ValueError("duplicate arm summary: " + summary["arm"])
            summaries[summary["arm"]] = summary
    return summaries


def render_markdown(report):
    lines = ["# Experiment C", "",
        ("Stub 管道验证：AST 假后端与字符估计不能用于模型效果推断。"
         if report["config"]["backend"] == "stub" else "逐请求 Token 来自服务端 usage。"),
        "", "全部分配题（包括失败和截断）保留在 ITT，按最终验收状态评分；截断另行计数。完成须通过精确答案校验，因此完成率与准确率在本协议中一致。",
        "", "| Arm | 题数 | 准确率 | 完成率 | 调用数 | 输入 Token | 截断调用 | 语义检出/计划 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, row in report["arms"].items():
        lines.append(f"| {arm} | {row['questions']} | {row['accuracy']} | {row['completion_rate']} | {row['total_model_calls']} | {row['total_prompt_tokens']} | {row['truncated_calls']} | {row['semantic_failures_detected']}/{row['semantic_injections_scheduled']} |")
    lines += ["", "配对差为 snapshot − dirty；95% CI 按题目成对重采样，执行与语义注入分别分层。输入 Token 为负表示节省；未知 usage 不填 0。",
              "", "| 分层 | 指标 | 配对题数 | 差值 | 95% CI |", "| --- | --- | ---: | ---: | --- |"]
    for stratum, comparison in report["paired_snapshot_minus_dirty"].items():
        for field, row in comparison["metrics"].items():
            lines.append(f"| {stratum} | {field} | {row['questions']} | {row['difference']} | {row['ci95']} |")
    lines += ["", "逐题失败、尝试分布、缺失 usage 覆盖与注入检出率详见 analysis.json。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    result = analyze(load_summaries(args.input), bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "analysis.json", result)
    (args.output / "analysis.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"analysis": str(args.output / "analysis.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
