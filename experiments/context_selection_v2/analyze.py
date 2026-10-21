"""Analyze frozen-plan context-selection runs without contacting any backend.

All inferential statistics resample question IDs, after averaging replicates.
The primary paired estimand uses questions for which every replicate is valid
in both arms. ITT additionally retains failed, non-truncated questions as wrong.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Sequence

from .common import ARMS, PROTOCOL_VERSION
from .freeze import sum_usage
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "cached_tokens")
INVARIANT_CONFIG = (
    "model", "seed", "repeats", "temperature", "max_tokens",
    "max_completion_tokens", "backend", "base_url", "prompt_version", "concurrency", "enable_thinking",
)
DEFAULT_SEED = 20260920


def _number(value: Any, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer" + (" or null" if nullable else ""))


def load_summaries(inputs: Sequence[Path | str]) -> dict[str, dict[str, Any]]:
    """Read explicit summary files, arm directories, or roots containing arm/summary.json."""
    paths: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            paths.add(path.resolve())
        elif (path / "summary.json").is_file():
            paths.add((path / "summary.json").resolve())
        elif path.is_dir():
            found = [path / arm / "summary.json" for arm in ARMS]
            paths.update(p.resolve() for p in found if p.is_file())
            if not any(p.is_file() for p in found):
                raise ValueError(f"no arm summaries found in {path}")
        else:
            raise ValueError(f"input does not exist: {path}")
    result = {}
    for path in sorted(paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"summary must be an object: {path}")
        arm = data.get("arm")
        if arm in result:
            raise ValueError(f"duplicate arm {arm!r}: {path}")
        result[arm] = data
    validate_summaries(result)
    return result


def validate_summaries(summaries: Mapping[str, Mapping[str, Any]]) -> None:
    """Reject incomparable runs instead of silently pairing unrelated experiments."""
    if "all" not in summaries or len(summaries) < 2:
        raise ValueError("analysis requires all baseline and at least one other arm")
    reference = summaries["all"]
    reference_keys = None
    for arm, summary in summaries.items():
        if arm not in ARMS or summary.get("arm") != arm:
            raise ValueError(f"invalid or mismatched arm: {arm!r}")
        if summary.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("incomparable protocol")
        if summary.get("schema_version") != 1:
            raise ValueError(f"unsupported schema_version in {arm}")
        for field in ("split", "manifest_sha256", "plans_sha256"):
            if not isinstance(summary.get(field), str) or not summary[field]:
                raise ValueError(f"missing {field} in {arm}")
            if summary[field] != reference.get(field):
                raise ValueError(f"incomparable {field} in {arm}")
        config = summary.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"missing config in {arm}")
        for field in INVARIANT_CONFIG:
            if config.get(field) != reference.get("config", {}).get(field):
                raise ValueError(f"incomparable config.{field} in {arm}")
        repeats = config.get("repeats")
        if repeats is not None:
            _number(repeats, f"{arm}.config.repeats")
            if repeats == 0:
                raise ValueError("repeats must be positive")
        rows = summary.get("questions")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"questions must be a nonempty list in {arm}")
        keys = set()
        repetitions: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("question_id"), str) or not row["question_id"]:
                raise ValueError(f"invalid question in {arm}")
            qid = row["question_id"]
            _number(row.get("replicate"), f"{arm}.{qid}.replicate")
            key = (qid, row["replicate"])
            if key in keys:
                raise ValueError(f"duplicate question/replicate in {arm}: {key}")
            keys.add(key)
            repetitions[qid].add(row["replicate"])
            for field in ("valid", "correct"):
                if not isinstance(row.get(field), bool):
                    raise ValueError(f"{arm}.{qid}.{field} must be boolean")
            for field in TOKEN_FIELDS:
                _number(row.get(field), f"{arm}.{qid}.{field}", nullable=True)
            for field in ("model_calls", "truncated_calls"):
                _number(row.get(field), f"{arm}.{qid}.{field}")
            for field in ("execution_truncated_calls", "planner_truncated_calls"):
                if field in row:
                    _number(row[field], f"{arm}.{qid}.{field}")
            if _execution_truncated(row) > row["model_calls"]:
                raise ValueError(f"execution truncated_calls exceeds model_calls in {arm}.{qid}")
            if not isinstance(row.get("failures", []), list):
                raise ValueError(f"{arm}.{qid}.failures must be a list")
            recall = row.get("required_recall")
            if recall is not None and (isinstance(recall, bool) or not isinstance(recall, (int, float)) or not math.isfinite(recall) or not 0 <= recall <= 1):
                raise ValueError(f"invalid required_recall in {arm}.{qid}")
            if row.get("complete_required") is not None and not isinstance(row["complete_required"], bool):
                raise ValueError(f"invalid complete_required in {arm}.{qid}")
        rep_sets = list(repetitions.values())
        if any(reps != rep_sets[0] for reps in rep_sets):
            raise ValueError(f"incomplete replicate grid in {arm}")
        if repeats is not None and len(rep_sets[0]) != repeats:
            raise ValueError(f"replicate count differs from config.repeats in {arm}")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(f"question/replicate grid differs in {arm}")
        planning = summary.get("planning")
        if (planning is None) != (reference.get("planning") is None):
            raise ValueError(f"shared planning presence differs in {arm}")
        if planning is not None:
            if not isinstance(planning, dict):
                raise ValueError(f"planning must be an object in {arm}")
            for field in ("model_calls", "truncated_calls", *TOKEN_FIELDS):
                _number(planning.get(field), f"{arm}.planning.{field}", nullable=True)
            for field in ("model_calls", "truncated_calls", *TOKEN_FIELDS):
                if planning.get(field) != (reference.get("planning") or {}).get(field):
                    raise ValueError(f"shared planning.{field} differs in {arm}")


def _execution_truncated(row: Mapping[str, Any]) -> int:
    return row.get("execution_truncated_calls", row["truncated_calls"])


def _truncated(row: Mapping[str, Any]) -> bool:
    return row["truncated_calls"] > 0 or row.get("planner_truncated_calls", 0) > 0 or _execution_truncated(row) > 0


def _mean_or_none(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def _questions(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["question_id"]].append(row)
    result = {}
    for qid, group in sorted(grouped.items()):
        truncated = any(_truncated(row) for row in group)
        valid = not truncated and all(row["valid"] for row in group)
        complete = [row.get("complete_required") for row in group]
        item = {
            "replicates": len(group), "valid": valid, "truncated": truncated,
            "accuracy": mean(float(row["correct"] and row["valid"] and not _truncated(row)) for row in group),
            "required_recall": _mean_or_none([row["required_recall"] for row in group if row.get("required_recall") is not None]),
            "complete_required": False if False in complete else (True if all(v is True for v in complete) else None),
        }
        for field in (*TOKEN_FIELDS, "model_calls"):
            item[field] = mean(row[field] for row in group) if all(row.get(field) is not None for row in group) else None
        result[qid] = item
    return result


def _quality(questions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    itt = [q for q in questions if not q["truncated"]]
    valid = [q for q in questions if q["valid"]]
    prompt = [q["prompt_tokens"] for q in valid if q["prompt_tokens"] is not None]
    return {
        "questions": len(questions),
        "itt_questions": len(itt), "itt_accuracy": _mean_or_none([q["accuracy"] for q in itt]),
        "valid_questions": len(valid), "valid_accuracy": _mean_or_none([q["accuracy"] for q in valid]),
        "mean_prompt_tokens_valid": _mean_or_none(prompt), "prompt_tokens_valid_questions": len(prompt),
        "mean_model_calls_valid": _mean_or_none([q["model_calls"] for q in valid]),
        "mean_prompt_tokens_all_questions": _mean_or_none([q["prompt_tokens"] for q in questions if q["prompt_tokens"] is not None]),
        "prompt_tokens_all_known_questions": sum(q["prompt_tokens"] is not None for q in questions),
        "mean_model_calls_all_questions": _mean_or_none([q["model_calls"] for q in questions]),
    }


def _cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"model_calls": sum(row["model_calls"] for row in rows)}
    for field in TOKEN_FIELDS:
        known = [row[field] for row in rows if row.get(field) is not None]
        result[field] = sum(known) if len(known) == len(rows) else None
        result[f"known_{field}"] = sum(known)
        result[f"missing_{field}_runs"] = len(rows) - len(known)
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _paired_metric(pairs: Sequence[tuple[float, float]], samples: int, seed: int, *, relative: bool = False) -> dict[str, Any]:
    """Each pair is (arm question mean, baseline question mean)."""
    if not pairs:
        return {"questions": 0, "arm_mean": None, "all_mean": None, "difference": None, "ci95": None,
                **({"relative_change": None, "relative_ci95": None} if relative else {})}
    n = len(pairs)
    arm_mean = mean(p[0] for p in pairs)
    all_mean = mean(p[1] for p in pairs)
    rng = random.Random(seed)
    differences = []
    relative_changes = []
    for _ in range(samples):
        draw = [pairs[rng.randrange(n)] for _ in range(n)]
        arm_sample = mean(p[0] for p in draw)
        all_sample = mean(p[1] for p in draw)
        differences.append(arm_sample - all_sample)
        if relative and all_sample != 0:
            relative_changes.append(arm_sample / all_sample - 1)
    result = {"questions": n, "arm_mean": arm_mean, "all_mean": all_mean,
              "difference": arm_mean - all_mean,
              "ci95": [_quantile(differences, .025), _quantile(differences, .975)]}
    if relative:
        result.update(relative_change=arm_mean / all_mean - 1 if all_mean else None,
                      relative_ci95=[_quantile(relative_changes, .025), _quantile(relative_changes, .975)] if len(relative_changes) == samples else None)
    return result


def _base_analysis(summaries: Mapping[str, Mapping[str, Any]], bootstrap_samples: int = 10000, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    validate_summaries(summaries)
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    questions = {arm: _questions(summary["questions"]) for arm, summary in summaries.items()}
    baseline = summaries["all"]
    report: dict[str, Any] = {
        "schema_version": 1, "split": baseline["split"], "manifest_sha256": baseline["manifest_sha256"],
        "plans_sha256": baseline["plans_sha256"], "config": {key: baseline["config"].get(key) for key in INVARIANT_CONFIG},
        "bootstrap": {"samples": bootstrap_samples, "seed": seed, "unit": "question_id", "confidence": .95, "method": "paired percentile"},
        "protocol": {
            "replicates": "average within question before any group mean or bootstrap",
            "valid": "all replicates valid and no truncated request for that question",
            "itt": "all non-truncated questions; invalid runs count as incorrect",
            "paired": "questions valid in both arms; token pairs additionally require known usage",
            "cost": "main comparison excludes shared planner; including_planning adds shared planning once per arm",
            "missing_usage": "never imputed as zero; known totals and missing-run counts retained",
        },
        "arms": {}, "comparisons": {},
    }
    for arm in ARMS:
        if arm not in summaries:
            continue
        summary = summaries[arm]
        rows = summary["questions"]
        qs = list(questions[arm].values())
        execution = _cost(rows)
        planning = dict(summary.get("planning") or {})
        for field in ("model_calls", "truncated_calls", *TOKEN_FIELDS):
            planning.setdefault(field, None)
        including = {field: execution[field] + planning[field] if execution.get(field) is not None and planning[field] is not None else None
                     for field in ("model_calls", *TOKEN_FIELDS)}
        execution_truncated = sum(_execution_truncated(row) for row in rows)
        planner_calls = planning["model_calls"]
        planner_truncated = planning["truncated_calls"]
        details = lambda row: {"question_id": row["question_id"], "replicate": row["replicate"],
                               "valid": row["valid"], "truncated_calls": row["truncated_calls"],
                               "execution_truncated_calls": _execution_truncated(row),
                               "planner_truncated_calls": row.get("planner_truncated_calls", 0), "failures": row.get("failures", [])}
        arm_report = {
            **_quality(qs), "runs": len(rows), "execution": execution, "planning": planning, "including_planning": including,
            "mean_required_recall": _mean_or_none([q["required_recall"] for q in qs if q["required_recall"] is not None]),
            "complete_required_questions": sum(q["complete_required"] is True for q in qs),
            "recall_measured_questions": sum(q["complete_required"] is not None for q in qs),
            "strata": {name: _quality([q for q in qs if q["complete_required"] is condition])
                       for name, condition in (("complete", True), ("incomplete", False), ("unknown", None))},
            "truncation": {
                "questions": sum(q["truncated"] for q in qs), "runs": sum(_truncated(row) for row in rows),
                "execution_calls": execution_truncated,
                "execution_rate": execution_truncated / execution["model_calls"] if execution["model_calls"] else None,
                "planning_calls": planner_truncated,
                "planning_rate": planner_truncated / planner_calls if planner_calls and planner_truncated is not None else None,
            },
            "truncated": [details(row) for row in rows if _truncated(row)],
            "invalid": [details(row) for row in rows if not row["valid"] or _truncated(row)],
            "failures": [details(row) for row in rows if row.get("failures")],
        }
        report["arms"][arm] = arm_report
    total_questions = len(questions["all"])
    for arm in ARMS[1:]:
        if arm not in summaries:
            continue
        common = sorted(qid for qid, q in questions[arm].items() if q["valid"] and questions["all"][qid]["valid"])
        accuracy_pairs = [(questions[arm][qid]["accuracy"], questions["all"][qid]["accuracy"]) for qid in common]
        token_ids = [qid for qid in common if questions[arm][qid]["prompt_tokens"] is not None and questions["all"][qid]["prompt_tokens"] is not None]
        token_pairs = [(questions[arm][qid]["prompt_tokens"], questions["all"][qid]["prompt_tokens"]) for qid in token_ids]
        report["comparisons"][arm] = {
            "baseline": "all", "questions": total_questions, "paired_valid_questions": len(common),
            "paired_valid_coverage": len(common) / total_questions,
            "paired_valid_question_ids": common,
            "excluded_question_ids": sorted(set(questions["all"]) - set(common)),
            "prompt_usage_coverage": len(token_ids) / total_questions,
            "accuracy": _paired_metric(accuracy_pairs, bootstrap_samples, seed),
            "prompt_tokens": _paired_metric(token_pairs, bootstrap_samples, seed, relative=True),
        }
    return report


def _fmt(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.2f}%" if percent else f"{value:,.2f}"


def _ci(value: Mapping[str, Any], *, percentage_points: bool = False) -> str:
    interval = value["ci95"]
    if interval is None:
        return "NA"
    scale = 100 if percentage_points else 1
    suffix = " pp" if percentage_points else ""
    return f"{value['difference'] * scale:.2f}{suffix} [{interval[0] * scale:.2f}, {interval[1] * scale:.2f}]"


def _base_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Context selection analysis", "",
             ("Backend: stub. These are fake answers and character-based token estimates; no model quality or token-saving conclusion can be drawn."
              if report["config"].get("backend") == "stub" else "Backend: provider; token counts come from reported request usage."), "",
             f"Split: `{report['split']}`; manifest: `{report['manifest_sha256']}`; plans: `{report['plans_sha256']}`.", "",
             "Replicates are averaged within each question. ITT includes every non-truncated question and counts failures as incorrect. Valid accuracy requires every replicate to be valid. Any truncated replicate excludes its entire question from ITT, valid means, and paired inference.", "",
             "| Arm | Questions / runs | ITT accuracy (n) | Valid accuracy (n) | Mean input / valid question replicate | Mean calls / valid question replicate | Execution truncation | Planner truncation |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, a in report["arms"].items():
        trunc = a["truncation"]
        lines.append(f"| {arm} | {a['questions']} / {a['runs']} | {_fmt(a['itt_accuracy'], percent=True)} ({a['itt_questions']}) | {_fmt(a['valid_accuracy'], percent=True)} ({a['valid_questions']}) | {_fmt(a['mean_prompt_tokens_valid'])} | {_fmt(a['mean_model_calls_valid'])} | {_fmt(trunc['execution_rate'], percent=True)} ({trunc['execution_calls']}) | {_fmt(trunc['planning_rate'], percent=True)} ({trunc['planning_calls']}) |")
    lines += ["", "Shared planner cost is excluded above and from paired comparisons. The table below shows recorded aggregate input, including failed and truncated runs. NA means usage was missing; it is never replaced with zero.", "",
              "| Arm | Execution calls | Execution input | Known input / missing runs | Shared planner calls / input | Including planner calls / input |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, a in report["arms"].items():
        ex, pl, total = a["execution"], a["planning"], a["including_planning"]
        lines.append(f"| {arm} | {_fmt(ex['model_calls'])} | {_fmt(ex['prompt_tokens'])} | {_fmt(ex['known_prompt_tokens'])} / {ex['missing_prompt_tokens_runs']} | {_fmt(pl['model_calls'])} / {_fmt(pl['prompt_tokens'])} | {_fmt(total['model_calls'])} / {_fmt(total['prompt_tokens'])} |")
    lines += ["", "Paired differences are arm minus all. A negative input difference denotes savings. Confidence intervals use question-level paired percentile bootstrap.", "",
              "| Arm vs all | Valid coverage | Accuracy difference [95% CI] | Input difference [95% CI] | Relative input change | Token coverage |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, c in report["comparisons"].items():
        lines.append(f"| {arm} | {c['paired_valid_questions']} / {c['questions']} ({_fmt(c['paired_valid_coverage'], percent=True)}) | {_ci(c['accuracy'], percentage_points=True)} | {_ci(c['prompt_tokens'])} | {_fmt(c['prompt_tokens']['relative_change'], percent=True)} | {_fmt(c['prompt_usage_coverage'], percent=True)} |")
    lines += ["", "| Arm | Required-document selection | Questions | ITT accuracy | Valid accuracy |",
              "| --- | --- | ---: | ---: | ---: |"]
    for arm, a in report["arms"].items():
        for name, stratum in a["strata"].items():
            lines.append(f"| {arm} | {name} | {stratum['questions']} | {_fmt(stratum['itt_accuracy'], percent=True)} | {_fmt(stratum['valid_accuracy'], percent=True)} |")
    lines += ["", "Invalid, truncated, and failed runs (question IDs and full failure details):", "", "```json",
              json.dumps({arm: {key: a[key] for key in ("invalid", "truncated", "failures")} for arm, a in report["arms"].items()}, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def paired_report(questions, arm, baseline, samples, seed, *, itt=False):
    common = sorted(qid for qid, q in questions[arm].items()
                    if (not q["truncated"] and not questions[baseline][qid]["truncated"] if itt
                        else q["valid"] and questions[baseline][qid]["valid"]))
    result = {"baseline": baseline, "question_ids": common, "questions": len(common)}
    for field in ("accuracy", "prompt_tokens", "model_calls"):
        pairs = [(questions[arm][qid][field], questions[baseline][qid][field]) for qid in common
                 if questions[arm][qid][field] is not None and questions[baseline][qid][field] is not None]
        result[field] = _paired_metric(pairs, samples, seed, relative=field == "prompt_tokens")
    return result


def enrich(summaries, samples, seed):
    report = _base_analysis(summaries, samples, seed)
    by_question = {a: _questions(s["questions"]) for a, s in summaries.items()}
    first = {r["question_id"]: r for r in summaries["all"]["questions"]}
    recalls = [r["planner_recall"] for r in first.values() if r["planner_recall"] is not None]
    report["planner_declaration"] = {
        "zero": sum(r == 0 for r in recalls), "partial": sum(0 < r < 1 for r in recalls),
        "complete": sum(r == 1 for r in recalls), "failed_unknown": len(first)-len(recalls),
        "mean_recall": _mean_or_none(recalls),
        "per_question": {qid: r["planner_recall"] for qid, r in first.items()}}
    report["planning_once"] = summaries["all"]["planning"]
    plans = summaries["all"]["planning_by_question"]
    report["planner_format"] = {"fence_stripped": sum(p.get("fence_stripped", 0) for p in plans.values()),
        "format_corrections": sum(p["format_corrections"] for p in plans.values()),
        "details": {qid: {k: p.get(k) for k in ("fence_details", "format_corrections", "error", "status")}
                    for qid, p in plans.items() if p.get("fence_stripped") or p.get("format_corrections") or p.get("error")}}
    report["paired_itt"] = {a: paired_report(by_question, a, "all", samples, seed, itt=True)
                            for a in summaries if a != "all"}
    for arm, summary in summaries.items():
        rows = summary["questions"]
        a = report["arms"][arm]
        a["planner_strata"] = {name: _quality([q for qid, q in by_question[arm].items()
            if first[qid]["planner_complete"] is value])
            for name, value in (("complete", True), ("incomplete", False), ("unknown", None))}
        events = [event for r in rows for event in r.get("context_requests", [])]
        reader_runs = sum(len(r.get("final_documents", {})) for r in rows)
        requested_readers = sum(len({e["task_id"] for e in r.get("context_requests", [])}) for r in rows)
        granted = sum(r.get("append_granted", 0) for r in rows)
        a["append"] = {
            "requests": len(events), "reader_runs": reader_runs, "requesting_reader_runs": requested_readers,
            "reader_request_rate": requested_readers / reader_runs if reader_runs else None,
            "question_run_request_rate": sum(bool(r.get("context_requests")) for r in rows) / len(rows),
            "granted_refs": granted, "required_granted_refs": sum(r.get("append_required", 0) for r in rows),
            "required_fraction_of_grants": sum(r.get("append_required", 0) for r in rows) / granted if granted else None,
            "denied_requests": sum(bool(e["denied"]) for e in events),
            "denied_refs": sum(len(e["denied"]) for e in events),
            "post_append_required_recall": _mean_or_none([q["required_recall"] for q in by_question[arm].values() if q["required_recall"] is not None]),
            "additional_calls_after_initial_reader_calls": sum(max(0, r["model_calls"] - len(r.get("final_documents", {})) - int("answerer" in r.get("actual_documents", {}))) for r in rows),
        }
        a["format"] = {"fence_stripped": sum(r.get("fence_stripped", 0) for r in rows),
            "details": [{"question_id": r["question_id"], "replicate": r["replicate"],
                         "events": r.get("fence_details", [])} for r in rows if r.get("fence_stripped")]}
    if "declared_append" in summaries and "declared" in summaries:
        report["append_minus_declared"] = paired_report(by_question, "declared_append", "declared", samples, seed)
        report["append_minus_declared_itt"] = paired_report(by_question, "declared_append", "declared", samples, seed, itt=True)
    return report


def analyze(summaries, bootstrap_samples=10000, seed=DEFAULT_SEED):
    validate_summaries(summaries)
    report = {"overall": enrich(summaries, bootstrap_samples, seed), "families": {}}
    families = sorted({r["family"] for r in summaries["all"]["questions"]})
    for family in families:
        subset = {}
        for arm, summary in summaries.items():
            rows = [r for r in summary["questions"] if r["family"] == family]
            ids = {r["question_id"] for r in rows}
            plans = {qid: p for qid, p in summary["planning_by_question"].items() if qid in ids}
            subset[arm] = {**summary, "questions": rows, "planning_by_question": plans,
                           "planning": sum_usage(list(plans.values()))}
        report["families"][family] = enrich(subset, bootstrap_samples, seed)
    return report


def render_markdown(report):
    sections = []
    for label, r in [("all questions", report["overall"]), *report["families"].items()]:
        sections += [f"# {label}", "", _base_markdown(r), "", "Planner declaration recall:", "",
                     "| Zero | Partial | Complete | Failed/unknown | Mean |",
                     "| ---: | ---: | ---: | ---: | ---: |"]
        p = r["planner_declaration"]
        sections.append(f"| {p['zero']} | {p['partial']} | {p['complete']} | {p['failed_unknown']} | {_fmt(p['mean_recall'])} |")
        sections += ["", "| Arm | Planner declaration | Questions | ITT accuracy | Valid accuracy |",
                     "| --- | --- | ---: | ---: | ---: |"]
        for arm, a in r["arms"].items():
            for name, s in a["planner_strata"].items():
                sections.append(f"| {arm} | {name} | {s['questions']} | {_fmt(s['itt_accuracy'], percent=True)} | {_fmt(s['valid_accuracy'], percent=True)} |")
        sections += ["", "| Arm | Reader request rate | Requests / denied | Granted / required | Required fraction | Recall after requests | Fence strips |",
                     "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for arm, a in r["arms"].items():
            x = a["append"]
            sections.append(f"| {arm} | {_fmt(x['reader_request_rate'], percent=True)} | {x['requests']} / {x['denied_requests']} | {x['granted_refs']} / {x['required_granted_refs']} | {_fmt(x['required_fraction_of_grants'], percent=True)} | {_fmt(x['post_append_required_recall'], percent=True)} | {a['format']['fence_stripped']} |")
        if "append_minus_declared" in r:
            c = r["append_minus_declared"]
            sections += ["", "declared_append − declared (paired valid questions):", "",
                         "| Accuracy difference [95% CI] | Input difference [95% CI] | Calls difference [95% CI] |",
                         "| ---: | ---: | ---: |",
                         f"| {_ci(c['accuracy'], percentage_points=True)} | {_ci(c['prompt_tokens'])} | {_ci(c['model_calls'])} |"]
        sections += ["", "Planner cost is shared once. Full format details (including each question/role) and ITT paired estimates are in JSON.", "",
                     "```json", json.dumps({"planning_once": r["planning_once"], "planner_format": r["planner_format"],
                         "execution_format": {a: x["format"] for a, x in r["arms"].items()}}, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(sections)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path, help="run root, arm directory, or summary JSON; repeatable")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        report = analyze(load_summaries(args.input), args.bootstrap_samples, args.seed)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "analysis.json"
    markdown_path = args.output / "analysis.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"analysis_json": str(json_path), "analysis_markdown": str(markdown_path), "arms": list(report["overall"]["arms"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
