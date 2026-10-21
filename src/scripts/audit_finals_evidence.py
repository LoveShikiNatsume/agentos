"""Read-only consistency audit of selected archived finals evidence.

This recomputes archived aggregates and checks file seals. It does not rerun a
model, attest original API calls, score SWE-bench patches, or activate sched_ext.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERIFIED_RUN = "national_experiments/20260709_swe_bench_verified_deepseek_30"
STAGE6_RUN = "member_a_sched_ext_validation/stage6_release/final-20260718-004616"
FRAMEWORKS = ("langgraph", "autogen", "openai_agents", "crewai")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_verified(root: Path) -> dict[str, Any]:
    summary = json.loads((root / "summary.json").read_text())
    rows = json.loads((root / "per_task.json").read_text())
    errors: list[str] = []
    if len(rows) != summary["n_tasks"] or len({row["instance_id"] for row in rows}) != len(rows):
        errors.append("task count or unique instance ids")
    per_framework: dict[str, Any] = {}
    totals = {mode: {key: 0 for key in ("prompt_tokens", "cache_hit_tokens", "cache_miss_tokens")} for mode in ("native", "runtime")}
    for fw in FRAMEWORKS:
        if fw not in summary["frameworks"] or any(fw not in row for row in rows):
            errors.append(f"missing framework: {fw}")
            continue
        def total(mode: str, metric: str) -> int:
            return sum(row[fw][mode][metric] for row in rows)
        archived = summary["frameworks"][fw]
        for metric in ("prompt_tokens", "cache_miss_tokens", "est_cost_usd"):
            before, after = total("native", metric), total("runtime", metric)
            if metric == "est_cost_usd":
                before, after = round(before, 4), round(after, 4)
            expected = {"native": before, "runtime": after,
                        "reduction_pct": round(100 * (1 - after / before), 1) if before else 0}
            if expected != archived["efficiency_no_fault"][metric]:
                errors.append(f"{fw}.{metric}: summary differs from per_task")
        resilience = archived["resilience_with_fault"]
        expected_completed = {"native": total("native_fault", "tasks_completed"),
                              "runtime": total("runtime_fault", "tasks_completed"),
                              "total": total("native_fault", "tasks_total")}
        expected_recovered = {"native": total("native_fault", "recovered_faults"),
                              "runtime": total("runtime_fault", "recovered_faults"),
                              "injected": total("runtime_fault", "faults_injected")}
        if expected_completed != resilience["tasks_completed"] or expected_recovered != resilience["recovered_faults"]:
            errors.append(f"{fw}: resilience totals differ")
        for row in rows:
            for mode in ("native", "runtime", "native_fault", "runtime_fault"):
                record = row[fw][mode]
                if record["cache_hit_tokens"] + record["cache_miss_tokens"] != record["prompt_tokens"]:
                    errors.append(f"{fw}/{row['instance_id']}/{mode}: hit+miss differs from prompt")
        for mode in totals:
            for metric in totals[mode]:
                totals[mode][metric] += total(mode, metric)
        per_framework[fw] = {
            "prompt_reduction_pct": archived["efficiency_no_fault"]["prompt_tokens"]["reduction_pct"],
            "native_prompt_tokens": total("native", "prompt_tokens"),
            "runtime_prompt_tokens": total("runtime", "prompt_tokens"),
        }
    ratios = {mode: round(100 * m["cache_hit_tokens"] / m["prompt_tokens"], 4)
              if m["prompt_tokens"] else 0 for mode, m in totals.items()}
    return {
        "passed": not errors, "errors": errors, "n_tasks": len(rows),
        "frameworks": per_framework, "pooled_usage": totals,
        "pooled_cache_hit_pct": ratios,
        "mean_framework_prompt_reduction_pct": round(sum(100 * (1 - v["runtime_prompt_tokens"] / v["native_prompt_tokens"]) for v in per_framework.values()) / len(per_framework), 4) if per_framework else None,
        "source_files": {p.name: _sha256(p) for p in root.iterdir() if p.is_file()},
        "archived_price_assumptions": summary.get("price_assumptions_usd_per_1m"),
        "scope": "six-role workload aggregates; not official SWE-bench patch correctness",
    }


def audit_stage6(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    checked = 0
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = (root / name.lstrip("*")).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f"missing or invalid sealed path: {name}")
            continue
        checked += 1
        if _sha256(path) != digest:
            errors.append(f"hash mismatch: {name}")
    if checked == 0:
        errors.append("empty evidence seal")
    archived = json.loads((root / "strict_verification.json").read_text())
    manifest = json.loads((root / "build_manifest.json").read_text())
    return {
        "seal_passed": not errors, "sealed_files_checked": checked, "errors": errors,
        "archived_verifier_passed": archived["passed"],
        "archived_checks": len(archived["checks"]),
        "source_commit": manifest["git"]["commit"],
        "source_kernel": manifest["kernel_release"],
        "scope": "file integrity and archived verifier result; no new kernel execution",
    }


def audit(runs_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "agentos_finals_evidence_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_runs_root": str(runs_root.resolve()),
    }
    for label, relative, function in (("verified", VERIFIED_RUN, audit_verified), ("stage6", STAGE6_RUN, audit_stage6)):
        try:
            result[label] = function(runs_root / relative)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result[label] = {"passed": False, "error_type": type(exc).__name__, "error": str(exc)}
    result["passed"] = result["verified"].get("passed", False) and result["stage6"].get("seal_passed", False) and result["stage6"].get("archived_verifier_passed", False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
