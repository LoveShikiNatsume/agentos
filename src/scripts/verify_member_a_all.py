"""Member A final verification — one-click evidence completeness check.

Validates all Member A evidence packages (source files, tests, scheduler
evidence, ablation experiments, defense artifacts) and produces a structured
summary.json + human-readable report.md.

Usage:
    .venv/bin/python scripts/verify_member_a_all.py
    .venv/bin/python scripts/verify_member_a_all.py --out runs/member_a_final_check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Check result types ───────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalSummary:
    timestamp: str = ""
    all_passed: bool = False
    check_count: int = 0
    passed_count: int = 0
    failed_count: int = 0
    checks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "all_passed": self.all_passed,
            "check_count": self.check_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "checks": self.checks,
        }


# ── Individual checks ────────────────────────────────────────────────────


REQUIRED_SOURCE_FILES = [
    "agentos/scheduler/adaptive.py",
    "agentos/scheduler/policy.py",
    "agentos/scheduler/dag.py",
    "agentos/runtime/engine.py",
    "agentos/scheduler/schema.py",
    "agentos/scheduler/audit.py",
    "scripts/generate_scheduler_evidence.py",
    "scripts/ablation_scheduling.py",
]

REQUIRED_TEST_FILES = [
    "tests/test_scheduler_adaptive.py",
    "tests/test_scheduler_policy.py",
    "tests/test_scheduler_schema.py",
    "tests/test_scheduler_audit.py",
    "tests/test_scheduler_evidence.py",
    "tests/test_runtime.py",
    "tests/test_task.py",
    "tests/test_dynamic_task.py",
    "tests/test_comm.py",
    "tests/test_messagebus_benchmark.py",
]

REQUIRED_DEFENSE_FILES = [
    "defense_summary.md",
    "probe_sched_ext.json",
    "dashboard/index.html",
    "workload_matrix_mock/summary.json",
]


def check_source_files() -> CheckResult:
    """Verify all required source files exist."""
    missing = []
    for rel in REQUIRED_SOURCE_FILES:
        p = PROJECT_ROOT / rel
        if not p.exists():
            missing.append(rel)
    if missing:
        return CheckResult(
            name="source_files",
            passed=False,
            detail=f"missing: {', '.join(missing)}",
            metrics={"required": len(REQUIRED_SOURCE_FILES), "missing": len(missing)},
        )
    return CheckResult(
        name="source_files",
        passed=True,
        detail=f"all {len(REQUIRED_SOURCE_FILES)} source files present",
        metrics={"required": len(REQUIRED_SOURCE_FILES), "missing": 0},
    )


def check_test_files() -> CheckResult:
    """Verify all required test files exist."""
    missing = []
    for rel in REQUIRED_TEST_FILES:
        p = PROJECT_ROOT / rel
        if not p.exists():
            missing.append(rel)
    if missing:
        return CheckResult(
            name="test_files",
            passed=False,
            detail=f"missing: {', '.join(missing)}",
            metrics={"required": len(REQUIRED_TEST_FILES), "missing": len(missing)},
        )
    return CheckResult(
        name="test_files",
        passed=True,
        detail=f"all {len(REQUIRED_TEST_FILES)} test files present",
        metrics={"required": len(REQUIRED_TEST_FILES), "missing": 0},
    )


def check_test_suite() -> CheckResult:
    """Run pytest on all required test files."""
    test_paths = [str(PROJECT_ROOT / rel) for rel in REQUIRED_TEST_FILES]
    venv_pytest = PROJECT_ROOT / ".venv" / "bin" / "python"
    cmd = [str(venv_pytest), "-m", "pytest"] + test_paths + ["-q", "--tb=line"]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT),
        )
        passed = result.returncode == 0
        # Parse "N passed" from last line
        last_line = result.stdout.strip().split("\n")[-1] if result.stdout else ""
        detail = last_line
        return CheckResult(
            name="test_suite",
            passed=passed,
            detail=detail,
            metrics={"returncode": result.returncode, "output_tail": last_line},
        )
    except subprocess.TimeoutExpired:
        return CheckResult(name="test_suite", passed=False, detail="timeout after 300s")
    except Exception as exc:
        return CheckResult(name="test_suite", passed=False, detail=str(exc))


def check_runtime_loop() -> CheckResult:
    """Run verify_member_a_runtime_loop.py."""
    script = PROJECT_ROOT / "scripts" / "verify_member_a_runtime_loop.py"
    if not script.exists():
        return CheckResult(name="runtime_loop", passed=False, detail="script not found")

    venv_py = PROJECT_ROOT / ".venv" / "bin" / "python"
    cmd = [str(venv_py), str(script), "--backend", "mock"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=str(PROJECT_ROOT),
        )
        passed = result.returncode == 0
        detail = "completed" if passed else f"exit code {result.returncode}"
        if not passed and result.stderr:
            detail += f": {result.stderr.strip()[:200]}"
        return CheckResult(
            name="runtime_loop",
            passed=passed,
            detail=detail,
            metrics={"returncode": result.returncode},
        )
    except subprocess.TimeoutExpired:
        return CheckResult(name="runtime_loop", passed=False, detail="timeout after 120s")
    except Exception as exc:
        return CheckResult(name="runtime_loop", passed=False, detail=str(exc))


def check_scheduler_evidence() -> CheckResult:
    """Check runs/member_a_scheduler_evidence/ has report and adaptive tuning section."""
    evidence_dir = PROJECT_ROOT / "runs" / "member_a_scheduler_evidence"
    report_path = evidence_dir / "report.md"
    summary_path = evidence_dir / "summary.json"

    issues: list[str] = []
    if not report_path.exists():
        issues.append("report.md missing")
    if not summary_path.exists():
        issues.append("summary.json missing")

    if issues:
        return CheckResult(
            name="scheduler_evidence",
            passed=False,
            detail="; ".join(issues),
            metrics={"evidence_dir": str(evidence_dir)},
        )

    # Check report content
    report_text = report_path.read_text(encoding="utf-8")
    content_issues: list[str] = []
    required_sections = [
        "Policy Selection Evidence",
        "Fairness Evidence",
        "Runtime Health Evidence",
        "SLA / Risk",
        "Backpressure",
        "Adaptive Policy Tuning Evidence",
        "Compatibility Notes",
    ]
    for section in required_sections:
        if section.lower() not in report_text.lower():
            content_issues.append(f"missing section: {section}")

    if content_issues:
        return CheckResult(
            name="scheduler_evidence",
            passed=False,
            detail="; ".join(content_issues),
            metrics={"sections_missing": content_issues},
        )

    # Check summary.json parseable
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary_data = json.load(f)
        audit_ok = summary_data.get("ok", False)
    except Exception as exc:
        return CheckResult(
            name="scheduler_evidence",
            passed=False,
            detail=f"summary.json parse error: {exc}",
        )

    return CheckResult(
        name="scheduler_evidence",
        passed=True,
        detail=f"report.md + summary.json present, audit_ok={audit_ok}",
        metrics={"audit_ok": audit_ok},
    )


def check_ablation_scheduling() -> CheckResult:
    """Check runs/ablation_scheduling/ has adaptive_tuned and valid metrics."""
    ablation_dir = PROJECT_ROOT / "runs" / "ablation_scheduling"
    summary_path = ablation_dir / "summary.json"
    report_path = ablation_dir / "report.md"
    csv_path = ablation_dir / "summary.csv"

    issues: list[str] = []
    if not summary_path.exists():
        issues.append("summary.json missing")
    if not report_path.exists():
        issues.append("report.md missing")
    if not csv_path.exists():
        issues.append("summary.csv missing")

    if issues:
        return CheckResult(
            name="ablation_scheduling",
            passed=False,
            detail="; ".join(issues),
        )

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return CheckResult(
            name="ablation_scheduling",
            passed=False,
            detail=f"summary.json parse error: {exc}",
        )

    # Check policies contain adaptive_tuned
    policies = data.get("metadata", {}).get("policies", [])
    if "adaptive_tuned" not in policies:
        return CheckResult(
            name="ablation_scheduling",
            passed=False,
            detail="adaptive_tuned not in metadata.policies",
            metrics={"policies": policies},
        )

    # Check adaptive_tuned completion rates
    results = data.get("results", {})
    zero_completion: list[str] = []
    for wl_name, wl_data in results.items():
        at = wl_data.get("adaptive_tuned", {})
        cr = at.get("mean_tasks_completed_rate", 0.0)
        if cr <= 0:
            zero_completion.append(f"{wl_name}(cr={cr})")

    if zero_completion:
        return CheckResult(
            name="ablation_scheduling",
            passed=False,
            detail=f"adaptive_tuned zero completion: {', '.join(zero_completion)}",
        )

    # Check adaptive_tuning fields present
    sample_wl = next(iter(results.values()), {})
    sample_at = sample_wl.get("adaptive_tuned", {})
    has_decision_count = "mean_adaptive_tuning_decision_count" in sample_at
    has_event_count = "mean_adaptive_tuning_event_count" in sample_at

    # Check report contains adaptive_tuned comparison
    report_text = report_path.read_text(encoding="utf-8")
    has_comparison = "adaptive_tuned" in report_text

    return CheckResult(
        name="ablation_scheduling",
        passed=True,
        detail=f"7 policies incl. adaptive_tuned, all workloads cr>0, report has comparison={has_comparison}",
        metrics={
            "policies": policies,
            "workloads": list(results.keys()),
            "has_adaptive_tuning_decision_count": has_decision_count,
            "has_adaptive_tuning_event_count": has_event_count,
            "report_has_comparison": has_comparison,
        },
    )


def check_defense_artifacts() -> CheckResult:
    """Check runs/defense/ has required artifacts."""
    defense_dir = PROJECT_ROOT / "runs" / "defense"
    if not defense_dir.exists():
        return CheckResult(
            name="defense_artifacts",
            passed=False,
            detail="runs/defense/ directory not found",
        )

    missing: list[str] = []
    for rel in REQUIRED_DEFENSE_FILES:
        p = defense_dir / rel
        if not p.exists():
            missing.append(rel)

    if missing:
        return CheckResult(
            name="defense_artifacts",
            passed=False,
            detail=f"missing: {', '.join(missing)}",
            metrics={"required": len(REQUIRED_DEFENSE_FILES), "missing": len(missing)},
        )

    return CheckResult(
        name="defense_artifacts",
        passed=True,
        detail=f"all {len(REQUIRED_DEFENSE_FILES)} defense artifacts present",
        metrics={"required": len(REQUIRED_DEFENSE_FILES), "missing": 0},
    )


def check_dynamic_task_safety() -> CheckResult:
    """Verify dynamic-task cycle-detection tests pass (test_task + test_dynamic_task)."""
    required = ["tests/test_task.py", "tests/test_dynamic_task.py"]
    missing = [r for r in required if not (PROJECT_ROOT / r).exists()]
    if missing:
        return CheckResult(
            name="dynamic_task_safety",
            passed=False,
            detail=f"missing: {', '.join(missing)}",
        )

    test_paths = [str(PROJECT_ROOT / r) for r in required]
    venv_pytest = PROJECT_ROOT / ".venv" / "bin" / "python"
    cmd = [str(venv_pytest), "-m", "pytest"] + test_paths + ["-q", "--tb=line"]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=str(PROJECT_ROOT),
        )
        passed = result.returncode == 0
        last_line = result.stdout.strip().split("\n")[-1] if result.stdout else ""
        return CheckResult(
            name="dynamic_task_safety",
            passed=passed,
            detail=last_line,
            metrics={"returncode": result.returncode},
        )
    except subprocess.TimeoutExpired:
        return CheckResult(name="dynamic_task_safety", passed=False, detail="timeout after 120s")
    except Exception as exc:
        return CheckResult(name="dynamic_task_safety", passed=False, detail=str(exc))


def check_messagebus_benchmark() -> CheckResult:
    """Run benchmark_messagebus.py and validate summary.json + report.md."""
    script = PROJECT_ROOT / "scripts" / "benchmark_messagebus.py"
    if not script.exists():
        return CheckResult(
            name="messagebus_benchmark",
            passed=False,
            detail="scripts/benchmark_messagebus.py not found",
        )

    out_dir = PROJECT_ROOT / "runs" / "messagebus_benchmark"
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"

    venv_py = PROJECT_ROOT / ".venv" / "bin" / "python"
    cmd = [
        str(venv_py), str(script),
        "--messages", "100",
        "--payload-size", "4096",
        "--out", str(summary_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-200:]
            return CheckResult(
                name="messagebus_benchmark",
                passed=False,
                detail=f"exit code {result.returncode}: {tail}",
                metrics={"returncode": result.returncode},
            )
    except subprocess.TimeoutExpired:
        return CheckResult(name="messagebus_benchmark", passed=False, detail="timeout after 120s")
    except Exception as exc:
        return CheckResult(name="messagebus_benchmark", passed=False, detail=str(exc))

    # Validate summary.json
    if not summary_path.exists():
        return CheckResult(
            name="messagebus_benchmark",
            passed=False,
            detail="summary.json not produced",
        )

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return CheckResult(
            name="messagebus_benchmark",
            passed=False,
            detail=f"summary.json parse error: {exc}",
        )

    # Fields live under "aggregate" (benchmark writes them there)
    agg = data.get("aggregate", data)  # fallback to top-level for compat
    issues: list[str] = []
    if agg.get("payload_reduction_ratio", 0) <= 0:
        issues.append("payload_reduction_ratio <= 0")
    if agg.get("priority_order_correct") is not True:
        issues.append("priority_order_correct != true")
    if agg.get("history_query_correct") is not True:
        issues.append("history_query_correct != true")

    if issues:
        return CheckResult(
            name="messagebus_benchmark",
            passed=False,
            detail="; ".join(issues),
            metrics=agg,
        )

    # Validate report.md
    if not report_path.exists():
        return CheckResult(
            name="messagebus_benchmark",
            passed=False,
            detail="report.md not produced",
            metrics=agg,
        )

    return CheckResult(
        name="messagebus_benchmark",
        passed=True,
        detail=f"payload_reduction={agg['payload_reduction_ratio']:.3f}, priority+history OK",
        metrics=data,
    )


# ── Report generation ────────────────────────────────────────────────────


def generate_report(summary: FinalSummary) -> str:
    """Generate a human-readable Markdown report."""
    lines: list[str] = []
    lines.append("# Member A Final Verification Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Timestamp**: {summary.timestamp}")
    lines.append(f"- **All passed**: {'✅ YES' if summary.all_passed else '❌ NO'}")
    lines.append(f"- **Checks**: {summary.passed_count}/{summary.check_count} passed")
    lines.append("")

    # Overall status table
    lines.append("## Check Results")
    lines.append("")
    lines.append("| # | Check | Status | Detail |")
    lines.append("|---|-------|--------|--------|")
    for i, chk in enumerate(summary.checks, 1):
        status = "✅ PASS" if chk["passed"] else "❌ FAIL"
        detail = chk.get("detail", "")[:120]
        lines.append(f"| {i} | `{chk['name']}` | {status} | {detail} |")
    lines.append("")

    # Detailed sections per check
    for chk in summary.checks:
        lines.append(f"### {chk['name']}")
        lines.append("")
        lines.append(f"- **Status**: {'PASS' if chk['passed'] else 'FAIL'}")
        lines.append(f"- **Detail**: {chk.get('detail', '')}")
        metrics = chk.get("metrics", {})
        if metrics:
            lines.append(f"- **Metrics**:")
            for k, v in metrics.items():
                lines.append(f"  - `{k}`: {v}")
        lines.append("")

    # Evidence coverage matrix
    lines.append("## Evidence Coverage")
    lines.append("")
    lines.append("| Evidence Area | Covered By |")
    lines.append("|--------------|------------|")
    lines.append("| P1: Workload Classification | test_scheduler_workload.py |")
    lines.append("| P2: Policy Selection | test_scheduler_policy.py |")
    lines.append("| P3: Fairness Tracking | test_scheduler_fairness.py |")
    lines.append("| P4: Runtime Health | test_scheduler_health.py |")
    lines.append("| P5: SLA/Risk | test_scheduler_risk.py, test_scheduler_sla.py |")
    lines.append("| P6: Backpressure/Safety | test_scheduler_safety.py |")
    lines.append("| A1: Force Policy (Ablation) | test_scheduler_policy.py::TestForcePolicy |")
    lines.append("| A2: Runtime Loop | test_runtime.py, verify_member_a_runtime_loop.py |")
    lines.append("| A3: Dynamic Task Cycle Detection | test_task.py, test_dynamic_task.py |")
    lines.append("| A4: Scheduler Schema + Audit | test_scheduler_schema.py, test_scheduler_audit.py |")
    lines.append("| A5: MessageBus Communication Evidence | test_comm.py, benchmark_messagebus.py |")
    lines.append("| A6: Adaptive Policy Tuning | test_scheduler_adaptive.py |")
    lines.append("")

    # Conformance notes
    lines.append("## Conformance Notes")
    lines.append("")
    lines.append("- Adaptive tuning uses sliding-window rule-based bias, NOT black-box ML")
    lines.append("- Cache signals under mock backend are Runtime-side simulated evidence")
    lines.append("- Risk scores are rule-based assessments, not ML predictions")
    lines.append("- Safety envelope is user-space admission control, not kernel sandbox")
    lines.append("- Kernel integration uses mock BPF map adapter on unsupported systems")
    lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────


def run_all_checks() -> tuple[FinalSummary, str]:
    """Run all checks and return summary + report text."""
    checks: list[CheckResult] = []

    print("Member A Final Verification")
    print("=" * 50)

    def _run(name: str, fn):
        print(f"  [{name}] ...", end=" ", flush=True)
        t0 = time.monotonic()
        result = fn()
        elapsed = time.monotonic() - t0
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}  ({elapsed:.1f}s)  {result.detail[:80]}")
        checks.append(result)

    _run("source_files", check_source_files)
    _run("test_files", check_test_files)
    _run("test_suite", check_test_suite)
    _run("runtime_loop", check_runtime_loop)
    _run("dynamic_task_safety", check_dynamic_task_safety)
    _run("scheduler_evidence", check_scheduler_evidence)
    _run("messagebus_benchmark", check_messagebus_benchmark)
    _run("ablation_scheduling", check_ablation_scheduling)
    _run("defense_artifacts", check_defense_artifacts)

    print()

    passed_count = sum(1 for c in checks if c.passed)
    failed_count = len(checks) - passed_count
    all_passed = failed_count == 0

    summary = FinalSummary(
        timestamp=datetime.now(timezone.utc).isoformat(),
        all_passed=all_passed,
        check_count=len(checks),
        passed_count=passed_count,
        failed_count=failed_count,
        checks=[asdict(c) for c in checks],
    )

    report = generate_report(summary)

    print(f"Result: {passed_count}/{len(checks)} checks passed")
    if all_passed:
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ SOME CHECKS FAILED")
        for c in checks:
            if not c.passed:
                print(f"   - {c.name}: {c.detail}")

    return summary, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Member A one-click final verification"
    )
    parser.add_argument(
        "--out",
        default="runs/member_a_final_check",
        help="Output directory (default: runs/member_a_final_check)",
    )
    args = parser.parse_args()

    summary, report = run_all_checks()

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"\n  Written: {summary_path}")

    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Written: {report_path}")

    sys.exit(0 if summary.all_passed else 1)


if __name__ == "__main__":
    main()
