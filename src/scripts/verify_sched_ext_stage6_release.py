#!/usr/bin/env python3
"""Strict offline verifier for the sched_ext Stage 6 release bundle."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.benchmarks.sched_ext_release import (  # noqa: E402
    SCHEMA_VERSION,
    aggregate_stage4,
    benchmark_csv,
    load_official_evidence,
    read_json,
    render_report,
    sha256_file,
)


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _seal(root: Path) -> None:
    paths = [path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in sorted(paths)),
        encoding="utf-8",
    )


def _manifest_checks(bundle: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    defense = bundle / "defense_bundle"
    manifest_path = defense / "manifest.json"
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        _check(checks, "manifest", False, str(exc))
        return {}
    _check(
        checks, "manifest_schema", manifest.get("schema") == SCHEMA_VERSION,
        f"schema={manifest.get('schema')}",
    )
    errors: list[str] = []
    for item in manifest.get("artifacts", []):
        path = (bundle / str(item.get("path", ""))).resolve()
        if not path.is_relative_to(bundle.resolve()):
            errors.append(f"outside:{item.get('path')}")
        elif not path.is_file():
            errors.append(f"missing:{item.get('path')}")
        elif sha256_file(path) != item.get("sha256"):
            errors.append(f"hash:{item.get('path')}")
    _check(checks, "manifest_hashes", not errors, "; ".join(errors[:5]) or "all match")
    return manifest


def _claim_violations(paths: list[Path]) -> list[str]:
    patterns = (
        r"替换(?:了)?\s*Linux\s*全部调度器",
        r"接管(?:了)?\s*(?:RT|实时调度类)",
        r"(?:已经|已|达到)生产级",
    )
    violations: list[str] = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "禁止" in line:
                continue
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
                violations.append(f"{path.name}:{number}")
    return violations


def verify(bundle: Path, *, frozen: bool) -> dict[str, Any]:
    bundle = bundle.resolve()
    checks: list[dict[str, Any]] = []
    manifest = _manifest_checks(bundle, checks)
    git = manifest.get("git", {}) if isinstance(manifest, dict) else {}
    git_ok = bool(git.get("commit")) and (not frozen or git.get("dirty") is False)
    _check(checks, "git_provenance", git_ok, f"commit={git.get('commit')} dirty={git.get('dirty')}")

    evidence = bundle / "defense_bundle/evidence/stage1-5"
    try:
        official = load_official_evidence(evidence)
        _check(checks, "official_evidence", True, "all official summaries present")
    except Exception as exc:
        _check(checks, "official_evidence", False, str(exc))
        official = {}

    if official:
        accepted = all(value.get("accepted") is True for value in official.values())
        _check(checks, "stages_1_to_5_accepted", accepted, "official accepted flags")
        stage2, stage3 = official["stage2"], official["stage3"]
        stage2_ok = (
            stage2.get("partial_mode") is True
            and stage2.get("scheduler_name") == "agentos_scx"
            and stage2.get("workers", {}).get("completed") == 100
            and stage2.get("workers", {}).get("ext_enabled") == 100
            and all(value == 0 for value in stage2.get("controls", {}).values())
        )
        _check(checks, "partial_takeover", stage2_ok, "100 workers and partial mode")
        stage3_ok = (
            stage3.get("workers", {}).get("completed") == 10
            and stage3.get("workers", {}).get("complete_associations") == 10
            and stage3.get("policy_readbacks_verified") == 10
            and stage3.get("policy_map_entries_after_runtime") == 0
        )
        _check(checks, "runtime_policy_chain", stage3_ok, "10 worker PID-policy bindings")
        stage4 = official["stage4_formal"]
        modes = stage4.get("modes", {})
        stage4_ok = (
            stage4.get("accepted") is True
            and len(modes) == 5
            and sum(value.get("workers_completed", 0) for value in modes.values()) == 500
            and stage4.get("checksum_consistent") is True
            and stage4.get("policy_leaks") == 0
            and stage4.get("task_state_leaks") == 0
            and all(value.get("affinity_violations") == 0 for value in modes.values())
        )
        _check(checks, "stage4_formal", stage4_ok, "five modes, 500 workers")
        stage5 = official["stage5_formal"]
        stage5_ok = (
            stage5.get("accepted") is True
            and stage5.get("passed") == stage5.get("trials") == 380
            and stage5.get("soak", {}).get("completed") is True
            and float(stage5.get("soak", {}).get("duration_seconds", 0)) >= 3600
            and stage5.get("soak", {}).get("task_loss") == 0
            and stage5.get("outcomes", {}).get("strict_failed") == 80
            and stage5.get("kernel_log_errors") == []
            and stage5.get("final_sched_ext_state") == "disabled"
            and stage5.get("systemd_failed_units") == 0
            and stage5.get("policy_leaks") == 0
            and stage5.get("task_state_leaks") == 0
            and all(stage5.get(key) == [] for key in (
                "residual_bpf_links", "residual_cgroups", "residual_pinned_maps",
                "residual_processes",
            ))
        )
        _check(checks, "stage5_reliability", stage5_ok, "380/380 plus one-hour soak")

    regression_path = bundle / "defense_bundle/evidence/stage6-regression/summary.json"
    try:
        regression = read_json(regression_path)
        inner = regression.get("summaries", {})
        regression_ok = (
            regression.get("accepted") is True
            and all(inner.get(name, {}).get("accepted") is True for name in (
                "runtime", "policy_effects", "ablation", "reliability"
            ))
            and inner.get("ablation", {}).get("iterations") == 3
            and inner.get("reliability", {}).get("passed") == 19
            and inner.get("reliability", {}).get("trials") == 19
            and regression.get("sysfs_after", {}).get("state") == "disabled"
        )
        observation_keys = {
            "cpu_utilization", "system_context_switches", "cpu_user_us",
            "cpu_system_us", "voluntary_context_switches",
            "involuntary_context_switches",
        }
        observations_ok = all(
            observation_keys <= set(mode.get("observability", {}))
            for mode in inner.get("ablation", {}).get("modes", {}).values()
        ) and len(inner.get("ablation", {}).get("modes", {})) == 5
        regression_ok = regression_ok and observations_ok
        _check(checks, "stage6_regression", regression_ok, "runtime/effects/15 trials/19 cases")
    except Exception as exc:
        _check(checks, "stage6_regression", False, str(exc))

    pytest_results = read_json(bundle / "pytest_results.json")
    pytest_ok = True
    for host in ("debian", "openeuler"):
        record = pytest_results.get(host, {})
        log = bundle / "logs" / str(record.get("log", ""))
        pytest_ok = pytest_ok and (
            record.get("passed") is True
            and record.get("exit_code") == 0
            and log.is_file()
            and sha256_file(log) == record.get("sha256")
        )
    _check(checks, "full_pytest", pytest_ok, "Debian and openEuler exit code 0")

    try:
        expected_summary, _raw = aggregate_stage4(evidence)
        actual_summary = read_json(bundle / "benchmark_summary.json")
        summary_ok = actual_summary == expected_summary
        csv_ok = (bundle / "benchmark_summary.csv").read_text(encoding="utf-8") == benchmark_csv(expected_summary)
        report_ok = (bundle / "report.md").read_text(encoding="utf-8") == render_report(official, expected_summary)
        _check(checks, "benchmark_traceability", summary_ok and csv_ok and report_ok, "JSON/CSV/report regenerated exactly")
    except Exception as exc:
        _check(checks, "benchmark_traceability", False, str(exc))

    events = [
        json.loads(line)
        for line in (bundle / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_counts = {
        name: sum(event.get("event") == name for event in events)
        for name in (
            "kernel.sched_ext_degraded", "kernel.sched_ext_fallback",
            "kernel.sched_ext_recovered",
        )
    }
    events_ok = (
        event_counts["kernel.sched_ext_fallback"] == 240
        and event_counts["kernel.sched_ext_recovered"] == 160
        and all(event.get("source_sha256") for event in events)
    )
    _check(checks, "lifecycle_events", events_ok, json.dumps(event_counts, sort_keys=True))
    docs = [bundle / "report.md", bundle / "defense_bundle/README.md"]
    violations = _claim_violations(docs)
    _check(checks, "claim_boundaries", not violations, ", ".join(violations) or "bounded claims")
    passed = all(item["passed"] for item in checks)
    return {"schema": SCHEMA_VERSION, "passed": passed, "checks": checks}


def _archive(bundle: Path) -> tuple[Path, Path]:
    archive = bundle.parent / f"member-a-sched-ext-stage6-release-{bundle.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name)
    sha = archive.with_suffix(archive.suffix + ".sha256")
    sha.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    return archive, sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify without rewriting the bundle or regenerating its archive",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.bundle, frozen=args.frozen)
    if args.check_only:
        result["check_only"] = True
    else:
        result_path = args.bundle / "strict_verification.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _seal(args.bundle)
        if result["passed"]:
            archive, sha = _archive(args.bundle.resolve())
            result["archive"] = str(archive)
            result["archive_sha256"] = str(sha)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for check in result["checks"]:
            print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['detail']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
