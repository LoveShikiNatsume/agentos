"""Verify Member A kernel adapter evidence in summary.json.

Checks that the workload matrix summary contains valid kernel adapter
evidence with expected fields and reasonable values.

Usage:
    .venv/bin/python scripts/verify_member_a_kernel_adapter.py \
      --summary runs/member_a_kernel_adapter/summary.json

    .venv/bin/python scripts/verify_member_a_kernel_adapter.py \
      --summary runs/member_a_kernel_adapter/summary.json --strict-cgroup

    .venv/bin/python scripts/verify_member_a_kernel_adapter.py \
      --summary runs/member_a_kernel_adapter/summary.json --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_VALID_PRESSURE_LEVELS = {"normal", "warning", "overloaded", "critical"}

# Composite adapter modes: <cgroup_part>+<bpf_part>
_VALID_ADAPTER_MODES = {
    "user_space+mock_bpf",
    "cgroup_v2+mock_bpf",
    "user_space+pinned_probe",
    "cgroup_v2+pinned_probe",
    "user_space+unavailable",
    "cgroup_v2+unavailable",
}

# Misleading claims that must not appear anywhere in the summary text
_FORBIDDEN_STRINGS = [
    "kernel takeover",
    "native scheduler enabled",
    "ebpf scheduling active",
]

# Required fields in probe dict (all must be non-empty strings)
_REQUIRED_PROBE_FIELDS = ["sched_ext", "bpf_map", "fallback_mode", "reason"]

# Standard event_counts field names (all must be non-negative ints, must be present)
_STANDARD_EVENT_COUNT_FIELDS = [
    "cgroup_quota_applied",
    "cgroup_unavailable",
    "cgroup_cleanup_failed",
    "pressure_sampled",
    "pressure_system",
    "pressure_agent_cgroup",
    "bpf_map_updated",
    "kernel_probe",
    "scheduler_decision",
]

# Top-level alias → event_counts field (must be equal)
_TOP_LEVEL_ALIASES = [
    ("cgroup_quota_applied_count", "cgroup_quota_applied"),
    ("pressure_sample_count", "pressure_sampled"),
    ("bpf_map_update_count", "bpf_map_updated"),
]


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_valid_nonneg_int(value: object) -> bool:
    """True if value is a non-negative integer; bool is NOT accepted."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def verify(summary_path: str, *, strict_cgroup: bool = False) -> list[dict]:
    """Run verification checks and return results.

    Each result is ``{"check": str, "status": "PASS"|"WARN"|"FAIL", "detail": str}``.
    """
    results: list[dict] = []
    path = Path(summary_path)

    # Check 1: file exists
    if not path.exists():
        results.append({
            "check": "summary_file_exists",
            "status": "FAIL",
            "detail": f"File not found: {path}",
        })
        return results
    results.append({"check": "summary_file_exists", "status": "PASS", "detail": str(path)})

    # Check 2: readable
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        results.append({
            "check": "summary_file_readable",
            "status": "FAIL",
            "detail": f"Cannot read file: {exc}",
        })
        return results
    except UnicodeError as exc:
        results.append({
            "check": "summary_file_readable",
            "status": "FAIL",
            "detail": f"Unicode decode error: {exc}",
        })
        return results

    # Check 3: valid JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        results.append({
            "check": "summary_valid_json",
            "status": "FAIL",
            "detail": f"Invalid JSON: {exc}",
        })
        return results
    results.append({"check": "summary_valid_json", "status": "PASS", "detail": "parsed successfully"})

    # Check 4: JSON root must be dict
    if not isinstance(data, dict):
        results.append({
            "check": "summary_root_is_dict",
            "status": "FAIL",
            "detail": f"JSON root is {type(data).__name__}, expected dict",
        })
        return results

    # Check 5: misleading strings anywhere in raw JSON
    raw_lower = raw.lower()
    for forbidden in _FORBIDDEN_STRINGS:
        if forbidden.lower() in raw_lower:
            results.append({
                "check": "summary_no_misleading_claims",
                "status": "FAIL",
                "detail": f"Summary contains forbidden claim: '{forbidden}'",
            })
            return results
    results.append({
        "check": "summary_no_misleading_claims",
        "status": "PASS",
        "detail": "no misleading kernel claims found",
    })

    # Check 6: kernel_adapter_evidence exists and is dict
    kae = data.get("kernel_adapter_evidence")
    if kae is None:
        results.append({
            "check": "kernel_adapter_evidence_exists",
            "status": "FAIL",
            "detail": "kernel_adapter_evidence not found in summary",
        })
        return results
    if not isinstance(kae, dict):
        results.append({
            "check": "kernel_adapter_evidence_exists",
            "status": "FAIL",
            "detail": f"kernel_adapter_evidence is {type(kae).__name__}, expected dict",
        })
        return results
    results.append({"check": "kernel_adapter_evidence_exists", "status": "PASS", "detail": "found"})

    # Check 7: event_counts must be dict
    event_counts = kae.get("event_counts")
    if not isinstance(event_counts, dict):
        results.append({
            "check": "event_counts_is_dict",
            "status": "FAIL",
            "detail": f"event_counts is {type(event_counts).__name__}, expected dict",
        })
        return results

    # Check 8: all standard event_counts fields must exist and be non-negative ints
    ec_errors: list[str] = []
    for field in _STANDARD_EVENT_COUNT_FIELDS:
        if field not in event_counts:
            ec_errors.append(f"{field}: missing")
        else:
            v = event_counts[field]
            if not _is_valid_nonneg_int(v):
                ec_errors.append(f"{field}={v!r} (must be non-negative int, not {type(v).__name__})")
    if ec_errors:
        for err in ec_errors:
            results.append({
                "check": "event_counts_valid",
                "status": "FAIL",
                "detail": err,
            })
    else:
        results.append({
            "check": "event_counts_valid",
            "status": "PASS",
            "detail": f"{len(_STANDARD_EVENT_COUNT_FIELDS)} fields valid",
        })

    # Check 9: top-level alias validation and consistency with event_counts
    for top_field, ec_field in _TOP_LEVEL_ALIASES:
        top_val = kae.get(top_field)
        ec_val = event_counts.get(ec_field) if isinstance(event_counts, dict) else None
        if not _is_valid_nonneg_int(top_val):
            results.append({
                "check": f"{top_field}_valid",
                "status": "FAIL",
                "detail": f"{top_field}={top_val!r} must be a non-negative integer",
            })
        elif isinstance(ec_val, int) and not isinstance(ec_val, bool) and top_val != ec_val:
            results.append({
                "check": f"{top_field}_consistent",
                "status": "FAIL",
                "detail": (
                    f"{top_field}={top_val} does not match "
                    f"event_counts.{ec_field}={ec_val}"
                ),
            })
        else:
            results.append({
                "check": f"{top_field}_valid",
                "status": "PASS",
                "detail": str(top_val),
            })

    # Check 10: schema version
    schema = kae.get("schema_version", "")
    if schema == "kernel_adapter_evidence.v1":
        results.append({"check": "evidence_schema_version", "status": "PASS", "detail": schema})
    else:
        results.append({
            "check": "evidence_schema_version",
            "status": "FAIL",
            "detail": f"unexpected schema: {schema}",
        })

    # Check 11: pressure_sample_count must be positive (already validated non-neg above)
    psc = kae.get("pressure_sample_count")
    if _is_valid_nonneg_int(psc) and psc > 0:
        results.append({"check": "pressure_sample_count_positive", "status": "PASS", "detail": str(psc)})
    elif _is_valid_nonneg_int(psc):
        results.append({
            "check": "pressure_sample_count_positive",
            "status": "FAIL",
            "detail": f"pressure_sample_count={psc}, expected > 0",
        })

    # Check 12: bpf_map_update_count must be positive
    bpf_count = kae.get("bpf_map_update_count")
    if _is_valid_nonneg_int(bpf_count) and bpf_count > 0:
        results.append({"check": "bpf_map_update_count_positive", "status": "PASS", "detail": str(bpf_count)})
    elif _is_valid_nonneg_int(bpf_count):
        results.append({
            "check": "bpf_map_update_count_positive",
            "status": "FAIL",
            "detail": f"bpf_map_update_count={bpf_count}, expected > 0",
        })

    # Check 13: probe evidence — must exist, be non-empty dict, all fields non-empty strings
    probe = kae.get("probe")
    if not probe or not isinstance(probe, dict):
        results.append({
            "check": "kernel_probe_evidence",
            "status": "FAIL",
            "detail": "probe evidence is missing or empty",
        })
    else:
        probe_errors: list[str] = []
        for field in _REQUIRED_PROBE_FIELDS:
            if field not in probe:
                probe_errors.append(f"{field}: missing")
            elif not _is_non_empty_string(probe[field]):
                v = probe[field]
                probe_errors.append(
                    f"{field}={v!r} must be a non-empty string (got {type(v).__name__})"
                )
        if probe_errors:
            for err in probe_errors:
                results.append({
                    "check": "kernel_probe_evidence",
                    "status": "FAIL",
                    "detail": err,
                })
        else:
            results.append({
                "check": "kernel_probe_evidence",
                "status": "PASS",
                "detail": f"sched_ext={probe.get('sched_ext')}, bpf_map={probe.get('bpf_map')}",
            })

    # Check 14: adapter_mode must be a valid composite mode
    adapter_mode = kae.get("adapter_mode", "")
    if adapter_mode in _VALID_ADAPTER_MODES:
        results.append({"check": "adapter_mode_valid", "status": "PASS", "detail": adapter_mode})
    else:
        results.append({
            "check": "adapter_mode_valid",
            "status": "FAIL",
            "detail": (
                f"invalid adapter_mode: '{adapter_mode}'. "
                f"Valid: {sorted(_VALID_ADAPTER_MODES)}"
            ),
        })

    # Check 15: worst_pressure_level valid
    wpl = kae.get("worst_pressure_level", "")
    if wpl in _VALID_PRESSURE_LEVELS:
        results.append({"check": "worst_pressure_level_valid", "status": "PASS", "detail": wpl})
    else:
        results.append({
            "check": "worst_pressure_level_valid",
            "status": "FAIL",
            "detail": f"invalid worst_pressure_level: {wpl}",
        })

    # Check 16: cgroup_cleanup_failed must be 0 in all modes
    cgroup_cleanup = event_counts.get("cgroup_cleanup_failed", 0) if isinstance(event_counts, dict) else 0
    if _is_valid_nonneg_int(cgroup_cleanup) and cgroup_cleanup > 0:
        results.append({
            "check": "cgroup_cleanup_failed",
            "status": "FAIL",
            "detail": f"cgroup_cleanup_failed={cgroup_cleanup}: cgroup cleanup errors indicate real failures",
        })

    # Check 17: cgroup quota
    cgroup_count = kae.get("cgroup_quota_applied_count", 0)
    if strict_cgroup:
        if _is_valid_nonneg_int(cgroup_count) and cgroup_count > 0 and (
            not _is_valid_nonneg_int(cgroup_cleanup) or cgroup_cleanup == 0
        ):
            results.append({
                "check": "strict_cgroup_quota",
                "status": "PASS",
                "detail": f"quota_applied={cgroup_count}, cleanup_failed={cgroup_cleanup}",
            })
        else:
            results.append({
                "check": "strict_cgroup_quota",
                "status": "FAIL",
                "detail": f"strict mode: quota_applied={cgroup_count}, cleanup_failed={cgroup_cleanup}",
            })
    else:
        # default mode: negative / wrong type → FAIL; 0 → WARN+PASS; >0 → PASS
        if not _is_valid_nonneg_int(cgroup_count):
            results.append({
                "check": "cgroup_quota",
                "status": "FAIL",
                "detail": f"cgroup_quota_applied_count={cgroup_count!r} must be a non-negative integer",
            })
        elif cgroup_count > 0:
            results.append({
                "check": "cgroup_quota",
                "status": "PASS",
                "detail": f"quota_applied={cgroup_count}",
            })
        else:
            results.append({
                "check": "cgroup_quota",
                "status": "WARN",
                "detail": "cgroup_quota_applied_count=0 (coroutine workload may not produce cgroup events)",
            })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify kernel adapter evidence")
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to summary.json",
    )
    parser.add_argument(
        "--strict-cgroup",
        action="store_true",
        help="Require cgroup_quota_applied_count > 0 and no cleanup failures",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    try:
        results = verify(args.summary, strict_cgroup=args.strict_cgroup)
    except Exception as exc:
        results = [{
            "check": "verifier_internal_error",
            "status": "FAIL",
            "detail": f"Unexpected error: {type(exc).__name__}: {exc}",
        }]

    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    pass_count = sum(1 for r in results if r["status"] == "PASS")

    if args.json:
        output = {
            "summary_path": args.summary,
            "strict_cgroup": args.strict_cgroup,
            "results": results,
            "passed": fail_count == 0,
            "fail_count": fail_count,
            "warn_count": warn_count,
            "pass_count": pass_count,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        for r in results:
            tag = r["status"]
            print(f"[{tag}] {r['check']}: {r['detail']}")

        print()
        if fail_count > 0:
            print(f"RESULT: FAIL ({fail_count} failed, {warn_count} warnings, {pass_count} passed)")
        else:
            print(f"RESULT: PASS ({pass_count} passed, {warn_count} warnings)")

    sys.exit(1 if fail_count > 0 else 0)


if __name__ == "__main__":
    main()
