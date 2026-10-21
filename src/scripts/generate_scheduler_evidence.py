"""Generate Member A scheduler evidence report.

Usage:
    .venv/bin/python scripts/generate_scheduler_evidence.py --events runs/demo/events.jsonl --out runs/defense/member_a_scheduler_evidence.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentos.scheduler.audit import audit_scheduler_events


def _worst_pressure_level(levels: list[str]) -> str:
    """Find the worst pressure level from a list."""
    if not levels:
        return "normal"
    priority = {"normal": 0, "warning": 1, "critical": 2, "overloaded": 3}
    worst = "normal"
    for level in levels:
        if priority.get(level, 0) > priority.get(worst, 0):
            worst = level
    return worst


def extract_evidence_data(events: list[dict]) -> dict:
    """Extract evidence data from events for the report."""
    policies = []
    workload_classes = []
    fairness_scores = []
    starvation_risks = []
    kernel_probe = None
    bpf_map_task_count = 0

    # Runtime health data
    health_scores = []
    pressure_levels = []
    max_queue_pressure = 0.0
    max_resource_pressure = 0.0
    max_fairness_pressure = 0.0
    max_dependency_pressure = 0.0
    all_health_warnings = set()

    # SLA/Risk data
    sla_counts = {"critical": 0, "interactive": 0, "batch": 0, "best_effort": 0}
    deadline_task_count = 0
    max_urgency_score = 0.0
    max_task_risk_score = 0.0
    high_risk_task_ids = set()
    all_risk_warnings = set()
    risk_aware_count = 0

    # Backpressure/Safety data
    max_backpressure_signal = "none"
    admission_counts = {"allow": 0, "defer": 0, "degrade": 0, "reject": 0}
    rejected_task_ids = set()
    degraded_task_ids = set()
    deferred_task_ids = set()
    all_safety_violations = set()

    # Adaptive tuning data
    adaptive_tuning_enabled_count = 0
    adaptive_adjusted_policies: list[str] = []
    all_adaptive_reasons: list[str] = []
    adaptive_policy_changed_count = 0

    for event in events:
        event_type = event.get("event_type", "")
        data = event.get("data", {})

        if event_type == "scheduler.decision":
            policy = data.get("policy")
            if policy and policy not in policies:
                policies.append(policy)
            if policy == "risk_aware":
                risk_aware_count += 1
            workload_class = data.get("workload_class")
            if workload_class and workload_class not in workload_classes:
                workload_classes.append(workload_class)
            fairness = data.get("fairness", {})
            if fairness:
                score = fairness.get("fairness_score")
                if score is not None:
                    fairness_scores.append(score)
                risk = fairness.get("starvation_risk")
                if risk and risk not in starvation_risks:
                    starvation_risks.append(risk)
            if "kernel_probe" in data and kernel_probe is None:
                kernel_probe = data["kernel_probe"]
            if "bpf_map" in data:
                bpf_map_task_count = max(bpf_map_task_count, data["bpf_map"].get("task_count", 0))

            # Extract SLA/Risk from decision
            sla = data.get("sla", {})
            if sla:
                for key in ["critical_count", "interactive_count", "batch_count", "best_effort_count"]:
                    sla_counts[key.replace("_count", "")] += sla.get(key, 0)
                deadline_task_count += sla.get("deadline_task_count", 0)
                max_urgency_score = max(max_urgency_score, sla.get("max_urgency_score", 0.0))

            risk = data.get("risk", {})
            if risk:
                max_task_risk_score = max(max_task_risk_score, risk.get("max_task_risk_score", 0.0))
                high_risk_task_ids.update(risk.get("high_risk_task_ids", []))
                all_risk_warnings.update(risk.get("risk_warnings", []))

            # Extract Backpressure/Safety from decision
            backpressure = data.get("backpressure", {})
            if backpressure:
                signal = backpressure.get("signal", "none")
                signal_priority = {"none": 0, "soft": 1, "hard": 2, "rejecting": 3}
                if signal_priority.get(signal, 0) > signal_priority.get(max_backpressure_signal, 0):
                    max_backpressure_signal = signal

            safety = data.get("safety", {})
            if safety:
                counts = safety.get("admission_counts", {})
                for key in admission_counts:
                    admission_counts[key] += counts.get(key, 0)
                rejected_task_ids.update(safety.get("rejected_task_ids", []))
                degraded_task_ids.update(safety.get("degraded_task_ids", []))
                deferred_task_ids.update(safety.get("safety_deferred_task_ids", []))
                all_safety_violations.update(safety.get("safety_violations", []))

        elif event_type == "scheduler.kernel_probe" and kernel_probe is None:
            kernel_probe = data

        elif event_type == "scheduler.adaptive_tuning":
            if data.get("observed_decisions", 0) > 0:
                adaptive_tuning_enabled_count += 1
            reasons = data.get("reasons", [])
            for reason in reasons:
                if reason not in all_adaptive_reasons:
                    all_adaptive_reasons.append(reason)
            before = data.get("selected_policy_before_bias")
            after = data.get("selected_policy_after_bias")
            if before and after and before != after:
                adaptive_policy_changed_count += 1
                adjusted_desc = f"{before} -> {after}"
                if adjusted_desc not in adaptive_adjusted_policies:
                    adaptive_adjusted_policies.append(adjusted_desc)

        elif event_type == "scheduler.runtime_health":
            score = data.get("runtime_health_score")
            if score is not None:
                health_scores.append(score)
            pl = data.get("pressure_level")
            if pl:
                pressure_levels.append(pl)
            qp = data.get("queue_pressure", 0.0)
            max_queue_pressure = max(max_queue_pressure, qp)
            rp = data.get("resource_pressure", 0.0)
            max_resource_pressure = max(max_resource_pressure, rp)
            fp = data.get("fairness_pressure", 0.0)
            max_fairness_pressure = max(max_fairness_pressure, fp)
            dp = data.get("dependency_pressure", 0.0)
            max_dependency_pressure = max(max_dependency_pressure, dp)
            hw = data.get("health_warnings", [])
            all_health_warnings.update(hw)

    return {
        "policies": policies,
        "workload_classes": workload_classes,
        "fairness_scores": fairness_scores,
        "starvation_risks": starvation_risks,
        "kernel_probe": kernel_probe,
        "bpf_map_task_count": bpf_map_task_count,
        "health_scores": health_scores,
        "pressure_levels": pressure_levels,
        "max_queue_pressure": max_queue_pressure,
        "max_resource_pressure": max_resource_pressure,
        "max_fairness_pressure": max_fairness_pressure,
        "max_dependency_pressure": max_dependency_pressure,
        "health_warnings": sorted(all_health_warnings),
        # SLA/Risk data
        "sla_counts": sla_counts,
        "deadline_task_count": deadline_task_count,
        "max_urgency_score": max_urgency_score,
        "max_task_risk_score": max_task_risk_score,
        "high_risk_task_ids": sorted(high_risk_task_ids),
        "risk_warnings": sorted(all_risk_warnings),
        "risk_aware_count": risk_aware_count,
        # Backpressure/Safety data
        "max_backpressure_signal": max_backpressure_signal,
        "admission_counts": admission_counts,
        "rejected_task_ids": sorted(rejected_task_ids),
        "degraded_task_ids": sorted(degraded_task_ids),
        "deferred_task_ids": sorted(deferred_task_ids),
        "safety_violations": sorted(all_safety_violations),
        # Adaptive tuning data
        "adaptive_tuning_enabled_count": adaptive_tuning_enabled_count,
        "adaptive_adjusted_policies": adaptive_adjusted_policies,
        "all_adaptive_reasons": all_adaptive_reasons,
        "adaptive_policy_changed_count": adaptive_policy_changed_count,
    }


def generate_report(events: list[dict]) -> str:
    """Generate a Markdown evidence report."""
    audit_result = audit_scheduler_events(events)
    evidence = extract_evidence_data(events)

    lines = []
    lines.append("# Member A Scheduler Evidence")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- scheduler decisions: {audit_result.decision_count}")
    lines.append(f"- policies seen: {', '.join(audit_result.policies_seen) or 'none'}")
    lines.append(f"- workload classes seen: {', '.join(audit_result.workload_classes_seen) or 'none'}")
    lines.append(f"- fairness warnings: {audit_result.fairness_warning_count}")
    lines.append(f"- kernel probe: {audit_result.kernel_probe_count}")
    lines.append(f"- bpf map updates: {audit_result.bpf_map_updated_count}")
    lines.append(f"- audit ok: {audit_result.ok}")
    lines.append("")

    # Policy Selection Evidence
    lines.append("## Policy Selection Evidence")
    lines.append("")
    if evidence["policies"]:
        lines.append("The scheduler selected the following policies based on workload classification:")
        lines.append("")
        for policy in evidence["policies"]:
            lines.append(f"- `{policy}`")
        lines.append("")
        lines.append("Policy selection is driven by:")
        lines.append("- Workload class (interactive, batch, ai_inference, etc.)")
        lines.append("- Priority and prefix heat (cache affinity)")
        lines.append("- Starvation risk (fairness balanced)")
    else:
        lines.append("No policy selection evidence found.")
    lines.append("")

    # Fairness Evidence
    lines.append("## Fairness Evidence")
    lines.append("")
    if evidence["fairness_scores"]:
        avg_score = sum(evidence["fairness_scores"]) / len(evidence["fairness_scores"])
        lines.append(f"- Average fairness score: {avg_score:.4f}")
        lines.append(f"- Starvation risks observed: {', '.join(evidence['starvation_risks']) or 'none'}")
        lines.append("")
        lines.append("Fairness tracking includes:")
        lines.append("- wait_ms: time tasks spend waiting")
        lines.append("- deferred_count: how many times a task was deferred")
        lines.append("- starvation_risk: low/medium/high based on thresholds")
    else:
        lines.append("No fairness evidence found.")
    lines.append("")

    # OS Integration Evidence
    lines.append("## OS Integration Evidence")
    lines.append("")
    if evidence["kernel_probe"]:
        lines.append("Kernel probe result:")
        lines.append("```json")
        lines.append(json.dumps(evidence["kernel_probe"], indent=2))
        lines.append("```")
        lines.append("")
        lines.append(f"Mock BPF map task count: {evidence['bpf_map_task_count']}")
    else:
        lines.append("No kernel probe evidence found.")
    lines.append("")

    # Runtime Health Evidence
    lines.append("## Runtime Health Evidence")
    lines.append("")
    lines.append("Runtime Health Score is a user-space Runtime health assessment metric.")
    lines.append("It does NOT represent Linux kernel health status.")
    lines.append("The metric is used to explain AgentOS Runtime scheduling pressure and risk.")
    lines.append("")
    if evidence["health_scores"]:
        avg_score = sum(evidence["health_scores"]) / len(evidence["health_scores"])
        min_score = min(evidence["health_scores"])
        worst_level = _worst_pressure_level(evidence["pressure_levels"])
        lines.append(f"- Average health score: {avg_score:.4f}")
        lines.append(f"- Minimum health score: {min_score:.4f}")
        lines.append(f"- Worst pressure level: {worst_level}")
        lines.append(f"- Max queue pressure: {evidence['max_queue_pressure']:.4f}")
        lines.append(f"- Max resource pressure: {evidence['max_resource_pressure']:.4f}")
        lines.append(f"- Max fairness pressure: {evidence['max_fairness_pressure']:.4f}")
        lines.append(f"- Max dependency pressure: {evidence['max_dependency_pressure']:.4f}")
        if evidence["health_warnings"]:
            lines.append(f"- Health warnings: {'; '.join(evidence['health_warnings'])}")
    else:
        lines.append("No runtime health evidence found.")
    lines.append("")

    # SLA / Risk-aware Scheduling Evidence
    lines.append("## SLA / Risk-aware Scheduling Evidence")
    lines.append("")
    lines.append("P5's risk score is a rule-based risk assessment, not a machine learning prediction.")
    lines.append("It does not directly represent real business value, but is used for Runtime scheduling-side risk awareness.")
    lines.append("")
    if evidence["max_urgency_score"] > 0 or evidence["max_task_risk_score"] > 0:
        lines.append(f"- SLA class distribution: critical={evidence['sla_counts'].get('critical', 0)}, interactive={evidence['sla_counts'].get('interactive', 0)}, batch={evidence['sla_counts'].get('batch', 0)}, best_effort={evidence['sla_counts'].get('best_effort', 0)}")
        lines.append(f"- Deadline task count: {evidence['deadline_task_count']}")
        lines.append(f"- Max urgency score: {evidence['max_urgency_score']:.4f}")
        lines.append(f"- Max task risk score: {evidence['max_task_risk_score']:.4f}")
        if evidence["high_risk_task_ids"]:
            lines.append(f"- High risk task IDs: {', '.join(evidence['high_risk_task_ids'])}")
        if evidence["risk_warnings"]:
            lines.append(f"- Risk warnings: {'; '.join(evidence['risk_warnings'])}")
        lines.append(f"- Risk-aware policy count: {evidence['risk_aware_count']}")
    else:
        lines.append("No SLA/Risk evidence found.")
    lines.append("")

    # Backpressure + Safety Envelope Evidence
    lines.append("## Backpressure + Safety Envelope Evidence")
    lines.append("")
    lines.append("P6 is a user-space Runtime safety layer, NOT a Linux kernel sandbox.")
    lines.append("P6's reject/defer/degrade decisions are admission control decisions, not system-call-level isolation.")
    lines.append("")
    if evidence["max_backpressure_signal"] != "none" or evidence["admission_counts"].get("reject", 0) > 0:
        lines.append(f"- Max backpressure signal: {evidence['max_backpressure_signal']}")
        lines.append(f"- Admission counts: allow={evidence['admission_counts'].get('allow', 0)}, defer={evidence['admission_counts'].get('defer', 0)}, degrade={evidence['admission_counts'].get('degrade', 0)}, reject={evidence['admission_counts'].get('reject', 0)}")
        if evidence["rejected_task_ids"]:
            lines.append(f"- Rejected task IDs: {', '.join(evidence['rejected_task_ids'])}")
        if evidence["degraded_task_ids"]:
            lines.append(f"- Degraded task IDs: {', '.join(evidence['degraded_task_ids'])}")
        if evidence["deferred_task_ids"]:
            lines.append(f"- Deferred task IDs: {', '.join(evidence['deferred_task_ids'])}")
        if evidence["safety_violations"]:
            lines.append(f"- Safety violations: {'; '.join(evidence['safety_violations'])}")
    else:
        lines.append("No backpressure/safety evidence found.")
    lines.append("")

    # Adaptive Policy Tuning Evidence
    lines.append("## Adaptive Policy Tuning Evidence")
    lines.append("")
    lines.append("Adaptive tuning uses a sliding-window, rule-based approach to adjust policy selection bias.")
    lines.append("It is NOT a black-box ML or reinforcement learning system. All adjustments are explainable,")
    lines.append("auditable, and based on transparent threshold rules.")
    lines.append("")
    if evidence["adaptive_tuning_enabled_count"] > 0:
        lines.append(f"- Adaptive tuning enabled decisions: {evidence['adaptive_tuning_enabled_count']}")
        lines.append(f"- Policy adjustments made: {evidence['adaptive_policy_changed_count']}")
        if evidence["adaptive_adjusted_policies"]:
            lines.append(f"- Adjusted policies: {'; '.join(evidence['adaptive_adjusted_policies'])}")
        if evidence["all_adaptive_reasons"]:
            lines.append(f"- Bias reasons: {'; '.join(evidence['all_adaptive_reasons'])}")
        lines.append("")
        lines.append("Adaptive rules implemented:")
        lines.append("1. Fairness adaptive: responds to increasing wait times and low fairness scores")
        lines.append("2. Cache affinity adaptive: responds to prefix/cache locality signals")
        lines.append("3. SLA/Risk adaptive: responds to deadline urgency and high-risk tasks")
        lines.append("4. Backpressure/Safety adaptive: responds to resource pressure and safety rejections")
    else:
        lines.append("No adaptive tuning evidence found.")
    lines.append("")
    lines.append("> Compatibility note: This is sliding-window rule-based tuning, not black-box ML.")
    lines.append("> Under mock backend, cache signals are Runtime-side simulated evidence,")
    lines.append("> not real cloud KV cache.")
    lines.append("")

    # Compatibility Notes
    lines.append("## Compatibility Notes")
    lines.append("")
    lines.append("- This is scheduler-side OS integration evidence.")
    lines.append("- It does not claim to replace Linux kernel scheduling.")
    lines.append("- Unsupported kernels fall back to mock BPF map mode.")
    lines.append("")

    # Kernel Integration Boundary
    lines.append("## Kernel Integration Boundary")
    lines.append("")
    lines.append("AgentOS Runtime implements a user-space scheduling control plane for multi-agent execution.")
    lines.append("It does not replace Linux CFS or take over kernel scheduling.")
    lines.append("")
    lines.append("- sched_ext/eBPF: capability probe + mock BPF map adapter")
    lines.append("- cgroup v2: real kernel interface for resource isolation (when available)")
    lines.append("- Runtime does NOT modify kernel scheduler policies")
    lines.append("")

    # Audit Result
    lines.append("## Audit Result")
    lines.append("")
    lines.append(f"- Audit OK: {audit_result.ok}")
    lines.append(f"- Decision count: {audit_result.decision_count}")
    lines.append(f"- Policies seen: {', '.join(audit_result.policies_seen) or 'none'}")
    lines.append(f"- Workload classes seen: {', '.join(audit_result.workload_classes_seen) or 'none'}")
    lines.append(f"- Error count: {audit_result.error_count}")
    lines.append(f"- Warning count: {audit_result.warning_count}")
    lines.append("")
    if audit_result.errors:
        lines.append("### Errors")
        lines.append("")
        for error in audit_result.errors:
            lines.append(f"- {error}")
        lines.append("")
    if audit_result.warnings:
        lines.append("### Warnings")
        lines.append("")
        for warning in audit_result.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Member A scheduler evidence report"
    )
    parser.add_argument(
        "--events",
        required=True,
        help="Path to events.jsonl file",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output Markdown file path",
    )
    args = parser.parse_args()

    events_path = Path(args.events)
    if not events_path.exists():
        print(f"Error: events file not found: {events_path}", file=sys.stderr)
        sys.exit(1)

    # Read events from JSONL
    events = []
    for line_num, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            events.append(event)
        except json.JSONDecodeError as e:
            print(f"Warning: skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)

    # Generate report
    report = generate_report(events)
    audit_result = audit_scheduler_events(events)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report saved to {out_path}")

    # Write summary.json
    summary_path = out_path.parent / "summary.json"
    summary_path.write_text(
        json.dumps(audit_result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Summary saved to {summary_path}")

    # Copy events to output dir
    events_out = out_path.parent / "events.jsonl"
    events_out.write_text(events_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Events copied to {events_out}")


if __name__ == "__main__":
    main()
