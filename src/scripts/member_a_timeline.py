#!/usr/bin/env python3
"""Generate a compact Member A runtime evidence timeline from events JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_INTERESTING_EVENTS = {
    "worker.started",
    "worker.finished",
    "worker.failed",
    "worker.timeout",
    "resource.cgroup_created",
    "resource.cgroup_pid_added",
    "resource.cgroup_quota_applied",
    "resource.usage_sampled",
    "resource.pressure_sampled",
    "resource.cgroup_feedback_applied",
    "resource.cgroup_removed",
    "resource.cgroup_cleanup_failed",
    "scheduler.decision",
    "kernel.decision_published",
    "scheduler.bpf_map_updated",
}


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        if isinstance(event, dict) and "event_type" in event:
            events.append(event)
    return events


def _summary(event: dict[str, Any]) -> str:
    data = event.get("data", {})
    event_type = event.get("event_type")
    if event_type == "worker.started":
        return f"pid={data.get('pid')} worker={data.get('worker_type')}"
    if event_type == "worker.finished":
        return f"pid={data.get('pid')} duration={data.get('duration_secs')}s"
    if event_type in {"worker.failed", "worker.timeout"}:
        return f"pid={data.get('pid')} error={data.get('error')}"
    if event_type == "resource.cgroup_created":
        return f"path={data.get('path')}"
    if event_type == "resource.cgroup_pid_added":
        return (
            f"pid={data.get('pid')} "
            f"verified={data.get('pid_membership_verified')}"
        )
    if event_type == "resource.cgroup_quota_applied":
        return f"mode={data.get('mode')} pid_attached={data.get('pid_attached')}"
    if event_type == "resource.usage_sampled":
        usage = data.get("usage", {})
        return (
            f"phase={data.get('phase')} "
            f"cpu_usec={usage.get('cpu_usage_usec')} "
            f"mem_peak_mb={usage.get('memory_peak_mb')}"
        )
    if event_type == "resource.pressure_sampled":
        pressure = data.get("pressure", {})
        return (
            f"scope={data.get('scope')} phase={data.get('phase')} "
            f"source={pressure.get('pressure_source')} "
            f"level={pressure.get('pressure_level')}"
        )
    if event_type == "resource.cgroup_feedback_applied":
        return (
            f"source={data.get('pressure_source')} "
            f"level={data.get('effective_pressure_level')}"
        )
    if event_type == "resource.cgroup_removed":
        return f"path={data.get('path')}"
    if event_type == "resource.cgroup_cleanup_failed":
        return f"reason={data.get('reason')}"
    if event_type == "scheduler.decision":
        pressure = data.get("pressure", {})
        backpressure = data.get("backpressure", {})
        return (
            f"policy={data.get('policy')} "
            f"pressure={pressure.get('pressure_source')} "
            f"backpressure={backpressure.get('signal')}"
        )
    if event_type == "kernel.decision_published":
        return (
            f"bpf_mode={data.get('bpf_mode')} "
            f"records={data.get('record_count')} "
            f"fallback={data.get('fallback_active')}"
        )
    if event_type == "scheduler.bpf_map_updated":
        return f"mode={data.get('mode')} task_count={data.get('task_count')}"
    return ""


def _evidence(events: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "worker_started": _count(events, "worker.started"),
        "cgroup_created": _count(events, "resource.cgroup_created"),
        "cgroup_pid_added": _count(events, "resource.cgroup_pid_added"),
        "cgroup_quota_applied": _count(events, "resource.cgroup_quota_applied"),
        "usage_sampled": _count(events, "resource.usage_sampled"),
        "pressure_sampled": _count(events, "resource.pressure_sampled"),
        "cgroup_feedback_applied": _count(events, "resource.cgroup_feedback_applied"),
        "scheduler_decision_with_cgroup_feedback": sum(
            1
            for event in events
            if event.get("event_type") == "scheduler.decision"
            and event.get("data", {}).get("pressure", {}).get("pressure_source")
            == "system+cgroup"
        ),
        "kernel_decision_published": _count(events, "kernel.decision_published"),
        "cgroup_removed": _count(events, "resource.cgroup_removed"),
        "cleanup_failed": _count(events, "resource.cgroup_cleanup_failed"),
    }


def _count(events: list[dict[str, Any]], event_type: str) -> int:
    return sum(1 for event in events if event.get("event_type") == event_type)


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Member A Runtime Timeline",
        "",
        "## Evidence Summary",
        "",
    ]
    for key, value in result["evidence"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Timeline", ""])
    lines.append("| Offset | Event | Agent | Task | Summary |")
    lines.append("| ---: | --- | --- | --- | --- |")
    for item in result["timeline"]:
        lines.append(
            "| "
            f"{item['offset_ms']:.3f} ms | "
            f"`{item['label']}` | "
            f"{item.get('agent_id') or '-'} | "
            f"{item.get('task_id') or '-'} | "
            f"{item['summary']} |"
        )
    return "\n".join(lines) + "\n"


def generate_timeline(
    events_path: str | Path,
    *,
    json_out: str | Path | None = None,
    markdown_out: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(events_path)
    events = _load_events(path)
    interesting = [
        event for event in events if event.get("event_type") in _INTERESTING_EVENTS
    ]
    first_timestamp = min(
        (float(event.get("timestamp", 0.0)) for event in interesting),
        default=0.0,
    )
    timeline = [
        {
            "offset_ms": round(
                (float(event.get("timestamp", 0.0)) - first_timestamp) * 1000,
                3,
            ),
            "label": str(event.get("event_type")),
            "agent_id": event.get("agent_id"),
            "task_id": event.get("task_id"),
            "summary": _summary(event),
        }
        for event in interesting
    ]
    result = {
        "source": str(path),
        "event_count": len(events),
        "timeline_count": len(timeline),
        "evidence": _evidence(events),
        "timeline": timeline,
    }

    if json_out is not None:
        out = Path(json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_out is not None:
        out = Path(markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_markdown(result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("events")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args(argv)

    result = generate_timeline(
        args.events,
        json_out=args.json_out,
        markdown_out=args.markdown_out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
