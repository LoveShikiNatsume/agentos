from __future__ import annotations

import json
import os
import time
from html import escape
from pathlib import Path
from typing import Any, Iterable

from agentos.events.log import EventLog
from agentos.events.types import RuntimeEvent


def load_events(paths: Iterable[str | Path]) -> list[RuntimeEvent]:
    events: list[RuntimeEvent] = []
    for path in paths:
        events.extend(EventLog(path).read())
    return sorted(events, key=lambda event: event.timestamp)


def load_benchmark_summary(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    summary_path = Path(path)
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_dashboard(
    *,
    log_paths: Iterable[str | Path],
    output_path: str | Path = "runs/dashboard/index.html",
    benchmark_summary_path: str | Path | None = None,
    benchmark_report_path: str | Path | None = None,
    workload_matrix_path: str | Path | None = None,
) -> Path:
    output = Path(output_path)
    events = load_events(log_paths)
    summary = load_benchmark_summary(benchmark_summary_path)
    matrix = load_benchmark_summary(workload_matrix_path)
    html = render_dashboard(
        events,
        benchmark_summary=summary,
        benchmark_report_path=benchmark_report_path,
        workload_matrix=matrix,
        source_logs=[str(path) for path in log_paths],
        output_path=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def render_dashboard(
    events: list[RuntimeEvent],
    *,
    benchmark_summary: dict[str, Any] | None = None,
    benchmark_report_path: str | Path | None = None,
    workload_matrix: dict[str, Any] | None = None,
    source_logs: list[str] | None = None,
    output_path: str | Path | None = None,
) -> str:
    data = _summarize_events(events)
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    report_link = _relative_link(benchmark_report_path, output_path)
    benchmark = _benchmark_view(benchmark_summary)

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>AgentOS Runtime Dashboard</title>",
            f"<style>{_CSS}</style>",
            "</head>",
            "<body>",
            '<main class="shell">',
            _header(generated_at, source_logs or [], report_link),
            _kpis(data, benchmark),
            _workload_matrix_section(workload_matrix),
            _benchmark_section(benchmark),
            _agent_section(data["agents"]),
            _resource_section(data["resource_events"]),
            _fault_section(data["fault_events"]),
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _summarize_events(events: list[RuntimeEvent]) -> dict[str, Any]:
    agents: dict[str, dict[str, Any]] = {}
    resource_events: list[dict[str, Any]] = []
    fault_events: list[dict[str, Any]] = []
    counts = {
        "events": len(events),
        "faults": 0,
        "recoveries": 0,
        "cgroup_created": 0,
        "cgroup_pid_added": 0,
        "cgroup_removed": 0,
    }

    for event in events:
        if event.agent_id is not None:
            row = agents.setdefault(
                event.agent_id,
                {
                    "agent_id": event.agent_id,
                    "task_id": event.task_id or "-",
                    "state": "-",
                    "role": "-",
                    "pid": "-",
                    "worker": "-",
                    "exit": "-",
                    "mem_peak_mb": "-",
                    "pids": "-",
                    "cgroup": "-",
                },
            )
            if event.task_id:
                row["task_id"] = event.task_id
            _update_agent_row(row, event)

        if event.event_type.startswith("fault."):
            counts["faults"] += 1
            if event.event_type == "fault.recovery_completed":
                counts["recoveries"] += 1
            fault_events.append(_event_row(event))
        elif event.event_type in {"worker.failed", "worker.timeout"}:
            fault_events.append(_event_row(event))

        if event.event_type.startswith("resource."):
            if event.event_type == "resource.cgroup_created":
                counts["cgroup_created"] += 1
            elif event.event_type == "resource.cgroup_pid_added":
                counts["cgroup_pid_added"] += 1
            elif event.event_type == "resource.cgroup_removed":
                counts["cgroup_removed"] += 1
            resource_events.append(_event_row(event))

    return {
        "agents": list(agents.values()),
        "resource_events": resource_events,
        "fault_events": fault_events,
        "counts": counts,
    }


def _update_agent_row(row: dict[str, Any], event: RuntimeEvent) -> None:
    if event.event_type == "agent.created":
        row["role"] = event.data.get("role", row["role"])
    elif event.event_type == "agent.state_changed":
        row["state"] = event.data.get("new_state", row["state"])
    elif event.event_type == "worker.started":
        row["pid"] = event.data.get("pid", row["pid"])
        row["worker"] = event.data.get("worker_type", row["worker"])
    elif event.event_type in {"worker.finished", "worker.failed", "worker.timeout"}:
        row["pid"] = event.data.get("pid", row["pid"])
        row["worker"] = event.data.get("worker_type", row["worker"])
        row["exit"] = event.data.get("exit_code", row["exit"])
    elif event.event_type in {
        "resource.cgroup_created",
        "resource.cgroup_unavailable",
        "resource.cgroup_pid_added",
    }:
        row["cgroup"] = event.data.get("path", row["cgroup"])
        row["pid"] = event.data.get("pid", row["pid"])
    elif event.event_type == "resource.usage_sampled":
        row["cgroup"] = event.data.get("path", row["cgroup"])
        usage = event.data.get("usage", {})
        if isinstance(usage, dict):
            row["mem_peak_mb"] = usage.get("memory_peak_mb", row["mem_peak_mb"])
            row["pids"] = usage.get("pids_current", row["pids"])


def _event_row(event: RuntimeEvent) -> dict[str, Any]:
    usage = event.data.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return {
        "time": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
        "event_type": event.event_type,
        "agent_id": event.agent_id or "-",
        "task_id": event.task_id or "-",
        "path": event.data.get("path", "-"),
        "pid": event.data.get("pid", "-"),
        "memory_peak_mb": usage.get("memory_peak_mb", "-"),
        "pids_current": usage.get("pids_current", "-"),
        "detail": _compact_detail(event.data),
    }


def _compact_detail(data: dict[str, Any]) -> str:
    compact: dict[str, Any] = {}
    for key in (
        "action",
        "attempt",
        "error",
        "reason",
        "timeout_seconds",
        "exit_code",
        "timed_out",
        "killed",
        "recovery_time_secs",
        "limits",
    ):
        if key in data and data[key] not in (None, "", {}, []):
            compact[key] = data[key]
    return json.dumps(compact, ensure_ascii=False) if compact else "-"


def _benchmark_view(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"available": False, "rows": [], "comparison": {}}
    rows = []
    for mode, values in summary.get("modes", {}).items():
        rows.append(
            {
                "mode": mode,
                "latency": values["total_latency_secs"]["mean"],
                "prompt_tokens": values["total_prompt_tokens"]["mean"],
                "completion_tokens": values["total_completion_tokens"]["mean"],
                "cache_hit_ratio": values["cache_hit_ratio"]["mean"],
                "faults": values["fault_count"]["mean"],
                "events": values["event_count"]["mean"],
            }
        )
    return {
        "available": True,
        "run_count": summary.get("run_count", 0),
        "rows": rows,
        "comparison": summary.get("comparison", {}),
    }


def _header(generated_at: str, source_logs: list[str], report_link: str | None) -> str:
    sources = "".join(f"<code>{_e(path)}</code>" for path in source_logs) or "<span>-</span>"
    report = (
        f'<a href="{_e(report_link)}">benchmark_report.md</a>'
        if report_link is not None
        else "<span>-</span>"
    )
    return (
        '<header class="topbar">'
        "<div>"
        "<h1>AgentOS Runtime Dashboard</h1>"
        f'<p class="meta">Generated { _e(generated_at) }</p>'
        "</div>"
        '<div class="sources">'
        f"<span>Logs</span>{sources}"
        f"<span>Report</span>{report}"
        "</div>"
        "</header>"
    )


def _kpis(data: dict[str, Any], benchmark: dict[str, Any]) -> str:
    counts = data["counts"]
    comparison = benchmark.get("comparison", {})
    latency_delta_pct = comparison.get("latency_delta_pct")
    cache_delta = comparison.get("cache_hit_ratio_delta")
    items = [
        ("Agents", len(data["agents"]), "accent-a"),
        ("Events", counts["events"], "accent-b"),
        ("Fault Events", counts["faults"], "accent-c"),
        ("Recoveries", counts["recoveries"], "accent-d"),
        ("Cgroups Created", counts["cgroup_created"], "accent-a"),
        ("PIDs Added", counts["cgroup_pid_added"], "accent-b"),
        ("Cgroups Removed", counts["cgroup_removed"], "accent-d"),
        ("Latency Delta", _fmt_pct(latency_delta_pct), "accent-c"),
        ("Cache Delta", _fmt_pct(cache_delta), "accent-b"),
    ]
    cells = "".join(
        f'<div class="kpi {accent}"><span>{_e(label)}</span><strong>{_e(value)}</strong></div>'
        for label, value, accent in items
    )
    return f'<section class="kpis">{cells}</section>'


def _workload_matrix_section(matrix: dict[str, Any] | None) -> str:
    if not matrix:
        return _section(
            "Workload Matrix",
            '<p class="empty">No workload matrix summary loaded.</p>',
        )

    workloads = matrix.get("workloads", {})
    aggregate = matrix.get("aggregate", {})
    meta = matrix.get("metadata", {})

    agg_strip = (
        '<div class="benchmark-strip">'
        f'<span>Backend <strong>{_e(meta.get("backend", "-"))}</strong></span>'
        f'<span>Duration <strong>{_e(_fmt_number(meta.get("total_duration_secs", 0)))}s</strong></span>'
        f'<span>Cache Improvement <strong>{_e(_fmt_pct(aggregate.get("cache_hit_ratio_improvement", 0)))}</strong></span>'
        f'<span>Fault Recovery <strong>{_e(aggregate.get("faults_recovered", 0))}/{_e(aggregate.get("faults_total", 0))}</strong></span>'
        f'<span>Tasks <strong>{_e(aggregate.get("total_tasks_completed_baseline", 0))} → {_e(aggregate.get("total_tasks_completed_runtime", 0))}</strong></span>'
        "</div>"
    )

    rows = []
    for wl_name, wl_data in workloads.items():
        bl = wl_data.get("baseline", {})
        rt = wl_data.get("runtime", {})
        comp = wl_data.get("comparison", {})
        fault_str = "RECOVERED" if rt.get("fault_recovered") else (
            "CRASHED" if bl.get("fault_injected") and not bl.get("fault_recovered") else "OK"
        )
        rows.append([
            wl_name,
            _fmt_number(bl.get("total_latency_secs", 0)),
            _fmt_number(rt.get("total_latency_secs", 0)),
            _fmt_pct(bl.get("cache_hit_ratio", 0)),
            _fmt_pct(rt.get("cache_hit_ratio", 0)),
            f'{bl.get("tasks_completed", 0)}/{bl.get("tasks_total", 0)}',
            f'{rt.get("tasks_completed", 0)}/{rt.get("tasks_total", 0)}',
            fault_str,
            str(rt.get("event_count", 0)),
        ])

    table = _table(
        [
            "Workload",
            "BL Latency(s)",
            "RT Latency(s)",
            "BL Cache Hit",
            "RT Cache Hit",
            "BL Tasks",
            "RT Tasks",
            "Fault",
            "Events",
        ],
        rows,
    )

    # Normalized metrics table
    norm_rows = []
    for wl_name, wl_data in workloads.items():
        bl_n = wl_data.get("baseline", {}).get("normalized", {})
        rt_n = wl_data.get("runtime", {}).get("normalized", {})
        norm_rows.append([
            wl_name,
            _fmt_number(bl_n.get("prompt_tokens_per_completed_task", 0), digits=0),
            _fmt_number(rt_n.get("prompt_tokens_per_completed_task", 0), digits=0),
            _fmt_number(bl_n.get("latency_per_completed_task_secs", 0)),
            _fmt_number(rt_n.get("latency_per_completed_task_secs", 0)),
            _fmt_pct(rt_n.get("cache_hit_ratio_improvement", 0)),
            "Yes" if rt_n.get("fault_recovery_rate", 0) > 0 else "No",
        ])
    norm_table = _table(
        ["Workload", "BL Tok/Task", "RT Tok/Task", "BL Lat/Task(s)", "RT Lat/Task(s)", "Cache Impr.", "Recovery"],
        norm_rows,
    )

    # Interaction overhead table
    oh_rows = []
    for wl_name, wl_data in workloads.items():
        rt_oh = wl_data.get("runtime", {}).get("interaction_overhead", {})
        oh_rows.append([
            wl_name,
            str(rt_oh.get("runtime_events", 0)),
            str(rt_oh.get("message_count", 0)),
            str(rt_oh.get("payload_bytes", 0)),
            str(rt_oh.get("contextref_bytes_saved", 0)),
            _fmt_number(rt_oh.get("event_latency_ms_avg", 0), digits=1),
            _fmt_number(rt_oh.get("worker_startup_ms_avg", 0), digits=1),
        ])
    oh_table = _table(
        ["Workload", "RT Events", "Messages", "Payload(B)", "CtxRef Saved(B)", "Evt Lat(ms)", "Worker(ms)"],
        oh_rows,
    )

    # Kernel probe strip
    kp = matrix.get("kernel_probe", {})
    probe_strip = (
        '<div class="benchmark-strip">'
        f'<span>sched_ext <strong>{_e(kp.get("sched_ext", "unavailable"))}</strong></span>'
        f'<span>BPF Map <strong>{_e(kp.get("bpf_map", "mock"))}</strong></span>'
        f'<span>Fallback <strong>{_e(kp.get("reason", "-")[:80])}</strong></span>'
        "</div>"
    )

    agg_oh = aggregate.get("interaction_overhead", {})
    oh_strip = (
        '<div class="benchmark-strip">'
        f'<span>Total RT Events <strong>{_e(agg_oh.get("total_runtime_events", 0))}</strong></span>'
        f'<span>Total Messages <strong>{_e(agg_oh.get("total_message_count", 0))}</strong></span>'
        f'<span>ContextRef Saved <strong>{_e(agg_oh.get("total_contextref_bytes_saved", 0))} B</strong></span>'
        f'<span>Tok/Task BL→RT <strong>'
        f'{_e(_fmt_number(aggregate.get("prompt_tokens_per_task_baseline", 0), digits=0))} → '
        f'{_e(_fmt_number(aggregate.get("prompt_tokens_per_task_runtime", 0), digits=0))}'
        f'</strong></span>'
        "</div>"
    )

    # Backend probe strip
    bp = matrix.get("backend_probe", {})
    backend_strip = ""
    if bp:
        avail = "Yes" if bp.get("available") else "No"
        reason = bp.get("fallback_reason", "")
        backend_strip = (
            '<h3>Real Backend Probe</h3>'
            '<div class="benchmark-strip">'
            f'<span>Backend <strong>{_e(bp.get("backend", "-"))}</strong></span>'
            f'<span>Available <strong>{_e(avail)}</strong></span>'
        )
        if bp.get("base_url"):
            backend_strip += f'<span>URL <strong>{_e(bp["base_url"])}</strong></span>'
        if bp.get("model"):
            backend_strip += f'<span>Model <strong>{_e(bp["model"])}</strong></span>'
        if bp.get("metrics_endpoint"):
            backend_strip += f'<span>Metrics <strong>accessible</strong></span>'
            backend_strip += f'<span>Prefix Cache <strong>{"Yes" if bp.get("prefix_cache_metrics") else "No"}</strong></span>'
        if bp.get("health_latency_ms"):
            backend_strip += f'<span>Latency <strong>{_e(_fmt_number(bp["health_latency_ms"], digits=1))}ms</strong></span>'
        if reason:
            backend_strip += f'<span>Fallback <strong>{_e(reason[:80])}</strong></span>'
        backend_strip += "</div>"

    # Interaction probe table
    ip = matrix.get("interaction_probe", {})
    ip_section = ""
    if ip:
        ip_section = (
            '<h3>MessageBus / ContextRef Communication Evidence</h3>'
            + _table(
                ["Metric", "Value"],
                [
                    ["Message Count", str(ip.get("message_count", 0))],
                    ["Context Ref Count", str(ip.get("context_ref_count", 0))],
                    ["Payload Bytes", str(ip.get("payload_bytes", 0))],
                    ["Inline Payload Bytes", str(ip.get("inline_payload_bytes", 0))],
                    ["Ref Payload Bytes", str(ip.get("ref_payload_bytes", 0))],
                    ["Bytes Saved", str(ip.get("bytes_saved", 0))],
                    ["Payload Reduction Ratio", _fmt_pct(ip.get("payload_reduction_ratio", 0))],
                    ["ContextRef Bytes Saved (total)", str(ip.get("contextref_bytes_saved_total", 0))],
                    ["Workload Count", str(ip.get("workload_count", 0))],
                ],
            )
        )

    # cgroup v2 info
    kpd = matrix.get("kernel_probe_detail", {})
    cgroup_strip = ""
    if kpd.get("cgroup_v2_mounted") is not None or (kpd.get("is_openeuler") and kpd.get("openeuler_version")):
        cgroup_strip = '<div class="benchmark-strip">'
        if kpd.get("cgroup_v2_mounted") is not None:
            cgroup_strip += f'<span>cgroup v2 <strong>{"mounted" if kpd["cgroup_v2_mounted"] else "not mounted"}</strong></span>'
        if kpd.get("cgroup_v2_path"):
            cgroup_strip += f'<span>Path <strong>{_e(kpd["cgroup_v2_path"])}</strong></span>'
        if kpd.get("cgroup_v2_controllers"):
            cgroup_strip += f'<span>Controllers <strong>{_e(", ".join(kpd["cgroup_v2_controllers"][:5]))}</strong></span>'
        if kpd.get("is_openeuler") and kpd.get("openeuler_version"):
            cgroup_strip += f'<span>openEuler <strong>{_e(kpd["openeuler_version"])}</strong></span>'
        cgroup_strip += "</div>"

    body = (
        agg_strip
        + backend_strip
        + "<h3>Per-Workload Comparison</h3>"
        + table
        + "<h3>Normalized Metrics (per completed task)</h3>"
        + oh_strip
        + norm_table
        + "<h3>Interaction Overhead</h3>"
        + oh_table
        + ip_section
        + "<h3>sched_ext / eBPF Probe</h3>"
        + probe_strip
        + cgroup_strip
    )
    return _section("Workload Matrix Evidence", body)


def _benchmark_section(benchmark: dict[str, Any]) -> str:
    if not benchmark["available"]:
        return _section("Benchmark", '<p class="empty">No benchmark summary loaded.</p>')
    rows = [
        [
            row["mode"],
            _fmt_number(row["latency"]),
            _fmt_number(row["prompt_tokens"], digits=1),
            _fmt_number(row["completion_tokens"], digits=1),
            _fmt_pct(row["cache_hit_ratio"]),
            _fmt_number(row["faults"], digits=1),
            _fmt_number(row["events"], digits=1),
        ]
        for row in benchmark["rows"]
    ]
    comparison = benchmark["comparison"]
    delta = _fmt_number(comparison.get("latency_delta_secs", 0.0))
    pct = _fmt_pct(comparison.get("latency_delta_pct", 0.0))
    body = (
        f'<div class="benchmark-strip"><span>Runs <strong>{_e(benchmark.get("run_count", 0))}</strong></span>'
        f"<span>Latency Delta <strong>{_e(delta)}s ({_e(pct)})</strong></span>"
        f'<span>Prompt Savings <strong>{_e(_fmt_number(comparison.get("prompt_token_savings", 0.0), digits=1))}</strong></span></div>'
        + _table(
            [
                "Mode",
                "Latency Mean(s)",
                "Prompt Tokens",
                "Completion Tokens",
                "Cache Hit Ratio",
                "Faults",
                "Events",
            ],
            rows,
        )
    )
    return _section("vLLM Benchmark", body)


def _agent_section(agents: list[dict[str, Any]]) -> str:
    rows = [
        [
            row["agent_id"],
            row["task_id"],
            row["state"],
            row["role"],
            row["pid"],
            row["worker"],
            row["exit"],
            row["mem_peak_mb"],
            row["pids"],
            row["cgroup"],
        ]
        for row in agents
    ]
    return _section(
        "Agents",
        _table(
            [
                "Agent",
                "Task",
                "State",
                "Role",
                "PID",
                "Worker",
                "Exit",
                "Mem Peak MB",
                "PIDs",
                "Cgroup",
            ],
            rows,
        ),
    )


def _resource_section(rows: list[dict[str, Any]]) -> str:
    body = _table(
        ["Time", "Event", "Agent", "Task", "PID", "Mem Peak MB", "PIDs", "Path"],
        [
            [
                row["time"],
                row["event_type"],
                row["agent_id"],
                row["task_id"],
                row["pid"],
                row["memory_peak_mb"],
                row["pids_current"],
                row["path"],
            ]
            for row in rows
        ],
    )
    return _section("Resource Isolation", body)


def _fault_section(rows: list[dict[str, Any]]) -> str:
    body = _table(
        ["Time", "Event", "Agent", "Task", "Detail"],
        [
            [row["time"], row["event_type"], row["agent_id"], row["task_id"], row["detail"]]
            for row in rows
        ],
    )
    return _section("Fault And Recovery", body)


def _section(title: str, body: str) -> str:
    return f'<section class="section"><h2>{_e(title)}</h2>{body}</section>'


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{_e(header)}</th>" for header in headers)
    if not rows:
        body = f'<tr><td colspan="{len(headers)}" class="empty">No rows.</td></tr>'
    else:
        body = "".join(
            "<tr>" + "".join(f"<td>{_e(value)}</td>" for value in row) + "</tr>"
            for row in rows
        )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _relative_link(path: str | Path | None, output_path: str | Path | None) -> str | None:
    if path is None:
        return None
    if output_path is None:
        return str(path)
    try:
        return os.path.relpath(Path(path), start=Path(output_path).parent)
    except ValueError:
        return str(path)


def _fmt_number(value: Any, *, digits: int = 3) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value * 100:.2f}%"


def _e(value: Any) -> str:
    return escape(str(value), quote=True)


_CSS = """
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f6f7f9;
  color: #182026;
}
* { box-sizing: border-box; }
body { margin: 0; background: #f6f7f9; }
.shell { width: min(1480px, calc(100vw - 40px)); margin: 0 auto; padding: 24px 0 36px; }
.topbar {
  display: grid;
  grid-template-columns: 1fr minmax(320px, 560px);
  gap: 24px;
  align-items: end;
  padding: 8px 0 18px;
  border-bottom: 1px solid #d9dee5;
}
h1 { margin: 0; font-size: 30px; line-height: 1.1; letter-spacing: 0; }
h2 { margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }
.meta { margin: 8px 0 0; color: #596672; font-size: 13px; }
.sources { display: grid; grid-template-columns: 64px 1fr; gap: 8px 12px; align-items: center; font-size: 13px; }
.sources span { color: #596672; }
code {
  display: inline-block;
  max-width: 100%;
  padding: 3px 6px;
  border: 1px solid #d9dee5;
  background: #ffffff;
  overflow-wrap: anywhere;
}
a { color: #1f6f78; text-decoration: none; }
a:hover { text-decoration: underline; }
.kpis {
  display: grid;
  grid-template-columns: repeat(9, minmax(112px, 1fr));
  gap: 10px;
  margin: 18px 0;
}
.kpi {
  min-height: 78px;
  padding: 12px;
  background: #ffffff;
  border: 1px solid #d9dee5;
  border-left-width: 4px;
}
.kpi span { display: block; color: #596672; font-size: 12px; margin-bottom: 10px; }
.kpi strong { display: block; font-size: 22px; line-height: 1.1; overflow-wrap: anywhere; }
.accent-a { border-left-color: #1f6f78; }
.accent-b { border-left-color: #6f7d1f; }
.accent-c { border-left-color: #9a4c1e; }
.accent-d { border-left-color: #28784a; }
.section { padding: 18px 0 4px; border-top: 1px solid #d9dee5; }
.benchmark-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 18px;
  margin-bottom: 12px;
  color: #596672;
  font-size: 13px;
}
.benchmark-strip strong { color: #182026; }
.table-wrap { width: 100%; overflow-x: auto; background: #ffffff; border: 1px solid #d9dee5; }
table { width: 100%; border-collapse: collapse; min-width: 860px; font-size: 13px; }
th, td { padding: 9px 10px; border-bottom: 1px solid #e8ebef; text-align: left; vertical-align: top; }
th { position: sticky; top: 0; background: #eef1f4; color: #34404b; font-weight: 650; }
td { color: #25313b; overflow-wrap: anywhere; }
tbody tr:hover { background: #f9fafb; }
.empty { color: #7b8791; text-align: left; }
@media (max-width: 1100px) {
  .topbar { grid-template-columns: 1fr; }
  .kpis { grid-template-columns: repeat(3, minmax(140px, 1fr)); }
}
@media (max-width: 640px) {
  .shell { width: min(100vw - 24px, 1480px); padding-top: 16px; }
  .kpis { grid-template-columns: repeat(2, minmax(128px, 1fr)); }
  h1 { font-size: 24px; }
}
"""
