from agentos.cli import build_parser, main
from agentos.events.log import EventLog
from agentos.events.types import EventCategory, RuntimeEvent, agent_state_changed, task_completed


def test_trace_command_prints_event_summary(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append(
        agent_state_changed(
            "agent-1",
            "ready",
            "running",
            task_id="task-1",
            trace_id="trace-1",
        )
    )
    log.append(task_completed("task-1", "agent-1", "done"))

    rc = main(["trace", "--log", str(path)])

    output = capsys.readouterr().out
    assert rc == 0
    assert "agent.state_changed" in output
    assert "task.completed" in output
    assert "agent-1" in output
    assert "task-1" in output
    assert "ready -> running" in output


def test_trace_command_filters_events(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append(RuntimeEvent(event_type="agent.state_changed", task_id="task-1"))
    log.append(RuntimeEvent(event_type="task.completed", task_id="task-2"))

    rc = main(["trace", "--log", str(path), "--task", "task-2"])

    output = capsys.readouterr().out
    assert rc == 0
    assert "task.completed" in output
    assert "agent.state_changed" not in output


def test_trace_command_formats_memory_pressure_fault(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append(
        RuntimeEvent(
            event_type="fault.memory_pressure",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.FAULT,
            data={
                "memory_mb": 4,
                "chunk_mb": 1,
                "hold_seconds": 0,
                "execution": "subprocess",
            },
        )
    )

    rc = main(["trace", "--log", str(path)])

    output = capsys.readouterr().out
    assert rc == 0
    assert "fault.memory_pressure" in output
    assert "memory_mb=4" in output
    assert "execution=subprocess" in output


def test_trace_command_handles_missing_log(tmp_path, capsys):
    rc = main(["trace", "--log", str(tmp_path / "missing.jsonl")])

    output = capsys.readouterr().out
    assert rc == 0
    assert "No events found" in output


def test_ps_command_aggregates_agent_state_and_llm_stats(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append(
        RuntimeEvent(
            event_type="agent.created",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.AGENT,
            data={"role": "coder", "name": "code"},
        )
    )
    log.append(agent_state_changed("agent-1", "ready", "running", task_id="task-1"))
    log.append(
        RuntimeEvent(
            event_type="llm.response_received",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.LLM,
            data={"model": "mock", "tokens_used": 12, "latency_secs": 0.01},
        )
    )

    rc = main(["ps", "--log", str(path)])

    output = capsys.readouterr().out
    assert rc == 0
    assert "agent-1" in output
    assert "task-1" in output
    assert "running" in output
    assert "mock" in output
    assert "12" in output


def test_ps_command_shows_worker_process_fields(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append(
        RuntimeEvent(
            event_type="agent.created",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.AGENT,
            data={"role": "echo"},
        )
    )
    log.append(
        RuntimeEvent(
            event_type="worker.started",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.AGENT,
            data={"pid": 1234, "worker_type": "echo"},
        )
    )
    log.append(
        RuntimeEvent(
            event_type="worker.finished",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.AGENT,
            data={"pid": 1234, "worker_type": "echo", "exit_code": 0},
        )
    )
    log.append(
        RuntimeEvent(
            event_type="resource.cgroup_created",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.RESOURCE,
            data={"path": "/sys/fs/cgroup/agentos/agent-1"},
        )
    )
    log.append(
        RuntimeEvent(
            event_type="resource.usage_sampled",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.RESOURCE,
            data={
                "path": "/sys/fs/cgroup/agentos/agent-1",
                "usage": {"memory_rss_mb": 12.5, "pids_current": 1},
            },
        )
    )

    rc = main(["ps", "--log", str(path)])

    output = capsys.readouterr().out
    assert rc == 0
    assert "PID" in output
    assert "WORKER" in output
    assert "CGROUP" in output
    assert "1234" in output
    assert "echo" in output
    assert "/sys/fs/cgroup/agentos/agent-1" in output
    assert "12.5" in output


def test_trace_command_formats_worker_events(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append(
        RuntimeEvent(
            event_type="worker.timeout",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.AGENT,
            data={
                "pid": 1234,
                "worker_type": "sleep",
                "exit_code": -9,
                "timed_out": True,
                "error": "worker timed out",
            },
        )
    )

    rc = main(["trace", "--log", str(path)])

    output = capsys.readouterr().out
    assert rc == 0
    assert "worker.timeout" in output
    assert "pid=1234" in output
    assert "worker_type=sleep" in output


def test_trace_command_formats_resource_events(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append(
        RuntimeEvent(
            event_type="resource.usage_sampled",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.RESOURCE,
            data={
                "path": "/sys/fs/cgroup/agentos/agent-1",
                "usage": {
                    "memory_rss_mb": 12.5,
                    "memory_peak_mb": 16.0,
                    "pids_current": 2,
                    "cpu_usage_usec": 12345,
                },
            },
        )
    )

    rc = main(["trace", "--log", str(path)])

    output = capsys.readouterr().out
    assert rc == 0
    assert "resource.usage_sampled" in output
    assert "mem_mb=12.5" in output
    assert "pids=2" in output
    assert "cpu_usec=12345" in output


def test_trace_command_formats_tool_events(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append(
        RuntimeEvent(
            event_type="tool.timeout",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.TOOL,
            data={
                "tool": "slow",
                "duration_secs": 0.01,
                "timed_out": True,
                "error": "tool timed out",
            },
        )
    )

    rc = main(["trace", "--log", str(path)])

    output = capsys.readouterr().out
    assert rc == 0
    assert "tool.timeout" in output
    assert "tool=slow" in output
    assert "timed_out=True" in output
    assert "error=tool timed out" in output


def test_context_command_prints_context_stats(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append(
        RuntimeEvent(
            event_type="context.stats",
            category=EventCategory.CONTEXT,
            data={
                "stats": {
                    "pool": {
                        "unique_segments": 1,
                        "total_hits": 3,
                        "token_savings": 42,
                    },
                    "total_refs": 1,
                }
            },
        )
    )

    rc = main(["context", "--log", str(path)])

    output = capsys.readouterr().out
    assert rc == 0
    assert "unique_segments" in output
    assert "token_savings" in output
    assert "42" in output


def test_inject_fault_command_appends_event(tmp_path, capsys):
    path = tmp_path / "events.jsonl"

    rc = main(
        [
            "inject-fault",
            "--log",
            str(path),
            "--type",
            "timeout",
            "--task",
            "task-1",
        ]
    )

    output = capsys.readouterr().out
    events = list(EventLog(path).read())
    assert rc == 0
    assert "fault.injected" in output
    assert events[0].event_type == "fault.injected"
    assert events[0].task_id == "task-1"
    assert events[0].data["fault_type"] == "timeout"


def test_inject_memory_pressure_fault_command_appends_event(tmp_path, capsys):
    path = tmp_path / "events.jsonl"

    rc = main(
        [
            "inject-fault",
            "--log",
            str(path),
            "--type",
            "memory_pressure",
            "--task",
            "task-1",
            "--memory-mb",
            "4",
            "--chunk-mb",
            "1",
            "--hold-seconds",
            "0",
        ]
    )

    output = capsys.readouterr().out
    events = list(EventLog(path).read())
    assert rc == 0
    assert "fault.injected" in output
    assert events[0].data["fault_type"] == "memory_pressure"
    assert events[0].data["memory_mb"] == 4
    assert events[0].data["chunk_mb"] == 1
    assert events[0].data["hold_seconds"] == 0.0


def test_dashboard_command_writes_static_html(tmp_path, capsys):
    log_path = tmp_path / "events.jsonl"
    summary_path = tmp_path / "benchmark_summary.json"
    output_path = tmp_path / "dashboard.html"
    log = EventLog(log_path)
    log.append(
        RuntimeEvent(
            event_type="agent.created",
            agent_id="agent-1",
            task_id="task-1",
            category=EventCategory.AGENT,
            data={"role": "echo"},
        )
    )
    summary_path.write_text(
        """
{
  "run_count": 1,
  "modes": {
    "baseline": {
      "total_latency_secs": {"mean": 2.0},
      "total_prompt_tokens": {"mean": 10.0},
      "total_completion_tokens": {"mean": 20.0},
      "cache_hit_ratio": {"mean": 0.1},
      "fault_count": {"mean": 0.0},
      "event_count": {"mean": 3.0}
    },
    "runtime": {
      "total_latency_secs": {"mean": 1.0},
      "total_prompt_tokens": {"mean": 8.0},
      "total_completion_tokens": {"mean": 18.0},
      "cache_hit_ratio": {"mean": 0.4},
      "fault_count": {"mean": 0.0},
      "event_count": {"mean": 4.0}
    }
  },
  "comparison": {
    "latency_delta_secs": -1.0,
    "latency_delta_pct": -0.5,
    "prompt_token_savings": 2.0,
    "cache_hit_ratio_delta": 0.3
  }
}
""",
        encoding="utf-8",
    )

    rc = main(
        [
            "dashboard",
            "--log",
            str(log_path),
            "--benchmark-summary",
            str(summary_path),
            "--output",
            str(output_path),
        ]
    )

    output = capsys.readouterr().out
    html = output_path.read_text(encoding="utf-8")
    assert rc == 0
    assert "dashboard written" in output
    assert "AgentOS Runtime Dashboard" in html
    assert "agent-1" in html
    assert "40.00%" in html


def test_live_subcommand_parser():
    parser = build_parser()
    args = parser.parse_args(["live", "--scenario", "devops", "--port", "9401"])
    assert args.command == "live"
    assert args.scenario == "devops"
    assert args.port == 9401


def test_live_subcommand_parser_with_new_scenarios():
    parser = build_parser()

    args = parser.parse_args(["live", "--scenario", "software_eng", "--backend", "cache-sim"])
    assert args.command == "live"
    assert args.scenario == "software_eng"
    assert args.backend == "cache-sim"

    args2 = parser.parse_args(["live", "--scenario", "research", "--backend", "mock"])
    assert args2.scenario == "research"
    assert args2.backend == "mock"


def test_live_subcommand_parser_backend_default():
    parser = build_parser()
    args = parser.parse_args(["live", "--scenario", "devops"])
    assert args.backend == "mock"
