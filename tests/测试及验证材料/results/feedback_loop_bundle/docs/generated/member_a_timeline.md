# Member A Runtime Timeline

## Evidence Summary

- `worker_started`: 2
- `cgroup_created`: 2
- `cgroup_pid_added`: 2
- `cgroup_quota_applied`: 2
- `usage_sampled`: 15
- `pressure_sampled`: 17
- `cgroup_feedback_applied`: 1
- `scheduler_decision_with_cgroup_feedback`: 1
- `kernel_decision_published`: 2
- `cgroup_removed`: 2
- `cleanup_failed`: 0

## Timeline

| Offset | Event | Agent | Task | Summary |
| ---: | --- | --- | --- | --- |
| 0.000 ms | `resource.pressure_sampled` | - | - | scope=system phase=None source=system level=normal |
| 0.225 ms | `scheduler.bpf_map_updated` | - | - | mode=mock task_count=1 |
| 0.226 ms | `kernel.decision_published` | - | - | bpf_mode=mock records=1 fallback=False |
| 0.236 ms | `scheduler.decision` | - | - | policy=latency_first pressure=system backpressure=none |
| 0.828 ms | `worker.started` | agent-pressure | pressure | pid=4 worker=sleep |
| 1.232 ms | `resource.cgroup_created` | agent-pressure | pressure | path=runs/member_a_feedback_loop_demo/cgroup/agent-pressure |
| 1.233 ms | `resource.cgroup_pid_added` | agent-pressure | pressure | pid=4 verified=False |
| 1.233 ms | `resource.cgroup_quota_applied` | agent-pressure | pressure | mode=cgroup_v2 pid_attached=True |
| 1.325 ms | `resource.usage_sampled` | agent-pressure | pressure | phase=initial cpu_usec=20000 mem_peak_mb=97.0 |
| 1.333 ms | `resource.pressure_sampled` | agent-pressure | pressure | scope=agent_cgroup phase=initial source=cgroup level=critical |
| 11.834 ms | `resource.usage_sampled` | agent-pressure | pressure | phase=running cpu_usec=40000 mem_peak_mb=98.0 |
| 11.846 ms | `resource.pressure_sampled` | agent-pressure | pressure | scope=agent_cgroup phase=running source=cgroup level=critical |
| 22.413 ms | `resource.usage_sampled` | agent-pressure | pressure | phase=running cpu_usec=60000 mem_peak_mb=99.0 |
| 22.432 ms | `resource.pressure_sampled` | agent-pressure | pressure | scope=agent_cgroup phase=running source=cgroup level=critical |
| 32.760 ms | `resource.usage_sampled` | agent-pressure | pressure | phase=running cpu_usec=80000 mem_peak_mb=100.0 |
| 32.774 ms | `resource.pressure_sampled` | agent-pressure | pressure | scope=agent_cgroup phase=running source=cgroup level=critical |
| 43.120 ms | `resource.usage_sampled` | agent-pressure | pressure | phase=running cpu_usec=100000 mem_peak_mb=101.0 |
| 43.138 ms | `resource.pressure_sampled` | agent-pressure | pressure | scope=agent_cgroup phase=running source=cgroup level=critical |
| 53.503 ms | `resource.usage_sampled` | agent-pressure | pressure | phase=running cpu_usec=120000 mem_peak_mb=102.0 |
| 53.519 ms | `resource.pressure_sampled` | agent-pressure | pressure | scope=agent_cgroup phase=running source=cgroup level=critical |
| 63.860 ms | `resource.usage_sampled` | agent-pressure | pressure | phase=running cpu_usec=140000 mem_peak_mb=103.0 |
| 63.875 ms | `resource.pressure_sampled` | agent-pressure | pressure | scope=agent_cgroup phase=running source=cgroup level=critical |
| 74.171 ms | `resource.usage_sampled` | agent-pressure | pressure | phase=running cpu_usec=160000 mem_peak_mb=104.0 |
| 74.183 ms | `resource.pressure_sampled` | agent-pressure | pressure | scope=agent_cgroup phase=running source=cgroup level=critical |
| 84.687 ms | `resource.usage_sampled` | agent-pressure | pressure | phase=running cpu_usec=180000 mem_peak_mb=105.0 |
| 84.702 ms | `resource.pressure_sampled` | agent-pressure | pressure | scope=agent_cgroup phase=running source=cgroup level=critical |
| 92.637 ms | `resource.usage_sampled` | agent-pressure | pressure | phase=final cpu_usec=200000 mem_peak_mb=106.0 |
| 92.651 ms | `resource.pressure_sampled` | agent-pressure | pressure | scope=agent_cgroup phase=final source=cgroup level=critical |
| 93.117 ms | `resource.cgroup_removed` | agent-pressure | pressure | path=runs/member_a_feedback_loop_demo/cgroup/agent-pressure |
| 93.123 ms | `worker.finished` | agent-pressure | pressure | pid=4 duration=0.09202811191789806s |
| 93.419 ms | `resource.cgroup_feedback_applied` | - | - | source=system+cgroup level=critical |
| 93.422 ms | `resource.pressure_sampled` | - | - | scope=system phase=None source=system level=normal |
| 93.913 ms | `scheduler.bpf_map_updated` | - | - | mode=mock task_count=2 |
| 93.915 ms | `kernel.decision_published` | - | - | bpf_mode=mock records=2 fallback=False |
| 93.941 ms | `scheduler.decision` | - | - | policy=risk_aware pressure=system+cgroup backpressure=hard |
| 94.707 ms | `worker.started` | agent-followup | followup | pid=6 worker=echo |
| 95.114 ms | `resource.cgroup_created` | agent-followup | followup | path=runs/member_a_feedback_loop_demo/cgroup/agent-followup |
| 95.115 ms | `resource.cgroup_pid_added` | agent-followup | followup | pid=6 verified=False |
| 95.115 ms | `resource.cgroup_quota_applied` | agent-followup | followup | mode=cgroup_v2 pid_attached=True |
| 95.191 ms | `resource.usage_sampled` | agent-followup | followup | phase=initial cpu_usec=220000 mem_peak_mb=107.0 |
| 95.199 ms | `resource.pressure_sampled` | agent-followup | followup | scope=agent_cgroup phase=initial source=cgroup level=critical |
| 105.743 ms | `resource.usage_sampled` | agent-followup | followup | phase=running cpu_usec=240000 mem_peak_mb=108.0 |
| 105.759 ms | `resource.pressure_sampled` | agent-followup | followup | scope=agent_cgroup phase=running source=cgroup level=critical |
| 116.110 ms | `resource.usage_sampled` | agent-followup | followup | phase=running cpu_usec=260000 mem_peak_mb=109.0 |
| 116.129 ms | `resource.pressure_sampled` | agent-followup | followup | scope=agent_cgroup phase=running source=cgroup level=critical |
| 126.445 ms | `resource.usage_sampled` | agent-followup | followup | phase=running cpu_usec=280000 mem_peak_mb=110.0 |
| 126.458 ms | `resource.pressure_sampled` | agent-followup | followup | scope=agent_cgroup phase=running source=cgroup level=critical |
| 136.375 ms | `resource.usage_sampled` | agent-followup | followup | phase=final cpu_usec=300000 mem_peak_mb=111.0 |
| 136.392 ms | `resource.pressure_sampled` | agent-followup | followup | scope=agent_cgroup phase=final source=cgroup level=critical |
| 136.921 ms | `resource.cgroup_removed` | agent-followup | followup | path=runs/member_a_feedback_loop_demo/cgroup/agent-followup |
| 136.929 ms | `worker.finished` | agent-followup | followup | pid=6 duration=0.041909956955350935s |
