# AgentOS-Runtime Workload Matrix Report

- **Timestamp**: 2026-06-24T13:37:12.375726+00:00
- **Backend**: cache-sim
- **Duration**: 1.029s
- **Workloads**: ai_inference, compile, compression, network

## Aggregate

| Metric | Baseline | Runtime | Delta |
|--------|----------|---------|-------|
| Cache Hit Ratio | 66.6% | 79.5% | +12.9% |
| Tasks Completed | 14/29 | 28/29 | +14 |
| Fault Recovery | 0/4 | 4/4 | +4 |

## Per-Workload Results

| Workload | BL Latency | RT Latency | BL Cache | RT Cache | BL Tasks | RT Tasks | Fault |
|----------|-----------|-----------|----------|----------|----------|----------|-------|
| ai_inference | 0.077s | 0.180s | 58.6% | 76.2% | 3/7 | 7/7 | RECOVERED |
| compile | 0.100s | 0.150s | 66.5% | 77.3% | 4/7 | 7/7 | RECOVERED |
| compression | 0.096s | 0.168s | 71.6% | 82.4% | 4/8 | 7/8 | RECOVERED |
| network | 0.076s | 0.174s | 59.8% | 78.1% | 3/7 | 7/7 | RECOVERED |

## Normalized Metrics (per completed task)

| Workload | BL Tok/Task | RT Tok/Task | BL Latency/Task | RT Latency/Task | Cache Improvement | Fault Recovery |
|----------|------------|------------|----------------|----------------|-------------------|---------------|
| ai_inference | 127 | 123 | 0.026s | 0.026s | +17.6% | Yes |
| compile | 122 | 116 | 0.025s | 0.021s | +10.8% | Yes |
| compression | 297 | 291 | 0.024s | 0.024s | +10.8% | Yes |
| network | 138 | 133 | 0.025s | 0.025s | +18.3% | Yes |

## Interaction Overhead Evidence

| Metric | Value |
|--------|-------|
| Total Runtime Events | 507 |
| Total MessageBus Messages | 0 |
| Total MessageBus Payload Bytes | 0 |
| ContextRef Bytes Saved (vs inline) | 14904 |

### Per-Workload Overhead

| Workload | RT Events | Messages | Payload Bytes | ContextRef Saved | Event Latency(ms) |
|----------|----------|----------|--------------|-----------------|-------------------|
| ai_inference | 127 | 0 | 0 | 2604 | 1.4 |
| compile | 123 | 0 | 0 | 2580 | 1.2 |
| compression | 127 | 0 | 0 | 6768 | 1.3 |
| network | 130 | 0 | 0 | 2952 | 1.3 |

## Real Backend Probe

- **Backend**: mock
- **Available**: Yes

## MessageBus / ContextRef Communication Evidence

| Metric | Value |
|--------|-------|
| Message Count | 70 |
| Context Ref Count | 35 |
| Payload Bytes | 47922 |
| Inline Payload Bytes | 46196 |
| Ref Payload Bytes | 1726 |
| Bytes Saved | 44470 |
| Payload Reduction Ratio | 96.3% |
| ContextRef Bytes Saved (total) | 14904 |
| Workload Count | 4 |

## sched_ext / eBPF Probe

- **sched_ext**: unavailable
- **BPF Map**: mock
- **Reason**: missing /sys/kernel/sched_ext; bpftool not found in PATH; missing CAP_BPF or unsupported kernel
- **cgroup v2**: mounted
- **cgroup v2 path**: /sys/fs/cgroup

> Note: sched_ext / eBPF status is probed at runtime. When unavailable, the system falls back to user-space mock adapters. The overhead evidence above measures user-space Runtime communication costs.

## Scheduler Evidence by Workload

| Workload | workload_class | policies_seen | policy_switches | fairness_score | starvation_warnings | kernel_integration | bpf_map_mode | fallback_reason |
|----------|---------------|---------------|-----------------|----------------|--------------------|--------------------|--------------|-----------------|
| ai_inference | ai_inference | latency_first, risk_aware | 1 | 1.0000 | 0 | mock | mock | forced mock mode |
| compile | compile | latency_first, throughput_first | 2 | 1.0000 | 0 | mock | mock | forced mock mode |
| compression | compression | latency_first | 0 | 1.0000 | 0 | mock | mock | forced mock mode |
| network | network | latency_first, risk_aware | 1 | 1.0000 | 0 | mock | mock | forced mock mode |


## Runtime Health Evidence by Workload

| Workload | avg_health_score | min_health_score | worst_pressure_level | max_queue_pressure | max_resource_pressure | max_fairness_pressure | max_dependency_pressure | health_warnings |
|----------|------------------|------------------|----------------------|--------------------|-----------------------|-----------------------|-------------------------|-----------------|
| ai_inference | 0.8853 | 0.8286 | warning | 0.0000 | 0.0000 | 0.0000 | 0.8571 | dependency pressure is high; dependency pressure is moderate |
| compile | 0.8991 | 0.8286 | warning | 0.0000 | 0.0000 | 0.0000 | 0.8571 | dependency pressure is high; dependency pressure is moderate |
| compression | 0.8812 | 0.8250 | warning | 0.0000 | 0.0000 | 0.0000 | 0.8750 | dependency pressure is high; dependency pressure is moderate |
| network | 0.8853 | 0.8286 | warning | 0.0000 | 0.0000 | 0.0000 | 0.8571 | dependency pressure is high; dependency pressure is moderate |


## SLA / Risk Evidence by Workload

| Workload | critical | interactive | batch | best_effort | deadline_tasks | max_urgency | max_risk | risk_aware_count | risk_warnings |
|----------|----------|-------------|-------|-------------|----------------|-------------|----------|------------------|---------------|
| ai_inference | 0 | 4 | 3 | 0 | 4 | 0.9000 | 0.2900 | 4 | - |
| compile | 0 | 0 | 7 | 0 | 0 | 0.4000 | 0.1300 | 0 | - |
| compression | 0 | 0 | 3 | 5 | 0 | 0.4000 | 0.1300 | 0 | - |
| network | 4 | 2 | 1 | 0 | 6 | 1.0000 | 0.3700 | 5 | critical task present |


## Backpressure / Safety Evidence by Workload

| Workload | max_backpressure | allow | defer | degrade | reject | safety_violations | violation_count |
|----------|------------------|-------|-------|---------|--------|-------------------|-----------------|
| ai_inference | soft | 7 | 0 | 0 | 0 | - | 0 |
| compile | soft | 7 | 0 | 0 | 0 | - | 0 |
| compression | soft | 7 | 0 | 0 | 1 | memory limit exceeded | 1 |
| network | soft | 7 | 0 | 0 | 0 | - | 0 |


## Key Findings

1. **Context pooling** improved cache hit ratio by 12.9% across all workloads
2. **Fault recovery** restored 4/4 faulted tasks vs 0 in baseline
3. **Task completion** improved from 14 to 28 tasks
4. **Per-task token efficiency** baseline 177 → runtime 166 tokens/task
5. **Runtime observability** collected 507 events across all workloads
6. **ContextRef savings** 14904 bytes saved via pass-by-reference across 4 workloads
