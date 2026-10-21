# AgentOS-Runtime Workload Matrix Report

- **Timestamp**: 2026-06-20T07:55:39.388956+00:00
- **Backend**: deepseek
- **Duration**: 435.047s
- **Workloads**: ai_inference, compile, compression, network

## Aggregate

| Metric | Baseline | Runtime | Delta |
|--------|----------|---------|-------|
| Cache Hit Ratio | 32.2% | 40.4% | +8.2% |
| Tasks Completed | 14/28 | 28/28 | +14 |
| Fault Recovery | 0/4 | 4/4 | +4 |

## Per-Workload Results

| Workload | BL Latency | RT Latency | BL Cache | RT Cache | BL Tasks | RT Tasks | Fault |
|----------|-----------|-----------|----------|----------|----------|----------|-------|
| ai_inference | 31.523s | 116.931s | 0.0% | 0.0% | 3/7 | 7/7 | RECOVERED |
| compile | 35.896s | 44.053s | 0.0% | 0.0% | 4/7 | 7/7 | RECOVERED |
| compression | 34.699s | 104.536s | 63.6% | 86.6% | 4/7 | 7/7 | RECOVERED |
| network | 20.728s | 46.675s | 0.0% | 0.0% | 3/7 | 7/7 | RECOVERED |

## Normalized Metrics (per completed task)

| Workload | BL Tok/Task | RT Tok/Task | BL Latency/Task | RT Latency/Task | Cache Improvement | Fault Recovery |
|----------|------------|------------|----------------|----------------|-------------------|---------------|
| ai_inference | 111 | 105 | 10.508s | 16.704s | +0.0% | Yes |
| compile | 112 | 106 | 8.974s | 6.293s | +0.0% | Yes |
| compression | 302 | 296 | 8.675s | 14.934s | +23.0% | Yes |
| network | 133 | 127 | 6.909s | 6.668s | +0.0% | Yes |

## Interaction Overhead Evidence

| Metric | Value |
|--------|-------|
| Total Runtime Events | 399 |
| Total MessageBus Messages | 0 |
| Total MessageBus Payload Bytes | 0 |
| ContextRef Bytes Saved (vs inline) | 14904 |

### Per-Workload Overhead

| Workload | RT Events | Messages | Payload Bytes | ContextRef Saved | Event Latency(ms) |
|----------|----------|----------|--------------|-----------------|-------------------|
| ai_inference | 100 | 0 | 0 | 2604 | 1169.3 |
| compile | 99 | 0 | 0 | 2580 | 445.0 |
| compression | 100 | 0 | 0 | 6768 | 1045.4 |
| network | 100 | 0 | 0 | 2952 | 466.7 |

## Real Backend Probe

- **Backend**: deepseek
- **Available**: Yes
- **Base URL**: https://api.deepseek.com
- **Model**: deepseek-chat
- **/v1/models**: accessible

## MessageBus / ContextRef Communication Evidence

| Metric | Value |
|--------|-------|
| Message Count | 56 |
| Context Ref Count | 28 |
| Payload Bytes | 38325 |
| Inline Payload Bytes | 36946 |
| Ref Payload Bytes | 1379 |
| Bytes Saved | 35567 |
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

## Key Findings

1. **Context pooling** improved cache hit ratio by 8.2% across all workloads
2. **Fault recovery** restored 4/4 faulted tasks vs 0 in baseline
3. **Task completion** improved from 14 to 28 tasks
4. **Per-task token efficiency** baseline 171 → runtime 158 tokens/task
5. **Runtime observability** collected 399 events across all workloads
6. **ContextRef savings** 14904 bytes saved via pass-by-reference across 4 workloads
