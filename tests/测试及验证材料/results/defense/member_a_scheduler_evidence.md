# Member A Scheduler Evidence

## Summary
- scheduler decisions: 1
- policies seen: latency_first
- workload classes seen: interactive
- fairness warnings: 0
- kernel probe: 1
- bpf map updates: 1
- audit ok: True

## Policy Selection Evidence

The scheduler selected the following policies based on workload classification:

- `latency_first`

Policy selection is driven by:
- Workload class (interactive, batch, ai_inference, etc.)
- Priority and prefix heat (cache affinity)
- Starvation risk (fairness balanced)

## Fairness Evidence

- Average fairness score: 1.0000
- Starvation risks observed: low

Fairness tracking includes:
- wait_ms: time tasks spend waiting
- deferred_count: how many times a task was deferred
- starvation_risk: low/medium/high based on thresholds

## OS Integration Evidence

Kernel probe result:
```json
{
  "sched_ext": "unavailable",
  "bpf_map": "mock",
  "fallback_mode": "mock_bpf_map",
  "reason": "forced mock mode"
}
```

Mock BPF map task count: 0

## Compatibility Notes

- This is scheduler-side OS integration evidence.
- It does not claim to replace Linux kernel scheduling.
- Unsupported kernels fall back to mock BPF map mode.
