# AgentOS sched_ext Stage 4 Policy Scheduler

This directory contains the experimental `agentos_scx` scheduler. It
uses sched_ext partial mode, so only tasks explicitly assigned policy
`SCHED_EXT` are managed by the BPF scheduler. Stage 4 implements latency FIFO,
normal weighted-vtime, and batch weighted-vtime DSQs. Policy lookup failures,
invalid fields, stale generations, and heartbeat expiry use the normal queue
with weight 100 and a 5 ms slice.

Batch starvation protection is tracked independently per CPU. Enqueueing on a
custom DSQ also kicks the task's associated CPU when it is idle, which is
required for single-CPU-affinity workers queued on shared DSQs.

## Pinned upstream

- Linux stable branch: `linux-6.12.y`
- Commit: `296aabce459470a4c1b68ffd0c0c0920e563aaad`
- Validated kernel release: `6.12.95-agentos-scx`
- Reference files: `tools/sched_ext/scx_simple.bpf.c`, `scx_simple.c`, the
  sched_ext Makefile, and headers under `tools/sched_ext/include/scx/`

The BPF scheduler and loader are GPL-2.0 adaptations of the upstream examples.
No Linux, libbpf, or sched_ext source is vendored here. The build consumes the
exact pinned kernel checkout through `KERNEL_SRC`.

## Build

```bash
make KERNEL_SRC=/opt/kernel-src/linux-6.12.y print-config
make KERNEL_SRC=/opt/kernel-src/linux-6.12.y all
```

Generated files are placed under `build/`:

```text
build/bin/agentos_scx
build/bin/agentos_scx_worker
build/bin/agentos_scx_bench_worker
build/bin/scx_simple_partial
build/include/agentos_scx.bpf.skel.h
```

The build fails if `KERNEL_SRC` is not the pinned git commit or does not contain
a built `vmlinux`.

## Manual smoke test

Keep the VM console available. In terminal 1:

```bash
sudo build/bin/agentos_scx --stats-interval-ms 250
```

In terminal 2:

```bash
sudo build/bin/agentos_scx_worker --duration-ms 500
cat /sys/kernel/sched_ext/state
cat /sys/kernel/sched_ext/switch_all
cat /sys/kernel/sched_ext/root/ops
```

Stop the loader with SIGINT or SIGTERM and verify that `state` returns to
`disabled`. Do not run this experimental scheduler on a production host.

## Automated VM acceptance

Run the verifier through `sudo` so it can both attach sched_ext and verify that
the loader rejects the original non-root user. Pass the PID of the invoking
shell as an unmanaged CFS control:

```bash
SHELL_PID=$$
sudo ../../.venv-sched-ext/bin/python \
  ../../scripts/verify_real_sched_ext.py \
  --unmanaged-pid "$SHELL_PID"
```

The strict Stage 2 run executes exactly 100 workers, exercises affinity and
SIGINT/SIGTERM cleanup, checks the kernel log window, and writes evidence under
`runs/member_a_sched_ext_validation/stage2_skeleton/`. Acceptance requires
`summary.json` to contain `"accepted": true` and the final sched_ext state to
be `disabled`.

After the Stage 2 skeleton remains accepted, verify the Stage 3 Runtime policy
binding path:

```bash
sudo ../../.venv-sched-ext/bin/python \
  ../../scripts/verify_sched_ext_runtime.py
```

This starts the loader, runs a real subprocess DAG in strict mode, verifies
every task-to-PID policy readback and `ext.enabled` transition, and requires the
policy map to be empty after Worker cleanup. Evidence is written to
`runs/member_a_sched_ext_validation/stage3_runtime/`.

## Stage 4 VM verification

First verify queue, weight, slice, and every fallback class:

```bash
sudo ../../.venv-sched-ext/bin/python \
  ../../scripts/verify_sched_ext_policy_effects.py \
  --scheduler-binary "$PWD/build/bin/agentos_scx" \
  --simple-binary "$PWD/build/bin/scx_simple_partial" \
  --worker-binary "$PWD/build/bin/agentos_scx_bench_worker"
```

Run the three-iteration smoke ablation before the formal experiment:

```bash
sudo ../../.venv-sched-ext/bin/python \
  ../../scripts/run_sched_ext_stage4_ablation.py \
  --scheduler-binary "$PWD/build/bin/agentos_scx" \
  --simple-binary "$PWD/build/bin/scx_simple_partial" \
  --worker-binary "$PWD/build/bin/agentos_scx_bench_worker" \
  --policy-effects-summary \
    ../../runs/member_a_sched_ext_validation/stage4_policy_effects/summary.json \
  --iterations 3 --duration-ms 3000
```

The Stage 4 trial controller, loader, and heartbeat thread are pinned to the
reserved CPU 0, preventing the controller from being starved while it releases
the startup barriers. Mixed-workload Workers use affinity set `{1,2}`, while
the background CFS pressure process is isolated on CPU 3. Keeping fair-class
pressure off the
Worker CPUs avoids partial-mode starvation of SCHED_EXT tasks direct-dispatched
to a per-CPU local DSQ. The ablation preflight requires CPUs 0-3 and records the
topology in `summary.json`. Weight comparison Workers are still co-located on
one CPU by the policy-effects verifier.
Each Worker also publishes `READY` after applying affinity, and the controller
verifies that the process is sleeping on its stdin barrier before assigning
`SCHED_EXT`. This prevents setup work from entering sched_ext before all
startup prerequisites are complete.

Formal acceptance uses `--iterations 20 --warmups 2 --duration-ms 12000`.
The policy-effects and ablation summaries must both contain
`"accepted": true`, their `SHA256SUMS` files must verify, and sched_ext must
finish in `disabled` state. Both smoke and formal acceptance reject a batch
kernel wait above 250 ms, a process progress gap above 500 ms, or a Worker
startup wait above 500 ms.

## Stage 5 reliability verification

Stage 5 adds a loader-owned one-second heartbeat, Runtime-owned 500 ms policy
leases, locked global-config updates, and the Runtime lifecycle states
`active`, `degraded`, and `fallback`. The loader remains externally managed;
the Runtime never starts or restarts it.

Run the complete one-iteration smoke matrix first. SysRq-S is destructive to
the active scheduler, so keep a VM console available and opt in explicitly:

```bash
sudo ../../.venv-sched-ext/bin/python \
  ../../scripts/verify_sched_ext_reliability.py \
  --scheduler-binary "$PWD/build/bin/agentos_scx" \
  --iterations 1 --allow-sysrq
```

The formal matrix runs all 15 auto cases and four strict cases 20 times, then
runs the one-hour soak:

```bash
sudo ../../.venv-sched-ext/bin/python \
  ../../scripts/verify_sched_ext_reliability.py \
  --scheduler-binary "$PWD/build/bin/agentos_scx" \
  --output-dir \
    ../../runs/member_a_sched_ext_validation/stage5_reliability/formal \
  --iterations 20 --allow-sysrq \
  --soak-seconds 3600 --require-soak
```

Every trial has its own loader and pinned maps. Evidence includes environment
and artifact hashes, raw trials, structured Runtime events, kernel log window,
sysfs snapshots, summary, and recursive `SHA256SUMS`. Acceptance requires
`summary.json` to contain `"accepted": true` and the final sched_ext state to
be `disabled`.

This remains an experimental partial-mode scheduler. Stage 5 does not add a
systemd unit and does not claim that SCHED_EXT outranks fair- or RT-class work.

## Stage 6 release observation

Stage 6 leaves the scheduling policy and BPF schemas unchanged. The benchmark
Worker adds observation-only `getrusage()` fields for user/system CPU time and
voluntary/involuntary context switches. Mixed trials additionally record the
CPU 0-3 utilization and system context-switch delta from `/proc/stat`.

These fields are additive and optional: the release aggregator accepts frozen
Stage 4 JSON without them and never converts a missing observation to zero.
The frozen Stage 4 20-run matrix remains the primary performance sample; the
Stage 6 three-run matrix is only a release-candidate regression.
