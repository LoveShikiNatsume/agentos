# 成员A真实内核通路验收报告

## 结论

成员A当前实现已经具备 Agent Runtime 的 OS-facing control plane：

- 真实 cgroup v2 控制：创建 cgroup、写入 CPU/内存/PID 配额、attach PID、验证 membership。
- PSI / cgroup usage 反馈：优先读取 cgroup-local PSI，缺失时回退系统 PSI。
- KernelSchedulerAdapter 边界：Scheduler 只产生决策，Runtime 通过 adapter 执行 OS-facing 操作。
- mock BPF map 保留：用于无权限开发、CI 和 fallback。
- 可选 bpftool pinned BPF map：用于验证真实 BPF map create/update/lookup/dump 通路。

本报告不宣称项目已经实现 sched_ext/eBPF 原生调度器接管 Linux 调度。

## 真实 cgroup v2 验收

验收脚本：

```bash
.venv/bin/python scripts/verify_real_cgroup_v2.py \
  --json-out runs/member_a_real_cgroup_v2.json
```

验收目标：

```json
{
  "available": true,
  "quota_applied": 4,
  "pid_attached": 1,
  "pid_membership_verified": true,
  "child_inheritance_verified": true,
  "initial_usage_samples": 1,
  "running_usage_samples": 1,
  "final_usage_samples": 1,
  "cgroup_local_pressure_samples": 1,
  "cleanup_failed": 0
}
```

判定口径：

- `available=true` 证明 cgroup v2 路径可写且 verifier 成功创建验证 cgroup。
- `quota_applied=4` 覆盖 `cpu.max`、`cpu.weight`、`memory.max`、`pids.max`。
- `pid_membership_verified=true` 证明不是只写 `cgroup.procs`，而是从 `/proc/<pid>/cgroup` 回查确认。
- `child_inheritance_verified=true` 证明 worker 子进程继承 cgroup 归属。
- `initial/running/final` 三阶段样本证明不是只在退出后采一次。
- `cleanup_failed=0` 证明 verifier 创建的 cgroup 已清理。

如果脚本返回 `skip_reason`，说明当前环境缺少安全 delegation 或必要权限，应在 openEuler/root/systemd delegation 环境重新运行。

当前工作区执行记录：

```json
{
  "available": false,
  "delegation_mode": "host_root",
  "skip_reason": "delegation=host_root; set AGENTOS_CGROUP_ALLOW_ROOT=1 to override"
}
```

该结果不是功能失败，而是当前 shell 没有安全 cgroup delegation。参赛验收时需要在 openEuler、容器 delegation、systemd delegation 或显式授权的宿主环境重跑。

## 真实 bpftool pinned map 验收

验收脚本：

```bash
.venv/bin/python scripts/verify_real_bpf_map.py \
  --json-out runs/member_a_real_bpf_map.json
```

验收目标：

```json
{
  "available": true,
  "bpffs_mounted": true,
  "map_created": true,
  "schema_verified": true,
  "updates": 1,
  "lookups": 1,
  "readback_verified": true,
  "dumps": 1,
  "cleanup_failed": 0
}
```

判定口径：

- `bpffs_mounted=true` 证明 `/sys/fs/bpf` 是真实 bpffs。
- `map_created=true` 证明 verifier 创建了真实 pinned BPF map。
- `updates/lookups/dumps` 证明调度记录可以经过 bpftool 写入、查询和 dump。
- `readback_verified=true` 证明 32 字节 key / 64 字节 value ABI 的关键字段可正确回读。

该验收只证明 BPF map 数据通路，不表示 eBPF scheduler 已接管内核调度。

当前工作区执行记录：

```json
{
  "available": false,
  "bpffs_mounted": false,
  "bpftool_version": null,
  "skip_reason": "bpftool not found in PATH"
}
```

该结果表示当前环境没有安装或暴露 `bpftool`，不是 mock BPF map 或 adapter 代码失败。参赛验收时需要在安装 `bpftool` 且 `/sys/fs/bpf` 为 bpffs 的环境重跑。

## P0 feedback-loop demo

无权限环境可运行确定性 demo：

```bash
.venv/bin/python scripts/demo_member_a_feedback_loop.py \
  --output-dir runs/member_a_feedback_loop_demo
```

输出：

- `runs/member_a_feedback_loop_demo/events.jsonl`
- `runs/member_a_feedback_loop_demo/summary.json`

该 demo 使用真实 Runtime、subprocess worker、KernelSchedulerAdapter 和事件流；cgroup-local PSI 由确定性 demo manager 提供，因此适合 CI 和本地无 root 环境。真实权限证明仍以 `verify_real_cgroup_v2.py` 为准。

验收目标：

```json
{
  "accepted": true,
  "local_sample_phases": ["final", "initial", "running"],
  "feedback_applied": 1,
  "affected_decisions": 1,
  "kernel_decision_published": 2,
  "cgroup_removed": 2
}
```

核心证据事件：

- `resource.cgroup_created`
- `resource.cgroup_pid_added`
- `resource.cgroup_quota_applied`
- `resource.usage_sampled`
- `resource.pressure_sampled`
- `resource.cgroup_feedback_applied`
- `scheduler.decision`
- `kernel.decision_published`
- `resource.cgroup_removed`

## 参赛验收清单

- [ ] 在 openEuler 或等价 Linux 特权环境运行 `verify_real_cgroup_v2.py`。
- [ ] 保存 `runs/member_a_real_cgroup_v2.json`。
- [ ] 在 bpffs + bpftool 可用环境运行 `verify_real_bpf_map.py`。
- [ ] 保存 `runs/member_a_real_bpf_map.json`。
- [ ] 在普通环境运行 `demo_member_a_feedback_loop.py`。
- [ ] 保存 `runs/member_a_feedback_loop_demo/summary.json` 和 `events.jsonl`。
- [ ] 答辩时明确区分 cgroup v2 真实控制、BPF map 通路验证和未实现的 sched_ext/eBPF 调度接管。
