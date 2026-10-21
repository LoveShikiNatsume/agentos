# 成员A Feedback Loop Demo

## 目的

这个 demo 用来展示成员A P0 阶段最关键的闭环：

```text
subprocess worker
-> cgroup quota applied
-> initial/running/final usage + PSI sampled
-> cgroup feedback accumulated
-> next scheduler round uses system+cgroup pressure
-> scheduler decision is published through KernelSchedulerAdapter
```

## 运行命令

```bash
.venv/bin/python scripts/demo_member_a_feedback_loop.py \
  --output-dir runs/member_a_feedback_loop_demo
```

输出文件：

```text
runs/member_a_feedback_loop_demo/events.jsonl
runs/member_a_feedback_loop_demo/summary.json
```

## Demo 场景

任务 DAG 包含两个任务：

1. `pressure`
   - subprocess worker。
   - sleep 一小段时间，确保 Runtime 可以采到 running 样本。
   - resource quota 包含 CPU、memory、pids。
   - demo manager 返回 cgroup-local memory PSI warning/critical 信号。

2. `followup`
   - 依赖 `pressure`。
   - 在下一轮 scheduler 中运行。
   - 该轮 scheduler 会消费上一轮 cgroup feedback。

## 为什么使用确定性 cgroup manager

普通开发环境通常没有写 `/sys/fs/cgroup` 和 `/sys/fs/bpf` 的权限。这个 demo 的目标是稳定展示 Runtime 闭环，所以：

- Runtime、subprocess worker、KernelSchedulerAdapter、EventLog 都是真实路径。
- cgroup-local PSI / usage 由 demo manager 确定性返回。
- 真实 cgroup v2 权限证明由 `scripts/verify_real_cgroup_v2.py` 单独完成。
- 真实 BPF map 权限证明由 `scripts/verify_real_bpf_map.py` 单独完成。

这避免了把“现场权限问题”误判成“Runtime 闭环不存在”。

## 验收字段

`summary.json` 的关键字段：

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

字段解释：

- `accepted=true`：demo 满足 P0 闭环验收。
- `local_sample_phases`：同一个 Runtime 流程中出现 initial、running、final 三阶段 cgroup-local pressure 样本。
- `feedback_applied`：Runtime 将 cgroup feedback 应用到 scheduler。
- `affected_decisions`：至少一个后续 `scheduler.decision` 使用 `pressure_source=system+cgroup`。
- `kernel_decision_published`：scheduler decision 被 KernelSchedulerAdapter 发布。
- `cgroup_removed`：worker 结束后 cgroup cleanup 路径被执行。

## 关键事件顺序

评委可以在 `events.jsonl` 中检查：

```text
worker.started
resource.cgroup_created
resource.cgroup_pid_added
resource.cgroup_quota_applied
resource.usage_sampled phase=initial
resource.pressure_sampled scope=agent_cgroup phase=initial
resource.usage_sampled phase=running
resource.pressure_sampled scope=agent_cgroup phase=running
resource.usage_sampled phase=final
resource.pressure_sampled scope=agent_cgroup phase=final
resource.cgroup_removed
resource.cgroup_feedback_applied pressure_source=system+cgroup
scheduler.decision pressure.pressure_source=system+cgroup
kernel.decision_published
```

## 和真实验证脚本的关系

| 文件 | 作用 |
| --- | --- |
| `scripts/demo_member_a_feedback_loop.py` | 无权限稳定展示 Runtime 闭环 |
| `scripts/verify_real_cgroup_v2.py` | 真实 cgroup v2 控制验收 |
| `scripts/verify_real_bpf_map.py` | 真实 bpftool pinned map 验收 |

答辩时建议先展示 demo 时间线，再展示两个 verifier 的 JSON 产物。
