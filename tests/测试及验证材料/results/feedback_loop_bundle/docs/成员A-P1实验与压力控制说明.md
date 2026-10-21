# 成员A P1 实验与压力控制说明

## P1 交付内容

P1 阶段补齐三类能力：

1. 事件时间线生成器。
2. 开闭环消融实验。
3. 压力驱动调度策略显式化。

这三类能力的目标是让成员A的 OS-facing Runtime 不只是“代码存在”，而是能被评委通过事件、数据和可解释规则看懂。

## 事件时间线生成器

命令：

```bash
.venv/bin/python scripts/member_a_timeline.py \
  runs/member_a_feedback_loop_demo/events.jsonl \
  --json-out runs/member_a_timeline.json \
  --markdown-out docs/generated/member_a_timeline.md
```

输出：

- `runs/member_a_timeline.json`
- `docs/generated/member_a_timeline.md`

时间线会提取以下关键事件：

- `worker.started`
- `resource.cgroup_created`
- `resource.cgroup_pid_added`
- `resource.cgroup_quota_applied`
- `resource.usage_sampled`
- `resource.pressure_sampled`
- `resource.cgroup_feedback_applied`
- `scheduler.decision`
- `kernel.decision_published`
- `resource.cgroup_removed`

答辩时可以用 `docs/generated/member_a_timeline.md` 直接解释 Runtime 如何从 worker 启动走到 cgroup feedback，再进入下一轮 scheduler decision。

## 开闭环消融实验

命令：

```bash
.venv/bin/python scripts/run_member_a_ablation.py \
  --output-dir runs/member_a_ablation \
  --iterations 3
```

输出：

- `runs/member_a_ablation/summary.json`
- `runs/member_a_ablation/report.md`

实验模式：

| 模式 | 含义 |
| --- | --- |
| `dag_only` | 只运行用户态 DAG 调度，不进入 subprocess/cgroup observation |
| `observation_only` | 运行 subprocess/cgroup usage + PSI observation，但没有下一轮 feedback decision |
| `feedback_control` | 运行完整 cgroup feedback -> next scheduler decision 闭环 |
| `feedback_bpf_publish` | 在 feedback control 基础上记录 kernel decision publish 证据 |

核心比较：

- `observation_adds_usage_samples`
- `feedback_adds_cgroup_decisions`
- `bpf_publish_records_kernel_events`

这些比较适合在 PPT 中作为“为什么要做闭环”的证据。

## 压力驱动调度策略

`DAGScheduler` 现在显式记录 `pressure_control`：

```json
{
  "pressure_control": {
    "enabled": true,
    "memory_pressure_deferred_task_ids": ["high-memory"],
    "cpu_pressure_batch_limit": 1,
    "reasons": [
      "memory pressure deferred high-memory non-critical tasks",
      "cpu pressure limited scheduler batch width"
    ]
  }
}
```

当前显式规则：

1. 当 memory PSI `avg10 >= 20` 且本轮有低内存任务可运行时，推迟非 critical 的高内存任务。
2. 当 CPU PSI `avg10 >= 20` 且本轮 batch 大于 1 时，将本轮 batch 收窄到 1。

这两个规则让 feedback loop 的效果更容易解释：

- memory pressure 保护系统免于继续放入高内存任务。
- CPU pressure 通过收窄并发降低进一步争用。

## 验证命令

P1 相关测试：

```bash
.venv/bin/python -m pytest -q \
  tests/test_member_a_timeline.py \
  tests/test_member_a_ablation.py \
  tests/test_scheduler_pressure_control.py \
  tests/test_member_a_feedback_loop_demo.py \
  tests/test_pressure_feedback.py \
  tests/test_scheduler.py \
  tests/test_scheduler_safety.py \
  tests/test_kernel_adapter.py \
  tests/test_kernel_adapter_boundary.py \
  tests/test_bpftool_adapter.py \
  tests/test_bpf_schema.py
```

当前通过标准：

```text
207 passed
```
