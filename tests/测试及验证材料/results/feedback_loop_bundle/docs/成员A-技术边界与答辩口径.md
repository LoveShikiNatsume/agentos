# 成员A技术边界与答辩口径

## 一句话定位

成员A实现的是 Agent Runtime 的 OS-facing control plane：Runtime 将 Agent worker 纳入 cgroup v2 控制，采集 PSI / usage 形成反馈闭环，并通过 KernelSchedulerAdapter 连接用户态调度策略与 Linux 资源控制机制。

## 可以明确宣称的能力

| 能力 | 状态 | 严谨表述 |
| --- | --- | --- |
| DAG Scheduler | 已实现 | 用户态 DAG 调度器，支持依赖、资源、workload、fairness、pressure、safety/backpressure 信号 |
| cgroup v2 控制 | 已实现 | Runtime 可创建 cgroup，写入 CPU/内存/PID 配额，并 attach worker PID |
| PID membership 验证 | 已实现 | 真实 `/sys/fs/cgroup` 路径下 attach 后回查 `/proc/<pid>/cgroup` |
| cgroup usage | 已实现 | 读取 `cpu.stat`、`memory.current`、`memory.peak`、`pids.current` |
| PSI 反馈 | 已实现 | 优先读取 cgroup-local `cpu.pressure`、`memory.pressure`、`io.pressure`，不可用时回退系统 PSI |
| feedback loop | 已实现 | initial/running/final 样本进入 accumulator，下一轮 scheduler 使用 `system+cgroup` pressure |
| KernelSchedulerAdapter | 已实现 | Scheduler 和 OS-facing 操作解耦，Runtime 统一调用 adapter |
| mock BPF map | 已实现 | 无权限环境和 CI 的稳定降级通路 |
| bpftool pinned map | 已实现可选通路 | 可验证真实 BPF map create/update/lookup/dump 和固定二进制 ABI |

## 不能夸大的能力

| 不应宣称 | 正确说法 |
| --- | --- |
| 已实现 Linux 内核调度器 | 实现了 Agent Runtime 用户态调度器 |
| 已实现 sched_ext 调度器 | 当前只做 sched_ext/BPF 能力 probe 和 fallback，不接管调度 |
| 已实现 eBPF scheduler | 当前实现的是可选 pinned BPF map 数据通路验证 |
| BPF map 会改变 Linux 调度行为 | BPF map 用来发布/验证调度记录，不改变 CFS/sched_ext 策略 |
| mock BPF 就是真实内核能力 | mock 是无权限开发和 fallback；真实 BPF map 由 bpftool verifier 证明 |

## 推荐答辩表述

> 成员A负责 Agent Runtime 的调度与 OS 资源控制。我们把 Agent task 视为可管理的执行单元，由 Runtime 启动 subprocess worker，并通过 cgroup v2 对 CPU、内存和 PID 做真实限制。Runtime 在 worker initial、running、final 三个阶段采集 usage 和 cgroup-local PSI，再把压力反馈给下一轮 DAG Scheduler，影响后续调度策略和安全准入。为了保持策略层和机制层分离，我们设计了 KernelSchedulerAdapter，把 cgroup、PSI、BPF map 发布统一封装在内核适配层。BPF 部分保留 mock fallback，并支持可选 bpftool pinned map 验证；我们不把它夸大为 sched_ext/eBPF 调度器接管 Linux 调度。

## 被 OS 专家追问时的回答

**问：你们有没有真正改 Linux 调度器？**

答：没有。我们当前实现的是 Agent Runtime 层的用户态调度与 OS 资源控制。真实 OS 能力在 cgroup v2 和 PSI，BPF map 是数据通路验证，不宣称替换 CFS 或实现 sched_ext 调度器。

**问：cgroup 是真控制还是模拟？**

答：Runtime 的 cgroup manager 会真实写 `cpu.max`、`cpu.weight`、`memory.max`、`pids.max`，并将 worker PID 写入 `cgroup.procs`。在真实 `/sys/fs/cgroup` 路径下，还会回查 `/proc/<pid>/cgroup` 验证 membership。无权限 demo 使用临时 root，真实证明由 `verify_real_cgroup_v2.py` 生成 JSON。

**问：PSI 反馈真的影响调度吗？**

答：是。worker 运行中产生的 cgroup-local PSI 会进入 `CgroupFeedbackAccumulator`，下一轮调度前 Runtime 调用 `_apply_scheduler_pressure()`，将 system PSI 与 cgroup-local PSI 合并，写入 scheduler 的 pressure state。随后 `scheduler.decision` 中会出现 `pressure_source=system+cgroup` 和对应 `cgroup_local` 证据。

**问：BPF map 的意义是什么？**

答：它是 Runtime 调度决策向内核侧数据结构发布的可选通路。mock 模式保证 CI 和无权限环境稳定；bpftool 模式可以在有权限环境创建 pinned BPF map，并进行 update/lookup/dump 验证。它证明了 ABI 和数据通路，不表示内核调度行为被改变。

## 评委可检查的证据

- `scripts/verify_real_cgroup_v2.py`
- `scripts/verify_real_bpf_map.py`
- `scripts/demo_member_a_feedback_loop.py`
- `tests/test_member_a_feedback_loop_demo.py`
- `tests/test_kernel_adapter_boundary.py`
- `tests/test_pressure_feedback.py`
- `tests/test_bpftool_adapter.py`
- `tests/test_bpf_schema.py`

## 底线

所有展示材料都应坚持：

```text
真实控制：cgroup v2
真实观测：cgroup usage + PSI
真实闭环：cgroup feedback -> next scheduler decision
可选内核数据通路：bpftool pinned BPF map
不宣称：sched_ext/eBPF scheduler 接管 Linux 调度
```
