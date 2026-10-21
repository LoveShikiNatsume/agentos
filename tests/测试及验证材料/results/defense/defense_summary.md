# AgentOS-Runtime 答辩证据总结

- **生成时间**: 2026-06-24T13:37:12.876012+00:00
- **Backend**: cache-sim
- **总运行时间**: 1.029s

## 一、四类 Workload 压力测试

| 工作负载 | 基线 Cache 命中 | Runtime Cache 命中 | 基线任务完成 | Runtime 任务完成 | 故障恢复 |
|---------|---------------|-------------------|------------|----------------|---------|
| ai_inference | 58.6% | 76.2% | 3/7 | 7/7 | 恢复 |
| compile | 66.5% | 77.3% | 4/7 | 7/7 | 恢复 |
| compression | 71.6% | 82.4% | 4/8 | 7/8 | 恢复 |
| network | 59.8% | 78.1% | 3/7 | 7/7 | 恢复 |

## 二、Baseline vs Runtime 对比

- **Cache 命中率提升**: 12.9%
- **任务完成数**: 基线 14 → Runtime 28
- **故障恢复率**: 4/4

## 三、Token/Task 归一化指标

- **基线 Token/Task**: 177
- **Runtime Token/Task**: 166
- **Token 节省**: 11 tokens/task

## 四、ContextRef 与 MessageBus 通信证据

- **MessageBus 消息总数**: 70
- **Inline Payload**: 46196 bytes
- **Ref Payload**: 1726 bytes
- **通信节省**: 44470 bytes
- **Payload 压缩比**: 96.3%
- **ContextRef 引用节省总量**: 14904 bytes

## 五、sched_ext / eBPF / cgroup 探测

- **sched_ext**: unavailable
- **BPF Map 模式**: mock
- **cgroup v2**: 已挂载
- **cgroup v2 controllers**: cpuset, cpu, io, memory, hugetlb, pids, rdma, misc
- **Fallback 原因**: missing /sys/kernel/sched_ext; bpftool not found in PATH; missing CAP_BPF or unsupported kernel

> **注意**: sched_ext / eBPF 状态为运行时探测结果。当不可用时，系统自动 fallback 到用户态 mock adapter。
> 此处仅为 probe/fallback/mock adapter 模式，**未接管 Linux 内核调度**。

## 六、Backend 探测

- **Backend**: mock
- **可用**: 是

## 七、关键结论

1. **共享上下文池化** 将 Cache 命中率提升了 12.9%
2. **故障恢复** 在 4 个注入故障中恢复了 4 个
3. **任务完成率** 从基线 14 提升到 Runtime 28
4. **Token 效率** 每任务 token 从 177 降到 166
5. **通信开销量化** MessageBus payload 压缩比 96.3%，ContextRef 节省 14904 bytes
