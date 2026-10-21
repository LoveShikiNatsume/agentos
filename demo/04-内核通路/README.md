# 内核通路

## 1 目的

本层说明 AgentOS Runtime 与 Linux 内核调度的衔接：运行时将 Agent Worker 的执行身份映射为内核调度策略，由基于 sched_ext 的 eBPF 调度器 `agentos_scx` 执行。该通路依赖指定版本的内核和 root 权限，在通用开发机和容器中无法复现，因此本目录只给出运行条件、入口与归档结果的位置，不提供可在任意环境直接运行的脚本。

## 2 运行条件

| 项目 | 要求 |
| --- | --- |
| 内核 | Linux 6.12.y 稳定分支，提交 `296aabce459470a4c1b68ffd0c0c0920e563aaad`；归档验证所用内核版本为 `6.12.95-agentos-scx` |
| 内核源码 | 上述提交的完整源码树，且已构建 `vmlinux` |
| 工具链 | gcc、clang、make、libelf、zlib |
| 权限 | root，用于加载 BPF 调度器和写入 cgroup v2 |
| CPU | 不少于 4 个逻辑 CPU，正式验收按 CPU 0 至 3 划分控制、Worker 与背景压力 |

`agentos_scx` 为实验性调度器，不应在生产主机上加载。sched_ext 不属于 openEuler 24.03 默认内核的能力，验证环境须另行安装上述内核。

## 3 入口

### 3.1 构建

```bash
make -C src/native/sched_ext \
  KERNEL_SRC=/opt/kernel-src/linux-6.12.y all
```

构建过程会校验内核源码的提交和 `vmlinux`，不符合要求时终止。内核源码与生成的二进制文件不随仓库分发。

### 3.2 验证

| 阶段 | 脚本 | 验证内容 |
| --- | --- | --- |
| 接管 | `src/scripts/verify_real_sched_ext.py` | 加载调度器，以 partial mode 接管 100 个 Worker，检查 CPU 亲和性、信号清理与内核日志 |
| 策略绑定 | `src/scripts/verify_sched_ext_runtime.py` | 运行真实的子进程任务图，核对任务到进程的策略写入与回读 |
| 策略效果 | `src/scripts/verify_sched_ext_policy_effects.py` | 核对三类调度队列、权重、时间片及各类回退 |
| 对比实验 | `src/scripts/run_sched_ext_stage4_ablation.py` | 五种调度模式的对比 |
| 归档校验 | `src/scripts/verify_sched_ext_stage6_release.py` | 对冻结归档进行离线一致性校验 |

各脚本的参数、执行顺序和验收条件见 [`src/native/sched_ext/README.md`](../../src/native/sched_ext/README.md)。

cgroup v2 资源控制与 PSI 压力采样不依赖 sched_ext，可在具备特权的 openEuler 容器中验证，入口见 [02-国产系统部署](../02-国产系统部署/)。

## 4 归档结果

验证在安装 `6.12.95-agentos-scx` 内核的 openEuler 环境中完成，结果位于 `tests/测试及验证材料/results/sched_ext_stage6/`，对应源码提交 `66910d96`，于 2026-07-18 冻结。

| 文件 | 内容 |
| --- | --- |
| `report.md` | 冻结报告，含五种调度模式的正式指标 |
| `takeover.json` | 接管与策略绑定阶段的验收记录 |
| `benchmark_summary.json`、`benchmark_raw.json` | 对比实验的汇总与原始数据 |
| `fallback.json` | 回退与可靠性验证记录 |
| `strict_verification.json` | 归档一致性校验的 14 项检查，全部通过 |
| `SHA256SUMS` 与 `.tar.gz` 归档包 | 完整证据包及其校验和 |

归档给出的主要结果如下：

- 接管：partial mode 下 100 个 Worker 全部进入 `SCHED_EXT`，未显式绑定的进程不受影响，调度器退出后内核状态恢复为 `disabled`。
- 尾延迟：在混合 CPU 负载的对比实验中，每种模式取 20 轮样本，三类调度队列策略的延迟 P95 为 1839.6 µs，CFS 为 5686.9 µs。
- 吞吐与公平性：同一实验中基准吞吐较 CFS 低约 0.8%，Jain 公平性指数为 0.939，CFS 为 0.987。
- 可靠性：380 项回退与恢复用例全部通过；持续 3603.3 s 的长时运行中任务丢失数为 0。

## 5 结论范围

上述结果支持的表述是：AgentOS Runtime 基于 sched_ext partial mode 接管显式绑定的 Agent Worker，并在调度器、心跳或策略映射失效时回退至默认调度。结果不支持下列表述：替换 Linux 的全部调度器；接管实时调度类；全部指标优于 CFS；达到生产级。对比实验中的吞吐为基准程序的迭代速率，不是模型请求吞吐。
