# AgentOS sched_ext Stage 6 冻结报告

- Schema: `sched_ext_release.v1`
- 接管边界：partial mode 仅接管显式绑定的 Agent Worker。
- 性能主样本：阶段 4 五模式各 20 轮；Stage 6 增量样本不并入主样本。
- 可靠性：`380/380`，soak `3603.301s`，task loss `0`。

## 五模式正式指标

| 模式 | latency P50 (us) | latency P95 (us) | latency P99 (us) | benchmark iter/s | Jain fairness |
| --- | ---: | ---: | ---: | ---: | ---: |
| cfs | 3283.168 | 5686.908 | 6483.916 | 73274274.296 | 0.986764 |
| scx_simple | 195222.192 | 466899.084 | 519975.165 | 74110825.625 | 0.999948 |
| agentos_normal | 3217.415 | 6496.042 | 6841.325 | 72359532.292 | 0.999998 |
| agentos_static_3dsq | 1513.882 | 1828.781 | 2078.791 | 72562664.083 | 0.939108 |
| agentos_full_dag_psi_fairness | 1518.649 | 1839.635 | 2064.618 | 72705833.267 | 0.939094 |

## 能力边界

允许表述：AgentOS 基于 sched_ext partial mode 真实接管 Agent Worker，并在异常时回退默认调度。

禁止表述：替换 Linux 全部调度器、接管实时调度类、达到生产级。
