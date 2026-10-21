# 成员A答辩证据包

- 生成时间: `2026-07-03T13:21:18.746075+00:00`
- 复制产物: `14`
- 缺失产物: `0`

## 这个证据包证明什么

- Runtime feedback loop 可以稳定复现：worker -> cgroup/usage/PSI -> feedback -> next scheduler decision。
- 开闭环消融实验可以区分 DAG only、observation、feedback control、BPF publish 证据。
- 时间线可以把 cgroup、PSI、scheduler decision、kernel publish 串成答辩可读链路。
- preflight 可以说明当前环境是否满足真实 cgroup v2 / bpftool pinned map 验收条件。

## 技术边界

- 已实现：cgroup v2 控制、PSI / usage 反馈、KernelSchedulerAdapter、mock BPF map、可选 bpftool pinned map 通路验证。
- 未实现 sched_ext/eBPF 调度接管，不能宣称替换 Linux 内核调度器。

## 产物清单

| 名称 | 状态 | 路径 | 字节 |
| --- | --- | --- | ---: |
| `preflight` | `copied` | `runs/member_a_preflight.json` | 1040 |
| `feedback_summary` | `copied` | `runs/member_a_feedback_loop_demo/summary.json` | 76884 |
| `feedback_events` | `copied` | `runs/member_a_feedback_loop_demo/events.jsonl` | 53192 |
| `ablation_summary` | `copied` | `runs/member_a_ablation/summary.json` | 5177 |
| `ablation_report` | `copied` | `runs/member_a_ablation/report.md` | 592 |
| `timeline_json` | `copied` | `runs/member_a_timeline.json` | 11255 |
| `timeline_markdown` | `copied` | `docs/generated/member_a_timeline.md` | 6408 |
| `real_cgroup_v2` | `copied` | `runs/member_a_real_cgroup_v2.json` | 609 |
| `real_bpf_map` | `copied` | `runs/member_a_real_bpf_map.json` | 344 |
| `boundary_doc` | `copied` | `docs/成员A-技术边界与答辩口径.md` | 4797 |
| `p15_report` | `copied` | `docs/成员A-P1.5收束报告.md` | 4821 |
| `kernel_report` | `copied` | `docs/成员A-真实内核通路验收报告.md` | 5065 |
| `feedback_doc` | `copied` | `docs/成员A-feedback-loop-demo.md` | 3445 |
| `p1_doc` | `copied` | `docs/成员A-P1实验与压力控制说明.md` | 3395 |

## 推荐答辩顺序

1. 先展示 `runs/member_a_preflight.json`，说明当前环境能力和缺口。
2. 展示 `docs/generated/member_a_timeline.md`，说明 Runtime feedback loop 事件链。
3. 展示 `runs/member_a_ablation/report.md`，说明 observation 与 feedback control 的差异。
4. 展示 `docs/成员A-技术边界与答辩口径.md`，主动说明真实 OS 能力和未宣称边界。
5. 如果目标 openEuler 环境已跑通 verifier，再展示 `runs/member_a_real_cgroup_v2.json` 和 `runs/member_a_real_bpf_map.json`。
