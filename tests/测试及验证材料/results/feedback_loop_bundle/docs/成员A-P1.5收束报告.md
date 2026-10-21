# 成员A P1.5 收束报告

## 目标

P1.5 的目标不是继续扩大功能，而是把 P0/P1 已实现能力收束成可验证、可答辩、可迁移到真实 openEuler 环境的证据链。

本阶段覆盖：

- 全量相关验证。
- P0/P1 产物检查。
- 真实环境 preflight。
- 当前环境限制记录。

## 验证矩阵

推荐验证命令：

```bash
.venv/bin/python -m pytest -q \
  tests/test_member_a_feedback_loop_demo.py \
  tests/test_member_a_timeline.py \
  tests/test_member_a_ablation.py \
  tests/test_scheduler_pressure_control.py \
  tests/test_pressure_feedback.py \
  tests/test_runtime_worker.py \
  tests/test_scheduler.py \
  tests/test_scheduler_safety.py \
  tests/test_kernel_adapter.py \
  tests/test_kernel_adapter_boundary.py \
  tests/test_bpftool_adapter.py \
  tests/test_bpf_schema.py \
  tests/test_verify_real_cgroup_v2.py \
  tests/test_verify_real_bpf_map.py \
  tests/test_member_a_preflight.py
```

当前验收标准：

```text
相关测试全部通过，无失败。
```

## 已确认产物

P0/P1 核心产物：

- `runs/member_a_feedback_loop_demo/summary.json`
- `runs/member_a_feedback_loop_demo/events.jsonl`
- `runs/member_a_ablation/summary.json`
- `runs/member_a_ablation/report.md`
- `runs/member_a_timeline.json`
- `docs/generated/member_a_timeline.md`
- `runs/member_a_preflight.json`
- `runs/member_a_real_cgroup_v2.json`
- `runs/member_a_real_bpf_map.json`

文档产物：

- `docs/成员A-真实内核通路验收报告.md`
- `docs/成员A-技术边界与答辩口径.md`
- `docs/成员A-feedback-loop-demo.md`
- `docs/成员A-P1实验与压力控制说明.md`
- `docs/成员A-P1.5收束报告.md`

## 当前 preflight 结果

当前环境：

```json
{
  "os": {
    "id": "debian",
    "kernel": "6.1.0-35-amd64",
    "pretty_name": "Debian GNU/Linux 12 (bookworm)"
  },
  "is_openeuler": false,
  "recommended_bpf_mode": "auto",
  "fallback_expected": true
}
```

当前可用能力：

- cgroup v2 hierarchy 存在。
- PSI 文件存在：`cpu`、`memory`、`io`。
- bpffs 已挂载在 `/sys/fs/bpf`。

当前缺失能力：

- cgroup delegation 不可用。
- `/sys/fs/cgroup` 当前不可写。
- `bpftool` 不在 PATH。
- 当前进程没有 `CAP_BPF`。

因此当前环境适合运行 mock/auto fallback demo，不适合作为真实 cgroup v2 或真实 pinned BPF map 的最终验收环境。

## 真实 verifier 状态

当前 cgroup verifier 输出：

```json
{
  "available": false,
  "delegation_mode": "host_root",
  "skip_reason": "delegation=host_root; set AGENTOS_CGROUP_ALLOW_ROOT=1 to override"
}
```

当前 BPF verifier 输出：

```json
{
  "available": false,
  "skip_reason": "bpftool not found in PATH"
}
```

这两个结果是环境 skip，不是 Runtime 代码路径失败。国赛前需要在 openEuler 或等价特权环境重跑。

## openEuler 目标环境要求

为了得到最终真实验收 JSON，目标环境建议满足：

- openEuler 或等价 Linux 发行版。
- cgroup v2 mounted。
- PSI enabled，存在 `/proc/pressure/cpu`、`/proc/pressure/memory`、`/proc/pressure/io`。
- 具备安全 cgroup delegation，或容器内拥有可写 delegated cgroup root。
- `/sys/fs/bpf` 为 bpffs。
- 安装 `bpftool`。
- 如需真实 BPF map create/update/lookup/dump，运行用户具备所需 BPF 权限，例如 `CAP_BPF` 或等价 root 权限。

## 推荐现场流程

1. 运行 preflight：

```bash
.venv/bin/python scripts/openeuler_member_a_preflight.py \
  --json-out runs/member_a_preflight.json
```

2. 如果 preflight 显示 cgroup 可用，运行：

```bash
.venv/bin/python scripts/verify_real_cgroup_v2.py \
  --json-out runs/member_a_real_cgroup_v2.json
```

3. 如果 preflight 显示 bpffs、bpftool、CAP_BPF 可用，运行：

```bash
.venv/bin/python scripts/verify_real_bpf_map.py \
  --json-out runs/member_a_real_bpf_map.json
```

4. 无论真实权限是否完整，都可以运行 deterministic feedback demo：

```bash
.venv/bin/python scripts/demo_member_a_feedback_loop.py \
  --output-dir runs/member_a_feedback_loop_demo
```

5. 生成时间线：

```bash
.venv/bin/python scripts/member_a_timeline.py \
  runs/member_a_feedback_loop_demo/events.jsonl \
  --json-out runs/member_a_timeline.json \
  --markdown-out docs/generated/member_a_timeline.md
```

## 答辩结论口径

当前本机环境只能证明：

- Runtime feedback loop 可稳定运行。
- 事件时间线完整。
- 消融实验能区分 observation 与 feedback control。
- preflight 能准确报告真实环境缺口。

不能把当前本机结果表述为：

- 真实 cgroup v2 verifier 已通过。
- 真实 bpftool pinned map verifier 已通过。
- sched_ext/eBPF 调度器已接管 Linux 调度。

最终国赛材料应以 openEuler 目标环境重新生成的 verifier JSON 为准。
