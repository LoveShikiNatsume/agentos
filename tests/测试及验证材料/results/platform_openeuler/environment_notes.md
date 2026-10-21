# openEuler / cgroup / sched_ext 环境验证说明

## 当前环境

- **OS**: Debian GNU/Linux 12 (bookworm)
- **Kernel**: 6.1.0-35-amd64
- **非 openEuler 环境**

## 探测结果

| 项目 | 状态 | 说明 |
|------|------|------|
| is_openeuler | false | 当前为 Debian，非 openEuler |
| openeuler_version | (空) | 非 openEuler 不填版本 |
| cgroup_v2_mounted | true | cgroup v2 已挂载 |
| cgroup_v2_path | /sys/fs/cgroup | 标准路径 |
| cgroup_v2_controllers | cpuset, cpu, io, memory, hugetlb, pids, rdma, misc | 8 个控制器可用 |
| sched_ext_available | false | Debian 6.1 内核无 sched_ext |
| bpf_available | true | /sys/fs/bpf 存在 |
| bpftool_available | false | 未安装 bpftool |
| cap_bpf | false | 无 CAP_BPF 权限 |
| bpf_map_mode | mock | 自动 fallback 到用户态 mock adapter |

## 已验证能力

1. **cgroup v2 检测**：正确识别挂载状态和 8 个 controllers
2. **sched_ext probe**：正确报告不可用，给出清晰 fallback reason
3. **BPF 检测**：区分 /sys/fs/bpf 存在 vs bpftool/CAP_BPF 不可用
4. **非 openEuler 保护**：不会误报 openEuler 版本号

## openEuler 实测命令（待真实环境）

```bash
# 在 openEuler 容器或虚拟机中执行
docker run --rm -it --privileged openeuler/openeuler:22.03-lts-sp3 bash

# 安装依赖后执行
uv run python scripts/probe_sched_ext.py --out runs/defense_openeuler/probe_openeuler_real.json
```

预期在 openEuler 22.03 LTS SP3 + 6.6+ 内核下：
- `is_openeuler: true`
- `openeuler_version: "22.03"`
- `sched_ext_available`: 取决于内核配置
- `cgroup_v2_controllers`: 应包含 cpu, memory, io, pids

## 保守表述

- 当前仅完成 **probe/fallback 验证**
- sched_ext/eBPF 为运行时能力探测，**未接管 Linux 内核调度**
- 不可用时自动 fallback 到用户态 mock adapter
