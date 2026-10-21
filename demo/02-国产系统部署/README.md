# 国产系统部署

## 1 目的

本层说明 AgentOS Runtime 在国内开源操作系统上的构建、安装与运行方式。仓库为 openEuler 24.03 LTS 和 openKylin 各提供一份容器构建文件：构建阶段在对应发行版的用户空间内安装系统依赖与本项目，容器启动后执行全量回归测试。针对 openEuler，另提供 cgroup v2 资源控制的验证脚本。

## 2 运行条件

| 项目 | 要求 |
| --- | --- |
| 容器运行时 | Docker |
| 网络 | 构建阶段需访问发行版软件源与 PyPI |
| 权限 | cgroup v2 验证需以 `--privileged` 启动容器 |

## 3 运行入口

以下命令均在仓库根目录执行。

### 3.1 openEuler

```bash
bash src/scripts/openeuler-verify.sh
```

该脚本依次完成四个步骤：

1. 依据 `src/Dockerfile.openeuler` 构建镜像，基础镜像为 `openeuler/openeuler:24.03-lts`；
2. 检查容器内 Python 版本不低于 3.11；
3. 在容器内执行全量回归测试；
4. 以特权模式运行 `src/scripts/openeuler-cgroup-verify.py`，在 cgroup v2 上创建资源组，写入并回读资源配额，读取用量统计，最后清理资源组。

各步骤亦可单独执行：

```bash
docker build -f src/Dockerfile.openeuler -t agentos-openeuler .
docker run --rm agentos-openeuler
docker run --rm --privileged agentos-openeuler \
  python3 src/scripts/openeuler-cgroup-verify.py
```

### 3.2 openKylin

```bash
docker build -f src/Dockerfile.openkylin -t agentos-openkylin .
docker run --rm agentos-openkylin
```

基础镜像为 `openkylin/openkylin:latest`。两个镜像的默认启动命令均为全量回归测试。

## 4 归档结果

平台验证的原始记录位于 `tests/测试及验证材料/results/`，本目录不重复存放。

| 平台 | 归档位置 | 内容 |
| --- | --- | --- |
| openKylin | `platform_openkylin/` | 2026-07-11 的镜像拉取、构建与测试日志；该批次测试结果为 1331 项通过 |
| openEuler | `sched_ext_stage6/pytest_results.json` | 源码提交 `66910d96` 在 openEuler 与 Debian 两个环境中的全量测试记录，退出码均为 0 |
| openEuler | `kernel_probe/` | 2026-07-03 在 openEuler 24.03 LTS 容器内对 cgroup v2 资源控制、cgroup 级 PSI 采样和 pinned BPF map 读写的验证记录 |

归档记录各自注明了日期与源码提交。在其他提交上重新运行时，测试项数量会随测试集的增补而变化，应以实际输出为准。

## 5 结论范围

容器与宿主机共享内核，因此容器内的构建与测试通过说明的是本项目对相应发行版用户空间（软件源、Python 版本、系统工具）的适配，不说明该发行版内核上的调度能力。与内核相关的验证见 [04-内核通路](../04-内核通路/)。
