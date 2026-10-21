# 离线功能演示

## 1 目的

本演示在不依赖模型服务凭据和特定内核的条件下，验证 AgentOS Runtime 的四项用户态功能：多 Agent 任务的依赖执行、按任务选择上下文、单分支故障隔离以及重试恢复。演示的输入材料固定，模型后端为 mock，输出由校验器逐项检查，因而同一命令在不同机器上应得到一致的判定结果。

## 2 验证内容

| 验证内容 | 代码位置 | 对应测试 |
| --- | --- | --- |
| 多 Agent 任务的依赖关系与执行结果 | `src/agentos/core/task.py`、`src/agentos/runtime/engine.py` | `tests/test_finals_demo.py` |
| 按任务选择上下文，并记录实际发送的请求 | `src/agentos/context/`、`src/agentos/scenarios/finals.py` | `tests/test_finals_demo.py`、`tests/test_context_ref.py` |
| 单分支故障隔离与基于上下文的重试恢复 | `src/agentos/fault/` | `tests/test_finals_runtime.py`、`tests/test_recovery.py` |
| 事件日志与离线事件面板 | `src/agentos/events/`、`src/agentos/dashboard/` | `tests/test_event_log.py`、`tests/test_dashboard.py` |

演示场景定义于 `src/agentos/scenarios/finals.py`，输入为合成的发布评审材料，共产生六次模型请求。

## 3 文件

| 文件 | 说明 |
| --- | --- |
| `run_offline_validation.sh` | 启动一次全新的离线演示，并在结束后调用校验器 |
| `verify_offline_bundle.py` | 检查输出目录的文件完整性、JSON 与 JSONL 结构以及数据来源标记 |
| `manifest.json` | 以机器可读形式记录验证范围、代码位置、测试入口与结论范围 |
| `结果/` | 随仓库归档的一次完整运行结果 |

## 4 运行方法

在仓库根目录安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e "./src[dev]"
```

执行离线演示：

```bash
bash demo/01-离线功能演示/run_offline_validation.sh
```

输出默认写入 `demo/output/离线功能演示/`，该路径已被 Git 忽略。输出目录也可由参数指定：

```bash
bash demo/01-离线功能演示/run_offline_validation.sh demo/output/my_run
```

为避免不同批次的结果相互混杂，脚本拒绝写入已存在的目录。

## 5 输出与判定

一次成功的运行在输出目录中生成下列文件：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 整体判定、运行环境、输入类型及各项功能检查的结果 |
| `report.md` | 供人工阅读的结果说明 |
| `requests.jsonl` | 六次模型请求及其回复 |
| `events.jsonl` | 带有 `demo_case` 标记的 Runtime 事件流 |
| `dashboard.html` | 由事件流生成的离线事件面板 |

校验器的通过条件为：`summary.passed` 为 `true`；后端为 `mock`；输入标记为合成材料；`all_shared`、`selected`、`isolation`、`retry` 四类演示事件均有记录。

## 6 归档结果

`结果/` 保存 2026-09-28 的一次运行，`summary.json` 中的判定为 `passed: true`。该批次的主要结果如下：

- 上下文选择：全部共享材料时输入为 6396 字符，按任务选择后为 2782 字符，减少 56.5%。
- 故障隔离：注入故障的可选任务失败，依赖该任务的分支被跳过，与其无依赖关系的评审任务和报告任务正常完成。
- 重试恢复：失败尝试的私有内容被清除，并发写入的共享更新得到保留，其他任务的私有内容保持隔离，三项检查均为 `true`。

上述结论可由 `events.jsonl`、`requests.jsonl` 和 `dashboard.html` 逐项核对。

## 7 结论范围

本演示使用 mock 后端和合成输入，其结果仅说明功能链路的正确性。字符数的减少反映的是该合成案例中材料选择的效果，不构成 Token 成本或性能方面的结论；本演示同样不涉及真实模型的输出质量、SWE-bench 通过率和内核调度。真实模型接口下的运行见 [03-真实模型场景](../03-真实模型场景/)，内核通路的验证见 [04-内核通路](../04-内核通路/)。
