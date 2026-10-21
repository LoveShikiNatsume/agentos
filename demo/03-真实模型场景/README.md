# 真实模型场景

## 1 目的

本层在真实模型接口下运行 AgentOS Runtime，用以说明运行时在实际模型调用、上层框架接入和多 Agent 应用场景中的行为。运行需要云端模型接口凭据或本地 vLLM 服务。凭据仅通过环境变量读取，不写入配置文件和仓库。

## 2 运行入口

以下命令均在仓库根目录执行，并假定已按根目录 `README.md` 完成安装：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e "./src[dev,app,langgraph]"
```

### 2.1 真实接口下的功能演示

该入口与 [01-离线功能演示](../01-离线功能演示/) 使用同一场景和同一组检查项，区别在于模型请求发往 DeepSeek 接口，Token 用量取自接口返回的 usage 字段。

```bash
export DEEPSEEK_API_KEY=<凭据>
.venv/bin/agentctl finals-demo --backend deepseek \
  --output demo/output/finals_deepseek
```

### 2.2 经 LangGraph 接入的仓库审阅

该场景说明上层框架接入运行时的方式。预先定义的审阅计划被转换为三个 Runtime 任务：两个审阅任务由运行时并发启动，各自由一个 LangGraph StateGraph 完成“计划—检查—回答”三个步骤；模型调用经过上下文管理器和模型网关，源码检查经过运行时的工具管理器；两个审阅任务的结果以上下文引用的形式发布，并通过消息总线通知汇总任务。

```bash
# 不需要模型服务的链路检查
.venv/bin/python -m agentos.scenarios.repository_review \
  --output demo/output/review_mock

# 本地 vLLM 服务
.venv/bin/python -m agentos.scenarios.repository_review \
  --backend vllm --base-url http://127.0.0.1:8000 --model <模型名> \
  --output demo/output/review_vllm
```

输出目录中的 `summary.json` 给出八项检查的结果，包括三个 Runtime 任务全部完成、观察到 LangGraph 内部步骤、工具调用经过运行时、六次模型调用全部经过网关等。

### 2.3 软件工程场景与实时面板

软件工程场景由 11 个任务组成，覆盖架构设计、数据库与接口设计、后端实现、测试、安全评审、文档、构建、静态检查、集成与部署。下列命令在运行场景的同时启动实时事件面板，并默认注入一次故障以展示恢复过程，`--no-fault` 可关闭故障注入：

```bash
export DEEPSEEK_API_KEY=<凭据>
.venv/bin/agentctl live --scenario software_eng --backend deepseek \
  --log demo/output/software_eng/events.jsonl
```

浏览器访问 `http://127.0.0.1:9400` 查看面板。将 `--backend` 改为 `mock` 可在没有凭据的条件下观察同一任务图的调度过程。`--scenario` 另有 `devops` 和 `research` 两个场景可选。

### 2.4 工作台与自然语言任务

工作台和 `agentctl exec` 接收以自然语言描述的仓库任务，由运行时组织多个 Agent 完成。模型、接口地址和执行限制由任务目录下的 `.agentos.toml` 给出，配置样例见 [`../examples/agentctl/deepseek-high.toml`](../examples/agentctl/deepseek-high.toml)。

```bash
cp demo/examples/agentctl/deepseek-high.toml /path/to/task/.agentos.toml
export DEEPSEEK_API_KEY=<凭据>

# 工作台
.venv/bin/agentctl app --cwd /path/to/task

# 单个任务
.venv/bin/agentctl exec --cwd /path/to/task "<任务描述>"

# 批量题单
.venv/bin/agentctl bench run --suite demo/examples/agentctl/tiny.json
```

样例配置启用了 bubblewrap 命令沙箱，运行环境须安装 bubblewrap 并允许用户命名空间。

## 3 归档结果

`结果/finals_deepseek_20260907/` 保存 2.1 节入口在 2026-09-07 的一次运行，模型为 `deepseek-chat`，`summary.json` 中的判定为 `passed: true`，`token_source` 为 `provider_usage`。

| 指标 | 全部共享材料 | 按任务选择 |
| --- | ---: | ---: |
| 模型调用次数 | 3 | 3 |
| 输入字符数 | 6396 | 2782 |
| 输入 Token 数 | 1266 | 552 |
| 缓存命中 Token 数 | 768 | 0 |

故障隔离与重试恢复的各项检查均通过，判定结果与离线演示一致。`requests.jsonl` 保存了实际发送的消息、回复和接口返回的 usage，可逐条核对。为便于在其他机器上查看，`dashboard.html` 中显示的事件日志路径已改为相对路径，其余内容与原始运行一致。

更大规模的真实模型实验不在本目录重复存放，其原始记录位于 `tests/测试及验证材料/results/`：

| 目录 | 内容 |
| --- | --- |
| `swe_bench_verified_30/` | 2026-07-09 的 SWE-bench Verified 30 任务批次，四种上层框架在接入运行时前后的 Token、缓存与故障恢复指标 |
| `workload_matrix_deepseek_real/` | 2026-06-20 的四类负载矩阵，DeepSeek 接口，含基线与运行时两组事件流 |
| `framework_benchmark_4fw_real/` | 四框架试运行的汇总与逐任务数据，共 10 个任务 |

## 4 结论范围

2.1 节的输入为合成材料，共六次模型调用，属于功能演示，不是性能基准或质量评测。两组请求按固定顺序执行，前缀长度和缓存预热状态不同，表中缓存命中 Token 数的差异不能用于比较缓存策略。2.2 节验证的是调用链路，不包含自主生成补丁。各批次归档实验所用的模型、日期和 Token 来源不同，引用时应以对应目录中的元数据为准。
