# 实验 C：失败后上下文恢复的单因素比较

本目录提供实验协议与桩验证，未运行真实模型。结论须由调度方审查代码、合并后用正式后端实验得出。

研究问题：同样的失败、校验器、反馈与最多两次尝试，重试前恢复任务起始私有上下文，对最终验收、调用次数和输入 Token 有何影响？

- `dirty`：`SemanticRecoveryPolicy(max_attempts=2, restore_context=False)`。
- `snapshot`：同一策略，唯一改变 `restore_context=True`，由 Runtime 的任务起始快照与 `ContextManager.import_agent_context(..., restore_shared=False)` 清除失败尝试新增的私有段。
- `no_retry`：最多一次尝试，作为不恢复的故障控制基准。每题首轮都被强制注入失败，因此无额外故障时此组验收完成率按设计为 0；它不是自然任务完成率估计。

每题只有一个 Runtime agent。按冻结 `required_refs` 顺序逐文档调用模型提取签名事实，真实写入 `ContextManager.add_private_ref`，最后一次模型调用根据私有事实回答。私有段由 Runtime 自动装入后续请求 system 内容，两组共用全部提示词。重试重新执行整段提取流程；反馈由 Runtime 传入，不包含标准答案。

为了隔离恢复变量，本实验给出所有必需文档，不进行实验 A 的检索策略或规划器比较。`manifest.json` 冻结实验 A 语料路径与 SHA256，复用已提交的 120 道标准库 AST 可判分题（probe 20 / formal 100），不复制 4.5 MB 语料；源清单改动时 loader 会拒绝继续。生成器按 split 用固定种子 20261019 洗牌，均衡分配一半执行、一半语义注入，冻结每题第一轮注入位置 K、错误事实与错误答案。

执行注入在 K 个事实段写入后抛异常；语义注入在 K 个事实段后追加错误的私有事实，再实际调用 answer 模型，将该次任务返回值覆写为冻结的错误答案，供 Runtime 校验器发现。模型原始 answer 仍完整保留在 calls.jsonl，覆写另有事件。这是显式、可重复的故障干预，不是声称模型自然产生了这些错误。

两组同样使用严格 JSON 与 Python 默认表达式 AST 规范化的标准答案校验。校验反馈只告知验收失败并要求重新读取源文档。由于 gold 校验器接受的结果才会标为 Runtime completed，本协议的准确率和验收完成率按设计一致；框架仍分别记录。若正式研究需要区分两者，应在另一个预注册协议中采用独立于 gold 的过程完成定义。

## 桩验证

在仓库根目录，使用指定虚拟环境（桩不访问 HTTP）：

```sh
export PYTHONPATH=src
PY=python
$PY -m experiments.semantic_recovery.run --stub --arm dirty --split all --output tests/results/semantic-c-smoke/dirty --concurrency 4 --thinking off
$PY -m experiments.semantic_recovery.run --stub --arm snapshot --split all --output tests/results/semantic-c-smoke/snapshot --concurrency 4 --thinking off
$PY -m experiments.semantic_recovery.run --stub --arm no_retry --split all --output tests/results/semantic-c-smoke/no_retry --concurrency 4 --thinking off
$PY -m experiments.semantic_recovery.analyze --input tests/results/semantic-c-smoke --output tests/results/semantic-c-smoke-analysis
$PY -m experiments.semantic_recovery.estimate --split formal
$PY -m pytest -q tests/test_semantic_recovery_experiment.py
```

输出目录须不存在，以防覆盖已运行的证据。每题保存 events.jsonl、calls.jsonl、summary.json；顶层 summary.json 含协议版本、manifest/提示词指纹、后端与配置。calls.jsonl 保存逐请求原始 usage、finish_reason、请求参数、发送消息、模型原始返回；max_tokens 默认 4096，最低 2048，隐藏传输重试为 0。测试阻断 socket.create_connection / OpenAIBackend.generate / HTTP request，验证桩不请求网络。

桩通过 AST 读取请求中的源文档，并以“相同符号保留最早私有事实”模拟可观察残留。该规则只是管道夹具，桩准确率、臂间差异及 char/4 Token 不能用来推断真实模型效果。

## 正式运行接口（交由调度方执行）

```sh
$PY -m experiments.semantic_recovery.run --base-url "$BASE_URL" --model "$MODEL" --arm dirty --split formal --output tests/results/semantic-c-formal/dirty --concurrency 4 --thinking off
# snapshot / no_retry 使用相同模型、seed、thinking、并发、max-tokens 配置与 formal split。
$PY -m experiments.semantic_recovery.analyze --input tests/results/semantic-c-formal --output tests/results/semantic-c-formal-analysis
```

支持与实验 A 相同的 `--base-url --model --arm --split --output --concurrency --thinking`，以及 `--stub --seed --max-tokens --manifest`。thinking 默认不向服务端发送开关；预注册后两组保持一致。seed 为请求层随机种子，不能保证所有后端完全确定性。正式运行应同一模型版本、服务配置、配额并交错/随机安排 arm 时间顺序，避免服务漂移。输出根目录应独立于版本库跟踪文件。

## 指标与分析

报告所有分配题（ITT）的准确率、验收完成率、每题调用次数与输入/输出 Token、尝试次数分布、截断调用/题数、语义注入检出比例（以计划注入与实际触达注入分别作分母），按 execution/semantic 注入类型分层。注入前若传输错误或截断阻断流程，题仍按分配类型进入 ITT，并单独记录 injection_reached，不能将未注入题当作校验器漏检。

比较 snapshot − dirty，按 question_id 进行配对 percentile bootstrap（默认 10000 次、95% CI），总体与两类注入分别计算。所有失败与截断题仍进入准确率/完成率配对，避免幸存者偏差。未知 usage 不填 0；仅 Token 配对排除任一臂 usage 未知的题，并报告覆盖及题号。截断单独计数，已有失败重试成功仍可验收完成。分析拒绝 manifest、提示词、split、题目集合/注入分配、可比模型配置不同或重复的 arm。

`estimate.py` 只读冻结清单，给出正常到达注入点且重试走完整流程时的请求预算；提前异常/截断可减少实际请求。输入是字符/4 加每个提取段 512 Token、反馈 64 Token 的明确假设，输出同时报告每调用 512 Token 的规划值与 calls × max_tokens 硬上限，不能当作实测。正式 token 使用 calls.jsonl 的 provider usage；服务端遗漏 usage 时保留未知。

## 协议限制

- 实验覆盖上下文段恢复，不回滚外部工具副作用；任务不调用工具。
- 每题都发生一次受控首轮失败，结果不能直接外推到自然故障率或全体 Agent 工作负载。
- 语义错误用固定错误事实和任务返回覆写注入；原始模型返回单独留档。
- gold 校验器只适用于这类可自动判分任务。真实应用需自行定义验收条件。
- 问题共享部分标准库文档；CI 的重采样单位仍是题目，不能消除共享文档相关性。
- 仅一轮重试，循环检测若提前终止会作为未完成记录；框架不额外绕过 Runtime 的终止机制。

## 本轮已验证产物与预算

2026-10-19 在隔离 worktree 中只用桩后端验证：

- 三臂全量 120 题（20 probe + 100 formal）：dirty 690 次、snapshot 690 次、no_retry 270 次，共 1650 次进程内桩调用；无 HTTP。
- 逐题与汇总证据位于 `tests/results/semantic-c-smoke/{dirty,snapshot,no_retry}/`；配对分析位于 `tests/results/semantic-c-smoke-analysis/analysis.{json,md}`（此 smoke 使用 1000 次 bootstrap，正式分析默认 10000）。
- `tests/test_semantic_recovery_experiment.py`：5 passed；包含禁止网络、真实 Runtime 恢复、首轮两组请求一致、dirty 私有残留实际进入请求、snapshot 清除、校验反馈、截断、未知 usage 以及配对配置拒绝。
- 正式预算产物为 `tests/results/semantic-c-formal-estimate.json`；重算命令为 `PYTHONPATH=src python -m experiments.semantic_recovery.estimate --split formal`。

formal 100 题，假设首轮到达冻结注入点且重试完整执行：

| Arm | 请求数 | 估计输入 Token | 输出规划（512/次） | 输出硬上限（4096/次） |
| --- | ---: | ---: | ---: | ---: |
| dirty | 573 | 1,656,941 | 293,376 | 2,347,008 |
| snapshot | 573 | 1,248,365 | 293,376 | 2,347,008 |
| no_retry | 223 | 470,045 | 114,176 | 913,408 |
| 合计 | 1,369 | 3,375,351 | 700,928 | 5,607,424 |

每组实际请求可能因注入点之前的服务错误或截断提前减少，离线估计区间分别为 100–573、100–573、100–223。输入值采用字符/4、每个私有提取段 512 Token、反馈 64 Token 的代理假设，不是测量或硬上限；输出硬上限受 max_tokens 控制，实际正式用量须读取服务端 usage。此处请求量也不含调度方另行执行的 probe、重复试验或其他模型。
