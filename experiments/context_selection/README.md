# 实验 A：非预设的按需上下文装配

本框架检验：在 AgentOS 的 `Runtime.run` 主链中，由模型生成的声明和系统选择器决定原文装配时，相对广播全部候选文档可减少多少输入，以及准确率变化。实验侧实现位于本目录；未修改 Runtime 核心。

## 冻结语料与题集

`manifest.json` 由服务器 `/usr/lib/python3.11` 的真实源码生成，seed 为 `20261019`。共 120 题，前 20 题为 `probe`，后 100 题为 `formal`；两个集合各有一半题涉及两个文档，另一半涉及三个文档。每题包含八个候选，干扰项按同模块、同定义类型及名称词重合排序，随机种子打破同分。

文档按完整函数、类或类成员边界切分，相邻完整定义可合并；超过长度上限的单个函数被舍弃。584 个使用中的文档保留原始文本，估算长度为 1001–2997 Token。估算方法为 `ceil(Unicode 字符数 / 4)`，不代表模型 tokenizer 的实测 Token。manifest 包含源文件相对路径、SHA256、LF 行号、字符数、估算方法、符号名称和定义作用域。

题目比较两个或三个不同文档中函数或方法的形参数量，要求输出唯一最多者、它的第一个带默认值的参数名及默认值表达式。计数包含仅限位置参数、普通参数、仅限关键字参数、`*args` 和 `**kwargs`，方法中的 `self`、`cls` 排除；按源码签名顺序确定第一个默认值。答案为严格 JSON：

```json
{"winner":"module.Class.method","parameter":"name","default":"Python expression"}
```

标准答案由 AST 生成，默认表达式经过 `ast.unparse`。判分要求字段恰好为上述三个字符串，拒绝重复键、额外字段、Markdown 包装和解释文字；标识符大小写敏感。表达式仅解析与重新输出，不执行，空白、引号及冗余括号归一化后精确匹配。全部冻结文档可解析，全部 120 个答案有从冻结文本重新推导的测试。

源码文件的原始缩进被保留。非空 `definition_scope` 的文档在测试中临时包入类作用域，避免 `dedent` 受多行字符串内部零缩进文本影响。源码行号按 LF 切分，避免 formfeed 被误当作换行。

manifest 文件字节 SHA256：
`249633df6696d2fd157c440ac1fbc7b24b36dec91998738a03a00c58401966c5`。
日志中的 `manifest_sha256` 是排序、紧凑 JSON 的语义 SHA256，值为
`2b2c04222aa8875ecde007089d2311987244051c19af2969e571c58631d06d85`；两者计算口径不同。

## 工作流与单因素协议

1. planner 通过一次单节点 `TaskDAG` 和 `Runtime.run` 执行。只接收题面、文档 ID、名称与一行符号目录，不接收原文、标准答案或所需文档标签。
2. planner 输出 1–4 个独立 reader，每个声明 `id`、`goal`、`context_refs`。校验候选引用、唯一 ID、字段及类型，格式错误仅纠正一次；截断或传输错误不进行格式重试。
3. 计划、原始请求、纠正记录及配置写入冻结 bundle。bundle 绑定 manifest、各题计划和全部角色提示词/文档包装模板的指纹。运行组必须传入 `--plans`，不允许静默重新规划或覆盖已有输出。
4. 各组每题每轮构建一个 `TaskDAG`：冻结 planner 节点仅重放计划，不调用模型；reader 依赖该节点，成功返回后由 `adapt_plan` 发布结果引用；answerer 依赖全部 reader，并读取它们的结果。
5. 四组只改变文档引用。提示词、文档包装与顺序、工具集合（空）、调度配置、温度 0、每轮 seed、输出上限相同。所有调用经过 `ctx.llm_call`、ContextManager 和 RecordedBackend。统一 `prefix_order=name`，关闭压缩、文档预算及缓存反馈，不改变宿主模型服务缓存配置。

| 组 | reader 原文 | answerer 原文 |
| --- | --- | --- |
| all | 八个候选全部广播 | 八个候选全部广播 |
| declared | planner 为该 reader 声明的文档 | 无，仅汇总 reader 结果 |
| bm25 | 以该 reader 的 goal 检索候选，取前 k 个 | 无，仅汇总 reader 结果 |
| oracle | 该题全部所需文档 | 该题全部所需文档 |

answerer 为固定汇总节点，planner 只声明 reader 的原文需求。这一定义允许比较广播原文与按角色装配，也避免 answerer 原文补回 BM25 检索遗漏。`oracle` 仅为所需文档已知时的参照，仍可能因 reader 抽取失败而答错，不表示质量必然达到 100%。

BM25 使用候选名称及原文，ASCII 字母/数字分词，拆分下划线，`k1=1.5,b=0.75`，分数相同按文档 ID 排序。对所选 split 的所有成功冻结 reader，设声明引用总数 D、reader 数 R。每个 reader 使用 `floor(D/R)` 个文档，再按 seed、题号和 reader ID 的 SHA256 排序，为 `D % R` 个 reader 增加一个名额。因此平均 k 精确等于 declared 平均声明数量；不使用标准答案、所需标签或任何组的输出调节 k。`bm25_budget` 保存具体规则与均值。

所有任务显式包含控制引用 `control`，空文档声明不会触发 Runtime 的“空列表回退全部共享上下文”。reader 不读取其他 reader 的输出；全部组的 answerer 都读取相同一组结果引用。原文包装标明源模块、路径和符号，以便解释完整限定名。

默认每题重复三次，seed 为基准 seed 加轮号 0、1、2，各组相同轮次使用同一 seed；计划只冻结一次。温度为 0，这些重复主要检验执行稳定性，不视作独立题目。真实后端必须支持发送的 seed 参数；请求不兼容会明确失败，不进行无 seed 回退。

输出上限默认 `4096`，CLI 禁止低于 `2048`；传输重试与任务恢复均为零。建议先用 probe 检查格式与截断，再确定正式协议；正式集开始后保持模板、参数、题集与计划冻结。

## 指标与统计

每个实际请求在 `calls.jsonl` 中保存完整 messages、实际选择引用、请求参数（含 seed）、原始 usage、校验后指标、完整响应、finish_reason、状态和耗时。`duration_seconds` 包含并发门等待；metrics 中的 latency 是后端/HTTP 调用时间。cached Token 使用核心的响应计数校验，缺失或不一致保留为 null，禁止当作实测零。

每题每轮记录精确判分、执行输入/输出/cached Token、模型请求数、实际发送文档、所需文档召回率、是否完整覆盖及失败原因。召回率取实际模型请求原文的并集；另记录 reader 阶段召回率。planner 与 answerer 汇总文本不当作原文召回。非法答案 JSON 判错并记录 `answer_format`，仍为有效执行。

任何 planner、reader 或 answerer 请求 `finish_reason=length` 都标记该题无效。分析时任一轮截断会从该题的所有质量均值和配对推断中剔除整题，保留题号、轮次及原因；planner 截断继承至各轮的有效性，但其请求数和成本只计算一次。传输或 Runtime 失败保留于非截断的 ITT 准确率分母并计错，同时报告有效子集，防止通过删除失败题提高质量。

分析先在题内平均重复，再对题号做固定 seed `20260920`、10000 次配对 percentile bootstrap，输出相对 all 的准确率差、输入 Token 差、相对输入变化与 95% CI。配对主比较使用两组共同有效题，并单列覆盖率；Token 比较另要求 usage 可用。按所需文档是否完整选择分层准确率，完整列出截断、无效与失败记录。主比较不含共享 planner；另报告每组加一次 planner 的总成本。四组合计 GPU 预算只需加一次共享 planner，不能加四次。

标准库可能被模型记忆；文档选择不完整但答对，不能解释为检索正确，需结合召回分层结果解读。此题族结论限于静态签名事实比较。

## 命令

以下命令均在 worktree 根目录运行，不需要安装依赖或修改服务器环境。

```bash
cd <仓库根目录>
PY=python
export PYTHONPATH=src
```

生成新题集（默认不覆盖现有文件；显式 `--overwrite` 才覆盖）：

```bash
"$PY" -m experiments.context_selection.generate \
  --source-root /usr/lib/python3.11 --seed 20261019 \
  --questions 120 --probe-size 20 --candidates 8 \
  --output .agentos/context-selection-new/manifest.json
```

桩后端全部组探测集，完全不发 HTTP；目录必须尚不存在：

```bash
OUT=.agentos/context-selection-smoke-new
"$PY" -m experiments.context_selection.freeze --stub --split all \
  --output "$OUT/planning" --concurrency 4
for arm in all declared bm25 oracle; do
  "$PY" -m experiments.context_selection.run --stub --arm "$arm" \
    --split probe --plans "$OUT/planning/plans.json" \
    --output "$OUT/$arm" --concurrency 4 --repeats 3
done
"$PY" -m experiments.context_selection.analyze --input "$OUT" --output "$OUT/analysis"
"$PY" -m experiments.context_selection.estimate --plans "$OUT/planning/plans.json" \
  --split formal --repeats 3 --reader-output-tokens 256 --output "$OUT/estimate-formal.json"
```

真实后端由实验调度方设置 `BASE_URL`、`MODEL`；以下命令为交付接口，本次未执行。先 probe，审查后使用独立输出目录将 `SPLIT` 改为 formal，并重新冻结对应 split 的真实计划。不得把桩计划用于真实组，配置校验会拒绝。

```bash
SPLIT=probe
OUT=.agentos/context-selection-real-probe
"$PY" -m experiments.context_selection.freeze \
  --base-url "${BASE_URL:?required}" --model "${MODEL:?required}" --split "$SPLIT" \
  --output "$OUT/planning" --concurrency 4 --max-tokens 4096 --seed 20261019
for arm in all declared bm25 oracle; do
  "$PY" -m experiments.context_selection.run \
    --base-url "$BASE_URL" --model "$MODEL" --arm "$arm" --split "$SPLIT" \
    --plans "$OUT/planning/plans.json" --output "$OUT/$arm" \
    --concurrency 4 --repeats 3 --max-tokens 4096 --seed 20261019
done
"$PY" -m experiments.context_selection.analyze --input "$OUT" --output "$OUT/analysis"
```

`--concurrency` 为实际同时进入后端请求的硬上限，同时限制运行中的题目数。API key 如需鉴权从 `OPENAI_API_KEY` 读取，日志会对其脱敏。分析也支持重复 `--input` 指定各臂目录或 summary 文件。

测试：

```bash
"$PY" -m pytest -q -o asyncio_mode=auto \
  experiments/context_selection/tests tests/test_workflow_adapter.py \
  tests/test_execution_backend_metrics.py tests/test_context_ref.py
```

## 已完成的桩验证与预算

2026-10-19 已验证输出：
`.agentos/context-selection-verified-20261019/`。

- `planning/plans.json`：120 个桩计划，120 次共享 planner 请求，全部成功。
- `{all,declared,bm25,oracle}/summary.json`：每组 20 题 × 3 轮，60 个有效执行、180 次请求；四组共 720 次执行请求。无截断或运行失败。
- `analysis/analysis.json` 与 `analysis/analysis.md`：完整汇总。固定假回答准确率均为 0%，符合桩后端预期，不能用于质量或节省结论。
- `estimate-formal.json`：正式集离线预算。
- 测试结果：88 passed，包含本目录及上述 Runtime 相关回归。

离线预算使用同一选择器、提示词和 ContextManager 装配，估算每条消息字符数除以四并加八个 framing Token；每个 reader 结果假定 256 Token。以下基于桩 planner 每题两个 reader、每 reader 声明一段、三轮重复，不能代替真实 planner 的冻结结果：

| 组 | 正式集执行请求数 | 估算输入 Token |
| --- | ---: | ---: |
| all | 900 | 12,470,001 |
| declared | 900 | 1,441,407 |
| bm25 | 900 | 1,435,239 |
| oracle | 900 | 4,440,459 |

正式集共享 planner 另需 100 次请求，至多一次格式纠正时最多 200 次。每组执行完成 Token 的硬上限为 900 × 4096 = 3,686,400。真实计划允许每题 1–4 个 reader，因此三轮正式集每组执行请求区间为 600–1500；拿到真实冻结计划后应重新运行 estimate。字符估算不能用于计费或性能结论，正式统计只使用后端 reported usage。原始输出在忽略目录 `.agentos/`，不进入提交。

当前未执行任何真实模型实验；真实端点 seed/usage 兼容性、实际截断率、模型质量与实测 Token 留待调度方验证。框架验证无已知阻断项。
