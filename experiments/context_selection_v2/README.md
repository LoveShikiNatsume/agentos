# 实验 A 第二版：不完整声明与 reader 追加声明

本实验在真实 `adapt_plan → TaskDAG → Runtime.run → ctx.llm_call` 链路中比较文档选择。代码与冻结题集独立于 `experiments/context_selection/`，不修改第一版协议、结果或模块。当前仅完成无网络桩验证与离线检索分析，不能据此判断模型质量或实测 Token 节省。

## 冻结题集与答案

`manifest.json`：seed `20261019`，源 `/usr/lib/python3.11`，Python `3.11.2`。沿用第一版 `collect_documents` 的完整定义切分、长度范围、原始缩进、源文件 SHA256 与行号。共 220 题、744 份使用中的文档；probe 20 题、formal 200 题，两个 split 内各一半 hop、一半 describe。每题恰好 2 份所需文档、8 个候选。manifest 语义 SHA256（排序紧凑 JSON）为：

`0a94f2ab18e15095fb80852b4634b1f5fc0d2ed5b520b606629802357897b93f`

- **hop**：题面只命名调用方。AST 查找其函数体中的裸名称调用；不进入嵌套函数、类、lambda，不算装饰器或签名中的调用，保守排除被参数或局部绑定遮蔽的名称。除自身外恰好调用一个同模块模块级函数，并要求调用方和被调方都是唯一的顶层定义、位于不同合格文档段。模块级函数名集合来自完整源文件，包括未进入切分文档的定义，避免遗漏其他被调函数。筛除被调函数叶名称以任何子串形式出现在题面的题目。共有 201 个合格候选，固定种子抽取 110 个不同调用方。为保证数量，不要求被调函数有默认值，也未扩展到方法调用。
- **describe**：题面给出两个函数/方法的 docstring 首句，不命名符号。先折叠空白，再以 `. ! ?` 后的空白分句；首句在完整解析语料的所有函数 docstring（包括长度过滤掉的函数及嵌套函数）中出现恰好一次，至少 5 个英文字母词、至多 500 字符。比较两个不同文档中的函数，要求参数最多者唯一且首个默认表达式至多 240 字符。整个题面不能包含任何候选函数/方法的叶名称：按不区分大小写的 Unicode 单词匹配，避免将 `f` 错认成所有包含字母 f 的单词。候选对满足此约束才接受。共有 3772 个满足首句基础筛选的函数，再从中筛选 110 对不同函数对；允许不同题复用函数或文档。

两族的目录均只有 ID、名称、符号列表（第一版相同的 `description`），没有 docstring、原文、标准答案或所需文档标签。干扰项不按词面相似度排序：先从目标所在模块的其他文档中均匀无放回抽取，不够再从同顶层包补，再从全语料补，最后打乱 8 个候选。describe 的禁符号约束会拒绝整个采样结果；这是明确登记的条件筛选，不是按检索得分挑干扰项。

hop 答案：

```json
{"callee":"module.function","parameter_count":3,"parameter":"name","default":"None"}
```

没有默认值时 `parameter`、`default` 必须同时为 JSON `null`。describe 沿用 `{"winner":"module.Class.method","parameter":"name","default":"expression"}`。参数计数包括 posonly、positional、kwonly、`*args`、`**kwargs`；仅方法排除 `self/cls`。默认表达式用 `ast.parse/unparse` 归一化，不执行。拒绝重复键、额外字段、错误类型、解释文字；字段精确匹配。测试从冻结文本重推导全部 220 个答案，生成器不 import/执行标准库源码。

## Runtime 接口与恢复语义

`AgentHandler.Context.request_context_refs(refs: list[str], *, reason="") -> dict` 为同步、任务局部的追加接口。`PlannedTask.metadata` 经适配器传给 Task：

```python
metadata={"context_request_policy": {
    "candidates": ["doc_a", "doc_b"], "max_requests": 2, "max_refs": 3,
}}
result = ctx.request_context_refs(["doc_b"], reason="caller refers to this function")
```

只有候选集合内、共享池中存在、可读、尚未声明的引用可以授予。名称和 `ctx://name` 等价；追加到 ACB 的 `context_refs`，按请求顺序去重，Task 的初始声明不变。下一次 `llm_call`/`llm_batch_call` 经 ContextManager 使用新声明装配。没有策略、空策略或非法策略一律拒绝。旧任务空列表表示全部共享上下文的兼容行为保持不变，此时池内引用已包含，不重复授予。

每个类型合法的接口请求（包括空列表和完全拒绝的请求）计一次请求次数；仅授予项消耗引用额度。类型不合法的调用抛 `TypeError`，不计为请求。额度在同一任务全部尝试之间累计；`SemanticRecoveryPolicy(restore_context=True)` / `CONTEXT_RETRY` 恢复初始声明，**不退还**次数或引用额度；普通重试及 `restore_context=False` 保留追加声明。私有上下文仍按原有恢复机制处理，不回滚共享池。该实验关闭任务恢复。

每次接口请求返回并写入 `context.refs_requested` 事件的 `data`：

```json
{
  "requested":["doc_b","outside"],
  "granted":["doc_b"],
  "denied":[{"ref":"outside","reason":"not_candidate"}],
  "remaining_requests":1,"remaining_refs":2,
  "requests_used":1,"refs_granted":1,
  "reason":"follow caller","attempt":1
}
```

事件还带 task_id、agent_id、trace_id、category=context。拒绝码为 `policy_disabled`、`invalid_policy`、`request_limit`、`duplicate_request`、`not_candidate`、`not_found`、`not_shared`、`not_readable`、`already_declared`、`ref_limit`，每个请求项有独立记录。

## 各组共同工作流

planner 每题通过一个 Runtime 节点产生 1–4 个独立 reader，只看题面和目录。格式错误最多纠正一次，明确提示“只输出裸 JSON，不要 Markdown 代码块、解释或额外字段”，并附具体校验错误。传输错误和截断不做格式重试。计划冻结一次，各组必须传入相同 bundle；bundle 绑定 manifest、规范化计划和全部角色提示词/文档包装/追加反馈模板指纹，以及模型、端点、seed、输出上限、温度、思考开关。运行不重新规划，也不覆盖已有输出。

每题每轮 DAG 包含重放冻结计划的 planner、依赖 planner 的 readers、依赖全部 reader 的 answerer。所有任务显式包含 `control`，空的文档声明不会触发全共享回退。reader 只读自己的原文及候选目录，不读其他 reader 的结果；answerer 只读所有 reader 的 `result/*` 引用和控制指令。

reader 所有组使用完全相同的提示词，返回以下两种 JSON 之一：

```json
{"facts":[{"symbol":"module.name","parameter_count":2,"first_default_parameter":"x","first_default":"None","docstring_first_sentence":"Description.","direct_calls":["module.other"]}]}
```

```json
{"need":["doc_id"],"reason":"need the callee signature"}
```

事实字段允许未知值为 null，`direct_calls` 支持调用链汇总。handler 校验 JSON 后调用 Runtime 追加接口，把结果作为下一条 user 消息，保留此前 assistant 的 need 响应，继续 `ctx.llm_call`。每 reader 最多 2 个 need、3 次模型调用；只在 `declared_append` 可授予，总计至多 3 个引用。任何拒绝（含部分拒绝）之后要求下一条返回 facts，未知填 null。拒绝后仍请求、第三次仍请求或非法 reader JSON 均判执行失败，原始响应仍在 calls 日志中；不再受理违反协议的 need。

| 组 | reader 初始原文 | 追加策略 | answerer 原文 |
| --- | --- | --- | --- |
| all | 全部 8 个候选 | 无，拒绝 | 无 |
| declared | 冻结的该 reader 声明 | 无，拒绝 | 无 |
| declared_append | 同 declared | 候选内，2 次/3 引用 | 无 |
| bm25 | 同 declared 平均数量的前 k 个 | 无，拒绝 | 无 |
| oracle | 该题全部 2 份所需文档 | 无，拒绝 | 无 |
| bm25_append（次要） | 同追加组最终平均数量的前 k 个 | 无，拒绝 | 无 |

所有组共享提示词、无工具、温度 0、每轮 seed=基准 seed+轮号、并发上限、文档包装/排序；差异只有文档引用和追加策略。统一 `prefix_order=name`，关闭压缩、文档预算、缓存反馈和任务恢复；不修改宿主服务的缓存配置。默认输出上限 4096，禁止低于 2048；`--thinking default|on|off` 可配置，真实运行建议明确 `off`。默认重复 3 轮，每题先平均再做统计。

planner、reader、answerer 都接受恰好一层包住整个响应的 Markdown 围栏（空语言或 json），剥掉后解析；不从解释文字里提取 JSON，不递归剥多层。各题/角色记录 `fence_stripped`、响应序号和事件 `experiment.fence_stripped`。这与第一版严格拒绝围栏不同，是第二版统一预先登记的变更。

## BM25 与离线诊断

执行查询为**题面 + reader goal**，索引为名称 + 符号列表 + 原文；沿用第一版分词（ASCII 字母/数字、下划线拆分）、`k1=1.5,b=0.75` 和 ID 平分排序。主 BM25 先算成功冻结 reader 的声明总数 D 和 reader 数 R，使用 floor(D/R)，按 seed、题号、reader ID 的 SHA256 排序为 D%R 个 reader 加一，平均 k 精确等于 declared。每轮复用此分配，允许 k=0。

次要 `bm25_append` 必须在 **同一 split、同一计划、同一模型配置、同一重复数的 declared_append 运行完成之后**运行，使用 `--append-summary`。从实际 `final_documents` 读取每个 reader 每轮的最终文档数，在 reader×轮次网格上做整数 floor/ceil 分配，平均数精确相等，输出约分后的均值和源 summary 指纹。它不读标准答案、准确率或 required_refs 来定 k。共享 planner 失败的题没有 reader，不进入预算，仍在各组计错；若有已执行 reader 的失败/截断或缺失最终观测，明确拒绝构造次要预算，避免静默用初始声明替代实际计数。两种 BM25 的并发门与其他组相同。

`retrieval_analysis.py` 完全不调模型，验证本地源码与冻结语料一致后，比较第一版原候选与按第二版中性规则重抽的一组候选，以及第二版题集；分别输出原索引、第二版索引、goal 查询、题面+goal 查询的 recall@1..8。按题平均，多个 reader 时先算各查询召回的均值，同时输出 reader 并集召回。可传 `--plans-v1/--plans-v2`；未传计划时用题面替代 goal，因查询使用词项集合，两种查询曲线相同。服务器没有嵌入模型，本实验不做向量检索。

已完成离线结果：第一版全部 120 题；无冻结真实计划文件，使用题面查询；第二版索引固定为名称+符号+原文。每题只重抽一次中性候选，不是多个重抽种子的置信区间。

| 候选集 | @1 | @2 | @3 | @4 | @5 | @6 | @7 | @8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 第一版原干扰项 | .3472 | .5903 | .7500 | .8417 | .9347 | .9778 | .9944 | 1 |
| 第一版中性干扰项 | .3528 | .6083 | .7569 | .8861 | .9472 | .9847 | .9931 | 1 |
| 第二版 220 题 | .4227 | .7114 | .8091 | .8659 | .9136 | .9636 | .9841 | 1 |

若保持第一版索引（名称+原文），第一版原/中性曲线分别为：
`[.2972,.5167,.6889,.7944,.8819,.9319,.9764,1]`、
`[.3083,.5264,.6889,.8347,.9139,.9597,.9875,1]`。

中性候选在多数 k 上提高召回，但不是每一点都提高；这里只隔离候选构造影响，不能用这些离线差值解释第一版全部模型准确率差，也不能将不同题集之间的曲线差异解释为机制收益。

## 日志、指标与统计

每次实际请求记录完整 messages、请求参数、原始响应、usage、finish_reason、实际装配引用；`events.jsonl` 记录所有 Runtime 装配、追加、DAG、结果发布、失败等事件。`summary.json` 与逐题文件包含初始选择、最终声明、实际发送文档并集、所需文档召回、每次追加及格式详情。实测统计只采用后端 reported usage；缺失/不一致的 usage 为 null，不能补零。桩后端的 usage 明确标识为字符估算。

`analyze` 输出 `analysis.json`、`analysis.md`，全体与 hop/describe 各自包含：

- planner 每题所需文档声明召回，0/部分/完整/计划失败的数量、分布与均值；共享 planner 调用、输入和格式成本单列一次。
- 各组非截断 ITT（失败计错）及有效子集准确率、每题每轮执行输入与调用数、总成本。
- 相对 all 的准确率和输入差及 CI；`declared_append − declared` 的准确率、输入和调用数差及 CI，另有 ITT 配对结果。
- 按 **planner 声明是否完整** 分层的各组准确率；另保留实际文档召回分层，避免混淆声明与执行选择。
- 按 reader×轮次计算的请求率、按题轮次请求率、请求/拒绝数量、授予引用数、其中 required 引用占比（每次 reader 授予为一个观测）、追加后召回、多出的调用及输入差。
- 所有截断、失败、无效和围栏剥离明细；declared 被拒绝请求数可表示 reader 表达材料不足的次数，不能等同于其诊断一定正确。

任一阶段 `finish_reason=length` 都使所在题轮次无效；任一重复轮次截断，该题从该组的全部质量均值及相应配对分析中剔除，题号/轮次/阶段原因单列。传输、reader 格式和 Runtime 失败在非截断 ITT 计错；answerer JSON 非法也计错并记录格式错误。先题内平均重复，再按题号固定 seed `20260920` 做 10000 次配对 percentile bootstrap 95% CI；有效比较使用共同有效题，Token 对另要求两侧 usage 完整，报告覆盖率。共享 planner 不进入执行差，也不能在六组总 GPU 预算中重复加六次。

## 与第一版的差异及限制

第一版符号比较题改为需要发现调用关系/匹配行为描述的两族题；中性干扰抽样代替名称词重合排序；BM25 同时用题面和 goal 并增加符号索引；所有 answerer 都移除原文（包括 all 和 oracle）；统一支持 reader need 和一层围栏容错。新增 declared_append 检验补声明，新增 bm25_append 对齐追加后的文档预算。第二版 all 与其他组的唯一实验变量因此回到文档引用/追加策略；两版准确率不直接可比。

描述题目录不含语义，planner 仍可能选择全部候选而获得完整声明，不能预设真实模型一定遗漏，须看实际召回分层。静态 AST 调用不是运行时调用图，不追踪别名、猴子补丁或动态分派。标准库可能被模型记忆；漏文档仍答对不等于检索成功。题之间可共享符号/文档，按题 bootstrap 未按模块聚类，跨模块推广需另做实验。本文未运行真实模型，真实 seed/usage/思考参数兼容性、准确率、截断和追加挽回率由调度方验证。

## 命令

从 worktree 根目录执行，所有输出目录须为新路径。默认 manifest 已冻结，不需要再生成。

```bash
PY=python
export PYTHONPATH=src
OUT=.agentos/context-selection-v2-new

# 可复现生成到一个新文件，不覆盖冻结文件
"$PY" -m experiments.context_selection_v2.generate \
  --source-root /usr/lib/python3.11 --seed 20261019 --questions 220 --probe-size 20 \
  --output "$OUT/manifest-check.json"

# 一次冻结；桩后端只检查收到的消息，不访问答案，不发 HTTP
"$PY" -m experiments.context_selection_v2.freeze --stub --thinking off --split all \
  --output "$OUT/planning" --concurrency 4
for arm in all declared declared_append bm25 oracle; do
  "$PY" -m experiments.context_selection_v2.run --stub --thinking off --arm "$arm" \
    --split probe --plans "$OUT/planning/plans.json" --output "$OUT/$arm" \
    --concurrency 4 --repeats 3
done
"$PY" -m experiments.context_selection_v2.run --stub --thinking off --arm bm25_append \
  --split probe --plans "$OUT/planning/plans.json" \
  --append-summary "$OUT/declared_append/summary.json" --output "$OUT/bm25_append" \
  --concurrency 4 --repeats 3
"$PY" -m experiments.context_selection_v2.analyze --input "$OUT" --output "$OUT/analysis"
"$PY" -m experiments.context_selection_v2.retrieval_analysis --output "$OUT/retrieval"

# 可选：用正式集追加桩运行的实际最终文档数估计次要 BM25 预算
"$PY" -m experiments.context_selection_v2.run --stub --thinking off --arm declared_append \
  --split formal --plans "$OUT/planning/plans.json" --output "$OUT/formal-budget/declared_append" \
  --concurrency 4 --repeats 3
"$PY" -m experiments.context_selection_v2.estimate --split formal --repeats 3 \
  --plans "$OUT/planning/plans.json" --append-summary "$OUT/formal-budget/declared_append/summary.json" \
  --output "$OUT/estimate-formal.json"
```

真实入口（仅给调度方使用，本次未执行）：先 probe 审阅，再用独立目录冻结 formal。所有 freeze/run 的模型配置一致；不能把桩计划用于真实模型。`BASE_URL` 和 `MODEL` 由调度方提供，不设默认服务地址。

```bash
SPLIT=probe
OUT=.agentos/context-selection-v2-real-probe
"$PY" -m experiments.context_selection_v2.freeze \
  --base-url "${BASE_URL:?required}" --model "${MODEL:?required}" --thinking off \
  --split "$SPLIT" --output "$OUT/planning" --concurrency 4 --seed 20261019 --max-tokens 4096
for arm in all declared declared_append bm25 oracle; do
  "$PY" -m experiments.context_selection_v2.run \
    --base-url "$BASE_URL" --model "$MODEL" --thinking off --arm "$arm" \
    --split "$SPLIT" --plans "$OUT/planning/plans.json" --output "$OUT/$arm" \
    --concurrency 4 --repeats 3 --seed 20261019 --max-tokens 4096
done
"$PY" -m experiments.context_selection_v2.run \
  --base-url "$BASE_URL" --model "$MODEL" --thinking off --arm bm25_append \
  --split "$SPLIT" --plans "$OUT/planning/plans.json" \
  --append-summary "$OUT/declared_append/summary.json" --output "$OUT/bm25_append" \
  --concurrency 4 --repeats 3 --seed 20261019 --max-tokens 4096
"$PY" -m experiments.context_selection_v2.analyze --input "$OUT" --output "$OUT/analysis"
```

## 已完成的桩验证与正式预算

最终输出：`.agentos/context-selection-v2-verified-20261019/`（忽略目录，不进提交）。`planning/plans.json` 有 220 个成功桩计划；每题两个 reader，每 reader 初始声明一份。probe 每组 20 题×3 轮，全部 60 个执行有效，无截断/失败：

| 组 | probe 执行调用数 |
| --- | ---: |
| all | 180 |
| declared | 300 |
| declared_append | 420 |
| bm25 | 300 |
| oracle | 300 |
| bm25_append | 300 |

追加组 probe 共 240 次请求、360 个授予；declared 共 120 次拒绝请求。其余组没有授予。桩固定返回空 facts 和假答案，所有组准确率为 0；它故意发 need、围栏响应以覆盖日志、预算、剥离及继续调用路径。`analysis/` 为全体和分题族 JSON/Markdown；`retrieval/` 为离线曲线。`formal-budget/declared_append/` 是额外的纯桩预算观测，不是正式模型实验。

`estimate-formal.json` 用同一装配器/提示词离线估算消息字符数÷4，另加每条消息 8 个 framing Token。假定 facts 256 Token、need 响应 128 Token；“请求额度用满”情景为每个非追加 reader 申请一次遭拒后作答，追加 reader 两次请求合计 3 份文档后作答；追加的原文取剩余最长文档以保守估计。此情景不是 tokenizer 的硬上界，也不是实测。

正式 200 题、3 轮、当前桩计划（2 readers/题）的预算如下。次要 BM25 的 k=4 来自正式追加桩的 1200 个 reader 轮次、4800 份最终文档；真实运行必须重新用真实追加 summary 计算。

| 组 | 无 need 请求数 | 无 need 估算输入 Token | 请求额度用满请求数 | 该情景估算输入 Token |
| --- | ---: | ---: | ---: | ---: |
| all | 1800 | 16,560,756 | 3000 | 32,935,902 |
| declared | 1800 | 3,715,824 | 3000 | 7,296,438 |
| declared_append | 1800 | 3,715,824 | 4200 | 22,899,396 |
| bm25 | 1800 | 3,823,422 | 3000 | 7,511,634 |
| oracle | 1800 | 5,539,749 | 3000 | 10,944,288 |
| bm25_append | 1800 | 9,473,997 | 3000 | 18,812,784 |

共享正式 planner 另计 200 次（一次格式纠正时最多 400 次），该桩的输入估算为 219,404 Token。单请求完成上限 4096；三种请求总数的完成 Token 硬上限分别为 7,372,800 / 12,288,000 / 17,203,200。真实冻结计划允许 1–4 readers/题：无 need 每组 1200–3000 请求，非追加组一次 need 后最多 1800–5400，追加组两次 need 后最多 2400–7800。拿到真实计划后应重新估算；request gate 保证 `--concurrency` 是同时进入后端的硬上限。

测试入口（新测试放在根 `tests/`，全量默认收集）：

```bash
"$PY" -m pytest -q
"$PY" -m pytest -q experiments/context_selection/tests
```

本受限沙箱拒绝 asyncio 自唤醒 socket 的 send（`EPERM`），标准全量命令停在既有 `test_real_tools_and_profile[runtime]` 的线程完成唤醒。验证时在忽略目录 `.agentos/pytest_wakeup.py` 临时给事件循环增加 1ms 定时唤醒，再运行 `PYTHONPATH=src:.agentos "$PY" -m pytest -q -p pytest_wakeup --timeout=60`。该插件不进入提交、不改断言/后端/Runtime；正常服务器环境应直接使用上面的标准命令。

最终验证结果（2026-10-19）：

- 不加临时插件的相关测试：`tests/test_context_requests.py`、`tests/test_context_selection_v2.py`、语义恢复、workflow adapter、第一版实验测试合计 **129 passed**；其中第一版单独回归 **63 passed**。
- 全量在定时唤醒插件及 `XDG_DATA_HOME=$PWD/.agentos/test-app-data` 条件下：**1812 passed、14 skipped、8 failed、8 errors**，耗时 57.69 秒。失败/错误全部为 `tests/test_comm_ipc.py` 的 10 个本地 Unix socket 用例、`tests/test_workbench_api.py` 的 6 个本地 HTTP socket 用例，系统调用返回 `EPERM`；跳过项为 13 个宿主不允许 bubblewrap namespace 的测试及 1 个缺失演示日志的测试。没有为使结果变绿而跳过失败测试、修改断言或放宽运行时权限。
- `relevant-tests.log`、`full-tests-final.log` 保存在上述最终输出目录；临时测试插件与依赖缓存均在 `.agentos/`。**全量测试全绿仍未完成**，需要调度方在允许本地 socket/namespace 的既有测试环境重跑标准命令。本次未访问 GPU、未请求模型服务、未发在线模型请求、未 push。

## 探测后修正（协议 v2.1，正式集前登记）

2026-10-19 用 Qwen3.5-9B（关闭思考）跑第一次探测集（20 题，原始记录 `~/agentos-runs/expA2/probe_qwen35/`，不进入正式结论），发现：9/20 题 planner 输出的 JSON 缺右花括号或引号（finish_reason 为 stop，纠正后仍错）；reader 把默认值写成 JSON 数字或布尔时被事实校验判为执行失败；answerer 照抄 reader 的事实字段名而不是题目要求的答案字段；声明不完整的调用链题中 reader 几乎不提出追加。据此修改，各组一致，题集不变：

- 所有角色请求带 `response_format={"type":"json_object"}`（vLLM 约束解码），只保证语法是 JSON，不约束字段。
- reader 事实只要求每条是带非空 `symbol` 的对象，其余字段不逐项校验（事实只交给 answerer 汇总）。
- answerer 指令按题族写明答案的全部字段。
- reader 指令写明在返回事实前先检查材料是否覆盖题目所需定义，缺失时返回 `need`，不要猜。
- 判分时 `default` 若为 JSON 数字或布尔，按 Python 字面量转为字符串（`1`→`"1"`，`false`→`"False"`）后再比较；其余仍严格。

协议版本改为 `context-selection-v2.1`，提示词指纹随之变化，计划须重新冻结。

第二次探测（协议 v2.1，原始记录 `~/agentos-runs/expA2/probe2_qwen35/`，不进入正式结论）planner 20/20 成功，但仍有 reader 截断：JSON 约束模式下模型在对象内部无限输出空白，或在收到 8 份原文时逐个列举并重复符号。再次修改（协议 v2.2）：

- vLLM 服务端加 `--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'`，（单请求的同名参数在 0.26.0 上不生效）。实测在 `json_object` 模式下输出仍带单个空格，这项设置是否起作用未单独验证；之后的探测没有再出现空白循环。这是服务端配置，不进入计划 bundle，正式结果文档写明。
- reader 指令要求只报告目标或题目需要的符号，每个符号一次，至多 6 条。
- reader 只接受 `facts`、`need`、`reason` 三个键：`need` 为非空列表时按追加请求处理（同时给出的 `facts` 丢弃），`need` 为 null 或空列表时按 `facts` 处理。
- 次要 BM25 组的预算不再要求追加组每题都有效：无效题仍按其 reader 实际最终文档数计入（无效题在各组都单独计为无效）。

第三次探测（协议 v2.2，`~/agentos-runs/expA2/probe3_qwen35/`）发现 v2.2 的 reader 校验把 `"need": null` 当作追加请求判失败（约 170 次），属于实验代码错误；修正后为协议 v2.3。
