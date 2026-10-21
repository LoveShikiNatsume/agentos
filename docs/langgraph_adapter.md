# LangGraph 静态图接入

`agentos.adapters.langgraph.adapt_state_graph` 将未编译的 LangGraph `StateGraph` 转换为 `(TaskDAG, handlers)`。实现针对项目固定依赖 LangGraph 1.2.8；LangGraph 仍为可选依赖，仅调用适配函数时导入。适配过程不调用 `compile()` 或 `validate()`，不修改传入图，不运行节点。

## 接口与示例

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from agentos.adapters.langgraph import adapt_state_graph
from agentos.llm.mock import MockLLMBackend
from agentos.llm.gateway import ModelGateway
from agentos.runtime.engine import Runtime

class State(TypedDict, total=False):
    text: str
    answer: str
async def answer(state, *, ctx):
    reply = await ctx.llm_call([{"role": "user", "content": state["text"]}])
    return {"answer": reply.content}
graph = StateGraph(State)
graph.add_node("answer", answer)
graph.add_edge(START, "answer")
graph.add_edge("answer", END)
dag, handlers = adapt_state_graph(graph, graph_name="qa", initial_state={"text": "hello"})
# 在异步入口执行；整个示例使用本地 mock，不访问模型服务。
gateway = ModelGateway({"default": MockLLMBackend(latency=0)})
results = await Runtime(gateway, handlers, enable_kernel_probe=False).run(dag)
```

参数：

- `graph_name`：必填非空图名，记录到任务来源和 ACB。
- `initial_state`：图输入 schema 范围内的 JSON 字典，默认空字典。适配时快照化，各节点使用独立副本。
- `base_context_refs`：每个任务额外选择的共享引用，例如 `("task", "policy")`，默认 `()`。引用应由调用方预先写入 Runtime 的 `ContextManager`。基础引用仅参与模型输入装配，不自动转换成 state 字段。
- `output_prefix`：结果引用前缀，默认 `"result/"`。同一上下文池运行多个图时，应使用不同前缀并避开已有引用。该参数不改变节点任务 ID；多个 DAG 合并到同一次 Runtime 执行前仍需由调用方处理 ID 命名空间。

## 任务与状态语义

每个节点生成一个同名 `Task` 与 handler。`edges` 和 `waiting_edges` 展开为直接前驱依赖，按拓扑顺序添加任务；无须按依赖顺序注册节点。`START` 和 `END` 本身不创建任务。入口任务无依赖，具有到 `END` 边的任务标记 `metadata["is_terminal"] = True`。所有节点必须从 `START` 可达且具有通往 `END` 的路径。每个节点每次 DAG 执行至多成功完成一次；重试沿用 Runtime 的故障策略。

节点支持普通同步、异步函数和可选的显式 `ctx` 参数，例如 `(state)`、`(state, *, ctx)`。同步函数在线程中运行，异步函数在 Runtime 的事件循环中等待；`ctx` 为 `AgentHandler.Context`。传入节点的字典按其 `input_schema` 字段筛选。节点返回值必须为仅含 JSON 值的字典，不接受元组、非字符串键、自定义对象和非有限浮点数。可返回部分字段，也可返回空字典；字段必须属于图注册的 channel。TypedDict 的值类型与必填字段不进行额外运行时校验。

节点成功返回后，现有 `RunnerHandler` 将原始返回字典发布到 `result/<node>`，随后 Runtime 才完成任务并释放后继依赖。运行失败或返回值不合法时不发布该节点的最终结果。Runtime 的任务结果仍为原始返回字典；适配器不额外生成聚合的图级最终 state，终点结果可按任务 ID 读取。

节点输入采用**因果状态**：初始 state 加该节点所有静态祖先的原始更新，按稳定拓扑顺序逐一合并，每个祖先只读取一次。合并使用 `ctx.context_manager.read()` 读取共享池中的原始结果，不使用经预算处理的模型上下文，也不在闭包中缓存运行中状态。例如 `a` 返回 `{"x": 1}`、`b` 返回 `{"y": 2}`，`a → b → c` 中的 `c` 仍能读取 `x` 和 `y`；菱形分支的共同祖先不会重复合并。有依赖先后的写入允许后继覆盖同一字段；无依赖先后的祖先写入同一字段会在汇合任务调用节点前抛出 `UnsupportedGraphError`，避免任意覆盖。

任务的 `context_refs` 始终是去重后的 `base_context_refs` 加**直接前驱**的结果引用，用于模型上下文装配。state 重建会额外读取传递祖先引用；当前共享引用的读取权限独立于 `context_refs`，后者是模型输入选择，不能视为访问控制列表。根节点也启用显式引用选择，因此空列表不会继承 Runtime 面向旧任务保留的“全部共享上下文”行为。

这是一种每节点执行一次、所有前驱完成后再执行后继的 DAG 语义，不保证与 LangGraph 全局 Pregel 超步、非因果分支可见性或多次触发语义等价。普通静态多入边和列表形式的 join 都转换为等待全部直接前驱。

## Runtime 接管范围

Runtime 接管任务就绪与资源准入、ACB 生命周期、结果发布、所选上下文装配、故障处理以及事件记录。每个 ACB 的 `metadata["source"]` 保存 `{"framework": "langgraph", "node": 节点名, "graph": 图名}`；`agent.created` 事件独立保存同一来源，经 JSONL EventLog 保留，可通过 `agent_id`、`task_id`、`trace_id` 关联后续事件。

节点通过 `await ctx.llm_call(...)` 调用模型时，请求进入 Runtime 的统一模型调用路径，经过上下文装配和调用事件。Runtime 应使用 `ModelGateway` 作为 `llm_backend`，如上例所示，才能进一步应用网关策略和计量；传入普通 backend 不会自动包装成网关。节点直接调用 LangChain 模型、供应商 SDK 或其他网络客户端时，Runtime 只调度该节点，**不接管这次模型调用**，不会自动获得其批处理、缓存统计、请求限制或模型调用记录。适配器不自动重写节点内部模型或工具调用。协程与线程 handler 共享宿主进程，不由此获得独立 PID 或 cgroup。

## 明确拒绝的结构

- 已编译图或已经执行过 `compile()` 的 builder；空图、未知边端点、不可达节点、无显式入口/终点、不能到达终点的节点、循环，以及同时由 `START` 和其他节点触发的入口。
- 条件边、`Send`、`Command` 和动态目的地。可静态识别的结构在适配时拒绝；未声明但由节点返回的动态对象在执行时拒绝，不发布结果。
- 非 `TypedDict` schema、reducer、非 `LastValue` channel、managed state 与 `context_schema`。
- 子图、复合或自定义配置的 Runnable、生成器，以及 LangGraph 的 `config`、`runtime`、`store`、`writer`、`previous`、`error` 参数注入。节点内部主动调用 LangGraph 专属运行期 API 也不受支持。
- 节点和图默认值中的 defer、retry、cache、timeout、error handler 策略。恢复与调度使用 Runtime 已有策略，不静默忽略 LangGraph 配置。

既有 `LangGraphHarness` 的动态规划与循环 worker 保持现有接入路径；本适配器不改写该 harness。复杂编排可继续通过 `adapt_plan` 显式划分 Runtime 任务。

## 离线验证

```sh
PYTHONPATH=src python -m pytest -c src/pyproject.toml -q tests/test_langgraph_adapter.py tests/test_workflow_adapter.py tests/test_runtime.py tests/test_event_log.py
```

测试使用 `MockLLMBackend` 或桩对象，不访问在线模型 API、GPU 或本地模型服务。

LangGraph 接口与原生 state/边语义参考：[StateGraph](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)、[官方 Graph API 文档](https://docs.langchain.com/oss/python/langgraph/graph-api)。支持边界以本适配器和项目固定的 1.2.8 实现为准。
