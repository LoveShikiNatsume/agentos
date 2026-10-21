# 测试

本目录包含自动化回归测试，以及支撑各项技术结论的测试与验证材料。

## 1 自动化测试

`test_*.py` 为可直接执行的回归测试，当前版本共 127 个文件、1674 项测试，按模块覆盖执行、调度、上下文、模型网关、通信、故障恢复、隔离、事件与工作台。

在仓库根目录安装依赖并执行全量测试：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e "./src[dev,app,langgraph]"
.venv/bin/python -m pytest -c src/pyproject.toml tests -q
```

按模块执行时指定测试文件：

```bash
.venv/bin/python -m pytest -c src/pyproject.toml tests/test_context*.py -q
.venv/bin/python -m pytest -c src/pyproject.toml tests/test_recovery.py tests/test_fault.py -q
```

在 openEuler 与 openKylin 容器内执行同一测试集的方法见 [`demo/02-国产系统部署`](../demo/02-国产系统部署/)。

## 2 测试及验证材料

[`测试及验证材料/`](测试及验证材料/) 存放固定输入、原始记录与实验结果，其内容不作为测试模块被收集。各子目录的说明见该目录下的 `README.md`。
