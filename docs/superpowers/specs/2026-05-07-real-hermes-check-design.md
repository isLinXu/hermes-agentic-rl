# real-hermes-check 设计文档

## 概述
本设计定义一个面向 `hermes-agentic-rl` 真实运行环境的辅助交付：提供一个可执行的自检脚本和一份配套文档，用于帮助用户在 **Python 3.11+** 环境中验证真实 `hermes-agent` 安装、公开入口可用性，以及当前仓库的 `integration=hermes` 路径是否具备运行前置条件。

该交付的目标不是替代真实 rollout 或联网调用，而是在真正开始执行前，快速判断环境是否已经满足最小要求，并明确指出阻塞点。

## 设计目标
- 提供 `scripts/check_real_hermes.py` 自检脚本
- 提供一份面向用户的真实 Hermes 安装与自检指南
- 检查 Python 版本、`run_agent` 导入、`AIAgent` 入口、`HermesRuntimeAdapter` 入口探测，以及示例配置/数据文件是否存在
- 以可读终端输出 + 结构化 JSON 摘要的形式返回检查结果
- 保持脚本不触发真实联网 LLM 请求

## 非目标
本设计明确不包括：

- 校验 API key 是否有效
- 自动安装 `hermes-agent`
- 自动安装 Python 3.11
- 发起真实模型请求
- 运行真正的 `rollout` / `train`
- 检查远程 provider 可用性

## 背景
当前仓库已经完成：

- fake runtime 与最小训练闭环
- `integration=hermes` 的运行时适配
- 基于注入式测试的真实入口接线验证
- `README` 中的基本运行说明

但在真实环境中，用户仍然需要回答几个关键问题：

1. 当前 Python 是否满足 `hermes-agent` 的版本要求
2. `hermes-agent` 是否真的已安装
3. 当前环境是否暴露了 `run_agent.AIAgent` 这一真实公开入口
4. 当前仓库的 `HermesRuntimeAdapter` 是否能发现并构造真实入口
5. 示例配置和示例数据是否就位

如果这些前置条件没有明确检查，用户会在后续 rollout 阶段才暴露问题，排查成本更高。

## 方案选择
### 方案 A：只写文档
优点：

- 实现最简单
- 不受环境影响

缺点：

- 用户必须手动逐项检查
- 不利于自动化或快速诊断

### 方案 B：只写脚本
优点：

- 一键执行
- 适合反复验证

缺点：

- 缺少上下文说明
- 用户看到失败结果后未必知道下一步怎么做

### 方案 C：脚本 + 文档，推荐
做法：

- 提供一个自检脚本
- 提供一份配套文档解释检查项、失败原因和下一步操作

优点：

- 自动化与说明文档兼得
- 最适合作为真实 Hermes 接线前的交付

缺点：

- 比单独脚本或文档多一层维护

### 结论
本设计采用方案 C：`脚本 + 文档`。

## 总体架构
建议新增以下文件：

```text
scripts/
└── check_real_hermes.py

docs/
└── real-hermes-check.md
```

同时建议补充对应测试：

```text
tests/
└── test_real_hermes_check.py
```

## 核心设计原则
### 检查项聚焦“本地可验证前置条件”
脚本只检查当前进程、当前文件系统和当前 Python 环境中可本地判断的条件。

不做：

- 网络连通性测试
- API 调用测试
- 真实 token 消耗操作

### 输出人类可读且机器可读
终端输出应能直接让用户看懂结论，同时保留 JSON 摘要，方便复制、保存或被其他脚本读取。

### 失败分级明确
建议至少分为：

- `ok`
- `warning`
- `error`

例如：

- Python 版本不足：`error`
- 样例数据缺失：`warning`
- 入口可用：`ok`

### 不依赖真实联网结果
脚本在离线环境下仍然应该能完成检查。

## 模块职责
### `scripts/check_real_hermes.py`
职责：

- 读取当前 Python 版本
- 检查 `run_agent` 是否可导入
- 检查 `AIAgent` 是否存在
- 调用当前仓库的 `find_hermes_entrypoint()` 与 `HermesRuntimeAdapter`
- 检查 `configs/terminal_grpo_hermes.yaml` 与 `data/minimal_terminal_tasks.jsonl` 是否存在
- 输出终端摘要和 JSON 结果

### `docs/real-hermes-check.md`
职责：

- 说明脚本的用途
- 说明前置条件
- 说明如何安装 `hermes-agent`
- 说明如何执行脚本
- 解释每个检查项失败时的含义
- 说明通过自检后如何运行真实 `rollout`

### `tests/test_real_hermes_check.py`
职责：

- 测试脚本核心检查函数，而不是仅测试打印文本
- 使用 monkeypatch 注入 `run_agent` 模块或缺失场景
- 保证脚本逻辑在当前 Python 3.10 环境中也能被测试

## 检查项设计
建议脚本至少输出以下结构：

```python
{
    "python_version": {
        "status": "ok" | "warning" | "error",
        "details": "...",
    },
    "run_agent_import": {
        "status": "...",
        "details": "...",
    },
    "ai_agent_symbol": {
        "status": "...",
        "details": "...",
    },
    "entrypoint_detection": {
        "status": "...",
        "details": "...",
    },
    "adapter_build": {
        "status": "...",
        "details": "...",
    },
    "sample_config": {
        "status": "...",
        "details": "...",
    },
    "sample_dataset": {
        "status": "...",
        "details": "...",
    },
}
```

### 说明
- `python_version`
  - Python < 3.11 时为 `error`
- `run_agent_import`
  - `import run_agent` 成功则 `ok`
- `ai_agent_symbol`
  - 存在 `AIAgent` 则 `ok`
- `entrypoint_detection`
  - `find_hermes_entrypoint()` 成功则 `ok`
- `adapter_build`
  - 在安全前提下尝试构造真实 runtime wrapper
  - 若构造失败，给出具体原因
- `sample_config`
  - 检查 `configs/terminal_grpo_hermes.yaml`
- `sample_dataset`
  - 检查 `data/minimal_terminal_tasks.jsonl`

## 安全边界
### 不发起真实会话
即使 `AIAgent` 可以构造，也不调用真正的 `chat()` 或联网 `run_conversation()`。

### adapter build 允许轻量失败
若构造时已经需要外部 provider 参数或环境准备，脚本应将其记为 `warning` 或 `error`，并输出原因。

## 输出形式
建议脚本输出两部分：

### 人类可读摘要

```text
[OK] Python version: 3.11.9
[OK] run_agent importable
[OK] AIAgent symbol found
[OK] Hermes entrypoint detected: run_agent.AIAgent
[WARN] Adapter build failed: missing OPENROUTER_API_KEY
[OK] Sample config exists
[OK] Sample dataset exists
```

### JSON 摘要
最后打印：

```json
{
  "overall_status": "warning",
  "checks": { ... }
}
```

## 文档内容结构
建议 `docs/real-hermes-check.md` 包含：

1. 目的
2. 前置条件
3. 安装命令
4. 运行脚本命令
5. 输出解释
6. 常见失败情况
7. 自检通过后如何执行真实 rollout

## 测试策略
### 单元测试
至少覆盖：

- Python 版本判断逻辑
- `run_agent` 可导入 / 不可导入
- `AIAgent` 存在 / 缺失
- 入口探测成功 / 失败
- 样例配置与样例数据存在性

### 不要求
- 真正安装 hermes-agent 后的集成测试
- 真实联网调用

## 验收标准
本交付完成后，必须满足：

1. 仓库中存在 `scripts/check_real_hermes.py`
2. 仓库中存在 `docs/real-hermes-check.md`
3. 脚本可以在当前环境中运行并输出结构化结果
4. 在 Python < 3.11 时，脚本能明确指出版本不满足
5. 在注入式 `run_agent` 场景下，测试可验证入口检查逻辑

## 风险与缓解
### 风险一：adapter build 过深依赖真实环境
缓解：

- 将 “build 是否成功” 作为一项可失败的检查，而不是脚本整体失败

### 风险二：输出太偏实现细节，用户难以理解
缓解：

- 使用简单、稳定的文案
- 详细上下文放入 JSON 的 `details`

### 风险三：当前环境 Python 3.10 导致脚本不可测
缓解：

- 测试中将 Python 版本判断封装为纯函数
- 用 monkeypatch 注入 `run_agent`

## 实现顺序建议
建议按以下顺序推进：

1. 设计脚本内部检查函数
2. 编写 `tests/test_real_hermes_check.py`
3. 实现 `scripts/check_real_hermes.py`
4. 编写 `docs/real-hermes-check.md`
5. 跑全量测试回归

## 结论
`real-hermes-check` 不是新的训练功能，而是一个面向真实 Hermes 接线落地的“前置环境可用性检查交付”。它能显著降低用户在 Python 3.11+ 环境中验证真实 `integration=hermes` 路径时的排查成本。
