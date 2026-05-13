# Real Hermes Check

## 目的

用于在 Python 3.11+ 环境中检查：

- `hermes-agent` 是否已安装
- `run_agent` 是否可导入
- `AIAgent` 是否存在
- 当前仓库的 `HermesRuntimeAdapter` 是否具备真实入口探测条件
- hermes 配置与示例数据是否存在

## 前置条件

- Python 3.11+
- 已安装本仓库
- 已执行：

```bash
git submodule update --init
```

- `hermes-agent` 已作为子项目放在 `subprojects/hermes-agent/`，或者你已设置：

```bash
export HERMES_AGENT_REPO=/path/to/hermes-agent
```

- 如需对比 pip 安装形态，也可以额外安装：

```bash
python -m pip install "git+https://github.com/NousResearch/hermes-agent.git"
```

## 运行方式

```bash
python scripts/check_real_hermes.py
```

或者：

```bash
python -m hermes_agentic_rl.cli.main hermes-preflight
```

`hermes-preflight` 的 JSON 输出会包含 `repo_source`，帮助你确认当前到底是从
`subprojects/hermes-agent`、环境变量覆盖路径，还是其他本地路径加载到 Hermes。

## 输出解释

- `[OK]`：该项通过
- `[WARN]`：该项未完全通过，但不一定阻止你继续手动检查
- `[ERROR]`：该项是硬阻塞

## 常见失败

### Python 版本不足

如果输出提示 Python 版本低于 3.11，说明当前环境不能真实安装 `hermes-agent`。

### run_agent 无法导入

说明 `hermes-agent` 尚未安装、子项目路径不对，或者当前虚拟环境不正确。

### adapter build failed

说明入口存在，但当前环境可能缺少 provider 配置、依赖或其他运行条件。

## 自检通过后执行真实 rollout

```bash
python -m hermes_agentic_rl.cli.main rollout --config configs/terminal_grpo_hermes.yaml --output outputs/hermes_trajectory.json
```
