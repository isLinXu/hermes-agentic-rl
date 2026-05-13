# FileSystem Verifier Reward 设计文档

## 目标
在现有 “rollout → reward → train sample JSONL 导出” 的基础上，增强数据质量验证能力：对文件/终端类任务，用**文件系统校验**作为主要成功判定信号，输出更可信的 reward 与训练统计，并支持质量门槛（quality gate）。

本阶段仍然属于**数据生成+验证**，不涉及真·GRPO 参数更新训练循环。

## 背景与问题
当前 reward 已兼容 Hermes/OpenAI 的输出结构，并能在最小示例任务上得到非零分。但仍存在问题：

- `expected_output="done"` 只能表示“任务成功”，无法验证实际产物（文件/内容）是否正确
- 仅依赖自然语言 `final_output` 或 tool_calls 结构，容易出现“说对了但没做对”

因此引入一个更强、可解释的 verifier：直接检查生成文件及内容。

## 数据集 Schema 变更（用户已选：显式字段）
为保证稳定性与可扩展性，任务数据新增字段 `expected_files`：

```json
{
  "task_id": "task-1",
  "instruction": "Create hello.txt and write hello",
  "expected_output": "done",
  "expected_files": [
    { "path": "hello.txt", "equals": "hello" }
  ]
}
```

字段约定：
- `expected_files`: list
  - `path` (string, required)：相对路径（相对执行根目录）
  - 内容校验（以下二选一，至少提供一个）：
    - `equals` (string)：文件内容（strip 后）需完全相等
    - `contains` (string)：文件内容需包含该片段
  - 可选扩展（本阶段不强制实现，但保留字段位）：
    - `regex` (string)：正则匹配
    - `min_bytes` / `max_bytes`

兼容性：
- 若数据项未提供 `expected_files`，verifier reward 将返回 `score=0` 且 reason 指明“no expected_files”，不阻塞其他 reward。

## 新增 Reward：FileSystemVerifierReward
新增 reward 组件 `FileSystemVerifierReward`，职责：

- 读取 `item["expected_files"]`
- 对每个期望文件：
  - 检查文件存在
  - 读取文本内容（UTF-8，失败则计为不通过并给出原因）
  - 按 `equals` 或 `contains` 判定
- 评分：
  - 全部通过：`score=1.0`
  - 任一失败：`score=0.0`
- metadata：
  - `checked_files`: N
  - `passed_files`: M
  - `failures`: [{path, reason}]

注意：本仓库当前 hermes/fake runtime 的写文件工具写入的是当前工作目录，因此 verifier 默认以进程工作目录为根（相对路径）。

## RewardManager 组合建议
保持当前两项 reward，并新增 verifier（权重可配置，默认推荐）：

- `OutcomeReward`：0.2
- `ToolcallReward`：0.2
- `FileSystemVerifierReward`：0.6

说明：verifier 更强、更“可被训练信赖”，因此权重更高。

## Train 统计增强
在 `train summary` 输出中新增：

- `verifier_pass_ratio`（仅统计包含 expected_files 的样本）
- `file_check_fail_ratio`（包含 expected_files 的样本中失败占比）

并新增质量门槛参数（CLI 与 config 都支持）：

- `--min-verifier-pass-ratio`

当 `verifier_pass_ratio` 低于阈值时，train 退出码非 0，提示“quality gate failed”。

## 示例数据更新
更新 `data/minimal_terminal_tasks.jsonl`，为两条样例补齐 `expected_files`：

- task-1: `hello.txt == "hello"`
- task-2: `notes/todo.txt == "buy milk"`

## 测试策略
新增测试覆盖：

1. verifier reward：
   - 文件不存在 → 0
   - 内容不匹配 → 0
   - equals 匹配 → 1
   - contains 匹配 → 1
2. train 多样本统计：
   - fake runtime 下，导出 N 行
   - verifier_pass_ratio == 1.0
3. hermes 真实回归：
   - 用 `configs/terminal_grpo_hermes.yaml` 跑 `train --limit 2 --overwrite`
   - 输出中 verifier_pass_ratio >= 阈值

## 验收标准
- 训练导出的样本中包含 `FileSystemVerifierReward` 的 reward component
- 示例数据两条任务均能通过文件系统校验
- `train` 输出包含 verifier 统计
- 质量门槛参数可用且能在失败时阻止“低质量数据继续产出”
