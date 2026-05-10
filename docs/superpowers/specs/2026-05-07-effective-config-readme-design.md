# effective-config + README 更新设计文档

## 目标
继续强化 “训练数据生成+验证管线” 的可用性与可追溯性：

1) 新增 `--print-effective-config` / `--print-effective-config-only`
   - 输出本次 `train` 真实生效的配置（含 CLI 覆盖后的 dataset/export/workdir/gates 等）
   - 便于实验复现、排查“为什么这次跑出来不一样”
2) 更新 `README.md`
   - 说明当前训练闭环已实现到“数据生成+验证”层（不包含真·GRPO 参数更新）
   - 文档化所有关键参数与推荐运行命令（含 mixed 数据集、quality gate、workdir-clean）

## 范围
仅改动：
- `hermes_agentic_rl/cli/main.py`
- `README.md`
- 新增 1 个测试文件验证 effective-config 输出

## CLI 设计
新增参数（train only）：
- `--print-effective-config`：在 train 开始前打印 effective config（JSON），然后继续执行训练
- `--print-effective-config-only`：只打印 effective config（JSON），然后退出（exit code 0）

输出内容（示例字段）：
```json
{
  "command": "train",
  "base_cwd": "...",
  "runtime": {"integration": "fake"},
  "environment": {"dataset_path": "..."},
  "trainer": {
    "export_training_path": "...",
    "workdir_base": "...",
    "overwrite": true,
    "max_samples": 10,
    "seed": 1,
    "min_nonzero_reward_ratio": 0.0,
    "min_verifier_pass_ratio": 0.9
  }
}
```

注意：
- 不输出任何 API key 的值
- 对相对路径输出解析后的绝对路径（与 train 内部一致）

## 测试策略
新增 `tests/test_train_effective_config.py`：
- 使用 fake runtime + 临时 config
- CLI 指定 `--dataset` / `--export` / `--workdir-base` / `--print-effective-config-only`
- 断言 stdout 是 JSON 且包含覆盖后的路径字段

## README 更新点
- 明确 “训练完成” 的定义：已完成数据生成+验证闭环；未实现参数更新训练
- 补充：
  - expected_files schema 与 verifier 能力
  - 常用命令：fake、hermes、mixed、CLI 覆盖、workdir-clean、print-effective-config
