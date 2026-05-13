# CLI overrides + workdir clean 设计文档

## 目标
提升训练数据生成/验证管线的“可操作性”和“可复现实验”能力：

1) 允许在命令行直接覆盖 dataset 与 export 路径，避免频繁改 YAML
2) 提供 `--workdir-clean`，在训练前安全清空 `workdir_base` 下的样本目录，避免历史残留污染下一次训练

## 范围
仅覆盖 `train` 子命令。

不修改 reward 逻辑、不引入新数据 schema。

## CLI 变更
新增参数（train only）：

- `--dataset <path>`：覆盖 `environment.dataset_path`
- `--export <path>`：覆盖 `trainer.export_training_path`
- `--workdir-base <path>`：已存在（继续保留）
- `--workdir-clean`：新增；在开始训练前清空 workdir_base

覆盖优先级：

CLI > config.yaml

## workdir-clean 的安全规则
为了避免误删，必须满足：

- 必须提供 workdir_base（CLI 或 config）
- workdir_base 解析为绝对路径后，必须位于 `base_cwd/outputs/` 之下
  - base_cwd 指启动 `train` 时的当前工作目录（repo 根）
- 仅删除 workdir_base 下的**子目录**（以及其内容）
  - 不删除 workdir_base 本身
  - 不删除 outputs 下其它文件（例如 jsonl）

## 测试策略
新增/扩展测试覆盖：

1) CLI dataset/export 覆盖：
   - 给一个临时 config.yaml（指向默认 dataset/export）
   - CLI 传 `--dataset` 指向另一个临时 jsonl
   - CLI 传 `--export` 指向相对路径，确保解析为 base_cwd 相对路径

2) workdir-clean：
   - 在 outputs/workdirs/test 下预创建一些 task-* 子目录和文件
   - 运行 train + `--workdir-clean`
   - 断言目录被清空，但 workdir_base 仍存在

3) 安全防护：
   - workdir_base 指向 outputs 之外（例如 tmp_path/evil）
   - `--workdir-clean` 必须失败退出

## 验收标准
- `hermes-agentic-rl train --dataset ... --export ...` 正常工作
- `--workdir-clean` 仅清理 outputs/workdirs/... 下的任务目录
- pytest 全量通过
- 真实 hermes 回归：使用 `--dataset data/terminal_tasks_50_mixed.jsonl` 跑一轮可用（不要求全过 gate）
