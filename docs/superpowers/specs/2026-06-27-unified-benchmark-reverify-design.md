# 统一真实 benchmark 复验入口设计

- 日期：2026-06-27
- 主题：统一 `reverify_real_benchmark.py` 的数据入口，使真实 benchmark 可在本地 parquet 缺失时自动切换到 Hugging Face 数据源
- 范围：`scripts/reverify_real_benchmark.py`、与其直接相关的测试、运行期临时配置生成逻辑
- 非范围：reward 设计调整、tool-call 稳定性优化、curriculum 改造、`hermes-preflight` 的 sandbox 兼容处理

## 背景

当前仓库已经出现了一个明显的分层不一致：

1. `hermes_agentic_rl/envs/hermes_reasoning_traces.py` 的环境配置已经支持两类真实数据入口：
   - 本地 `dataset_path`
   - 远程 `repo_id/config_name/split`，并支持 `streaming` 与 `rows_api_only`
2. 通过实际命令验证，当前代码已经可以直接从 `lambda/hermes-agent-reasoning-traces` 加载真实 trace 样本并展开为训练项。
3. 但 `scripts/reverify_real_benchmark.py` 仍然将统一复验流程绑定到 `data/hermes_reasoning_traces/train.parquet`。

这导致仓库出现一种尴尬状态：

- 训练与环境层已经具备远程真实数据能力
- 一键复验入口却仍然把“是否存在本地 parquet”当成硬前置条件

从工程角度看，这不是训练内核能力不足，而是入口编排没有跟上环境能力。

## 目标

本批次目标只有四个：

1. 让 `reverify_real_benchmark.py` 在默认 parquet 缺失时，不再直接失败，而是自动切换到 Hugging Face 数据源。
2. 保持现有本地 parquet 工作流不变，避免破坏已经存在的离线复验路径。
3. 让 `eval-rl` 与 `benchmark-suite` 在 HF 模式下仍能沿用现有正式配置的大部分逻辑，只覆盖数据入口相关字段。
4. 用回归测试锁定：
   - 数据源优先级
   - dry-run 行为
   - 临时配置生成
   - benchmark-suite 的内部配置重写

## 成功标准

本批次完成后，至少满足以下标准：

1. `python scripts/reverify_real_benchmark.py --dry-run` 在本地 parquet 缺失时返回 `0`，并明确输出当前将采用 HF 数据源。
2. `python scripts/reverify_real_benchmark.py` 在本地 parquet 缺失但网络路径可用时，能顺序执行：
   - `hermes-preflight`
   - `eval-rl --config <临时 eval 配置>`
   - `eval-rl --config <临时 stage2 配置>`
   - `benchmark-suite --config <临时 benchmark-suite 配置>`
3. 显式传入 `--dataset /path/to/train.parquet` 时，仍优先走本地 parquet，不进入 HF fallback。
4. HF 模式下生成的临时配置不改动 reward、backend、metrics、promotion gate 等评估逻辑，只改数据入口及必要的配置引用。
5. 对应测试能覆盖 HF fallback 与 parquet 优先级，防止后续回归。

## 方案对比

### 方案 A：脚本层运行时重写配置，推荐

做法：

1. `reverify_real_benchmark.py` 先解析最终数据源模式。
2. 若走 parquet，则保持当前行为。
3. 若走 HF，则读取现有正式 YAML 配置，生成运行期临时副本。
4. `eval-rl` 与 `benchmark-suite` 都只消费这些临时副本。

优点：

- 改动集中在脚本层，最小侵入
- 不污染长期正式配置
- 最符合“先补统一入口”的目标

缺点：

- 需要维护临时配置生成逻辑
- `benchmark_suite.yaml` 内部引用的 `config_path` 也要同步重写

### 方案 B：直接修改正式 eval / benchmark 配置

做法：

- 将 `configs/hermes_reasoning_traces_eval_rl.yaml`
- `configs/hermes_reasoning_traces_eval_rl_terminal_command_stage2.yaml`
- `configs/benchmark_suite.yaml`

统一改为同时支持 parquet 与 HF 参数。

优点：

- 表面上配置更统一

缺点：

- 会把“环境选择”混入长期配置
- 更容易影响现有 parquet 复验工作流
- 不利于维持“正式训练配置稳定、运行时编排灵活”的边界

### 方案 C：新增一套 HF 专用配置

做法：

- 新增 `*_hf.yaml`
- 复验脚本根据模式切换到不同配置

优点：

- 最容易理解
- 问题排查直观

缺点：

- 配置重复
- 后续极易漂移
- 与“优先最小修改”的目标不符

### 结论

本批次采用 **方案 A：脚本层运行时重写配置**。

## 总体设计

### 一、数据源模式

引入两种明确模式：

1. `parquet`
2. `hf`

数据源选择优先级固定如下：

1. 如果显式传 `--dataset`，强制走 `parquet`
2. 如果未传 `--dataset`，且默认 `data/hermes_reasoning_traces/train.parquet` 存在，走 `parquet`
3. 如果默认 parquet 不存在，自动走 `hf`

这个优先级保证：

- 已有本地离线工作流完全不受影响
- 真实 benchmark 一键复验能力不再被本地 parquet 硬绑定

### 二、CLI 参数设计

保留现有参数：

- `--dataset`
- `--skip-preflight`
- `--dry-run`

新增 HF 参数，并提供安全默认值：

- `--hf-repo-id`
  - 默认：`lambda/hermes-agent-reasoning-traces`
- `--hf-config-name`
  - 默认：`kimi`
- `--hf-split`
  - 默认：`train`
- `--hf-streaming`
  - 默认：开启
- `--hf-rows-api-only`
  - 默认：开启

新增参数只影响 HF 模式，不影响 parquet 模式。

### 三、运行期配置物化

脚本内部新增运行时配置物化逻辑：

1. 若模式为 `parquet`
   - 复用现有正式配置路径
   - 若显式传 `--dataset`，继续创建或刷新默认数据软链
2. 若模式为 `hf`
   - 在临时目录中生成以下配置副本：
     - `eval` 配置副本
     - `stage2 eval` 配置副本
     - `benchmark-suite` 配置副本

临时 `eval` 配置的改动只允许覆盖 `environment` 中与数据源有关的字段：

- 删除或忽略 `dataset_path`
- 写入：
  - `repo_id`
  - `config_name`
  - `split`
  - `streaming`
  - `rows_api_only`

必要时允许写入：

- `rows_api_timeout`
- `rows_api_retries`

但不主动修改以下内容：

- `reward_mode`
- `tool_call_reward_weight`
- `text_reward_weight`
- `backend`
- `eval_rl`
- `metrics`
- `promotion_gate`
- `capability_axes`

### 四、benchmark-suite 重写

HF 模式下，`benchmark_suite.yaml` 不能继续引用正式 `eval` 配置路径，而应改为引用临时 `eval` 配置副本。

因此，临时 benchmark-suite 配置只做一类重写：

- 将 `benchmark_suite.benchmarks[].config_path` 从正式路径替换为临时 eval 配置路径

除此之外不改 benchmark 权重、阈值、name、tags 等逻辑。

### 五、命令构建方式

当前 `build_commands()` 把正式配置路径写死在脚本常量中。

本批次改为：

1. 先解析“最终运行时配置路径”
2. 再把这些路径传给 `build_commands()`

也就是说，`build_commands()` 不再依赖固定常量，而是显式接收：

- `eval_config_path`
- `stage2_eval_config_path`
- `benchmark_suite_config_path`
- `include_preflight`

这样可以保证：

- parquet 模式与 HF 模式共用同一组命令编排逻辑
- dry-run 与实际执行看到的是同一套命令

### 六、dry-run 语义

`--dry-run` 的行为要从“只检查本地 parquet 是否存在”改成“展示最终将采用的真实运行方案”。

dry-run 输出至少应包含：

1. 当前选择的数据源模式：`parquet` 或 `hf`
2. 数据源说明：
   - parquet 模式下显示实际数据路径
   - HF 模式下显示 `repo_id/config_name/split`
3. 最终要执行的命令
4. 若为 HF 模式，说明会生成临时配置并展示其路径

dry-run 在以下情况下才返回错误：

1. 显式传 `--dataset`，但目标文件不存在
2. 临时配置生成失败

dry-run 不应再因为“默认 parquet 不存在”而失败。

## 详细模块设计

### `resolve_data_source()`

职责：

1. 根据 CLI 参数与默认 parquet 存在性，返回最终数据源模式
2. 产出结构化结果对象，至少包含：
   - `mode`
   - `description`
   - `dataset_path` 或 HF 参数
   - 是否需要创建软链

### `materialize_runtime_configs()`

职责：

1. parquet 模式下直接返回正式配置路径
2. HF 模式下生成临时配置目录与副本，并返回运行时配置路径集合

输出结构建议包含：

- `eval_config_path`
- `stage2_eval_config_path`
- `benchmark_suite_config_path`
- `temp_dir`

### `build_commands()`

职责：

1. 基于运行时配置路径组装命令
2. 保持现有顺序：
   - 可选 `hermes-preflight`
   - `eval-rl`
   - `eval-rl stage2`
   - `benchmark-suite`

### `run_commands()`

职责：

1. 保持当前清理 `PYTHONHOME` / `PYTHONPATH` 的行为
2. 执行构造好的命令
3. 不负责数据源判断与配置生成

## 测试设计

本批次严格遵循“先测试，再实现”。

第一轮至少新增以下测试：

1. `test_main_dry_run_falls_back_to_hf_when_default_parquet_missing`
   - 默认 parquet 不存在
   - `--dry-run` 返回 `0`
   - 输出包含 HF 数据源信息

2. `test_explicit_dataset_keeps_parquet_priority`
   - 显式传 `--dataset`
   - 即使同时存在 HF 参数，也仍走 parquet

3. `test_materialize_runtime_configs_builds_hf_eval_configs_without_dataset_path`
   - HF 模式下生成临时 eval 配置
   - 其中不再保留 `dataset_path`
   - 正确写入 `repo_id/config_name/split`

4. `test_materialize_runtime_configs_rewrites_benchmark_suite_config_paths`
   - 临时 benchmark-suite 配置中
   - `config_path` 指向临时 eval 配置而不是正式路径

5. `test_build_commands_uses_runtime_config_paths`
   - 命令构造函数使用的是传入路径
   - 而不是脚本内硬编码常量

必要时允许补充：

6. `test_main_returns_error_when_explicit_dataset_missing`
7. `test_parquet_mode_reuses_default_configs_without_temp_rewrite`

## 文件修改范围

预计允许修改的文件：

1. `scripts/reverify_real_benchmark.py`
2. `tests/test_reverify_real_benchmark_script.py`

如实现需要少量辅助函数，也允许新增一个与脚本紧邻的轻量辅助模块，但优先保持逻辑留在现有脚本内，避免无意义拆分。

预计不修改的文件：

1. `hermes_agentic_rl/envs/hermes_reasoning_traces.py`
2. `hermes_agentic_rl/eval/*`
3. `configs/hermes_reasoning_traces_eval_rl.yaml`
4. `configs/hermes_reasoning_traces_eval_rl_terminal_command_stage2.yaml`
5. `configs/benchmark_suite.yaml`

## 风险与回退

### 风险

1. HF 模式依赖远程数据访问，如果运行环境网络不可用，则统一入口虽然不再被 parquet 阻塞，但执行阶段仍可能失败。
2. benchmark-suite 内部若未来新增更多嵌套结构，当前的 `config_path` 重写逻辑需要同步维护。
3. 若临时配置覆盖字段过多，可能无意改变现有评估逻辑。

### 回退策略

1. 所有正式配置保持不变，回退成本低。
2. 数据入口改动集中在脚本层，一旦发现问题，可直接回退该脚本与测试。
3. 实现时严格限制 HF 模式下只覆盖 `environment` 数据字段与 `benchmark_suite` 的配置引用，降低副作用。

## 预期输出

本批次完成后，应能产出以下结果：

1. 一个支持 parquet / HF 双入口自动切换的 `reverify_real_benchmark.py`
2. 覆盖 fallback、优先级和临时配置重写的测试
3. 更真实的 dry-run 输出，用于说明最终会如何复验

## 自检结论

已完成自检，结论如下：

1. 文档没有保留 `TODO`、`TBD` 等占位项。
2. 范围被明确限制在统一复验入口，不与 reward 或训练策略优化混合。
3. 数据源优先级、命令构造、临时配置生成、测试范围已经定义清楚。
4. 设计采用脚本层最小侵入方案，符合当前先补齐一键复验能力的目标。
