# Hermes-Native Training Framework

`hermes-agentic-rl` 的框架层不是重造一套通用训练系统，而是把 Hermes-agent
相关能力收束成两条稳定流水线：

- `EnvTrainingPipeline`: `rollout -> reward -> export`
- `SessionTrainingPipeline`: `session record -> turn sample -> replay buffer`
- `SessionTrainingPipeline` 也能直接提交 / flush / close `session_sidecar`

这样做的目标是把 Hermes-agent 的原生 session trace、reward 语义和 replay 语义
保持在同一套数据结构里，方便训练、回放、评估和后续 self-evolution。

## 入口

```python
from hermes_agentic_rl.framework import build_framework

framework = build_framework(config)
trajectory, summary = await framework.env.collect_and_judge(item)
samples = framework.session.train_samples_from_record(record)
```

## 价值

- 环境训练和 session replay 共用同一套配置
- turn sample 可以直接回灌到 replay buffer
- 未来拆分异步 sidecar、judge 服务或 trainer 服务时，不需要改掉核心数据结构
- 同一份 session trace 可以继续喂给 prompt / skill / policy 的后续改进阶段
- `session-eval-export` 会把这些 traces 转成 GEPA-friendly 的 `task_input` /
  `expected_behavior` 数据集
