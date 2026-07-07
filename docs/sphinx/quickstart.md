# Quick Start

## Minimal GRPO Training

```python
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig
from hermes_agentic_rl.backends.tiny import TinyBackend
from hermes_agentic_rl.envs.base_env import SimpleEnv
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward

# Build components
backend = TinyBackend()
env = SimpleEnv(prompts=["What is 2+2?", "What is 3+3?"])
reward_manager = RewardManager(rewards=[OutcomeReward(weight=1.0)])

# Configure training
config = GRPOTrainerConfig(
    n_iters=20,
    group_size=4,
    prompts_per_iter=2,
    lr=1e-3,
    max_new_tokens=16,
)

# Train
trainer = GRPOTrainer(
    policy=backend,
    env=env,
    reward_manager=reward_manager,
    cfg=config,
)
trainer.train()
```

## YAML Configuration

Create a `config.yaml`:

```yaml
backend: hf
model_name: Qwen/Qwen2.5-1.5B-Instruct

n_iters: 100
group_size: 8
prompts_per_iter: 4
lr: 1e-4
max_new_tokens: 256

rewards:
  components:
    - type: ToolcallReward
      weight: 1.0
    - type: OutcomeReward
      weight: 2.0
  composer:
    normalize:
      toolcall_reward: true
      outcome_reward: true
    aggregator: weighted_sum

curriculum:
  auto_advance: true
  min_iters_per_stage: 10
```

Run with:

```bash
python -m hermes_agentic_rl.yaml_config --config config.yaml
```

## SFT + RL Pipeline

```python
from hermes_agentic_rl.examples.sft_plus_rl import run_sft_then_rl

run_sft_then_rl(
    sft_data_path="data/sft.jsonl",
    rl_prompts=["Solve: {problem}"],
    n_rl_iters=50,
)
```

See `examples/minimal_grpo.py` and `examples/sft_plus_rl.py` for complete examples.
