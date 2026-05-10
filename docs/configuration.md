# `train-rl` YAML configuration reference

Top-level keys recognized by `hermes_agentic_rl.cli.train_rl.run_train_rl`:

```yaml
algo: grpo | ppo              # default "grpo"

backend:
  name: tiny | hf
  # tiny options
  dim: 32
  n_heads: 4
  n_layers: 2
  max_len: 128
  seed: 0
  with_value_head: false      # auto-forced true for PPO
  # hf options
  model_name_or_path: gpt2
  device: cpu                 # cpu | cuda | mps
  dtype: float32
  trust_remote_code: false

environment:
  type: echo | sim_tool | letter_counting | curriculum
  # echo / sim_tool / letter_counting
  dataset_path: path/to/tasks.jsonl   # optional
  dataset_size: 16                    # sim_tool only
  dataset_seed: 0
  max_level: 10                       # letter_counting only
  n_samples: 200                      # letter_counting only
  # curriculum
  levels:                             # list[dict] of sub-env specs
    - { type: echo }
    - { type: sim_tool, dataset_size: 16 }
  window: 20
  promote_threshold: 0.6
  demote_threshold: null
  allow_demote: false

agent_loop:
  type: policy | multi_turn           # default "policy"
  max_turns: 3                        # multi_turn only
  max_new_tokens_per_turn: 16         # multi_turn only

train_rl:
  # --- core ---
  n_iters: 30
  group_size: 4
  prompts_per_iter: 2
  lr: 5.0e-3
  max_new_tokens: 8
  temperature: 1.0
  clip_eps: 0.2
  grad_clip: 1.0
  log_every: 5
  save_every: 0                       # legacy flat state_dict dump
  seed: 0
  multi_turn: false
  output_dir: outputs/run_1           # overridden by --output CLI arg

  # --- reference policy / KL ---
  use_reference: false
  kl_coef: 0.0
  kl_estimator: k1 | k2 | k3          # default k1; k3 recommended for new runs

  # --- loss aggregation ---
  loss_agg: mean_token | sum_token | dr_grpo
  max_len_for_dr_grpo: 256

  # --- GRPO-only ---
  entropy_coef: 0.0
  advantage_norm: group | batch | whiten
  per_token_advantage: false

  # --- PPO-only ---
  vf_coef: 0.5
  vf_clip_eps: 0.2
  gamma: 1.0
  lam: 0.95
  normalize_advantage: true
  whiten_advantage: false             # v0.7
  advantage_clip: 3.0                 # v0.7

  # --- interleaved SFT (GRPO) ---
  interleave_sft_every: 0
  interleave_sft_samples: 32
  interleave_sft_lr: 1.0e-4

  # --- batch generate ---
  batch_generate: false

  # --- v0.6: checkpoint / resume ---
  checkpoint_every: 0                 # iters between full ckpts; 0 disables
  keep_last_checkpoints: 3
  resume_from: null | int | "latest"
  auto_resume: false                  # True → pick newest ckpt if any

# v0.6: metrics backends — all optional, all isolation-fault-safe
metrics:
  jsonl: true                         # → <output_dir>/metrics.jsonl
  stdout: false
  tensorboard: true | "<path>"        # → <output_dir>/tb/
  wandb: true | { prefix: train }     # requires external wandb.init()

# v0.3: live dashboard (stdlib HTTP + Chart.js CDN)
dashboard:
  enabled: false
  host: 127.0.0.1
  port: 8765

# v0.6: learned reward model (Bradley-Terry RM head)
reward_model:
  enabled: false
  backend: tiny | hf
  # tiny options (same as top-level backend)
  dim: 32
  n_heads: 4
  n_layers: 2
  max_len: 256
  seed: 0
  # hf options
  model_name_or_path: gpt2
  device: cpu
  dtype: float32
  trust_remote_code: false
  # loading a pre-trained head
  head_path: path/to/rm_head.pt
  weight: 1.0
  freeze_base: true
```

## Minimal runnable example

See `configs/echo_grpo_mvp.yaml` (v0.2 MVP, still works verbatim).

## Annotated v0.6 example

See `configs/echo_grpo_v06.yaml` for a config that exercises:
- K3 KL estimator
- Checkpoint + auto-resume
- Pluggable metrics writers
