# hermes-agentic-rl — Documentation

| Section | What's inside |
|---|---|
| [architecture.md](architecture.md) | 14-layer architecture overview, data flow, extension points |
| [hermes-native-training-framework.md](hermes-native-training-framework.md) | Hermes-native training facade and pipeline mapping |
| [experiments.md](experiments.md) | Run snapshots, W&B links, and benchmark notes |
| [hermes-agentic-rl-optimization-plan.md](hermes-agentic-rl-optimization-plan.md) | Article-based optimization roadmap for Hermes-style self-evolution + RL |
| [self-evolution-export.md](self-evolution-export.md) | Export Hermes session traces into self-evolution-friendly eval datasets |
| [training_loop.md](training_loop.md) | Full `OnPolicyTrainer` lifecycle: rollout → reward → loss → step |
| [configuration.md](configuration.md) | YAML schema reference for `train-rl` |
| [ROADMAP.md](ROADMAP.md) | v1.0 API stability boundary and release phases |
| [adr/](adr/) | Architecture Decision Records (one file per high-impact decision) |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Dev setup, coding conventions, PR flow |

External repositories live under `subprojects/` as git submodules. Run
`git submodule update --init subprojects/hermes-agent subprojects/atropos subprojects/tinker-atropos`
after cloning, then use
`hermes-preflight` and `atropos-preflight` to validate local wiring.

For usage examples, see the top-level [README.md](../README.md) and `configs/`.
