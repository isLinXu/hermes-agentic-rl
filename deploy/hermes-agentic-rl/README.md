# hermes-agentic-rl Helm Chart

Helm chart for deploying [hermes-agentic-rl](https://github.com/gatilin/hermes-agentic-rl) — an Agentic RL framework supporting GRPO, PPO, multi-turn training, curriculum learning, and self-evolution loops.

## Prerequisites

- Kubernetes 1.25+
- Helm 3.12+

## Installation

### Quick start (one-shot training Job)

```bash
helm install hermes-train ./deploy/hermes-agentic-rl \
  --namespace hermes \
  --create-namespace \
  --set workloadType=job \
  --set command=train-rl \
  --set image.repository=hermes-agentic-rl \
  --set image.tag=latest
```

### Scheduled training (CronJob)

```bash
helm install hermes-cron ./deploy/hermes-agentic-rl \
  --namespace hermes \
  --create-namespace \
  --set workloadType=cronjob \
  --set cronjob.schedule="0 2 * * *" \
  --set command=train-rl
```

### Long-running service (Deployment)

```bash
helm install hermes-worker ./deploy/hermes-agentic-rl \
  --namespace hermes \
  --create-namespace \
  --set workloadType=deployment \
  --set command=online-cycle \
  --set deployment.replicaCount=2
```

## Workload Types

| Type | Use case | Kubernetes resource |
|------|----------|-------------------|
| `job` | One-shot training / evaluation | Job |
| `cronjob` | Scheduled recurring training | CronJob |
| `deployment` | Long-running worker / service | Deployment |

## Commands

Supported CLI commands (passed via `--set command=<cmd>`):

- `train-rl` — RL training (GRPO / PPO)
- `eval-rl` — RL evaluation
- `eval-gate` — Quality gate evaluation
- `benchmark-suite` — Run benchmark suite
- `online-cycle` — Online self-evolution loop
- `session-train-worker` — Session-based training worker
- `self-evolution-batch` — Batch self-evolution
- `train` — Export rollouts as JSONL
- `rollout` — Single rollout

## Configuration

### Inline config (default)

Override the default training config via `--set config.inline=...`:

```bash
helm install hermes-train ./deploy/hermes-agentic-rl \
  --set config.inline="$(cat my-config.yaml)"
```

### External ConfigMap

```bash
kubectl create configmap my-config --from-file=config.yaml=my-config.yaml
helm install hermes-train ./deploy/hermes-agentic-rl \
  --set config.existingConfigMap=my-config
```

### GPU training

```bash
helm install hermes-train ./deploy/hermes-agentic-rl \
  --set resources.limits.\"nvidia.com/gpu\"=1 \
  --set resources.requests.\"nvidia.com/gpu\"=1 \
  --set nodeSelector.\"nvidia.com/gpu.present\"=true \
  --set 'tolerations[0].key=nvidia.com/gpu' \
  --set 'tolerations[0].operator=Exists' \
  --set 'tolerations[0].effect=NoSchedule'
```

### Persistent storage

Enable PVC for outputs and checkpoints (default: enabled):

```bash
helm install hermes-train ./deploy/hermes-agentic-rl \
  --set persistence.enabled=true \
  --set persistence.size=50Gi \
  --set persistence.storageClass=fast-ssd
```

### Environment variables (secrets)

```bash
helm install hermes-train ./deploy/hermes-agentic-rl \
  --set 'env[0].name=WANDB_API_KEY' \
  --set 'env[0].valueFrom.secretKeyRef.name=wandb-secret' \
  --set 'env[0].valueFrom.secretKeyRef.key=api-key'
```

## Values

See [values.yaml](values.yaml) for full configuration reference.

## Uninstall

```bash
helm uninstall hermes-train -n hermes
```
