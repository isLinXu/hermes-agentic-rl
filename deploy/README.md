# hermes-agentic-rl Kubernetes Deployment

Kubernetes manifests for deploying `hermes-agentic-rl` v1.0.0-rc1 in a cluster.

## Quick Start

```bash
# 1. Build the Docker image (from repo root)
docker build -t hermes-agentic-rl:1.0.0-rc1 .

# 2. Push to your registry (optional, if not using local cluster)
docker tag hermes-agentic-rl:1.0.0-rc1 <registry>/hermes-agentic-rl:1.0.0-rc1
docker push <registry>/hermes-agentic-rl:1.0.0-rc1

# 3. Update secret values
kubectl create secret generic hermes-secrets -n hermes-agentic-rl \
  --from-literal=HF_TOKEN=hf_xxx \
  --from-literal=WANDB_API_KEY=xxx \
  --dry-run=client -o yaml | kubectl apply -f -

# 4. Deploy everything
kubectl apply -k deploy/

# 5. Port-forward TensorBoard
kubectl port-forward -n hermes-agentic-rl svc/hermes-tensorboard 6006:6006
# Open http://localhost:6006
```

## Architecture

| Component        | Kind        | Purpose                                           |
|------------------|-------------|---------------------------------------------------|
| Namespace        | Namespace   | `hermes-agentic-rl` isolation                     |
| ServiceAccount   | SA + RBAC   | Least-privilege for training pods                 |
| Secret           | Secret      | HF_TOKEN, WANDB_API_KEY, OPENAI_API_KEY           |
| ConfigMap        | ConfigMap   | Training configs (`echo_grpo_mvp.yaml`, etc.)     |
| PVC              | PVC         | 50Gi shared volume for outputs & checkpoints      |
| Job (GRPO)       | Job         | CPU-friendly GRPO training on TinyCausalLM        |
| Job (MTGRPO)     | Job         | Multi-turn GRPO with curriculum                   |
| Job (Eval)       | Job         | RL evaluation / gate pass                         |
| Job (Benchmark)  | Job         | Benchmark suite execution                         |
| CronJob          | CronJob     | Daily curriculum learning (`online-self-evolve`)  |
| TensorBoard      | Deployment  | Real-time metrics dashboard                       |
| HPA              | HPA         | Autoscales TensorBoard 1→3 replicas               |
| NetworkPolicy    | NetworkPolicy | Default-deny + same-namespace allow             |

## Running Training

```bash
# GRPO training
kubectl apply -f deploy/job-training-grpo.yaml

# MTGRPO training
kubectl apply -f deploy/job-training-mtgrpo.yaml

# Check logs
kubectl logs -n hermes-agentic-rl job/hermes-training-grpo -f

# Wait for completion
kubectl wait -n hermes-agentic-rl --for=condition=complete job/hermes-training-grpo --timeout=30m
```

## GPU Training

To schedule on GPU nodes, patch the Job with nodeSelector / tolerations / resource limits:

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
nodeSelector:
  accelerator: nvidia-a100
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

## Kustomize Overlays

Create environment-specific overlays:

```
deploy/
├── base/          # (optional) move base manifests here
└── overlays/
    ├── dev/
    │   └── kustomization.yaml
    └── prod/
        └── kustomization.yaml
```

Example `overlays/prod/kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../
images:
  - name: hermes-agentic-rl
    newName: ghcr.io/gatilin/hermes-agentic-rl
    newTag: "1.0.0-rc1"
patches:
  - target:
      kind: Job
      name: hermes-training-grpo
    patch: |
      - op: replace
        path: /spec/template/spec/containers/0/resources/limits/memory
        value: 32Gi
```

## Cleanup

```bash
kubectl delete -k deploy/
```

## Security Notes

- All pods run as `nobody` (uid 65534) with `runAsNonRoot: true`.
- Root filesystems are mounted read-only where possible.
- Secrets are injected via `secretKeyRef` with `optional: true` to allow local testing without real tokens.
- NetworkPolicy default-denies cross-namespace ingress.
