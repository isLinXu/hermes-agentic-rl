"""Quick smoke: SmolLM2-360M-Instruct + LoRA on MPS (no atropos dep)."""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def main():
    from hermes_agentic_rl.backends.hf import HFBackendConfig, HFCausalLMBackend
    from hermes_agentic_rl.peft.lora import LoRAConfig, inject_lora

    MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
    DEVICE = "mps"

    print("=== 1. Loading SmolLM2-360M-Instruct on MPS ===")
    cfg = HFBackendConfig(model_name_or_path=MODEL, device=DEVICE, dtype="float32")
    backend = HFCausalLMBackend(cfg)
    total = sum(p.numel() for p in backend.model.parameters())
    print(f"  Vocab: {backend.tokenizer.vocab_size}, Params: {total:,}")

    print("\n=== 2. LoRA r=16 q_proj,v_proj,o_proj ===")
    lora_cfg = LoRAConfig(r=16, alpha=32.0, target_patterns=("q_proj", "v_proj", "o_proj"))
    adapter = inject_lora(backend.model, lora_cfg)
    trainable = sum(p.numel() for p in backend.trainable_parameters())
    print(f"  Modules: {len(adapter.modules)}, LoRA params: {adapter.num_parameters():,}")
    print(f"  Trainable: {trainable:,} (base frozen: {total - trainable:,})")

    print("\n=== 3. Generate ===")
    prompt = backend.tokenizer.encode("How many e's are in 'elephant'?")
    gen = backend.generate(prompt, max_new_tokens=20, temperature=1.0, seed=42)
    text = backend.tokenizer.decode(gen.response_ids)
    print(f"  Response: {text[:100]}")

    gen2 = backend.generate(prompt, max_new_tokens=20, temperature=1.0, seed=42)
    assert gen.response_ids == gen2.response_ids, "SEED BROKEN!"
    print("  Seed OK")

    print("\n=== 4. Score ===")
    resp_ids = backend.tokenizer.encode("3")
    lp = backend.score(prompt, resp_ids)
    print(f"  score('3'): {lp.tolist()}")

    print("\n=== 5. Gradient ===")
    loss = lp.sum()
    loss.backward()
    grads = sum(
        1 for p in backend.trainable_parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"  Params with grad: {grads}")

    print("\n=== ALL PASSED ===")


if __name__ == "__main__":
    main()
