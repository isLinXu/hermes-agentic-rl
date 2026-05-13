"""Parameter-efficient fine-tuning: LoRA injection for any nn.Linear.

Usage::

    from hermes_agentic_rl.peft import inject_lora, LoRAConfig

    backend = TinyCausalLMBackend(...)
    adapter = inject_lora(
        backend.model,
        LoRAConfig(r=4, alpha=8, target_patterns=("qkv", "proj", "ff.0", "ff.2", "head")),
    )
    # Now only LoRA params require_grad; base is frozen.

    # Train via standard optimizer over adapter.parameters().
    # Save / load:
    adapter.save(Path("out/adapter.pt"))
    adapter.load(Path("out/adapter.pt"))
    # Merge back into base weights for deployment:
    adapter.merge_into_base()
"""

from hermes_agentic_rl.peft.lora import (
    LoRAAdapter,
    LoRAConfig,
    LoRALinear,
    inject_lora,
)

__all__ = ["LoRAAdapter", "LoRAConfig", "LoRALinear", "inject_lora"]
