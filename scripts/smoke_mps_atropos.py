"""Smoke test: SmolLM2-360M-Instruct + LoRA + MPS + letter_counting env.

Validates:
  1. HF backend loads on MPS
  2. LoRA injection + trainable_params count
  3. Single generate (reproducible with seed)
  4. score() returns logprobs
  5. AtroposLetterCountingEnv.get_next_item() + format_prompt()
  6. AtroposLetterCountingReward.evaluate()
"""

import asyncio
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "subprojects" / "atropos"))


async def main():
    from hermes_agentic_rl.backends.hf import HFBackendConfig, HFCausalLMBackend
    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.core.types import Trajectory
    from hermes_agentic_rl.envs.atropos_letter_counting import (
        AtroposLetterCountingEnv,
        AtroposLetterCountingReward,
    )
    from hermes_agentic_rl.peft.lora import LoRAConfig, inject_lora

    MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
    DEVICE = "mps"

    # ---- 1. Load model on MPS ----
    print("=== 1. Loading model ===")
    cfg = HFBackendConfig(
        model_name_or_path=MODEL,
        device=DEVICE,
        dtype="float32",
    )
    backend = HFCausalLMBackend(cfg)
    print(f"  Model loaded, vocab={backend.tokenizer.vocab_size}")
    total_params = sum(p.numel() for p in backend.model.parameters())
    print(f"  Total params: {total_params:,}")

    # ---- 2. Inject LoRA ----
    print("\n=== 2. Inject LoRA ===")
    lora_cfg = LoRAConfig(
        r=16,
        alpha=32.0,
        dropout=0.0,
        target_patterns=("q_proj", "v_proj", "o_proj"),
    )
    adapter = inject_lora(backend.model, lora_cfg)
    trainable = sum(p.numel() for p in backend.trainable_parameters())
    lora_params = adapter.num_parameters()
    print(f"  LoRA injected: {len(adapter.modules)} modules")
    print(f"  LoRA params: {lora_params:,}")
    print(f"  Trainable params (total): {trainable:,}")
    assert trainable == lora_params, "trainable should equal LoRA params only"

    # ---- 3. Generate ----
    print("\n=== 3. Generate ===")
    prompt = backend.tokenizer.encode("How many e's are in the word 'elephant'?")
    print(f"  Prompt ids: {prompt[:10]}... (len={len(prompt)})")
    gen_out = backend.generate(prompt, max_new_tokens=30, temperature=1.0, seed=42)
    text = backend.tokenizer.decode(gen_out.response_ids)
    print(f"  Response (seed=42): {text[:80]}...")
    print(f"  Logprobs: {gen_out.logprobs[:5]}...")

    # reproducibility
    gen2 = backend.generate(prompt, max_new_tokens=30, temperature=1.0, seed=42)
    assert gen_out.response_ids == gen2.response_ids, "seed reproducibility failed!"
    print("  Seed reproducibility: OK")

    # ---- 4. Score ----
    print("\n=== 4. Score ===")
    resp = backend.tokenizer.encode("3")
    logp = backend.score(prompt, resp)
    print(f"  score('3'): {logp.tolist()}")

    # ---- 5. Atropos env ----
    print("\n=== 5. AtroposLetterCountingEnv ===")
    env = AtroposLetterCountingEnv(
        backend._hf_tok,
        base_dir=str(PROJECT / "subprojects" / "atropos"),
    )
    await env.setup()
    item = await env.get_next_item()
    print(f"  item keys: {list(item.keys())}")
    print(f"  text: {item['text']}")
    print(f"  target_letters: {item['target_letters']}")
    print(f"  correct_counts: {item['correct_counts']}")
    print(f"  difficulty: {item['difficulty_level']}")

    formatted = env.format_prompt(item)
    print(f"  formatted_prompt: {formatted[:120]}...")

    # ---- 6. Reward ----
    print("\n=== 6. AtroposLetterCountingReward ===")
    reward = AtroposLetterCountingReward(weight=1.0)
    rm = RewardManager([reward])

    # test correct answer
    tgt = item["target_letters"]
    if len(tgt) == 1:
        correct_ans = f"<answer>{item['correct_counts'][tgt[0]]}</answer>"
    else:
        import json
        correct_ans = f"<answer>{json.dumps(dict(item['correct_counts']))}</answer>"
    print(f"  correct answer string: {correct_ans}")

    traj = Trajectory(
        turns_used=1,
        final_output=correct_ans,
        metadata={"runtime": {"rl": {"prompt_ids": [], "response_ids": [], "old_logprobs": []}}},
    )
    summary = await rm.evaluate(item, traj, tool_context=None)
    print(f"  correct answer reward: {summary.final_score}")

    # test wrong answer
    wrong_traj = Trajectory(
        turns_used=1,
        final_output="<answer>999</answer>",
        metadata={"runtime": {"rl": {"prompt_ids": [], "response_ids": [], "old_logprobs": []}}},
    )
    summary2 = await rm.evaluate(item, wrong_traj, tool_context=None)
    print(f"  wrong answer reward: {summary2.final_score}")

    await env.close()
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
