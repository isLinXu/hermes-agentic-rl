"""Quick integration test: letter_counting env + tiny backend + GRPO.

Validates the full pipeline without loading a real HF model.
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def main():
    import asyncio
    from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.envs.letter_counting import LetterCountingEnv, LetterCountingReward
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

    async def _setup():
        env = LetterCountingEnv()
        reward = LetterCountingReward(weight=1.0)
        rm = RewardManager([reward])
        backend = TinyCausalLMBackend(TinyBackendConfig(dim=32, n_heads=4, n_layers=2, max_len=256, seed=42))
        return env, rm, backend

    env, rm, backend = asyncio.run(_setup())

    print("=== Letter Counting + Tiny Backend + GRPO ===\n")

    # verify env works
    async def _smoke():
        item = await env.get_next_item()
        print(f"Item: text='{item['text'][:30]}...' targets={item['target_letters']} counts={item['correct_counts']}")
        prompt = env.format_prompt(item)
        print(f"Prompt: {prompt[:100]}...")
        prompt_ids = backend.tokenizer.encode(prompt)
        gen = backend.generate(prompt_ids, max_new_tokens=30, temperature=1.0, seed=42)
        response = backend.tokenizer.decode(gen.response_ids)
        print(f"Response: {response[:80]}...")
        from hermes_agentic_rl.core.types import RolloutStep, Trajectory
        traj = Trajectory(
            task_id=item["task_id"],
            prompt=prompt,
            steps=[RolloutStep(turn_index=0, assistant_message=response)],
            final_output=response,
            finished_naturally=gen.finished,
            turns_used=1,
            metadata={"runtime": {"rl": {"prompt_ids": prompt_ids, "response_ids": gen.response_ids, "old_logprobs": gen.logprobs}}},
        )
        summary = await rm.evaluate(item, traj, tool_context=None)
        print(f"Reward: {summary.final_score:.4f} ({summary.components[0].reason if summary.components else 'N/A'})")
    asyncio.run(_smoke())

    # short training (train() is sync, uses its own event loop internally)
    print("\n--- Short GRPO training (5 iters) ---")
    tcfg = GRPOTrainerConfig(
        n_iters=5, group_size=4, prompts_per_iter=2,
        lr=1e-3, max_new_tokens=16, temperature=1.0,
        log_every=1, seed=42,
    )
    trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=tcfg)
    stats = trainer.train()
    print(f"\nDone! Rewards: {[r['mean_reward'] for r in stats.iters]}")
    print("PASSED")


if __name__ == "__main__":
    main()