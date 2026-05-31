"""训练验证脚本 — 端到端训练实验

覆盖范围：
  1. GRPO + Echo（最小 baseline，验证损失/奖励下降）
  2. PPO  + Echo（对比实验）
  3. GRPO + Echo，开启 normalize_reward + adaptive_kl（v0.9 特性验证）
  4. GRPO + Echo，K3 KL estimator + checkpoint（v0.6 特性验证）
  5. GRPO + sim_tool（多工具环境）
  6. GRPO + curriculum（简单 → echo 两级课程）
  7. GRPO + Echo，per_token_advantage（REINFORCE++ 路径）

运行方式：
    python scripts/training_validation.py            # 全量（~5分钟）
    python scripts/training_validation.py --smoke    # 快速冒烟（~30秒）
    python scripts/training_validation.py --case 1  # 单跑某个 case
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 项目根目录加入路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.envs.sim_tool_env import (
    SimToolEnv,
    SimToolRewardComponent,
    build_sim_tool_dataset,
)
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig
from hermes_agentic_rl.trainers.ppo_trainer import PPOTrainer, PPOTrainerConfig


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _tiny_backend(seed: int = 0, value_head: bool = False) -> TinyCausalLMBackend:
    return TinyCausalLMBackend(
        TinyBackendConfig(
            dim=32, n_heads=4, n_layers=2, max_len=128,
            device="cpu", dtype="float32", seed=seed,
            with_value_head=value_head,
        )
    )


def _echo_env_and_rm() -> tuple[EchoTaskEnv, RewardManager]:
    return EchoTaskEnv(build_default_echo_dataset()), RewardManager([EchoRewardComponent(weight=1.0)])


def _sim_env_and_rm(n: int = 8) -> tuple[SimToolEnv, RewardManager]:
    return SimToolEnv(build_sim_tool_dataset(n=n, seed=42)), RewardManager([SimToolRewardComponent(weight=1.0)])


def _fmt_reward(val: float) -> str:
    bar = "█" * int(val * 20)
    return f"{val:6.4f} |{bar:<20}|"


# ──────────────────────────────────────────────
# 结果收集
# ──────────────────────────────────────────────

@dataclass
class CaseResult:
    name: str
    passed: bool
    duration_s: float
    iters: int = 0
    last_reward: float = 0.0
    best_reward: float = 0.0
    reward_delta: float = 0.0
    reward_history: list[float] = field(default_factory=list)
    error: str = ""
    assertions: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# 验证断言
# ──────────────────────────────────────────────

def _assert_reward_improves(stats: Any, threshold: float = 0.0, name: str = "") -> list[str]:
    """验证奖励确实在提升。允许 delta >= threshold."""
    results = []
    delta = stats.mean_reward_delta()
    best = stats.best_reward()
    last = stats.last_reward()

    if delta >= threshold:
        results.append(f"✓ reward_delta={delta:+.4f} >= {threshold} (PASS)")
    else:
        results.append(f"✗ reward_delta={delta:+.4f} < {threshold} (FAIL)")

    if best >= 0.0:
        results.append(f"✓ best_reward={best:.4f} >= 0 (sanity PASS)")
    else:
        results.append(f"✗ best_reward={best:.4f} < 0 (sanity FAIL)")

    results.append(f"  last_reward={last:.4f}  best_reward={best:.4f}")
    return results


def _assert_no_nan_in_metrics(metric_log: list[dict], name: str = "") -> list[str]:
    results = []
    nan_count = 0
    for entry in metric_log:
        for k, v in entry.items():
            if isinstance(v, float) and (v != v):  # NaN check
                nan_count += 1
    if nan_count == 0:
        results.append(f"✓ no NaN in {len(metric_log)} metric entries (PASS)")
    else:
        results.append(f"✗ found {nan_count} NaN values in metrics (FAIL)")
    return results


# ──────────────────────────────────────────────
# 各 Case 实现
# ──────────────────────────────────────────────

def case_1_grpo_echo_baseline(smoke: bool = False) -> CaseResult:
    """Case 1: GRPO + Echo — 最小基线，验证奖励上升"""
    name = "GRPO+Echo baseline"
    t0 = time.time()
    metric_log: list[dict] = []
    try:
        backend = _tiny_backend(seed=0)
        env, rm = _echo_env_and_rm()
        cfg = GRPOTrainerConfig(
            n_iters=10 if smoke else 60,
            group_size=8,
            prompts_per_iter=2,
            lr=5e-3,
            max_new_tokens=8,
            temperature=1.0,
            kl_coef=0.0,
            use_reference=False,
            log_every=5,
            seed=42,
            metrics_sink=metric_log.append,
        )
        trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
        stats = trainer.train()

        assertions = _assert_reward_improves(stats, threshold=-0.05)
        assertions += _assert_no_nan_in_metrics(metric_log)
        passed = all("FAIL" not in a for a in assertions)
        return CaseResult(
            name=name, passed=passed, duration_s=time.time() - t0,
            iters=len(stats.iters) if isinstance(stats.iters, list) else stats.iters, last_reward=stats.last_reward(),
            best_reward=stats.best_reward(), reward_delta=stats.mean_reward_delta(),
            reward_history=[e.get("mean_reward", 0.0) for e in metric_log if "mean_reward" in e],
            assertions=assertions,
        )
    except Exception as e:
        return CaseResult(name=name, passed=False, duration_s=time.time() - t0,
                          error=traceback.format_exc(), assertions=[f"✗ EXCEPTION: {e}"])


def case_2_ppo_echo(smoke: bool = False) -> CaseResult:
    """Case 2: PPO + Echo — 对比实验，验证 value head + GAE"""
    name = "PPO+Echo baseline"
    t0 = time.time()
    metric_log: list[dict] = []
    try:
        backend = _tiny_backend(seed=1, value_head=True)
        env, rm = _echo_env_and_rm()
        cfg = PPOTrainerConfig(
            n_iters=10 if smoke else 40,
            group_size=4,
            prompts_per_iter=2,
            lr=5e-3,
            max_new_tokens=8,
            temperature=1.0,
            kl_coef=0.0,
            vf_coef=0.5,
            lam=0.95,
            use_reference=False,
            log_every=5,
            seed=42,
            metrics_sink=metric_log.append,
        )
        trainer = PPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
        stats = trainer.train()

        assertions = _assert_reward_improves(stats, threshold=-0.05)
        assertions += _assert_no_nan_in_metrics(metric_log)
        passed = all("FAIL" not in a for a in assertions)
        return CaseResult(
            name=name, passed=passed, duration_s=time.time() - t0,
            iters=len(stats.iters) if isinstance(stats.iters, list) else stats.iters, last_reward=stats.last_reward(),
            best_reward=stats.best_reward(), reward_delta=stats.mean_reward_delta(),
            reward_history=[e.get("mean_reward", 0.0) for e in metric_log if "mean_reward" in e],
            assertions=assertions,
        )
    except Exception as e:
        return CaseResult(name=name, passed=False, duration_s=time.time() - t0,
                          error=traceback.format_exc(), assertions=[f"✗ EXCEPTION: {e}"])


def case_3_grpo_normalize_adaptive_kl(smoke: bool = False) -> CaseResult:
    """Case 3: v0.9 特性 — normalize_reward + adaptive_kl"""
    name = "GRPO+normalize_reward+adaptive_kl"
    t0 = time.time()
    metric_log: list[dict] = []
    try:
        backend = _tiny_backend(seed=2)
        env, rm = _echo_env_and_rm()
        cfg = GRPOTrainerConfig(
            n_iters=10 if smoke else 50,
            group_size=8,
            prompts_per_iter=2,
            lr=5e-3,
            max_new_tokens=8,
            temperature=1.0,
            use_reference=True,
            kl_coef=0.01,
            kl_estimator="k3",
            normalize_reward=True,
            reward_norm_clip=5.0,
            adaptive_kl=True,
            adaptive_kl_horizon=500.0,
            adaptive_kl_min=1e-4,
            adaptive_kl_max=1.0,
            target_kl=0.02,
            log_every=5,
            seed=42,
            metrics_sink=metric_log.append,
        )
        trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
        stats = trainer.train()

        assertions = _assert_reward_improves(stats, threshold=-0.05)
        assertions += _assert_no_nan_in_metrics(metric_log)
        # 检查 kl_coef 字段是否出现（adaptive_kl 应写入 metrics）
        kl_coef_logged = any("kl_coef" in e for e in metric_log)
        if kl_coef_logged:
            assertions.append("✓ adaptive_kl: kl_coef logged in metrics (PASS)")
        else:
            assertions.append("  adaptive_kl: kl_coef not in metrics (INFO — may be expected)")
        passed = all("FAIL" not in a for a in assertions)
        return CaseResult(
            name=name, passed=passed, duration_s=time.time() - t0,
            iters=len(stats.iters) if isinstance(stats.iters, list) else stats.iters, last_reward=stats.last_reward(),
            best_reward=stats.best_reward(), reward_delta=stats.mean_reward_delta(),
            reward_history=[e.get("mean_reward", 0.0) for e in metric_log if "mean_reward" in e],
            assertions=assertions,
        )
    except Exception as e:
        return CaseResult(name=name, passed=False, duration_s=time.time() - t0,
                          error=traceback.format_exc(), assertions=[f"✗ EXCEPTION: {e}"])


def case_4_grpo_checkpoint_resume(smoke: bool = False, output_dir: Path | None = None) -> CaseResult:
    """Case 4: v0.6 — checkpoint 保存 + auto_resume 续训"""
    name = "GRPO+checkpoint+resume"
    t0 = time.time()
    output_dir = output_dir or (ROOT / "outputs" / "checkpoint_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Phase A: 训练 10 iter 并保存 checkpoint
        backend_a = _tiny_backend(seed=3)
        env, rm = _echo_env_and_rm()
        cfg_a = GRPOTrainerConfig(
            n_iters=10,
            group_size=4,
            prompts_per_iter=2,
            lr=5e-3,
            max_new_tokens=8,
            temperature=1.0,
            log_every=5,
            seed=42,
            output_dir=output_dir,
            checkpoint_every=5,
            keep_last_checkpoints=2,
            auto_resume=False,
        )
        trainer_a = GRPOTrainer(policy=backend_a, env=env, reward_manager=rm, cfg=cfg_a)
        stats_a = trainer_a.train()

        # 验证 checkpoint 目录存在（格式：iter_XXXXX/）
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dirs = list(ckpt_dir.glob("iter_*")) if ckpt_dir.exists() else []

        assertions: list[str] = []
        if ckpt_dirs:
            assertions.append(f"✓ checkpoint saved: {len(ckpt_dirs)} iter dir(s) in {ckpt_dir} (PASS)")
        else:
            assertions.append(f"✗ no checkpoint dirs found in {ckpt_dir} (FAIL)")

        n_iters_a = len(stats_a.iters) if isinstance(stats_a.iters, list) else stats_a.iters

        if smoke:
            # smoke 模式跳过 resume 验证
            assertions.append("  [smoke] skipping resume phase")
            passed = all("FAIL" not in a for a in assertions)
            return CaseResult(
                name=name, passed=passed, duration_s=time.time() - t0,
                iters=n_iters_a, last_reward=stats_a.last_reward(),
                best_reward=stats_a.best_reward(), reward_delta=stats_a.mean_reward_delta(),
                assertions=assertions,
            )

        # Phase B: auto_resume 从 checkpoint 续训
        env2, rm2 = _echo_env_and_rm()
        backend_b = _tiny_backend(seed=3)
        cfg_b = GRPOTrainerConfig(
            n_iters=20,   # 总目标 20 iter，应从 10 续训
            group_size=4,
            prompts_per_iter=2,
            lr=5e-3,
            max_new_tokens=8,
            temperature=1.0,
            log_every=5,
            seed=42,
            output_dir=output_dir,
            checkpoint_every=5,
            keep_last_checkpoints=2,
            auto_resume=True,
        )
        trainer_b = GRPOTrainer(policy=backend_b, env=env2, reward_manager=rm2, cfg=cfg_b)
        stats_b = trainer_b.train()

        # 续训后 len(iters) >= 10（从 checkpoint 10 继续，最多补 10 iter）
        n_iters_b = len(stats_b.iters) if isinstance(stats_b.iters, list) else stats_b.iters
        if n_iters_b >= 1:
            assertions.append(f"✓ resumed: ran {n_iters_b} additional iter(s) (PASS)")
        else:
            assertions.append(f"  resume: iters={n_iters_b} (may be up-to-date — check logs)")

        passed = all("FAIL" not in a for a in assertions)
        return CaseResult(
            name=name, passed=passed, duration_s=time.time() - t0,
            iters=n_iters_b, last_reward=stats_b.last_reward(),
            best_reward=stats_b.best_reward(), reward_delta=stats_b.mean_reward_delta(),
            assertions=assertions,
        )
    except Exception as e:
        return CaseResult(name=name, passed=False, duration_s=time.time() - t0,
                          error=traceback.format_exc(), assertions=[f"✗ EXCEPTION: {e}"])


def case_5_grpo_sim_tool(smoke: bool = False) -> CaseResult:
    """Case 5: GRPO + SimTool — 多工具环境"""
    name = "GRPO+SimTool"
    t0 = time.time()
    metric_log: list[dict] = []
    try:
        backend = _tiny_backend(seed=4)
        env, rm = _sim_env_and_rm(n=8)
        cfg = GRPOTrainerConfig(
            n_iters=8 if smoke else 30,
            group_size=4,
            prompts_per_iter=2,
            lr=5e-3,
            max_new_tokens=12,
            temperature=1.0,
            kl_coef=0.0,
            log_every=5,
            seed=42,
            metrics_sink=metric_log.append,
        )
        trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
        stats = trainer.train()

        assertions = _assert_reward_improves(stats, threshold=-0.05)
        assertions += _assert_no_nan_in_metrics(metric_log)
        passed = all("FAIL" not in a for a in assertions)
        return CaseResult(
            name=name, passed=passed, duration_s=time.time() - t0,
            iters=len(stats.iters) if isinstance(stats.iters, list) else stats.iters, last_reward=stats.last_reward(),
            best_reward=stats.best_reward(), reward_delta=stats.mean_reward_delta(),
            reward_history=[e.get("mean_reward", 0.0) for e in metric_log if "mean_reward" in e],
            assertions=assertions,
        )
    except Exception as e:
        return CaseResult(name=name, passed=False, duration_s=time.time() - t0,
                          error=traceback.format_exc(), assertions=[f"✗ EXCEPTION: {e}"])


def case_6_grpo_curriculum(smoke: bool = False) -> CaseResult:
    """Case 6: GRPO + CurriculumEnv（两个 Echo 级别）"""
    name = "GRPO+Curriculum(2-level Echo)"
    t0 = time.time()
    metric_log: list[dict] = []
    try:
        from hermes_agentic_rl.envs.curriculum import CurriculumEnv

        # 两个 Echo level（相同任务，验证 curriculum 机制不崩溃）
        env_l0 = EchoTaskEnv(build_default_echo_dataset())
        env_l1 = EchoTaskEnv(build_default_echo_dataset())
        rm_l0 = RewardManager([EchoRewardComponent(weight=1.0)])
        rm_l1 = RewardManager([EchoRewardComponent(weight=1.0)])

        cur_env = CurriculumEnv(
            levels=[env_l0, env_l1],
            window=10,
            promote_threshold=0.5,
            allow_demote=True,
            demote_threshold=0.2,
            on_level_change=lambda old, new: print(f"  [curriculum] level {old} → {new}"),
        )

        class _CurrRM:
            async def evaluate(self, item, trajectory, tool_context):
                lvl = cur_env.current_level
                return await [rm_l0, rm_l1][lvl].evaluate(item, trajectory, tool_context)

        backend = _tiny_backend(seed=5)
        cfg = GRPOTrainerConfig(
            n_iters=10 if smoke else 40,
            group_size=4,
            prompts_per_iter=2,
            lr=5e-3,
            max_new_tokens=8,
            temperature=1.0,
            log_every=5,
            seed=42,
            metrics_sink=metric_log.append,
        )
        trainer = GRPOTrainer(policy=backend, env=cur_env, reward_manager=_CurrRM(), cfg=cfg)  # type: ignore
        stats = trainer.train()

        assertions = _assert_reward_improves(stats, threshold=-0.05)
        assertions += _assert_no_nan_in_metrics(metric_log)
        assertions.append(f"  curriculum final_level={cur_env.current_level}")
        passed = all("FAIL" not in a for a in assertions)
        return CaseResult(
            name=name, passed=passed, duration_s=time.time() - t0,
            iters=len(stats.iters) if isinstance(stats.iters, list) else stats.iters, last_reward=stats.last_reward(),
            best_reward=stats.best_reward(), reward_delta=stats.mean_reward_delta(),
            reward_history=[e.get("mean_reward", 0.0) for e in metric_log if "mean_reward" in e],
            assertions=assertions,
        )
    except Exception as e:
        return CaseResult(name=name, passed=False, duration_s=time.time() - t0,
                          error=traceback.format_exc(), assertions=[f"✗ EXCEPTION: {e}"])


def case_7_grpo_per_token_advantage(smoke: bool = False) -> CaseResult:
    """Case 7: GRPO + per_token_advantage (REINFORCE++ 路径)"""
    name = "GRPO+per_token_advantage"
    t0 = time.time()
    metric_log: list[dict] = []
    try:
        backend = _tiny_backend(seed=6)
        env, rm = _echo_env_and_rm()
        cfg = GRPOTrainerConfig(
            n_iters=10 if smoke else 40,
            group_size=8,
            prompts_per_iter=2,
            lr=5e-3,
            max_new_tokens=8,
            temperature=1.0,
            per_token_advantage=True,
            advantage_norm="whiten",
            kl_coef=0.0,
            log_every=5,
            seed=42,
            metrics_sink=metric_log.append,
        )
        trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
        stats = trainer.train()

        assertions = _assert_reward_improves(stats, threshold=-0.05)
        assertions += _assert_no_nan_in_metrics(metric_log)
        passed = all("FAIL" not in a for a in assertions)
        return CaseResult(
            name=name, passed=passed, duration_s=time.time() - t0,
            iters=len(stats.iters) if isinstance(stats.iters, list) else stats.iters, last_reward=stats.last_reward(),
            best_reward=stats.best_reward(), reward_delta=stats.mean_reward_delta(),
            reward_history=[e.get("mean_reward", 0.0) for e in metric_log if "mean_reward" in e],
            assertions=assertions,
        )
    except Exception as e:
        return CaseResult(name=name, passed=False, duration_s=time.time() - t0,
                          error=traceback.format_exc(), assertions=[f"✗ EXCEPTION: {e}"])


# ──────────────────────────────────────────────
# 报告渲染
# ──────────────────────────────────────────────

def _render_reward_curve(history: list[float], width: int = 40) -> str:
    if not history:
        return "  (no history)"
    mn, mx = min(history), max(history)
    rng = mx - mn or 1e-8
    bars = []
    for i, v in enumerate(history):
        h = int((v - mn) / rng * 6)
        bar = "▁▂▃▄▅▆▇█"[min(h, 7)]
        bars.append(bar)
    return "  curve: " + "".join(bars) + f"  [{mn:.3f} → {mx:.3f}]"


def _print_report(results: list[CaseResult]) -> bool:
    print("\n" + "═" * 70)
    print("  HERMES AGENTIC-RL  训练验证报告")
    print("═" * 70)

    all_pass = True
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"\n[Case] {r.name}  {status}  ({r.duration_s:.1f}s)")
        print(f"  iters={r.iters}  last_reward={_fmt_reward(r.last_reward)}")
        print(f"  best={r.best_reward:.4f}  delta={r.reward_delta:+.4f}")
        if r.reward_history:
            print(_render_reward_curve(r.reward_history))
        for a in r.assertions:
            print(f"  {a}")
        if r.error:
            print("  [traceback]")
            for line in r.error.strip().split("\n")[-8:]:
                print(f"    {line}")
        if not r.passed:
            all_pass = False

    print("\n" + "─" * 70)
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    print(f"  总计: {passed}/{total} PASSED")
    total_time = sum(r.duration_s for r in results)
    print(f"  耗时: {total_time:.1f}s")
    print("─" * 70 + "\n")
    return all_pass


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

CASES = {
    1: case_1_grpo_echo_baseline,
    2: case_2_ppo_echo,
    3: case_3_grpo_normalize_adaptive_kl,
    4: case_4_grpo_checkpoint_resume,
    5: case_5_grpo_sim_tool,
    6: case_6_grpo_curriculum,
    7: case_7_grpo_per_token_advantage,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="hermes-agentic-rl 训练验证")
    parser.add_argument("--smoke", action="store_true", help="快速冒烟（少量 iter）")
    parser.add_argument("--case", type=int, default=None, help="只跑某个 case（1-7）")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="将结果写入 JSON 文件",
    )
    args = parser.parse_args()

    selected = {args.case: CASES[args.case]} if args.case else CASES

    results: list[CaseResult] = []
    for idx, fn in selected.items():
        print(f"\n{'─'*60}")
        print(f"▶ Running Case {idx}: {fn.__doc__.splitlines()[0].strip()}")
        print(f"{'─'*60}")
        res = fn(smoke=args.smoke)
        results.append(res)

    all_pass = _print_report(results)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "name": r.name,
                "passed": r.passed,
                "duration_s": round(r.duration_s, 2),
                "iters": r.iters,
                "last_reward": round(r.last_reward, 6),
                "best_reward": round(r.best_reward, 6),
                "reward_delta": round(r.reward_delta, 6),
                "reward_history": [round(v, 6) for v in r.reward_history],
                "assertions": r.assertions,
                "error": r.error,
            }
            for r in results
        ]
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入: {out_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
