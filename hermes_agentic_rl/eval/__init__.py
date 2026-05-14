"""Evaluation + version management + paired A/B testing."""

from hermes_agentic_rl.eval.ab_test import paired_welch_t, run_ab
from hermes_agentic_rl.eval.benchmark_suite import run_benchmark_suite
from hermes_agentic_rl.eval.harness import EvalConfig, EvalHarness, EvalReport
from hermes_agentic_rl.eval.rl_eval import (
    run_eval_gate,
    run_eval_rl,
    select_items_by_group,
    split_items_by_source_trace_id,
)
from hermes_agentic_rl.eval.version_manager import VersionInfo, VersionManager

__all__ = [
    "EvalConfig",
    "EvalHarness",
    "EvalReport",
    "VersionInfo",
    "VersionManager",
    "paired_welch_t",
    "run_ab",
    "run_benchmark_suite",
    "run_eval_gate",
    "run_eval_rl",
    "select_items_by_group",
    "split_items_by_source_trace_id",
]
