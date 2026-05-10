"""Evaluation + version management + paired A/B testing."""

from hermes_agentic_rl.eval.ab_test import paired_welch_t, run_ab
from hermes_agentic_rl.eval.harness import EvalConfig, EvalHarness, EvalReport
from hermes_agentic_rl.eval.version_manager import VersionInfo, VersionManager

__all__ = [
    "EvalConfig",
    "EvalHarness",
    "EvalReport",
    "VersionInfo",
    "VersionManager",
    "paired_welch_t",
    "run_ab",
]
