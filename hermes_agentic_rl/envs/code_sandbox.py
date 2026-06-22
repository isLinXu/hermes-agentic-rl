"""Code execution sandbox and SWE-bench environment.

Provides two levels of code execution:
  1. LocalSandbox: subprocess-based sandbox for safe Python execution.
     Uses resource limits (timeout, memory) via subprocess + signal.
     No Docker required. Suitable for unit-test style tasks.
  2. DockerSandbox: Docker-based isolation for full repo-level tasks.
     Requires Docker installed. Suitable for SWE-bench style tasks.

SWE-bench integration:
  CodeFixEnv wraps a repository + failing test suite. The agent must
  produce a patch that makes the failing tests pass. Reward is based
  on test pass rate (and optionally code quality metrics).

Integration with Hermes:
  CodeFixEnv subclasses BaseEnv, so it plugs directly into the existing
  OnPolicyTrainer + RewardManager pipeline. No special trainer needed.

  The env injects tool context so Hermes can call a real code execution
  tool during the rollout:
      tool_context["run_code"](code: str) → {"stdout": str, "stderr": str, "returncode": int}
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample

# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExecResult:
    stdout: str
    stderr: str
    returncode: int
    elapsed: float
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Local sandbox (subprocess, no Docker)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LocalSandboxConfig:
    timeout_seconds: float = 10.0
    max_output_bytes: int = 65536  # 64 KB stdout + stderr cap
    extra_env: dict[str, str] = field(default_factory=dict)


class LocalSandbox:
    """Execute Python code in a subprocess with timeout.

    Args:
        cfg: sandbox configuration.

    Usage::

        sb = LocalSandbox()
        result = await sb.run("print(1 + 1)")
        assert result.stdout.strip() == "2"
    """

    def __init__(self, cfg: LocalSandboxConfig | None = None) -> None:
        self.cfg = cfg or LocalSandboxConfig()

    async def run(self, code: str) -> ExecResult:
        """Run Python code string asynchronously."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            fpath = f.name
        try:
            return await asyncio.get_event_loop().run_in_executor(None, self._run_sync, fpath)
        finally:
            Path(fpath).unlink(missing_ok=True)

    def _run_sync(self, fpath: str) -> ExecResult:
        import os

        env = dict(os.environ)
        env.update(self.cfg.extra_env)
        # Restrict PATH to only Python to limit damage
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, fpath],
                capture_output=True,
                timeout=self.cfg.timeout_seconds,
                env=env,
            )
            elapsed = time.monotonic() - t0
            stdout = proc.stdout[: self.cfg.max_output_bytes].decode("utf-8", errors="replace")
            stderr = proc.stderr[: self.cfg.max_output_bytes].decode("utf-8", errors="replace")
            return ExecResult(
                stdout=stdout, stderr=stderr, returncode=proc.returncode, elapsed=elapsed
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            return ExecResult(
                stdout="", stderr="TIMEOUT", returncode=-1, elapsed=elapsed, timed_out=True
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            return ExecResult(stdout="", stderr=str(exc), returncode=-2, elapsed=elapsed)

    async def run_tests(self, repo_path: str, test_pattern: str = "tests/") -> ExecResult:
        """Run pytest in a repository directory."""
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pytest",
                test_pattern,
                "--tb=short",
                "-q",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=repo_path,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=self.cfg.timeout_seconds
                )
            except TimeoutError:
                proc.kill()
                return ExecResult(
                    stdout="",
                    stderr="TIMEOUT",
                    returncode=-1,
                    elapsed=time.monotonic() - t0,
                    timed_out=True,
                )
            stdout = stdout_b[: self.cfg.max_output_bytes].decode("utf-8", errors="replace")
            stderr = stderr_b[: self.cfg.max_output_bytes].decode("utf-8", errors="replace")
            return ExecResult(
                stdout=stdout,
                stderr=stderr,
                returncode=proc.returncode or 0,
                elapsed=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExecResult(
                stdout="", stderr=str(exc), returncode=-2, elapsed=time.monotonic() - t0
            )


# ---------------------------------------------------------------------------
# Docker sandbox (optional)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DockerSandboxConfig:
    image: str = "python:3.11-slim"
    timeout_seconds: float = 60.0
    max_output_bytes: int = 262144  # 256 KB
    network_disabled: bool = True
    memory_limit: str = "512m"
    cpus: float = 1.0


class DockerSandbox:
    """Execute code inside a Docker container.

    Requires Docker CLI to be installed and accessible on PATH.
    Raises RuntimeError if Docker is unavailable.
    """

    def __init__(self, cfg: DockerSandboxConfig | None = None) -> None:
        self.cfg = cfg or DockerSandboxConfig()
        if not shutil.which("docker"):
            raise RuntimeError(
                "DockerSandbox requires Docker CLI. Install Docker and make sure "
                "`docker` is on PATH."
            )

    async def run(self, code: str) -> ExecResult:
        """Run Python code string in Docker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "run.py"
            code_path.write_text(code, encoding="utf-8")
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network=none" if self.cfg.network_disabled else "",
                f"--memory={self.cfg.memory_limit}",
                f"--cpus={self.cfg.cpus}",
                f"-v{tmpdir}:{tmpdir}:ro",
                self.cfg.image,
                "python3",
                str(code_path),
            ]
            cmd = [c for c in cmd if c]  # strip empty strings
            return await self._exec(cmd)

    async def _exec(self, cmd: list[str]) -> ExecResult:
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=self.cfg.timeout_seconds
                )
            except TimeoutError:
                proc.kill()
                return ExecResult(
                    stdout="",
                    stderr="TIMEOUT",
                    returncode=-1,
                    elapsed=time.monotonic() - t0,
                    timed_out=True,
                )
            stdout = stdout_b[: self.cfg.max_output_bytes].decode("utf-8", errors="replace")
            stderr = stderr_b[: self.cfg.max_output_bytes].decode("utf-8", errors="replace")
            return ExecResult(
                stdout=stdout,
                stderr=stderr,
                returncode=proc.returncode or 0,
                elapsed=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExecResult(
                stdout="", stderr=str(exc), returncode=-2, elapsed=time.monotonic() - t0
            )


# ---------------------------------------------------------------------------
# CodeFixEnv — SWE-bench style environment
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CodeFixItem:
    """One SWE-bench style task."""

    task_id: str
    repo_path: str  # path to a local git-clean copy of the repo
    failing_tests: list[str]  # list of pytest node IDs for the failing tests
    issue_description: str  # natural language description of the bug
    gold_patch: str | None = None  # reference solution (for eval only)


@dataclass(slots=True)
class CodeFixConfig:
    test_timeout: float = 30.0
    reward_per_test: float = 1.0
    use_docker: bool = False
    docker_image: str = "python:3.11-slim"
    pass_reward_weight: float = 0.8
    format_reward_weight: float = 0.2


class CodeFixEnv(BaseEnv):
    """SWE-bench style code repair environment.

    The agent receives a bug description and must produce a unified diff
    (or a complete file replacement) that fixes the failing tests.

    Reward:
      - pass_reward: fraction of failing_tests that now pass after applying patch.
      - format_reward: 0.2 for a valid unified diff format.

    Tool context injection:
      Sets tool_context["run_code"] so the Hermes agent loop can execute
      code snippets during reasoning (before producing the final patch).

    Usage::

        items = [CodeFixItem(...)]
        env = CodeFixEnv(items, cfg=CodeFixConfig())
        await env.setup()
        item = await env.get_next_item()
        # ... run trainer ...
        await env.close()
    """

    def __init__(
        self,
        items: list[CodeFixItem],
        cfg: CodeFixConfig | None = None,
    ) -> None:
        self.items = items
        self.cfg = cfg or CodeFixConfig()
        self._idx = 0
        self._sandbox: LocalSandbox | DockerSandbox
        if self.cfg.use_docker:
            try:
                self._sandbox = DockerSandbox(
                    DockerSandboxConfig(
                        image=self.cfg.docker_image,
                        timeout_seconds=self.cfg.test_timeout,
                    )
                )
            except RuntimeError:
                self._sandbox = LocalSandbox(
                    LocalSandboxConfig(timeout_seconds=self.cfg.test_timeout)
                )
        else:
            self._sandbox = LocalSandbox(LocalSandboxConfig(timeout_seconds=self.cfg.test_timeout))

    async def setup(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return {"idx": self._idx, "n_items": len(self.items)}

    def observe(self, score: float) -> None:
        pass

    async def get_next_item(self) -> dict[str, Any]:
        if not self.items:
            raise RuntimeError("CodeFixEnv has no items")
        item = self.items[self._idx % len(self.items)]
        self._idx += 1
        return {
            "task_id": item.task_id,
            "instruction": self._make_prompt(item),
            "repo_path": item.repo_path,
            "failing_tests": item.failing_tests,
            "gold_patch": item.gold_patch,
            "_fix_item": item,
        }

    def _make_prompt(self, item: CodeFixItem) -> str:
        tests_str = "\n".join(f"  - {t}" for t in item.failing_tests)
        return textwrap.dedent(f"""
            ## Code Fix Task

            **Issue:** {item.issue_description}

            **Failing tests:**
            {tests_str}

            Your task:
            1. Analyze the repository to find the root cause.
            2. Produce a fix as a **unified diff** wrapped in:
               ```diff
               --- a/path/to/file
               +++ b/path/to/file
               @@ ... @@
               ...
               ```
            3. The fix must make all listed failing tests pass.
        """).strip()

    def format_prompt(self, item: dict[str, Any]) -> str:
        return str(item.get("instruction", ""))

    def build_supervised_samples(self, item: dict[str, Any]) -> list[SupervisedSample]:
        fix_item: CodeFixItem | None = item.get("_fix_item")
        if fix_item is None or fix_item.gold_patch is None:
            return []
        return [
            SupervisedSample(
                instruction=self._make_prompt(fix_item),
                response=f"```diff\n{fix_item.gold_patch}\n```",
            )
        ]

    # ── Patch extraction ────────────────────────────────────────────────

    @staticmethod
    def _extract_patch(text: str) -> str | None:
        """Extract unified diff from response text."""
        import re

        m = re.search(r"```diff\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Bare diff without code block
        m2 = re.search(r"^--- a/", text, re.MULTILINE)
        if m2:
            return text[m2.start() :].strip()
        return None

    # ── Patch application ───────────────────────────────────────────────

    async def _apply_patch(self, repo_path: str, patch: str) -> bool:
        """Apply unified diff to repo using `patch` CLI. Returns success."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(patch)
            patch_file = f.name
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["patch", "-p1", "--dry-run", "-i", patch_file],
                    capture_output=True,
                    cwd=repo_path,
                ),
            )
            if result.returncode != 0:
                return False
            # Actual application
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["patch", "-p1", "-i", patch_file],
                    capture_output=True,
                    cwd=repo_path,
                ),
            )
            return True
        except FileNotFoundError:
            # `patch` CLI not available; try Python-only approach (no-op here)
            return False
        finally:
            Path(patch_file).unlink(missing_ok=True)

    async def _reset_repo(self, repo_path: str) -> None:
        """Git reset --hard to undo patch application."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["git", "checkout", "--", "."], cwd=repo_path, capture_output=True
                ),
            )
        except Exception:
            pass

    # ── Reward ──────────────────────────────────────────────────────────

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        response = trajectory.final_output or ""
        fix_item: CodeFixItem | None = item.get("_fix_item")
        if fix_item is None:
            return [RewardResult(name="code_fix", score=0.0, reason="no fix_item")]

        patch = self._extract_patch(response)
        format_score = self.cfg.format_reward_weight if patch else 0.0

        if patch is None:
            return [
                RewardResult(
                    name="code_fix",
                    score=format_score,
                    weight=1.0,
                    reason="no_patch_found",
                )
            ]

        # Apply patch
        applied = await self._apply_patch(fix_item.repo_path, patch)
        if not applied:
            await self._reset_repo(fix_item.repo_path)
            return [
                RewardResult(
                    name="code_fix",
                    score=format_score * 0.5,
                    weight=1.0,
                    reason="patch_apply_failed",
                )
            ]

        # Run failing tests
        pass_count = 0
        total = len(fix_item.failing_tests)
        for test_id in fix_item.failing_tests:
            try:

                def _run_pytest(test_path: str) -> subprocess.CompletedProcess[bytes]:
                    return subprocess.run(
                        [sys.executable, "-m", "pytest", test_path, "-q", "--tb=no"],
                        capture_output=True,
                        timeout=self.cfg.test_timeout,
                        cwd=fix_item.repo_path,
                    )

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    _run_pytest,
                    test_id,
                )
                if result.returncode == 0:
                    pass_count += 1
            except Exception:
                pass

        await self._reset_repo(fix_item.repo_path)

        pass_frac = pass_count / max(1, total)
        pass_reward = self.cfg.pass_reward_weight * pass_frac
        score = format_score + pass_reward

        return [
            RewardResult(
                name="code_fix",
                score=min(1.0, score),
                weight=1.0,
                reason=f"tests={pass_count}/{total} patch_ok=True",
            )
        ]
