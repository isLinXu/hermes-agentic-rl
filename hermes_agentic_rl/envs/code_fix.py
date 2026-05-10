"""SWE-bench style code environment: bug-fixing with sandbox execution.

This is a lightweight code-fix environment for RL training. It presents the
model with a buggy function, runs the generated fix in a sandboxed Python
subprocess, and returns a reward based on test pass rate.

Key design choices:
  - Pure stdlib sandbox: subprocess with timeout, no Docker dependency.
  - Deterministic tests: each bug has N known test cases.
  - Reward decomposition: reward = test_pass_ratio + format_bonus.
  - Curriculum difficulty: bugs range from simple (typo fix) to complex
    (logic error in multi-function module).

Usage (standalone)::

    env = CodeFixEnv(seed=42, n_samples=100)
    item = await env.get_next_item()
    prompt = env.format_prompt(item)
    # ... generate response ...
    reward = env.compute_reward(item, response_text)
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.core.reward_manager import BaseReward, RewardResult, RewardSummary
from hermes_agentic_rl.envs.base_env import BaseEnv

# ---------------------------------------------------------------------------
# Bug database (curriculum levels 0-4)
# ---------------------------------------------------------------------------

@dataclass
class CodeBug:
    """A single bug-fixing task."""
    bug_id: str
    level: int  # 0=easy (typo), 1=medium (missing line), 2=hard (logic), 3=very hard
    description: str
    buggy_code: str
    test_code: str  # assert-based test, returns True if all pass
    expected_keywords: list[str] = field(default_factory=list)


# Built-in bug library
_BUILTIN_BUGS: list[CodeBug] = [
    # Level 0: typo / simple syntax
    CodeBug(
        bug_id="typo_add",
        level=0,
        description="Fix the typo in the function that adds two numbers.",
        buggy_code=textwrap.dedent("""\
        def add(a, b):
            retrun a + b  # typo: retrun → return
        """),
        test_code=textwrap.dedent("""\
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(0, 0) == 0
        """),
        expected_keywords=["return", "add"],
    ),
    CodeBug(
        bug_id="typo_multiply",
        level=0,
        description="Fix the typo in the function that multiplies two numbers.",
        buggy_code=textwrap.dedent("""\
        def multiply(a, b):
            resutl = a * b  # typo: resutl → result
            return resutl
        """),
        test_code=textwrap.dedent("""\
        assert multiply(3, 4) == 12
        assert multiply(-2, 5) == -10
        assert multiply(0, 100) == 0
        """),
        expected_keywords=["result", "return"],
    ),
    # Level 1: missing line / incomplete logic
    CodeBug(
        bug_id="missing_abs",
        level=1,
        description="Fix the absolute value function. It currently returns the input unchanged.",
        buggy_code=textwrap.dedent("""\
        def my_abs(x):
            # missing: handle negative case
            return x
        """),
        test_code=textwrap.dedent("""\
        assert my_abs(5) == 5
        assert my_abs(-5) == 5
        assert my_abs(0) == 0
        assert my_abs(-3.14) == 3.14
        """),
        expected_keywords=["abs", "negative", "-x"],
    ),
    CodeBug(
        bug_id="missing_max_check",
        level=1,
        description="Fix the max-of-three function. It only compares the first two.",
        buggy_code=textwrap.dedent("""\
        def max_of_three(a, b, c):
            if a > b:
                return a
            return b
        """),
        test_code=textwrap.dedent("""\
        assert max_of_three(1, 2, 3) == 3
        assert max_of_three(3, 2, 1) == 3
        assert max_of_three(2, 3, 1) == 3
        assert max_of_three(5, 5, 5) == 5
        """),
        expected_keywords=["c", "max"],
    ),
    # Level 2: logic error
    CodeBug(
        bug_id="logic_fib",
        level=2,
        description="Fix the Fibonacci function. It returns wrong values for n >= 2.",
        buggy_code=textwrap.dedent("""\
        def fibonacci(n):
            if n <= 0:
                return 0
            if n == 1:
                return 1
            return fibonacci(n - 1) + fibonacci(n)  # bug: infinite recursion
        """),
        test_code=textwrap.dedent("""\
        assert fibonacci(0) == 0
        assert fibonacci(1) == 1
        assert fibonacci(2) == 1
        assert fibonacci(5) == 5
        assert fibonacci(10) == 55
        """),
        expected_keywords=["fibonacci", "n-2", "recursive"],
    ),
    CodeBug(
        bug_id="logic_is_prime",
        level=2,
        description="Fix the prime-check function. It incorrectly classifies some numbers.",
        buggy_code=textwrap.dedent("""\
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, n):
                if n % i == 0:
                    return True  # bug: returns True for composites
            return False
        """),
        test_code=textwrap.dedent("""\
        assert is_prime(2) == True
        assert is_prime(3) == True
        assert is_prime(4) == False
        assert is_prime(17) == True
        assert is_prime(100) == False
        """),
        expected_keywords=["False", "prime", "True"],
    ),
    # Level 3: multi-function / complex logic
    CodeBug(
        bug_id="complex_binary_search",
        level=3,
        description="Fix the binary search function. It has an off-by-one error.",
        buggy_code=textwrap.dedent("""\
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid  # bug: should be mid + 1
                else:
                    right = mid  # bug: should be mid - 1
            return -1
        """),
        test_code=textwrap.dedent("""\
        assert binary_search([1, 2, 3, 4, 5], 3) == 2
        assert binary_search([1, 2, 3, 4, 5], 1) == 0
        assert binary_search([1, 2, 3, 4, 5], 5) == 4
        assert binary_search([1, 2, 3, 4, 5], 6) == -1
        assert binary_search([1], 1) == 0
        assert binary_search([1], 2) == -1
        """),
        expected_keywords=["mid+1", "mid-1", "binary"],
    ),
    CodeBug(
        bug_id="complex_merge_sorted",
        level=3,
        description="Fix the merge-two-sorted-lists function. It skips some elements.",
        buggy_code=textwrap.dedent("""\
        def merge_sorted(a, b):
            result = []
            i = j = 0
            while i < len(a) and j < len(b):
                if a[i] <= b[j]:
                    result.append(a[i])
                else:
                    result.append(b[j])
                i += 1  # bug: should increment i only when a[i] is used
                j += 1  # bug: should increment j only when b[j] is used
            result.extend(a[i:])
            result.extend(b[j:])
            return result
        """),
        test_code=textwrap.dedent("""\
        assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
        assert merge_sorted([1], [2]) == [1, 2]
        assert merge_sorted([1, 2], []) == [1, 2]
        assert merge_sorted([], [3, 4]) == [3, 4]
        """),
        expected_keywords=["i", "j", "merge"],
    ),
]


# ---------------------------------------------------------------------------
# Sandbox executor
# ---------------------------------------------------------------------------


def _run_tests_in_sandbox(
    fixed_code: str,
    test_code: str,
    timeout: float = 5.0,
) -> tuple[bool, str]:
    """Execute the fixed code + test code in a subprocess sandbox.

    Returns (all_pass, error_output).
    """
    full_script = textwrap.dedent(f"""\
    import sys
    import traceback
    import builtins
    import os
    import io

    # Restricted builtins: disable dangerous functions
    _safe_builtins = {{
        k: v for k, v in builtins.__dict__.items()
        if k not in {{'open', 'exec', 'eval', 'compile', '__import__',
                     'input', 'breakpoint', 'memoryview'}}
    }}

    # Run in restricted namespace
    _ns = {{"__builtins__": _safe_builtins}}

    try:
        exec({json.dumps(fixed_code)}, _ns)
    except Exception as e:
        print(f"COMPILE_ERROR: {{e}}", file=sys.stderr)
        sys.exit(1)

    errors = []
    try:
        exec({json.dumps(test_code)}, _ns)
    except AssertionError as e:
        errors.append(f"TEST_FAIL: {{e}}")
    except Exception as e:
        errors.append(f"TEST_ERROR: {{type(e).__name__}}: {{e}}")

    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        sys.exit(2)
    sys.exit(0)
    """)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if proc.returncode == 0:
            return True, ""
        return False, proc.stderr.strip()[:500]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: execution exceeded time limit"
    except Exception as e:
        return False, f"SANDBOX_ERROR: {e}"


# ---------------------------------------------------------------------------
# CodeFixEnv
# ---------------------------------------------------------------------------


class CodeFixEnv(BaseEnv):
    """SWE-bench style code-fixing environment for RL training.

    Each item is a buggy function + test suite. The agent must generate a
    corrected version of the function.
    """

    def __init__(
        self,
        bugs: list[CodeBug] | None = None,
        seed: int = 42,
        n_samples: int = 100,
        max_level: int = 3,
    ):
        self._bugs = bugs or _BUILTIN_BUGS
        self._rng = random.Random(seed)
        self._n_samples = n_samples
        self._max_level = max_level
        self._items: list[dict[str, Any]] = self._generate_items()

    def _generate_items(self) -> list[dict[str, Any]]:
        items = []
        available = [b for b in self._bugs if b.level <= self._max_level]
        if not available:
            available = self._bugs[:1]

        for i in range(self._n_samples):
            bug = self._rng.choice(available)
            items.append({
                "id": f"bug_{i:04d}",
                "bug_id": bug.bug_id,
                "level": bug.level,
                "description": bug.description,
                "buggy_code": bug.buggy_code,
                "test_code": bug.test_code,
                "expected_keywords": bug.expected_keywords,
            })
        return items

    async def get_next_item(self) -> dict[str, Any]:
        return self._rng.choice(self._items)

    def format_prompt(self, item: dict[str, Any]) -> str:
        return textwrap.dedent(f"""\
        You are a code-fixing assistant. Below is a buggy Python function.
        Output ONLY the corrected function code inside <code>...</code> tags.

        Description: {item['description']}

        Buggy code:
        ```
        {item['buggy_code'].strip()}
        ```

        Please output the fixed code:
        <code>
        """).strip()

    def compute_reward(self, item: dict[str, Any], response: str) -> float:
        """Compute reward: test_pass_ratio (0-1) + format_bonus (0-0.2)."""
        # Extract code from response
        import re

        m = re.search(r"<code>(.*?)</code>", response, re.DOTALL | re.IGNORECASE)
        if not m:
            # Try without tags
            m = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
        if not m:
            # No code found: partial credit for mentioning expected keywords
            kw_bonus = sum(
                0.1 for kw in item.get("expected_keywords", [])
                if kw.lower() in response.lower()
            )
            return min(kw_bonus, 0.3)

        fixed_code = m.group(1).strip()
        if not fixed_code:
            return 0.1  # empty code block

        # Sandbox execution
        all_pass, error = _run_tests_in_sandbox(
            fixed_code, item["test_code"], timeout=5.0
        )

        if all_pass:
            return 1.0

        # Partial credit: if code compiles and runs (some tests may pass)
        # Count how many test assertions pass by running each separately
        test_lines = [
            line.strip()
            for line in item["test_code"].strip().split("\n")
            if line.strip().startswith("assert")
        ]
        if not test_lines:
            return 0.2  # no tests defined

        passed = 0
        for test_line in test_lines:
            all_pass_single, _ = _run_tests_in_sandbox(
                fixed_code, test_line, timeout=3.0
            )
            if all_pass_single:
                passed += 1

        return passed / max(len(test_lines), 1)

    def compute_sequence_reward(
        self, item: dict[str, Any], sequence: str
    ) -> dict[str, Any]:
        reward = self.compute_reward(item, sequence)
        return {
            "reward": reward,
            "passed": reward >= 1.0,
            "partial": 0 < reward < 1.0,
        }


# ---------------------------------------------------------------------------
# Reward component for CodeFixEnv
# ---------------------------------------------------------------------------


class CodeFixReward(BaseReward):
    """Reward component for code-fix tasks."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Any,
        tool_context: Any = None,
    ) -> RewardResult:
        # Extract final response text from trajectory
        if hasattr(trajectory, "response_text"):
            text = trajectory.response_text
        elif hasattr(trajectory, "final_output"):
            text = trajectory.final_output
        elif isinstance(trajectory, str):
            text = trajectory
        else:
            text = str(trajectory)

        score = _compute_codefix_score(item, text)
        return RewardResult(
            score=score * self.weight,
            components={"codefix_score": score},
        )


def _compute_codefix_score(item: dict[str, Any], response: str) -> float:
    """Compute code-fix score from response text."""
    import re

    m = re.search(r"<code>(.*?)</code>", response, re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    if not m:
        return 0.0

    fixed_code = m.group(1).strip()
    all_pass, _ = _run_tests_in_sandbox(fixed_code, item["test_code"], timeout=5.0)
    return 1.0 if all_pass else 0.0


# ---------------------------------------------------------------------------
# RewardSummary for integration
# ---------------------------------------------------------------------------


class CodeFixRewardSummary(RewardSummary):
    """Reward summary for code-fix training."""

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Any,
        tool_context: Any = None,
    ) -> RewardResult:
        if hasattr(trajectory, "response_text"):
            text = trajectory.response_text
        elif hasattr(trajectory, "final_output"):
            text = trajectory.final_output
        else:
            text = str(trajectory)
        score = _compute_codefix_score(item, text)
        return RewardResult(
            score=score,
            components={"codefix_score": score},
        )
