from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward


class FileSystemVerifierReward(BaseReward):
    name = "filesystem_verifier_reward"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        del trajectory, tool_context

        expected_files = item.get("expected_files")
        if not expected_files:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="no expected_files provided; skipped",
                weight=0.0,
                metadata={"checked_files": 0, "passed_files": 0, "failures": []},
            )

        checked = 0
        passed = 0
        failures: list[dict[str, str]] = []
        root = Path.cwd()

        for spec in expected_files:
            checked += 1
            path = str(spec.get("path", "")).strip()
            if not path:
                failures.append({"path": "", "reason": "missing path"})
                continue

            full_path = root / path
            if not full_path.exists():
                failures.append({"path": path, "reason": "file missing"})
                continue

            try:
                size_bytes = full_path.stat().st_size
            except Exception as exc:
                failures.append({"path": path, "reason": f"stat failed: {exc}"})
                continue

            min_bytes = spec.get("min_bytes")
            if min_bytes is not None:
                try:
                    if size_bytes < int(min_bytes):
                        failures.append({"path": path, "reason": "file too small"})
                        continue
                except Exception:
                    failures.append({"path": path, "reason": "invalid min_bytes"})
                    continue

            max_bytes = spec.get("max_bytes")
            if max_bytes is not None:
                try:
                    if size_bytes > int(max_bytes):
                        failures.append({"path": path, "reason": "file too large"})
                        continue
                except Exception:
                    failures.append({"path": path, "reason": "invalid max_bytes"})
                    continue

            try:
                content = full_path.read_text(encoding="utf-8")
            except Exception as exc:
                failures.append({"path": path, "reason": f"read failed: {exc}"})
                continue

            equals = spec.get("equals")
            contains = spec.get("contains")
            regex = spec.get("regex")
            has_any_rule = any(
                value is not None for value in (equals, contains, regex, min_bytes, max_bytes)
            )
            if equals is not None:
                if content.strip() == str(equals).strip():
                    # do not early-continue; allow combining with other rules (contains/regex)
                    pass
                else:
                    failures.append({"path": path, "reason": "content mismatch (equals)"})
                    continue

            if contains is not None:
                if str(contains) not in content:
                    failures.append({"path": path, "reason": "content mismatch (contains)"})
                    continue

            if regex is not None:
                try:
                    if re.search(str(regex), content) is None:
                        failures.append({"path": path, "reason": "content mismatch (regex)"})
                        continue
                except re.error as exc:
                    failures.append({"path": path, "reason": f"invalid regex: {exc}"})
                    continue

            if not has_any_rule:
                failures.append({"path": path, "reason": "no verification rule provided"})
                continue

            passed += 1

        score = 1.0 if checked > 0 and passed == checked else 0.0
        reason = "all expected_files verified" if score == 1.0 else "file verification failed"
        return RewardResult(
            name=self.name,
            score=score,
            reason=reason,
            weight=self.weight,
            metadata={"checked_files": checked, "passed_files": passed, "failures": failures},
        )
