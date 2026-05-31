"""Rule-based OPD / next-state judges for LetterCountingEnv.

These judges are intentionally local and deterministic. They cover the common
feedback shape produced by letter-counting evaluators:

    "Wrong answer; correct answer is 7"
    "expected: <answer>{\"a\": 2}</answer>"

The OPD judge extracts a directive hint. The next-state PRM judge also compares
the response against the expected answer when it can, so hints containing words
like "correct answer" do not accidentally flip a BAD vote to GOOD.
"""

from __future__ import annotations

import json
import re

from hermes_agentic_rl.algos.opd import OPDJudge, wrap_hint
from hermes_agentic_rl.rewards.next_state_prm import BAD, GOOD, NEUTRAL, NextStateJudge, PRMVote


def extract_letter_counting_expected_answer(next_state: str) -> str | None:
    """Extract the expected answer payload from textual next-state feedback."""
    text = str(next_state or "").strip()
    if not text:
        return None

    patterns = [
        r"(?:correct|expected)(?:\s+answer)?\s*(?:is|=|:)\s*<answer>(.*?)</answer>",
        r"(?:correct|expected)(?:\s+answer)?\s*(?:is|=|:)\s*(\{.*?\})",
        r"(?:correct|expected)(?:\s+answer)?\s*(?:is|=|:)\s*(-?\d+)",
        r"answer\s*(?:is|=|:)\s*(-?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    tag_match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip()
    return None


def build_letter_counting_hint(expected_answer: str) -> str:
    """Build a concise OPD hint for the expected answer payload."""
    payload = expected_answer.strip()
    return (
        f"The expected answer is <answer>{payload}</answer>. "
        "Return exactly that answer format."
    )


async def letter_counting_opd_judge_fn(response: str, next_state: str) -> str | None:
    """Rule function compatible with OPDJudge."""
    del response
    expected = extract_letter_counting_expected_answer(next_state)
    if expected is None:
        return None
    return wrap_hint(build_letter_counting_hint(expected))


class LetterCountingOPDJudge(OPDJudge):
    """OPD hint extractor for LetterCountingEnv feedback."""

    def __init__(self) -> None:
        super().__init__(letter_counting_opd_judge_fn)


class LetterCountingNextStateJudge(NextStateJudge):
    """GOOD/BAD/NEUTRAL next-state judge for LetterCountingEnv feedback."""

    def __init__(self) -> None:
        super().__init__(self._judge)

    async def _judge(self, response: str, next_state: str) -> PRMVote:
        expected = extract_letter_counting_expected_answer(next_state)
        if expected is not None:
            hint = build_letter_counting_hint(expected)
            actual = _extract_response_answer(response)
            vote = GOOD if actual is not None and _answers_equal(actual, expected) else BAD
            return PRMVote(vote=vote, hint=hint, raw=_label(vote))

        state_upper = str(next_state or "").upper()
        if any(marker in state_upper for marker in ("WRONG", "INCORRECT", "FAIL")):
            return PRMVote(vote=BAD, hint=None, raw="BAD")
        if any(marker in state_upper for marker in ("CORRECT", "PASS", "SUCCESS")):
            return PRMVote(vote=GOOD, hint=None, raw="GOOD")
        return PRMVote(vote=NEUTRAL, hint=None, raw="NEUTRAL")


def _extract_response_answer(response: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", str(response or ""), re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _answers_equal(actual: str, expected: str) -> bool:
    actual_norm = _normalise_answer(actual)
    expected_norm = _normalise_answer(expected)
    return actual_norm == expected_norm


def _normalise_answer(value: str) -> object:
    stripped = value.strip()
    tag_match = re.fullmatch(
        r"<answer>(.*?)</answer>",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tag_match:
        stripped = tag_match.group(1).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return int(stripped)
        except ValueError:
            return stripped
    if isinstance(parsed, dict):
        return {str(key): _normalise_json_value(val) for key, val in sorted(parsed.items())}
    return parsed


def _normalise_json_value(value: object) -> object:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return value


def _label(vote: int) -> str:
    if vote == GOOD:
        return "GOOD"
    if vote == BAD:
        return "BAD"
    return "NEUTRAL"
