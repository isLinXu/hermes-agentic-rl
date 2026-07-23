from __future__ import annotations

import asyncio
import json

from hermes_agentic_rl.rewards.letter_counting_judge import (
    LetterCountingNextStateJudge,
    LetterCountingOPDJudge,
    extract_letter_counting_expected_answer,
)
from hermes_agentic_rl.rewards.llm_judge import (
    OpenAICompatibleJudgeConfig,
    OpenAICompatibleNextStateJudge,
    OpenAICompatibleOPDJudge,
)
from hermes_agentic_rl.rewards.next_state_prm import BAD


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_opd_judge_extracts_hint(monkeypatch):
    captured: dict = {}

    def _urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.headers)
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "[HINT_START]Use the required tool-call format.[HINT_END]"
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("JUDGE_KEY", "secret-key")
    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    judge = OpenAICompatibleOPDJudge(
        OpenAICompatibleJudgeConfig(
            model="judge-model",
            base_url="http://judge.local/v1",
            api_key_env="JUDGE_KEY",
            retries=0,
            timeout=3.0,
        )
    )

    hint = asyncio.run(judge.extract_hint("bad response", "format error"))

    assert hint == "Use the required tool-call format."
    assert captured["url"] == "http://judge.local/v1/chat/completions"
    assert captured["timeout"] == 3.0
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["payload"]["model"] == "judge-model"


def test_openai_compatible_next_state_judge_parses_vote(monkeypatch):
    def _urlopen(_req, timeout):
        del timeout
        return _FakeHTTPResponse(
            {"choices": [{"message": {"content": "BAD\n[HINT_START]Return valid JSON.[HINT_END]"}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    judge = OpenAICompatibleNextStateJudge(
        OpenAICompatibleJudgeConfig(
            model="judge-model",
            base_url="http://judge.local/v1",
            api_key=None,
            api_key_env=None,
            retries=0,
        )
    )

    vote = asyncio.run(judge.judge("not json", "parser failed"))

    assert vote.vote == BAD
    assert vote.hint == "Return valid JSON."


def test_next_state_parser_prioritizes_explicit_bad_vote_with_correct_hint():
    from hermes_agentic_rl.rewards.next_state_prm import NextStateJudge

    async def _judge(_response: str, _next_state: str) -> str:
        return "BAD\n[HINT_START]The correct answer is <answer>4</answer>.[HINT_END]"

    vote = asyncio.run(NextStateJudge(_judge).judge("bad", "wrong"))

    assert vote.vote == BAD
    assert vote.hint == "The correct answer is <answer>4</answer>."


def test_letter_counting_opd_judge_extracts_expected_answer_hint():
    judge = LetterCountingOPDJudge()

    hint = asyncio.run(
        judge.extract_hint("<answer>3</answer>", "Wrong answer. The correct answer is 4.")
    )

    assert hint == ("The expected answer is <answer>4</answer>. Return exactly that answer format.")


def test_letter_counting_next_state_judge_marks_mismatch_bad():
    judge = LetterCountingNextStateJudge()

    vote = asyncio.run(judge.judge("<answer>3</answer>", "Wrong answer. correct answer is 4"))

    assert vote.vote == BAD
    assert vote.hint is not None
    assert "<answer>4</answer>" in vote.hint


def test_letter_counting_next_state_judge_marks_matching_json_good():
    from hermes_agentic_rl.rewards.next_state_prm import GOOD

    judge = LetterCountingNextStateJudge()

    vote = asyncio.run(
        judge.judge(
            '<answer>{"b": 2, "a": 1}</answer>',
            'expected answer: {"a": 1, "b": 2}',
        )
    )

    assert vote.vote == GOOD


def test_letter_counting_expected_answer_extractor_supports_answer_tags():
    assert extract_letter_counting_expected_answer("expected: <answer>7</answer>") == "7"


def test_letter_counting_expected_answer_prefers_correct_marker_over_prior_tag():
    assert (
        extract_letter_counting_expected_answer(
            "Your answer <answer>3</answer> was wrong. Correct answer is 4."
        )
        == "4"
    )
