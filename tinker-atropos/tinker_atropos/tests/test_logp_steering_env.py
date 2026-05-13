"""Tests for the logp_steering environment logic (no network, no tinker)."""
import pytest


class TestExtractFirstTurn:
    """Test _extract_first_turn helper."""

    def _make_env(self):
        """Create a minimal LogpSteeringEnv without full init."""
        from tinker_atropos.environments.logp_steering import LogpSteeringEnv

        env = object.__new__(LogpSteeringEnv)
        return env

    def test_single_user_message(self):
        env = self._make_env()
        item = {
            "conversation": [
                {"role": "user", "content": "Hello there"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        result = env._extract_first_turn(item)
        assert result is not None
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello there"

    def test_system_then_user(self):
        env = self._make_env()
        item = {
            "conversation": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What's up"},
                {"role": "assistant", "content": "Not much"},
            ]
        }
        result = env._extract_first_turn(item)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_empty_conversation(self):
        env = self._make_env()
        item = {"conversation": []}
        result = env._extract_first_turn(item)
        assert result is None

    def test_no_user_message(self):
        env = self._make_env()
        item = {
            "conversation": [
                {"role": "system", "content": "You are helpful"},
            ]
        }
        result = env._extract_first_turn(item)
        assert result is None

    def test_multi_turn_only_takes_first(self):
        env = self._make_env()
        item = {
            "conversation": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
                {"role": "assistant", "content": "Second answer"},
            ]
        }
        result = env._extract_first_turn(item)
        assert len(result) == 1
        assert result[0]["content"] == "First question"

    def test_missing_conversation_key(self):
        env = self._make_env()
        item = {}
        result = env._extract_first_turn(item)
        assert result is None


class TestSteeringPrefixConstruction:
    """Test that steering prefix + student tokens produces valid teacher input."""

    def test_prefix_prepend(self):
        """Teacher tokens = prefix + student tokens, no retokenization."""
        prefix = [151644, 8948, 198, 100, 200, 151645, 198]
        student_tokens = [151644, 872, 198, 50, 60, 151645, 198, 151644, 77091, 198, 70, 80]

        teacher_tokens = prefix + student_tokens
        assert len(teacher_tokens) == len(prefix) + len(student_tokens)
        # student tokens are exactly preserved at the end
        assert teacher_tokens[-len(student_tokens) :] == student_tokens

    def test_alignment_tail_slice(self):
        """Taking last N from teacher logprobs gives exact student alignment."""
        prefix_len = 25
        student_len = 100
        teacher_logprobs = [-0.5] * (prefix_len + student_len)

        aligned = teacher_logprobs[-student_len:]
        assert len(aligned) == student_len

    def test_distill_format_k1(self):
        """distill arrays should be [seq_len][1] for tinker K=1."""
        student_tokens = [1, 2, 3, 4, 5]
        teacher_lps = [-0.5, -0.3, -0.2, -0.4, -0.1]

        distill_ids = [[tid] for tid in student_tokens]
        distill_lps = [[lp] for lp in teacher_lps]

        assert len(distill_ids) == len(student_tokens)
        assert len(distill_lps) == len(teacher_lps)
        assert all(len(row) == 1 for row in distill_ids)
        assert all(len(row) == 1 for row in distill_lps)

    def test_score_computation(self):
        """Score = mean(teacher - student) on non-prompt tokens."""
        student_lps = [1.0, 1.0, -0.5, -0.3, -0.2]  # 2 prompt, 3 gen
        teacher_lps = [-0.1, -0.1, -0.8, -0.6, -0.4]

        diffs = []
        for s, t in zip(student_lps, teacher_lps):
            if s != 1.0:
                diffs.append(t - s)

        score = sum(diffs) / len(diffs)
        # (-0.8 - -0.5) + (-0.6 - -0.3) + (-0.4 - -0.2) = -0.3 + -0.3 + -0.2 = -0.8
        # -0.8 / 3 ≈ -0.267
        assert score == pytest.approx(-0.8 / 3, rel=1e-6)
        assert len(diffs) == 3  # only gen tokens counted
