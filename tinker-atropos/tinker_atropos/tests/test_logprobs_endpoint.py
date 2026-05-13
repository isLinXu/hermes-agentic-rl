import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    trainer.base_model = "meta-llama/Llama-3.1-8B-Instruct"

    trainer.tokenizer = MagicMock()
    trainer.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])
    trainer.tokenizer.decode = MagicMock(side_effect=lambda tokens: f"tok_{tokens[0]}")

    trainer.current_sampling_client = MagicMock()
    trainer.current_sampling_client.compute_logprobs_async = AsyncMock(
        return_value=[-0.1, -0.5, -0.3, -0.8, -0.2]
    )

    return trainer


@pytest.fixture
def app_with_trainer(mock_trainer):
    import tinker_atropos.trainer as trainer_module

    original_trainer = trainer_module.trainer
    trainer_module.trainer = mock_trainer
    yield trainer_module.app
    trainer_module.trainer = original_trainer


@pytest.fixture
def client(app_with_trainer):
    return TestClient(app_with_trainer)


class TestLogprobsEndpoint:
    def test_logprobs_with_input_ids(self, client, mock_trainer):
        """Basic logprobs request with raw token IDs."""
        response = client.post("/logprobs", json={"input_ids": [1, 2, 3, 4, 5]})
        assert response.status_code == 200

        data = response.json()
        assert data["num_tokens"] == 5
        assert len(data["logprobs"]) == 5
        assert data["logprobs"][0]["token_id"] == 1
        assert data["logprobs"][0]["logprob"] == pytest.approx(-0.1)
        assert data["logprobs"][0]["token"] is None  # return_text=False by default

    def test_logprobs_with_text(self, client, mock_trainer):
        """Logprobs from text input (tokenized server-side)."""
        response = client.post("/logprobs", json={"text": "Hello world"})
        assert response.status_code == 200

        data = response.json()
        assert data["num_tokens"] == 5
        mock_trainer.tokenizer.encode.assert_called_once_with(
            "Hello world", add_special_tokens=False
        )

    def test_logprobs_with_return_text(self, client, mock_trainer):
        """Logprobs with decoded token strings."""
        response = client.post(
            "/logprobs",
            json={"input_ids": [10, 20, 30], "return_text": True},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["logprobs"][0]["token"] == "tok_10"
        assert data["logprobs"][1]["token"] == "tok_20"
        assert data["logprobs"][2]["token"] == "tok_30"

    def test_logprobs_no_input(self, client):
        """Should 400 when neither input_ids nor text provided."""
        response = client.post("/logprobs", json={})
        assert response.status_code == 400
        assert (
            "input_ids" in response.json()["detail"].lower()
            or "text" in response.json()["detail"].lower()
        )

    def test_logprobs_empty_input(self, client):
        """Should 400 on empty token list."""
        response = client.post("/logprobs", json={"input_ids": []})
        assert response.status_code == 400
        assert "at least one token" in response.json()["detail"].lower()

    def test_logprobs_without_trainer(self, app_with_trainer):
        """Should 503 when trainer not initialized."""
        import tinker_atropos.trainer as trainer_module

        original = trainer_module.trainer
        trainer_module.trainer = None
        try:
            client = TestClient(app_with_trainer)
            response = client.post("/logprobs", json={"input_ids": [1, 2, 3]})
            assert response.status_code == 503
            assert "Trainer not initialized" in response.json()["detail"]
        finally:
            trainer_module.trainer = original

    def test_logprobs_handles_none_values(self, client, mock_trainer):
        """Should handle None logprob values from compute_logprobs."""
        mock_trainer.current_sampling_client.compute_logprobs_async = AsyncMock(
            return_value=[None, -0.5, None]
        )
        response = client.post("/logprobs", json={"input_ids": [1, 2, 3]})
        assert response.status_code == 200

        data = response.json()
        assert data["logprobs"][0]["logprob"] == 0.0  # None -> 0.0
        assert data["logprobs"][1]["logprob"] == pytest.approx(-0.5)
        assert data["logprobs"][2]["logprob"] == 0.0

    def test_logprobs_single_token(self, client, mock_trainer):
        """Edge case: single token input."""
        mock_trainer.current_sampling_client.compute_logprobs_async = AsyncMock(return_value=[-1.5])
        response = client.post("/logprobs", json={"input_ids": [42]})
        assert response.status_code == 200

        data = response.json()
        assert data["num_tokens"] == 1
        assert len(data["logprobs"]) == 1
        assert data["logprobs"][0]["token_id"] == 42
        assert data["logprobs"][0]["logprob"] == pytest.approx(-1.5)

    def test_logprobs_compute_failure(self, client, mock_trainer):
        """Should 500 if compute_logprobs raises."""
        mock_trainer.current_sampling_client.compute_logprobs_async = AsyncMock(
            side_effect=RuntimeError("tinker exploded")
        )
        response = client.post("/logprobs", json={"input_ids": [1, 2, 3]})
        assert response.status_code == 500
        assert "tinker exploded" in response.json()["detail"]

    def test_logprobs_input_ids_takes_precedence(self, client, mock_trainer):
        """When both input_ids and text given, input_ids wins."""
        response = client.post(
            "/logprobs",
            json={"input_ids": [99, 100], "text": "should be ignored"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["num_tokens"] == 2
        assert data["logprobs"][0]["token_id"] == 99
        # tokenizer.encode should NOT have been called
        mock_trainer.tokenizer.encode.assert_not_called()
