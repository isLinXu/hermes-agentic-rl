"""
Integration tests for serve.py — starts it as a real subprocess and hits
it with actual HTTP requests against the real tinker API.

Requires TINKER_API_KEY to be set. Skipped if not available.

Run with: .venv/bin/python -m pytest tinker_atropos/tests/test_serve_integration.py -v -s
"""

import os
import time
import signal
import subprocess

import pytest
import requests

pytestmark = pytest.mark.skipif(
    not os.environ.get("TINKER_API_KEY"),
    reason="TINKER_API_KEY not set — skipping tinker integration tests",
)

SERVE_PORT = 18199  # unusual port to avoid conflicts
BASE_URL = f"http://localhost:{SERVE_PORT}"
MODEL = "Qwen/Qwen3-30B-A3B"
VENV_PYTHON = os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "bin", "python")
SERVE_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "serve.py")


@pytest.fixture(scope="module")
def serve_process():
    """Start serve.py as a real subprocess, wait for it to be healthy, tear down after."""
    env = os.environ.copy()
    proc = subprocess.Popen(
        [VENV_PYTHON, SERVE_SCRIPT, "--model", MODEL, "--port", str(SERVE_PORT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
    )

    # Wait for server to be ready
    deadline = time.time() + 90  # tinker client init can take a while
    healthy = False
    while time.time() < deadline:
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=2)
            if resp.status_code == 200 and resp.json().get("ready"):
                healthy = True
                break
        except (requests.ConnectionError, requests.Timeout):
            pass
        # Check if process died
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            pytest.fail(f"serve.py exited early (code={proc.returncode}):\n{stdout}")
        time.sleep(2)

    if not healthy:
        proc.kill()
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        pytest.fail(f"serve.py didn't become healthy within 90s:\n{stdout[-2000:]}")

    yield proc

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class TestServeHealth:
    def test_health(self, serve_process):
        resp = requests.get(f"{BASE_URL}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == MODEL
        assert data["ready"] is True


class TestServeChatCompletions:
    def test_chat_generates_text(self, serve_process):
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hi in one word."}],
                "max_tokens": 10,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) == 1
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert data["model"] == MODEL

    def test_chat_multi_sample(self, serve_process):
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Pick a number"}],
                "max_tokens": 5,
                "temperature": 1.0,
                "n": 2,
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["choices"]) == 2

    def test_chat_with_system_prompt(self, serve_process):
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You only respond with 'yes'."},
                    {"role": "user", "content": "Is water wet?"},
                ],
                "max_tokens": 10,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["choices"][0]["message"]["content"]) > 0


class TestServeCompletions:
    def test_completion(self, serve_process):
        resp = requests.post(
            f"{BASE_URL}/v1/completions",
            json={
                "prompt": "The capital of France is",
                "max_tokens": 10,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) == 1
        assert len(data["choices"][0]["text"]) > 0

    def test_batch_completion(self, serve_process):
        resp = requests.post(
            f"{BASE_URL}/v1/completions",
            json={
                "prompt": ["Hello", "World"],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["choices"]) == 2


class TestServeLogprobs:
    def test_logprobs_from_ids(self, serve_process):
        # Tokenize something first via completions to get valid token ids,
        # or just use known Qwen token IDs
        resp = requests.post(
            f"{BASE_URL}/logprobs",
            json={
                "text": "Hello world",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_tokens"] > 0
        assert len(data["logprobs"]) == data["num_tokens"]

        # Non-first tokens should have negative logprobs
        for entry in data["logprobs"][1:]:
            assert entry["logprob"] < 0.0
            assert isinstance(entry["token_id"], int)

    def test_logprobs_return_text(self, serve_process):
        resp = requests.post(
            f"{BASE_URL}/logprobs",
            json={
                "text": "The quick brown fox",
                "return_text": True,
            },
        )
        assert resp.status_code == 200
        for entry in resp.json()["logprobs"]:
            assert entry["token"] is not None

    def test_logprobs_empty_400(self, serve_process):
        resp = requests.post(f"{BASE_URL}/logprobs", json={"input_ids": []})
        assert resp.status_code == 400

    def test_logprobs_no_input_400(self, serve_process):
        resp = requests.post(f"{BASE_URL}/logprobs", json={})
        assert resp.status_code == 400

    def test_steering_prefix_changes_logprobs(self, serve_process):
        """The whole point: a system prompt prefix changes the distribution."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)

        bare_tokens = tok.encode("The world is beautiful", add_special_tokens=False)

        sys_tpl = tok.apply_chat_template(
            [{"role": "system", "content": "You speak only about darkness and despair."}],
            tokenize=True,
            add_generation_prompt=False,
        )
        prefix = sys_tpl["input_ids"] if hasattr(sys_tpl, "input_ids") else list(sys_tpl)
        steered_tokens = prefix + bare_tokens

        bare_resp = requests.post(f"{BASE_URL}/logprobs", json={"input_ids": bare_tokens})
        steered_resp = requests.post(f"{BASE_URL}/logprobs", json={"input_ids": steered_tokens})

        assert bare_resp.status_code == 200
        assert steered_resp.status_code == 200

        bare_lps = [e["logprob"] for e in bare_resp.json()["logprobs"]]
        steered_lps = [e["logprob"] for e in steered_resp.json()["logprobs"]]
        steered_aligned = steered_lps[-len(bare_tokens) :]

        diffs = [abs(a - b) for a, b in zip(bare_lps[1:], steered_aligned[1:])]
        max_diff = max(diffs) if diffs else 0
        assert max_diff > 0.001, f"Steering had no effect (max_diff={max_diff})"
