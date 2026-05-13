from __future__ import annotations

import io
import json

from hermes_agentic_rl.datasets.hf_loader import load_hf_dataset


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def test_load_hf_dataset_streaming_uses_rows_endpoint(monkeypatch) -> None:
    payload = {
        "rows": [
            {"row": {"task_id": "trace-1", "value": "a"}},
            {"row": {"task_id": "trace-2", "value": "b"}},
        ]
    }

    def fake_urlopen(url: str, timeout: int = 30):
        assert "datasets-server.huggingface.co/rows" in url
        assert "dataset=lambda%2Frepo" in url
        assert "config=kimi" in url
        assert "length=2" in url
        assert timeout == 30
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("hermes_agentic_rl.datasets.hf_loader.urlopen", fake_urlopen)

    rows = load_hf_dataset(
        "lambda/repo",
        config_name="kimi",
        split="train",
        limit=2,
        streaming=True,
    )

    assert [row["task_id"] for row in rows] == ["trace-1", "trace-2"]


def test_load_hf_dataset_streaming_pages_rows_endpoint(monkeypatch) -> None:
    requested_urls: list[str] = []

    def fake_urlopen(url: str, timeout: int = 30):
        requested_urls.append(url)
        if "offset=0" in url:
            rows = [{"row": {"task_id": f"trace-{idx}"}} for idx in range(100)]
        elif "offset=100" in url:
            rows = [{"row": {"task_id": f"trace-{idx}"}} for idx in range(100, 128)]
        else:
            rows = []
        assert timeout == 30
        return _FakeResponse(json.dumps({"rows": rows}).encode("utf-8"))

    monkeypatch.setattr("hermes_agentic_rl.datasets.hf_loader.urlopen", fake_urlopen)

    rows = load_hf_dataset(
        "lambda/repo",
        config_name="kimi",
        split="train",
        limit=128,
        streaming=True,
    )

    assert len(rows) == 128
    assert "length=100" in requested_urls[0]
    assert "offset=0" in requested_urls[0]
    assert "length=28" in requested_urls[1]
    assert "offset=100" in requested_urls[1]
