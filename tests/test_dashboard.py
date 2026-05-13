"""Smoke test for the pure-stdlib live dashboard."""

from __future__ import annotations

import json
import socket
import urllib.request

from hermes_agentic_rl.monitor.dashboard import LiveDashboard


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_dashboard_serves_metrics_and_html():
    port = _free_port()
    with LiveDashboard(host="127.0.0.1", port=port) as d:
        d.record({"iter": 0, "mean_reward": 0.1, "loss": 0.5, "algo": "grpo"})
        d.record({"iter": 1, "mean_reward": 0.2, "loss": 0.4, "algo": "grpo"})
        # HTML endpoint
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:
            body = r.read().decode("utf-8")
            assert "hermes-agentic-rl" in body
            assert "chart.js" in body.lower()
        # metrics endpoint
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics") as r:
            data = json.loads(r.read())
            assert len(data) == 2
            assert data[-1]["iter"] == 1
            assert data[-1]["mean_reward"] == 0.2
