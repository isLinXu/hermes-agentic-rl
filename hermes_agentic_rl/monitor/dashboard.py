"""Pure-stdlib live training dashboard.

Starts an HTTP server on a background thread; exposes:

    GET /           → HTML page with Chart.js (CDN) rendering live curves
    GET /metrics    → JSON list of all records recorded so far

The dashboard is opt-in — callers pass ``metrics_sink=dashboard.record`` to
the trainer config. If the dashboard is never started, the sink is a no-op.
Zero external dependencies (Chart.js loaded from CDN, not vendored).
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>hermes-agentic-rl live dashboard</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 24px; }
  h1 { font-size: 18px; margin: 0 0 16px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .panel { border: 1px solid #ddd; border-radius: 6px; padding: 12px; }
  canvas { max-height: 260px; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 10px;
         background: #eef; color: #226; font-size: 12px; margin-right: 6px; }
</style>
</head>
<body>
<h1>hermes-agentic-rl <span class="tag" id="algo">algo</span><span class="tag" id="iter">iter=—</span></h1>
<div class="grid">
  <div class="panel"><canvas id="reward"></canvas></div>
  <div class="panel"><canvas id="loss"></canvas></div>
  <div class="panel"><canvas id="kl"></canvas></div>
  <div class="panel"><canvas id="value"></canvas></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
const make = (id, label, color) => new Chart(document.getElementById(id), {
  type: 'line',
  data: { labels: [], datasets: [{ label, data: [], borderColor: color, tension: 0.2, pointRadius: 0 }] },
  options: { animation: false, responsive: true, scales: { y: { beginAtZero: false } } }
});
const charts = {
  reward: make('reward', 'mean_reward', '#0a7'),
  loss:   make('loss',   'loss',        '#c40'),
  kl:     make('kl',     'kl',          '#a0a'),
  value:  make('value',  'value_loss',  '#09c'),
};
async function tick() {
  try {
    const r = await fetch('/metrics');
    const data = await r.json();
    for (const k of ['reward','loss','kl','value']) {
      charts[k].data.labels = data.map(d => d.iter);
    }
    charts.reward.data.datasets[0].data = data.map(d => d.mean_reward ?? 0);
    charts.loss.data.datasets[0].data   = data.map(d => d.loss ?? 0);
    charts.kl.data.datasets[0].data     = data.map(d => d.kl ?? 0);
    charts.value.data.datasets[0].data  = data.map(d => d.value_loss ?? 0);
    for (const k of Object.keys(charts)) charts[k].update('none');
    if (data.length) {
      document.getElementById('iter').textContent = 'iter=' + data[data.length - 1].iter;
      document.getElementById('algo').textContent = data[data.length - 1].algo || 'algo';
    }
  } catch (e) { console.error(e); }
  setTimeout(tick, 1000);
}
tick();
</script>
</body></html>
"""


class LiveDashboard:
    """Thread-safe ring buffer + HTTP server for training metrics."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8765, maxlen: int = 5000) -> None:
        self.host = host
        self.port = port
        self._records: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # --- public API ---

    def record(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self._records.append(dict(rec))

    def start(self) -> str:
        if self._server is not None:
            return self.url
        dashboard = self
        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:  # silence stderr spam
                return
            def do_GET(self) -> None:
                if self.path == "/metrics":
                    with dashboard._lock:
                        body = json.dumps(list(dashboard._records)).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path in ("/", "/index.html"):
                    body = _HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(404)
        self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def __enter__(self) -> LiveDashboard:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


def make_metrics_sink(dashboard: LiveDashboard) -> Callable[[dict[str, Any]], None]:
    return dashboard.record
