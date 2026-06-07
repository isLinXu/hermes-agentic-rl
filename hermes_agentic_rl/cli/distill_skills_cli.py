"""CLI entry for ``hermes-rl distill-skills``.

A zero-config, one-command path: point it at raw agent traces and get
installable Skill candidate packages + a human-readable report. No YAML,
no model, no training required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hermes_agentic_rl.collectors.distill_skills import distill_skills


def run_distill_skills(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hermes-agentic-rl distill-skills",
        description=(
            "Distill installable Skill candidates from raw agent traces. "
            "No training required."
        ),
    )
    parser.add_argument(
        "--traces",
        "-t",
        nargs="+",
        required=True,
        help="one or more raw trace files (.json / .jsonl)",
    )
    parser.add_argument(
        "--out",
        "-o",
        required=True,
        help="output directory for skill packages + reports",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="minimum per-turn reward to keep a candidate (default: 0.0)",
    )
    parser.add_argument(
        "--status",
        choices=["ready_for_review", "draft", "blocked"],
        default=None,
        help="highlight only skills with this quality status in the report",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the full summary JSON to stdout",
    )
    args = parser.parse_args(argv)

    missing = [p for p in args.traces if not Path(p).exists()]
    if missing:
        print(f"error: trace file(s) not found: {', '.join(missing)}")
        return 1

    summary = distill_skills(
        trace_paths=list(args.traces),
        output_dir=args.out,
        min_reward=args.min_reward,
        status_filter=args.status,
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        mining = summary.get("mining", {})
        q = summary.get("quality", {}).get("status_counts", {})
        print(
            "[distill-skills] "
            f"traces={mining.get('trace_files', 0)} "
            f"turns_mined={mining.get('turns_mined', 0)} "
            f"candidates={summary.get('candidate_records', 0)} "
            f"skills={summary.get('skills_exported', 0)} "
            f"(ready={q.get('ready_for_review', 0)} "
            f"draft={q.get('draft', 0)} blocked={q.get('blocked', 0)})"
        )
        print(f"[distill-skills] report: {summary.get('report_path')}")
        print(f"[distill-skills] output: {summary.get('output_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_distill_skills(sys.argv[1:]))
