from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CapabilityMetric:
    name: str
    weight: float = 1.0


@dataclass(frozen=True)
class CapabilityAxis:
    name: str
    description: str
    metrics: tuple[CapabilityMetric, ...]


DEFAULT_CAPABILITY_AXES: tuple[CapabilityAxis, ...] = (
    CapabilityAxis(
        name="task_success",
        description="Held-out reward and explicit success-threshold performance.",
        metrics=(
            CapabilityMetric("mean_reward", 0.6),
            CapabilityMetric("success_rate", 0.4),
        ),
    ),
    CapabilityAxis(
        name="tool_use_reliability",
        description="Hermes tool-call structure, tool selection, and argument fidelity.",
        metrics=(
            CapabilityMetric("metadata/tool_call_parse_ok", 0.25),
            CapabilityMetric("metadata/tool_name_match", 0.25),
            CapabilityMetric("metadata/argument_key_overlap", 0.25),
            CapabilityMetric("metadata/argument_value_similarity", 0.25),
        ),
    ),
    CapabilityAxis(
        name="interaction_control",
        description="Whether the agent finishes cleanly and exposes the configured success signal.",
        metrics=(
            CapabilityMetric("finished_naturally_rate", 0.5),
            CapabilityMetric("success_metric_found_rate", 0.5),
        ),
    ),
    CapabilityAxis(
        name="self_evolution_signal",
        description="Signals useful for deciding whether experience should become replay data or a skill.",
        metrics=(
            CapabilityMetric("success_score_mean", 0.5),
            CapabilityMetric("mean_reward", 0.5),
        ),
    ),
    CapabilityAxis(
        name="prompt_context",
        description="Long-context retention, constraint preservation, distractor avoidance, and concise synthesis.",
        metrics=(
            CapabilityMetric("metadata/context_required_fact_recall", 0.30),
            CapabilityMetric("metadata/context_constraint_satisfaction", 0.25),
            CapabilityMetric("metadata/context_tool_summary_retention", 0.20),
            CapabilityMetric("metadata/context_distractor_avoidance", 0.15),
            CapabilityMetric("metadata/context_compression_ok", 0.10),
        ),
    ),
)

OBJECTIVE_AXIS_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tool_use_reliability", ("tool", "argument", "parse", "command")),
    ("task_success", ("success", "reward", "score", "completion")),
    ("interaction_control", ("finish", "turn", "recovery", "stop")),
    ("prompt_context", ("prompt", "context", "memory", "compress")),
    ("skill_learning", ("skill", "self_evolution", "self-evolution", "replay")),
)


def normalize_capability_axes(raw: Any) -> list[CapabilityAxis]:
    if raw is False:
        return []
    if raw is None or raw == "":
        return list(DEFAULT_CAPABILITY_AXES)

    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = [(str(idx), item) for idx, item in enumerate(raw)]
    else:
        raise ValueError("capability_axes must be a mapping, list, false, or omitted")

    axes: list[CapabilityAxis] = []
    for fallback_name, payload in items:
        if isinstance(payload, str):
            axes.append(
                CapabilityAxis(
                    name=str(payload),
                    description="",
                    metrics=(CapabilityMetric(str(payload), 1.0),),
                )
            )
            continue
        if not isinstance(payload, dict):
            raise ValueError("each capability_axes entry must be a mapping or metric string")

        name = str(payload.get("name") or fallback_name)
        description = str(payload.get("description") or "")
        raw_metrics = payload.get("metrics")
        if not isinstance(raw_metrics, list) or not raw_metrics:
            raise ValueError(f"capability_axes.{name}.metrics must be a non-empty list")

        metrics: list[CapabilityMetric] = []
        for metric in raw_metrics:
            if isinstance(metric, str):
                metrics.append(CapabilityMetric(name=metric, weight=1.0))
                continue
            if not isinstance(metric, dict):
                raise ValueError(f"capability_axes.{name}.metrics entries must be strings or mappings")
            metric_name = str(metric.get("name") or metric.get("metric") or "").strip()
            if not metric_name:
                raise ValueError(f"capability_axes.{name}.metrics entry is missing name")
            metrics.append(
                CapabilityMetric(
                    name=metric_name,
                    weight=float(metric.get("weight", 1.0)),
                )
            )
        axes.append(CapabilityAxis(name=name, description=description, metrics=tuple(metrics)))
    return axes


def build_capability_report(
    policy_reports: list[dict[str, Any]],
    *,
    axes: list[CapabilityAxis],
) -> dict[str, Any] | None:
    if not axes:
        return None

    policies: list[dict[str, Any]] = []
    for report in policy_reports:
        axis_scores = {
            axis.name: _axis_score(dict(report.get("metrics", {})), axis)
            for axis in axes
        }
        present_scores = [
            payload["score"]
            for payload in axis_scores.values()
            if payload.get("score") is not None
        ]
        overall_score = (
            sum(float(score) for score in present_scores) / len(present_scores)
            if present_scores
            else None
        )
        policies.append(
            {
                "name": report.get("name"),
                "checkpoint_path": report.get("checkpoint_path"),
                "overall_score": overall_score,
                "axes": axis_scores,
            }
        )

    baseline = policies[0] if policies else None
    candidates = [
        policy
        for policy in policies[1:]
        if policy.get("overall_score") is not None
    ]
    best_candidate = max(
        candidates,
        key=lambda item: float(item.get("overall_score") or float("-inf")),
        default=None,
    )
    deltas: dict[str, Any] = {}
    if baseline and best_candidate:
        for axis in axes:
            base_score = baseline["axes"][axis.name].get("score")
            candidate_score = best_candidate["axes"][axis.name].get("score")
            delta = (
                None
                if base_score is None or candidate_score is None
                else float(candidate_score) - float(base_score)
            )
            deltas[axis.name] = {
                "baseline": base_score,
                "candidate": candidate_score,
                "delta": delta,
            }

    return {
        "axes": [
            {
                "name": axis.name,
                "description": axis.description,
                "metrics": [
                    {"name": metric.name, "weight": metric.weight}
                    for metric in axis.metrics
                ],
            }
            for axis in axes
        ],
        "baseline": baseline.get("name") if baseline else None,
        "candidate": best_candidate.get("name") if best_candidate else None,
        "policies": policies,
        "deltas": deltas,
    }


def capability_report_markdown(report: dict[str, Any] | None) -> str:
    if not report:
        return "# Capability Report\n\nCapability axes are disabled.\n"

    lines = ["# Capability Report", ""]
    lines.append(f"- Baseline: `{report.get('baseline')}`")
    lines.append(f"- Candidate: `{report.get('candidate')}`")
    lines.append("")
    lines.append("| axis | baseline | candidate | delta |")
    lines.append("|---|---|---|---|")
    deltas = report.get("deltas") if isinstance(report.get("deltas"), dict) else {}
    for axis in report.get("axes", []):
        name = str(axis.get("name"))
        payload = deltas.get(name, {}) if isinstance(deltas, dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format_optional_float(payload.get("baseline")),
                    _format_optional_float(payload.get("candidate")),
                    _format_optional_float(payload.get("delta"), signed=True),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def infer_objective_axes(target_metrics: Any) -> list[str]:
    if not isinstance(target_metrics, list):
        return []

    axes: list[str] = []
    for metric in target_metrics:
        text = str(metric).lower()
        for axis, keywords in OBJECTIVE_AXIS_KEYWORDS:
            if axis in axes:
                continue
            if any(keyword in text for keyword in keywords):
                axes.append(axis)
    return axes


def _axis_score(metrics: dict[str, float], axis: CapabilityAxis) -> dict[str, Any]:
    values: dict[str, float] = {}
    missing: list[str] = []
    weighted = 0.0
    total_weight = 0.0
    for spec in axis.metrics:
        value = metrics.get(spec.name)
        if value is None:
            missing.append(spec.name)
            continue
        numeric = float(value)
        values[spec.name] = numeric
        weighted += numeric * spec.weight
        total_weight += abs(spec.weight)
    return {
        "score": weighted / total_weight if total_weight else None,
        "metrics": values,
        "missing_metrics": missing,
    }


def _format_optional_float(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:+.4f}" if signed else f"{numeric:.4f}"
