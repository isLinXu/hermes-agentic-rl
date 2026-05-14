from __future__ import annotations

from hermes_agentic_rl.eval.capability_axes import (
    DEFAULT_CAPABILITY_AXES,
    infer_objective_axes,
    normalize_capability_axes,
)


def test_normalize_capability_axes_uses_defaults_when_omitted() -> None:
    axes = normalize_capability_axes(None)

    assert [axis.name for axis in axes] == [axis.name for axis in DEFAULT_CAPABILITY_AXES]
    assert "prompt_context" in [axis.name for axis in axes]


def test_normalize_capability_axes_accepts_yaml_mapping_shape() -> None:
    axes = normalize_capability_axes(
        {
            "tool_use_reliability": {
                "description": "Hermes tool-call structure and argument fidelity.",
                "metrics": [
                    "metadata/tool_call_parse_ok",
                    {"name": "metadata/tool_name_match", "weight": 2.0},
                ],
            }
        }
    )

    assert len(axes) == 1
    assert axes[0].name == "tool_use_reliability"
    assert axes[0].metrics[0].name == "metadata/tool_call_parse_ok"
    assert axes[0].metrics[0].weight == 1.0
    assert axes[0].metrics[1].name == "metadata/tool_name_match"
    assert axes[0].metrics[1].weight == 2.0


def test_normalize_capability_axes_accepts_list_shape_and_disabled_flag() -> None:
    axes = normalize_capability_axes(
        [
            {
                "name": "command_content",
                "description": "Terminal command fidelity.",
                "metrics": ["metadata/argument_value_similarity"],
            }
        ]
    )

    assert axes[0].name == "command_content"
    assert axes[0].metrics[0].name == "metadata/argument_value_similarity"
    assert normalize_capability_axes(False) == []


def test_infer_objective_axes_maps_target_metrics_to_agent_capabilities() -> None:
    assert infer_objective_axes(
        [
            "tool_call_parse_ok",
            "argument_value_similarity",
            "finished_naturally_rate",
            "success_rate",
            "context_required_fact_recall",
        ]
    ) == [
        "tool_use_reliability",
        "interaction_control",
        "task_success",
        "prompt_context",
    ]
