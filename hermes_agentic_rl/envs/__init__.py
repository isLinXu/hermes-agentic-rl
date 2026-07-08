from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_TOOLS",
    "AtroposLetterCountingEnv",
    "AtroposLetterCountingReward",
    "BaseEnv",
    "ContextBenchmarkConfig",
    "ContextBenchmarkEnv",
    "ContextBenchmarkReward",
    "CurriculumEnv",
    "CurriculumState",
    "EchoRewardComponent",
    "EchoTaskEnv",
    "Fable5TraceConfig",
    "Fable5TraceEnv",
    "Fable5TraceReward",
    "Hermes3DatasetEnv",
    "HermesReasoningTraceEnv",
    "HermesReasoningTraceReward",
    "HermesReasoningTracesConfig",
    "LetterCountingConfig",
    "LetterCountingEnv",
    "LetterCountingReward",
    "MCPToolEnv",
    "MCPToolRegistry",
    "MCPToolReward",
    "MCPToolSpec",
    "MCPToolTask",
    "SimToolEnv",
    "SimToolRewardComponent",
    "TerminalTaskEnv",
    "build_context_benchmark_dataset",
    "build_mcp_tool_dataset",
    "build_sim_tool_dataset",
    "calc_tool",
    "load_hermes_reasoning_trace_turns",
    "safe_eval",
]


def __getattr__(name: str) -> Any:
    if name in {"AtroposLetterCountingEnv", "AtroposLetterCountingReward"}:
        from hermes_agentic_rl.envs.atropos_letter_counting import (
            AtroposLetterCountingEnv,
            AtroposLetterCountingReward,
        )

        return {
            "AtroposLetterCountingEnv": AtroposLetterCountingEnv,
            "AtroposLetterCountingReward": AtroposLetterCountingReward,
        }[name]
    if name == "BaseEnv":
        from hermes_agentic_rl.envs.base_env import BaseEnv

        return BaseEnv
    if name in {
        "ContextBenchmarkConfig",
        "ContextBenchmarkEnv",
        "ContextBenchmarkReward",
        "build_context_benchmark_dataset",
    }:
        from hermes_agentic_rl.envs.context_benchmark import (
            ContextBenchmarkConfig,
            ContextBenchmarkEnv,
            ContextBenchmarkReward,
            build_context_benchmark_dataset,
        )

        return {
            "ContextBenchmarkConfig": ContextBenchmarkConfig,
            "ContextBenchmarkEnv": ContextBenchmarkEnv,
            "ContextBenchmarkReward": ContextBenchmarkReward,
            "build_context_benchmark_dataset": build_context_benchmark_dataset,
        }[name]
    if name in {"CurriculumEnv", "CurriculumState"}:
        from hermes_agentic_rl.envs.curriculum import CurriculumEnv, CurriculumState

        return {
            "CurriculumEnv": CurriculumEnv,
            "CurriculumState": CurriculumState,
        }[name]
    if name in {"EchoRewardComponent", "EchoTaskEnv"}:
        from hermes_agentic_rl.envs.echo_task_env import EchoRewardComponent, EchoTaskEnv

        return {
            "EchoRewardComponent": EchoRewardComponent,
            "EchoTaskEnv": EchoTaskEnv,
        }[name]
    if name in {"Fable5TraceConfig", "Fable5TraceEnv", "Fable5TraceReward"}:
        from hermes_agentic_rl.envs.fable5_traces import (
            Fable5TraceConfig,
            Fable5TraceEnv,
            Fable5TraceReward,
        )

        return {
            "Fable5TraceConfig": Fable5TraceConfig,
            "Fable5TraceEnv": Fable5TraceEnv,
            "Fable5TraceReward": Fable5TraceReward,
        }[name]
    if name == "Hermes3DatasetEnv":
        from hermes_agentic_rl.envs.hermes3_dataset_env import Hermes3DatasetEnv

        return Hermes3DatasetEnv
    if name in {
        "HermesReasoningTraceEnv",
        "HermesReasoningTraceReward",
        "HermesReasoningTracesConfig",
        "load_hermes_reasoning_trace_turns",
    }:
        from hermes_agentic_rl.envs.hermes_reasoning_traces import (
            HermesReasoningTraceEnv,
            HermesReasoningTraceReward,
            HermesReasoningTracesConfig,
            load_hermes_reasoning_trace_turns,
        )

        return {
            "HermesReasoningTraceEnv": HermesReasoningTraceEnv,
            "HermesReasoningTraceReward": HermesReasoningTraceReward,
            "HermesReasoningTracesConfig": HermesReasoningTracesConfig,
            "load_hermes_reasoning_trace_turns": load_hermes_reasoning_trace_turns,
        }[name]
    if name in {"LetterCountingConfig", "LetterCountingEnv", "LetterCountingReward"}:
        from hermes_agentic_rl.envs.letter_counting import (
            LetterCountingConfig,
            LetterCountingEnv,
            LetterCountingReward,
        )

        return {
            "LetterCountingConfig": LetterCountingConfig,
            "LetterCountingEnv": LetterCountingEnv,
            "LetterCountingReward": LetterCountingReward,
        }[name]
    if name in {
        "DEFAULT_TOOLS",
        "SimToolEnv",
        "SimToolRewardComponent",
        "build_sim_tool_dataset",
        "calc_tool",
        "safe_eval",
    }:
        from hermes_agentic_rl.envs.sim_tool_env import (
            DEFAULT_TOOLS,
            SimToolEnv,
            SimToolRewardComponent,
            build_sim_tool_dataset,
            calc_tool,
            safe_eval,
        )

        return {
            "DEFAULT_TOOLS": DEFAULT_TOOLS,
            "SimToolEnv": SimToolEnv,
            "SimToolRewardComponent": SimToolRewardComponent,
            "build_sim_tool_dataset": build_sim_tool_dataset,
            "calc_tool": calc_tool,
            "safe_eval": safe_eval,
        }[name]
    if name == "TerminalTaskEnv":
        from hermes_agentic_rl.envs.terminal_task_env import TerminalTaskEnv

        return TerminalTaskEnv
    if name in {
        "MCPToolEnv",
        "MCPToolRegistry",
        "MCPToolReward",
        "MCPToolSpec",
        "MCPToolTask",
        "build_mcp_tool_dataset",
    }:
        from hermes_agentic_rl.envs.mcp_tool_env import (
            MCPToolEnv,
            MCPToolRegistry,
            MCPToolReward,
            MCPToolSpec,
            MCPToolTask,
            build_mcp_tool_dataset,
        )

        return {
            "MCPToolEnv": MCPToolEnv,
            "MCPToolRegistry": MCPToolRegistry,
            "MCPToolReward": MCPToolReward,
            "MCPToolSpec": MCPToolSpec,
            "MCPToolTask": MCPToolTask,
            "build_mcp_tool_dataset": build_mcp_tool_dataset,
        }[name]
    raise AttributeError(name)
