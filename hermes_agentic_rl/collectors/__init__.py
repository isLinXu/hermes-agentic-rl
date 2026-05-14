__all__ = [
    "LocalSessionSidecar",
    "LocalSessionSidecarConfig",
    "SessionTurnSample",
    "build_preference_pairs_from_samples",
    "build_session_reward_results",
    "build_sidecar_from_runtime_config",
    "build_tokenizer_from_config",
    "close_all_sidecars",
    "collect_session_turn_samples",
    "get_or_create_sidecar",
    "judge_session_turn_sample",
    "mine_session_turn_sample",
    "normalize_replay_mining_config",
    "record_to_session_turn_samples",
    "records_to_train_samples",
    "render_messages_for_training",
    "replay_jsonl_payloads_from_record",
    "session_turn_sample_to_train_sample",
    "session_turn_samples_to_replay_buffer",
    "summarize_replay_mining",
    "trajectory_to_session_turn_samples",
]


def __getattr__(name: str):
    if name in {"SessionTurnSample", "collect_session_turn_samples"}:
        from hermes_agentic_rl.collectors.conversation_collector import (
            SessionTurnSample,
            collect_session_turn_samples,
        )

        return {
            "SessionTurnSample": SessionTurnSample,
            "collect_session_turn_samples": collect_session_turn_samples,
        }[name]
    if name in {
        "render_messages_for_training",
        "session_turn_sample_to_train_sample",
        "session_turn_samples_to_replay_buffer",
        "trajectory_to_session_turn_samples",
    }:
        from hermes_agentic_rl.collectors.trajectory_adapter import (
            render_messages_for_training,
            session_turn_sample_to_train_sample,
            session_turn_samples_to_replay_buffer,
            trajectory_to_session_turn_samples,
        )

        return {
            "render_messages_for_training": render_messages_for_training,
            "session_turn_sample_to_train_sample": session_turn_sample_to_train_sample,
            "session_turn_samples_to_replay_buffer": session_turn_samples_to_replay_buffer,
            "trajectory_to_session_turn_samples": trajectory_to_session_turn_samples,
        }[name]
    if name in {
        "build_tokenizer_from_config",
        "record_to_session_turn_samples",
        "replay_jsonl_payloads_from_record",
    }:
        from hermes_agentic_rl.collectors.replay_export import (
            build_tokenizer_from_config,
            record_to_session_turn_samples,
            replay_jsonl_payloads_from_record,
        )

        return {
            "build_tokenizer_from_config": build_tokenizer_from_config,
            "record_to_session_turn_samples": record_to_session_turn_samples,
            "replay_jsonl_payloads_from_record": replay_jsonl_payloads_from_record,
        }[name]
    if name in {"build_preference_pairs_from_samples", "records_to_train_samples"}:
        from hermes_agentic_rl.collectors.preference_mining import (
            build_preference_pairs_from_samples,
            records_to_train_samples,
        )

        return {
            "build_preference_pairs_from_samples": build_preference_pairs_from_samples,
            "records_to_train_samples": records_to_train_samples,
        }[name]
    if name in {"build_session_reward_results", "judge_session_turn_sample"}:
        from hermes_agentic_rl.collectors.session_judge import (
            build_session_reward_results,
            judge_session_turn_sample,
        )

        return {
            "build_session_reward_results": build_session_reward_results,
            "judge_session_turn_sample": judge_session_turn_sample,
        }[name]
    if name in {
        "mine_session_turn_sample",
        "normalize_replay_mining_config",
        "summarize_replay_mining",
    }:
        from hermes_agentic_rl.collectors.replay_mining import (
            mine_session_turn_sample,
            normalize_replay_mining_config,
            summarize_replay_mining,
        )

        return {
            "mine_session_turn_sample": mine_session_turn_sample,
            "normalize_replay_mining_config": normalize_replay_mining_config,
            "summarize_replay_mining": summarize_replay_mining,
        }[name]
    if name in {
        "LocalSessionSidecar",
        "LocalSessionSidecarConfig",
        "build_sidecar_from_runtime_config",
        "close_all_sidecars",
        "get_or_create_sidecar",
    }:
        from hermes_agentic_rl.collectors.sidecar import (
            LocalSessionSidecar,
            LocalSessionSidecarConfig,
            build_sidecar_from_runtime_config,
            close_all_sidecars,
            get_or_create_sidecar,
        )

        return {
            "LocalSessionSidecar": LocalSessionSidecar,
            "LocalSessionSidecarConfig": LocalSessionSidecarConfig,
            "build_sidecar_from_runtime_config": build_sidecar_from_runtime_config,
            "close_all_sidecars": close_all_sidecars,
            "get_or_create_sidecar": get_or_create_sidecar,
        }[name]
    raise AttributeError(name)
