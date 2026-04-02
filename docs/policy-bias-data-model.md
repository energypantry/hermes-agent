# Policy Bias Engine Data Model

## Database

SQLite database path:

- `~/.hermes/policy_bias.db`

Schema version is tracked in `schema_version`.
Current schema version: `4`.

## Table: `moments`

Represents outcome-bearing runtime events.

Key columns:

- `id`
- `profile_id`
- `session_id`
- `timestamp`
- `task_type`
- `platform`
- `context_summary`
- `action_trace_summary`
- `tool_path`
- `decision_class`
- `outcome_class`
- `reward_score`
- `confidence_score`
- `user_feedback_signal`
- `error_signal`
- `side_effect_level`
- `latency_ms`
- `cost_estimate`
- `evidence_refs`
- `extracted_tags`
- `bias_candidate_key`

Important indexes:

- `idx_moments_profile_time`
- `idx_moments_profile_candidate`
- `idx_moments_session`

## Table: `biases`

Represents durable policy priors synthesized from moments.

Key columns:

- `id`
- `profile_id`
- `scope`
- `condition_signature`
- `preferred_policy`
- `anti_policy`
- `rationale_summary`
- `confidence`
- `support_count`
- `avg_reward`
- `recency_score`
- `decay_rate`
- `status`
- `source_moment_ids`
- `created_at`
- `updated_at`
- `last_triggered_at`
- `trigger_count`
- `rollback_parent_id`
- `version`
- `bias_candidate_key`
- `disabled_reason`

Important indexes:

- `idx_biases_profile_status_scope`
- `idx_biases_profile_updated`
- unique candidate index per profile:
  `idx_biases_profile_candidate`

## Table: `decision_traces`

Represents per-decision explainability records.

Key columns:

- `id`
- `profile_id`
- `session_id`
- `turn_index`
- `task_type`
- `platform`
- `user_message_excerpt`
- `retrieved_bias_ids`
- `injected_bias_ids`
- `shadow_bias_ids`
- `planner_effects`
- `tool_weight_deltas`
- `risk_actions`
- `response_effects`
- `evidence_summary`
- `created_at`

Important indexes:

- `idx_decision_traces_profile_created`
- `idx_decision_traces_session_created`

## Table: `bias_history`

Represents versioned snapshots for audit and rollback.

Key columns:

- `id`
- `bias_id`
- `profile_id`
- `version`
- `operation`
- `snapshot`
- `created_at`

Important indexes:

- `idx_bias_history_bias_version`
- `idx_bias_history_profile_created`

## Profile Isolation

Every runtime table stores `profile_id`.

This means:

- moments do not leak across profiles
- bias retrieval is profile scoped
- decision traces are profile scoped
- exports are profile scoped
- history and rollback stay inside the originating profile

## Lifecycle Mapping

The persistence model supports:

- moment accumulation
- synthesis into a single current bias row per `(profile_id, bias_candidate_key)`
- version increments on updates
- explicit status transitions
- historical snapshots for audit and rollback
- response-policy effects captured alongside planner/tool/risk traces for later explanation

## Export Shape

Stable bias export returns compact records with:

- bias ID and scope
- condition signature
- preferred and anti-policy text
- confidence, support count, average reward
- version
- candidate key
- supporting moment IDs
