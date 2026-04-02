"""SQLite schema and migrations for the Policy Bias Engine."""

from __future__ import annotations

SCHEMA_VERSION = 5

BASE_SCHEMA_VERSION = 1

BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS moments (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    task_type TEXT NOT NULL,
    platform TEXT NOT NULL,
    context_summary TEXT NOT NULL,
    action_trace_summary TEXT NOT NULL,
    tool_path TEXT NOT NULL,
    decision_class TEXT NOT NULL,
    outcome_class TEXT NOT NULL,
    reward_score REAL NOT NULL,
    confidence_score REAL NOT NULL,
    user_feedback_signal REAL DEFAULT 0,
    error_signal REAL DEFAULT 0,
    side_effect_level TEXT NOT NULL DEFAULT 'none',
    latency_ms INTEGER,
    cost_estimate REAL,
    evidence_refs TEXT NOT NULL,
    extracted_tags TEXT NOT NULL,
    bias_candidate_key TEXT
);

CREATE INDEX IF NOT EXISTS idx_moments_profile_time
    ON moments(profile_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_moments_profile_candidate
    ON moments(profile_id, bias_candidate_key, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_moments_session
    ON moments(session_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS biases (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    condition_signature TEXT NOT NULL,
    preferred_policy TEXT NOT NULL,
    anti_policy TEXT,
    rationale_summary TEXT NOT NULL,
    confidence REAL NOT NULL,
    support_count INTEGER NOT NULL DEFAULT 0,
    avg_reward REAL NOT NULL DEFAULT 0,
    recency_score REAL NOT NULL DEFAULT 0,
    decay_rate REAL NOT NULL DEFAULT 0.01,
    status TEXT NOT NULL,
    source_moment_ids TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_triggered_at REAL,
    trigger_count INTEGER NOT NULL DEFAULT 0,
    rollback_parent_id TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    bias_candidate_key TEXT
);

CREATE INDEX IF NOT EXISTS idx_biases_profile_status_scope
    ON biases(profile_id, status, scope, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_biases_profile_updated
    ON biases(profile_id, updated_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_biases_profile_candidate
    ON biases(profile_id, bias_candidate_key)
    WHERE bias_candidate_key IS NOT NULL;
"""

MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE biases ADD COLUMN disabled_reason TEXT",
        """
        CREATE TABLE IF NOT EXISTS decision_traces (
            id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL DEFAULT 0,
            task_type TEXT NOT NULL,
            platform TEXT NOT NULL,
            user_message_excerpt TEXT NOT NULL,
            retrieved_bias_ids TEXT NOT NULL,
            injected_bias_ids TEXT NOT NULL,
            shadow_bias_ids TEXT NOT NULL,
            planner_effects TEXT NOT NULL,
            tool_weight_deltas TEXT NOT NULL,
            risk_actions TEXT NOT NULL,
            evidence_summary TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_decision_traces_profile_created
            ON decision_traces(profile_id, created_at DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_decision_traces_session_created
            ON decision_traces(session_id, created_at DESC)
        """,
    ],
    3: [
        """
        CREATE TABLE IF NOT EXISTS bias_history (
            id TEXT PRIMARY KEY,
            bias_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            operation TEXT NOT NULL,
            snapshot TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_bias_history_bias_version
            ON bias_history(bias_id, version DESC, created_at DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_bias_history_profile_created
            ON bias_history(profile_id, created_at DESC)
        """,
    ],
    4: [
        "ALTER TABLE decision_traces ADD COLUMN response_effects TEXT NOT NULL DEFAULT '[]'",
    ],
    5: [
        """
        CREATE TABLE IF NOT EXISTS policy_state_dimensions (
            id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            dimension_key TEXT NOT NULL,
            value REAL NOT NULL,
            confidence REAL NOT NULL,
            support_count INTEGER NOT NULL DEFAULT 0,
            avg_reward REAL NOT NULL DEFAULT 0,
            recency_score REAL NOT NULL DEFAULT 0,
            decay_rate REAL NOT NULL DEFAULT 0.01,
            status TEXT NOT NULL,
            source_moment_ids TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_triggered_at REAL,
            trigger_count INTEGER NOT NULL DEFAULT 0,
            rollback_parent_id TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            status_note TEXT
        )
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_policy_state_dimensions_profile_key
            ON policy_state_dimensions(profile_id, dimension_key)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_policy_state_dimensions_profile_status
            ON policy_state_dimensions(profile_id, status, updated_at DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_policy_state_dimensions_profile_updated
            ON policy_state_dimensions(profile_id, updated_at DESC)
        """,
        """
        CREATE TABLE IF NOT EXISTS policy_state_updates (
            id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            dimension_id TEXT,
            dimension_key TEXT NOT NULL,
            moment_id TEXT,
            session_id TEXT,
            timestamp REAL NOT NULL,
            task_type TEXT NOT NULL,
            platform TEXT NOT NULL,
            decision_class TEXT NOT NULL,
            outcome_class TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            delta REAL NOT NULL,
            value_before REAL NOT NULL,
            value_after REAL NOT NULL,
            confidence_before REAL NOT NULL,
            confidence_after REAL NOT NULL,
            support_delta INTEGER NOT NULL DEFAULT 1,
            reward_score REAL NOT NULL DEFAULT 0,
            reason TEXT NOT NULL,
            source_moment_ids TEXT NOT NULL,
            evidence_refs TEXT NOT NULL,
            update_source TEXT NOT NULL DEFAULT 'moment',
            bias_candidate_key TEXT,
            created_at REAL NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_policy_state_updates_profile_dimension_time
            ON policy_state_updates(profile_id, dimension_key, timestamp DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_policy_state_updates_profile_created
            ON policy_state_updates(profile_id, created_at DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_policy_state_updates_moment
            ON policy_state_updates(moment_id, timestamp DESC)
        """,
    ],
}
