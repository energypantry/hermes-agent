"""Core models for the Policy Bias Engine."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

BIAS_SCOPES = (
    "planning",
    "tool_use",
    "communication",
    "risk",
    "user_specific",
    "platform_specific",
    "workflow_specific",
)

BIAS_STATUSES = ("active", "shadow", "disabled", "archived")

POLICY_STATE_KEYS = (
    "inspect_tendency",
    "risk_aversion",
    "local_first_tendency",
    "decomposition_tendency",
    "retry_switch_tendency",
    "directness_tendency",
    "verbosity_budget",
    "findings_first_tendency",
    "single_step_tendency",
    "shared_channel_caution",
)

POLICY_STATE_STATUSES = ("active", "shadow", "disabled", "archived")


def now_ts() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def derive_profile_id() -> str:
    """Return a stable profile identifier derived from HERMES_HOME."""
    home = get_hermes_home().resolve()
    default_home = (Path.home() / ".hermes").resolve()
    profiles_root = (Path.home() / ".hermes" / "profiles").resolve()

    if home == default_home:
        return "profile:default"

    try:
        rel = home.relative_to(profiles_root)
        if rel.parts:
            return f"profile:{rel.parts[0]}"
    except ValueError:
        pass

    digest = hashlib.sha1(str(home).encode("utf-8")).hexdigest()[:12]
    return f"profile:custom:{digest}"


@dataclass(slots=True)
class PolicyBiasConfig:
    enabled: bool = True
    retrieval_top_k: int = 4
    max_prompt_tokens: int = 500
    shadow_mode_default: bool = False
    min_support_count: int = 3
    min_avg_reward: float = 0.25
    strong_signal_reward: float = 0.8
    confidence_decay_per_day: float = 0.01
    scopes_enabled: tuple[str, ...] = (
        "planning",
        "tool_use",
        "risk",
        "communication",
        "user_specific",
    )
    require_inspect_before_execute_for_external_actions: bool = True
    high_side_effect_shadow_only: bool = False
    log_bias_triggers: bool = True
    expose_explanations: bool = True

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> "PolicyBiasConfig":
        raw = raw or {}
        synthesis = raw.get("synthesis") or {}
        risk = raw.get("risk_controls") or {}
        obs = raw.get("observability") or {}
        scopes = raw.get("scopes_enabled") or cls.scopes_enabled
        if not isinstance(scopes, (list, tuple)):
            scopes = cls.scopes_enabled
        normalized_scopes = tuple(
            scope for scope in scopes if isinstance(scope, str) and scope in BIAS_SCOPES
        ) or cls.scopes_enabled
        return cls(
            enabled=bool(raw.get("enabled", True)),
            retrieval_top_k=max(1, int(raw.get("retrieval_top_k", 4))),
            max_prompt_tokens=max(80, int(raw.get("max_prompt_tokens", 500))),
            shadow_mode_default=bool(raw.get("shadow_mode_default", False)),
            min_support_count=max(1, int(synthesis.get("min_support_count", 3))),
            min_avg_reward=float(synthesis.get("min_avg_reward", 0.25)),
            strong_signal_reward=float(synthesis.get("strong_signal_reward", 0.8)),
            confidence_decay_per_day=max(
                0.0, float(synthesis.get("confidence_decay_per_day", 0.01))
            ),
            scopes_enabled=normalized_scopes,
            require_inspect_before_execute_for_external_actions=bool(
                risk.get("require_inspect_before_execute_for_external_actions", True)
            ),
            high_side_effect_shadow_only=bool(
                risk.get("high_side_effect_shadow_only", False)
            ),
            log_bias_triggers=bool(obs.get("log_bias_triggers", True)),
            expose_explanations=bool(obs.get("expose_explanations", True)),
        )


@dataclass(slots=True)
class PolicyMoment:
    id: str
    profile_id: str
    session_id: str
    timestamp: float
    task_type: str
    platform: str
    context_summary: str
    action_trace_summary: str
    tool_path: str
    decision_class: str
    outcome_class: str
    reward_score: float
    confidence_score: float
    user_feedback_signal: float = 0.0
    error_signal: float = 0.0
    side_effect_level: str = "none"
    latency_ms: Optional[int] = None
    cost_estimate: Optional[float] = None
    evidence_refs: list[str] = field(default_factory=list)
    extracted_tags: list[str] = field(default_factory=list)
    bias_candidate_key: Optional[str] = None


@dataclass(slots=True)
class PolicyBias:
    id: str
    profile_id: str
    scope: str
    condition_signature: str
    preferred_policy: str
    anti_policy: Optional[str]
    rationale_summary: str
    confidence: float
    support_count: int
    avg_reward: float
    recency_score: float
    decay_rate: float
    status: str
    source_moment_ids: list[str]
    created_at: float
    updated_at: float
    last_triggered_at: Optional[float]
    trigger_count: int
    rollback_parent_id: Optional[str]
    version: int
    bias_candidate_key: Optional[str] = None
    status_note: Optional[str] = None


@dataclass(slots=True)
class PolicyStateDimension:
    id: str
    profile_id: str
    dimension_key: str
    value: float
    confidence: float
    support_count: int
    avg_reward: float
    recency_score: float
    decay_rate: float
    status: str
    source_moment_ids: list[str]
    created_at: float
    updated_at: float
    last_triggered_at: Optional[float]
    trigger_count: int
    rollback_parent_id: Optional[str]
    version: int
    status_note: Optional[str] = None


@dataclass(slots=True)
class PolicyStateUpdate:
    id: str
    profile_id: str
    dimension_id: Optional[str]
    dimension_key: str
    moment_id: Optional[str]
    session_id: Optional[str]
    timestamp: float
    task_type: str
    platform: str
    decision_class: str
    outcome_class: str
    signal_type: str
    delta: float
    value_before: float
    value_after: float
    confidence_before: float
    confidence_after: float
    support_delta: int
    reward_score: float
    reason: str
    source_moment_ids: list[str]
    evidence_refs: list[str]
    update_source: str
    bias_candidate_key: Optional[str] = None
    created_at: float = field(default_factory=now_ts)


@dataclass(slots=True)
class PolicyStateRebuildResult:
    profile_id: str
    moment_count: int
    dimensions: list[PolicyStateDimension] = field(default_factory=list)
    updates: list[PolicyStateUpdate] = field(default_factory=list)
    created_at: float = field(default_factory=now_ts)


@dataclass(slots=True)
class PolicyStatePlan:
    active_dimensions: list[PolicyStateDimension] = field(default_factory=list)
    effective_values: dict[str, float] = field(default_factory=dict)
    planning_priority: float = 0.0
    execution_caution: float = 0.0
    local_first_priority: float = 0.0
    retry_avoidance: float = 0.0
    planner_mode: str = "direct"
    clarify_priority: float = 0.0
    max_tool_calls_per_turn: int = 0
    max_parallel_tools: int = 0
    execution_mode_scores: dict[str, float] = field(default_factory=dict)
    runtime_coverage_score: float = 0.0
    response_directness: float = 0.0
    findings_first_priority: float = 0.0
    single_step_priority: float = 0.0
    shared_channel_caution: float = 0.0
    require_sequential: bool = False
    preferred_risk_mode: str = "direct"
    prompt_mode: str = "minimal"
    available_tools: list[str] = field(default_factory=list)
    runtime_surfaces: list[str] = field(default_factory=list)
    prompt_hint_keys: list[str] = field(default_factory=list)
    tool_class_weights: dict[str, float] = field(default_factory=dict)
    response_controls: dict[str, Any] = field(default_factory=dict)
    conflict_resolutions: list[dict[str, Any]] = field(default_factory=list)
    arbitration_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ToolWeightDelta:
    tool_name: str
    weight_delta: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RiskAction:
    tool_name: str
    decision: str
    reason: str
    suggested_tool: Optional[str] = None
    bias_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DecisionTrace:
    id: str
    profile_id: str
    session_id: str
    turn_index: int
    task_type: str
    platform: str
    user_message_excerpt: str
    retrieved_bias_ids: list[str] = field(default_factory=list)
    injected_bias_ids: list[str] = field(default_factory=list)
    shadow_bias_ids: list[str] = field(default_factory=list)
    planner_effects: list[dict[str, Any]] = field(default_factory=list)
    tool_weight_deltas: list[dict[str, Any]] = field(default_factory=list)
    risk_actions: list[dict[str, Any]] = field(default_factory=list)
    response_effects: list[dict[str, Any]] = field(default_factory=list)
    evidence_summary: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=now_ts)


@dataclass(slots=True)
class BiasHistoryEntry:
    id: str
    bias_id: str
    profile_id: str
    version: int
    operation: str
    snapshot: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=now_ts)


@dataclass(slots=True)
class RetrievalResult:
    active_biases: list[PolicyBias] = field(default_factory=list)
    shadow_biases: list[PolicyBias] = field(default_factory=list)
    scored_biases: list[tuple[PolicyBias, float]] = field(default_factory=list)


@dataclass(slots=True)
class BiasDecisionContext:
    session_id: str
    turn_index: int
    task_type: str
    platform: str
    user_message: str
    active_biases: list[PolicyBias] = field(default_factory=list)
    shadow_biases: list[PolicyBias] = field(default_factory=list)
    decision_priors: str = ""
    ranked_tools: list[dict[str, Any]] = field(default_factory=list)
    tool_weight_deltas: list[ToolWeightDelta] = field(default_factory=list)
    trace_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
