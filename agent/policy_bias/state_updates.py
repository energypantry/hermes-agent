"""Deterministic policy-state update helpers for V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .models import (
    POLICY_STATE_KEYS,
    PolicyMoment,
    PolicyStateDimension,
    PolicyStateRebuildResult,
    PolicyStateUpdate,
    new_id,
    now_ts,
)

_ACTIVE_STATUS = "active"
_DEFAULT_DECAY_RATE = 0.01
_SOURCE_LIMIT = 20


@dataclass(slots=True)
class _PolicySignal:
    dimension_key: str
    delta: float
    reward_score: float
    signal_type: str
    reason: str


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalized_strength(moment: PolicyMoment) -> float:
    strength = 0.60
    outcome = (moment.outcome_class or "").lower()
    if outcome in {"success", "completed", "checkpoint"}:
        strength += 0.08
    elif outcome in {"failure", "error", "blocked"}:
        strength += 0.22
    if moment.error_signal > 0:
        strength += min(0.18, float(moment.error_signal) * 0.12)
    if moment.user_feedback_signal != 0:
        strength += min(0.16, abs(float(moment.user_feedback_signal)) * 0.10)
    if moment.confidence_score > 0:
        strength += min(0.12, float(moment.confidence_score) * 0.08)
    return max(0.40, min(1.25, strength))


def _moment_tokens(moment: PolicyMoment) -> tuple[str, set[str], str, str, str, str]:
    candidate = (moment.bias_candidate_key or "").lower()
    tags = {str(tag).lower() for tag in moment.extracted_tags}
    tool_path = (moment.tool_path or "").lower()
    task_type = (moment.task_type or "").lower()
    platform = (moment.platform or "").lower()
    outcome = (moment.outcome_class or "").lower()
    return candidate, tags, tool_path, task_type, platform, outcome


def _has_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def _shared_platform(platform: str) -> bool:
    return _has_any(platform, ("slack", "discord", "teams", "group", "channel"))


def _signals_for_moment(moment: PolicyMoment) -> list[_PolicySignal]:
    candidate, tags, tool_path, task_type, platform, outcome = _moment_tokens(moment)
    signals: list[_PolicySignal] = []
    strength = _normalized_strength(moment)

    def emit(
        dimension_key: str,
        base_delta: float,
        signal_type: str,
        reason: str,
        *,
        reward_score: Optional[float] = None,
    ) -> None:
        if dimension_key not in POLICY_STATE_KEYS:
            return
        delta = base_delta * strength
        if abs(delta) < 0.01:
            return
        signals.append(
            _PolicySignal(
                dimension_key=dimension_key,
                delta=delta,
                reward_score=float(moment.reward_score if reward_score is None else reward_score),
                signal_type=signal_type,
                reason=reason,
            )
        )

    inspect_tokens = (
        "planning.inspect_before_edit",
        "risk.inspect_before_execute",
        "inspect",
        "search",
        "read_file",
        "browser_snapshot",
        "web_search",
    )
    if _has_any(candidate, inspect_tokens) or _has_any(tool_path, inspect_tokens):
        if outcome in {"success", "completed", "checkpoint"}:
            emit(
                "inspect_tendency",
                0.22,
                "inspect_success",
                "Successful inspect/search behavior reinforced an inspect-first tendency.",
            )

    if _has_any(candidate, ("risk.inspect_before_execute",)) or (
        moment.side_effect_level in {"medium", "high"}
        and (outcome in {"failure", "error", "blocked"} or moment.error_signal > 0)
    ):
        emit(
            "risk_aversion",
            0.24 if outcome in {"failure", "error", "blocked"} else 0.16,
            "side_effect_risk",
            "Outcome-bearing side-effect behavior increased risk caution.",
        )

    if task_type == "repo_modification":
        if _has_any(tool_path, ("read_file", "search_files", "patch", "write_file", "terminal")):
            emit(
                "local_first_tendency",
                0.18,
                "local_repo_work",
                "Repository-local work reinforced a local-first tendency.",
            )
        if _has_any(tool_path, ("web_search", "web_extract", "browser_navigate")):
            emit(
                "local_first_tendency",
                -0.10,
                "external_for_local_work",
                "External browsing on a local repo task reduced the local-first tendency.",
            )

    if _has_any(candidate, ("workflow_specific.decompose_before_act",)) or _has_any(
        tool_path,
        ("todo", "clarify"),
    ):
        emit(
            "decomposition_tendency",
            0.18,
            "decompose_before_act",
            "Planning behavior before action reinforced task decomposition.",
        )

    if _has_any(candidate, ("tool_use.change_strategy_after_retries",)) or (
        outcome in {"failure", "error"} and _has_any(" ".join(tags), ("retry", "fallback", "repeat"))
    ):
        emit(
            "retry_switch_tendency",
            0.20,
            "change_strategy_after_retry",
            "Repeated failure patterns reinforced changing strategy instead of retrying the same path.",
        )

    if _has_any(candidate, ("communication.concise_first", "user_specific.directness_over_fluff")) or _has_any(
        " ".join(tags),
        ("concise", "direct", "brief", "findings_first"),
    ):
        emit(
            "directness_tendency",
            0.16,
            "concise_direct_feedback",
            "Concise/direct feedback reinforced a direct communication tendency.",
        )
        emit(
            "verbosity_budget",
            0.12,
            "concise_direct_feedback",
            "Concise/direct feedback reinforced a tighter verbosity budget.",
        )

    if _has_any(candidate, ("workflow_specific.structured_debugging_output",)) or (
        task_type == "repo_modification"
        and _has_any(" ".join(tags), ("debug", "review", "bug", "failure", "regression"))
    ):
        emit(
            "findings_first_tendency",
            0.18,
            "structured_debugging",
            "Debug/review behavior reinforced findings-first responses.",
        )

    if _has_any(candidate, ("user_specific.one_step_at_a_time",)) or _has_any(
        " ".join(tags),
        ("one_step", "single_step"),
    ):
        emit(
            "single_step_tendency",
            0.15,
            "single_step_workflow",
            "Single-step workflow preference was reinforced.",
        )

    if _has_any(candidate, ("platform_specific.group_chat_caution",)) or _shared_platform(platform):
        if moment.side_effect_level in {"medium", "high"} or outcome in {"failure", "blocked", "error"}:
            emit(
                "shared_channel_caution",
                0.18,
                "shared_channel_context",
                "Shared-channel context with side effects reinforced caution.",
            )

    return signals


def _seed_dimension(moment: PolicyMoment, dimension_key: str) -> PolicyStateDimension:
    created_at = float(moment.timestamp or now_ts())
    return PolicyStateDimension(
        id=new_id("psd"),
        profile_id=moment.profile_id,
        dimension_key=dimension_key,
        value=0.0,
        confidence=0.0,
        support_count=0,
        avg_reward=0.0,
        recency_score=0.0,
        decay_rate=_DEFAULT_DECAY_RATE,
        status=_ACTIVE_STATUS,
        source_moment_ids=[],
        created_at=created_at,
        updated_at=created_at,
        last_triggered_at=None,
        trigger_count=0,
        rollback_parent_id=None,
        version=1,
        status_note=None,
    )


def apply_policy_state_update(
    dimension: PolicyStateDimension,
    update: PolicyStateUpdate,
) -> PolicyStateDimension:
    if dimension.status in {"disabled", "archived"}:
        return dimension

    source_moment_ids = list(dict.fromkeys(dimension.source_moment_ids + update.source_moment_ids))[-_SOURCE_LIMIT:]
    support_count = max(0, int(dimension.support_count + update.support_delta))
    avg_reward = update.reward_score
    if support_count > 1:
        total_reward = (dimension.avg_reward * max(0, dimension.support_count)) + update.reward_score
        avg_reward = total_reward / support_count

    return PolicyStateDimension(
        id=dimension.id,
        profile_id=dimension.profile_id,
        dimension_key=dimension.dimension_key,
        value=clamp01(update.value_after),
        confidence=clamp01(update.confidence_after),
        support_count=support_count,
        avg_reward=avg_reward,
        recency_score=clamp01(max(dimension.recency_score, 0.0) * (1.0 - dimension.decay_rate) + 0.08),
        decay_rate=max(0.0, float(dimension.decay_rate)),
        status=dimension.status,
        source_moment_ids=source_moment_ids,
        created_at=dimension.created_at,
        updated_at=float(update.timestamp),
        last_triggered_at=float(update.timestamp),
        trigger_count=dimension.trigger_count + 1,
        rollback_parent_id=dimension.rollback_parent_id,
        version=dimension.version + 1,
        status_note=dimension.status_note,
    )


def derive_policy_state_updates_from_moment(
    moment: PolicyMoment,
    state_by_key: Optional[dict[str, PolicyStateDimension]] = None,
) -> list[PolicyStateUpdate]:
    if state_by_key is None:
        state_by_key = {}
    updates: list[PolicyStateUpdate] = []
    for signal in _signals_for_moment(moment):
        current = state_by_key.get(signal.dimension_key) or _seed_dimension(moment, signal.dimension_key)
        value_before = current.value
        value_after = clamp01(value_before + signal.delta)
        confidence_before = current.confidence
        confidence_after = clamp01(
            max(
                confidence_before,
                min(1.0, confidence_before * 0.85 + abs(signal.delta) * 1.1 + max(0.0, moment.reward_score) * 0.05),
            )
        )
        support_delta = 1
        update = PolicyStateUpdate(
            id=new_id("psu"),
            profile_id=moment.profile_id,
            dimension_id=current.id,
            dimension_key=signal.dimension_key,
            moment_id=moment.id,
            session_id=moment.session_id,
            timestamp=float(moment.timestamp),
            task_type=moment.task_type,
            platform=moment.platform,
            decision_class=moment.decision_class,
            outcome_class=moment.outcome_class,
            signal_type=signal.signal_type,
            delta=signal.delta,
            value_before=value_before,
            value_after=value_after,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            support_delta=support_delta,
            reward_score=signal.reward_score,
            reason=signal.reason,
            source_moment_ids=[moment.id],
            evidence_refs=list(moment.evidence_refs),
            update_source="moment",
            bias_candidate_key=moment.bias_candidate_key,
            created_at=float(moment.timestamp),
        )
        updates.append(update)
        state_by_key[signal.dimension_key] = apply_policy_state_update(current, update)
    return updates


def build_policy_state_from_moments(
    moments: Iterable[PolicyMoment],
    *,
    initial_state_by_key: Optional[dict[str, PolicyStateDimension]] = None,
) -> PolicyStateRebuildResult:
    ordered = sorted(list(moments), key=lambda item: item.timestamp)
    if not ordered:
        if initial_state_by_key:
            dimensions = sorted(initial_state_by_key.values(), key=lambda item: item.dimension_key)
            return PolicyStateRebuildResult(
                profile_id=dimensions[0].profile_id,
                moment_count=0,
                dimensions=dimensions,
                updates=[],
                created_at=now_ts(),
            )
        return PolicyStateRebuildResult(profile_id="profile:unknown", moment_count=0)

    profile_id = ordered[0].profile_id
    state_by_key: dict[str, PolicyStateDimension] = dict(initial_state_by_key or {})
    updates: list[PolicyStateUpdate] = []
    for moment in ordered:
        for update in derive_policy_state_updates_from_moment(moment, state_by_key):
            updates.append(update)

    dimensions = sorted(state_by_key.values(), key=lambda item: item.dimension_key)
    return PolicyStateRebuildResult(
        profile_id=profile_id,
        moment_count=len(ordered),
        dimensions=dimensions,
        updates=updates,
        created_at=now_ts(),
    )
