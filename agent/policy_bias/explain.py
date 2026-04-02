"""Explainability helpers for policy-bias decisions."""

from __future__ import annotations

from .models import DecisionTrace, PolicyBias, PolicyMoment


def bias_summary(bias: PolicyBias) -> dict[str, object]:
    return {
        "id": bias.id,
        "scope": bias.scope,
        "status": bias.status,
        "confidence": round(bias.confidence, 3),
        "support_count": bias.support_count,
        "avg_reward": round(bias.avg_reward, 3),
        "preferred_policy": bias.preferred_policy,
        "anti_policy": bias.anti_policy,
        "condition_signature": bias.condition_signature,
        "source_moment_ids": list(bias.source_moment_ids),
    }


def moment_summary(moment: PolicyMoment) -> dict[str, object]:
    return {
        "id": moment.id,
        "timestamp": moment.timestamp,
        "task_type": moment.task_type,
        "decision_class": moment.decision_class,
        "outcome_class": moment.outcome_class,
        "reward_score": round(moment.reward_score, 3),
        "tool_path": moment.tool_path,
        "context_summary": moment.context_summary,
        "evidence_refs": list(moment.evidence_refs),
        "candidate_key": moment.bias_candidate_key,
    }


def decision_trace_summary(trace: DecisionTrace) -> dict[str, object]:
    return {
        "id": trace.id,
        "session_id": trace.session_id,
        "turn_index": trace.turn_index,
        "task_type": trace.task_type,
        "platform": trace.platform,
        "user_message_excerpt": trace.user_message_excerpt,
        "retrieved_bias_ids": list(trace.retrieved_bias_ids),
        "injected_bias_ids": list(trace.injected_bias_ids),
        "shadow_bias_ids": list(trace.shadow_bias_ids),
        "planner_effects": list(trace.planner_effects),
        "tool_weight_deltas": list(trace.tool_weight_deltas),
        "risk_actions": list(trace.risk_actions),
        "response_effects": list(trace.response_effects),
        "evidence_summary": list(trace.evidence_summary),
        "created_at": trace.created_at,
    }


def build_explanation_payload(
    trace: DecisionTrace,
    biases: list[PolicyBias],
    moments: list[PolicyMoment],
) -> dict[str, object]:
    return {
        "trace": decision_trace_summary(trace),
        "biases": [bias_summary(bias) for bias in biases],
        "moments": [moment_summary(moment) for moment in moments],
    }
