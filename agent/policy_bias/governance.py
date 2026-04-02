"""Governance operations for policy-bias state."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .explain import build_explanation_payload
from .models import PolicyBias, PolicyBiasConfig
from .store import PolicyBiasStore
from .synthesis import get_boundary_metadata, synthesize_bias


def list_biases(
    store: PolicyBiasStore,
    *,
    profile_id: str,
    limit: int = 50,
    status: Optional[str] = None,
) -> list[PolicyBias]:
    statuses = [status] if status else None
    return store.list_biases(profile_id, statuses=statuses, limit=limit)


def inspect_bias(
    store: PolicyBiasStore,
    *,
    bias_id: str,
) -> dict[str, object] | None:
    bias = store.get_bias(bias_id)
    if bias is None:
        return None
    moments = [
        moment
        for moment in (
            store.get_moment(moment_id) for moment_id in bias.source_moment_ids
        )
        if moment is not None
    ]
    return {
        "bias": bias,
        "moments": moments,
        "history": store.list_bias_history(bias_id, limit=20),
        "boundary": describe_bias_boundary(bias),
    }


def describe_bias_boundary(bias: PolicyBias) -> dict[str, object]:
    return get_boundary_metadata(bias.bias_candidate_key or "")


def set_bias_enabled(
    store: PolicyBiasStore,
    *,
    bias_id: str,
    enabled: bool,
) -> bool:
    return store.set_bias_status(
        bias_id,
        status="active" if enabled else "disabled",
        note=None if enabled else "disabled via governance command",
    )


def archive_bias(
    store: PolicyBiasStore,
    *,
    bias_id: str,
) -> bool:
    return store.set_bias_status(
        bias_id,
        status="archived",
        note="archived via governance command",
    )


def rescore_biases(
    store: PolicyBiasStore,
    *,
    config: PolicyBiasConfig,
    profile_id: str,
    bias_id: Optional[str] = None,
) -> list[PolicyBias]:
    if bias_id:
        existing = store.get_bias(bias_id)
        if existing is None:
            return []
        if not existing.bias_candidate_key:
            return [existing]
        moments = store.get_moments_by_candidate(profile_id, existing.bias_candidate_key)
        rebuilt = synthesize_bias(
            config=config,
            profile_id=profile_id,
            candidate_key=existing.bias_candidate_key,
            moments=moments,
            existing=existing,
        )
        if rebuilt is None:
            return [existing]
        store.upsert_bias(rebuilt)
        return [rebuilt]

    return rebuild_biases(store, config=config, profile_id=profile_id)


def rebuild_biases(
    store: PolicyBiasStore,
    *,
    config: PolicyBiasConfig,
    profile_id: str,
) -> list[PolicyBias]:
    grouped = defaultdict(list)
    for moment in store.iter_all_moments(profile_id):
        if moment.bias_candidate_key:
            grouped[moment.bias_candidate_key].append(moment)

    rebuilt: list[PolicyBias] = []
    for candidate_key, moments in grouped.items():
        existing = store.find_bias_by_candidate_key(profile_id, candidate_key)
        bias = synthesize_bias(
            config=config,
            profile_id=profile_id,
            candidate_key=candidate_key,
            moments=moments,
            existing=existing,
        )
        if bias is None:
            continue
        store.upsert_bias(bias)
        rebuilt.append(bias)
    return rebuilt


def recent_moments(
    store: PolicyBiasStore,
    *,
    profile_id: str,
    limit: int = 20,
) -> list[dict[str, object]]:
    return [
        {
            "id": moment.id,
            "timestamp": moment.timestamp,
            "task_type": moment.task_type,
            "decision_class": moment.decision_class,
            "outcome_class": moment.outcome_class,
            "reward_score": moment.reward_score,
            "tool_path": moment.tool_path,
            "candidate_key": moment.bias_candidate_key,
        }
        for moment in store.list_recent_moments(profile_id, limit=limit)
    ]


def bias_history(
    store: PolicyBiasStore,
    *,
    bias_id: str,
    limit: int = 20,
) -> list[dict[str, object]]:
    return [
        {
            "id": entry.id,
            "bias_id": entry.bias_id,
            "version": entry.version,
            "operation": entry.operation,
            "snapshot": entry.snapshot,
            "created_at": entry.created_at,
        }
        for entry in store.list_bias_history(bias_id, limit=limit)
    ]


def audit_bias_boundaries(
    store: PolicyBiasStore,
    *,
    profile_id: str,
    limit: int = 100,
    status: Optional[str] = None,
) -> list[dict[str, object]]:
    statuses = [status] if status else None
    audited: list[dict[str, object]] = []
    for bias in store.list_biases(profile_id, statuses=statuses, limit=limit):
        boundary = describe_bias_boundary(bias)
        audited.append(
            {
                "id": bias.id,
                "status": bias.status,
                "scope": bias.scope,
                "candidate_key": bias.bias_candidate_key,
                "classification": boundary["classification"],
                "action_surfaces": boundary["action_surfaces"],
                "why_not_memory": boundary["why_not_memory"],
                "why_not_skill": boundary["why_not_skill"],
            }
        )
    return audited


def rollback_bias(
    store: PolicyBiasStore,
    *,
    bias_id: str,
    version: int,
) -> bool:
    return store.rollback_bias(bias_id, version=version)


def export_stable_biases(
    store: PolicyBiasStore,
    *,
    config: PolicyBiasConfig,
    profile_id: str,
    min_confidence: float = 0.55,
) -> list[dict[str, object]]:
    exported = []
    for bias in store.list_biases(profile_id, statuses=["active"], limit=500):
        if bias.support_count < config.min_support_count:
            continue
        if bias.confidence < min_confidence:
            continue
        exported.append(
            {
                "id": bias.id,
                "scope": bias.scope,
                "condition_signature": bias.condition_signature,
                "preferred_policy": bias.preferred_policy,
                "anti_policy": bias.anti_policy,
                "rationale_summary": bias.rationale_summary,
                "confidence": bias.confidence,
                "support_count": bias.support_count,
                "avg_reward": bias.avg_reward,
                "version": bias.version,
                "candidate_key": bias.bias_candidate_key,
                "source_moment_ids": list(bias.source_moment_ids),
            }
        )
    return exported


def explain_recent(
    store: PolicyBiasStore,
    *,
    profile_id: str,
    session_id: Optional[str] = None,
    limit: int = 5,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    traces = store.get_recent_decision_traces(profile_id, session_id=session_id, limit=limit)
    for trace in traces:
        bias_ids = list(dict.fromkeys(trace.retrieved_bias_ids + trace.shadow_bias_ids))
        biases = [
            bias
            for bias in (
                store.get_bias(bias_id) for bias_id in bias_ids
            )
            if bias is not None
        ]
        moment_ids = []
        for bias in biases:
            moment_ids.extend(bias.source_moment_ids)
        deduped_ids = list(dict.fromkeys(moment_ids))[:20]
        moments = [
            moment
            for moment in (store.get_moment(moment_id) for moment_id in deduped_ids)
            if moment is not None
        ]
        payloads.append(build_explanation_payload(trace, biases, moments))
    return payloads
