"""Bias retrieval and ranking."""

from __future__ import annotations

import time
from typing import Iterable

from .models import BIAS_SCOPES, PolicyBias, PolicyBiasConfig, RetrievalResult
from .scoring import (
    apply_confidence_decay,
    lexical_overlap_score,
    semantic_overlap_score,
)
from .store import PolicyBiasStore


def _structured_match_score(
    bias: PolicyBias,
    *,
    task_type: str,
    platform: str,
    available_tools: Iterable[str],
) -> float:
    score = 0.0
    available_tools = set(available_tools or [])
    signature = (bias.condition_signature or "").lower()

    if task_type and task_type in signature:
        score += 0.35
    if platform and platform in signature:
        score += 0.25
    if bias.scope == "tool_use" and available_tools:
        score += 0.15
    if bias.scope == "risk" and any(
        tool in available_tools
        for tool in ("terminal", "send_message", "cronjob", "ha_call_service", "write_file", "patch")
    ):
        score += 0.25
    if bias.scope == "communication":
        score += 0.10
    if bias.scope == "user_specific":
        score += 0.18
    return min(score, 1.0)


def _select_scope_balanced(
    scored: list[tuple[PolicyBias, float]],
    *,
    top_k: int,
    status: str,
    enabled_scopes: tuple[str, ...],
) -> list[PolicyBias]:
    if top_k <= 0:
        return []

    scope_order = {scope: index for index, scope in enumerate(BIAS_SCOPES)}
    allowed_scopes = tuple(scope for scope in enabled_scopes if scope in BIAS_SCOPES) or BIAS_SCOPES

    scope_heads: list[tuple[str, tuple[PolicyBias, float]]] = []
    seen_scopes: set[str] = set()
    for bias, score in scored:
        if bias.status != status or bias.scope not in allowed_scopes or bias.scope in seen_scopes:
            continue
        scope_heads.append((bias.scope, (bias, score)))
        seen_scopes.add(bias.scope)

    scope_heads.sort(
        key=lambda item: (
            item[1][1],
            -scope_order.get(item[0], len(scope_order)),
        ),
        reverse=True,
    )

    selected: list[PolicyBias] = []
    selected_ids: set[str] = set()
    for _scope, (bias, _score) in scope_heads:
        selected.append(bias)
        selected_ids.add(bias.id)
        if len(selected) >= top_k:
            return selected

    for bias, _score in scored:
        if bias.status != status or bias.scope not in allowed_scopes or bias.id in selected_ids:
            continue
        selected.append(bias)
        selected_ids.add(bias.id)
        if len(selected) >= top_k:
            break

    return selected


def retrieve_biases(
    store: PolicyBiasStore,
    *,
    config: PolicyBiasConfig,
    profile_id: str,
    user_message: str,
    task_type: str,
    platform: str,
    available_tools: Iterable[str],
    include_shadow: bool = True,
) -> RetrievalResult:
    statuses = ["active"]
    if include_shadow:
        statuses.append("shadow")
    now = time.time()

    candidates = store.list_biases(
        profile_id,
        statuses=statuses,
        scopes=config.scopes_enabled,
        limit=200,
    )

    query_text = " ".join(
        filter(
            None,
            [user_message, task_type, platform, " ".join(sorted(set(available_tools or [])))],
        )
    )
    scored: list[tuple[PolicyBias, float]] = []
    for bias in candidates:
        decayed_conf = apply_confidence_decay(
            bias.confidence,
            updated_at=bias.updated_at,
            now=now,
            decay_per_day=bias.decay_rate,
        )
        lexical = lexical_overlap_score(query_text, bias)
        semantic = semantic_overlap_score(query_text, bias)
        structured = _structured_match_score(
            bias,
            task_type=task_type,
            platform=platform,
            available_tools=available_tools,
        )
        reward_factor = max(0.0, (bias.avg_reward + 1.0) / 2.0)
        score = (
            0.35 * decayed_conf
            + 0.22 * lexical
            + 0.23 * semantic
            + 0.10 * structured
            + 0.10 * reward_factor
        )
        scored.append((bias, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    active_biases = _select_scope_balanced(
        scored,
        top_k=config.retrieval_top_k,
        status="active",
        enabled_scopes=config.scopes_enabled,
    )
    shadow_biases = _select_scope_balanced(
        scored,
        top_k=config.retrieval_top_k,
        status="shadow",
        enabled_scopes=config.scopes_enabled,
    ) if include_shadow else []

    return RetrievalResult(
        active_biases=active_biases,
        shadow_biases=shadow_biases,
        scored_biases=scored,
    )
