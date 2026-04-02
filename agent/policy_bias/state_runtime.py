"""Runtime helpers for consuming Policy Bias V2 policy state."""

from __future__ import annotations

from typing import Iterable, Sequence

from .models import PolicyStateDimension, PolicyStatePlan, now_ts


_ACTIVE_STATUSES = {"active", "shadow"}
_MIN_ACTIVE_MAGNITUDE = 0.12
_DEBUG_MARKERS = (
    "debug",
    "review",
    "bug",
    "failure",
    "failing",
    "regression",
    "issue",
    "risk",
    "错误",
    "问题",
    "风险",
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _effective_strength(
    dimension: PolicyStateDimension,
    *,
    now: float | None = None,
) -> float:
    now = float(now if now is not None else now_ts())
    updated_at = float(dimension.updated_at or now)
    elapsed_days = max(0.0, (now - updated_at) / 86400.0)
    decay_factor = max(0.0, 1.0 - (float(dimension.decay_rate) * elapsed_days))
    confidence_factor = 0.60 + (0.40 * _clamp01(dimension.confidence))
    return _clamp01(float(dimension.value) * decay_factor * confidence_factor)


def _is_debug_request(*, user_message: str, task_type: str) -> bool:
    lowered = (user_message or "").lower()
    if any(marker in lowered for marker in _DEBUG_MARKERS):
        return True
    return task_type == "repo_modification"


def _looks_shared_platform(platform: str) -> bool:
    platform = (platform or "").lower()
    return any(token in platform for token in ("slack", "discord", "teams", "group", "channel"))


def active_dimensions(
    dimensions: Iterable[PolicyStateDimension] | None,
    *,
    include_shadow: bool = False,
    now: float | None = None,
) -> list[PolicyStateDimension]:
    allowed_statuses = {"active", "shadow"} if include_shadow else {"active"}
    active: list[PolicyStateDimension] = []
    for dimension in dimensions or []:
        if dimension.status not in allowed_statuses:
            continue
        if _effective_strength(dimension, now=now) < _MIN_ACTIVE_MAGNITUDE:
            continue
        active.append(dimension)
    return active


def dimension_map(
    dimensions: Iterable[PolicyStateDimension] | None,
    *,
    include_shadow: bool = False,
    now: float | None = None,
) -> dict[str, PolicyStateDimension]:
    return {
        dimension.dimension_key: dimension
        for dimension in active_dimensions(dimensions, include_shadow=include_shadow, now=now)
    }


def effective_value_map(
    dimensions: Iterable[PolicyStateDimension] | None,
    *,
    include_shadow: bool = False,
) -> dict[str, float]:
    return {
        key: _effective_strength(dimension)
        for key, dimension in dimension_map(
            dimensions,
            include_shadow=include_shadow,
        ).items()
    }


def dimension_value(
    dimensions: Iterable[PolicyStateDimension] | None,
    key: str,
    *,
    include_shadow: bool = False,
) -> float:
    return effective_value_map(
        dimensions,
        include_shadow=include_shadow,
    ).get(key, 0.0)


def serialize_dimensions(
    dimensions: Iterable[PolicyStateDimension] | None,
    *,
    include_shadow: bool = False,
) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for dimension in active_dimensions(dimensions, include_shadow=include_shadow):
        serialized.append(
            {
                "id": dimension.id,
                "dimension_key": dimension.dimension_key,
                "value": round(float(dimension.value), 4),
                "effective_value": round(_effective_strength(dimension), 4),
                "confidence": round(float(dimension.confidence), 4),
                "support_count": int(dimension.support_count),
                "status": dimension.status,
                "updated_at": float(dimension.updated_at),
            }
        )
    return serialized


def evidence_summary(
    dimensions: Iterable[PolicyStateDimension] | None,
    *,
    include_shadow: bool = False,
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for dimension in active_dimensions(dimensions, include_shadow=include_shadow):
        summary.append(
            {
                "kind": "policy_state",
                "id": dimension.id,
                "dimension_key": dimension.dimension_key,
                "value": round(float(dimension.value), 3),
                "effective_value": round(_effective_strength(dimension), 3),
                "confidence": round(float(dimension.confidence), 3),
                "support_count": int(dimension.support_count),
                "status": dimension.status,
            }
        )
    return summary


def compile_state_plan(
    dimensions: Iterable[PolicyStateDimension] | None,
    *,
    task_type: str,
    platform: str,
    user_message: str,
    available_tools: Sequence[str] | None = None,
    recent_failed_tools: Sequence[str] | None = None,
) -> PolicyStatePlan:
    active = active_dimensions(dimensions)
    values = effective_value_map(active)
    available = set(available_tools or [])
    recent_failed = set(recent_failed_tools or [])
    notes: list[str] = []

    inspect = values.get("inspect_tendency", 0.0)
    risk = values.get("risk_aversion", 0.0)
    local_first = values.get("local_first_tendency", 0.0)
    decompose = values.get("decomposition_tendency", 0.0)
    retry_switch = values.get("retry_switch_tendency", 0.0)
    directness = values.get("directness_tendency", 0.0)
    verbosity_budget = values.get("verbosity_budget", 0.0)
    findings_first = values.get("findings_first_tendency", 0.0)
    single_step = values.get("single_step_tendency", 0.0)
    shared_caution = values.get("shared_channel_caution", 0.0) if _looks_shared_platform(platform) else 0.0

    planning_priority = _clamp01(max(inspect, decompose, single_step * 0.85, shared_caution * 0.60))
    execution_caution = _clamp01(max(risk, shared_caution, inspect * 0.35, single_step * 0.30))
    local_first_priority = _clamp01(local_first if task_type == "repo_modification" else 0.0)
    retry_avoidance = _clamp01(retry_switch if recent_failed else 0.0)
    response_directness = _clamp01(max(directness, verbosity_budget * 0.85))
    findings_first_priority = _clamp01(findings_first if _is_debug_request(user_message=user_message, task_type=task_type) else 0.0)
    single_step_priority = _clamp01(single_step)

    if execution_caution >= 0.72:
        preferred_risk_mode = "confirm"
        notes.append("High execution caution biases the agent toward confirmation before side effects.")
    elif execution_caution >= 0.45:
        preferred_risk_mode = "inspect"
        notes.append("Moderate execution caution biases the agent toward inspect/simulate before side effects.")
    else:
        preferred_risk_mode = "direct"

    require_sequential = False
    if single_step_priority >= 0.35:
        require_sequential = True
        notes.append("Single-step tendency forces sequential tool execution.")
    elif planning_priority >= 0.55 and {"todo", "clarify"} & available and recent_failed:
        require_sequential = True
        notes.append("Planning-first arbitration serializes execution after recent failures.")
    elif planning_priority >= 0.65 and inspect >= 0.45:
        require_sequential = True
        notes.append("Inspect-first arbitration serializes mixed inspect and execute batches.")

    response_controls: dict[str, object] = {}
    if response_directness >= 0.35:
        response_controls["strip_leading_acknowledgement"] = True
        response_controls["drop_trailing_offer"] = True
        notes.append("Response directness strips acknowledgement and trailing offer boilerplate.")
    if findings_first_priority >= 0.35:
        response_controls["findings_first_heading"] = True
        notes.append("Findings-first tendency restructures debug/review responses.")
    if single_step_priority >= 0.35:
        response_controls["prefer_single_step"] = True

    prompt_hint_keys: list[str] = []
    if execution_caution >= 0.72:
        prompt_hint_keys.append("risk_aversion")
    elif planning_priority >= 0.72:
        prompt_hint_keys.append("inspect_tendency" if inspect >= decompose else "decomposition_tendency")
    if local_first_priority >= 0.70 and {"web_search", "web_extract", "browser_navigate"} & available:
        prompt_hint_keys.append("local_first_tendency")
    if findings_first_priority >= 0.70:
        prompt_hint_keys.append("findings_first_tendency")
    if response_directness >= 0.75 and not findings_first_priority:
        prompt_hint_keys.append("directness_tendency")
    if shared_caution >= 0.65:
        prompt_hint_keys.append("shared_channel_caution")
    prompt_hint_keys = list(dict.fromkeys(prompt_hint_keys))[:3]

    prompt_mode = "off"
    if prompt_hint_keys:
        prompt_mode = "minimal"
        notes.append("Prompt translation is reduced to only the highest-signal state hints.")

    return PolicyStatePlan(
        active_dimensions=active,
        effective_values=values,
        planning_priority=planning_priority,
        execution_caution=execution_caution,
        local_first_priority=local_first_priority,
        retry_avoidance=retry_avoidance,
        response_directness=response_directness,
        findings_first_priority=findings_first_priority,
        single_step_priority=single_step_priority,
        shared_channel_caution=shared_caution,
        require_sequential=require_sequential,
        preferred_risk_mode=preferred_risk_mode,
        prompt_mode=prompt_mode,
        prompt_hint_keys=prompt_hint_keys,
        response_controls=response_controls,
        arbitration_notes=notes,
    )


def plan_summary(plan: PolicyStatePlan) -> dict[str, object]:
    return {
        "kind": "policy_state_plan",
        "planning_priority": round(plan.planning_priority, 3),
        "execution_caution": round(plan.execution_caution, 3),
        "local_first_priority": round(plan.local_first_priority, 3),
        "retry_avoidance": round(plan.retry_avoidance, 3),
        "response_directness": round(plan.response_directness, 3),
        "findings_first_priority": round(plan.findings_first_priority, 3),
        "single_step_priority": round(plan.single_step_priority, 3),
        "shared_channel_caution": round(plan.shared_channel_caution, 3),
        "require_sequential": bool(plan.require_sequential),
        "preferred_risk_mode": plan.preferred_risk_mode,
        "prompt_mode": plan.prompt_mode,
        "prompt_hint_keys": list(plan.prompt_hint_keys),
        "arbitration_notes": list(plan.arbitration_notes),
    }


def prompt_hint_lines(
    dimensions: Iterable[PolicyStateDimension] | None,
    *,
    plan: PolicyStatePlan | None = None,
) -> list[str]:
    hints: list[tuple[str, str]] = []
    values = effective_value_map(dimensions)
    allowed_keys = set(plan.prompt_hint_keys) if plan is not None else set()
    if plan is not None and plan.prompt_mode == "off":
        return []

    def _hint(key: str, scope: str, text: str, *, threshold: float = 0.35) -> None:
        value = values.get(key)
        if value is None or float(value) < threshold:
            return
        if allowed_keys and key not in allowed_keys:
            return
        hints.append(
            (
                key,
                f"- [scope={scope} state={key} value={value:.2f}] {text}",
            )
        )

    _hint(
        "inspect_tendency",
        "planning",
        "Prefer inspection and cheap verification before mutation.",
    )
    _hint(
        "risk_aversion",
        "risk",
        "Raise certainty before external or side-effect actions.",
    )
    _hint(
        "local_first_tendency",
        "tool_use",
        "Prefer local reads/search before external browsing on code-local tasks.",
    )
    _hint(
        "decomposition_tendency",
        "workflow_specific",
        "Break work into planning or inspect steps before execution.",
    )
    _hint(
        "directness_tendency",
        "communication",
        "Keep responses direct and low-fluff.",
    )
    _hint(
        "findings_first_tendency",
        "workflow_specific",
        "Lead with findings or risks before broader explanation.",
    )
    _hint(
        "single_step_tendency",
        "user_specific",
        "Prefer one executable next step at a time.",
    )
    _hint(
        "shared_channel_caution",
        "platform_specific",
        "Be more cautious before acting in shared or group channels.",
    )
    return [line for _key, line in hints]
