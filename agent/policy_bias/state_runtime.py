"""Runtime helpers for consuming Policy Bias V2 policy state."""

from __future__ import annotations

from typing import Iterable, Sequence

from .models import PolicyStateDimension, PolicyStatePlan, now_ts


_ACTIVE_STATUSES = {"active", "shadow"}
_MIN_ACTIVE_MAGNITUDE = 0.12
_AMBIGUITY_MARKERS = (
    "maybe",
    "probably",
    "not sure",
    "if needed",
    "should we",
    "can you",
    "maybe send",
    "不确定",
    "如果需要",
    "要不要",
)
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


def looks_ambiguous_request(user_message: str) -> bool:
    text = (user_message or "").lower()
    return any(token in text for token in _AMBIGUITY_MARKERS)


def _record_conflict(
    conflicts: list[dict[str, object]],
    *,
    between: tuple[str, str],
    winner: str,
    loser: str,
    rationale: str,
) -> None:
    conflicts.append(
        {
            "between": list(between),
            "winner": winner,
            "loser": loser,
            "rationale": rationale,
        }
    )


def _rounded_weights(weights: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 3) for key, value in weights.items() if abs(float(value)) > 1e-6}


def _rounded_scores(scores: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 3) for key, value in scores.items()}


def _top_score_key(scores: dict[str, float]) -> str:
    if not scores:
        return "direct"
    return max(scores.items(), key=lambda item: (float(item[1]), item[0]))[0]


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
    ambiguous = looks_ambiguous_request(user_message)
    notes: list[str] = []
    conflicts: list[dict[str, object]] = []

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
    clarify_priority = 0.0
    if "clarify" in available:
        clarify_priority = _clamp01(
            (0.72 if ambiguous else 0.0)
            * max(execution_caution, shared_caution, single_step_priority * 0.85, decompose * 0.75)
            + (0.20 if recent_failed and retry_avoidance >= 0.35 else 0.0)
        )
        if ambiguous and execution_caution >= 0.45:
            clarify_priority = max(clarify_priority, 0.48)

    if execution_caution >= 0.55 and response_directness >= 0.45:
        _record_conflict(
            conflicts,
            between=("risk_aversion", "directness_tendency"),
            winner="risk_aversion",
            loser="directness_tendency",
            rationale="High side-effect caution overrides terse execution behavior for risky decisions.",
        )
        notes.append("Risk caution overrides directness when side effects are on the line.")

    if single_step_priority >= 0.35 and decompose >= 0.45:
        _record_conflict(
            conflicts,
            between=("single_step_tendency", "decomposition_tendency"),
            winner="single_step_tendency",
            loser="decomposition_tendency",
            rationale="Decomposition remains preferred, but execution is serialized into one next step at a time.",
        )
        notes.append("Single-step tendency serializes decomposition into one next action at a time.")

    if findings_first_priority >= 0.35 and single_step_priority >= 0.35:
        _record_conflict(
            conflicts,
            between=("findings_first_tendency", "single_step_tendency"),
            winner="findings_first_tendency",
            loser="single_step_tendency",
            rationale="Findings-first structure is preserved, but the numbered next-step budget is capped to one.",
        )
        notes.append("Findings-first structure is preserved while the next-step budget is capped at one.")

    if execution_caution >= 0.72:
        preferred_risk_mode = "confirm"
        notes.append("High execution caution biases the agent toward confirmation before side effects.")
    elif execution_caution >= 0.45:
        preferred_risk_mode = "inspect"
        notes.append("Moderate execution caution biases the agent toward inspect/simulate before side effects.")
    else:
        preferred_risk_mode = "direct"

    planner_mode = "direct"
    if clarify_priority >= 0.45:
        planner_mode = "clarify_first"
        notes.append("Clarify-first arbitration activates under ambiguity and elevated caution.")
    elif planning_priority >= 0.70 and decompose >= inspect and {"todo", "clarify"} & available:
        planner_mode = "decompose_first"
        notes.append("Decompose-first arbitration prioritizes planning surfaces before execution.")
    elif planning_priority >= 0.55 and inspect >= 0.45 and {
        "read_file",
        "search_files",
        "browser_snapshot",
        "web_search",
        "web_extract",
    } & available:
        planner_mode = "inspect_first"
        notes.append("Inspect-first arbitration prioritizes cheap verification before mutation.")
    elif local_first_priority >= 0.70 and {"read_file", "search_files", "patch", "write_file", "terminal"} & available:
        planner_mode = "local_first"
        notes.append("Local-first arbitration biases toward local repo surfaces before external browsing.")

    require_sequential = False
    if planner_mode == "clarify_first":
        require_sequential = True
        notes.append("Clarify-first arbitration serializes execution until ambiguity is reduced.")
    elif single_step_priority >= 0.35:
        require_sequential = True
        notes.append("Single-step tendency forces sequential tool execution.")
    elif planning_priority >= 0.55 and {"todo", "clarify"} & available and recent_failed:
        require_sequential = True
        notes.append("Planning-first arbitration serializes execution after recent failures.")
    elif planning_priority >= 0.65 and inspect >= 0.45:
        require_sequential = True
        notes.append("Inspect-first arbitration serializes mixed inspect and execute batches.")

    max_tool_calls_per_turn = 0
    max_parallel_tools = 0
    if planner_mode == "clarify_first":
        max_tool_calls_per_turn = 1
        max_parallel_tools = 1
        notes.append("Clarify-first arbitration caps the turn to one tool decision at a time.")
    elif single_step_priority >= 0.55 or execution_caution >= 0.82:
        max_tool_calls_per_turn = 1
        max_parallel_tools = 1
        notes.append("High caution or one-step preference caps execution to a single tool call.")
    elif require_sequential:
        max_tool_calls_per_turn = 2 if planning_priority >= 0.55 else 1
        max_parallel_tools = 1
        notes.append("Sequential arbitration constrains the tool batch before the next model turn.")
    elif planning_priority >= 0.60 or retry_avoidance >= 0.45:
        max_tool_calls_per_turn = 2
        max_parallel_tools = 2
        notes.append("Planner pressure reduces tool-batch breadth even when parallelism stays available.")

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
        response_controls["max_numbered_steps"] = 1

    tool_class_weights = _rounded_weights(
        {
            "inspect": _clamp01(
                (0.85 * planning_priority)
                + (0.35 * execution_caution)
                + (0.15 if planner_mode == "inspect_first" else 0.0)
            ),
            "planning": _clamp01(
                (0.75 * planning_priority)
                + (0.35 * single_step_priority)
                + (0.20 if planner_mode in {"decompose_first", "clarify_first"} else 0.0)
            ),
            "clarify": _clamp01(
                clarify_priority + (0.15 if planner_mode == "clarify_first" else 0.0)
            ),
            "local": _clamp01(
                (0.90 * local_first_priority) + (0.10 * planning_priority)
            ) if task_type == "repo_modification" else 0.0,
            "mutating": -_clamp01(
                (0.30 * planning_priority)
                + (0.45 * execution_caution)
                + (0.20 * single_step_priority)
                + (0.15 if planner_mode == "clarify_first" else 0.0)
            ),
            "external": -_clamp01(
                (0.25 * execution_caution)
                + (0.35 * local_first_priority)
                + (0.30 * shared_caution)
                + (0.15 if planner_mode == "clarify_first" else 0.0)
            ),
            "retry_failed": -_clamp01(0.95 * retry_avoidance),
        }
    )

    execution_mode_scores = _rounded_scores(
        {
            "direct": _clamp01(
                (1.0 - execution_caution) * 0.92
                + (0.12 if not ambiguous and shared_caution < 0.2 else 0.0)
                - (0.18 if planner_mode == "clarify_first" else 0.0)
            ),
            "inspect": _clamp01(
                (0.68 * planning_priority)
                + (0.42 * execution_caution)
                + (0.16 if planner_mode == "inspect_first" else 0.0)
            ),
            "simulate": _clamp01(
                (0.58 * execution_caution)
                + (0.18 * planning_priority)
                + (0.14 * shared_caution)
                + (0.10 if planner_mode == "inspect_first" else 0.0)
            ),
            "confirm": _clamp01(
                (0.86 * execution_caution)
                + (0.16 * shared_caution)
                + (0.08 if planner_mode == "clarify_first" else 0.0)
            ),
            "clarify": _clamp01(
                clarify_priority + (0.12 if planner_mode == "clarify_first" else 0.0)
            ),
        }
    )

    runtime_surfaces: list[str] = []
    if planner_mode != "direct" or tool_class_weights:
        runtime_surfaces.append("planner")
    if preferred_risk_mode != "direct":
        runtime_surfaces.append("risk")
    if response_controls:
        runtime_surfaces.append("response")
    if max_tool_calls_per_turn > 0 or max_parallel_tools > 0:
        runtime_surfaces.append("execution_budget")

    runtime_coverage_score = _clamp01(
        (0.34 * max(planning_priority, float(bool(tool_class_weights))))
        + (0.28 * max(execution_mode_scores.get("inspect", 0.0), execution_mode_scores.get("confirm", 0.0), execution_mode_scores.get("clarify", 0.0)))
        + (0.18 * max(response_directness, findings_first_priority, single_step_priority))
        + (0.20 * max(float(bool(max_tool_calls_per_turn)), float(bool(max_parallel_tools and max_parallel_tools < 3))))
    )

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
    if prompt_hint_keys and runtime_coverage_score < 0.72:
        prompt_mode = "minimal"
        notes.append("Prompt translation is reduced to only the highest-signal state hints.")
    elif prompt_hint_keys:
        notes.append("Runtime coverage is strong enough to suppress state prompt translation for this turn.")
    if runtime_coverage_score >= 0.72:
        notes.append(
            f"Runtime coverage score {runtime_coverage_score:.2f} keeps policy state primarily off-prompt."
        )
    dominant_execution_mode = _top_score_key(execution_mode_scores)
    if dominant_execution_mode != "direct":
        notes.append(
            f"Execution-mode scoring currently leans toward {dominant_execution_mode} before direct action."
        )

    return PolicyStatePlan(
        active_dimensions=active,
        effective_values=values,
        planning_priority=planning_priority,
        execution_caution=execution_caution,
        local_first_priority=local_first_priority,
        retry_avoidance=retry_avoidance,
        planner_mode=planner_mode,
        clarify_priority=clarify_priority,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        max_parallel_tools=max_parallel_tools,
        execution_mode_scores=execution_mode_scores,
        runtime_coverage_score=runtime_coverage_score,
        response_directness=response_directness,
        findings_first_priority=findings_first_priority,
        single_step_priority=single_step_priority,
        shared_channel_caution=shared_caution,
        require_sequential=require_sequential,
        preferred_risk_mode=preferred_risk_mode,
        prompt_mode=prompt_mode,
        available_tools=sorted(available),
        runtime_surfaces=runtime_surfaces,
        prompt_hint_keys=prompt_hint_keys,
        tool_class_weights=tool_class_weights,
        response_controls=response_controls,
        conflict_resolutions=conflicts,
        arbitration_notes=notes,
    )


def plan_summary(plan: PolicyStatePlan) -> dict[str, object]:
    return {
        "kind": "policy_state_plan",
        "planning_priority": round(plan.planning_priority, 3),
        "execution_caution": round(plan.execution_caution, 3),
        "local_first_priority": round(plan.local_first_priority, 3),
        "retry_avoidance": round(plan.retry_avoidance, 3),
        "planner_mode": plan.planner_mode,
        "clarify_priority": round(plan.clarify_priority, 3),
        "max_tool_calls_per_turn": int(plan.max_tool_calls_per_turn),
        "max_parallel_tools": int(plan.max_parallel_tools),
        "execution_mode_scores": dict(plan.execution_mode_scores),
        "runtime_coverage_score": round(plan.runtime_coverage_score, 3),
        "response_directness": round(plan.response_directness, 3),
        "findings_first_priority": round(plan.findings_first_priority, 3),
        "single_step_priority": round(plan.single_step_priority, 3),
        "shared_channel_caution": round(plan.shared_channel_caution, 3),
        "require_sequential": bool(plan.require_sequential),
        "preferred_risk_mode": plan.preferred_risk_mode,
        "prompt_mode": plan.prompt_mode,
        "runtime_surfaces": list(plan.runtime_surfaces),
        "tool_class_weights": dict(plan.tool_class_weights),
        "prompt_hint_keys": list(plan.prompt_hint_keys),
        "conflict_resolutions": list(plan.conflict_resolutions),
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
