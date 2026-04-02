"""Explicit planner/tool weighting hooks driven by policy biases."""

from __future__ import annotations

import copy
from typing import Iterable

from .models import PolicyBias, PolicyStateDimension, PolicyStatePlan, RiskAction, ToolWeightDelta
from .scoring import side_effect_level_for_tool
from .state_runtime import dimension_map, looks_ambiguous_request

_INSPECT_TOOLS = {"read_file", "search_files", "browser_snapshot", "web_search", "web_extract", "ha_get_state", "ha_list_entities", "ha_list_services"}
_LOCAL_TOOLS = {"read_file", "search_files", "patch", "write_file", "terminal"}
_EXTERNAL_TOOLS = {"web_search", "web_extract", "browser_navigate", "browser_click", "browser_type", "browser_press", "send_message", "cronjob", "ha_call_service"}
_MUTATING_TOOLS = {"write_file", "patch", "terminal", "browser_click", "browser_type", "browser_press", "send_message", "cronjob", "ha_call_service"}
_PLANNING_TOOLS = {"todo", "clarify"}


def _looks_shared_platform(platform: str) -> bool:
    platform = (platform or "").lower()
    return any(token in platform for token in ("slack", "discord", "teams", "group", "channel"))


def _has_explicit_execute_intent(user_message: str, function_args: dict[str, object]) -> bool:
    text = " ".join(
        filter(
            None,
            [
                user_message or "",
                str(function_args.get("message", "") or ""),
                str(function_args.get("command", "") or ""),
            ],
        )
    ).lower()
    return any(
        token in text
        for token in (
            "please send",
            "go ahead",
            "proceed",
            "send it",
            "run it",
            "do it",
            "现在发送",
            "直接执行",
            "就这么做",
        )
    )


def _candidate_keys(biases: Iterable[PolicyBias]) -> set[str]:
    return {bias.bias_candidate_key or "" for bias in biases}


def rerank_tools(
    tool_defs: list[dict],
    biases: list[PolicyBias],
    *,
    user_message: str,
    task_type: str,
    platform: str,
    recent_failed_tools: Iterable[str],
    policy_state: list[PolicyStateDimension] | None = None,
    policy_state_plan: PolicyStatePlan | None = None,
) -> tuple[list[dict], list[ToolWeightDelta], list[dict]]:
    if not tool_defs:
        return [], [], []

    keys = _candidate_keys(biases)
    state = dimension_map(policy_state)
    state_values = dict(policy_state_plan.effective_values) if policy_state_plan is not None else {
        key: float(dimension.value)
        for key, dimension in state.items()
    }
    recent_failed = set(recent_failed_tools or [])
    deltas: list[ToolWeightDelta] = []
    planner_effects: list[dict] = []
    scored: list[tuple[float, int, dict]] = []
    plan_weights = dict(policy_state_plan.tool_class_weights) if policy_state_plan is not None else {}
    planner_mode = policy_state_plan.planner_mode if policy_state_plan is not None else "direct"

    for index, tool_def in enumerate(tool_defs):
        tool_name = tool_def.get("function", {}).get("name", "")
        delta = 0.0
        reasons: list[str] = []

        if task_type == "repo_modification" and "planning.inspect_before_edit" in keys:
            if tool_name in {"read_file", "search_files"}:
                delta += 1.2
                reasons.append("inspect-before-edit")
            elif tool_name in {"write_file", "patch"}:
                delta -= 0.2
                reasons.append("defer-edits-until-inspection")

        if task_type == "repo_modification" and "tool_use.patch_before_rewrite" in keys:
            if tool_name == "patch":
                delta += 1.1
                reasons.append("prefer-patch")
            elif tool_name == "write_file":
                delta -= 0.7
                reasons.append("avoid-full-rewrite")

        if task_type == "repo_modification" and "tool_use.local_before_external_code" in keys:
            if tool_name in _LOCAL_TOOLS:
                delta += 0.9
                reasons.append("local-before-external")
            elif tool_name in {"web_search", "web_extract", "browser_navigate"}:
                delta -= 0.8
                reasons.append("penalize-external-for-local-task")

        if task_type == "current_info" and "planning.search_before_fresh_answer" in keys:
            if tool_name in {"web_search", "web_extract", "browser_navigate"}:
                delta += 1.0
                reasons.append("search-before-fresh-answer")

        if "risk.inspect_before_execute" in keys:
            if tool_name in _INSPECT_TOOLS:
                delta += 0.7
                reasons.append("risk-inspect-first")
            elif tool_name in _MUTATING_TOOLS:
                delta -= 0.4
                reasons.append("risk-penalty")

        if "workflow_specific.decompose_before_act" in keys:
            if tool_name in _PLANNING_TOOLS:
                delta += 1.0
                reasons.append("decompose-before-act")
            elif tool_name in _MUTATING_TOOLS | _EXTERNAL_TOOLS:
                delta -= 0.35
                reasons.append("delay-execution-until-decomposed")

        if "workflow_specific.structured_debugging_output" in keys and task_type == "repo_modification":
            if tool_name in {"read_file", "search_files", "session_search"}:
                delta += 0.45
                reasons.append("findings-first-evidence-gathering")
            elif tool_name in {"write_file", "patch"}:
                delta -= 0.20
                reasons.append("avoid-editing-before-findings")

        if "user_specific.one_step_at_a_time" in keys:
            if tool_name in _PLANNING_TOOLS | _INSPECT_TOOLS:
                delta += 0.30
                reasons.append("one-step-at-a-time")
            elif tool_name in _MUTATING_TOOLS | _EXTERNAL_TOOLS:
                delta -= 0.20
                reasons.append("avoid-multi-pronged-execution")

        if "platform_specific.group_chat_caution" in keys and _looks_shared_platform(platform):
            if tool_name in _MUTATING_TOOLS | {"send_message", "cronjob", "ha_call_service"}:
                delta -= 0.45
                reasons.append("shared-channel-caution")
            elif tool_name in _INSPECT_TOOLS | _PLANNING_TOOLS:
                delta += 0.25
                reasons.append("shared-channel-inspect-first")

        if "tool_use.change_strategy_after_retries" in keys and tool_name in recent_failed:
            delta -= 1.0
            reasons.append("recent-failing-path")

        if policy_state_plan is not None and plan_weights:
            inspect_weight = max(0.0, float(plan_weights.get("inspect", 0.0)))
            if inspect_weight:
                if tool_name in _INSPECT_TOOLS:
                    delta += 0.95 * inspect_weight
                    reasons.append("policy-plan:inspect")
                elif tool_name in _MUTATING_TOOLS:
                    delta -= 0.22 * inspect_weight
                    reasons.append("policy-plan:defer-mutation")

            planning_weight = max(0.0, float(plan_weights.get("planning", 0.0)))
            if planning_weight:
                if tool_name in _PLANNING_TOOLS:
                    delta += 0.95 * planning_weight
                    reasons.append("policy-plan:planning")
                elif tool_name in _MUTATING_TOOLS | _EXTERNAL_TOOLS:
                    delta -= 0.18 * planning_weight
                    reasons.append("policy-plan:delay-execution")

            clarify_weight = max(0.0, float(plan_weights.get("clarify", 0.0)))
            if clarify_weight and tool_name == "clarify":
                delta += 1.10 * clarify_weight
                reasons.append("policy-plan:clarify")

            local_weight = max(0.0, float(plan_weights.get("local", 0.0)))
            if local_weight and task_type == "repo_modification":
                if tool_name in _LOCAL_TOOLS:
                    delta += 0.90 * local_weight
                    reasons.append("policy-plan:local-first")
                elif tool_name in {"web_search", "web_extract", "browser_navigate"}:
                    delta -= 0.70 * local_weight
                    reasons.append("policy-plan:penalize-external")

            mutating_weight = float(plan_weights.get("mutating", 0.0))
            if mutating_weight < 0 and tool_name in _MUTATING_TOOLS:
                delta += mutating_weight
                reasons.append("policy-plan:mutating-penalty")

            external_weight = float(plan_weights.get("external", 0.0))
            if external_weight < 0 and tool_name in _EXTERNAL_TOOLS:
                delta += external_weight
                reasons.append("policy-plan:external-penalty")

            retry_failed_weight = float(plan_weights.get("retry_failed", 0.0))
            if retry_failed_weight < 0 and tool_name in recent_failed:
                delta += retry_failed_weight
                reasons.append("policy-plan:avoid-retry-loop")

            if planner_mode == "clarify_first" and tool_name in _MUTATING_TOOLS | _EXTERNAL_TOOLS:
                delta -= 0.20
                reasons.append("policy-plan:clarify-first")
        else:
            inspect_tendency = max(0.0, float(state_values.get("inspect_tendency", 0.0)))
            if inspect_tendency:
                if tool_name in _INSPECT_TOOLS:
                    delta += 0.85 * inspect_tendency
                    reasons.append("policy-state:inspect-tendency")
                elif tool_name in _MUTATING_TOOLS:
                    delta -= 0.20 * inspect_tendency
                    reasons.append("policy-state:defer-mutation")

            risk_aversion = max(0.0, float(state_values.get("risk_aversion", 0.0)))
            if risk_aversion:
                if tool_name in _INSPECT_TOOLS:
                    delta += 0.55 * risk_aversion
                    reasons.append("policy-state:risk-aversion")
                elif tool_name in _MUTATING_TOOLS | _EXTERNAL_TOOLS:
                    delta -= 0.35 * risk_aversion
                    reasons.append("policy-state:risk-penalty")

            local_first = max(0.0, float(state_values.get("local_first_tendency", 0.0)))
            if local_first and task_type == "repo_modification":
                if tool_name in _LOCAL_TOOLS:
                    delta += 0.85 * local_first
                    reasons.append("policy-state:local-first")
                elif tool_name in {"web_search", "web_extract", "browser_navigate"}:
                    delta -= 0.65 * local_first
                    reasons.append("policy-state:penalize-external")

            decomposition = max(0.0, float(state_values.get("decomposition_tendency", 0.0)))
            if decomposition:
                if tool_name in _PLANNING_TOOLS:
                    delta += 0.90 * decomposition
                    reasons.append("policy-state:decompose")
                elif tool_name in _MUTATING_TOOLS | _EXTERNAL_TOOLS:
                    delta -= 0.25 * decomposition
                    reasons.append("policy-state:delay-execution")

            retry_switch = max(0.0, float(state_values.get("retry_switch_tendency", 0.0)))
            if retry_switch and tool_name in recent_failed:
                delta -= 1.10 * retry_switch
                reasons.append("policy-state:avoid-retry-loop")

            shared_channel_caution = max(0.0, float(state_values.get("shared_channel_caution", 0.0)))
            if shared_channel_caution and _looks_shared_platform(platform):
                if tool_name in _MUTATING_TOOLS | {"send_message", "cronjob", "ha_call_service"}:
                    delta -= 0.40 * shared_channel_caution
                    reasons.append("policy-state:shared-channel-caution")
                elif tool_name in _INSPECT_TOOLS | _PLANNING_TOOLS:
                    delta += 0.25 * shared_channel_caution
                    reasons.append("policy-state:shared-channel-inspect")

        if delta:
            deltas.append(ToolWeightDelta(tool_name=tool_name, weight_delta=delta, reasons=reasons))
            planner_effects.append(
                {"tool_name": tool_name, "weight_delta": round(delta, 3), "reasons": reasons}
            )

        scored.append((delta, -index, copy.deepcopy(tool_def)))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ranked = [item[2] for item in scored]
    return ranked, deltas, planner_effects


def rerank_tool_calls(
    parsed_calls: list[tuple[object, str, dict]],
    biases: list[PolicyBias],
    *,
    recent_failed_tools: Iterable[str],
    policy_state: list[PolicyStateDimension] | None = None,
    policy_state_plan: PolicyStatePlan | None = None,
) -> tuple[list[tuple[object, str, dict]], list[dict]]:
    if len(parsed_calls) <= 1:
        return parsed_calls, []

    keys = _candidate_keys(biases)
    state = dimension_map(policy_state)
    state_values = dict(policy_state_plan.effective_values) if policy_state_plan is not None else {
        key: float(dimension.value)
        for key, dimension in state.items()
    }
    recent_failed = set(recent_failed_tools or [])
    plan_weights = dict(policy_state_plan.tool_class_weights) if policy_state_plan is not None else {}
    if not keys and not state and policy_state_plan is None:
        return parsed_calls, []

    def _priority(item: tuple[object, str, dict]) -> tuple[int, int]:
        _tool_call, tool_name, _args = item
        score = 0
        if "risk.inspect_before_execute" in keys or "planning.inspect_before_edit" in keys:
            if tool_name in _INSPECT_TOOLS:
                score += 30
            elif tool_name in _MUTATING_TOOLS:
                score -= 10
        if "workflow_specific.decompose_before_act" in keys:
            if tool_name in _PLANNING_TOOLS:
                score += 22
            elif tool_name in _MUTATING_TOOLS | _EXTERNAL_TOOLS:
                score -= 8
        if "user_specific.one_step_at_a_time" in keys:
            if tool_name in _PLANNING_TOOLS | _INSPECT_TOOLS:
                score += 14
            elif tool_name in _MUTATING_TOOLS | _EXTERNAL_TOOLS:
                score -= 6
        if "tool_use.change_strategy_after_retries" in keys and tool_name in recent_failed:
            score -= 25
        if policy_state_plan is not None and plan_weights:
            if tool_name in _INSPECT_TOOLS:
                score += int(18 * max(0.0, float(plan_weights.get("inspect", 0.0))))
            if tool_name in _PLANNING_TOOLS:
                score += int(16 * max(0.0, float(plan_weights.get("planning", 0.0))))
            if tool_name == "clarify":
                score += int(20 * max(0.0, float(plan_weights.get("clarify", 0.0))))
            if tool_name in _LOCAL_TOOLS:
                score += int(14 * max(0.0, float(plan_weights.get("local", 0.0))))
            if tool_name in _MUTATING_TOOLS:
                score += int(16 * float(plan_weights.get("mutating", 0.0)))
            if tool_name in _EXTERNAL_TOOLS:
                score += int(16 * float(plan_weights.get("external", 0.0)))
            if tool_name in recent_failed:
                score += int(18 * float(plan_weights.get("retry_failed", 0.0)))
        else:
            inspect_tendency = max(0.0, float(state_values.get("inspect_tendency", 0.0)))
            if inspect_tendency:
                if tool_name in _INSPECT_TOOLS:
                    score += int(18 * inspect_tendency)
                elif tool_name in _MUTATING_TOOLS:
                    score -= int(7 * inspect_tendency)
            decomposition = max(0.0, float(state_values.get("decomposition_tendency", 0.0)))
            if decomposition:
                if tool_name in _PLANNING_TOOLS:
                    score += int(16 * decomposition)
                elif tool_name in _MUTATING_TOOLS | _EXTERNAL_TOOLS:
                    score -= int(6 * decomposition)
            retry_switch = max(0.0, float(state_values.get("retry_switch_tendency", 0.0)))
            if retry_switch and tool_name in recent_failed:
                score -= int(18 * retry_switch)
        if policy_state_plan is not None and policy_state_plan.require_sequential:
            if tool_name in _PLANNING_TOOLS | _INSPECT_TOOLS:
                score += 8
            elif tool_name in _MUTATING_TOOLS | _EXTERNAL_TOOLS:
                score -= 4
        return (score, 0)

    ordered = sorted(parsed_calls, key=_priority, reverse=True)
    if ordered == parsed_calls:
        return parsed_calls, []

    planner_effects = [
        {
            "action": "reordered_tool_calls",
            "before": [name for _, name, _ in parsed_calls],
            "after": [name for _, name, _ in ordered],
        }
    ]
    return ordered, planner_effects


def evaluate_risk_gate(
    tool_name: str,
    function_args: dict[str, object],
    biases: list[PolicyBias],
    *,
    require_inspect_first: bool,
    has_recent_inspection: bool,
    user_message: str = "",
    platform: str = "",
    policy_state: list[PolicyStateDimension] | None = None,
    policy_state_plan: PolicyStatePlan | None = None,
) -> RiskAction | None:
    keys = _candidate_keys(biases)
    state = dimension_map(policy_state)
    side_effect_level = side_effect_level_for_tool(tool_name, function_args)
    if side_effect_level not in {"medium", "high"}:
        return None
    if tool_name in _INSPECT_TOOLS:
        return None

    bias_ids = [bias.id for bias in biases if (bias.bias_candidate_key or "") in {
        "risk.inspect_before_execute",
        "platform_specific.group_chat_caution",
    }]
    explicit_intent = _has_explicit_execute_intent(user_message, function_args)
    ambiguous = looks_ambiguous_request(user_message)
    state_values = dict(policy_state_plan.effective_values) if policy_state_plan is not None else {
        key: float(dimension.value)
        for key, dimension in state.items()
    }
    shared_channel_state = float(state_values.get("shared_channel_caution", 0.0))
    shared_platform = (
        "platform_specific.group_chat_caution" in keys or shared_channel_state >= 0.35
    ) and _looks_shared_platform(platform)
    risk_aversion = max(0.0, float(state_values.get("risk_aversion", 0.0)))
    inspect_tendency = max(0.0, float(state_values.get("inspect_tendency", 0.0)))
    preferred_risk_mode = policy_state_plan.preferred_risk_mode if policy_state_plan is not None else "direct"
    clarify_priority = policy_state_plan.clarify_priority if policy_state_plan is not None else 0.0
    available_tools = set(policy_state_plan.available_tools) if policy_state_plan is not None else set()
    execution_mode_scores = dict(policy_state_plan.execution_mode_scores) if policy_state_plan is not None else {}
    direct_score = float(execution_mode_scores.get("direct", 0.0))
    inspect_score = float(execution_mode_scores.get("inspect", 0.0))
    simulate_score = float(execution_mode_scores.get("simulate", 0.0))
    confirm_score = float(execution_mode_scores.get("confirm", 0.0))
    clarify_score = float(execution_mode_scores.get("clarify", 0.0))
    clarify_tool = "clarify" if "clarify" in available_tools else None

    if shared_platform and tool_name in {"send_message", "cronjob", "ha_call_service"} and not explicit_intent:
        return RiskAction(
            tool_name=tool_name,
            decision="confirm",
            reason="Shared-channel caution bias requires explicit confirmation before taking an external side-effect action in a group or channel context.",
            suggested_tool=clarify_tool or "web_search",
            bias_ids=bias_ids,
        )

    require_state_inspect = require_inspect_first and (
        risk_aversion >= 0.35 or inspect_tendency >= 0.45 or preferred_risk_mode in {"inspect", "confirm"}
    )
    if "risk.inspect_before_execute" not in keys and not require_state_inspect:
        return None
    if has_recent_inspection:
        if side_effect_level == "high" and (
            ambiguous
            or risk_aversion >= 0.55
            or preferred_risk_mode == "confirm"
            or confirm_score >= max(direct_score + 0.05, 0.55)
            or (tool_name in {"send_message", "cronjob", "ha_call_service"} and not explicit_intent)
        ):
            return RiskAction(
                tool_name=tool_name,
                decision="confirm",
                reason="Policy bias requires stronger certainty before a high-side-effect external action can proceed, even after inspection.",
                suggested_tool=clarify_tool or "web_search",
                bias_ids=bias_ids,
            )
        return None

    suggested_tool = "browser_snapshot" if tool_name.startswith("browser_") else "read_file"
    if tool_name in {"send_message", "cronjob", "ha_call_service"}:
        suggested_tool = "web_search"
    if tool_name == "terminal":
        suggested_tool = "search_files"
    if ambiguous and clarify_tool and clarify_priority >= 0.45:
        suggested_tool = clarify_tool

    decision = "inspect"
    reason = "Policy bias requires an inspect/search step before executing a mutating or external action."
    if tool_name in {"browser_click", "browser_type", "browser_press"}:
        if confirm_score >= max(simulate_score + 0.10, 0.82) and (ambiguous or shared_platform):
            decision = "confirm"
            reason = "Policy state currently prefers confirmation before this browser side-effect action."
        elif simulate_score >= 0.35 or not execution_mode_scores:
            decision = "simulate"
            reason = "Policy state currently prefers a simulate/preview step before this browser side-effect action."
    elif tool_name in {"send_message", "cronjob", "ha_call_service"}:
        if clarify_tool and ambiguous and clarify_score >= max(confirm_score - 0.05, 0.45):
            decision = "confirm"
            reason = "Policy state prefers clarifying intent before this external side-effect action."
        elif (
            ambiguous
            or not explicit_intent
            or risk_aversion >= 0.55
            or preferred_risk_mode == "confirm"
            or confirm_score >= max(inspect_score, direct_score + 0.05, 0.45)
        ):
            decision = "confirm"
            reason = "Policy state prefers stronger certainty before this external side-effect action."
    elif tool_name == "terminal":
        command = str(function_args.get("command", "") or "").lower()
        if any(token in command for token in ("rm ", "git reset", "git clean", "truncate ", "dd ")):
            decision = "confirm"
            reason = "Policy state prefers explicit confirmation before destructive terminal commands."
        elif inspect_score >= max(direct_score, 0.35):
            decision = "inspect"
            reason = "Policy state prefers verifying the terminal action context before execution."
    elif side_effect_level == "high" and confirm_score >= max(inspect_score + 0.12, direct_score + 0.12, 0.72):
        decision = "confirm"
        reason = "Policy state currently prefers confirmation before this high-side-effect action."

    return RiskAction(
        tool_name=tool_name,
        decision=decision,
        reason=reason,
        suggested_tool=suggested_tool,
        bias_ids=bias_ids,
    )


def make_blocked_tool_result(risk_action: RiskAction) -> str:
    import json

    return json.dumps(
        {
            "success": False,
            "status": "blocked_by_policy_bias",
            "decision": risk_action.decision,
            "error": risk_action.reason,
            "suggested_next_step": risk_action.suggested_tool,
            "tool_name": risk_action.tool_name,
            "bias_ids": risk_action.bias_ids,
        },
        ensure_ascii=False,
    )
