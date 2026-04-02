"""Explicit planner/tool weighting hooks driven by policy biases."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Iterable

from .models import PolicyBias, RiskAction, ToolWeightDelta
from .scoring import side_effect_level_for_tool

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


def _is_high_ambiguity_request(user_message: str) -> bool:
    text = (user_message or "").lower()
    return any(
        token in text
        for token in (
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
) -> tuple[list[dict], list[ToolWeightDelta], list[dict]]:
    if not tool_defs:
        return [], [], []

    keys = _candidate_keys(biases)
    recent_failed = set(recent_failed_tools or [])
    deltas: list[ToolWeightDelta] = []
    planner_effects: list[dict] = []
    scored: list[tuple[float, int, dict]] = []

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
) -> tuple[list[tuple[object, str, dict]], list[dict]]:
    if len(parsed_calls) <= 1:
        return parsed_calls, []

    keys = _candidate_keys(biases)
    recent_failed = set(recent_failed_tools or [])
    if not keys:
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
) -> RiskAction | None:
    keys = _candidate_keys(biases)
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
    ambiguous = _is_high_ambiguity_request(user_message)
    shared_platform = "platform_specific.group_chat_caution" in keys and _looks_shared_platform(platform)

    if shared_platform and tool_name in {"send_message", "cronjob", "ha_call_service"} and not explicit_intent:
        return RiskAction(
            tool_name=tool_name,
            decision="confirm",
            reason="Shared-channel caution bias requires explicit confirmation before taking an external side-effect action in a group or channel context.",
            suggested_tool="web_search",
            bias_ids=bias_ids,
        )

    if "risk.inspect_before_execute" not in keys or not require_inspect_first:
        return None
    if has_recent_inspection:
        if side_effect_level == "high" and (ambiguous or (tool_name in {"send_message", "cronjob", "ha_call_service"} and not explicit_intent)):
            return RiskAction(
                tool_name=tool_name,
                decision="confirm",
                reason="Policy bias requires stronger certainty before a high-side-effect external action can proceed, even after inspection.",
                suggested_tool="web_search",
                bias_ids=bias_ids,
            )
        return None

    suggested_tool = "browser_snapshot" if tool_name.startswith("browser_") else "read_file"
    if tool_name in {"send_message", "cronjob", "ha_call_service"}:
        suggested_tool = "web_search"
    if tool_name == "terminal":
        suggested_tool = "search_files"

    decision = "inspect"
    reason = "Policy bias requires an inspect/search step before executing a mutating or external action."
    if tool_name in {"browser_click", "browser_type", "browser_press"}:
        decision = "simulate"
        reason = "Policy bias requires a simulate/preview step before taking a browser action with side effects."
    elif tool_name in {"send_message", "cronjob", "ha_call_service"} and (ambiguous or not explicit_intent):
        decision = "confirm"
        reason = "Policy bias requires stronger certainty before executing this external side-effect action."
    elif tool_name == "terminal":
        command = str(function_args.get("command", "") or "").lower()
        if any(token in command for token in ("rm ", "git reset", "git clean", "truncate ", "dd ")):
            decision = "confirm"
            reason = "Policy bias requires explicit confirmation before destructive terminal commands."

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
