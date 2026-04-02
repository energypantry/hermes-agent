"""Deterministic response-policy hooks driven by policy biases."""

from __future__ import annotations

import re
from typing import Iterable

from .models import PolicyBias, PolicyStateDimension, PolicyStatePlan
from .state_runtime import dimension_map

_LEADING_ACK_PATTERNS = (
    "got it",
    "understood",
    "sure",
    "absolutely",
    "of course",
    "sounds good",
    "done",
    "好的",
    "明白",
    "没问题",
    "收到",
)

_TRAILING_OFFER_PATTERNS = (
    "let me know if you want",
    "if you want, i can",
    "i can also",
    "happy to",
    "需要的话我可以",
    "如果你需要",
    "如果要的话我可以",
)

_DEBUG_REQUEST_MARKERS = (
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
    "review",
)

_NUMBERED_STEP_RE = re.compile(r"^\s*\d+\.\s+")


def derive_response_controls(
    biases: Iterable[PolicyBias],
    *,
    task_type: str,
    user_message: str,
    policy_state: Iterable[PolicyStateDimension] | None = None,
    policy_state_plan: PolicyStatePlan | None = None,
) -> dict[str, object]:
    if policy_state_plan is not None and policy_state_plan.response_controls:
        return dict(policy_state_plan.response_controls)

    keys = {
        bias.bias_candidate_key or ""
        for bias in biases
    }
    state = dimension_map(policy_state)
    controls: dict[str, object] = {}
    if {"communication.concise_first", "user_specific.directness_over_fluff"} & keys:
        controls["strip_leading_acknowledgement"] = True
        controls["drop_trailing_offer"] = True
    directness = float(state.get("directness_tendency").value) if state.get("directness_tendency") else 0.0
    if directness >= 0.35:
        controls["strip_leading_acknowledgement"] = True
        controls["drop_trailing_offer"] = True
    if "workflow_specific.structured_debugging_output" in keys and _looks_debug_request(
        user_message=user_message,
        task_type=task_type,
    ):
        controls["findings_first_heading"] = True
    findings_first = (
        float(state.get("findings_first_tendency").value)
        if state.get("findings_first_tendency")
        else 0.0
    )
    if findings_first >= 0.35 and _looks_debug_request(
        user_message=user_message,
        task_type=task_type,
    ):
        controls["findings_first_heading"] = True
    if "user_specific.one_step_at_a_time" in keys:
        controls["prefer_single_step"] = True
    single_step = (
        float(state.get("single_step_tendency").value)
        if state.get("single_step_tendency")
        else 0.0
    )
    if single_step >= 0.35:
        controls["prefer_single_step"] = True
    return controls


def apply_response_controls(
    text: str,
    controls: dict[str, object] | None,
) -> tuple[str, list[dict[str, object]]]:
    content = (text or "").strip()
    if not content or not controls:
        return content, []

    effects: list[dict[str, object]] = []
    lines = content.splitlines()

    if controls.get("strip_leading_acknowledgement"):
        while len(lines) > 1 and _looks_like_ack_line(lines[0]):
            removed = lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
            effects.append(
                {
                    "effect": "strip_leading_acknowledgement",
                    "removed": removed.strip(),
                }
            )

    if controls.get("drop_trailing_offer"):
        while len(lines) > 1 and not lines[-1].strip():
            lines.pop()
        if len(lines) > 1 and _looks_like_trailing_offer(lines[-1]):
            removed = lines.pop()
            while lines and not lines[-1].strip():
                lines.pop()
            effects.append(
                {
                    "effect": "drop_trailing_offer",
                    "removed": removed.strip(),
                }
            )

    content = "\n".join(lines).strip()

    if controls.get("findings_first_heading") and content:
        lowered = content.lower()
        if not (lowered.startswith("**findings**") or lowered.startswith("findings")):
            content = f"**Findings**\n{content}"
            effects.append({"effect": "prepend_findings_heading"})

    max_numbered_steps = int(controls.get("max_numbered_steps") or 0)
    if max_numbered_steps > 0 and content:
        trimmed, changed = _limit_numbered_steps(content, max_numbered_steps)
        if changed:
            content = trimmed
            effects.append({"effect": "limit_numbered_steps", "limit": max_numbered_steps})

    if controls.get("prefer_single_step") and content:
        trimmed, changed = _trim_to_single_step(content)
        if changed:
            content = trimmed
            effects.append({"effect": "trim_to_single_step"})

    return content, effects


def _looks_debug_request(*, user_message: str, task_type: str) -> bool:
    lowered = (user_message or "").lower()
    if any(marker in lowered for marker in _DEBUG_REQUEST_MARKERS):
        return True
    return task_type == "repo_modification"


def _looks_like_ack_line(line: str) -> bool:
    lowered = (line or "").strip().lower().strip(" .!,:;")
    if not lowered:
        return False
    return any(lowered == token or lowered.startswith(token + " ") for token in _LEADING_ACK_PATTERNS)


def _looks_like_trailing_offer(line: str) -> bool:
    lowered = (line or "").strip().lower().strip(" .!,:;")
    if not lowered:
        return False
    return any(lowered.startswith(token) for token in _TRAILING_OFFER_PATTERNS)


def _trim_to_single_step(content: str) -> tuple[str, bool]:
    return _limit_numbered_steps(content, 1)


def _limit_numbered_steps(content: str, limit: int) -> tuple[str, bool]:
    if limit <= 0:
        return content, False
    lines = content.splitlines()
    step_indexes = [
        index
        for index, line in enumerate(lines)
        if _NUMBERED_STEP_RE.match(line.strip())
    ]
    if len(step_indexes) <= limit:
        return content, False
    cutoff = step_indexes[limit]
    trimmed = "\n".join(lines[:cutoff]).strip()
    return trimmed or content, bool(trimmed and trimmed != content)
