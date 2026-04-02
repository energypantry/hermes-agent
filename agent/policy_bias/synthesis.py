"""Moment-to-bias synthesis logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .models import PolicyBias, PolicyBiasConfig, PolicyMoment, new_id, now_ts
from .scoring import compute_bias_confidence, recency_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CandidateDescriptor:
    scope: str
    condition_signature: str
    preferred_policy: str
    anti_policy: Optional[str]
    rationale_summary: str


_CANDIDATE_DESCRIPTORS: dict[str, CandidateDescriptor] = {
    "planning.inspect_before_edit": CandidateDescriptor(
        scope="planning",
        condition_signature="task=repo_modification;strategy=inspect_before_edit",
        preferred_policy="For repo modification tasks, inspect file structure and conventions before editing.",
        anti_policy="Do not patch or rewrite code before reading the relevant files.",
        rationale_summary="Positive repo-edit turns repeatedly succeeded when Hermes inspected local files before editing.",
    ),
    "planning.search_before_fresh_answer": CandidateDescriptor(
        scope="planning",
        condition_signature="task=current_info;strategy=search_before_answer",
        preferred_policy="In ambiguous or fresh-topic requests, search/browse before answering.",
        anti_policy="Do not answer fresh or current-topic requests from stale assumptions.",
        rationale_summary="Fresh-topic tasks produced better outcomes when Hermes searched first.",
    ),
    "tool_use.local_before_external_code": CandidateDescriptor(
        scope="tool_use",
        condition_signature="task=repo_modification;tool_path=local_before_external",
        preferred_policy="For code-local tasks, prefer local read/search tools before external browser or web fetches.",
        anti_policy="Avoid defaulting to external browsing when the task is local to the repo.",
        rationale_summary="Code-local tasks consistently resolved faster when Hermes stayed local first.",
    ),
    "tool_use.patch_before_rewrite": CandidateDescriptor(
        scope="tool_use",
        condition_signature="task=repo_modification;edit_strategy=targeted_patch",
        preferred_policy="Prefer targeted patch updates over destructive full-file rewrites.",
        anti_policy="Avoid full rewrites when a narrow patch will do.",
        rationale_summary="Targeted patches preserved context and reduced regressions compared with broad rewrites.",
    ),
    "tool_use.change_strategy_after_retries": CandidateDescriptor(
        scope="tool_use",
        condition_signature="failure_pattern=repeated_same_tool_retry",
        preferred_policy="When repeated tool retries fail similarly, change strategy instead of retrying the same path.",
        anti_policy="Do not keep retrying the same failing tool path without new information.",
        rationale_summary="Repeated retries without strategy change led to poorer outcomes than switching approach.",
    ),
    "risk.inspect_before_execute": CandidateDescriptor(
        scope="risk",
        condition_signature="side_effects=external_or_mutating;strategy=inspect_before_execute",
        preferred_policy="When an action has external side effects, inspect/search/simulate before execute.",
        anti_policy="Avoid direct external or mutating actions before inspection.",
        rationale_summary="External-action tasks were safer and more reliable when Hermes inspected before executing.",
    ),
    "communication.concise_first": CandidateDescriptor(
        scope="communication",
        condition_signature="user_pref=concise_first",
        preferred_policy="Prefer a concise direct answer first, with deeper detail only when needed.",
        anti_policy="Avoid leading with unnecessary verbosity for this user.",
        rationale_summary="The user repeatedly signaled preference for concise, direct answers.",
    ),
    "user_specific.directness_over_fluff": CandidateDescriptor(
        scope="user_specific",
        condition_signature="user_pref=directness_over_fluff",
        preferred_policy="Prioritize directness and production-ready output over conceptual framing for this user.",
        anti_policy="Avoid fluff, cheerleading, or unnecessary framing for this user.",
        rationale_summary="Repeated user instructions favored direct, execution-focused responses.",
    ),
    "user_specific.one_step_at_a_time": CandidateDescriptor(
        scope="user_specific",
        condition_signature="user_pref=one_executable_step_at_a_time",
        preferred_policy="When ambiguity remains, prefer one concrete executable step at a time for this user.",
        anti_policy="Avoid jumping to multi-branch plans when the user wants a single next action.",
        rationale_summary="The user repeatedly asked for the next minimal executable step instead of a large plan.",
    ),
    "platform_specific.group_chat_caution": CandidateDescriptor(
        scope="platform_specific",
        condition_signature="platform=group_chat;caution=high",
        preferred_policy="In group-chat contexts, be more cautious with assumptions and external actions.",
        anti_policy="Avoid casual high-side-effect actions in shared channels.",
        rationale_summary="Shared-channel interactions require stronger caution than direct CLI sessions.",
    ),
    "workflow_specific.decompose_before_act": CandidateDescriptor(
        scope="workflow_specific",
        condition_signature="workflow=complex_task;strategy=decompose_before_act",
        preferred_policy="Decompose multi-part tasks before acting when the workflow is complex.",
        anti_policy="Avoid jumping straight into execution on complex workflows without decomposition.",
        rationale_summary="Complex workflows performed better when Hermes decomposed the work before acting.",
    ),
    "workflow_specific.structured_debugging_output": CandidateDescriptor(
        scope="workflow_specific",
        condition_signature="workflow=debugging;output=structured_findings_first",
        preferred_policy="For debugging and review tasks, present findings and risks first in structured form.",
        anti_policy="Avoid burying actionable debugging findings behind broad summaries.",
        rationale_summary="Debugging tasks were clearer when Hermes led with structured findings first.",
    ),
}


def get_candidate_descriptor(candidate_key: str) -> Optional[CandidateDescriptor]:
    return _CANDIDATE_DESCRIPTORS.get(candidate_key)


def synthesize_bias(
    *,
    config: PolicyBiasConfig,
    profile_id: str,
    candidate_key: str,
    moments: list[PolicyMoment],
    existing: Optional[PolicyBias] = None,
    now: Optional[float] = None,
) -> Optional[PolicyBias]:
    descriptor = get_candidate_descriptor(candidate_key)
    if descriptor is None or not moments:
        return None

    now = now if now is not None else now_ts()
    moments = sorted(moments, key=lambda item: item.timestamp)
    support_count = len(moments)
    avg_reward = sum(moment.reward_score for moment in moments) / support_count
    recency = (
        sum(
            recency_score(
                moment.timestamp,
                now=now,
                decay_per_day=config.confidence_decay_per_day,
            )
            for moment in moments
        )
        / support_count
    )
    confidence = compute_bias_confidence(
        support_count=support_count,
        avg_reward=avg_reward,
        recency=recency,
    )

    status = "shadow"
    has_strong_signal = any(
        abs(moment.reward_score) >= config.strong_signal_reward for moment in moments
    )
    if avg_reward >= config.min_avg_reward and support_count >= config.min_support_count:
        status = "active"
    elif has_strong_signal or config.shadow_mode_default:
        status = "shadow"

    if avg_reward < 0:
        status = "shadow" if support_count < (config.min_support_count + 1) else "disabled"
        confidence = min(confidence, 0.45)

    if (
        descriptor.scope == "risk"
        and config.high_side_effect_shadow_only
        and status == "active"
    ):
        status = "shadow"

    if existing is not None and existing.status in {"disabled", "archived"} and avg_reward >= 0:
        status = existing.status

    source_moment_ids = [moment.id for moment in moments[-12:]]
    rationale = descriptor.rationale_summary
    if avg_reward < 0:
        rationale = (
            f"{descriptor.rationale_summary} Recent negative evidence is suppressing this bias."
        )

    bias = PolicyBias(
        id=existing.id if existing else new_id("bias"),
        profile_id=profile_id,
        scope=descriptor.scope,
        condition_signature=descriptor.condition_signature,
        preferred_policy=descriptor.preferred_policy,
        anti_policy=descriptor.anti_policy if avg_reward < 0 else descriptor.anti_policy,
        rationale_summary=rationale,
        confidence=confidence,
        support_count=support_count,
        avg_reward=avg_reward,
        recency_score=recency,
        decay_rate=config.confidence_decay_per_day,
        status=status,
        source_moment_ids=source_moment_ids,
        created_at=existing.created_at if existing else now,
        updated_at=now,
        last_triggered_at=existing.last_triggered_at if existing else None,
        trigger_count=existing.trigger_count if existing else 0,
        rollback_parent_id=existing.rollback_parent_id if existing else None,
        version=existing.version if existing else 1,
        bias_candidate_key=candidate_key,
        status_note=existing.status_note if existing else None,
    )

    logger.debug(
        "Synthesized bias %s status=%s support=%s avg_reward=%.3f confidence=%.3f",
        candidate_key,
        bias.status,
        bias.support_count,
        bias.avg_reward,
        bias.confidence,
    )
    return bias
