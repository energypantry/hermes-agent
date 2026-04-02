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
    action_surfaces: tuple[str, ...]
    why_not_memory: str
    why_not_skill: str


_CANDIDATE_DESCRIPTORS: dict[str, CandidateDescriptor] = {
    "planning.inspect_before_edit": CandidateDescriptor(
        scope="planning",
        condition_signature="task=repo_modification;strategy=inspect_before_edit",
        preferred_policy="For repo modification tasks, inspect file structure and conventions before editing.",
        anti_policy="Do not patch or rewrite code before reading the relevant files.",
        rationale_summary="Positive repo-edit turns repeatedly succeeded when Hermes inspected local files before editing.",
        action_surfaces=("prompt", "tool_ranking", "tool_batch"),
        why_not_memory="This is a reusable decision prior about how to approach repo edits, not a factual note about the user or environment.",
        why_not_skill="It biases ordering and gating for many repo-edit tasks, rather than encoding a full task-specific procedure.",
    ),
    "planning.search_before_fresh_answer": CandidateDescriptor(
        scope="planning",
        condition_signature="task=current_info;strategy=search_before_answer",
        preferred_policy="In ambiguous or fresh-topic requests, search/browse before answering.",
        anti_policy="Do not answer fresh or current-topic requests from stale assumptions.",
        rationale_summary="Fresh-topic tasks produced better outcomes when Hermes searched first.",
        action_surfaces=("prompt", "tool_ranking"),
        why_not_memory="This governs how to act on fresh-topic requests under uncertainty, not a stable fact to remember.",
        why_not_skill="It is a compact planning prior, not a reusable multi-step recipe for one narrow workflow.",
    ),
    "tool_use.local_before_external_code": CandidateDescriptor(
        scope="tool_use",
        condition_signature="task=repo_modification;tool_path=local_before_external",
        preferred_policy="For code-local tasks, prefer local read/search tools before external browser or web fetches.",
        anti_policy="Avoid defaulting to external browsing when the task is local to the repo.",
        rationale_summary="Code-local tasks consistently resolved faster when Hermes stayed local first.",
        action_surfaces=("prompt", "tool_ranking"),
        why_not_memory="This changes tool-selection behavior for code-local work instead of storing a fact about a specific repo state.",
        why_not_skill="It is a cross-task tool preference, not a complete procedure for a single class of task.",
    ),
    "tool_use.patch_before_rewrite": CandidateDescriptor(
        scope="tool_use",
        condition_signature="task=repo_modification;edit_strategy=targeted_patch",
        preferred_policy="Prefer targeted patch updates over destructive full-file rewrites.",
        anti_policy="Avoid full rewrites when a narrow patch will do.",
        rationale_summary="Targeted patches preserved context and reduced regressions compared with broad rewrites.",
        action_surfaces=("prompt", "tool_ranking"),
        why_not_memory="This is an action preference about edit strategy, not a durable fact to inject as memory.",
        why_not_skill="It nudges choice among edit paths; it does not replace a reusable end-to-end coding workflow.",
    ),
    "tool_use.change_strategy_after_retries": CandidateDescriptor(
        scope="tool_use",
        condition_signature="failure_pattern=repeated_same_tool_retry",
        preferred_policy="When repeated tool retries fail similarly, change strategy instead of retrying the same path.",
        anti_policy="Do not keep retrying the same failing tool path without new information.",
        rationale_summary="Repeated retries without strategy change led to poorer outcomes than switching approach.",
        action_surfaces=("prompt", "tool_ranking", "tool_batch"),
        why_not_memory="This responds to repeated-failure dynamics during execution, rather than recording a fact about the world.",
        why_not_skill="It is a compact fallback prior for many tool paths, not a standalone reusable playbook.",
    ),
    "risk.inspect_before_execute": CandidateDescriptor(
        scope="risk",
        condition_signature="side_effects=external_or_mutating;strategy=inspect_before_execute",
        preferred_policy="When an action has external side effects, inspect/search/simulate before execute.",
        anti_policy="Avoid direct external or mutating actions before inspection.",
        rationale_summary="External-action tasks were safer and more reliable when Hermes inspected before executing.",
        action_surfaces=("prompt", "tool_ranking", "risk_gate"),
        why_not_memory="This is a safety policy for how to act around side effects, not a remembered fact.",
        why_not_skill="It applies broadly across tools and sessions as a gating prior, not as a narrow procedural recipe.",
    ),
    "communication.concise_first": CandidateDescriptor(
        scope="communication",
        condition_signature="user_pref=concise_first",
        preferred_policy="Prefer a concise direct answer first, with deeper detail only when needed.",
        anti_policy="Avoid leading with unnecessary verbosity for this user.",
        rationale_summary="The user repeatedly signaled preference for concise, direct answers.",
        action_surfaces=("prompt", "response_policy"),
        why_not_memory="The remembered fact is that the user likes brevity; the bias is the decision rule to default to concise-first responses.",
        why_not_skill="This is a stable response prior, not an executable workflow that should live as a skill.",
    ),
    "user_specific.directness_over_fluff": CandidateDescriptor(
        scope="user_specific",
        condition_signature="user_pref=directness_over_fluff",
        preferred_policy="Prioritize directness and production-ready output over conceptual framing for this user.",
        anti_policy="Avoid fluff, cheerleading, or unnecessary framing for this user.",
        rationale_summary="Repeated user instructions favored direct, execution-focused responses.",
        action_surfaces=("prompt", "response_policy"),
        why_not_memory="Memory can store the user's preference; this bias is the operational rule to apply that preference by default.",
        why_not_skill="It changes cross-task response policy rather than describing a reusable procedure.",
    ),
    "user_specific.one_step_at_a_time": CandidateDescriptor(
        scope="user_specific",
        condition_signature="user_pref=one_executable_step_at_a_time",
        preferred_policy="When ambiguity remains, prefer one concrete executable step at a time for this user.",
        anti_policy="Avoid jumping to multi-branch plans when the user wants a single next action.",
        rationale_summary="The user repeatedly asked for the next minimal executable step instead of a large plan.",
        action_surfaces=("prompt", "response_policy"),
        why_not_memory="The fact that the user prefers incremental guidance can be remembered, but this bias is the default decision policy it should trigger.",
        why_not_skill="It affects how Hermes sequences help across many tasks, not a standalone workflow to save as a skill.",
    ),
    "platform_specific.group_chat_caution": CandidateDescriptor(
        scope="platform_specific",
        condition_signature="platform=group_chat;caution=high",
        preferred_policy="In group-chat contexts, be more cautious with assumptions and external actions.",
        anti_policy="Avoid casual high-side-effect actions in shared channels.",
        rationale_summary="Shared-channel interactions require stronger caution than direct CLI sessions.",
        action_surfaces=("prompt", "risk_gate"),
        why_not_memory="This is a channel-specific operating prior, not a factual memory entry.",
        why_not_skill="It applies broadly whenever the platform changes, rather than teaching a reusable procedure.",
    ),
    "workflow_specific.decompose_before_act": CandidateDescriptor(
        scope="workflow_specific",
        condition_signature="workflow=complex_task;strategy=decompose_before_act",
        preferred_policy="Decompose multi-part tasks before acting when the workflow is complex.",
        anti_policy="Avoid jumping straight into execution on complex workflows without decomposition.",
        rationale_summary="Complex workflows performed better when Hermes decomposed the work before acting.",
        action_surfaces=("prompt", "tool_ranking"),
        why_not_memory="This is a decision prior about approaching complex work, not a remembered task fact.",
        why_not_skill="It is a general planning bias across many workflows, not a concrete step-by-step procedure.",
    ),
    "workflow_specific.structured_debugging_output": CandidateDescriptor(
        scope="workflow_specific",
        condition_signature="workflow=debugging;output=structured_findings_first",
        preferred_policy="For debugging and review tasks, present findings and risks first in structured form.",
        anti_policy="Avoid burying actionable debugging findings behind broad summaries.",
        rationale_summary="Debugging tasks were clearer when Hermes led with structured findings first.",
        action_surfaces=("prompt", "response_policy"),
        why_not_memory="This controls how Hermes should present debugging decisions, not a fact that belongs in memory.",
        why_not_skill="It is a cross-task output prior, not a reusable procedural asset with its own supporting files.",
    ),
}


def get_candidate_descriptor(candidate_key: str) -> Optional[CandidateDescriptor]:
    return _CANDIDATE_DESCRIPTORS.get(candidate_key)


def descriptor_qualifies_for_policy_bias(descriptor: CandidateDescriptor) -> bool:
    """Guardrail to keep policy bias distinct from memory and skills."""
    return bool(
        descriptor.action_surfaces
        and descriptor.why_not_memory.strip()
        and descriptor.why_not_skill.strip()
    )


def get_boundary_metadata(candidate_key: str) -> dict[str, object]:
    descriptor = get_candidate_descriptor(candidate_key)
    if descriptor is None:
        return {
            "candidate_key": candidate_key,
            "classification": "manual_review",
            "action_surfaces": [],
            "why_not_memory": "No descriptor is registered for this candidate key.",
            "why_not_skill": "No descriptor is registered for this candidate key.",
        }
    return {
        "candidate_key": candidate_key,
        "classification": (
            "policy_bias"
            if descriptor_qualifies_for_policy_bias(descriptor)
            else "manual_review"
        ),
        "action_surfaces": list(descriptor.action_surfaces),
        "why_not_memory": descriptor.why_not_memory,
        "why_not_skill": descriptor.why_not_skill,
    }


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
    if not descriptor_qualifies_for_policy_bias(descriptor):
        logger.warning(
            "Skipping candidate %s because it does not qualify as a policy bias boundary-safe descriptor",
            candidate_key,
        )
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
