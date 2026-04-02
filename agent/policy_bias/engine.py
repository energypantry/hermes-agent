"""Runtime coordinator for the Policy Bias Engine."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Iterable, Optional

from .explain import bias_summary
from .governance import explain_recent
from .injector import build_decision_priors
from .models import (
    BiasDecisionContext,
    DecisionTrace,
    PolicyBiasConfig,
    PolicyMoment,
    RetrievalResult,
    RiskAction,
    ToolWeightDelta,
    derive_profile_id,
    new_id,
    now_ts,
)
from .planner_hooks import evaluate_risk_gate, rerank_tool_calls, rerank_tools
from .response_hooks import derive_response_controls
from .retrieval import retrieve_biases
from .scoring import (
    classify_task_type,
    compute_moment_confidence,
    compute_reward_score,
    detect_feedback_signal,
    is_correction_message,
    side_effect_level_for_tool,
)
from .state_runtime import evidence_summary as state_evidence_summary
from .state_runtime import serialize_dimensions
from .store import PolicyBiasStore
from .synthesis import synthesize_bias

logger = logging.getLogger(__name__)

_INSPECT_TOOLS = {"read_file", "search_files", "browser_snapshot", "web_search", "web_extract", "ha_get_state", "ha_list_entities", "ha_list_services"}
_WEB_TOOLS = {"web_search", "web_extract", "browser_navigate"}
_LOCAL_TOOLS = {"read_file", "search_files", "patch", "write_file", "terminal"}


class PolicyBiasEngine:
    """Co-ordinates moments, synthesis, retrieval, injection, and governance."""

    def __init__(
        self,
        config: Optional[dict] = None,
        *,
        store: Optional[PolicyBiasStore] = None,
    ):
        self.config = (
            config if isinstance(config, PolicyBiasConfig) else PolicyBiasConfig.from_dict(config)
        )
        self.profile_id = derive_profile_id()
        self.enabled = bool(self.config.enabled)
        self._store_available = False
        self.store: Optional[PolicyBiasStore] = None
        if self.enabled:
            try:
                self.store = store or PolicyBiasStore()
                self._store_available = True
            except Exception as exc:
                logger.warning("Policy bias store init failed; feature disabled: %s", exc)
                self.enabled = False
        self._seen_feedback_fingerprints: set[str] = set()

    def close(self) -> None:
        if self.store:
            try:
                self.store.close()
            except Exception:
                pass

    def is_enabled(self) -> bool:
        return self.enabled and self._store_available and self.store is not None

    def begin_turn(
        self,
        *,
        session_id: str,
        turn_index: int,
        user_message: str,
        platform: str,
        available_tools: Iterable[str],
        tool_defs: list[dict],
    ) -> BiasDecisionContext:
        if not self.is_enabled():
            return BiasDecisionContext(
                session_id=session_id,
                turn_index=turn_index,
                task_type=classify_task_type(user_message, available_tools),
                platform=platform or "cli",
                user_message=user_message,
                ranked_tools=tool_defs,
            )

        self._capture_feedback_preferences(
            session_id=session_id,
            user_message=user_message,
            platform=platform or "cli",
            turn_index=turn_index,
            available_tools=available_tools,
        )

        task_type = classify_task_type(user_message, available_tools)
        recent_moments = self.store.list_recent_moments(
            self.profile_id,
            limit=20,
            session_id=session_id,
        )
        recent_failed_tools = [
            self._extract_terminal_tool_name(moment.tool_path)
            for moment in recent_moments
            if moment.outcome_class in {"failure", "error", "blocked"}
        ]
        has_recent_inspection = any(
            self._extract_terminal_tool_name(moment.tool_path) in _INSPECT_TOOLS
            and moment.outcome_class in {"success", "completed", "checkpoint"}
            for moment in recent_moments[:6]
        )

        retrieval = retrieve_biases(
            self.store,
            config=self.config,
            profile_id=self.profile_id,
            user_message=user_message,
            task_type=task_type,
            platform=platform or "cli",
            available_tools=available_tools,
            include_shadow=True,
        )
        state_dimensions = self._list_policy_state_dimensions(include_shadow=True)
        decision_priors, injected_ids = build_decision_priors(
            retrieval.active_biases,
            max_prompt_tokens=self.config.max_prompt_tokens,
            policy_state=state_dimensions,
        )
        ranked_tools, tool_deltas, planner_effects = rerank_tools(
            tool_defs,
            retrieval.active_biases,
            user_message=user_message,
            task_type=task_type,
            platform=platform or "cli",
            recent_failed_tools=recent_failed_tools,
            policy_state=state_dimensions,
        )
        evidence_items = [bias_summary(bias) for bias in retrieval.active_biases]
        evidence_items.extend(state_evidence_summary(state_dimensions))
        trace = DecisionTrace(
            id=new_id("trace"),
            profile_id=self.profile_id,
            session_id=session_id,
            turn_index=turn_index,
            task_type=task_type,
            platform=platform or "cli",
            user_message_excerpt=(user_message or "")[:240],
            retrieved_bias_ids=[bias.id for bias in retrieval.active_biases],
            injected_bias_ids=injected_ids,
            shadow_bias_ids=[bias.id for bias in retrieval.shadow_biases],
            planner_effects=planner_effects,
            tool_weight_deltas=[
                {"tool_name": delta.tool_name, "weight_delta": delta.weight_delta, "reasons": delta.reasons}
                for delta in tool_deltas
            ],
            risk_actions=[],
            response_effects=[],
            evidence_summary=evidence_items,
        )
        self.store.save_decision_trace(trace)
        for bias in retrieval.active_biases:
            try:
                self.store.touch_bias_trigger(bias.id)
            except Exception:
                logger.debug("Failed to update trigger stats for policy bias %s", bias.id)
        if self.config.log_bias_triggers and (
            retrieval.active_biases or retrieval.shadow_biases
        ):
            logger.info(
                "Policy bias retrieval session=%s active=%s shadow=%s",
                session_id,
                [bias.bias_candidate_key or bias.id for bias in retrieval.active_biases],
                [bias.bias_candidate_key or bias.id for bias in retrieval.shadow_biases],
            )

        return BiasDecisionContext(
            session_id=session_id,
            turn_index=turn_index,
            task_type=task_type,
            platform=platform or "cli",
            user_message=user_message,
            active_biases=retrieval.active_biases,
            shadow_biases=retrieval.shadow_biases,
            decision_priors=decision_priors,
            ranked_tools=ranked_tools,
            tool_weight_deltas=tool_deltas,
            trace_id=trace.id,
            metadata={
                "recent_failed_tools": [name for name in recent_failed_tools if name],
                "has_recent_inspection": has_recent_inspection,
                "planner_effects": planner_effects,
                "turn_tool_names": [],
                "risk_actions": [],
                "_policy_state_dimensions": state_dimensions,
                "policy_state_dimensions": serialize_dimensions(state_dimensions),
                "response_controls": derive_response_controls(
                    retrieval.active_biases,
                    task_type=task_type,
                    user_message=user_message,
                    policy_state=state_dimensions,
                ),
                "response_effects": [],
                "blocked_tool_names": [],
                "injected_bias_ids": list(injected_ids),
                "evidence_summary": evidence_items,
            },
        )

    def rerank_tool_calls(
        self,
        context: Optional[BiasDecisionContext],
        parsed_calls: list[tuple[object, str, dict]],
    ) -> tuple[list[tuple[object, str, dict]], list[dict]]:
        if context is None or not self.is_enabled():
            return parsed_calls, []
        ordered, planner_effects = rerank_tool_calls(
            parsed_calls,
            context.active_biases,
            recent_failed_tools=context.metadata.get("recent_failed_tools", []),
            policy_state=context.metadata.get("_policy_state_dimensions", []),
        )
        if planner_effects:
            context.metadata.setdefault("planner_effects", []).extend(planner_effects)
            self._save_trace_update(context)
        return ordered, planner_effects

    def evaluate_risk(
        self,
        context: Optional[BiasDecisionContext],
        *,
        tool_name: str,
        function_args: dict[str, object],
    ) -> Optional[RiskAction]:
        if context is None or not self.is_enabled():
            return None
        risk_action = evaluate_risk_gate(
            tool_name,
            function_args,
            context.active_biases,
            require_inspect_first=self.config.require_inspect_before_execute_for_external_actions,
            has_recent_inspection=bool(context.metadata.get("has_recent_inspection")),
            user_message=context.user_message,
            platform=context.platform,
            policy_state=context.metadata.get("_policy_state_dimensions", []),
        )
        if risk_action is not None:
            context.metadata.setdefault("risk_actions", []).append(
                {
                    "tool_name": risk_action.tool_name,
                    "decision": risk_action.decision,
                    "reason": risk_action.reason,
                    "suggested_tool": risk_action.suggested_tool,
                    "bias_ids": risk_action.bias_ids,
                }
            )
            context.metadata.setdefault("blocked_tool_names", []).append(tool_name)
            self._save_trace_update(context)
        return risk_action

    def record_tool_result(
        self,
        context: Optional[BiasDecisionContext],
        *,
        tool_name: str,
        function_args: dict[str, object],
        result: str,
        duration_ms: int,
    ) -> None:
        if context is None or not self.is_enabled():
            return

        result_status = self._tool_result_status(result)
        is_guard_block = result_status in {"approval_required", "blocked_by_policy_bias"}
        failure = False if is_guard_block else self._tool_result_failed(result)
        outcome_class = "blocked" if is_guard_block else "failure" if failure else "success"
        error_signal = 0.0 if is_guard_block else 1.0 if failure else 0.0
        side_effect_level = side_effect_level_for_tool(tool_name, function_args)
        reward = (
            0.0
            if is_guard_block
            else compute_reward_score(
                outcome_class=outcome_class,
                error_signal=error_signal,
                latency_ms=duration_ms,
                side_effect_level=side_effect_level,
            )
        )
        context.metadata.setdefault("turn_tool_names", []).append(tool_name)
        if tool_name in _INSPECT_TOOLS and not failure:
            context.metadata["has_recent_inspection"] = True

        candidate_key = None
        repeated_failures = context.metadata.setdefault("tool_failures", {})
        if failure:
            repeated_failures[tool_name] = repeated_failures.get(tool_name, 0) + 1
            if repeated_failures[tool_name] >= 2:
                candidate_key = "tool_use.change_strategy_after_retries"

        moment = PolicyMoment(
            id=new_id("moment"),
            profile_id=self.profile_id,
            session_id=context.session_id,
            timestamp=now_ts(),
            task_type=context.task_type,
            platform=context.platform,
            context_summary=(context.user_message or "")[:220],
            action_trace_summary=f"{tool_name} completed with {'failure' if failure else 'success'}",
            tool_path=tool_name,
            decision_class="tool_use",
            outcome_class=outcome_class,
            reward_score=reward,
            confidence_score=compute_moment_confidence(
                error_signal=error_signal,
                has_evidence=True,
                repeated_pattern=bool(candidate_key),
            ),
            error_signal=error_signal,
            side_effect_level=side_effect_level,
            latency_ms=duration_ms,
            evidence_refs=[f"session:{context.session_id}:turn:{context.turn_index}:tool:{tool_name}"],
            extracted_tags=[context.task_type, tool_name, side_effect_level, outcome_class],
            bias_candidate_key=candidate_key,
        )
        self._record_moment(moment)
        if candidate_key:
            self._synthesize_candidate(candidate_key)

    def record_checkpoint(
        self,
        *,
        session_id: str,
        turn_index: int,
        platform: str,
        task_type: str,
        label: str,
    ) -> None:
        if not self.is_enabled():
            return
        moment = PolicyMoment(
            id=new_id("moment"),
            profile_id=self.profile_id,
            session_id=session_id,
            timestamp=now_ts(),
            task_type=task_type,
            platform=platform or "cli",
            context_summary=label[:220],
            action_trace_summary=label,
            tool_path="checkpoint",
            decision_class="checkpoint",
            outcome_class="checkpoint",
            reward_score=0.20,
            confidence_score=0.60,
            evidence_refs=[f"session:{session_id}:turn:{turn_index}:checkpoint"],
            extracted_tags=[task_type, "checkpoint"],
        )
        self._record_moment(moment)

    def record_turn_outcome(
        self,
        context: Optional[BiasDecisionContext],
        *,
        final_response: str,
        completed: bool,
        interrupted: bool,
    ) -> None:
        if context is None or not self.is_enabled():
            return

        tool_names = list(context.metadata.get("turn_tool_names", []))
        outcome_class = "completed" if completed and not interrupted else "partial" if interrupted else "failure"
        reward = compute_reward_score(
            outcome_class=outcome_class,
            side_effect_level="high" if any(name in {"send_message", "cronjob", "ha_call_service"} for name in tool_names) else "none",
            retry_count=sum(context.metadata.get("tool_failures", {}).values()),
        )
        base_moment = PolicyMoment(
            id=new_id("moment"),
            profile_id=self.profile_id,
            session_id=context.session_id,
            timestamp=now_ts(),
            task_type=context.task_type,
            platform=context.platform,
            context_summary=(context.user_message or "")[:220],
            action_trace_summary=" > ".join(tool_names) if tool_names else "no_tools",
            tool_path=" > ".join(tool_names) if tool_names else "no_tools",
            decision_class="turn_outcome",
            outcome_class=outcome_class,
            reward_score=reward,
            confidence_score=compute_moment_confidence(
                has_evidence=True,
                repeated_pattern=bool(context.metadata.get("tool_failures")),
            ),
            evidence_refs=[f"session:{context.session_id}:turn:{context.turn_index}"],
            extracted_tags=[context.task_type, outcome_class] + tool_names[:5],
        )
        self._record_moment(base_moment)

        for candidate_key in self._derive_turn_candidates(context, completed=completed):
            candidate_moment = PolicyMoment(
                id=new_id("moment"),
                profile_id=self.profile_id,
                session_id=context.session_id,
                timestamp=now_ts(),
                task_type=context.task_type,
                platform=context.platform,
                context_summary=(context.user_message or "")[:220],
                action_trace_summary=(final_response or "")[:220],
                tool_path=" > ".join(tool_names) if tool_names else "no_tools",
                decision_class="policy_candidate",
                outcome_class=outcome_class,
                reward_score=reward,
                confidence_score=compute_moment_confidence(
                    has_evidence=True,
                    repeated_pattern=True,
                ),
                side_effect_level="high" if candidate_key == "risk.inspect_before_execute" else "none",
                evidence_refs=[f"session:{context.session_id}:turn:{context.turn_index}"],
                extracted_tags=[context.task_type, context.platform, candidate_key],
                bias_candidate_key=candidate_key,
            )
            self._record_moment(candidate_moment)
            self._synthesize_candidate(candidate_key)

    def recent_explanations(
        self,
        *,
        session_id: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, object]]:
        if not self.is_enabled():
            return []
        return explain_recent(
            self.store,
            profile_id=self.profile_id,
            session_id=session_id,
            limit=limit,
        )

    def _capture_feedback_preferences(
        self,
        *,
        session_id: str,
        user_message: str,
        platform: str,
        turn_index: int,
        available_tools: Iterable[str],
    ) -> None:
        feedback_signal = detect_feedback_signal(user_message)
        markers: list[str] = []
        candidate_keys: list[str] = []
        lowered = (user_message or "").lower()
        is_correction = is_correction_message(user_message)
        fingerprint = hashlib.sha1(
            f"{session_id}:{turn_index}:{user_message or ''}".encode("utf-8")
        ).hexdigest()
        if fingerprint in self._seen_feedback_fingerprints:
            return
        concise_request = any(
            tok in lowered
            for tok in (
                "be concise",
                "keep it concise",
                "keep it brief",
                "answer briefly",
                "concise and direct",
                "be direct",
                "简短一点",
                "直接一点",
                "别废话",
                "no fluff",
            )
        )
        one_step_request = any(
            tok in lowered
            for tok in (
                "one step at a time",
                "next step only",
                "step by step",
                "single next step",
                "一步一步",
                "下一步就行",
            )
        )
        findings_first_request = any(
            tok in lowered
            for tok in (
                "findings first",
                "risks first",
                "structured findings",
                "review findings first",
                "debug findings first",
                "先给结论",
                "先说问题",
            )
        )
        if concise_request:
            candidate_keys.extend(["communication.concise_first", "user_specific.directness_over_fluff"])
            markers.append("concise-first")
        if one_step_request:
            candidate_keys.append("user_specific.one_step_at_a_time")
            markers.append("one-step")
        if findings_first_request:
            candidate_keys.append("workflow_specific.structured_debugging_output")
            markers.append("findings-first")
        if is_correction:
            markers.append("correction")

        if not candidate_keys and feedback_signal == 0.0 and not markers:
            return
        self._seen_feedback_fingerprints.add(fingerprint)
        if len(self._seen_feedback_fingerprints) > 2048:
            self._seen_feedback_fingerprints.clear()
            self._seen_feedback_fingerprints.add(fingerprint)

        feedback_outcome = "partial" if feedback_signal < 0 or is_correction else "success"
        feedback_reward = -0.25 if feedback_signal < 0 or is_correction else 0.20
        base_feedback_moment = PolicyMoment(
            id=new_id("moment"),
            profile_id=self.profile_id,
            session_id=session_id,
            timestamp=now_ts(),
            task_type=classify_task_type(user_message, available_tools),
            platform=platform,
            context_summary=(user_message or "")[:220],
            action_trace_summary="user feedback / correction signal",
            tool_path="user_feedback",
            decision_class="user_feedback",
            outcome_class=feedback_outcome,
            reward_score=feedback_reward,
            confidence_score=compute_moment_confidence(
                user_feedback_signal=feedback_signal,
                has_evidence=True,
                repeated_pattern=is_correction,
            ),
            user_feedback_signal=feedback_signal,
            evidence_refs=[f"session:{session_id}:turn:{turn_index}:user_feedback"],
            extracted_tags=markers + list(dict.fromkeys(candidate_keys)),
        )
        self._record_moment(base_feedback_moment)

        for candidate_key in dict.fromkeys(candidate_keys):
            moment = PolicyMoment(
                id=new_id("moment"),
                profile_id=self.profile_id,
                session_id=session_id,
                timestamp=now_ts(),
                task_type=classify_task_type(user_message, available_tools),
                platform=platform,
                context_summary=(user_message or "")[:220],
                action_trace_summary="user preference / correction signal",
                tool_path="user_feedback",
                decision_class="user_feedback",
                outcome_class=feedback_outcome,
                reward_score=-0.35 if feedback_signal < 0 or is_correction else 0.65,
                confidence_score=compute_moment_confidence(
                    user_feedback_signal=feedback_signal,
                    has_evidence=True,
                    repeated_pattern=is_correction,
                ),
                user_feedback_signal=feedback_signal,
                evidence_refs=[f"session:{session_id}:turn:{turn_index}:user_feedback"],
                extracted_tags=markers + [candidate_key],
                bias_candidate_key=candidate_key,
            )
            self._record_moment(moment)
            self._synthesize_candidate(candidate_key)

    def _derive_turn_candidates(
        self,
        context: BiasDecisionContext,
        *,
        completed: bool,
    ) -> list[str]:
        tool_names = list(context.metadata.get("turn_tool_names", []))
        recent_failures = context.metadata.get("tool_failures", {})
        candidates: list[str] = []

        if context.task_type == "repo_modification":
            try:
                first_inspect = min(tool_names.index(name) for name in tool_names if name in {"read_file", "search_files"})
            except ValueError:
                first_inspect = None
            try:
                first_edit = min(tool_names.index(name) for name in tool_names if name in {"patch", "write_file"})
            except ValueError:
                first_edit = None

            if completed and first_inspect is not None and first_edit is not None and first_inspect < first_edit:
                candidates.append("planning.inspect_before_edit")
            if completed and "patch" in tool_names:
                candidates.append("tool_use.patch_before_rewrite")
            if completed and any(name in _LOCAL_TOOLS for name in tool_names) and not any(name in _WEB_TOOLS for name in tool_names):
                candidates.append("tool_use.local_before_external_code")

        if context.task_type == "current_info" and completed and any(name in _WEB_TOOLS for name in tool_names):
            candidates.append("planning.search_before_fresh_answer")

        if any(name in {"send_message", "cronjob", "ha_call_service", "terminal", "browser_click", "browser_type", "browser_press"} for name in tool_names):
            if completed and context.metadata.get("has_recent_inspection"):
                candidates.append("risk.inspect_before_execute")

        if completed and any(count >= 2 for count in recent_failures.values()):
            candidates.append("tool_use.change_strategy_after_retries")
        elif not completed and any(count >= 2 for count in recent_failures.values()):
            candidates.append("tool_use.change_strategy_after_retries")

        planning_tools = {"todo", "clarify"}
        non_planning = [name for name in tool_names if name not in planning_tools]
        if completed and any(name in planning_tools for name in tool_names) and len(non_planning) >= 2:
            candidates.append("workflow_specific.decompose_before_act")
        if self._looks_shared_platform(context.platform) and any(
            name in {"send_message", "cronjob", "ha_call_service", "browser_click", "browser_type", "browser_press"}
            for name in tool_names
        ):
            candidates.append("platform_specific.group_chat_caution")

        return list(dict.fromkeys(candidates))

    def _record_moment(self, moment: PolicyMoment) -> None:
        if not self.is_enabled():
            return
        self.store.add_moment(moment)
        self._apply_policy_state_updates(moment)
        if self.config.log_bias_triggers:
            logger.info(
                "Policy bias moment created id=%s class=%s outcome=%s candidate=%s",
                moment.id,
                moment.decision_class,
                moment.outcome_class,
                moment.bias_candidate_key,
            )

    def _synthesize_candidate(self, candidate_key: str) -> None:
        if not self.is_enabled():
            return
        moments = self.store.get_moments_by_candidate(self.profile_id, candidate_key)
        existing = self.store.find_bias_by_candidate_key(self.profile_id, candidate_key)
        bias = synthesize_bias(
            config=self.config,
            profile_id=self.profile_id,
            candidate_key=candidate_key,
            moments=moments,
            existing=existing,
        )
        if bias is not None:
            self.store.upsert_bias(bias)
            if self.config.log_bias_triggers:
                logger.info(
                    "Policy bias synthesized key=%s status=%s confidence=%.3f support=%s avg_reward=%.3f",
                    candidate_key,
                    bias.status,
                    bias.confidence,
                    bias.support_count,
                    bias.avg_reward,
                )

    def _save_trace_update(self, context: BiasDecisionContext) -> None:
        if not self.is_enabled() or not context.trace_id:
            return
        trace = DecisionTrace(
            id=context.trace_id,
            profile_id=self.profile_id,
            session_id=context.session_id,
            turn_index=context.turn_index,
            task_type=context.task_type,
            platform=context.platform,
            user_message_excerpt=(context.user_message or "")[:240],
            retrieved_bias_ids=[bias.id for bias in context.active_biases],
            injected_bias_ids=list(context.metadata.get("injected_bias_ids", [])),
            shadow_bias_ids=[bias.id for bias in context.shadow_biases],
            planner_effects=context.metadata.get("planner_effects", []),
            tool_weight_deltas=[
                {"tool_name": delta.tool_name, "weight_delta": delta.weight_delta, "reasons": delta.reasons}
                for delta in context.tool_weight_deltas
            ],
            risk_actions=context.metadata.get("risk_actions", []),
            response_effects=context.metadata.get("response_effects", []),
            evidence_summary=context.metadata.get(
                "evidence_summary",
                [bias_summary(bias) for bias in context.active_biases],
            ),
        )
        self.store.save_decision_trace(trace)

    def record_response_effects(
        self,
        context: Optional[BiasDecisionContext],
        response_effects: list[dict[str, object]],
    ) -> None:
        if context is None or not self.is_enabled() or not response_effects:
            return
        context.metadata.setdefault("response_effects", []).extend(response_effects)
        self._save_trace_update(context)

    def _list_policy_state_dimensions(self, *, include_shadow: bool) -> list[object]:
        if not self.is_enabled():
            return []
        store = self.store
        if store is None:
            return []
        method_names = (
            "list_policy_state_dimensions",
            "list_policy_state",
            "list_state_dimensions",
        )
        statuses = ["active", "shadow"] if include_shadow else ["active"]
        for method_name in method_names:
            method = getattr(store, method_name, None)
            if not callable(method):
                continue
            try:
                return list(method(self.profile_id, statuses=statuses))
            except TypeError:
                try:
                    return list(method(profile_id=self.profile_id, statuses=statuses))
                except TypeError:
                    try:
                        return list(method(self.profile_id))
                    except Exception as exc:
                        logger.debug("Policy state listing via %s failed: %s", method_name, exc)
                        return []
            except Exception as exc:
                logger.debug("Policy state listing via %s failed: %s", method_name, exc)
                return []
        return []

    def _apply_policy_state_updates(self, moment: PolicyMoment) -> None:
        if not self.is_enabled():
            return
        store = self.store
        if store is None:
            return
        method_names = (
            "apply_policy_state_from_moment",
            "apply_moment_state_updates",
            "record_policy_state_from_moment",
            "update_policy_state_from_moment",
        )
        for method_name in method_names:
            method = getattr(store, method_name, None)
            if not callable(method):
                continue
            try:
                updates = method(moment)
            except TypeError:
                try:
                    updates = method(moment=moment)
                except Exception as exc:
                    logger.debug("Policy state update via %s failed: %s", method_name, exc)
                    return
            except Exception as exc:
                logger.debug("Policy state update via %s failed: %s", method_name, exc)
                return
            if self.config.log_bias_triggers and updates:
                logger.info(
                    "Policy state updated moment=%s updates=%s",
                    moment.id,
                    [
                        getattr(update, "dimension_key", None)
                        or (update.get("dimension_key") if isinstance(update, dict) else None)
                        for update in updates
                    ],
                )
            return

    @staticmethod
    def _tool_result_failed(result: str) -> bool:
        if not result:
            return False
        lowered = result[:500].lower()
        if lowered.startswith("error") or '"error"' in lowered or '"failed"' in lowered:
            return True
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                if parsed.get("success") is False:
                    return True
                exit_code = parsed.get("exit_code")
                if exit_code is not None and int(exit_code) != 0:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _tool_result_status(result: str) -> str | None:
        if not result:
            return None
        try:
            parsed = json.loads(result)
        except Exception:
            return None
        if isinstance(parsed, dict):
            status = parsed.get("status")
            if isinstance(status, str):
                return status
        return None

    @staticmethod
    def _extract_terminal_tool_name(tool_path: str) -> str:
        if not tool_path:
            return ""
        if ">" in tool_path:
            return tool_path.split(">")[-1].strip()
        return tool_path.strip()

    @staticmethod
    def _looks_shared_platform(platform: str) -> bool:
        platform = (platform or "").lower()
        return any(token in platform for token in ("slack", "discord", "teams", "group", "channel"))
