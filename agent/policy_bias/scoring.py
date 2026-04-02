"""Scoring and classification helpers for policy-bias synthesis and retrieval."""

from __future__ import annotations

import math
import re
from typing import Iterable

from .models import PolicyBias

_TOKEN_RE = re.compile(r"[a-z0-9_./:-]+")

_POSITIVE_FEEDBACK_MARKERS = (
    "thanks",
    "thank you",
    "nice",
    "good",
    "great",
    "perfect",
    "works",
    "that helped",
    "exactly",
    "不错",
    "可以",
    "好",
    "对",
)

_NEGATIVE_FEEDBACK_MARKERS = (
    "wrong",
    "not what i asked",
    "doesn't work",
    "bad",
    "failed",
    "incorrect",
    "too verbose",
    "stop",
    "不对",
    "错",
    "不是这个",
)

_CORRECTION_MARKERS = (
    "actually",
    "i meant",
    "instead",
    "no,",
    "not that",
    "修正",
    "改成",
    "不是",
)

_SEMANTIC_EXPANSIONS = {
    "inspect": {"inspect", "read", "search", "review", "check", "snapshot"},
    "edit": {"edit", "patch", "write", "rewrite", "modify", "change"},
    "current": {"latest", "today", "current", "recent", "fresh", "news"},
    "risk": {"risk", "unsafe", "dangerous", "side-effect", "destructive"},
    "external": {"external", "send", "delete", "schedule", "service", "command"},
    "concise": {"concise", "short", "brief", "direct"},
    "local": {"local", "repo", "file", "code", "workspace"},
    "browser": {"browse", "browser", "web", "search", "extract"},
}


def normalize_tokens(text: str) -> set[str]:
    text = (text or "").lower()
    return {tok for tok in _TOKEN_RE.findall(text) if tok}


def expand_semantic_tokens(tokens: Iterable[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        for concept, aliases in _SEMANTIC_EXPANSIONS.items():
            if token == concept or token in aliases:
                expanded.add(concept)
                expanded.update(aliases)
    return expanded


def semantic_overlap_score(query_text: str, bias: PolicyBias) -> float:
    query = expand_semantic_tokens(normalize_tokens(query_text))
    if not query:
        return 0.0
    bias_text = " ".join(
        filter(
            None,
            [
                bias.scope,
                bias.condition_signature,
                bias.preferred_policy,
                bias.anti_policy or "",
                bias.rationale_summary,
                bias.bias_candidate_key or "",
            ],
        )
    )
    target = expand_semantic_tokens(normalize_tokens(bias_text))
    if not target:
        return 0.0
    overlap = len(query & target)
    union = len(query | target)
    return overlap / union if union else 0.0


def lexical_overlap_score(query_text: str, bias: PolicyBias) -> float:
    query = normalize_tokens(query_text)
    if not query:
        return 0.0
    target = normalize_tokens(
        " ".join(
            filter(
                None,
                [
                    bias.condition_signature,
                    bias.preferred_policy,
                    bias.anti_policy or "",
                    bias.rationale_summary,
                    bias.bias_candidate_key or "",
                ],
            )
        )
    )
    if not target:
        return 0.0
    overlap = len(query & target)
    return overlap / max(len(query), 1)


def classify_task_type(user_message: str, tool_names: Iterable[str] | None = None) -> str:
    text = (user_message or "").lower()
    tool_names = set(tool_names or [])
    if any(tok in text for tok in ("latest", "today", "recent", "current", "news")):
        return "current_info"
    if any(tok in text for tok in ("repo", "code", "file", "bug", "test", "branch", "commit")):
        return "repo_modification"
    if any(tok in text for tok in ("send", "delete", "remove", "schedule", "cron")):
        return "external_action"
    if {"read_file", "search_files", "patch", "write_file"} & tool_names:
        return "repo_modification"
    if {"send_message", "cronjob", "ha_call_service"} & tool_names:
        return "external_action"
    return "general"


def detect_feedback_signal(user_message: str) -> float:
    text = (user_message or "").lower()
    positive = any(marker in text for marker in _POSITIVE_FEEDBACK_MARKERS)
    negative = any(marker in text for marker in _NEGATIVE_FEEDBACK_MARKERS)
    if positive and not negative:
        return 1.0
    if negative and not positive:
        return -1.0
    return 0.0


def is_correction_message(user_message: str) -> bool:
    text = (user_message or "").lower()
    return any(marker in text for marker in _CORRECTION_MARKERS)


def compute_reward_score(
    *,
    outcome_class: str,
    user_feedback_signal: float = 0.0,
    error_signal: float = 0.0,
    latency_ms: int | None = None,
    cost_estimate: float | None = None,
    side_effect_level: str = "none",
    retry_count: int = 0,
) -> float:
    reward = 0.0

    if outcome_class in {"success", "completed", "checkpoint"}:
        reward += 0.45
    elif outcome_class in {"failure", "error", "blocked"}:
        reward -= 0.6
    elif outcome_class == "partial":
        reward -= 0.15

    reward += 0.30 * user_feedback_signal
    reward -= 0.35 * error_signal
    reward -= min(0.20, retry_count * 0.08)

    if latency_ms:
        reward -= min(0.12, latency_ms / 30000.0 * 0.10)
    if cost_estimate:
        reward -= min(0.08, cost_estimate * 0.01)

    if side_effect_level == "high" and outcome_class in {"success", "completed"}:
        reward += 0.10
    elif side_effect_level == "high" and outcome_class in {"failure", "error", "blocked"}:
        reward -= 0.10

    return max(-1.0, min(1.0, reward))


def compute_moment_confidence(
    *,
    user_feedback_signal: float = 0.0,
    error_signal: float = 0.0,
    has_evidence: bool = True,
    repeated_pattern: bool = False,
) -> float:
    confidence = 0.45
    if has_evidence:
        confidence += 0.15
    confidence += 0.15 * abs(user_feedback_signal)
    confidence += 0.10 * abs(error_signal)
    if repeated_pattern:
        confidence += 0.10
    return max(0.05, min(1.0, confidence))


def recency_score(timestamp: float, *, now: float, decay_per_day: float) -> float:
    if not timestamp:
        return 0.0
    days = max(0.0, (now - timestamp) / 86400.0)
    return math.exp(-max(0.0, decay_per_day) * days)


def apply_confidence_decay(
    confidence: float,
    *,
    updated_at: float,
    now: float,
    decay_per_day: float,
) -> float:
    return max(0.0, min(1.0, confidence * recency_score(updated_at, now=now, decay_per_day=decay_per_day)))


def compute_bias_confidence(
    *,
    support_count: int,
    avg_reward: float,
    recency: float,
) -> float:
    support_factor = min(1.0, support_count / 6.0)
    reward_factor = max(0.0, min(1.0, (avg_reward + 1.0) / 2.0))
    confidence = 0.20 + (0.35 * support_factor) + (0.30 * reward_factor) + (0.15 * recency)
    return max(0.0, min(1.0, confidence))


def side_effect_level_for_tool(tool_name: str, function_args: dict[str, object] | None) -> str:
    function_args = function_args or {}
    if tool_name in {"send_message", "cronjob", "ha_call_service"}:
        return "high"
    if tool_name in {"browser_click", "browser_type", "browser_press", "write_file", "patch"}:
        return "medium"
    if tool_name == "terminal":
        command = str(function_args.get("command", "") or "").lower()
        if any(token in command for token in ("rm ", "mv ", "git reset", "git clean", "truncate ", "dd ")):
            return "high"
        if command.strip():
            return "medium"
    return "none"
