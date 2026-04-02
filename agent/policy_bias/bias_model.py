"""Minimal bias model used to seed and inspect stable decision priors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Bias:
    id: str
    description: str
    scope: str
    confidence: float


SEEDED_BIASES: tuple[Bias, ...] = (
    Bias(
        id="inspect-before-modify",
        description="Always inspect before modifying code",
        scope="planning",
        confidence=0.98,
    ),
    Bias(
        id="avoid-repeating-failed-tool-actions",
        description="Avoid repeating failed tool actions",
        scope="tool_use",
        confidence=0.96,
    ),
    Bias(
        id="search-before-answer",
        description="For ambiguous queries, search before answering",
        scope="planning",
        confidence=0.94,
    ),
)


BIAS_ID_TO_CANDIDATE_KEY = {
    "inspect-before-modify": "planning.inspect_before_edit",
    "avoid-repeating-failed-tool-actions": "tool_use.change_strategy_after_retries",
    "search-before-answer": "planning.search_before_fresh_answer",
}

CANDIDATE_KEY_TO_BIAS_ID = {
    candidate_key: bias_id for bias_id, candidate_key in BIAS_ID_TO_CANDIDATE_KEY.items()
}

SEEDED_BIAS_BY_ID = {bias.id: bias for bias in SEEDED_BIASES}
