"""Runtime helpers for consuming Policy Bias V2 policy state."""

from __future__ import annotations

from typing import Iterable

from .models import PolicyStateDimension


_ACTIVE_STATUSES = {"active", "shadow"}
_MIN_ACTIVE_MAGNITUDE = 0.12


def active_dimensions(
    dimensions: Iterable[PolicyStateDimension] | None,
    *,
    include_shadow: bool = False,
) -> list[PolicyStateDimension]:
    allowed_statuses = {"active", "shadow"} if include_shadow else {"active"}
    active: list[PolicyStateDimension] = []
    for dimension in dimensions or []:
        if dimension.status not in allowed_statuses:
            continue
        if abs(float(dimension.value)) < _MIN_ACTIVE_MAGNITUDE:
            continue
        active.append(dimension)
    return active


def dimension_map(
    dimensions: Iterable[PolicyStateDimension] | None,
    *,
    include_shadow: bool = False,
) -> dict[str, PolicyStateDimension]:
    return {
        dimension.dimension_key: dimension
        for dimension in active_dimensions(dimensions, include_shadow=include_shadow)
    }


def dimension_value(
    dimensions: Iterable[PolicyStateDimension] | None,
    key: str,
    *,
    include_shadow: bool = False,
) -> float:
    dimension = dimension_map(dimensions, include_shadow=include_shadow).get(key)
    return float(dimension.value) if dimension is not None else 0.0


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
                "confidence": round(float(dimension.confidence), 3),
                "support_count": int(dimension.support_count),
                "status": dimension.status,
            }
        )
    return summary


def prompt_hint_lines(
    dimensions: Iterable[PolicyStateDimension] | None,
) -> list[str]:
    hints: list[tuple[str, str]] = []
    values = dimension_map(dimensions)

    def _hint(key: str, scope: str, text: str, *, threshold: float = 0.35) -> None:
        dimension = values.get(key)
        if dimension is None or float(dimension.value) < threshold:
            return
        hints.append(
            (
                key,
                f"- [scope={scope} state={key} value={dimension.value:.2f}] {text}",
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
