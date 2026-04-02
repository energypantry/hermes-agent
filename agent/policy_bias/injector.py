"""Prompt injection helpers for policy biases."""

from __future__ import annotations

from agent.model_metadata import estimate_tokens_rough

from .models import PolicyBias


def build_decision_priors(
    biases: list[PolicyBias],
    *,
    max_prompt_tokens: int,
) -> tuple[str, list[str]]:
    if not biases:
        return "", []

    lines = ["Decision Priors"]
    injected_ids: list[str] = []
    for bias in biases:
        line = (
            f"- [scope={bias.scope} confidence={bias.confidence:.2f}] "
            f"{bias.preferred_policy}"
        )
        candidate = "\n".join(lines + [line])
        if estimate_tokens_rough(candidate) > max_prompt_tokens:
            break
        lines.append(line)
        injected_ids.append(bias.id)

    if len(lines) == 1:
        return "", []
    return "\n".join(lines), injected_ids
