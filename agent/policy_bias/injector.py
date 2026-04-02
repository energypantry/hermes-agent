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
        prefix = f"- [scope={bias.scope} confidence={bias.confidence:.2f}] "
        policy_text = bias.preferred_policy
        line = prefix + policy_text
        candidate = "\n".join(lines + [line])
        if estimate_tokens_rough(candidate) > max_prompt_tokens:
            if injected_ids:
                break

            words = policy_text.split()
            truncated_line = None
            while len(words) > 3:
                words = words[:-1]
                trial = prefix + " ".join(words).rstrip(" .,;:") + "..."
                trial_candidate = "\n".join(lines + [trial])
                if estimate_tokens_rough(trial_candidate) <= max_prompt_tokens:
                    truncated_line = trial
                    break
            if truncated_line is None:
                break
            line = truncated_line
        lines.append(line)
        injected_ids.append(bias.id)

    if len(lines) == 1:
        return "", []
    return "\n".join(lines), injected_ids
