"""Prompt injection helpers for policy biases."""

from __future__ import annotations

from agent.model_metadata import estimate_tokens_rough

from .models import PolicyBias, PolicyStateDimension, PolicyStatePlan
from .state_runtime import prompt_hint_lines


def build_decision_priors(
    biases: list[PolicyBias],
    *,
    max_prompt_tokens: int,
    policy_state: list[PolicyStateDimension] | None = None,
    policy_state_plan: PolicyStatePlan | None = None,
) -> tuple[str, list[str]]:
    if not biases and not policy_state and policy_state_plan is None:
        return "", []

    lines = ["Decision Priors:"]
    injected_ids: list[str] = []
    for bias in list(biases or [])[:3]:
        policy_text = (bias.preferred_policy or "").strip()
        if not policy_text:
            continue
        line = f"- {policy_text}"
        candidate = "\n".join(lines + [line])
        if estimate_tokens_rough(candidate) > max_prompt_tokens:
            if injected_ids:
                break

            words = policy_text.split()
            truncated_line = None
            while len(words) > 3:
                words = words[:-1]
                trial = "- " + " ".join(words).rstrip(" .,;:") + "..."
                trial_candidate = "\n".join(lines + [trial])
                if estimate_tokens_rough(trial_candidate) <= max_prompt_tokens:
                    truncated_line = trial
                    break
            if truncated_line is None:
                break
            line = truncated_line
        lines.append(line)
        injected_ids.append(bias.id)

    state_lines = prompt_hint_lines(policy_state, plan=policy_state_plan)
    for line in state_lines[:2]:
        candidate = "\n".join(lines + [line])
        if estimate_tokens_rough(candidate) > max_prompt_tokens:
            break
        lines.append(line)

    if len(lines) == 1:
        return "", []
    return "\n".join(lines), injected_ids
