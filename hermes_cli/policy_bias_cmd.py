"""CLI utilities for the Policy Bias Engine."""

from __future__ import annotations

import json
from typing import Any

from agent.policy_bias.governance import (
    archive_bias,
    bias_history,
    explain_recent,
    export_stable_biases,
    inspect_bias,
    list_biases,
    recent_moments,
    rebuild_biases,
    rollback_bias,
    rescore_biases,
    set_bias_enabled,
)
from agent.policy_bias.models import PolicyBiasConfig, derive_profile_id
from agent.policy_bias.store import PolicyBiasStore
from hermes_cli.config import load_config


def _load_store_and_config() -> tuple[PolicyBiasStore, PolicyBiasConfig, str]:
    cfg = load_config()
    policy_cfg = PolicyBiasConfig.from_dict(cfg.get("policy_bias"))
    profile_id = derive_profile_id()
    return PolicyBiasStore(), policy_cfg, profile_id


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def run_policy_bias_command(args) -> None:
    store, policy_cfg, profile_id = _load_store_and_config()
    try:
        action = getattr(args, "policy_bias_action", None) or "list"

        if action == "list":
            biases = list_biases(
                store,
                profile_id=profile_id,
                limit=getattr(args, "limit", 50),
                status=getattr(args, "status", None),
            )
            if getattr(args, "json", False):
                _print_json(
                    [
                        {
                            "id": bias.id,
                            "status": bias.status,
                            "scope": bias.scope,
                            "confidence": bias.confidence,
                            "support_count": bias.support_count,
                            "avg_reward": bias.avg_reward,
                            "candidate_key": bias.bias_candidate_key,
                            "preferred_policy": bias.preferred_policy,
                        }
                        for bias in biases
                    ]
                )
                return
            if not biases:
                print("No policy biases found.")
                return
            for bias in biases:
                print(
                    f"{bias.id}  [{bias.status}] [{bias.scope}] "
                    f"conf={bias.confidence:.2f} support={bias.support_count} "
                    f"reward={bias.avg_reward:.2f}"
                )
                print(f"  key: {bias.bias_candidate_key or '-'}")
                print(f"  {bias.preferred_policy}")
            return

        if action == "inspect":
            payload = inspect_bias(store, bias_id=args.bias_id)
            if payload is None:
                print(f"Bias not found: {args.bias_id}")
                return
            bias = payload["bias"]
            moments = payload["moments"]
            history = payload["history"]
            if getattr(args, "json", False):
                _print_json(
                    {
                        "bias": {
                            "id": bias.id,
                            "status": bias.status,
                            "scope": bias.scope,
                            "confidence": bias.confidence,
                            "support_count": bias.support_count,
                            "avg_reward": bias.avg_reward,
                            "preferred_policy": bias.preferred_policy,
                            "anti_policy": bias.anti_policy,
                            "rationale_summary": bias.rationale_summary,
                            "condition_signature": bias.condition_signature,
                            "source_moment_ids": bias.source_moment_ids,
                        },
                        "moments": [
                            {
                                "id": moment.id,
                                "timestamp": moment.timestamp,
                                "outcome_class": moment.outcome_class,
                                "reward_score": moment.reward_score,
                                "tool_path": moment.tool_path,
                                "context_summary": moment.context_summary,
                            }
                            for moment in moments
                        ],
                        "history": [
                            {
                                "id": entry.id,
                                "version": entry.version,
                                "operation": entry.operation,
                                "snapshot": entry.snapshot,
                                "created_at": entry.created_at,
                            }
                            for entry in history
                        ],
                    }
                )
                return
            print(f"{bias.id}  [{bias.status}] [{bias.scope}]")
            print(f"confidence: {bias.confidence:.2f}")
            print(f"support_count: {bias.support_count}")
            print(f"avg_reward: {bias.avg_reward:.2f}")
            print(f"candidate_key: {bias.bias_candidate_key or '-'}")
            print(f"condition_signature: {bias.condition_signature}")
            print(f"preferred_policy: {bias.preferred_policy}")
            if bias.anti_policy:
                print(f"anti_policy: {bias.anti_policy}")
            print(f"rationale: {bias.rationale_summary}")
            print("evidence moments:")
            for moment in moments:
                print(
                    f"  - {moment.id} outcome={moment.outcome_class} "
                    f"reward={moment.reward_score:.2f} tool_path={moment.tool_path}"
                )
            if history:
                print("history:")
                for entry in history[:5]:
                    print(
                        f"  - v{entry.version} {entry.operation} "
                        f"at {entry.created_at:.3f}"
                    )
            return

        if action == "enable":
            changed = set_bias_enabled(store, bias_id=args.bias_id, enabled=True)
            print("Enabled." if changed else "Bias not found.")
            return

        if action == "disable":
            changed = set_bias_enabled(store, bias_id=args.bias_id, enabled=False)
            print("Disabled." if changed else "Bias not found.")
            return

        if action == "archive":
            changed = archive_bias(store, bias_id=args.bias_id)
            print("Archived." if changed else "Bias not found.")
            return

        if action == "rescore":
            updated = rescore_biases(
                store,
                config=policy_cfg,
                profile_id=profile_id,
                bias_id=getattr(args, "bias_id", None),
            )
            _print_json(
                [
                    {
                        "id": bias.id,
                        "status": bias.status,
                        "confidence": bias.confidence,
                        "support_count": bias.support_count,
                        "avg_reward": bias.avg_reward,
                        "candidate_key": bias.bias_candidate_key,
                    }
                    for bias in updated
                ]
            )
            return

        if action == "rebuild":
            rebuilt = rebuild_biases(store, config=policy_cfg, profile_id=profile_id)
            _print_json(
                [
                    {
                        "id": bias.id,
                        "status": bias.status,
                        "confidence": bias.confidence,
                        "support_count": bias.support_count,
                        "avg_reward": bias.avg_reward,
                        "candidate_key": bias.bias_candidate_key,
                    }
                    for bias in rebuilt
                ]
            )
            return

        if action == "moments":
            payload = recent_moments(
                store,
                profile_id=profile_id,
                limit=getattr(args, "limit", 20),
            )
            _print_json(payload)
            return

        if action == "explain":
            payload = explain_recent(
                store,
                profile_id=profile_id,
                session_id=getattr(args, "session_id", None),
                limit=getattr(args, "limit", 5),
            )
            _print_json(payload)
            return

        if action == "history":
            payload = bias_history(
                store,
                bias_id=args.bias_id,
                limit=getattr(args, "limit", 20),
            )
            _print_json(payload)
            return

        if action == "rollback":
            changed = rollback_bias(
                store,
                bias_id=args.bias_id,
                version=args.version,
            )
            print("Rolled back." if changed else "Bias/version not found.")
            return

        if action == "export":
            payload = export_stable_biases(
                store,
                config=policy_cfg,
                profile_id=profile_id,
                min_confidence=getattr(args, "min_confidence", 0.55),
            )
            _print_json(payload)
            return

        raise SystemExit(f"Unknown policy-bias action: {action}")
    finally:
        store.close()
