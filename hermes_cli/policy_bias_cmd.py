"""CLI utilities for the Policy Bias Engine."""

from __future__ import annotations

import json
from typing import Any

from agent.policy_bias.governance import (
    archive_bias,
    audit_bias_boundaries,
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


def _safe_close_store(store: Any) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        close()


def _value_from(obj: Any, key: str, default: Any = None, *aliases: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj.get(key, default)
        for alias in aliases:
            if alias in obj:
                return obj.get(alias, default)
        return default
    if hasattr(obj, key):
        return getattr(obj, key, default)
    for alias in aliases:
        if hasattr(obj, alias):
            return getattr(obj, alias, default)
    return default


def _serialize_policy_state_dimension(dimension: Any) -> dict[str, Any]:
    return {
        "id": _value_from(dimension, "id"),
        "profile_id": _value_from(dimension, "profile_id"),
        "dimension_key": _value_from(dimension, "dimension_key"),
        "value": _value_from(dimension, "value"),
        "confidence": _value_from(dimension, "confidence"),
        "support_count": _value_from(dimension, "support_count"),
        "avg_reward": _value_from(dimension, "avg_reward"),
        "recency_score": _value_from(dimension, "recency_score"),
        "decay_rate": _value_from(dimension, "decay_rate"),
        "status": _value_from(dimension, "status"),
        "updated_at": _value_from(dimension, "updated_at", None, "last_updated_at"),
        "last_triggered_at": _value_from(dimension, "last_triggered_at"),
        "version": _value_from(dimension, "version"),
        "rollback_parent_id": _value_from(dimension, "rollback_parent_id"),
        "source_moment_ids": _value_from(dimension, "source_moment_ids", []),
        "status_note": _value_from(dimension, "status_note"),
        "state_metadata": _value_from(dimension, "state_metadata", {}),
    }


def _serialize_policy_state_update(update: Any) -> dict[str, Any]:
    moment_id = _value_from(update, "moment_id", None, "source_moment_id")
    value_before = _value_from(update, "value_before", None, "previous_value")
    value_after = _value_from(update, "value_after", None, "new_value")
    reason = _value_from(update, "reason", None, "update_reason")
    return {
        "id": _value_from(update, "id"),
        "profile_id": _value_from(update, "profile_id"),
        "dimension_key": _value_from(update, "dimension_key"),
        "moment_id": moment_id,
        "source_moment_id": moment_id,
        "delta": _value_from(update, "delta"),
        "value_before": value_before,
        "previous_value": value_before,
        "value_after": value_after,
        "new_value": value_after,
        "reason": reason,
        "update_reason": reason,
        "confidence_before": _value_from(update, "confidence_before"),
        "confidence_after": _value_from(update, "confidence_after"),
        "confidence_delta": _value_from(update, "confidence_delta"),
        "reward_score": _value_from(update, "reward_score"),
        "signal_type": _value_from(update, "signal_type"),
        "created_at": _value_from(update, "created_at"),
        "state_metadata": _value_from(update, "state_metadata", {}),
    }


def _state_api(store: Any, *names: str):
    for name in names:
        method = getattr(store, name, None)
        if callable(method):
            return method
    return None


def _state_api_unavailable(action: str) -> None:
    print(
        f"Policy state API is not available for '{action}'. "
        "The CLI surface is ready, but the backing store/runtime methods have not been wired in this build."
    )


def _run_policy_state_command(args, store: Any, profile_id: str) -> None:
    action = getattr(args, "state_action", None) or "list"

    list_dimensions = _state_api(
        store,
        "list_policy_state_dimensions",
        "list_policy_state",
        "get_policy_state_dimensions",
    )
    get_dimension = _state_api(
        store,
        "find_policy_state_dimension",
        "inspect_policy_state_dimension",
        "get_policy_state_dimension",
    )
    list_updates = _state_api(
        store,
        "list_policy_state_updates",
        "list_recent_policy_state_updates",
        "recent_policy_state_updates",
    )
    rebuild_state = _state_api(
        store,
        "rebuild_policy_state",
        "rebuild_policy_state_from_moments",
        "recompute_policy_state",
    )
    reset_state = _state_api(
        store,
        "reset_policy_state",
        "reset_policy_state_dimension",
    )
    explain_state = _state_api(
        store,
        "explain_policy_state_decision",
        "explain_policy_state_trace",
        "explain_policy_state_change",
    )

    if action == "list":
        if list_dimensions is None:
            _state_api_unavailable("list")
            return
        dimensions = list_dimensions(
            profile_id=profile_id,
            statuses=[getattr(args, "status")] if getattr(args, "status", None) else None,
            limit=getattr(args, "limit", 50),
        ) or []
        if getattr(args, "json", False):
            _print_json(
                {
                    "profile_id": profile_id,
                    "dimensions": [_serialize_policy_state_dimension(dim) for dim in dimensions],
                }
            )
            return
        if not dimensions:
            print("No policy state dimensions found.")
            return
        for dimension in dimensions:
            payload = _serialize_policy_state_dimension(dimension)
            print(
                f"{payload['dimension_key']}  [{payload['status']}] "
                f"value={payload['value']:.3f} conf={payload['confidence']:.2f} "
                f"support={payload['support_count']} reward={payload['avg_reward']:.2f}"
            )
            print(
                f"  recency={payload['recency_score']:.2f} decay={payload['decay_rate']:.3f} "
                f"version={payload['version']} updated_at={payload['updated_at']}"
            )
        return

    if action == "inspect":
        if get_dimension is None:
            _state_api_unavailable("inspect")
            return
        dimension = get_dimension(
            profile_id=profile_id,
            dimension_key=getattr(args, "dimension_key"),
        )
        if dimension is None:
            print(f"Policy state dimension not found: {getattr(args, 'dimension_key')}")
            return
        updates = []
        if list_updates is not None:
            updates = list_updates(
                profile_id=profile_id,
                dimension_key=getattr(args, "dimension_key"),
                limit=getattr(args, "limit", 10),
            ) or []
        if getattr(args, "json", False):
            _print_json(
                {
                    "dimension": _serialize_policy_state_dimension(dimension),
                    "updates": [_serialize_policy_state_update(update) for update in updates],
                }
            )
            return
        payload = _serialize_policy_state_dimension(dimension)
        print(f"{payload['dimension_key']}  [{payload['status']}]")
        print(f"  value: {payload['value']:.3f}")
        print(f"  confidence: {payload['confidence']:.2f}")
        print(f"  support_count: {payload['support_count']}")
        print(f"  avg_reward: {payload['avg_reward']:.2f}")
        print(f"  recency_score: {payload['recency_score']:.2f}")
        print(f"  decay_rate: {payload['decay_rate']:.3f}")
        print(f"  version: {payload['version']}")
        print(f"  updated_at: {payload['updated_at']}")
        print(f"  state_metadata: {payload['state_metadata'] or {}}")
        if updates:
            print("recent updates:")
            for update in updates[:10]:
                up = _serialize_policy_state_update(update)
                print(
                    f"  - {up['id']} delta={up['delta']:+.3f} "
                    f"prev={up['value_before']:.3f} new={up['value_after']:.3f} "
                    f"reason={up['reason']}"
                )
        return

    if action == "updates":
        if list_updates is None:
            _state_api_unavailable("updates")
            return
        updates = list_updates(
            profile_id=profile_id,
            dimension_key=getattr(args, "dimension_key", None),
            limit=getattr(args, "limit", 20),
        ) or []
        if getattr(args, "json", False):
            _print_json(
                {
                    "profile_id": profile_id,
                    "updates": [_serialize_policy_state_update(update) for update in updates],
                }
            )
            return
        if not updates:
            print("No policy state updates found.")
            return
        for update in updates:
            payload = _serialize_policy_state_update(update)
            print(
                f"{payload['id']}  [{payload['dimension_key']}] "
                f"delta={payload['delta']:+.3f} prev={payload['value_before']:.3f} "
                f"new={payload['value_after']:.3f}"
            )
            print(
                f"  reason: {payload['reason']} source_moment={payload['moment_id']} "
                f"at={payload['created_at']}"
            )
        return

    if action == "rebuild":
        if rebuild_state is None:
            _state_api_unavailable("rebuild")
            return
        rebuilt = rebuild_state(profile_id=profile_id) or []
        if hasattr(rebuilt, "dimensions"):
            rebuilt_dimensions = list(getattr(rebuilt, "dimensions", []))
        else:
            rebuilt_dimensions = list(rebuilt)
        if getattr(args, "json", False):
            _print_json(
                {
                    "profile_id": profile_id,
                    "dimensions": [_serialize_policy_state_dimension(dim) for dim in rebuilt_dimensions],
                }
            )
            return
        print(f"Rebuilt {len(rebuilt_dimensions)} policy state dimensions.")
        return

    if action == "reset":
        if reset_state is None:
            _state_api_unavailable("reset")
            return
        all_dimensions = bool(getattr(args, "all", False) or not getattr(args, "dimension_key", None))
        payload = reset_state(
            profile_id=profile_id,
            dimension_key=None if all_dimensions else getattr(args, "dimension_key", None),
            all_dimensions=all_dimensions,
        )
        if getattr(args, "json", False):
            _print_json({"profile_id": profile_id, "result": payload})
            return
        target = "all policy state" if all_dimensions else getattr(args, "dimension_key", "policy state")
        print(f"Reset {target}.")
        return

    if action == "explain":
        if explain_state is None:
            _state_api_unavailable("explain")
            return
        payload = explain_state(
            profile_id=profile_id,
            trace_id=getattr(args, "trace_id", None),
            limit=getattr(args, "limit", 10),
        )
        if getattr(args, "json", False):
            _print_json(payload if payload is not None else {})
            return
        if not payload:
            print("No policy-state explanation data found.")
            return
        _print_json(payload)
        return

    raise SystemExit(f"Unknown policy-state action: {action}")


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
            boundary = payload["boundary"]
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
                        "boundary": boundary,
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
            print(
                "boundary: "
                f"{boundary['classification']} surfaces={', '.join(boundary['action_surfaces']) or '-'}"
            )
            print(f"why_not_memory: {boundary['why_not_memory']}")
            print(f"why_not_skill: {boundary['why_not_skill']}")
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

        if action == "audit":
            payload = audit_bias_boundaries(
                store,
                profile_id=profile_id,
                limit=getattr(args, "limit", 50),
                status=getattr(args, "status", None),
            )
            if getattr(args, "json", False):
                _print_json(payload)
                return
            if not payload:
                print("No policy biases found.")
                return
            for entry in payload:
                surfaces = ", ".join(entry["action_surfaces"]) or "-"
                print(
                    f"{entry['id']}  [{entry['status']}] [{entry['scope']}] "
                    f"classification={entry['classification']} surfaces={surfaces}"
                )
                print(f"  key: {entry['candidate_key'] or '-'}")
                print(f"  why_not_memory: {entry['why_not_memory']}")
                print(f"  why_not_skill: {entry['why_not_skill']}")
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

        if action == "state":
            _run_policy_state_command(args, store, profile_id)
            return

        raise SystemExit(f"Unknown policy-bias action: {action}")
    finally:
        _safe_close_store(store)
