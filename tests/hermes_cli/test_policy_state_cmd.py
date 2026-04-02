"""Tests for the V2 policy-state CLI surfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from agent.policy_bias.models import BIAS_SCOPES, PolicyBiasConfig
from hermes_cli import main as cli_main
from hermes_cli import policy_bias_cmd


@dataclass
class _Dimension:
    id: str
    profile_id: str
    dimension_key: str
    value: float
    confidence: float
    support_count: int
    avg_reward: float
    recency_score: float
    decay_rate: float
    status: str
    last_updated_at: float
    last_triggered_at: float | None
    version: int
    rollback_parent_id: str | None = None
    state_metadata: dict | None = None


@dataclass
class _Update:
    id: str
    profile_id: str
    dimension_key: str
    source_moment_id: str
    delta: float
    previous_value: float
    new_value: float
    update_reason: str
    confidence_delta: float
    created_at: float
    state_metadata: dict | None = None


class _FakePolicyStateStore:
    def __init__(self):
        self.closed = False
        self.calls: list[tuple] = []
        self.dimensions = {
            "inspect_tendency": _Dimension(
                id="dim_1",
                profile_id="profile:test",
                dimension_key="inspect_tendency",
                value=0.74,
                confidence=0.81,
                support_count=8,
                avg_reward=0.63,
                recency_score=0.92,
                decay_rate=0.01,
                status="active",
                last_updated_at=10.0,
                last_triggered_at=20.0,
                version=3,
                state_metadata={"source": "moment_update"},
            ),
            "risk_aversion": _Dimension(
                id="dim_2",
                profile_id="profile:test",
                dimension_key="risk_aversion",
                value=0.51,
                confidence=0.62,
                support_count=4,
                avg_reward=0.38,
                recency_score=0.56,
                decay_rate=0.02,
                status="shadow",
                last_updated_at=11.0,
                last_triggered_at=None,
                version=2,
                state_metadata={"source": "shadow"},
            ),
        }
        self.updates = [
            _Update(
                id="upd_1",
                profile_id="profile:test",
                dimension_key="inspect_tendency",
                source_moment_id="moment_1",
                delta=0.08,
                previous_value=0.66,
                new_value=0.74,
                update_reason="successful inspect-before-edit pattern",
                confidence_delta=0.04,
                created_at=21.0,
                state_metadata={"scope": "planning"},
            ),
            _Update(
                id="upd_2",
                profile_id="profile:test",
                dimension_key="risk_aversion",
                source_moment_id="moment_2",
                delta=0.05,
                previous_value=0.46,
                new_value=0.51,
                update_reason="external side-effect near-miss",
                confidence_delta=0.02,
                created_at=22.0,
                state_metadata={"scope": "risk"},
            ),
        ]

    def list_policy_state_dimensions(self, *, profile_id, statuses=None, limit=50):
        self.calls.append(("list", profile_id, statuses, limit))
        values = list(self.dimensions.values())
        if statuses:
            values = [dim for dim in values if dim.status in set(statuses)]
        return values[:limit]

    def get_policy_state_dimension(self, *, profile_id, dimension_key):
        self.calls.append(("inspect", profile_id, dimension_key))
        return self.dimensions.get(dimension_key)

    def list_policy_state_updates(self, *, profile_id, dimension_key=None, limit=20):
        self.calls.append(("updates", profile_id, dimension_key, limit))
        values = [upd for upd in self.updates if dimension_key in (None, upd.dimension_key)]
        return values[:limit]

    def rebuild_policy_state(self, *, profile_id):
        self.calls.append(("rebuild", profile_id))
        return list(self.dimensions.values())

    def reset_policy_state(self, *, profile_id, dimension_key=None, all_dimensions=False):
        self.calls.append(("reset", profile_id, dimension_key, all_dimensions))
        return {"dimension_key": dimension_key, "all_dimensions": all_dimensions}

    def explain_policy_state_decision(self, *, profile_id, trace_id=None, limit=10):
        self.calls.append(("explain", profile_id, trace_id, limit))
        return {
            "profile_id": profile_id,
            "trace_id": trace_id,
            "limit": limit,
            "active_dimensions": ["inspect_tendency"],
            "effects": [{"surface": "tool_selection", "delta": 0.25}],
        }

    def close(self):
        self.closed = True


def _loader_factory(store, profile_id: str):
    def _loader():
        return (
            store,
            PolicyBiasConfig.from_dict(
                {
                    "enabled": True,
                    "scopes_enabled": list(BIAS_SCOPES),
                }
            ),
            profile_id,
        )

    return _loader


def test_policy_state_cli_list_inspect_updates_json(tmp_path, monkeypatch, capsys):
    store = _FakePolicyStateStore()
    monkeypatch.setattr(
        policy_bias_cmd,
        "_load_store_and_config",
        _loader_factory(store, "profile:test"),
    )

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(
            policy_bias_action="state",
            state_action="list",
            status="active",
            limit=10,
            json=True,
        )
    )
    list_payload = json.loads(capsys.readouterr().out)
    assert list_payload["profile_id"] == "profile:test"
    assert list_payload["dimensions"][0]["dimension_key"] == "inspect_tendency"
    assert list_payload["dimensions"][0]["status"] == "active"

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(
            policy_bias_action="state",
            state_action="inspect",
            dimension_key="inspect_tendency",
            limit=5,
            json=True,
        )
    )
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["dimension"]["dimension_key"] == "inspect_tendency"
    assert inspect_payload["updates"][0]["update_reason"].startswith("successful inspect")

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(
            policy_bias_action="state",
            state_action="updates",
            dimension_key="risk_aversion",
            limit=5,
            json=True,
        )
    )
    updates_payload = json.loads(capsys.readouterr().out)
    assert updates_payload["updates"][0]["dimension_key"] == "risk_aversion"

    assert store.closed is True


def test_policy_state_cli_rebuild_reset_and_explain(tmp_path, monkeypatch, capsys):
    store = _FakePolicyStateStore()
    monkeypatch.setattr(
        policy_bias_cmd,
        "_load_store_and_config",
        _loader_factory(store, "profile:test"),
    )

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(
            policy_bias_action="state",
            state_action="rebuild",
            json=True,
        )
    )
    rebuild_payload = json.loads(capsys.readouterr().out)
    assert rebuild_payload["dimensions"][0]["dimension_key"] == "inspect_tendency"

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(
            policy_bias_action="state",
            state_action="reset",
            dimension_key="inspect_tendency",
            all=False,
            json=True,
        )
    )
    reset_payload = json.loads(capsys.readouterr().out)
    assert reset_payload["result"]["dimension_key"] == "inspect_tendency"
    assert reset_payload["result"]["all_dimensions"] is False

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(
            policy_bias_action="state",
            state_action="reset",
            all=True,
            json=True,
        )
    )
    reset_all_payload = json.loads(capsys.readouterr().out)
    assert reset_all_payload["result"]["dimension_key"] is None
    assert reset_all_payload["result"]["all_dimensions"] is True

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(
            policy_bias_action="state",
            state_action="explain",
            trace_id="trace_1",
            limit=3,
            json=True,
        )
    )
    explain_payload = json.loads(capsys.readouterr().out)
    assert explain_payload["trace_id"] == "trace_1"
    assert explain_payload["active_dimensions"] == ["inspect_tendency"]


def test_policy_state_cli_gracefully_handles_missing_backend(monkeypatch, capsys):
    class _NoStateStore:
        def close(self):
            pass

    monkeypatch.setattr(
        policy_bias_cmd,
        "_load_store_and_config",
        _loader_factory(_NoStateStore(), "profile:test"),
    )

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(
            policy_bias_action="state",
            state_action="list",
            limit=10,
            json=False,
        )
    )
    output = capsys.readouterr().out
    assert "Policy state API is not available" in output


def test_policy_bias_main_routes_policy_state_subcommand(monkeypatch):
    captured: dict[str, object] = {}

    def _runner(args):
        captured["policy_bias_action"] = args.policy_bias_action
        captured["state_action"] = args.state_action

    monkeypatch.setattr(policy_bias_cmd, "run_policy_bias_command", _runner)
    monkeypatch.setattr(
        cli_main.sys,
        "argv",
        ["hermes", "policy-bias", "state", "list", "--json"],
    )

    cli_main.main()

    assert captured["policy_bias_action"] == "state"
    assert captured["state_action"] == "list"
