"""Tests for the policy-bias CLI utilities."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from agent.policy_bias.models import PolicyBias
from agent.policy_bias.models import BIAS_SCOPES, PolicyBiasConfig, PolicyMoment
from agent.policy_bias.store import PolicyBiasStore
from agent.policy_bias.synthesis import get_candidate_descriptor
from hermes_cli import policy_bias_cmd


def _loader_factory(db_path, profile_id: str):
    def _loader():
        return (
            PolicyBiasStore(db_path=db_path),
            PolicyBiasConfig.from_dict(
                {
                    "enabled": True,
                    "scopes_enabled": list(BIAS_SCOPES),
                }
            ),
            profile_id,
        )

    return _loader


def _make_bias(profile_id: str, candidate_key: str) -> PolicyBias:
    descriptor = get_candidate_descriptor(candidate_key)
    assert descriptor is not None
    return PolicyBias(
        id=f"bias_{candidate_key.replace('.', '_')}",
        profile_id=profile_id,
        scope=descriptor.scope,
        condition_signature=descriptor.condition_signature,
        preferred_policy=descriptor.preferred_policy,
        anti_policy=descriptor.anti_policy,
        rationale_summary=descriptor.rationale_summary,
        confidence=0.82,
        support_count=4,
        avg_reward=0.62,
        recency_score=1.0,
        decay_rate=0.01,
        status="active",
        source_moment_ids=["moment_1"],
        created_at=1.0,
        updated_at=1.0,
        last_triggered_at=None,
        trigger_count=0,
        rollback_parent_id=None,
        version=1,
        bias_candidate_key=candidate_key,
    )


def _make_moment(profile_id: str, candidate_key: str, *, reward: float) -> PolicyMoment:
    return PolicyMoment(
        id=f"moment_{candidate_key.replace('.', '_')}_{abs(int(reward * 100))}_{int(time.time() * 1000)}",
        profile_id=profile_id,
        session_id="session-1",
        timestamp=time.time(),
        task_type="repo_modification",
        platform="cli",
        context_summary="fix repo file",
        action_trace_summary="read_file > patch",
        tool_path="read_file > patch",
        decision_class="policy_candidate",
        outcome_class="success" if reward >= 0 else "failure",
        reward_score=reward,
        confidence_score=0.8,
        evidence_refs=["session:session-1:turn:1"],
        extracted_tags=[candidate_key],
        bias_candidate_key=candidate_key,
    )


def test_policy_bias_cli_list_json(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "policy_bias.db"
    store = PolicyBiasStore(db_path=db_path)
    try:
        bias = _make_bias("profile:test", "planning.inspect_before_edit")
        store.upsert_bias(bias)
    finally:
        store.close()

    monkeypatch.setattr(
        policy_bias_cmd,
        "_load_store_and_config",
        _loader_factory(db_path, "profile:test"),
    )

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(policy_bias_action="list", limit=10, status=None, json=True)
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload[0]["candidate_key"] == "planning.inspect_before_edit"
    assert payload[0]["status"] == "active"


def test_policy_bias_cli_disable_updates_store(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "policy_bias.db"
    store = PolicyBiasStore(db_path=db_path)
    try:
        bias = _make_bias("profile:test", "risk.inspect_before_execute")
        store.upsert_bias(bias)
    finally:
        store.close()

    monkeypatch.setattr(
        policy_bias_cmd,
        "_load_store_and_config",
        _loader_factory(db_path, "profile:test"),
    )

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(policy_bias_action="disable", bias_id=bias.id)
    )
    output = capsys.readouterr().out

    reopened = PolicyBiasStore(db_path=db_path)
    try:
        updated = reopened.get_bias(bias.id)
        assert updated is not None
        assert updated.status == "disabled"
        assert "Disabled." in output
    finally:
        reopened.close()


def test_policy_bias_cli_export_and_rollback(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "policy_bias.db"
    store = PolicyBiasStore(db_path=db_path)
    try:
        bias = _make_bias("profile:test", "planning.inspect_before_edit")
        store.upsert_bias(bias)
        store.set_bias_status(bias.id, status="disabled", note="manual disable")
    finally:
        store.close()

    monkeypatch.setattr(
        policy_bias_cmd,
        "_load_store_and_config",
        _loader_factory(db_path, "profile:test"),
    )

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(policy_bias_action="export", min_confidence=0.5)
    )
    exported = json.loads(capsys.readouterr().out)
    assert exported == []

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(policy_bias_action="rollback", bias_id=bias.id, version=1)
    )
    output = capsys.readouterr().out

    reopened = PolicyBiasStore(db_path=db_path)
    try:
        restored = reopened.get_bias(bias.id)
        assert restored is not None
        assert restored.status == "active"
        assert "Rolled back." in output
    finally:
        reopened.close()


def test_policy_bias_cli_audit_and_inspect_include_boundary_metadata(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "policy_bias.db"
    store = PolicyBiasStore(db_path=db_path)
    try:
        bias = _make_bias("profile:test", "planning.inspect_before_edit")
        store.upsert_bias(bias)
    finally:
        store.close()

    monkeypatch.setattr(
        policy_bias_cmd,
        "_load_store_and_config",
        _loader_factory(db_path, "profile:test"),
    )

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(policy_bias_action="audit", limit=10, status=None, json=True)
    )
    audit_payload = json.loads(capsys.readouterr().out)
    assert audit_payload[0]["classification"] == "policy_bias"
    assert "tool_ranking" in audit_payload[0]["action_surfaces"]

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(policy_bias_action="inspect", bias_id=bias.id, json=True)
    )
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["boundary"]["classification"] == "policy_bias"
    assert "why_not_memory" in inspect_payload["boundary"]


def test_policy_bias_cli_enable_archive_history_and_rebuild(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "policy_bias.db"
    store = PolicyBiasStore(db_path=db_path)
    try:
        bias = _make_bias("profile:test", "planning.inspect_before_edit")
        store.upsert_bias(bias)
        for reward in (0.7, 0.75, 0.8):
            store.add_moment(_make_moment("profile:test", "planning.inspect_before_edit", reward=reward))
    finally:
        store.close()

    monkeypatch.setattr(
        policy_bias_cmd,
        "_load_store_and_config",
        _loader_factory(db_path, "profile:test"),
    )

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(policy_bias_action="archive", bias_id=bias.id)
    )
    assert "Archived." in capsys.readouterr().out

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(policy_bias_action="enable", bias_id=bias.id)
    )
    assert "Enabled." in capsys.readouterr().out

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(policy_bias_action="history", bias_id=bias.id, limit=10)
    )
    history_payload = json.loads(capsys.readouterr().out)
    assert history_payload
    assert history_payload[0]["snapshot"]["status"] in {"active", "archived"}

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(policy_bias_action="rescore", bias_id=bias.id)
    )
    rescore_payload = json.loads(capsys.readouterr().out)
    assert rescore_payload[0]["candidate_key"] == "planning.inspect_before_edit"

    policy_bias_cmd.run_policy_bias_command(
        SimpleNamespace(policy_bias_action="rebuild")
    )
    rebuild_payload = json.loads(capsys.readouterr().out)
    assert rebuild_payload
    assert rebuild_payload[0]["candidate_key"] == "planning.inspect_before_edit"
