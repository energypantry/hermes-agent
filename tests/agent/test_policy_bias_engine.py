"""Tests for the Policy Bias Engine."""

from __future__ import annotations

import sqlite3
import time

from agent.policy_bias.engine import PolicyBiasEngine
from agent.policy_bias.governance import audit_bias_boundaries, inspect_bias, rebuild_biases
from agent.policy_bias.injector import build_decision_priors
from agent.policy_bias.migrations import BASE_SCHEMA_SQL, BASE_SCHEMA_VERSION, SCHEMA_VERSION
from agent.policy_bias.models import BIAS_SCOPES, PolicyBias, PolicyBiasConfig, PolicyMoment, now_ts
from agent.policy_bias.planner_hooks import evaluate_risk_gate, rerank_tools
from agent.policy_bias.retrieval import retrieve_biases
from agent.policy_bias.store import PolicyBiasStore
from agent.policy_bias.synthesis import (
    descriptor_qualifies_for_policy_bias,
    get_boundary_metadata,
    get_candidate_descriptor,
    synthesize_bias,
)


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_engine(
    tmp_path,
    *,
    profile_id: str = "profile:test",
    retrieval_top_k: int = 4,
    db_name: str = "policy_bias.db",
    config: dict | None = None,
):
    cfg = {
        "enabled": True,
        "retrieval_top_k": retrieval_top_k,
        "max_prompt_tokens": 500,
        "shadow_mode_default": False,
        "scopes_enabled": list(BIAS_SCOPES),
        "synthesis": {
            "min_support_count": 3,
            "min_avg_reward": 0.25,
            "strong_signal_reward": 0.8,
            "confidence_decay_per_day": 0.01,
        },
        "risk_controls": {
            "require_inspect_before_execute_for_external_actions": True,
            "high_side_effect_shadow_only": False,
        },
        "observability": {
            "log_bias_triggers": True,
            "expose_explanations": True,
        },
    }
    if config:
        cfg.update(config)
    store = PolicyBiasStore(db_path=tmp_path / db_name)
    engine = PolicyBiasEngine(cfg, store=store)
    engine.profile_id = profile_id
    return engine, store


def _make_bias(
    profile_id: str,
    candidate_key: str,
    *,
    status: str = "active",
    confidence: float = 0.82,
    support_count: int = 4,
    avg_reward: float = 0.62,
    updated_at: float | None = None,
) -> PolicyBias:
    descriptor = get_candidate_descriptor(candidate_key)
    assert descriptor is not None
    ts = updated_at if updated_at is not None else now_ts()
    return PolicyBias(
        id=f"bias_{candidate_key.replace('.', '_')}_{int(ts * 1000)}",
        profile_id=profile_id,
        scope=descriptor.scope,
        condition_signature=descriptor.condition_signature,
        preferred_policy=descriptor.preferred_policy,
        anti_policy=descriptor.anti_policy,
        rationale_summary=descriptor.rationale_summary,
        confidence=confidence,
        support_count=support_count,
        avg_reward=avg_reward,
        recency_score=1.0,
        decay_rate=0.01,
        status=status,
        source_moment_ids=[f"moment_{candidate_key}_1"],
        created_at=ts,
        updated_at=ts,
        last_triggered_at=None,
        trigger_count=0,
        rollback_parent_id=None,
        version=1,
        bias_candidate_key=candidate_key,
    )


def _make_moment(
    profile_id: str,
    candidate_key: str,
    *,
    reward: float,
    timestamp: float,
) -> PolicyMoment:
    return PolicyMoment(
        id=f"moment_{candidate_key.replace('.', '_')}_{int(timestamp * 1000)}_{abs(int(reward * 100))}",
        profile_id=profile_id,
        session_id="session-1",
        timestamp=timestamp,
        task_type="repo_modification",
        platform="cli",
        context_summary="fix repo file",
        action_trace_summary="read_file > patch",
        tool_path="read_file > patch",
        decision_class="policy_candidate",
        outcome_class="success" if reward >= 0 else "failure",
        reward_score=reward,
        confidence_score=0.75,
        evidence_refs=[f"session:session-1:ts:{timestamp}"],
        extracted_tags=["repo_modification", candidate_key],
        bias_candidate_key=candidate_key,
    )


def _simulate_repo_success(engine: PolicyBiasEngine, *, session_id: str, turn_index: int) -> None:
    tool_defs = _tool_defs("patch", "read_file", "web_search")
    ctx = engine.begin_turn(
        session_id=session_id,
        turn_index=turn_index,
        user_message="Fix the repo bug in this file",
        platform="cli",
        available_tools=["patch", "read_file", "web_search"],
        tool_defs=tool_defs,
    )
    engine.record_tool_result(
        ctx,
        tool_name="read_file",
        function_args={"path": "app.py"},
        result='{"success": true}',
        duration_ms=10,
    )
    engine.record_tool_result(
        ctx,
        tool_name="patch",
        function_args={"path": "app.py"},
        result='{"success": true}',
        duration_ms=20,
    )
    engine.record_turn_outcome(
        ctx,
        final_response="done",
        completed=True,
        interrupted=False,
    )


def test_store_migrates_v1_schema_to_current(tmp_path):
    db_path = tmp_path / "policy_bias.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(BASE_SCHEMA_SQL)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (BASE_SCHEMA_VERSION,))
        conn.commit()
    finally:
        conn.close()

    store = PolicyBiasStore(db_path=db_path)
    try:
        version = store._conn.execute("SELECT version FROM schema_version").fetchone()["version"]
        columns = {
            row["name"]
            for row in store._conn.execute("PRAGMA table_info(biases)").fetchall()
        }
        tables = {
            row["name"]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert version == SCHEMA_VERSION
        assert "disabled_reason" in columns
        assert "decision_traces" in tables
    finally:
        store.close()


def test_delete_biases_respects_status_filter(tmp_path):
    store = PolicyBiasStore(db_path=tmp_path / "policy_bias.db")
    try:
        active = _make_bias("profile:test", "planning.inspect_before_edit", status="active")
        shadow = _make_bias("profile:test", "tool_use.patch_before_rewrite", status="shadow")
        store.upsert_bias(active)
        store.upsert_bias(shadow)

        deleted = store.delete_biases("profile:test", statuses=["active"])

        assert deleted == 1
        assert store.get_bias(active.id) is None
        assert store.get_bias(shadow.id) is not None
    finally:
        store.close()


def test_synthesize_bias_lifecycle_and_negative_suppression():
    config = PolicyBiasConfig.from_dict(
        {
            "enabled": True,
            "scopes_enabled": list(BIAS_SCOPES),
            "synthesis": {
                "min_support_count": 3,
                "min_avg_reward": 0.25,
                "strong_signal_reward": 0.8,
                "confidence_decay_per_day": 0.01,
            },
        }
    )
    base_ts = now_ts() - 30
    one_strong = [
        _make_moment(
            "profile:test",
            "planning.search_before_fresh_answer",
            reward=0.9,
            timestamp=base_ts,
        )
    ]
    shadow_bias = synthesize_bias(
        config=config,
        profile_id="profile:test",
        candidate_key="planning.search_before_fresh_answer",
        moments=one_strong,
        now=base_ts + 1,
    )
    active_bias = synthesize_bias(
        config=config,
        profile_id="profile:test",
        candidate_key="planning.search_before_fresh_answer",
        moments=[
            _make_moment("profile:test", "planning.search_before_fresh_answer", reward=0.7, timestamp=base_ts),
            _make_moment("profile:test", "planning.search_before_fresh_answer", reward=0.8, timestamp=base_ts + 1),
            _make_moment("profile:test", "planning.search_before_fresh_answer", reward=0.75, timestamp=base_ts + 2),
        ],
        now=base_ts + 3,
    )
    disabled_bias = synthesize_bias(
        config=config,
        profile_id="profile:test",
        candidate_key="planning.search_before_fresh_answer",
        moments=[
            _make_moment("profile:test", "planning.search_before_fresh_answer", reward=-0.7, timestamp=base_ts),
            _make_moment("profile:test", "planning.search_before_fresh_answer", reward=-0.8, timestamp=base_ts + 1),
            _make_moment("profile:test", "planning.search_before_fresh_answer", reward=-0.75, timestamp=base_ts + 2),
            _make_moment("profile:test", "planning.search_before_fresh_answer", reward=-0.85, timestamp=base_ts + 3),
        ],
        now=base_ts + 4,
    )

    assert shadow_bias is not None and shadow_bias.status == "shadow"
    assert active_bias is not None and active_bias.status == "active"
    assert disabled_bias is not None and disabled_bias.status == "disabled"
    assert disabled_bias.confidence <= 0.45


def test_candidate_descriptors_carry_boundary_metadata():
    descriptor = get_candidate_descriptor("planning.inspect_before_edit")

    assert descriptor is not None
    assert descriptor_qualifies_for_policy_bias(descriptor) is True
    assert "tool_ranking" in descriptor.action_surfaces
    assert "not a factual note" in descriptor.why_not_memory

    boundary = get_boundary_metadata("planning.inspect_before_edit")
    assert boundary["classification"] == "policy_bias"
    assert "tool_batch" in boundary["action_surfaces"]


def test_retrieval_is_top_k_and_profile_isolated(tmp_path):
    store = PolicyBiasStore(db_path=tmp_path / "policy_bias.db")
    try:
        profile_a = "profile:a"
        profile_b = "profile:b"
        now = time.time()
        store.upsert_bias(
            _make_bias(profile_a, "planning.inspect_before_edit", updated_at=now)
        )
        store.upsert_bias(
            _make_bias(profile_a, "tool_use.patch_before_rewrite", updated_at=now - 5)
        )
        store.upsert_bias(
            _make_bias(profile_a, "risk.inspect_before_execute", updated_at=now - 10)
        )
        store.upsert_bias(
            _make_bias(profile_b, "communication.concise_first", updated_at=now)
        )

        config = PolicyBiasConfig.from_dict(
            {"enabled": True, "retrieval_top_k": 2, "scopes_enabled": list(BIAS_SCOPES)}
        )
        result = retrieve_biases(
            store,
            config=config,
            profile_id=profile_a,
            user_message="Fix the repo file with a small patch",
            task_type="repo_modification",
            platform="cli",
            available_tools=["read_file", "patch"],
            include_shadow=True,
        )

        assert len(result.active_biases) == 2
        assert all(bias.profile_id == profile_a for bias in result.active_biases)
        assert {bias.bias_candidate_key for bias in result.active_biases} <= {
            "planning.inspect_before_edit",
            "tool_use.patch_before_rewrite",
            "risk.inspect_before_execute",
        }
    finally:
        store.close()


def test_retrieval_preserves_scope_coverage_before_duplicate_scope_fill(tmp_path):
    store = PolicyBiasStore(db_path=tmp_path / "policy_bias.db")
    try:
        profile_id = "profile:test"
        now = time.time()
        biases = [
            _make_bias(profile_id, "planning.inspect_before_edit", updated_at=now, confidence=0.92),
            _make_bias(profile_id, "tool_use.patch_before_rewrite", updated_at=now - 1, confidence=0.95),
            _make_bias(profile_id, "tool_use.local_before_external_code", updated_at=now - 2, confidence=0.94),
            _make_bias(profile_id, "risk.inspect_before_execute", updated_at=now - 3, confidence=0.78),
        ]
        for bias in biases:
            store.upsert_bias(bias)

        config = PolicyBiasConfig.from_dict(
            {"enabled": True, "retrieval_top_k": 3, "scopes_enabled": list(BIAS_SCOPES)}
        )
        result = retrieve_biases(
            store,
            config=config,
            profile_id=profile_id,
            user_message="Inspect the repo, patch the file, and avoid risky external actions",
            task_type="repo_modification",
            platform="cli",
            available_tools=["read_file", "patch", "send_message"],
            include_shadow=False,
        )

        selected_keys = {bias.bias_candidate_key for bias in result.active_biases}
        selected_scopes = {bias.scope for bias in result.active_biases}

        assert len(result.active_biases) == 3
        assert selected_scopes == {"planning", "tool_use", "risk"}
        assert "risk.inspect_before_execute" in selected_keys
        assert sum(1 for bias in result.active_biases if bias.scope == "tool_use") == 1
    finally:
        store.close()


def test_build_decision_priors_is_bounded():
    biases = [
        _make_bias("profile:test", "planning.inspect_before_edit"),
        _make_bias("profile:test", "tool_use.patch_before_rewrite"),
        _make_bias("profile:test", "risk.inspect_before_execute"),
    ]
    for bias in biases:
        bias.preferred_policy = bias.preferred_policy + " " + ("extra-detail " * 20)

    priors, injected_ids = build_decision_priors(biases, max_prompt_tokens=45)

    assert priors.startswith("Decision Priors")
    assert injected_ids
    assert len(injected_ids) < len(biases)


def test_planner_reranking_and_risk_gate_apply_active_biases():
    biases = [
        _make_bias("profile:test", "planning.inspect_before_edit"),
        _make_bias("profile:test", "tool_use.patch_before_rewrite"),
        _make_bias("profile:test", "risk.inspect_before_execute"),
        _make_bias("profile:test", "workflow_specific.decompose_before_act"),
    ]
    ranked_tools, deltas, _planner_effects = rerank_tools(
        _tool_defs("send_message", "patch", "read_file", "todo"),
        biases,
        user_message="Fix the repo file and then notify the user",
        task_type="repo_modification",
        platform="cli",
        recent_failed_tools=[],
    )

    assert [tool["function"]["name"] for tool in ranked_tools][:3] == ["read_file", "todo", "patch"]
    assert any(delta.tool_name == "send_message" and delta.weight_delta < 0 for delta in deltas)

    blocked = evaluate_risk_gate(
        "send_message",
        {"message": "done"},
        biases,
        require_inspect_first=True,
        has_recent_inspection=False,
        user_message="Send a message to the customer",
        platform="cli",
    )
    allowed = evaluate_risk_gate(
        "send_message",
        {"message": "done"},
        biases,
        require_inspect_first=True,
        has_recent_inspection=True,
        user_message="Please send it to the customer now.",
        platform="cli",
    )
    simulated = evaluate_risk_gate(
        "browser_click",
        {"selector": "#submit"},
        biases,
        require_inspect_first=True,
        has_recent_inspection=False,
        user_message="Click submit",
        platform="cli",
    )

    assert blocked is not None
    assert blocked.decision == "confirm"
    assert blocked.suggested_tool == "web_search"
    assert allowed is None
    assert simulated is not None
    assert simulated.decision == "simulate"


def test_repeated_success_promotes_active_bias_and_influences_next_turn(tmp_path):
    engine, store = _make_engine(tmp_path)
    try:
        for turn_index in range(1, 4):
            _simulate_repo_success(engine, session_id="session-1", turn_index=turn_index)

        bias = store.find_bias_by_candidate_key(
            engine.profile_id,
            "planning.inspect_before_edit",
        )
        assert bias is not None
        assert bias.status == "active"

        next_ctx = engine.begin_turn(
            session_id="session-1",
            turn_index=4,
            user_message="Fix the repo bug in this file",
            platform="cli",
            available_tools=["patch", "read_file", "web_search"],
            tool_defs=_tool_defs("patch", "read_file", "web_search"),
        )

        assert "Decision Priors" in next_ctx.decision_priors
        assert "inspect file structure and conventions before editing" in next_ctx.decision_priors
        assert next_ctx.ranked_tools[0]["function"]["name"] == "read_file"
    finally:
        engine.close()


def test_negative_outcomes_disable_existing_bias(tmp_path):
    engine, store = _make_engine(tmp_path)
    try:
        base_ts = now_ts() - 10
        for offset, reward in enumerate((0.7, 0.75, 0.8)):
            store.add_moment(
                _make_moment(
                    engine.profile_id,
                    "planning.inspect_before_edit",
                    reward=reward,
                    timestamp=base_ts + offset,
                )
            )
        rebuild_biases(store, config=engine.config, profile_id=engine.profile_id)
        active = store.find_bias_by_candidate_key(engine.profile_id, "planning.inspect_before_edit")
        assert active is not None and active.status == "active"

        for offset, reward in enumerate((-0.9, -0.8, -0.85, -0.75), start=4):
            store.add_moment(
                _make_moment(
                    engine.profile_id,
                    "planning.inspect_before_edit",
                    reward=reward,
                    timestamp=base_ts + offset,
                )
            )
        rebuild_biases(store, config=engine.config, profile_id=engine.profile_id)

        suppressed = store.find_bias_by_candidate_key(
            engine.profile_id,
            "planning.inspect_before_edit",
        )
        assert suppressed is not None
        assert suppressed.status == "disabled"
        assert suppressed.avg_reward < 0
    finally:
        engine.close()


def test_disabling_bias_stops_influencing_behavior(tmp_path):
    engine, store = _make_engine(tmp_path)
    try:
        bias = _make_bias(engine.profile_id, "risk.inspect_before_execute", status="active")
        store.upsert_bias(bias)

        ctx = engine.begin_turn(
            session_id="session-1",
            turn_index=1,
            user_message="Send a message to the customer",
            platform="cli",
            available_tools=["send_message", "web_search"],
            tool_defs=_tool_defs("send_message", "web_search"),
        )
        blocked = engine.evaluate_risk(
            ctx,
            tool_name="send_message",
            function_args={"message": "hello"},
        )
        assert blocked is not None

        store.set_bias_status(bias.id, status="disabled")
        ctx = engine.begin_turn(
            session_id="session-1",
            turn_index=2,
            user_message="Send a message to the customer",
            platform="cli",
            available_tools=["send_message", "web_search"],
            tool_defs=_tool_defs("send_message", "web_search"),
        )
        unblocked = engine.evaluate_risk(
            ctx,
            tool_name="send_message",
            function_args={"message": "hello"},
        )

        assert not ctx.active_biases
        assert unblocked is None
    finally:
        engine.close()


def test_policy_guard_blocks_do_not_count_as_failures_or_create_risk_evidence(tmp_path):
    engine, store = _make_engine(tmp_path)
    try:
        bias = _make_bias(engine.profile_id, "risk.inspect_before_execute", status="active")
        store.upsert_bias(bias)

        ctx = engine.begin_turn(
            session_id="session-1",
            turn_index=1,
            user_message="Send a message to the customer",
            platform="cli",
            available_tools=["send_message", "web_search"],
            tool_defs=_tool_defs("send_message", "web_search"),
        )
        blocked = engine.evaluate_risk(
            ctx,
            tool_name="send_message",
            function_args={"message": "hello"},
        )
        assert blocked is not None

        engine.record_tool_result(
            ctx,
            tool_name="send_message",
            function_args={"message": "hello"},
            result='{"success": false, "status": "blocked_by_policy_bias"}',
            duration_ms=1,
        )
        engine.record_turn_outcome(
            ctx,
            final_response="",
            completed=False,
            interrupted=False,
        )

        assert ctx.metadata.get("tool_failures", {}) == {}
        assert store.get_moments_by_candidate(engine.profile_id, "risk.inspect_before_execute") == []
        assert store.get_moments_by_candidate(engine.profile_id, "tool_use.change_strategy_after_retries") == []
    finally:
        engine.close()


def test_bias_history_and_rollback_restore_prior_version(tmp_path):
    store = PolicyBiasStore(db_path=tmp_path / "policy_bias.db")
    try:
        bias = _make_bias("profile:test", "planning.inspect_before_edit", status="active")
        store.upsert_bias(bias)
        store.set_bias_status(bias.id, status="disabled", note="manual disable")

        history = store.list_bias_history(bias.id, limit=10)
        assert [entry.version for entry in history][:2] == [2, 1]
        assert history[0].snapshot["status"] == "disabled"
        assert history[1].snapshot["status"] == "active"

        rolled_back = store.rollback_bias(bias.id, version=1)
        restored = store.get_bias(bias.id)
        history = store.list_bias_history(bias.id, limit=10)

        assert rolled_back is True
        assert restored is not None
        assert restored.status == "active"
        assert restored.version == 3
        assert restored.rollback_parent_id == f"{bias.id}:v2"
        assert history[0].operation == "rollback_to_v1"
    finally:
        store.close()


def test_inspect_and_boundary_audit_explain_why_bias_is_not_memory_or_skill(tmp_path):
    store = PolicyBiasStore(db_path=tmp_path / "policy_bias.db")
    try:
        bias = _make_bias("profile:test", "tool_use.patch_before_rewrite")
        store.upsert_bias(bias)

        inspected = inspect_bias(store, bias_id=bias.id)
        audited = audit_bias_boundaries(store, profile_id="profile:test", limit=10)

        assert inspected is not None
        assert inspected["boundary"]["classification"] == "policy_bias"
        assert "edit strategy" in inspected["boundary"]["why_not_memory"]
        assert audited[0]["candidate_key"] == "tool_use.patch_before_rewrite"
        assert "tool_ranking" in audited[0]["action_surfaces"]
    finally:
        store.close()


def test_shadow_bias_is_logged_but_does_not_change_behavior(tmp_path, caplog):
    engine, store = _make_engine(tmp_path)
    try:
        bias = _make_bias(engine.profile_id, "risk.inspect_before_execute", status="shadow")
        store.upsert_bias(bias)

        with caplog.at_level("INFO"):
            ctx = engine.begin_turn(
                session_id="session-1",
                turn_index=1,
                user_message="Send a message to the customer",
                platform="cli",
                available_tools=["send_message", "web_search"],
                tool_defs=_tool_defs("send_message", "web_search"),
            )

        assert not ctx.active_biases
        assert [shadow.id for shadow in ctx.shadow_biases] == [bias.id]
        assert ctx.decision_priors == ""
        assert [tool["function"]["name"] for tool in ctx.ranked_tools] == [
            "send_message",
            "web_search",
        ]
        assert "Policy bias retrieval" in caplog.text

        trace = store.get_recent_decision_traces(engine.profile_id, limit=1)[0]
        assert trace.shadow_bias_ids == [bias.id]
        assert trace.injected_bias_ids == []
    finally:
        engine.close()


def test_feedback_moments_are_deduplicated_within_a_turn(tmp_path):
    engine, store = _make_engine(tmp_path)
    try:
        kwargs = {
            "session_id": "session-1",
            "turn_index": 1,
            "user_message": "Be concise and direct",
            "platform": "cli",
            "available_tools": ["read_file"],
            "tool_defs": _tool_defs("read_file"),
        }
        engine.begin_turn(**kwargs)
        engine.begin_turn(**kwargs)

        concise_moments = store.get_moments_by_candidate(
            engine.profile_id,
            "communication.concise_first",
        )
        direct_moments = store.get_moments_by_candidate(
            engine.profile_id,
            "user_specific.directness_over_fluff",
        )

        assert len(concise_moments) == 1
        assert len(direct_moments) == 1
    finally:
        engine.close()


def test_generic_debug_words_do_not_promote_structured_output_bias(tmp_path):
    engine, store = _make_engine(tmp_path)
    try:
        engine.begin_turn(
            session_id="session-1",
            turn_index=1,
            user_message="Please review this bug in the repo",
            platform="cli",
            available_tools=["read_file", "patch"],
            tool_defs=_tool_defs("read_file", "patch"),
        )

        findings_moments = store.get_moments_by_candidate(
            engine.profile_id,
            "workflow_specific.structured_debugging_output",
        )
        assert findings_moments == []
    finally:
        engine.close()
