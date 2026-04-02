"""Policy-bias integration tests for the AIAgent conversation loop."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.policy_bias.models import BiasDecisionContext
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
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


def _mock_response(content: str = "done"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _mock_tool_response(*tool_calls):
    msg = SimpleNamespace(content="", tool_calls=list(tool_calls))
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _tool_call(name: str, arguments: str = "{}"):
    return SimpleNamespace(
        id=f"call_{name}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _StubPolicyBiasEngine:
    def __init__(self, context: BiasDecisionContext):
        self.context = context
        self.turn_outcomes: list[dict] = []
        self.recorded_planner_effects: list[dict] = []

    def is_enabled(self) -> bool:
        return True

    def begin_turn(self, **_kwargs):
        return self.context

    def record_turn_outcome(self, context, **kwargs) -> None:
        self.turn_outcomes.append({"context": context, **kwargs})

    def record_checkpoint(self, **_kwargs) -> None:
        return None

    def evaluate_risk(self, *_args, **_kwargs):
        return None

    def record_tool_result(self, *_args, **_kwargs) -> None:
        return None

    def rerank_tool_calls(self, context, parsed_calls):
        return parsed_calls, []

    def record_planner_effects(self, context, planner_effects) -> None:
        self.recorded_planner_effects.extend(planner_effects)


def test_run_conversation_injects_decision_priors_and_ranked_tools():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("patch", "read_file"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()

    context = BiasDecisionContext(
        session_id="session-1",
        turn_index=1,
        task_type="repo_modification",
        platform="cli",
        user_message="Fix the repo file",
        decision_priors=(
            "Decision Priors\n"
            "- [scope=planning confidence=0.92] Inspect local files before editing."
        ),
        ranked_tools=_make_tool_defs("read_file", "patch"),
    )
    stub_engine = _StubPolicyBiasEngine(context)
    agent._policy_bias_engine = stub_engine
    agent._base_tools = _make_tool_defs("patch", "read_file")
    agent.valid_tool_names = {"patch", "read_file"}

    captured_api_kwargs = {}

    def _fake_call(api_kwargs):
        captured_api_kwargs.update(api_kwargs)
        return _mock_response("done")

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=_fake_call),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_honcho_sync"),
        patch.object(agent, "_queue_honcho_prefetch"),
    ):
        result = agent.run_conversation("Fix the repo file", sync_honcho=False)

    assert result["final_response"] == "done"
    assert captured_api_kwargs["messages"][0]["role"] == "system"
    assert "Decision Priors" in captured_api_kwargs["messages"][0]["content"]
    assert [tool["function"]["name"] for tool in captured_api_kwargs["tools"]] == [
        "read_file",
        "patch",
    ]
    assert stub_engine.turn_outcomes


def test_repo_modification_request_reads_before_patch_when_bias_engine_is_live():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("patch", "read_file"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()

    execution_order: list[str] = []

    def _fake_call(_api_kwargs):
        if not hasattr(_fake_call, "count"):
            _fake_call.count = 0
        _fake_call.count += 1
        if _fake_call.count == 1:
            return _mock_tool_response(
                _tool_call("patch", '{"path": "app.py"}'),
                _tool_call("read_file", '{"path": "app.py"}'),
            )
        return _mock_response("done")

    def _fake_tool_handler(function_name, function_args, *_args, **_kwargs):
        execution_order.append(function_name)
        return '{"success": true}'

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=_fake_call),
        patch("run_agent.handle_function_call", side_effect=_fake_tool_handler),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_honcho_sync"),
        patch.object(agent, "_queue_honcho_prefetch"),
    ):
        result = agent.run_conversation("Please modify app.py to fix the bug", sync_honcho=False)

    assert result["final_response"] == "done"
    assert execution_order[:2] == ["read_file", "patch"]


def test_one_step_bias_forces_sequential_tool_batches():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("patch", "read_file", "todo"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    context = BiasDecisionContext(
        session_id="session-1",
        turn_index=1,
        task_type="repo_modification",
        platform="cli",
        user_message="Fix the repo file one step at a time",
        active_biases=[
            SimpleNamespace(bias_candidate_key="user_specific.one_step_at_a_time"),
        ],
    )
    agent._policy_turn_context = context

    tool_calls = [
        SimpleNamespace(function=SimpleNamespace(name="todo")),
        SimpleNamespace(function=SimpleNamespace(name="patch")),
    ]

    assert agent._policy_requires_sequential_tool_batch(tool_calls) is True


def test_policy_state_forces_sequential_tool_batches():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("patch", "read_file", "todo"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    context = BiasDecisionContext(
        session_id="session-1",
        turn_index=1,
        task_type="repo_modification",
        platform="cli",
        user_message="Fix the repo file one step at a time",
        metadata={
            "_policy_state_dimensions": [
                SimpleNamespace(dimension_key="single_step_tendency", value=0.8),
            ]
        },
    )
    agent._policy_turn_context = context

    tool_calls = [
        SimpleNamespace(function=SimpleNamespace(name="todo")),
        SimpleNamespace(function=SimpleNamespace(name="patch")),
    ]

    assert agent._policy_requires_sequential_tool_batch(tool_calls) is True


def test_policy_state_plan_forces_sequential_tool_batches():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("patch", "read_file", "todo"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    context = BiasDecisionContext(
        session_id="session-1",
        turn_index=1,
        task_type="repo_modification",
        platform="cli",
        user_message="Fix the repo file one step at a time",
        metadata={
            "_policy_state_plan": SimpleNamespace(require_sequential=True),
        },
    )
    agent._policy_turn_context = context

    tool_calls = [
        SimpleNamespace(function=SimpleNamespace(name="todo")),
        SimpleNamespace(function=SimpleNamespace(name="patch")),
    ]

    assert agent._policy_requires_sequential_tool_batch(tool_calls) is True


def test_policy_state_plan_limits_tool_batch_before_execution():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("clarify", "patch", "send_message"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    context = BiasDecisionContext(
        session_id="session-1",
        turn_index=1,
        task_type="repo_modification",
        platform="slack",
        user_message="Maybe send the update after you inspect the regression.",
        metadata={
            "_policy_state_plan": SimpleNamespace(max_tool_calls_per_turn=1),
        },
    )
    stub_engine = _StubPolicyBiasEngine(context)
    agent._policy_turn_context = context
    agent._policy_bias_engine = stub_engine

    tool_calls = [
        _tool_call("clarify"),
        _tool_call("patch"),
        _tool_call("send_message"),
    ]

    limited = agent._policy_limit_tool_batch(tool_calls)

    assert [tool.function.name for tool in limited] == ["clarify"]
    assert stub_engine.recorded_planner_effects
    assert stub_engine.recorded_planner_effects[0]["kind"] == "tool_batch_limit"
    assert stub_engine.recorded_planner_effects[0]["omitted_tool_names"] == ["patch", "send_message"]


def test_policy_state_plan_caps_parallel_workers():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("read_file", "search_files", "web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent._policy_turn_context = BiasDecisionContext(
        session_id="session-1",
        turn_index=1,
        task_type="repo_modification",
        platform="cli",
        user_message="Inspect this issue.",
        metadata={
            "_policy_state_plan": SimpleNamespace(max_parallel_tools=2),
        },
    )

    limit = agent._policy_parallel_tool_worker_limit(
        [_tool_call("read_file"), _tool_call("search_files"), _tool_call("web_search")]
    )

    assert limit == 2


def test_policy_response_controls_strip_fluff_and_add_findings_heading():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("read_file"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()

    context = BiasDecisionContext(
        session_id="session-1",
        turn_index=1,
        task_type="repo_modification",
        platform="cli",
        user_message="Review this bug and give findings first. Be concise and direct.",
        ranked_tools=_make_tool_defs("read_file"),
        metadata={
            "response_controls": {
                "strip_leading_acknowledgement": True,
                "drop_trailing_offer": True,
                "findings_first_heading": True,
            },
        },
    )
    stub_engine = _StubPolicyBiasEngine(context)
    stub_engine.record_response_effects = MagicMock()
    agent._policy_bias_engine = stub_engine
    agent._base_tools = _make_tool_defs("read_file")
    agent.valid_tool_names = {"read_file"}

    response_text = (
        "Got it.\n\n"
        "The failure is caused by a missing guard in the file loader.\n\n"
        "Let me know if you want a deeper walkthrough."
    )

    with (
        patch.object(agent, "_interruptible_api_call", return_value=_mock_response(response_text)),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_honcho_sync"),
        patch.object(agent, "_queue_honcho_prefetch"),
    ):
        result = agent.run_conversation(
            "Review this bug and give findings first. Be concise and direct.",
            sync_honcho=False,
        )

    assert result["final_response"] == (
        "**Findings**\nThe failure is caused by a missing guard in the file loader."
    )
    stub_engine.record_response_effects.assert_called_once()
