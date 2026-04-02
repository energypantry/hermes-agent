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


class _StubPolicyBiasEngine:
    def __init__(self, context: BiasDecisionContext):
        self.context = context
        self.turn_outcomes: list[dict] = []

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
