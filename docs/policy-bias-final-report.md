# Policy Bias Engine Final Report

## Architecture Implemented

Implemented a first-class Policy Bias Engine with:

- moment capture in SQLite
- deterministic synthesis from repeated/high-signal moments
- bounded retrieval by profile, scope-balanced selection, and top-k ranking
- explicit `Decision Priors` prompt injection
- planner/tool weighting hooks
- risk gating for external and mutating actions, including inspect / simulate / confirm decisions
- deterministic response-policy hooks for concise/direct and findings-first behavior, with traceable response effects
- policy-driven tool-batch preparation so reordered and limited tool calls are persisted consistently before execution
- explainability traces
- governance controls for enable, disable, archive, history, rollback, rebuild, and export
- boundary metadata and audit tooling to keep policy bias separate from memory and skills
- a V2 policy-state foundation with persistent dimensions, update history, compiler-oriented arbitration, and runtime consumption
- a V2.1 compiler surface with explicit `planner_mode`, `clarify_priority`, `tool_class_weights`, response budgets, and conflict-resolution metadata
- a V2.2 runtime substrate with action-budget controls (`max_tool_calls_per_turn`, `max_parallel_tools`) and stronger state-prompt suppression when runtime coverage is already high
- a V2.3 execution substrate with explicit `execution_mode_scores` and `runtime_coverage_score`, allowing risk behavior to consume graded preferences instead of only discrete modes
- a V2.4 substrate-scoring layer with explicit `action_surface_scores` and `response_shape_scores`, allowing planner and response shaping to consume shared compiled surfaces instead of only class weights and boolean flags
- a `policy-bias state ...` CLI subtree backed by the policy-state store for inspection, rebuild, reset, and explain flows

The engine is integrated into the live Hermes conversation loop in `run_agent.py`, not just a standalone prototype module.

## Files Created

- `agent/policy_bias/__init__.py`
- `agent/policy_bias/models.py`
- `agent/policy_bias/migrations.py`
- `agent/policy_bias/store.py`
- `agent/policy_bias/scoring.py`
- `agent/policy_bias/synthesis.py`
- `agent/policy_bias/retrieval.py`
- `agent/policy_bias/injector.py`
- `agent/policy_bias/planner_hooks.py`
- `agent/policy_bias/explain.py`
- `agent/policy_bias/governance.py`
- `agent/policy_bias/engine.py`
- `agent/policy_bias/response_hooks.py`
- `hermes_cli/policy_bias_cmd.py`
- `docs/policy-bias-implementation-plan.md`
- `docs/policy-bias.md`
- `docs/policy-bias-ops.md`
- `docs/policy-bias-data-model.md`
- `docs/policy-bias-v2-foundation-plan.md`
- `docs/policy-bias-v2-prd-prompt.md`
- `tests/agent/test_policy_bias_engine.py`
- `tests/hermes_cli/test_policy_bias_cmd.py`
- `tests/hermes_cli/test_policy_state_cmd.py`
- `tests/test_run_agent_policy_bias.py`

## Files Modified

- `agent/policy_bias/models.py`
- `agent/policy_bias/engine.py`
- `agent/policy_bias/state_runtime.py`
- `agent/policy_bias/planner_hooks.py`
- `agent/policy_bias/response_hooks.py`
- `run_agent.py`
- `hermes_cli/config.py`
- `hermes_cli/main.py`
- `tests/tools/test_browser_camofox_state.py`
- `tests/agent/test_policy_bias_engine.py`
- `tests/test_run_agent_policy_bias.py`
- `hermes_cli/policy_bias_cmd.py`
- `docs/policy-bias.md`
- `docs/policy-bias-ops.md`
- `docs/policy-bias-final-report.md`

## Migration Notes

- Added dedicated SQLite database: `~/.hermes/policy_bias.db`
- Added schema versioning through `agent/policy_bias/migrations.py`
- Current schema version after this work: `5`

Schema evolution:

1. base schema: `moments` and `biases`
2. v2: `decision_traces` and `disabled_reason`
3. v3: `bias_history` for audit and rollback
4. v4: `response_effects` on `decision_traces`
5. v5: `policy_state_dimensions` and `policy_state_updates`

Rollback notes:

- disabling or archiving a bias is reversible
- `rollback` restores a prior snapshot as a new current version
- audit history is preserved during rollback

## Test Coverage Summary

Added tests for:

- schema migration and status-filtered deletion
- synthesis lifecycle and negative suppression
- descriptor boundary validation and audit metadata
- retrieval top-k behavior, scope-balanced selection, and profile isolation
- prompt injection bounding
- planner reranking and risk gating
- staged risk decisions for inspect / simulate / confirm
- deterministic response-policy controls and trace recording
- policy-state action-budget limits for tool batch breadth, worker caps, and state-only runtime coverage
- policy-state execution-mode scoring for `direct / inspect / simulate / confirm / clarify`
- policy-state action-surface scoring for `inspect_local / inspect_external / plan / clarify / mutate_local / mutate_external`
- policy-state response-shape scoring for `concise / findings_first / single_step / structured_debug`
- repeated-success promotion into active bias
- disabling a bias and verifying influence stops
- shadow bias observability without behavior impact
- feedback moment deduplication within a turn
- bias history and rollback
- CLI list / disable / export / rollback flows
- CLI audit / inspect boundary reporting
- run-loop integration for `Decision Priors` prompt injection and ranked tool surfaces
- run-loop integration for final-response post-processing under active communication / workflow biases

Verification completed in this environment:

- `py_compile` passed for all new policy-bias modules, CLI entrypoints, and tests
- policy-bias focused pytest suite passed:
  - `tests/agent/test_policy_bias_engine.py`
  - `tests/hermes_cli/test_policy_bias_cmd.py`
  - `tests/test_run_agent_policy_bias.py`
- current focused result after compiler, action-budget, execution-mode-score, and substrate-score validation: `47 passed`
- full project `pytest` was executed in a provisioned local virtualenv and produced:
  - `7625 passed`
  - `220 skipped`
  - `1 xpassed`
  - `37 failed`

The remaining full-suite failures in this environment are concentrated outside the policy-bias focused surfaces and still require separate mainline cleanup.

## Known Limitations

- CI or mainline merge validation is still required even though local full-suite execution was completed
- semantic retrieval is lightweight token-overlap based, not embedding-based
- rollback restores a prior bias snapshot, but there is no separate visual diff renderer yet
- export is JSON-oriented governance output, not a full offline training pipeline on its own
- concise/directness and findings-first biases still function more like response-policy surfaces than deep control-plan dimensions
- the current policy-state compiler is intentionally deterministic and lightweight; it is a substrate for deeper arbitration rather than a learned policy model

## Future Work Toward Offline Weight Updates

1. join exported stable biases with decision traces and supporting moments
2. bucket by scope and candidate key
3. generate preference / policy-training examples with positive and anti-policy pairs
4. evaluate whether certain biases should graduate from online policy control into offline training data
5. evolve the policy-state compiler into a richer arbitration layer while keeping the online Policy Bias Engine as the auditable, reversible control plane even after offline training begins
