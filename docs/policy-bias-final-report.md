# Policy Bias Engine Final Report

## Architecture Implemented

Implemented a first-class Policy Bias Engine with:

- moment capture in SQLite
- deterministic synthesis from repeated/high-signal moments
- bounded retrieval by profile, scope, and top-k ranking
- explicit `Decision Priors` prompt injection
- planner/tool weighting hooks
- risk gating for external and mutating actions
- explainability traces
- governance controls for enable, disable, archive, history, rollback, rebuild, and export

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
- `hermes_cli/policy_bias_cmd.py`
- `docs/policy-bias-implementation-plan.md`
- `docs/policy-bias.md`
- `docs/policy-bias-ops.md`
- `docs/policy-bias-data-model.md`
- `tests/agent/test_policy_bias_engine.py`
- `tests/hermes_cli/test_policy_bias_cmd.py`
- `tests/test_run_agent_policy_bias.py`

## Files Modified

- `run_agent.py`
- `hermes_cli/config.py`
- `hermes_cli/main.py`
- `tests/tools/test_browser_camofox_state.py`

## Migration Notes

- Added dedicated SQLite database: `~/.hermes/policy_bias.db`
- Added schema versioning through `agent/policy_bias/migrations.py`
- Current schema version after this work: `3`

Schema evolution:

1. base schema: `moments` and `biases`
2. v2: `decision_traces` and `disabled_reason`
3. v3: `bias_history` for audit and rollback

Rollback notes:

- disabling or archiving a bias is reversible
- `rollback` restores a prior snapshot as a new current version
- audit history is preserved during rollback

## Test Coverage Summary

Added tests for:

- schema migration and status-filtered deletion
- synthesis lifecycle and negative suppression
- retrieval top-k behavior and profile isolation
- prompt injection bounding
- planner reranking and risk gating
- repeated-success promotion into active bias
- disabling a bias and verifying influence stops
- shadow bias observability without behavior impact
- feedback moment deduplication within a turn
- bias history and rollback
- CLI list / disable / export / rollback flows
- run-loop integration for `Decision Priors` prompt injection and ranked tool surfaces

Verification completed in this environment:

- `py_compile` passed for all new policy-bias modules, CLI entrypoints, and tests
- custom smoke scripts passed for:
  - repeated-success synthesis to active bias
  - next-turn decision-prior injection and tool reranking
  - governance export / disable / rollback history flow

Full `pytest` execution was not possible in the current sandbox because:

- no local project virtualenv was present
- `pytest` and runtime deps were unavailable in system Python
- outbound network resolution to PyPI / GitHub was unavailable, so `uv` could not fetch missing packages

## Known Limitations

- full test suite execution is still required in a properly provisioned Hermes dev environment
- semantic retrieval is lightweight token-overlap based, not embedding-based
- rollback restores a prior bias snapshot, but there is no separate visual diff renderer yet
- export is JSON-oriented governance output, not a full offline training pipeline on its own

## Future Work Toward Offline Weight Updates

1. join exported stable biases with decision traces and supporting moments
2. bucket by scope and candidate key
3. generate preference / policy-training examples with positive and anti-policy pairs
4. evaluate whether certain biases should graduate from online policy control into offline training data
5. keep the online Policy Bias Engine as the auditable, reversible control layer even after offline training begins
