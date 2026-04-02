# Policy Bias Engine Implementation Plan

## Objective

Add a production-safe, profile-isolated `Policy Bias Engine` to Hermes so repeated or high-signal interaction patterns become compact decision priors that influence prompt construction, tool/path preference, planner heuristics, and external-action risk behavior without modifying model weights.

## Current Architecture Findings

- `run_agent.py` owns the real execution path:
  - `AIAgent.__init__()` loads config, tools, session persistence, memory, and context compression.
  - `_build_system_prompt()` assembles the cached system prompt.
  - `run_conversation()` builds per-turn ephemeral context and executes the main tool-calling loop.
  - `_execute_tool_calls_*()` is the real tool execution path.
  - `_compress_context()` and `flush_memories()` are the best existing checkpoint hooks.
- `hermes_state.py` already provides a SQLite pattern with WAL, migrations, and profile-scoped storage under `HERMES_HOME`.
- `model_tools.py` controls tool schema exposure and the final dispatcher, but the best weighting hook is in `AIAgent` before API submission and before tool execution.
- `hermes_cli/config.py` is the canonical config source and migration entry point.
- `hermes_cli/main.py` is the correct place for operational/admin subcommands.

## High-Level Design

### Storage

- Use a dedicated SQLite database at `~/.hermes/policy_bias.db` rather than extending `state.db`.
- Reasoning:
  - avoids coupling bias schema migrations to the existing session schema version;
  - keeps graceful degradation simple;
  - preserves backward compatibility for current session persistence.
- Reuse Hermes SQLite patterns:
  - WAL mode
  - local migration/version table
  - profile-scoped path derived from `HERMES_HOME`

### New Module Namespace

- `agent/policy_bias/models.py`
- `agent/policy_bias/store.py`
- `agent/policy_bias/migrations.py`
- `agent/policy_bias/scoring.py`
- `agent/policy_bias/synthesis.py`
- `agent/policy_bias/retrieval.py`
- `agent/policy_bias/injector.py`
- `agent/policy_bias/planner_hooks.py`
- `agent/policy_bias/explain.py`
- `agent/policy_bias/governance.py`

One thin façade service will be defined inside the namespace and instantiated by `AIAgent`.

## Data Model

### Moments

Store outcome-bearing events with:

- `id`
- `profile_id`
- `session_id`
- `timestamp`
- `task_type`
- `platform`
- `context_summary`
- `action_trace_summary`
- `tool_path`
- `decision_class`
- `outcome_class`
- `reward_score`
- `confidence_score`
- `user_feedback_signal`
- `error_signal`
- `side_effect_level`
- `latency_ms`
- `cost_estimate`
- `evidence_refs`
- `extracted_tags`
- `bias_candidate_key`

### Biases

Store durable priors with:

- `id`
- `profile_id`
- `scope`
- `condition_signature`
- `preferred_policy`
- `anti_policy`
- `rationale_summary`
- `confidence`
- `support_count`
- `avg_reward`
- `recency_score`
- `decay_rate`
- `status`
- `source_moment_ids`
- `created_at`
- `updated_at`
- `last_triggered_at`
- `trigger_count`
- `rollback_parent_id`
- `version`

### Decision Trace / Explainability

Add a separate trace table so every bias-triggered decision is inspectable:

- `decision_trace_id`
- `profile_id`
- `session_id`
- `message_turn_index`
- `task_type`
- `retrieved_bias_ids`
- `injected_bias_ids`
- `shadow_bias_ids`
- `planner_effects`
- `tool_weight_deltas`
- `risk_actions`
- `evidence_summary`
- `created_at`

## Integration Plan

### 1. Agent bootstrap

In `AIAgent.__init__()`:

- load `policy_bias` config;
- instantiate `PolicyBiasStore` and `PolicyBiasEngine`;
- fail open if the subsystem cannot initialize.

### 2. Prompt injection

In `run_conversation()` before `_build_api_kwargs()`:

- derive a turn context snapshot;
- retrieve top-k relevant biases;
- build a compact `Decision Priors` block;
- append it to the ephemeral system prompt for the current turn only.

This keeps retrieval scoped and token-bounded while preserving the cached stable system prompt.

### 3. Planner / next-step influence

Hermes has no standalone planner object, so planner influence will be implemented through explicit turn heuristics:

- compute a planning policy context from retrieved biases;
- bias tool definition ordering before API submission;
- bias multi-tool execution ordering before execution;
- bias retry behavior when similar failing paths recur.

This satisfies the requirement for non-prompt-only action influence.

### 4. Tool selection weighting

Before `_build_api_kwargs()`:

- score each available tool against retrieved tool-use/risk/platform biases;
- sort tool definitions by weighted preference;
- record the deltas into a decision trace.

### 5. External-action risk behavior

Before each tool execution:

- classify the tool call by side-effect level;
- apply risk biases and config gates;
- for high-risk calls, require inspect/simulate/confirm behavior when applicable.

Concrete first-pass policy:

- terminal destructive commands: require safer inspect-first path or approval-shaped error payload;
- browser click/type/press, `send_message`, `cronjob`, `ha_call_service`, destructive file ops: treated as external or high-side-effect candidates;
- repeated failing side-effect paths produce anti-policy signals.

### 6. Moment extraction

Create moments from:

- successful final turn completion;
- tool completion and tool failure;
- repeated failing tool paths;
- explicit positive/negative user feedback;
- correction-like user follow-ups;
- context compression / memory flush checkpoints;
- interrupted or partial/error outcomes where signal is meaningful.

### 7. Bias synthesis lifecycle

Implement deterministic promotion:

- raw moments accumulate by `bias_candidate_key`;
- repeated strong positive signals promote `candidate -> shadow -> active`;
- repeated negative signals reduce confidence or produce `anti_policy`;
- conflicting biases within a scope are resolved by weighted confidence, recency, and reward;
- decayed or disabled biases remain auditable.

### 8. Governance and controls

Add `hermes policy-bias ...` subcommands:

- `list`
- `inspect <bias-id>`
- `enable <bias-id>`
- `disable <bias-id>`
- `rescore [<bias-id>]`
- `rebuild`
- `moments`
- `explain`

These commands will operate directly on the policy-bias store and print structured summaries.

## Config Changes

Add a new top-level `policy_bias` section to `DEFAULT_CONFIG` and bump `_config_version`.

Planned defaults:

```yaml
policy_bias:
  enabled: true
  retrieval_top_k: 4
  max_prompt_tokens: 500
  shadow_mode_default: false
  synthesis:
    min_support_count: 3
    min_avg_reward: 0.25
    strong_signal_reward: 0.8
    confidence_decay_per_day: 0.01
  scopes_enabled:
    - planning
    - tool_use
    - risk
    - communication
    - user_specific
  risk_controls:
    require_inspect_before_execute_for_external_actions: true
    high_side_effect_shadow_only: false
  observability:
    log_bias_triggers: true
    expose_explanations: true
```

## Testing Plan

### Unit tests

- schema + migrations
- scoring / decay / confidence updates
- moment-to-bias promotion
- retrieval filtering and top-k behavior
- prompt injection token bounding
- planner/tool weighting deltas
- risk gating behavior

### Integration / end-to-end tests

- repeated successful pattern becomes active bias
- repeated negative outcome suppresses or anti-promotes a bias
- only top-k biases appear in `Decision Priors`
- profile isolation across separate homes / stores
- disabled bias no longer affects ranking or injection
- shadow bias logs retrieval but does not change behavior

### Regression coverage

- existing config loading remains backward-compatible
- AIAgent still runs when policy-bias store is unavailable
- session persistence, memory, and tool loading remain intact

## Documentation Deliverables

- `docs/policy-bias.md`
- `docs/policy-bias-ops.md`
- `docs/policy-bias-data-model.md`
- `docs/policy-bias-final-report.md`

## Implementation Order

1. Build policy-bias models, store, migrations, scoring, and synthesis.
2. Wire config and AIAgent bootstrap.
3. Add retrieval, injection, planner/tool weighting, and risk hooks.
4. Add explainability traces and governance CLI.
5. Add tests.
6. Add docs and final report.
