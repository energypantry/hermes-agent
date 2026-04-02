# Policy Bias V2 PRD Prompt

Use this prompt for a future implementation pass that evolves Hermes from V1 bias objects toward V2 long-lived policy state.

---

You are working inside the latest `feature/policy-bias-engine` branch of `NousResearch/hermes-agent`.

Your mission is to evolve the existing Policy Bias Engine into a V2 “Policy State / Agent Substrate” system that behaves more like long-lived background inclination than explicit prompt text.

## Product Goal

Hermes already has:

- short-term context
- memory / facts
- skills / reusable procedures
- a V1 Policy Bias Engine with moment capture, bias objects, retrieval, prompt injection, planner/tool/risk hooks, governance, and tests

You must build the next layer:

- persistent policy state that survives across turns and profiles
- lower-bandwidth than text bias objects
- less dependent on prompt injection
- still explainable, auditable, editable, and safe

This V2 layer should move Hermes closer to durable “底蕴” without modifying base model weights.

## Non-Goals

- Do not fine-tune the base model in this task.
- Do not remove V1 bias objects outright.
- Do not replace memory or skills.
- Do not turn this into a hidden black-box behavior system.

## Core Concept

Introduce `PolicyState` as a first-class runtime substrate.

The primary runtime control flow should become:

`moments -> state updates -> planner/tool/risk/response control -> optional prompt translation`

instead of:

`moments -> text bias retrieval -> prompt-first influence`

## Required V2 Capabilities

### 1. Policy State Dimensions

Add persistent, profile-scoped policy dimensions such as:

- `inspect_tendency`
- `risk_aversion`
- `local_first_tendency`
- `decomposition_tendency`
- `retry_switch_tendency`
- `directness_tendency`
- `verbosity_budget`
- `findings_first_tendency`
- `single_step_tendency`
- `shared_channel_caution`

These dimensions must:

- persist in SQLite
- update gradually over time
- decay over time
- be bounded and explainable
- remain profile-isolated

### 2. State Update Engine

Implement deterministic updates from moments into policy state.

Requirements:

- bounded update deltas
- confidence and support tracking
- recency weighting
- decay
- rebuild/recompute from stored moments
- evidence references for explainability

### 3. Runtime State Consumption

Make runtime systems consume `PolicyState` directly:

- planner weighting
- tool selection weighting
- tool-batch ordering
- risk gating
- response controls

The runtime should prefer policy state over prompt text where possible.

### 4. Prompt Translation as Secondary Layer

Keep `Decision Priors`, but make it a secondary translation layer:

- only emit compact prompt hints when needed
- do not treat prompt injection as the primary effect path
- keep token use bounded

### 5. Governance

Add governance surfaces for policy state:

- list current state dimensions
- inspect one dimension
- show recent state updates
- rebuild state from moments
- reset one dimension
- reset all state for a profile
- explain how state affected a decision

Expose these through the existing Hermes CLI as a `policy-bias state ...` sub-tree so V1 bias governance and V2 policy-state governance coexist in the same operational surface.

### 6. Explainability

Every state-driven behavior change must be explainable.

The explanation layer must show:

- which state dimensions were active
- what their values/confidence were
- what planner/tool/risk/response effects they caused
- which moments updated them

If the underlying policy-state store is not yet present, the CLI surface should fail gracefully and clearly report that the backing API is pending, rather than crashing or silently ignoring the command.

## Storage Requirements

Add SQLite-backed persistence for:

- `policy_state_dimensions`
- `policy_state_updates`

Provide migrations and backward compatibility with the existing V1 database.

## Backward Compatibility

You must preserve:

- existing V1 bias objects
- current moments
- current governance commands
- current profile behavior
- graceful degradation if the new state store fails

V1 and V2 should coexist during migration.

## Suggested Module Structure

Extend or add under `agent/policy_bias/`:

- `state_models.py`
- `state_store.py`
- `state_updates.py`
- `state_runtime.py`
- `state_explain.py`
- `state_governance.py`

Or cleanly extend the existing modules if that produces a better integration.

## Tests

Add tests for:

- schema migration
- bounded state updates
- decay behavior
- rebuild from moments
- runtime weighting based on state dimensions
- prompt translation fallback
- explainability payloads
- profile isolation
- reset/rebuild governance flows

## Deliverables

You must produce:

1. code implementing Policy Bias V2 foundation
2. SQLite migration(s)
3. tests
4. docs explaining Policy State architecture
5. a final report summarizing what changed and how V1/V2 coexist

## Working Style

- inspect the existing V1 implementation first
- preserve current production-safe conventions
- prefer deterministic control logic over vague LLM-only behavior
- keep the system auditable and reversible

---

Recommended first step:

Create a short implementation plan describing how V2 will coexist with V1 during transition, then implement schema + state update primitives before changing runtime behavior.
