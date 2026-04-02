# Policy Bias V2 Foundation Plan

## Goal

Evolve the current Policy Bias Engine from a bias-object and prompt-assisted control system into a long-lived policy-state layer that is compiled into runtime policy decisions rather than re-expressed as prompt text.

The key shift is:

- V1: `moments -> bias objects -> retrieval -> prompt/hook application`
- V2: `moments -> policy state updates -> policy-state compiler / arbitration -> low-bandwidth control surfaces -> optional prompt translation`

This does **not** mean immediate model fine-tuning. It means Hermes gains a persistent, profile-scoped internal policy state that can shape behavior with far less prompt dependence.

## Why V2 Exists

V1 already adds real value over memory and skills:

- it stores action priors separately from memory and skills
- it influences planner/tool/risk behavior, not just phrasing
- it is auditable, reversible, and bounded

But V1 still has two limitations:

1. Biases are still represented primarily as explicit text objects.
2. Some behavior still depends on prompt translation (`Decision Priors`).

The desired V2 direction is closer to "底蕴":

- long-lived
- low-bandwidth
- partially implicit
- not re-read as a large text document every turn
- able to influence many future decisions through stable tendencies

## Design Principle Shift

V1 design principle:

- durable policy priors should be explicit, explainable objects

V2 design principle:

- durable policy priors should remain explainable, but the *primary runtime representation* should be structured policy state, with a compiler/arbitration layer producing runtime controls and only then prompt hints

That means:

- bias text becomes one governance/export surface
- policy state becomes the main runtime control surface

## V2 Core Concept

Introduce a new first-class layer:

- `PolicyState`

This layer is profile-scoped, persistent, decayed over time, evidence-backed, and continuously updated by moments.

`PolicyState` is not a collection of prose rules.
It is a compact vector-like state object composed of policy dimensions that feed a policy-state compiler.

Examples:

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

Each dimension persists across turns and profiles, changes gradually, and influences multiple downstream systems.

### 2.5 Policy-State Compiler

Add a runtime compiler that turns policy state into a per-turn policy plan.

Inputs:

- active policy dimensions
- task type
- platform
- available tools
- recent failure and inspection history

Outputs:

- tool weights and ordering deltas
- risk thresholds and gates
- response budget and structure hints
- a bounded prompt translation payload when needed

The compiler should resolve conflicts deterministically. When dimensions disagree, arbitration should prefer the combination of:

- task relevance
- confidence
- support count
- recency
- side-effect risk
- profile-specific overrides

### 2.6 V2.1 Runtime Compiler Upgrade

The next concrete upgrade after the initial V2 substrate is to move from
"effective values + a few hooks" to a more unified per-turn compiler output.

The compiler should now emit:

- `planner_mode`
- `clarify_priority`
- `tool_class_weights`
- `response budget controls`
- `conflict_resolutions`

Examples:

- ambiguous + high-risk + shared-channel context should produce `planner_mode=clarify_first`
- directness should not suppress risk confirmation thresholds
- findings-first and one-step-at-a-time can co-exist by preserving findings structure while capping numbered next steps to one
- local-first and inspect-first should strengthen local inspection weights without depending on prompt text

### 2.7 V2.2 Action Budget and Runtime Coverage

The next upgrade is to let the compiler control action amplitude, not just action ordering.

The compiler should now also emit:

- `max_tool_calls_per_turn`
- `max_parallel_tools`
- `runtime_surfaces`

Examples:

- clarify-first or high-caution turns should cap execution to one tool decision before the next model turn
- inspect-first / decomposition-heavy turns can still allow two tools, but should constrain them to sequential execution
- when planner, risk, response, and execution-budget surfaces are already covered by the runtime, prompt translation for policy state should switch fully off

This makes policy state feel less like prompt text and more like a low-bandwidth behavioral substrate that governs how much the agent acts before re-evaluating.

### 2.8 V2.3 Execution-Mode Scoring

The next upgrade is to stop relying only on discrete runtime modes and instead compile a stable execution-mode score map for each turn.

The compiler should now also emit:

- `execution_mode_scores`
- `runtime_coverage_score`

Examples:

- risky ambiguous shared-channel turns should push `confirm` and `clarify` scores above `direct`
- browser-side-effect turns should preserve a strong `simulate` default even when inspection is also valuable
- prompt translation should stay off when runtime coverage is already strong enough to carry planner, risk, response, and execution-budget behavior

This pushes the substrate closer to an internal behavioral field: the agent is no longer just picking a named mode, it is carrying graded execution preferences that can be consumed by downstream control surfaces.

### 2.9 V2.4 Action-Surface and Response-Shape Scoring

The next upgrade is to stop relying on a narrow set of class weights and boolean response flags and instead compile shared score maps for action surfaces and response shapes.

The compiler should now also emit:

- `action_surface_scores`
- `response_shape_scores`

Examples:

- repo-local inspection pressure should raise `inspect_local` above `inspect_external`, so planner and tool routing can favor local evidence surfaces before browsing
- ambiguity plus high caution should raise `clarify` as an action surface, not just a planner mode label
- concise/debug/review tendencies should compile into `concise`, `findings_first`, `single_step`, and `structured_debug` response-shape scores before they are translated into deterministic response controls
- planner, response, and tool re-ranking should consume these compiled scores first, with prompt translation staying suppressed when runtime coverage is already strong

This is the next step toward “底蕴”: the runtime is being driven less by isolated rules and more by a compact graded substrate that can influence many downstream actions without becoming a large prompt artifact.

## Proposed Architecture

### 1. Moment Layer

Keep V1 moments, but extend them to support V2 state updates.

New V2 interpretation:

- moments are no longer only evidence for synthesizing a named bias
- moments are also direct update events for one or more policy dimensions

Each moment can emit:

- `candidate_key` for V1 compatibility
- `state_update_signals` for V2 evolution

Examples:

- repeated successful inspect-before-edit moments increase `inspect_tendency`
- repeated side-effect near-misses increase `risk_aversion`
- repeated "be concise" and "findings first" feedback updates `verbosity_budget`, `directness_tendency`, and `findings_first_tendency`

### 2. Policy State Layer

Add a new persistent table and models:

- `policy_state_dimensions`
- optional `policy_state_history`

Each row represents one dimension for one profile:

- `profile_id`
- `dimension_key`
- `value`
- `confidence`
- `support_count`
- `last_updated_at`
- `decay_rate`
- `source_summary`
- `status`
- `version`

Optional related records:

- recent update contributors
- per-dimension evidence references
- rollback metadata

### 3. State Update Layer

Add a deterministic state updater:

- `moments -> dimension deltas`

This replaces "bias synthesis only" as the primary learning path.

Desired properties:

- bounded update step size
- saturating values
- decay over time
- profile isolation
- explainable update reasons
- conflict-aware balancing

Example shape:

- dimensions live in `[-1.0, 1.0]` or `[0.0, 1.0]`
- updates use weighted EMA / bounded additive deltas
- confidence and support evolve separately from raw value

### 4. Runtime Control Layer

The main V2 change:

- planner
- tool router
- risk gate
- response controls

should read from the compiled `PolicyState` plan first.

Examples:

- `inspect_tendency` raises inspect tool scores
- `risk_aversion` raises confirm/simulate thresholds
- `local_first_tendency` penalizes browser/web paths on code-local tasks
- `decomposition_tendency` favors planning tools and sequential batches
- `verbosity_budget` constrains final response shaping
- `findings_first_tendency` changes review/debug output structure

### 5. Prompt Translation Layer

`Decision Priors` should remain, but become secondary.

New role:

- translate the surviving compiler output into compact prompt hints only when needed

That means:

- not every dimension should become prompt text
- prompt translation should be late, sparse, and bounded
- runtime control should not depend on prompt translation
- prompt text is a fallback output of the compiler, not the source of truth

### 6. Governance Layer

V2 must keep V1 explainability strengths.

Operators must be able to inspect:

- current policy dimensions
- their values and confidence
- the compiler/arbitration inputs that shaped the current plan
- which moments changed them
- how they affected planner/tool/risk/response decisions
- how they decayed over time

Governance needs:

- list policy state
- inspect one dimension
- show recent updates
- replay / recompute state
- inspect the compiled policy plan for a turn
- reset one dimension
- reset all policy state for a profile
- compare V1 bias objects vs V2 state effects

### 7. CLI / Admin Surfaces

The V2 governance surface should be exposed through the existing `policy-bias` command family rather than a separate brand-new top-level command.

Proposed command tree:

- `hermes policy-bias state list`
- `hermes policy-bias state inspect <dimension_key>`
- `hermes policy-bias state updates`
- `hermes policy-bias state rebuild`
- `hermes policy-bias state reset [<dimension_key>] [--all]`
- `hermes policy-bias state explain <trace_id>`

Operational contract:

- these commands should remain profile-scoped
- they should prefer JSON when `--json` is passed
- they should fail gracefully if the policy-state backend has not been wired yet
- they should coexist with V1 bias governance rather than replacing it

## V2 Data Model

### New Model: `PolicyStateDimension`

Suggested fields:

- `id`
- `profile_id`
- `dimension_key`
- `value`
- `confidence`
- `support_count`
- `avg_reward`
- `recency_score`
- `decay_rate`
- `status` (`active`, `shadow`, `disabled`, `archived`)
- `last_updated_at`
- `last_triggered_at`
- `trigger_count`
- `version`
- `rollback_parent_id`
- `state_metadata`

### New Model: `PolicyStateUpdate`

Suggested fields:

- `id`
- `profile_id`
- `dimension_key`
- `source_moment_id`
- `delta`
- `previous_value`
- `new_value`
- `update_reason`
- `confidence_delta`
- `created_at`

This gives a clean audit trail without requiring full state snapshot rewrites for every tiny update.

## Runtime Strategy

### Phase 1: Hybrid V1/V2

Do not delete V1 bias objects.

Instead:

- keep V1 bias synthesis for governance and backward compatibility
- add V2 policy state in parallel
- let runtime control prefer V2 state when available
- keep `Decision Priors` as a fallback translation layer

This reduces migration risk.

### Phase 2: State-First Runtime

Once V2 proves stable:

- planner/tool/risk/response controls read primarily from the compiled `PolicyState` plan
- bias object retrieval becomes secondary / governance-facing
- prompt translation only emits top few translated priors

### Phase 3: Optional Training Path

After enough evidence:

- export policy-state trajectories
- train reranker / sidecar policy model / offline adapter if desired

## What V2 Is Not

V2 is not:

- a giant prompt file
- hidden arbitrary behavior with no traceability
- a direct fine-tune of the base model
- a replacement for memory or skills

## Compatibility Requirements

V2 must preserve:

- current profiles
- existing sessions
- memory and skill flows
- V1 bias governance
- graceful degradation when state store fails

## Suggested Module Additions

Under `agent/policy_bias/`:

- `state_models.py`
- `state_store.py`
- `state_updates.py`
- `state_runtime.py`
- `state_explain.py`
- `state_governance.py`

Or, if cleaner:

- extend existing `models.py`, `store.py`, `engine.py`, and `governance.py`

## Implementation Phases

### Workstream A: State Schema and Persistence

- add V2 tables and migrations
- add `PolicyStateDimension` and `PolicyStateUpdate`
- add persistence APIs

### Workstream B: State Update Engine

- map moments to dimension deltas
- implement decay and bounded updates
- add recompute/rebuild logic

### Workstream C: Runtime Consumption

- make planner/tool/risk/response controls read policy state
- add state-aware prompt translation as a fallback output
- add deterministic arbitration for conflicting dimensions
- preserve V1 fallback behavior

### Workstream D: Governance and Explainability

- CLI/admin surfaces for listing, inspecting, rebuilding, and resetting state
- explainability for state-driven decisions
- preserve existing `policy-bias` bias governance and add a `state` sub-tree for V2

### Workstream E: Tests and Docs

- unit tests for state updates and decay
- integration tests for runtime effects
- migration coverage
- operator docs

## Success Criteria

V2 is successful when:

- Hermes has persistent policy state independent of memory and skills
- the primary runtime driver is a compiled policy plan derived from structured policy state rather than prompt text
- `Decision Priors` becomes a secondary translation layer
- policy-state arbitration is deterministic, bounded, and explainable
- planner/tool/risk/response behavior can be explained via state dimensions
- V1 remains backward-compatible during transition

## Task Breakdown

The implementation should be split into these parallel task groups:

1. Schema + models
2. State update engine
3. Runtime integration
4. Governance + explainability
5. Tests + docs

These task groups have mostly disjoint write scopes and can be handled by separate worker agents in parallel.
