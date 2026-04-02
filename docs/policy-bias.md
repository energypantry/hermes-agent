# Policy Bias Engine

## Purpose

The Policy Bias Engine turns repeated or high-signal moments into durable, auditable decision priors that affect how Hermes plans, chooses tools, and handles risk.

Policy bias is intentionally separate from:

- short-term context
- memory and facts
- persona text
- user profile storage
- skills and reusable procedures

Biases are compact, evidence-backed priors. They are not free-form reflections and they are not a long markdown memory file.

## Boundary Rules

Hermes now treats the four learning layers as distinct storage targets:

- short-term context: what matters for the current turn only
- memory / facts: durable facts about the user, environment, repo, or platform
- skills / procedures: reusable step-by-step workflows and supporting assets
- policy bias: compact action priors that tilt future decisions under uncertainty

Use this rule of thumb:

- if the artifact answers "what is true?", it belongs in memory
- if it answers "how do I do this class of task end-to-end?", it belongs in a skill
- if it answers "when unsure, which path should Hermes lean toward?", it belongs in policy bias

Examples:

- "The user prefers concise answers." can live in `USER.md`
- "Default to concise-first responses for this user." is a policy bias because it is the action rule triggered by that fact
- "How to migrate a Stripe webhook integration safely." is a skill
- "Inspect before mutating when side effects are present." is a policy bias

Every registered policy-bias candidate now carries explicit boundary metadata:

- which decision surfaces it affects
- why it is not memory
- why it is not a skill

Those boundary explanations are exposed through `hermes policy-bias inspect` and `hermes policy-bias audit`.

## Architecture

### 1. Moment Layer

Runtime events are normalized into `PolicyMoment` records and stored in SQLite. Current moment sources include:

- tool success and failure
- repeated tool failure patterns
- turn completion / failure
- user feedback and correction signals
- context compression checkpoints

Each moment carries structured fields such as task type, platform, tool path, decision class, reward, confidence, side-effect level, latency, evidence refs, and an optional `bias_candidate_key`.

### 2. Bias Layer

Durable priors are stored as `PolicyBias` objects, keyed by profile and candidate signature. Each bias tracks:

- scope
- preferred policy and optional anti-policy
- confidence, support count, average reward, and recency
- lifecycle status: `active`, `shadow`, `disabled`, `archived`
- source moment IDs
- trigger counts and last-triggered time
- version and rollback metadata

Bias candidates are only synthesized when their descriptor passes the boundary guardrail: it must declare action surfaces plus a rationale for why the artifact belongs in policy bias instead of memory or skills.

### 3. Retrieval Layer

At decision time Hermes retrieves a bounded bias set using:

- structured signals: task type, platform, tool surface
- lexical overlap
- semantic overlap
- profile isolation
- active vs shadow filtering

Retrieval is top-k bounded, profile-isolated, and scope-balanced through config. Hermes first preserves high-signal representatives across relevant scopes, then fills remaining slots by score.

### 4. Injection Layer

Retrieved active biases are applied in four places:

1. Prompt injection
   Hermes adds a dedicated `Decision Priors` block to the per-call system prompt.
2. Planner weighting
   Tool ordering is re-ranked before the model sees the tool list.
3. Tool batch ordering
   Tool-call batches can be reordered so inspect/search actions happen before mutating ones.
4. Risk behavior
   External or mutating tools can be forced into inspect, simulate, or explicit-confirm paths before execution.

Hermes now also applies deterministic response controls for relevant active biases:

- concise/directness biases can strip low-signal acknowledgements and trailing filler offers from the final answer
- structured debugging biases can enforce a findings-first heading on debugging/review style replies
- those response effects are captured in decision traces for explainability

Shadow biases are retrieved and traced, but do not influence action selection.

### 5. Synthesis Layer

Bias promotion is deterministic and score-governed:

- repeated good moments raise support and confidence
- repeated bad moments suppress or disable a bias
- one-off strong signals can land in `shadow`
- high-side-effect biases can be forced into `shadow` by config

The LLM is not used to silently mutate policy state. Numeric lifecycle rules remain system-owned.

### 6. Governance and Explainability

The subsystem keeps decision traces and bias history so operators can:

- explain which biases were retrieved, injected, or only shadowed
- inspect supporting moments
- disable, enable, archive, and rollback a bias
- export stable active biases for offline training-corpus generation
- review version history for audit and rollback

## Runtime Integration

The engine is wired into the real Hermes flow in `run_agent.py`:

- `AIAgent.__init__()` loads config and instantiates the engine
- each `run_conversation()` turn calls `begin_turn()`
- the per-call system prompt gets a `Decision Priors` block
- tool definitions are re-ranked before the API call
- tool batches are reordered when inspect-first policies apply
- risk gating runs before tool execution
- risk gating can return inspect / simulate / confirm decisions with suggested next steps
- final responses can be post-processed by deterministic response-policy hooks when active biases require concise/direct or findings-first behavior
- tool results create moments
- turn completion creates moments and triggers synthesis
- context compression creates checkpoint moments

## Lifecycle

Bias lifecycle is:

`candidate evidence -> shadow -> active -> disabled / archived`

Operationally:

- new evidence accumulates in `moments`
- candidate keys with enough support and reward synthesize into `biases`
- updates create new bias versions and append to `bias_history`
- rollback restores a prior snapshot as a new current version

## Config

Policy Bias Engine config lives under `policy_bias` in `~/.hermes/config.yaml`.

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

## Safety Properties

- Feature-flagged: the engine can be disabled globally.
- Bounded: retrieval is top-k and prompt injection is token-limited.
- Profile-isolated: moments, biases, traces, and exports are profile scoped.
- Explainable: traces and evidence are stored explicitly.
- Backward-compatible: memory, skills, sessions, and existing tool flows remain intact.
- Fail-open: if the policy-bias store fails to initialize, Hermes continues without crashing.

## Future Offline Training Path

Stable active biases can be exported as compact supervision records. The intended future pipeline is:

1. export stable active biases plus supporting moments
2. bucket by scope and candidate key
3. join with decision traces and success outcomes
4. build a high-signal corpus for offline preference / policy training
5. keep online Policy Bias Engine as the live, auditable control plane
