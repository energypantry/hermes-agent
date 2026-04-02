# Policy Bias Engine Operations

## Storage

- Primary database: `~/.hermes/policy_bias.db`
- Schema version table: `schema_version`
- Main tables: `moments`, `biases`, `decision_traces`, `bias_history`

## CLI Commands

Hermes exposes operational controls through `hermes policy-bias ...`.

### List / Inspect

```bash
hermes policy-bias list
hermes policy-bias list --status active --json
hermes policy-bias inspect <bias_id>
hermes policy-bias inspect <bias_id> --json
hermes policy-bias audit
hermes policy-bias audit --status active --json
```

### Enable / Disable / Archive

```bash
hermes policy-bias enable <bias_id>
hermes policy-bias disable <bias_id>
hermes policy-bias archive <bias_id>
```

### Re-score / Rebuild

```bash
hermes policy-bias rescore
hermes policy-bias rescore <bias_id>
hermes policy-bias rebuild
```

### Moments / Explainability

```bash
hermes policy-bias moments --limit 20
hermes policy-bias explain --limit 5
hermes policy-bias explain --session-id <session_id>
```

### History / Rollback / Export

```bash
hermes policy-bias history <bias_id> --limit 20
hermes policy-bias rollback <bias_id> <version>
hermes policy-bias export --min-confidence 0.55
```

## What To Watch

Useful log events:

- moment creation
- synthesis decisions
- retrieval hits
- planner/tool weighting deltas
- risk gating blocks
- response-policy effects
- explainability traces

With default config these are emitted through normal Hermes logging.

## Common Operator Workflows

### A bias is too aggressive

1. Run `hermes policy-bias inspect <bias_id>`
2. Run `hermes policy-bias explain --limit 5`
3. Disable it with `hermes policy-bias disable <bias_id>`
4. If needed, archive it after review

### A stored item looks like memory or a skill instead of a policy bias

1. Run `hermes policy-bias audit`
2. Inspect the `action_surfaces`, `why_not_memory`, and `why_not_skill` fields
3. If the classification looks wrong, disable the bias and move the content to `USER.md`, `MEMORY.md`, or a skill as appropriate
4. Rebuild or rescore after cleaning up the underlying moments if needed

### A bias regressed after recent evidence

1. Inspect history with `hermes policy-bias history <bias_id>`
2. Roll back to a prior version with `hermes policy-bias rollback <bias_id> <version>`
3. Re-run `inspect` and `explain` to verify the restored state

### Rebuild after reward or synthesis rule changes

1. Update config thresholds
2. Run `hermes policy-bias rebuild`
3. Review active/shadow counts with `list`
4. Inspect a few representative biases before leaving the rebuild in place

### Build an export set for offline training

1. Run `hermes policy-bias export`
2. Filter or post-process by scope, confidence, and support count
3. Join with decision traces and supporting moments as needed

## Debugging

If a bias does not seem to affect behavior:

1. Confirm `policy_bias.enabled: true`
2. Check `scopes_enabled`
3. Inspect whether the bias is `active` vs `shadow`
4. Review recent decision traces with `explain`
5. Check whether prompt injection was token-capped
6. Check whether response-policy effects were recorded in the trace when the bias should shape the final answer
7. Check whether the task/tool context actually matches the bias signature

If you are checking for duplicate storage across learning layers:

1. Run `hermes policy-bias audit`
2. Confirm the bias changes a decision surface rather than only restating a fact
3. If it looks like a factual preference, move it to memory
4. If it looks like a reusable playbook, move it to a skill

If the database is corrupted or unavailable:

- Hermes should continue operating without policy-bias influence
- repair by moving aside `~/.hermes/policy_bias.db`
- Hermes will recreate schema on next startup
- restoring from backup is safe because all state is SQLite-backed

## Safety Notes

- `risk.inspect_before_execute` can drive `inspect`, `simulate`, or `confirm` requirements before mutating or external tools proceed.
- Shadow biases are observable but do not affect action selection.
- Rollback restores a prior snapshot as a new current version; it does not delete audit history.
