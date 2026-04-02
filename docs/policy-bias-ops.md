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
- explainability traces

With default config these are emitted through normal Hermes logging.

## Common Operator Workflows

### A bias is too aggressive

1. Run `hermes policy-bias inspect <bias_id>`
2. Run `hermes policy-bias explain --limit 5`
3. Disable it with `hermes policy-bias disable <bias_id>`
4. If needed, archive it after review

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
6. Check whether the task/tool context actually matches the bias signature

If the database is corrupted or unavailable:

- Hermes should continue operating without policy-bias influence
- repair by moving aside `~/.hermes/policy_bias.db`
- Hermes will recreate schema on next startup
- restoring from backup is safe because all state is SQLite-backed

## Safety Notes

- `risk.inspect_before_execute` can block mutating or external tools until an inspect/search step occurs.
- Shadow biases are observable but do not affect action selection.
- Rollback restores a prior snapshot as a new current version; it does not delete audit history.
