# Strategy Contract

## Purpose

Register a falsifiable strategy hypothesis before coding and bind its decision
report to the exact specification, research protocol, data snapshot, and code
revision. The field source of truth is
`src/orb/strategy_spec_template.json`.

## CLI Usage

```bash
python3 scripts/validate_strategy_contract.py \
  --spec path/to/spec.json \
  --report path/to/report.json \
  --summary-output reports/strategy-summary.md
```

The validator accepts only `Candidate`, `No-Go`, `Exploratory`, or `Invalid`.
The output summary is create-only and must be regenerated from JSON when the
source changes.

## Operations

1. Fill the specification before implementing a strategy.
2. Freeze the universe, features, label timing, holding period, costs,
   parameter search space, protocol hash, data manifest hash, and budget.
3. Compute the spec hash and include it in every decision report.
4. Validate the report before publishing it; retain JSON and Markdown together.
5. Record gate statuses as `passed`, `failed`, `not-run`, or `skipped`.

## Limitations

This contract does not evaluate profitability, download data, consume retention,
or run a model. It verifies declarations and provenance only. A code commit is
required as a non-empty identifier but is not fetched or verified by the local
validator.

## Rollback

Before results exist, revert the feature branch and correct the specification.
After a report is published, do not overwrite it. Create a new `spec_id` or
experiment report, document the invalidation, and render a new summary.
