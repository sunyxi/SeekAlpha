# Data Source Policy

## Decision

Databento is the provisional primary source and Alpaca is the fallback. The
decision is `selected_with_blockers`, not an authorization to download data.
The source of truth is `src/orb/data_source_policy.json`.

## CLI Usage

```bash
python3 scripts/validate_data_source_policy.py
```

## Operations

Before any bulk download, verify the provider contract, non-display research
rights, retention rights, corporate-action coverage, and historical sector
coverage. Read credentials only from environment variables. Store hashes and
provider snapshot IDs in manifests; never commit raw data or credentials.

Rebuild membership by `instrument_id` and half-open listing intervals. A ticker
is only a date-scoped alias. Include delisted instruments when valid at the
research date, and never substitute current constituents for historical members.

## Limitations

Databento sector-history entitlement and final contract terms are not configured
in this repository. Alpaca is a fallback for bars/quotes/corporate actions, not
a proof of point-in-time membership. No data download or provider authentication
is performed by this issue.

## Rollback

Before any download, correct the policy in a new branch and update its ADR. After
data is acquired, do not rewrite a manifest or reinterpret membership; create a
new policy ID and invalidate affected experiments. Delete no credentials from
Git history as a substitute for secret-rotation procedures.
