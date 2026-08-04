# ADR-006: Optional Microsoft Qlib research adapter

**Date**: 2026-08-04
**Status**: Accepted
**Deciders**: project maintainers

## Context

SeekAlpha needs a controlled way to evaluate Microsoft Qlib factors and models
against its existing cached market data. The ORB core and walk-forward evaluator
are frozen, standard-library-only paths. Qlib also has a broad dependency tree
and is not an execution or broker integration layer.

## Decision

Add `pyqlib==0.9.7`, pandas, and numpy in a separate `qlib` optional dependency
group. Keep all integration code under `src/orb/qlib_adapter/`, use lazy runtime
imports, and export a deterministic `(datetime, instrument)` DataFrame from the
existing `DailyPanel` without downloading Qlib sample data.

The initial POC is limited to data conversion and runtime compatibility. Qlib
must not be imported directly or transitively by `src/orb/core/orb_core.py` or
`scripts/wf_select.py`. Any factor or model experiment remains subject to the
training-window, purge, embargo, frozen-search-space, and paired-comparison rules
in `AGENT.md`.

## Consequences

- Researchers can install and validate Qlib independently with `.[qlib]`.
- Normal ORB simulation and walk-forward evaluation do not install Qlib.
- Existing cache files remain the source of truth and are never rewritten.
- The exported CSV is an interchange artifact, not a production Qlib binary
  store and not evidence of strategy improvement.
- `pyqlib` and Qlib are MIT licensed; the POC does not copy Qlib source code.

## Alternatives considered

- **Replace the existing research engine with Qlib**: rejected because it would
  invalidate frozen behavior and couple execution assumptions to a research
  framework.
- **Use Qlib sample data**: rejected because data provenance and consistency
  with existing ORB results would be lost.
- **Add Qlib to the default or test runtime**: rejected because the package is
  optional and its dependency cost should not affect the core path.
- **Export directly to Qlib binary storage**: deferred until the tabular
  compatibility boundary and data semantics are independently reviewed.
