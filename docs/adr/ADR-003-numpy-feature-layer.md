# ADR-003 — numpy as computation engine for the feature layer

**Status:** Accepted
**Date:** 2026-08-03
**Deciders:** project maintainers
**Related:** [ADR-001](ADR-001-stdlib-only-core.md)

---

## Context

Alpha-101 cross-sectional factors require:

- Rolling statistics (correlation, covariance, standard deviation) over panels of
  shape (T, N) where T ≈ 300–1300 and N = 31.
- Cross-sectional rank normalization per time step.
- Sliding-window views for point-in-time computation without data copies.

Implementing these correctly in pure Python would require either:
(a) O(T·N·d) nested loops — too slow for interactive research; or  
(b) Re-implementing a substantial subset of NumPy internally — a maintenance burden
    with no benefit.

ADR-001 mandates stdlib-only for `src/orb/core/` and `scripts/wf_select.py`.  No
prior ADR covers the yet-to-be-created feature layer.

## Decision

**numpy may be imported in `src/orb/features/` only.**

All other modules keep their existing dependency policy:

| Module / script | numpy allowed? |
| --- | --- |
| `src/orb/core/` | ❌ stdlib only (ADR-001) |
| `scripts/wf_select.py` | ❌ stdlib only (ADR-001) |
| `src/orb/calendar.py` | ❌ stdlib only |
| `src/orb/cache.py` | ❌ stdlib only |
| `src/orb/quality.py` | ❌ stdlib only |
| **`src/orb/features/`** | ✅ numpy allowed |
| `scripts/local_pump.py` | ❌ (imports features only lazily if needed) |

numpy is added as an **optional** dependency:

```toml
[project.optional-dependencies]
features = ["numpy>=1.24"]
test     = ["pytest", "numpy>=1.24"]
```

Runtime installs that do not need factors use `pip install -e "."` and never
import `src/orb/features/`.

## Consequences

**Positive:**
- `numpy.lib.stride_tricks.sliding_window_view` gives O(1) memory-efficient
  rolling windows without data copies (view semantics).
- Vectorised rank, correlation, and covariance are ≥ 100× faster than pure Python
  for the panel sizes used in this project.
- numpy is ubiquitous in Python quant stacks and adds no supply-chain risk.

**Negative / mitigations:**
- `pip install -e "."` (no extras) will fail to import `orb.features`; this is
  expected and intentional — callers must use `pip install -e ".[features]"`.
- Test CI installs `.[test]` which pulls numpy; this is unchanged since the
  test suite already tests feature code.
- Any accidental `import numpy` in core modules will be caught by the existing
  stdlib-only linting step (planned for M4).

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| pandas | Heavier than numpy; introduces an additional optional dep with its own ABI concerns |
| polars | New ABI, Rust dependency, overkill for single-panel factor computation |
| stdlib only (array module) | No broadcasting, no sliding windows; too slow |
| Cython / numba | Compilation step in CI; overkill for 42 factors |
