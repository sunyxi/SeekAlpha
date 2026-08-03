# ADR-001: stdlib-only core and walk-forward evaluator

**Date**: 2026-08-03  
**Status**: Accepted  
**Deciders**: project lead

## Context

The ORB research engine (`src/orb/core/orb_core.py`) and the walk-forward
evaluator (`scripts/wf_select.py`) must run in any Python 3.12+ environment
without a package manager step, without network access, and without version
conflicts introduced by third-party packages. The core is also the unit that
carries the strongest determinism requirement: the same input must always
produce the same output, regardless of the environment.

Third-party numerical libraries (numpy, pandas, scipy) ship with compiled
extensions whose behaviour can differ across platforms (BLAS linkage, floating-
point contraction, SIMD availability). Relying on them in the core would make
determinism harder to guarantee and audit.

## Decision

`src/orb/core/orb_core.py` and `scripts/wf_select.py` import only the Python
standard library. No third-party package may be imported, even transitively,
in these two files. This constraint is permanent and is enforced by CI.

## Consequences

**Easier**:
- Determinism is guaranteed at the language level; no BLAS or SIMD variability.
- The engine can be copied into any Python 3.12+ environment and run immediately.
- Security surface is minimal: no transitive C extensions to audit.
- Unit tests run with zero installation overhead.

**Harder**:
- Vectorised computation (e.g. rolling statistics) must be implemented in pure
  Python loops or `statistics`/`collections` from stdlib. This is acceptable
  because the engine processes one symbol at a time and performance is not the
  bottleneck at the current scale.
- Contributors familiar with numpy idioms must translate them to stdlib idioms
  when touching the core.

## Alternatives considered

- **numpy in the core**: rejected because it introduces platform-dependent
  floating-point behaviour and a mandatory install step.
- **pandas in the core**: rejected for the same reasons, plus pandas carries a
  much larger transitive dependency tree.
- **Cython/C extension**: rejected as overkill for the current scale and would
  break the "copy-and-run" property.
