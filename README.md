# FSD_v2

`FSD_v2` is a modular rewrite of the `FastSecDec/FSD` prototype.  It keeps the
same triangle and box examples, but separates:

- declarative sector generation,
- black-box U/F topology evaluators,
- generic sector processing,
- Havana integration,
- formatting and benchmarking.

The purpose is to demonstrate that sector-decomposed integrands can be built
from sector metadata plus numerical U/F evaluators.  U and F Symbolica
expressions are retained in `TopologyDefinition` for display and evaluator
construction, but `SectorProcessor` only calls evaluators and never
symbolically manipulates U or F.

Do not import `scipy` or `sympy` in this project.

## Documentation

The derivation and implementation notes are kept in `docs/`:

- `docs/FastSecDec.tex`: LaTeX source,
- `docs/FastSecDec.pdf`: compiled PDF.

## Setup

OneLOopBridge is required and external.  It is not vendored.

Use an existing checkout:

```sh
cd FSD_v2
ONELOOPBRIDGE_SRC=/path/to/OneLOopBridge ./install.sh
```

Or let the script clone it into ignored `.deps/`:

```sh
cd FSD_v2
./install.sh --clone-oneloopbridge
```

The script creates `.venv`, installs `requirements.txt`, builds the
OneLOopBridge Python bindings, and verifies `import oneloop_bridge`.

## Tests

The pytest suite requires the same configured `.venv` and external
OneLOopBridge binding as normal CLI runs.  Run it from `FSD_v2` with:

```sh
.venv/bin/python -m pytest -q
```

The current tests are deterministic low-statistics smoke tests.  They cover:

- triangle massive, `C0(s;m^2)`,
- triangle massless Euclidean, `C0(s;0)`,
- box massive, `D0(0,0,0,0,s12,s23;m^2)`,
- box massless Euclidean, `D0(0,0,0,0,s12,s23;0)`,
- rejection of unsupported massless timelike triangle and box kinematics.

For the four supported integrals, the tests validate sector counts and
singular-axis metadata, run one short Havana integration, and compare all
Laurent coefficients to OneLOopBridge with an MC-aware pull tolerance.

## Usage

Triangle massive:

```sh
.venv/bin/python FSD.py --s 1.0 --m 1.0
```

Triangle massless:

```sh
.venv/bin/python FSD.py --s -1.0 --m 0.0
```

Box massive:

```sh
.venv/bin/python FSD.py --integral box --s12 0.5 --s23 0.7 --m 1.0
```

Box massless:

```sh
.venv/bin/python FSD.py --integral box --s12 -1.0 --s23 -2.0 --m 0.0
```

Display the Feynman-normalized prefactor convention:

```sh
.venv/bin/python FSD.py --s 1.0 --m 1.0 --prefactor-convention feynman
```

Cap vectorized processor task size:

```sh
.venv/bin/python FSD.py --s 1.0 --m 1.0 --batch-size 10000
```

The default `--batch-size 0` keeps the previous behavior: each Havana
iteration is split only by worker, then grouped by sector inside that worker
chunk.  With a positive batch size, each worker chunk is further split into
tasks of at most that many Monte Carlo samples.

Stop when the summed relative MC error reaches a percent target:

```sh
.venv/bin/python FSD.py --s -1.0 --m 0 --target-rel-accuracy 0.03
```

The target is in percent and applies to the same quantity shown as `err%`.
When this option is enabled, the progress percentage and ETA use the current
error estimate extrapolated with `err ~ 1/sqrt(N)`.  If a finite `--max-iter`
sample budget would be reached first, the ETA and progress bar use that shorter
completion time instead.

Use an unbounded run:

```sh
.venv/bin/python FSD.py --s -1.0 --m 0 --max-iter -1 --target-rel-accuracy 0.03
```

With `--max-iter -1` and no target relative accuracy, the run is deliberately
unbounded and must be interrupted by the user.

Enable Symbolica evaluator JIT compilation experimentally:

```sh
.venv/bin/python FSD.py --s 1.0 --m 1.0 --jit-compile-evaluators
```

This is disabled by default because the currently tested Symbolica batch JIT
path can mis-evaluate simple row-wise expressions.  The standalone
`jit_compile_MRE.py` script demonstrates the issue:

```sh
.venv/bin/python jit_compile_MRE.py
```

## Output

Before integration, non-JSON runs print a coloured structured summary:

- run configuration and Havana settings,
- retained U/F polynomials,
- evaluator parameter order and F dual shapes,
- all sector maps, regular Jacobians, monomials, singular axes, and subtraction
  type,
- validation and benchmark availability.

Use `--quiet-summary` to suppress this summary.  Use `--json` for JSON output;
JSON output includes the same summary data fields but suppresses both the
summary table and progress bar.

During integration, `progressbar2` reports compact coloured labels:

- `it`: current iteration, with a sample-based progress bar that advances during an
  iteration as batches finish,
- `smpl`: accumulated sample count against `max_iter * samples_per_iter`, shown
  with 3-significant-digit `K`, `M`, or `B` units when useful,
- `err%`: live relative MC error in percent, computed as the sum of absolute MC errors
  over all Laurent coefficients divided by the sum of absolute central values,
  optionally followed by the blue target value,
- `pull`: live pull maxed over Laurent coefficients,
- `t`: total elapsed wall time,
- `eta`: ETA to the sample budget, or to the target relative accuracy when
  `--target-rel-accuracy` is enabled,
- `eval μs/smpl/wkr`: average evaluator time per sample per worker in `μs`;
  `EvalT` is worker-summed, so this is normalized by the total sample count,
- `prof py|eval|hav`: live profile `(python | evaluator | havana)`, where Havana includes grid
  sampling, cloned-grid training accumulation, merge, and update time.

When `--target-rel-accuracy` is enabled, the target check is performed after
each accumulated batch, not only at full iteration boundaries.  The final result
includes all completed batches up to the stopping point, and `--batch-size`
therefore controls the mid-iteration stopping granularity.

The final table reports the selected prefactor convention only.  Values with
Monte Carlo uncertainty use parenthesis notation with two significant error
digits.  The `MC err` column reports the relative one-sigma MC error in
percent.  OneLOopBridge benchmark values are always computed and compared.
The timing footer reports total Symbolica evaluator time `EvalT`, measured
Python hot-path time `PythonT`, Havana time `HavanaT`, and the corresponding
profile percentages.

## Current Scope

Supported examples:

- `C0(s;m^2)`, massive finite, with `s < 4 m^2`,
- `C0(s;0)`, massless endpoint-subtracted, with Euclidean `s < 0`,
- `D0(0,0,0,0,s12,s23;m^2)`, massive finite, with `s12,s23 < 4 m^2`,
- `D0(0,0,0,0,s12,s23;0)`, massless endpoint-subtracted, with Euclidean
  `s12 < 0` and `s23 < 0`.

Timelike and threshold massless kinematics are intentionally rejected because
the prototype has no contour deformation or threshold regularization.
