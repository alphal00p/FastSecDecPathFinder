# FSD

`FSD` is a modular black-box sector-decomposition prototype for the triangle
and box examples, with an experimental GammaLoop `.dot` path backed by
pySecDec for Symanzik polynomial construction and sector generation.  It
separates:

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

`TopologyDefinition` also stores the general parametric metadata needed beyond
the current one-loop examples: loop count, propagator powers, the affine
dimension, the global prefactor description, and the epsilon-dependent powers
of `U` and `F`.  `SectorDefinition` stores monomial powers from `U`, `F`,
Jacobian/measure, and numerator factors separately, so endpoint powers can be
assembled generically before applying a subtraction strategy.

Do not import `scipy` or `sympy` in FSD-owned code.  pySecDec is an external
backend and may use its own symbolic stack internally.

## Documentation

The derivation and implementation notes are kept in `docs/`:

- `docs/FastSecDec.tex`: LaTeX source,
- `docs/FastSecDec.pdf`: compiled PDF.
- `docs/performance.md`: low-statistics generation/runtime measurements for
  built-in examples, DOT examples, and the double/triple box ladder probes.

## Setup

OneLOopBridge is required and external.  It is not vendored.  DOT mode also
requires pySecDec, `pydot`, and `pyyaml`.  The `geometric` pySecDec
decomposition method requires a Normaliz executable on `PATH` or supplied
through `--normaliz-executable`; FSD does not install Normaliz automatically.
The CLI default is `--sector-method iterative`, so Normaliz is only needed
when `--sector-method geometric` is requested explicitly.  The `geometric_ku`
and `iterative` methods may be used when Normaliz is not available.

Use an existing checkout:

```sh
cd FSD
ONELOOPBRIDGE_SRC=/path/to/OneLOopBridge ./install.sh
```

Or let the script clone it into ignored `.deps/`:

```sh
cd FSD
./install.sh --clone-oneloopbridge
```

The script creates `.venv`, installs `requirements.txt`, builds the
OneLOopBridge Python bindings, and verifies `import oneloop_bridge`.  On macOS,
pySecDec source builds can be sensitive to locale/toolchain settings.  If the
build fails while compiling GiNaC documentation with `LC_ALL=C.UTF-8`, retry
with a supported locale such as `LC_ALL=C LANG=C`.

## Tests

The pytest suite requires the same configured `.venv` and external
OneLOopBridge binding as normal CLI runs.  Run it from the repository root with:

```sh
.venv/bin/python -m pytest -q
```

The current tests are deterministic low-statistics smoke tests.  They cover:

- triangle massive, `C0(s;m^2)`,
- triangle massless Euclidean, `C0(s;0)`,
- box massive, `D0(0,0,0,0,s12,s23;m^2)`,
- box massless Euclidean, `D0(0,0,0,0,s12,s23;0)`,
- rejection of unsupported massless timelike triangle and box kinematics,
- DOT parsing and pySecDec sector generation for triangle, box, finite
  two-point and three-point two-loop examples, and finite two-point and
  three-point three-loop examples,
- recursive endpoint subtraction for one through four logarithmic axes,
- dual evaluator generation modes, including one envelope evaluator for mixed
  singular-axis sector sets,
- DOT/FSD integration without re-entering pySecDec after generation,
- generation timing headline buckets,
- retention of regular positive monomial factors in finite sectors.

For the four supported integrals, the tests validate sector counts and
singular-axis metadata, run one short Havana integration, and compare all
Laurent coefficients to OneLOopBridge with an MC-aware pull tolerance.

Optional generated-pySecDec numerical comparisons for small 2-loop and 3-loop
DOT examples are skipped by default because they compile external packages.
Enable them explicitly with:

```sh
FSD_RUN_PYSECDEC_COMPARE=1 .venv/bin/python -m pytest -q
```

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

GammaLoop DOT-file topology path with FSD/Havana integration:

```sh
.venv/bin/python FSD.py \
  --dot-file examples/dot/triangle.dot \
  --kinematics examples/dot/triangle_kinematics.yaml \
  --sector-method iterative \
  --dot-engine fsd
```

DOT mode uses:

- `dot_parser.py` for GammaLoop-style invisible external half-edges,
  edge ordering, unit propagator powers, and `mass` attributes,
- `kinematics.py` for YAML `values` and scalar-product `replacements`,
  evaluated with Symbolica,
- `pysecdec_bridge.py` as the only module importing pySecDec,
- the same generic `SectorProcessor` used by the hard-coded examples.

Run pySecDec’s generated integrator instead:

```sh
.venv/bin/python FSD.py \
  --dot-file examples/dot/box.dot \
  --kinematics examples/dot/box_kinematics.yaml \
  --sector-method iterative \
  --dot-engine pysecdec
```

Run both engines and compare in the same display convention:

```sh
.venv/bin/python FSD.py \
  --dot-file examples/dot/box.dot \
  --kinematics examples/dot/box_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

Smaller multi-loop DOT examples are included for the current generic path.
They are deliberately massive, Euclidean examples so that pySecDec can run with
contour deformation disabled while the package build stays modest enough for
smoke comparisons:

```sh
.venv/bin/python FSD.py \
  --dot-file examples/dot/kite_2loop.dot \
  --kinematics examples/dot/kite_2loop_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

```sh
.venv/bin/python FSD.py \
  --dot-file examples/dot/self_energy_3loop.dot \
  --kinematics examples/dot/self_energy_3loop_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

```sh
.venv/bin/python FSD.py \
  --dot-file examples/dot/three_point_2loop.dot \
  --kinematics examples/dot/three_point_2loop_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

```sh
.venv/bin/python FSD.py \
  --dot-file examples/dot/three_point_3loop.dot \
  --kinematics examples/dot/three_point_3loop_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

The progressively larger three-point examples are:

```sh
.venv/bin/python FSD.py \
  --dot-file examples/dot/three_point_2loop_6line.dot \
  --kinematics examples/dot/three_point_2loop_6line_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

```sh
.venv/bin/python FSD.py \
  --dot-file examples/dot/three_point_3loop_8line.dot \
  --kinematics examples/dot/three_point_3loop_8line_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

The DOT/YAML schema used by the examples is:

```yaml
values:
  s12: -1.0
  s23: -1.0
  mt: 0.0
replacements:
  p1*p1: 0
  p2*p2: 0
  p1*p2: s12/2
```

DOT edge masses are read from `mass`.  Symbolic masses must appear in
`values`; numeric masses can be written directly.  No PDG-to-mass inference is
performed.  In DOT mode these mass symbols are resolved to the numeric YAML
values before pySecDec generation, so massless and massive sector structures
are not confused.  `ext -> v` half-edges are incoming and `v -> ext`
half-edges are outgoing; pySecDec receives bare external momentum symbols and
the sign convention is encoded by the scalar-product replacements.

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

Choose how topology-level dualized U/F evaluators are generated:

```sh
.venv/bin/python FSD.py --s -1.0 --m 0 --pregenerate-dual-evaluators
.venv/bin/python FSD.py --s -1.0 --m 0 --lazy-dual-evaluators-generation
.venv/bin/python FSD.py --s -1.0 --m 0 --pregenerate-single-overall-dual-evaluator
.venv/bin/python FSD.py --s -1.0 --m 0 --symbolic-derivatives
```

`--pregenerate-dual-evaluators` is the default and builds one dualized U/F
evaluator per unique sector dual shape before integration.  Lazy mode keeps
the current cache-on-first-use behavior and reports first-use dualization time
as `TaylorGen`, separate from `EvalT`.  Single-overall mode builds one padded
envelope evaluator per integration dimension and remaps its Taylor columns
back to the sector-native shape; this is useful for proving that runtime
evaluation can be made generation-free even when sectors ask for different
dual shapes.  `--symbolic-derivatives` builds ordinary non-dual Symbolica
evaluators for symbolic U/F partial derivatives with respect to the original
Feynman parameters and then composes them with sector-map Taylor jets by an
explicit chain rule.  The derivative evaluators are shared across sectors.

Choose the endpoint-subtraction backend:

```sh
.venv/bin/python FSD.py --s -1.0 --m 0 --subtraction-backend recursive
.venv/bin/python FSD.py --s -1.0 --m 0 --subtraction-backend formula
.venv/bin/python FSD.py --s -1.0 --m 0 --subtraction-backend projector-formula
```

`recursive` uses the vectorized Python/Numpy localized Taylor subtraction
sum.  `formula` builds full Symbolica subtraction evaluators whose signatures
include the sector-specific U/F/J monomial and Taylor-coefficient layout.
`projector-formula` builds lower-signature Symbolica endpoint projectors keyed
only by endpoint powers, Taylor orders, and Laurent range; sector-specific
regular coefficients are still obtained from the black-box U/F/J Taylor path
and then passed into the shared projector evaluator.

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

Control the integrated Laurent range and comparison target:

```sh
.venv/bin/python FSD.py --s -1.0 --m 0 --max-eps-order -1 \
  --target -1.0 0.0 0.0 0.0
```

The deepest order is always `eps^(-2*loop_count)`.  `--max-eps-order` selects
the highest order included in the Monte Carlo accumulators.  Numeric
`--target` entries are real/imaginary pairs ordered from the deepest pole
upward; unspecified trailing coefficients are set to zero.  `--target
result.json` reads a previous result file in the same displayed prefactor
convention.  In DOT mode, `--target pysecdec` first runs pySecDec using the
configured pySecDec controls and uses that result as the live comparison.

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
- evaluator parameter order, U/F Taylor shapes, Taylor evaluator mode, and
  Taylor evaluator build time,
- DOT generation timing buckets when applicable:
  `Generation U and F polynomial`, `Generating sectors`, and
  `Generating Symbolica evaluators`,
- all sector maps, regular sector prefactors, U/F monomials, endpoint powers,
  singular axes, and subtraction type,
- DOT sector statistics when applicable, including total sector count, capped
  sector display (`showing 20/N sectors`), singular-axis distribution, U/F
  monomial tuple counts, endpoint pole depth, max dimension, and max pole
  order,
- validation and benchmark availability.

Use `--quiet-summary` to suppress this summary.  Use `--json` for JSON output;
JSON output includes the same summary data fields but suppresses both the
summary table and progress bar.

Every completed run writes a pretty `result.json` atomically.  Built-in
triangle/box runs write to the current working directory; DOT runs write next
to the DOT file.  Use `--result-path PATH` to override this, which is useful
when a previous result file is also being used as `--target`.  The file
contains request/config metadata, input kinematics, topology and sector
summaries, generation/runtime timings, target metadata, aggregate Laurent
coefficients, and additive per-sector Laurent coefficients.
Inspect a stored result without regenerating anything with:

```sh
.venv/bin/python FSD.py --show-results result.json \
  --sort-sector-results abs-error
```

Sector rows can be sorted by `index`, `abs-central`, or `abs-error`.

During integration, `progressbar2` reports compact coloured labels:

- `it`: current iteration, with a sample-based progress bar that advances during an
  iteration as batches finish,
- `smpl`: accumulated sample count against `max_iter * samples_per_iter`, shown
  with 3-significant-digit `K`, `M`, or `B` units when useful,
- `err%`: live relative MC error in percent, computed as the sum of absolute MC errors
  over all Laurent coefficients divided by the sum of absolute central values,
  optionally followed by the blue target value,
- `pull`: live pull maxed over Laurent coefficients, or `N/A` when no target is available,
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
percent.  Explicit `--target` values override all built-in references.
Without `--target`, built-in triangle/box runs use OneLOopBridge, while
DOT/FSD-only runs report `N/A` unless `--dot-engine both` or `--target
pysecdec` is used.
The timing footer reports total Symbolica evaluator time `EvalT`, measured
Python hot-path time `PythonT`, Havana time `HavanaT`, Taylor evaluator setup
time `TaylorGen`, and the corresponding profile percentages.

## Current Scope

Supported examples:

- `C0(s;m^2)`, massive finite, with `s < 4 m^2`,
- `C0(s;0)`, massless endpoint-subtracted, with Euclidean `s < 0`,
- `D0(0,0,0,0,s12,s23;m^2)`, massive finite, with `s12,s23 < 4 m^2`,
- `D0(0,0,0,0,s12,s23;0)`, massless endpoint-subtracted, with Euclidean
  `s12 < 0` and `s23 < 0`.

Non-Euclidean massless kinematics are intentionally rejected in the current
prototype.

Experimental DOT scope:

- scalar Euclidean topologies only,
- unit propagator powers only,
- no tensor numerator support in the FSD processor,
- endpoint powers must be negative integers regulated by epsilon.  The generic
  processor applies localized Taylor subtraction, with logarithmic plus
  distributions as the `N=0` special case,
- positive monomial powers factored by pySecDec are treated as regular
  multiplicative factors in `g_s`,
- no contour deformation in FSD DOT mode,
- pySecDec global prefactors are currently convolved only when regular at
  epsilon zero; examples with global Gamma prefactor poles are intentionally
  not part of the validated DOT set yet,
- pySecDec may be run separately with `--dot-engine pysecdec` or compared with
  `--dot-engine both`,
- FSD uses pySecDec decomposition data to build declarative sectors, but the
  hot-path processor still treats U and F as black-box Symbolica evaluators,
- in FSD DOT mode, pySecDec is used before integration only.  Prepared
  `TopologyDefinition` and `SectorDefinition` objects are inherited by workers
  through a fork context; if fork is unavailable, multi-worker DOT integration
  fails clearly instead of regenerating sectors at runtime.
