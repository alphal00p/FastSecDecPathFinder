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
- `docs/validation.md`: current target comparisons and endpoint-stability
  probes.
- `docs/universal_cache.md`: formula-cache warmup command, 1L/2L verification
  timings, and the current 3L cache-generation estimate.
- `cache_generation_project.md`: cluster handoff for generating and packaging
  the distributable universal formula cache.

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

Large DOT runs can reuse a precomputed formula cache.  The cache is not tracked
in this repository; install it from a local archive or a hosted URL:

```sh
./install.sh --cache-tar /path/to/FSD_cache.tar.gz
./install.sh --cache-url https://example.invalid/FSD_cache.tar.gz
```

The same inputs may be supplied with `FSD_CACHE_TARBALL` or `FSD_CACHE_URL`.
Archives may contain either `cache/` or `subtraction_formulae/`; both are
installed under the top-level `cache/` directory.  Cold generation falls back to
building any missing formula and writes it into `cache/subtraction_formulae`.

To explicitly warm and verify the topology-independent projector cache on the
shipped DOT examples, use the `cache` subcommand:

```sh
.venv/bin/python FSD.py cache \
  --cache-loop-counts 1 2 \
  --cache-verify-samples-per-sector 8 \
  --cache-report-path docs/universal_cache_report.json
```

This covers endpoint-projector, regular Taylor, and chain-rule formula assets,
then runs a low-stat democratic check over each selected topology.  The detailed
report is documented in `docs/universal_cache.md`.

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

Reusable run presets are available under `examples/runs`.  YAML option keys
match long CLI option names without the leading `--`, and explicit CLI flags
override the YAML values:

```sh
.venv/bin/python FSD.py --run examples/runs/dot_box.yaml --max-iter 1
```

The corresponding DOT graphs and kinematics live in `examples/graphs`, while
persistent targets and run outputs live in `examples/outputs`.  The preset set
also includes the off-shell Euclidean triple box,
`examples/runs/dot_triple_box_offshell.yaml`, as the no-threshold three-loop
triple-box case to keep in future performance and stability sweeps.

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
  --dot-file examples/graphs/triangle.dot \
  --kinematics examples/graphs/triangle_kinematics.yaml \
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

For heavier DOT topologies, prefer the two-stage prepared-bundle workflow:

```sh
.venv/bin/python FSD.py generate \
  --dot-file examples/graphs/triple_box.dot \
  --kinematics examples/graphs/triple_box_kinematics.yaml \
  --sector-method iterative \
  --subtraction-backend projector-formula \
  --ibp-reduce-to-log-endpoint \
  --direct-projector-cache-term-threshold 0 \
  --symbolic-derivatives \
  --chain-rule-formula-signature-limit 4096 \
  --chain-rule-formula-output-length-limit 288 \
  --max-eps-order 0 \
  --output examples/outputs/prepared_triple_box

.venv/bin/python FSD.py integrate \
  --output examples/outputs/prepared_triple_box \
  --samples-per-iter 100000 \
  --batch-size 10000 \
  --workers 10
```

`generate` is DOT-only.  It writes `manifest.json`, topology/sector metadata,
reference Symbolica expression strings, generation timings, and serialized
Symbolica evaluator bytes under the selected `--output` directory.  Hot
JIT/compiled evaluator artifacts are stored together with an eager evaluator
fallback, because arbitrary-precision endpoint rescue uses Symbolica's eager
`evaluate_with_prec`/`evaluate_complex_with_prec` APIs.  `integrate` is strict:
it loads evaluator bytes from that bundle and does not call pySecDec, rebuild
Symanzik polynomials, or generate Symbolica formula/evaluator artifacts.  In
other words, the integration process is a disk-only consumer of the prepared
bundle, apart from lazy loading the serialized evaluator files.
The default result path for `integrate` is
`<output>/result.json`; use `--result-path` to override it.  Evaluator artifacts
are loaded lazily through an LRU cache controlled by `--evaluator-lru-size`
(`0` means unlimited).  The current prepared loader uses the Laurent range
stored in the bundle by default.  An explicit `--max-eps-order` in
`integrate` may request any lower or equal maximum order; for example a bundle
prepared through `eps^0` can be reused for a leading-pole result view through
`eps^-1`.  The current strict loader still evaluates the prepared formula
artifacts at their stored range and trims the displayed/persisted coefficients;
it does not rebuild smaller formulas.  Requesting an order above the prepared
range fails strictly.

Two sector-evaluator comparison backends are available for DOT prepared
bundles:

- `--sector-evaluator-backend two-stage-explicit` keeps FSD's derivative-source
  evaluator and endpoint assembler as two prepared Symbolica calls per
  singular sector.
- `--explicit` is shorthand for `--sector-evaluator-backend explicit`; it
  builds one fully substituted multi-output Symbolica evaluator per sector,
  closer to the pySecDec generation/runtime trade-off.  It is the default
  runtime backend for DOT runs.
- `--projector-generation` selects the fast-generation black-box projector path
  where U/F are never opened by the sector processor.  This remains the
  conceptual FSD path, but it is slower per sample than explicit evaluators.

Current 3-loop triple-box status: a full `eps^-6..eps^0` prepared bundle was
generated under a 30 GiB process-tree memory cap.  The bundle contains 1972
sectors and 22996 serialized evaluator artifacts, occupies about 4.1 GiB on
disk, and recorded 0.198 s for U/F construction, 1.191 s for sector
generation, 288.842 s for Symbolica evaluator preparation, plus 457.193 s for
evaluator serialization.  A one-point all-sector diagnostic completed 1746
sectors, hit the 30 s classification cap in 226 sectors, and triggered no
precision rescue in completed sectors.  This is a subtraction/stability and
runtime diagnostic rather than a precision validation.

For sector-coverage diagnostics, use democratic sampling instead of the
default Havana sector sampling:

```sh
.venv/bin/python FSD.py integrate \
  --output examples/outputs/prepared_triple_box \
  --sampling-mode democratic \
  --democratic-samples-per-sector 1000 \
  --batch-size 1000 \
  --workers 10
```

This forces the same number of uniform points in every selected sector and
records per-sector sample counts, timing, precision-rescue fractions, and
maximum observed weights in `result.json`.  It is intended as a diagnostic
mode; the default adaptive mode remains the production integration path.

QMC integration is also available through QMCPy's randomized shifted rank-1
lattices with the Korobov periodizing transform used in the pySecDec/QMC
literature.  This path is independent of pySecDec's QMC internals; pySecDec is
only used for DOT sector finding or as an external generated-integrator
baseline.  In this mode `--samples-per-iter` is the number of lattice points
per sector and per random shift, while `--qmc-shifts` is the number of
independent shifts used for the one-sigma error estimate.  The lattice point
count must be a power of two for the current QMCPy backend:

```sh
.venv/bin/python FSD.py \
  --run examples/runs/dot_box.yaml \
  --sampling-mode qmc \
  --qmc-shifts 16 \
  --qmc-korobov-alpha 3 \
  --samples-per-iter 8192 \
  --batch-size 8192 \
  --workers 10
```

For a direct one-loop comparison against pySecDec's generated QMC backend, use:

```sh
.venv/bin/python scripts/compare_qmc_pysecdec.py \
  --run-file examples/runs/dot_box.yaml \
  --kinematics-file examples/graphs/box_kinematics.yaml \
  --target-source oneloop-sector \
  --oneloop-integral box \
  --fsd-prefactor-convention sector \
  --pysecdec-shared .pysecdec_build/fsd_psd_box/fsd_psd_box_pylink.so \
  --sample-counts 1024 4096 \
  --qmc-shifts 16 \
  --workers 10
```

With `--target-source oneloop-sector`, the comparison target is taken from
OneLOopBridge at the matching built-in triangle/box kinematics and converted to
the DOT sector convention.  The script prints both the public pySecDec QMC
budget and FSD's raw sector-sample count, because pySecDec does not expose the
same per-sector accounting through the pylink API.

The transform is applied in vectorized NumPy before the existing batched
Symbolica sector evaluator call.  The estimator treats each shifted lattice as
one sector estimate and combines sector errors in quadrature; deterministic
lattice points are not treated as independent Monte Carlo samples.

Run pySecDec’s generated integrator instead:

```sh
.venv/bin/python FSD.py \
  --dot-file examples/graphs/box.dot \
  --kinematics examples/graphs/box_kinematics.yaml \
  --sector-method iterative \
  --dot-engine pysecdec
```

Run both engines and compare in the same display convention:

```sh
.venv/bin/python FSD.py \
  --dot-file examples/graphs/box.dot \
  --kinematics examples/graphs/box_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

Smaller multi-loop DOT examples are included for the current generic path.
They are deliberately massive, Euclidean examples so that pySecDec can run with
contour deformation disabled while the package build stays modest enough for
smoke comparisons:

```sh
.venv/bin/python FSD.py \
  --dot-file examples/graphs/kite_2loop.dot \
  --kinematics examples/graphs/kite_2loop_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

```sh
.venv/bin/python FSD.py \
  --dot-file examples/graphs/self_energy_3loop.dot \
  --kinematics examples/graphs/self_energy_3loop_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

```sh
.venv/bin/python FSD.py \
  --dot-file examples/graphs/three_point_2loop.dot \
  --kinematics examples/graphs/three_point_2loop_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

```sh
.venv/bin/python FSD.py \
  --dot-file examples/graphs/three_point_3loop.dot \
  --kinematics examples/graphs/three_point_3loop_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

The progressively larger three-point examples are:

```sh
.venv/bin/python FSD.py \
  --dot-file examples/graphs/three_point_2loop_6line.dot \
  --kinematics examples/graphs/three_point_2loop_6line_kinematics.yaml \
  --sector-method iterative \
  --dot-engine both
```

```sh
.venv/bin/python FSD.py \
  --dot-file examples/graphs/three_point_3loop_8line.dot \
  --kinematics examples/graphs/three_point_3loop_8line_kinematics.yaml \
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
For selected hard DOT sectors, FSD can pregenerate Symbolica chain-rule
composition formulas as well; their build time is reported as `ChainGen` and
large all-sector request sets are guarded to avoid impractical generation.

The default endpoint-subtraction backend is now `projector-formula`, which is
the black-box path used by the DOT examples.  The older full-formula and
recursive backends remain useful diagnostics:

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

For sectors with higher endpoint powers, enable the IBP-lowered projector path:

```sh
.venv/bin/python FSD.py \
  --run examples/runs/dot_triple_box.yaml \
  --IBP_reduce_to_log_endpoint
```

The legacy flag above is equivalent to `--ibp-power-goal -1`, which lowers
higher endpoint powers to logarithmic `y^(-1+c eps)` projectors.  A numeric
goal can stop the lowering earlier; for example `--ibp-power-goal -3` lowers
only until every endpoint base power is at least `-3`, then applies the usual
projector subtraction to the remaining endpoint powers.  This is useful when
the fully logarithmic IBP tree is more complex than the residual projector.
Goals greater than `-1` are rejected for now; omit the option, or pass
`--no-ibp-reduce-to-log-endpoint`, to disable IBP lowering.

Run files can be overridden from the CLI.  The triple-box preset enables IBP
lowering, but a selected-sector comparison can disable it explicitly:

```sh
.venv/bin/python FSD.py \
  --run examples/runs/dot_triple_box.yaml \
  --no-ibp-reduce-to-log-endpoint \
  --sectors 62
```

The IBP mode lowers endpoints such as `y^(-2+c eps)` and `y^(-3+c eps)` to
logarithmic child projectors plus boundary and derivative terms.  Independently
of that toggle, the top-level `cache/subtraction_formulae` directory stores
universal endpoint-projector, regular-Taylor, and chain-rule formula assets:
reference Symbolica expression metadata where practical and serialized
Symbolica evaluator bytes for fast reuse.  The older
`assets/subtraction_formulae` location is still searched as a read-only
fallback for legacy and curated assets, but new cold-cache generation writes to
`cache/subtraction_formulae`.
When IBP lowering would produce a large compound tree, FSD automatically uses a
curated direct endpoint projector instead if one is shipped for that exact
universal signature.  The default switch point is controlled by:

```sh
--direct-projector-cache-term-threshold 54
```

Set the threshold to `0` to always prefer the IBP compound projector.  The
summary reports how many sectors and endpoint signatures used the direct
curated override.

The curated endpoint-projector set currently contains 488 universal formulae,
about 49 MB.  They are small enough to be treated as shipped FSD code, so a
matching sector uses the direct formula by default rather than requiring a
warm-cache or opt-in mode.  On the triple-box leading-pole smoke this switches
all depth-six endpoint sectors to the direct endpoint-projector path.

The first curated regular-Taylor signatures are the six small `PSD213`-class
signatures that showed evaluator-dominated runtime; they add only about
100 KiB.  Exploratory JSON cache files in the cache root are ignored by git by
default because the full triple-box regular-Taylor cache can be hundreds of MB
and should only be promoted after a runtime comparison justifies it.  The
loader prefers a curated copy over a local generated copy with the same
signature.  The regular-Taylor layer is guarded by
`--regular-taylor-signature-limit`,
`--regular-taylor-formula-volume-limit`, and
`--regular-taylor-formula-axis-limit`.  After the Symbolica dev dualization
fix, U/F dual evaluator preparation is no longer the main blocker for six-axis
Taylor boxes.  The cold regular-Taylor formula itself can still be expensive:
some triple-box signatures spend minutes in formula extraction or in
dualizing the generated regular expression.  The caps therefore remain useful
for fallback and memory studies, and lowering them deliberately leaves
high-axis or large Taylor-box signatures on the Python fallback path.  A
curated signature also bypasses the cold-build signature-count cap because it
is treated as already generated source data rather than new work for the
current run.  The additional
`--chain-rule-formula-signature-limit` guard controls the Symbolica
chain-rule composition formulas used by the symbolic-derivative path.  These
signatures are topology-independent: they are keyed by active-coordinate
count, sector-variable rank, and requested Taylor shape.  The additional
`--chain-rule-formula-output-length-limit` leaves very large cold signatures on
the strict Python sparse Taylor fallback while still reusing any matching
cached evaluator artifacts.

For cache-warming experiments, the guarded path can be disabled explicitly:

```sh
.venv/bin/python FSD.py \
  --run examples/runs/dot_triple_box.yaml \
  --sectors 62 \
  --max-eps-order -5 \
  --force-regular-taylor-formulas
```

This is intentionally not the default.  The resulting regular-Taylor formulae
are universal for their endpoint/Taylor signature and are stored under
`cache/subtraction_formulae`, so a cold build can be expensive once while a
warm-cache run may be useful for runtime studies.  The generated cache can be
packaged as `FSD_cache.tar.gz` and installed later with `install.sh`; the
generation summary reports how many formulae were loaded from cache and how
many had to be generated.  The lower-level cap
flags remain available for partial comparisons such as “six axes, but only
small Taylor volume”.

To inspect generated versus curated formula assets:

```sh
scripts/inspect_subtraction_cache.py
```

To promote a validated generated formula into the curated source asset set:

```sh
scripts/promote_subtraction_formula_asset.py regular_taylor_<hash>.json
```

Use `--dry-run` first when checking large assets.  Promotion is deliberately
explicit because curated regular-Taylor formulae become default behavior for the
matching universal signature.

For exploratory long runs, especially Symbolica dualization probes that may
not react to `Ctrl-C`, launch FSD through the local watchdog wrapper:

```sh
./run_with_memory_watch.py \
  --limit-gb 30 \
  --timeout-seconds 600 \
  --poll-seconds 0.5 \
  -- .venv/bin/python FSD.py --run examples/runs/dot_triple_box.yaml
```

The wrapper owns the child process group and can interrupt it on timeout or
memory limit without a separate external `kill` command.  It prefers `psutil`
for process-tree RSS accounting, so it can run inside the normal sandbox once
the requirements are installed.  To stop it manually, create the watched stop
file from the same directory:

```sh
touch stop.order
```

The wrapper removes stale stop files at startup and removes the observed stop
file before terminating the child group.  In restricted sandboxes RSS polling
may be unavailable; the wall-time timeout and stop-file path still work.

Near endpoint points can be promoted to multiprecision evaluator calls:

```sh
.venv/bin/python FSD.py \
  --run examples/runs/dot_double_box.yaml \
  --stability-threshold 1e-3 \
  --medium-precision-stability-threshold 1e-6 \
  --high-precision-stability-threshold 1e-8
```

The defaults are `1e-3` at 32 digits, `1e-6` at 100 digits, and `1e-8` at
1000 digits.  The result JSON records global and per-sector precision-tier hit
counts.

Stop when the summed relative MC error reaches a percent target:

```sh
.venv/bin/python FSD.py --s -1.0 --m 0 --target-rel-accuracy 0.03
```

The target is in percent and applies to the same quantity shown as `err%`.
When this option is enabled, the progress percentage and ETA use the current
error estimate extrapolated with `err ~ 1/sqrt(N)`.  If a finite `--max-iter`
sample budget would be reached first, the ETA and progress bar use that shorter
completion time instead.

The same stopping quantity can also be specified as a dimensionless ratio, and
an absolute summed-error target can be requested in the selected prefactor
convention:

```sh
.venv/bin/python FSD.py --s -1.0 --m 0 --target-rel-error 3e-4
.venv/bin/python FSD.py --s -1.0 --m 0 --target-abs-error 1e-8
```

To request a wall-time budget, use:

```sh
.venv/bin/python FSD.py --s -1.0 --m 0 --target-integration-time 600
```

For Havana and QMC, FSD first performs a short same-worker-count warm-up and
then adjusts the effective sample statistics to get close to the requested
wall time.  The elapsed-time stop remains active as a guard and the warm-up
tuning is recorded in `result.json`.

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

Choose the Symbolica evaluator backend:

```sh
.venv/bin/python FSD.py --s 1.0 --m 1.0 --jit-compile
.venv/bin/python FSD.py --s 1.0 --m 1.0 --eager-evaluator
.venv/bin/python FSD.py --run examples/runs/dot_triangle.yaml --compile
```

`--jit-compile --real-evaluator` is the default.  `--eager-evaluator` disables
JIT and keeps plain eager evaluators.  `--compile` builds a shared-library f64
hot path where Symbolica supports it; FSD still stores the eager evaluator next
to the compiled artifact so precision rescue remains available when strict
prepared bundles are loaded later.  `--complex-evaluator` can be used to force
complex-valued f64 evaluator calls even for real kinematics.

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
  `--target-rel-accuracy`, `--target-rel-error`, `--target-abs-error`, or
  `--target-integration-time` is enabled,
- `eval μs/smpl/wkr`: average evaluator time per sample per worker in `μs`;
  `EvalT` is worker-summed, so this is normalized by the total sample count,
- `prof py|eval|hav`: live profile `(python | evaluator | havana)`, where Havana includes grid
  sampling, cloned-grid training accumulation, merge, and update time.  In
  `--sampling-mode qmc`, `hav` is zero because QMCPy sampling is accounted for
  in the Python timing bucket.

When target stopping is enabled, Havana performs the target check after each
accumulated batch, not only at full iteration boundaries.  The final result
includes all completed batches up to the stopping point, and `--batch-size`
therefore controls the mid-iteration stopping granularity.  In correlated QMC
mode, the meaningful aggregate error/pull is available only after a complete
random-shift sector sum has been registered; before that first complete
aggregate, the progress bar shows `pending` for both `err%` and the live value,
pulls are suppressed, and keyboard interruption returns the last complete
aggregate estimate.

The final table reports the selected prefactor convention only.  Values with
Monte Carlo uncertainty use parenthesis notation with two significant error
digits.  The `MC err` column reports the relative one-sigma MC error in
percent.  Explicit `--target` values override all built-in references.
Without `--target`, built-in triangle/box runs use OneLOopBridge, while
DOT/FSD-only runs report `N/A` unless `--dot-engine both` or `--target
pysecdec` is used.
The timing footer reports total Symbolica evaluator time `EvalT`, measured
Python hot-path time `PythonT`, Havana time `HavanaT`, Taylor evaluator setup
time `TaylorGen`, chain-rule composition formula setup time `ChainGen`, and
the corresponding profile percentages.

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

- Euclidean denominator topologies only, with optional momentum-space
  dot-product numerators reduced to Feynman-parameter numerator polynomials,
- unit propagator powers only,
- numerator support is currently restricted to dot products of loop/external
  momenta; non-polynomial numerator callbacks and unreduced tensor structures
  are outside this phase,
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
- in legacy single-shot FSD DOT mode, pySecDec is used before integration only.
  Prepared `TopologyDefinition` and `SectorDefinition` objects are inherited by
  workers through a fork context; if fork is unavailable, multi-worker DOT
  integration fails clearly instead of regenerating sectors at runtime,
- in two-stage `integrate` mode, the runtime boots from serialized evaluator
  artifacts in `--output` and fails if any required artifact is missing.  It
  never regenerates missing pySecDec sectors or Symbolica evaluators.
