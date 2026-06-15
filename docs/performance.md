# FSD Performance Notes

This document records the current low-statistics performance measurements for
the DOT-backed FSD path.  The numbers are development measurements, not final
precision benchmarks.  They are meant to show generation cost, hot-path cost,
and the effect of the endpoint-projector formula cache and optional IBP
endpoint lowering.

All FSD-owned code uses Symbolica evaluators and Havana sampling.  In DOT/FSD
mode pySecDec is used only during generation, for Symanzik polynomial
construction and sector generation.  Once `TopologyDefinition` and
`SectorDefinition` objects are prepared, integration does not re-enter
pySecDec.

## Environment

| item | value |
|---|---|
| date | 2026-06-15 |
| machine | Darwin arm64 |
| Python | 3.12.6 |
| Symbolica | 2.0.0 |
| pySecDec | 1.6.6 |
| Normaliz | not found on `PATH` |
| default dual mode | `--pregenerate-dual-evaluators` |
| heavy DOT derivative mode | `--symbolic-derivatives` |
| default precision thresholds | `1e-8` and `1e-12` |
| default precision digits | `100` and `1000` |

Current fast test suite:

```text
97 passed, 2 skipped
```

The skipped tests are optional generated-pySecDec comparisons and one sandboxed
multiprocessing guard that can be unavailable on restricted macOS runners.

## Run Presets

Common configurations live in `examples/runs`.  CLI options can override any
YAML entry:

```sh
.venv/bin/python FSD.py --run examples/runs/dot_box.yaml --max-iter 1
```

The DOT graphs and kinematics are under `examples/graphs`.  Persistent targets
and run outputs are under `examples/outputs`.

Long exploratory runs should be launched through the local watchdog wrapper:

```sh
./run_with_memory_watch.py \
  --limit-gb 30 \
  --timeout-seconds 600 \
  --poll-seconds 0.5 \
  -- .venv/bin/python FSD.py --run examples/runs/dot_triple_box.yaml
```

The wrapper starts the FSD command in a child process group and terminates that
group itself if the timeout or memory limit is exceeded.  It prefers `psutil`
for process-tree RSS accounting and falls back to shell `ps` only if `psutil`
is unavailable.  This is important for
Symbolica native calls that may not react to `Ctrl-C`.  To stop it manually,
create the watched stop file from the same directory:

```sh
touch stop.order
```

The wrapper removes stale stop files at startup and removes the observed stop
file before terminating the child group.  In restricted sandboxes RSS polling
may be unavailable; the timeout and stop-file path still work.

## Generation Summary

The FSD generation buckets are:

- `Generation U and F polynomial`: DOT parse, kinematics, pySecDec
  `LoopIntegralFromGraph`, U/F extraction, prefactor metadata, and Symbolica
  expression conversion.
- `Generating sectors`: pySecDec sector decomposition and conversion to
  declarative `SectorDefinition` objects.
- `Generating Symbolica evaluators`: scalar U/F evaluators, sector-map and
  Jacobian evaluators, Taylor derivative evaluators, and endpoint-projector
  formula evaluators.

| topology | input | sectors | Laurent range | FSD generation [s] | pySecDec generated-integrator generation [s] | notes |
|---|---|---:|---|---:|---:|---|
| triangle | DOT | 3 | `eps^-2..eps^0` | 0.223 | 9.095 | low-stat comparison available |
| box | DOT | 12 | `eps^-2..eps^0` | 0.240 | 9.075 | low-stat comparison available |
| double box | DOT | 140 | `eps^-4..eps^0` | 0.615 | 272.81 | stored pySecDec target available |
| triple box | DOT iterative | 1972 | `eps^-6..eps^-1` current 30 GiB preset; `eps^-6..eps^0` supported | 5.582 generation smoke; 5.862 sampled leading-pole run; 56.11 capped six-order run | not completed | endpoint-projector cache plus curated small regular-Taylor assets; current 10-worker preset uses lower-memory IBP compound projectors |

For the triple box, the current leading-pole all-sector generation smoke
finished in 5.582 s with six curated regular-Taylor source assets used by
default and direct curated endpoint projectors selected for 382 sectors across
55 signatures.  A sampled all-sector run with 2000 samples generated in
5.862 s and then spent 137.6 s in integration.  The chain-rule pregeneration
guard still skips the huge active-sector request set.
The lowered chain-rule signature removes sector names and map expression
strings from the cache key, but the full active set still exceeds the default
cap.  A 1024-cap probe also skipped after collecting 1032 unique map-jet
signatures, so full all-sector chain-rule pregeneration is not yet a good
default.  The current guarded generation breakdown was:

| component | time [s] |
|---|---:|
| Generation U and F polynomial | 0.179 |
| Generating sectors | 1.110 |
| Generating Symbolica evaluators | 4.293 |
| total | 5.582 |

The 2000-sample all-sector run had a similar generation profile:

| component | time [s] |
|---|---:|
| Generation U and F polynomial | 0.176 |
| Generating sectors | 1.127 |
| Generating Symbolica evaluators | 4.559 |
| total | 5.862 |

Before child-projector sharing and cache reuse, endpoint-projector generation
for the same triple-box setup was about 14.55 s.  A cache-miss run with the
current endpoint-projector formula family took about 8.100 s total generation.
The generated cache root is intentionally not tracked wholesale.  It stores
parseable expression strings for both endpoint-projector formulas and
regular-Taylor formulas; compiled evaluator serialization is not attempted.
The validated endpoint-projector files and the first small validated
regular-Taylor files are shipped under `assets/subtraction_formulae/curated`
and are therefore normal FSD source assets.  The current curated regular set is
the six `PSD213`-class signatures, about 100 KiB in total.  Larger
regular-Taylor cache files remain selective: only signatures with a measured
runtime benefit should be promoted into the curated directory.

The Symbolica evaluator `functions=` API was checked as a possible route for
delegating more sparse-series algebra.  It supplies symbolic function
definitions that are inlined into the evaluator, for example `f(x)=x^2+2`.
An undefined `f(x)` cannot be called as a runtime Python callback from an
evaluator.  The viable route is therefore still to build coarser Symbolica
evaluators for reusable algebraic pieces, not to invoke Python sparse
multiplication from inside Symbolica.  The remaining Python-heavy sectors need
smaller universal regular-Taylor expressions, Symbolica-side factorizations
that avoid huge expanded direct formulae, or a native sparse-series primitive.

The runtime IBP child-term cache computes one Taylor envelope per boundary/zero
projector.  For the hard triple-box sector `PSD213`, this reduced ordinary
regular-Taylor reconstructions from 192 to 108.  The symbolic derivative
backend also shares one sector-map Taylor context between U and F.  A newer
regular-Taylor formula layer moves the regular `g_s` power/log/epsilon algebra
into pregenerated Symbolica evaluators and evaluates only the ancestor-closed
Taylor shape actually required by each formula.  For `PSD213`, these changes
cut a three-sample profile from 29.5 s to 10.6 s and then below one second of
hot Python time.  For selected sectors with at most 256 regular source requests,
FSD also pregenerates U/F dual evaluators for those smaller regular-Taylor
shapes, moving the chain-rule composition itself into Symbolica.

The regular-Taylor layer is guarded for all-sector runs.  The original
sector-specific signature would require 101018 regular signatures in the full
triple-box setup.  Lower residual-input signatures and sparse output requests
make the prepared formula set much smaller.  The current guarded all-sector
run prepares 147 regular-Taylor signatures, skips 29 harder high-axis
signatures, and switches 382 sectors to shipped direct endpoint-projector
assets.  This avoids multi-minute cold Symbolica builds while still using
cached Symbolica formula evaluators for the cheaper part of the sector family.
The caps are controlled by
`--regular-taylor-formula-axis-limit` and
`--regular-taylor-formula-volume-limit`.
For cache-warming studies, `--force-regular-taylor-formulas` lifts those
guards.  That mode is deliberately separate from the default: the expensive
six-axis formulae are universal for their signature and can be reused from the
`assets/subtraction_formulae` JSON cache, so cold-generation time and warm-cache
runtime should be compared as two different questions.  The ignored generated
cache root currently contains local exploratory regular-Taylor formula files
plus generated copies of the endpoint-projector files, about 849 MB total in
this development checkout.  The same 488 endpoint-projector files and six
vetted regular-Taylor files are now also curated source assets, about 49 MB
plus 100 KiB.  The remaining regular-Taylor
cache is too large and too uneven in runtime behavior to treat as an unfiltered
source asset.  Additional curated regular-Taylor files can still be promoted under
`assets/subtraction_formulae/curated`, which the formula loader searches in
addition to the generated cache root, once they show a clear runtime win.  A
curated regular-Taylor file is considered FSD code: it is preferred over a
local generated copy and bypasses the cold-build axis/volume guard for that
exact signature, so the corresponding direct formula becomes default without
requiring `--force-regular-taylor-formulas`.  This is the intended default-on
path for universal formulas that work well: ship the vetted subset as code,
keep the large exploratory cache local.
Symbolica evaluator `functions=` were inspected as a possible bridge for the
remaining fallback work.  In the installed Python API, this mechanism defines
symbolic subfunctions at evaluator construction time; it is not a runtime
callback interface.  It can structure one Symbolica expression, but it cannot
make a compiled subtraction evaluator dynamically call the separate U/F/J
coefficient evaluators or remove the coefficient-input boundary.  The
residual-input regular-Taylor builder already uses the compact available route
for cached direct formulae: build one scalar Symbolica expression and dualize
it for the requested Taylor/Laurent coefficients.  The remaining
hard-sector cost, when no direct endpoint-projector asset is curated, is
therefore the uncached large-signature regular residual algebra and the IBP
child-projector traversal, not a missing Python callback hook.
Use `scripts/inspect_subtraction_cache.py` to summarize generated versus
curated files by schema, signature version, axis count, mode, and size.  Once a
signature has a measured runtime benefit, promote exactly that generated JSON
with `scripts/promote_subtraction_formula_asset.py`; the promoted file becomes a
source asset and the direct formula is then default for that universal
signature.

An all-sector forced-formula triple-box probe with the existing warm cache tried
to prepare 226 regular-Taylor signatures.  It was still in the regular-Taylor
formula stage after 150 s, with RSS fluctuating between about 4 and 9.5 GiB,
and was stopped through `stop.order`.  A more focused `PSD1057` forced-formula
probe was also negative: one sector, 400 requested samples, and one worker
still produced no `result.json` after 301 s, while RSS plateaued near
11.4 GiB before the watchdog timeout terminated the process.  This makes the
current all-sector default clear: keep the guarded formula set and use forced
formula generation only for selected-sector or curated-cache studies.

The all-sector non-IBP path was re-probed after the cache had been warmed.  It
still timed out after 180 s inside endpoint-projector formula preparation at
low RSS, so the all-sector run preset remains IBP-lowered.  Selected-sector
no-IBP runs are still useful diagnostics, but they are not the default
triple-box steering mode.

The useful part of the non-IBP probe is now folded back into the default path:
when IBP lowering would create at least
`--direct-projector-cache-term-threshold` child projectors and a vetted direct
endpoint-projector asset is shipped for that universal signature, the sector
uses the direct cached projector.  This keeps the all-sector cold-generation
guard while letting known-good direct formulae become default FSD source data.
For the current triple-box leading-pole smoke this selected 382 sectors and 55
direct endpoint signatures.

For the leading two triple-box coefficients, only sectors with endpoint pole
depth five or six can contribute.  The current split is:

| class | sectors | regular-Taylor status |
|---|---:|---|
| endpoint depth below five | 1556 | skipped for `eps^-6..eps^-5` |
| endpoint depth five | 304 | prepared by regular formulas or direct endpoint assets |
| endpoint depth six | 112 | direct endpoint assets under the default guard |

The 112 depth-six sectors are not uniform.  Earlier default-IBP probes showed
that 324-request and 486-request sectors were stable but Python-heavy, while
729-request sectors were the practical blocker.  With the current threshold-54
direct cache policy, representative depth-six signatures now use curated direct
endpoint-projector assets:

| sector | class | samples | outcome |
|---|---|---:|---|
| `PSD814` | depth six, direct endpoint | 4000, 10 workers | `eps^-6=-0.000680(45)`, `eps^-5=-0.0204(10)`; no precision rescue |
| `PSD855` | depth six, direct endpoint | 4000, 10 workers | `eps^-6=-0.000680(45)`, `eps^-5=-0.0024(11)`; no precision rescue |
| `PSD649` | older 729-request fallback chronology | 10 | curated direct endpoint override gave `eps^-6=-0.00050(36)`, `eps^-5=-0.0032(75)`; no precision rescue |
| `PSD649` | 729-request sparse fallback, pre-direct-cache chronology | 1 requested | sparse fallback timed out after 180 s before returning one sample |
| `PSD649` | forced direct regular formulas, pre-direct-endpoint chronology | 1 forced-direct sample | generation 130.6 s; one sample took 381.7 s evaluator time |
| `PSD649` | child-batched fallback, pre-direct-cache chronology | 1 requested | timed out after 240 s before returning one sample |
| `PSD649` | child-batched fallback plus integer-power shortcut | 1 requested | timed out after 240 s before returning one sample |
| `PSD649` | dense sparse-convolution fallback chronology | 1 requested | timed out after 240 s before returning one sample; RSS stayed below 4 GiB |
| `PSD649` | clustered zero-set fallback chronology | 1 requested | returned one ordinary-precision sample in 104 s; `99.98%` Python |
| `PSD649` | adaptive chunks and split U/F/J source shapes | 1 requested | returned one ordinary-precision sample in 64.5 s; `99.97%` Python |
| `PSD649` | adaptive chunks chronology | 2 requested | two ordinary-precision samples in 124.6 s Python time; no precision rescue |
| `PSD649` | curated direct endpoint-projector override | 10 requested | ten ordinary-precision samples; `eps^-6=-0.00050(36)`, `eps^-5=-0.0032(75)`; `PythonT=0.368 s`, `EvalT=0.054 s`; no precision rescue |
| `PSD649` | Symbolica chain-rule formulas plus grouped zero chunks | 2 requested | generation 7.79 s with `ChainGen=6.34 s`; runtime PythonT 14.1 s, EvalT 0.43 s; no precision rescue |
| `PSD649` | structural map-layout reuse plus ancestor-cache shape building | 10 requested | generation 5.19 s with `ChainGen=3.82 s`; runtime PythonT 32.2 s, EvalT 1.35 s; `eps^-6=-0.00050(36)`, `eps^-5=-0.0031(59)`; no precision rescue |
| `PSD649` | multi-worker chronology | 100 requested, 10 workers | generation 5.25 s with `ChainGen=3.86 s`; hot summed PythonT 1125.7 s, EvalT 25.7 s; `eps^-6=-0.00018(14)`, `eps^-5=0.0006(32)`; no precision rescue |
| `PSD649` | forced direct regular formulas after chain-rule updates | 1 requested | watchdog timeout after 120 s under 35 GiB before completing the selected-sector run |
| `PSD184` | all-sector top contributor, focused probe | 4000, 10 workers | `eps^-6=0.00174(41)`, `eps^-5=-0.052(11)`; no precision rescue |
| `PSD80` | all-sector top contributor, focused probe | 4000, 10 workers | `eps^-6=0.00220(41)`, `eps^-5=-0.048(11)`; no precision rescue |
| `PSD1057` | all-sector top contributor, focused probe | 4000, 10 workers | `eps^-6=0.00056(14)`, `eps^-5=0.0270(50)`; no precision rescue; PythonT 107.59 s, EvalT 23.56 s |

The `PSD649` chronology separates two different kinds of direct formulae.  A
forced direct regular-Taylor formula made the final evaluator huge and was not
a viable default.  A curated direct endpoint-projector formula is different:
it bypasses hundreds of IBP child projectors while still using the black-box
Taylor source path, and the selected ten-sample diagnostic drops to
`PythonT=0.368 s`, `EvalT=0.054 s`.  The older intermediate chain-rule formula
layer is still useful for uncurated signatures: it lowered the selected
`PSD649` diagnostic from 124.6 s of Python time for two samples to 32.2 s for
ten samples while keeping all evaluations at ordinary precision.  The
remaining hard cases are the high-axis signatures that do not yet have a
curated direct endpoint asset; those still need a lower-signature Symbolica
function decomposition, a different sparse-series implementation, or further
algebraic factorization before an all-sector triple-box convergence run is
practical.

The current direct-endpoint hard sectors were profiled again after making
direct endpoint-projector assets the default.  For `PSD814`, a 200-sample
single-worker profile spends its hot time in 64 distinct zero-projector regular
source reconstructions.  Boundary stacking does not help this sector because
the direct endpoint formula has 64 coefficient groups and 64 different zero
sets.  A direct chain-rule lookup cache avoids rebuilding the structural
chain-rule signature after pregeneration and reduces the short-probe hot
`PythonT` from about 2.55 s to about 2.35 s.  Reusing the same sparse
convolution ladder for integer powers and logarithms in the regular fallback,
and evaluating all requested original-parameter derivatives through one
Symbolica multi-output evaluator, reduces the selected 4000-sample CLI probe to
`PythonT=41.78 s`, `EvalT=16.45 s`, and `4.11e3` evaluator microseconds per
sample per worker.  The path remains Python-heavy, but this is a real
improvement in the current guarded fallback.
Forcing direct regular-Taylor formulas for the same selected sector was stopped
by the watchdog after 180 s before integration, at about 4.4 GiB RSS.  That
confirms the current default tradeoff: shipped endpoint projectors are useful
source assets, while unvetted high-axis regular-Taylor formulas are still too
expensive to enable or curate blindly.
Switching the same selected sector back to pregenerated dual U/F evaluators was
also stopped after 180 s before integration, at about 5.2 GiB RSS.  The
symbolic-derivative path therefore remains the practical default for these hard
DOT sectors: it is not fully evaluator-dominated yet, but it reaches runtime
with minute-scale generation while the dual route does not.

After the forced-direct probe, the ordinary-precision IBP path was changed to
batch child endpoint-projector evaluations by child signature.  This reduces
the worst signature from hundreds of tiny child evaluator calls to one call per
distinct child projector signature.  The change is algebraically validated by a
term-by-term regression test, but it does not solve `PSD649`: the sparse
fallback still timed out after 240 s.  A micro-profile of the first 20 regular
groups showed typical group timings of `0.03..0.17 s`, with only about
`1.5e-4 s` in Symbolica evaluators.  The bottleneck is therefore Python
sparse-series construction of the regular `g_s` powers and logarithms.

The sparse fallback now detects integer regular powers and evaluates them with
truncated products or binomial reciprocal series instead of going through
`exp(power*log(series))`.  This is algebraically cheaper for the scalar
triple-box powers (`U^2 F^-4`), but it did not make `PSD649` runtime-ready: the
one-sample probe still timed out after 240 s.  That confirms the remaining
problem is the 729 distinct six-axis residual-source assemblies, not only the
integer power representation.

The sparse convolution kernel now also precomputes valid Taylor split lists for
each output support, capped by a small LRU cache so one-off six-axis signatures
do not become a persistent memory cost.  A direct profile over the `PSD649`
regular groups reached only 400 of 729 groups in 223 s before the 240 s
watchdog stopped the diagnostic.  RSS stayed below 3.3 GiB, so this is a CPU
and Python-algebra issue rather than a memory blow-up or precision rescue
problem.

The follow-up dense sparse-convolution implementation stores the requested
Taylor support in compact arrays and uses precomputed split-index plans.  On a
representative `PSD649` group this reduced the group time from about 5.16 s to
about 2.16 s, but the full one-sample `PSD649` probe still timed out after
240 s.  The improvement is real but not enough to make the fallback viable for
the hardest six-axis class; those signatures need either a curated direct
formula asset that has already been proven fast, or a lower-level compiled or
Symbolica-function implementation of the residual series algebra.

The current ordinary-precision fallback batches IBP boundary configurations by
zero set and adds exact constant/single-term sparse multiplication fast paths.
For a representative `PSD649` zero cluster, the time went from about 14.8 s to
about 4.7 s.  A later adaptive boundary-chunk rule stacks more rows for shallow
zero projectors and keeps smaller chunks for the two-axis cases where
over-stacking is slower.  The same path now returns one selected `PSD649`
sample in about `63..65 s`, with generation about `1.44 s` and no
precision-rescue samples.  With pregenerated Symbolica chain-rule composition
formulae and one zero-chunk assembly, a ten-sample diagnostic gives
`eps^-6=-0.00050(36)` and `eps^-5=-0.0031(59)`, with `96.2%` Python time and
no precision rescue.  The lowered signature plus cached map-jet layouts reduce
selected-sector chain-rule generation from about `6.34 s` to `3.82 s`, and
structural map-layout reuse brings the ten-sample Python time down to `32.2 s`.
A 100-sample, 10-worker probe gives `eps^-6=-0.00018(14)` and
`eps^-5=0.0006(32)`, again with no precision rescue.  Relative error on
`eps^-5` is not very meaningful when the central value is compatible with zero;
using the absolute `eps^-6` error, reaching about `1e-5` would require order
`2e4` selected-sector samples, or order hours at the measured rate.  This is enough
to diagnose the sector without timing out, but it is not enough for a
practical all-sector Monte Carlo run.  The remaining profile is still almost
entirely Python and not evaluator time, so the next required step is to move
the symbolic-derivative
chain-rule/sparse-series composition into a compiled kernel or into Symbolica
function evaluators.

The axis guard is necessary.  The current v3 sparse-output regular formula
signature avoids requesting the full Cartesian Taylor box and closes only the
actually needed output coefficients under Symbolica's dual-shape ancestor
rules.  This is correct and reduces Python overhead once the formula exists,
but it does not make the hardest six-axis formulas cheap to build or reload:
a selected `PSD62` run still spent about 226 s in Symbolica regular-Taylor
evaluator generation for eight six-axis signatures.  Once built, the one-sample
runtime became evaluator-dominated (`87.7%` evaluator, `12.3%` Python).  With
IBP lowering enabled, `PSD62` asks for 324 six-axis sparse regular-Taylor
signatures.  Their sparse output volumes range only from 8 to 54 coefficients
(`4` at volume `8`, `24` at `12`, `8` at `16`, `48` at `18`, `48` at `24`,
`32` at `27`, `96` at `36`, and `64` at `54`), but all are above the default
five-axis cold-build guard.  This makes them good candidates for deliberate
curation only after selected-signature runtime measurements show a win.
With the default axis cap, the same sector skips those signatures and finishes
generation in about 1.4 s.  Its runtime uses a sparse fallback that carries
endpoint output-pair shapes down to the U/F/J Taylor source requests.  This
reduced a one-sample `PSD62` fallback from about 36 s to about 11 s.  After
excluding identically-zero Taylor output columns from the sparse monomial
groups, a 10000-sample no-IBP control run spent 11.54 s in Python and 4.248 s
in Symbolica evaluator calls.  The fallback is still Python-heavy, but less so
than before this sparse grouping pass.

## Subtraction Backends

| backend | role |
|---|---|
| `recursive` | original Python/Numpy inclusion-exclusion over endpoint projectors |
| `formula` | one Symbolica formula evaluator per full sector signature |
| `projector-formula` | lower-signature Symbolica endpoint projector receiving black-box Taylor coefficients |
| `projector-formula + IBP` | optional lowering of `y^(-n+c eps)` endpoints to logarithmic child projectors |

`projector-formula` is the CLI default.  The recursive and full-formula
backends are kept as cross-checks and diagnostics.

The projector formula cache is universal in the relevant sense: it does not
depend on U, F, sector-map names, masses, or kinematics.  Its signature depends
on endpoint powers, Taylor orders, Laurent orders, and whether IBP lowering is
enabled.  Evaluators are rebuilt from cached expression strings during
generation; compiled evaluator serialization is not attempted.

Regular-Taylor formulas are cached with the same storage policy.  Their cache
does not contain U or F expressions; it only stores the algebra that maps
already-computed black-box Taylor coefficients to the regular coefficients fed
into the endpoint projector.

## IBP Endpoint Lowering

The option

```sh
--IBP_reduce_to_log_endpoint
```

is valid with `--subtraction-backend projector-formula`.  It lowers higher
endpoint powers such as `y^(-2+c eps)` and `y^(-3+c eps)` to logarithmic child
projectors plus boundary and derivative terms.  The extra derivatives are the
same kind of Taylor coefficients already requested by the projector
subtraction, so the dual and symbolic-derivative evaluator paths cover them.
Run-file settings can be overridden with `--no-ibp-reduce-to-log-endpoint`;
this is useful for selected-sector diagnostics because the non-IBP projector
can have fewer child Taylor requests even when the all-sector non-IBP formula
generation is too expensive.  A direct all-sector non-IBP triple-box generation
probe was stopped by the watchdog after 180 s still in endpoint-projector
formula build, so the shipped all-sector triple-box preset keeps IBP enabled.

The endpoint stability improvement is visible in the hard triple-box sector
`PSD213`.  For a near-endpoint sample with all singular coordinates at
`1e-12`, the ordinary double-precision full Laurent evaluation produced `NaN`
lower coefficients, while the high-precision path gave finite values:

| sector | scale | precision | representative result |
|---|---:|---|---|
| `PSD213` | `1e-12` | double | lower coefficients became `NaN` |
| `PSD213` | `1e-12` | 32 digits | `eps^0 = 6.351e90`, unstable |
| `PSD213` | `1e-12` | 80 digits | `eps^0 = -1.444e41`, unstable |
| `PSD213` | `1e-12` | 160 digits | `eps^-4 = 0.0325116`, `eps^-3 = 0.409428`, `eps^0 = 2.2229e4` |
| `PSD213` | `1e-12` | 300 digits | agrees with 160 digits in the displayed coefficients |

For the leading two coefficients of the same sector, ordinary double precision
already agrees with the high-precision rescue down to `1e-14`.  The lower
coefficients are much more sensitive.  In the `1e-12` point probe, 160 and 300
digits agreed, while 32 and 80 digits did not.  The CLI high tier remains
conservative at 1000 digits for samples closer than `1e-12`.

A same-sample Monte Carlo comparison of projector mode with and without IBP
for `PSD213`, performed before the current sparse residual-input signature switch, used
400 samples, the same seed, symbolic derivatives, and the leading two
coefficients.  It gave identical central values within MC precision:

| mode | samples | regular signatures | generation [s] | PythonT [s] | EvalT [s] | `eps^-5` |
|---|---:|---:|---:|---:|---:|---:|
| legacy selected-sector no IBP | 400 | 32 | 6.064 | 0.877 | 5.628 | `0.00068(40)` |
| legacy selected-sector IBP | 400 | 108 | 9.360 | 1.910 | 7.380 | `0.00068(40)` |

This answers the runtime question: at the same precision level, the current IBP
path is not favorable for this hard sector.  It is algebraically useful because
it lowers high endpoint powers to logarithmic child projectors, but it also
introduces boundary and derivative child terms.  In this implementation those
terms require more regular Taylor formulas and more high-precision child input
assembly.  The observed speedups below come from the regular-Taylor formula
layer and smaller Taylor shapes, not from the IBP identity by itself.

The same conclusion appears in the hard fallback sector `PSD62`.  The
IBP-lowered path is the all-sector default because all-sector non-IBP endpoint
projector generation is too expensive, but for this one selected sector the
no-IBP control is faster.  The IBP and no-IBP central values agree within the
large low-statistics errors:

| `PSD62` leading-pole run | samples | `eps^-5` estimate | relative err | PythonT [s] | EvalT [s] | precision rescue |
|---|---:|---:|---:|---:|---:|---|
| no IBP control | 1000 | `-0.049(22)` | 45.76% | 1.889 | 0.447 | none |
| IBP-lowered default | 1000 | `-0.050(23)` | 45.33% | 19.96 | 2.367 | none |
| IBP-lowered default | 2500 | `-0.055(14)` | 26.14% | 39.43 | 5.657 | none |
| IBP-lowered default | 5000 | `-0.055(10)` | 18.59% | 51.11 | 10.72 | none |
| IBP-lowered default | 15000 | `-0.0643(57)` | 8.92% | 163.03 | 33.26 | none |
| IBP + forced regular formulas, warm cache | 250 | `-0.056(45)` | 80.83% | 9.084 | 23.17 | none |

The IBP-lowered `PSD62` errors decrease from 18.59% to 8.92% when the sample
count increases from 5000 to 15000, close to the expected `1/sqrt(N)` trend
and enough to reach the requested order-10% target for this selected sector.
The 15000-sample point took 196 s wall time on one worker and remained entirely
ordinary precision.  The forced regular-formula path proves the warm-cache idea
can shift most runtime into Symbolica (`71.83%` evaluator in the 250-sample
run), but its direct evaluator is much heavier per sample (`92.7 ms/smpl/wkr`)
than the guarded fallback for this sector.  It therefore remains an opt-in
cache viability mode, not the default.

For ordinary Monte Carlo samples of the same sector, the envelope cache gives a
clearer runtime win:

| `PSD213` leading-pole run | samples | `eps^-5` estimate | MC err | relative err | PythonT [s] | EvalT [s] |
|---|---:|---:|---:|---:|---:|---:|
| before Taylor-envelope reuse | 100 | `8.26e-4` | `7.45e-4` | 90.25% | 90.03 | 0.257 |
| after Taylor-envelope reuse | 100 | `8.26e-4` | `7.45e-4` | 90.25% | 39.22 | 0.0997 |
| after Taylor-envelope + shared U/F map context | 100 | `8.26e-4` | `7.45e-4` | 90.25% | 25.25 | 0.119 |
| after Taylor-envelope + shared U/F map context | 400 | `7.4e-4` | `4.1e-4` | 55.22% | 53.88 | 0.367 |
| after regular-Taylor formula + minimized shape | 100 | `8.26e-4` | `7.45e-4` | 90.25% | 1.989 | 0.191 |
| after regular-Taylor dual evaluators | 100 | `8.26e-4` | `7.45e-4` | 90.25% | 1.578 | 1.937 |
| after regular-Taylor formula + minimized shape | 400 | `7.4e-4` | `4.1e-4` | 55.22% | 4.111 | 0.647 |
| after regular-Taylor dual evaluators | 400 | `6.8e-4` | `4.0e-4` | 58.81% | 0.877 | 5.628 |
| current sparse residual-input signatures | 100 | `1.51e-3` | `8.9e-4` | 58.59% | 0.170 | 1.099 |
| current sparse residual-input signatures | 400 | `6.8e-4` | `4.0e-4` | 58.81% | 0.343 | 4.242 |
| current IBP sparse residual-input signatures | 1600 | `5.94e-4` | `1.90e-4` | 32.03% | 2.242 | 72.984 |

The 100-to-400 sample error reduction is compatible with the beginning of
Monte Carlo \(1/\sqrt{N}\) scaling for this sector: the absolute error fell by
a factor about 1.8, close to the ideal factor 2.  Extrapolating the 400-sample
relative error to 10% requires roughly
\[
  N_{10\%} \simeq 400 \left({55.22\over 10}\right)^2 \simeq 1.2\times 10^4
\]
samples for this sector from the older 400-sample point.  The newer 1600-sample
IBP point has 32.03% relative error, which extrapolates to about `1.6e4`
samples for 10% and about 12.5 minutes of one-worker hot integration time at
the measured 1600-sample rate.  This path is already evaluator-dominated
(`97.02%` evaluator), so the profile shape is the desired one.
With efficient multi-worker execution this should be substantially shorter,
but the current sandbox prevented a reliable multi-worker timing run.

## Hot Runtime

`EvalT` and `PythonT` are worker-summed work times, so they can exceed elapsed
wall time in multi-worker runs.  The `avg` column is the worker-local
evaluator time per accepted sample: `EvalT / Nsamples`.  It is not divided by
the worker count again because `EvalT` has already been accumulated over the
workers that evaluated those samples.

| topology | setup | samples | workers | avg [us/smpl/wkr] | profile | notes |
|---|---|---:|---:|---:|---|---|
| built-in triangle massless | pregenerated duals | 4096 | 1 | 2.64 | mixed | one-loop endpoint sectors |
| built-in box massless | pregenerated duals | 4096 | 1 | 2.72 | mixed | one-loop endpoint sectors |
| DOT triangle | pregenerated duals | 4096 | 1 | 0.502 | mixed | generated sectors |
| DOT box | pregenerated duals | 4096 | 1 | 1.26 | mixed | generated sectors |
| DOT double box | symbolic derivatives, projector formula | 20000 | 4 | 18.23 | mostly evaluator/Python mixed | target available |
| DOT triple box `PSD389` | projector formula, no IBP | 100 | 1 | 10.7 | 72.67% Python, 24.63% evaluator | one `y^-3` endpoint, leading two orders zero |
| DOT triple box `PSD386` | projector formula, no IBP | 100 | 1 | 20.4 | 80.40% Python, 18.29% evaluator | two `y^-3` endpoints, leading two orders zero |
| DOT triple box `PSD213` | sparse projector formula, no IBP | 400 | 1 | 1.06e4 | 7.49% Python, 92.50% evaluator | hard nonzero sector, leading two orders |
| DOT triple box `PSD213` | current IBP-lowered prepared formulas | 400 | 1 | 4.19e4 | 3.61% Python, 96.39% evaluator | `eps^-5 = 0.00074(41)`, no precision rescue |
| DOT triple box `PSD213` | IBP-lowered prepared formulas | 1600 | 1 | 4.56e4 | 2.98% Python, 97.02% evaluator | `eps^-5 = 0.00059(19)`, no precision rescue |
| DOT triple box `PSD1` | IBP-lowered sparse fallback | 1000 | 1 | 2.36e3 | 89.10% Python, 10.90% evaluator | 324 skipped regular requests; no precision rescue |
| DOT triple box `PSD7` | IBP-lowered sparse fallback | 1000 | 1 | 3.58e3 | 96.46% Python, 3.54% evaluator | 486 skipped regular requests; no precision rescue |
| DOT triple box all sectors | current guarded sampled smoke | 2000 | 10 | 3.58e4 | 93.26% Python, 6.59% evaluator | generation 5.862 s; integration 137.6 s; `eps^-6=0.221(108)`, `eps^-5=2.1(2.9)`; no non-finite sector entries; no precision rescue |
| DOT triple box all sectors | 30 GiB capped six-order IBP preset | 370 | 10 | 2.07e6 | 56.03% Python, 43.95% evaluator | interrupted by 30 GiB guard at 324 s wall; generation 56.11 s; `eps^-6=0.15(15)`, `eps^-5=2.0(3.0)`, `eps^-4=-72(67)`, `eps^-3=-805(593)`, `eps^-2=-2159(4303)`, `eps^-1=-984(24956)`; no precision rescue |
| DOT triple box `PSD814` | current direct endpoint cache | 4000 | 10 | 4.11e3 | 71.75% Python, 28.25% evaluator | multi-output derivative evaluator plus shared power/log ladder; `eps^-6=-0.000680(45)`, `eps^-5=-0.0204(10)`; no precision rescue |
| DOT triple box `PSD855` | current direct endpoint cache | 4000 | 10 | 4.63e3 | 80.67% Python, 19.33% evaluator | `eps^-6=-0.000680(45)`, `eps^-5=-0.0024(11)`; no precision rescue |
| DOT triple box `PSD184` | all-sector top contributor, focused probe | 4000 | 10 | 5.02e3 | 22.12% Python, 77.88% evaluator | `eps^-6=0.00174(41)`, `eps^-5=-0.052(11)`; no precision rescue |
| DOT triple box `PSD80` | all-sector top contributor, focused probe | 4000 | 10 | 1.09e4 | 22.53% Python, 77.47% evaluator | `eps^-6=0.00220(41)`, `eps^-5=-0.048(11)`; no precision rescue |
| DOT triple box `PSD1057` | all-sector top contributor, focused probe | 4000 | 10 | 5.89e3 | 82.03% Python, 17.97% evaluator | endpoint-plan cache active; `eps^-6=0.00056(14)`, `eps^-5=0.0270(50)`; no precision rescue |
| DOT triple box `PSD62` | current IBP-lowered default | 2000 | 1 | 6.19e4 | 32.62% Python, 67.38% evaluator | `eps^-6 = 0.00161(57)`, `eps^-5 = -0.056(16)`, no precision rescue |
| DOT triple box `PSD62` | default sparse fallback | 100 | 1 | 2.66e3 | 98.00% Python, 2.00% evaluator | older pre-chain-rule diagnostic; generation 1.396 s; no precision rescue |
| DOT triple box `PSD62` | `--no-ibp`, sparse fallback control | 10000 | 1 | 425 | 73.08% Python, 26.90% evaluator | `eps^-5 = -0.0734(70)`, 9.52%; no precision rescue |
| DOT triple box `PSD62` | IBP-lowered sparse fallback | 2500 | 1 | 2.26e3 | 87.45% Python, 12.55% evaluator | `eps^-5 = -0.055(14)`, 26.14%; no precision rescue |
| DOT triple box `PSD62` | IBP-lowered sparse fallback | 15000 | 1 | 2.22e3 | 83.05% Python, 16.94% evaluator | `eps^-6 = 0.00213(21)`, `eps^-5 = -0.0643(57)`; no precision rescue |
| DOT triple box `PSD62` | IBP + `--force-regular-taylor-formulas`, warm cache | 250 | 1 | 9.27e4 | 28.16% Python, 71.83% evaluator | regular formula build 0.189 s; direct evaluator too heavy for default |
| DOT triple box all sectors | IBP + `--force-regular-taylor-formulas`, warm cache | n/a | 1 | n/a | n/a | stopped after 150 s in regular-formula preparation; 226 signatures requested |
| DOT triple box `PSD1057` | IBP + `--force-regular-taylor-formulas`, warm cache | 400 requested | 1 | n/a | n/a | no result after 301 s; RSS about 11.4 GiB; stopped by watchdog before sampling |
| DOT triple box `PSD62` | `--no-ibp`, force six-axis formulas up to volume 2 | 100 | 1 | 520 | 95.56% Python, 4.43% evaluator | 1 regular formula built, 7 skipped; no speedup |
| DOT triple box `PSD62` | `--no-ibp`, force six-axis formulas up to volume 6 | 100 | 1 | 740 | 93.61% Python, 6.38% evaluator | 3 regular formulas built, 5 skipped; no speedup |
| DOT triple box all sectors | current guarded default generation probe | 1 | 1 | n/a | generation-only smoke | generation 5.582 s; 147 regular formulas built, 6 curated, 29 skipped, 382 direct endpoint overrides |
| DOT triple box `PSD62` | six-axis v3 formula forced | 1 | 1 | 2.24e6 | 12.26% Python, 87.74% evaluator | generation 227 s; proves formula runtime is evaluator-dominated but cold build is too slow |
| DOT triple box `PSD649` | forced direct regular formulas | 1 | 1 | 3.82e8 | 11.62% Python, 88.38% evaluator | generation 130.6 s; evaluator too large for runtime |
| DOT triple box `PSD649` | IBP-lowered sparse fallback | 1 requested | 1 | n/a | no completed sample | 729 skipped regular requests; timed out after 180 s |
| DOT triple box `PSD649` | child-batched sparse fallback | 1 requested | 1 | n/a | no completed sample | timed out after 240 s; bottleneck is regular series algebra |
| DOT triple box `PSD649` | dense sparse-convolution fallback | 1 requested | 1 | n/a | no completed sample | timed out after 240 s; representative group improved 2.4x but full sector still too slow |
| DOT triple box `PSD649` | clustered zero-set fallback plus sparse fast paths | 1 | 1 | 1.9e4 | 99.98% Python, 0.02% evaluator | generation 1.585 s; runtime 104 s; no precision rescue |
| DOT triple box `PSD649` | adaptive chunks and split U/F/J source shapes | 1 | 1 | 2.02e4 | 99.97% Python, 0.03% evaluator | generation 1.444 s; runtime 64.5 s; no precision rescue |
| DOT triple box `PSD649` | adaptive chunks, two-sample diagnostic | 2 | 1 | 2.07e4 | 99.97% Python, 0.03% evaluator | `eps^-6=-0.00103(75)`, `eps^-5=0.0017(78)`; no precision rescue |
| DOT triple box `PSD649` | pregenerated chain-rule formulas and one zero-chunk assembly | 2 | 1 | 2.15e5 evaluator μs/smpl/wkr | 97.04% Python, 2.96% evaluator | generation 7.792 s, `ChainGen=6.335 s`; PythonT 14.12 s, EvalT 0.430 s; no precision rescue |
| DOT triple box `PSD649` | pre-direct-endpoint structural map-layout reuse, ancestor cache, sparse-support key cache | 10 | 1 | 1.39e5 | 95.95% Python, 4.05% evaluator | generation 5.31 s, `ChainGen=3.88 s`; `eps^-6=-0.00050(36)`, `eps^-5=-0.0031(59)`; no precision rescue |
| DOT triple box `PSD649` | current curated direct endpoint-projector override | 10 | 1 | 5.45e3 | 87.10% Python, 12.88% evaluator | generation 1.962 s; `eps^-6=-0.00050(36)`, `eps^-5=-0.0032(75)`; no precision rescue |
| DOT triple box `PSD649` | pre-direct-endpoint multi-worker convergence probe | 100 | 10 | 2.57e5 | 97.77% Python, 2.23% evaluator | `eps^-6=-0.00018(14)`, `eps^-5=0.0006(32)`; no precision rescue |
| DOT triple box `PSD649` | forced direct regular formulas, guarded watchdog | 1 requested | 1 | n/a | no completed sample | stopped at 120 s under 35 GiB before finishing the selected-sector run |
| DOT triple box all sectors | chain-rule pregeneration guard | 1 | 1 | n/a | generation smoke | generation 5.582 s; chain-rule formulas skipped by limit for huge active set |

The full triple-box all-sector integration remains runtime-heavy.  Generation
is now fast enough to iterate on the integrand, and the hard-sector MC error is
showing the expected downward trend.  The selected hard-sector runtime is much
faster after the regular-Taylor evaluator layer and is evaluator-dominated when
the selected sector stays below the regular-formula caps.  The complete
six-order 10-worker run is currently memory-limited: the faster direct cached
endpoint-projector path can spike above 30 GiB, so the shipped preset uses
lower-memory IBP compound projectors and batch size 1.  Even then, the final
30 GiB capped run interrupted at 370 samples and is not a convergence result.
The remaining runtime problem is the high-axis regular-source assembly that
feeds hard endpoint sectors; it is stable, but not yet sufficiently fused into
Symbolica evaluators.  The current all-sector 2000-sample smoke gives naive
10% error estimates of about
`4.8e4` samples for `eps^-6` and `3.8e5` samples for `eps^-5`, or roughly
55 minutes and 7.3 hours at the measured all-sector rate before adaptive-grid
improvements.  Leading-pole probes now skip
sectors whose endpoint pole depth is too shallow to contribute to the requested
Laurent range, so shallow sectors no longer pay the subtraction cost just to
return exact zero.  A profile of `PSD62` showed that the old dense symbolic-
derivative fallback spent tens of seconds composing rectangular Taylor boxes.
The current sparse fallback passes endpoint output-pair requests down to the
source Taylor layer and uses sparse truncation for the regular power/log
series.  It is much faster, but selected direct-endpoint depth-six sectors are
still dominated by Python sparse-series composition rather than evaluator calls.
Batching child projectors by
signature removes redundant endpoint-projector evaluator calls, but it does
not address the dominant regular-series construction for the hardest 729-group
class.  Partial preparation of only the
cheapest six-axis signatures is also not useful: volume limits 2 and 6 build
quickly, but the remaining high-volume signatures still dominate the Python
sparse-series fallback and the small prepared formula calls add more overhead
than they remove in the 100-sample probes.

The selected-sector result JSON was also compacted.  Earlier result files
serialized every sector dual-shape multi-index in `summary.symanzik`, which
made a `PSD62` 1000-sample diagnostic about 109 MB and caused JSON writing to
dominate cProfile.  The summary now records compact dual-shape statistics
instead; the same diagnostic result file is about 7.5 MB while retaining all
aggregate and per-sector coefficient results.

Symbolica custom numeric functions were also checked.  A symbol declared with
`S("f", eval={"complex": callback, "decimal_complex": callback})` can be used
inside `evaluate_complex` and `evaluate_complex_with_prec`.  This is a useful
future route for a more fused evaluator, but the Python callback is invoked per
function occurrence and sample, so it is not automatically faster than the
current batched Symbolica derivative evaluators unless the callback is coarse or
implemented outside Python.
