# FSD Performance Notes

These are development measurements for the DOT-backed FSD path as of
2026-06-22.  They are performance and stability diagnostics, not final
precision benchmarks.  FSD-owned code remains free of SciPy and SymPy imports;
pySecDec is used only at the DOT generation boundary.

## Environment

| item | value |
|---|---|
| machine | Apple M3 Pro, 12 logical CPUs, Darwin arm64 |
| Python | 3.12.6 |
| Symbolica | 2.0.0 local `symbolica-community` wheel patched to `symbolica/dev` |
| Symbolica dev commit | `5f61332d8b21391f40712f42e499b3cf6a9ae7fa` |
| pySecDec | 1.6.6 |
| Normaliz | not found on `PATH`; iterative/geometric_ku paths used |
| default precision thresholds | `1e-3`, `1e-6`, and `1e-8` |
| default precision digits | 32, 100, and 1000 |

## Symbolica Dev Dualization Check

The standalone reproducer `U_dualization_slowdown.py` was used before and
after installing a local `symbolica-community` wheel patched to
`symbolica/dev`.  The default case is the triple-box U polynomial with the
six-axis dual shape `[3,3,3,3,3,4]`, i.e. 5120 requested Taylor coefficients.

| Symbolica source | scalar evaluator build [s] | copied evaluator dualize [s] | speedup |
|---|---:|---:|---:|
| previous venv wheel | 0.000282 | 191.316 | 1x |
| local community/dev wheel | 0.000196 | 11.926 | 16.0x |

This removes the original several-minute U/F dualization bottleneck.  It does
not by itself make every high-axis regular-source formula fast: the cost has
moved to how much source algebra/evaluator fragmentation we ask Symbolica and
Python to perform.

The current local wheel was built in release mode from `symbolica-community`
with its `Cargo.toml` patched to the Symbolica dev commit above.  The
standalone `MRE_JIT_compile_real_bug.py` still reproduces a real-JIT
multi-output evaluation bug on the DOT double-box `PSD50` sector: real
`evaluate(...)` differs from the eager evaluator at order one, while complex
`evaluate_complex(...)` agrees at about `1e-11`.  FSD therefore keeps using the
complex JIT entry point as the f64 hot path when `--jit-compile
--real-evaluator` is selected.

## Generation Timing

FSD generation is reported in three headline buckets:

| bucket | meaning |
|---|---|
| Generation U and F polynomial | DOT parsing, kinematics, pySecDec loop-integral construction, U/F extraction, prefactor metadata, Symbolica expression conversion |
| Generating sectors | pySecDec sector decomposition and conversion to declarative `SectorDefinition` metadata |
| Generating Symbolica evaluators | scalar evaluators, sector map/Jacobian evaluators, derivative evaluators, endpoint projectors, regular Taylor formulas |

Current topology overview:

| topology | input | sectors | Laurent range | FSD generation [s] | pySecDec generated-integrator generation [s] | FSD timing notes |
|---|---|---:|---|---:|---:|---|
| triangle | DOT | 2 | `eps^-2..eps^0` | 0.223 | 9.095 | pySecDec symmetry-reduced sectors |
| box | DOT | 3 | `eps^-2..eps^0` | 0.240 | 9.075 | pySecDec symmetry-reduced sectors |
| double box | DOT | 96 | `eps^-4..eps^0` | 9.031 prepared explicit bundle | 272.81 historical package build | corrected prefactor bundle |
| triple box | DOT iterative | 1972 | `eps^-6..eps^0` | 38.46 recorded generation + 30.61 serialization | not completed | compressed prepared bundle, 30 GiB guard |

## QMC Integration Probe

FSD has an experimental `--sampling-mode qmc` based on randomized shifted
rank-1 lattices.  The implementation is independent of pySecDec's QMC
internals: FSD obtains shifted lattice points, applies a Korobov periodizing
map, evaluates the prepared Symbolica sector evaluators in batches, and
estimates errors from the shift-to-shift spread.

As of 2026-06-23, QMC defaults are deliberately conservative: unless the user
explicitly requests an evaluator mode, QMC uses eager complex Symbolica
evaluators.  This avoids the known real-JIT evaluator bug documented by the
standalone MREs.  The default lattice backend is now the local
`cbcpt-dn1-100` CBC/PT generating-vector table because it is much closer to
pySecDec's QMC convergence on the double box.  `--qmc-lattice-backend qmcpy`
remains available and is still useful for independent one-loop checks.

### One-loop parity

The DOT triangle and box were rechecked on 2026-06-23 with the current safe
QMC defaults: eager complex FSD evaluators and the bundled CBC/PT lattice table.
Both agree with the stored pySecDec/OneLOop-compatible targets.  The comparison
helper now infers the matching kinematics file from the run YAML, so the
generated pySecDec box package receives all three real parameters
(`s12`, `s23`, `mt`).

| topology | N/shift requested -> used | shifts | FSD support groups | FSD raw samples | FSD eps^0 diff | FSD eps^0 err | pySecDec effective samples | pySecDec eps^0 diff | pySecDec eps^0 err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| triangle | `1024 -> 1123` | 16 | 2 | 35,936 | `+9.53e-10` | `8.42e-10` | 89,840 | `+1.36e-10` | `7.12e-10` |
| box | `1024 -> 1123` | 16 | 5 | 89,840 | `+1.36e-5` | `8.77e-6` | 143,744 | `+4.88e-6` | `6.06e-6` |

These rows come from fixed-work pySecDec comparisons with loose pySecDec
tolerances (`epsrel=epsabs=1e99`), so pySecDec does not refine to a hidden
accuracy target.  They confirm that FSD QMC still has one-loop convergence
parity in sample-count terms.

### Double-box QMC status

The scalar DOT double box was also compared in the same pySecDec prefactor
convention.  The earlier stored file
`examples/outputs/dot_double_box_pysecdec_target.json` must not be treated as a
truth value: it is itself a finite-statistics result and has no reliable error
stored with it.  The best available reference in the current logs is instead
the longer pySecDec run in `/tmp/pysecdec_double_box_178070.json`.  This is a
practical numerical reference, not an analytic ground truth.

A bug found during the QMC comparison was in FSD's DOT global-prefactor series:
`-gamma(3+2 eps)` was routed through the generic Symbolica Gamma-series
fallback, which gave inaccurate coefficients from `eps^2` onward.  The
analytic affine-Gamma path now handles optional signs and numeric scales.  The
correct pySecDec-prefactor coefficients for this double-box run are:

```text
[-2.0,
 -3.6911373403938685,
 -4.985859983805386,
 -4.599953351364897,
 -3.681199052065825]
```

| engine | run setting | wall/integration time | sample accounting | eps^0 value | eps^0 quoted err | note |
|---|---|---:|---|---:|---:|---|
| FSD QMC, corrected prefactor bundle | CBC/PT, `N=17807`, 16 shifts | 125.9 s | 30.2M raw sector-group samples | `-14.8630462698` | `4.08e-3` | all coefficients within `2.5 sigma` |
| pySecDec QMC, fixed-work tolerances | CBC/PT, `N=17807`, 16 shifts | 3.68 s | 86.0M parsed sector/order samples | `-14.8446386726` | `3.78e-3` | same `n`, loose tolerances |
| FSD QMC, corrected prefactor bundle | CBC/PT, `N=34687`, 32 shifts | 471.1 s | 117.7M raw sector-group samples | `-14.8480115525` | `2.83e-3` | 7.9-minute 10-core run |
| pySecDec QMC, fixed-work tolerances | CBC/PT, `N=34687`, 32 shifts | 11.5 s | 335.2M parsed sector/order samples | `-14.8529765646` | `1.90e-3` | same `n`, loose tolerances |
| pySecDec QMC | CBC/PT, `N=178070`, 16 shifts | 645.6 s | public budget 2.85M; verbose sample accounting not captured | `-14.8531901024` | `2.48e-7` | best current reference |

The corrected FSD runs are compatible with the long pySecDec numerical
reference for all five Laurent coefficients.  In sample-count terms, FSD is
close to pySecDec: at `N=34687`, 32 shifts, pySecDec uses about `2.85x` more
parsed sector/order samples and quotes a finite-part error only about `1.5x`
smaller.  In wall-clock terms pySecDec remains far faster because its generated
C++ kernels are much cheaper than FSD's Symbolica evaluator route.

The source of the earlier apparent non-parity was twofold:

1. FSD expanded the signed pySecDec global prefactor `-gamma(3+2 eps)`
   incorrectly from `eps^2` onward.  That is now fixed by the signed/scaled
   affine-Gamma path.
2. The comparison script used `epsrel=epsabs=1e-99` for pySecDec.  That forced
   pySecDec to refine internally, so nominal low-`N` runs were not fixed-work
   runs.  For example, a comparison run requested `N=1024` and `N=4096` with
   16 shifts:

| requested `N` | FSD raw sector samples | FSD eps^0 diff / err | pySecDec effective samples parsed from verbose log | pySecDec eps^0 diff / err |
|---:|---:|---:|---:|---:|
| 1024 | 1.90M | `+5.02e-1 / 2.06e0` | 86.0M | `+3.63e-4 / 4.12e-3` |
| 4096 | 7.23M | `+7.60e-2 / 2.07e-1` | 326.6M | `+1.00e-4 / 4.53e-4` |

This explains why pySecDec appeared to converge much faster at equal
user-facing `N`: the generated integrator was also allocating substantially more
effective sector/order work.  If the target is equal wall time rather than
equal sampled work, FSD still needs either much faster sector kernels or a
compiled pySecDec-style backend.  If the target is equal sample count, the
current fixed-work runs show close parity.

### Havana comparison

A current Havana probe with the same corrected double-box target used
`--target-integration-time 120` after the target-time warmup adjustment.  The
integration phase ran for `208.3 s` after generation and accumulated `50.6M`
samples.  It is still much less reliable than QMC on this topology:

| sampler | samples | integration time [s] | eps^0 diff to long pySecDec | eps^0 err | note |
|---|---:|---:|---:|---:|---:|
| FSD QMC | 30.2M raw sector-group samples | 125.9 | `-9.86e-3` | `4.08e-3` | corrected prefactor bundle |
| FSD QMC | 117.7M raw sector-group samples | 471.1 | `+5.18e-3` | `2.83e-3` | 7.9-minute 10-core run |
| FSD Havana | 50.6M samples | 208.3 | `+1.25e-1` | `0.320` | corrected prefactor bundle |

Havana still needs more training/iterations for the multi-loop endpoint
structure; QMC is the trustworthy default for the current double-box
convergence studies.  The target-time warmup remains an estimate and overshot
this run, but the sampler conclusion is unambiguous: at comparable sampled work
Havana is orders of magnitude less efficient than QMC for the finite part.

## Explicit Backend And Numerator Timing

The `--explicit` backend substitutes the sector maps into each sector
integrand and builds one multi-output Symbolica evaluator per sector.  This is
the pySecDec-like comparison path: it deliberately gives up the FSD black-box
U/F derivative construction in exchange for a faster runtime evaluator.

The one-loop timing tables in this subsection are older backend microbenchmarks
and were recorded before the current pySecDec-style sector symmetry reduction
was applied globally.  They remain useful for numerator/runtime trends, but the
current scalar sector counts are the symmetry-reduced counts in the generation
overview above.

The table below compares the explicit FSD path against pySecDec's generated
integrator on one-loop scalar and numerator examples.  All pySecDec runs were
launched through `run_with_memory_watch.py --limit-gb 30`.  The pySecDec
runtime column is computed as the recorded pySecDec integration wall time
divided by `--pysecdec-maxeval 1000`; pySecDec's public result JSON does not
expose the exact number of integrand calls, so this should be read as a
normalized wall-time proxy rather than a strict per-call profiler.

| case | sectors | FSD explicit generation [s] | FSD explicit avg [us/sample] | FSD explicit min/max [us/sample] | pySecDec package generation [s] | pySecDec setup incl. compile [s] | pySecDec integration/maxeval [us] |
|---|---:|---:|---:|---:|---:|---:|---:|
| triangle | 3 | 0.190 | 5.58 | 2.42 / 10.21 | 0.651 | 5.524 | 42.15 |
| box | 12 | 0.199 | 2.68 | 1.45 / 9.71 | 0.588 | 5.707 | 64.66 |
| triangle numerator | 4 | 0.169 | 4.06 | 1.94 / 8.24 | 0.593 | 5.872 | 47.58 |
| box numerator | 12 | 0.325 | 2.95 | 1.66 / 10.80 | 0.798 | 8.558 | 173.15 |
| box rank-2 numerator | 12 | 0.227 | 3.05 | 1.74 / 10.35 | 0.809 | 7.912 | 156.73 |
| box high-rank numerator | 12 | 0.398 | 3.05 | 2.02 / 8.90 | 414.177 | 424.298 | 344.88 |

The high-rank box numerator is intentionally exaggerated.  It is still below
the 10-minute pySecDec-generation cutoff requested for this comparison, but it
already shows the generation/runtime trade-off: FSD explicit generation stays
sub-second because it asks Symbolica to build eager sector evaluators directly,
while pySecDec spends most of its time producing and compiling a generated C++
package.

The corresponding FSD projector timings on the same examples are much slower
at one loop, because the projector path is intentionally factored into
black-box U/F source evaluation plus universal endpoint algebra:

| case | FSD projector generation [s] | FSD projector avg [us/sample] | FSD projector min/max [us/sample] |
|---|---:|---:|---:|
| triangle | 0.179 | 372.91 | 64.13 / 789.38 |
| box | 0.196 | 217.15 | 63.79 / 1280.44 |
| triangle numerator | 0.171 | 213.89 | 18.82 / 719.78 |
| box numerator | 0.184 | 186.17 | 64.11 / 851.43 |
| box rank-2 numerator | 0.200 | 190.85 | 65.13 / 921.13 |
| box high-rank numerator | 0.390 | 183.42 | 65.62 / 882.12 |

This confirms that the explicit backend is the right comparison point when the
question is pySecDec-style runtime speed.  It is now the default DOT runtime
backend, but it is not the black-box FSD strategy: it relies on explicitly
substituting sector maps into U/F, whereas the projector backend preserves the
path-finder goal of treating U/F as numerical black-box evaluators.

## Scalar Double-Box Three-Way Timing

The scalar Euclidean double-box three-way timing below is a historical
pre-symmetry-reduction microbenchmark from 2026-06-22.  It should not be mixed
with the current 96-sector QMC convergence comparison above.  It is retained
only because it records the relative cost of the projector and explicit
evaluator backends on the same machine.  FSD runtimes used the `benchmark`
subcommand with 5 ordinary f64 interior samples per sector.  pySecDec was run
through `--dot-engine pysecdec` with `--keep-pysecdec-workdir`, then the
generated shared library was loaded directly for an independent verbose timing
pass.

| method | sectors / generated integrals | generation setup [s] | compile [s] | integration/runtime metric |
|---|---:|---:|---:|---|
| FSD projector | 140 sectors | 0.445 | n/a | avg 583.43 us/FSD-sector sample; median 254.71; min/max 18.21 / 4501.27 |
| FSD explicit | 140 sectors | 12.583 | n/a | avg 3.99 us/FSD-sector sample; median 2.54; min/max 1.49 / 18.27 |
| pySecDec generated | 302 sector/order integrals | 7.837 | 338.752 | 3.682 s recorded integration wall time |

The pySecDec direct-library cross-check is more informative than simply
dividing by `--pysecdec-maxeval`, but the most precise comparison is the direct
generated-kernel benchmark now exposed through FSD's `benchmark` subcommand:

```sh
.venv/bin/python FSD.py benchmark \
  --run examples/runs/dot_double_box.yaml \
  --dot-engine pysecdec \
  --pysecdec-workdir .pysecdec_build_double_box \
  --keep-pysecdec-workdir \
  --benchmark-samples-per-sector 1000000 \
  --quiet-summary \
  --no-progress
```

This compiles a temporary C++ driver against the persistent pySecDec generated
static library and directly times the generated sector/order kernels.  A
verbose public `IntegralLibrary` call is still useful as a whole-package timing
cross-check, but it is not the same metric.

For an apples-to-apples sector-kernel comparison, a small C++ driver was
compiled directly against the generated pySecDec sector headers in the
persistent `.pysecdec_build_double_box` artifact and run with 1,000,000 fixed
interior sample points on the Apple M3 Pro listed above.  The same-index
pySecDec `sector_50` is not the relevant comparison for the standalone PSD50
MRE: it only has three order kernels and very small generated source files.
The relevant raw-kernel reference is the slow generated pySecDec sector,
`sector_53`, which has the same five-order Laurent structure (`eps^-4..eps^0`)
and much larger generated source.

| pySecDec generated sector | generated order kernels | direct C++ timing [us / sector point] | note |
|---|---:|---:|---|
| `sector_53` | `eps^-4` through `eps^0` | 3.644 | relevant MRE reference and slowest pySecDec sector in the all-sector raw-kernel sweep |
| `sector_50` | `eps^-2`, `eps^-1`, `eps^0` | 0.102 | same numeric index only; too simple for the PSD50 MRE comparison |

The pySecDec all-sector raw-kernel sweep over the same artifact reported 96
generated sectors and 302 sector/order kernels.  Grouping order kernels by
sector gave min/median/average/max sector-point timings of
`0.011 / 0.157 / 0.538 / 3.716 us`, with `sector_53` as the slowest sector.
At 100,000 points the CLI benchmark reproduces the same hierarchy, reporting
slowest `sector_53 = 3.65 us`; the larger 1,000,000-point run is the number
used in the table above.

The previously slow explicit FSD sector in this comparison was `PSD50`.  It was
remeasured separately with 1000 f64 samples on the local community/dev
Symbolica wheel:

| `PSD50` backend | evaluator generation [s] | sector-processor runtime [us/sample] | Symbolica eval share |
|---|---:|---:|---:|
| eager | 14.105 | 43.33 | 99.37% |
| JIT, real API requested | 13.121 | 47.73 | 99.45% |
| JIT with heavier optimizer env knobs | 13.136 | 48.57 | 99.87% |
| assembly `--compile` | 376.557 | 7.92 | 99.34% |

The direct JIT evaluator call, bypassing the rest of the sector processor, is
stable at about `14.7..15.3 us/sample` for batch sizes `16..8192`; batch size
1 costs about `23.6 us/sample`.  The raw real-valued JIT path is currently
known to return wrong values on the standalone MRE expression set; FSD does not
silently route it through the complex evaluator anymore.  Use
`--complex-evaluator` explicitly when checking the current correctness
workaround.  The larger `~48 us/sample` benchmark value is therefore mostly
the full FSD sector-processor route around the evaluator, not Python arithmetic
in the evaluator itself.

The standalone MRE can print the recorded pySecDec reference line and can also
run the same generated-kernel benchmark when the persistent pySecDec package is
available:

```sh
.venv/bin/python MRE/MRE_poor_eager_performance.py \
  --modes eager jit-real jit-complex \
  --batch-sizes 1000000 \
  --repeats 1 \
  --run-pysecdec-kernel-benchmark \
  --pysecdec-kernel-sectors 53
```

An auxiliary sweep at `maxeval = 100, 300, 1000, 3000` remained nearly flat at
about 3.5 s per pySecDec call.  This shows that `integration wall time /
maxeval` is not a reliable per-sample metric for this double-box package:
pySecDec enforces a minimum-work floor (`mineval`/QMC refinement) and the
actual exposed work count is the verbose `n = 10061` per generated
sector/order integral.

The latest completed compressed triple-box bundle was generated with
pregenerated dual evaluators, IBP endpoint lowering, and no chain-rule formula
backend:

```sh
.venv/bin/python FSD.py generate \
  --dot-file examples/graphs/triple_box.dot \
  --kinematics examples/graphs/triple_box_kinematics.yaml \
  --output examples/outputs/prepared_triple_box_dual_stream_probe \
  --sector-method iterative \
  --prefactor-convention pysecdec \
  --subtraction-backend projector-formula \
  --ibp-reduce-to-log-endpoint \
  --direct-projector-cache-term-threshold 0 \
  --pregenerate-dual-evaluators \
  --regular-taylor-signature-limit 100000 \
  --regular-taylor-formula-volume-limit 100000 \
  --regular-taylor-formula-axis-limit 5 \
  --max-eps-order 0
```

Prepared triple-box artifact counts:

| artifact | count / size |
|---|---:|
| sectors | 1972 |
| endpoint-projector formulas | 360 |
| regular-Taylor formulas | 166 |
| serialized evaluator files | 30572 |
| prepared bundle size | 27 GiB |
| generated top-level cache size | 22 GiB |
| legacy asset cache size | 11 GiB |

Raw `.bin` evaluator sidecars were tested and rejected: they reduced
compression CPU cost but grew the partial prepared bundle to roughly 30 GiB
after only about 5000 streamed evaluator files.  Compressed sidecars remain the
practical prepared-bundle format.

The `--pregenerate-single-overall-dual-evaluator` probe was also rejected for
the triple box: it still prepared more than 1200 streamed evaluator artifacts
and had not completed after 10 minutes.  It did not solve the source-evaluator
preparation bottleneck.

## PSD2 Direct Formula Probe

`PSD2` is a six-axis triple-box sector with singular axes `[0,1,3,5,7,8]`.
The current compressed bundle evaluates it through the sparse fallback for the
regular source algebra.  Repeated one-point timings separate cold evaluator
loading from warm steady state:

| path | repeat set | wall [s] | Symbolica eval [s] | Python/glue [s] |
|---|---|---:|---:|---:|
| sparse fallback | cold repeat 0 | 10.74 | 9.59 | 1.15 |
| sparse fallback | warm median repeats 2..4 | 1.15 | 0.760 | 0.390 |
| injected direct regular formulas | preparation repeat 0 | 69.57 | 20.58 | 48.99 |
| injected direct regular formulas | warm median repeats 2..5 | 10.58 | 9.97 | 0.612 |

The direct formula probe injected the 8 unique six-axis regular formula
signatures needed by PSD2.  The formulas themselves are universal and small in
count, but the direct path required thousands of source dual shapes and many
separate evaluator calls.  It reduced Python time but made total runtime much
worse.  This is the clearest evidence that the next optimization must fuse the
regular-source computation rather than creating many standalone coefficient
evaluators.

## Cache Strategy

Empty-cache generation time is no longer treated as the main user experience.
The intended distribution model is:

1. Build universal formula caches offline, potentially on a cluster.
2. Ship/download the cache under top-level `cache/`.
3. Generate topology-specific prepared bundles from that cache.
4. Run strict `integrate --output ...` with no pySecDec or evaluator
   generation.

The current local cache already shows the scale: tens of GiB, not source-repo
size.  That is acceptable for an optional downloaded cache archive.

## Practical Conclusion

The prepared DOT bundle path works for triangle, box, double box, and the full
triple-box sector list.  Strict prepared integration performs no pySecDec work
and no evaluator generation.  The remaining performance weak point is
high-axis source assembly for the triple box.  Blindly moving every sparse
coefficient into separate Symbolica formula evaluators is not sufficient; the
needed improvement is a coarser fused evaluator/source path or a native
Symbolica sparse-series primitive.
