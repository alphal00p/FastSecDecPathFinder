# FSD Performance Notes

These are development measurements for the DOT-backed FSD path as of
2026-06-22.  They are performance and stability diagnostics, not final
precision benchmarks.  FSD-owned code remains free of SciPy and SymPy imports;
pySecDec is used only at the DOT generation boundary.

## Environment

| item | value |
|---|---|
| machine | Darwin arm64 |
| Python | 3.12.6 |
| Symbolica | 2.0.0 local `symbolica-community` wheel patched to `symbolica/dev` |
| Symbolica dev commit | `07f1de5fc119b01e2875c8d0163b25eacabadf21` |
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
| local community/dev wheel | 0.000200 | 11.385 | 16.8x |

This removes the original several-minute U/F dualization bottleneck.  It does
not by itself make every high-axis regular-source formula fast: the cost has
moved to how much source algebra/evaluator fragmentation we ask Symbolica and
Python to perform.

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
| triangle | DOT | 3 | `eps^-2..eps^0` | 0.223 | 9.095 | avg 2.35 us/smpl/wkr |
| box | DOT | 12 | `eps^-2..eps^0` | 0.240 | 9.075 | avg 7.00 us/smpl/wkr |
| double box | DOT | 140 | `eps^-4..eps^0` | 0.615 | 272.81 | avg 18.23 us/smpl/wkr |
| triple box | DOT iterative | 1972 | `eps^-6..eps^0` | 38.46 recorded generation + 30.61 serialization | not completed | compressed prepared bundle, 30 GiB guard |

## QMC Integration Probe

FSD now has an experimental `--sampling-mode qmc` based on QMCPy's randomized
shifted rank-1 lattices.  FSD applies the Korobov periodizing map in vectorized
NumPy batches and then calls the usual batched Symbolica sector evaluators.
Each random shift is treated as one sector estimate; errors are estimated from
the shift-to-shift spread and combined across sectors in quadrature.

The one-loop DOT triangle was compared directly against pySecDec's generated
QMC backend using:

```sh
.venv/bin/python scripts/compare_qmc_pysecdec.py \
  --sample-counts 1024 4096 16384 \
  --qmc-shifts 16 \
  --workers 10 \
  --output-json /tmp/fsd_qmc_triangle_compare_latest.json
```

The reference target was the stored pySecDec-convention DOT-triangle target in
`examples/outputs/dot_triangle_pysecdec_target.json`.  The finite coefficient
is shown below.  FSD raw samples are `sectors * N/shift * shifts`; pySecDec
only exposes the public QMC `maxeval` budget through the pylink API, not the
same per-sector accounting.

| N/shift | FSD raw sector samples | FSD eps^0 diff | FSD eps^0 err | pySecDec maxeval | pySecDec eps^0 diff | pySecDec eps^0 err |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 49,152 | 8.696e-10 | 4.281e-9 | 16,384 | 7.783e-14 | 2.348e-14 |
| 4,096 | 196,608 | 2.675e-10 | 6.781e-9 | 65,536 | 7.561e-14 | 1.378e-14 |
| 16,384 | 786,432 | 4.026e-13 | 5.267e-9 | 262,144 | 6.650e-14 | 8.530e-15 |

This confirms that the FSD QMC implementation converges to the same
pySecDec-convention one-loop result.  It is not competitive with pySecDec on
this simple generated-C++ case: pySecDec's mature QMC implementation and
generated integrand are substantially more accurate at the same public lattice
sizes.  The next useful test is therefore the two-loop double box, where FSD's
generation/runtime trade-off is more relevant.

For the two-loop DOT double box, a prepared explicit FSD bundle was integrated
strictly from disk, so the timing below is runtime only:

```sh
.venv/bin/python FSD.py integrate \
  --output /tmp/fsd_prepared_double_box_explicit_qmc \
  --sampling-mode qmc \
  --qmc-shifts 128 \
  --samples-per-iter 1024 \
  --max-iter 1 \
  --batch-size 8192 \
  --workers 10 \
  --target examples/outputs/dot_double_box_pysecdec_target.json \
  --no-progress \
  --quiet-summary \
  --json \
  --result-path /tmp/fsd_qmc_double_explicit_1024x128_latest.json
```

This run used `140 * 1024 * 128 = 18,350,080` raw sector samples and completed
in `104.97 s` (`38.47 us/sample/worker`).  The same total sample budget with
`4096` points and only `32` random shifts had larger shift-estimated errors;
for this topology, more randomized shifts are a better short-run setting.

| order | FSD QMC | MC err | pySecDec target | pull |
|---|---:|---:|---:|---:|
| `eps^-4` | 3.167e-4 | 8.03e-4 | 0 | 0.39 |
| `eps^-3` | 1.48838 | 9.76e-3 | 1.50018 | 1.21 |
| `eps^-2` | 1.20705 | 5.64e-2 | 1.26841 | 1.09 |
| `eps^-1` | 2.85675 | 1.89e-1 | 2.99703 | 0.74 |
| `eps^0` | -14.8463 | 5.27e-1 | -14.8579 | 0.02 |

The precision-rescue fractions for this double-box QMC run were `83.93%`
ordinary f64, `13.16%` at 32 digits, `1.98%` at 100 digits, and `0.92%` at
1000 digits.  All five Laurent coefficients are within `1.3 sigma` of the
stored pySecDec target in this roughly two-minute run.

## Explicit Backend And Numerator Timing

The `--explicit` backend substitutes the sector maps into each sector
integrand and builds one multi-output Symbolica evaluator per sector.  This is
the pySecDec-like comparison path: it deliberately gives up the FSD black-box
U/F derivative construction in exchange for a faster runtime evaluator.

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
question is pySecDec-style runtime speed.  It is not the default FSD strategy,
because it relies on explicitly substituting sector maps into U/F, whereas the
projector backend preserves the path-finder goal of treating U/F as numerical
black-box evaluators.

## Scalar Double-Box Three-Way Timing

The scalar Euclidean double box was rerun on 2026-06-22 with a 30 GiB watchdog
and no wall-time timeout.  FSD runtimes use the `benchmark` subcommand with 5
ordinary f64 interior samples per sector.  pySecDec was run through
`--dot-engine pysecdec` with `--keep-pysecdec-workdir`, then the generated
shared library was loaded directly for an independent verbose timing pass.

| method | sectors / generated integrals | generation setup [s] | compile [s] | integration/runtime metric |
|---|---:|---:|---:|---|
| FSD projector | 140 sectors | 0.445 | n/a | avg 583.43 us/FSD-sector sample; median 254.71; min/max 18.21 / 4501.27 |
| FSD explicit | 140 sectors | 12.583 | n/a | avg 3.99 us/FSD-sector sample; median 2.54; min/max 1.49 / 18.27 |
| pySecDec generated | 302 sector/order integrals | 7.837 | 338.752 | 3.682 s recorded integration wall time |

The pySecDec direct-library cross-check is more informative than simply
dividing by `--pysecdec-maxeval`.  A verbose direct call reported 302
generated sector/order integrals, each with `n = 10061`, for 3,038,422
generated-integrand evaluations.  The measured wall time was 3.552 s, giving
about 1.17 us per generated sector/order integrand evaluation.  The summed
per-integral timings gave 3.426 s, or 1.13 us/evaluation.  Per-integral timing
spread was 0.33 to 11.78 us/evaluation, with a median of 0.56 us.

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
