# PSD2 Runtime Experiment

This folder is a standalone experiment for sector `PSD2` of the massless Euclidean three-loop triple-box DOT example.  The sector data in `inputs/psd2_sector.json` hard-codes the PSD2 map, monomials, and the topology-level `U` and `F` polynomials as Symbolica expression strings.

The point of the experiment is to compare two deliberately different implementations of the same sector integrand:

- **FSD-style:** use the prepared FSD bundle and evaluate `PSD2` through the black-box `SectorProcessor`.  `U` and `F` stay behind evaluator calls; regular Taylor/source assembly and endpoint projector algebra are staged.
- **Fused style:** explicitly substitute the PSD2 map into `U` and `F`, build the full IBP-lowered endpoint subtraction as Symbolica expressions, and lower one Symbolica evaluator per selected Laurent coefficient.  This intentionally violates the FSD black-box boundary and estimates the pySecDec-style runtime ceiling for this sector.

## Sector Summary

`PSD2` has nine sector variables

```text
x1, x2, x3, x4, x5, x6, x7, x8, x9
```

with sector map

```text
(1, x1, x2*x1, x2*x3*x1, x2*x4*x1,
 x2*x5*x4*x1, x2*x6*x4*x1,
 x2*x6*x9*x4*x8*x7, x2*x8*x1, x2*x9*x4*x1).
```

The declared monomial data is:

| source | powers in sector variables |
| --- | --- |
| `F` monomial | `(3, 3, 0, 2, 0, 1, 0, 1, 1)` |
| `U` monomial | `(2, 2, 0, 1, 0, 0, 0, 0, 0)` |
| Jacobian monomial | `(7, 7, 0, 4, 0, 1, 0, 1, 1)` |
| singular axes | `(0, 1, 3, 5, 7, 8)` |
| endpoint Taylor orders | `(0, 0, 1, 2, 2, 2)` on the singular axes |

Combining these powers with the three-loop parametric exponents gives the IBP-lowered endpoint signature

```text
y0^(-1-eps) y1^(-1-eps) y3^(-2-2 eps)
y5^(-3-3 eps) y7^(-3-3 eps) y8^(-3-3 eps).
```

The IBP lowering expands this into 54 logarithmic child-projector terms.  The fully active last term is the expensive one:

```text
boundary = ()
derivative multi-index = (0, 0, 1, 2, 2, 2)
active singular positions = (0, 1, 2, 3, 4, 5)
```

## Commands

The full fused expressions were generated once, without lowering evaluators:

```bash
.venv/bin/python run_with_memory_watch.py \
  --limit-gb 30 \
  --timeout-seconds 1900 \
  --poll-seconds 15 \
  --log-file PSD2_runtime_experiment/fused_expr_only_plain_watch.log \
  -- \
  .venv/bin/python PSD2_runtime_experiment/psd2_runtime_experiment.py \
    --skip-fsd \
    --skip-fused-evaluator-build \
    --fused-max-build-seconds 1800
```

The first-five comparison, excluding `eps^-1` and `eps^0`, was then run from the saved expression artifacts:

```bash
.venv/bin/python run_with_memory_watch.py \
  --limit-gb 30 \
  --timeout-seconds 600 \
  --poll-seconds 5 \
  --log-file PSD2_runtime_experiment/compare_first5_fixed_sample_clean_watch.log \
  -- \
  .venv/bin/python PSD2_runtime_experiment/psd2_runtime_experiment.py \
    --fsd-max-eps-order -2 \
    --load-fused-expressions \
    --fused-evaluator-orders -6 -5 -4 -3 -2 \
    --repeats 4 \
    --points 1 \
    --sample 0.37 0.42 0.58 0.63 0.29 0.51 0.74 0.68 0.46 \
    --evaluator-cpe-iterations 1 \
    --evaluator-max-horner-vars 128 \
    --evaluator-n-cores 1 \
    --results-json PSD2_runtime_experiment/results_first5_fsd_truncated.json
```

The `--fsd-max-eps-order -2` option narrows the prepared-bundle FSD path to the same active Laurent range as the fused path.  The prepared bundle still contains formula artifacts through `eps^0`, but the runtime path evaluates only active prepared projector outputs and only requests regular-source coefficients needed for `eps^-6..eps^-2`.

Generated artifacts and JSON/log outputs are ignored by git because the full fused expression cache is large and reproducible.

The FSD-only timing was also checked in separate full-range and truncated-range runs with the same fixed sample point and no fused work:

```bash
.venv/bin/python PSD2_runtime_experiment/psd2_runtime_experiment.py \
  --skip-fused \
  --repeats 8 \
  --points 1 \
  --sample 0.37 0.42 0.58 0.63 0.29 0.51 0.74 0.68 0.46 \
  --results-json PSD2_runtime_experiment/results_fsd_full_probe.json

.venv/bin/python PSD2_runtime_experiment/psd2_runtime_experiment.py \
  --skip-fused \
  --fsd-max-eps-order -2 \
  --repeats 8 \
  --points 1 \
  --sample 0.37 0.42 0.58 0.63 0.29 0.51 0.74 0.68 0.46 \
  --results-json PSD2_runtime_experiment/results_fsd_truncated_probe.json
```

An intermediate source-coefficient fused path was also tested.  This keeps the
U/F boundary intact: regular endpoint Taylor coefficients are acquired from the
prepared black-box machinery, then a single Symbolica evaluator receives those
coefficients and performs all IBP child-projector algebra and Laurent assembly:

```bash
.venv/bin/python PSD2_runtime_experiment/psd2_runtime_experiment.py \
  --skip-fused \
  --run-source-fused \
  --fsd-max-eps-order -2 \
  --source-fused-max-eps-order -2 \
  --repeats 4 \
  --points 1 \
  --sample 0.37 0.42 0.58 0.63 0.29 0.51 0.74 0.68 0.46 \
  --evaluator-cpe-iterations 1 \
  --evaluator-max-horner-vars 128 \
  --evaluator-n-cores 1 \
  --results-json PSD2_runtime_experiment/results_source_fused_first5_probe.json
```

and with a small batch:

```bash
.venv/bin/python PSD2_runtime_experiment/psd2_runtime_experiment.py \
  --skip-fused \
  --run-source-fused \
  --fsd-max-eps-order -2 \
  --source-fused-max-eps-order -2 \
  --repeats 3 \
  --points 8 \
  --evaluator-cpe-iterations 1 \
  --evaluator-max-horner-vars 128 \
  --evaluator-n-cores 1 \
  --results-json PSD2_runtime_experiment/results_source_fused_first5_batch8_probe.json
```

Finally, a true two-evaluator path was tested.  It uses:

1. one Symbolica evaluator that computes all `3840` regular Taylor source
   coefficients needed by PSD2 for `eps^-6..eps^-2`;
2. one Symbolica evaluator that consumes those coefficients and directly
   returns the five Laurent coefficients.

In this standalone experiment the first evaluator is built from the explicit
PSD2 U/F expressions.  In an FSD implementation this is the slot that should be
filled by a generated black-box U/F derivative/source evaluator.

```bash
.venv/bin/python run_with_memory_watch.py \
  --limit-gb 30 \
  --poll-seconds 5 \
  --log-file PSD2_runtime_experiment/two_stage_first5_watch.log \
  -- \
  .venv/bin/python PSD2_runtime_experiment/psd2_runtime_experiment.py \
    --skip-fused \
    --run-two-stage-fused \
    --fsd-max-eps-order -2 \
    --two-stage-max-eps-order -2 \
    --repeats 3 \
    --points 1 \
    --sample 0.37 0.42 0.58 0.63 0.29 0.51 0.74 0.68 0.46 \
    --evaluator-cpe-iterations 1 \
    --evaluator-max-horner-vars 128 \
    --evaluator-n-cores 1 \
    --results-json PSD2_runtime_experiment/results_two_stage_first5_probe.json
```

Two additional generic-path probes were run after this:

```bash
.venv/bin/python run_with_memory_watch.py \
  --limit-gb 30 \
  --poll-seconds 5 \
  --log-file PSD2_runtime_experiment/dual_envelope_first5_watch.log \
  -- \
  .venv/bin/python PSD2_runtime_experiment/psd2_runtime_experiment.py \
    --skip-fused \
    --run-dual-envelope-source \
    --fsd-max-eps-order -2 \
    --dual-envelope-max-eps-order -2 \
    --repeats 2 \
    --points 1 \
    --sample 0.37 0.42 0.58 0.63 0.29 0.51 0.74 0.68 0.46 \
    --evaluator-cpe-iterations 1 \
    --evaluator-max-horner-vars 128 \
    --evaluator-n-cores 1 \
    --results-json PSD2_runtime_experiment/results_dual_envelope_first5_probe.json
```

and a cold `symbolic-derivatives` processor probe was stopped after about ten minutes while it was still building universal chain-rule formulae:

```bash
.venv/bin/python run_with_memory_watch.py \
  --limit-gb 30 \
  --poll-seconds 5 \
  --log-file PSD2_runtime_experiment/symbolic_processor_first5_watch.log \
  -- \
  .venv/bin/python - <<'PY'
# Instantiate PSD2 with dual_evaluator_mode='symbolic-derivatives',
# prepare endpoint, regular-Taylor, and chain-rule formulae, then evaluate.
PY
```

## Results

Full fused expression generation completed under the 30 GB guard:

| quantity | value |
| --- | ---: |
| full expression generation wall time | `1339.5 s` |
| symbolic expression construction time | `1191.1 s` |
| full compressed expression footprint | `173.1 MB` |
| peak observed RSS | `< 30 GB` |

Expression footprint by Laurent order:

| order | compressed text | compressed binary expression | raw text size |
| --- | ---: | ---: | ---: |
| `eps^-6` | `287 B` | `2.0 KB` | `1.8 KB` |
| `eps^-5` | `6.5 KB` | `10 KB` | `108 KB` |
| `eps^-4` | `101 KB` | `94 KB` | `2.30 MB` |
| `eps^-3` | `949 KB` | `701 KB` | `25.4 MB` |
| `eps^-2` | `5.4 MB` | `4.0 MB` | `172 MB` |
| `eps^-1` | `21 MB` | `18 MB` | `786 MB` |
| `eps^0` | `62 MB` | `53 MB` | `2.60 GB` |

First-five evaluator lowering and evaluation:

| implementation | coefficients | one-time evaluator setup | eval wall per sample |
| --- | --- | ---: | ---: |
| FSD-style prepared bundle | full `eps^-6..eps^0` | already prepared in bundle | `1.419 s` |
| FSD-style evaluator part | full `eps^-6..eps^0` | already prepared in bundle | `0.983 s` |
| FSD-style Python part | full `eps^-6..eps^0` | already prepared in bundle | `0.438 s` |
| FSD-style prepared bundle | only `eps^-6..eps^-2` | already prepared in bundle | `1.390 s` |
| FSD-style evaluator part | only `eps^-6..eps^-2` | already prepared in bundle | `0.984 s` |
| FSD-style Python part | only `eps^-6..eps^-2` | already prepared in bundle | `0.402 s` |
| source-coefficient fused total | only `eps^-6..eps^-2` | `0.112 s` | `1.275 s` |
| source-coefficient acquisition | only `eps^-6..eps^-2` | included above | `1.275 s` |
| source-coefficient fused assembler | only `eps^-6..eps^-2` | included above | `45 us` |
| two-evaluator source+assembler | only `eps^-6..eps^-2` | `733.5 s` | `27.4 ms` |
| two-evaluator source evaluator | only `eps^-6..eps^-2` | included above | `27.1 ms` |
| two-evaluator assembler | only `eps^-6..eps^-2` | included above | `0.14 ms` |
| fused Symbolica evaluator | only `eps^-6..eps^-2` | `12.29 s` | `1.90 ms` |

The selected fused evaluator artifacts for `eps^-6..eps^-2` occupy `2.49 MB`; the selected fused expression artifacts occupy `5.05 MB`.

At the fixed interior sample point, the two implementations agree well:

| order | fused value | fused - FSD-style |
| --- | ---: | ---: |
| `eps^-6` | `2.332160301583e-03` | `-1.82e-16` |
| `eps^-5` | `-3.523577879123e-03` | `5.50e-16` |
| `eps^-4` | `-3.090948131190e-01` | `-6.44e-15` |
| `eps^-3` | `6.855016565023e-01` | `-1.63e-13` |
| `eps^-2` | `9.415475415132e-01` | `4.46e-11` |

The fused first-five evaluator is about `730x` faster than the intended truncated FSD-style wall time for this one-point PSD2 test, and about `516x` faster than the truncated FSD-style Symbolica evaluator time alone.  This speedup is not free: the full fused expressions took roughly 22 minutes to build for a single sector and the finite-part expression is very large.

The full-range and truncated FSD timings are deliberately close, but they are not expected to be identical.  Dropping `eps^-1` and `eps^0` removes two output coefficients and some regular-source columns, but it does not remove the dominant source of work for PSD2: the six-axis IBP endpoint structure already requires the high-order boundary/source envelopes needed by the leading coefficients.  In particular, the expensive child with derivative multi-index `(0, 0, 1, 2, 2, 2)` is already needed before the finite part.  The current black-box FSD path therefore pays most of the U/F Taylor-source and child-projector assembly cost even when the requested range stops at `eps^-2`; the observed reduction is mostly in Python-side input assembly.

The source-coefficient fused experiment confirms the same diagnosis.  The
single assembler evaluator has `3849` inputs, of which `3840` are regular
Taylor coefficients.  It builds in `0.066 s`, lowers in `0.047 s`, and evaluates
in about `45 us` for one point.  The total time remains around `1.28 s/sample`
because acquiring the regular Taylor coefficients still costs `1.275 s/sample`
in this standalone path.  It agrees with the FSD-style coefficients at the
fixed sample with max difference `3.7e-14`.

For a batch of 8 points, the same pattern holds:

| implementation | coefficients | wall/sample | evaluator/sample | Python/sample |
| --- | --- | ---: | ---: | ---: |
| FSD-style prepared bundle | `eps^-6..eps^-2` | `0.954 s` | `0.880 s` | `0.0736 s` |
| source-coefficient fused | `eps^-6..eps^-2` | `1.247 s` | `1.175 s` | `0.0724 s` |
| source-coefficient fused assembler only | `eps^-6..eps^-2` | `32 us` | `32 us` | n/a |

So this middle-ground fusion removes the final projector-assembly overhead, but
that overhead is already small after batching.  The large gain only appears in
the fully fused pySecDec-style evaluator because that also bakes in the
regular-source/chain-rule construction that FSD currently keeps outside the
final evaluator.

The true two-evaluator path changes that conclusion substantially.  Its source
evaluator is expensive to generate for this standalone PSD2 test:

| quantity | value |
| --- | ---: |
| source coefficient outputs | `3840` |
| source expression build | `636.9 s` |
| source evaluator lowering | `96.6 s` |
| source expression raw text footprint | `2.45 GB` |
| source evaluator bytes | `47.1 MB` |
| assembler expression build | `0.070 s` |
| assembler evaluator lowering | `0.047 s` |
| assembler evaluator bytes | `123 KB` |

But its runtime is much closer to the fully fused pySecDec-style ceiling:

| implementation | first-five runtime/sample | relative to two-stage |
| --- | ---: | ---: |
| FSD-style prepared bundle | `1.48 s` in this run | `54x` slower |
| two-evaluator source+assembler | `27.4 ms` | reference |
| fully fused pySecDec-style evaluator | `1.90 ms` | `14x` faster |

The two-evaluator output agrees with FSD-style at the fixed sample with max
difference `4.4e-11`.  Python overhead is essentially gone: `0.16 ms/sample`.
The remaining `27.1 ms/sample` is almost entirely the first source evaluator
call.  This supports the direction suggested in the discussion: the useful FSD
runtime target is not a Python loop over source coefficients, but a generated
source evaluator plus a generated assembler evaluator.  The open engineering
question is how to generate and cache the source evaluator generically without
creating multi-GB parseable expressions for every hard sector.

The first attempt at a fully black-box two-call variant used one global
dualized envelope for all endpoint groups.  It is correct but not viable:

| quantity | value |
| --- | ---: |
| active Laurent range | `eps^-6..eps^-2` |
| endpoint groups | `324` |
| envelope Taylor columns | `4096` |
| scalar/evaluator setup | `0.001 s` |
| envelope shape construction | `0.302 s` |
| U/F dual evaluator preparation | `13.1 s` |
| runtime/sample | `25.7 s` |
| Symbolica evaluator time/sample | `25.0 s` |
| Python time/sample | `0.693 s` |
| max difference vs FSD-style | `5.1e-14` |

This failed for a useful reason.  A single envelope makes generation simple, but
it forces Symbolica to evaluate a large 4096-coefficient jet at every one of the
324 endpoint rows.  The better black-box path must keep sparse per-signature
source shapes, not one maximal envelope.

The current strict symbolic-derivative processor path is closer to the desired
architecture: U/F are represented by derivative evaluators, while chain-rule
and regular-source algebra are universal formulae.  A one-endpoint-group
standalone diagnostic already showed sub-millisecond runtime:

| diagnostic | value |
| --- | ---: |
| endpoint group | `boundary=(), zero=()` |
| derivative slots | `60` |
| assembler expression build | `13.6 s` |
| runtime/sample | `0.707 ms` |
| source evaluator/sample | `0.689 ms` |
| assembler/sample | `2.1 us` |

However, building the full cold chain-rule formula set for PSD2 was stopped
after roughly `626 s` under the 30 GB watchdog while still in Python
expression-series accumulation.  This does not invalidate the architecture,
because these chain-rule formulae are topology-independent by signature, but it
does mean they must be generated offline into the universal cache.  They are
not acceptable as ordinary per-run generation work.

For the full `eps^-6..eps^0` range, the reliable current reference remains the
prepared-bundle FSD path:

| implementation | coefficients | wall/sample | evaluator/sample | Python/sample |
| --- | --- | ---: | ---: | ---: |
| FSD-style prepared bundle | `eps^-6..eps^0` | `1.44 s` | `1.06 s` | `0.389 s` |

The source-coefficient fused assembler was also run through `eps^0`.  It builds
quickly (`0.265 s` expression construction and `0.265 s` evaluator lowering)
and uses only `4800` coefficient inputs.  An older local cache entry made the
standalone assembler appear to disagree with the FSD reference for `eps^-1` and
`eps^0`; that cache file had the right signature payload but only a truncated
Laurent expression payload.  The cache loader now rejects such stale projector
entries by checking both the Laurent order list and the number of output
expressions.

The direct two-evaluator source+assembler path was then pushed through
`eps^0` as a generation feasibility test:

```bash
.venv/bin/python run_with_memory_watch.py \
  --limit-gb 30 \
  --poll-seconds 10 \
  --log-file PSD2_runtime_experiment/two_stage_full_watch.log \
  -- \
  .venv/bin/python PSD2_runtime_experiment/psd2_runtime_experiment.py \
    --skip-fsd \
    --skip-fused \
    --run-two-stage-fused \
    --two-stage-max-eps-order 0 \
    --repeats 1 \
    --points 1 \
    --sample 0.37 0.42 0.58 0.63 0.29 0.51 0.74 0.68 0.46 \
    --evaluator-cpe-iterations 1 \
    --evaluator-max-horner-vars 128 \
    --evaluator-n-cores 1 \
    --results-json PSD2_runtime_experiment/results_two_stage_full_probe.json
```

Generation completed under the 30 GB watchdog:

| quantity | value |
| --- | ---: |
| source coefficient outputs | `4800` |
| source expression build | `882.5 s` |
| source evaluator lowering | `133.7 s` |
| source expression raw text footprint | `4.06 GB` |
| source evaluator bytes | `73.9 MB` |
| assembler expression build | `0.272 s` |
| assembler evaluator lowering | `0.257 s` |
| assembler evaluator bytes | `428 KB` |
| runtime/sample | `29.6 ms` |
| source evaluator/sample | `24.5 ms` |
| assembler/sample | `4.79 ms` |
| Python/sample | `0.234 ms` |

So the answer to the generation question is yes: the direct two-evaluator PSD2
experiment can be generated all the way to `eps^0` on this machine within the
30 GB guard, taking about `16.9 min` for expression construction plus evaluator
lowering.  With the stale-cache guard in place, the cached two-stage evaluator
agrees with a fresh FSD-style sector-processor evaluation at the fixed interior
sample point through the finite part:

| order | two-evaluator full-range diff vs fresh FSD |
| --- | ---: |
| `eps^-6` | `-2.8e-16` |
| `eps^-5` | `-1.9e-15` |
| `eps^-4` | `-1.4e-14` |
| `eps^-3` | `-6.8e-14` |
| `eps^-2` | `4.4e-11` |
| `eps^-1` | `-2.2e-10` |
| `eps^0` | `1.2e-9` |

## Saved Evaluator Cache

The PSD2 steering script now has an explicit evaluator cache interface:

```bash
--save-evaluators-to PSD2_runtime_experiment/artifacts/two_stage_full_eps0
--load-evaluators-from PSD2_runtime_experiment/artifacts/two_stage_full_eps0
```

The underscore spellings are also accepted:

```bash
--save_evaluators_to ...
--load_evaluators_from ...
```

For the two-stage source+assembler experiment this saves two independent
Symbolica evaluator artifacts, keyed by the requested maximum epsilon order:

| artifact | role |
| --- | --- |
| `source_epsmax_0.bin.gz` | evaluator A, producing all regular Taylor/source coefficients |
| `source_epsmax_0.json` | evaluator A metadata and coefficient-key layout |
| `assembler_epsmax_0.bin.gz` | evaluator B, assembling the Laurent coefficients from source inputs |
| `assembler_epsmax_0.json` | evaluator B metadata, input names, Laurent orders, and coefficient-key layout |

The loader verifies the sector id, requested maximum epsilon order, and the full
coefficient-key layout before reusing an evaluator.  This is deliberately strict:
a stale source evaluator with the wrong coefficient order would otherwise give a
numerically plausible but meaningless result.

A cache load check was run with:

```bash
.venv/bin/python PSD2_runtime_experiment/psd2_runtime_experiment.py \
  --skip-fsd \
  --skip-fused \
  --run-two-stage-fused \
  --two-stage-max-eps-order 0 \
  --load-evaluators-from PSD2_runtime_experiment/artifacts/two_stage_full_eps0 \
  --points 1 \
  --repeats 1 \
  --sample 0.37 0.42 0.58 0.63 0.29 0.51 0.74 0.68 0.46 \
  --results-json PSD2_runtime_experiment/results_two_stage_loaded_check.json
```

Both evaluator stages were loaded from disk.  No expression generation or
evaluator lowering was performed in this check:

| quantity | loaded-cache value |
| --- | ---: |
| source coefficient outputs | `5376` |
| source evaluator cache | `loaded` |
| assembler evaluator cache | `loaded` |
| total wall time/sample | `20.31 ms` |
| evaluator time/sample | `20.27 ms` |
| Python time/sample | `37.7 us` |
| source evaluator/sample | `20.13 ms` |
| assembler evaluator/sample | `0.146 ms` |

This is the intended future cache-recycling shape for FSD: expensive evaluator
construction can be done once, while later runtime experiments bootstrap
directly from serialized evaluator bytes plus validated layout metadata.

## Current FSD-Style Versus pySecDec-Style Timing

The latest candidate to port back into FSD is the two-evaluator
source+assembler layout:

```text
evaluator A: sector coordinates -> regular Taylor/source coefficients
evaluator B: sector coordinates + source coefficients -> Laurent coefficients
```

The PSD2 standalone implementation still hard-codes PSD2-specific source
expressions, so it is a proxy for the intended generic FSD construction rather
than the final implementation.  Its important property is the runtime structure:
the final sector evaluation is two Symbolica evaluator calls and almost no
Python arithmetic.

For the full `eps^-6..eps^0` range, the latest saved two-stage cache has:

| item | value |
| --- | ---: |
| Laurent coefficients | `7` |
| source coefficient outputs | `5376` |
| source expression construction | `1454.09 s` |
| source evaluator lowering | `184.89 s` |
| assembler expression construction | `0.323 s` |
| assembler evaluator lowering | `0.356 s` |
| total one-time generation | `1639.65 s` (`27.33 min`) |
| source evaluator artifact | `77.6 MB` raw, `27.0 MB` gzipped |
| assembler evaluator artifact | `179 KB` gzipped |
| double runtime/sample | `20.31 ms` |
| source evaluator/sample | `20.13 ms` |
| assembler evaluator/sample | `0.146 ms` |
| Python overhead/sample | `37.7 us` |

The same evaluator objects support precision escalation without regeneration.
At the fixed interior point:

| precision | source evaluator | assembler evaluator | total evaluator time |
| --- | ---: | ---: | ---: |
| double | `20.1 ms` | `0.15 ms` | `20.3 ms` |
| `prec=32` | `266 ms` | `8.3 ms` | `274 ms` |
| `prec=1000` | `14.7 s` | `0.10 s` | `14.8 s` |

For comparison, the explicit pySecDec-style fused PSD2 path substitutes the PSD2
sector map into `U` and `F` and bakes the whole sector integrand into direct
Laurent-coefficient evaluators.  That path is not the FSD architecture, but it
is the useful upper-performance reference:

| implementation | coefficient range | one-time generation/evaluator setup | runtime/sample |
| --- | --- | ---: | ---: |
| current staged FSD processor | `eps^-6..eps^0` | prepared bundle already loaded | `12.80 s` cold fixed sample |
| current staged FSD processor | `eps^-6..eps^-2` | prepared bundle already loaded | `1.26 s` warm median |
| FSD-style two evaluator | `eps^-6..eps^0` | `27.33 min` | `20.31 ms` |
| FSD-style two evaluator | `eps^-6..eps^-2` | `733.5 s` | `27.4 ms` |
| pySecDec-style fused evaluator | `eps^-6..eps^-2` | full expression cache `1339.5 s`; selected evaluator lowering `12.29 s` | `1.90 ms` |

So for PSD2, the two-evaluator FSD-style path is now roughly:

```text
~630x faster than the current staged FSD processor through eps^0,
~14x slower than the fully fused pySecDec-style evaluator for eps^-6..eps^-2.
```

The pySecDec-style finite-part expression exists in the artifact cache
(`eps^0` is `62 MB` compressed text / `53 MB` compressed binary expression /
`2.60 GB` raw text), but a finite-part fused evaluator has not yet been lowered
successfully in this experiment.  Therefore the only measured pySecDec-style
runtime comparison is the validated first-five range `eps^-6..eps^-2`.

## Arbitrary-Precision Endpoint Probe

The saved full-range two-stage evaluator was also used to probe the finite part
with Symbolica's arbitrary-precision evaluator path.  The same evaluator objects
were reused; only the input payloads were promoted to padded `Decimal` values and
the calls were routed through `evaluate_complex_with_prec(..., prec)`.

At the fixed interior point, double precision, 32 digits, and 1000 digits all
agree through `eps^0`:

| order | double | `prec=32` | `prec=1000` | `|double-1000|` |
| --- | ---: | ---: | ---: | ---: |
| `eps^-6` | `2.332160301583e-03` | `2.332160301583e-03` | `2.332160301583e-03` | `2.3e-16` |
| `eps^-5` | `-3.523577879126e-03` | `-3.523577879124e-03` | `-3.523577879124e-03` | `1.1e-15` |
| `eps^-4` | `-3.090948131190e-01` | `-3.090948131190e-01` | `-3.090948131190e-01` | `3.6e-15` |
| `eps^-3` | `6.855016565024e-01` | `6.855016565024e-01` | `6.855016565024e-01` | `4.3e-14` |
| `eps^-2` | `9.415475415127e-01` | `9.415475415130e-01` | `9.415475415130e-01` | `2.5e-13` |
| `eps^-1` | `3.804078661884e+00` | `3.804078661886e+00` | `3.804078661886e+00` | `1.4e-12` |
| `eps^0` | `-3.990081844668e+00` | `-3.990081844662e+00` | `-3.990081844662e+00` | `6.1e-12` |

The hard stability test is the simultaneous six-axis corner where all singular
coordinates are set to the same small number and nonsingular coordinates are
kept at the fixed interior values.  This is a stress test for cancellation, not
a representative Monte Carlo point:

| singular-axis value | `eps^0`, double or 32 digits | `eps^0`, `prec=1000` | conclusion |
| --- | ---: | ---: | --- |
| `1e-3` | double `-7.21e4`; 32 digits `-4.235e4` | `-4.235e4` | double unstable, 32 digits enough |
| `1e-4` | double `-5.23e10`; 32 digits `-1.2064e5` | `-1.2064e5` | double catastrophic, 32 digits enough |
| `1e-5` | 32 digits `-2.348373e5` | `-2.348380e5` | 32 digits still usable |
| `1e-6` | 32 digits `4.590e6` | `-3.628e5` | 32 digits no longer enough |
| `1e-8` | 32 digits `-1.537e18` | `-4.792e5` | 1000-digit rescue required |

The important observation is that the 1000-digit finite part grows moderately
as the six-axis corner is approached.  That is consistent with logarithmic
endpoint remnants after plus-distribution subtraction.  The double-precision
and insufficient-precision failures instead show apparent power-like blow-ups,
which disappear once the source coefficients and the subtraction assembler are
both evaluated at high precision.

A more explicit scaling check evaluated the same simultaneous six-axis corner
over several decades with `prec=1000`:

| `delta` on all singular axes | `log(1/delta)` | `eps^0` at `prec=1000` | `delta * eps^0` |
| ---: | ---: | ---: | ---: |
| `1e-2` | `4.605` | `-5.496e3` | `-5.496e1` |
| `1e-3` | `6.908` | `-4.235e4` | `-4.235e1` |
| `1e-4` | `9.210` | `-1.206e5` | `-1.206e1` |
| `1e-5` | `11.513` | `-2.348e5` | `-2.348e0` |
| `1e-6` | `13.816` | `-3.628e5` | `-3.628e-1` |
| `1e-7` | `16.118` | `-4.643e5` | `-4.643e-2` |
| `1e-8` | `18.421` | `-4.792e5` | `-4.792e-3` |

This is the strongest current evidence that the finite part is properly
subtracted.  If a power endpoint singularity such as `1/delta` had survived,
`delta * eps^0` would approach a nonzero constant rather than dropping by four
orders of magnitude across this scan.  Stronger missed powers would be even more
obvious.  The remaining growth is compatible with logarithmic endpoint remnants
from the plus-distribution expansion.

Single-axis endpoint probes are much less severe.  Setting one singular
coordinate at a time to `1e-6`, while keeping the other singular coordinates at
the interior point, gives double-precision `eps^0` values agreeing with the
1000-digit result at roughly `1e-3` absolute or better, and 32 digits agrees
with 1000 digits at the displayed precision.  The problematic case is therefore
the simultaneous multi-axis corner, as expected for PSD2.

## Derivative-Symbol Generic Split Probe

A stricter generic split was added with:

```text
--run-derivative-fused
```

This builds evaluator A from stacked x-space U/F derivative values and evaluator
B from derivative symbols, sector-map Taylor algebra, residual monomial
division, U/F powers/logs, IBP projectors, and final Laurent assembly.  In other
words, evaluator B contains no U/F expression.

A one-endpoint-group diagnostic is viable:

| diagnostic | value |
| --- | ---: |
| endpoint group | `boundary=(), zero=()` |
| derivative slots | `60` |
| assembler expression build | `13.6 s` |
| runtime/sample | `0.707 ms` |
| source evaluator/sample | `0.689 ms` |
| assembler/sample | `2.1 us` |

The second endpoint group, `boundary=(), zero=(0,)`, exposes the current
blocker.  The naive derivative-symbol assembler spent more than `90 s` inside
the Python sparse expression-series multiplication for the regular
`prefactor * log_power` expansion before it was stopped.  This is a generation
problem, not a runtime problem, and it is exactly the part that should become a
cached topology-independent formula rather than being rebuilt by a Python
series loop per sector/group.

This diagnostic is why the derivative-symbol builder has not been promoted to
the generic DOT path yet.  The validated fast route is:

```text
topology-specific source evaluator -> universal/factorized assembler evaluator
```

but the universal assembler must be built from reusable cached subformulae
instead of the current naive group-by-group expansion.

## Endpoint Stability Caveat

A random sample with one singular coordinate near `6e-4` showed that the fused double-precision expression can lose accuracy in `eps^-2`; the largest observed first-five discrepancy was `1.38`.  The fixed interior sample above avoids that cancellation and agrees to `4.5e-11`.

This is not surprising.  The fully fused expression evaluates the final algebra as one large rational/log expression in double precision.  The staged FSD-style construction keeps the regular Taylor coefficients and endpoint projector algebra separated, which is numerically safer near endpoint boundaries and can route near-boundary samples through higher precision.

## How To Include `eps^-1` And `eps^0`

The script already exposes the evaluator controls needed to continue the comparison.  `eps^-1` can be lowered with lower-memory evaluator settings by disabling direct translation:

```bash
.venv/bin/python run_with_memory_watch.py \
  --limit-gb 30 \
  --timeout-seconds 900 \
  --poll-seconds 2 \
  --log-file PSD2_runtime_experiment/lower_epsm1_nodirect_watch.log \
  -- \
  .venv/bin/python PSD2_runtime_experiment/psd2_runtime_experiment.py \
    --skip-fsd \
    --load-fused-expressions \
    --fused-evaluator-orders -1 \
    --evaluator-cpe-iterations 0 \
    --evaluator-max-horner-vars 0 \
    --evaluator-n-cores 1 \
    --evaluator-max-common-pair-cache-entries 10000 \
    --evaluator-max-common-pair-distance 20 \
    --no-evaluator-direct-translation
```

This completed locally with `eps^-1` evaluator lowering time `166.3 s` and a peak observed RSS below the 30 GB watchdog limit.  The same strategy is the next thing to try for `eps^0`:

```bash
.venv/bin/python run_with_memory_watch.py \
  --limit-gb 30 \
  --timeout-seconds 1800 \
  --poll-seconds 2 \
  --log-file PSD2_runtime_experiment/lower_eps0_nodirect_watch.log \
  -- \
  .venv/bin/python PSD2_runtime_experiment/psd2_runtime_experiment.py \
    --skip-fsd \
    --load-fused-expressions \
    --fused-evaluator-orders 0 \
    --evaluator-cpe-iterations 0 \
    --evaluator-max-horner-vars 0 \
    --evaluator-n-cores 1 \
    --evaluator-max-common-pair-cache-entries 10000 \
    --evaluator-max-common-pair-distance 20 \
    --no-evaluator-direct-translation
```

If that succeeds, a full comparison can be run by replacing the first-five order list with:

```text
--fused-evaluator-orders -6 -5 -4 -3 -2 -1 0
```

For `eps^-1` and especially `eps^0`, the most important tuning knobs are:

```text
--no-evaluator-direct-translation
--evaluator-cpe-iterations 0
--evaluator-max-horner-vars 0
--evaluator-n-cores 1
--evaluator-max-common-pair-cache-entries 10000
--evaluator-max-common-pair-distance 20
--evaluator-verbose
```

The finite expression is the hard case: its raw parseable text is about `2.60 GB`, so evaluator lowering may need more RAM, different Symbolica evaluator settings, or a split/factorized fused construction rather than one monolithic finite-part evaluator.

## Current Conclusion

For `eps^-6..eps^-2`, the fastest validated PSD2 implementation in this folder
is the direct two-evaluator source+assembler path:

```text
explicit PSD2 source evaluator -> universal endpoint assembler evaluator
```

It is runtime-acceptable for a path-finder benchmark: `27.4 ms/sample`, about
`14x` slower than the fully fused pySecDec-style evaluator and roughly `54x`
faster than the current prepared-bundle FSD implementation for this sector.
The result agrees with FSD at the fixed interior point to `4.4e-11`.

This implementation is not the final FSD design because its source evaluator
was built by explicitly substituting the PSD2 sector map into U/F.  The design
that should generalize is instead:

```text
shared U/F derivative evaluators
  -> cached universal chain-rule/source formulae
  -> cached universal endpoint assembler evaluator
```

The PSD2 probes support that direction but do not yet complete it:

- one global dual envelope is black-box and correct, but too slow at
  `25.7 s/sample`;
- the cold symbolic-derivative chain-rule builder is cacheable but too slow to
  run interactively for PSD2;
- the existing endpoint assembler is already cheap, `45 us/sample` for the
  first five coefficients and `0.45 ms/sample` through `eps^0`.

So the implementation to move into FSD is not the global-envelope path.  It is
a cache-backed symbolic-derivative/source path where the expensive universal
formulae are shipped or generated offline.  With that cache in place, the PSD2
one-group diagnostic suggests the source side can be in the millisecond regime;
without it, Python expression-series generation remains the bottleneck.
