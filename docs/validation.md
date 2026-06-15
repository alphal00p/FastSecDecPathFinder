# Validation Notes

This note records the current validation status of FSD on 2026-06-15.  The
goal is to keep the report topology-centric: commands and full metadata are
stored in the corresponding `result.json` files, while this document records
the main agreement and stability conclusions.

All coefficient values below use the selected display prefactor convention.
Errors are one-sigma Monte Carlo errors.

## Static Checks

```sh
.venv/bin/python -m pytest -q
rg "import (sympy|scipy)|from (sympy|scipy)" . -g'*.py'
```

| check | result |
|---|---:|
| `pytest -q` | 97 passed, 2 skipped |
| FSD-owned SciPy/SymPy import guard | no matches |

The skipped tests are optional generated-pySecDec comparisons and one
sandbox-sensitive multiprocessing regression guard.

## Overview

The triangle and box rows use DOT inputs, not the built-in shortcuts, so both
FSD and pySecDec generation times are comparable.

| topology | input | sectors | coefficients | target | FSD generation [s] | pySecDec generation [s] | avg FSD runtime |
|---|---|---:|---|---|---:|---:|---:|
| triangle | DOT | 3 | `eps^-2..eps^0` | pySecDec generated integrator | 0.223 | 9.095 | 2.35 us/smpl/wkr |
| box | DOT | 12 | `eps^-2..eps^0` | pySecDec generated integrator | 0.240 | 9.075 | 7.00 us/smpl/wkr |
| double box | DOT | 140 | `eps^-4..eps^0` | stored pySecDec-convention target | 0.615 | 272.81 | 18.23 us/smpl/wkr |
| triple box | DOT iterative | 1972 | `eps^-6..eps^-1` current capped run; `eps^-6..eps^0` supported | no pySecDec target completed | 5.582 generation smoke; 5.862 sampled leading-pole run; 56.11 capped six-order run | not completed | selected-sector probes plus 30 GiB capped all-sector probes |

For the triple box, the latest leading-pole all-sector generation smoke
finished in 5.582 s after the chain-rule pregeneration guard skipped the huge
active-sector request set.  That run included 0.179 s for U/F construction,
1.110 s for sector generation, and 4.293 s for Symbolica evaluator
preparation.  The current default prepares 147 regular Taylor signatures, uses
6 curated regular-Taylor source assets, skips 29 harder high-axis signatures,
and uses curated direct endpoint projectors for 382 sectors across 55
signatures.  The skipped regular-Taylor signatures are controlled by
`--regular-taylor-formula-axis-limit` and
`--regular-taylor-formula-volume-limit`.  This replaces the previous
all-or-nothing fallback and keeps all-sector generation below 10 s.  A
selected `PSD62` run showed why the axis cap is needed: trying to build eight
six-axis regular signatures still took about 226 s even with the v3 sparse
output signature, while the guarded run finished generation in about 1.5 s.
The complementary all-sector non-IBP path was re-probed with the warm cache and
still timed out after 180 s inside endpoint-projector formula preparation, so
the shipped triple-box run preset remains IBP-lowered.
The generated pySecDec triple-box package did not complete within the local
memory and wall-time limits used for this pathfinder run.

A current all-sector leading-pole run with 2000 samples and 10 workers completed
under the watchdog in 137.6 s after 5.862 s of generation.  It hit 1261 of 1972
sectors, produced no non-finite sector entries, and never left ordinary
precision.  Its aggregate statistical errors are still large:
`eps^-6 = 0.221(108)` and `eps^-5 = 2.1(2.9)`.  Naive `1/sqrt(N)` scaling from
that run gives roughly `4.8e4` samples for 10% relative error on `eps^-6` and
`3.8e5` samples for 10% relative error on `eps^-5`, corresponding to about
55 minutes and 7.3 hours respectively at the measured all-sector rate before
any benefit from adaptation.  This is not a precision result, but it is useful
runtime evidence: the current all-sector integrand is finite and stable, while
the remaining issue is variance and Python-heavy hard-sector source assembly.

The requested full three-loop all-sector run was then repeated for the six
orders `eps^-6..eps^-1` on 10 workers under a process-tree memory watchdog.
The final safe preset uses symbolic derivatives, the `projector-formula`
backend, IBP endpoint lowering, `batch-size: 1`, and
`direct-projector-cache-term-threshold: 0` so that high-memory direct endpoint
projectors are not used in the complete all-sector run.  The exact 30 GiB
watchdog interrupted at 324 s wall time when RSS reached 30.19 GiB, and FSD
wrote the partial accumulated result with 370 samples.  Generation took
56.11 s, split into 0.196 s for U/F construction, 1.139 s for sector
generation, and 54.78 s for Symbolica evaluator construction.  No sample used
precision rescue.  The aggregate `pysecdec`-convention coefficients were:

| order | FSD | MC err | rel err |
|---|---:|---:|---:|
| `eps^-6` | `0.1454` | `0.1536` | 105.62% |
| `eps^-5` | `2.047` | `3.023` | 147.69% |
| `eps^-4` | `-72.44` | `67.00` | 92.49% |
| `eps^-3` | `-804.9` | `592.9` | 73.67% |
| `eps^-2` | `-2159` | `4303` | 199.32% |
| `eps^-1` | `-984.1` | `2.496e4` | 2536.08% |

This is explicitly not a validation-quality result.  It is the best current
memory-safe 10-worker all-sector six-order probe, and it shows that the
present implementation is still memory/runtime limited before it is statistics
limited.  The top few one-sample sector contributions in the partial result
were `PSD875`, `PSD364`, `PSD1130`, `PSD148`, `PSD165`, and `PSD814`, so the
next debugging pass should focus on fusing or lowering the regular-source
assembly for those high-depth sectors.

For the leading two triple-box coefficients, the sector population splits into
1556 sectors that are too shallow to contribute, 304 endpoint-depth-five sectors,
and 112 endpoint-depth-six sectors.  With the current curated source-asset
policy, all depth-six representative probes below use direct endpoint
projectors rather than the old large IBP child trees.  The hard-sector
remaining issue is runtime profile, not numerical stability: the regular
source assembly feeding some direct endpoint projectors is still Python-heavy.

## DOT One-Loop Agreement

Massless triangle, pySecDec convention:

| order | FSD | MC err | pySecDec target | rel diff |
|---|---:|---:|---:|---:|
| `eps^-2` | -0.999855 | 0.001414 | -1.000000 | 0.0145% |
| `eps^-1` | 0.577866 | 0.003796 | 0.577216 | 0.112% |
| `eps^0` | 0.656049 | 0.005555 | 0.655878 | 0.0261% |

Massless box, pySecDec convention:

| order | FSD | MC err | pySecDec target | rel diff |
|---|---:|---:|---:|---:|
| `eps^-2` | 4.001006 | 0.003465 | 4.000000 | 0.0251% |
| `eps^-1` | -2.304872 | 0.004860 | -2.308863 | 0.173% |
| `eps^0` | -12.494261 | 0.011108 | -12.493117 | 0.00916% |

Both DOT one-loop examples reproduce the generated pySecDec targets at
sub-percent level.

## DOT Double Box Agreement

The double box uses `--subtraction-backend projector-formula` and the target in
`examples/outputs/dot_double_box_pysecdec_target.json`.

| order | FSD | MC err | target | rel diff | pull |
|---|---:|---:|---:|---:|---:|
| `eps^-4` | -8.29e-4 | 9.01e-4 | 0 | n/a | 0.92 |
| `eps^-3` | 1.495225 | 0.011449 | 1.500180 | 0.330% | 0.43 |
| `eps^-2` | 1.255916 | 0.071973 | 1.268409 | 0.985% | 0.17 |
| `eps^-1` | 3.129779 | 0.272310 | 2.997033 | 4.43% | 0.49 |
| `eps^0` | -14.064221 | 0.979197 | -14.857887 | 5.34% | 0.81 |

All double-box target pulls are below one sigma in this 3M-sample run.  The
leading nonzero coefficients are already close to percent-level agreement; the
subleading finite-side coefficients still need more statistics or variance
reduction for percent-level central-value claims.

## Triple Box Status

The old all-sector triple-box MC probes showed very large errors and should
not be used as physics validation.  The current useful result is instead a
focused stability check of the hard endpoint sector `PSD213`.

For the selected sector `PSD213`, a same-sample Monte Carlo comparison of
`projector-formula` with and without IBP endpoint lowering gives identical
central values within MC precision for the leading two coefficients:

| mode | samples | regular signatures | generation [s] | PythonT [s] | EvalT [s] | `eps^-5` |
|---|---:|---:|---:|---:|---:|---:|
| projector, no IBP | 400 | 32 | 6.064 | 0.877 | 5.628 | `0.00068(40)` |
| projector, IBP | 400 | 108 | 9.360 | 1.910 | 7.380 | `0.00068(40)` |

The same sector also exposes why high precision is necessary.  In the full
Laurent range at scale `1e-12`, ordinary double precision produced `NaN` lower
coefficients.  At the same point, 32 and 80 digits were still unstable, while
160 and 300 digits agreed in the displayed coefficients:

| precision | `eps^-4` | `eps^-3` | `eps^0` | status |
|---:|---:|---:|---:|---|
| 32 | `3.996e3` | `-6.594e28` | `6.351e90` | unstable |
| 80 | `3.251e-2` | `4.094e-1` | `-1.444e41` | unstable in lower orders |
| 160 | `3.251e-2` | `4.094e-1` | `2.223e4` | stable vs 300 digits |
| 300 | `3.251e-2` | `4.094e-1` | `2.223e4` | stable |

The high-precision path promotes the sample, sector-map, U/F/J Taylor
coefficients, endpoint projector inputs, and final projector evaluator call to
the requested Decimal precision before casting the final weight back to double.
With that full-path promotion, the high-tier default of 1000 digits is
conservative for the probed point.

For the leading two coefficients of `PSD213`, 100 and 1000 digits agreed at
scales `1e-8` and `1e-10` within displayed precision.  The 1000-digit path was
roughly ten times slower, so the default thresholds are now conservative:

| tier | threshold | digits |
|---|---:|---:|
| stability | `1e-8` | 100 |
| high precision | `1e-12` | 1000 |

The IBP child-term runtime cache computes one Taylor envelope per boundary/zero
projector, but IBP was not a runtime win in the `PSD213` comparison performed
before the current sparse residual-input signature switch: it used more regular signatures and had
larger Python and evaluator time at the same sample count.  The current
selected-sector regular-Taylor formula layer uses the sparse residual-input
signature by default.  It moves the regular `g_s` power/log/epsilon algebra
into Symbolica and evaluates only the ancestor-closed Taylor shape each formula
needs.  For `PSD213`, this path prepares 6 regular signatures and is
runtime evaluator-dominated.

Current selected-sector Monte Carlo runs show the beginning of the expected
variance scaling and no evidence of endpoint numerical instability.  The
current curated direct endpoint path has been checked on two depth-six
representatives with 4000 samples on 10 workers:

| sector | mode | samples | coefficient | estimate | MC err | relative err | PythonT [s] | EvalT [s] | precision rescue |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `PSD814` | direct endpoint cache | 4000 | `eps^-6` | `-6.80e-4` | `4.5e-5` | 6.68% | 41.78 | 16.45 | none |
| `PSD814` | direct endpoint cache | 4000 | `eps^-5` | `-2.04e-2` | `1.0e-3` | 5.13% | 41.78 | 16.45 | none |
| `PSD855` | direct endpoint cache | 4000 | `eps^-6` | `-6.80e-4` | `4.5e-5` | 6.68% | 77.28 | 18.52 | none |
| `PSD855` | direct endpoint cache | 4000 | `eps^-5` | `-2.4e-3` | `1.1e-3` | 48.0% | 77.28 | 18.52 | none |

The same sectors were rerun at 1000 and 4000 samples with fixed one-iteration
settings to check MC scaling:

| sector | coefficient | MC err at 1000 | MC err at 4000 | error ratio | ideal ratio |
|---|---|---:|---:|---:|---:|
| `PSD814` | `eps^-6` | `8.51e-5` | `4.54e-5` | 1.873 | 2.000 |
| `PSD814` | `eps^-5` | `1.99e-3` | `1.05e-3` | 1.897 | 2.000 |
| `PSD855` | `eps^-6` | `8.51e-5` | `4.54e-5` | 1.873 | 2.000 |
| `PSD855` | `eps^-5` | `2.16e-3` | `1.15e-3` | 1.884 | 2.000 |

The identical `eps^-6` estimates arise from symmetry-related representative
sectors.  The `eps^-5` central value is sector-dependent and can be close to
zero, so its relative error is less useful as a stability diagnostic.  These
runs show order-10% deepest-pole errors without any precision rescue, but they
also show the remaining optimization target: these direct-endpoint sectors are
still Python-heavy because the regular Taylor source assembly is not fully
fused into Symbolica.
After the latest sparse-series and derivative-batching updates, the selected
`PSD814` 4000-sample timing is `PythonT=41.78 s`, `EvalT=16.45 s`,
and `4.11e3` evaluator microseconds per sample per worker.  This is still
Python-heavy, but it removes a repeated sparse convolution from the guarded
fallback and reduces per-derivative evaluator dispatch.
Forcing the high-axis regular-Taylor formula layer on representative `PSD814`
was stopped after 180 s before integration, so the current endpoint stability
claim rests on the guarded direct-endpoint path rather than on an unfiltered
regular-formula cache.

The earlier `PSD62` fallback runs remain useful as a convergence chronology.
The sector reaches order-10% relative MC error for the leading two coefficients
with 15000 samples and never leaves ordinary precision:

| sector | mode | samples | coefficient | estimate | MC err | relative err | PythonT [s] | EvalT [s] | precision rescue |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `PSD62` | current IBP-lowered default | 2000 | `eps^-6` | `1.61e-3` | `5.69e-4` | 35.48% | 59.93 | 123.78 | none |
| `PSD62` | current IBP-lowered default | 2000 | `eps^-5` | `-5.58e-2` | `1.59e-2` | 28.46% | 59.93 | 123.78 | none |
| `PSD62` | IBP-lowered default | 2500 | `eps^-5` | `-5.5e-2` | `1.4e-2` | 26.14% | 39.43 | 5.657 | none |
| `PSD62` | IBP-lowered default | 5000 | `eps^-5` | `-5.52e-2` | `1.03e-2` | 18.59% | 51.11 | 10.72 | none |
| `PSD62` | IBP-lowered default | 15000 | `eps^-6` | `2.13e-3` | `2.08e-4` | 9.79% | 163.03 | 33.26 | none |
| `PSD62` | IBP-lowered default | 15000 | `eps^-5` | `-6.43e-2` | `5.74e-3` | 8.92% | 163.03 | 33.26 | none |

The `PSD62` error reduction from 5000 to 15000 samples is close to
`1/sqrt(N)`: the ideal factor is `sqrt(3)=1.73`, while the observed
`eps^-5` relative-error ratio is 2.08.  The fresh 2000-sample run uses the
current chain-rule and sparse-source implementation and is now mostly evaluator
time (`67.38%` evaluator).  The older 15000-sample run was performed before
those changes and remains useful only for the observed error scaling down to
order 10%.  The sector is stable and statistically convergent; making all
high-depth regular-source assembly evaluator-dominated remains an optimization
target.

The 2000-sample all-sector smoke identified `PSD184`, `PSD80`, and `PSD1057`
among the largest one-sample contributors after `PSD62`.  Focused 4000-sample
runs of those sectors also stayed entirely at ordinary precision and returned
finite estimates:

| sector | samples | coefficient | estimate | MC err | relative err | PythonT [s] | EvalT [s] | precision rescue |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `PSD184` | 4000 | `eps^-6` | `1.74e-3` | `4.1e-4` | 23.6% | 5.70 | 20.08 | none |
| `PSD184` | 4000 | `eps^-5` | `-5.23e-2` | `1.13e-2` | 21.5% | 5.70 | 20.08 | none |
| `PSD80` | 4000 | `eps^-6` | `2.20e-3` | `4.1e-4` | 18.7% | 12.66 | 43.53 | none |
| `PSD80` | 4000 | `eps^-5` | `-4.84e-2` | `1.13e-2` | 23.4% | 12.66 | 43.53 | none |
| `PSD1057` | 4000 | `eps^-6` | `5.58e-4` | `1.36e-4` | 24.5% | 107.59 | 23.56 | none |
| `PSD1057` | 4000 | `eps^-5` | `2.70e-2` | `4.95e-3` | 18.3% | 107.59 | 23.56 | none |

These sectors require about 1.3e4 to 2.4e4 samples each for 10% relative error
under naive `1/sqrt(N)` scaling.  `PSD184` and `PSD80` are evaluator-dominated;
`PSD1057` is stable but still Python-heavy.

The selected sector `PSD213` exercises the prepared regular-Taylor formula
path.  It is already runtime evaluator-dominated and also stays entirely on
ordinary precision:
The six small regular-Taylor signatures needed by this sector are now curated
under `assets/subtraction_formulae/curated`, so this direct-formula path is
default behavior rather than an opt-in warm-cache accident.

| sector | mode | samples | coefficient | estimate | MC err | relative err | PythonT [s] | EvalT [s] | evaluator share |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `PSD213` | current IBP-lowered default | 400 | `eps^-5` | `7.40e-4` | `4.09e-4` | 55.22% | 0.627 | 16.753 | 96.39% |
| `PSD213` | IBP-lowered default | 1600 | `eps^-5` | `5.94e-4` | `1.90e-4` | 32.03% | 2.242 | 72.984 | 97.02% |

Extrapolating the 1600-sample `PSD213` point with `1/sqrt(N)` gives roughly
`1.6e4` samples for 10% relative error on `eps^-5`, or about 12.5 minutes on
one worker at the measured hot rate.  The important implementation evidence is
that the profile shape is now correct for sectors whose regular formulae are
prepared: most time is spent in Symbolica evaluators, not Python glue.

Two older fallback-sector probes cover larger skipped-request classes and are
kept here as chronology:

| sector | skipped regular requests | samples | coefficient | estimate | MC err | relative err | PythonT [s] | EvalT [s] | precision rescue |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| `PSD1` | 324 | 1000 | `eps^-6` | `-1.91e-3` | `2.57e-4` | 13.49% | 19.26 | 2.36 | none |
| `PSD1` | 324 | 1000 | `eps^-5` | `7.63e-2` | `4.45e-3` | 5.83% | 19.26 | 2.36 | none |
| `PSD7` | 486 | 1000 | `eps^-6` | `-6.36e-4` | `8.58e-5` | 13.49% | 97.46 | 3.58 | none |
| `PSD7` | 486 | 1000 | `eps^-5` | `-2.39e-2` | `2.34e-3` | 9.76% | 97.46 | 3.58 | none |

The 729-request class was qualitatively worse before direct endpoint-projector
cache selection.  Representative sector `PSD649` originally timed out in
sparse-fallback mode after 180 s before returning even one sample, despite low
RSS.  After sparse-convolution, shared-chain-product, adaptive IBP chunking,
and split U/F/J source-shape optimizations, one sample returned in about
`63..65 s`.  The chain-rule composition layer, structural map-layout reuse,
ancestor-cache shape building, and cached sparse support keys reduced the
ten-sample probe to 32.9 s of Python time plus 1.39 s of evaluator time, with
`ChainGen=3.88 s` and no precision rescue.  The curated direct
endpoint-projector override for the same universal signature is much better:
the selected ten-sample probe now takes `PythonT=0.368 s`, `EvalT=0.054 s`,
with compatible estimates `eps^-6=-0.00050(36)` and `eps^-5=-0.0032(75)`.
No precision rescue was triggered.  This makes curated direct endpoint
projectors the right default for validated high-IBP signatures.  With the
current curated source-asset set, the depth-six representative sectors tested
above use that path by default.

Forcing direct regular-Taylor formulas for the same sector is still not the
answer: the guarded selected-sector experiment was stopped after 120 s under a
35 GiB memory cap before completing the run.  Earlier direct-evaluator probes
did return one sample, but only with minute-scale evaluator time.  This is not
a precision-rescue problem.  It is the remaining all-sector triple-box
bottleneck and shows that simple cache promotion is not enough for this class:
it needs a lower-signature Symbolica function decomposition, a different
sparse-series implementation, or further algebraic factorization before it can
be part of a practical all-sector convergence run.

The older `PSD649` diagnostics are useful mainly as a chronology of what was
fixed.  Plain sparse fallback timed out after 180--240 s, integer-power
shortcuts were correct but insufficient, and grouped sparse-convolution
lowered one sample to about 64 s while remaining almost entirely Python-bound.
The chain-rule formula layer moved the mapped-derivative composition into
Symbolica but still left too much IBP child-projector assembly in Python.  The
direct endpoint-projector cache removes that particular child-projector
overhead for curated signatures.  Endpoint stability is not the observed
problem: no precision rescue is triggered in these probes.

Additional sector-selected probes were run on high-endpoint-power sectors.
They are not convergence tests because the leading two orders are zero at the
displayed precision, but they exercise different endpoint-projector signatures
without triggering precision rescue:

| sector | samples | Laurent range | result | precision rescue | generation [s] |
|---|---:|---|---|---|---:|
| `PSD389` | 100 | `eps^-6..eps^-5` | both displayed coefficients `0(0)` | none | 1.458 |
| `PSD386` | 100 | `eps^-6..eps^-5` | both displayed coefficients `0(0)` | none | 1.561 |
| `PSD30` | 100 | `eps^-6..eps^-5` | both displayed coefficients `0(0)` | none | 1.829 |
| `PSD59` | 100 | `eps^-6..eps^-5` | both displayed coefficients `0(0)` | none | 1.539 |

The original all-sector guarded-default probe requested only one sample, so it
is not a Monte Carlo validation point.  It is useful for generation timing: all
1972 sectors and all prepared formulae were generated in 5.582 s, with 147
regular signatures prepared, 6 of them loaded from curated source assets,
29 high-axis signatures skipped by the default guards, and 382 sectors switched
to shipped direct endpoint-projector assets.  A follow-up 2000-sample
all-sector smoke generated in 5.862 s and returned finite sector results for
every sector that was sampled.  Together these runs confirm the generation side
is now comfortably in the requested `O(min)` regime; runtime convergence remains
statistics-limited.

The most statistically useful fallback chronology is still `PSD62`.  With the axis
cap, generation is about 1.5 s for selected-sector runs and about 5.6 s for
the current all-sector triple-box smoke.  The fallback carries endpoint output-pair
requests down to the U/F/J Taylor source layer and uses sparse truncation for
the regular power/log series.  The remaining fallback cost is still Python
sparse-series composition rather than evaluator calls; this is the main reason
the 15000-sample `PSD62` run is 83% Python even though it is stable and
convergent.

Forcing the v3 sparse regular-Taylor formula path on `PSD62` changes the
runtime profile but not the generation bottleneck.  The formula request is
sparse and ancestor-closed, so it no longer asks for a dense six-axis Taylor
box.  However, rebuilding and dualizing the Symbolica evaluator for those eight
six-axis signatures still takes about 226 s.  After that build, the one-sample
runtime is evaluator-dominated (`EvalT = 2.24 s`, `PythonT = 0.31 s`), which is
the right runtime shape but not an acceptable default generation tradeoff yet.

The all-sector non-IBP generation path was also re-probed with the warm cache
and stopped by the watchdog after 180 s still inside endpoint-projector formula
preparation.  The all-sector triple-box preset therefore remains IBP-lowered.
However, the useful part of the non-IBP result is now part of the default:
when a shipped direct endpoint-projector asset exists and the IBP compound tree
would exceed `--direct-projector-cache-term-threshold`, FSD uses the direct
endpoint projector for that sector.

The complementary all-sector forced-formula probe is also negative evidence for
making direct regular-Taylor formulae the default today.  With the existing warm
formula cache, the run requested 226 regular-Taylor signatures and was still in
formula preparation after 150 s, with RSS between roughly 4 and 9.5 GiB.  It was
stopped with `stop.order`.  A focused `PSD1057` forced-formula run was also
stopped by the watchdog before sampling: after 301 s and about 11.4 GiB RSS it
had produced no result file for 400 requested samples.  Selected-sector force
mode is useful for cache viability studies, but all-sector forced mode needs a
curated smaller asset set, lower-signature generated expressions, or a native
sparse-series primitive before it can replace the guarded default.
Curated files under `assets/subtraction_formulae/curated` are now treated as
source assets.  The endpoint-projector cache set is curated and shipped because
it is universal and modest in size.  For sector `PSD649`, the curated direct
endpoint projector changes the selected ten-sample diagnostic from the old
chain-rule/IBP path (`PythonT=32.9 s`, `EvalT=1.39 s`) to
`PythonT=0.368 s`, `EvalT=0.054 s`, with the compatible estimates
`eps^-6=-0.00050(36)` and `eps^-5=-0.0032(75)`.  No precision rescue was
triggered.  A hard regular-Taylor signature can also be curated; if it has a
vetted curated file, it bypasses the default cold-build guard and the direct
formula is used without `--force-regular-taylor-formulas`.  The earlier
`PSD649` forced-direct regular-Taylor probe shows the important caveat: a
curated file only removes cold expression generation.  It should only be
promoted when the resulting evaluator is also fast enough in the hot path.

This validates the endpoint stability mechanism for the problematic sector
classes and shows a plausible convergence trend.  The current all-sector
leading-pole smoke is finite but far from precision convergence at 2000 samples.
The selected-sector diagnostics are evaluator-dominated when they stay below
the regular-formula guards; the new direct endpoint depth-six probes are stable
and much faster than the old IBP fallback, but some sectors such as `PSD1057`
remain Python-heavy.  The next optimization target is to fuse or replace that
sparse Python regular-source composition with smaller Symbolica-generated
pieces or native sparse-series support.  The Symbolica `functions=` evaluator
option was checked and is not a runtime Python-callback mechanism; it inlines
symbolic function definitions into the evaluator.
