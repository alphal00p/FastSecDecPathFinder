# Validation Runs

This note records the validation runs performed on 2026-06-14 on the local
10-worker setup.  The target was to get below 1% when feasible, while keeping
each expensive setup to about 30 minutes on 10 cores.  Some subleading
multi-loop coefficients did not reach 1% in that budget; those cases are
reported explicitly rather than rescaled or hidden.

All FSD coefficient tables use the displayed prefactor convention selected by
the command.  Errors are one-sigma Monte Carlo errors.  `rel err` is
`abs(error) / abs(central)`.  For coefficients whose target is zero, relative
target agreement is not meaningful; the pull and absolute error are the useful
checks.

## Commands

Static checks:

```bash
.venv/bin/python -m pytest -q
rg "import (sympy|scipy)|from (sympy|scipy)" . -g'*.py'
```

Built-in massless triangle, benchmarked with OneLOopBridge:

```bash
.venv/bin/python FSD.py \
  --integral triangle \
  --mode massless \
  --s -1.0 \
  --m 0.0 \
  --prefactor-convention raw \
  --subtraction-backend projector-formula \
  --samples-per-iter 1000000 \
  --batch-size 20000 \
  --max-iter 1 \
  --min-iter 1 \
  --workers 10 \
  --json \
  --no-progress \
  --quiet-summary \
  --stability-threshold 0 \
  --high-precision-stability-threshold 0 \
  --result-path /tmp/fsd_validation_triangle_1M.json
```

Built-in massless box, benchmarked with OneLOopBridge:

```bash
.venv/bin/python FSD.py \
  --integral box \
  --mode massless \
  --s12 -1.0 \
  --s23 -2.0 \
  --m 0.0 \
  --prefactor-convention raw \
  --subtraction-backend projector-formula \
  --samples-per-iter 1000000 \
  --batch-size 20000 \
  --max-iter 1 \
  --min-iter 1 \
  --workers 10 \
  --json \
  --no-progress \
  --quiet-summary \
  --stability-threshold 0 \
  --high-precision-stability-threshold 0 \
  --result-path /tmp/fsd_validation_box_1M.json
```

DOT double box with FSD.  The target is read from
`examples/dot/result_reference.json`, which stores pySecDec-convention numeric
reference coefficients:

```bash
.venv/bin/python FSD.py \
  --dot-file examples/dot/double_box.dot \
  --kinematics examples/dot/double_box_kinematics.yaml \
  --dot-engine fsd \
  --prefactor-convention pysecdec \
  --sector-method iterative \
  --symbolic-derivatives \
  --subtraction-backend projector-formula \
  --target examples/dot/result_reference.json \
  --samples-per-iter 1000000 \
  --batch-size 20000 \
  --max-iter 3 \
  --min-iter 1 \
  --workers 10 \
  --json \
  --no-progress \
  --quiet-summary \
  --stability-threshold 0 \
  --high-precision-stability-threshold 0 \
  --result-path /tmp/fsd_validation_double_box_projector_3M.json
```

DOT double box with pySecDec itself, run only to obtain a generation/runtime
comparison:

```bash
.venv/bin/python FSD.py \
  --dot-file examples/dot/double_box.dot \
  --kinematics examples/dot/double_box_kinematics.yaml \
  --dot-engine pysecdec \
  --prefactor-convention pysecdec \
  --sector-method iterative \
  --symbolic-derivatives \
  --subtraction-backend recursive \
  --pysecdec-epsrel 1.0 \
  --pysecdec-maxeval 1000 \
  --pysecdec-workdir /tmp/fsd_pysecdec_double_validation \
  --result-path /tmp/fsd_validation_double_box_pysecdec_lowstat.json \
  --quiet-summary \
  --log-level INFO
```

Attempted triple-box pySecDec target for the first two leading poles.  This
timed out in pySecDec package generation before producing coefficients:

```bash
.venv/bin/python FSD.py \
  --dot-file examples/dot/triple_box.dot \
  --kinematics examples/dot/triple_box_kinematics.yaml \
  --dot-engine pysecdec \
  --prefactor-convention pysecdec \
  --sector-method geometric_ku \
  --symbolic-derivatives \
  --subtraction-backend recursive \
  --max-eps-order -5 \
  --pysecdec-epsrel 1.0 \
  --pysecdec-maxeval 1000 \
  --pysecdec-workdir /tmp/fsd_pysecdec_triple_leading2 \
  --result-path /tmp/fsd_validation_triple_box_pysecdec_leading2.json \
  --quiet-summary \
  --log-level INFO
```

DOT triple box with FSD for the first two leading poles:

```bash
.venv/bin/python FSD.py \
  --dot-file examples/dot/triple_box.dot \
  --kinematics examples/dot/triple_box_kinematics.yaml \
  --dot-engine fsd \
  --prefactor-convention pysecdec \
  --sector-method geometric_ku \
  --symbolic-derivatives \
  --subtraction-backend recursive \
  --samples-per-iter 200 \
  --batch-size 1 \
  --max-iter 10 \
  --min-iter 1 \
  --workers 10 \
  --max-eps-order -5 \
  --json \
  --no-progress \
  --quiet-summary \
  --result-path /tmp/fsd_validation_triple_box_leading2_2k.json
```

## Test And Import Guard

| check | result |
|---|---:|
| `pytest -q` | 57 passed, 2 skipped |
| FSD-owned SciPy/SymPy import guard | no matches |

## Generation And Runtime Summary

`FSD gen` is the total FSD-side generation time.  For DOT inputs this is the
sum of U/F construction, sector generation/conversion, and Symbolica evaluator
construction.  `pySecDec gen` is package generation plus compile plus load for
the pySecDec generated integrator, when that path was run.  `avg FSD runtime`
is the evaluator time per sample per worker reported by FSD.

| setup | engine | sectors | samples | wall [s] | FSD gen [s] | pySecDec gen [s] | pySecDec integ [s] | avg FSD runtime [us/smpl/wkr] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| built-in triangle | FSD | 2 | 1,000,000 | 1.60 | n/a | n/a | n/a | 6.38 |
| built-in box | FSD | 12 | 1,000,000 | 1.72 | n/a | n/a | n/a | 6.37 |
| DOT double box | FSD | 140 | 3,000,000 | 84.28 | 0.615 | n/a | n/a | 18.23 |
| DOT double box | pySecDec | 140 | n/a | 276.99 | 0.205 | 272.81 | 3.37 | n/a |
| DOT triple box, leading two poles | FSD | 2792 | 2,000 | 1064.10 | 22.80 | n/a | n/a | 3177.71 |
| DOT triple box, target attempt | pySecDec | 2792 | n/a | 600.04 timeout | 23.7 | >600 timeout | n/a | n/a |

The double-box pySecDec generation time is dominated by compilation:
6.07 s package generation, 266.73 s compile, and 0.005 s load.  The triple-box
pySecDec run reached package generation but did not finish within the 600 s
target-generation cap, so no pySecDec target was available for the FSD
triple-box coefficients.

## Built-In One-Loop Validation

### Triangle

Massless triangle, raw convention, target from OneLOopBridge.

| order | FSD | MC err | target | pull |
|---|---:|---:|---:|---:|
| `eps^-2` | -1.000000 | 0 | -1.000000 | 0.00 |
| `eps^-1` | 2.97e-16 | 1.58e-16 | 0 | 1.88 |
| `eps^0` | 2.01e-4 | 1.74e-4 | 0 | 1.15 |

The nonzero pole is exact in this setup.  The remaining coefficients have zero
targets and are consistent with zero in pull.

### Box

Massless box, raw convention, target from OneLOopBridge.

| order | FSD | MC err | rel err | target | rel diff | pull |
|---|---:|---:|---:|---:|---:|---:|
| `eps^-2` | 2.002548 | 0.002262 | 0.113% | 2.000000 | 0.127% | 1.13 |
| `eps^-1` | -0.685744 | 0.004445 | 0.648% | -0.693147 | 1.07% | 1.67 |
| `eps^0` | -4.931948 | 0.011869 | 0.241% | -4.934802 | 0.058% | 0.24 |

All nonzero box coefficients have sub-percent MC relative errors.  The
`eps^-1` central value is 1.07% away from the target but still only 1.67 sigma
from the benchmark.

## DOT Double Box Validation

The DOT double box used the current `projector-formula` subtraction backend and
a target loaded from `examples/dot/result_reference.json` in pySecDec
convention.

| order | FSD | MC err | rel err | target | rel diff | pull |
|---|---:|---:|---:|---:|---:|---:|
| `eps^-4` | -8.29e-4 | 9.01e-4 | n/a | 0 | n/a | 0.92 |
| `eps^-3` | 1.495225 | 0.011449 | 0.766% | 1.500180 | 0.330% | 0.43 |
| `eps^-2` | 1.255916 | 0.071973 | 5.73% | 1.268409 | 0.985% | 0.17 |
| `eps^-1` | 3.129779 | 0.272310 | 8.70% | 2.997033 | 4.43% | 0.49 |
| `eps^0` | -14.064221 | 0.979197 | 6.96% | -14.857887 | 5.34% | 0.81 |

Within the 3M-sample run, the `eps^-3` coefficient is already below 1% MC
relative error.  The deeper finite-side coefficients are not below 1% yet, but
all target pulls are below one sigma.  This suggests the central values are
compatible with the reference, while the current variance is still too large
for percent-level claims on every Laurent order in this budget.

The low-stat pySecDec comparison run produced:

| order | pySecDec | MC err | rel err |
|---|---:|---:|---:|
| `eps^-4` | -3.98e-15 | 2.24e-15 | n/a |
| `eps^-3` | 1.499928 | 0.000358 | 0.0239% |
| `eps^-2` | 1.270090 | 0.001717 | 0.135% |
| `eps^-1` | 3.000086 | 0.006473 | 0.216% |
| `eps^0` | -14.822466 | 0.024256 | 0.164% |

## DOT Triple Box Leading Poles

The triple box was generated and integrated by FSD for the first two leading
Laurent coefficients only.  A matching pySecDec target was attempted but was
not available: pySecDec did not finish package generation in the 600 s target
budget.

| order | FSD | MC err | rel err | target |
|---|---:|---:|---:|---|
| `eps^-6` | 0.136694 | 0.0956 | 69.9% | unavailable |
| `eps^-5` | -1.952625 | 3.29 | 169% | unavailable |

This is not a precision validation yet.  It validates that the DOT-to-sector
path can generate the 2792-sector triple box and run through the leading two
orders under the 30-minute budget, but the statistics are far from converged.
The missing pySecDec target is currently the main blocker for a numerical
agreement statement on this topology.

## Diagnostic Runs

A longer double-box run with default endpoint stabilization was interrupted
smoothly after 8M samples and wrote a partial result.  It did not improve the
validation: rare endpoint samples generated enormous variance in the subleading
coefficients.  The run used the same target and current projector backend, took
1239.97 s, and reported 75.50 us/sample/worker.  The result is useful as a
stress test of the endpoint region, but not as the best validation row.

This diagnostic is consistent with the current interpretation: additional
statistics and sector training do help some coefficients, but the hardest
multi-axis endpoint sectors still need variance reduction or a more robust
subtraction/evaluation strategy before all Laurent orders can be validated at
the percent level in the requested wall-time.
