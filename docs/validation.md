# Validation Notes

This document records the current validation state of FSD as of 2026-06-23.
Detailed commands and full metadata live in generated `result.json` files; this
note keeps the main agreement, stability, and current limitation conclusions.

All values below use the displayed prefactor convention of the run.  Errors are
one-sigma Monte Carlo errors.

## Static Checks

The intended checks are:

```sh
.venv/bin/python -m pytest -q
rg "import (sympy|scipy)|from (sympy|scipy)" . -g'*.py'
```

The last full targeted run was:

```text
.venv/bin/python -m pytest tests/test_integrals.py -q
150 passed, 2 skipped
```

The current development branch also passes the focused prefactor tests covering
the signed/scaled Gamma prefactor expansion used by DOT-generated pySecDec
normalizations.

## Overview

The triangle and box rows use DOT inputs, not built-in shortcuts, so the
generation comparison with pySecDec is meaningful.

| topology | input | sectors | coefficients | target | FSD generation [s] | pySecDec generation [s] | runtime summary |
|---|---|---:|---|---|---:|---:|---|
| triangle | DOT | 2 | `eps^-2..eps^0` | pySecDec generated integrator / OneLOop-compatible kinematics | 0.22 | 9.10 | QMC agrees with target |
| box | DOT | 3 | `eps^-2..eps^0` | pySecDec generated integrator / OneLOop-compatible kinematics | 0.24 | 9.08 | QMC agrees with target |
| double box | DOT | 96 | `eps^-4..eps^0` | high-stat pySecDec numerical reference | 9.03 prepared explicit bundle | 272.81 historical package build | QMC compatible after prefactor fix, but much less precise than pySecDec |
| triple box | DOT iterative | 1972 | `eps^-6..eps^0` | no pySecDec target completed | 38.46 recorded generation + 30.61 serialization | not completed | prepared bundle builds; performance study ongoing |

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

Both DOT one-loop examples reproduce generated pySecDec targets at
sub-percent level.

## DOT Double Box Status

There is no analytic truth value in the current FSD workflow for the two-loop
double box.  OneLOopBridge is only used for one-loop triangle/box checks.  The
best current double-box reference is the longer pySecDec numerical run
`/tmp/pysecdec_double_box_178070.json`, whose reported finite part is
`eps^0 = -14.8531901024 +/- 2.48e-7`.  This is the practical reference used
below, not an exact value.

The main FSD discrepancy seen before 2026-06-23 was traced to the DOT global
prefactor series, not to sector subtraction.  The pySecDec double-box prefactor
contains `-gamma(3+2 eps)`.  FSD's generic Symbolica Gamma-series fallback
expanded that signed factor inaccurately from `eps^2` onward.  The analytic
single-affine-Gamma path now handles optional signs and numeric scales:

```text
-gamma(3+2 eps)
  -> [-2.0, -3.6911373403938685, -4.985859983805386,
      -4.599953351364897, -3.681199052065825]
```

With a regenerated prepared bundle using the corrected prefactor, a QMC run at
`N=17807` with 16 random shifts gives:

| order | FSD QMC | MC err | pySecDec reference | pull |
|---|---:|---:|---:|---:|---:|
| `eps^-4` | 8.24e-16 | 7.25e-16 | 6.94e-17 | 1.04 |
| `eps^-3` | 1.499916 | 6.56e-5 | 1.500000 | 1.27 |
| `eps^-2` | 1.268155 | 2.51e-4 | 1.268353 | 0.79 |
| `eps^-1` | 3.002546 | 7.20e-4 | 3.003641 | 1.52 |
| `eps^0` | -14.863046 | 4.08e-3 | -14.853190 | 2.41 |

This run is statistically compatible with the high-stat pySecDec reference for
all five coefficients.

A higher-stat FSD run at `N=34687` with 32 random shifts accumulated `117.7M`
raw sector-group samples in `471.1 s` and gave:

| order | FSD QMC | MC err | pySecDec reference | pull |
|---|---:|---:|---:|---:|
| `eps^-4` | -6.11e-17 | 1.91e-16 | 6.94e-17 | 0.68 |
| `eps^-3` | 1.500016 | 1.37e-5 | 1.500000 | 1.13 |
| `eps^-2` | 1.268395 | 4.97e-5 | 1.268353 | 0.85 |
| `eps^-1` | 3.004745 | 5.34e-4 | 3.003641 | 2.07 |
| `eps^0` | -14.848012 | 2.83e-3 | -14.853190 | 1.83 |

The matching fixed-work pySecDec run at the same `N=34687`, 32 shifts used
`335.2M` parsed sector/order samples and quoted `eps^0` error `1.90e-3`.
Thus FSD is close to pySecDec in sample-count convergence, although pySecDec is
still much faster in wall time because its generated kernels are cheaper.

## Triple Box Status

The full triple-box prepared bundle now builds all 1972 sectors through
`eps^0` under the 30 GiB guard.  The latest completed compressed bundle
contains:

| artifact | count / size |
|---|---:|
| sectors | 1972 |
| endpoint-projector formula signatures | 360 |
| regular-Taylor formula signatures | 166 |
| serialized evaluator files | 30572 |
| prepared bundle size | 27 GiB |

Older notes used the word "skipped" for guarded formula cold-builds, not for
missing sectors.  The current prepared bundle includes all sector ids.  Strict
`integrate --output ...` loads the bundle without pySecDec or evaluator
generation.

The completed one-point sector scans did not show endpoint precision rescue
events in the evaluated points.  This supports the statement that the current
problem is not an obvious endpoint-subtraction instability.  It does not prove
triple-box convergence: several hard sectors remain too slow for a meaningful
democratic high-statistics scan.

## PSD2 Diagnostic

`PSD2` is a representative six-axis sector.  The completed compressed bundle
uses the sparse regular-source fallback for this sector.  A focused experiment
then injected the 8 unique six-axis regular formulas needed by PSD2 and allowed
missing source dual shapes to be built in memory.

| path | warm median wall | warm median Symbolica eval | warm median Python/glue | conclusion |
|---|---:|---:|---:|---|
| sparse fallback in prepared bundle | 1.15 s | 0.760 s | 0.390 s | current practical path |
| injected direct regular formulas | 10.58 s | 9.97 s | 0.612 s | lower Python, much worse total |

The direct formula path is algebraically valid for the sampled points, but it
is not a performance win.  It fragments the work across many formula evaluators
and thousands of source dual shapes.  This is a useful validation of the design
boundary: "move more work into Symbolica" is only helpful if it is fused at the
right granularity.

## Current Conclusion

FSD is validated on DOT triangle, DOT box, and DOT double box against pySecDec
targets at the current precision.  The triple box can be generated as a strict
prepared bundle and loaded without pySecDec at runtime.  The remaining
triple-box work is performance/convergence engineering: build a fused
Symbolica-side regular-source path or native sparse-series primitive, then
repeat the democratic all-sector scan and only then launch a long Monte Carlo
run for all seven Laurent coefficients.
