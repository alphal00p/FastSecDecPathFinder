# Validation Notes

This document records the current validation state of FSD as of 2026-06-17.
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

The last full recorded result was `103 passed, 2 skipped`.  The current
development branch also passes the focused regular-Taylor tests covering the
dualized and sparse formula builders.

## Overview

The triangle and box rows use DOT inputs, not built-in shortcuts, so the
generation comparison with pySecDec is meaningful.

| topology | input | sectors | coefficients | target | FSD generation [s] | pySecDec generation [s] | runtime summary |
|---|---|---:|---|---|---:|---:|---|
| triangle | DOT | 3 | `eps^-2..eps^0` | pySecDec generated integrator | 0.223 | 9.095 | avg 2.35 us/smpl/wkr |
| box | DOT | 12 | `eps^-2..eps^0` | pySecDec generated integrator | 0.240 | 9.075 | avg 7.00 us/smpl/wkr |
| double box | DOT | 140 | `eps^-4..eps^0` | stored pySecDec-convention target | 0.615 | 272.81 | avg 18.23 us/smpl/wkr |
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
finite-side coefficients still need more statistics or variance reduction for
percent-level central-value claims.

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
