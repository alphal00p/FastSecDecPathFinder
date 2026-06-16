# Validation Notes

This document records the current validation state of FSD on 2026-06-16.
Detailed commands and full metadata are stored in the generated `result.json`
files; this note keeps only the main agreement and stability conclusions.

All values below use the displayed prefactor convention of the run.  Errors are
one-sigma Monte Carlo errors.

## Static Checks

The current intended checks are:

```sh
.venv/bin/python -m pytest -q
rg "import (sympy|scipy)|from (sympy|scipy)" . -g'*.py'
```

Current result: `103 passed, 2 skipped`.

## Overview

The triangle and box rows use DOT inputs, not the built-in shortcuts, so the
generation comparison with pySecDec is meaningful.

| topology | input | sectors | coefficients | target | FSD generation [s] | pySecDec generation [s] | runtime summary |
|---|---|---:|---|---|---:|---:|---|
| triangle | DOT | 3 | `eps^-2..eps^0` | pySecDec generated integrator | 0.223 | 9.095 | avg 2.35 us/smpl/wkr |
| box | DOT | 12 | `eps^-2..eps^0` | pySecDec generated integrator | 0.240 | 9.075 | avg 7.00 us/smpl/wkr |
| double box | DOT | 140 | `eps^-4..eps^0` | stored pySecDec-convention target | 0.615 | 272.81 | avg 18.23 us/smpl/wkr |
| triple box | DOT iterative | 1972 | `eps^-6..eps^0` | no pySecDec target completed | 290.23 + 457.19 serialization | not completed | one-point sector scan, no precision rescue in completed sectors |

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
finite-side coefficients still need more statistics or variance reduction for
percent-level central-value claims.

## Triple Box Status

The full triple-box prepared bundle now builds all 1972 sectors through
`eps^0` under the 30 GiB guard.  The latest prepared bundle contains:

| artifact | count |
|---|---:|
| endpoint-projector formula signatures | 360 |
| regular-Taylor formula signatures | 160 |
| universal chain-rule formula signatures | 181 |
| serialized evaluator files | 22996 |

Generation timing:

| component | time |
|---|---:|
| Generation U and F polynomial | 0.198 s |
| Generating sectors | 1.191 s |
| Generating Symbolica evaluators | 288.842 s |
| evaluator serialization | 457.193 s |

The word "skipped" in older notes referred to guarded formula cold-builds, not
to missing sectors.  The latest bundle includes all 1972 sectors.  It still
does not prepare every possible universal chain-rule signature: signatures
whose output expression count exceeded the current cap of 288 are left to the
strict Python sparse Taylor fallback.

A deterministic one-point scan attempted every sector with 10 workers and a
30 s per-sector cap:

| metric | value |
|---|---:|
| completed sectors | 1746 |
| sectors hitting 30 s cap | 226 |
| precision rescue events | 0 |
| completed wall min / median / p90 / p99 / max [s] | 0.0020 / 0.206 / 10.13 / 25.23 / 29.12 |
| completed Symbolica eval min / median / p90 / p99 / max [s] | 0.0010 / 0.111 / 6.76 / 19.34 / 27.52 |
| completed `max|coefficient|` min / median / p90 / p99 / max | `2.83e-8` / 0.155 / 3.50 / 81.9 / `2.15e3` |

Largest completed one-point sector weights:

| sector | max coefficient | wall [s] | Symbolica eval [s] | Python/glue [s] |
|---|---:|---:|---:|---:|
| `PSD350` | `2.15e3` | 28.63 | 18.28 | 10.35 |
| `PSD2` | `5.80e2` | 13.31 | 0.011 | 13.30 |
| `PSD671` | `5.73e2` | 9.05 | 7.86 | 1.19 |
| `PSD201` | `5.57e2` | 8.37 | 4.68 | 3.69 |
| `PSD106` | `3.90e2` | 12.59 | 0.010 | 12.58 |

The completed-sector scan supports two claims:

1. The prepared bundle is complete enough to load and evaluate every sector id
   without pySecDec or evaluator generation at integration time.
2. No completed deterministic point showed endpoint precision failure; all
   completed points remained in ordinary double precision.

It does not yet prove triple-box convergence.  The 226 capped sectors and the
large one-point coefficients mean that a full precision run still needs
substantial optimization.  The main remaining technical risk is the high-axis
chain-rule/source assembly path, especially sectors where Symbolica evaluation
is cheap but the Python sparse Taylor fallback is tens of seconds per point.

The next validation target is therefore not more blind statistics.  It is to
move the remaining large universal chain-rule/source signatures into
reusable Symbolica evaluators or native sparse-series operations, then repeat
the democratic sector scan and only then launch a long all-coefficient
Monte Carlo run.
