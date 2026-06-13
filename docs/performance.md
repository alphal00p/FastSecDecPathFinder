# FSD Performance Notes

This document records the performance measurements used while developing the
DOT-backed FSD path.  The numbers are not precision benchmarks; they are
low-statistics smoke measurements meant to show which topologies generate, how
large the sector sets are, and where runtime is spent.

All FSD-owned runtime rows below use Symbolica evaluators, Havana sampling, and
the explicit `--pregenerate-dual-evaluators` mode unless stated otherwise.  DOT
topologies use pySecDec only for U/F construction and sector generation in
`--dot-engine fsd` mode.  The generated pySecDec integrator was not run for the
double-box or triple-box ladder examples because package generation/integration
is too heavy for this pathfinder note.

The runtime tables report hot integration time only.  Taylor evaluator setup
time, shown as `TaylorGen` by the CLI, is recorded separately and is included in
the DOT generation bucket `Generating Symbolica evaluators`; it is not part of
`EvalT`, `PythonT`, `HavanaT`, or `elapsed`.

## Environment

| item | value |
|---|---|
| machine | Darwin arm64, kernel 24.0.0 |
| Python | 3.12.6 |
| Symbolica | 2.0.0 |
| pySecDec | 1.6.6 |
| Normaliz | not found on `PATH` |
| default Taylor mode | `--pregenerate-dual-evaluators` |

The LaTeX/Python environment was the project `.venv`.  The default pytest suite
at the time of this report was:

```text
32 passed, 2 skipped
```

The skipped tests are optional low-stat generated-pySecDec comparisons for
small 2-loop and 3-loop DOT examples.  They are enabled with
`FSD_RUN_PYSECDEC_COMPARE=1`.

## Built-In One-Loop Runs

Command shape:

```sh
.venv/bin/python FSD.py ... \
  --samples-per-iter 4096 --batch-size 1024 \
  --max-iter 1 --min-iter 1 --workers 1 \
  --json --no-progress --quiet-summary \
  --pregenerate-dual-evaluators
```

| case | samples | elapsed [s] | EvalT [s] | PythonT [s] | HavanaT [s] | avg [us/smpl/wkr] |
|---|---:|---:|---:|---:|---:|---:|
| triangle massive, `s=1,m=1` | 4096 | 0.0121 | 0.00277 | 0.00646 | 0.00280 | 0.675 |
| triangle massless, `s=-1,m=0` | 4096 | 0.0207 | 0.0108 | 0.00905 | 0.000826 | 2.64 |
| box massive, `s12=0.5,s23=0.7,m=1` | 4096 | 0.00715 | 0.000916 | 0.00511 | 0.00108 | 0.224 |
| box massless, `s12=-1,s23=-2,m=0` | 4096 | 0.0365 | 0.0111 | 0.0243 | 0.00105 | 2.72 |

| case | Laurent coefficient estimates |
|---|---|
| triangle massive | `eps^0 = -0.5400008 +/- 0.0063783`; poles zero |
| triangle massless | `eps^-2 = -1`; `eps^-1 = -4.44e-17 +/- 4.21e-17`; `eps^0 = 0.003096 +/- 0.002719` |
| box massive | `eps^0 = 0.194516 +/- 0.003546`; poles zero |
| box massless | `eps^-2 = 2.03614 +/- 0.03590`; `eps^-1 = -0.68037 +/- 0.07079`; `eps^0 = -4.9910 +/- 0.1871` |

## DOT Generation Statistics

The generation buckets are:

- `Generation U and F polynomial`: DOT parse, kinematics, pySecDec
  `LoopIntegralFromGraph`, and U/F extraction.
- `Generating sectors`: pySecDec sector decomposition and FSD
  `SectorDefinition` conversion.
- `Generating Symbolica evaluators`: scalar U/F evaluators, sector
  map/Jacobian evaluators, and the selected pregenerated Taylor evaluator
  backend.

| case | method | loops | params | dim | sectors | axes | Laurent | U/F gen [s] | sectors gen [s] | Symbolica gen [s] | total [s] |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| DOT triangle | iterative | 1 | 3 | 2 | 3 | 1,2 | `eps^-2..eps^0` | 0.168 | 0.00036 | 0.00076 | 0.169 |
| DOT box | iterative | 1 | 4 | 3 | 12 | 1,2 | `eps^-2..eps^0` | 0.0263 | 0.00204 | 0.00092 | 0.0293 |
| kite 2-loop | iterative | 2 | 5 | 4 | 16 | 0 | `eps^0` | 0.0149 | 0.00239 | 0.00115 | 0.0185 |
| self-energy 3-loop | iterative | 3 | 7 | 6 | 117 | 0 | `eps^0` | 0.0185 | 0.0254 | 0.00307 | 0.0470 |
| three-point 2-loop | iterative | 2 | 5 | 4 | 16 | 0 | `eps^0` | 0.0200 | 0.00229 | 0.00040 | 0.0227 |
| three-point 3-loop | iterative | 3 | 7 | 6 | 117 | 0 | `eps^0` | 0.0248 | 0.0246 | 0.00231 | 0.0517 |
| three-point 2-loop, 6-line | iterative | 2 | 6 | 5 | 22 | 0 | `eps^0` | 0.0215 | 0.00364 | 0.00079 | 0.0259 |
| three-point 3-loop, 8-line | iterative | 3 | 8 | 7 | 162 | 0 | `eps^0` | 0.0270 | 0.0448 | 0.00394 | 0.0758 |
| double box | iterative | 2 | 7 | 6 | 140 | 0,1,2,3,4 | `eps^-4..eps^0` | 0.164 | 0.0306 | 0.300 | 0.494 |
| triple box | geometric_ku | 3 | 10 | 9 | 298 | 0,1,2,3,4 | `eps^-4..eps^0` | 0.191 | 2.92 | 11.8 | 14.9 |

The `geometric` sector method was not used because Normaliz was not available
on `PATH`.  `geometric_ku` was used for the triple box because iterative
generation did not return within a practical bound during exploration.

## DOT FSD Runtime Statistics

Light DOT examples used:

```sh
.venv/bin/python FSD.py \
  --dot-file examples/dot/<case>.dot \
  --kinematics examples/dot/<case>_kinematics.yaml \
  --sector-method iterative --dot-engine fsd \
  --prefactor-convention sector \
  --samples-per-iter 4096 --batch-size 1024 \
  --max-iter 1 --min-iter 1 --workers 1 \
  --json --no-progress --quiet-summary \
  --pregenerate-dual-evaluators
```

The ladder examples used the same FSD engine but with larger batches and four
workers:

- double box: `--samples-per-iter 20000 --batch-size 5000 --workers 4`
- triple box: `--sector-method geometric_ku --samples-per-iter 10000 --batch-size 2000 --workers 4`

| case | samples | workers | elapsed [s] | EvalT [s] | PythonT [s] | HavanaT [s] | avg [us/smpl/wkr] |
|---|---:|---:|---:|---:|---:|---:|---:|
| DOT triangle | 4096 | 1 | 0.0102 | 0.00206 | 0.00728 | 0.000801 | 0.502 |
| DOT box | 4096 | 1 | 0.0299 | 0.00515 | 0.0237 | 0.00101 | 1.26 |
| kite 2-loop | 4096 | 1 | 0.00796 | 0.00130 | 0.00541 | 0.00122 | 0.318 |
| self-energy 3-loop | 4096 | 1 | 0.0392 | 0.00420 | 0.0286 | 0.00629 | 1.03 |
| three-point 2-loop | 4096 | 1 | 0.00800 | 0.00123 | 0.00543 | 0.00131 | 0.300 |
| three-point 3-loop | 4096 | 1 | 0.0380 | 0.00403 | 0.0287 | 0.00521 | 0.983 |
| three-point 2-loop, 6-line | 4096 | 1 | 0.00997 | 0.00150 | 0.00675 | 0.00168 | 0.366 |
| three-point 3-loop, 8-line | 4096 | 1 | 0.0515 | 0.00551 | 0.0380 | 0.00794 | 1.35 |
| double box | 20000 | 4 | 2.62 | 5.77 | 4.31 | 0.0218 | 289 |
| triple box | 10000 | 4 | 29.2 | 27.0 | 48.2 | 0.0508 | 2698 |

`EvalT` and `PythonT` are worker-summed work times, so they can exceed elapsed
wall time in multi-worker runs.  The `avg` column normalizes `EvalT` by the
accepted sample count.  The Taylor evaluator setup time is not included in this
table.

## Taylor Evaluator Mode Comparison

The default mode uses pregenerated dualized U/F evaluators.  The
`--symbolic-derivatives` mode instead builds ordinary non-dual Symbolica
evaluators for symbolic U/F partial derivatives with respect to the original
Feynman parameters, then composes those derivatives with sector-map Taylor jets
by explicit chain rules.  Both modes below were run with the same sample count
and seed; `TaylorGen` is pre-integration setup time and is not included in
`elapsed`, `EvalT`, `PythonT`, or `HavanaT`.

| case | mode | samples | elapsed [s] | EvalT [s] | PythonT [s] | HavanaT [s] | TaylorGen [s] | avg [us/smpl/wkr] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| built-in triangle massless | pregenerated duals | 4096 | 0.0207 | 0.0108 | 0.00905 | 0.000826 | 0.0000486 | 2.64 |
| built-in triangle massless | symbolic derivatives | 4096 | 0.0276 | 0.0122 | 0.0146 | 0.000758 | 0.000231 | 2.99 |
| built-in box massless | pregenerated duals | 4096 | 0.0365 | 0.0111 | 0.0243 | 0.00105 | 0.0000534 | 2.72 |
| built-in box massless | symbolic derivatives | 4096 | 0.0596 | 0.0142 | 0.0443 | 0.00108 | 0.000423 | 3.46 |
| DOT triangle | pregenerated duals | 4096 | 0.0102 | 0.00206 | 0.00728 | 0.000801 | 0.0000343 | 0.502 |
| DOT triangle | symbolic derivatives | 4096 | 0.0124 | 0.00281 | 0.00876 | 0.000787 | 0.000148 | 0.686 |
| DOT box | pregenerated duals | 4096 | 0.0299 | 0.00515 | 0.0237 | 0.00101 | 0.0000334 | 1.26 |
| DOT box | symbolic derivatives | 4096 | 0.0428 | 0.00704 | 0.0347 | 0.00101 | 0.000220 | 1.72 |

For these endpoint-subtracted one-loop examples the symbolic derivative mode is
slower.  It is useful as a cross-check and as an explicit demonstration of the
chain-rule construction, but the dualized U/F evaluator path remains the faster
runtime backend in the current implementation.

## Subtraction Backend Comparison

Three endpoint-subtraction backends are now available:

- `recursive`: the original vectorized Python/Numpy inclusion-exclusion over
  endpoint projectors.
- `formula`: a full Symbolica formula evaluator whose signature includes the
  full sector U/F/J monomial and Taylor-coefficient layout.
- `projector-formula`: a lower-signature Symbolica evaluator for only the
  endpoint projector algebra.  Sector-specific regular-function coefficients
  `g_{S,alpha,r}` are still assembled by the existing black-box Taylor path and
  then passed to the generic projector evaluator.

The lower signature is visible in the double-box DOT case: 133 singular sectors
produce 133 full formula signatures but only 20 endpoint-projector signatures.
The triple-box result file has 2674 singular sectors; metadata counting gives
about 863 current full signatures versus 158 ordered endpoint-projector
signatures, or 109 if endpoint-axis permutations are canonicalized.

All rows below use low statistics and disabled precision-rescue thresholds so
the backend costs are directly comparable.  The double-box rows use
`--symbolic-derivatives` to avoid measuring large dualization shapes.

| case | backend | samples | workers | elapsed [s] | EvalT [s] | PythonT [s] | formula build [s] | full sigs | projector sigs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| triangle massless | recursive | 4096 | 1 | 0.0239 | 0.0115 | 0.0113 | 0 | 0 | 0 |
| triangle massless | formula | 4096 | 1 | 0.0245 | 0.0167 | 0.00688 | 0.00109 | 1 | 0 |
| triangle massless | projector-formula | 4096 | 1 | 0.0211 | 0.0121 | 0.00815 | 0.000705 | 0 | 1 |
| box massless | recursive | 4096 | 1 | 0.0406 | 0.0117 | 0.0274 | 0 | 0 | 0 |
| box massless | formula | 4096 | 1 | 0.0287 | 0.0172 | 0.0101 | 0.00115 | 2 | 0 |
| box massless | projector-formula | 4096 | 1 | 0.0318 | 0.0126 | 0.0178 | 0.000706 | 0 | 2 |
| double box DOT | recursive | 5000 | 1 | 14.61 | 0.114 | 14.49 | 0 | 0 | 0 |
| double box DOT | formula | 5000 | 1 | 11.55 | 0.179 | 11.36 | 16.22 | 133 | 0 |
| double box DOT | projector-formula | 5000 | 1 | 11.13 | 0.121 | 11.00 | 0.379 | 0 | 20 |

The lower-signature projector is therefore the better generated-formula path
for the double box: it keeps the runtime benefit of moving endpoint
inclusion-exclusion into Symbolica, while reducing formula-generation time from
16.2 s to 0.38 s in this run.  It does not yet remove all Python overhead,
because the sector-specific `g_{S,alpha,r}` coefficient assembly is still the
same Python/Numpy regular-function Taylor layer used by the recursive backend.

The projector backend was also run against the saved double-box reference
target:

```sh
.venv/bin/python FSD.py \
  --dot-file examples/dot/double_box.dot \
  --kinematics examples/dot/double_box_kinematics.yaml \
  --dot-engine fsd \
  --prefactor-convention pysecdec \
  --sector-method iterative \
  --symbolic-derivatives \
  --subtraction-backend projector-formula \
  --target examples/dot/result_reference.json \
  --samples-per-iter 20000 \
  --batch-size 5000 \
  --max-iter 1 \
  --min-iter 1 \
  --workers 4 \
  --stability-threshold 0 \
  --high-precision-stability-threshold 0
```

The aggregate result matched the reference within one sigma for every
coefficient:

| coefficient | FSD projector-formula | MC error | reference | pull |
|---|---:|---:|---:|---:|
| `eps^-4` | 0.0113 | 0.0149 | 0 | 0.759 |
| `eps^-3` | 1.6189 | 0.1830 | 1.5002 | 0.649 |
| `eps^-2` | 0.6404 | 1.1986 | 1.2684 | 0.524 |
| `eps^-1` | 2.9641 | 4.3587 | 2.9970 | 0.0076 |
| `eps^0` | -18.7566 | 14.2495 | -14.8579 | 0.274 |

With the same seed and sample count, the recursive backend gave the same
coefficients and MC errors up to numerical roundoff, confirming that the new
projector changes the endpoint-subtraction implementation but not the
integrand being sampled.

## DOT FSD Coefficient Estimates

These coefficient estimates are in the FSD `sector` display convention.  For
the ladder examples, errors are intentionally large at the low statistics used
here; these rows show that the generated sectors execute, not that the
integrals have converged.

| case | coefficient estimates |
|---|---|
| DOT triangle | `eps^-2 = 0.9741 +/- 0.0219`; `eps^-1 = -0.0474 +/- 0.0465`; `eps^0 = -1.6517 +/- 0.0377` |
| DOT box | `eps^-2 = 4.0317 +/- 0.0543`; `eps^-1 = -3.9804 +/- 0.0530`; `eps^0 = -12.6592 +/- 0.1317` |
| kite 2-loop | `eps^0 = 0.69265 +/- 0.01217` |
| self-energy 3-loop | `eps^0 = 1.51497 +/- 0.03671` |
| three-point 2-loop | `eps^0 = 0.70707 +/- 0.01227` |
| three-point 3-loop | `eps^0 = 1.51199 +/- 0.03668` |
| three-point 2-loop, 6-line | `eps^0 = 0.146775 +/- 0.004406` |
| three-point 3-loop, 8-line | `eps^0 = 0.229907 +/- 0.008416` |
| double box | `eps^-4 = 0.3146 +/- 0.0155`; `eps^-3 = -4.39 +/- 5.05`; `eps^-2 = -269.5 +/- 260.5`; `eps^-1 = -5.59e3 +/- 7.72e3`; `eps^0 = -7.39e4 +/- 1.29e5` |
| triple box | `eps^-4 = 0.2330 +/- 0.0193`; `eps^-3 = 403.5 +/- 252.6`; `eps^-2 = 1.075e5 +/- 6.329e4`; `eps^-1 = 8.89e7 +/- 8.69e7`; `eps^0 = 4.36e9 +/- 4.30e9` |

## pySecDec Comparison Policy

The built-in triangle and box tests compare against OneLOopBridge.  DOT mode
uses pySecDec for U/F and sector generation.  The generated pySecDec integrator
can be run with `--dot-engine pysecdec` or `--dot-engine both`, but it was not
run for the double-box and triple-box ladder measurements in this document.

Default tests cover DOT generation for the smaller 2-loop and 3-loop examples.
Optional generated-pySecDec comparisons are available for a small subset and
are skipped unless explicitly enabled:

```sh
FSD_RUN_PYSECDEC_COMPARE=1 .venv/bin/python -m pytest -q
```
