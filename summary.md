# FastSecDec Path Finder Summary

This is the current short status of FSD after the DOT, prepared-bundle,
universal-formula-cache, and triple-box diagnostic work.  The central
path-finder rule is unchanged: sector integrands are built from black-box
numerical evaluators of the Symanzik polynomials.  The integration processor
does not substitute sector maps into `U` or `F` symbolically.

## Current Workflow

For DOT inputs the preferred workflow is now two-stage:

```sh
.venv/bin/python FSD.py generate \
  --dot-file examples/graphs/triple_box.dot \
  --kinematics examples/graphs/triple_box_kinematics.yaml \
  --output examples/outputs/prepared_triple_box \
  --sector-method iterative \
  --subtraction-backend projector-formula \
  --ibp-reduce-to-log-endpoint \
  --symbolic-derivatives \
  --max-eps-order 0

.venv/bin/python FSD.py integrate \
  --output examples/outputs/prepared_triple_box \
  --workers 10 \
  --samples-per-iter 100000 \
  --batch-size 1000
```

`integrate --output ...` is strict disk-only with respect to generation: it
loads topology metadata, sectors, expression metadata, and serialized
Symbolica evaluator bytes from the prepared bundle.  It does not call pySecDec,
reconstruct `U` or `F`, or build Symbolica formula/evaluator artifacts during
integration.  A strict integrate run can still use the generic Python sparse
Taylor fallback when a deliberately unprepared universal chain-rule signature
is missing from the bundle.

Long exploratory runs should use the local process-tree memory guard.  It can
be stopped by creating `stop.order` in the working directory, so no interactive
permission or `Ctrl-C` is needed:

```sh
./run_with_memory_watch.py \
  --limit-gb 30 \
  --poll-seconds 30 \
  -- .venv/bin/python FSD.py integrate --output examples/outputs/prepared_triple_box
```

## Current Coverage

The triangle, box, and double-box rows use DOT inputs.  The pySecDec column is
the generated-integrator/package generation timing from the validation runs.

| topology | sectors | Laurent range | FSD generation | pySecDec generation | status |
|---|---:|---|---:|---:|---|
| triangle | 3 | `eps^-2..eps^0` | 0.223 s | 9.095 s | agrees with pySecDec target |
| box | 12 | `eps^-2..eps^0` | 0.240 s | 9.075 s | agrees with pySecDec target |
| double box | 140 | `eps^-4..eps^0` | 0.615 s | 272.81 s | all target pulls below 1 sigma in current run |
| triple box | 1972 | `eps^-6..eps^0` | 290.23 s + 457.19 s serialization | not completed | full prepared bundle builds; convergence not demonstrated |

The latest triple-box prepared bundle was generated under a 30 GiB memory
guard with `--chain-rule-formula-output-length-limit 288`.  It contains all
1972 sectors, 360 endpoint-projector formula signatures, 160 regular-Taylor
formula signatures, 181 universal chain-rule formula signatures, and 22996
serialized evaluator files.  The bundle and the local generated cache are each
about 4.1 GiB.

Triple-box generation breakdown:

| component | time |
|---|---:|
| Generation U and F polynomial | 0.198 s |
| Generating sectors | 1.191 s |
| Generating Symbolica evaluators | 288.842 s |
| evaluator serialization | 457.193 s |

The expensive evaluator/formula cache is now rooted at `cache/`, which is
ignored by git and intended to become a distributable tarball.  `install.sh`
can already unpack a local or remote cache archive.  Generation falls back to
rebuilding missing formulas and adds them to the cache.

## Triple-Box Diagnostics

A deterministic one-point scan touched all 1972 prepared triple-box sectors
with 10 workers, a 30 GiB process-tree memory guard, and a 30 s per-sector
classification cap.  It is a coverage and stability diagnostic, not a Monte
Carlo precision result.

| metric | value |
|---|---:|
| completed sectors | 1746 |
| sectors hitting 30 s cap | 226 |
| precision rescue events | 0 |
| completed-sector wall time | 0.0020 s min, 0.206 s median, 29.12 s max |
| completed-sector Symbolica eval time | 0.0010 s min, 0.111 s median, 27.52 s max |
| completed-sector `max|coefficient|` | `2.83e-8` min, 0.155 median, `2.15e3` max |

Timeouts are concentrated in the high-axis sector bands:

| sector-id bin | timeouts |
|---:|---:|
| 0..99 | 8 |
| 100..199 | 12 |
| 300..399 | 4 |
| 400..499 | 6 |
| 500..599 | 10 |
| 600..699 | 24 |
| 700..799 | 32 |
| 800..899 | 45 |
| 900..999 | 27 |
| 1000..1099 | 38 |
| 1100..1199 | 20 |

The largest completed one-point coefficients were finite and stayed in
ordinary double precision:

| sector | largest coefficient | wall time | Symbolica eval time | Python/glue time |
|---|---:|---:|---:|---:|
| `PSD350` | `2.15e3` | 28.63 s | 18.28 s | 10.35 s |
| `PSD2` | `5.80e2` | 13.31 s | 0.011 s | 13.30 s |
| `PSD671` | `5.73e2` | 9.05 s | 7.86 s | 1.19 s |
| `PSD201` | `5.57e2` | 8.37 s | 4.68 s | 3.69 s |
| `PSD106` | `3.90e2` | 12.59 s | 0.010 s | 12.58 s |

This confirms that the triple-box obstruction is not pySecDec re-entry and not
visible endpoint numerical instability in the completed sectors.  The current
weak point is the high-depth source assembly for unprepared universal
chain-rule signatures.  Some hard sectors are genuinely Symbolica-evaluator
heavy; others, such as `PSD2` and `PSD106`, are dominated by the Python sparse
Taylor fallback.

The latest hard-sector probe focused on `PSD649`, a six-axis sector with
missing regular/source formulas.  In the existing strict prepared bundle it
still takes about 53 s per one-point repeat, almost entirely in Python sparse
Taylor composition.  A diagnostic direct-U/F-dual run reduces Python time to
about 7 s for one point, but moves the work into Symbolica evaluator calls and
does not improve 10-point batch throughput.  A cache-warming attempt for the
first missing `PSD649` regular-Taylor formula was stopped after about 170 s
while building/dualizing the generated regular expression.  So the remaining
problem is not the original U/F dualization slowdown; it is the missing fused
regular-source evaluator for hard six-axis signatures.

The requested `1972 x 1000` democratic triple-box scan is therefore not yet a
near-1000 s task with this implementation.  The next useful step is to finish
moving the remaining large chain-rule/source assembly into reusable Symbolica
evaluators or a native sparse-series primitive while keeping the prepared
bundle memory bounded.
