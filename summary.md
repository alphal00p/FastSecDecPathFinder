# FastSecDec Path Finder Summary

This is the current short status of FSD after the DOT, prepared-bundle,
universal-formula-cache, Symbolica-dev, and triple-box diagnostic work.  The
core path-finder rule is unchanged: the sector processor treats `U` and `F` as
black-box numerical evaluators.  It never substitutes sector maps into those
polynomials symbolically during integration.

## Preferred Workflow

For DOT inputs the preferred workflow is two-stage:

```sh
.venv/bin/python FSD.py generate \
  --dot-file examples/graphs/triple_box.dot \
  --kinematics examples/graphs/triple_box_kinematics.yaml \
  --output examples/outputs/prepared_triple_box \
  --sector-method iterative \
  --subtraction-backend projector-formula \
  --ibp-reduce-to-log-endpoint \
  --pregenerate-dual-evaluators \
  --max-eps-order 0

.venv/bin/python FSD.py integrate \
  --output examples/outputs/prepared_triple_box \
  --workers 10 \
  --samples-per-iter 100000 \
  --batch-size 1000
```

`integrate --output ...` is strict disk-only with respect to generation: it
loads topology metadata, sectors, formula metadata, and serialized Symbolica
evaluator bytes from the prepared bundle.  It does not call pySecDec,
reconstruct `U`/`F`, or build Symbolica formula/evaluator artifacts during
integration.

Long exploratory runs should use the process-tree memory guard:

```sh
./run_with_memory_watch.py \
  --limit-gb 30 \
  --poll-seconds 30 \
  -- .venv/bin/python FSD.py integrate --output examples/outputs/prepared_triple_box
```

The wrapper terminates the child process group if the memory limit is exceeded
or if `stop.order` appears in the working directory.

## Cache Model

The relevant user experience is now the shipped-cache experience.  Building an
empty universal cache locally can be very expensive and is not the intended
default.  It is acceptable for an offline cluster job to generate the universal
formula cache once and ship it as a downloadable archive.

Current local cache sizes:

| artifact | size |
|---|---:|
| generated top-level cache `cache/subtraction_formulae` | 22 GiB |
| legacy/source asset cache `assets/subtraction_formulae` | 11 GiB |
| completed compressed triple-box prepared bundle | 27 GiB |

The cache entries are universal in the endpoint/projector signatures.  The
topology-specific part remains the source evaluator preparation for the actual
`U`, `F`, sector maps, and Jacobians.  Those belong in a prepared bundle and
are reused by strict `integrate`.

## Current Coverage

The triangle, box, and double-box rows use DOT inputs.  The pySecDec column is
the generated-integrator/package generation timing from validation runs.

| topology | sectors | Laurent range | FSD generation | pySecDec generation | status |
|---|---:|---|---:|---:|---|
| triangle | 3 | `eps^-2..eps^0` | 0.223 s | 9.095 s | agrees with pySecDec target |
| box | 12 | `eps^-2..eps^0` | 0.240 s | 9.075 s | agrees with pySecDec target |
| double box | 140 | `eps^-4..eps^0` | 0.615 s | 272.81 s | target pulls below 1 sigma in current run |
| triple box | 1972 | `eps^-6..eps^0` | 38.46 s recorded generation plus 30.61 s bundle serialization | not completed | prepared bundle builds; convergence not demonstrated |

The latest completed compressed triple-box bundle contains all 1972 sectors,
360 endpoint-projector formula signatures, 166 regular-Taylor formula
signatures, and 30572 serialized evaluator files.  It was generated with
pregenerated dual evaluators, IBP endpoint lowering, and no chain-rule formula
backend.

## PSD2 Focused Experiment

`PSD2` is a six-axis sector with monomial powers
`F: [3,3,0,2,0,1,0,1,1]` and singular axes `[0,1,3,5,7,8]`.  It is useful
because the older scan made it look Python-dominated.

With the current completed compressed bundle, warm PSD2 repeats are no longer
the old 13 s Python bottleneck:

| path | warm median wall | warm median Symbolica eval | warm median Python/glue |
|---|---:|---:|---:|
| sparse fallback in prepared bundle | 1.15 s | 0.760 s | 0.390 s |
| injected direct regular formulas | 10.58 s | 9.97 s | 0.612 s |

The direct formula experiment loaded/injected the 8 unique six-axis regular
formula signatures for PSD2.  It reduced Python-side work, but it made total
runtime much worse because the evaluation became fragmented across many
Symbolica evaluator calls and thousands of U/F source dual shapes.  This is a
concrete counterexample to blindly baking every coefficient into separate
formula evaluators.

The right next optimization is not "more small formula evaluators"; it is a
fused evaluator/source path that keeps the algebra inside Symbolica while
reducing evaluator-call granularity.

## Triple-Box Status

The full triple-box sector list can be generated and loaded from a prepared
bundle under the 30 GiB guard.  Earlier one-point scans found no precision
rescue events in completed sectors.  The remaining issue is performance and
variance, not a known endpoint-subtraction instability.

The main weak point is still high-axis source assembly.  Some hard sectors are
genuinely Symbolica-evaluator heavy; others still pay Python sparse-series
composition.  The PSD2 experiment shows that replacing that Python work by
many standalone Symbolica evaluators can be slower, so the next implementation
step should fuse the regular-source computation more aggressively.
