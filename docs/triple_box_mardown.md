# Triple-box investigation

This note records the triple-box checks run on 2026-06-13.  The purpose was not
to claim a converged physics result, but to isolate the cost of the DOT/pySecDec
generation phase, Symbolica evaluator generation, and the current subtraction
runtime path for a demanding three-loop example.

## Topology and command

The FSD run used the massless Euclidean triple box from
`examples/dot/triple_box.dot` with `examples/dot/triple_box_kinematics.yaml`.
The Laurent range was the full three-loop range through the finite part,
`eps^-6 ... eps^0`.

```bash
.venv/bin/python FSD.py \
  --dot-file examples/dot/triple_box.dot \
  --kinematics examples/dot/triple_box_kinematics.yaml \
  --dot-engine fsd \
  --prefactor-convention pysecdec \
  --sector-method geometric_ku \
  --symbolic-derivatives \
  --subtraction-backend recursive \
  --samples-per-iter 600 \
  --batch-size 1 \
  --max-iter 1 \
  --min-iter 1 \
  --workers 10 \
  --max-eps-order 0 \
  --quiet-summary \
  --log-level WARNING \
  --stability-threshold 0 \
  --high-precision-stability-threshold 0 \
  --result-path docs/triple_box_fsd_result.json
```

The result file is `docs/triple_box_fsd_result.json`.  It contains all 2792
per-sector accumulators and is therefore large.

## Generation timings

FSD uses pySecDec only to construct the scalar parametric data and sectors.  The
runtime path then works from FSD `TopologyDefinition` and `SectorDefinition`
objects and does not re-enter pySecDec.

| Stage | Time |
|---|---:|
| Generation U and F polynomial | 0.194 s |
| Generating sectors | 24.901 s |
| Generating Symbolica evaluators | 0.189 s |
| Total FSD generation | 25.284 s |

Detailed generation breakdown:

| Detail | Time |
|---|---:|
| DOT parse | 0.156 s |
| Kinematics load/evaluation | 0.001 s |
| pySecDec `LoopIntegralFromGraph` | 0.022 s |
| U/F extraction | 0.015 s |
| Symbolica scalar evaluator build | 0.001 s |
| pySecDec sector decomposition (`geometric_ku`) | 23.961 s |
| FSD `SectorDefinition` conversion | 0.940 s |
| Symbolica sector evaluator build | 0.113 s |
| Symbolica Taylor evaluator build (`--symbolic-derivatives`) | 0.075 s |

Sector statistics:

| Quantity | Value |
|---|---:|
| Total sectors | 2792 |
| Integration dimension | 9 |
| Laurent range | `eps^-6 ... eps^0` |
| Max endpoint Taylor order | 2 |
| 0-axis sectors | 118 |
| 1-axis sectors | 314 |
| 2-axis sectors | 587 |
| 3-axis sectors | 669 |
| 4-axis sectors | 544 |
| 5-axis sectors | 400 |
| 6-axis sectors | 160 |

## FSD integration probe

The 600-sample run lasted about 202 s wall time.  Since there are 2792 sectors,
most sectors received zero or one sample, so this is a runtime probe rather than
a convergence claim.

| Quantity | Value |
|---|---:|
| Samples | 600 |
| Workers | 10 |
| Wall time | 202.335 s |
| Accumulated evaluator time | 1.817 s |
| Accumulated Python time | 1866.684 s |
| Accumulated Havana time | 0.323 s |
| Average evaluator time | 3029 μs / sample / worker |
| Python fraction | 99.89% |
| Evaluator fraction | 0.10% |
| Havana fraction | 0.02% |

Precision-rescue thresholds were explicitly disabled in this timing run, so the
sample fractions were:

| Precision path | Samples | Fraction |
|---|---:|---:|
| Ordinary | 600 | 100.00% |
| 32 digits | 0 | 0.00% |
| 1000 digits | 0 | 0.00% |

The aggregate FSD coefficients in the `pysecdec` display convention were:

| Coefficient | Central value | MC error | Relative MC error |
|---|---:|---:|---:|
| `eps^-6` | 0.0797849906941 | 0.205506365538 | 257.58% |
| `eps^-5` | -1.74481438599 | 3.96354152179 | 227.16% |
| `eps^-4` | -53.3590224913 | 54.3058438752 | 101.77% |
| `eps^-3` | -42.702198166 | 980.772948531 | 2296.77% |
| `eps^-2` | 16139.4087847 | 18455.5741263 | 114.35% |
| `eps^-1` | 178834.451525 | 161287.93589 | 90.19% |
| `eps^0` | 992346.098538 | 873486.546585 | 88.02% |

The large errors are expected for this short probe.  The finite-part and
subleading-pole values are dominated by isolated large-weight sector samples.

## pySecDec leading-pole attempt

A pySecDec package-generation/integration attempt was launched for the leading
pole only:

```bash
.venv/bin/python FSD.py \
  --dot-file examples/dot/triple_box.dot \
  --kinematics examples/dot/triple_box_kinematics.yaml \
  --dot-engine pysecdec \
  --prefactor-convention pysecdec \
  --sector-method geometric_ku \
  --symbolic-derivatives \
  --subtraction-backend recursive \
  --max-eps-order -6 \
  --pysecdec-epsrel 1.0 \
  --pysecdec-maxeval 1000 \
  --pysecdec-workdir docs/.pysecdec_triple_box_leading \
  --result-path docs/triple_box_pysecdec_leading.json \
  --quiet-summary \
  --log-level INFO
```

The pySecDec path timed out after 600.03 s during generated-package creation,
before producing `docs/triple_box_pysecdec_leading.json`.  The last log entries
were still in `sum_package` / `make_package` for `fsd_psd_triple_box_integral`.
This means no leading-pole target was obtained from pySecDec within the 10-minute
budget.

The important comparison is therefore generation-stage only:

| Path | Outcome |
|---|---|
| FSD DOT + pySecDec sector metadata + Symbolica evaluators | Runtime-ready in 25.284 s |
| pySecDec generated package path | No result within 600.03 s |

No pySecDec coefficient table is available for this run:

| Coefficient | pySecDec central value | pySecDec error | Status |
|---|---:|---:|---|
| `eps^-6` | n/a | n/a | package generation timed out before integration |

## Symbolic derivatives versus dualization

The large sector dual shape `[3, 3, 3, 3, 3, 4]` is not an envelope artifact.  It
appears natively in sectors such as `PSD1092`, where the extracted endpoint
powers require Taylor orders through second order on several axes and through
third order on one axis after combining U/F residual requirements.

Direct Symbolica dualization of such high-rank shapes is currently too expensive
for generation:

| Evaluator | Shape length | Observed dualization time |
|---|---:|---:|
| U dual evaluator | 5120 | about 194 s |
| F dual evaluator | 5120 | about 190 s |
| Constant Jacobian dual evaluator | 5120 | about 184 s |

The near-identical timing for a constant expression shows that the current
bottleneck is the requested dual shape itself, not U/F expression complexity.
The standalone reproducer is `U_dualization_slowdown.py`.

The `--symbolic-derivatives` path is much more promising for generation.  For
the triple box it built all derivative evaluators in about 0.075 s.  The reason
is structural: U and F are multilinear in the original Feynman parameters, so
the symbolic derivative path needs many mixed first derivatives, but avoids deep
per-original-parameter derivatives and avoids Symbolica dualization of the large
sector-coordinate jet shapes.

## Subtraction backend conclusion

The current `--subtraction-backend recursive` path is a Python/Numpy runtime
implementation of the localized Taylor subtraction sum.  It calls Symbolica to
obtain U/F/J Taylor coefficient inputs, but the inclusion-exclusion over endpoint
subtractions, Taylor multi-indices, denominator expansions, logarithms, and
Laurent coefficient accumulation is performed by Python loops at sampling time.

That explains the observed profile:

```text
Python  99.89%
Symbolica evaluator 0.10%
Havana 0.02%
```

The `--subtraction-backend formula` path now builds the subtraction formula in
`subtraction_formula.py`.  It keeps U/F as black-box Taylor-coefficient inputs
and uses Symbolica replacement rules, series expansion, and coefficient
extraction to construct the endpoint-subtracted Laurent expressions.  The older
Python expression-construction builder is retained as
`build_subtraction_formula_legacy` for testing and profiling.

Representative builder timings on the triple-box sectors, after fixing the
Taylor-coefficient extraction bug in the Symbolica-template generator, were:

| Sector | Axes | Taylor coefficient shape | New Symbolica-template builder | Legacy Python-expression builder |
|---|---:|---:|---:|---:|
| `PSD0` | 3 | 64 | 5.45 s | 0.297 s |
| `PSD1` | 4 | 256 | >180 s, interrupted | 152.9 s |

This validates the replacement-rule generator, but the current single full
formula is not yet the scalable route.  Keeping coefficient extraction correct
requires Symbolica series coefficient extraction on factored expressions, and
the resulting full formula is still a large scalar expression specialized to the
full sector signature, with thousands of U/F/J coefficient input symbols and
many endpoint projector terms.

That design has now been implemented as
`--subtraction-backend projector-formula`.  It splits the subtraction path into
two layers:

1. The existing black-box Taylor path builds the sector-specific
   `g_{S,alpha,r}` coefficients from U/F/J evaluator data.
2. A lower-signature Symbolica endpoint-projector evaluator receives those
   coefficients and performs the endpoint inclusion-exclusion, analytic
   denominator factors, logarithms, and Laurent projection.

The endpoint-projector cache key depends only on the number of singular axes,
the endpoint powers, the endpoint Taylor orders, and the requested Laurent
range.  It does not contain the sector map, U/F monomial layout, Jacobian
monomial layout, or native U/F/J Taylor coefficient shape.

The measured signature reduction is substantial:

| Case | Singular sectors | Full formula signatures | Endpoint-projector signatures |
|---|---:|---:|---:|
| DOT double box | 133 | 133 | 20 |
| DOT triple box metadata | 2674 | about 863 | 158 ordered / 109 canonicalized |

The double-box low-stat benchmark shows why this matters:

| Backend | Samples | Workers | Runtime [s] | Formula build [s] | Full sigs | Projector sigs |
|---|---:|---:|---:|---:|---:|---:|
| `recursive` | 5000 | 1 | 14.61 | 0 | 0 | 0 |
| `formula` | 5000 | 1 | 11.55 | 16.22 | 133 | 0 |
| `projector-formula` | 5000 | 1 | 11.13 | 0.379 | 0 | 20 |

The projector backend was also run against the saved double-box reference
target with 20000 samples and 4 workers.  All coefficients were within one
standard deviation:

| Coefficient | FSD projector | MC error | Reference | Pull |
|---|---:|---:|---:|---:|
| `eps^-4` | 0.0113 | 0.0149 | 0 | 0.759 |
| `eps^-3` | 1.6189 | 0.1830 | 1.5002 | 0.649 |
| `eps^-2` | 0.6404 | 1.1986 | 1.2684 | 0.524 |
| `eps^-1` | 2.9641 | 4.3587 | 2.9970 | 0.0076 |
| `eps^0` | -18.7566 | 14.2495 | -14.8579 | 0.274 |

With the same seed and sampling configuration, the recursive backend gives the
same double-box coefficients and MC errors up to numerical roundoff.  This
confirms that the new backend changes only the endpoint-subtraction
implementation, not the sampled integrand.

The remaining bottleneck is still the sector-specific `g_{S,alpha,r}` assembly,
which is shared with the recursive backend and is performed in Python/Numpy.
The endpoint projector has moved the generic subtraction algebra into
Symbolica, but a future fully generated backend would also generate the
regular-coefficient layer.
