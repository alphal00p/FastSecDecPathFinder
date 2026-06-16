# FSD Performance Notes

These are the current development measurements for the DOT-backed FSD path.
They are performance and stability diagnostics, not final precision
benchmarks.  All FSD-owned code remains free of SciPy and SymPy imports;
pySecDec is used only at the DOT generation boundary.

## Environment

| item | value |
|---|---|
| date | 2026-06-16 |
| machine | Darwin arm64 |
| Python | 3.12.6 |
| Symbolica | 2.0.0 local `symbolica-community` wheel patched to `symbolica/dev` |
| Symbolica dev commit | `07f1de5fc119b01e2875c8d0163b25eacabadf21` |
| pySecDec | 1.6.6 |
| Normaliz | not found on `PATH`; iterative/geometric_ku paths used |
| heavy DOT derivative mode | `--symbolic-derivatives` |
| default precision thresholds | `1e-8` and `1e-12` |
| default precision digits | 100 and 1000 |

## Symbolica Dev Dualization Check

The standalone reproducer `U_dualization_slowdown.py` was used before and
after installing a local `symbolica-community` wheel patched to
`symbolica/dev` commit `07f1de5fc119b01e2875c8d0163b25eacabadf21`.  The
default case is the triple-box U polynomial with the six-axis dual shape
`[3,3,3,3,3,4]`, i.e. 5120 requested Taylor coefficients.

| Symbolica source | scalar evaluator build [s] | copied evaluator dualize [s] | speedup |
|---|---:|---:|---:|
| previous venv wheel | 0.000282 | 191.316 | 1x |
| local community/dev wheel | 0.000200 | 11.385 | 16.8x |

The quick two-axis sanity case stayed sub-millisecond after the replacement
(`dualize = 0.000214 s`).  This removes the previous several-minute U/F
dualization bottleneck for the exact six-axis shape that motivated the
standalone script.  It does not automatically make the generated
regular-Taylor formula path cheap: some six-axis regular-source expressions
are still costly to build or dualize.  The regular-Taylor defaults now allow
attempting these six-axis boxes (`--regular-taylor-formula-volume-limit 8192`,
`--regular-taylor-formula-axis-limit 6`), while the guards remain configurable
for fallback studies.

Common run presets live in `examples/runs`.  CLI options override YAML values:

```sh
.venv/bin/python FSD.py --run examples/runs/dot_box.yaml --max-iter 1
```

Long runs should use the memory watchdog:

```sh
./run_with_memory_watch.py \
  --limit-gb 30 \
  --poll-seconds 30 \
  -- .venv/bin/python FSD.py --run examples/runs/dot_triple_box.yaml
```

The wrapper terminates the child process group if the memory limit is exceeded
or if `stop.order` appears in the working directory.

## Generation Timing

FSD generation is reported in three headline buckets:

| bucket | meaning |
|---|---|
| Generation U and F polynomial | DOT parsing, kinematics, pySecDec loop-integral construction, U/F extraction, prefactor metadata, Symbolica expression conversion |
| Generating sectors | pySecDec sector decomposition and conversion to declarative `SectorDefinition` metadata |
| Generating Symbolica evaluators | scalar evaluators, sector map/Jacobian evaluators, derivative evaluators, endpoint projectors, regular Taylor formulas, chain-rule formulas |

Current topology overview:

| topology | input | sectors | Laurent range | FSD generation [s] | pySecDec generated-integrator generation [s] | FSD timing notes |
|---|---|---:|---|---:|---:|---|
| triangle | DOT | 3 | `eps^-2..eps^0` | 0.223 | 9.095 | avg 2.35 us/smpl/wkr |
| box | DOT | 12 | `eps^-2..eps^0` | 0.240 | 9.075 | avg 7.00 us/smpl/wkr |
| double box | DOT | 140 | `eps^-4..eps^0` | 0.615 | 272.81 | avg 18.23 us/smpl/wkr |
| triple box | DOT iterative | 1972 | `eps^-6..eps^0` | 290.23 + 457.19 serialization | not completed | one-point sector scan below |

The latest triple-box prepared bundle was generated with:

```sh
.venv/bin/python FSD.py generate \
  --dot-file examples/graphs/triple_box.dot \
  --kinematics examples/graphs/triple_box_kinematics.yaml \
  --output examples/outputs/prepared_triple_box_universal_eps0_limit288 \
  --sector-method iterative \
  --prefactor-convention pysecdec \
  --subtraction-backend projector-formula \
  --ibp-reduce-to-log-endpoint \
  --direct-projector-cache-term-threshold 0 \
  --symbolic-derivatives \
  --chain-rule-formula-signature-limit 4096 \
  --chain-rule-formula-output-length-limit 288 \
  --max-eps-order 0
```

Prepared triple-box artifact counts:

| artifact | count |
|---|---:|
| sectors | 1972 |
| endpoint-projector formulas | 360 |
| regular-Taylor formulas | 160 |
| universal chain-rule formulas | 181 |
| serialized evaluator files | 22996 |
| prepared bundle size | 4.1 GiB |
| generated cache size | 4.1 GiB |

Triple-box generation breakdown:

| component | time [s] |
|---|---:|
| Generation U and F polynomial | 0.198 |
| Generating sectors | 1.191 |
| Generating Symbolica evaluators | 288.842 |
| evaluator serialization | 457.193 |

The generated formula cache is now under top-level `cache/`.  It stores
reference Symbolica expression strings and serialized evaluator sidecars where
available.  It is ignored by git and intended as a downloadable distribution
cache; missing formulas are regenerated and added locally.

## Triple-Box One-Point Sector Scan

The prepared bundle was scanned by evaluating one deterministic point in every
sector with 10 workers and a 30 s per-sector diagnostic cap.  This scan is a
runtime/stability classification pass.

| metric | completed sectors |
|---|---:|
| count | 1746 |
| wall time min / median / p90 / p99 / max [s] | 0.0020 / 0.206 / 10.13 / 25.23 / 29.12 |
| Symbolica eval time min / median / p90 / p99 / max [s] | 0.0010 / 0.111 / 6.76 / 19.34 / 27.52 |
| Python/glue time min / median / p90 / p99 / max [s] | 0.00040 / 0.083 / 1.93 / 12.71 / 25.32 |
| `max|coefficient|` min / median / p90 / p99 / max | `2.83e-8` / 0.155 / 3.50 / 81.9 / `2.15e3` |
| precision rescue events | 0 |

The remaining 226 sectors reached the 30 s cap.  Those are not numerical
failures; they are the sectors still dominated by uncached high-axis
chain-rule/source assembly or very expensive evaluator formulas.

Representative completed high-weight sectors:

| sector | max coefficient | wall [s] | Symbolica eval [s] | Python/glue [s] |
|---|---:|---:|---:|---:|
| `PSD350` | `2.15e3` | 28.63 | 18.28 | 10.35 |
| `PSD2` | `5.80e2` | 13.31 | 0.011 | 13.30 |
| `PSD671` | `5.73e2` | 9.05 | 7.86 | 1.19 |
| `PSD201` | `5.57e2` | 8.37 | 4.68 | 3.69 |
| `PSD106` | `3.90e2` | 12.59 | 0.010 | 12.58 |

Representative slow completed sectors:

| sector | wall [s] | Symbolica eval [s] | Python/glue [s] | max coefficient |
|---|---:|---:|---:|---:|
| `PSD364` | 29.12 | 19.46 | 9.65 | 67.6 |
| `PSD363` | 28.99 | 18.38 | 10.62 | 4.21 |
| `PSD697` | 28.91 | 27.52 | 1.39 | 2.20 |
| `PSD349` | 28.90 | 17.73 | 11.16 | 0.326 |
| `PSD350` | 28.63 | 18.28 | 10.35 | `2.15e3` |

After installing the local `symbolica-community` wheel patched to
`symbolica/dev`, the old prepared-bundle `PSD649` one-point profile was rerun
without regenerating the bundle:

| sector | mode | wall [s] | Symbolica eval [s] | Python/glue [s] | max coefficient |
|---|---|---:|---:|---:|---:|
| `PSD649` | existing strict bundle, symbolic-derivative fallback | 80.36 | 0.034 | 80.32 | 21.0 |
| `PSD649` | repeated strict bundle, symbolic-derivative fallback | 53.06 | 0.005 | 53.05 | 2.24 |
| `PSD649` | diagnostic direct U/F duals, second one-point repeat | 46.62 | 39.39 | 7.23 | 13.7 |
| `PSD649` | diagnostic direct U/F duals, 10-point repeat | 143.11 | 106.28 | 36.83 | `1.68e7` |

This confirms that the old prepared artifact is not bottlenecked by U/F
`Evaluator.dualize()` at runtime.  It is still bottlenecked by missing fused
source/chain evaluators.  Switching to direct U/F duals moves most of the
one-point cost into Symbolica, but does not improve vectorized batch cost for
this sector.

A cache-warming probe for the first missing `PSD649` six-axis regular-Taylor
signature was stopped after about 170 s.  The scalar-dualized construction was
interrupted inside `Evaluator.dualize(...)`; an explicit coefficient-extraction
construction was also still building after a similar time.  This is the
current blocker for a fully baked triple-box runtime bundle.

This shows two distinct bottlenecks.  Some hard sectors are real Symbolica
evaluator work.  Others, such as `PSD2` and `PSD106`, spend almost all time in
Python sparse Taylor composition because their universal chain-rule signature
was not prepared under the current output-length cap.

## Practical Conclusion

The prepared DOT bundle path now works for triangle, box, double box, and the
full triple-box sector list.  It cleanly separates generation from integration:
strict prepared integration performs no pySecDec work and no evaluator
generation.

The remaining performance weak point is the high-depth triple-box source
assembly.  The next optimization should either cache more universal
chain-rule/source formulas as Symbolica evaluators or move sparse-series
composition into a native Symbolica-side primitive.  Until that is done, the
full `1972 x 1000` democratic triple-box scan is not a 1000 s run.
