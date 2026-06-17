# FSD Performance Notes

These are development measurements for the DOT-backed FSD path as of
2026-06-17.  They are performance and stability diagnostics, not final
precision benchmarks.  FSD-owned code remains free of SciPy and SymPy imports;
pySecDec is used only at the DOT generation boundary.

## Environment

| item | value |
|---|---|
| machine | Darwin arm64 |
| Python | 3.12.6 |
| Symbolica | 2.0.0 local `symbolica-community` wheel patched to `symbolica/dev` |
| Symbolica dev commit | `07f1de5fc119b01e2875c8d0163b25eacabadf21` |
| pySecDec | 1.6.6 |
| Normaliz | not found on `PATH`; iterative/geometric_ku paths used |
| default precision thresholds | `1e-8` and `1e-12` |
| default precision digits | 100 and 1000 |

## Symbolica Dev Dualization Check

The standalone reproducer `U_dualization_slowdown.py` was used before and
after installing a local `symbolica-community` wheel patched to
`symbolica/dev`.  The default case is the triple-box U polynomial with the
six-axis dual shape `[3,3,3,3,3,4]`, i.e. 5120 requested Taylor coefficients.

| Symbolica source | scalar evaluator build [s] | copied evaluator dualize [s] | speedup |
|---|---:|---:|---:|
| previous venv wheel | 0.000282 | 191.316 | 1x |
| local community/dev wheel | 0.000200 | 11.385 | 16.8x |

This removes the original several-minute U/F dualization bottleneck.  It does
not by itself make every high-axis regular-source formula fast: the cost has
moved to how much source algebra/evaluator fragmentation we ask Symbolica and
Python to perform.

## Generation Timing

FSD generation is reported in three headline buckets:

| bucket | meaning |
|---|---|
| Generation U and F polynomial | DOT parsing, kinematics, pySecDec loop-integral construction, U/F extraction, prefactor metadata, Symbolica expression conversion |
| Generating sectors | pySecDec sector decomposition and conversion to declarative `SectorDefinition` metadata |
| Generating Symbolica evaluators | scalar evaluators, sector map/Jacobian evaluators, derivative evaluators, endpoint projectors, regular Taylor formulas |

Current topology overview:

| topology | input | sectors | Laurent range | FSD generation [s] | pySecDec generated-integrator generation [s] | FSD timing notes |
|---|---|---:|---|---:|---:|---|
| triangle | DOT | 3 | `eps^-2..eps^0` | 0.223 | 9.095 | avg 2.35 us/smpl/wkr |
| box | DOT | 12 | `eps^-2..eps^0` | 0.240 | 9.075 | avg 7.00 us/smpl/wkr |
| double box | DOT | 140 | `eps^-4..eps^0` | 0.615 | 272.81 | avg 18.23 us/smpl/wkr |
| triple box | DOT iterative | 1972 | `eps^-6..eps^0` | 38.46 recorded generation + 30.61 serialization | not completed | compressed prepared bundle, 30 GiB guard |

The latest completed compressed triple-box bundle was generated with
pregenerated dual evaluators, IBP endpoint lowering, and no chain-rule formula
backend:

```sh
.venv/bin/python FSD.py generate \
  --dot-file examples/graphs/triple_box.dot \
  --kinematics examples/graphs/triple_box_kinematics.yaml \
  --output examples/outputs/prepared_triple_box_dual_stream_probe \
  --sector-method iterative \
  --prefactor-convention pysecdec \
  --subtraction-backend projector-formula \
  --ibp-reduce-to-log-endpoint \
  --direct-projector-cache-term-threshold 0 \
  --pregenerate-dual-evaluators \
  --regular-taylor-signature-limit 100000 \
  --regular-taylor-formula-volume-limit 100000 \
  --regular-taylor-formula-axis-limit 5 \
  --max-eps-order 0
```

Prepared triple-box artifact counts:

| artifact | count / size |
|---|---:|
| sectors | 1972 |
| endpoint-projector formulas | 360 |
| regular-Taylor formulas | 166 |
| serialized evaluator files | 30572 |
| prepared bundle size | 27 GiB |
| generated top-level cache size | 22 GiB |
| legacy asset cache size | 11 GiB |

Raw `.bin` evaluator sidecars were tested and rejected: they reduced
compression CPU cost but grew the partial prepared bundle to roughly 30 GiB
after only about 5000 streamed evaluator files.  Compressed sidecars remain the
practical prepared-bundle format.

The `--pregenerate-single-overall-dual-evaluator` probe was also rejected for
the triple box: it still prepared more than 1200 streamed evaluator artifacts
and had not completed after 10 minutes.  It did not solve the source-evaluator
preparation bottleneck.

## PSD2 Direct Formula Probe

`PSD2` is a six-axis triple-box sector with singular axes `[0,1,3,5,7,8]`.
The current compressed bundle evaluates it through the sparse fallback for the
regular source algebra.  Repeated one-point timings separate cold evaluator
loading from warm steady state:

| path | repeat set | wall [s] | Symbolica eval [s] | Python/glue [s] |
|---|---|---:|---:|---:|
| sparse fallback | cold repeat 0 | 10.74 | 9.59 | 1.15 |
| sparse fallback | warm median repeats 2..4 | 1.15 | 0.760 | 0.390 |
| injected direct regular formulas | preparation repeat 0 | 69.57 | 20.58 | 48.99 |
| injected direct regular formulas | warm median repeats 2..5 | 10.58 | 9.97 | 0.612 |

The direct formula probe injected the 8 unique six-axis regular formula
signatures needed by PSD2.  The formulas themselves are universal and small in
count, but the direct path required thousands of source dual shapes and many
separate evaluator calls.  It reduced Python time but made total runtime much
worse.  This is the clearest evidence that the next optimization must fuse the
regular-source computation rather than creating many standalone coefficient
evaluators.

## Cache Strategy

Empty-cache generation time is no longer treated as the main user experience.
The intended distribution model is:

1. Build universal formula caches offline, potentially on a cluster.
2. Ship/download the cache under top-level `cache/`.
3. Generate topology-specific prepared bundles from that cache.
4. Run strict `integrate --output ...` with no pySecDec or evaluator
   generation.

The current local cache already shows the scale: tens of GiB, not source-repo
size.  That is acceptable for an optional downloaded cache archive.

## Practical Conclusion

The prepared DOT bundle path works for triangle, box, double box, and the full
triple-box sector list.  Strict prepared integration performs no pySecDec work
and no evaluator generation.  The remaining performance weak point is
high-axis source assembly for the triple box.  Blindly moving every sparse
coefficient into separate Symbolica formula evaluators is not sufficient; the
needed improvement is a coarser fused evaluator/source path or a native
Symbolica sparse-series primitive.
