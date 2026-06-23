# FastSecDec Path Finder

FastSecDec is a prototype for sector-decomposed numerical integration of
scalar Feynman-parameter integrals.  The main supported workflow is:

1. read a GammaLoop-style DOT graph or a direct Symanzik `U/F` input,
2. use pySecDec for graph-to-parametric data and sector generation,
3. generate explicit Symbolica sector evaluators,
4. integrate with FSD's Havana or QMC drivers, or run native pySecDec for
   comparison.

The default FSD generation path builds explicit sector integrand evaluators.
The native pySecDec mode is available with `--dot-engine pysecdec`; in that
mode FSD only prepares the DOT/kinematics boundary and lets pySecDec generate
and run its own integrator.

`FSD.py` is the top-level CLI entry point.  The implementation modules live in
`src/`.

## Common Runs

These examples use `--jit-compile --complex-evaluator` for FSD runs.  That
keeps JIT enabled while avoiding the current real-valued JIT evaluator issue.
`--output` stores reusable FSD generated artifacts.  Native pySecDec uses
`--pysecdec-workdir` instead.

One-loop massless box from DOT, integrated with FSD/QMC for about 30 seconds:

```sh
.venv/bin/python FSD.py \
  --run examples/runs/dot_box.yaml \
  --sampling-mode qmc \
  --target-integration-time 30 \
  --workers 10 \
  --result-path examples/outputs/dot_box_qmc_30s.json \
  --jit-compile --complex-evaluator \
  --output MyFSDOutputBox \
  --restart
```

The same one-loop box, integrated natively by pySecDec:

```sh
.venv/bin/python FSD.py \
  --run examples/runs/dot_box.yaml \
  --dot-engine pysecdec \
  --workers 10 \
  --pysecdec-workdir MyPySecDecOutput \
  --keep-pysecdec-workdir
```

Massless two-loop double box from DOT, integrated with FSD/Havana:

```sh
.venv/bin/python FSD.py \
  --run examples/runs/dot_double_box.yaml \
  --sampling-mode havana \
  --workers 10 \
  --samples-per-iter 1000000 \
  --batch-size 100000 \
  --max-iter 10 \
  --result-path examples/outputs/dot_double_box_havana.json \
  --jit-compile --complex-evaluator \
  --output MyFSDOutputDoubleBox \
  --restart
```

The same double box supplied directly through `U` and `F` polynomials:

```sh
.venv/bin/python FSD.py \
  --run examples/runs/double_box_from_U_and_F.yaml \
  --sampling-mode havana \
  --workers 10 \
  --samples-per-iter 1000000 \
  --batch-size 100000 \
  --max-iter 10 \
  --result-path examples/outputs/double_box_from_U_and_F_havana.json \
  --jit-compile --complex-evaluator \
  --output MyFSDOutputDoubleBoxFromUandF \
  --restart
```

## Setup

Create the local environment and install the external OneLOopBridge binding:

```sh
./install.sh --clone-oneloopbridge
```

or point to an existing checkout:

```sh
ONELOOPBRIDGE_SRC=/path/to/OneLOopBridge ./install.sh
```

DOT mode requires pySecDec, pydot, and PyYAML.  The default sector method is
`iterative`; pySecDec's `geometric` method additionally requires Normaliz on
`PATH` or `--normaliz-executable`.

Large formula caches are intentionally not tracked.  If you have a packaged
cache, install it with:

```sh
./install.sh --cache-tar /path/to/FSD_cache.tar.gz
```

## Inputs

Run presets live in `examples/runs/`.  Paths in a run YAML are resolved
relative to that YAML file, and explicit CLI options override YAML values.

DOT examples and kinematics are in `examples/graphs/`.  Direct `U/F` input is
shown in `examples/runs/double_box_from_U_and_F.yaml`; that mode also requires
parametric metadata such as loop count, propagator powers, `U/F` exponents, and
the global prefactor.

Tracked target files under `examples/outputs/` are fixtures.  New run products
should be written under an ignored output location, for example with
`--output MyFSDOutput...` and `--result-path examples/outputs/...` for explicit
result files.

## Useful Commands

Run the test suite:

```sh
.venv/bin/python -m pytest -q
```

Show a saved result:

```sh
.venv/bin/python FSD.py --show-results examples/outputs/dot_double_box_pysecdec_target.json
```

Force native pySecDec output to stream to the terminal instead of the default
captured generation log:

```sh
.venv/bin/python FSD.py --run examples/runs/dot_box.yaml --dot-engine pysecdec --show-pysecdec-output
```

## More Detail

The old long-form README is kept as `exhaustive_README.md`.  The derivation
notes are in `docs/FastSecDec.tex` and `docs/FastSecDec.pdf`.
