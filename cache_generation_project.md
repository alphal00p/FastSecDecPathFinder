# FSD Universal Cache Generation Project

This is the cluster handoff for generating the distributable FastSecDec
formula cache.  It is written for a fresh worker where an agent will clone the
repository, build the Python environment, install a custom `symbolica-community`
wheel against latest `symbolica/dev`, run cache generation under a memory
watchdog, and package the resulting cache.

## Current Status

FSD now has a first-class cache-warming mode:

```sh
.venv/bin/python FSD.py cache ...
```

This mode exercises the topology-independent pieces used by the
`projector-formula` backend:

- endpoint-projector formulae;
- regular Taylor source formulae;
- mapped-derivative chain-rule formulae.

The cache root is:

```text
cache/subtraction_formulae
```

The cache itself is intentionally ignored by git.  The repository stores only
the code, documentation, and timing reports; the generated cache should be
compressed as `FSD_cache.tar.gz` and distributed separately.

The current local verification covered the complete shipped 1L and 2L DOT
examples in two variants:

- direct/default endpoint projectors:
  `docs/universal_cache_report.json`;
- IBP endpoint lowering:
  `docs/universal_cache_report_ibp.json`.

Both variants verified successfully on the 1L/2L examples.  The 2L double-box
is the first case that generated substantial new universal cache entries.  The
3L family is expected to be dominated by six-axis signatures and should be run
on a cluster rather than a laptop.

## Fresh Clone

Clone the repository and enter it:

```sh
git clone git@github.com:alphal00p/FastSecDecPathFinder.git
cd FastSecDecPathFinder
```

If SSH keys are not available on the cluster, use HTTPS instead:

```sh
git clone https://github.com/alphal00p/FastSecDecPathFinder.git
cd FastSecDecPathFinder
```

## System Requirements

Install or load modules for:

- Python 3.12 or newer;
- a recent Rust toolchain (`cargo`, `rustc`);
- C/C++ compiler toolchain;
- `git`;
- enough RAM per worker for heavy Symbolica formula generation;
- optional but useful: `tmux` or `screen`.

Normaliz is not required for the commands below because they use
`--sector-method iterative`.  If you switch to `--sector-method geometric`,
make sure `normaliz` is on `PATH` or pass `--normaliz-executable`.

## Python Environment And OneLOopBridge

The normal setup creates `.venv`, installs Python dependencies, and builds the
required external OneLOopBridge bindings:

```sh
./install.sh --clone-oneloopbridge
```

If you already have a OneLOopBridge checkout:

```sh
ONELOOPBRIDGE_SRC=/path/to/OneLOopBridge ./install.sh
```

The install script verifies `import oneloop_bridge`.  Cache generation itself
does not use OneLOopBridge, but the FSD environment is expected to have it
available.

## Install Symbolica Community Against Latest Symbolica Dev

The cache-generation runs should use the latest `symbolica-community` built
against `symbolica/dev`, because the dev branch contains the dualization speed
fix relevant to high-axis sector work.

Build and install the wheel inside the FSD `.venv`:

```sh
mkdir -p .deps
git clone https://github.com/symbolica-dev/symbolica-community .deps/symbolica-community-dev

SYMBOLICA_DEV_COMMIT="$(git ls-remote https://github.com/symbolica-dev/symbolica.git refs/heads/dev | awk '{print $1}')"
echo "Using symbolica/dev commit ${SYMBOLICA_DEV_COMMIT}"

.venv/bin/python - <<'PY'
from pathlib import Path
import os
import re

commit = os.environ["SYMBOLICA_DEV_COMMIT"]
cargo = Path(".deps/symbolica-community-dev/Cargo.toml")
text = cargo.read_text()
patch = (
    '[patch.crates-io]\n'
    f'symbolica = {{ git = "https://github.com/symbolica-dev/symbolica.git", rev = "{commit}" }}\n'
)

if "[patch.crates-io]" in text:
    pattern = r"(?ms)^\[patch\.crates-io\]\n(?:^[^\[].*\n?)*"
    text = re.sub(pattern, patch, text)
else:
    text = text.rstrip() + "\n\n" + patch
cargo.write_text(text)
print(cargo)
PY

(
  cd .deps/symbolica-community-dev
  ../../.venv/bin/python -m pip install --upgrade maturin
  ../../.venv/bin/maturin build --release
)

.venv/bin/python -m pip install --force-reinstall .deps/symbolica-community-dev/target/wheels/symbolica-*.whl
```

Verify the installed Symbolica and rerun the dualization reproduction:

```sh
.venv/bin/python - <<'PY'
import symbolica
print(getattr(symbolica, "__version__", "n/a"))
print(symbolica.__file__)
PY

.venv/bin/python U_dualization_slowdown.py --quick
.venv/bin/python U_dualization_slowdown.py
```

The full `U_dualization_slowdown.py` run uses the old problematic
six-axis `[3,3,3,3,3,4]` dual shape.  With the fixed Symbolica backend, the
`dualize copied evaluator` stage should no longer take minutes.

## Baseline Checks

Before launching long cache jobs:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile FSD.py cache_warm.py integrand.py
```

## Memory Watchdog

Use the repository watchdog for long cache jobs:

```sh
.venv/bin/python run_with_memory_watch.py --limit-gb 500 --poll-seconds 30 -- \
  .venv/bin/python FSD.py cache --help
```

The wrapper monitors the full child process tree RSS.  It has no wall-time
timeout unless `--timeout-seconds` is explicitly supplied, so the default
cluster workflow should let the scheduler wall time control the job.

The wrapper also watches `./stop.order`.  Creating that file requests a clean
stop without needing an external `kill`; the wrapper owns the child process
group and will interrupt/terminate it itself:

```sh
touch stop.order
```

At startup the wrapper removes a stale `stop.order` file if one is present, so
it is safe to reuse the same working directory for repeated attempts.

For Slurm, also request a memory limit at the scheduler level.  Example:

```sh
srun --cpus-per-task=1 --mem=550G --time=24:00:00 \
  .venv/bin/python run_with_memory_watch.py --limit-gb 500 --poll-seconds 30 -- \
  .venv/bin/python FSD.py cache --cache-cases self_energy_3loop --cache-verify-samples-per-sector 1
```

## Generate The 1L And 2L Universal Cache

This is already done locally, but rerun it on the cluster to validate the fresh
environment and fill any missing cache entries:

```sh
.venv/bin/python FSD.py cache \
  --cache-loop-counts 1 2 \
  --cache-verify-samples-per-sector 8 \
  --cache-report-path docs/cluster_cache_1l2l_direct.json \
  --cache-workdir .cache_warm_cluster_1l2l_direct \
  --no-progress
```

Generate the IBP endpoint-lowering variant as well:

```sh
.venv/bin/python FSD.py cache \
  --cache-loop-counts 1 2 \
  --cache-verify-samples-per-sector 4 \
  --cache-report-path docs/cluster_cache_1l2l_ibp.json \
  --cache-workdir .cache_warm_cluster_1l2l_ibp \
  --ibp-reduce-to-log-endpoint \
  --direct-projector-cache-term-threshold 0 \
  --no-progress
```

## Generate The 3L Universal Cache

The 3L objective is to generate all universal formula signatures needed by the
shipped 3L DOT examples, especially the triple box.  Run direct/default and IBP
variants separately.  Start with the smaller 3L examples, then the triple box.

Direct/default projector cache:

```sh
.venv/bin/python run_with_memory_watch.py \
  --limit-gb 500 \
  --poll-seconds 30 \
  --log-file cluster_cache_3l_direct.watch.log \
  -- \
  .venv/bin/python FSD.py cache \
    --cache-cases self_energy_3loop three_point_3loop three_point_3loop_8line \
    --cache-verify-samples-per-sector 1 \
    --cache-report-path docs/cluster_cache_3l_direct_small.json \
    --cache-workdir .cache_warm_cluster_3l_direct_small \
    --force-regular-taylor-formulas \
    --chain-rule-formula-signature-limit 1000000 \
    --chain-rule-formula-output-length-limit 0 \
    --no-progress
```

Triple-box direct/default projector cache:

```sh
.venv/bin/python run_with_memory_watch.py \
  --limit-gb 500 \
  --poll-seconds 30 \
  --log-file cluster_cache_3l_triple_box_direct.watch.log \
  -- \
  .venv/bin/python FSD.py cache \
    --cache-cases triple_box \
    --cache-verify-samples-per-sector 1 \
    --cache-report-path docs/cluster_cache_3l_triple_box_direct.json \
    --cache-workdir .cache_warm_cluster_3l_triple_box_direct \
    --force-regular-taylor-formulas \
    --chain-rule-formula-signature-limit 1000000 \
    --chain-rule-formula-output-length-limit 0 \
    --no-progress
```

IBP endpoint-lowering 3L cache:

```sh
.venv/bin/python run_with_memory_watch.py \
  --limit-gb 500 \
  --poll-seconds 30 \
  --log-file cluster_cache_3l_ibp.watch.log \
  -- \
  .venv/bin/python FSD.py cache \
    --cache-loop-counts 3 \
    --cache-verify-samples-per-sector 1 \
    --cache-report-path docs/cluster_cache_3l_ibp.json \
    --cache-workdir .cache_warm_cluster_3l_ibp \
    --ibp-reduce-to-log-endpoint \
    --direct-projector-cache-term-threshold 0 \
    --force-regular-taylor-formulas \
    --chain-rule-formula-signature-limit 1000000 \
    --chain-rule-formula-output-length-limit 0 \
    --no-progress
```

Notes:

- `--force-regular-taylor-formulas` lifts local laptop guards.  This is the
  cluster mode for actually producing complete universal signatures.
- `--chain-rule-formula-signature-limit 1000000` and
  `--chain-rule-formula-output-length-limit 0` avoid skipping cold chain-rule
  signatures.
- `--cache-verify-samples-per-sector 1` keeps verification cheap while proving
  that each generated asset can be loaded and used.  Increase this only after
  the cache is complete.
- If a job hits the 500 GiB watchdog, split by `--cache-cases` and rerun the
  failing topology on a larger memory node.

## Expected Runtime Scale

The local 1L/2L cache pass generated the double-box universal assets in about
10 seconds and verified all 1L/2L DOT examples.  This is not predictive for
3L six-axis signatures.

Current calibrated planning estimates are:

- optimistic 2L-rate lower bound for triple-box-like 3L cache: about 431 s;
- hard six-axis lower calibration: about 50 h serial;
- hard direct-formula calibration: up to about 457 h serial.

The large range reflects whether the cluster run mainly hits source-group style
formulae or full direct six-axis formulae.  The work is signature-local and can
be parallelized by case/signature family.  On roughly 100 independent workers,
the current calibrated range is approximately 0.5 to 4.6 hours before IO and
per-worker RAM constraints.

## Package The Cache

After successful generation:

```sh
tar -czf FSD_cache.tar.gz cache
du -sh cache FSD_cache.tar.gz
```

Validate the archive in a fresh clone or temporary directory:

```sh
git clone git@github.com:alphal00p/FastSecDecPathFinder.git /tmp/FSD_cache_check
cd /tmp/FSD_cache_check
./install.sh --clone-oneloopbridge --cache-tar /path/to/FSD_cache.tar.gz
.venv/bin/python FSD.py cache \
  --cache-cases triple_box \
  --cache-verify-samples-per-sector 1 \
  --cache-report-path /tmp/FSD_cache_check_triple_box_report.json \
  --no-progress
```

If the report shows mostly cache hits and no missing-evaluator failures, the
archive is suitable for distribution.

## What To Commit Back

Commit code and markdown reports, not the cache itself.  The cache directory is
ignored.  Useful files to push back after the cluster run:

- `docs/cluster_cache_*.json`;
- any updated timing markdown;
- the exact `symbolica/dev` commit used;
- notes about RAM limits, failed signatures, or topology splits.

Do not commit:

- `cache/`;
- `.cache_warm*`;
- `.deps/`;
- generated pySecDec build directories;
- large watchdog logs unless they contain concise failure diagnostics.
