# `benchmarks/`

Standalone, runnable reproductions of the standard ice-sheet model
benchmarks in Yelmo.jl. Each subdirectory is a self-contained
benchmark with its own Julia project, scripts, and (regenerable)
outputs. Unlike `test/`, these are **not** asserted against fixtures
or analytical solutions and are **not** run as part of CI — they
exist for inspection, qualitative comparison, and publication-grade
reproductions.

For asserted regression tests with tolerances against fixtures or
analytical references, see `test/benchmarks/` instead.

## Layout

- `IceSheetBenchmarks/` — vendored portable Julia package providing
  the model-agnostic benchmark specs (`AbstractBenchmark`,
  `EISMINT1MovingBenchmark`, …) and a `YelmoBenchmarks` package
  extension that adds `Yelmo.YelmoModel(::AbstractBenchmark, t)`
  when both packages are loaded. Long-term home is the standalone
  [fesmc/IceSheetBenchmarks.jl](https://github.com/fesmc/IceSheetBenchmarks.jl)
  repo.
- `<benchmark-name>/` — one directory per benchmark, each a
  self-contained Julia project with its own `Project.toml` /
  `Manifest.toml`, plus `run.jl` (simulation), `summary.jl`
  (post-processing), `summary.json` (committed reference values),
  and a `README.md`.
- `new.jl` — scaffolder; see "Adding a new benchmark" below.

## Available benchmarks

| Benchmark                           | Status   | Description |
| ----------------------------------- | -------- | ----------- |
| [`eismint1-moving`](eismint1-moving)| ✅       | EISMINT-1 moving-margin (Huybrechts et al. 1996), SIA + adaptive PC. |
| [`mismip3d-stnd`](mismip3d-stnd)    | ⚠️ BROKEN | MISMIP3D Stnd (Pattyn et al. 2013), SSA + grounding-line dynamics on a sloped marine bed. Currently does not reproduce the reference — SSA velocities saturate at the 5000 m/yr clamp. |

## Running a benchmark

Each benchmark is a self-contained Julia project. From the benchmark
directory:

```bash
cd benchmarks/<name>
julia --project=. run.jl
julia --project=. summary.jl
```

The first run will instantiate the project (resolving Yelmo.jl as a
local `develop`-ed dependency from the repo root). Outputs are
written to `output/` (NetCDF time series, restart snapshots) and
`plots/` (PNG figures, when generated). Both directories are
gitignored. A small `summary.json` with high-level statistics
(committed) summarises the run for cross-version comparison.

## Adding a new benchmark

Use the scaffolder to create the directory layout, project files, and
template scripts:

```bash
julia benchmarks/new.jl <name>
```

This creates `benchmarks/<name>/` with:

```
README.md       # benchmark-specific notes, references, expected outputs
Project.toml    # local Julia project (Yelmo.jl + plotting/IO deps)
Manifest.toml   # committed for reproducibility
run.jl          # main run script — flags configured at the top
summary.jl      # post-processing: reads output/ → writes summary.json
summary.json    # committed reference values + run metadata
.gitignore      # output/, plots/
```

After scaffolding, edit `run.jl` and `summary.jl` to implement the
specific benchmark, then add a row to the table above and commit.

While developing, the benchmark spec (`MyBenchmark <: AbstractBenchmark`)
can live inline at the top of `run.jl` — once stable, move it into
`IceSheetBenchmarks/src/` so other benchmarks and downstream consumers
can use it.

## Conventions

- **Outputs in `output/`** (NetCDF time series, restart files) —
  gitignored, regenerable.
- **Plots in `plots/`** (PNG, PDF) — gitignored. Generation logic
  may live in `summary.jl` or a separate script.
- **`summary.json`** is committed and contains physics-only summary
  statistics (final volume, dome height, GL position, …) plus
  metadata (benchmark name, date, Yelmo version). Hardware-dependent
  fields (wall-clock, thread count) belong in the per-run NetCDF
  attributes inside `output/`, not in `summary.json`.
- **Configuration in `run.jl` directly** — no separate `config.toml`.
  Constants at the top of the file, with comments explaining the
  choice and pointing at the reference / Fortran namelist where
  relevant.
