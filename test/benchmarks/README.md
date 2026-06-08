# `test/benchmarks/`

Validation infrastructure for `YelmoModel` against the standard
ice-sheet benchmarks (BUELER, EISMINT, ISMIP-HOM, slab, trough,
MISMIP+, CalvingMIP). The benchmark **specs** (struct + closed-form
state + reference geometry) live in
[`IceSheetBenchmarks`](../../benchmarks/IceSheetBenchmarks/); this
directory holds the **Yelmo-side scaffolding** that drives YelmoModel
/ YelmoMirror through those specs, generates / loads reference
fixtures, and runs the actual `@testset` assertions.

## How it works

Each benchmark uses one of two backends:

  - **AbstractBenchmark** (e.g. `BuelerBenchmark <: AbstractBenchmark`)
    — a model-agnostic spec from `IceSheetBenchmarks` that implements:
      - `state(b, t)` → analytical state at time `t` as a NamedTuple
        of schema-keyed fields,
      - `write_fixture!(b, path; times)` → serialise the state to a
        NetCDF restart,
      - `analytical_velocity(b, t)` → analytical depth-averaged ice
        velocity (where defined; a stub error otherwise).
    Plus a generic `YelmoModel(b, t; …)` constructor — provided by
    the `YelmoBenchmarks` package extension inside ISB — that builds
    a `YelmoModel` directly from `state(b, t)` with **no** NetCDF
    round-trip. No Fortran library required. Used for benchmarks
    with closed-form analytical solutions (BUELER, HOM-C).
  - **YelmoMirror** (driven by `BenchmarkSpec` in
    [`harness.jl`](harness.jl) via `generate_fixture!`) — drive
    YelmoMirror with a Yelmo namelist + synthetic-grid axes + an
    `setup_initial_state!` callback, and write a restart at each
    output time. Used for benchmarks without analytical solutions
    (Trough, MISMIP3D `t > 0`, EISMINT `t > 0`, CalvingMIP). Requires
    `libyelmo_c_api.so` to expose `yelmo_init_grid` and
    `yelmo_restart_write`.

Both backends produce restart-format NetCDFs that the existing
`YelmoModel(restart_file, time; …)` constructor can load. Tests can
also load the analytical state directly via `YelmoModel(b, t; …)` and
skip NetCDF entirely. CI never invokes
[`regenerate.jl`](regenerate.jl) — fixtures are pre-committed under
[`fixtures/`](fixtures/).

## Layout

```
test/benchmarks/
├── README.md
├── harness.jl              # YelmoBenchmarkHarness module umbrella:
│                           #   BenchmarkSpec, generate_fixture!,
│                           #   load_fixture, _spec_name dispatch
├── yelmo/                  # per-benchmark Yelmo-side glue:
│   ├── trough.jl           #   _setup_*_initial_state! IC callbacks,
│   ├── eismint_moving.jl   #   state(b, t > 0) fixture readers,
│   ├── mismip3d.jl         #   write_fixture!(b, t > 0) Mirror branches
│   └── calvingmip.jl
├── regenerate.jl           # entry point (dispatches on spec type)
├── regen_*.jl, bench_*.jl, diff_*.jl   # debug / perf helpers
├── test_*.jl               # @testset assertions (run by CI)
├── specs/                  # Yelmo Fortran .nml namelists
│   ├── yelmo_TROUGH.nml
│   ├── yelmo_MISMIP3D.nml
│   ├── yelmo_EISMINT_moving.nml
│   └── yelmo_CalvingMIP_exp*.nml
└── fixtures/               # committed reference NetCDFs
    └── *.nc
```

## Adding a new benchmark

### Analytical (closed-form) benchmark

1. Add a `<MyBenchmark> <: AbstractBenchmark` struct in
   `benchmarks/IceSheetBenchmarks/src/`. Fields carry the parameters
   needed by the analytical formula.
2. Implement `state(b, t)` — return a NamedTuple of schema-keyed
   fields (e.g. `H_ice`, `z_bed`, `smb_ref`). Coordinates `xc` /
   `yc` (in metres) drive grid construction; other keys are routed
   into the appropriate component group via the variable-meta tables
   in `src/variables/model/`.
3. Implement `write_fixture!(b, path; times)` if the default NetCDF
   layout in `bueler.jl` doesn't fit. Most benchmarks can copy that
   structure verbatim and just swap in their own `state` call.
4. Add an `IceSheetBenchmarks.include(...)` + export entry to
   `IceSheetBenchmarks.jl`.
5. Add a Yelmo-side `_spec_name(::MyBenchmark)` in `harness.jl` (or
   under `yelmo/`) — that string becomes the fixture-filename stem.
6. Push an instance into `SPECS` in `regenerate.jl`.
7. Run `julia --project=test test/benchmarks/regenerate.jl <name>`
   locally to produce the fixture; commit under `fixtures/`.
8. Add a `test_*.jl` that loads the fixture via the file-based and
   in-memory paths and validates against the analytical formula.

### Yelmo-driven (YelmoMirror-reference) benchmark

1. Add the bare spec struct (geometry + forcing parameters) to ISB.
   `state(b, t > 0)` is host-provided.
2. Add a Yelmo namelist under `specs/`. Disable all file loads
   (`init_topo_load`, `basins_load`, `regions_load`, every `pd_*_load`)
   — initial state comes from Julia.
3. Add a `yelmo/<benchmark>.jl` file with:
     - a `_<benchmark>_namelist_path(b)` resolver,
     - `_spec_name(b)` / `_<benchmark>_fixture_path(b, t)`,
     - `state(b, t > 0)` that reads the committed fixture,
     - `_setup_<benchmark>_initial_state!(ymirror, b, time)` IC callback,
     - `write_fixture!(b, path; times)` that calls `generate_fixture!`
       with a `BenchmarkSpec` built from the above.
   `include` it from `harness.jl`.
4. Push the spec into `SPECS` in `regenerate.jl`.
5. Run `julia --project=test test/benchmarks/regenerate.jl <name>`
   locally to produce the fixture; commit under `fixtures/`.
6. Add a `test_*.jl` that loads via `load_fixture` and validates
   against the YelmoMirror reference.

## Per-milestone benchmark mapping

The benchmark portfolio grows incrementally with the dyn solvers:

| Solver milestone | Benchmark added | Reason |
|---|---|---|
| 3a (kinematics)  | none yet                              | no velocity solver to validate |
| 3b (N_eff / cb)  | none                                   | restart-derived consistency only |
| **scaffolding**  | `bueler_b` (3c step 1 / step 2)       | proves the API works |
| 3c (SIA)         | `bueler_b` SIA convergence            | SIA-only flow vs analytical Halfar |
| 3d (SSA)         | `trough`, `mismip+`                   | marine ice-stream tests |
| 3f (DIVA)        | `bueler_b_full` (analytical lockstep) | hybrid SIA/SSA flow |
| post-3a          | `calvingmip`                          | calving + dyn coupled |

## Regenerating fixtures

```bash
# From repo root, regenerate everything (refuses to clobber):
julia --project=test test/benchmarks/regenerate.jl

# Force regenerate one spec:
julia --project=test test/benchmarks/regenerate.jl bueler_b --overwrite
```
