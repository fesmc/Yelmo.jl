# `test/benchmarks/`

Validation infrastructure for `YelmoModel` against Fortran-internally-generated
benchmarks (BUELER, EISMINT, ISMIP-HOM, slab, trough, MISMIP+, CalvingMIP).

## How it works

Each benchmark uses one of two backends, dispatched on type:

  - **AbstractBenchmark** (`BuelerBenchmark <: AbstractBenchmark`) —
    a struct that carries the parameters of a closed-form benchmark
    and implements:
      - `state(b, t)` → analytical state at time `t` as a NamedTuple
        of schema-keyed fields,
      - `write_fixture!(b, path; times)` → serialize the state to a
        NetCDF restart,
      - `analytical_velocity(b, t)` → analytical depth-averaged ice
        velocity (where defined; a stub error otherwise).
    Plus a generic `YelmoModel(b, t; …)` constructor that builds a
    `YelmoModel` directly from `state(b, t)` with **no** NetCDF
    round-trip. No Fortran library required. Used for benchmarks
    with closed-form solutions (BUELER-B Halfar today; future
    BUELER-A / BUELER-C variants slot in alongside).
  - **YelmoMirror** (`BenchmarkSpec` in [`helpers.jl`](helpers.jl)) —
    drive YelmoMirror with a Yelmo namelist + synthetic-grid axes +
    an `setup_initial_state!` callback, and write a restart at each
    output time. Used for benchmarks without analytical solutions
    (EISMINT, ISMIP-HOM, slab, trough, MISMIP+, CalvingMIP).
    Requires `libyelmo_c_api.so` to expose `yelmo_init_grid` (already
    in the API) and `yelmo_restart_write` (added for this milestone).

Both backends produce restart-format NetCDFs that the existing
`YelmoModel(restart_file, time; …)` constructor can load. Tests can
also load the analytical state directly via `YelmoModel(b, t; …)` and
skip NetCDF entirely. CI never invokes [`regenerate.jl`](regenerate.jl)
— fixtures are pre-committed under [`fixtures/`](fixtures/).

## Layout

```
test/benchmarks/
├── README.md
├── helpers.jl              # YelmoBenchmarks module umbrella
│                           #   + BenchmarkSpec + run_mirror_benchmark! + load_fixture
├── benchmarks.jl           # AbstractBenchmark interface contract
│                           #   + generic YelmoModel(::AbstractBenchmark, t)
├── bueler.jl               # Halfar / Bueler analytical formulas
│                           #   + BuelerBenchmark <: AbstractBenchmark
├── regenerate.jl           # entry point (dispatches on spec type)
├── test_smoke.jl           # smoke test (loads fixture, basic checks)
└── fixtures/
    ├── README.md
    └── bueler_b_t1000.nc   # ← committed binary (see fixtures/README.md)
```

## Adding a new benchmark

### Analytical (closed-form) benchmark

1. Add a `<MyBenchmark> <: AbstractBenchmark` struct to a new file
   under `test/benchmarks/` (or alongside `bueler.jl` if related).
   Fields carry the parameters needed by the analytical formula.
2. Implement `state(b, t)` — return a NamedTuple of schema-keyed
   fields (e.g. `H_ice`, `z_bed`, `smb_ref`). Coordinates `xc` /
   `yc` (in metres) drive grid construction; other keys are routed
   into the appropriate component group via the variable-meta
   tables in `src/variables/model/`.
3. Implement `write_fixture!(b, path; times)` if the default NetCDF
   layout in `bueler.jl` doesn't fit. Most benchmarks can copy that
   structure verbatim and just swap in their own `state` call.
4. Push an instance of the struct into `SPECS` in `regenerate.jl`.
5. Run `julia --project=test test/benchmarks/regenerate.jl <name>`
   locally to produce the fixture; commit it under `fixtures/`.
6. Add a test that loads the fixture via the file-based and
   in-memory paths and validates against the analytical formula.

### YelmoMirror (Fortran-internally-generated) benchmark

1. Add a Yelmo namelist under `specs/` (e.g. `specs/yelmo_EISMINT-moving.nml`).
   Disable all file loads (`init_topo_load`, `basins_load`, `regions_load`,
   every `pd_*_load`) — initial state comes from Julia.
2. Add a spec module under `specs/` defining a `BenchmarkSpec` const and an
   `setup_initial_state!` callback. The callback mutates `ymirror.tpo.H_ice`
   and `ymirror.bnd.*` then calls `yelmo_sync!(ymirror)`.
3. Push the spec into `SPECS` in `regenerate.jl`.
4. Run `julia --project=test test/benchmarks/regenerate.jl <spec_name>` locally
   to produce the fixture.
5. Commit the fixture under `fixtures/`.
6. Add a test that loads the fixture via `load_fixture` and validates against
   the YelmoMirror reference state (the file you just generated).

## Per-milestone benchmark mapping

The benchmark portfolio grows incrementally with the dyn solvers:

| Solver milestone | Benchmark added | Reason |
|---|---|---|
| 3a (kinematics)  | none yet                              | no velocity solver to validate |
| 3b (N_eff / cb)  | none                                   | restart-derived consistency only |
| **scaffolding**  | `bueler_b` (3c step 1 / step 2)       | proves the API works |
| 3c (SIA)         | `bueler_b` SIA convergence (Commit 4) | SIA-only flow vs analytical Halfar |
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
