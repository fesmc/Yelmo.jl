# `test/benchmarks/`

Validation infrastructure for `YelmoModel` against Fortran-internally-generated
benchmarks (BUELER, EISMINT, ISMIP-HOM, slab, trough, MISMIP+, CalvingMIP).

## How it works

1. Each benchmark is described by a [`BenchmarkSpec`](helpers.jl) — namelist path,
   synthetic-grid axes, run length / output cadence, and an
   `setup_initial_state!` callback that seeds H_ice and boundary fields after
   YelmoMirror is constructed.
2. [`regenerate.jl`](regenerate.jl) runs each spec via YelmoMirror and writes a
   NetCDF restart fixture into [`fixtures/`](fixtures/) at every output time.
   Requires `libyelmo_c_api.so` to be present and to expose `yelmo_init_grid`
   (already in the API) and `yelmo_restart_write` (added for this milestone).
3. CI never invokes `regenerate.jl` — fixtures are pre-committed under
   `fixtures/`. Tests load them via [`load_fixture`](helpers.jl), which calls
   the existing `YelmoModel(restart_file, time; …)` constructor.

## Layout

```
test/benchmarks/
├── README.md
├── helpers.jl              # BenchmarkSpec + run_mirror_benchmark! + load_fixture
├── bueler.jl               # Halfar / Bueler analytical solutions
├── regenerate.jl           # entry point (run YelmoMirror → fixtures)
├── test_smoke.jl           # smoke test (loads fixture, basic checks)
├── specs/
│   ├── yelmo_BUELER-B.nml  # Yelmo namelist for BUELER-B
│   └── bueler_b_smoke.jl   # BenchmarkSpec definition
└── fixtures/
    ├── README.md
    └── bueler_b_smoke__t1000.nc   # ← committed binary (see fixtures/README.md)
```

## Adding a new benchmark

1. Add a Yelmo namelist under `specs/` (e.g. `specs/yelmo_EISMINT-moving.nml`).
   Disable all file loads (`init_topo_load`, `basins_load`, `regions_load`,
   every `pd_*_load`) — initial state comes from Julia.
2. Add a spec module under `specs/` defining a `BenchmarkSpec` const and an
   `setup_initial_state!` callback. The callback mutates `ymirror.tpo.H_ice`
   and `ymirror.bnd.*` then calls `yelmo_sync!(ymirror)`.
3. Push the spec into the registry in `regenerate.jl`.
4. Run `julia --project=test test/benchmarks/regenerate.jl <spec_name>` locally
   to produce the fixture.
5. Commit the fixture under `fixtures/`.
6. Add a test that loads the fixture via `load_fixture` and validates against
   either the analytical solution (BUELER) or the YelmoMirror reference state
   (the file you just generated).

## Per-milestone benchmark mapping

The benchmark portfolio grows incrementally with the dyn solvers:

| Solver milestone | Benchmark added | Reason |
|---|---|---|
| 3a (kinematics)  | none yet                              | no velocity solver to validate |
| 3b (N_eff / cb)  | none                                   | restart-derived consistency only |
| **scaffolding**  | `bueler_b_smoke` (this PR)            | proves the API works |
| 3c (SIA)         | `eismint_moving`, `ismiphom_a`        | SIA-only flow |
| 3d (SSA)         | `trough`, `mismip+`                   | marine ice-stream tests |
| 3f (DIVA)        | `bueler_b_full` (analytical lockstep) | hybrid SIA/SSA flow |
| post-3a          | `calvingmip`                          | calving + dyn coupled |

## Regenerating fixtures

```bash
# From repo root, regenerate everything (refuses to clobber):
julia --project=test test/benchmarks/regenerate.jl

# Force regenerate one spec:
julia --project=test test/benchmarks/regenerate.jl bueler_b_smoke --overwrite
```
