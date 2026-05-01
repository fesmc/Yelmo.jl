# Yelmo.jl

`Yelmo.jl` is a Julia interface to the Yelmo ice-sheet model. It exposes
two backends behind a common abstract type, [`AbstractYelmoModel`](@ref):

- [`YelmoMirror`](api/mirror.md) — a thin Julia wrapper around the
  **Fortran** Yelmo model (`yelmo/libyelmo`). Each `step!` call
  passes through to the Fortran solver via a C interface; Julia owns
  only the field arrays pulled back across the boundary.
- [`YelmoModel`](@ref) — a **pure-Julia** ice-sheet solver, built on
  Oceananigans `Field`s. It is the focus of an in-progress port of the
  Fortran physics, organised into per-component `<comp>_step!`
  functions called in fixed phase order from `step!(::YelmoModel, dt)`.

Both backends share the same six-component state layout (`bnd`, `dta`,
`dyn`, `mat`, `thrm`, `tpo`), the same NetCDF restart format, and the
same output module ([`init_output`](@ref) / [`write_output!`](@ref)),
so the two are interchangeable at the run-script level.

## What this site covers

- **Getting started** — install, link the Fortran tree, run a five-step
  smoke test, run the analytical BUELER-B benchmark.
- **Concepts** — the model state layout, the staggered grid, parameter
  groups, and the Mirror-vs-Model split.
- **Usage** — loading a model from a restart, stepping it, writing
  NetCDF output, and lockstep-comparing two backends.
- **Physics** — narrative + math for each ported component: the
  topography step's per-phase pipeline, advection, mass-balance
  bookkeeping, the subgrid grounded fraction (with derivation of the
  analytical-limit fixes), relaxation, level-set calving, and the
  dynamics step (driving stress, lateral BC, `N_eff`, bed-roughness
  chain, SIA solver).
- **API reference** — auto-generated from the package's docstrings,
  one page per module.
- **Variables** — the canonical variable tables for the six state
  groups (used by the constructor to allocate fields with correct
  dimensions and grid staggering).

## Implementation status

The Fortran-backed [`YelmoMirror`](api/mirror.md) supports the full
Yelmo physics chain (anything Fortran Yelmo can do).

The pure-Julia [`YelmoModel`](@ref) is in active development. As of
the current milestone:

| Component | Status |
|---|---|
| `tpo` advection (explicit upwind)                          | Done |
| `tpo` mass balance: SMB / BMB / FMB / DMB pipeline         | Done (DMB stub) |
| `tpo` subgrid grounded fraction (linear / area / CISM)     | Done |
| `tpo` distance / mask / surface diagnostics                | Done |
| `tpo` relaxation                                            | Done (modes 1–3, `topo_rel == 4` deferred) |
| `tpo` calving (level-set flux form, 3 laws)                | Done |
| `tpo` predictor-corrector wrapper, `impl-lis` solver       | Deferred |
| `dyn` driving + lateral BC stress                          | Done |
| `dyn` effective pressure `N_eff` (6 methods)               | Done (subgrid sampling deferred) |
| `dyn` bed-roughness chain `cb_tgt → cb_ref → c_bed`        | Done |
| `dyn` SIA solver (`solver = "sia"`, Option C convention)   | Done |
| `dyn` SSA / hybrid / DIVA solvers                          | Deferred |
| `dyn` velocity Jacobian, `uz`, strain-rate tensor          | Deferred (milestone 3h) |
| `mat`, `thrm`                                              | Deferred (per-component step shells in place) |

See the [physics overview](physics/index.md) for the per-step phase
ordering, the [topography page](physics/topography.md) for the
full topography pipeline, and the [dynamics page](physics/dynamics.md)
for the velocity-solver chain.

## Quick example

```julia
using Yelmo

# Construct from a Yelmo Fortran restart NetCDF.
y = YelmoModel("yelmo_restart.nc", 0.0;
               alias = "demo",
               groups = (:bnd, :dyn, :mat, :thrm, :tpo),
               strict = false)

init_state!(y, 0.0)

# Open an output NetCDF.
out = init_output(y, "demo.nc")

# Step the model and write a slice each year.
for k in 1:5
    step!(y, 1.0)
    write_output!(out, y)
end

close(out)
```

A complete worked example with parameter overrides and selective
output is in the [getting-started](getting-started.md) guide.
