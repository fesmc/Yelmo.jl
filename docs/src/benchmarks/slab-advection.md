# Slab advection stability

**Test file**: `test/benchmarks/test_slab_advection.jl`  
**Solver tested**: Advection kernel only (`advect_tracer!`)  
**Validation**: Unconditional stability of the implicit upwind scheme

## Setup

This benchmark exercises the advection kernel in isolation — velocities are
prescribed analytically (uniform slab flow at `u₀ = 100 m/yr`) so the test
is completely decoupled from the SSA / SIA / friction pipeline.  The domain is
a tiny 32 × 4 periodic-x grid with `dx = 1 km`.

A Gaussian perturbation `ΔH` is placed at `x = L/4`:

```math
\Delta H(x) = \Delta H_\mathrm{amp}\, e^{-(x - x_0)^2 / (2\sigma^2)}, \quad
\Delta H_\mathrm{amp} = 10\text{ m}, \quad \sigma = 4\text{ km}
```

The test advects `ΔH` through the kernel and measures the amplitude factor
`√(∑ΔH²_final) / √(∑ΔH²_initial)`.  Under upwind diffusion, `factor < 1`
is expected for stable schemes; `factor ≫ 1` or NaN signals instability.

## What it tests

Three test sets cover the stability envelope:

| Test | CFL | Expected behaviour |
|---|---|---|
| Sub-CFL (`CFL = 0.5`) | 0.5 | Both implicit and explicit stable (`factor < 1`) |
| Super-CFL (`CFL = 5`) | 5.0 | Implicit still stable; explicit (no substepping) blows up |
| Extreme CFL (`CFL = 50`) | 50 | Implicit: one giant step produces a finite, decayed field |

The super-CFL test explicitly disables the explicit scheme's internal
substepper (`cfl_safety = 1e6`) to expose the raw kernel instability.
With the default `cfl_safety`, the explicit scheme subdivides into
`ceil(CFL / cfl_safety)` sub-steps and is also stable — but that measures
the substepper, not the scheme.

## Motivation

The implicit upwind advection scheme (`ytopo.solver = "impl"`) was added to
allow stable integration at large `dt` — necessary for adaptive-timestepping
runs where the PI42 controller may occasionally request steps with `CFL ≫ 1`
on isolated cells near the margin.  The slab test directly gates the key
property: backward-Euler implicit upwind is unconditionally stable.

## How to run

```bash
julia --project=test test/benchmarks/test_slab_advection.jl
```

No fixtures required.  The test constructs synthetic grids and fields
in-memory.

## API

```julia
# Direct kernel calls (no YelmoModel):
cache = init_advection_cache(grid)
advect_tracer!(c, u, v, dt; scheme=:upwind_implicit, cache=cache)
advect_tracer!(c, u, v, dt; scheme=:upwind_explicit)
advect_tracer!(c, u, v, dt; scheme=:upwind_explicit, cfl_safety=1e6)
```

The `cache` pre-allocates the sparse linear system; passing it avoids
per-call allocation for the implicit scheme.
