# Advection

Yelmo.jl uses an **explicit first-order upwind** scheme for 2D tracer
advection on a regular Arakawa C-staggered grid. The same kernel
([`advect_tracer!`](@ref)) is used twice in a single `topo_step!`:

- Phase 2 — to advect ice thickness `H_ice` at the depth-averaged
  ice velocity `(ux_bar, uy_bar)`.
- Calving phase 6 — to advect the level-set function `lsf` at the
  calving-front velocity `(ux_bar + cr_acx, uy_bar + cr_acy)`.

This page describes the math, the CFL self-protection, and the design
choices that differ from the Fortran reference.

## Continuous problem

We integrate the conservative tracer-advection equation

```math
\frac{\partial c}{\partial t} + \nabla \cdot (\mathbf{u}\, c) = 0
```

in 2D over a time interval `[0, dt]`, with `c(x, y, t)` cell-centred
(aa-node) and the velocity `(u, v)` on the staggered acx / acy faces.
The velocity is held fixed during the call.

For ice thickness this is mass conservation (tracer = thickness, no
diffusive term in this kernel — sources / sinks like SMB and BMB are
applied separately as tendencies after advection; see the
[mass-balance page](mass-balance.md)).

For the level-set function this is a kinematic transport: we advect
`lsf` passively at the front velocity, and the zero level set
co-moves with the ice front.

## Discretisation

The kernel uses Forward Euler in time and
`Oceananigans.Advection.UpwindBiased(order=1)` in space. Per
sub-step:

```math
c_{ij}^{n+1} = c_{ij}^{n} - \Delta t_\mathrm{sub}\,
  \bigl[\nabla \cdot (\mathbf{U} c)\bigr]_{ij}^n
```

with the upwind divergence computed via Oceananigans' `div_Uc`
operator. The scheme is:

- **Conservative** (flux-form upwind on a uniform grid).
- **Monotone**: produces no new extrema. For `H_ice ≥ 0` initial
  data, `H_ice` stays non-negative.
- **First-order accurate** in space and time. Diffusive — sharp
  fronts smear over a few cells per advected length. Adequate for
  ice-sheet thickness, where the leading-order behaviour is set by
  mass-balance sources rather than gradients of `H`. For
  cleaner level-set fronts, redistancing restores the slope each
  step (see the [calving page](calving.md)).

## CFL self-protection

The kernel is **internally CFL-aware**: each call to
[`advect_tracer!`](@ref) sub-steps until the requested outer `dt` is
reached, with each sub-step bounded by

```math
\Delta t_\mathrm{sub} \le \mathrm{cfl\_safety} \cdot
  \min\!\left(\frac{\Delta x}{|u|_\mathrm{max}},
              \frac{\Delta y}{|v|_\mathrm{max}}\right).
```

This means a caller can pass any outer `dt` (e.g. one year) without
thinking about stability. The default `cfl_safety = 0.1` matches
Fortran Yelmo's `ytopo.cfl_max`, leaving comfortable headroom under
`u + v` advection. Setting it higher trades stability margin for
fewer sub-steps; lower buys more headroom at the cost of throughput.

A zero-velocity call is a no-op (sub-step `dt = Inf` is clamped to
the outer `dt`, the upwind divergence is identically zero).

## Boundary handling

Boundary conditions are imposed via Oceananigans' halo machinery, not
per-cell branches. The Yelmo `H_ice` field is constructed with
**Dirichlet `H = 0`** halos on every domain edge (via
`_dirichlet_2d_field` inside the constructor), so the upwind stencil
reads zero past the boundary without any special-case code in the
kernel. Velocity halos are filled once at the start of each
[`advect_tracer!`](@ref) call (controlled by
`fill_velocity_halos`).

The level-set function uses default Oceananigans halo handling
(extrapolation), since `lsf` should not jump to a special value past
the domain edge — it just continues in its current trend.

## Differences from Fortran

The Fortran reference offers two solvers (`expl-upwind` and
`impl-lis`). Only the explicit form is ported. The
[`advect_tracer!`](@ref) signature accepts a `scheme` keyword for an
implicit drop-in later — call sites do not need to change when the
implicit solver lands.

The Fortran kernel does not sub-step internally; it requires the
caller to honour the CFL constraint via the predictor / corrector
adaptive timestep. The Julia kernel sub-steps so that callers
unfamiliar with the CFL still produce stable runs.

## Limitations

The current kernel requires a **uniform** grid spacing in `x` and
`y`. Stretched grids error out with a clear message (see
[`advect_tracer!`](@ref)). Generalising to stretched grids is a
straightforward replacement of the per-cell `dx`, `dy` lookups; not
yet on the milestone list.

## Tests

`test/test_yelmo_topo.jl` covers `advect_tracer!` via:

- Direct kernel-level unit tests on synthetic velocity fields
  (passive transport at uniform `u`, mass conservation under a
  closed flow, no advance with zero velocity).
- Indirect coverage through `topo_step!` integration tests
  (slab conservation, Greenland 16km five-step smoke).
- The level-set test inventory exercises advection again for the
  calving pipeline (see the [calving page](calving.md)).
