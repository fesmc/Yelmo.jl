# Physics overview

The Yelmo ice-sheet model evolves five coupled state components on a
2D Arakawa C-staggered grid. In `Yelmo.jl` they are exposed as the six
state groups `bnd`, `dta`, `dyn`, `mat`, `thrm`, `tpo`
(`bnd` is read-only forcing; the rest are evolved). One full
`step!(y, dt)` is, conceptually, an ordered phase chain:

```
step!(y, dt)
  ├─ topo_step! : ice-thickness + grounding + mass-balance bookkeeping
  ├─ dyn_step!  : velocity solve (currently SIA; SSA / DIVA upcoming)
  ├─ mat_step!  : strain rates, viscosity, anisotropy           [stub]
  └─ therm_step!: temperature advection + diffusion + basal melt [stub]
```

Coupling between the four components within a single step is handled
by Yelmo's predictor / corrector wrapping (currently in `YelmoMirror`,
not yet ported to `YelmoModel`). Each component reads the others' end-
of-previous-step state from the shared `y.{bnd,dyn,mat,thrm,tpo}`
groups.

## Status of the Julia port

Two of the four components are now wired through in
[`YelmoModel`](@ref). The remaining two have stub forwards (so they
plug into `step!(y, dt)` without changing call sites once the kernels
land):

| Phase | Module | Status | Notes |
|---|---|---|---|
| `topo_step!` | `Yelmo.YelmoModelTopo` | **Done** | See the [topography page](topography.md) for the full pipeline. |
| `dyn_step!`  | `Yelmo.YelmoModelDyn` | **Partial (SIA)** | Driving / lateral / basal stress, `N_eff`, bed-roughness, SIA solver dispatch. SSA / hybrid / DIVA deferred. See the [dynamics page](dynamics.md). |
| `mat_step!`  | (future) | Deferred | Strain-rate / viscosity / anisotropy. |
| `therm_step!`| (future) | Deferred | 3D advection-diffusion + basal melt. |

For users running production setups today, the
[`YelmoMirror`](../api/mirror.md) backend supports the complete
physics chain and exposes the same step / output API as `YelmoModel`.

## What is documented here

The pages under "Physics" describe the **mathematical content** of
each ported kernel — not just the function signature. Each page
covers:

- The physical quantity being computed.
- The discretisation choice (upwind, bilinear, half-plane clip,
  depth-recurrence, …).
- The Fortran reference being ported (function name, file, line).
- Numerical notes — corner cases, stabilisations, where the Julia
  port differs from the Fortran reference.
- The relevant tests in `test/test_yelmo_topo.jl`,
  `test/test_yelmo_dyn.jl`, and `test/test_yelmo_sia.jl`.

| Page | Topic |
|---|---|
| [Topography step](topography.md)        | The full pipeline that orchestrates every other topography page on a per-step basis. |
| [Advection](advection.md)                | Generic 2D tracer advection — used for `H_ice` and the calving level-set. |
| [Mass balance](mass-balance.md)          | SMB, BMB, FMB, DMB sources; `apply_tendency!`; residual cleanup. |
| [Grounded fraction](grounded-fraction.md)| Subgrid `f_grnd` via the CISM bilinear-interpolation scheme; analytical-limit fixes vs the Fortran reference. |
| [Relaxation](relaxation.md)              | Optional ice-thickness relaxation toward a reference state. |
| [Calving](calving.md)                    | Level-set front evolution, calving-front velocity laws, redistancing. |
| [Dynamics](dynamics.md)                  | Driving stress, lateral BC stress, effective basal pressure, bed-roughness chain, the SIA velocity solver under the Option C vertical-stagger convention. |

The order roughly follows the per-step phase ordering: topography
runs first, dynamics second.
