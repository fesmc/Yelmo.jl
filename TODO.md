# Yelmo.jl porting & improvement TODO

Items to bring Yelmo.jl closer to feature parity with Fortran Yelmo, in priority
order. Items 1–7 are straightforward ports or already-operational features in
Fortran. Items 8–10 are larger efforts.

## Planned

1. **Advection allocation cleanup** — apply the wrapper + parametric-kernel
   template to `src/topo/advection.jl` to drop per-timestep allocations.
2. ~~**Hybrid solver dispatch**~~ — *already operational* (`dyn_step!`
   handles `"hybrid"` and the SIA+SSA additivity test was already
   passing). The leftover sub-task was to add the `"diva-noslip"`
   variant, which mirrors Fortran's `solver = "diva-noslip"` keyword
   (forces `no_slip = true` regardless of `y.p.ydyn.no_slip`).
3. ~~**Principal stresses → von-Mises (`vm-m16`) calving**~~ — done.
   `mat_step!` already computes `strs2D_tau_eig_1`; the calving stub
   was replaced with a faithful port of Fortran's
   `calc_calving_rate_vonmises_m16` and threaded through
   `calving_step!` via `_dispatch_calving!`.
4. **Calving parameter dispatch** — validate calving method names at init
   time so unsupported choices fail fast (currently runtime error).
5. **Thread Picard / matrix assembly loops** in SSA/DIVA solvers.
6. **Ice age / passive tracer** — port `ice_tracer.f90` (3D advection,
   column solvers, BMB coupling).
7. **Regions infrastructure** — port `yelmo_regions.f90` (regional masks,
   regional analysis utilities).
8. **Enthalpy solver validation** — benchmark the current port against
   reference data.
9. **Regridding infrastructure** — port `yelmo_regridding.f90`.
10. **Predictor–corrector schemes FE-SBE / AB-SAM** — finish the timestepping
    extension (Lever 1 in the PC refactor design).

## Explicitly out of scope

- L1L2 solver
- `calving_aa` (vertical-MB calving on aa-cells)
- Subgrid grounding-line discharge schemes
