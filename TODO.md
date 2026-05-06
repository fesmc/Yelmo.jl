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
4. ~~**Calving parameter dispatch**~~ — done. `ycalv_params(...)` now
   validates `calv_flt_method` / `calv_grnd_method` at construction
   when `use_lsf = true`, with separate error messages for unknown
   names vs known-but-unported Fortran methods (`vm-l19`, `simple`,
   `flux`, `kill`, `kill-pos`).
5. **Thread Picard / matrix assembly loops** in SSA/DIVA solvers.
   - Done (scope A): four hot per-cell kernels with no shared writes
     are now `@threads` on the row axis: `_picard_relax_visc_kernel!`,
     `_picard_relax_vel_kernel!`, `set_inactive_margins!`,
     `calc_basal_stress!`.
   - Out of scope (deferred): the SSA matrix assembly itself
     (`_assemble_ssa_matrix_kernel!` line 591) uses a shared
     monotonic counter `k` for COO triplet writes, and the COO→CSC
     reduction sums duplicate entries into the same `nzval` slot.
     Threading either requires a per-cell offset pre-scan and is a
     real refactor — leave for a future commit.
   - Out of scope: convergence-norm reductions
     (`picard_calc_convergence_l2`) need per-thread accumulators.
6. **Ice age / passive tracer** — partial port (scope A).
   Done: `calc_tracer_3D!` driver + implicit Crank-Nicolson column
   solver + horizontal 2nd-order upwind + Rybak-Huybrechts basal BC
   + per-column rate-of-change limiter, wired into `mat_step!` for
   `calc_age = true && tracer_method = "impl"` (`y.mat.dep_time`).
   Out of scope (deferred): explicit solver `tracer_method = "expl"`,
   `calc_isochrones` (`depth_iso` diagnostic), and the
   `enh_method ∈ {*-tracer}` paths.
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
