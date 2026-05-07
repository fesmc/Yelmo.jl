# Yelmo.jl porting & improvement TODO

Items to bring Yelmo.jl closer to feature parity with Fortran Yelmo,
in priority order. Items 1–7, 9 and the bonus scope are done on this
branch; items 8 and 10 are deferred.

## Status

- ✅ 1, 2, 3, 4, 5, 6, 7, 9 + bonus (fill / filter helpers,
  aligned-resolution remapping)
- ⏸ 8 (enthalpy validation), 10 (FE-SBE / AB-SAM PC schemes) —
  deferred to a separate effort

## Done on this branch

1. ~~**Advection allocation cleanup**~~ — renamed
   `ImplicitAdvectionCache` → `AdvectionCache`, lifted the
   explicit-path `tend` buffer into the cache so the production
   `advect_tracer!` path is zero-alloc on both schemes.
2. ~~**Hybrid solver dispatch**~~ — already operational; added the
   `"diva-noslip"` variant that mirrors Fortran's solver-keyword
   override of the `no_slip` flag.
3. ~~**Principal stresses → vm-m16 calving**~~ — kernel ported
   (Morlighem 2016) and threaded through `calving_step!` via
   `y.mat.strs2D_tau_eig_1`.
4. ~~**Calving parameter dispatch validation**~~ — `ycalv_params(...)`
   validates `calv_flt_method` / `calv_grnd_method` at construction
   when `use_lsf = true`, with separate "unknown" vs
   "known-but-unported" error messages.
5. ~~**Thread SSA Picard kernels (scope A)**~~ — `@threads` on four
   hot per-cell kernels with no shared writes:
   `_picard_relax_visc_kernel!`, `_picard_relax_vel_kernel!`,
   `set_inactive_margins!`, `calc_basal_stress!`. The matrix
   assembly itself (shared COO counter) and the L2 reduction
   (per-thread accumulators) stay single-threaded — flagged as
   future work.
6. ~~**Ice age tracer (scope A)**~~ — `calc_tracer_3D!` + implicit
   Crank-Nicolson column solver + 2nd-order upwind horizontal
   advection + Rybak-Huybrechts basal BC + per-column rate-of-change
   limiter. Wired into `mat_step!` for `calc_age = true &&
   tracer_method = "impl"`. The explicit solver, isochrones, and
   the `enh_method ∈ {*-tracer}` paths are deferred.
7. ~~**Regions infrastructure (full port)**~~ — `src/regions/` with
   `init_regions`, `add_region!`, `update_regions!`,
   `write_regions!`, and `calc_region_diagnostics!`. Default
   whole-domain region auto-created. NetCDF output is one file per
   region with the mask written as a static 2D variable plus 39
   time-series scalar variables (units mirror the Fortran writer).
9. ~~**Regridding infrastructure**~~ — `src/utils/scrip_map.jl`
   vendors the core of `palma-ice/ScripMap.jl` (load + apply, plus
   pure-Julia replacements for `fill_weighted!` / `fill_nearest!` /
   `gaussian_filter!`) so the YelmoModel restart loader can
   regrid SCRIP-format weights generated externally by CDO. New
   constructor kwargs: `target_grid_file`, `maps_dir`,
   `regrid_method`. When ScripMap.jl is registered upstream the
   vendored copy can be deleted in favour of `using ScripMap` —
   the API is held byte-identical.

### Bonus (not on the original list)

- ~~**Aligned-resolution conservative remapping**~~ —
  `src/utils/grid_scale.jl` adds `GridScaleWeights` plus
  `map_field_to_lo` / `map_field_to_hi` for the special case where
  one grid is an integer refinement of another. Stencil helpers +
  `@threads` outer loop. Plus `refine_grid` / `coarsen_grid` for
  one-line construction of a companion grid from an existing one.

## Deferred

8. **Enthalpy solver validation** — benchmark the current port
   against reference data. Requires running the model and comparing
   against YelmoMirror — defer to a dedicated validation campaign.
10. **Predictor–corrector schemes FE-SBE / AB-SAM** — finish the
    timestepping extension (Lever 1 in the PC refactor design memo).
    Larger-than-one-commit refactor.

## Explicitly out of scope (not planned)

- L1L2 solver
- `calving_aa` (vertical-MB calving on aa-cells)
- Subgrid grounding-line discharge schemes
