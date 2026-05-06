# Plan — vertical convention refactor (T_ice / ux / etc. → interior + boundary split)

## TL;DR

Yelmo.jl's grid loader builds an Oceananigans `Center`-staggered z
axis from the file's `zeta_ac` (Face values, length 11), and then
loads file `T_ice` / `ux` / etc. (length 10) into Yelmo's
`Nz=10` Center fields by **verbatim index copy**. Because Oceananigans
`Center` cells are forced to be midpoints of consecutive Face values,
the file's center positions (which include `z=0` and `z=1` boundary
endpoints) and Yelmo's center positions (interior midpoints) **don't
align**. The data ends up at slightly-wrong vertical positions
throughout the column, with the worst case at the surface where the
file's `z=1` value gets stored at Yelmo's `z=0.948`.

This mis-aligned convention has been the de-facto state for all 3D ice
fields (dyn `ux`/`uy`/`uz_star`, mat `visc`/`ATT`/`enh`, thrm
`T_ice`/`enth`/`T_pmp`/...). The thrm port (PR1–PR7) adopts the same
buggy convention because it uses `T_ice[:,:,1]` and `T_ice[:,:,Nz]` as
basal/surface BC values in the implicit column solver — which works
because the file values at `z=0` and `z=1` end up close to those
positions, but is technically wrong.

This document specifies the work to fix it correctly: introduce true
interior + 2D-boundary separation across all 3D ice and bedrock
fields, with NetCDF I/O that combines on write and splits on read so
the file format stays Mirror-compatible.

## Empirical evidence

Verified on a 16 km Greenland restart at `RESTART_PATH`:

```
File zeta (centers, length 10): [0.0,    0.0123, 0.0494, 0.1111, 0.1975, 0.3086, 0.4444, 0.6049, 0.7901, 1.0]
Yelmo zeta_aa (Center, length 10): [0.0031, 0.0185, 0.0556, 0.1173, 0.2037, 0.3148, 0.4506, 0.6111, 0.7963, 0.9475]
Δ at the 10 corresponding indices:
  k=1:  +0.003   (file BASE → near-base interior cell)
  k=2..9: +0.006 (small interior offset)
  k=10: -0.053   (file SURFACE → 5.3% below surface)
```

The file's `zeta_ac` (Face, length 11) matches Yelmo's Face exactly.
What's wrong: the file's `zeta` (centers, length 10) is *ignored* in
the grid construction; Yelmo derives its own center positions as
midpoints of `zeta_ac`. Then the file's `T_ice[:,:,k]` (which the
file claims is at file zeta[k]) gets stored at Yelmo's
`zeta_aa[k]`, a different physical position.

Verbatim-copy proof:

```
file ux[1..10]   = [0.09, 0.01, -0.23, ..., -1.29]
Yelmo ux[1..10]  = [0.05, -0.12, -0.56, ..., -2.51]   # different values for a *different* model state
file ux_b        = 0.09     Yelmo ux_b = 0.05
file ux_s        = -1.29    Yelmo ux_s = -2.51
Yelmo ux[1] vs ux_b: 0.0498 vs 0.0498  ← exact equality
Yelmo ux[10] vs ux_s: -2.5137 vs -2.5137  ← exact equality
```

So `ux_b` is currently a **redundant 2D copy** of `ux[:,:,1]`, not the
properly-stored basal boundary value. Same for `ux_s`. The thrm port
has the same redundancy without even storing `T_ice_b` / `T_ice_s` at
all.

## What "Path A" cannot fix

Just adding `T_ice_b` / `T_ice_s` 2D fields as shortcuts to
`T_ice[:,:,1]` / `T_ice[:,:,Nz]` (matching the current `ux_b` /
`ux_s` pattern) keeps the underlying mis-alignment. Any consumer that
treats `T_ice[:,:,1]` as an *interior cell value* gets a small but
real systematic error in physical position (~0.3% of column depth).
The implicit thrm column solver is one such consumer that happens to
treat it as a *boundary* value — but mat's rate factor (`rf_method=1`)
and any future analysis that integrates over the column will see the
mis-aligned data.

`Oceananigans.Center` stagger fundamentally cannot include the boundary
endpoints — it's defined as midpoints of consecutive Faces. The
"Center field with endpoints" interpretation isn't implementable with
stock Oceananigans.

## Path B — the fix

**Internal convention:**
- Yelmo's grid `Nz` = file `zeta` length − 2. So Greenland 16 km
  becomes Nz = 8 (was Nz = 10).
- All 3D ice fields (`T_ice`, `enth`, `T_pmp`, `cp`, `kt`, `omega`,
  `Q_strn`, `dQsdT`, `advecxy`, `ux`, `uy`, `uz_star`, `jvel_*`,
  `mat.visc`, `mat.ATT`, `mat.enh`, ...) drop to Nz=8 INTERIOR
  cells.
- 2D `_b` / `_s` boundary fields hold the actual basal and surface
  values: `T_ice_b`, `T_ice_s`, `enth_b`, `enth_s`, `T_pmp_b`,
  `T_pmp_s` for thrm. `ux_b`, `ux_s`, `uy_b`, `uy_s` already exist
  in dyn — change them from "redundant copy" to "true storage".
- `omega_b` / `omega_s` and `cp_b` / `cp_s` and `kt_b` / `kt_s`
  derived in place from `T_b` / `T_s` (no separate storage needed).
- Bedrock: `T_rock` interior + `T_rock_b` 2D for the deep boundary.
  `T_rock_s` ≡ `T_ice_b` (the bed-ice interface is one shared
  surface) — store once on the ice side, bedrock solver reads from
  there.

**File / I/O convention:**
- File on disk stays unchanged: `T_ice` is a length-10 zeta field
  including endpoints. Mirror compatibility preserved.
- A boundary-fields registry lists which file fields combine an
  interior + 2D _b / 2D _s on Yelmo's side:
  ```
  "T_ice"  => (:T_ice_b,  :T_ice,  :T_ice_s)
  "enth"   => (:enth_b,   :enth,   :enth_s)
  "T_pmp"  => (:T_pmp_b,  :T_pmp,  :T_pmp_s)
  "ux"     => (:ux_b,     :ux,     :ux_s)
  "uy"     => (:uy_b,     :uy,     :uy_s)
  "uz_star"=> (:uz_b_star, :uz_star, :uz_s_star)   # if needed
  ...
  ```
- Read: detect a registered field in the restart → load full slab →
  slice [:,:,1] → `_b`, [:,:,Nz_file] → `_s`, [:,:,2..Nz_file-1] →
  interior. Length checks match.
- Write: detect a registered field on output → glue `_b` + interior +
  `_s` along the zeta axis → write under the unified field name.
- YelmoMirror sync: same registry drives the C-API marshalling.

**Grid construction:**
- Loader takes file `zeta` (length Nz_file). Yelmo Nz = Nz_file − 2.
- Yelmo `Center` zeta_aa = file zeta[2..Nz_file-1] **exactly** (no
  midpoint recomputation — these are the file's interior centers and
  we keep them).
- Yelmo `Face` zeta_ac built so that midpoints of consecutive Faces
  recover zeta_aa, with endpoints at 0 and 1. For non-uniformly
  spaced file zeta this means the Faces aren't the file's `zeta_ac`
  — accept this departure (file `zeta_ac` is no longer authoritative
  for the Yelmo Center grid).

  **Open design question**: alternative is to add a custom grid that
  explicitly stores both the centers and the half-cell faces from
  the file. Discuss.

## Affected components (file-level checklist)

### `src/` — core grid + I/O

- `src/YelmoCore.jl::load_grids_from_restart` — change Nz derivation,
  build zeta_aa from file's interior, build zeta_ac to be consistent.
- `src/YelmoCore.jl::yelmo_define_grids` (if it constructs from a
  benchmark spec rather than restart) — same.
- `src/YelmoCore.jl::load_field_from_dataset_3D` (or wherever the
  field load happens) — apply boundary registry: interior + _b + _s.
- `src/YelmoCore.jl::make_field` etc. — handle the new 2D _b / _s
  field allocation.
- `src/YelmoIO.jl::write_output!` — apply boundary registry on write
  to recombine into unified slab.
- `src/YelmoIO.jl::init_output` — register the unified-axis variable
  names (not the _b / _s components individually, unless the user
  asks for them).

### `src/variables/model/`

- `yelmo-variables-ytherm.md`: add `T_ice_b`, `T_ice_s`, `enth_b`,
  `enth_s`, `T_pmp_b`, `T_pmp_s`, `T_rock_b` (no `T_rock_s` —
  shared with `T_ice_b`).
- `yelmo-variables-ydyn.md`: clarify that `ux_b` etc. are now true
  boundary storage, not slice-shortcuts. (May already say so —
  check.)
- `yelmo-variables-ymat.md`: `visc_b`, `visc_s`, `ATT_b`, `ATT_s`,
  `enh_b`, `enh_s`? Decide based on whether mat's basal/surface
  reads matter (`visc_int` integrates over the column → needs
  half-cell values).
- `yelmo-variables-ytopo.md`: probably no impact (topo is mostly 2D).

### `src/dyn/` — velocity solvers

- `velocity_sia.jl`, `velocity_ssa.jl`, `velocity_diva.jl`,
  `velocity_uz.jl`, `basal_dragging.jl`, `viscosity.jl`,
  `deformation.jl`, `lateral_stress.jl`, `driving_stress.jl`,
  `diagnostics.jl` — anywhere these read `ux[:,:,1]` /
  `ux[:,:,end]` / etc., switch to `ux_b` / `ux_s`. Anywhere they
  write to `ux[:,:,1]` / `[:,:,end]` (probably zero), drop the writes
  (those slots no longer exist in interior storage).
- `dyn_step!` body: trace each phase and check if any reads slice 1 or
  Nz of a 3D field. The `_calc_uz_3D_jac!` and `calc_velocity_sia!`
  vertical loops in particular need careful eyes.

### `src/mat/`

- `viscosity.jl`: `visc_bar` / `visc_int` depth-averages. The current
  trapezoidal rule needs to include the new _b / _s endpoints with
  their actual zeta positions (0 and 1). May already be doing this
  via `vert_int_trapz_boundary!` (`integration.jl`) — verify.
- `enhancement.jl`: same for `enh_bar`.
- `rate_factor.jl`: `rf_method=1` (deferred) reads basal T → should
  use `T_ice_b` directly under the new convention.

### `src/thrm/`

- `YelmoModelThrm.jl`: `therm_step!` body. After Path B:
  - Properties update writes `T_pmp` interior + `T_pmp_b` +
    `T_pmp_s`; `cp` interior only (no _b / _s) because cp is derived
    in place per layer; same for `kt`.
  - Method dispatch passes `T_b`, `T_s` scalars to the column solver.
  - `_extrapolate_thrm_margin!` operates on _b / _s fields too.
  - `_calc_T_prime_3D!` writes `T_prime` interior + `T_prime_b`
    (already exists) + `T_prime_s` (new — or derive in place).
- `column_solver.jl` (`_calc_temp_column_internal!`): change BC
  formulation. Current code sets `solution[1] = val_base - T_ref`
  (Dirichlet at zeta_aa[1]); new code should treat val_base as a
  TRUE basal value at z=0 and discretise the BC across the gap from
  z=0 to zeta_aa[1] (which is non-trivial but smaller than current
  half-cell). For Neumann the formulation already uses dz to
  zeta_aa[2] which is fine.
- `temp_solver.jl`, `enth_solver.jl`: signature change to take T_b,
  T_s + interior; write back to interior + _b + _s.
- `bedrock.jl`: same split. `define_temp_bedrock_3D!` writes
  interior + T_rock_b. `T_rock_s` reads from `y.thrm.T_ice_b`
  (shared surface).
- `solvers_analytic.jl`: `define_temp_linear_3D!` and
  `define_temp_robin_3D!` write interior + _b + _s.
- `advection.jl`: `calc_advec_horizontal_3D!` reads var (T or enth)
  with interior + _b + _s — needs to advect the boundary slices too
  if they vary horizontally (probably yes). Or pass _b / _s as
  separate 2D fields and don't advect them (questionable physically).
- `helpers.jl`: `_calc_cts_height_column` already takes the
  interior column — fine. `_extrapolate_thrm_margin!` needs to
  extrapolate _b and _s too.

### `src/topo/`

- Anywhere it reads `dyn` 3D fields' top or bottom slice — point at
  `ux_s` / `uy_s` (for surface velocity) or `ux_b` / `uy_b`. The
  basal calving / discharge kernels are likely consumers.

### `src/YelmoMirrorCoreFields.jl`

- C-API sync: `yelmo_set_var3D!("thrm_T_ice", ...)` currently passes
  the Yelmo `T_ice` (length 10) to Fortran which expects length 10.
  After Path B Yelmo has length 8 + _b + _s; the marshalling needs
  to glue them back to a length-10 buffer for the C API call. Same
  for the read side `yelmo_get_var3D!`.
- This is where the boundary registry pays off — drive both reads and
  writes from the same lookup table.

### `test/`

- `test_yelmo_thrm.jl` (newly added): the column-solver tests still
  apply but the BC formulation may shift. Update as needed.
- `test_yelmo_topo.jl`, `test_yelmo_model.jl`,
  `test_yelmo_mat_*.jl`, `test_yelmo_dyn*.jl`: any place that
  asserts on a specific `[:,:,1]` or `[:,:,end]` slice of a 3D
  field — review whether the assertion still applies (probably needs
  to point at `_b` / `_s` instead).
- `test/benchmarks/test_*.jl`: same. The Mirror-lockstep tests will
  need re-running and tolerances may need re-tightening (the current
  tolerances absorb the mis-alignment).

### `test/benchmarks/fixtures/`

- All committed `.nc` fixtures were generated against the buggy
  convention. Re-generate via `regenerate.jl <benchmark> --overwrite`
  for each (the file format itself doesn't change, but the values
  change because Mirror runs Fortran which has the correct
  convention; before Path B, Mirror values were being silently
  re-interpreted by Yelmo as being at the wrong z positions).

## Recommended commit sequence

Best to do in a fresh worktree off main (NOT on top of the thrm
port branch). Suggested:

1. **commit 1: variables + grid construction**
   - Add the 6 new ytherm boundary fields to the schema.
   - Update `load_grids_from_restart` to build Nz = file Nz − 2 grid
     with file's interior centers as Yelmo zeta_aa.
   - Update `make_field` / allocator for the new 2D _b / _s fields.
   - Yelmo loads cleanly but solvers don't yet know about the new
     fields. Existing tests will mostly fail — that's expected.

2. **commit 2: I/O registry (read + write)**
   - Add the boundary registry.
   - Update field loader to apply registry on read (split file's
     length-10 into _b + interior + _s).
   - Update output writer to recombine on write.
   - YelmoMirror sync uses same registry for C-API marshalling.
   - At this point, restart load and output write are fully
     symmetric and Mirror-compatible.

3. **commit 3: dyn refactor**
   - Update all dyn kernels to consume `ux_b` / `ux_s` etc. as true
     boundary fields rather than slice 1 / Nz of `ux`.
   - Re-run dyn unit tests; expect tolerance shifts.

4. **commit 4: mat refactor**
   - Update `visc_bar` / `enh_bar` integrals to include _b / _s.
   - Update `rate_factor.jl` (when `rf_method=1` lands).

5. **commit 5: thrm refactor**
   - Update `therm_step!`, column solver, advection, properties.
   - Re-run `test_yelmo_thrm.jl` (1D unit tests); update as needed.

6. **commit 6: topo + final polish**
   - Update topo reads of dyn fields.
   - Re-run all tests; update tolerances and fixtures as needed.

7. **commit 7: regenerate benchmark fixtures**
   - Run `regenerate.jl <benchmark> --overwrite` for each.
   - Verify Mirror-lockstep tests pass under new convention with
     reasonable tolerances.

## Validation strategy

- After commit 1: `using Yelmo` precompiles; `YelmoModel(restart)`
  constructs without error; `step!(y, 0.0)` doesn't crash.
- After commit 2: round-trip — load restart, write to NetCDF, load
  the written file, compare to the original (should match to F32
  precision for Float32 vars).
- After commit 3: dyn tests pass; `dyn_step!` allocates no more than
  before.
- After commit 5: `test_yelmo_thrm.jl` passes; Greenland-restart smoke
  test produces sensible T_ice for all six methods (same as PR1–7).
- After commit 7: Mirror lockstep tests pass with documented
  tolerances. Specifically:
  - `test_eismint_moving_lockstep.jl` H_ice + uxy comparisons.
  - `test_mismip3d_stnd_lockstep.jl` H_ice + ux_b + f_grnd
    comparisons.
- Independent: a "vertical convention round-trip" unit test that
  builds a synthetic file with known `T_ice[:,:,1] = 273` (basal),
  loads via Yelmo, asserts `T_ice_b == 273` (basal value goes to the
  right place), writes back, asserts the round-trip restores
  `T_ice[:,:,1] = 273`.

## Open design questions

1. **Grid Face convention**: should Yelmo's `zeta_ac` (Face) be the
   file's `zeta_ac` directly (preserving file authority for both
   centers and faces)? That would mean the new Yelmo Nz_face = file
   Nz_face = 11 and Yelmo Nz_center = file Nz - 2 = 8. The midpoint
   relationship between Centers and Faces gets broken (Yelmo Face has
   11 entries but only 9 of them are between consecutive Centers —
   the other 2 are above the surface Center and below the basal
   Center). Oceananigans may not accept this layout.
   - Alternative: accept that Yelmo `zeta_ac` is recomputed as
     midpoints between Centers + endpoints, and may differ from file's
     `zeta_ac`. File's `zeta_ac` is then only used at I/O for the
     boundary slices, not for the grid.

2. **Bedrock surface sharing**: enforce `T_rock_s ≡ T_ice_b` how?
   - (a) Don't allocate `T_rock_s`; bedrock solver reads `T_ice_b`
     for its top BC. Cleanest.
   - (b) Allocate both, sync after thrm step.
   - (c) Make `T_rock_s` an alias / view of `T_ice_b`.
   - Recommend (a). Enth_rock_s ≡ enth_b similarly.

3. **Advection of boundary fields**: should
   `calc_advec_horizontal_3D!` advect `T_ice_b` and `T_ice_s` too?
   - Yes physically: surface T at one cell may differ from neighbour
     cells, so horizontal advection of `T_ice_s` is real. But for
     `T_srf` (Dirichlet from boundary forcing) we override post-solve
     anyway, so advection of `_s` may be moot.
   - For `T_ice_b`: advection matters when the basal value changes
     across cells (always the case for grounded ice). Recommend
     yes.

4. **Solver BC discretisation**: the gap from z=0 (boundary) to
   zeta_aa[1] (first interior center) is now a half-cell of
   different size than between consecutive interior centers. The
   implicit solver's first-row dz needs to use that half-cell
   thickness. Mirror Fortran does this implicitly because Fortran's
   T_ice[1] IS the boundary; the first-cell dz is `zeta_aa[2] -
   zeta_aa[1]` (interior-to-boundary distance). Under Path B,
   recompute as `zeta_aa[1] - 0` for the basal half-cell.

5. **Performance**: the current per-step `collect(zeta)` allocations
   profiled at ~340 KB/step. Path B's grid change keeps `collect`
   per-step unless we also add the scratch struct
   (`y.thrm.scratch.zeta_aa::Vector{Float64}` populated once at
   init). Recommend adding the scratch struct as part of commit 5.

6. **Backward-compat shim?**: do we want Yelmo to be able to read a
   restart file that was generated by Yelmo.jl pre-Path-B (which is
   itself the same format as Mirror — so this is moot if file format
   is unchanged)? Probably no shim needed since file format doesn't
   change.

## Effort estimate

- Pure code changes: ~5-8 days for someone familiar with the codebase.
- Re-validation against Mirror lockstep tests + regen of fixtures:
  ~2-3 days.
- Total: ~1.5-2 weeks of focused work.

## How to start

```bash
git -C /Users/alrobi001/models/Yelmo.jl worktree add \
    .claude/worktrees/vert-split main
cd .claude/worktrees/vert-split
# Read this plan + the empirical-evidence section of the doc-only
# commit on the blissful-wilson-d37e1c branch (commit 9234... — see
# git log for the exact sha).
# Start with commit 1 (variables + grid).
```
