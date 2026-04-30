# `topo_step!` — phase-level reference

`topo_step!(y::YelmoModel, dt)` advances the topography component
`y.tpo` by `dt` years. Phase order matches the Fortran reference
[`yelmo_topography.f90:calc_ytopo_pc`](../yelmo/src/yelmo_topography.f90)
predictor/corrector body (line 172 onward), one phase per
`apply_tendency!` call so each contribution to `mb_net` is realised
and recorded individually.

## Phase pipeline

| # | Phase | Helper(s) | Output | Notes |
|---|---|---|---|---|
| 1 | Snapshot `H_ice` | — | `H_prev`, `tpo.H_ice_n` | `H_ice_n` feeds the `topo_rel_field == "H_ice_n"` relaxation target. |
| 2 | Advection | `advect_tracer!` | `tpo.H_ice` | Skipped if `ytopo.topo_fixed`. Generic 2D tracer advection; reused by `lsf_update!`. |
| 3 | Mask post-step | `_apply_mask_ice_pass!` | `tpo.H_ice` | `bnd.mask_ice ∈ {NONE, FIXED, DYNAMIC}` per cell. |
| 4 | Snapshot for `dHidt_dyn` | — | `H_after_dyn` | Captures the dynamic contribution before any tendency. |
| 5 | `f_ice` refresh | `calc_f_ice!` | `tpo.f_ice` | Binary stub for v1; fractional later. |
| 6 | **SMB** | `mbal_tendency!`, `apply_tendency!` | `tpo.smb`, `tpo.H_ice` | Source: `bnd.smb_ref`. |
| 7 | `f_ice` refresh | `calc_f_ice!` | `tpo.f_ice` | |
| 8 | **BMB** | `calc_H_grnd!`, `determine_grounded_fractions!`, `calc_bmb_total!`, `mbal_tendency!`, `apply_tendency!` | `tpo.H_grnd`, `tpo.f_grnd_bmb`, `tpo.bmb_ref`, `tpo.bmb`, `tpo.H_ice` | Refreshes `H_grnd` and `f_grnd_bmb` from the *current* state. Combines `thrm.bmb_grnd` and `bnd.bmb_shlf` per `ytopo.bmb_gl_method`. Skipped if `ytopo.use_bmb == false`. |
| 9 | `f_ice` refresh | `calc_f_ice!` | `tpo.f_ice` | |
| 10 | **FMB** | `calc_fmb_total!`, `mbal_tendency!`, `apply_tendency!` | `tpo.fmb_ref`, `tpo.fmb`, `tpo.H_ice` | Submerged-front parameterisation. Same `use_bmb` gate as Fortran. |
| 11 | `f_ice` refresh | `calc_f_ice!` | `tpo.f_ice` | |
| 12 | **DMB** | `calc_mb_discharge!`, `mbal_tendency!`, `apply_tendency!` | `tpo.dmb_ref`, `tpo.dmb`, `tpo.H_ice` | v1 stub: only `dmb_method = 0` (no-op) is implemented. |
| 13 | `f_ice` refresh | `calc_f_ice!` | `tpo.f_ice` | |
| 14 | **Calving** | `calving_step!` | `tpo.cmb`, `tpo.lsf`, `tpo.cr_acx`, `tpo.cr_acy`, `tpo.dlsfdt`, `tpo.cmb_flt`, `tpo.cmb_grnd`, `tpo.H_ice` | Level-set flux method only (no aa/mb-form). Gated on `ycalv.use_lsf`. See [`calving.md`](./calving.md). |
| 15 | **Relaxation** (optional) | `set_tau_relax!`, `calc_G_relaxation!`, `apply_tendency!` | `tpo.tau_relax`, `tpo.mb_relax`, `tpo.H_ice` | Skipped when `ytopo.topo_rel == 0`. |
| 16 | `f_ice` refresh | `calc_f_ice!` | `tpo.f_ice` | |
| 17 | **Residual cleanup** | `resid_tendency!`, `apply_tendency!` | `tpo.mb_resid`, `tpo.H_ice` | Min-thickness margins, islands, neighbour cap. |
| 18 | `f_ice` refresh | `calc_f_ice!` | `tpo.f_ice` | |
| 19 | Net mass balance | — | `tpo.mb_net` | `smb + bmb + fmb + dmb + mb_relax + mb_resid`. |
| 20 | Diagnostics | `_update_diagnostics!` | `tpo.H_grnd`, `tpo.f_grnd`, `tpo.z_srf`, `tpo.z_base`, `tpo.dHidt`, `tpo.dHidt_dyn` | Final-state refresh. `f_grnd` is subgrid (CISM scheme). |
| 21 | `y.time += dt` | — | — | |

Mass-conservation invariant: `dHidt = dHidt_dyn + mb_net` to within
`apply_tendency!` clipping tolerance. Verified in the integration tests.

## Implementation status (Milestone 2)

**Done**

- Phase 1 (advection): `advect_tracer!` — generic 2D tracer advection
  (explicit upwind via Oceananigans operators), used both for `H_ice`
  here and for `lsf` in calving.
- Phases 2–6, 8–13, 15–18 (the per-cell mass-balance pipeline).
- Subgrid `f_grnd` via the full CISM bilinear-interpolation scheme,
  with a numerically stable `_calc_fraction_above_zero` kernel that
  improves on the Fortran reference. See
  [`grounded-fraction.md`](./grounded-fraction.md) for the math.
- Optional ice-thickness relaxation toward `bnd.H_ice_ref` or
  `tpo.H_ice_n`, supporting `topo_rel ∈ {-1, 1, 2, 3}`.

**Done (milestone 2c — calving)**

- Phase 14 — level-set flux calving via `calving_step!`. Three laws
  ported (`equil`, `threshold`, `vm-m16` stub). Sussman/Osher
  redistancing replaces the Fortran neighbour-snap reset and `dt_lsf`
  re-flag. Full pipeline documented in [`calving.md`](./calving.md).

**Deferred to later milestones**

- Phase 12 (DMB): the Calov+ 2015 kernel needs `dist_grline` and
  `dist_margin` distance-to-feature fields, which are not yet
  computed on the Julia side. The `calc_mb_discharge!` signature
  already mirrors Fortran for drop-in completion later.
- Phase 8 (BMB) `bmb_gl_method = "pmpt"` (subgrid tidal-zone
  parameterisation) — needs `calc_subgrid_array`. The other four
  methods (`fcmp`, `fmp`, `pmp`, `nmp`) are wired through.
- Phase 15 relaxation `topo_rel == 4` — needs `mask_grz` from the
  grounding-zone diagnostic.
- Predictor-corrector wrapping (`pred`/`corr` substructs).
- Implicit advection solver (`impl-lis`).

## Inputs and outputs

**Read** (from other components):

- `dyn`: `ux_bar`, `uy_bar` (advection)
- `bnd`: `smb_ref`, `bmb_shlf`, `fmb_shlf`, `z_bed`, `z_sl`,
  `z_bed_sd`, `H_ice_ref`, `tau_relax`, `ice_allowed`, `mask_ice`
- `thrm`: `bmb_grnd`

**Written** (`tpo` group): the entire ytopo state — `H_ice`, `H_grnd`,
`H_ice_n`, `f_ice`, `f_grnd`, `f_grnd_acx`, `f_grnd_acy`,
`f_grnd_bmb`, `tau_relax`, `z_srf`, `z_base`, `dHidt`, `dHidt_dyn`,
`mb_net`, `smb`, `bmb`, `fmb`, `dmb`, `cmb`, `mb_relax`, `mb_resid`,
`bmb_ref`, `fmb_ref`, `dmb_ref`, plus the calving sub-state (`lsf`,
`lsf_n`, `dlsfdt`, `cr_acx`, `cr_acy`, `cmb_flt`, `cmb_flt_acx`,
`cmb_flt_acy`, `cmb_grnd`, `cmb_grnd_acx`, `cmb_grnd_acy`).

## Tests

`test/test_yelmo_topo.jl` covers:

- Kernel-level: `advect_tracer!`, `calc_f_ice!`, `calc_H_grnd!`,
  `determine_grounded_fractions!`, `calc_bmb_total!`,
  `calc_fmb_total!`, `calc_mb_discharge!`, `set_tau_relax!`,
  `calc_G_relaxation!`.
- `mask_ice` post-step pass (12 cell-state combinations).
- Real-restart 5-step smoke (Greenland 16km).
- Slab conservation tests for SMB, BMB, and relaxation, each
  prescribing one tendency and verifying realised thinning + the
  mass-balance accounting `dHidt = dHidt_dyn + mb_net`.
- Analytical benchmarks for `determine_grounded_fractions!`:
  - Linear-GL (Tier 1): exact to 1e-12 across 8 parameterised
    `(a, b, c)` triples.
  - Circular-GL convergence (Tier 2): clean O(dx²) at all four
    refinement levels (rates 2.02 / 2.01 / 2.01).
- Calving (phase 14): nine testsets covering `lsf_init!`, the
  ocean-extrapolation sweeps, Sussman/Osher redistancing
  (`|∇φ| = 1` recovery and zero-set preservation), passive LSF
  transport, every law (`equil`, `threshold`, `vm-m16` error path),
  the merge logic, and an end-to-end kill on a synthetic shelf with
  `mass-balance closure to 1e-9`. See [`calving.md`](./calving.md)
  for the test inventory.
