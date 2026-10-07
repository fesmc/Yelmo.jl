# Topography stages — phase-level reference

The topography is advanced in the stages of the predictor-corrector time
loop (`src/timestepping.jl`), a port of Fortran `calc_ytopo_pc`
(`yelmo_topography.f90`):

| Stage | Call | Result |
|---|---|---|
| Predictor | `topo_step!(y, dt, PCPredictor(); β1, β2)` | `H_pred` (live state and `tpo.pc.pred`), for the velocity solve |
| Corrector | `topo_step!(y, dt, PCCorrector(); β3, β4)` | `H_corr` (`tpo.pc.corr`); live state back to `H_n` |
| Advance | `topo_step!(y, dt, PCAdvance(); use_H_pred)` | `H_{n+1}` = predictor or corrector record |

The predictor and corrector transport `H_n` with the transport velocity
(the depth-averaged velocity, filtered with `yelmo.pc_filter_vel`, faces
into ice-free cells closed) and mixed advective rates

- predictor: `dHidt_dyn = β1·f(H_n, u_n) + β2·f_{n-1}`
- corrector: `dHidt_dyn = β3·f(H_pred, u*) + β4·f(H_n, u_n)`

(`f(H_n, u_n)` is `tpo.pc.dHidt_dyn_raw`, `f_{n-1}` is `dHidt_dyn_raw_n`;
the β come from `yelmo.pc_method`), then run the mass-balance cascade below,
one `apply_tendency!` per contribution so each is realised and recorded.

## Phase pipeline (predictor and corrector)

| # | Phase | Helper(s) | Output | Notes |
|---|---|---|---|---|
| 1 | Store `H_ice_n`, `z_srf_n`, `lsf_n` | — | `tpo.H_ice_n`, … | Predictor only. `H_ice_n` is also the `topo_rel_field == "H_ice_n"` relaxation target. |
| 2 | Advective rate | `advection_tendency!` | `tpo.pc.dHidt_dyn_raw` / `tpo.dHidt_dyn` | Zero with `ytopo.solver = "none"`. |
| 3 | Transport | `apply_tendency!(…; mb_clip)` | `tpo.H_ice`, `tpo.dHidt_dyn`, `tpo.mb_clip` | Mixed rate applied to `H_n`; the clip of negative thickness is booked in `mb_clip`. |
| 4 | `f_ice` refresh | `calc_f_ice!` | `tpo.f_ice` | Binary (`front_subgrid = "none"`). |
| 5 | **SMB** | `mbal_tendency!`, `apply_tendency!` | `tpo.smb`, `tpo.H_ice` | Source: `bnd.smb_ref`. |
| 6 | **BMB** | `calc_H_grnd!`, `determine_grounded_fractions!`, `calc_bmb_total!`, `mbal_tendency!`, `apply_tendency!` | `tpo.H_grnd`, `tpo.f_grnd_bmb`, `tpo.bmb_ref`, `tpo.bmb`, `tpo.H_ice` | Combines `thrm.bmb_grnd` and `bnd.bmb_shlf` per `ytopo.bmb_gl_method`. Skipped if `ytopo.use_bmb == false`. |
| 7 | **FMB** | `calc_fmb_total!`, `mbal_tendency!`, `apply_tendency!` | `tpo.fmb_ref`, `tpo.fmb`, `tpo.H_ice` | Same `use_bmb` gate as Fortran. |
| 8 | **DMB** | `calc_mb_discharge!`, `mbal_tendency!`, `apply_tendency!` | `tpo.dmb_ref`, `tpo.dmb`, `tpo.H_ice` | Only `dmb_method = 0` (no-op) is implemented. |
| 9 | **Calving** | `calving_step!` | `tpo.cmb`, `tpo.lsf`, `tpo.cr_acx`, `tpo.cr_acy`, `tpo.cmb_flt`, `tpo.cmb_grnd`, `tpo.H_ice` | Level-set flux method only. Gated on `ycalv.use_lsf`. See [the calving page](calving.md). |
| 10 | **Relaxation** (optional) | `set_tau_relax!`, `calc_G_relaxation!`, `apply_tendency!` | `tpo.tau_relax`, `tpo.mb_relax`, `tpo.H_ice` | Skipped when `ytopo.topo_rel == 0`. |
| 11 | **Residual cleanup** | `resid_tendency!`, `apply_tendency!` | `tpo.mb_resid`, `tpo.H_ice` | `bnd.mask_ice` (no ice / fixed thickness), minimum-thickness margins, islands. |
| 12 | Net mass balance | — | `tpo.mb_net` | `smb + bmb + fmb + dmb + mb_relax + mb_resid` (calving `cmb` is separate). |
| 13 | Rates | `_stage_rates!` | `tpo.dHidt`, `tpo.dlsfdt`, `tpo.mb_err` | Relative to `H_ice_n`, `lsf_n`. |
| 14 | Diagnostics | `update_diagnostics!` | `tpo.H_grnd`, `tpo.f_grnd*`, `tpo.f_grnd_pin`, `tpo.z_srf`, `tpo.z_base`, `tpo.dist_grline`, `tpo.dist_margin`, `tpo.mask_grz`, `tpo.mask_bed`, `tpo.mask_frnt`, `tpo.dzsdx/dy`, `tpo.dHidx/y`, `tpo.dzbdx/dy`, `tpo.H_ice_dyn`, `tpo.f_ice_dyn` | `f_ice` is refreshed after every phase that changes `H_ice`. |

Mass-conservation invariant: `dHidt = dHidt_dyn + mb_clip + mb_net + cmb`
(`mb_err ≈ 0`). Verified in the integration tests.

## Implementation status (Milestone 2)

**Done**

- Advection: `advect_tracer!` — generic 2D tracer advection
  (explicit upwind via Oceananigans operators, or implicit), used both
  for `H_ice` here and for `lsf` in calving.
- The predictor-corrector stages and the per-cell mass-balance pipeline.
- Subgrid `f_grnd` via the full CISM bilinear-interpolation scheme,
  with a numerically stable `_calc_fraction_above_zero` kernel that
  improves on the Fortran reference. See
  [the grounded-fraction page](grounded-fraction.md) for the math.
- Optional ice-thickness relaxation toward `bnd.H_ice_ref` or
  `tpo.H_ice_n`, supporting `topo_rel ∈ {-1, 1, 2, 3}`.

**Done (milestone 2c — calving)**

- Level-set flux calving via `calving_step!`. Three laws
  ported (`equil`, `threshold`, `vm-m16` stub). Sussman/Osher
  redistancing (Fortran `lsf_method = "redist"`). Full pipeline documented in [the calving page](calving.md).

**Deferred to later milestones**

- DMB: the Calov+ 2015 kernel needs `dist_grline` and
  `dist_margin` distance-to-feature fields, which are not yet
  computed on the Julia side. The `calc_mb_discharge!` signature
  already mirrors Fortran for drop-in completion later.
- BMB `bmb_gl_method = "pmpt"` (subgrid tidal-zone
  parameterisation) — needs `calc_subgrid_array`. The other four
  methods (`fcmp`, `fmp`, `pmp`, `nmp`) are wired through.
- Relaxation `topo_rel == 4` — needs `mask_grz` from the
  grounding-zone diagnostic.

## Inputs and outputs

**Read** (from other components):

- `c` (`YelmoConstants`): `rho_ice`, `rho_sw` — used by `calc_H_grnd!`,
  `calc_fmb_total!`, the surface/base elevation update, and
  `calving_step!`. Defaults to Yelmo Fortran `&phys`; can be
  overridden by passing `c=YelmoConstants(...)` to the constructor.
- `dyn`: `ux_bar`, `uy_bar` (advection)
- `bnd`: `smb_ref`, `bmb_shlf`, `fmb_shlf`, `z_bed`, `z_sl`,
  `z_bed_sd`, `H_ice_ref`, `tau_relax`, `mask_ice`
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
  `mass-balance closure to 1e-9`. See [the calving page](calving.md)
  for the test inventory.
