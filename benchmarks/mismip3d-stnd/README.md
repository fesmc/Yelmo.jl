# `mismip3d-stnd`

MISMIP3D Stnd benchmark (Pattyn et al. 2013). Steady-state buildup of a
2D marine ice sheet on a y-invariant downward-sloping bed
(`z_bed = -100 - x_km`, m) with a fixed calving front at the eastern
edge, under SSA + Coulomb-friction grounding-line dynamics.

## Reference

- Pattyn, F., et al. (2013). *Grounding-line migration in plan-view
  marine ice-sheet models*, J. Glaciol. **59**(215), 410–422.
- Yelmo Fortran reference: `yelmo/par/yelmo_MISMIP3D.nml`,
  `yelmo/tests/mismip3D.f90`, `yelmo/tests/yelmo_mismip.f90`.

## Geometry

- Domain: x ∈ [0, 800] km (Bounded), y ∈ [-50, +50] km (Periodic-y).
- Grid: dx = 16 km → Nx = 51, Ny = 7 (odd Ny → centerline j = 4).
- Bed: y-invariant, z_bed = −100 − x_km (slope −1/1000).

## Initial condition

This benchmark uses Fortran's *commented thicker-grounded* IC variant
(`mismip3D.f90:62-64`, currently disabled in production) rather than
the literal 10 m all-floating slab:

```
H_ice = max(0, 1000 - 0.9·z_bed)   for z_bed < 0
H_ice = 0                           where z_bed < z_bed_floor (= −500 m)
```

The literal Fortran IC is rank-deficient under SSA — every cell
floating, no driving stress, BiCGStab returns spurious clamped
velocities — and Yelmo.jl's forward-Euler pipeline cannot recover
without adaptive timestepping. The thicker IC grounds all marine
cells from t=0 and produces a well-posed SSA system. Fortran's own
test machinery uses the same workaround on production runs.

## Forcing (constant in time)

- `smb_ref = 0.5 m/yr`
- `T_srf = 273.15 K`
- `Q_geo = 42 mW/m²`
- Calving boundary at the eastern column (`ice_allowed[Nx, :] = 0`).

## Model configuration

- Solver: SSA, Picard with `picard_iter_max = 20`, `picard_tol = 1e-3`,
  Krylov `rtol = 1e-6`, `itmax = 500`.
- `beta_method = 4` (regularized Coulomb, q-exponent), `beta_q = 1/3`,
  `beta_u0 = 1`, `beta_gl_stag = 3`, `ssa_lat_bc = "floating"`.
- Material: `n_glen = 3`, constant `A_glen = 3.1536e-18 Pa⁻³ yr⁻¹`,
  `cb_ref = 3.165176e4`, `N_eff = 1 Pa`.
- Topo: explicit upwind advection.
- Therm: disabled (`method = "fixed"`).
- Time stepping: forward-Euler, `dt = 1 yr`.

## How to run

```bash
cd benchmarks/mismip3d-stnd
julia --project=. run.jl
julia --project=. summary.jl
```

## Outputs

- `output/timeseries.nc` — per-sample diagnostics: `time, max_H,
  mean_H, mean_f_grnd, gl_x_km, max_ux_abs, max_uy_center_abs,
  picard_iters_last`.
- `output/restart_final.nc` — final 2D state (H_ice, z_bed, …).
- `summary.json` — committed reference values for cross-version
  comparison.

## Notes

This is a longer reproduction of the regression test
[test/benchmarks/test_mismip3d_stnd.jl](../../test/benchmarks/test_mismip3d_stnd.jl).
`run.jl` calls `Yelmo.init_state!(y, 0.0; thrm_method = "robin")` after
constructing the model — mirrors Fortran's `yelmo_init_state` call in
`yelmo/tests/yelmo_mismip.f90`. Without that, default-zero `T_ice_b`
collides with `calc_c_bed!`'s `scale_T = 1` thermal-scaling branch
and collapses basal friction; the SSA solver then saturates at the
5000 m/yr velocity clamp. See PR #66 for the bisect history.

Next benchmarks: **mismip3d-stnd-att-ramp** (same setup with phased
changes to the Glen rate factor — see
[examples/mismip3d_att_ramp/run_ramp.jl](../../examples/mismip3d_att_ramp/run_ramp.jl)
for a pre-benchmarks prototype) and CalvingMIP-Exp1.
