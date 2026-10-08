# Adaptive predictor-corrector timestepping

**Test file**: `test/benchmarks/test_adaptive_dt.jl`  
**Solver tested**: Adaptive timestepping (HEUN, FE-SBE, AB-SAM)  
**Validation**: Redo reference round-trip + MISMIP3D Stnd 500-yr comparison

## Overview

`YelmoModel` steps with the predictor-corrector time loop of Fortran
`yelmo_update` ([Stepping](../usage/stepping.md)). The scheme is
`yelmo.pc_method`; each mixes the advective rates of the predictor and
corrector topography stages with its β coefficients:

| Scheme | β1, β2 (predictor) | β3, β4 (corrector) | Order | `tau` |
|---|---|---|---|---|
| `"FE-SBE"` | 1, 0 | 1, 0 | 1 | `(H_corr − H_pred)/(2·dt)` |
| `"AB-SAM"` | 1 + ζ/2, −ζ/2 | ½, ½ | 2 | `ζ·(H_corr − H_pred)/((3ζ + 3)·dt)` |
| `"HEUN"` | 1, 0 | ½, ½ | 2 | `(H_corr − H_pred)/(6·dt)` |

(ζ = dt/dt_prev; AB-SAM runs as FE-SBE on a cold start.) The step is
redone with a smaller `dt` if the norm `η` of the truncation error `tau`
exceeds `pc_tol`; with `dt_method = 2` the controller (`pc_controller`)
chooses the next `dt` from the last values of `η`, aiming at `pc_eps`.

`η` [1/yr] is the RMS of the scaled errors `|tau|/(1 m + 0.01·H_corr)` over
the points of the pc mask (ice at least `pc_eta_H_min` thick and
`pc_eta_u_min` fast in both states, no partly covered cell in the 3×3
neighbourhood, grounded in both states, not an isolated outlier), without
the `pc_eta_trim` fraction of largest ones (Fortran `set_pc_mask`,
`calc_pc_eta`).

## What it tests

Three test sets:

### 1. Redo reference round-trip

Takes a few fixed-dt steps (so velocities are non-trivial), stores the
topography and dynamics in a `RedoRef`, advances further, then calls
`restore!`. Asserts that the fields come back exactly — the time loop relies
on this to redo rejected steps.

### 2. 500-yr MISMIP3D Stnd trajectory (all three schemes)

Runs MISMIP3D Standard to `t = 500 yr` with adaptive PC (outer step `dt = 1 yr`)
and compares against a fixed-forward-Euler reference run:

| Metric | Tolerance |
|---|---|
| `max(H)` relative difference | < 10% |
| `mean(H)` relative difference | < 10% |
| `mean(f_grnd)` absolute difference | < 5 percentage points |

The adaptive and fixed-FE runs need not produce bit-identical output — they
converge to the same attractor but via different trajectories.  The ±10%
tolerance confirms they land in the same neighbourhood.

Observed (albedo, yelmo dev time loop):

| Quantity | Fixed dt (`dt_method = 0`) | FE-SBE | HEUN | AB-SAM |
|---|---|---|---|---|
| `max(H)` | 1576.27 m | 1576.29 m | 1576.29 m | 1576.27 m |
| `mean(H)` | 840.63 m | 841.26 m | 841.25 m | 841.20 m |
| `mean(f_grnd)` | 0.4902 | 0.4902 | 0.4902 | 0.4902 |

### 3. Rollback path actually fires on the cliff IC

The MISMIP3D thicker IC produces a velocity cliff on the first step
(unconstrained SSA gives ~5000 m/yr at the calving column, then
`ssa_vel_max` clips it).  The first outer step should trigger at least one
adaptive rejection or sub-step.  The test asserts
`n_rejections > 0` OR `n_steps_taken > 1` OR `min(pc_dt) < 1 yr`.

## Step-size controller details

As in Fortran `set_adaptive_timestep_pc`, the controller gives the ratio
`ρ = dt_{n+1}/dt_n` from the last three `η` (and `dt`). The default PI42
(Söderlind & Wang, 2006):

```math
\rho = \left(\frac{\varepsilon}{\eta_n}\right)^{k_i + k_p}
       \left(\frac{\varepsilon}{\eta_{n-1}}\right)^{-k_p},
\qquad k_i = \frac{2}{5\,k},\ k_p = \frac{1}{5\,k}
```

with `ε = pc_eps` and `k` the order of the scheme. `"H312b"`, `"H312PID"`,
`"H321PID"` and `"PID1"` are also available. Then `ρ ≤ pc_rho_max`, the
step is capped at Courant number `pc_cfl_max` of the transport velocity,
fitted to `[dt_min, time left]` (two equal steps rather than a big and a
tiny one at the end of the call) and rounded down to 4 decimals.

| Parameter | Description | Default |
|---|---|---|
| `pc_tol` | Redo threshold on `η` | 1.0 /yr |
| `pc_eps` | Target `η` | 0.02 /yr |
| `pc_n_redo` | Max attempts per step | 5 |
| `pc_rho_max` | Max growth of `dt` per step | 2.0 |
| `pc_cfl_max` | Courant-number cap | 0.5 |
| `dt_min` | Minimum step | 0.1 yr |
| `cfl_max` | Courant number of `dt_method = 1` | 0.1 |

## How to run

```bash
julia --project=test test/benchmarks/test_adaptive_dt.jl
```

No fixtures required.  The test builds the MISMIP3D model in-memory and runs
both the fixed-dt reference and the adaptive runs within the same script.

## Configuring adaptive timestepping

```julia
using Yelmo
using Yelmo.YelmoPar: yelmo_params, YelmoParameters

p = YelmoParameters("my_run";
    yelmo = yelmo_params(
        dt_method     = 2,          # adaptive PC (the default)
        pc_method     = "AB-SAM",   # or "HEUN" / "FE-SBE"
        pc_controller = "PI42",
        pc_tol        = 1.0,
        pc_eps        = 0.02,
    ),
    # ... other parameter groups
)
```

`dt_method = 0` takes one step per `step!` call, `dt_method = 1` Courant
steps (`cfl_max`).
