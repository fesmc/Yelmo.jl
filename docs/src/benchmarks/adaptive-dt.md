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
exceeds `pc_tol`; with `dt_method = 2` the PI42 controller
(`pc_controller = "PI42"`) chooses the next `dt` from the last values of
`η`, aiming at `pc_eps`.

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

The PI42 controller (same as Fortran Yelmo's `dt_method = 2`) adjusts
the next `dt` as:

```math
dt_{n+1} = dt_n \cdot
\left(\frac{\varepsilon_0}{\eta_n}\right)^{k_1}
\left(\frac{\varepsilon_0}{\eta_{n-1}}\right)^{k_2}
```

with `k₁ = 0.4/q`, `k₂ = 0.2/q` (order `q = 2` for PI42).  The key
parameters:

| Parameter | Description | Typical value |
|---|---|---|
| `pc_tol` | Rejection threshold on `η` | 5.0 m/yr |
| `pc_eps` | Controller floor `ε₀` | 1.0 m/yr |
| `pc_n_redo` | Max retries per outer step | 5 |
| `dt_min` | Minimum allowed sub-step | 0.01 yr |
| `cfl_max` | Maximum CFL for the next step | 0.1 |

## How to run

```bash
julia --project=test test/benchmarks/test_adaptive_dt.jl
```

No fixtures required.  The test builds the MISMIP3D model in-memory and runs
both the fixed-FE reference and the adaptive-PC runs within the same script.

## Configuring adaptive timestepping

```julia
using Yelmo
using Yelmo.YelmoPar: yelmo_params, YelmoParameters

p = YelmoParameters("my_run";
    yelmo = yelmo_params(
        dt_method     = 2,          # adaptive PC
        pc_method     = "FE-SBE",   # or "HEUN" / "AB-SAM"
        pc_controller = "PI42",
        pc_tol        = 5.0,
        pc_eps        = 1.0,
        pc_n_redo     = 5,
        dt_min        = 0.01,
        cfl_max       = 0.1,
    ),
    # ... other parameter groups
)
```

Set `dt_method = 0` (the default) to use fixed forward Euler.
