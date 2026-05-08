# Adaptive predictor-corrector timestepping

**Test file**: `test/benchmarks/test_adaptive_dt.jl`  
**Solver tested**: Adaptive timestepping (HEUN, FE-SBE, AB-SAM)  
**Validation**: Snapshot/restore round-trip + MISMIP3D Stnd 500-yr comparison

## Overview

Yelmo.jl supports three adaptive predictor-corrector (PC) schemes,
selectable via `yelmo.pc_method`:

| Scheme | Description |
|---|---|
| `"HEUN"` | Heun's method (explicit trapezoidal rule) — predictor step + corrector average |
| `"FE-SBE"` | Forward Euler predictor + Semi-Backward Euler corrector |
| `"AB-SAM"` | Adams-Bashforth predictor + Semi-Adams-Moulton corrector |

All three share the same PI42 step-size controller (`pc_controller = "PI42"`)
that adjusts `dt` based on the velocity change between predictor and corrector
steps:

```math
\eta = \frac{\|\mathbf{u}_\mathrm{pred} - \mathbf{u}_\mathrm{corr}\|_\infty}{u_\mathrm{ref}}
```

The step is rejected and retried with a smaller `dt` if `η > pc_tol`;
otherwise `dt` is grown for the next step.

## What it tests

Three test sets:

### 1. Snapshot / restore round-trip

Takes a few fixed-dt steps (so velocities are non-trivial), snapshots the full
model state, advances further, then calls `restore!`.  Asserts that every
snapshotted field is recovered to < 10⁻¹² absolute error.  This is the deepest
sanity check on the rollback machinery — the adaptive PC relies on `restore!`
to roll back rejected steps.

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

Observed agreement (typical):

| Quantity | Fixed FE | HEUN | FE-SBE | AB-SAM |
|---|---|---|---|---|
| `max(H)` | ~1575 m | ~1540 m | ~1530 m | ~1545 m |
| `mean(f_grnd)` | ~0.490 | ~0.488 | ~0.486 | ~0.489 |

### 3. Rollback path actually fires on the cliff IC

The MISMIP3D thicker IC produces a velocity cliff on the first step
(unconstrained SSA gives ~5000 m/yr at the calving column, then
`ssa_vel_max` clips it).  The first outer step should trigger at least one
adaptive rejection or sub-step.  The test asserts
`n_rejections > 0` OR `n_steps_taken > 1` OR `min(dt_history) < 1 yr`.

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
using Yelmo.YelmoModelPar: yelmo_params, YelmoModelParameters

p = YelmoModelParameters("my_run";
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
