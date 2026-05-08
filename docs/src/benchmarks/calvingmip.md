# CalvingMIP Experiment 1

**Test file**: `test/benchmarks/test_calvingmip_exp1.jl`  
**Fixture**: `test/benchmarks/fixtures/calvingmip_exp1_t1000.nc`  
**Solver tested**: SSA + level-set calving (via hook)  
**Validation**: YelmoMirror reference fixture at t = 1000 yr (ice-covered
cell count within ±5%)

## Setup

CalvingMIP (Calving Model Intercomparison Project) Experiment 1 is a
circular-domain calving benchmark.  A parabolic bowl bed drives ice
accumulation; a velocity-equilibrium calving law caps the ice extent at
a prescribed radius.

| Parameter | Value |
|---|---|
| Domain | 64 × 64, `dx` = 25 km, x/y ∈ [−800, 800] km |
| Topology | Bounded |
| Bed | `z_bed(r) = 900 − 2900 (r/R₀)²`  (`R₀` = 800 km) |
| `z_bed` at `r = 0` | +900 m |
| `z_bed` at `r = R₀` | −2000 m |
| SMB | 0.3 m/yr uniform |
| `T_srf` | 223.15 K |
| `Q_geo` | 42 mW m⁻² |
| IC | `H_ice = 0`, `lsf = +1` (all ocean) |
| Calving radius | 750 km |

## Calving law (Exp 1)

The calving rate `cr` is set to `−u` everywhere (velocity-equilibrium:
calving flux equals ice flux into the calving front), then zeroed out for
cells more than one `dx` inside the 750 km radius:

```julia
y.hooks.calv_flt = (cx, cy, ux, uy, Hi, fi, lsf, t) ->
    calvmip_exp1!(cx, cy, ux, uy, Hi, fi, lsf, t; xc=xc, yc=yc)
```

This hook pattern is how experiment-specific calving laws are attached to
`YelmoModel` without modifying the model core.  At each `topo_step!` call
the model invokes `y.hooks.calv_flt` to populate `tpo.cr_acx` / `tpo.cr_acy`
before the level-set advection phase.

## What it tests

- `calvmip_bed_circular` geometry formula (unit test against known values
  at `r = 0` and `r = R₀`).
- Analytical IC at `t = 0`: `H_ice = 0`, `lsf = +1`, bed parabola correct.
- **Regression** at `t = 1000 yr`: ice-covered cell count agrees with the
  committed YelmoMirror fixture within ±5%.

The ±5% tolerance is intentionally loose; exact cell-count agreement is not
expected because Yelmo.jl and YelmoMirror use slightly different
advection/calving pipelines.  The test confirms the calving-hook mechanism
and the level-set front-tracking are working qualitatively correctly.

## How to run

```bash
julia --project=test test/benchmarks/test_calvingmip_exp1.jl
```

Requires `fixtures/calvingmip_exp1_t1000.nc`.  To regenerate:

```bash
julia --project=test test/benchmarks/regenerate.jl calvingmip_exp1 --overwrite
```

## Benchmark struct

```julia
include("test/benchmarks/helpers.jl")
using .YelmoBenchmarks

b = CalvingMIPBenchmark(:exp1; dx_km = 25.0)
y = YelmoModel(b, 0.0; p=p, boundaries=:bounded)
# Attach the experiment-specific calving hook:
xc, yc = b.xc, b.yc
y.hooks.calv_flt = (cx, cy, ux, uy, Hi, fi, lsf, t) ->
    calvmip_exp1!(cx, cy, ux, uy, Hi, fi, lsf, t; xc=xc, yc=yc)
init_state!(y, 0.0; thrm_method="robin")
```

## Experiment variants

The `benchmarks/` directory under the repo root contains namelist
configurations for CalvingMIP Experiments 1–4 in
`benchmarks/calvingmip-exp{1,2,3,4}/`.  These are the Fortran-side
namelists used by `YelmoMirror` to generate reference fixtures for each
experiment.  Only Exp 1 has a committed fixture and a test file in CI;
Exps 2–4 are available for local validation runs.
