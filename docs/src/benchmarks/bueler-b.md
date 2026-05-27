# BUELER-B — Halfar dome

**Test file**: `test/benchmarks/test_sia.jl`  
**Fixture**: `test/benchmarks/fixtures/bueler_b_t1000.nc`  
**Solver tested**: SIA  
**Validation**: Analytical Halfar (1983) / Bueler et al. (2005) closed-form solution

## Setup

The BUELER-B benchmark is a radially symmetric isothermal ice dome on a flat
bed with a prescribed accumulation pattern chosen so that the exact solution is
a scaled Halfar similarity profile.  The physical parameters follow Bueler et al. (2005)
Test B:

| Parameter | Value |
|---|---|
| Glen exponent `n` | 3 |
| Rate factor `A` | 1 × 10⁻¹⁶ Pa⁻³ yr⁻¹ (isothermal) |
| Ice density `ρ_ice` | 910 kg m⁻³ |
| Dome radius `R₀` | 750 km |
| Dome height `H₀` | 3600 m |
| Evaluation time `t` | 1000 yr |

The exact ice thickness at time `t` is the Halfar similarity solution:

```math
H(r, t) = H_0 \left(\frac{t_0}{t}\right)^{1/9}
\left[1 - \left(\frac{r}{R_0}\left(\frac{t_0}{t}\right)^{-1/18}\right)^{4/3}\right]_+^{3/7}
```

where `t_0` is the Halfar reference time computed from the other parameters.
The depth-averaged SIA velocity on the dome interior is:

```math
\bar{u}_x(r) = -\frac{2A}{n+2}(\rho g)^n H^{n+1}
\left(\frac{\partial H}{\partial r}\right)^{n-1}
\frac{\partial H}{\partial x}
```

and likewise for `ū_y`.

## What it tests

Two independent test sets:

1. **Closed-form derivative correctness** — checks that the private
   `_halfar_dHdr_closed` formula matches a centred finite-difference of
   `bueler_test_BC!`-evaluated `H` on a 1D radial slice (relative error
   < 0.1%).  This gates sign / exponent / factor errors in the analytical
   formula before using it as the velocity reference.

2. **SIA solver convergence** — drives `dyn_step!` (SIA only) on the Halfar
   dome at three resolutions (`dx` = 50, 20, 10 km) and evaluates the
   relative L2 error on the dome interior (`H > 100 m` mask to avoid the
   singular margin).  The test asserts:
   - Monotone error decrease: `err(20 km) < err(50 km)` and
     `err(10 km) < err(20 km)`.
   - Absolute threshold at the finest grid: relative L2 error < 10%.

## Expected results

Empirical errors at the production solver settings stabilise at:

| `dx` | Relative L2 error `ux` | Relative L2 error `uy` |
|---|---|---|
| 50 km | ~0.25 | ~0.25 |
| 20 km | ~0.12 | ~0.12 |
| 10 km | ~0.074 | ~0.065 |

The residual ~7% error at 10 km is dominated by the finite-difference
truncation error in the SIA stencil at the coarse resolution of these
benchmarks; it decreases toward zero as `dx → 0`.

## How to run

```bash
julia --project=test test/benchmarks/test_sia.jl
```

The benchmark constructs the `YelmoModel` in-memory from the analytical
state at `t = 1000 yr` (no NetCDF required).  The committed fixture
`bueler_b_t1000.nc` is used only by the smoke test (`test_smoke.jl`) and
can be regenerated with:

```bash
julia --project=test test/benchmarks/regenerate.jl bueler_b --overwrite
```

## Benchmark struct

```julia
using Yelmo
include("test/benchmarks/helpers.jl")
using .YelmoBenchmarks

b = BuelerBenchmark(:B; dx_km = 10.0, R0_km = 750.0, H0 = 3600.0)

# In-memory YelmoModel directly from the analytical state — no NetCDF.
y = YelmoModel(b, 1000.0; p = YelmoParameters("bueler_b"))

# Analytical velocity at the same time for comparison.
ux_ref, uy_ref = analytical_velocity(b, 1000.0)
```
