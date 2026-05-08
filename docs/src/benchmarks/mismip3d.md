# MISMIP3D Standard

**Test files**:
- `test/benchmarks/test_mismip3d_stnd.jl` — 500-yr standalone trajectory
- `test/benchmarks/test_mismip3d_stnd_lockstep.jl` — YelmoMirror lockstep
- `test/benchmarks/test_mismip3d_stnd_att.jl` — ATT ramp grounding-line
  migration
- `test/benchmarks/test_mismip3d_stnd_implicit.jl` — implicit vs explicit
  advection comparison

**Fixtures**: `test/benchmarks/fixtures/mismip3d_stnd_t0.nc`,
`mismip3d_stnd_t500.nc`, `mismip3d_stnd_att_t1500.nc`  
**Solver tested**: SSA with Regularised Coulomb friction (+ adaptive PC, implicit
advection)  
**Validation**: YelmoMirror lockstep at t = 500 yr

## Setup

MISMIP3D Standard (Pattyn et al. 2013) is a marine-ice-sheet benchmark for
SSA solvers on a linearly sloped bed.

| Parameter | Value |
|---|---|
| Domain x | [0, 800] km, Bounded (`Nx` = 51, `dx` = 16 km) |
| Domain y | [−50, +50] km, Periodic (`Ny` = 7) |
| Bed | `z_bed(x) = −100 − x_km` (linear slope seaward) |
| SMB | 0.5 m/yr uniform |
| Rate factor `rf_const` | 3.1536 × 10⁻¹⁸ Pa⁻³ yr⁻¹ |
| Friction | Regularised Coulomb (`beta_method = 4`, `q = 1/3`, `u₀ = 1 m/yr`) |
| Calving | Kill-column at `i = Nx` (eastern boundary) |

## Initial condition

The literal Fortran IC (10 m uniform slab) leaves the SSA system
rank-deficient at `t = 0` (every cell floating, driving stress ≈ 0,
BiCGStab saturates at the velocity clamp).  Yelmo.jl therefore uses Fortran's
commented-out alternative thicker IC:

```math
H_\mathrm{ice} = \max\!\left(0,\; 1000 - 0.9\,z_\mathrm{bed}\right)
\quad \text{where } z_\mathrm{bed} \geq -500\text{ m}
```

This grounds all marine cells from `t = 0`, giving a well-posed SSA from
the first step.  The literal IC is preserved in
`MISMIP3DBenchmark.state(b, 0)` for completeness; re-enabling it would
require the adaptive-dt / `topo_fixed` infrastructure not yet present in
Yelmo.jl.

## What the test files exercise

### `test_mismip3d_stnd.jl` — 500-yr trajectory

- SSA Picard solver stability (`max|ux|` bounded, no NaN).
- y-symmetry preservation: `ux(i, j) = ux(i, Ny+1−j)` (geometry is
  y-invariant), `uy ≈ 0` on the centreline.  Tolerance 10% of max velocity.
- Trajectory growth: `max(H) > 50 m`, `mean(f_grnd) > 0.05` at `t = 500 yr`.
- NetCDF output round-trip.

Observed trajectory at 500 yr with the thicker IC:

| Quantity | Value |
|---|---|
| `max(H)` | ~1574 m |
| `mean(H)` | ~831 m |
| `mean(f_grnd)` | ~0.49 |
| `max\|ux_b\|` | ~670 m/yr |
| `max\|uy_bar\|` on centreline | < 1 × 10⁻¹² m/yr (y-symmetry) |

### `test_mismip3d_stnd_lockstep.jl` — YelmoMirror comparison

Runs the same 500-yr trajectory and compares cell-by-cell against the
committed YelmoMirror reference:

| Field | Tolerance |
|---|---|
| `H_ice` | relative < 50% |
| `f_grnd` | absolute < 50% |
| `ux_b` | relative < 50% |

The 50% tolerance reflects the simplified Yelmo.jl pipeline (constant
`ATT`, constant `cb_ref`, no per-step `mat_step!` or `thrm_step!`) versus
the Fortran backend's adaptive viscosity and full coupled pipeline.  It will
be tightened as the material and thermal modules are completed.

At the current milestone, the typical lockstep errors are:

| Field | Observed relative error |
|---|---|
| `H_ice` | ~2% |
| `f_grnd` | ~0.02 |
| `ux_b` | ~14% |

### `test_mismip3d_stnd_att.jl` — ATT ramp grounding-line migration

Exercises the three-phase MISMIP3D ice-dynamics protocol compressed into
three 500-yr windows:

- **Phase 0** (`t` = 0–500 yr): Standard equilibration at baseline
  `A₀ = 3.1536 × 10⁻¹⁸` Pa⁻³ yr⁻¹.
- **Phase 1** (`t` = 500–1000 yr): Rate factor reduced 10× to `A_low =
  3.1536 × 10⁻¹⁹` Pa⁻³ yr⁻¹ (stiffer ice → slower shelf flow →
  grounding line **advances**).
- **Phase 2** (`t` = 1000–1500 yr): Rate factor restored to `A₀`
  (softer ice → grounding line **retreats** toward baseline).

The test asserts:
- `mean(f_grnd)` increases during Phase 1 (advance) and decreases during
  Phase 2 (retreat).
- At `t = 1500 yr`, `mean(f_grnd)` is within 5 percentage points of the
  Phase 0 baseline (hysteresis < 5%).

### `test_mismip3d_stnd_implicit.jl` — implicit vs explicit advection

Runs two 100-yr trajectories side-by-side — one with `ytopo.solver = "expl"`,
one with `"impl"` — and checks that `H_ice` agrees to within a tight
tolerance.  This gates the implicit upwind advection scheme on a real
2D benchmark with a flowing ice shelf.

## How to run

```bash
julia --project=test test/benchmarks/test_mismip3d_stnd.jl
julia --project=test test/benchmarks/test_mismip3d_stnd_lockstep.jl
julia --project=test test/benchmarks/test_mismip3d_stnd_att.jl
julia --project=test test/benchmarks/test_mismip3d_stnd_implicit.jl
```

The standalone and ATT ramp tests build the model in-memory.
The lockstep test requires `fixtures/mismip3d_stnd_t500.nc`; the ATT
ramp lockstep requires `mismip3d_stnd_att_t1500.nc`.  Both are committed.

## Benchmark struct

```julia
include("test/benchmarks/helpers.jl")
using .YelmoBenchmarks

b = MISMIP3DBenchmark(:Stnd; dx_km = 16.0)
# Boundaries: Bounded in x (open ocean on east), Periodic in y.
y = YelmoModel(b, 0.0; p=p, boundaries=:periodic_y)
```
