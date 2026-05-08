# ISMIP-HOM-C — SSA rotational symmetry

**Test file**: `test/benchmarks/test_hom_c.jl`  
**Fixture**: `test/benchmarks/fixtures/ismiphom_c_l80_t0.nc`  
**Solver tested**: SSA  
**Validation**: 180° rotational symmetry of the SSA solution

## Setup

ISMIP-HOM Experiment C (Pattyn et al. 2008) is a diagnostic SSA benchmark
on a periodic domain with spatially varying basal friction.  The geometry is a
1000 m uniform slab on a uniformly-sloped bed; the friction pattern is
sinusoidal:

```math
\beta(x, y) = \beta_0 + \beta_\mathrm{amp}\sin\!\left(\frac{2\pi x}{L}\right)
              \sin\!\left(\frac{2\pi y}{L}\right)
```

| Parameter | Value |
|---|---|
| Domain | `L` = 80 km, `dx` = 2 km → 40 × 40 cells |
| Topology | Fully periodic in x and y |
| Bed slope | `α` = 0.1° |
| Slab thickness `H` | 1000 m |
| Rate factor `A` | 1 × 10⁻¹⁶ Pa⁻³ yr⁻¹ (isothermal) |
| `β₀` = `β_amp` | 1000 Pa yr m⁻¹ |

The expected depth-averaged velocity magnitude at the centre of the slab is
approximately `τ_d / β₀ ≈ 15.6 m/yr`.

## What it tests

ISMIP-HOM-C has no closed-form solution.  Instead the test exploits the
**180° rotational symmetry** of the problem: under `(x, y) → (L − x, L − y)`,
the uniform driving stress `τ_d^x = ρgH tan α` is invariant and the
`β`-pattern is invariant (both `sin` factors flip sign, their product does
not).  Therefore:

```math
u_x(x, y) = u_x(L - x,\, L - y), \qquad
u_y(x, y) = u_y(L - x,\, L - y).
```

The test:

1. Checks that `update_diagnostics!` produces a uniform `dzsdx = −tan α`
   at every face (including across the periodic wrap), confirming that the
   `dzsdx_periodic_offset` mechanism works correctly.
2. Checks that `mean(|ux_bar|) ∈ [10, 50]` m/yr (sanity bound on driving
   stress / friction balance).
3. Asserts the rotational symmetry residual (maximum element-wise
   difference divided by maximum velocity magnitude) is below **1 × 10⁻⁷**.

The asymmetry residual at production solver settings (`rtol = 1e-8`,
`picard_tol = 1e-6`) is ~2 × 10⁻⁸ — this is the iterative-solver noise
floor, not a structural error.  The 1 × 10⁻⁷ threshold sits 5× above that
noise floor while detecting any recurrence of the periodic-wrap clamp bug
it was written to catch (that bug produced residuals of ~5 × 10⁻²).

## Periodic-slope offset

The benchmark geometry is a uniformly-sloped surface `z_s = −x tan α`.
On a periodic domain, the gradient kernel `calc_gradient_acx!` reads the
halo from the opposing face across the wrap.  Without correction, the
wrap-face gradient would see a discontinuous jump from `z_s(Nx·dx) ≈ 0` back
to `z_s(0) = 0`, producing zero driving stress on that column instead of the
uniform `−tan α`.

The fix is the `ytopo.dzsdx_periodic_offset` parameter: the gradient
kernel adds `offset = −tan α · L_x` to the halo read at the wrap face, so
every face (interior and wrap) recovers the correct uniform gradient.

## How to run

```bash
julia --project=test test/benchmarks/test_hom_c.jl
```

The model is built in-memory from the analytical state (no NetCDF
round-trip).

## Benchmark struct

```julia
include("test/benchmarks/helpers.jl")
using .YelmoBenchmarks

b = HOMCBenchmark(:C; L_km = 80.0, dx_km = 2.0)
y = YelmoModel(b, 0.0; p=p, boundaries=:periodic)
_setup_hom_c_beta!(y, b)    # fills y.dyn.c_bed / cb_ref from the sinusoidal β
```
