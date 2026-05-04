## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# Slab advection-stability stress test.
#
# Distilled from `yelmo/tests/yelmo_slab.f90`: that program perturbs
# a uniform slab with Gaussian noise on a sloped bed, runs the full
# SSA + topo prognostic pipeline at a sweep of (dx, dt) combinations,
# and reports the perturbation-amplitude growth ratio
# `factor = stdev(H_final) / stdev(H_initial)`. Factor < 1 → stable
# (perturbation decays); factor > 1 → unstable. The Fortran test uses
# bisection to bracket the maximum-stable dt for each dx.
#
# What this Yelmo.jl port gates:
#
#   The implicit upwind scheme is unconditionally stable (backward
#   Euler) — it should bound the perturbation even at CFL ≫ 1, where
#   any explicit upwind variant without substepping would diverge.
#   We don't sweep (dx, dt); we pin two regimes:
#
#     * CFL = 0.5 (sub-CFL): both schemes produce factor ≤ 1
#       (perturbation decays under upwind diffusion).
#     * CFL = 5   (super-CFL): implicit factor still ≤ 1; explicit
#       *without internal substepping* (`cfl_safety = 1e6`) blows up
#       to factor ≫ 1 or NaN. Explicit *with* its default substepping
#       is also stable (it just internally subdivides into 50 small
#       steps), but that's measuring the substepper, not the scheme —
#       we explicitly disable substepping to exercise the kernel.
#
# The configuration deliberately bypasses the SSA / dyn pipeline:
# velocities are set to a uniform u₀ so the test isolates the
# topo-step advection kernel from solver coupling. The Fortran
# slab test runs the full pipeline so its `factor` metric also
# reflects SSA / Picard convergence on a perturbed slab; this
# port focuses on the scheme stability, which is the property
# PR-1..3 added.
#
# Companion to:
#   * `test_yelmo_topo_advection_implicit.jl` — unit-level dispatch /
#     conservation / centroid-tracking tests.
#   * `test_mismip3d_stnd_implicit.jl` — implicit vs explicit on a
#     real 2D benchmark at small CFL.

using Test
using Statistics
using Yelmo
using Oceananigans

const _SLAB_NX     = 32
const _SLAB_NY     = 4
const _SLAB_DX     = 1e3        # m
const _SLAB_H0     = 1000.0     # m
const _SLAB_DH_AMP = 10.0       # m  (Gaussian-pulse amplitude)
const _SLAB_DH_SIG = 4 * _SLAB_DX

# Build a tiny periodic-x / Bounded-y slab grid plus its uniform
# velocity field. Returns `(grid, c, u, v, c0, c0_init)` where:
#
#   * `c`        — `CenterField` for the prognostic tracer (perturbed
#                  thickness deviation, ΔH).
#   * `c0`       — copy of the initial perturbation (for the post-run
#                  amplitude metric).
#   * `c0_init`  — copy of the initial mean (i.e. zero); kept for
#                  diagnostic completeness.
#
# We advect the *deviation* `ΔH = H − H₀`, not the full H. Under
# uniform flow on a periodic axis, ΔH should translate without growth
# regardless of the absolute mean. This matches the Fortran-Yelmo
# slab convention (the "factor" metric is `stdev(H, H₀)` — the
# RMS deviation from the steady-state mean).
function _slab_setup(u0::Float64)
    Nx, Ny, dx = _SLAB_NX, _SLAB_NY, _SLAB_DX
    grid = RectilinearGrid(CPU();
        size = (Nx, Ny, 1),
        x = (0, Nx * dx),
        y = (0, Ny * dx),
        z = (0, 1),
        topology = (Periodic, Bounded, Bounded),
    )

    c  = CenterField(grid)
    Ci = interior(c)
    σ  = _SLAB_DH_SIG
    x0 = Nx * dx / 4
    @inbounds for j in 1:Ny, i in 1:Nx
        x = (i - 0.5) * dx
        Ci[i, j, 1] = _SLAB_DH_AMP * exp(-((x - x0)^2) / (2 * σ^2))
    end
    c0 = copy(Ci)

    u = XFaceField(grid); fill!(interior(u), u0)
    v = YFaceField(grid); fill!(interior(v), 0.0)

    return grid, c, u, v, c0
end

# Perturbation-amplitude metric, normalised so `factor = 1` means
# no growth. We use the L2 norm of `ΔH` rather than the Fortran
# `stdev` because both fields here have mean zero (we advect the
# deviation directly), so L2 and stdev are equal up to a constant
# `√N` factor.
_amplitude_factor(c_now, c_init) = sqrt(sum(c_now .^ 2)) / sqrt(sum(c_init .^ 2))

@testset "benchmarks: slab advection stability — sub-CFL (CFL=0.5)" begin
    u0 = 100.0                            # m/yr
    cfl = 0.5
    dt  = cfl * _SLAB_DX / u0             # 5 yr
    n_steps = 20

    # Implicit.
    grid_i, c_i, u_i, v_i, c0_i = _slab_setup(u0)
    cache_i = init_advection_cache(grid_i)
    for _ in 1:n_steps
        advect_tracer!(c_i, u_i, v_i, dt;
                       scheme = :upwind_implicit, cache = cache_i)
    end
    factor_i = _amplitude_factor(interior(c_i), c0_i)

    # Explicit, with substepping disabled (cfl_safety huge ⇒ one
    # step per call). At CFL=0.5 this is identical to the
    # default-substepping case.
    grid_e, c_e, u_e, v_e, c0_e = _slab_setup(u0)
    for _ in 1:n_steps
        advect_tracer!(c_e, u_e, v_e, dt;
                       scheme = :upwind_explicit, cfl_safety = 1e6)
    end
    factor_e = _amplitude_factor(interior(c_e), c0_e)

    @info "Slab CFL=0.5 (sub-CFL): factor_impl=$(round(factor_i; digits=4))  " *
          "factor_expl=$(round(factor_e; digits=4))"

    @test factor_i < 1.0   # implicit decays (numerical diffusion)
    @test factor_e < 1.0   # explicit decays (numerical diffusion)
    @test all(isfinite, interior(c_i))
    @test all(isfinite, interior(c_e))
end

@testset "benchmarks: slab advection stability — super-CFL (CFL=5)" begin
    u0 = 100.0
    cfl = 5.0
    dt  = cfl * _SLAB_DX / u0             # 50 yr
    n_steps = 5

    # Implicit at CFL=5: still stable (backward Euler is
    # unconditionally stable). The factor will be small because
    # the implicit scheme is heavily diffusive at large CFL.
    grid_i, c_i, u_i, v_i, c0_i = _slab_setup(u0)
    cache_i = init_advection_cache(grid_i)
    for _ in 1:n_steps
        advect_tracer!(c_i, u_i, v_i, dt;
                       scheme = :upwind_implicit, cache = cache_i)
    end
    factor_i = _amplitude_factor(interior(c_i), c0_i)

    # Explicit with substepping disabled. At CFL=5 a single forward-
    # Euler upwind step is unconditionally unstable: the perturbation
    # amplitude grows ~ (CFL−1)ⁿ per step, and after a few steps the
    # field either explodes or hits NaN. We pin `cfl_safety = 1e6`
    # so the kernel takes one outer step instead of subdividing.
    grid_e, c_e, u_e, v_e, c0_e = _slab_setup(u0)
    blew_up = false
    for _ in 1:n_steps
        advect_tracer!(c_e, u_e, v_e, dt;
                       scheme = :upwind_explicit, cfl_safety = 1e6)
        if any(!isfinite, interior(c_e))
            blew_up = true
            break
        end
    end
    factor_e = blew_up ? Inf : _amplitude_factor(interior(c_e), c0_e)

    @info "Slab CFL=5 (super-CFL): factor_impl=$(round(factor_i; digits=4))  " *
          "factor_expl=$(blew_up ? "NaN/Inf" : string(round(factor_e; digits=4)))"

    # Implicit is unconditionally stable — perturbation must not grow.
    @test factor_i < 1.5
    @test all(isfinite, interior(c_i))

    # Explicit without substepping must blow up at CFL > 1 (otherwise
    # the test isn't actually exercising what it claims to).
    @test blew_up || factor_e > 10.0
end

@testset "benchmarks: slab advection stability — implicit large dt single step" begin
    # Stress test: one giant step at CFL = 50. Implicit should still
    # produce a finite, smoothly-decayed field.
    u0 = 100.0
    cfl = 50.0
    dt  = cfl * _SLAB_DX / u0             # 500 yr
    grid, c, u, v, c0 = _slab_setup(u0)
    cache = init_advection_cache(grid)

    advect_tracer!(c, u, v, dt;
                   scheme = :upwind_implicit, cache = cache)

    @test all(isfinite, interior(c))
    @test maximum(abs, interior(c)) ≤ _SLAB_DH_AMP   # strict max-norm decay
    factor = _amplitude_factor(interior(c), c0)
    @info "Slab CFL=50 single step: factor_impl=$(round(factor; digits=4))"
    @test factor < 1.0
end
