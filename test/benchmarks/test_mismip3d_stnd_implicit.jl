## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# MISMIP3D Stnd implicit-vs-explicit advection regression.
#
# Companion to `test_mismip3d_stnd.jl` (the 500-yr default-default
# trajectory test). This file is the targeted regression for the
# implicit upwind scheme on a real benchmark — it runs two YelmoModels
# side-by-side, identical except for `ytopo.solver` (`"impl"` vs
# `"expl"`), and checks that the two trajectories agree closely on
# H_ice over a 100-yr run.
#
# Why a comparison test rather than a strict-conservation test:
#
#   - Strict mass conservation of the implicit advection operator is
#     already validated in `test_yelmo_topo_advection_implicit.jl`
#     ("solve_advection! — uniform translation conserves total"), on
#     a periodic configuration where the only mass sinks are the
#     prescribed source terms.
#   - On MISMIP3D Stnd (Bounded-x with a calving column at i=Nx),
#     the matrix has Dirichlet identity rows on the lateral edges by
#     construction. Mass that "would" enter the boundary cell from
#     the upstream interior is dropped — exactly what the calving
#     boundary should do, matching Fortran impl-lis behaviour. So
#     ∑H drift relative to ∑MB·dt·dA at this boundary is expected
#     and quantitatively reflects the boundary outflux, not a
#     non-conservation bug.
#   - What a benchmark test like this *can* meaningfully gate is:
#     "does the implicit scheme produce essentially the same answer
#     as the explicit scheme at small CFL, on a non-trivial
#     2D problem with realistic dynamics?" — the consistency
#     question.
#
# Setup: same SSA / topo / till config as `test_mismip3d_stnd.jl`,
# 100-step trajectory at dt = 1 yr (CFL ≈ 0.02, well within explicit
# stability so explicit and implicit upwind should agree closely).
# We pin `solver` explicitly in both configs so the test is
# resilient to future default changes.

using Test
using Statistics
using Yelmo
using Oceananigans: interior

include("helpers.jl")
using .YelmoBenchmarks

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           yneff_params, ytill_params, ytopo_params

const _SPEC = MISMIP3DBenchmark(:Stnd; dx_km=16.0)

function _params_with_solver(solver_str::String, alias::String)
    return YelmoModelParameters(alias;
        ydyn = ydyn_params(
            solver         = "ssa",
            visc_method    = 1,
            beta_method    = 4,
            beta_q         = 1.0/3.0,
            beta_u0        = 1.0,
            beta_gl_scale  = 0,
            beta_gl_stag   = 3,
            beta_min       = 0.0,
            ssa_lat_bc     = "floating",
            eps_0          = 1e-6,
            taud_lim       = 1e6,
            ssa_solver     = SSASolver(precond         = :jacobi,
                                       picard_tol      = 1e-3,
                                       picard_iter_max = 20,
                                       picard_relax    = 0.7,
                                       rtol            = 1e-6,
                                       itmax           = 500),
        ),
        yneff = yneff_params(method = 0, const_ = 1.0),
        ytill = ytill_params(method = -1),
        ymat  = ymat_params(
            n_glen     = 3.0,
            rf_const   = 3.1536e-18,
            visc_min   = 1e3,
            de_max     = 0.5,
            enh_shear  = 1.0,
            enh_stream = 1.0,
            enh_shlf   = 1.0,
        ),
        ytopo = ytopo_params(solver = solver_str),
    )
end

function _build_mismip3d(spec, p)
    y = YelmoModel(spec, 0.0; p=p, boundaries = :periodic_y)
    Nx, Ny = length(spec.xc), length(spec.yc)
    fill!(interior(y.mat.ATT),    p.ymat.rf_const)
    fill!(interior(y.dyn.cb_ref), spec.cf_ref)

    # Same thicker grounded IC override as the default-default test.
    H_int     = interior(y.tpo.H_ice)
    z_bed_int = interior(y.bnd.z_bed)
    @inbounds for j in 1:Ny, i in 1:Nx
        zb = z_bed_int[i, j, 1]
        H_int[i, j, 1] = (zb < spec.z_bed_floor) ? 0.0 : max(0.0, 1000.0 - 0.9 * zb)
    end

    Yelmo.update_diagnostics!(y)
    return y
end

@testset "benchmarks: MISMIP3D Stnd implicit vs explicit advection" begin
    b = _SPEC

    p_impl = _params_with_solver("impl", "mismip3d_stnd_impl")
    p_expl = _params_with_solver("expl", "mismip3d_stnd_expl")

    @test p_impl.ytopo.solver == "impl"
    @test p_expl.ytopo.solver == "expl"

    y_impl = _build_mismip3d(b, p_impl)
    y_expl = _build_mismip3d(b, p_expl)

    # Ensure the two start identical to the bit (same IC, no run yet).
    @test interior(y_impl.tpo.H_ice) == interior(y_expl.tpo.H_ice)

    n_steps = 100
    dt      = 1.0
    nan_seen = false

    for k in 1:n_steps
        Yelmo.step!(y_impl, dt)
        Yelmo.step!(y_expl, dt)
        H_i = interior(y_impl.tpo.H_ice)
        H_e = interior(y_expl.tpo.H_ice)
        if any(isnan, H_i) || any(isnan, H_e)
            nan_seen = true
            @info "NaN at step $k (impl=$(any(isnan, H_i)), expl=$(any(isnan, H_e)))."
            break
        end
    end

    @test !nan_seen

    H_impl = interior(y_impl.tpo.H_ice)
    H_expl = interior(y_expl.tpo.H_ice)

    # Element-wise agreement metric. CFL is ≈ 0.02 here, so explicit
    # and implicit upwind should produce nearly identical H (the
    # numerical diffusion difference scales with CFL — small at low
    # CFL, large at CFL ≥ 1).
    abs_diff = abs.(H_impl .- H_expl)
    max_abs  = maximum(abs_diff)
    max_H    = maximum(H_expl)
    rel_err  = max_abs / max(max_H, eps())
    mean_err = mean(abs_diff) / max(max_H, eps())

    @info "MISMIP3D Stnd impl-vs-expl after $n_steps yr: " *
          "max|ΔH|=$(round(max_abs; digits=3)) m  " *
          "max|ΔH|/max(H)=$(round(rel_err; sigdigits=4))  " *
          "mean|ΔH|/max(H)=$(round(mean_err; sigdigits=4))  " *
          "max(H_impl)=$(round(maximum(H_impl); digits=2))  " *
          "max(H_expl)=$(round(maximum(H_expl); digits=2))"

    # At CFL ≈ 0.02 over 100 steps the schemes should agree to within
    # a few percent on the relative max-H scale. Keep the bound
    # generous enough to absorb the implicit scheme's slightly higher
    # numerical diffusion without being so loose that a real
    # divergence would slip through.
    @test rel_err  < 0.05
    @test mean_err < 0.01

    # Both runs must produce a sensible final state.
    @test maximum(H_impl) > 100.0
    @test maximum(H_expl) > 100.0
    @test mean(interior(y_impl.tpo.f_grnd)) > 0.05
    @test mean(interior(y_expl.tpo.f_grnd)) > 0.05
end
