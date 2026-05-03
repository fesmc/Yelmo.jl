## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# Trough-F17 DIVA lockstep test.
#
# Sibling to `test_trough.jl` (which exercises SSA on the same
# fixture). The Fortran TROUGH-F17 namelist sets `solver = "diva"`
# (yelmo_TROUGH.nml:128), so the committed `trough_f17_t1000.nc`
# fixture is a Fortran-DIVA-generated reference. This test runs
# Yelmo.jl's port (`solver = "diva"`) against it for a single
# `dyn_step!`, mirroring the SSA test's structure.
#
# Loose tolerance characterising the first observation; tighten in a
# follow-up. The previous SSA-on-DIVA-fixture comparison saw
# `rel_ux ≈ 0.07`. The DIVA-on-DIVA comparison should be tighter
# (since the physics now matches the fixture).

using Test
using Statistics
using Yelmo
using Oceananigans: interior

include("helpers.jl")
using .YelmoBenchmarks

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           yneff_params, ytill_params, ytopo_params

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))
const _T_OUT_DIVA  = 1000.0
const _SPEC_DIVA   = TroughBenchmark(:F17; dx_km = 8.0)

# Same physics block as `test_trough.jl::_trough_yelmo_params`, but
# with `solver = "diva"` (matching the Fortran TROUGH-F17 namelist
# default) and the new `no_slip = false` flag explicit.
function _trough_diva_params()
    return YelmoModelParameters("trough_f17_diva_load";
        ydyn = ydyn_params(
            solver         = "diva",
            visc_method    = 1,
            beta_method    = 2,
            beta_const     = 1e3,
            beta_q         = 1.0/3.0,
            beta_u0        = 31556926.0,
            beta_gl_stag   = 3,
            beta_min       = 0.0,
            ssa_lat_bc     = "floating",
            no_slip        = false,
            ssa_solver     = SSASolver(rtol            = 1e-4,
                                       itmax           = 200,
                                       picard_tol      = 1e-3,
                                       picard_iter_max = 20,
                                       picard_relax    = 0.7),
        ),
        yneff = yneff_params(method = -1, const_ = 1e7),
        ytill = ytill_params(
            method    = 1,  scale_zb = 0,  scale_sed = 0,
            is_angle  = true, n_sd = 1,    f_sed = 1.0,
            sed_min   = 5.0, sed_max  = 15.0,
            z0        = -300.0,  z1   = 200.0,
            cf_min    = 5.0, cf_ref   = 10.0,
        ),
        ymat  = ymat_params(
            n_glen = 3.0, rf_const = 3.1536e-18,
            de_max = 0.5, enh_shear = 1.0, enh_stream = 1.0, enh_shlf = 1.0,
        ),
    )
end

@testset "benchmarks: TroughBenchmark DIVA dyn_step lockstep" begin
    fixture_path = joinpath(FIXTURES_DIR, "trough_f17_t1000.nc")
    @assert isfile(fixture_path) "Trough fixture missing: $fixture_path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl trough_f17 --overwrite` first."

    p = _trough_diva_params()
    y = YelmoModel(fixture_path, _T_OUT_DIVA;
                   alias  = "trough_f17_diva_load",
                   p      = p,
                   boundaries = :periodic_y,
                   groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                   strict = false)

    @test y isa AbstractYelmoModel
    @test y.time == _T_OUT_DIVA
    @test all(isfinite, interior(y.tpo.H_ice))
    @test all(isfinite, interior(y.dyn.ux_b))

    # Refresh diagnostics (mirror test_trough.jl's pattern).
    Yelmo.update_diagnostics!(y)

    # Stash the YelmoMirror DIVA reference depth-averaged velocities
    # *before* dyn_step! mutates them.
    ref_ux_bar = copy(interior(y.dyn.ux_bar))
    ref_uy_bar = copy(interior(y.dyn.uy_bar))
    ref_ux_b   = copy(interior(y.dyn.ux_b))
    ref_uy_b   = copy(interior(y.dyn.uy_b))

    Yelmo.YelmoModelDyn.dyn_step!(y, 1.0)

    iter_count = y.dyn.scratch.ssa_iter_now[]
    @info "Trough DIVA Picard iterations: $iter_count"
    @test iter_count > 0
    @test iter_count <= y.p.ydyn.ssa_solver.picard_iter_max

    jl_ux_bar = interior(y.dyn.ux_bar)
    jl_uy_bar = interior(y.dyn.uy_bar)
    jl_ux_b   = interior(y.dyn.ux_b)
    jl_uy_b   = interior(y.dyn.uy_b)
    jl_ux     = interior(y.dyn.ux)
    jl_ux_i   = interior(y.dyn.ux_i)

    @test all(isfinite, jl_ux_bar)
    @test all(isfinite, jl_uy_bar)
    @test all(isfinite, jl_ux_b)
    @test all(isfinite, jl_ux)
    @test all(isfinite, jl_ux_i)

    # Lockstep error vs YelmoMirror DIVA reference.
    err_ux_bar = maximum(abs.(jl_ux_bar .- ref_ux_bar))
    err_uy_bar = maximum(abs.(jl_uy_bar .- ref_uy_bar))
    rel_ux_bar = err_ux_bar / max(maximum(abs.(ref_ux_bar)), eps())
    rel_uy_bar = err_uy_bar / max(maximum(abs.(ref_uy_bar)), eps())

    err_ux_b   = maximum(abs.(jl_ux_b .- ref_ux_b))
    err_uy_b   = maximum(abs.(jl_uy_b .- ref_uy_b))
    rel_ux_b   = err_ux_b / max(maximum(abs.(ref_ux_b)), eps())
    rel_uy_b   = err_uy_b / max(maximum(abs.(ref_uy_b)), eps())

    @info "Trough DIVA dyn_step! vs YelmoMirror DIVA reference (single step):\n" *
          "  ux_bar:  abs=$(round(err_ux_bar; digits=3))  rel=$(round(rel_ux_bar; digits=4))" *
          "  ref max=$(round(maximum(abs.(ref_ux_bar)); digits=2))  jl max=$(round(maximum(abs.(jl_ux_bar)); digits=2))\n" *
          "  uy_bar:  abs=$(round(err_uy_bar; digits=3))  rel=$(round(rel_uy_bar; digits=4))" *
          "  ref max=$(round(maximum(abs.(ref_uy_bar)); digits=2))  jl max=$(round(maximum(abs.(jl_uy_bar)); digits=2))\n" *
          "  ux_b:    abs=$(round(err_ux_b;   digits=3))  rel=$(round(rel_ux_b;   digits=4))" *
          "  ref max=$(round(maximum(abs.(ref_ux_b));   digits=2))  jl max=$(round(maximum(abs.(jl_ux_b));   digits=2))\n" *
          "  uy_b:    abs=$(round(err_uy_b;   digits=3))  rel=$(round(rel_uy_b;   digits=4))" *
          "  ref max=$(round(maximum(abs.(ref_uy_b));   digits=2))  jl max=$(round(maximum(abs.(jl_uy_b));   digits=2))"

    # First-observation tolerances. ux is the dominant flow direction
    # in this geometry (along-trough); uy is a small cross-flow
    # component (~68 m/yr magnitude vs ux's ~420 m/yr) where pointwise
    # rel error inflates from a small absolute error. Use rel for ux
    # (along-flow signal-to-noise good) and abs for uy (compare absolute
    # error against the ux scale). Tighten in a follow-up once we
    # understand the residual dynamics differences.
    ref_ux_scale = max(maximum(abs.(ref_ux_bar)), 1.0)
    @test rel_ux_bar < 0.20
    @test rel_ux_b   < 0.20
    @test err_uy_bar < 0.10 * ref_ux_scale   # ≈ 42 m/yr at this scale
    @test err_uy_b   < 0.10 * ref_ux_scale

    # 3D shear diagnostic should be non-trivial (DIVA's distinguishing
    # feature vs SSA — non-zero shearing component).
    max_ux_i = maximum(abs, jl_ux_i)
    @info "DIVA 3D shear (ux_i): max=$(round(max_ux_i; digits=4)) m/yr"
    @test max_ux_i > 0.0

    # Write the post-dyn_step! YelmoModel state for inspection.
    logs_dir = abspath(joinpath(@__DIR__, "..", "..", "logs"))
    mkpath(logs_dir)
    jl_out_path = joinpath(logs_dir, "trough_f17_diva_jl_t1000.nc")
    isfile(jl_out_path) && rm(jl_out_path)
    out = init_output(y, jl_out_path;
                      selection = OutputSelection(groups=[:tpo, :dyn, :thrm, :mat, :bnd]))
    write_output!(out, y)
    close(out)
    @info "Trough DIVA post-dyn_step state written to $jl_out_path " *
          "(compare against fixture at $fixture_path)"
end
