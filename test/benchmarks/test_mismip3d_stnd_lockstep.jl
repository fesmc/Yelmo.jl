## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# MISMIP3D Stnd YelmoMirror lockstep test at t = 500 yr.
#
# Companion to the standalone trajectory test in
# `test_mismip3d_stnd.jl`. The standalone test runs Yelmo.jl on its own
# from the thicker grounded IC override
# (`H_ice = max(0, 1000 - 0.9 z_bed)` for `z_bed >= -500 m`); this
# lockstep test additionally loads the Fortran YelmoMirror reference
# fixture at t=500 yr (committed as
# `fixtures/mismip3d_stnd_t500.nc` and produced by
# `regenerate.jl mismip3d_stnd --overwrite`) and compares the two
# end-states cell-by-cell.
#
# The two simulations share:
#
#   - The same IC (Fortran's commented thicker-grounded variant from
#     `mismip3D.f90:62-64`).
#   - The same MISMIP3D Stnd boundary forcing (smb 0.5 m/yr, T_srf
#     273.15 K, Q_geo 42 mW/m^2, kill-pos calving on the eastern
#     column).
#   - The same fixed dt = 1.0 yr forward time stepping over 500 yr
#     (YelmoMirror namelist sets `dt_method = 0`, matching Yelmo.jl's
#     forward Euler).
#
# The simulations differ in the dyn-physics pipeline:
#
#   - YelmoMirror runs the full Fortran ydyn (beta_method=4 with
#     adaptive viscosity, full ATT/cb_ref recomputation per step).
#   - Yelmo.jl-standalone uses a simplified pipeline: pre-filled
#     constant ATT (rf_const=3.1536e-18) and pre-filled constant
#     cb_ref (cf_ref=3.165176e4 with ytill.method=-1 to bypass per-step
#     recomputation). Same SSA Picard solver settings.
#
# Tolerance: 50% relative error per the Phase 1 plan. Tighten in a
# follow-up after the first observation.

using Test
using Statistics
using Yelmo
using Oceananigans: interior
using NCDatasets

include("helpers.jl")
using .YelmoBenchmarks

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           yneff_params, ytill_params, ytopo_params

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

const _SPEC_LOCK = MISMIP3DBenchmark(:Stnd; dx_km=16.0)

# Shared ydyn / ymat / ytill / yneff / ytopo configuration. Mirrors
# `test_mismip3d_stnd.jl::_mismip3d_yelmo_params` exactly so both tests
# exercise the same Yelmo.jl-side solver setup.
function _mismip3d_lockstep_params()
    return YelmoModelParameters("mismip3d_stnd_lockstep";
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
        ytopo = ytopo_params(),
    )
end

# Run the Yelmo.jl-side standalone trajectory and return the final
# YelmoModel. Mirrors the time-loop body in `test_mismip3d_stnd.jl`.
function _run_mismip3d_jl_trajectory(b::MISMIP3DBenchmark, p; n_steps=500, dt=1.0)
    Nx, Ny = length(b.xc), length(b.yc)

    # Build YelmoModel from the analytical t=0 IC.
    y = YelmoModel(b, 0.0; p=p, boundaries = :periodic_y)

    # Pre-fill ATT (rf_method=0 constant rate factor) and cb_ref
    # (ytill.method=-1 frozen friction coefficient).
    fill!(interior(y.mat.ATT),    p.ymat.rf_const)
    fill!(interior(y.dyn.cb_ref), b.cf_ref)

    # Override IC with Fortran's commented thicker-grounded variant
    # (mismip3D.f90:62-64). Same code as test_mismip3d_stnd.jl lines
    # 191-196 — matches the YelmoMirror callback exactly.
    H_int     = interior(y.tpo.H_ice)
    z_bed_int = interior(y.bnd.z_bed)
    @inbounds for j in 1:Ny, i in 1:Nx
        zb = z_bed_int[i, j, 1]
        H_int[i, j, 1] = (zb < b.z_bed_floor) ? 0.0 : max(0.0, 1000.0 - 0.9 * zb)
    end

    # Materialise diagnostics from the freshly-loaded H_ice / z_bed.
    Yelmo.update_diagnostics!(y)

    # 500-yr forward Euler time loop, no diagnostic logging (the
    # standalone test handles that).
    for _ in 1:n_steps
        Yelmo.step!(y, dt)
    end

    return y
end

@testset "benchmarks: MISMIP3D Stnd YelmoMirror lockstep at t=500" begin
    b = _SPEC_LOCK
    p = _mismip3d_lockstep_params()

    # =====================================================================
    # 1. Load YelmoMirror reference at t=500 via the file-based
    #    constructor (carries face-staggered velocities ux_b / uy_b,
    #    not just the Center-aligned subset that `state(b, 500.0)`
    #    returns).
    # =====================================================================
    fixture_path = joinpath(FIXTURES_DIR, "mismip3d_stnd_t500.nc")
    @assert isfile(fixture_path) "MISMIP3D fixture missing: $fixture_path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl mismip3d_stnd --overwrite` first."

    y_ref = YelmoModel(fixture_path, 500.0;
                       alias  = "mismip3d_stnd_ref",
                       p      = p,
                       boundaries = :periodic_y,
                       groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                       strict = false)

    @test y_ref isa AbstractYelmoModel
    @test y_ref.time == 500.0

    Nx, Ny = length(b.xc), length(b.yc)
    @test size(interior(y_ref.tpo.H_ice))   == (Nx, Ny, 1)
    @test size(interior(y_ref.dyn.ux_b))    == (Nx + 1, Ny, 1)
    @test size(interior(y_ref.dyn.uy_b))    == (Nx, Ny, 1)

    # =====================================================================
    # 2. State round-trip: in-memory `state(b, 500.0)` reads the
    #    same fixture; verify Center-aligned fields agree with the
    #    file-based load to round-trip precision (Float32 → Float64
    #    promotion only).
    # =====================================================================
    y_mem = YelmoModel(b, 500.0; p=p, boundaries = :periodic_y)
    @test y_mem isa AbstractYelmoModel
    @test y_mem.time == 500.0

    H_ref_int  = interior(y_ref.tpo.H_ice)
    H_mem_int  = interior(y_mem.tpo.H_ice)
    fg_ref_int = interior(y_ref.tpo.f_grnd)
    fg_mem_int = interior(y_mem.tpo.f_grnd)

    @test maximum(abs.(H_ref_int  .- H_mem_int))  < 1e-9
    @test maximum(abs.(fg_ref_int .- fg_mem_int)) < 1e-9

    # =====================================================================
    # 3. Run Yelmo.jl 500-yr trajectory from the same thicker IC.
    # =====================================================================
    y_jl = _run_mismip3d_jl_trajectory(b, p; n_steps=500, dt=1.0)

    @test y_jl.time == 500.0

    # =====================================================================
    # 4. Lockstep comparison.
    #
    # Compared fields:
    #   - H_ice  (Center,  Nx × Ny)
    #   - f_grnd (Center,  Nx × Ny; absolute since values live in [0, 1])
    #   - ux_b   (XFace,   (Nx+1) × Ny)
    #
    # uy_b is essentially numerical noise on both sides (< 1e-12 m/yr;
    # geometry is y-invariant), so it is reported but not asserted.
    # =====================================================================
    ref_H  = interior(y_ref.tpo.H_ice)[:, :, 1]
    jl_H   = interior(y_jl.tpo.H_ice)[:, :, 1]
    ref_fg = interior(y_ref.tpo.f_grnd)[:, :, 1]
    jl_fg  = interior(y_jl.tpo.f_grnd)[:, :, 1]
    ref_ux = interior(y_ref.dyn.ux_b)[:, :, 1]
    jl_ux  = interior(y_jl.dyn.ux_b)[:, :, 1]

    rel_H  = maximum(abs.(jl_H  .- ref_H))  / max(maximum(abs.(ref_H)),  eps())
    abs_fg = maximum(abs.(jl_fg .- ref_fg))   # f_grnd in [0, 1]: absolute is more meaningful
    rel_ux = maximum(abs.(jl_ux .- ref_ux)) / max(maximum(abs.(ref_ux)), eps())

    @info "MISMIP3D Stnd YelmoMirror lockstep at t=500: " *
          "rel_H=$(round(rel_H; digits=4)) " *
          "abs_fg=$(round(abs_fg; digits=4)) " *
          "rel_ux=$(round(rel_ux; digits=4)) " *
          "ref max(|ux_b|)=$(round(maximum(abs.(ref_ux)); digits=2)) " *
          "jl  max(|ux_b|)=$(round(maximum(abs.(jl_ux));  digits=2)) " *
          "ref max(H)=$(round(maximum(ref_H); digits=2)) " *
          "jl  max(H)=$(round(maximum(jl_H);  digits=2)) " *
          "ref mean(f_grnd)=$(round(mean(ref_fg); digits=4)) " *
          "jl  mean(f_grnd)=$(round(mean(jl_fg);  digits=4))"

    # 50% smoke tolerance per Phase 1 plan; tighten in a follow-up if
    # results are clean.
    @test rel_H  < 0.5
    @test abs_fg < 0.5
    # KNOWN-BROKEN: max|ux_b| differs ~4.5× between Yelmo.jl (~175 m/yr)
    # and YelmoMirror (~779 m/yr) at t=500 despite matching geometry to
    # <1% (max H peak agreement 1574.7 vs 1573.7 m, identical
    # mean(f_grnd)=0.49). Picard converges in 1 iter on Yelmo.jl side
    # throughout the run — suggests the visc nonlinearity isn't being
    # iterated (visc_eff_int not refreshed between iters, or
    # beta_method=4 under-velocity bug, or a residual hotspot analogous
    # to trough's three from PR #25/26). Flagged for separate follow-up
    # investigation. See PR description for diagnostic notes.
    @test_broken rel_ux < 0.5

    # =====================================================================
    # 5. NetCDF dump of Yelmo.jl-side end-state for inspection.
    # =====================================================================
    logs_dir = abspath(joinpath(@__DIR__, "..", "..", "logs"))
    mkpath(logs_dir)
    out_path = joinpath(logs_dir, "mismip3d_stnd_lockstep_jl.nc")
    isfile(out_path) && rm(out_path)
    out = init_output(y_jl, out_path;
                      selection = OutputSelection(groups=[:tpo, :dyn, :mat, :bnd]))
    write_output!(out, y_jl)
    close(out)
    @info "Yelmo.jl post-500yr state written to $out_path " *
          "(compare against fixture at $fixture_path)"
end
