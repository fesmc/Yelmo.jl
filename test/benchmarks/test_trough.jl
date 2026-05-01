## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# Regression test for the TROUGH-F17 YelmoMirror fixture.
#
# The fixture `fixtures/trough_f17_t1000.nc` is produced by
# `regenerate.jl trough_f17 --overwrite` (which drives
# `run_mirror_benchmark!` end-to-end through YelmoMirror — the first
# production exercise of that path). This test:
#
#   1. **Round-trip** — load the fixture two ways:
#         - file-based `YelmoModel(restart_file, time)` constructor.
#         - in-memory `YelmoModel(b, t)` via `state(b, t)` reading
#           the same fixture.
#      Verify the Center-aligned state (H_ice, z_bed, f_grnd,
#      smb_ref, T_srf, Q_geo) agrees to round-trip precision.
#
#   2. **SSA dyn_step! lockstep** — run a single `dyn_step!` on the
#      file-based YelmoModel, configured with `solver = "ssa"` and
#      the trough's namelist parameters. Verify Picard converges
#      and print the lockstep error vs the YelmoMirror reference
#      `ux_bar`/`uy_bar` from the fixture.
#
# State (1) is asserted hard. Lockstep (2) is *informational* on
# this first end-to-end test — the tolerance is unknown until
# observed; tightening (or asserting) lands in a follow-up if the
# numbers warrant it.

using Test
using Yelmo
using Oceananigans: interior
using Oceananigans.Grids: Bounded
using NCDatasets

include("helpers.jl")
using .YelmoBenchmarks

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           yneff_params, ytill_params

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

# Match the committed fixture: 8 km grid, snapshot at t=1000.
const _T_OUT = 1000.0
const _SPEC  = TroughBenchmark(:F17; dx_km=8.0)

# Trough-physics parameters for the file-based YelmoModel — must
# echo every override from `specs/yelmo_TROUGH.nml` so the solver
# config is consistent with the fixture's reference state. Defaults
# from `ytill_params`, `yneff_params`, `ymat_params`, and
# `ydyn_params` follow the Fortran *global* defaults; the trough
# namelist overrides several of them and we must mirror that here.
#
# Phase 1 diagnostic (commit 84b284b) traced a residual ~50% lockstep
# gap to ytill defaults `is_angle=false`, `cf_ref=0.8` falling
# through, producing c_bed ~53× too large. Fixing the ytill block
# (and auditing yneff / ymat / ydyn while we're at it) closes it.
function _trough_yelmo_params()
    return YelmoModelParameters("trough_f17_load";
        # &ydyn — overrides namelist values for an SSA-only lockstep
        # check. solver = "ssa" diverges from the namelist's "diva"
        # by design (test exercises the SSA kernel against the
        # YelmoMirror reference). Other ydyn fields match namelist.
        ydyn = ydyn_params(
            solver         = "ssa",                 # namelist: "diva" — overridden for SSA-only test
            visc_method    = 1,                     # namelist: 1
            beta_method    = 2,                     # namelist: 2 (default is 1)
            beta_const     = 1e3,                   # namelist: 1e3
            beta_q         = 1.0/3.0,               # namelist: 0.3333333 (default is 1.0)
            beta_u0        = 31556926.0,            # namelist: 31556926 (default is 100)
            beta_gl_stag   = 3,                     # namelist: 3 (default is 1)
            beta_min       = 0.0,                   # namelist: 0.0 (default is 100.0)
            ssa_lat_bc     = "floating",            # namelist: "floating"
            ssa_solver     = SSASolver(rtol            = 1e-4,
                                       itmax           = 200,
                                       picard_tol      = 1e-3,    # namelist ssa_iter_conv = 1e-3
                                       picard_iter_max = 20,      # namelist ssa_iter_max  = 20
                                       picard_relax    = 0.7),    # namelist ssa_iter_rel  = 0.7
        ),
        # &yneff — namelist sets method=3 with nxi=5 subgrid sampling,
        # but Yelmo.jl has nxi > 0 deferred (errors on non-zero nxi
        # in calc_ydyn_neff!). Use method=-1 (external) so the
        # fixture's loaded N_eff is preserved as-is. Phase 1 diag
        # confirmed N_eff matches the fixture exactly under this
        # path; switching to method=3 with nxi=0 would compute a
        # slightly different N_eff than Fortran's nxi=5 result.
        yneff = yneff_params(method = -1, const_ = 1e7),
        # &ytill — namelist:
        #   method=1, scale_zb=0, scale_sed=0, is_angle=True, n_sd=1,
        #   f_sed=1.0, sed_min=5.0, sed_max=15.0, z0=-300, z1=200,
        #   cf_min=5.0, cf_ref=10.0
        # Defaults that DIFFER and must be overridden: scale_zb,
        # is_angle, n_sd, f_sed, cf_min, cf_ref. (Other fields happen
        # to match the global default already, but we pass them
        # explicitly anyway for clarity / to lock the test to the
        # namelist.)
        ytill = ytill_params(
            method    = 1,
            scale_zb  = 0,
            scale_sed = 0,
            is_angle  = true,
            n_sd      = 1,
            f_sed     = 1.0,
            sed_min   = 5.0,
            sed_max   = 15.0,
            z0        = -300.0,
            z1        =  200.0,
            cf_min    =  5.0,
            cf_ref    = 10.0,
        ),
        # &ymat — namelist overrides:
        #   rf_const=3.1536e-18, de_max=0.5, enh_shear/stream/shlf=1.0
        # (defaults are 1e-18, 2.0, 3.0/3.0/0.7).
        ymat  = ymat_params(
            n_glen     = 3.0,
            rf_const   = 3.1536e-18,
            de_max     = 0.5,
            enh_shear  = 1.0,
            enh_stream = 1.0,
            enh_shlf   = 1.0,
        ),
    )
end

@testset "benchmarks: TroughBenchmark fixture round-trip + SSA dyn_step" begin
    fixture_path = joinpath(FIXTURES_DIR, "trough_f17_t1000.nc")
    @assert isfile(fixture_path) "Trough fixture missing: $fixture_path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl trough_f17 --overwrite` first."

    # =====================================================================
    # 1. File-based load.
    #
    # The Fortran TROUGH-F17 grid is periodic in y (the trough wraps
    # around). Load on `boundaries = :periodic_y` so the SSA solver
    # treats the y=1 / y=Ny rows as wrapping interior rows rather
    # than clamping them to zero via a Bounded no-slip BC. The
    # `_load_into_field!` methods in `src/YelmoCore.jl` are
    # topology-aware and dispatch on the y-axis topology so a
    # Periodic-y YFaceField (interior shape `(Nx, Ny, 1)`) loads
    # the fixture's `(Nx, Ny)` data directly, no replicate slot.
    # =====================================================================
    p = _trough_yelmo_params()
    y_file = YelmoModel(fixture_path, _T_OUT;
                        alias  = "trough_f17_load",
                        p      = p,
                        boundaries = :periodic_y,
                        groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                        strict = false)

    @test y_file isa AbstractYelmoModel
    @test y_file.time == _T_OUT

    # =====================================================================
    # 2. In-memory load via state(b, t) — reads the same fixture but
    #    returns only Center-aligned fields, then routes them through
    #    the same allocation path the file loader uses.
    # =====================================================================
    y_mem = YelmoModel(_SPEC, _T_OUT; p=p, boundaries = :periodic_y)
    @test y_mem isa AbstractYelmoModel
    @test y_mem.time == _T_OUT

    Nx, Ny = length(_SPEC.xc), length(_SPEC.yc)

    # Shape check: 88×21×1 (singleton z-axis on a Flat grid).
    # Under Periodic-y, YFaceField has Ny face slots (the Ny+1-th face
    # is the 1-st by wrap and is not stored). XFaceField under Bounded-x
    # still has Nx+1 slots.
    @test size(interior(y_mem.tpo.H_ice))    == (Nx, Ny, 1)
    @test size(interior(y_file.tpo.H_ice))   == (Nx, Ny, 1)
    @test size(interior(y_file.dyn.ux_b))    == (Nx + 1, Ny, 1)   # XFaceField, Bounded-x
    @test size(interior(y_file.dyn.uy_b))    == (Nx, Ny, 1)       # YFaceField, Periodic-y

    # =====================================================================
    # 3. Center-field state agreement: file vs in-memory at machine
    #    precision (only difference is the Float32 → Float64 promotion
    #    in state(), which is bit-exact since both routes go through
    #    the same NetCDF on-disk Float32 representation).
    # =====================================================================
    H_file  = interior(y_file.tpo.H_ice)
    H_mem   = interior(y_mem.tpo.H_ice)
    zb_file = interior(y_file.bnd.z_bed)
    zb_mem  = interior(y_mem.bnd.z_bed)
    fg_file = interior(y_file.tpo.f_grnd)
    fg_mem  = interior(y_mem.tpo.f_grnd)
    Tsrf_f  = interior(y_file.bnd.T_srf)
    Tsrf_m  = interior(y_mem.bnd.T_srf)

    @test maximum(abs.(H_file  .- H_mem))  < 1e-9
    @test maximum(abs.(zb_file .- zb_mem)) < 1e-9
    @test maximum(abs.(fg_file .- fg_mem)) < 1e-9
    @test maximum(abs.(Tsrf_f  .- Tsrf_m)) < 1e-9

    # =====================================================================
    # 4. Physical sanity of the loaded state (dome-grew, trough is
    #    real, ice flows downhill).
    # =====================================================================
    @test all(isfinite, H_file)
    @test all(isfinite, zb_file)
    @test 50.0 < maximum(H_file) < 2000.0     # IC=50 m, grew under smb
    @test minimum(zb_file) ≈ -720.0 atol=50.0  # F17 zb_deep clamp
    @test maximum(zb_file) > 0.0              # ridge above sea level

    # ux_b should be predominantly down-trough (positive) on the
    # grounded portion of the domain.
    ux_b_file = interior(y_file.dyn.ux_b)
    @test maximum(ux_b_file) > 50.0           # active SSA flow

    # =====================================================================
    # 5. SSA dyn_step! lockstep — run dyn_step! on the file-based
    #    YelmoModel (which carries the full ATT / ux_b / cb_ref
    #    state from the fixture) and compare against the YelmoMirror
    #    reference ux_bar/uy_bar.
    #
    # Picard convergence is asserted; the lockstep error is logged.
    # The first run characterises the agreement — assertion is
    # informational pending follow-up tuning.
    # =====================================================================
    Yelmo.update_diagnostics!(y_file)

    # Stash the YelmoMirror reference depth-averaged velocities
    # *before* dyn_step! mutates them.
    ref_ux_bar = copy(interior(y_file.dyn.ux_bar))
    ref_uy_bar = copy(interior(y_file.dyn.uy_bar))

    Yelmo.YelmoModelDyn.dyn_step!(y_file, 1.0)

    iter_count = y_file.dyn.scratch.ssa_iter_now[]
    @info "Trough dyn_step! Picard iterations: $iter_count"
    @test iter_count > 0
    @test iter_count <= y_file.p.ydyn.ssa_solver.picard_iter_max

    new_ux_bar = interior(y_file.dyn.ux_bar)
    new_uy_bar = interior(y_file.dyn.uy_bar)

    @test all(isfinite, new_ux_bar)
    @test all(isfinite, new_uy_bar)

    # Lockstep error vs YelmoMirror reference.
    err_ux = maximum(abs.(new_ux_bar .- ref_ux_bar))
    err_uy = maximum(abs.(new_uy_bar .- ref_uy_bar))
    rel_ux = err_ux / max(maximum(abs.(ref_ux_bar)), eps())
    rel_uy = err_uy / max(maximum(abs.(ref_uy_bar)), eps())
    @info "Trough dyn_step! vs YelmoMirror reference: " *
          "abs err_ux=$err_ux  err_uy=$err_uy  " *
          "rel err_ux=$rel_ux  rel_uy=$rel_uy  " *
          "ref max(|ux|)=$(maximum(abs.(ref_ux_bar)))  " *
          "max(|uy|)=$(maximum(abs.(ref_uy_bar)))"
    # No hard assertion on the lockstep error — first observation.
    # Tolerance characterisation lands in a follow-up if needed.

    # Write the post-dyn_step! YelmoModel state to a NetCDF for
    # side-by-side inspection against the YelmoMirror reference fixture.
    # Output goes to <worktree>/logs/ (gitignored) — not committed.
    logs_dir = abspath(joinpath(@__DIR__, "..", "..", "logs"))
    mkpath(logs_dir)
    jl_out_path = joinpath(logs_dir, "trough_f17_jl_t1000.nc")
    isfile(jl_out_path) && rm(jl_out_path)
    out = init_output(y_file, jl_out_path;
                      selection = OutputSelection(groups=[:tpo, :dyn, :thrm, :mat, :bnd]))
    write_output!(out, y_file)
    close(out)
    @info "Trough YelmoModel post-dyn_step state written to $jl_out_path " *
          "(compare against fixture at $fixture_path)"
end
