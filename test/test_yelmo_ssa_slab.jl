## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Diagnostic: SSA on the SLAB-S06 setup (yelmo_slab.f90) with
# imposed-constant viscosity and imposed-constant beta.
#
# Goal: isolate WHERE the trough SSA divergence vs YelmoMirror lives.
# By forcing `visc_method = 0` (constant `visc_const`) and
# `beta_method = 0` (constant `beta_const`), the matrix-assembly +
# linear solve are the *only* dynamic components of the SSA Picard
# iteration. The viscosity / beta recompute paths are bypassed.
#
# Setup (from `par/yelmo_SLAB-S06.nml` and `tests/yelmo_slab.f90`):
#   - Grid: 51 x 41 cells, dx = 2 km (100 km x 80 km extent).
#   - Geometry: uniform H_ice = 1000 m, z_bed = -alpha * x_metres
#                (linear downward slope in +x), z_srf = z_bed + H.
#   - alpha = 1e-3 (so dz_s/dx = -alpha).
#   - visc_const = 1e7 [Pa yr], beta_const = 1e3 [Pa yr / m].
#   - All grounded, no calving front, ssa_lat_bc = "none".
#   - solver = "ssa", n_glen = 3.0.
#
# Pure-SSA analytical (no SIA shear contribution):
#   u_b = rho_ice * g * H * alpha / beta
#       = 910 * 9.81 * 1000 * 1e-3 / 1e3 ~= 8.927 m/yr   (uniform interior).
#
# This file is intentionally NOT under test/benchmarks/ -- it is a
# debugging probe, not a fixture-backed regression. It writes the
# Yelmo.jl post-`dyn_step!` state to `<worktree>/logs/` for the user
# to inspect in ncview / NCDatasets.
#
# Loose assertions only: finiteness, Picard iteration count, and the
# x-only flow direction (uy_bar << ux_bar). The analytical match is
# *characterised* (printed numbers) -- NOT hard-asserted, since the
# trough divergence is being investigated.

using Test
using Yelmo
using Oceananigans: interior
using Oceananigans.Grids: Bounded
using NCDatasets
using Statistics: mean

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                            yneff_params, ytill_params

# ----------------------------------------------------------------------
# Synthetic restart fixture for the SLAB-S06 setup.
# ----------------------------------------------------------------------
# Mirrors `_write_ssa_slab_fixture!` in `test_yelmo_ssa.jl` but with
# the larger domain (51 x 41) and the SLAB-S06 slope direction.
function _write_slab_s06_fixture!(path::AbstractString;
                                  Nx::Int, Ny::Int, dx::Float64,
                                  H_const::Float64, alpha::Float64,
                                  Nz::Int)
    xc_m = collect(range(0.5*dx, (Nx - 0.5)*dx; length=Nx))
    yc_m = collect(range(0.5*dx, (Ny - 0.5)*dx; length=Ny))
    zeta_ac = collect(range(0.0, 1.0; length=Nz + 1))
    zeta_rock_ac = collect(range(0.0, 1.0; length=5))

    NCDataset(path, "c") do ds
        defDim(ds, "xc",           Nx)
        defDim(ds, "yc",           Ny)
        defDim(ds, "zeta",         Nz)
        defDim(ds, "zeta_ac",      Nz + 1)
        defDim(ds, "zeta_rock",    length(zeta_rock_ac) - 1)
        defDim(ds, "zeta_rock_ac", length(zeta_rock_ac))

        xv = defVar(ds, "xc", Float64, ("xc",))
        xv[:] = xc_m ./ 1e3
        xv.attrib["units"] = "km"
        yv = defVar(ds, "yc", Float64, ("yc",))
        yv[:] = yc_m ./ 1e3
        yv.attrib["units"] = "km"

        zc = defVar(ds, "zeta", Float64, ("zeta",))
        zc[:] = 0.5 .* (zeta_ac[1:end-1] .+ zeta_ac[2:end])
        zc.attrib["units"] = "1"
        zac = defVar(ds, "zeta_ac", Float64, ("zeta_ac",))
        zac[:] = zeta_ac
        zac.attrib["units"] = "1"
        zrc = defVar(ds, "zeta_rock", Float64, ("zeta_rock",))
        zrc[:] = 0.5 .* (zeta_rock_ac[1:end-1] .+ zeta_rock_ac[2:end])
        zrc.attrib["units"] = "1"
        zrac = defVar(ds, "zeta_rock_ac", Float64, ("zeta_rock_ac",))
        zrac[:] = zeta_rock_ac
        zrac.attrib["units"] = "1"

        # SLAB-S06: uniform H, linear slope -alpha in +x.
        H     = fill(H_const, Nx, Ny)
        z_bed = zeros(Nx, Ny)
        f_ice = ones(Nx, Ny)
        # All-grounded: z_sl far below z_bed.
        z_sl  = fill(-1e6, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            z_bed[i, j] = -alpha * xc_m[i]
        end

        for (name, arr) in (("H_ice", H), ("z_bed", z_bed),
                            ("f_ice", f_ice), ("z_sl", z_sl))
            v = defVar(ds, name, Float64, ("xc", "yc"))
            v[:, :] = arr
        end
    end
    return path
end

@testset "diagnostic: SSA SLAB-S06 with imposed visc/beta" begin
    # ----- SLAB-S06 parameters (yelmo_SLAB-S06.nml) -----
    Nx, Ny     = 51, 41
    dx         = 2_000.0          # [m]   2 km grid
    H_const    = 1000.0           # [m]
    alpha      = 1e-3             # [-]   dimensionless slope magnitude
    visc_const = 1e7              # [Pa yr]
    beta_const = 1e3              # [Pa yr / m]
    n_glen     = 3.0
    Nz         = 4

    # Default Yelmo constants (rho_ice=910, g=9.81).
    rho_ice = 910.0
    g_acc   = 9.81

    # Pure-SSA analytical plug flow: u_b = rho g H alpha / beta.
    # (The Fortran formula `(1 + beta*F2)/beta` adds a SIA contribution
    # `beta*F2 * u_b` where F2 = H/(3*eta); for solver="ssa" the SIA
    # branch is not invoked, so the SIA contribution is absent.)
    u_b_analytical = rho_ice * g_acc * H_const * alpha / beta_const

    @info "SLAB-S06 setup" Nx Ny dx_km=dx/1e3 H_const alpha visc_const beta_const u_b_analytical

    # ----- Build YelmoModel from synthetic restart -----
    tdir = mktempdir(; prefix="ssa_slab_s06_")
    path = joinpath(tdir, "ssa_slab_s06_restart.nc")
    _write_slab_s06_fixture!(path; Nx=Nx, Ny=Ny, dx=dx,
                             H_const=H_const, alpha=alpha, Nz=Nz)

    p = YelmoModelParameters("ssa_slab_diag";
        ydyn = ydyn_params(
            solver         = "ssa",
            visc_method    = 0,                # constant viscosity = visc_const
            visc_const     = visc_const,
            beta_method    = 0,                # constant beta = beta_const
            beta_const     = beta_const,
            beta_gl_scale  = 0,                # no GL scaling
            beta_min       = 0.0,
            ssa_lat_bc     = "none",           # try "none" first per prompt
            ssa_solver     = SSASolver(rtol            = 1e-8,
                                       itmax           = 500,
                                       picard_tol      = 1e-6,
                                       picard_iter_max = 50,
                                       picard_relax    = 0.7),
        ),
        yneff = yneff_params(method = -1, const_ = 1e7),
        ytill = ytill_params(method = -1),
        ymat  = ymat_params(n_glen = n_glen),
    )

    y = YelmoModel(path, 0.0;
                   rundir     = tdir,
                   alias      = "ssa_slab_s06",
                   p          = p,
                   boundaries = :bounded,    # SLAB-S06 (Bounded, Bounded)
                   strict     = false)

    # External fields used by visc_method=1/2 / cb_ref / N_eff are
    # filled but should not influence the result with imposed
    # visc/beta. Harmless.
    fill!(interior(y.mat.ATT), 1e-16)
    fill!(interior(y.dyn.cb_ref), 1.0)
    fill!(interior(y.dyn.N_eff), 1e7)

    # Initial guess: zero -- stress-test the solver.
    fill!(interior(y.dyn.ux_b),   0.0)
    fill!(interior(y.dyn.uy_b),   0.0)
    fill!(interior(y.dyn.ux_bar), 0.0)
    fill!(interior(y.dyn.uy_bar), 0.0)

    Yelmo.update_diagnostics!(y)

    # ----- Run a single dyn_step! -----
    Yelmo.YelmoModelDyn.dyn_step!(y, 1.0)

    iter_count = y.dyn.scratch.ssa_iter_now[]
    picard_iter_max = y.p.ydyn.ssa_solver.picard_iter_max
    @info "SSA SLAB-S06 Picard iterations" iter_count picard_iter_max

    # ----- Smoke checks (loose, diagnostic) -----
    Ux_bar = interior(y.dyn.ux_bar)
    Uy_bar = interior(y.dyn.uy_bar)

    @test all(isfinite, Ux_bar)
    @test all(isfinite, Uy_bar)
    @test iter_count > 0
    @test iter_count < picard_iter_max

    # ----- Verify imposed visc / beta were applied -----
    visc_eff = interior(y.dyn.visc_eff)
    beta_aa  = interior(y.dyn.beta)
    @info "Imposed-coefficient check" visc_eff_min=minimum(visc_eff) visc_eff_max=maximum(visc_eff) beta_min=minimum(beta_aa) beta_max=maximum(beta_aa)
    # visc_method=0 should fill with visc_const; beta_method=0 with beta_const.
    @test maximum(abs.(visc_eff .- visc_const)) < 1e-6 * visc_const
    @test maximum(abs.(beta_aa  .- beta_const))  < 1e-6 * beta_const

    # ----- Diagnostic: per-cell observations -----
    # Trim a 2-cell border (Dirichlet-affected) to define the
    # "interior" used for the analytical comparison.
    nb = 2
    Ux_int = Ux_bar[(1 + nb):(end - nb), (1 + nb):(end - nb), 1]
    Uy_int = Uy_bar[(1 + nb):(end - nb), (1 + nb):(end - nb), 1]

    ux_min  = minimum(Ux_int)
    ux_max  = maximum(Ux_int)
    ux_mean = mean(Ux_int)
    uy_max_abs = maximum(abs.(Uy_int))
    err_abs = maximum(abs.(Ux_int .- u_b_analytical))
    err_rel = err_abs / abs(u_b_analytical)

    @info "Interior ux_bar (border=$nb trimmed)" ux_min ux_max ux_mean u_b_analytical err_abs err_rel
    @info "Interior |uy_bar| max" uy_max_abs ratio_uy_to_ux=uy_max_abs / max(ux_max, eps())

    # Per-row x-profile (middle y-row) and per-column y-profile (middle x-col).
    j_mid = (Ny ÷ 2) + 1
    # ux_bar is XFaceField -> first dim is Nx+1 (or Nx for periodic-x).
    # Print across the full face stencil.
    println("\n=== x-profile of ux_bar at j=$j_mid (middle y-row) ===")
    println(Ux_bar[:, j_mid, 1])

    i_mid = (Nx ÷ 2) + 1
    println("\n=== y-profile of ux_bar at i=$i_mid (middle x-col) ===")
    println(Ux_bar[i_mid, :, 1])

    println("\n=== y-profile of uy_bar at i=$i_mid (middle x-col) ===")
    println(Uy_bar[i_mid, :, 1])

    println("\n=== x-profile of uy_bar at j=$j_mid (middle y-row) ===")
    println(Uy_bar[:, j_mid, 1])

    # ----- Hard-assert: cross-coupling sanity -----
    # The slope is in x only; uy_bar should be << ux_bar in magnitude.
    # If this fails, the bug is in the matrix cross-coupling.
    @test uy_max_abs < 0.1 * max(ux_max, eps())

    # NOTE: the analytical-match assertion is intentionally loose /
    # informational on this first run. The deviation pattern (uniform?
    # near-zero upstream + fast downstream?) is the diagnostic signal.
    # Tolerance lands in a follow-up after the bug is identified.

    # ----- Write Yelmo.jl post-dyn_step! state to logs/ for inspection -----
    logs_dir = abspath(joinpath(@__DIR__, "..", "logs"))
    mkpath(logs_dir)
    jl_out_path = joinpath(logs_dir, "ssa_slab_jl_dyn_step.nc")
    isfile(jl_out_path) && rm(jl_out_path)
    out = init_output(y, jl_out_path;
                      selection = OutputSelection(groups = [:tpo, :dyn, :mat, :bnd]))
    write_output!(out, y)
    close(out)
    @info "Yelmo.jl post-dyn_step! state written" path=jl_out_path
end
