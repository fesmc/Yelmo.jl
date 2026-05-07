## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Cross-method equivalence test for SSASolver(method = :energy_quadratic)
# vs (method = :residual) on the SLAB-S06 fixture.
#
# Goal: verify the energy-functional assembly produces the same `u`
# solution as the strong-form residual assembly, since (per the
# IceSheetStencils.jl `energy_functional_demo.jl`) the inner-stencil
# Hessian K equals -A_residual·dx·dy and the RHS satisfies
# b_K = -b_residual·dx·dy. The same `u` must satisfy both systems.
#
# Setup mirrors `test_yelmo_ssa_slab.jl`: 51×41 SLAB-S06 with constant
# viscosity / beta (visc_method = 0, beta_method = 0), `boundaries =
# :bounded` (no-slip Dirichlet on all 4 edges). With method =
# :energy_quadratic the Dirichlet edges + mask = 0 / -1 rows use the
# κ-penalty form; the inner stencil is the symmetric Hessian.
#
# Tolerance: `picard_tol = 1e-6` for both runs; we then assert that
# the converged `ux_bar` arrays agree to 1e-3 (m/yr) absolute and
# 1e-4 relative on the interior. The κ-penalty BC introduces tiny
# leak (∝ 1/κ); we use κ = 1e25 so this is well below 1e-6.

using Test
using Yelmo
using Oceananigans: interior
using Oceananigans.Grids: Bounded
using NCDatasets

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                            yneff_params, ytill_params

# SLAB-S06 fixture writer — duplicated from test_yelmo_ssa_slab.jl
# rather than included to keep this test self-contained.
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
        xv = defVar(ds, "xc", Float64, ("xc",));      xv[:] = xc_m ./ 1e3; xv.attrib["units"] = "km"
        yv = defVar(ds, "yc", Float64, ("yc",));      yv[:] = yc_m ./ 1e3; yv.attrib["units"] = "km"
        zc = defVar(ds, "zeta", Float64, ("zeta",))
        zc[:] = 0.5 .* (zeta_ac[1:end-1] .+ zeta_ac[2:end]); zc.attrib["units"] = "1"
        zac = defVar(ds, "zeta_ac", Float64, ("zeta_ac",));  zac[:] = zeta_ac;  zac.attrib["units"] = "1"
        zrc = defVar(ds, "zeta_rock", Float64, ("zeta_rock",))
        zrc[:] = 0.5 .* (zeta_rock_ac[1:end-1] .+ zeta_rock_ac[2:end]); zrc.attrib["units"] = "1"
        zrac = defVar(ds, "zeta_rock_ac", Float64, ("zeta_rock_ac",))
        zrac[:] = zeta_rock_ac; zrac.attrib["units"] = "1"

        H     = fill(H_const, Nx, Ny)
        z_bed = zeros(Nx, Ny)
        f_ice = ones(Nx, Ny)
        z_sl  = fill(-1e6, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            z_bed[i, j] = -alpha * xc_m[i]
        end
        for (name, arr) in (("H_ice", H), ("z_bed", z_bed),
                            ("f_ice", f_ice), ("z_sl", z_sl))
            v = defVar(ds, name, Float64, ("xc", "yc")); v[:, :] = arr
        end
    end
    return path
end

function _build_slab_model(path; method::Symbol, linear_method::Symbol,
                                  precond::Symbol = :jacobi)
    p = YelmoModelParameters("ssa_slab_energy";
        ydyn = ydyn_params(
            solver         = "ssa",
            visc_method    = 0,
            visc_const     = 1e7,
            beta_method    = 0,
            beta_const     = 1e3,
            beta_gl_scale  = 0,
            beta_min       = 0.0,
            ssa_lat_bc     = "none",
            ssa_solver     = SSASolver(method          = method,
                                       linear_method   = linear_method,
                                       precond         = precond,
                                       rtol            = 1e-10,
                                       itmax           = 1000,
                                       picard_tol      = 1e-6,
                                       picard_iter_max = 100,
                                       picard_relax    = 0.7),
        ),
        yneff = yneff_params(method = -1, const_ = 1e7),
        ytill = ytill_params(method = -1),
        ymat  = ymat_params(n_glen = 3.0),
    )
    tdir = mktempdir(; prefix="ssa_slab_energy_$(method)_")
    y = YelmoModel(path, 0.0;
                   rundir     = tdir,
                   alias      = "ssa_slab_energy_$(method)",
                   p          = p,
                   boundaries = :bounded,
                   strict     = false)
    fill!(interior(y.mat.ATT), 1e-16)
    fill!(interior(y.dyn.cb_ref), 1.0)
    fill!(interior(y.dyn.N_eff), 1e7)
    fill!(interior(y.dyn.ux_b),   0.0)
    fill!(interior(y.dyn.uy_b),   0.0)
    fill!(interior(y.dyn.ux_bar), 0.0)
    fill!(interior(y.dyn.uy_bar), 0.0)
    Yelmo.update_diagnostics!(y)
    return y
end

@testset "SSA energy_quadratic vs residual: SLAB-S06 equivalence" begin
    Nx, Ny = 51, 41
    dx = 2_000.0
    fdir = mktempdir(; prefix="ssa_slab_s06_fixture_")
    path = joinpath(fdir, "restart.nc")
    _write_slab_s06_fixture!(path; Nx=Nx, Ny=Ny, dx=dx,
                             H_const=1000.0, alpha=1e-3, Nz=4)

    # Run with the existing :residual + bicgstab path (reference).
    y_res = _build_slab_model(path; method = :residual,
                                     linear_method = :bicgstab,
                                     precond = :jacobi)
    Yelmo.YelmoModelDyn.dyn_step!(y_res, 1.0)
    ux_res = copy(interior(y_res.dyn.ux_bar))
    iter_res = y_res.dyn.scratch.ssa_iter_now[]
    @info "residual run" iter_res ux_max = maximum(abs, ux_res)

    # Run with the new :energy_quadratic + cg path.
    y_eng = _build_slab_model(path; method = :energy_quadratic,
                                     linear_method = :cg,
                                     precond = :jacobi)
    Yelmo.YelmoModelDyn.dyn_step!(y_eng, 1.0)
    ux_eng = copy(interior(y_eng.dyn.ux_bar))
    iter_eng = y_eng.dyn.scratch.ssa_iter_now[]
    @info "energy_quadratic run" iter_eng ux_max = maximum(abs, ux_eng)

    @test all(isfinite, ux_eng)
    @test iter_eng < 100
    # Strict equivalence: same `u` solution to within Picard tol.
    diff_abs = maximum(abs.(ux_eng .- ux_res))
    diff_rel = diff_abs / max(maximum(abs, ux_res), eps())
    @info "ux_bar equivalence" diff_abs diff_rel
    @test diff_abs < 1e-3            # m/yr
    @test diff_rel < 1e-4
end
