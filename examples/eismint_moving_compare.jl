## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("../test")
#########################################################

# EISMINT-1 moving-margin comparison plot script.
#
# Generates three CairoMakie figures comparing Yelmo.jl's 25-kyr
# end-state to:
#
#   - the Fortran YelmoMirror reference (committed fixture at
#     `test/benchmarks/fixtures/eismint_moving_t25000.nc`), and
#   - the Huybrechts EISMINT-1 community-reference data in the
#     `alex-robinson/ice-benchmarks` repository (assumed cloned at
#     `/Users/alrobi001/models/ice-benchmarks/EISMINT1-moving/`).
#
# Plots produced (under `examples/figures/eismint_moving/`):
#
#   - `x-H.png`        : centerline ice-thickness cross-section vs the
#                         exactmargin / type1 / type2 reference curves.
#   - `divide-uz.png`  : vertical-velocity profile at the dome divide
#                         cell vs `divide_uz.txt`.
#   - `x-uxy.png`      : centerline depth-averaged horizontal velocity
#                         magnitude vs `x-uxy_mean.txt`.
#
# A 25-kyr Yelmo.jl trajectory takes ~3 min on an M-series Mac. The
# script caches the end-state to
# `<repo>/logs/eismint_moving_compare/yelmo_t25000.jld2`-equivalent
# (a small NetCDF snapshot) so re-plotting is instant. Pass `--rerun`
# on the command line to force a fresh trajectory.
#
# The reference uxy_mean / x-H / divide-uz data files are
# whitespace-separated `(x, value)` columns with one header line.

using NCDatasets
using DelimitedFiles
using Yelmo
using Oceananigans
using Oceananigans: interior, Face
using Oceananigans.Grids: znodes

include("../test/benchmarks/helpers.jl")
using .YelmoBenchmarks

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           yneff_params, ytill_params, ytopo_params,
                           yelmo_params

import CairoMakie as CM

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
const REPO_ROOT      = abspath(joinpath(@__DIR__, ".."))
const FIXTURES_DIR   = joinpath(REPO_ROOT, "test", "benchmarks", "fixtures")
const MIRROR_FIXTURE = joinpath(FIXTURES_DIR, "eismint_moving_t25000.nc")

const ICE_BENCH_DIR = "/Users/alrobi001/models/ice-benchmarks/EISMINT1-moving"

const FIG_DIR    = joinpath(@__DIR__, "figures", "eismint_moving")
const CACHE_DIR  = joinpath(REPO_ROOT, "logs", "eismint_moving_compare")
const YELMO_CACHE = joinpath(CACHE_DIR, "yelmo_t25000.nc")

mkpath(FIG_DIR)
mkpath(CACHE_DIR)

const RERUN = "--rerun" in ARGS

# ----------------------------------------------------------------------
# Yelmo.jl 25-kyr trajectory: run + cache (or load cached)
# ----------------------------------------------------------------------
function _eismint_moving_params()
    return YelmoModelParameters("eismint_moving_compare";
        yelmo = yelmo_params(
            dt_method     = 2,
            pc_method     = "HEUN",
            pc_controller = "PI42",
            pc_tol        = 5.0,
            pc_eps        = 1.0,
            pc_n_redo     = 5,
            dt_min        = 0.01,
            cfl_max       = 0.5,
        ),
        ydyn = ydyn_params(
            solver       = "sia",
            uz_method    = 3,
            visc_method  = 1,
            eps_0        = 1e-6,
            taud_lim     = 2e5,
        ),
        ytopo = ytopo_params(solver = "expl", use_bmb = false),
        yneff = yneff_params(method = 0, const_ = 1.0),
        ytill = ytill_params(method = -1),
        ymat  = ymat_params(
            n_glen     = 3.0,
            rf_const   = 1e-16,
            visc_min   = 1e3,
            de_max     = 0.5,
            enh_method = "shear3D",
            enh_shear  = 1.0,
            enh_stream = 1.0,
            enh_shlf   = 1.0,
        ),
    )
end

function _run_yelmo_25kyr(b::EISMINT1MovingBenchmark)
    p = _eismint_moving_params()
    y = YelmoModel(b, 0.0; p = p, boundaries = :bounded)
    fill!(interior(y.mat.ATT), p.ymat.rf_const)
    fill!(interior(y.dyn.cb_ref), 0.0)
    fill!(interior(y.dyn.N_eff),  1.0)

    @info "Running Yelmo.jl 25-kyr EISMINT-moving trajectory…"
    t0 = time()
    for k in 1:250
        step!(y, 100.0)
        k % 50 == 0 && @info "  Yelmo.jl t=$(round(Int, y.time)) yr  max(H)=$(round(maximum(interior(y.tpo.H_ice)), digits=2)) m"
    end
    wall = time() - t0
    @info "Yelmo.jl 25-kyr wall-clock: $(round(wall, digits=1)) s"
    return y, wall
end

# Cache `H_ice`, `uxy_bar`, `uz`, `zeta_ac` from the Yelmo.jl end-state
# into a small NetCDF so the plot can be re-run without re-running the
# 3-min trajectory.
function _save_yelmo_cache(y, wall::Float64; path = YELMO_CACHE)
    isfile(path) && rm(path)
    H   = Array{Float64}(interior(y.tpo.H_ice)[:, :, 1])
    uxy = Array{Float64}(interior(y.dyn.uxy_bar)[:, :, 1])
    uz  = Array{Float64}(interior(y.dyn.uz)[:, :, :])
    zac = collect(Float64, znodes(y.gt, Face()))
    NCDataset(path, "c") do ds
        Nx, Ny = size(H); Nz_ac = length(zac)
        defDim(ds, "xc",       Nx)
        defDim(ds, "yc",       Ny)
        defDim(ds, "zeta_ac",  Nz_ac)
        defVar(ds, "H_ice",    Float64, ("xc", "yc"))[:, :]    = H
        defVar(ds, "uxy_bar",  Float64, ("xc", "yc"))[:, :]    = uxy
        defVar(ds, "uz",       Float64, ("xc", "yc", "zeta_ac"))[:, :, :] = uz
        defVar(ds, "zeta_ac",  Float64, ("zeta_ac",))[:]       = zac
        ds.attrib["yelmo_wallclock_seconds"] = wall
    end
    @info "Cached Yelmo.jl end-state to $(path)"
    return path
end

function _load_yelmo_cache(path::String)
    NCDataset(path, "r") do ds
        return (
            H_ice    = Array{Float64}(ds["H_ice"][:, :]),
            uxy_bar  = Array{Float64}(ds["uxy_bar"][:, :]),
            uz       = Array{Float64}(ds["uz"][:, :, :]),
            zeta_ac  = Array{Float64}(ds["zeta_ac"][:]),
            wall_s   = Float64(get(ds.attrib, "yelmo_wallclock_seconds", NaN)),
        )
    end
end

# ----------------------------------------------------------------------
# Mirror fixture loader
# ----------------------------------------------------------------------
function _load_mirror(path::String)
    NCDataset(path, "r") do ds
        H   = Array{Float64}(ds["H_ice"][:, :, 1])
        uxy = Array{Float64}(ds["uxy_bar"][:, :, 1])
        # Plot Mirror's `uz` (kinematic vertical velocity, the field
        # that should match Yelmo.jl's `uz` and the Huybrechts
        # reference). NOT `uz_star` — that adds `ux·c_x + uy·c_y +
        # c_t` corrections used by the thermodynamic advection step,
        # not the kinematic uz.
        #
        # Note: Mirror's `uz_b` at the divide is +0.25 m/yr at
        # t = 25 kyr instead of the kinematically expected ~0 (flat
        # bed, no slip, no melt). This appears to be a Fortran-side
        # restart artefact: the formula `dzbdt = dzsdt − dHidt` is
        # used at line 234 of `velocity_general.f90` to derive a
        # synthetic bed-elevation rate, but the stored `dzsdt` field
        # is 0 in the restart while `dHidt = -0.38` (dome oscillating
        # around equilibrium), giving a non-physical `dzbdt = +0.38`
        # that drives `uz_b ≠ 0`. Yelmo.jl's `dzsdt` and `dHidt` are
        # both 0 at t = 25 kyr (true steady state via the adaptive
        # PC), so the Yelmo.jl `uz_b ≈ 0` is correct.
        uz  = Array{Float64}(ds["uz"][:, :, :, 1])
        zac = Array{Float64}(ds["zeta_ac"][:])
        wall = Float64(get(ds.attrib, "mirror_wallclock_seconds", NaN))
        return (; H_ice = H, uxy_bar = uxy, uz = uz, zeta_ac = zac, wall_s = wall)
    end
end

# ----------------------------------------------------------------------
# ice-benchmarks reference loader
# ----------------------------------------------------------------------
function _load_ref_table(path::String)
    isfile(path) || error("Reference data missing: $path")
    raw = readdlm(path; comments=false, skipstart=1)   # skip header line
    return Float64.(raw[:, 1]), Float64.(raw[:, 2])    # (x, value)
end

# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------

# Convert a 2D centerline cross-section (j = jcenter) to a (radial-x,
# value) 1D series indexed by distance from the dome summit.
function _radial_centerline(b, field::AbstractMatrix)
    Nx = length(b.xc)
    jcenter = (Nx + 1) ÷ 2
    xc_km = b.xc ./ 1e3
    x_summit_km = b.x_summit / 1e3
    x_rad = abs.(xc_km .- x_summit_km)
    vals = field[:, jcenter]
    return x_rad, vals
end

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

b = EISMINT1MovingBenchmark()

# 1. Yelmo.jl end-state (run or load cache).
yelmo = if RERUN || !isfile(YELMO_CACHE)
    y, wall = _run_yelmo_25kyr(b)
    _save_yelmo_cache(y, wall)
    _load_yelmo_cache(YELMO_CACHE)
else
    @info "Loading cached Yelmo.jl 25-kyr end-state from $(YELMO_CACHE)"
    _load_yelmo_cache(YELMO_CACHE)
end

# 2. Mirror end-state.
isfile(MIRROR_FIXTURE) || error(
    "Mirror fixture missing at $(MIRROR_FIXTURE). Run " *
    "`julia --project=test test/benchmarks/regenerate.jl eismint_moving --overwrite`.")
mirror = _load_mirror(MIRROR_FIXTURE)

@info "Wall-clock summary" yelmo_jl_s=yelmo.wall_s mirror_s=mirror.wall_s ratio=yelmo.wall_s / mirror.wall_s

# 3. Reference data.
ref_xH_exact = _load_ref_table(joinpath(ICE_BENCH_DIR, "EISMINT1-moving_x-H_exactmargin.txt"))
ref_xH_t1    = _load_ref_table(joinpath(ICE_BENCH_DIR, "EISMINT1-moving_x-H_type1.txt"))
ref_xH_t2    = _load_ref_table(joinpath(ICE_BENCH_DIR, "EISMINT1-moving_x-H_type2.txt"))
ref_uz_div   = _load_ref_table(joinpath(ICE_BENCH_DIR, "EISMINT1-moving_divide_uz.txt"))
ref_xuxy     = _load_ref_table(joinpath(ICE_BENCH_DIR, "EISMINT1-moving_x-uxy_mean.txt"))

# 4. Centerline cross-sections.
x_rad_y, H_y_line   = _radial_centerline(b, yelmo.H_ice)
_,        H_m_line   = _radial_centerline(b, mirror.H_ice)
_,        uxy_y_line = _radial_centerline(b, yelmo.uxy_bar)
_,        uxy_m_line = _radial_centerline(b, mirror.uxy_bar)

Nx = length(b.xc)
jcenter = (Nx + 1) ÷ 2
icenter = (Nx + 1) ÷ 2
uz_y_div   = yelmo.uz[icenter, jcenter, :]
uz_m_div   = mirror.uz[icenter, jcenter, :]
zeta_y_ac  = yelmo.zeta_ac
zeta_m_ac  = mirror.zeta_ac

# 5. Plots — three separate PNG figures.

# ------ x-H profile ------
let
    fig = CM.Figure(size = (700, 500))
    ax = CM.Axis(fig[1, 1];
        xlabel = "Distance from dome centre [km]",
        ylabel = "Ice thickness H [m]",
        title  = "EISMINT-1 moving — H cross-section at t = 25 kyr")

    CM.lines!(ax, ref_xH_t1[1], ref_xH_t1[2];
              color = :gray60, linestyle = :dash,    label = "Huybrechts type-1")
    CM.lines!(ax, ref_xH_t2[1], ref_xH_t2[2];
              color = :gray40, linestyle = :dot,     label = "Huybrechts type-2")
    CM.lines!(ax, ref_xH_exact[1], ref_xH_exact[2];
              color = :black,  linestyle = :solid,   label = "Huybrechts exact-margin")
    CM.scatterlines!(ax, x_rad_y, H_y_line;
              color = :tomato,    markersize = 8,    label = "Yelmo.jl (25 kyr)")
    CM.scatterlines!(ax, x_rad_y, H_m_line;
              color = :steelblue, markersize = 6,    marker = :diamond,
              label = "YelmoMirror (Fortran 25 kyr)")

    CM.axislegend(ax; position = :rt)
    CM.xlims!(ax, 0, 800); CM.ylims!(ax, 0, 3200)

    out = joinpath(FIG_DIR, "x-H.png")
    CM.save(out, fig)
    @info "Saved $(out)"
end

# ------ divide uz profile ------
let
    fig = CM.Figure(size = (700, 500))
    ax = CM.Axis(fig[1, 1];
        xlabel = "Vertical velocity uz [m/yr]",
        ylabel = "ζ (sigma — 0 = bed, 1 = surface)",
        title  = "EISMINT-1 moving — vertical velocity at the divide, t = 25 kyr")

    CM.lines!(ax, ref_uz_div[1], ref_uz_div[2];
              color = :black, linestyle = :solid,   label = "Huybrechts reference")
    CM.scatterlines!(ax, uz_y_div, zeta_y_ac;
              color = :tomato,    markersize = 8,    label = "Yelmo.jl")
    CM.scatterlines!(ax, uz_m_div, zeta_m_ac;
              color = :steelblue, markersize = 6,    marker = :diamond,
              label = "YelmoMirror")

    CM.axislegend(ax; position = :lt)

    out = joinpath(FIG_DIR, "divide-uz.png")
    CM.save(out, fig)
    @info "Saved $(out)"
end

# ------ x-uxy_mean profile ------
let
    fig = CM.Figure(size = (700, 500))
    ax = CM.Axis(fig[1, 1];
        xlabel = "Distance from dome centre [km]",
        ylabel = "Depth-averaged |u| [m/yr]",
        title  = "EISMINT-1 moving — depth-averaged velocity at t = 25 kyr")

    CM.lines!(ax, ref_xuxy[1], ref_xuxy[2];
              color = :black, linestyle = :solid,   label = "Huybrechts reference")
    CM.scatterlines!(ax, x_rad_y, uxy_y_line;
              color = :tomato,    markersize = 8,    label = "Yelmo.jl")
    CM.scatterlines!(ax, x_rad_y, uxy_m_line;
              color = :steelblue, markersize = 6,    marker = :diamond,
              label = "YelmoMirror")

    CM.axislegend(ax; position = :rt)
    CM.xlims!(ax, 0, 800)

    out = joinpath(FIG_DIR, "x-uxy.png")
    CM.save(out, fig)
    @info "Saved $(out)"
end

# Summary text dropped alongside the figures — captures the wall-clock
# comparison in plain text so the reader can see it without opening
# the plots.
let
    summary_path = joinpath(FIG_DIR, "summary.txt")
    open(summary_path, "w") do io
        println(io, "EISMINT-1 moving 25-kyr benchmark — comparison summary")
        println(io, "=" ^ 60)
        println(io, "Yelmo.jl wall-clock:  $(round(yelmo.wall_s, digits=1)) s")
        println(io, "Fortran  wall-clock:  $(round(mirror.wall_s, digits=1)) s")
        println(io, "Speed ratio (Yelmo.jl / Fortran): " *
                    "$(round(yelmo.wall_s / mirror.wall_s, digits=2))×")
        println(io)
        println(io, "Yelmo.jl  max(H)         = $(round(maximum(yelmo.H_ice), digits=2)) m")
        println(io, "Mirror    max(H)         = $(round(maximum(mirror.H_ice), digits=2)) m")
        println(io, "Yelmo.jl  max(uxy_bar)   = $(round(maximum(yelmo.uxy_bar), digits=2)) m/yr")
        println(io, "Mirror    max(uxy_bar)   = $(round(maximum(mirror.uxy_bar), digits=2)) m/yr")
    end
    @info "Saved $(summary_path)"
end

@info "Done. Figures in: $(FIG_DIR)"
