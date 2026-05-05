# TROUGH-F17 1000-yr DIVA benchmark — WARM-START WORKAROUND.
#
# Yelmo.jl's SSA solver fails to converge on the cold-start IC (uniform
# H=50 m, trough z_bed) — see the spawned task "Investigate SSA cold-
# start failure on TROUGH-F17 IC". Workaround: load the Fortran-Mirror
# fixture at t=1000 (where the system has equilibrated and SSA solves
# cleanly) as Yelmo.jl's IC, re-set time to 0, run for 1000 more yr.
# Compare to a separately-regenerated Mirror fixture at t=2000 (i.e.
# Mirror's equivalent of "1000 more yr from the warm state").
#
# Three runs:
#   - default:  dt_method = 2 (adaptive Heun + PI42), dt_outer = 5 yr
#               (matches the Fortran TROUGH spec's `dt = 5.0`)
#   - tight:    dt_method = 0 (fixed Heun), dt_outer = 1 yr
#               Reference for the "near-converged" accuracy check.
#   - mirror:   read regenerated TROUGH-F17 fixture at t=2000 plus
#               Fortran's per-step timestep log. Mirror wallclock for
#               the warm window is computed from the timestep log
#               (sum of `dt_now / speed * 3.6` for entries in
#               [1000, 2000)).
#
# Reports:
#   - wallclock + ratio vs Fortran (warm window)
#   - eta statistics (Yelmo.jl + Fortran, both from their timestep logs)
#   - H rel-L∞ vs Mirror end-state at t=2000
#   - H rel-L∞ vs Yelmo.jl tight (the "near-converged" reference)
cd(@__DIR__)
import Pkg; Pkg.activate("..")

using Yelmo
using Statistics
using Oceananigans: interior
using NCDatasets

include("helpers.jl")
using .YelmoBenchmarks

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           yneff_params, ytill_params, ytopo_params,
                           yelmo_params
using Yelmo.YelmoSolvers: SSASolver

const WITH_LOG = "--with-log" in ARGS

const T_END         = 1000.0    # Yelmo.jl coordinate. Mirror coord = 1000 + T_END.
const RUNDIR_ROOT   = abspath(joinpath(@__DIR__, "..", "..", "logs", "bench_diva_trough"))
const IC_PATH         = joinpath(@__DIR__, "fixtures", "trough_f17_t1000.nc")
const MIRROR_END_PATH = joinpath(@__DIR__, "fixtures", "trough_f17_t2000.nc")
const MIRROR_TS_LOG   = joinpath(@__DIR__, "fixtures", "trough_f17_timesteps.nc")
const PLOT_PATH       = joinpath(RUNDIR_ROOT, "bench_diva_trough.png")

function build_params(dt_method::Int; log_timestep::Bool = false)
    # Mirrors `test_trough_diva.jl::_trough_diva_params` plus
    # log_timestep + the adaptive PC machinery.
    return YelmoModelParameters("trough_f17_$(dt_method)";
        yelmo = yelmo_params(
            dt_method     = dt_method,
            pc_method     = "HEUN",
            pc_controller = "PI42",
            pc_tol        = 5.0,
            pc_eps        = 1.0,
            pc_n_redo     = 5,
            dt_min        = 0.01,
            cfl_max       = 0.5,
            log_timestep  = log_timestep,
        ),
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
            method   = 1,  scale_zb = 0,  scale_sed = 0,
            is_angle = true, n_sd = 1,    f_sed = 1.0,
            sed_min  = 5.0, sed_max  = 15.0,
            z0       = -300.0, z1    = 200.0,
            cf_min   = 5.0, cf_ref   = 10.0,
        ),
        ymat = ymat_params(
            n_glen = 3.0, rf_const = 3.1536e-18,
            de_max = 0.5, enh_shear = 1.0, enh_stream = 1.0, enh_shlf = 1.0,
        ),
    )
end

function run_yelmo(label::String, dt_method::Int, dt_outer::Float64;
                   log_timestep::Bool = false)
    rundir = joinpath(RUNDIR_ROOT, label)
    mkpath(rundir)
    p = build_params(dt_method; log_timestep = log_timestep)
    # Warm-start: load Mirror's t=1000 fixture as IC, then re-set
    # `y.time = 0` so the Yelmo.jl run is in [0, T_END] coordinates.
    # Trough uses periodic_y boundaries (per the Fortran namelist /
    # test_trough_diva.jl).
    y = YelmoModel(IC_PATH, 0.0; p = p, boundaries = :periodic_y,
                   rundir = rundir,
                   groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                   strict = false)
    fill!(interior(y.mat.ATT), p.ymat.rf_const)
    Yelmo.update_diagnostics!(y)
    n_steps = Int(round(T_END / dt_outer))
    @info "[$label] running" dt_method dt_outer n_steps
    t0 = time()
    for k in 1:n_steps
        Yelmo.step!(y, dt_outer)
        if (n_steps > 100 && k % (n_steps ÷ 10) == 0) || k == n_steps
            @info "[$label]" t=y.time max_H=round(maximum(interior(y.tpo.H_ice)), digits=2) elapsed=round(time()-t0, digits=1)
        end
    end
    wallclock = time() - t0
    log = y.dyn.scratch.timestep_log[]
    log === nothing || close(log)
    log_path = joinpath(rundir, "yelmo_timesteps.nc")
    return y, wallclock, isfile(log_path) ? log_path : ""
end

function load_mirror()
    H, uxy = NCDataset(MIRROR_END_PATH, "r") do ds
        H = Array{Float64}(ds["H_ice"][:, :, 1])
        uxy = if haskey(ds, "uxy_bar")
            Array{Float64}(ds["uxy_bar"][:, :, 1])
        else
            ux = Array{Float64}(ds["ux_bar"][:, :, 1])
            uy = Array{Float64}(ds["uy_bar"][:, :, 1])
            sqrt.(ux.^2 .+ uy.^2)
        end
        H, uxy
    end
    # Compute warm-window wallclock from the Fortran timestep log:
    # sum dt_now / speed * 3.6 over entries in [1000, 2000).
    # `speed` is in kyr/hr; dt_now in yr → wallclock_s = dt_now * 3.6 / speed.
    wc_warm = if isfile(MIRROR_TS_LOG)
        NCDataset(MIRROR_TS_LOG, "r") do ds
            t  = Array{Float64}(ds["time"][:])
            dt = Array{Float64}(ds["dt_now"][:])
            spd = Array{Float64}(ds["speed"][:])
            mask = (t .>= 1000.0) .& (t .< 2000.0 + 1e-6)
            # Avoid division by zero on the first few rows where speed==0.
            valid = mask .& (spd .> 0)
            sum(dt[valid] .* 3.6 ./ spd[valid])
        end
    else
        NaN
    end
    (H=H, uxy=uxy, wallclock=wc_warm)
end

function read_eta_stats(log_path; eta_var = "pc_eta", dt_var = "dt_now")
    NCDataset(log_path, "r") do ds
        eta = Array{Float64}(ds[eta_var][:])
        dt  = Array{Float64}(ds[dt_var][:])
        wc  = haskey(ds, "wallclock_s") ? Array{Float64}(ds["wallclock_s"][:]) : Float64[]
        ssa = haskey(ds, "ssa_iter")    ? Array{Int}(ds["ssa_iter"][:])      : Int[]
        redo = haskey(ds, "iter_redo")  ? Array{Int}(ds["iter_redo"][:])     : Int[]
        (n=length(eta), eta=eta, dt=dt, wc=wc, ssa=ssa, redo=redo)
    end
end

function rel_linf(a, b; mask = nothing)
    if mask === nothing
        diff = maximum(abs.(a .- b)); ref = maximum(abs.(b))
    else
        diff = maximum(abs.(a[mask] .- b[mask])); ref = maximum(abs.(b[mask]))
    end
    ref > 0 ? diff / ref : diff
end

# -------- Run --------
mkpath(RUNDIR_ROOT)
y_def,  wc_def,  log_def  = run_yelmo("default", 2, 5.0; log_timestep = WITH_LOG)
y_tight, wc_tight, log_tight = run_yelmo("tight",   0, 1.0; log_timestep = false)

mirror = load_mirror()

# -------- Report --------
println("\n" * "="^72)
println("TROUGH-F17 1000-yr DIVA benchmark")
println("="^72)

for (label, y, wc, log_path) in (
        ("default (dt=5,  adaptive Heun + DIVA)", y_def,   wc_def,   log_def),
        ("tight   (dt=1,  fixed Heun + DIVA)",    y_tight, wc_tight, log_tight))
    H = Array{Float64}(interior(y.tpo.H_ice)[:, :, 1])
    uxy = Array{Float64}(interior(y.dyn.uxy_bar)[:, :, 1])

    println("\n--- $label ---")
    println("  wallclock_s        = $(round(wc, digits=2))   ratio_vs_mirror = $(isnan(mirror.wallclock) ? "N/A" : round(wc/mirror.wallclock, digits=2))×")
    println("  end max(H)         = $(round(maximum(H), digits=2)) m   (mirror $(round(maximum(mirror.H), digits=2)))")
    println("  end mean(H)        = $(round(mean(H), digits=2)) m")
    println("  end max(uxy_bar)   = $(round(maximum(uxy), digits=2)) m/yr  (mirror $(round(maximum(mirror.uxy), digits=2)))")
    if !isempty(log_path) && isfile(log_path)
        s = read_eta_stats(log_path)
        println("  PC steps taken     = $(s.n)")
        println("  mean dt            = $(round(mean(s.dt), digits=4)) yr")
        println("  mean eta           = $(round(mean(s.eta), sigdigits=4)) m/yr")
        println("  median eta         = $(round(median(s.eta), sigdigits=4)) m/yr")
        println("  max  eta           = $(round(maximum(s.eta), sigdigits=4)) m/yr")
        println("  mean ssa_iter      = $(isempty(s.ssa) ? "N/A" : round(mean(s.ssa), digits=2))")
        println("  PC retries (>1)    = $(isempty(s.redo) ? "N/A" : count(>(1), s.redo))")
    else
        println("  (timestep log disabled for this run)")
    end
    common = (H .≥ 100.0) .& (mirror.H .≥ 100.0)
    println("  rel L∞ H vs mirror, full   = $(round(rel_linf(H, mirror.H), sigdigits=4))")
    println("  rel L∞ H vs mirror, common = $(round(rel_linf(H, mirror.H; mask=common), sigdigits=4))")
end

H_def   = Array{Float64}(interior(y_def.tpo.H_ice)[:, :, 1])
H_tight = Array{Float64}(interior(y_tight.tpo.H_ice)[:, :, 1])
common = (H_def .≥ 100.0) .& (H_tight .≥ 100.0)
println("\n--- Yelmo.jl-default vs Yelmo.jl-tight (own near-converged reference) ---")
println("  rel L∞ H, full    = $(round(rel_linf(H_def, H_tight), sigdigits=4))")
println("  rel L∞ H, common  = $(round(rel_linf(H_def, H_tight; mask=common), sigdigits=4))")

println("\n--- Mirror reference ---")
println("  wallclock_s = $(isnan(mirror.wallclock) ? "N/A — fixture missing wallclock" : round(mirror.wallclock, digits=2))")
println("  end max(H)  = $(round(maximum(mirror.H), digits=2)) m")
println("  end mean(H) = $(round(mean(mirror.H), digits=2)) m")

if isfile(MIRROR_TS_LOG)
    # Read raw, then filter to the warm window [1000, 2000).
    NCDataset(MIRROR_TS_LOG, "r") do ds
        t   = Array{Float64}(ds["time"][:])
        dt  = Array{Float64}(ds["dt_now"][:])
        eta = Array{Float64}(ds["pc_eta"][:])
        redo = Array{Int}(ds["iter_redo"][:])
        ssa  = Array{Int}(ds["ssa_iter"][:])
        mask = (t .>= 1000.0) .& (t .< 2000.0 + 1e-6)
        n = count(mask)
        println("\n--- Mirror Fortran timestep log (warm window t∈[1000,2000]) ---")
        println("  PC steps           = $n")
        println("  mean dt            = $(round(mean(dt[mask]), digits=4)) yr")
        println("  mean eta           = $(round(mean(eta[mask]), sigdigits=4)) m/yr")
        println("  median eta         = $(round(median(eta[mask]), sigdigits=4)) m/yr")
        println("  max  eta           = $(round(maximum(eta[mask]), sigdigits=4)) m/yr")
        println("  mean ssa_iter      = $(round(mean(ssa[mask]), digits=2))")
        println("  PC retries (>1)    = $(count(>(1), redo[mask]))")
    end
else
    println("\n(Mirror Fortran timestep log not present — run regen_trough_diva.jl first.)")
end

# -------- Plots (only if --with-log + Mirror log present) --------
# Yelmo.jl run is in [0, T_END] coordinates (we re-set time = 0 from the
# warm-start fixture); Mirror's matching window is [1000, 2000]. To
# overlay them on the same time axis, shift Mirror's time by -1000.
if WITH_LOG && !isempty(log_def) && isfile(log_def) && isfile(MIRROR_TS_LOG)
    include("bench_plots.jl")

    # Build a temp Mirror log with time shifted by -1000 for plotting.
    mirror_shifted = joinpath(RUNDIR_ROOT, "mirror_warm_window_shifted.nc")
    isfile(mirror_shifted) && rm(mirror_shifted)
    NCDataset(MIRROR_TS_LOG, "r") do src
        NCDataset(mirror_shifted, "c") do dst
            t   = Array{Float64}(src["time"][:])
            dt  = Array{Float64}(src["dt_now"][:])
            eta = Array{Float64}(src["pc_eta"][:])
            mask = (t .>= 1000.0) .& (t .< 2000.0 + 1e-6)
            n = count(mask)
            defDim(dst, "time", n)
            tv = defVar(dst, "time", Float64, ("time",));     tv[:] = t[mask] .- 1000.0
            dv = defVar(dst, "dt_now", Float64, ("time",));   dv[:] = dt[mask]
            ev = defVar(dst, "pc_eta", Float64, ("time",));   ev[:] = eta[mask]
        end
    end

    make_bench_plots(log_def, mirror_shifted, PLOT_PATH;
                     window = (0.0, T_END),
                     label  = "TROUGH-F17 1000-yr DIVA  warm-start  (Yelmo.jl HEUN+PI42 vs Fortran HEUN+PI42)")
elseif WITH_LOG
    @warn "log_def or MIRROR_TS_LOG missing — skipping plot"
end

println("\nDone.")
