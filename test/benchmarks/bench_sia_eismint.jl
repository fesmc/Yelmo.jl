# EISMINT-moving 25-kyr SIA benchmark.
#
# Three runs:
#   - default:  dt_method = 2 (adaptive Heun + PI42), dt_outer = 100 yr,
#               pc settings of the Mirror spec (specs/yelmo_EISMINT_moving.nml)
#   - tight:    dt_method = 0 (fixed Heun, no controller), dt_outer = 1 yr.
#               Reference: many small Heun steps approximate the converged
#               solution well enough for an accuracy comparison.
#   - mirror:   read Fortran-fixture end state + recorded wallclock from
#               eismint_moving_t25000.nc.
#
# Default run does NOT log per-step (so the wallclock is a clean speed
# comparison). Pass `--with-log` to enable Yelmo.jl's `log_timestep` and
# emit a 4-panel comparison plot vs Mirror's Fortran log.
#
# Per-run scratch (rundir, optional yelmo_timesteps.nc, optional plot)
# lives under `<repo>/logs/bench_sia_eismint/`. Logs are gitignored.
#
# Reports:
#   - wallclock + ratio vs Fortran
#   - end-state max/mean H + max uxy
#   - rel L∞ H vs Mirror end state
#   - rel L∞ H Yelmo.jl-default vs Yelmo.jl-tight (own near-converged ref)
#   - With `--with-log`: η statistics + plot.
cd(@__DIR__)
import Pkg; Pkg.activate("..")

using Yelmo
using Statistics
using Oceananigans: interior
using NCDatasets

include("harness.jl")
using .YelmoBenchmarkHarness

using Yelmo.YelmoPar: YelmoParameters, ydyn_params, ymat_params, ytherm_params,
                           yhyd_params, ytill_params, ytopo_params,
                           yelmo_params

const WITH_LOG    = "--with-log" in ARGS
const SKIP_TIGHT  = "--skip-tight" in ARGS

const T_END = 25_000.0
const RUNDIR_ROOT = abspath(joinpath(@__DIR__, "..", "..", "logs", "bench_sia_eismint"))
const FIXTURE_PATH = joinpath(@__DIR__, "fixtures", "eismint_moving_t25000.nc")
const MIRROR_TS_LOG = joinpath(@__DIR__, "fixtures", "eismint_moving_timesteps.nc")
const PLOT_PATH    = joinpath(RUNDIR_ROOT, "bench_sia_eismint.png")

function build_params(dt_method::Int; log_timestep::Bool = false)
    return with_ported_options(YelmoParameters("eismint_moving_$(dt_method)";
        yelmo = yelmo_params(phys_const = "EISMINT",
            dt_method     = dt_method,
            pc_method     = "HEUN",
            pc_controller = "PI42",
            pc_filter_vel = false,      # EISMINT spec values
            pc_tol        = 2.0,
            pc_eps        = 1e-3,
            pc_n_redo     = 5,
            dt_min        = 0.01,
            cfl_max       = 0.5,
            log_timestep  = log_timestep,
            domain = "Greenland", grid_name = "GRL-16KM"
        ),
        ytopo = ytopo_params(solver="expl", use_bmb=false),
        ycalv = ycalv_params(use_lsf = false, calv_flt_method = "vm-l19", calv_grnd_method = "zero", H_min_grnd = 0.0, H_min_flt = 0.0),
        ydyn = ydyn_params(solver="sia", uz_method=3, visc_method=1,
                            eps_0=1e-6, taud_lim=2e5,
            ssa_solver = SSASolver(method = :residual),
            ssa_lat_bc = "floating", ssa_vel_max = 5000.0, ssa_iter_max = 50),
        ytill = ytill_params(method=1),   # Mirror spec; no sliding under SIA
        yhyd = yhyd_params(bkt_N_closure = 0, const_N = 1.0),
        ymat = ymat_params(n_glen=3.0, rf_const=1e-16, visc_min=1e3, de_max=0.5,
                            enh_method="shear3D", enh_shear=1.0,
                            enh_stream=1.0, enh_shlf=1.0,
            rf_method = -1),
        # Mirror Fortran's EISMINT-moving config: `ytherm.method = "fixed"`,
        # so `therm_step!` is a no-op for this benchmark.
        ytherm = ytherm_params(method="fixed"),
    ))
end

function run_yelmo(label::String, dt_method::Int, dt_outer::Float64;
                   log_timestep::Bool = false)
    rundir = joinpath(RUNDIR_ROOT, label)
    mkpath(rundir)
    p = build_params(dt_method; log_timestep = log_timestep)
    b = EISMINT1MovingBenchmark()
    y = YelmoModel(b, 0.0; p=p, boundaries=:bounded, rundir=rundir)
    fill!(interior(y.mat.ATT), p.ymat.rf_const)
    fill!(interior(y.dyn.cb_ref), 0.0)
    fill!(interior(y.dyn.N_eff),  1.0)
    n_steps = Int(round(T_END / dt_outer))
    @info "[$label] running" dt_method dt_outer n_steps log_timestep
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
    NCDataset(FIXTURE_PATH, "r") do ds
        H = Array{Float64}(ds["H_ice"][:, :, 1])
        uxy = if haskey(ds, "uxy_bar")
            Array{Float64}(ds["uxy_bar"][:, :, 1])
        else
            ux = Array{Float64}(ds["ux_bar"][:, :, 1])
            uy = Array{Float64}(ds["uy_bar"][:, :, 1])
            sqrt.(ux.^2 .+ uy.^2)
        end
        wc = haskey(ds.attrib, "mirror_wallclock_seconds") ?
                Float64(ds.attrib["mirror_wallclock_seconds"]) : NaN
        (H=H, uxy=uxy, wallclock=wc)
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

function read_eta_stats(log_path)
    NCDataset(log_path, "r") do ds
        dt  = Array{Float64}(ds["dt_now"][:])
        k   = dt .> 0                  # drop the initial row
        eta = Array{Float64}(ds["pc_eta"][:])[k]
        ssa  = Array{Int}(ds["ssa_iter"][:])[k]
        redo = Array{Int}(ds["iter_redo"][:])[k]
        (n=count(k), eta=eta, dt=dt[k], ssa=ssa, redo=redo)
    end
end

# -------- Run --------
mkpath(RUNDIR_ROOT)
y_def,  wc_def,  log_def  = run_yelmo("default", 2, 100.0; log_timestep = WITH_LOG)
y_tight, wc_tight, log_tight = if SKIP_TIGHT
    (nothing, NaN, "")
else
    run_yelmo("tight", 0, 1.0; log_timestep = false)
end

mirror = load_mirror()

# -------- Report --------
println("\n" * "="^72)
println("EISMINT-moving 25-kyr SIA benchmark$(WITH_LOG ? "  (with log_timestep)" : "")")
println("="^72)

runs_to_report = SKIP_TIGHT ?
    (("default (dt=100, adaptive Heun)", y_def, wc_def, log_def),) :
    (("default (dt=100, adaptive Heun)", y_def,   wc_def,   log_def),
     ("tight   (dt=1,   fixed Heun)",    y_tight, wc_tight, log_tight))

for (label, y, wc, log_path) in runs_to_report
    H = Array{Float64}(interior(y.tpo.H_ice)[:, :, 1])
    uxy = Array{Float64}(interior(y.dyn.uxy_bar)[:, :, 1])

    println("\n--- $label ---")
    println("  wallclock_s        = $(round(wc, digits=2))   ratio_vs_mirror = $(round(wc/mirror.wallclock, digits=2))×")
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
        println("  steps with redos   = $(count(>(0), s.redo))")
    else
        println("  (timestep log disabled for this run; pass --with-log to enable)")
    end
    common = (H .≥ 100.0) .& (mirror.H .≥ 100.0)
    println("  rel L∞ H vs mirror, full   = $(round(rel_linf(H, mirror.H), sigdigits=4))")
    println("  rel L∞ H vs mirror, common = $(round(rel_linf(H, mirror.H; mask=common), sigdigits=4))")
end

if !SKIP_TIGHT
    H_def   = Array{Float64}(interior(y_def.tpo.H_ice)[:, :, 1])
    H_tight = Array{Float64}(interior(y_tight.tpo.H_ice)[:, :, 1])
    common = (H_def .≥ 100.0) .& (H_tight .≥ 100.0)
    println("\n--- Yelmo.jl-default vs Yelmo.jl-tight (own near-converged reference) ---")
    println("  rel L∞ H, full    = $(round(rel_linf(H_def, H_tight), sigdigits=4))")
    println("  rel L∞ H, common  = $(round(rel_linf(H_def, H_tight; mask=common), sigdigits=4))")
end

println("\n--- Mirror reference ---")
println("  wallclock_s = $(round(mirror.wallclock, digits=2))")
println("  end max(H)  = $(round(maximum(mirror.H), digits=2)) m")
println("  end mean(H) = $(round(mean(mirror.H), digits=2)) m")

# -------- Plots (only if --with-log + Mirror log present) --------
if WITH_LOG && !isempty(log_def) && isfile(log_def)
    if isfile(MIRROR_TS_LOG)
        include("bench_plots.jl")
        make_bench_plots(log_def, MIRROR_TS_LOG, PLOT_PATH;
                         label = "EISMINT-moving 25-kyr SIA  (Yelmo.jl HEUN+PI42 vs Fortran HEUN+PI42)")
    else
        @warn "Mirror timestep log not found — skipping plot." path=MIRROR_TS_LOG
    end
end

println("\nDone.")
