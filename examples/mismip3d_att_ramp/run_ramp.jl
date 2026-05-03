## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("../../test")
#########################################################

# MISMIP3D Stnd + ATT-ramp full demonstration run.
#
# Stretched 3-phase variant of the CI test
# `test/benchmarks/test_mismip3d_stnd_att.jl`. Same protocol shape
# pivoted around the established Stnd baseline, with each phase
# extended from 500 yr → 2000 yr so the GL has time to migrate
# noticeably and (partially) re-equilibrate. Total run = 6000 yr at
# dt = 1 yr ≈ 10 min on an M-series Mac.
#
# Outputs (under `<repo>/logs/mismip3d_att_ramp/`, gitignored):
#   - `timeseries.nc`  : every 10 yr — `time, A, gl_idx, gl_x_km,
#                         max_H, mean_H, max_ux_b, n_grnd, n_shelf`
#   - `ramp.png`       : 2-panel CairoMakie plot (top: A(t) on log
#                         scale, bottom: GL_x(t) in km).
#
# Demonstration only — no assertions. The CI test
# `test_mismip3d_stnd_att.jl` already covers the qualitative
# advance/retreat invariants and YelmoMirror lockstep at t=1500.
#
# When adaptive-dt / predictor-corrector lands, the natural follow-up
# is the literal Fortran 71-kyr Pattyn-2017 protocol from
# `yelmo/tests/yelmo_mismip_new.f90:148-247` (15-kyr Stnd
# equilibration + 3 × 2-kyr ATT phases at A ∈ [1e-16, 1e-17, 1e-16]
# Pa^-3 yr^-1 + 50-kyr buffer). At dt=1 forward-Euler that's
# infeasible (~2 hr); under PC adaptive dt it should fit in the same
# coffee-break budget as this stretched-3-phase variant.

using Yelmo
using Statistics
using Oceananigans: interior
using NCDatasets
using CairoMakie
using Printf

include(joinpath(@__DIR__, "..", "..", "test", "benchmarks", "helpers.jl"))
using .YelmoBenchmarks

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           yneff_params, ytill_params, ytopo_params

# ----------------------------------------------------------------------
# Phase schedule and sampling cadence.
# ----------------------------------------------------------------------

const A_BASELINE = 3.1536e-18              # Pa^-3 yr^-1
const PHASES = [
    (label = "Stnd",     duration_yr = 2000.0, A = A_BASELINE),
    (label = "A_low",    duration_yr = 2000.0, A = A_BASELINE * 0.1),
    (label = "recovery", duration_yr = 2000.0, A = A_BASELINE),
]
const SAMPLE_DT_YR = 10.0
const DT_YR        = 1.0
const J_CENTER     = 4

const OUTPUT_DIR = abspath(joinpath(@__DIR__, "..", "..", "logs",
                                    "mismip3d_att_ramp"))

# ----------------------------------------------------------------------
# Model build (matches test_mismip3d_stnd_att.jl exactly).
# ----------------------------------------------------------------------

function _params()
    return YelmoModelParameters("mismip3d_stnd_att_ramp";
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
            rf_const   = A_BASELINE,
            visc_min   = 1e3,
            de_max     = 0.5,
            enh_shear  = 1.0,
            enh_stream = 1.0,
            enh_shlf   = 1.0,
        ),
        ytopo = ytopo_params(),
    )
end

function _build(b, p)
    Nx, Ny = length(b.xc), length(b.yc)
    y = YelmoModel(b, 0.0; p=p, boundaries = :periodic_y)

    fill!(interior(y.mat.ATT),    p.ymat.rf_const)
    fill!(interior(y.dyn.cb_ref), b.cf_ref)

    H_int     = interior(y.tpo.H_ice)
    z_bed_int = interior(y.bnd.z_bed)
    @inbounds for j in 1:Ny, i in 1:Nx
        zb = z_bed_int[i, j, 1]
        H_int[i, j, 1] = (zb < b.z_bed_floor) ? 0.0 :
                                                 max(0.0, 1000.0 - 0.9 * zb)
    end
    Yelmo.update_diagnostics!(y)
    return y
end

# ----------------------------------------------------------------------
# Sampling helpers.
# ----------------------------------------------------------------------

# Easternmost grounded-cell index along centerline j.
function _gl_index(y; j::Int = J_CENTER)
    Fg = interior(y.tpo.f_grnd)[:, :, 1]
    last_grnd = 0
    @inbounds for i in 1:size(Fg, 1)
        Fg[i, j] > 0.5 && (last_grnd = i)
    end
    return last_grnd
end

# Snapshot the diagnostics we want to track.
function _snapshot(y, A_now; j::Int = J_CENTER)
    H  = interior(y.tpo.H_ice)[:, :, 1]
    Fg = interior(y.tpo.f_grnd)[:, :, 1]
    Ux = interior(y.dyn.ux_b)[:, :, 1]
    gl_idx = _gl_index(y; j=j)
    # Cell centres in km (xc is stored in m).
    xc_km = y.g.xᶜᵃᵃ ./ 1e3
    gl_x_km = gl_idx > 0 ? xc_km[gl_idx] : NaN
    return (
        time     = y.time,
        A        = A_now,
        gl_idx   = gl_idx,
        gl_x_km  = gl_x_km,
        max_H    = maximum(H),
        mean_H   = mean(H),
        max_ux_b = maximum(abs, Ux),
        n_grnd   = count(>(0.5), Fg),
        n_shelf  = count(i -> H[i] > 0 && Fg[i] <= 0.5, eachindex(H)),
    )
end

# ----------------------------------------------------------------------
# NetCDF + plot writers.
# ----------------------------------------------------------------------

function _write_timeseries_nc(samples, path)
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        n = length(samples)
        defDim(ds, "time", n)

        ds.attrib["title"] = "MISMIP3D Stnd + ATT-ramp time series"
        ds.attrib["dt_yr"] = DT_YR
        ds.attrib["sample_dt_yr"] = SAMPLE_DT_YR
        ds.attrib["j_center"] = J_CENTER

        # Define variables. Float64 throughout; Int for the discrete counters.
        defVar(ds, "time",     Float64, ("time",), attrib = ["units" => "yr"])
        defVar(ds, "A",        Float64, ("time",), attrib = ["units" => "Pa^-3 yr^-1"])
        defVar(ds, "gl_idx",   Int64,   ("time",))
        defVar(ds, "gl_x_km",  Float64, ("time",), attrib = ["units" => "km"])
        defVar(ds, "max_H",    Float64, ("time",), attrib = ["units" => "m"])
        defVar(ds, "mean_H",   Float64, ("time",), attrib = ["units" => "m"])
        defVar(ds, "max_ux_b", Float64, ("time",), attrib = ["units" => "m/yr"])
        defVar(ds, "n_grnd",   Int64,   ("time",))
        defVar(ds, "n_shelf",  Int64,   ("time",))

        ds["time"][:]     = [s.time     for s in samples]
        ds["A"][:]        = [s.A        for s in samples]
        ds["gl_idx"][:]   = [s.gl_idx   for s in samples]
        ds["gl_x_km"][:]  = [s.gl_x_km  for s in samples]
        ds["max_H"][:]    = [s.max_H    for s in samples]
        ds["mean_H"][:]   = [s.mean_H   for s in samples]
        ds["max_ux_b"][:] = [s.max_ux_b for s in samples]
        ds["n_grnd"][:]   = [s.n_grnd   for s in samples]
        ds["n_shelf"][:]  = [s.n_shelf  for s in samples]
    end
    return path
end

function _plot_ramp(samples, path)
    t   = [s.time     for s in samples]
    A   = [s.A        for s in samples]
    glx = [s.gl_x_km  for s in samples]

    fig = Figure(size = (900, 600))

    ax_A = Axis(fig[1, 1];
                title  = "Glen rate factor A(t)",
                xlabel = "time (yr)",
                ylabel = "A (Pa⁻³ yr⁻¹)",
                yscale = log10,
                xticklabelsvisible = false)
    lines!(ax_A, t, A; color = :black, linewidth = 1.5)

    ax_gl = Axis(fig[2, 1];
                 title  = "Grounding-line x-position (centerline j=$(J_CENTER))",
                 xlabel = "time (yr)",
                 ylabel = "GL x (km)")
    lines!(ax_gl, t, glx; color = :steelblue, linewidth = 1.5)

    linkxaxes!(ax_A, ax_gl)

    save(path, fig)
    return path
end

# ----------------------------------------------------------------------
# Main run.
# ----------------------------------------------------------------------

function main()
    println("MISMIP3D Stnd + ATT-ramp full demonstration run")
    println("Phases: ", join(["$(p.label) $(p.duration_yr) yr @ A=$(p.A)" for p in PHASES], "; "))

    b = MISMIP3DBenchmark(:Stnd; dx_km = 16.0)
    p = _params()
    y = _build(b, p)

    samples = NamedTuple[]
    push!(samples, _snapshot(y, PHASES[1].A))   # t=0 snapshot

    # Steps per outer SAMPLE_DT_YR — guaranteed integer because both
    # SAMPLE_DT_YR and DT_YR are picked to divide PHASES.duration_yr.
    steps_per_sample = Int(SAMPLE_DT_YR / DT_YR)

    for phase in PHASES
        @printf("\n== Phase: %-10s  duration=%.0f yr  A=%.4e ==\n",
                phase.label, phase.duration_yr, phase.A)

        # Switch the rate factor for this phase.
        fill!(interior(y.mat.ATT), phase.A)

        n_steps_phase = Int(phase.duration_yr / DT_YR)
        n_samples_phase = div(n_steps_phase, steps_per_sample)

        for k in 1:n_samples_phase
            for _ in 1:steps_per_sample
                Yelmo.step!(y, DT_YR)
            end
            s = _snapshot(y, phase.A)
            push!(samples, s)

            # Periodic console heartbeat (every 10 samples = 100 yr).
            if k % 10 == 0
                @printf("  t=%6.0f  A=%.2e  gl_x=%6.1f km  max_H=%7.1f  max_ux=%7.2f m/yr  n_grnd=%3d n_shelf=%3d\n",
                        s.time, s.A, s.gl_x_km, s.max_H, s.max_ux_b,
                        s.n_grnd, s.n_shelf)
            end
        end
    end

    println("\nFinal sample: ", samples[end])

    mkpath(OUTPUT_DIR)
    nc_path  = joinpath(OUTPUT_DIR, "timeseries.nc")
    png_path = joinpath(OUTPUT_DIR, "ramp.png")

    _write_timeseries_nc(samples, nc_path)
    _plot_ramp(samples, png_path)

    println("\nWrote $(nc_path)")
    println("Wrote $(png_path)")

    return samples
end

main()
