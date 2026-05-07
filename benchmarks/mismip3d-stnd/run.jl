# benchmarks/mismip3d-stnd/run.jl
#
# MISMIP3D Stnd benchmark — 2000 yr SSA + grounding-line dynamics on a
# y-invariant downward-sloping bed. Reproduces the steady-state buildup
# phase of Pattyn et al. (2013), J. Glaciol. 59(215).
#
# Outputs (under output/, gitignored):
#   - timeseries.nc    — every SAMPLE_DT_YR yr: time, max_H, mean_H,
#                        mean_f_grnd, gl_x_km, max_ux_abs,
#                        max_uy_center_abs, picard_iters_last.
#   - restart_final.nc — final 2D state (H_ice, z_bed, …).

using IceSheetBenchmarks
using Yelmo
using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           ytherm_params, yneff_params, ytill_params,
                           ytopo_params
using Oceananigans: interior
using NCDatasets
using Statistics
using Printf

# ----------------------------------------------------------------------
# Configuration (edit in place).
# ----------------------------------------------------------------------

const T_END_YR     = 2000.0     # total simulated time [yr]
const DT_YR        = 1.0        # forward-Euler timestep [yr]
const SAMPLE_DT_YR = 10.0       # cadence for time-series snapshots [yr]

const DX_KM       = 16.0        # → Nx=51, Ny=7 (centerline j=4)

const OUTPUT_DIR    = abspath(joinpath(@__DIR__, "output"))
const TIMESERIES_NC = joinpath(OUTPUT_DIR, "timeseries.nc")
const RESTART_FINAL = joinpath(OUTPUT_DIR, "restart_final.nc")

# ----------------------------------------------------------------------
# Model parameters — SSA + Coulomb friction, therm disabled.
# Mirrors test/benchmarks/test_mismip3d_stnd.jl and the Fortran namelist.
# ----------------------------------------------------------------------

function _mismip3d_params(b)
    return YelmoModelParameters("mismip3d_stnd";
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
            rf_const   = b.A_glen,
            visc_min   = 1e3,
            de_max     = 0.5,
            enh_shear  = 1.0,
            enh_stream = 1.0,
            enh_shlf   = 1.0,
        ),
        ytopo  = ytopo_params(),
        ytherm = ytherm_params(method = "fixed"),
    )
end

function _build(b, p)
    Nx, Ny = length(b.xc), length(b.yc)
    y = Yelmo.YelmoModel(b, 0.0; p = p, boundaries = :periodic_y)

    # Pre-fill mat.ATT (rf_method=0) and dyn.cb_ref (till bypass).
    fill!(interior(y.mat.ATT),    p.ymat.rf_const)
    fill!(interior(y.dyn.cb_ref), b.cf_ref)

    # Override IC with Fortran's commented thicker-grounded variant
    # (mismip3D.f90:62-64). The literal 10 m all-floating IC is
    # rank-deficient under SSA — see README for full discussion.
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
# Sampling — diagnostic snapshot.
# ----------------------------------------------------------------------

# Easternmost grounded cell along the centerline j (cell-centred xc).
function _gl_index(Fg, j::Int)
    last_grnd = 0
    @inbounds for i in axes(Fg, 1)
        Fg[i, j] > 0.5 && (last_grnd = i)
    end
    return last_grnd
end

function _snapshot(y, b, j_center::Int)
    H  = interior(y.tpo.H_ice)[:, :, 1]
    Fg = interior(y.tpo.f_grnd)[:, :, 1]
    Ux = interior(y.dyn.ux_bar)[:, :, 1]
    Uy = interior(y.dyn.uy_bar)[:, :, 1]

    gl_idx  = _gl_index(Fg, j_center)
    gl_x_km = gl_idx > 0 ? b.xc[gl_idx] / 1e3 : NaN

    return (
        time              = y.time,
        max_H             = maximum(H),
        mean_H            = mean(H),
        mean_f_grnd       = mean(Fg),
        gl_x_km           = gl_x_km,
        max_ux_abs        = maximum(abs, Ux),
        max_uy_center_abs = maximum(abs, Uy[:, j_center]),
        picard_iters_last = Int(y.dyn.scratch.ssa_iter_now[]),
    )
end

# ----------------------------------------------------------------------
# Output writers.
# ----------------------------------------------------------------------

function _write_timeseries_nc(samples, path)
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        n = length(samples)
        defDim(ds, "time", n)

        ds.attrib["title"]        = "MISMIP3D Stnd time series"
        ds.attrib["benchmark"]    = "mismip3d-stnd"
        ds.attrib["dt_yr"]        = DT_YR
        ds.attrib["sample_dt_yr"] = SAMPLE_DT_YR
        ds.attrib["t_end_yr"]     = T_END_YR

        defVar(ds, "time",              Float64, ("time",), attrib = ["units" => "yr"])
        defVar(ds, "max_H",             Float64, ("time",), attrib = ["units" => "m"])
        defVar(ds, "mean_H",            Float64, ("time",), attrib = ["units" => "m"])
        defVar(ds, "mean_f_grnd",       Float64, ("time",), attrib = ["units" => "1"])
        defVar(ds, "gl_x_km",           Float64, ("time",), attrib = ["units" => "km"])
        defVar(ds, "max_ux_abs",        Float64, ("time",), attrib = ["units" => "m/yr"])
        defVar(ds, "max_uy_center_abs", Float64, ("time",), attrib = ["units" => "m/yr"])
        defVar(ds, "picard_iters_last", Int64,   ("time",))

        ds["time"][:]              = [s.time              for s in samples]
        ds["max_H"][:]             = [s.max_H             for s in samples]
        ds["mean_H"][:]            = [s.mean_H            for s in samples]
        ds["mean_f_grnd"][:]       = [s.mean_f_grnd       for s in samples]
        ds["gl_x_km"][:]           = [s.gl_x_km           for s in samples]
        ds["max_ux_abs"][:]        = [s.max_ux_abs        for s in samples]
        ds["max_uy_center_abs"][:] = [s.max_uy_center_abs for s in samples]
        ds["picard_iters_last"][:] = [s.picard_iters_last for s in samples]
    end
    return path
end

function _write_final_restart(y, b, path)
    write_fixture!(b, path; times = [0.0])
    H_final     = interior(y.tpo.H_ice)[:, :, 1]
    f_grnd      = interior(y.tpo.f_grnd)[:, :, 1]
    NCDataset(path, "a") do ds
        ds["H_ice"][:, :] = H_final
        # f_grnd not in the IC fixture — add it.
        if !haskey(ds, "f_grnd")
            v = defVar(ds, "f_grnd", Float64, ("xc", "yc"))
            v[:, :] = f_grnd
            v.attrib["units"]     = "1"
            v.attrib["long_name"] = "Grounded fraction"
        else
            ds["f_grnd"][:, :] = f_grnd
        end
        ds.attrib["solution_type"] = "yelmojl-trajectory"
        ds.attrib["time_yr"]       = y.time
    end
    return path
end

# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------

function main()
    mkpath(OUTPUT_DIR)
    @info "mismip3d-stnd — t_end=$(T_END_YR) yr, dt=$(DT_YR) yr, dx=$(DX_KM) km"

    b = MISMIP3DBenchmark(:Stnd; dx_km = DX_KM)
    p = _mismip3d_params(b)
    y = _build(b, p)

    Nx, Ny = length(b.xc), length(b.yc)
    j_center = div(Ny + 1, 2)

    n_steps          = Int(round(T_END_YR / DT_YR))
    steps_per_sample = max(Int(SAMPLE_DT_YR / DT_YR), 1)

    samples = NamedTuple[]
    push!(samples, _snapshot(y, b, j_center))

    for k in 1:n_steps
        Yelmo.step!(y, DT_YR)
        if k % steps_per_sample == 0
            s = _snapshot(y, b, j_center)
            push!(samples, s)
            if k % (steps_per_sample * 10) == 0
                @printf("  t=%5.0f  max_H=%7.1f  mean_H=%7.1f  mean(fgrnd)=%.3f  GLx=%6.1f km  max|ux|=%7.2f  picard=%d\n",
                        s.time, s.max_H, s.mean_H, s.mean_f_grnd,
                        s.gl_x_km, s.max_ux_abs, s.picard_iters_last)
            end
        end
    end

    @info "final sample" sample = samples[end]

    _write_timeseries_nc(samples, TIMESERIES_NC)
    _write_final_restart(y, b, RESTART_FINAL)

    @info "wrote outputs" timeseries = TIMESERIES_NC restart = RESTART_FINAL

    return samples
end

main()
