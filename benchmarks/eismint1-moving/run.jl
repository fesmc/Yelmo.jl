# benchmarks/eismint1-moving/run.jl
#
# EISMINT-1 moving-margin benchmark — full 25 kyr SIA + adaptive
# predictor-corrector trajectory. Reproduction of the dome experiment
# in Huybrechts et al. (1996), Annals of Glaciology 23.
#
# Outputs (under output/, gitignored):
#   - timeseries.nc   — time, max_H, mean_H, total_volume_m3,
#                       dome_x_km, dome_y_km, max_uz_abs sampled every
#                       SAMPLE_DT_YR yr.
#   - restart_final.nc — full state at T_END_YR (NetCDF restart written
#                       via Yelmo's existing restart writer).

using IceSheetBenchmarks
using Yelmo                       # loading both activates the YelmoBenchmarks extension
using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           ytherm_params, yneff_params, ytill_params,
                           ytopo_params, yelmo_params
using Oceananigans: interior
using NCDatasets
using Statistics
using Printf

# ----------------------------------------------------------------------
# Configuration (edit in place).
# ----------------------------------------------------------------------

const T_END_YR     = 25_000.0     # total simulated time [yr] — full equilibrium
const DT_OUTER_YR  = 100.0        # outer-loop dt [yr]; adaptive PC sub-steps inside
const SAMPLE_DT_YR = 100.0        # cadence for time-series snapshots [yr]

const DX_KM = 50.0                # EISMINT-1 default grid spacing
const A_GLEN = 1e-16              # Pa^-3 yr^-1; matches yelmo_EISMINT_moving.nml rf_const

const OUTPUT_DIR     = abspath(joinpath(@__DIR__, "output"))
const TIMESERIES_NC  = joinpath(OUTPUT_DIR, "timeseries.nc")
const RESTART_FINAL  = joinpath(OUTPUT_DIR, "restart_final.nc")

# ----------------------------------------------------------------------
# Model parameters — SIA + adaptive HEUN/PI42, therm disabled.
# Mirror of the EISMINT-1 namelist subset relevant to this configuration.
# ----------------------------------------------------------------------

function _eismint_moving_params()
    return YelmoModelParameters("eismint1_moving";
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
        ytopo = ytopo_params(
            solver  = "expl",
            use_bmb = false,
        ),
        yneff  = yneff_params(method = 0, const_ = 1.0),
        ytill  = ytill_params(method = -1),
        ymat   = ymat_params(
            n_glen     = 3.0,
            rf_const   = A_GLEN,
            visc_min   = 1e3,
            de_max     = 0.5,
            enh_method = "shear3D",
            enh_shear  = 1.0,
            enh_stream = 1.0,
            enh_shlf   = 1.0,
        ),
        ytherm = ytherm_params(method = "fixed"),
    )
end

# Build a Yelmo model from a benchmark spec and pre-fill the rate factor
# / sliding fields that this configuration assumes constant.
function _build(b, p)
    y = Yelmo.YelmoModel(b, 0.0; p = p, boundaries = :bounded)
    fill!(interior(y.mat.ATT),    p.ymat.rf_const)
    fill!(interior(y.dyn.cb_ref), 0.0)
    fill!(interior(y.dyn.N_eff),  1.0)
    return y
end

# ----------------------------------------------------------------------
# Sampling — high-level diagnostic snapshot.
# ----------------------------------------------------------------------

function _snapshot(y, b)
    H  = interior(y.tpo.H_ice)[:, :, 1]
    Uz = interior(y.dyn.uz)
    Nx, Ny = size(H)

    cell_area_m2 = (b.dx_km * 1e3)^2
    total_volume_m3 = sum(H) * cell_area_m2

    Hmax_idx = argmax(H)
    i_max, j_max = Tuple(Hmax_idx)
    dome_x_km = b.xc[i_max] / 1e3
    dome_y_km = b.yc[j_max] / 1e3

    return (
        time            = y.time,
        max_H           = maximum(H),
        mean_H          = mean(H),
        total_volume_m3 = total_volume_m3,
        dome_x_km       = dome_x_km,
        dome_y_km       = dome_y_km,
        max_uz_abs      = maximum(abs, Uz),
    )
end

function _write_timeseries_nc(samples, path)
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        n = length(samples)
        defDim(ds, "time", n)

        ds.attrib["title"]        = "EISMINT-1 moving-margin time series"
        ds.attrib["benchmark"]    = "eismint1-moving"
        ds.attrib["dt_outer_yr"]  = DT_OUTER_YR
        ds.attrib["sample_dt_yr"] = SAMPLE_DT_YR
        ds.attrib["t_end_yr"]     = T_END_YR

        defVar(ds, "time",            Float64, ("time",), attrib = ["units" => "yr"])
        defVar(ds, "max_H",           Float64, ("time",), attrib = ["units" => "m"])
        defVar(ds, "mean_H",          Float64, ("time",), attrib = ["units" => "m"])
        defVar(ds, "total_volume_m3", Float64, ("time",), attrib = ["units" => "m^3"])
        defVar(ds, "dome_x_km",       Float64, ("time",), attrib = ["units" => "km"])
        defVar(ds, "dome_y_km",       Float64, ("time",), attrib = ["units" => "km"])
        defVar(ds, "max_uz_abs",      Float64, ("time",), attrib = ["units" => "m/yr"])

        ds["time"][:]            = [s.time            for s in samples]
        ds["max_H"][:]           = [s.max_H           for s in samples]
        ds["mean_H"][:]          = [s.mean_H          for s in samples]
        ds["total_volume_m3"][:] = [s.total_volume_m3 for s in samples]
        ds["dome_x_km"][:]       = [s.dome_x_km       for s in samples]
        ds["dome_y_km"][:]       = [s.dome_y_km       for s in samples]
        ds["max_uz_abs"][:]      = [s.max_uz_abs      for s in samples]
    end
    return path
end

function _write_final_restart(y, b, path)
    # Persist the final state via IceSheetBenchmarks' fixture writer
    # for the 2D fields we control here, then overwrite H_ice with the
    # simulated value. Keeping it lightweight — full Yelmo restart
    # serialization is out of scope for this benchmark first cut.
    write_fixture!(b, path; times = [0.0])
    H_final = interior(y.tpo.H_ice)[:, :, 1]
    NCDataset(path, "a") do ds
        ds["H_ice"][:, :] = H_final
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
    @info "eismint1-moving — t_end=$(T_END_YR) yr, dt_outer=$(DT_OUTER_YR) yr, dx=$(DX_KM) km"

    b = EISMINT1MovingBenchmark(; dx_km = DX_KM, A_glen = A_GLEN)
    p = _eismint_moving_params()
    y = _build(b, p)

    n_steps         = Int(round(T_END_YR / DT_OUTER_YR))
    steps_per_sample = Int(SAMPLE_DT_YR / DT_OUTER_YR)
    n_steps_per_sample = max(steps_per_sample, 1)

    samples = NamedTuple[]
    push!(samples, _snapshot(y, b))

    @info "step 0: " sample = samples[end]

    for k in 1:n_steps
        Yelmo.step!(y, DT_OUTER_YR)
        if k % n_steps_per_sample == 0
            s = _snapshot(y, b)
            push!(samples, s)
            if k % (n_steps_per_sample * 10) == 0
                @printf("  t=%6.0f  max_H=%7.2f m  mean_H=%6.2f m  vol=%.3e m^3  dome=(%.0f,%.0f) km  max|uz|=%.3f m/yr\n",
                        s.time, s.max_H, s.mean_H, s.total_volume_m3,
                        s.dome_x_km, s.dome_y_km, s.max_uz_abs)
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
