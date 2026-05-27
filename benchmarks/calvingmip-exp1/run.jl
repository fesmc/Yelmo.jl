# benchmarks/calvingmip-exp1/run.jl
#
# CalvingMIP Experiment 1 — circular domain, equilibrium calving pinned
# at r = 750 km. Ice grows from a cold start under constant SMB until
# it reaches the pinned front, where calvmip_exp1! cancels outflow.
#
# Reference: CalvingMIP wiki — https://github.com/JRowanJordan/CalvingMIP/wiki
#
# Outputs (under output/, gitignored):
#   - timeseries.nc     — time, ice volume, max H, ice cell count.
#   - restart_final.nc  — full state at T_END_YR; consumed by
#                          benchmarks/calvingmip-exp2/run.jl.

using IceSheetBenchmarks
using Yelmo
using Yelmo: step!, init_state!
using Yelmo.YelmoPar: YelmoParameters
using Oceananigans: interior
using NCDatasets
using Statistics
using Printf

# ----------------------------------------------------------------------
# Configuration (edit in place).
# ----------------------------------------------------------------------

const T_END_YR     = 10_000.0    # full equilibrium spin-up [yr]
const DT_OUTER_YR  = 5.0         # outer-loop dt [yr]
const SAMPLE_DT_YR = 100.0       # cadence for time-series snapshots [yr]

const DX_KM          = 25.0      # CalvingMIP standard grid spacing
const NAMELIST_PATH  = abspath(joinpath(@__DIR__, "yelmo_calvingmip_exp1.nml"))

const OUTPUT_DIR     = abspath(joinpath(@__DIR__, "output"))
const TIMESERIES_NC  = joinpath(OUTPUT_DIR, "timeseries.nc")
const RESTART_FINAL  = joinpath(OUTPUT_DIR, "restart_final.nc")

# ----------------------------------------------------------------------
# Build the model.
# ----------------------------------------------------------------------

function _build()
    b = CalvingMIPBenchmark(:exp1; dx_km = DX_KM)
    p = YelmoParameters(NAMELIST_PATH, "calvingmip_exp1")
    y = YelmoModel(b, 0.0; p = p, boundaries = :bounded)

    # Attach the exp1 calving-rate hook (captures b's xc/yc).
    xc = b.xc; yc = b.yc
    y.hooks.calv_flt = (cx, cy, ux, uy, Hi, fi, lsf, t) ->
        calvmip_exp1!(cx, cy, ux, uy, Hi, fi, lsf, t; xc = xc, yc = yc)

    init_state!(y, 0.0; thrm_method = "robin")
    return y, b
end

# ----------------------------------------------------------------------
# Sampling.
# ----------------------------------------------------------------------

function _snapshot(y, b)
    H   = interior(y.tpo.H_ice)[:, :, 1]
    L   = interior(y.tpo.lsf)[:, :, 1]
    cell_area_m2 = (b.dx_km * 1e3)^2
    return (
        time           = y.time,
        max_H          = maximum(H),
        mean_H         = mean(H),
        total_volume_m3 = sum(H) * cell_area_m2,
        n_ice_cells    = count(>(0.0), H),
        n_ocean_cells  = count(>(0.0), L),
    )
end

function _write_timeseries_nc(samples, path)
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        n = length(samples)
        defDim(ds, "time", n)

        ds.attrib["title"]        = "CalvingMIP Exp1 time series"
        ds.attrib["benchmark"]    = "calvingmip-exp1"
        ds.attrib["dt_outer_yr"]  = DT_OUTER_YR
        ds.attrib["sample_dt_yr"] = SAMPLE_DT_YR
        ds.attrib["t_end_yr"]     = T_END_YR

        for (name, units) in (
            ("time",            "yr"),
            ("max_H",           "m"),
            ("mean_H",          "m"),
            ("total_volume_m3", "m^3"),
            ("n_ice_cells",     "1"),
            ("n_ocean_cells",   "1"),
        )
            defVar(ds, name, Float64, ("time",), attrib = ["units" => units])
        end
        ds["time"][:]            = [s.time            for s in samples]
        ds["max_H"][:]           = [s.max_H           for s in samples]
        ds["mean_H"][:]          = [s.mean_H          for s in samples]
        ds["total_volume_m3"][:] = [s.total_volume_m3 for s in samples]
        ds["n_ice_cells"][:]     = [Float64(s.n_ice_cells)   for s in samples]
        ds["n_ocean_cells"][:]   = [Float64(s.n_ocean_cells) for s in samples]
    end
    return path
end

# ----------------------------------------------------------------------
# Restart writer.  Persists the 2D state needed by the exp2 run.jl
# (H_ice, lsf, z_bed, plus boundary forcing). Lightweight format —
# exp2 reads back via NCDataset and re-injects into a fresh YelmoModel.
# ----------------------------------------------------------------------

function _write_restart_nc(y, b, path)
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        Nx = length(b.xc); Ny = length(b.yc)
        defDim(ds, "xc", Nx); defDim(ds, "yc", Ny)

        defVar(ds, "xc", Float64, ("xc",))[:] = b.xc
        defVar(ds, "yc", Float64, ("yc",))[:] = b.yc

        for (name, src) in (
            ("H_ice",    interior(y.tpo.H_ice)[:, :, 1]),
            ("lsf",      interior(y.tpo.lsf)[:, :, 1]),
            ("z_bed",    interior(y.bnd.z_bed)[:, :, 1]),
            ("z_sl",     interior(y.bnd.z_sl)[:, :, 1]),
            ("smb_ref",  interior(y.bnd.smb_ref)[:, :, 1]),
            ("T_srf",    interior(y.bnd.T_srf)[:, :, 1]),
            ("Q_geo",    interior(y.bnd.Q_geo)[:, :, 1]),
            ("bmb_shlf", interior(y.bnd.bmb_shlf)[:, :, 1]),
        )
            defVar(ds, name, Float64, ("xc", "yc"))[:, :] = src
        end

        ds.attrib["benchmark"] = "calvingmip-exp1"
        ds.attrib["time_yr"]   = y.time
        ds.attrib["dx_km"]     = b.dx_km
    end
    return path
end

# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------

function main()
    mkpath(OUTPUT_DIR)
    @info "calvingmip-exp1 — t_end=$(T_END_YR) yr, dt_outer=$(DT_OUTER_YR) yr, dx=$(DX_KM) km"

    y, b = _build()

    n_steps           = Int(round(T_END_YR / DT_OUTER_YR))
    steps_per_sample  = max(Int(round(SAMPLE_DT_YR / DT_OUTER_YR)), 1)

    samples = NamedTuple[]
    push!(samples, _snapshot(y, b))
    @info "step 0: " sample = samples[end]

    for k in 1:n_steps
        step!(y, DT_OUTER_YR)
        if k % steps_per_sample == 0
            s = _snapshot(y, b)
            push!(samples, s)
            if k % (steps_per_sample * 10) == 0
                @printf("  t=%6.0f  max_H=%7.2f m  mean_H=%6.2f m  vol=%.3e m^3  n_ice=%d\n",
                        s.time, s.max_H, s.mean_H, s.total_volume_m3, s.n_ice_cells)
                flush(stdout)
            end
        end
    end

    @info "final sample" sample = samples[end]
    _write_timeseries_nc(samples, TIMESERIES_NC)
    _write_restart_nc(y, b, RESTART_FINAL)
    @info "wrote outputs" timeseries = TIMESERIES_NC restart = RESTART_FINAL

    return samples
end

main()
