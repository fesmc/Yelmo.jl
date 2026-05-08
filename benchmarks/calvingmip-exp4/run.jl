# benchmarks/calvingmip-exp4/run.jl
#
# CalvingMIP Experiment 4 — Thule domain, oscillating calving front,
# chained from the Exp3 spin-up restart. Same calving law as Exp2
# (calvmip_exp2!) but on the Thule bed geometry.
#
#   Phase 1 (Exp3, run separately): equilibrium calving pins the front
#                                    at r = 750 km. Restart written to
#                                    benchmarks/calvingmip-exp3/output/restart_final.nc.
#   Phase 2 (this script):          calvmip_exp2! oscillating law,
#                                    w = (u/|u|) · (−300 sin(2π t / 1000)) m/yr.
#                                    Period = 1 000 yr. Full 5-cycle protocol.
#
# Reference: CalvingMIP wiki — https://github.com/JRowanJordan/CalvingMIP/wiki
# Fortran:   yelmo/src/physics/calving/calving_ac.f90:484 (calvmip_exp2,
#            "Experiment 2 & 4 of CalvMIP")
#
# An 8-direction front-radius metric is recorded each snapshot for
# diagnostic purposes. On the Thule domain the front is not expected to
# be rotationally symmetric (the bed is not), so the radii are reported
# as diagnostics rather than symmetry metrics — no threshold check.
#
# Outputs (under output/, gitignored):
#   - timeseries.nc       — time series of the 8 front radii + asym metrics.
#   - snapshots_phase4.nc — periodic 2D snapshots (H_ice, lsf, ux_bar, uy_bar).
#   - restart_final.nc    — full 2D state at end of run.

using IceSheetBenchmarks
using Yelmo
using Yelmo: step!, init_state!
using Yelmo.YelmoModelPar: YelmoModelParameters
using Oceananigans: interior
using NCDatasets
using Statistics: mean
using Printf

# ----------------------------------------------------------------------
# Configuration.
# ----------------------------------------------------------------------

const DT_OUTER_YR     = 5.0
const SAMPLE_DT_YR    = 50.0
const T_PHASE4_MAX_YR = 5000.0          # full CalvingMIP 5-cycle protocol

const DX_KM         = 25.0
const NAMELIST_PATH = abspath(joinpath(@__DIR__, "yelmo_calvingmip_exp4.nml"))

const EXP3_RESTART = abspath(joinpath(
    @__DIR__, "..", "calvingmip-exp3", "output", "restart_final.nc"))

const OUTPUT_DIR    = abspath(joinpath(@__DIR__, "output"))
const TIMESERIES_NC = joinpath(OUTPUT_DIR, "timeseries.nc")
const SNAPSHOTS_NC  = joinpath(OUTPUT_DIR, "snapshots_phase4.nc")
const RESTART_FINAL = joinpath(OUTPUT_DIR, "restart_final.nc")

# ----------------------------------------------------------------------
# 8-direction front-radius scan (shared with exp2; copied here so exp4
# has no external dependency on exp2's run.jl).
# ----------------------------------------------------------------------

function _scan_front(vals::AbstractVector, r_km::AbstractVector)
    @inbounds for k in 1:length(vals)-1
        if vals[k] <= 0.0 && vals[k+1] > 0.0
            frac = -vals[k] / (vals[k+1] - vals[k])
            return r_km[k] + frac * (r_km[k+1] - r_km[k])
        end
    end
    return NaN
end

function _bilinear(L, xc_km, yc_km, xq, yq)
    Nx, Ny = size(L, 1), size(L, 2)
    dx = xc_km[2] - xc_km[1]
    dy = yc_km[2] - yc_km[1]
    fi = (xq - xc_km[1]) / dx + 1.0
    fj = (yq - yc_km[1]) / dy + 1.0
    i0 = clamp(Int(floor(fi)), 1, Nx - 1)
    j0 = clamp(Int(floor(fj)), 1, Ny - 1)
    s = clamp(fi - i0, 0.0, 1.0)
    t = clamp(fj - j0, 0.0, 1.0)
    return ((1 - s) * (1 - t) * L[i0,   j0,   1] +
                  s  * (1 - t) * L[i0+1, j0,   1] +
            (1 - s) *      t   * L[i0,   j0+1, 1] +
                  s  *      t   * L[i0+1, j0+1, 1])
end

function _front_radii_8(L, xc_km, yc_km;
                         step_km::Float64 = 1.0,
                         r_max_km::Float64 = 800.0)
    @assert size(L, 3) == 1
    n = Int(floor(r_max_km / step_km))
    rs = collect(0.0:step_km:r_max_km)

    function _ray(θ)
        cx, cy = cos(θ), sin(θ)
        vals = [_bilinear(L, xc_km, yc_km, r * cx, r * cy) for r in rs]
        return _scan_front(vals, rs)
    end
    r_E  = _ray(0.0)
    r_NE = _ray(π/4)
    r_N  = _ray(π/2)
    r_NW = _ray(3π/4)
    r_W  = _ray(π)
    r_SW = _ray(5π/4)
    r_S  = _ray(3π/2)
    r_SE = _ray(7π/4)
    return (E=r_E, NE=r_NE, N=r_N, NW=r_NW, W=r_W, SW=r_SW, S=r_S, SE=r_SE)
end

function _asym(radii)
    vals = filter(!isnan, collect(values(radii)))
    isempty(vals) && return NaN
    return (maximum(vals) - minimum(vals)) / mean(vals)
end

function _asym_4fold(radii)
    cards = filter(!isnan, [radii.E, radii.N, radii.W, radii.S])
    diags = filter(!isnan, [radii.NE, radii.NW, radii.SW, radii.SE])
    a_card = isempty(cards) ? NaN : (maximum(cards) - minimum(cards)) / mean(cards)
    a_diag = isempty(diags) ? NaN : (maximum(diags) - minimum(diags)) / mean(diags)
    return max(a_card, a_diag)
end

# ----------------------------------------------------------------------
# Build a YelmoModel from the IC at t = 0 and overwrite the prognostic
# fields from the Exp3 restart. Time is reset to 0 so phase-4 sin runs
# naturally; the calvmip_exp2! law only uses time modulo 1000 yr.
# ----------------------------------------------------------------------

function _build_from_exp3_restart()
    isfile(EXP3_RESTART) || error(
        "Missing Exp3 restart at $EXP3_RESTART. Run\n" *
        "  julia --project=. ../calvingmip-exp3/run.jl\n" *
        "first.")

    b = CalvingMIPBenchmark(:exp4; dx_km = DX_KM)
    p = YelmoModelParameters(NAMELIST_PATH, "calvingmip_exp4")
    y = YelmoModel(b, 0.0; p = p, boundaries = :bounded)

    NCDataset(EXP3_RESTART, "r") do ds
        for (name, dst) in (
            ("H_ice",    y.tpo.H_ice),
            ("lsf",      y.tpo.lsf),
            ("z_bed",    y.bnd.z_bed),
            ("z_sl",     y.bnd.z_sl),
            ("smb_ref",  y.bnd.smb_ref),
            ("T_srf",    y.bnd.T_srf),
            ("Q_geo",    y.bnd.Q_geo),
            ("bmb_shlf", y.bnd.bmb_shlf),
        )
            haskey(ds, name) || continue
            interior(dst)[:, :, 1] .= ds[name][:, :]
        end
    end

    # Attach exp2/4 calving hook (captures b's xc/yc).
    xc = b.xc; yc = b.yc
    y.hooks.calv_flt = (cx, cy, ux, uy, Hi, fi, lsf, t) ->
        calvmip_exp2!(cx, cy, ux, uy, Hi, fi, lsf, t; xc = xc, yc = yc)

    init_state!(y, 0.0; thrm_method = "robin")
    return y, b
end

# ----------------------------------------------------------------------
# Restart writer.
# ----------------------------------------------------------------------

function _write_restart_nc(y, b, path; reason::AbstractString = "")
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        Nx = length(b.xc); Ny = length(b.yc)
        defDim(ds, "xc", Nx); defDim(ds, "yc", Ny)
        defDim(ds, "xc_face", Nx + 1); defDim(ds, "yc_face", Ny + 1)
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
            v = defVar(ds, name, Float64, ("xc", "yc"))
            v[:, :] = src
        end

        defVar(ds, "ux_bar", Float64, ("xc_face", "yc"))[:, :] =
            interior(y.dyn.ux_bar)[:, :, 1]
        defVar(ds, "uy_bar", Float64, ("xc", "yc_face"))[:, :] =
            interior(y.dyn.uy_bar)[:, :, 1]

        ds.attrib["benchmark"] = "calvingmip-exp4"
        ds.attrib["time_yr"]   = y.time
        ds.attrib["dx_km"]     = b.dx_km
        ds.attrib["reason"]    = reason
    end
    return path
end

# ----------------------------------------------------------------------
# Time-series writer.
# ----------------------------------------------------------------------

function _write_timeseries_nc(samples, path)
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        n = length(samples)
        defDim(ds, "time", n)

        ds.attrib["title"]         = "CalvingMIP Exp4 time series + 8-direction front radii"
        ds.attrib["benchmark"]     = "calvingmip-exp4"
        ds.attrib["dt_outer_yr"]   = DT_OUTER_YR
        ds.attrib["sample_dt_yr"]  = SAMPLE_DT_YR

        for (name, units) in (
            ("time",   "yr"),
            ("max_H",  "m"),
            ("vol",    "m^3"),
            ("asym",   "1"),
            ("asym4",  "1"),
            ("front_E",  "km"), ("front_NE", "km"),
            ("front_N",  "km"), ("front_NW", "km"),
            ("front_W",  "km"), ("front_SW", "km"),
            ("front_S",  "km"), ("front_SE", "km"),
        )
            defVar(ds, name, Float64, ("time",), attrib = ["units" => units])
        end
        ds["time"][:]  = [s.time for s in samples]
        ds["max_H"][:] = [s.max_H for s in samples]
        ds["vol"][:]   = [s.vol for s in samples]
        ds["asym"][:]  = [s.asym for s in samples]
        ds["asym4"][:] = [s.asym4 for s in samples]
        for d in (:E, :NE, :N, :NW, :W, :SW, :S, :SE)
            ds["front_$(d)"][:] = [getproperty(s.r, d) for s in samples]
        end
    end
    return path
end

# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------

function main()
    mkpath(OUTPUT_DIR)
    @info "calvingmip-exp4 — t_max=$(T_PHASE4_MAX_YR) yr, dt_outer=$(DT_OUTER_YR), Thule domain"

    y, b = _build_from_exp3_restart()
    xc_km = b.xc ./ 1e3; yc_km = b.yc ./ 1e3
    Nx = length(b.xc); Ny = length(b.yc)

    function _take_sample()
        H = interior(y.tpo.H_ice)[:, :, 1]
        L = interior(y.tpo.lsf)
        cell_area = (b.dx_km * 1e3)^2
        r = _front_radii_8(L, xc_km, yc_km)
        return (
            time  = y.time,
            max_H = maximum(H),
            vol   = sum(H) * cell_area,
            asym  = _asym(r),
            asym4 = _asym_4fold(r),
            r     = r,
        )
    end

    # Streaming 2D snapshots.
    isfile(SNAPSHOTS_NC) && rm(SNAPSHOTS_NC)
    snap_ds = NCDataset(SNAPSHOTS_NC, "c")
    defDim(snap_ds, "xc", Nx); defDim(snap_ds, "yc", Ny)
    defDim(snap_ds, "xc_face", Nx + 1); defDim(snap_ds, "yc_face", Ny + 1)
    defDim(snap_ds, "time", Inf)
    defVar(snap_ds, "xc",   Float64, ("xc",))[:]   = b.xc
    defVar(snap_ds, "yc",   Float64, ("yc",))[:]   = b.yc
    defVar(snap_ds, "time", Float64, ("time",), attrib = ["units" => "yr"])
    defVar(snap_ds, "H_ice",  Float64, ("xc", "yc", "time"),
           attrib = ["units" => "m",    "long_name" => "Ice thickness"])
    defVar(snap_ds, "lsf",    Float64, ("xc", "yc", "time"),
           attrib = ["units" => "1",    "long_name" => "Level-set function"])
    defVar(snap_ds, "ux_bar", Float64, ("xc_face", "yc", "time"),
           attrib = ["units" => "m/yr", "long_name" => "Vertically-averaged x-velocity"])
    defVar(snap_ds, "uy_bar", Float64, ("xc", "yc_face", "time"),
           attrib = ["units" => "m/yr", "long_name" => "Vertically-averaged y-velocity"])
    snap_ds.attrib["title"]        = "CalvingMIP Exp4 2D snapshots"
    snap_ds.attrib["benchmark"]    = "calvingmip-exp4"
    snap_ds.attrib["sample_dt_yr"] = SAMPLE_DT_YR
    snap_ds.attrib["dx_km"]        = b.dx_km

    function _write_snapshot!(snap_idx)
        snap_ds["time"][snap_idx]         = y.time
        snap_ds["H_ice"][:, :, snap_idx]  = interior(y.tpo.H_ice)[:, :, 1]
        snap_ds["lsf"][:, :, snap_idx]    = interior(y.tpo.lsf)[:, :, 1]
        snap_ds["ux_bar"][:, :, snap_idx] = interior(y.dyn.ux_bar)[:, :, 1]
        snap_ds["uy_bar"][:, :, snap_idx] = interior(y.dyn.uy_bar)[:, :, 1]
        return nothing
    end

    samples = NamedTuple[]
    push!(samples, _take_sample())
    s0 = samples[1]
    @printf("  t = %5.0f  asym4 = %5.3f%%  asym8 = %5.2f%%\n",
            s0.time, 100 * s0.asym4, 100 * s0.asym)
    flush(stdout)
    snap_idx = 1
    _write_snapshot!(snap_idx)

    n_max        = Int(round(T_PHASE4_MAX_YR / DT_OUTER_YR))
    sample_every = max(Int(round(SAMPLE_DT_YR / DT_OUTER_YR)), 1)

    for k in 1:n_max
        step!(y, DT_OUTER_YR)
        if k % sample_every == 0
            s = _take_sample()
            push!(samples, s)
            @printf("  t = %5.0f  asym4 = %5.3f%%  asym8 = %5.2f%%  vol = %.3e m^3\n",
                    s.time, 100 * s.asym4, 100 * s.asym, s.vol)
                flush(stdout)

            snap_idx += 1
            _write_snapshot!(snap_idx)
        end
    end

    close(snap_ds)
    _write_timeseries_nc(samples, TIMESERIES_NC)
    _write_restart_nc(y, b, RESTART_FINAL; reason = "end of 5-cycle run")
    @info "wrote outputs" timeseries = TIMESERIES_NC snapshots = SNAPSHOTS_NC restart = RESTART_FINAL

    return samples
end

main()
