# benchmarks/calvingmip-exp2/run.jl
#
# CalvingMIP Experiment 2 — oscillating calving front, chained from
# the Exp1 spin-up restart.
#
#   Phase 1 (Exp1, run separately): equilibrium calving pins the front
#                                    at r = 750 km. Restart written to
#                                    benchmarks/calvingmip-exp1/output/restart_final.nc.
#   Phase 2 (this script):          calvmip_exp2! oscillating law,
#                                    w = (u/|u|) · (−300 sin(2π t / 1000)) m/yr.
#                                    Period = 1 000 yr.
#
# An 8-direction front-radius asymmetry metric is recorded each
# snapshot. The run terminates once the metric exceeds
# ASYM_THRESHOLD (default 5 %) — there is no need to finish the full
# 5-cycle protocol; once asymmetry develops it grows monotonically.
#
# Outputs (under output/, gitignored):
#   - timeseries.nc                  — time series of the 8 front radii
#                                       and the asymmetry metric.
#   - restart_at_threshold.nc        — full state at the moment the
#                                       threshold is crossed.
#   - snapshots_phase2.nc (optional) — periodic 2D snapshots.

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

const DT_OUTER_YR    = 5.0
const SAMPLE_DT_YR   = 50.0
const T_PHASE2_MAX_YR = 1500.0          # cap if the threshold is never hit
const ASYM_THRESHOLD = 0.05             # 5 % — (max−min)/mean over 8 radii

const DX_KM         = 25.0
const NAMELIST_PATH = abspath(joinpath(@__DIR__, "yelmo_calvingmip_exp2.nml"))

const EXP1_RESTART = abspath(joinpath(
    @__DIR__, "..", "calvingmip-exp1", "output", "restart_final.nc"))

const OUTPUT_DIR    = abspath(joinpath(@__DIR__, "output"))
const TIMESERIES_NC = joinpath(OUTPUT_DIR, "timeseries.nc")
const RESTART_NC    = joinpath(OUTPUT_DIR, "restart_at_threshold.nc")
const SNAPSHOTS_NC  = joinpath(OUTPUT_DIR, "snapshots_phase2.nc")

# ----------------------------------------------------------------------
# Asymmetry metric: 8-direction front radii (E, NE, N, NW, W, SW, S, SE)
# from the centre of the domain to the first lsf 0-crossing.
# ----------------------------------------------------------------------

# Linearly interpolate the radius at which a 1-D `vals` profile crosses
# from ≤ 0 to > 0. Returns NaN if no crossing along the ray.
function _scan_front(vals::AbstractVector, r_km::AbstractVector)
    @inbounds for k in 1:length(vals)-1
        if vals[k] <= 0.0 && vals[k+1] > 0.0
            frac = -vals[k] / (vals[k+1] - vals[k])
            return r_km[k] + frac * (r_km[k+1] - r_km[k])
        end
    end
    return NaN
end

# Cardinal + intercardinal radii, with the centre (imid, jmid) chosen
# to be the cell whose centre is closest to the origin.
function _front_radii_8(L, xc_km, yc_km, imid, jmid)
    Nx, Ny = size(L, 1), size(L, 2)
    r_E = _scan_front(L[imid:Nx,   jmid, 1],    xc_km[imid:Nx])
    r_W = _scan_front(L[imid:-1:1, jmid, 1], -xc_km[imid:-1:1])
    r_N = _scan_front(L[imid, jmid:Ny,   1],    yc_km[jmid:Ny])
    r_S = _scan_front(L[imid, jmid:-1:1, 1], -yc_km[jmid:-1:1])

    function _diag(di, dj)
        n = min(di > 0 ? Nx - imid : imid - 1,
                dj > 0 ? Ny - jmid : jmid - 1)
        lsf = [L[imid + di*k, jmid + dj*k, 1] for k in 0:n]
        r   = [sqrt(xc_km[imid + di*k]^2 + yc_km[jmid + dj*k]^2)
               for k in 0:n]
        return _scan_front(lsf, r)
    end
    r_NE = _diag(+1, +1); r_NW = _diag(-1, +1)
    r_SW = _diag(-1, -1); r_SE = _diag(+1, -1)

    return (E=r_E, NE=r_NE, N=r_N, NW=r_NW, W=r_W, SW=r_SW, S=r_S, SE=r_SE)
end

function _asym(radii)
    vals = filter(!isnan, collect(values(radii)))
    isempty(vals) && return NaN
    return (maximum(vals) - minimum(vals)) / mean(vals)
end

# ----------------------------------------------------------------------
# Build a YelmoModel from the IC at t = 0 and overwrite the prognostic
# fields from the Exp1 restart. Time is reset to 0 so phase-2 sin runs
# naturally; the calvmip_exp2! law only uses time modulo 1000 yr.
# ----------------------------------------------------------------------

function _build_from_exp1_restart()
    isfile(EXP1_RESTART) || error(
        "Missing Exp1 restart at $EXP1_RESTART. Run\n" *
        "  julia --project=. ../calvingmip-exp1/run.jl\n" *
        "first.")

    b = CalvingMIPBenchmark(:exp2; dx_km = DX_KM)
    p = YelmoModelParameters(NAMELIST_PATH, "calvingmip_exp2")
    y = YelmoModel(b, 0.0; p = p, boundaries = :bounded)

    NCDataset(EXP1_RESTART, "r") do ds
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

    # Attach exp2 calving hook (captures b's xc/yc).
    xc = b.xc; yc = b.yc
    y.hooks.calv_flt = (cx, cy, ux, uy, Hi, fi, lsf, t) ->
        calvmip_exp2!(cx, cy, ux, uy, Hi, fi, lsf, t; xc = xc, yc = yc)

    init_state!(y, 0.0; thrm_method = "robin")
    return y, b
end

# ----------------------------------------------------------------------
# Restart writer (mirrors calvingmip-exp1).
# ----------------------------------------------------------------------

function _write_restart_nc(y, b, path; reason::AbstractString = "")
    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        Nx = length(b.xc); Ny = length(b.yc)
        defDim(ds, "xc", Nx); defDim(ds, "yc", Ny)
        defVar(ds, "xc", Float64, ("xc",))[:] = b.xc
        defVar(ds, "yc", Float64, ("yc",))[:] = b.yc

        for (name, src) in (
            ("H_ice",    interior(y.tpo.H_ice)[:, :, 1]),
            ("lsf",      interior(y.tpo.lsf)[:, :, 1]),
            ("ux_bar",   interior(y.dyn.ux_bar)[:, :, 1]),
            ("uy_bar",   interior(y.dyn.uy_bar)[:, :, 1]),
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

        ds.attrib["benchmark"] = "calvingmip-exp2"
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

        ds.attrib["title"]         = "CalvingMIP Exp2 time series + asymmetry metric"
        ds.attrib["benchmark"]     = "calvingmip-exp2"
        ds.attrib["dt_outer_yr"]   = DT_OUTER_YR
        ds.attrib["sample_dt_yr"]  = SAMPLE_DT_YR
        ds.attrib["asym_threshold"] = ASYM_THRESHOLD

        for (name, units) in (
            ("time",   "yr"),
            ("max_H",  "m"),
            ("vol",    "m^3"),
            ("asym",   "1"),
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
    @info "calvingmip-exp2 — t_max=$(T_PHASE2_MAX_YR) yr, dt_outer=$(DT_OUTER_YR), threshold=$(ASYM_THRESHOLD)"

    y, b = _build_from_exp1_restart()
    Nx = length(b.xc); Ny = length(b.yc)
    xc_km = b.xc ./ 1e3; yc_km = b.yc ./ 1e3

    # Use the cell-centre indices closest to the origin (cell-centred
    # grid: with even Nx the origin sits between two cells).
    imid = argmin(abs.(b.xc))
    jmid = argmin(abs.(b.yc))

    function _take_sample()
        H = interior(y.tpo.H_ice)[:, :, 1]
        L = interior(y.tpo.lsf)
        cell_area = (b.dx_km * 1e3)^2
        r = _front_radii_8(L, xc_km, yc_km, imid, jmid)
        return (
            time  = y.time,
            max_H = maximum(H),
            vol   = sum(H) * cell_area,
            asym  = _asym(r),
            r     = r,
        )
    end

    samples = NamedTuple[]
    push!(samples, _take_sample())
    s0 = samples[1]
    @printf("  t = %5.0f  asym = %5.2f%%\n", s0.time, 100 * s0.asym)

    n_max          = Int(round(T_PHASE2_MAX_YR / DT_OUTER_YR))
    sample_every   = max(Int(round(SAMPLE_DT_YR / DT_OUTER_YR)), 1)
    threshold_path = nothing

    for k in 1:n_max
        step!(y, DT_OUTER_YR)
        if k % sample_every == 0
            s = _take_sample()
            push!(samples, s)
            @printf("  t = %5.0f  asym = %5.2f%%  vol = %.3e m^3\n",
                    s.time, 100 * s.asym, s.vol)

            if !isnan(s.asym) && s.asym > ASYM_THRESHOLD
                threshold_path = _write_restart_nc(y, b, RESTART_NC;
                    reason = @sprintf("asym=%.3f exceeds %.3f", s.asym, ASYM_THRESHOLD))
                @info "Asymmetry threshold crossed — stopping" t = s.time asym = s.asym
                break
            end
        end
    end

    _write_timeseries_nc(samples, TIMESERIES_NC)
    @info "wrote outputs" timeseries = TIMESERIES_NC restart = threshold_path

    return samples, threshold_path
end

main()
