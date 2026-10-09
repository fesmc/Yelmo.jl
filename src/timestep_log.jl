# ----------------------------------------------------------------------
# Per-step timestep log (NetCDF), enabled by `y.p.yelmo.log_timestep`.
#
# Port of Fortran `yelmo_timestep_write_init` / `yelmo_timestep_write`
# (yelmo_timesteps.f90): same dimensions (`pt`, `xc`/`yc` in km,
# unlimited `time`), `pc_eps` on `pt`, and one row per accepted step
# with the variables of `TIMESTEP_LOG_VARS`. A first row at the start
# time holds the controller state (`dt_pi = pc_dt[1]`, `pc_eta[1]`).
#
# Rows are buffered in memory and written to `<rundir>/yelmo_timesteps.nc`
# by `close(log)` (one NetCDF access per row would cost ~28 ms each).
# The first `close` of a log creates the file (replacing one of an earlier
# run, as Fortran does at init); later calls append.
# ----------------------------------------------------------------------

using NCDatasets
using Oceananigans.Grids: xnodes, ynodes, Center

export TimestepLog, init_timestep_log!, write_timestep_row!

# (name, type, units, long_name), in the order of the Fortran writer.
const TIMESTEP_LOG_VARS = (
    (:speed,        Float64, "kyr/hr", "Yelmo model speed"),
    (:speed_tpo,    Float64, "kyr/hr", "Yelmo topo speed"),
    (:speed_dyn,    Float64, "kyr/hr", "Yelmo dyn speed"),
    (:dt_now,       Float64, "yr",     "Timestep"),
    (:dt_adv,       Float64, "yr",     "Timestep (CFL criterion)"),
    (:dt_pi,        Float64, "yr",     "Timestep (PI controller)"),
    (:pc_eta,       Float64, "1/yr",   "eta (pc error norm: RMS of pc_tau/(1 m + 0.01 H))"),
    (:ssa_iter,     Int64,   "",       "Picard iterations for SSA convergence"),
    (:iter_redo,    Int64,   "",       "Number of redo iterations needed"),
    (:ssa_lin_iter, Int64,   "",       "Linear solver iterations of the SSA solve (summed over Picard iterations)"),
    (:ssa_lin_fail, Int64,   "",       "SSA linear solves at breakdown or the iteration limit"),
    (:ssa_lim_n,    Int64,   "",       "SSA faces at the velocity limit (drag active or clipped)"),
    (:adv_lin_iter, Int64,   "",       "Linear solver iterations of the thickness advection (predictor + corrector)"),
    (:adv_lin_fail, Int64,   "",       "Advection linear solves at breakdown or the iteration limit"),
)

const _TIMESTEP_LOG_NAMES = map(first, TIMESTEP_LOG_VARS)

mutable struct TimestepLog
    path::String
    pc_eps::Float64
    xc::Vector{Float64}          # [m]
    yc::Vector{Float64}          # [m]
    time::Vector{Float64}
    cols::NamedTuple             # one buffer per `TIMESTEP_LOG_VARS` entry
    created::Bool                # file written by this log
end

"""
    init_timestep_log!(y; filename = "yelmo_timesteps.nc") -> TimestepLog

Empty log buffer for `<y.rundir>/filename`; the file is written by
`close(log)`.
"""
function init_timestep_log!(y; filename::String = "yelmo_timesteps.nc")
    rundir = isempty(y.rundir) ? "." : y.rundir
    isdir(rundir) || mkpath(rundir)
    cols = NamedTuple{_TIMESTEP_LOG_NAMES}(map(v -> v[2][], TIMESTEP_LOG_VARS))
    return TimestepLog(joinpath(rundir, filename), Float64(y.p.yelmo.pc_eps),
                       collect(Float64, xnodes(y.g, Center())),
                       collect(Float64, ynodes(y.g, Center())),
                       Float64[], cols, false)
end

"""
    write_timestep_row!(log, time; kwargs...) -> log

Buffer one row at `time`; the keywords are the variables of
`TIMESTEP_LOG_VARS`, all required.
"""
function write_timestep_row!(log::TimestepLog, time::Real; kwargs...)
    Set(keys(kwargs)) == Set(_TIMESTEP_LOG_NAMES) ||
        error("write_timestep_row!: expected the keywords $(_TIMESTEP_LOG_NAMES), got $(keys(kwargs)).")
    push!(log.time, Float64(time))
    for (name, T, _, _) in TIMESTEP_LOG_VARS
        push!(log.cols[name], T(kwargs[name]))
    end
    return log
end

# Write the buffered rows to NetCDF (creating the file on the first call)
# and empty the buffers.
function Base.close(log::TimestepLog)
    n = length(log.time)
    n == 0 && return log
    log.created || (_create_timestep_file(log); log.created = true)
    NCDataset(log.path, "a") do ds
        n0 = length(ds["time"])
        r = (n0 + 1):(n0 + n)
        ds["time"][r] = log.time
        for name in _TIMESTEP_LOG_NAMES
            ds[String(name)][r] = log.cols[name]
        end
    end
    empty!(log.time)
    foreach(empty!, log.cols)
    return log
end

function _create_timestep_file(log::TimestepLog)
    NCDataset(log.path, "c") do ds
        defDim(ds, "pt", 1)
        defDim(ds, "xc", length(log.xc))
        defDim(ds, "yc", length(log.yc))
        defDim(ds, "time", Inf)
        v = defVar(ds, "pt", Float64, ("pt",)); v[:] = [1.0]; v.attrib["units"] = "point"
        for (name, x, axis) in (("xc", log.xc, "GeoX"), ("yc", log.yc, "GeoY"))
            v = defVar(ds, name, Float64, (name,))
            v[:] = x .* 1e-3
            v.attrib["units"] = "kilometers"
            v.attrib["_CoordinateAxisType"] = axis
        end
        v = defVar(ds, "time", Float64, ("time",)); v.attrib["units"] = "years"
        v = defVar(ds, "pc_eps", Float64, ("pt",)); v[:] = [log.pc_eps]
        for (name, T, units, long_name) in TIMESTEP_LOG_VARS
            v = defVar(ds, String(name), T, ("time",))
            v.attrib["units"] = units
            v.attrib["long_name"] = long_name
        end
    end
    return nothing
end
