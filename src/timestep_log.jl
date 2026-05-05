# ----------------------------------------------------------------------
# Per-PC-step timestep log (NetCDF), enabled by `y.p.yelmo.log_timestep`.
#
# Mirrors Fortran Yelmo's `yelmo_timestep_write` family at
# `yelmo_timesteps.f90:1214-1308`. Writes one row per accepted PC
# attempt with: `time`, `dt_now`, `pc_eta`, `ssa_iter`, `iter_redo`,
# and per-step `wallclock_s`. Used by the comparison-vs-Fortran
# benchmarks to dump the controller's dt + eta history alongside the
# Mirror reference.
#
# File layout: `<y.rundir>/yelmo_timesteps.nc`. Created lazily on the
# first PC step when `log_timestep = true`. The NetCDF dataset is
# kept open for the lifetime of the model (closed via `Base.close` /
# garbage collection).
#
# This is a Yelmo.jl port of the Fortran scaffolding only — Yelmo.jl
# does not yet plumb `dt_adv` (CFL-based limit) or `dt_pi` (PI's
# pre-clamp recommendation) separately, so those columns are omitted.
# Add them later if the comparison needs them.
# ----------------------------------------------------------------------

using NCDatasets

export TimestepLog, init_timestep_log!, write_timestep_row!

mutable struct TimestepLog
    ds::NCDataset
    path::String
    n::Int                  # next time-index to write (1-based)
end

# Lazily create the timestep-log NetCDF at `<rundir>/yelmo_timesteps.nc`.
# Returns the `TimestepLog` handle. Caller is expected to cache it on
# `y.dyn.scratch.timestep_log[]`.
function init_timestep_log!(y; filename::String = "yelmo_timesteps.nc")
    rundir = y.rundir
    isempty(rundir) && (rundir = ".")
    isdir(rundir) || mkpath(rundir)
    path = joinpath(rundir, filename)
    isfile(path) && rm(path)

    ds = NCDataset(path, "c")
    defDim(ds, "time", Inf)

    pc_eps = y.p === nothing ? 0.0 : Float64(y.p.yelmo.pc_eps)
    ds.attrib["pc_eps"] = pc_eps

    tv = defVar(ds, "time", Float64, ("time",))
    tv.attrib["units"]     = "yr"
    tv.attrib["long_name"] = "model time at end of accepted PC step"

    dv = defVar(ds, "dt_now", Float64, ("time",))
    dv.attrib["units"]     = "yr"
    dv.attrib["long_name"] = "PC sub-step dt taken"

    ev = defVar(ds, "pc_eta", Float64, ("time",))
    ev.attrib["units"]     = "m yr^-1"
    ev.attrib["long_name"] = "PC truncation-error proxy (eta)"

    sv = defVar(ds, "ssa_iter", Int64, ("time",))
    sv.attrib["units"]     = "1"
    sv.attrib["long_name"] = "Picard iterations the SSA / DIVA solver took on this step"

    rv = defVar(ds, "iter_redo", Int64, ("time",))
    rv.attrib["units"]     = "1"
    rv.attrib["long_name"] = "PC retry count (1 = accepted on first try)"

    wv = defVar(ds, "wallclock_s", Float64, ("time",))
    wv.attrib["units"]     = "s"
    wv.attrib["long_name"] = "wall-clock seconds for this PC sub-step"

    return TimestepLog(ds, path, 1)
end

# Append one accepted-PC-step row.
function write_timestep_row!(log::TimestepLog, y;
                             dt_now::Real, eta::Real,
                             iter_redo::Integer, wallclock_s::Real)
    ssa_iter = try
        Int(y.dyn.scratch.ssa_iter_now[])
    catch
        0
    end

    n = log.n
    log.ds["time"][n]        = Float64(y.time)
    log.ds["dt_now"][n]      = Float64(dt_now)
    log.ds["pc_eta"][n]      = Float64(eta)
    log.ds["ssa_iter"][n]    = Int64(ssa_iter)
    log.ds["iter_redo"][n]   = Int64(iter_redo)
    log.ds["wallclock_s"][n] = Float64(wallclock_s)
    log.n = n + 1
    return log
end

Base.close(log::TimestepLog) = close(log.ds)
