# ----------------------------------------------------------------------
# Per-PC-step timestep log (NetCDF), enabled by `y.p.yelmo.log_timestep`.
#
# Mirrors Fortran Yelmo's `yelmo_timestep_write` family at
# `yelmo_timesteps.f90:1214-1308`. Records one row per accepted PC
# attempt with: `time`, `dt_now`, `pc_eta`, `ssa_iter`, `iter_redo`,
# and per-step `wallclock_s`. Used by the comparison-vs-Fortran
# benchmarks to dump the controller's dt + eta history alongside the
# Mirror reference.
#
# **Buffered writes**: each `write_timestep_row!` push only appends
# to in-memory vectors (essentially free). The NetCDF file is created
# and populated in one go by `Base.close(log)` — call this when the
# benchmark is done. Per-row write to NetCDF (the previous design)
# costs ~28 ms apiece (open / nc_put_vara / close per access), which
# dominated the wallclock for adaptive runs with many sub-steps. With
# buffering, log overhead drops to negligible.
#
# File layout: `<y.rundir>/yelmo_timesteps.nc`. Created on the first
# `close(log)` call.
#
# This is a Yelmo.jl port of the Fortran scaffolding only — Yelmo.jl
# does not yet plumb `dt_adv` (CFL-based limit) or `dt_pi` (PI's
# pre-clamp recommendation) separately, so those columns are omitted.
# Add later if the comparison needs them.
# ----------------------------------------------------------------------

using NCDatasets

export TimestepLog, init_timestep_log!, write_timestep_row!

mutable struct TimestepLog
    path::String
    pc_eps::Float64
    times::Vector{Float64}
    dt_now::Vector{Float64}
    pc_eta::Vector{Float64}
    ssa_iter::Vector{Int64}
    iter_redo::Vector{Int64}
    wallclock_s::Vector{Float64}
end

# Lazily initialise the in-memory log buffer. The NetCDF file is NOT
# created here — that happens at `close(log)` time. Returns the
# `TimestepLog` handle. Caller is expected to cache it on
# `y.dyn.scratch.timestep_log[]`.
function init_timestep_log!(y; filename::String = "yelmo_timesteps.nc")
    rundir = y.rundir
    isempty(rundir) && (rundir = ".")
    isdir(rundir) || mkpath(rundir)
    path = joinpath(rundir, filename)
    pc_eps = y.p === nothing ? 0.0 : Float64(y.p.yelmo.pc_eps)
    return TimestepLog(path, pc_eps,
                       Float64[], Float64[], Float64[],
                       Int64[],   Int64[],   Float64[])
end

# Append one accepted-PC-step row to the in-memory buffer.
function write_timestep_row!(log::TimestepLog, y;
                             dt_now::Real, eta::Real,
                             iter_redo::Integer, wallclock_s::Real)
    ssa_iter = try
        Int(y.dyn.scratch.ssa_iter_now[])
    catch
        0
    end
    push!(log.times,       Float64(y.time))
    push!(log.dt_now,      Float64(dt_now))
    push!(log.pc_eta,      Float64(eta))
    push!(log.ssa_iter,    Int64(ssa_iter))
    push!(log.iter_redo,   Int64(iter_redo))
    push!(log.wallclock_s, Float64(wallclock_s))
    return log
end

# Flush the buffered rows to NetCDF and clear the buffer. Safe to
# call multiple times — subsequent calls append to the existing file
# and clear the buffer between calls (so a long-running session can
# checkpoint partial logs without losing data).
function Base.close(log::TimestepLog)
    n = length(log.times)
    n == 0 && return log

    if isfile(log.path)
        # Append to existing file: open, find current length, extend.
        NCDataset(log.path, "a") do ds
            tv = ds["time"]
            n_existing = length(tv)
            range = (n_existing + 1):(n_existing + n)
            tv[range]              = log.times
            ds["dt_now"][range]    = log.dt_now
            ds["pc_eta"][range]    = log.pc_eta
            ds["ssa_iter"][range]  = log.ssa_iter
            ds["iter_redo"][range] = log.iter_redo
            ds["wallclock_s"][range] = log.wallclock_s
        end
    else
        NCDataset(log.path, "c") do ds
            defDim(ds, "time", Inf)
            ds.attrib["pc_eps"] = log.pc_eps
            # NCDatasets won't grow an unlimited dim of current size 0
            # via `tv[:] = data`; index explicitly so the dim extends.
            r = 1:n

            tv = defVar(ds, "time", Float64, ("time",))
            tv.attrib["units"]     = "yr"
            tv.attrib["long_name"] = "model time at end of accepted PC step"
            tv[r] = log.times

            dv = defVar(ds, "dt_now", Float64, ("time",))
            dv.attrib["units"]     = "yr"
            dv.attrib["long_name"] = "PC sub-step dt taken"
            dv[r] = log.dt_now

            ev = defVar(ds, "pc_eta", Float64, ("time",))
            ev.attrib["units"]     = "m yr^-1"
            ev.attrib["long_name"] = "PC truncation-error proxy (eta)"
            ev[r] = log.pc_eta

            sv = defVar(ds, "ssa_iter", Int64, ("time",))
            sv.attrib["units"]     = "1"
            sv.attrib["long_name"] = "Picard iterations the SSA / DIVA solver took on this step"
            sv[r] = log.ssa_iter

            rv = defVar(ds, "iter_redo", Int64, ("time",))
            rv.attrib["units"]     = "1"
            rv.attrib["long_name"] = "PC retry count (1 = accepted on first try)"
            rv[r] = log.iter_redo

            wv = defVar(ds, "wallclock_s", Float64, ("time",))
            wv.attrib["units"]     = "s"
            wv.attrib["long_name"] = "wall-clock seconds for this PC sub-step"
            wv[r] = log.wallclock_s
        end
    end

    # Clear the buffer so subsequent close() calls don't double-write.
    empty!(log.times); empty!(log.dt_now); empty!(log.pc_eta)
    empty!(log.ssa_iter); empty!(log.iter_redo); empty!(log.wallclock_s)
    return log
end
