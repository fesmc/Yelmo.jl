# ----------------------------------------------------------------------
# EISMINT1MovingBenchmark — Yelmo-side glue.
#
# The spec struct (`EISMINT1MovingBenchmark`), the radial SMB helper
# (`eismint_moving_smb`), the analytical zero-ice IC (`state(b, 0)`),
# and the analytical `write_fixture!(b, [0])` live in IceSheetBenchmarks.
# This file adds:
#
#   - `_spec_name` / `_eismint_moving_fixture_path` — fixture naming.
#   - `state(b, t > 0)` — read the committed YelmoMirror fixture.
#   - `write_fixture!(b, path; times=[t > 0])` — drive YelmoMirror to
#     `t` and write the reference restart.
#   - `_setup_eismint_moving_initial_state!` — YelmoMirror IC callback.
# ----------------------------------------------------------------------

using IceSheetBenchmarks: IceSheetBenchmarks, EISMINT1MovingBenchmark,
                           eismint_moving_smb,
                           _eismint_moving_analytical_state,
                           _write_eismint_moving_analytical_fixture!

# Re-export so existing tests using `.YelmoBenchmarkHarness` still see the name.
export EISMINT1MovingBenchmark, eismint_moving_smb

const _DEFAULT_EISMINT_MOVING_NAMELIST = abspath(joinpath(@__DIR__, "..", "specs",
                                                            "yelmo_EISMINT_moving.nml"))
_eismint_moving_namelist_path(::EISMINT1MovingBenchmark) =
    _DEFAULT_EISMINT_MOVING_NAMELIST

_spec_name(::EISMINT1MovingBenchmark) = "eismint_moving"

const _EISMINT_MOVING_FIXTURES_DIR = abspath(joinpath(@__DIR__, "..", "fixtures"))

function _eismint_moving_fixture_path(b::EISMINT1MovingBenchmark, t::Real;
                                       fixtures_dir::AbstractString = _EISMINT_MOVING_FIXTURES_DIR)
    return joinpath(fixtures_dir, "$(_spec_name(b))_t$(Int(round(Float64(t)))).nc")
end

"""
    state(b::EISMINT1MovingBenchmark, t::Real) -> NamedTuple

For `t = 0`, delegates to the IceSheetBenchmarks analytical zero-ice IC.
For `t > 0`, reads the committed YelmoMirror reference fixture at
`_eismint_moving_fixture_path(b, t)`.

Errors with a hint to run `regenerate.jl` if the fixture is missing.
"""
function state(b::EISMINT1MovingBenchmark, t::Real)
    t_f = Float64(t)
    t_f == 0.0 && return _eismint_moving_analytical_state(b)

    path = _eismint_moving_fixture_path(b, t_f)
    isfile(path) || error(
        "EISMINT1MovingBenchmark.state: fixture missing at $path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl " *
        "$(_spec_name(b)) --overwrite` first.")
    NCDataset(path, "r") do ds
        out = Dict{Symbol,Any}(:xc => b.xc, :yc => b.yc)
        for name in ("H_ice", "z_bed", "smb_ref", "T_srf", "Q_geo")
            if haskey(ds, name)
                raw  = ds[name][:, :, :]
                arr2 = ndims(raw) == 3 ? raw[:, :, 1] : raw
                out[Symbol(name)] = Array{Float64}(arr2)
            end
        end
        return NamedTuple(out)
    end
end

"""
    write_fixture!(b::EISMINT1MovingBenchmark, path; times=[0.0]) -> Vector{String}

For `t = 0`, delegates to the IceSheetBenchmarks analytical-IC writer.
For `t > 0`, drives YelmoMirror from the EISMINT-moving IC + namelist
to `t`, writes the restart to `path`, and appends performance metadata
as NetCDF attributes so the lockstep test can read Mirror wall-clock
without rerunning Fortran.

Single-time only — pass one time per call.
"""
function write_fixture!(b::EISMINT1MovingBenchmark, path::AbstractString;
                        times::AbstractVector{<:Real} = [0.0])
    length(times) == 1 ||
        error("write_fixture!(EISMINT1MovingBenchmark, …): pass `times` " *
              "with exactly one entry per call (got $(length(times))).")
    t = Float64(first(times))
    if t == 0.0
        _write_eismint_moving_analytical_fixture!(b, path, t)
        return [path]
    end
    return _write_eismint_moving_mirror_fixture!(b, path, t)
end

# YelmoMirror fixture writer: drive Fortran from the EISMINT-moving
# IC + namelist to `t_out` (e.g. 25_000 yr), write the restart to
# `path`, then append performance metadata as NetCDF attributes so the
# lockstep test can compare wall-clock against the Yelmo.jl run.
function _write_eismint_moving_mirror_fixture!(b::EISMINT1MovingBenchmark,
                                                 path::AbstractString,
                                                 t_out::Float64)
    namelist_path = _eismint_moving_namelist_path(b)
    isfile(namelist_path) || error(
        "EISMINT1MovingBenchmark.write_fixture!: namelist not found at " *
        "$(namelist_path).")

    spec = BenchmarkSpec(
        name           = _spec_name(b),
        namelist_path  = namelist_path,
        grid           = (xc = b.xc, yc = b.yc, grid_name = "EISMINT"),
        time_init      = 0.0,
        end_time       = t_out,
        output_times   = [t_out],
        # Fortran outer-loop dt; matches `&ctrl dtt = 100.0` in the
        # spec namelist. The internal adaptive PC sub-steps inside.
        dt             = 100.0,
        setup_initial_state! = (ymirror, t) ->
            _setup_eismint_moving_initial_state!(ymirror, b, t),
    )

    fixtures_dir = dirname(path)
    mkpath(fixtures_dir)
    isfile(path) && rm(path)

    wall_clock_start = time()
    paths = generate_fixture!(spec; fixtures_dir = fixtures_dir,
                                  overwrite = true)
    wall_clock_s = time() - wall_clock_start

    src = paths[1]
    if src != path
        mv(src, path; force = true)
        paths = [path]
    end

    cpu_info = try
        first(Sys.cpu_info()).model
    catch
        "unknown"
    end
    NCDataset(path, "a") do ds
        ds.attrib["mirror_wallclock_seconds"] = wall_clock_s
        ds.attrib["mirror_t_init"]            = 0.0
        ds.attrib["mirror_t_end"]             = t_out
        ds.attrib["mirror_dt_outer_yr"]       = spec.dt
        ds.attrib["mirror_n_outer_steps"]     = Int(round(t_out / spec.dt))
        ds.attrib["mirror_julia_version"]     = string(VERSION)
        ds.attrib["mirror_cpu_model"]         = cpu_info
        ds.attrib["mirror_n_julia_threads"]   = Threads.nthreads()
    end

    @info "EISMINT-moving Mirror fixture written" path t_out_yr=t_out wall_clock_s

    return paths
end

# YelmoMirror initial-state callback for EISMINT-moving.
function _setup_eismint_moving_initial_state!(ymirror, b::EISMINT1MovingBenchmark,
                                                time::Real)
    Nx, Ny = length(b.xc), length(b.yc)

    fill!(interior(ymirror.tpo.H_ice),     0.0)
    fill!(interior(ymirror.bnd.z_bed),     0.0)
    fill!(interior(ymirror.bnd.z_sl),      0.0)
    fill!(interior(ymirror.bnd.bmb_shlf),  0.0)
    fill!(interior(ymirror.bnd.H_sed),     0.0)
    fill!(interior(ymirror.bnd.T_shlf),    b.T_srf_const)
    fill!(interior(ymirror.bnd.T_srf),     b.T_srf_const)
    fill!(interior(ymirror.bnd.Q_geo),     b.Q_geo_const)

    smb_int = interior(ymirror.bnd.smb_ref)
    @inbounds for j in 1:Ny, i in 1:Nx
        smb_int[i, j, 1] = eismint_moving_smb(b, b.xc[i], b.yc[j])
    end

    fill!(interior(ymirror.bnd.calv_mask), 0.0)

    yelmo_sync!(ymirror)
    return ymirror
end
