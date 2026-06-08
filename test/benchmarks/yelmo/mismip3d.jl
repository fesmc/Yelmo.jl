# ----------------------------------------------------------------------
# MISMIP3DBenchmark — Yelmo-side glue.
#
# The spec struct (`MISMIP3DBenchmark`) and the analytical Stnd IC
# (`state(b, 0)`, `write_fixture!(b, [0])`) live in IceSheetBenchmarks.
# This file adds:
#
#   - `_spec_name` / `_mismip3d_fixture_path` — fixture naming.
#   - `state(b, t > 0)` — read the committed YelmoMirror fixture.
#   - `write_fixture!(b, path; times=[t > 0])` — drive YelmoMirror to
#     `t` and write the reference restart.
#   - `_setup_mismip3d_initial_state!` — YelmoMirror IC callback (the
#     thicker grounded-IC override; see commentary inside).
#   - `_write_mismip3d_mirror_fixture_att!` — multi-phase ATT-ramp
#     fixture writer used by the MISMIP3D-ATT test.
# ----------------------------------------------------------------------

using IceSheetBenchmarks: IceSheetBenchmarks, MISMIP3DBenchmark,
                           _mismip3d_analytical_state,
                           _write_mismip3d_analytical_fixture!

# Re-export so existing tests using `.YelmoBenchmarkHarness` still see the name.
export MISMIP3DBenchmark

const _DEFAULT_MISMIP3D_NAMELIST = abspath(joinpath(@__DIR__, "specs",
                                                     "yelmo_MISMIP3D.nml"))
_mismip3d_namelist_path(::MISMIP3DBenchmark) = _DEFAULT_MISMIP3D_NAMELIST

_spec_name(b::MISMIP3DBenchmark) = "mismip3d_$(lowercase(string(b.variant)))"

const _MISMIP3D_FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

function _mismip3d_fixture_path(b::MISMIP3DBenchmark, t::Real;
                                fixtures_dir::AbstractString = _MISMIP3D_FIXTURES_DIR)
    return joinpath(fixtures_dir,
                    "$(_spec_name(b))_t$(Int(round(Float64(t)))).nc")
end

"""
    state(b::MISMIP3DBenchmark, t::Real) -> NamedTuple

For `t = 0`, delegates to the IceSheetBenchmarks analytical Stnd IC.
For `t > 0`, reads the committed YelmoMirror reference fixture at
`_mismip3d_fixture_path(b, t)`.

See the `TroughBenchmark.state` docstring for the face-staggered-field
exclusion rationale.

Errors with a hint to run `regenerate.jl` if the fixture is missing.
"""
function state(b::MISMIP3DBenchmark, t::Real)
    t_f = Float64(t)
    t_f == 0.0 && return _mismip3d_analytical_state(b)

    path = _mismip3d_fixture_path(b, t_f)
    isfile(path) || error(
        "MISMIP3DBenchmark.state: fixture missing at $path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl " *
        "$(_spec_name(b)) --overwrite` first.")

    NCDataset(path, "r") do ds
        out = Dict{Symbol,Any}(:xc => b.xc, :yc => b.yc)
        for name in ("H_ice", "z_bed", "f_grnd",
                     "smb_ref", "T_srf", "Q_geo")
            if haskey(ds, name)
                raw = ds[name][:, :, :]
                arr2d = ndims(raw) == 3 ? raw[:, :, 1] : raw
                out[Symbol(name)] = Array{Float64}(arr2d)
            end
        end
        return NamedTuple(out)
    end
end

"""
    write_fixture!(b::MISMIP3DBenchmark, path; times=[0.0]) -> Vector{String}

For `t = 0`, delegates to the IceSheetBenchmarks analytical-IC writer.
For `t > 0`, drives YelmoMirror from the analytical IC through
`_setup_mismip3d_initial_state!` + the namelist, integrates to `t`,
and writes the restart.

Single-time only — pass one time per call. `regenerate.jl` dispatches
`[0.0, 500.0]` via one call each.
"""
function write_fixture!(b::MISMIP3DBenchmark, path::AbstractString;
                        times::AbstractVector{<:Real} = [0.0])
    length(times) == 1 ||
        error("write_fixture!(MISMIP3DBenchmark, …): pass `times` with " *
              "exactly one entry per call (got $(length(times))). " *
              "regenerate.jl loops over [0.0, 500.0] one t at a time.")
    t = Float64(first(times))
    if t == 0.0
        _write_mismip3d_analytical_fixture!(b, path, t)
        return [path]
    end
    return _write_mismip3d_mirror_fixture!(b, path, t)
end

# ----------------------------------------------------------------------
# YelmoMirror initial-state callback for MISMIP3D Stnd.
#
# Mirrors the Yelmo.jl-side test override (test_mismip3d_stnd.jl
# lines 184-196): apply the thicker grounded-IC variant from the
# Fortran source's commented-out section in mismip3D.f90:62-64
#   `H_ice = max(0, 1000 - 0.9·z_bed)` for z_bed < 0
# so the SSA system is well-posed from step 1 (the literal Fortran
# 10 m all-floating IC is rank-deficient — see test header for the
# full discussion). z_bed below `b.z_bed_floor` stays ice-free.
#
# Also sets the constant Stnd boundary forcing (T_srf, smb_ref, Q_geo,
# z_sl, bmb_shlf, H_sed) plus `calv_mask[Nx, :] = 1` for the Fortran
# kill-pos calving of the eastern column.
# ----------------------------------------------------------------------

# `time` is unused (Stnd IC has no time dependence); kept for the
# `(ymirror, time)` callback signature `BenchmarkSpec.setup_initial_state!`
# expects.
function _setup_mismip3d_initial_state!(ymirror, b::MISMIP3DBenchmark, time::Real)
    Nx = length(b.xc)
    Ny = length(b.yc)

    z_bed = zeros(Nx, Ny)
    H_ice = zeros(Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        x_km = b.xc[i] / 1e3
        zb   = b.bed_intercept - b.bed_slope * x_km
        z_bed[i, j] = zb
        H_ice[i, j] = (zb < b.z_bed_floor) ? 0.0 :
                                              max(0.0, 1000.0 - 0.9 * zb)
    end

    _assign_field!(ymirror.bnd.z_bed, z_bed)
    _assign_field!(ymirror.tpo.H_ice, H_ice)

    fill!(interior(ymirror.bnd.z_sl),     0.0)
    fill!(interior(ymirror.bnd.bmb_shlf), 0.0)
    fill!(interior(ymirror.bnd.H_sed),    0.0)
    fill!(interior(ymirror.bnd.T_shlf),   b.T_srf_const)
    fill!(interior(ymirror.bnd.T_srf),    b.T_srf_const)
    fill!(interior(ymirror.bnd.smb_ref),  b.smb_const)
    fill!(interior(ymirror.bnd.Q_geo),    b.Q_geo_const)

    # Calving mask: True only on the eastern (i=Nx) column (Fortran's
    # kill-pos calv_mask). Stored as Float64 (0/1) in the YelmoMirror.
    calv_mask = zeros(Nx, Ny)
    calv_mask[Nx, :] .= 1.0
    _assign_field!(ymirror.bnd.calv_mask, calv_mask)

    yelmo_sync!(ymirror)

    return ymirror
end

# Single-phase YelmoMirror fixture writer.
function _write_mismip3d_mirror_fixture!(b::MISMIP3DBenchmark,
                                         path::AbstractString,
                                         t_out::Float64)
    namelist_path = _mismip3d_namelist_path(b)
    isfile(namelist_path) || error(
        "MISMIP3DBenchmark.write_fixture!: namelist not found at " *
        "$(namelist_path).")

    spec = BenchmarkSpec(
        name           = _spec_name(b),
        namelist_path  = namelist_path,
        grid           = (xc = b.xc, yc = b.yc,
                          grid_name = "MISMIP3D"),
        time_init      = 0.0,
        end_time       = t_out,
        output_times   = [t_out],
        dt             = 1.0,
        setup_initial_state! = (ymirror, t) ->
            _setup_mismip3d_initial_state!(ymirror, b, t),
    )

    fixtures_dir = dirname(path)
    mkpath(fixtures_dir)
    isfile(path) && rm(path)

    paths = generate_fixture!(spec; fixtures_dir = fixtures_dir,
                                  overwrite = true)
    src = paths[1]
    if src != path
        mv(src, path; force = true)
        paths = [path]
    end
    return paths
end

# ----------------------------------------------------------------------
# Multi-phase ATT-ramp fixture writer.
#
# Drives YelmoMirror through a sequence of phases, each with its own
# constant `rf_const` (Glen rate factor). The Fortran-side mechanism
# requires a one-shot namelist patch (`rf_method = -1`, "external rate
# factor") so Fortran's `calc_ymat` (yelmo_material.f90:201-211)
# does not overwrite `mat%now%ATT` from the scalar `mat%par%rf_const`
# on every step. With `rf_method = -1`, Fortran trusts whatever's in
# the `mat.ATT` field; we push the new uniform value through the C-API
# at each phase boundary.
#
# Outputs ONE NetCDF restart at `times_phase_end[end]`. Filename is the
# user-supplied `path`; intermediate phase boundaries are not snapshot.
# ----------------------------------------------------------------------

function _write_mismip3d_mirror_fixture_att!(b::MISMIP3DBenchmark,
                                             path::AbstractString,
                                             times_phase_end::AbstractVector{<:Real},
                                             rf_phases::AbstractVector{<:Real})
    @assert length(times_phase_end) == length(rf_phases) (
        "phase-count mismatch: times=$(length(times_phase_end)) " *
        "rf=$(length(rf_phases))")
    @assert !isempty(times_phase_end) "need at least one phase"
    @assert all(times_phase_end .> 0) "phase end times must be positive"
    @assert issorted(times_phase_end) "phase end times must be sorted"
    namelist_path = _mismip3d_namelist_path(b)
    isfile(namelist_path) || error(
        "_write_mismip3d_mirror_fixture_att!: namelist not found at " *
        "$(namelist_path).")

    # Patch the source namelist: switch `rf_method` to -1 (external).
    src = read(namelist_path, String)
    patched = replace(src,
                      r"rf_method\s*=\s*-?\d+" => "rf_method            = -1")
    if !occursin("rf_method            = -1", patched)
        error("_write_mismip3d_mirror_fixture_att!: failed to patch " *
              "rf_method in $(namelist_path) — check the namelist " *
              "still has an `rf_method = N` line.")
    end
    tmpdir = mktempdir(; prefix = "mismip3d_att_nml_")
    patched_nml = joinpath(tmpdir, "yelmo_MISMIP3D_att.nml")
    write(patched_nml, patched)

    spec = BenchmarkSpec(
        name           = _spec_name(b) * "_att",
        namelist_path  = patched_nml,
        grid           = (xc = b.xc, yc = b.yc, grid_name = "MISMIP3D"),
        time_init      = 0.0,
        end_time       = Float64(times_phase_end[end]),
        output_times   = [Float64(times_phase_end[end])],
        dt             = 1.0,
        setup_initial_state! = (ymirror, t) ->
            _setup_mismip3d_initial_state!(ymirror, b, t),
    )

    fixtures_dir = dirname(path)
    mkpath(fixtures_dir)
    isfile(path) && rm(path)

    p = Yelmo.YelmoMirrorPar.read_nml(spec.namelist_path)
    rundir = mktempdir(; prefix = "bench_$(spec.name)_")
    yelmo_input = abspath(joinpath(dirname(Yelmo.YelmoMirrorCore.yelmolib),
                                   "..", "..", "input"))
    isdir(yelmo_input) || error(
        "_write_mismip3d_mirror_fixture_att!: yelmo input dir not found " *
        "at $yelmo_input.")
    symlink(yelmo_input, joinpath(rundir, "input"))

    cwd_orig = pwd()
    cd(rundir)
    ymirror = try
        YelmoMirror(p, spec.time_init;
                    grid      = spec.grid,
                    alias     = "ylmo1",
                    rundir    = rundir,
                    overwrite = true)
    catch
        cd(cwd_orig)
        rethrow()
    end

    try
        spec.setup_initial_state!(ymirror, spec.time_init)

        # Pre-fill mat.ATT to Phase-0 rf BEFORE init_state! so the
        # rf_method=-1 path sees a valid value when Fortran's init
        # runs `calc_ymat`.
        fill!(interior(ymirror.mat.ATT), Float64(rf_phases[1]))
        Yelmo.YelmoMirrorCore._set_var!(ymirror.mat.ATT, ymirror.v.mat.ATT,
                                        ymirror.buffers, ymirror.calias)

        init_state!(ymirror, spec.time_init)

        # Re-push baseline ATT after init_state! in case it cleared
        # the field during initialisation.
        fill!(interior(ymirror.mat.ATT), Float64(rf_phases[1]))
        Yelmo.YelmoMirrorCore._set_var!(ymirror.mat.ATT, ymirror.v.mat.ATT,
                                        ymirror.buffers, ymirror.calias)

        t = spec.time_init
        for (phase_idx, t_target) in enumerate(times_phase_end)
            rf_val = Float64(rf_phases[phase_idx])

            fill!(interior(ymirror.mat.ATT), rf_val)
            Yelmo.YelmoMirrorCore._set_var!(ymirror.mat.ATT,
                                            ymirror.v.mat.ATT,
                                            ymirror.buffers,
                                            ymirror.calias)

            while t < t_target - 1e-9
                dt_step = min(spec.dt, t_target - t)
                step!(ymirror, dt_step)
                t = ymirror.time
            end
        end

        yelmo_write_restart!(ymirror, path; time = t)
    finally
        cd(cwd_orig)
    end

    return [path]
end
