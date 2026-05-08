# ----------------------------------------------------------------------
# CalvingMIP — YelmoMirror fixture-generation glue.
#
# The benchmark spec (`CalvingMIPBenchmark`), bed geometry, analytical
# IC, and calving-law hooks (`calvmip_exp1!`, `calvmip_exp2!`) live in
# the standalone `IceSheetBenchmarks` package under `benchmarks/`. This
# file only adds the test-suite-side glue:
#
#   - a YelmoMirror initial-state callback;
#   - a `state(b, t > 0)` method that reads the committed Mirror fixture;
#   - a `write_fixture!(b, …)` extension that drives YelmoMirror so
#     `regenerate.jl` can refresh the committed fixture under
#     `test/benchmarks/fixtures/`.
#
# References:
#   - CalvingMIP wiki: https://github.com/JRowanJordan/CalvingMIP/wiki
#   - Fortran driver:  yelmo/tests/yelmo_calving.f90
#   - Calving laws:    yelmo/src/physics/calving/calving_ac.f90
# ----------------------------------------------------------------------

using IceSheetBenchmarks: IceSheetBenchmarks, CalvingMIPBenchmark,
                           calvmip_bed_circular, calvmip_bed_thule,
                           calvmip_exp1!, calvmip_exp2!,
                           _calvingmip_analytical_state
using Oceananigans.Fields: interior

# Re-export so existing tests (e.g. `test_calvingmip_exp1.jl`) that do
# `using .YelmoBenchmarks` still see these names.
export CalvingMIPBenchmark, calvmip_bed_circular, calvmip_bed_thule
export calvmip_exp1!, calvmip_exp2!

# -----------------------------------------------------------------------
# Per-experiment Fortran namelists (Mirror-only; ignored by the in-
# memory Yelmo.jl path).
# -----------------------------------------------------------------------

const _CALVINGMIP_NAMELIST_DIR = abspath(joinpath(@__DIR__, "specs"))
_calvingmip_namelist_path(exp::Symbol) = joinpath(
    _CALVINGMIP_NAMELIST_DIR,
    exp === :exp1 ? "yelmo_CalvingMIP_exp1.nml" :
    exp === :exp2 ? "yelmo_CalvingMIP_exp2.nml" :
    error("calvingmip: unsupported exp = $exp."))

calvingmip_namelist_path(b::CalvingMIPBenchmark) = _calvingmip_namelist_path(b.exp)

# -----------------------------------------------------------------------
# Spec name + fixture path (used by regenerate.jl)
# -----------------------------------------------------------------------

_spec_name(b::CalvingMIPBenchmark) = "calvingmip_$(lowercase(string(b.exp)))"

const _CALVINGMIP_FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

function _calvingmip_fixture_path(b::CalvingMIPBenchmark, t::Real;
                                   fixtures_dir = _CALVINGMIP_FIXTURES_DIR)
    return joinpath(fixtures_dir,
                    "$(_spec_name(b))_t$(Int(round(Float64(t)))).nc")
end

# -----------------------------------------------------------------------
# state(b, t > 0): read the YelmoMirror fixture (same convention as
# other test-side benchmarks). The t = 0 path is delegated to the
# IceSheetBenchmarks `state(b, 0.0)` analytical IC.
# -----------------------------------------------------------------------

function state(b::CalvingMIPBenchmark, t::Real)
    t_f = Float64(t)
    t_f == 0.0 && return _calvingmip_analytical_state(b)
    path = _calvingmip_fixture_path(b, t_f)
    isfile(path) || error(
        "CalvingMIPBenchmark.state: fixture missing at $path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl " *
        "$(_spec_name(b)) --overwrite` first.")
    NCDataset(path, "r") do ds
        out = Dict{Symbol,Any}(:xc => b.xc, :yc => b.yc)
        for name in ("H_ice", "z_bed", "f_grnd", "smb_ref", "T_srf", "Q_geo")
            haskey(ds, name) || continue
            raw = ds[name][:, :, :]
            arr2d = ndims(raw) == 3 ? raw[:, :, 1] : raw
            out[Symbol(name)] = Array{Float64}(arr2d)
        end
        return NamedTuple(out)
    end
end

# -----------------------------------------------------------------------
# YelmoMirror initial-state callback
# -----------------------------------------------------------------------

function _setup_calvingmip_initial_state!(ymirror, b::CalvingMIPBenchmark,
                                          time::Real)
    Nx = length(b.xc); Ny = length(b.yc)

    # Bed geometry from the analytical formula.
    z_bed = [calvmip_bed_circular(b.xc[i], b.yc[j]) for i in 1:Nx, j in 1:Ny]
    _assign_field!(ymirror.bnd.z_bed, z_bed)

    # No ice at the start.
    fill!(interior(ymirror.tpo.H_ice), 0.0)

    # lsf = +1 (all ocean). The Fortran LSFinit sets lsf=+1 where H=0,
    # so this matches; calving_step!'s above-SL pin will force lsf=−1
    # over land each step.
    fill!(interior(ymirror.tpo.lsf), 1.0)

    # Constant boundary forcing.
    fill!(interior(ymirror.bnd.z_sl),     0.0)
    fill!(interior(ymirror.bnd.bmb_shlf), 0.0)
    fill!(interior(ymirror.bnd.H_sed),    0.0)
    fill!(interior(ymirror.bnd.T_shlf),   b.T_srf_const)
    fill!(interior(ymirror.bnd.T_srf),    b.T_srf_const)
    fill!(interior(ymirror.bnd.smb_ref),  b.smb_const)
    fill!(interior(ymirror.bnd.Q_geo),    b.Q_geo_const)

    # Sync Julia fields → Fortran before init_state!.
    yelmo_sync!(ymirror)

    return ymirror
end

# -----------------------------------------------------------------------
# YelmoMirror fixture writer — extends `IceSheetBenchmarks.write_fixture!`
# so `regenerate.jl` can call it uniformly.
# -----------------------------------------------------------------------

function write_fixture!(b::CalvingMIPBenchmark,
                         path::AbstractString;
                         times::AbstractVector{<:Real} = [0.0])
    length(times) == 1 ||
        error("write_fixture!(CalvingMIPBenchmark): pass one time per call.")
    t = Float64(first(times))
    if t == 0.0
        # No CalvingMIP t = 0 fixture file is generated — the in-memory
        # `state(b, 0.0)` IC is constructed on the fly. (Mirror always
        # advances at least one step from t = 0 too.)
        error("write_fixture!(CalvingMIPBenchmark): t = 0 has no fixture " *
              "file (analytical IC is computed on-the-fly). Pass t > 0.")
    end
    return _write_calvingmip_mirror_fixture!(b, path, t)
end

function _write_calvingmip_mirror_fixture!(b::CalvingMIPBenchmark,
                                            path::AbstractString,
                                            t_out::Float64)
    namelist_path = calvingmip_namelist_path(b)
    isfile(namelist_path) || error(
        "CalvingMIPBenchmark.write_fixture!: namelist not found at " *
        "$namelist_path.")

    spec = BenchmarkSpec(
        name          = _spec_name(b),
        namelist_path = namelist_path,
        grid          = (xc = b.xc, yc = b.yc, grid_name = "CALVINGMIP"),
        time_init     = 0.0,
        end_time      = t_out,
        output_times  = [t_out],
        dt            = 1.0,
        setup_initial_state! = (ymirror, t) ->
            _setup_calvingmip_initial_state!(ymirror, b, t),
    )

    fixtures_dir = dirname(path)
    mkpath(fixtures_dir)
    isfile(path) && rm(path)

    paths = run_mirror_benchmark!(spec; fixtures_dir = fixtures_dir,
                                   overwrite = true)
    src = paths[1]
    if src != path
        mv(src, path; force = true)
        paths = [path]
    end
    return paths
end
