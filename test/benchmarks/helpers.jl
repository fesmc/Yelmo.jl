# ----------------------------------------------------------------------
# Benchmark scaffolding for YelmoModel validation.
#
# Workflow:
#   1. Define a `BenchmarkSpec` for a Fortran-internally-generated
#      benchmark (e.g. BUELER-B). The spec carries the Yelmo namelist
#      path, the synthetic-grid axes, the run length / output cadence,
#      and a `setup_initial_state!` callback that sets H_ice / boundary
#      fields after YelmoMirror is constructed but before init_state!.
#   2. `run_mirror_benchmark!(spec; fixtures_dir)` runs YelmoMirror on
#      the spec, writing a NetCDF restart at each `output_time` via
#      `yelmo_write_restart!` (a thin Julia wrapper around the
#      `yelmo_restart_write` C API). Fixture paths are returned.
#   3. `load_fixture(spec; index)` constructs a `YelmoModel` from the
#      fixture for downstream validation against the YelmoMirror
#      reference (or, for BUELER tests, against the analytical
#      solution).
#
# CI does NOT invoke `run_mirror_benchmark!` — fixtures are committed
# under `test/benchmarks/fixtures/`. The `regenerate.jl` script
# refreshes them locally and requires `libyelmo_c_api.so` to be present.
#
# Halfar / Bueler analytical solutions live in `bueler.jl` (included
# below), ported from `yelmo/tests/ice_benchmarks.f90`. The
# `AbstractBenchmark` interface contract (and the in-memory
# `YelmoModel(::AbstractBenchmark, t)` constructor) lives in
# `benchmarks.jl`.
# ----------------------------------------------------------------------

module YelmoBenchmarks

using Yelmo
using Oceananigans: interior

include("benchmarks.jl")
include("bueler.jl")
include("trough.jl")
include("hom_c.jl")
include("mismip3d.jl")

export BenchmarkSpec
export AbstractBenchmark, BuelerBenchmark, TroughBenchmark, HOMCBenchmark,
       MISMIP3DBenchmark
export run_mirror_benchmark!, load_fixture
export state, write_fixture!, analytical_velocity
export bueler_test_BC!, bueler_gamma
export _setup_hom_c_beta!

# Registry callback signature: `(ymirror, time)` → mutate `ymirror`
# fields and push back to Fortran. The spec author is responsible for
# calling `yelmo_sync!` (or per-field `_set_var!`s) at the end of the
# callback so the Fortran state matches the Julia-side mutation before
# `init_state!` runs.
const InitialStateSetup = Function

"""
    BenchmarkSpec(; name, namelist_path, grid, end_time, output_times,
                    setup_initial_state!, time_init=0.0, dt=1.0, alias)

Describe one Fortran-internally-generated benchmark.

Fields:

  - `name`              : short identifier; doubles as the alias and
                          the fixture-file stem (e.g. `"bueler_b_smoke"`).
  - `namelist_path`     : absolute path to a Yelmo Fortran namelist
                          (`.nml`). YelmoMirror reads this verbatim.
                          The namelist must set `init_topo_load = .false.`
                          so `yelmo_init` doesn't try to read topography
                          from a (nonexistent) file.
  - `grid`              : synthetic-grid spec (NamedTuple) passed to the
                          new `YelmoMirror(p, time; grid=...)` path.
                          Required keys: `xc`, `yc` (cell centres in
                          metres). Optional: `grid_name`, `lon`, `lat`,
                          `area`.
  - `end_time`          : final simulation time in years.
  - `output_times`      : strictly-increasing vector of times (yr) at
                          which to write restart fixtures. Default:
                          `[end_time]` (final snapshot only). EISMINT-
                          flavoured benchmarks may want N evenly-
                          spaced times to capture transient evolution.
  - `setup_initial_state!`: callback `(ymirror, time_init) -> nothing`.
                          Sets H_ice / bnd fields per the benchmark and
                          calls `yelmo_sync!` at the end.
  - `time_init`         : starting time in years (default 0.0). For
                          BUELER analytical setups this is `t - t₀`,
                          where `t₀` is the Halfar reference time.
  - `dt`                : main loop time step in years (default 1.0).
  - `alias`             : YelmoMirror alias (default `name`). One of
                          `"ylmo1"` / `"ylmo2"` reserved by the C API.
"""
Base.@kwdef struct BenchmarkSpec
    name::String
    namelist_path::String
    grid::NamedTuple
    end_time::Float64
    output_times::Vector{Float64} = Float64[]
    setup_initial_state!::InitialStateSetup
    time_init::Float64 = 0.0
    dt::Float64        = 1.0
    alias::String      = ""    # defaults to `name` when empty
end

# Resolve a missing alias to the spec name. The C API treats alias
# `"ylmo2"` as a second persistent state slot; everything else maps
# to `ylmo1`. Benchmarks normally just want `ylmo1` (the default).
_alias(spec::BenchmarkSpec) = isempty(spec.alias) ? "ylmo1" : spec.alias

# Resolve the actual list of output times: default to `[end_time]`
# (final snapshot only) when the user left `output_times` empty.
function _output_times(spec::BenchmarkSpec)
    isempty(spec.output_times) ? [spec.end_time] : spec.output_times
end

# Stable file path for a fixture at the given output time. Filename is
# `<spec.name>__t<time-yr>.nc` (rounded to nearest year). For
# single-snapshot specs this collapses to `<name>__t<end>.nc`.
function _fixture_path(fixtures_dir::String, spec::BenchmarkSpec, time::Float64)
    rounded = Int(round(time))
    return joinpath(fixtures_dir, "$(spec.name)__t$(rounded).nc")
end

"""
    run_mirror_benchmark!(spec; fixtures_dir, overwrite=false) -> Vector{String}

Run `spec` via YelmoMirror, writing a NetCDF restart fixture at each
output time. Returns a `Vector{String}` of the produced fixture paths.

Refuses to overwrite an existing fixture unless `overwrite = true`.
This protects committed fixtures from accidental clobber when running
the smoke tests against pre-existing data.

Requires `libyelmo_c_api.so` to be present (i.e. the Yelmo Fortran
library has been built locally) and to expose the `yelmo_init_grid`
and `yelmo_restart_write` symbols. CI does not call this function —
it loads pre-committed fixtures via [`load_fixture`](@ref).
"""
function run_mirror_benchmark!(spec::BenchmarkSpec;
                                fixtures_dir::String,
                                overwrite::Bool = false)
    isfile(spec.namelist_path) ||
        error("BenchmarkSpec.namelist_path not found: $(spec.namelist_path)")
    mkpath(fixtures_dir)

    out_times = sort!(unique!(_output_times(spec)))
    out_times[1] >= spec.time_init ||
        error("First output_time ($(out_times[1])) precedes time_init " *
              "($(spec.time_init))")
    out_times[end] <= spec.end_time + 1e-9 ||
        error("Last output_time ($(out_times[end])) exceeds end_time " *
              "($(spec.end_time))")

    # Pre-check: refuse overwrite of any existing fixture.
    paths = [_fixture_path(fixtures_dir, spec, t) for t in out_times]
    if !overwrite
        existing = filter(isfile, paths)
        isempty(existing) || error(
            "run_mirror_benchmark!: fixtures already exist; pass " *
            "`overwrite=true` to clobber. Existing: " * join(existing, ", "))
    end

    # 1. Build YelmoMirror with synthetic grid.
    p = Yelmo.YelmoPar.read_nml(spec.namelist_path)
    rundir = mktempdir(; prefix="bench_$(spec.name)_")
    # Yelmo Fortran reads `input/yelmo_phys_const.nml` (and other
    # input files) as a CWD-relative path. Symlink the yelmo source
    # tree's `input/` into the rundir so the Fortran lookup resolves
    # regardless of where Julia was invoked from. Then `cd` to rundir
    # for the actual YelmoMirror call.
    yelmo_input = abspath(joinpath(dirname(Yelmo.YelmoMirrorCore.yelmolib),
                                   "..", "..", "input"))
    isdir(yelmo_input) || error(
        "run_mirror_benchmark!: yelmo input directory not found at $yelmo_input. " *
        "Expected the Yelmo Fortran source tree to be available alongside libyelmo.")
    symlink(yelmo_input, joinpath(rundir, "input"))

    cwd_orig = pwd()
    cd(rundir)
    ymirror = try
        YelmoMirror(p, spec.time_init;
                    grid      = spec.grid,
                    alias     = _alias(spec),
                    rundir    = rundir,
                    overwrite = true)
    catch
        cd(cwd_orig)
        rethrow()
    end

    try
        # 2. Apply the spec's initial-state setup (sets H_ice, bnd
        #    fields in Julia and pushes them back to Fortran).
        spec.setup_initial_state!(ymirror, spec.time_init)

        # 3. Initialise Fortran state derivatives from the now-set
        #    inputs.
        init_state!(ymirror, spec.time_init)

        # 4. Time-step loop. Step in `spec.dt` increments; whenever
        #    we cross an `output_time` write a fixture.
        t = spec.time_init
        next_out = 1
        while next_out <= length(out_times)
            target = out_times[next_out]
            while t < target - 1e-9
                dt_step = min(spec.dt, target - t)
                step!(ymirror, dt_step)
                t = ymirror.time
            end
            yelmo_write_restart!(ymirror, paths[next_out]; time=t)
            next_out += 1
        end

        # 5. Optional: continue stepping past the last output if
        #    end_time is later (no further fixtures, just end-state
        #    cleanup so step! invariants don't drift).
        while ymirror.time < spec.end_time - 1e-9
            dt_step = min(spec.dt, spec.end_time - ymirror.time)
            step!(ymirror, dt_step)
        end
    finally
        cd(cwd_orig)
    end

    return paths
end

"""
    load_fixture(spec; index=:final, kwargs...) -> YelmoModel

Construct a `YelmoModel` from a previously-written benchmark fixture.
`index` selects which output time:

  - `:final` (default) → the last entry of `_output_times(spec)`.
  - `Integer`         → 1-based index into `_output_times(spec)`.
  - `Real`            → match by time (within 1e-6 yr).

Extra kwargs flow through to `YelmoModel(...)`. The `groups` and
`strict` defaults match the existing dyn / topo regression tests:
load all groups except `dta`, with `strict = false` (some restart
fields may be schema-defined but absent).

The fixture must already exist; this function does NOT regenerate.
"""
function load_fixture(spec::BenchmarkSpec;
                      fixtures_dir::String,
                      index = :final,
                      groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                      strict::Bool = false,
                      kwargs...)
    out_times = sort!(unique!(_output_times(spec)))
    t = if index === :final
        out_times[end]
    elseif index isa Integer
        out_times[index]
    elseif index isa Real
        idx = findfirst(τ -> abs(τ - Float64(index)) < 1e-6, out_times)
        idx === nothing && error(
            "load_fixture: no output_time matches $(index) (available: $out_times)")
        out_times[idx]
    else
        error("load_fixture: `index` must be :final, an Integer, or a Real (got $(typeof(index)))")
    end
    path = _fixture_path(fixtures_dir, spec, t)
    isfile(path) || error("load_fixture: fixture missing at $path; run regenerate.jl first.")
    return YelmoModel(path, t;
                      alias  = "$(spec.name)_load",
                      groups = groups,
                      strict = strict,
                      kwargs...)
end

end # module YelmoBenchmarks
