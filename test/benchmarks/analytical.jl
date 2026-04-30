# ----------------------------------------------------------------------
# Analytical-solution fixture writer.
#
# For benchmarks with closed-form analytical solutions (BUELER-A,
# BUELER-B/C Halfar, etc.) we don't need to actually run YelmoMirror
# to produce a fixture — the analytical formula IS the fixture.
# Skips the entire YelmoMirror dependency: no Fortran library
# required, no Yelmo namelist gymnastics, no synthetic-grid plumbing.
#
# Format: a NetCDF restart that `YelmoModel(restart_file, time;
# strict=false)` can load directly. Only the variables the spec's
# `write_fields!` callback explicitly sets are populated; everything
# else is omitted from the file and `load_state!(strict=false)` skips
# them, leaving the corresponding YelmoModel fields at their default
# (zero) allocation.
#
# Used for the BUELER-B smoke benchmark in this PR. Future benchmarks
# without analytical solutions (EISMINT-moving, ISMIP-HOM, slab,
# trough, MISMIP+, CalvingMIP) use the YelmoMirror path in
# `helpers.jl::run_mirror_benchmark!` instead.
# ----------------------------------------------------------------------

using NCDatasets

export AnalyticalSpec, write_analytical_fixture!

"""
    AnalyticalSpec(; name, output_times, xc, yc, zeta_ac=nothing,
                     zeta_rock=nothing, write_fields!)

Describe one analytical-solution benchmark. `write_fields!(ds, time)`
is invoked once per `output_time`; it should `defVar`-and-write any
state variables (commonly just `H_ice`, optionally a few `bnd` /
`tpo` companions) that the smoke / lockstep test will assert against.

Coordinates:

  - `xc`, `yc`: cell-centre coordinates in **metres**. Written to the
    NetCDF in **kilometres** (the Yelmo restart convention; YelmoModel
    converts back to metres on load via the `units = "km"` attribute).
  - `zeta_ac`: face coordinates of the ice z-axis. Defaults to the
    standard 11-point `[0, 0.1, ..., 1.0]` (uniformly-spaced sigma
    layers) when `nothing`.
  - `zeta_rock`: face coordinates of the bedrock z-axis. Default
    omits the rock-grid 3D entirely (YelmoModel constructor errors if
    no rock vertical axis is found, so we default to 5 layers spaced
    at uniform sigma).

`output_times`: list of times (yr) at which to write a fixture.
Single-element list collapses to a single fixture file.

`name`: identifier; fixture filenames are `<name>__t<time>.nc`,
matching the YelmoMirror path's convention so `load_fixture` finds
them either way.
"""
Base.@kwdef struct AnalyticalSpec
    name::String
    output_times::Vector{Float64}
    xc::Vector{Float64}
    yc::Vector{Float64}
    zeta_ac::Vector{Float64}   = collect(range(0.0, 1.0; length=11))
    zeta_rock::Vector{Float64} = collect(range(0.0, 1.0; length=5))
    write_fields!::Function    # (ds::NCDataset, time::Float64) -> nothing
end

"""
    write_analytical_fixture!(spec; fixtures_dir, overwrite=false)
        -> Vector{String}

Write one NetCDF fixture per `spec.output_time`. Each file carries
the full YelmoModel coordinate structure (xc, yc, zeta_ac,
zeta_rock_ac) plus any state variables the spec's `write_fields!`
callback dropped in.

Returns the list of fixture paths, in the same order as
`spec.output_times`.
"""
function write_analytical_fixture!(spec::AnalyticalSpec;
                                   fixtures_dir::String,
                                   overwrite::Bool = false)
    mkpath(fixtures_dir)
    out_times = sort!(unique!(copy(spec.output_times)))

    paths = [joinpath(fixtures_dir,
                      "$(spec.name)__t$(Int(round(t))).nc")
             for t in out_times]

    if !overwrite
        existing = filter(isfile, paths)
        isempty(existing) || error(
            "write_analytical_fixture!: fixtures already exist; pass " *
            "`overwrite=true` to clobber. Existing: " *
            join(existing, ", "))
    end

    for (path, t) in zip(paths, out_times)
        isfile(path) && rm(path)   # overwrite=true case
        NCDataset(path, "c") do ds
            _write_coordinates!(ds, spec)
            spec.write_fields!(ds, t)
        end
    end

    return paths
end

# Lay down the four coordinate dimensions / variables YelmoModel's
# `load_grids_from_restart` looks for, with units that match the
# Greenland-restart convention (xc/yc in km, zeta in dimensionless).
function _write_coordinates!(ds::NCDataset, spec::AnalyticalSpec)
    Nx, Ny       = length(spec.xc), length(spec.yc)
    Nz_ac        = length(spec.zeta_ac)
    Nz_rock_ac   = length(spec.zeta_rock)

    defDim(ds, "xc",          Nx)
    defDim(ds, "yc",          Ny)
    defDim(ds, "zeta",        Nz_ac - 1)
    defDim(ds, "zeta_ac",     Nz_ac)
    defDim(ds, "zeta_rock",   Nz_rock_ac - 1)
    defDim(ds, "zeta_rock_ac", Nz_rock_ac)

    # Convert xc/yc from metres → km for the saved file. YelmoModel's
    # `load_grids_from_restart` reads the `units` attribute and
    # multiplies by 1000 on load when it sees "km".
    xv = defVar(ds, "xc", Float64, ("xc",))
    xv[:] = spec.xc ./ 1e3
    xv.attrib["units"] = "km"

    yv = defVar(ds, "yc", Float64, ("yc",))
    yv[:] = spec.yc ./ 1e3
    yv.attrib["units"] = "km"

    # Cell-centre zeta (used as a fallback when zeta_ac is absent).
    zc = defVar(ds, "zeta", Float64, ("zeta",))
    zc[:] = 0.5 .* (spec.zeta_ac[1:end-1] .+ spec.zeta_ac[2:end])
    zc.attrib["units"] = "1"

    zac = defVar(ds, "zeta_ac", Float64, ("zeta_ac",))
    zac[:] = spec.zeta_ac
    zac.attrib["units"] = "1"

    zrc = defVar(ds, "zeta_rock", Float64, ("zeta_rock",))
    zrc[:] = 0.5 .* (spec.zeta_rock[1:end-1] .+ spec.zeta_rock[2:end])
    zrc.attrib["units"] = "1"

    zracv = defVar(ds, "zeta_rock_ac", Float64, ("zeta_rock_ac",))
    zracv[:] = spec.zeta_rock
    zracv.attrib["units"] = "1"

    return nothing
end

"""
    load_analytical_fixture(spec; fixtures_dir, index=:final, kwargs...)
        -> YelmoModel

Same shape as `load_fixture` but for an `AnalyticalSpec`. Looks up
the `<spec.name>__t<time>.nc` file matching `index`, then loads via
the existing `YelmoModel(restart_file, time; ...)` constructor with
`strict = false` (so unset schema variables fall back to default
allocation).
"""
function load_analytical_fixture(spec::AnalyticalSpec;
                                  fixtures_dir::String,
                                  index = :final,
                                  groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                                  strict::Bool = false,
                                  kwargs...)
    out_times = sort!(unique!(copy(spec.output_times)))
    t = if index === :final
        out_times[end]
    elseif index isa Integer
        out_times[index]
    elseif index isa Real
        idx = findfirst(τ -> abs(τ - Float64(index)) < 1e-6, out_times)
        idx === nothing && error(
            "load_analytical_fixture: no output_time matches $(index) " *
            "(available: $out_times)")
        out_times[idx]
    else
        error("load_analytical_fixture: `index` must be :final, Integer, or Real (got $(typeof(index)))")
    end
    path = joinpath(fixtures_dir, "$(spec.name)__t$(Int(round(t))).nc")
    isfile(path) || error("load_analytical_fixture: fixture missing at $path; run regenerate.jl first.")
    return YelmoModel(path, t;
                      alias  = "$(spec.name)_load",
                      groups = groups,
                      strict = strict,
                      kwargs...)
end
