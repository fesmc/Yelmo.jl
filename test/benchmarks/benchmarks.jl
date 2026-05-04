# ----------------------------------------------------------------------
# AbstractBenchmark — interface contract for analytical benchmarks.
#
# An `AbstractBenchmark` carries the parameters of a closed-form
# benchmark (BUELER-B / BUELER-C Halfar dome, etc.) and is the source
# of truth for both:
#
#   - the analytical state at any time `t` (used directly to construct
#     a `YelmoModel` in memory, no NetCDF round-trip), and
#   - the on-disk fixture file (a NetCDF restart that the file-based
#     `YelmoModel(restart_file, time; …)` constructor can load).
#
# Concrete subtypes provide the `state(b, t)` analytical-state method;
# the rest of the interface (writing fixtures, building YelmoModels)
# is generic and lives here.
#
# YelmoMirror-driven benchmarks (EISMINT, ISMIP-HOM, slab, trough,
# MISMIP+, CalvingMIP) use `BenchmarkSpec` from `helpers.jl` instead —
# different backend, different scaffolding.
# ----------------------------------------------------------------------

using Oceananigans
using Oceananigans: interior
using Oceananigans.Grids: RectilinearGrid, Bounded, Flat
using Yelmo
using Yelmo: resolve_boundaries, MASK_ICE_DYNAMIC

# Pull the package-private allocation helpers from YelmoCore. They are
# not part of the Yelmo public API (underscore-prefixed) but the
# in-memory `YelmoModel(::AbstractBenchmark, t)` constructor needs to
# share them with the file-based path so the resulting group
# NamedTuples are bit-identical between the two routes.
const _load_yelmo_variable_meta = Yelmo.YelmoCore._load_yelmo_variable_meta
const _alloc_yelmo_groups       = Yelmo.YelmoCore._alloc_yelmo_groups

using NCDatasets

export AbstractBenchmark
export state, write_fixture!, analytical_velocity

"""
    AbstractBenchmark

Supertype for benchmarks with closed-form analytical solutions.

Concrete subtypes (e.g. `BuelerBenchmark`) carry the parameters of the
benchmark (grid axes, dome scale, Glen-flow parameters, …) and must
implement:

  - `state(b, t)` — analytical state at time `t`, returned as a
    `NamedTuple` whose keys are variable names matching the
    `src/variables/model/yelmo-variables-*.md` schema (e.g. `:H_ice`,
    `:z_bed`, `:smb_ref`).
  - `write_fixture!(b, path; times)` — serialize the analytical state
    at one or more times to a NetCDF restart file.
  - `analytical_velocity(b, t)` (optional) — analytical
    depth-averaged ice-velocity field.

`YelmoModel(b, t; …)` is a generic constructor (defined here) that
calls `state(b, t)` and routes the resulting fields into the
appropriate component group via the same allocation path the
file-based constructor uses.
"""
abstract type AbstractBenchmark end

# Interface stubs — concrete subtypes must implement.
function state end             # state(b::AbstractBenchmark, t::Real) -> NamedTuple
function write_fixture! end    # write_fixture!(b, path; times=[t]) -> Vector{String}
function analytical_velocity end

# Default error stub for benchmarks without an analytical velocity.
analytical_velocity(b::AbstractBenchmark, t::Real) = error(
    "analytical_velocity not implemented for $(typeof(b)). " *
    "Use a concrete benchmark subtype with a closed-form velocity solution.")

# ----------------------------------------------------------------------
# In-memory YelmoModel constructor: skip NetCDF entirely, build a
# YelmoModel directly from the analytical state of an AbstractBenchmark.
# ----------------------------------------------------------------------

# Build a 1D `range(z_face[1], z_face[end]; length=length(z_face))`-style
# Bounded vertical axis when zeta_ac is regular, falling back to the
# face-array form when irregular. Keeps the synthetic grid construction
# short in the common case.
function _zeta_axis(zeta_ac::AbstractVector{<:Real})
    return collect(Float64, zeta_ac)
end

# Build (g, gt, gr) RectilinearGrids from raw cell-centre / face axes.
# Mirrors `load_grids_from_restart` but takes the arrays as direct
# arguments, with no NetCDF in the loop.
function _grids_from_axes(xc::AbstractVector{<:Real},
                          yc::AbstractVector{<:Real},
                          zeta_ac::AbstractVector{<:Real},
                          zeta_rock_ac::AbstractVector{<:Real};
                          boundaries = :bounded)
    Nx = length(xc)
    Ny = length(yc)
    dx = xc[2] - xc[1]
    dy = yc[2] - yc[1]
    xlims = (xc[1] - dx/2, xc[end] + dx/2)
    ylims = (yc[1] - dy/2, yc[end] + dy/2)

    Tx, Ty = resolve_boundaries(boundaries)

    g = RectilinearGrid(size=(Nx, Ny),
                        x=xlims, y=ylims,
                        topology=(Tx, Ty, Flat))

    z_ice = _zeta_axis(zeta_ac)
    gt = RectilinearGrid(size=(Nx, Ny, length(z_ice) - 1),
                         x=xlims, y=ylims, z=z_ice,
                         topology=(Tx, Ty, Bounded))

    z_rock = _zeta_axis(zeta_rock_ac)
    gr = RectilinearGrid(size=(Nx, Ny, length(z_rock) - 1),
                         x=xlims, y=ylims, z=z_rock,
                         topology=(Tx, Ty, Bounded))

    return g, gt, gr
end

# Look up which component group `key` belongs to via the variable-meta
# tables. Returns the group symbol (`:bnd`, `:dyn`, …) or `nothing` if
# the key is unknown / a coordinate axis (`:xc`, `:yc`, `:zeta_ac`).
function _group_for_var(key::Symbol, v_meta)
    for gname in (:bnd, :dta, :dyn, :mat, :thrm, :tpo)
        haskey(getfield(v_meta, gname), key) && return gname
    end
    return nothing
end

# Default coordinate axes for analytical fixtures: 11-point uniform
# zeta_ac (10 ice layers) and 5-point uniform zeta_rock_ac (4 rock
# layers), matching the shape baked into the BUELER-B reference fixture.
const _DEFAULT_ZETA_AC      = collect(range(0.0, 1.0; length=11))
const _DEFAULT_ZETA_ROCK_AC = collect(range(0.0, 1.0; length=5))

"""
    YelmoModel(b::AbstractBenchmark, t::Real;
               alias::String = "...",
               rundir::String = "./",
               p = nothing,
               c::YelmoConstants = YelmoConstants(),
               boundaries = :bounded,
               kwargs...) -> YelmoModel

Build a `YelmoModel` directly from the analytical state of `b` at
time `t`, with **no** NetCDF round-trip. Calls `state(b, t)` and
writes the resulting NamedTuple's fields into the appropriate
component group via the same allocation path the file-based
constructor uses.

The state NamedTuple may carry coordinate-axis arrays (`:xc`, `:yc`,
`:zeta_ac`, `:zeta_rock_ac`) — these drive the grid construction.
If absent, defaults are taken from `b`'s `xc` / `yc` fields and from
`_DEFAULT_ZETA_AC` / `_DEFAULT_ZETA_ROCK_AC`. Any other key matching
a Yelmo schema variable is routed into the corresponding group.

`mask_ice` is set to `MASK_ICE_DYNAMIC` everywhere.
"""
function Yelmo.YelmoModel(b::AbstractBenchmark, t::Real;
                          alias::String = _default_alias(b, t),
                          rundir::String = "./",
                          p = nothing,
                          c::YelmoConstants = YelmoConstants(),
                          boundaries = :bounded)

    if p === nothing
        p = YelmoModelParameters(alias)
    end

    s = state(b, Float64(t))

    # Resolve coordinate axes: prefer s.xc / s.yc (in metres); fall
    # back to b.xc / b.yc; zeta defaults are uniform 11-point /
    # 5-point sigma layers.
    xc = haskey(s, :xc)            ? collect(Float64, s.xc)            : collect(Float64, b.xc)
    yc = haskey(s, :yc)            ? collect(Float64, s.yc)            : collect(Float64, b.yc)
    zeta_ac      = haskey(s, :zeta_ac)      ? collect(Float64, s.zeta_ac)      : copy(_DEFAULT_ZETA_AC)
    zeta_rock_ac = haskey(s, :zeta_rock_ac) ? collect(Float64, s.zeta_rock_ac) : copy(_DEFAULT_ZETA_ROCK_AC)

    g, gt, gr = _grids_from_axes(xc, yc, zeta_ac, zeta_rock_ac; boundaries=boundaries)

    v_meta = _load_yelmo_variable_meta()
    bnd, dta, dyn, mat, thrm, tpo = _alloc_yelmo_groups(g, gt, gr, v_meta)

    # Default-positional struct constructor (NOT the `restart_file` /
    # `b::AbstractBenchmark` keyword overloads; we want the raw field
    # constructor exposed by the mutable struct definition itself).
    timer = Yelmo.YelmoTimer(enabled = p.yelmo.timing)
    y = Yelmo.YelmoCore.YelmoModel(alias, rundir, Float64(t), p, c,
                                   g, gt, gr, v_meta,
                                   bnd, dta, dyn, mat, thrm, tpo, timer)

    # Default mask_ice to all-dynamic (the file-based loader does the
    # same before calling load_state!).
    fill!(interior(y.bnd.mask_ice), Float64(MASK_ICE_DYNAMIC))

    # Walk the state NamedTuple and route every schema-matching entry
    # into the appropriate group's field. Coordinate axes and unknown
    # keys are silently skipped.
    coord_keys = (:xc, :yc, :zeta, :zeta_ac, :zeta_rock, :zeta_rock_ac)
    for key in keys(s)
        key in coord_keys && continue
        gname = _group_for_var(key, v_meta)
        gname === nothing && continue
        group = getfield(y, gname)
        haskey(group, key) || continue
        field = getfield(group, key)
        arr = s[key]
        _assign_field!(field, arr)
    end

    return y
end

# Build a default alias from a benchmark + time. Subtypes can override
# by passing `alias = "..."` at the call site; this just provides a
# reasonable default like `"bueler_b_t1000"`.
_default_alias(b::AbstractBenchmark, t::Real) = "$(typeof(b))_t$(Int(round(t)))"

# Write `arr` into `field`'s interior, broadcasting over the trailing
# dimension when `arr` is 2D and `field` interior has a singleton 3rd
# axis (e.g. `H_ice` is 2D in fixtures but laid out as `(Nx, Ny, 1)` in
# Oceananigans Center fields with Flat-z).
function _assign_field!(field, arr::AbstractArray)
    iv = interior(field)
    if ndims(arr) == ndims(iv)
        iv .= arr
    elseif ndims(arr) == 2 && ndims(iv) == 3 && size(iv, 3) == 1
        iv[:, :, 1] .= arr
    elseif ndims(arr) == 3 && ndims(iv) == 3
        iv .= arr
    else
        error("_assign_field!: incompatible shapes — arr=$(size(arr)) field=$(size(iv))")
    end
    return field
end
