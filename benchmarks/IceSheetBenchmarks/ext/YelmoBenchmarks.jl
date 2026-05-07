module YelmoBenchmarks

# Package extension: activates when both IceSheetBenchmarks and Yelmo
# are loaded. Adds a Yelmo.YelmoModel(::AbstractBenchmark, t) method
# that builds a YelmoModel directly from a benchmark spec (no NetCDF
# round-trip).

using IceSheetBenchmarks: IceSheetBenchmarks, AbstractBenchmark, state
using Yelmo
using Yelmo: YelmoConstants, MASK_ICE_DYNAMIC, resolve_boundaries
using Yelmo.YelmoModelPar: YelmoModelParameters
using Oceananigans
using Oceananigans: interior
using Oceananigans.Grids: RectilinearGrid, Bounded, Flat

const _load_yelmo_variable_meta = Yelmo.YelmoCore._load_yelmo_variable_meta
const _alloc_yelmo_groups       = Yelmo.YelmoCore._alloc_yelmo_groups

const _DEFAULT_ZETA_AC      = collect(range(0.0, 1.0; length=11))
const _DEFAULT_ZETA_ROCK_AC = collect(range(0.0, 1.0; length=5))

function _zeta_axis(zeta_ac::AbstractVector{<:Real})
    return collect(Float64, zeta_ac)
end

function _grids_from_axes(xc::AbstractVector{<:Real},
                          yc::AbstractVector{<:Real},
                          zeta_ac::AbstractVector{<:Real},
                          zeta_rock_ac::AbstractVector{<:Real};
                          boundaries = :bounded)
    Nx = length(xc); Ny = length(yc)
    dx = xc[2] - xc[1]; dy = yc[2] - yc[1]
    xlims = (xc[1] - dx/2, xc[end] + dx/2)
    ylims = (yc[1] - dy/2, yc[end] + dy/2)

    Tx, Ty = resolve_boundaries(boundaries)

    g = RectilinearGrid(size=(Nx, Ny), x=xlims, y=ylims,
                        topology=(Tx, Ty, Flat))

    z_ice  = _zeta_axis(zeta_ac)
    gt = RectilinearGrid(size=(Nx, Ny, length(z_ice) - 1),
                         x=xlims, y=ylims, z=z_ice,
                         topology=(Tx, Ty, Bounded))

    z_rock = _zeta_axis(zeta_rock_ac)
    gr = RectilinearGrid(size=(Nx, Ny, length(z_rock) - 1),
                         x=xlims, y=ylims, z=z_rock,
                         topology=(Tx, Ty, Bounded))

    return g, gt, gr
end

function _group_for_var(key::Symbol, v_meta)
    for gname in (:bnd, :dta, :dyn, :mat, :thrm, :tpo)
        haskey(getfield(v_meta, gname), key) && return gname
    end
    return nothing
end

_default_alias(b::AbstractBenchmark, t::Real) = "$(typeof(b))_t$(Int(round(t)))"

function _assign_field!(field, arr::AbstractArray)
    iv = interior(field)
    if ndims(arr) == ndims(iv)
        iv .= arr
    elseif ndims(arr) == 2 && ndims(iv) == 3 && size(iv, 3) == 1
        iv[:, :, 1] .= arr
    elseif ndims(arr) == 3 && ndims(iv) == 3
        iv .= arr
    else
        error("YelmoBenchmarks._assign_field!: incompatible shapes — " *
              "arr=$(size(arr)) field=$(size(iv))")
    end
    return field
end

"""
    Yelmo.YelmoModel(b::AbstractBenchmark, t::Real;
                     alias::String = "...",
                     rundir::String = "./",
                     p = nothing,
                     c::YelmoConstants = YelmoConstants(),
                     boundaries = :bounded) -> YelmoModel

Build a `YelmoModel` directly from the analytical state of `b` at
time `t`, with no NetCDF round-trip. Calls `state(b, t)` and writes
the resulting NamedTuple's fields into the appropriate component
group via the same allocation path the file-based constructor uses.

Coordinate-axis entries in the state NamedTuple (`:xc`, `:yc`,
`:zeta_ac`, `:zeta_rock_ac`) drive grid construction; defaults fall
back to `b.xc / b.yc` and uniform 11-/5-point sigma layers.
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

    xc = haskey(s, :xc) ? collect(Float64, s.xc) : collect(Float64, b.xc)
    yc = haskey(s, :yc) ? collect(Float64, s.yc) : collect(Float64, b.yc)
    zeta_ac      = haskey(s, :zeta_ac)      ? collect(Float64, s.zeta_ac)      : copy(_DEFAULT_ZETA_AC)
    zeta_rock_ac = haskey(s, :zeta_rock_ac) ? collect(Float64, s.zeta_rock_ac) : copy(_DEFAULT_ZETA_ROCK_AC)

    g, gt, gr = _grids_from_axes(xc, yc, zeta_ac, zeta_rock_ac; boundaries=boundaries)

    v_meta = _load_yelmo_variable_meta()
    bnd, dta, dyn, mat, thrm, tpo = _alloc_yelmo_groups(g, gt, gr, v_meta)

    timer = Yelmo.YelmoTimer(enabled = p.yelmo.timing)
    y = Yelmo.YelmoCore.YelmoModel(alias, rundir, Float64(t), p, c,
                                   g, gt, gr, v_meta,
                                   bnd, dta, dyn, mat, thrm, tpo, timer)

    fill!(interior(y.bnd.mask_ice), Float64(MASK_ICE_DYNAMIC))

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

end # module YelmoBenchmarks
