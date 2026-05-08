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
using Oceananigans.BoundaryConditions: fill_halo_regions!

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
                                   bnd, dta, dyn, mat, thrm, tpo, timer,
                                   Yelmo.YelmoHooks.YelmoHooks())

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

# -----------------------------------------------------------------------
# CalvingMIP calving-law hooks (Yelmo `YelmoHooks.calv_flt` signature).
#
# These laws operate on Oceananigans XFaceField / YFaceField and need
# Yelmo's hook plumbing, so they live in the package extension rather
# than the model-agnostic `IceSheetBenchmarks/src/calvingmip.jl`. A
# generic array-only version may be added to IceSheetBenchmarks later
# once a non-Yelmo host needs it.
#
# Both hooks take the same positional signature as
# `Yelmo.YelmoHooks.calv_flt`: `(cr_x, cr_y, u_bar, v_bar, H_ice,
# f_ice, lsf, time)`. Keyword args (`xc`, `yc`, `r_lim`) are captured
# at hook installation time:
#
#   y.hooks.calv_flt = (cx, cy, ux, uy, Hi, fi, lsf, t) ->
#       IceSheetBenchmarks.calvmip_exp1!(cx, cy, ux, uy, Hi, fi, lsf, t;
#                                        xc = b.xc, yc = b.yc)
# -----------------------------------------------------------------------

"""
    IceSheetBenchmarks.calvmip_exp1!(cr_x, cr_y, u_bar, v_bar,
                                     H_ice, f_ice, lsf, time;
                                     xc, yc, r_lim=750e3)

CalvingMIP Exp1/3 calving-rate law (port of `calvmip_exp1` in
`yelmo/src/physics/calving/calving_ac.f90:395-482`).

Algorithm:
  1. `cr = −u` everywhere (velocity-equilibrium calving).
  2. For aa-centres inside `r < r_lim`, zero the calving rate on faces
     whose other neighbour is also inside `r_lim`. Faces straddling the
     `r_lim` boundary keep `cr = −u`. Result: front pinned at the
     `r_lim` circle (default 750 km).
"""
function IceSheetBenchmarks.calvmip_exp1!(cr_x, cr_y,
                                           u_bar, v_bar,
                                           H_ice, f_ice, lsf, time::Float64;
                                           xc::AbstractVector{<:Real},
                                           yc::AbstractVector{<:Real},
                                           r_lim::Real = 750e3)
    fill_halo_regions!(u_bar)
    fill_halo_regions!(v_bar)

    Cx = interior(cr_x); Cy = interior(cr_y)
    Ux = interior(u_bar); Uy = interior(v_bar)

    Nx = length(xc); Ny = length(yc)

    # Step 1: cr = −u (velocity equilibrium).
    @inbounds for j in axes(Cx, 2), i in axes(Cx, 1)
        Cx[i, j, 1] = -Ux[i, j, 1]
    end
    @inbounds for j in axes(Cy, 2), i in axes(Cy, 1)
        Cy[i, j, 1] = -Uy[i, j, 1]
    end

    # Step 2: zero faces that lie strictly inside `r_lim`.
    # XFaceField indexing: face `i` sits between centres i−1 and i;
    # the face to the RIGHT of centre `i` is `cr_x[i+1, j]`.
    @inbounds for j in 1:Ny, i in 1:Nx
        r_ij = sqrt(xc[i]^2 + yc[j]^2)
        r_ij >= r_lim && continue

        if i < Nx
            r_right = sqrt(xc[i+1]^2 + yc[j]^2)
            if r_right < r_lim
                Cx[i+1, j, 1] = 0.0
            end
        end
        if i > 1
            r_left = sqrt(xc[i-1]^2 + yc[j]^2)
            if r_left < r_lim
                Cx[i, j, 1] = 0.0
            end
        end
        if j < Ny
            r_top = sqrt(xc[i]^2 + yc[j+1]^2)
            if r_top < r_lim
                Cy[i, j+1, 1] = 0.0
            end
        end
        if j > 1
            r_bot = sqrt(xc[i]^2 + yc[j-1]^2)
            if r_bot < r_lim
                Cy[i, j, 1] = 0.0
            end
        end
    end
    return cr_x, cr_y
end

"""
    IceSheetBenchmarks.calvmip_exp2!(cr_x, cr_y, u_bar, v_bar,
                                     H_ice, f_ice, lsf, time; xc, yc)

CalvingMIP Exp2 oscillating-front calving-rate law (port of
`calvmip_exp2` in `yelmo/src/physics/calving/calving_ac.f90:484-533`).

Net front velocity in the radial direction:
  w  = u + cr = (u/|u|) · wv,
  wv = −300 sin(2π t / 1000)   [m/yr],

so the front oscillates between ±300 m/yr radially with period 1000 yr.
The face-local speed `|u|` uses the cross-staggered partner velocity
(4-point average of the orthogonal face values), regularised with
`max(|u|, 1e-8)`. The simpler "face-normal magnitude" form gives a √2
speed bias at 45° and is unstable when the normal velocity is near
zero — see commit history for details.
"""
function IceSheetBenchmarks.calvmip_exp2!(cr_x, cr_y,
                                           u_bar, v_bar,
                                           H_ice, f_ice, lsf, time::Float64;
                                           xc::AbstractVector{<:Real},
                                           yc::AbstractVector{<:Real})
    fill_halo_regions!(u_bar)
    fill_halo_regions!(v_bar)

    Cx = interior(cr_x); Cy = interior(cr_y)
    Ux = interior(u_bar); Uy = interior(v_bar)

    wv = -300.0 * sinpi(2.0 * time / 1000.0)

    Nxu, Nyu = size(Ux, 1), size(Ux, 2)   # XFaceField: Nx+1, Ny
    Nxv, Nyv = size(Uy, 1), size(Uy, 2)   # YFaceField: Nx,   Ny+1

    # x-faces: cross-stagger v from the 4 surrounding y-faces.
    @inbounds for j in 1:Nyu, i in 1:Nxu
        u    = Ux[i, j, 1]
        i1   = max(1, i - 1);  i2 = min(Nxv, i)
        jp1  = min(Nyv, j + 1)
        vcrs = 0.25 * (Uy[i1, j, 1] + Uy[i1, jp1, 1] +
                       Uy[i2, j, 1] + Uy[i2, jp1, 1])
        uxy  = max(1e-8, sqrt(u*u + vcrs*vcrs))
        Cx[i, j, 1] = -u + (u / uxy) * wv
    end

    # y-faces: cross-stagger u from the 4 surrounding x-faces.
    @inbounds for j in 1:Nyv, i in 1:Nxv
        v    = Uy[i, j, 1]
        ip1  = min(Nxu, i + 1)
        jm1  = max(1, j - 1);  jj = min(Nyu, j)
        ucrs = 0.25 * (Ux[i,   jm1, 1] + Ux[ip1, jm1, 1] +
                       Ux[i,   jj,  1] + Ux[ip1, jj,  1])
        uxy  = max(1e-8, sqrt(v*v + ucrs*ucrs))
        Cy[i, j, 1] = -v + (v / uxy) * wv
    end
    return cr_x, cr_y
end

end # module YelmoBenchmarks
