# ----------------------------------------------------------------------
# Multi-valued bed-state diagnostics.
#
#   - `calc_grounding_line_zone!` — `mask_grz` from `dist_grline`.
#   - `gen_mask_bed!`             — `mask_bed` from per-cell ice /
#                                    flotation / PMP state, plus the
#                                    grounding-line cell flag.
#   - `calc_ice_front!`           — `mask_frnt` distinguishing
#                                    floating, marine and grounded
#                                    fronts plus their adjacent
#                                    ice-free cells.
#
# All three kernels write Float64 fields whose values are integer
# enums (`MASK_BED_*` from YelmoConst for `mask_bed`, `mask_grz ∈
# {-2,-1,0,1,2}`, `mask_frnt ∈ {-2, -1, 0, 1, 2, 3}`).
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.BoundaryConditions: fill_halo_regions!

using ..YelmoConst: MASK_BED_OCEAN, MASK_BED_LAND, MASK_BED_FROZEN,
                    MASK_BED_STREAM, MASK_BED_GRLINE, MASK_BED_FLOAT,
                    MASK_BED_PARTIAL,
                    MASK_FRNT_ICE_FREE, MASK_FRNT_ICE_FREE_LAND, MASK_FRNT_NONE,
                    MASK_FRNT_FLOAT, MASK_FRNT_MARINE, MASK_FRNT_GRND
using ..YelmoUtils: _neighbor_im1, _neighbor_ip1, _neighbor_jm1, _neighbor_jp1
using Oceananigans.Grids: AbstractTopology, topology

export calc_grounding_line_zone!, gen_mask_bed!, calc_ice_front!

# Float64 forms of the enum integers — eligible for in-kernel `==`
# comparisons against the stored CenterField values.
const _MASK_BED_OCEAN_F   = Float64(MASK_BED_OCEAN)
const _MASK_BED_LAND_F    = Float64(MASK_BED_LAND)
const _MASK_BED_FROZEN_F  = Float64(MASK_BED_FROZEN)
const _MASK_BED_STREAM_F  = Float64(MASK_BED_STREAM)
const _MASK_BED_GRLINE_F  = Float64(MASK_BED_GRLINE)
const _MASK_BED_FLOAT_F   = Float64(MASK_BED_FLOAT)
const _MASK_BED_PARTIAL_F = Float64(MASK_BED_PARTIAL)

"""
    calc_grounding_line_zone!(mask_grz, dist_gl, dist_grz_m) -> mask_grz

Bin the signed grounding-line-distance field `dist_gl` (in **metres**)
into a 5-valued zone mask:

| value | meaning                              |
|------:|--------------------------------------|
|  `-2` | floating cell outside grounding zone |
|  `-1` | floating cell inside grounding zone  |
|   `0` | grounding-line cell                  |
|  `+1` | grounded cell inside grounding zone  |
|  `+2` | grounded cell outside grounding zone |

`dist_grz_m` is the zone half-width in **metres**. The namelist
parameter `ytopo.dist_grz` lives in km; convert at the call site
(`1e3 * dist_grz`).

Port of `physics/topography.f90:1524 calc_grounding_line_zone`.
"""
function calc_grounding_line_zone!(mask_grz, dist_gl, dist_grz_m::Real)
    M = @view interior(mask_grz)[:, :, 1]
    D = @view interior(dist_gl)[:, :, 1]
    nx, ny = size(M)
    @assert size(D) == (nx, ny)
    thresh = Float64(dist_grz_m)

    @inbounds for j in 1:ny, i in 1:nx
        d = D[i, j]
        if d == 0.0
            M[i, j] = 0.0
        elseif d > 0.0
            M[i, j] = d <= thresh ? 1.0 : 2.0
        else  # d < 0
            M[i, j] = abs(d) <= thresh ? -1.0 : -2.0
        end
    end
    return mask_grz
end

"""
    gen_mask_bed!(mask_bed, f_ice, f_pmp, f_grnd, mask_grz) -> mask_bed

Fill the multi-valued bed mask cell-wise according to the Fortran
`gen_mask_bed` decision tree:

  1. **Grounding-line cell** (`mask_grz == 0`) → `MASK_BED_GRLINE`.
  2. **Ice-free cell** (`f_ice == 0`):
     - grounded (`f_grnd > 0`) → `MASK_BED_LAND`
     - floating              → `MASK_BED_OCEAN`
  3. **Partially ice-covered** (`0 < f_ice < 1`) → `MASK_BED_PARTIAL`.
  4. **Fully ice-covered** (`f_ice == 1`):
     - grounded, temperate base (`f_pmp > 0.5`) → `MASK_BED_STREAM`
     - grounded, frozen base                    → `MASK_BED_FROZEN`
     - floating                                 → `MASK_BED_FLOAT`

The `MASK_BED_ISLAND` value is reserved (the Fortran routine does not
currently emit it; the `find_connected_mask` helper that would is
flagged TO-DO upstream).

Port of `physics/topography.f90:61 gen_mask_bed`.
"""
function gen_mask_bed!(mask_bed, f_ice, f_pmp, f_grnd, mask_grz)
    Mb = @view interior(mask_bed)[:, :, 1]
    Fi = @view interior(f_ice)[:, :, 1]
    Fp = @view interior(f_pmp)[:, :, 1]
    Fg = @view interior(f_grnd)[:, :, 1]
    Mg = @view interior(mask_grz)[:, :, 1]
    nx, ny = size(Mb)
    @assert size(Fi) == (nx, ny)
    @assert size(Fp) == (nx, ny)
    @assert size(Fg) == (nx, ny)
    @assert size(Mg) == (nx, ny)

    @inbounds for j in 1:ny, i in 1:nx
        fi = Fi[i, j]
        fg = Fg[i, j]

        if Mg[i, j] == 0.0
            Mb[i, j] = _MASK_BED_GRLINE_F
        elseif fi == 0.0
            Mb[i, j] = (fg > 0.0) ? _MASK_BED_LAND_F : _MASK_BED_OCEAN_F
        elseif fi < 1.0
            Mb[i, j] = _MASK_BED_PARTIAL_F
        else
            # fi == 1 (fully ice-covered)
            if fg > 0.0
                Mb[i, j] = (Fp[i, j] > 0.5) ?
                           _MASK_BED_STREAM_F : _MASK_BED_FROZEN_F
            else
                Mb[i, j] = _MASK_BED_FLOAT_F
            end
        end
    end
    return mask_bed
end

const _MASK_FRNT_ICE_FREE_F      = Float64(MASK_FRNT_ICE_FREE)
const _MASK_FRNT_ICE_FREE_LAND_F = Float64(MASK_FRNT_ICE_FREE_LAND)
const _MASK_FRNT_NONE_F          = Float64(MASK_FRNT_NONE)
const _MASK_FRNT_FLOAT_F         = Float64(MASK_FRNT_FLOAT)
const _MASK_FRNT_MARINE_F        = Float64(MASK_FRNT_MARINE)
const _MASK_FRNT_GRND_F          = Float64(MASK_FRNT_GRND)

"""
    calc_ice_front!(mask_frnt, f_ice, f_grnd, z_bed, z_sl) -> mask_frnt

Mark the ice fronts with the `MASK_FRNT_*` codes (YelmoConst):

| value | meaning                                                    |
|------:|------------------------------------------------------------|
|  `-2` | ice-free cell next to a front, land (bed at or above sea level) |
|  `-1` | ice-free cell next to a front, ocean (bed below sea level) |
|   `0` | not a front cell                                           |
|  `+1` | floating ice front                                         |
|  `+2` | ice front grounded below sea level (marine)                |
|  `+3` | ice front grounded above sea level                         |

A front cell is a fully covered cell (`f_ice == 1`) with at least one
direct neighbour `f_ice < 1`; the ice-free neighbours are marked as ocean
or land from their own bed, so that the front type can be decided per
face (`set_ssa_masks!`). The model passes the dynamic cover `f_ice_dyn`.
Neighbours follow the grid topology (clamped when Bounded, wrapped when
Periodic).

Port of `physics/topography.f90:calc_ice_front` (yelmo dev).
"""
function calc_ice_front!(mask_frnt, f_ice, f_grnd, z_bed, z_sl)
    Mf = interior(mask_frnt)
    Fi = interior(f_ice)
    Fg = interior(f_grnd)
    Zb = interior(z_bed)
    Zs = interior(z_sl)
    Nx, Ny = size(Mf, 1), size(Mf, 2)
    Tx = topology(mask_frnt.grid, 1)
    Ty = topology(mask_frnt.grid, 2)
    _calc_ice_front_kernel!(Mf, Fi, Fg, Zb, Zs, Tx, Ty, Nx, Ny)
    return mask_frnt
end

# Ice-free cell next to a front: ocean if the bed is below sea level.
@inline _ice_free_code(z_bed, z_sl) =
    z_bed < z_sl ? _MASK_FRNT_ICE_FREE_F : _MASK_FRNT_ICE_FREE_LAND_F

function _calc_ice_front_kernel!(Mf, Fi, Fg, Zb, Zs,
                                 ::Type{Tx}, ::Type{Ty}, Nx::Int, Ny::Int
                                ) where {Tx<:AbstractTopology, Ty<:AbstractTopology}
    fill!(Mf, _MASK_FRNT_NONE_F)

    # A front cell writes the codes of its ice-free neighbours. They never
    # collide with front codes (a front cell is fully covered) and two front
    # cells write the same code into a shared neighbour.
    @inbounds for j in 1:Ny, i in 1:Nx
        Fi[i, j, 1] == 1.0 || continue

        im1 = _neighbor_im1(i, Nx, Tx)
        ip1 = _neighbor_ip1(i, Nx, Tx)
        jm1 = _neighbor_jm1(j, Ny, Ty)
        jp1 = _neighbor_jp1(j, Ny, Ty)

        fW = Fi[im1, j, 1]
        fE = Fi[ip1, j, 1]
        fS = Fi[i, jm1, 1]
        fN = Fi[i, jp1, 1]
        (fW < 1.0 || fE < 1.0 || fS < 1.0 || fN < 1.0) || continue

        if Fg[i, j, 1] > 0.0 && Zs[i, j, 1] <= Zb[i, j, 1]
            Mf[i, j, 1] = _MASK_FRNT_GRND_F
        elseif Fg[i, j, 1] > 0.0
            Mf[i, j, 1] = _MASK_FRNT_MARINE_F
        else
            Mf[i, j, 1] = _MASK_FRNT_FLOAT_F
        end

        fW < 1.0 && (Mf[im1, j, 1] = _ice_free_code(Zb[im1, j, 1], Zs[im1, j, 1]))
        fE < 1.0 && (Mf[ip1, j, 1] = _ice_free_code(Zb[ip1, j, 1], Zs[ip1, j, 1]))
        fS < 1.0 && (Mf[i, jm1, 1] = _ice_free_code(Zb[i, jm1, 1], Zs[i, jm1, 1]))
        fN < 1.0 && (Mf[i, jp1, 1] = _ice_free_code(Zb[i, jp1, 1], Zs[i, jp1, 1]))
    end
    return nothing
end
