# ----------------------------------------------------------------------
# Multi-valued bed-state diagnostics.
#
#   - `calc_grounding_line_zone!` — `mask_grz` from `dist_grline`.
#   - `gen_mask_bed!`             — `mask_bed` from per-cell ice /
#                                    flotation / PMP state, plus the
#                                    grounding-line cell flag.
#
# Both kernels write Float64 fields whose values are the integer enum
# constants from YelmoConst (`MASK_BED_*`, plus the implicit
# `mask_grz ∈ {-2,-1,0,1,2}`).
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

using ..YelmoConst: MASK_BED_OCEAN, MASK_BED_LAND, MASK_BED_FROZEN,
                    MASK_BED_STREAM, MASK_BED_GRLINE, MASK_BED_FLOAT,
                    MASK_BED_PARTIAL

export calc_grounding_line_zone!, gen_mask_bed!

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
