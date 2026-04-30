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
# {-2,-1,0,1,2}`, `mask_frnt ∈ {-1, 0, 1, 3}`).
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.BoundaryConditions: fill_halo_regions!

using ..YelmoConst: MASK_BED_OCEAN, MASK_BED_LAND, MASK_BED_FROZEN,
                    MASK_BED_STREAM, MASK_BED_GRLINE, MASK_BED_FLOAT,
                    MASK_BED_PARTIAL

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

# Per-side `mask_frnt` enum values, matching the integer parameters
# in `physics/topography.f90:calc_ice_front:555-558`. The Fortran
# routine reuses `+1` for both floating and marine fronts (with a
# commented-out `+2 = marine` alternative); we follow the active
# numbering.
const _MASK_FRNT_ICE_FREE  = -1.0
const _MASK_FRNT_FLOATING  =  1.0
const _MASK_FRNT_MARINE    =  1.0   # same value as FLOATING in Fortran
const _MASK_FRNT_GROUNDED  =  3.0
const _MASK_FRNT_INTERIOR  =  0.0

"""
    calc_ice_front!(mask_frnt, f_ice, f_grnd, z_bed, z_sl) -> mask_frnt

Mark the ice-front cells of the domain with an integer-valued mask:

| value | meaning                                                  |
|------:|----------------------------------------------------------|
|  `-1` | ice-free cell adjacent to a fully-covered front cell     |
|   `0` | interior cell (no front nearby)                          |
|  `+1` | floating ice front, *or* marine grounded ice front       |
|  `+3` | grounded ice front above sea level                       |

A *front* cell is a fully-ice-covered cell (`f_ice == 1`) that has at
least one direct (4-conn) neighbour with `f_ice < 1`. The cell type
key (floating / marine / grounded) is set from `f_grnd` and the bed
elevation vs sea level. Adjacent ice-free cells are marked `-1`.

Halo handling: `f_ice` halos are filled via `fill_halo_regions!` so
front detection at domain edges respects the grid's topology + BCs.
The mark-adjacent-cells pass writes into the interior only; the
ice-free flag does not propagate through the halo.

Port of `physics/topography.f90:533 calc_ice_front`.
"""
function calc_ice_front!(mask_frnt, f_ice, f_grnd, z_bed, z_sl)
    Mf = @view interior(mask_frnt)[:, :, 1]
    Fg = @view interior(f_grnd)[:, :, 1]
    Zb = @view interior(z_bed)[:, :, 1]
    Zs = @view interior(z_sl)[:, :, 1]
    nx, ny = size(Mf)

    fill_halo_regions!(f_ice)

    # Initialise to interior (0). The kernel below writes `-1` /
    # `+1` / `+3` only at relevant cells.
    fill!(Mf, _MASK_FRNT_INTERIOR)

    @inbounds for j in 1:ny, i in 1:nx
        # Centre cell must be fully ice-covered to be a front.
        f_ice[i, j, 1] == 1.0 || continue

        fW = f_ice[i-1, j,   1]
        fE = f_ice[i+1, j,   1]
        fS = f_ice[i,   j-1, 1]
        fN = f_ice[i,   j+1, 1]

        # `<` not `==` matches the Fortran `f_neighb .lt. 1.0`.
        n_open = (fW < 1.0 ? 1 : 0) + (fE < 1.0 ? 1 : 0) +
                 (fS < 1.0 ? 1 : 0) + (fN < 1.0 ? 1 : 0)
        n_open == 0 && continue

        # Classify the front cell type.
        if Fg[i, j] > 0.0
            Mf[i, j] = (Zs[i, j] <= Zb[i, j]) ?
                       _MASK_FRNT_GROUNDED : _MASK_FRNT_MARINE
        else
            Mf[i, j] = _MASK_FRNT_FLOATING
        end

        # Mark the ice-free neighbours `-1`. Don't write through the
        # halo — only interior cells.
        if fW < 1.0 && i > 1
            Mf[i-1, j] = _MASK_FRNT_ICE_FREE
        end
        if fE < 1.0 && i < nx
            Mf[i+1, j] = _MASK_FRNT_ICE_FREE
        end
        if fS < 1.0 && j > 1
            Mf[i, j-1] = _MASK_FRNT_ICE_FREE
        end
        if fN < 1.0 && j < ny
            Mf[i, j+1] = _MASK_FRNT_ICE_FREE
        end
    end
    return mask_frnt
end
