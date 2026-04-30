# ----------------------------------------------------------------------
# Frontal mass balance (FMB) at the marine ice front.
#
# `calc_fmb_total!` computes the per-cell frontal melt rate that arises
# at the lateral boundary between marine ice and open ocean. Three
# methods, mirroring `physics/topography.f90:1658:calc_fmb_total`:
#
#   - method 0: pass-through `fmb = fmb_shlf` (boundary forcing).
#   - method 1: scale `bmb_shlf` of ice-free neighbours by the area
#     of the cell's submerged front face.
#   - method 2: scale `fmb_shlf` of the cell by the same submerged-
#     front-face area ratio.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calc_fmb_total!

# Effective ice thickness — `H_ice / f_ice` for partially-covered cells,
# `H_ice` otherwise. Mirrors `calc_H_eff` in
# `physics/topography.f90:781`.
@inline function _calc_H_eff(H_ice::Real, f_ice::Real)
    return f_ice > 0.0 ? H_ice / f_ice : H_ice
end

# Saturated lookup: `H_ice` at (i,j) if in-bounds, else 0.
@inline function _h_ice_or_zero(H::AbstractArray, i::Int, j::Int,
                                nx::Int, ny::Int)
    return (1 <= i <= nx && 1 <= j <= ny) ? H[i, j, 1] : 0.0
end

@inline function _bmb_or_zero(B::AbstractArray, i::Int, j::Int,
                              nx::Int, ny::Int)
    return (1 <= i <= nx && 1 <= j <= ny) ? B[i, j, 1] : 0.0
end

"""
    calc_fmb_total!(fmb, fmb_shlf, bmb_shlf, H_ice, H_grnd, f_ice,
                    fmb_method, fmb_scale, rho_ice, rho_sw, dx) -> fmb

Compute the frontal mass balance `fmb` (m/yr) for marine-ice cells at
the ice front. The per-cell rate vanishes for cells that are either
land-grounded (`H_grnd ≥ H_ice` ⇒ above flotation), ice-free, or
without an ice-free neighbour.

  - `fmb_method == 0` (pass-through): `fmb = fmb_shlf` everywhere.
  - `fmb_method == 1`: scale the mean `bmb_shlf` over ice-free
    neighbours by the area-of-front fraction `n_margin*dz*dx /
    (dx*dx)`, then by `fmb_scale`.
  - `fmb_method == 2`: scale the local `fmb_shlf` by the same area
    fraction.

Where `dz` is the submerged front depth: `H_eff * rho_ice/rho_sw` for
floating cells, `max(H_eff - H_grnd, 0) * rho_ice/rho_sw` for grounded
marine cells.

Out-of-domain neighbour reads of `H_ice` resolve to 0 (open ocean), so
boundary cells correctly count the domain-edge as an ice-free
neighbour.

Port of `physics/topography.f90:1658:calc_fmb_total`.
"""
function calc_fmb_total!(fmb, fmb_shlf, bmb_shlf,
                         H_ice, H_grnd, f_ice,
                         fmb_method::Integer,
                         fmb_scale::Real,
                         rho_ice::Real, rho_sw::Real,
                         dx::Real)
    F   = interior(fmb)
    Fs  = interior(fmb_shlf)
    Bs  = interior(bmb_shlf)
    H   = interior(H_ice)
    Hg  = interior(H_grnd)
    Fi  = interior(f_ice)

    nx = size(F, 1)
    ny = size(F, 2)

    rho_ratio_iw = rho_ice / rho_sw
    area_tot = dx * dx

    if fmb_method == 0
        @inbounds for j in 1:ny, i in 1:nx
            F[i, j, 1] = Fs[i, j, 1]
        end
        return fmb
    end

    if fmb_method != 1 && fmb_method != 2
        error("calc_fmb_total!: unknown fmb_method=$fmb_method. " *
              "Supported: 0 (pass-through), 1 (bmb_shlf-scaled), 2 (fmb_shlf-scaled).")
    end

    @inbounds for j in 1:ny, i in 1:nx
        h_here  = H[i, j, 1]
        hg_here = Hg[i, j, 1]
        fi_here = Fi[i, j, 1]

        # Marine ice front: ice-covered, [floating or grounded below
        # sea level], and bordering ≥ 1 ice-free cell.
        marine_ice = (h_here > 0.0) && (hg_here < h_here)

        if !marine_ice
            F[i, j, 1] = 0.0
            continue
        end

        nW = _h_ice_or_zero(H, i-1, j,   nx, ny)
        nE = _h_ice_or_zero(H, i+1, j,   nx, ny)
        nS = _h_ice_or_zero(H, i,   j-1, nx, ny)
        nN = _h_ice_or_zero(H, i,   j+1, nx, ny)

        n_margin = (nW == 0.0 ? 1 : 0) + (nE == 0.0 ? 1 : 0) +
                   (nS == 0.0 ? 1 : 0) + (nN == 0.0 ? 1 : 0)

        if n_margin == 0
            F[i, j, 1] = 0.0
            continue
        end

        H_eff = _calc_H_eff(h_here, fi_here)

        # Submerged-front depth: floating ⇒ full ice draft;
        # grounded marine ⇒ depth above the bed-flotation reference.
        dz = if hg_here < 0.0
            H_eff * rho_ratio_iw
        else
            max((H_eff - hg_here) * rho_ratio_iw, 0.0)
        end

        area_flt = n_margin * dz * dx

        if fmb_method == 1
            bmb_shlf_W = _bmb_or_zero(Bs, i-1, j,   nx, ny)
            bmb_shlf_E = _bmb_or_zero(Bs, i+1, j,   nx, ny)
            bmb_shlf_S = _bmb_or_zero(Bs, i,   j-1, nx, ny)
            bmb_shlf_N = _bmb_or_zero(Bs, i,   j+1, nx, ny)
            bmb_sum = (nW == 0.0 ? bmb_shlf_W : 0.0) +
                      (nE == 0.0 ? bmb_shlf_E : 0.0) +
                      (nS == 0.0 ? bmb_shlf_S : 0.0) +
                      (nN == 0.0 ? bmb_shlf_N : 0.0)
            bmb_eff = bmb_sum / n_margin
            F[i, j, 1] = bmb_eff * (area_flt / area_tot) * fmb_scale
        else  # fmb_method == 2
            F[i, j, 1] = Fs[i, j, 1] * (area_flt / area_tot)
        end
    end

    return fmb
end
