# ----------------------------------------------------------------------
# Lateral boundary stress at the ice front.
#
# `calc_lateral_bc_stress_2D!` computes the depth-integrated lateral
# stress (`taul_int_acx`, `taul_int_acy` — units: Pa·m) on staggered
# ac-faces wherever a fully-covered front cell `mask_frnt > 0` borders
# an ice-free neighbour `mask_frnt < 0`. Elsewhere the field is zero.
#
# The face-value formula at a marine/floating front (Lipscomb et al.
# 2019, Eqs. 11–12; Winkelmann et al. 2011, Eq. 27) is:
#
#     τ_bc_int = ½ρ_i g H_ice² − ½ρ_w g H_ocn²
#
# where `H_ocn = H_ice · (1 − min((z_srf − z_sl)/H_ice, 1))` is the
# submerged ocean column at the cell. For a fully-grounded front above
# sea level (`z_srf > z_sl`) the second term collapses to zero.
#
# Indexing follows `calc_driving_stress!`: Fortran face `(i, j)` ↔
# Julia `interior(taul_int_acx)[i+1, j, 1]` for x-faces (likewise y).
#
# Port of `velocity_general.f90:1450 calc_lateral_bc_stress_2D` and
# `:1565 calc_lateral_bc_stress`.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.BoundaryConditions: fill_halo_regions!

export calc_lateral_bc_stress_2D!

# Per-cell depth-integrated lateral stress at the ice front. `H_ice`,
# `z_srf`, `z_sl` are the centred-cell values at the front cell;
# returns `tau_bc_int` in Pa·m. Mirrors `calc_lateral_bc_stress`.
@inline function _calc_lateral_bc_stress(H_ice::Float64, z_srf::Float64, z_sl::Float64,
                                         rho_ice::Float64, rho_sw::Float64, g::Float64)
    # Submerged fraction of the ice column. The `min(.., 1)` clamp
    # handles partially-grounded fronts where the surface sits above
    # sea level (in which case `f_submerged ≤ 0` and `H_ocn = 0`).
    f_submerged = 1.0 - min((z_srf - z_sl) / H_ice, 1.0)
    H_ocn       = H_ice * f_submerged
    return 0.5 * rho_ice * g * H_ice^2 -
           0.5 * rho_sw  * g * H_ocn^2
end

"""
    calc_lateral_bc_stress_2D!(taul_int_acx, taul_int_acy,
                               mask_frnt, H_ice, f_ice,
                               z_srf, z_sl,
                               rho_ice, rho_sw, g)
        -> (taul_int_acx, taul_int_acy)

Compute the depth-integrated lateral boundary stress on staggered
ac-faces. The acx-face between cells `(i, j)` and `(i+1, j)` is set to
`_calc_lateral_bc_stress` at the front cell exactly when the two cells
straddle a front: `(mask_frnt[i,j] > 0 ∧ mask_frnt[i+1,j] < 0)` or the
reverse. All non-front faces are zeroed. The acy direction is
analogous.

`mask_frnt` follows the encoding from `calc_ice_front!`:
`+1`/`+3` = front, `-1` = ice-free margin neighbour, `0` = interior.

Port of `velocity_general.f90:1450 calc_lateral_bc_stress_2D`.
"""
function calc_lateral_bc_stress_2D!(taul_int_acx, taul_int_acy,
                                    mask_frnt, H_ice, f_ice,
                                    z_srf, z_sl,
                                    rho_ice::Real, rho_sw::Real, g::Real)
    # Halos: the kernel reads (i+1, j) / (i, j+1) of mask_frnt and the
    # selected front cell's H_ice / z_srf / z_sl. f_ice is unused by
    # the active branch (the commented-out f_ice variant is dead in
    # Fortran) but kept in the signature for parity.
    fill_halo_regions!(mask_frnt)
    fill_halo_regions!(H_ice)
    fill_halo_regions!(z_srf)
    fill_halo_regions!(z_sl)

    rho_ice_f = Float64(rho_ice)
    rho_sw_f  = Float64(rho_sw)
    g_f       = Float64(g)

    Tx = interior(taul_int_acx)
    Ty = interior(taul_int_acy)
    Nx = size(interior(mask_frnt), 1)
    Ny = size(interior(mask_frnt), 2)

    # Reset the entire interior — the Fortran initialises to zero and
    # only writes at front faces.
    fill!(Tx, 0.0)
    fill!(Ty, 0.0)

    @inbounds for j in 1:Ny, i in 1:Nx
        m0 = mask_frnt[i,   j, 1]
        mE = mask_frnt[i+1, j, 1]
        mN = mask_frnt[i,   j+1, 1]

        # x-direction: front face between (i, j) and (i+1, j).
        if (m0 > 0.0 && mE < 0.0) || (m0 < 0.0 && mE > 0.0)
            i1 = m0 < 0.0 ? i + 1 : i
            Tx[i+1, j, 1] = _calc_lateral_bc_stress(
                Float64(H_ice[i1, j, 1]),
                Float64(z_srf[i1, j, 1]),
                Float64(z_sl[i1, j, 1]),
                rho_ice_f, rho_sw_f, g_f)
        end

        # y-direction: front face between (i, j) and (i, j+1).
        if (m0 > 0.0 && mN < 0.0) || (m0 < 0.0 && mN > 0.0)
            j1 = m0 < 0.0 ? j + 1 : j
            Ty[i, j+1, 1] = _calc_lateral_bc_stress(
                Float64(H_ice[i, j1, 1]),
                Float64(z_srf[i, j1, 1]),
                Float64(z_sl[i, j1, 1]),
                rho_ice_f, rho_sw_f, g_f)
        end
    end

    # Replicate first-face slot per the YelmoMirror loader convention.
    @views Tx[1, :, :] .= Tx[2, :, :]
    @views Ty[:, 1, :] .= Ty[:, 2, :]

    return taul_int_acx, taul_int_acy
end
