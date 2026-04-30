# ----------------------------------------------------------------------
# Surface elevation diagnostics.
#
#   - `calc_z_srf!` — per-cell surface elevation from `H_ice`, `f_ice`,
#     `z_bed`, `z_sl` using the Pattyn (2017, Eq. 1) max-of-grounded-or-
#     floating formula. Sub-grid `f_ice < 1` cells collapse the
#     effective thickness to zero (the cell's surface is the bare
#     bed/sea-level).
#
# `z_base = z_srf - H_ice` lives in `_update_diagnostics!` directly —
# it's a single line and depends on the just-computed `z_srf`.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calc_z_srf!

"""
    calc_z_srf!(z_srf, H_ice, f_ice, z_bed, z_sl, rho_ice, rho_sw) -> z_srf

Per-cell surface elevation:

    z_srf = max(z_bed + H_eff,  z_sl + (1 - rho_ice/rho_sw) · H_eff)

where the effective ice thickness is:

  - `H_eff = H_ice / f_ice` when `f_ice ≥ 1` (fully ice-covered)
  - `H_eff = 0`             when `f_ice < 1` (collapse partial /
    ice-free cells to the bare bed-or-sea-level surface)

The first branch dominates for grounded ice (where the bed sits above
flotation), the second for floating ice. For ice-free cells (`f_ice = 0`),
both terms reduce to `max(z_bed, z_sl)`.

Port of `physics/topography.f90:655 calc_z_srf_max` — the variant
selected by `calc_ytopo_diagnostic` for the standard (non-sub-grid)
GL configuration.
"""
function calc_z_srf!(z_srf, H_ice, f_ice, z_bed, z_sl,
                     rho_ice::Real, rho_sw::Real)
    Z   = interior(z_srf)
    H   = interior(H_ice)
    Fi  = interior(f_ice)
    Zb  = interior(z_bed)
    Zsl = interior(z_sl)

    rho_ice_sw = rho_ice / rho_sw

    @inbounds for j in axes(Z, 2), i in axes(Z, 1)
        H_eff = Fi[i, j, 1] >= 1.0 ? H[i, j, 1] / Fi[i, j, 1] : 0.0
        zb    = Zb[i, j, 1]
        zsl   = Zsl[i, j, 1]
        Z[i, j, 1] = max(zb + H_eff, zsl + (1.0 - rho_ice_sw) * H_eff)
    end
    return z_srf
end
