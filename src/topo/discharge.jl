# ----------------------------------------------------------------------
# Subgrid discharge mass balance (DMB).
#
# Stub for milestone 2c: only `dmb_method == 0` (no discharge) is
# wired through. The full Calov+ 2015 implementation
# (`physics/discharge.f90:calc_mb_discharge`) requires `dist_grline`
# and `dist_margin` distance-to-feature fields, which are not yet
# computed on the Julia side. Calling with any other method raises a
# deferred-implementation error.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calc_mb_discharge!

"""
    calc_mb_discharge!(dmb, H_ice, z_srf, z_bed_sd, dist_grline,
                       dist_margin, f_ice, method, dx, alpha_max,
                       tau_mbd, sigma_ref, m_d, m_r) -> dmb

Subgrid calving discharge rate (Calov+ 2015).

  - `method == 0`: zero out `dmb` (no discharge). The arguments other
    than `dmb` are unused on this branch.
  - `method != 0`: raises an error — the full kernel needs distance
    fields (`dist_grline`, `dist_margin`) that the v1 port has not
    yet computed.

Signature mirrors `physics/discharge.f90:calc_mb_discharge` for
forward compatibility; once the distance fields land, dropping in
the full body is a self-contained follow-up.
"""
function calc_mb_discharge!(dmb,
                            H_ice, z_srf, z_bed_sd,
                            dist_grline, dist_margin, f_ice,
                            method::Integer,
                            dx::Real,
                            alpha_max::Real,
                            tau_mbd::Real,
                            sigma_ref::Real,
                            m_d::Real,
                            m_r::Real)
    if method == 0
        fill!(interior(dmb), 0.0)
        return dmb
    end

    error("calc_mb_discharge!: dmb_method=$method not yet ported. " *
          "The Calov+ 2015 kernel requires dist_grline / dist_margin " *
          "fields which are not yet computed in Yelmo.jl. Set " *
          "ytopo.dmb_method = 0 (default) to disable.")
end
