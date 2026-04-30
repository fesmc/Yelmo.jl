# ----------------------------------------------------------------------
# Calving-rate laws on ac-nodes.
#
# Each law fills a pair of staggered fields `(cr_x, cr_y)` with a
# horizontal calving-front velocity in m/yr, signed *opposite* to the
# ice flow (so a positive front velocity acts as retreat). Outputs feed
# `merge_calving_rates!`, which combines floating and marine-grounded
# rates into the single `(cr_acx, cr_acy)` used to advect the level-set
# function `lsf` at `w = u_bar + cr`.
#
# Laws ported from `yelmo/src/physics/calving/calving_ac.f90`:
#
#   - `calc_calving_equil_ac!`     — `cr = -u_bar` (front-fixing)
#   - `calc_calving_threshold_ac!` — ice-thickness threshold (Hc)
#   - `calc_calving_vonmises_m16_ac!` — Morlighem et al. (2016)
#     stress-based; not yet wired through `mat`, errors at call.
#
# `merge_calving_rates!` ports the merge / above-SL pin block from
# `yelmo_topography.f90:744-782`.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.BoundaryConditions: fill_halo_regions!

export calc_calving_equil_ac!,
       calc_calving_threshold_ac!,
       calc_calving_vonmises_m16_ac!,
       merge_calving_rates!

"""
    calc_calving_equil_ac!(cr_x, cr_y, u_bar, v_bar) -> (cr_x, cr_y)

Equilibrium calving law: `cr = -u_bar`. Pinning the calving front in
place is the intended use — combined with `merge_calving_rates!` and
`lsf_update!`, the net front velocity `w = u_bar + cr ≡ 0`, so `lsf`
is stationary.
"""
function calc_calving_equil_ac!(cr_x, cr_y, u_bar, v_bar)
    Cx = interior(cr_x)
    Cy = interior(cr_y)
    Ux = interior(u_bar)
    Uy = interior(v_bar)
    @inbounds for j in axes(Cx, 2), i in axes(Cx, 1)
        Cx[i, j, 1] = -Ux[i, j, 1]
    end
    @inbounds for j in axes(Cy, 2), i in axes(Cy, 1)
        Cy[i, j, 1] = -Uy[i, j, 1]
    end
    return cr_x, cr_y
end

"""
    calc_calving_threshold_ac!(cr_x, cr_y, u_bar, v_bar, H_ice, f_ice, Hc)
        -> (cr_x, cr_y)

Ice-thickness threshold calving law (port of
`calc_calving_threshold_lsf`). The calving rate ramps linearly from
zero at `H ≥ 2·Hc` to `-u_bar` at `H = Hc` to `-2·u_bar` at `H = 0`:

    cr_x = -u_bar · max(0, 1 + (Hc − H_acx)/Hc)

`H_acx` / `H_acy` are the ice thickness staggered onto the ac-face. At
ice/ocean borders the staggering picks the ice side directly so that
ocean cells (where `H = 0`) cannot drag the face value to zero.

`Hc` is the reference thickness (`ycalv.Hc_ref_flt` for floating,
`Hc_ref_grnd` for grounded). `H_ice` and `f_ice` live on aa-nodes.
"""
function calc_calving_threshold_ac!(cr_x, cr_y,
                                    u_bar, v_bar,
                                    H_ice, f_ice,
                                    Hc::Real)
    Hc > 0 || error("calc_calving_threshold_ac!: Hc must be > 0 (got $Hc).")

    Cx = interior(cr_x)
    Cy = interior(cr_y)
    Ux = interior(u_bar)
    Uy = interior(v_bar)

    fill_halo_regions!(H_ice)
    fill_halo_regions!(f_ice)

    # XFaceField indexing: face `i` sits between centres `i-1` and `i`.
    @inbounds for j in axes(Cx, 2), i in axes(Cx, 1)
        H_l = H_ice[i - 1, j, 1]
        H_r = H_ice[i,     j, 1]
        F_l = f_ice[i - 1, j, 1]
        F_r = f_ice[i,     j, 1]
        H_acx = if F_l > 0.0 && F_r == 0.0
            H_l                              # ice → ocean: pick ice
        elseif F_l == 0.0 && F_r > 0.0
            H_r                              # ocean → ice: pick ice
        else
            0.5 * (H_l + H_r)
        end
        wv = max(0.0, 1.0 + (Hc - H_acx) / Hc)
        Cx[i, j, 1] = -Ux[i, j, 1] * wv
    end

    # YFaceField indexing: face `j` sits between centres `j-1` and `j`.
    @inbounds for j in axes(Cy, 2), i in axes(Cy, 1)
        H_d = H_ice[i, j - 1, 1]
        H_u = H_ice[i, j,     1]
        F_d = f_ice[i, j - 1, 1]
        F_u = f_ice[i, j,     1]
        H_acy = if F_d > 0.0 && F_u == 0.0
            H_d
        elseif F_d == 0.0 && F_u > 0.0
            H_u
        else
            0.5 * (H_d + H_u)
        end
        wv = max(0.0, 1.0 + (Hc - H_acy) / Hc)
        Cy[i, j, 1] = -Uy[i, j, 1] * wv
    end
    return cr_x, cr_y
end

"""
    calc_calving_vonmises_m16_ac!(cr_x, cr_y, u_bar, v_bar, tau_1, f_ice, tau_ice)
        -> (cr_x, cr_y)

Morlighem et al. (2016) von-Mises calving (port of
`calc_calving_rate_vonmises_m16`):

    cr_x = -u_bar · max(0, tau1_acx / tau_ice)

Requires the 1st principal stress `tau_1` from `mat`. Not yet wired
in the Yelmo.jl port — calling this errors. Stub kept here so the
dispatch in `calving_step!` can route to it once `mat` lands.
"""
function calc_calving_vonmises_m16_ac!(cr_x, cr_y,
                                       u_bar, v_bar,
                                       tau_1, f_ice,
                                       tau_ice::Real)
    error("calc_calving_vonmises_m16_ac!: vm-m16 calving requires `mat` " *
          "(1st principal stress `tau_1`), which is not yet ported. " *
          "Use `calv_flt_method = \"threshold\"` or `\"equil\"`.")
end

"""
    merge_calving_rates!(cr_acx, cr_acy,
                         cmb_flt_acx, cmb_flt_acy,
                         cmb_grnd_acx, cmb_grnd_acy,
                         u_bar, v_bar,
                         f_grnd_acx, f_grnd_acy,
                         z_bed, z_sl) -> (cr_acx, cr_acy)

Combine the per-direction floating and grounded calving rates into
the single merged front velocity `(cr_acx, cr_acy)` used to advect
`lsf`. Port of the merge block at
`yelmo/src/physics/calving/calving_ac.f90` callers in
`yelmo_topography.f90:744-782`.

Per ac-face:

  - if `f_grnd_ac == 0` (floating face):    use `cmb_flt_*`
  - else if mean bed elevation across the face is above sea level:
    pin the front by overriding `cr = -u_bar` (so `w = u + cr = 0`)
  - else (marine-terminating grounded face): use `cmb_grnd_*`

The above-SL pin matches Fortran lines 758–774; without it,
land-terminating grounded ice would be allowed to advance into
above-sea-level cells via the LSF advection, which is unphysical.
"""
function merge_calving_rates!(cr_acx, cr_acy,
                              cmb_flt_acx, cmb_flt_acy,
                              cmb_grnd_acx, cmb_grnd_acy,
                              u_bar, v_bar,
                              f_grnd_acx, f_grnd_acy,
                              z_bed, z_sl)
    Cx  = interior(cr_acx)
    Cy  = interior(cr_acy)
    Fx  = interior(cmb_flt_acx)
    Fy  = interior(cmb_flt_acy)
    Gx  = interior(cmb_grnd_acx)
    Gy  = interior(cmb_grnd_acy)
    Ux  = interior(u_bar)
    Uy  = interior(v_bar)
    Gax = interior(f_grnd_acx)
    Gay = interior(f_grnd_acy)

    fill_halo_regions!(z_bed)
    fill_halo_regions!(z_sl)

    # x-face merge: face `i` between centres `i-1` and `i`.
    @inbounds for j in axes(Cx, 2), i in axes(Cx, 1)
        if Gax[i, j, 1] == 0.0
            Cx[i, j, 1] = Fx[i, j, 1]
        else
            zb_face  = 0.5 * (z_bed[i - 1, j, 1] + z_bed[i, j, 1])
            zsl_face = 0.5 * (z_sl[i - 1, j, 1]  + z_sl[i, j, 1])
            Cx[i, j, 1] = zb_face > zsl_face ? -Ux[i, j, 1] : Gx[i, j, 1]
        end
    end

    # y-face merge: face `j` between centres `j-1` and `j`.
    @inbounds for j in axes(Cy, 2), i in axes(Cy, 1)
        if Gay[i, j, 1] == 0.0
            Cy[i, j, 1] = Fy[i, j, 1]
        else
            zb_face  = 0.5 * (z_bed[i, j - 1, 1] + z_bed[i, j, 1])
            zsl_face = 0.5 * (z_sl[i, j - 1, 1]  + z_sl[i, j, 1])
            Cy[i, j, 1] = zb_face > zsl_face ? -Uy[i, j, 1] : Gy[i, j, 1]
        end
    end
    return cr_acx, cr_acy
end
