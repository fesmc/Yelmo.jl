# ----------------------------------------------------------------------
# Per-region scalar diagnostics (port of Fortran
# `yelmo/src/yelmo_regions.f90:208 yelmo_calc_region`).
#
# Computes ~38 mean / sum / max diagnostics over three sub-masks
# derived from `mask`:
#
#   mask_tot  = mask & (H_ice  > 0)
#   mask_grnd = mask & (H_ice  > 0) & (f_grnd  > 0)
#   mask_flt  = mask & (H_ice  > 0) & (f_grnd == 0)
#
# When a sub-mask has zero active cells, the corresponding fields
# are set to 0.0 (matches Fortran). The caller-supplied `mask` is
# assumed to be a 2D `Bool` matrix of shape (Nx, Ny).
#
# Conversion factors:
#
#   m^3 → km^3       :  1e-9
#   m^2 → km^2       :  1e-6
#   km^3 → m sle     :  (1e-3) / 394.7   (Fortran yelmo_boundaries.f90:74)
#   km^3/yr → Sv     :  1e-6 * (1e9 * rho_w / rho_ice) / sec_year
# ----------------------------------------------------------------------

const _M3_TO_KM3       = 1e-9
const _M2_TO_KM2       = 1e-6
const _KM3_TO_M_SLE    = 1.0e-3 / 394.7

# Helper: get cell areas dx, dy from the model.
@inline function _cell_dx_dy(y::YelmoModel)
    dx_g = y.g.Δxᶜᵃᵃ
    dy_g = y.g.Δyᵃᶜᵃ
    dx = abs(Float64(dx_g isa Number ? dx_g :
        error("calc_region_diagnostics!: stretched x-grid not supported.")))
    dy = abs(Float64(dy_g isa Number ? dy_g :
        error("calc_region_diagnostics!: stretched y-grid not supported.")))
    return dx, dy
end

# Mean of a 2D Center-field interior over a Bool mask. Returns 0.0
# when the mask has no active cells.
@inline function _masked_mean(F::AbstractArray{Float64,3},
                              mask::AbstractMatrix{Bool})
    Nx, Ny, _ = size(F)
    s = 0.0
    n = 0
    @inbounds for j in 1:Ny, i in 1:Nx
        if mask[i, j]
            s += F[i, j, 1]
            n += 1
        end
    end
    return n == 0 ? 0.0 : s / n
end

# Sum of a 2D Center-field interior over a Bool mask.
@inline function _masked_sum(F::AbstractArray{Float64,3},
                             mask::AbstractMatrix{Bool})
    Nx, Ny, _ = size(F)
    s = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        if mask[i, j]
            s += F[i, j, 1]
        end
    end
    return s
end

# Max of a 2D Center-field interior over a Bool mask. Returns 0.0
# when the mask has no active cells (matches Fortran which returns
# the implicit `0` since the variable is zeroed in the else-branch).
@inline function _masked_max(F::AbstractArray{Float64,3},
                             mask::AbstractMatrix{Bool})
    Nx, Ny, _ = size(F)
    m = -Inf
    n = 0
    @inbounds for j in 1:Ny, i in 1:Nx
        if mask[i, j]
            v = F[i, j, 1]
            if v > m
                m = v
            end
            n += 1
        end
    end
    return n == 0 ? 0.0 : m
end

# Count active cells in a Bool mask.
@inline function _count_mask(mask::AbstractMatrix{Bool})
    n = 0
    @inbounds for x in mask
        x && (n += 1)
    end
    return n
end

# Compute ice-thickness above flotation:
#   H_af = max(0, H_ice + min(0, z_bed - z_sl) * (rho_sw / rho_ice))
# Used for `V_sl` (volume above sea level). Mirrors Fortran
# `yelmo/src/physics/topography.f90 calc_H_af` with `use_f_ice = false`.
@inline function _calc_H_af(H_ice::Float64, z_bed::Float64, z_sl::Float64,
                            rho_ice::Float64, rho_sw::Float64)
    z_diff = min(0.0, z_bed - z_sl)
    H_af = H_ice + z_diff * (rho_sw / rho_ice)
    return max(0.0, H_af)
end

"""
    calc_region_diagnostics!(diag::RegionDiagnostics,
                             y::YelmoModel,
                             mask::AbstractMatrix{Bool}) -> diag

Recompute every field of `diag` from the current `y` state, using
`mask` as the user-supplied region selector. Mirrors Fortran
`yelmo_calc_region`.
"""
function calc_region_diagnostics!(diag::RegionDiagnostics,
                                  y::YelmoModel,
                                  mask::AbstractMatrix{Bool})
    Nx = size(y.tpo.H_ice, 1)
    Ny = size(y.tpo.H_ice, 2)
    size(mask) == (Nx, Ny) ||
        error("calc_region_diagnostics!: mask shape $(size(mask)) does not " *
              "match grid ($(Nx), $(Ny)).")

    dx, dy = _cell_dx_dy(y)
    cell_area_m2 = dx * dy

    # Pull interiors once.
    H_ice    = interior(y.tpo.H_ice)
    z_srf    = interior(y.tpo.z_srf)
    dHidt    = interior(y.tpo.dHidt)
    dzsdt    = interior(y.tpo.dzsdt)
    dmb      = interior(y.tpo.dmb)
    cmb      = interior(y.tpo.cmb)
    cmb_flt  = interior(y.tpo.cmb_flt)
    cmb_grnd = interior(y.tpo.cmb_grnd)
    bmb_2d   = interior(y.tpo.bmb)
    f_grnd   = interior(y.tpo.f_grnd)

    uxy_bar  = interior(y.dyn.uxy_bar)
    uxy_s    = interior(y.dyn.uxy_s)
    uxy_b    = interior(y.dyn.uxy_b)

    z_bed    = interior(y.bnd.z_bed)
    z_sl     = interior(y.bnd.z_sl)
    smb_ref  = interior(y.bnd.smb_ref)
    T_srf    = interior(y.bnd.T_srf)
    T_shlf   = interior(y.bnd.T_shlf)

    f_pmp    = interior(y.thrm.f_pmp)
    H_w      = interior(y.thrm.H_w)

    rho_ice  = y.c.rho_ice
    rho_sw   = y.c.rho_sw
    rho_w    = y.c.rho_w
    sec_year = y.c.sec_year

    conv_km3a_Sv = 1e-6 * (1e9 * rho_w / rho_ice) / sec_year

    # Build sub-masks. mask_tot has shape (Nx, Ny). We allocate three
    # local Bool matrices — small (Nx*Ny bytes) compared to the
    # field interiors, no need to scratchify yet.
    mask_tot  = falses(Nx, Ny)
    mask_grnd = falses(Nx, Ny)
    mask_flt  = falses(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if mask[i, j] && H_ice[i, j, 1] > 0.0
            mask_tot[i, j] = true
            if f_grnd[i, j, 1] > 0.0
                mask_grnd[i, j] = true
            else
                mask_flt[i, j]  = true
            end
        end
    end

    npts_tot  = _count_mask(mask_tot)
    npts_grnd = _count_mask(mask_grnd)
    npts_flt  = _count_mask(mask_flt)

    # ===== Total ice =====
    if npts_tot > 0
        diag.H_ice     = _masked_mean(H_ice, mask_tot)
        diag.z_srf     = _masked_mean(z_srf, mask_tot)
        diag.dHidt     = _masked_mean(dHidt, mask_tot)
        diag.H_ice_max = _masked_max(H_ice, mask_tot)
        diag.dzsdt     = _masked_mean(dzsdt, mask_tot)

        sum_H = _masked_sum(H_ice, mask_tot)
        diag.V_ice = sum_H * cell_area_m2 * _M3_TO_KM3            # km^3
        diag.A_ice = npts_tot * cell_area_m2 * _M2_TO_KM2          # km^2
        diag.dVidt = _masked_sum(dHidt, mask_tot) * cell_area_m2 * _M3_TO_KM3
        diag.fwf   = -diag.dVidt * conv_km3a_Sv

        diag.dmb      = _masked_sum(dmb,      mask_tot) * cell_area_m2  # m^3/yr
        diag.cmb      = _masked_sum(cmb,      mask_tot) * cell_area_m2
        diag.cmb_flt  = _masked_sum(cmb_flt,  mask_tot) * cell_area_m2
        diag.cmb_grnd = _masked_sum(cmb_grnd, mask_tot) * cell_area_m2

        # Volume above flotation (V_sl) — sum H_af over mask_tot.
        s_Haf = 0.0
        @inbounds for j in 1:Ny, i in 1:Nx
            if mask_tot[i, j]
                s_Haf += _calc_H_af(H_ice[i, j, 1], z_bed[i, j, 1],
                                    z_sl[i, j, 1], rho_ice, rho_sw)
            end
        end
        diag.V_sl  = s_Haf * cell_area_m2 * _M3_TO_KM3
        diag.V_sle = diag.V_sl * _KM3_TO_M_SLE

        diag.uxy_bar = _masked_mean(uxy_bar, mask_tot)
        diag.uxy_s   = _masked_mean(uxy_s,   mask_tot)
        diag.uxy_b   = _masked_mean(uxy_b,   mask_tot)

        diag.z_bed = _masked_mean(z_bed,   mask_tot)
        diag.smb   = _masked_mean(smb_ref, mask_tot)
        diag.T_srf = _masked_mean(T_srf,   mask_tot)
        diag.bmb   = _masked_mean(bmb_2d,  mask_tot)
    else
        diag.H_ice = 0.0; diag.z_srf = 0.0; diag.dHidt = 0.0
        diag.H_ice_max = 0.0; diag.dzsdt = 0.0
        diag.V_ice = 0.0; diag.A_ice = 0.0; diag.dVidt = 0.0; diag.fwf = 0.0
        diag.dmb = 0.0; diag.cmb = 0.0; diag.cmb_flt = 0.0; diag.cmb_grnd = 0.0
        diag.V_sl = 0.0; diag.V_sle = 0.0
        diag.uxy_bar = 0.0; diag.uxy_s = 0.0; diag.uxy_b = 0.0
        diag.z_bed = 0.0; diag.smb = 0.0; diag.T_srf = 0.0; diag.bmb = 0.0
    end

    # ===== Grounded ice =====
    if npts_grnd > 0
        diag.H_ice_g  = _masked_mean(H_ice, mask_grnd)
        diag.z_srf_g  = _masked_mean(z_srf, mask_grnd)
        diag.V_ice_g  = _masked_sum(H_ice, mask_grnd) * cell_area_m2 * _M3_TO_KM3
        diag.A_ice_g  = npts_grnd * cell_area_m2 * _M2_TO_KM2
        diag.uxy_bar_g = _masked_mean(uxy_bar, mask_grnd)
        diag.uxy_s_g   = _masked_mean(uxy_s,   mask_grnd)
        diag.uxy_b_g   = _masked_mean(uxy_b,   mask_grnd)
        diag.f_pmp     = _masked_mean(f_pmp,   mask_grnd)
        diag.H_w       = _masked_mean(H_w,     mask_grnd)
        diag.bmb_g     = _masked_mean(bmb_2d,  mask_grnd)
    else
        diag.H_ice_g = 0.0; diag.z_srf_g = 0.0
        diag.V_ice_g = 0.0; diag.A_ice_g = 0.0
        diag.uxy_bar_g = 0.0; diag.uxy_s_g = 0.0; diag.uxy_b_g = 0.0
        diag.f_pmp = 0.0; diag.H_w = 0.0; diag.bmb_g = 0.0
    end

    # ===== Floating ice =====
    if npts_flt > 0
        diag.H_ice_f  = _masked_mean(H_ice, mask_flt)
        diag.V_ice_f  = _masked_sum(H_ice, mask_flt) * cell_area_m2 * _M3_TO_KM3
        diag.A_ice_f  = npts_flt * cell_area_m2 * _M2_TO_KM2
        diag.uxy_bar_f = _masked_mean(uxy_bar, mask_flt)
        diag.uxy_s_f   = _masked_mean(uxy_s,   mask_flt)
        diag.uxy_b_f   = _masked_mean(uxy_b,   mask_flt)
        diag.z_sl      = _masked_mean(z_sl,    mask_flt)
        diag.bmb_shlf  = _masked_mean(bmb_2d,  mask_flt)
        diag.T_shlf    = _masked_mean(T_shlf,  mask_flt)
    else
        diag.H_ice_f = 0.0
        diag.V_ice_f = 0.0; diag.A_ice_f = 0.0
        diag.uxy_bar_f = 0.0; diag.uxy_s_f = 0.0; diag.uxy_b_f = 0.0
        diag.z_sl = 0.0; diag.bmb_shlf = 0.0; diag.T_shlf = 0.0
    end

    return diag
end
