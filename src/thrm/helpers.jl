# ----------------------------------------------------------------------
# Small per-column helpers used by the implicit `temp`/`enth` solvers
# and by the bedrock + diagnostic post-pass.
#
# Direct ports:
#
#   - `_calc_wtd_harmonic_mean` ← `calc_wtd_harmonic_mean`
#     (`physics/ice_enthalpy.f90:1088`). Weighted harmonic mean of two
#     scalars, used to combine kappa from neighbouring aa-nodes onto
#     an ac-node interface.
#   - `_get_cts_index`           ← `get_cts_index`
#     (`physics/ice_enthalpy.f90:1244`). Highest k for which the
#     column is at-or-above PMP enthalpy (1 means cold base, nz means
#     fully temperate, 0 not used in the Fortran convention).
#   - `_calc_cts_height_column`  ← `calc_cts_height`
#     (`physics/ice_enthalpy.f90:933`). Linear-interpolation CTS
#     height between the highest temperate aa-node and the layer above.
#   - `_calc_bmb_grounded`       ← `calc_bmb_grounded`
#     (`physics/thermodynamics.f90:51`). Per-cell basal mass balance
#     from the heat-flux net at the base.
# ----------------------------------------------------------------------

"""
    _calc_wtd_harmonic_mean(var1, var2, wt1, wt2) -> avg

Weighted harmonic mean. Faithful port of Fortran's expression — does
not include the `+ tol` softening that's commented out in the source.
"""
@inline function _calc_wtd_harmonic_mean(var1::Float64, var2::Float64,
                                         wt1::Float64,  wt2::Float64)
    return ((wt1 / var1 + wt2 / var2) / (wt1 + wt2))^(-1)
end

"""
    _get_cts_index(enth_col, enth_pmp_col) -> k_cts

Return the largest k (1-based) for which `enth_col[k] >= enth_pmp_col[k]`,
counting upward from the base. Returns 1 if even the base is cold (or
mostly cold) per Fortran's loop initialisation; returns `nz` if every
layer is at-or-above PMP enthalpy.
"""
function _get_cts_index(enth_col::AbstractVector{Float64},
                        enth_pmp_col::AbstractVector{Float64})
    nz    = length(enth_col)
    k_cts = 1
    @inbounds for k in 1:nz
        if enth_col[k] >= enth_pmp_col[k]
            k_cts = k
        else
            break
        end
    end
    return k_cts
end

"""
    _calc_cts_height_column(enth_col, T_pmp_col, cp_col, H_ice, zeta_aa) -> H_cts

Per-column cold-temperate-transition surface height in metres. The
search starts from a per-layer enthalpy threshold `enth_pmp_col[k] =
T_pmp_col[k] * cp_col[k]` (water content zero at PMP). When the
highest temperate cell is fully interior, a linear interpolation
locates the CTS within the cell above.
"""
function _calc_cts_height_column(enth_col::AbstractVector{Float64},
                                 T_pmp_col::AbstractVector{Float64},
                                 cp_col::AbstractVector{Float64},
                                 H_ice::Float64,
                                 zeta_aa::Vector{Float64})
    # NOTE: `enth_col` etc. are views over halo-inclusive
    # OffsetArrays — `length(enth_col)` would include z-halos.
    # Use `length(zeta_aa)` which is the materialised Float64 vector
    # of interior z nodes.
    nz = length(zeta_aa)
    @inbounds begin
        # Find highest k with enth >= T_pmp*cp.
        k_cts = 1
        for k in 1:nz
            if enth_col[k] >= T_pmp_col[k] * cp_col[k]
                k_cts = k
            else
                if k == 1
                    k_cts = 0  # No temperate ice in the column.
                end
                break
            end
        end
        if k_cts == 0
            return 0.0
        elseif k_cts == nz
            return H_ice
        end
        e0     = enth_col[k_cts]
        e1     = enth_col[k_cts + 1]
        ep0    = T_pmp_col[k_cts]     * cp_col[k_cts]
        ep1    = T_pmp_col[k_cts + 1] * cp_col[k_cts + 1]
        denom  = (e1 - e0) - (ep1 - ep0)
        if denom != 0.0
            f_lin = (ep0 - e0) / denom
            f_lin < 1e-2 && (f_lin = 0.0)
        else
            f_lin = 1.0
        end
        zeta_cts = zeta_aa[k_cts] + f_lin * (zeta_aa[k_cts + 1] - zeta_aa[k_cts])
        return H_ice * zeta_cts
    end
end

"""
    _calc_bmb_grounded(T_prime_b, Q_ice_b_now, Q_b_now, Q_rock_now,
                       rho_ice, L_ice) -> bmb_grnd

Per-cell basal mass balance from the heat-flux net at the ice base
(Cuffey & Patterson 2010 Eq. 9.38). Heat fluxes in J a^-1 m^-2;
returns m/a (positive = accretion). Two safeties applied in order:

  1. If the basal temperature is more than 1 K below PMP and the
     formula returns net melt, force `bmb_grnd = 0` (rare init
     pathology; mirrors Fortran).
  2. Underflow guard: snap `|bmb_grnd| < 1e-5` to 0.
"""
# Extrapolate the thermodynamic state (enth / T_ice / omega / T_pmp)
# from fully-iced neighbours into ice-margin / ice-free cells. Fortran
# `calc_ytherm_enthalpy_3D:444-477`. The extrapolation gives the rate
# factor in `mat` (rf_method=1) reasonable starting values at cells
# that have just been advected ice, improving stability.
function _extrapolate_thrm_margin!(enth_d, T_d, om_d, Tp_d, fi_d,
                                   Nx::Int, Ny::Int, Nz::Int)
    @inbounds for j in 2:(Ny - 1), i in 2:(Nx - 1)
        if fi_d[i, j, 1] < 1.0
            # 3×3 neighbour count of fully-iced cells.
            wt_tot = 0.0
            for dj in -1:1, di in -1:1
                if fi_d[i + di, j + dj, 1] == 1.0
                    wt_tot += 1.0
                end
            end
            if wt_tot > 0.0
                inv_wt = 1.0 / wt_tot
                for k in 1:Nz
                    s_e = 0.0; s_T = 0.0; s_o = 0.0; s_p = 0.0
                    for dj in -1:1, di in -1:1
                        if fi_d[i + di, j + dj, 1] == 1.0
                            s_e += enth_d[i + di, j + dj, k]
                            s_T += T_d[i + di, j + dj, k]
                            s_o += om_d[i + di, j + dj, k]
                            s_p += Tp_d[i + di, j + dj, k]
                        end
                    end
                    enth_d[i, j, k] = s_e * inv_wt
                    T_d[i, j, k]    = s_T * inv_wt
                    om_d[i, j, k]   = s_o * inv_wt
                    Tp_d[i, j, k]   = s_p * inv_wt
                end
            end
        end
    end
    return nothing
end

# Marine-shelf basal temperature approximation (Jenkins 1991, modified
# to approach T_pmp as the grounding line is approached). Direct port
# of Fortran `calc_T_base_shlf_approx` (`thermodynamics.f90:1052`).
@inline function _calc_T_base_shlf_approx(H_ice::Float64, T_pmp::Float64,
                                          H_grnd::Float64, T0::Float64,
                                          rho_ice::Float64, rho_sw::Float64)
    a1 = -0.0575
    b1 =  0.0901
    c1 =  7.61e-4
    S0 = 34.75
    T_base_shlf = a1 * S0 + b1 + c1 * (rho_ice / rho_sw) * H_ice + T0
    H_grnd_lim  = -100.0
    f_scalar    = (H_grnd - H_grnd_lim) / H_grnd_lim
    f_scalar    = min(max(f_scalar, 0.0), 1.0)
    T_base_shlf = f_scalar * T_base_shlf + (1.0 - f_scalar) * T_pmp
    T_base_shlf = min(T_base_shlf, T_pmp)
    return T_base_shlf
end

@inline function _calc_bmb_grounded(T_prime_b::Float64,
                                    Q_ice_b_now::Float64,
                                    Q_b_now::Float64,
                                    Q_rock_now::Float64,
                                    rho_ice::Float64,
                                    L_ice::Float64)
    Q_net = Q_rock_now + Q_b_now - Q_ice_b_now
    bmb   = -Q_net / (rho_ice * L_ice)
    if T_prime_b < -1.0 && bmb < 0.0
        bmb = 0.0
    end
    if abs(bmb) < 1e-5
        bmb = 0.0
    end
    return bmb
end
