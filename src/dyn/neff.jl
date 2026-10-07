# ----------------------------------------------------------------------
# Effective basal pressure `N_eff` on aa-cell centres.
#
# `calc_ydyn_neff!` dispatches on `yhyd.bkt_N_closure` (FastHydrology's
# N closures, `closures.f90`):
#
#   - `-1`  → EXTERNAL: no-op (`N_eff` is set externally).
#   - ` 0`  → CONST: `N_eff = const_N`.
#   - ` 1`  → OVERBURDEN: `N_eff = ρ_i g H_eff`.
#   - ` 2`  → MARINE: marine connectivity (Leguy et al. 2014, Eq. 14).
#   - ` 3`  → TILL: till basal pressure (van Pelt & Bueler 2015, Eq. 23).
#   - ` 4`  → TWO_VALUE: `f_pmp · (δ P_0) + (1 - f_pmp) · P_0`
#             (`δ = two_value_delta`; standalone-only in Fortran Yelmo).
#
# All methods scale `H_ice` to "effective" thickness (zero for
# partially-covered cells, full thickness for fully-covered) before
# computing the overburden, mirroring `calc_H_eff(set_frac_zero=true)`.
# Floating cells (`f_grnd == 0`) get `N_eff = 0` (except CONST).
#
# **Subgrid sampling is NOT yet ported.** When `ydyn.neff_nxi > 0`, the
# Fortran reference samples the water thickness over the cell using either
# Gaussian quadrature (`nxi == 1`) or a uniform `nxi × nxi` grid
# (`nxi > 1`); `check_ported` rejects it.
#
# Port of `yelmo/src/yelmo_dynamics.f90 calc_ydyn_neff` (v1.15) and the
# `calc_effective_pressure_*` closures.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calc_ydyn_neff!

# Effective ice thickness: full `H_ice` only on fully-covered cells,
# zero otherwise. Mirrors `calc_H_eff(..., set_frac_zero=true)`.
@inline function _H_eff(H_ice::Float64, f_ice::Float64)
    return f_ice >= 1.0 ? H_ice : 0.0
end

# OVERBURDEN (1) — overburden pressure on grounded cells.
@inline function _neff_overburden(H_ice::Float64, f_ice::Float64,
                                  f_grnd::Float64,
                                  rho_ice::Float64, g::Float64)
    f_grnd > 0.0 || return 0.0
    return rho_ice * g * _H_eff(H_ice, f_ice)
end

# MARINE (2) — marine connectivity (Leguy et al. 2014, Eq. 14).
@inline function _neff_marine(H_ice::Float64, f_ice::Float64,
                              z_bed::Float64, z_sl::Float64, H_w::Float64,
                              p::Float64,
                              rho_ice::Float64, rho_sw::Float64, g::Float64)
    rho_sw_ice = rho_sw / rho_ice
    H_float = max(0.0, rho_sw_ice * (z_sl - z_bed))
    H_eff   = _H_eff(H_ice, f_ice)

    p_w = if H_eff == 0.0
        0.0
    elseif H_eff < H_float
        rho_ice * g * H_eff   # floating: water pressure equals ice pressure
    else
        x = min(1.0, H_float / H_eff)
        rho_ice * g * H_eff * (1.0 - (1.0 - x)^p)
    end
    return rho_ice * g * H_eff - p_w
end

# TILL (3) — till basal pressure (van Pelt & Bueler 2015, Eq. 23).
# `H_w` is the till water thickness, `H_w_max` the bucket capacity
# (`yhyd.W_til_max`).
@inline function _neff_till(H_w::Float64, H_ice::Float64, f_ice::Float64,
                            f_grnd::Float64, H_w_max::Float64,
                            N0::Float64, delta::Float64,
                            e0::Float64, Cc::Float64,
                            rho_ice::Float64, g::Float64)
    f_grnd > 0.0 || return 0.0
    H_eff = _H_eff(H_ice, f_ice)
    P0    = rho_ice * g * H_eff
    s     = min(H_w / H_w_max, 1.0)
    # Cap the exponent to avoid overflow at very low Cc / high e0.
    q1    = min((e0 / Cc) * (1.0 - s), 10.0)
    return min(N0 * (delta * P0 / N0)^s * 10.0^q1, P0)
end

# TWO_VALUE (4) — two-valued blend via `f_pmp` (fraction of cell at
# pressure-melting). At `f_pmp = 0` (frozen): `N_eff = P0`; at
# `f_pmp = 1` (temperate): `N_eff = δ · P0`.
@inline function _neff_two_value(f_pmp::Float64, H_ice::Float64, f_ice::Float64,
                                 f_grnd::Float64, delta::Float64,
                                 rho_ice::Float64, g::Float64)
    f_grnd > 0.0 || return 0.0
    H_eff = _H_eff(H_ice, f_ice)
    P0 = rho_ice * g * H_eff
    P1 = P0 * delta
    return P0 * (1.0 - f_pmp) + P1 * f_pmp
end

"""
    calc_ydyn_neff!(y::YelmoModel) -> y

Compute `y.dyn.N_eff` from the current state, dispatching on
`y.p.yhyd.bkt_N_closure`. Reads `H_ice_dyn`, `f_ice_dyn`, `f_grnd` from
`tpo`; `z_bed`, `z_sl` from `bnd`; `H_w`, `f_pmp` from `thrm`;
constants from `y.c`. Closure parameters come from `y.p.yhyd`.
"""
function calc_ydyn_neff!(y)
    hyd    = y.p.yhyd
    method = hyd.bkt_N_closure
    -1 <= method <= 4 || error(
        "calc_ydyn_neff!: yhyd.bkt_N_closure must be in [-1, 4]; got $method")

    method == -1 && return y       # EXTERNAL: `N_eff` set externally — leave alone

    N_int = interior(y.dyn.N_eff)

    if method == 0
        fill!(N_int, Float64(hyd.const_N))
        return y
    end

    W_til_max = Float64(hyd.W_til_max)

    rho_ice = Float64(y.c.rho_ice)
    rho_sw  = Float64(y.c.rho_sw)
    g       = Float64(y.c.g)

    H_ice_dyn = interior(y.tpo.H_ice_dyn)
    f_ice_dyn = interior(y.tpo.f_ice_dyn)
    f_grnd    = interior(y.tpo.f_grnd)
    z_bed     = interior(y.bnd.z_bed)
    z_sl      = interior(y.bnd.z_sl)
    H_w       = interior(y.thrm.H_w)
    f_pmp     = interior(y.thrm.f_pmp)

    Nx, Ny = size(N_int, 1), size(N_int, 2)

    if method == 1
        @inbounds for j in 1:Ny, i in 1:Nx
            N_int[i, j, 1] = _neff_overburden(
                H_ice_dyn[i, j, 1], f_ice_dyn[i, j, 1], f_grnd[i, j, 1],
                rho_ice, g)
        end
    elseif method == 2
        p_lk = Float64(hyd.marine_p)
        @inbounds for j in 1:Ny, i in 1:Nx
            N_int[i, j, 1] = _neff_marine(
                H_ice_dyn[i, j, 1], f_ice_dyn[i, j, 1],
                z_bed[i, j, 1], z_sl[i, j, 1], H_w[i, j, 1],
                p_lk, rho_ice, rho_sw, g)
        end
    elseif method == 3
        N0    = Float64(hyd.till_N0)
        delta = Float64(hyd.till_delta)
        e0    = Float64(hyd.till_e0)
        Cc    = Float64(hyd.till_Cc)
        @inbounds for j in 1:Ny, i in 1:Nx
            N_int[i, j, 1] = _neff_till(
                H_w[i, j, 1], H_ice_dyn[i, j, 1], f_ice_dyn[i, j, 1],
                f_grnd[i, j, 1], W_til_max,
                N0, delta, e0, Cc, rho_ice, g)
        end
    else  # method == 4
        delta = Float64(hyd.two_value_delta)
        @inbounds for j in 1:Ny, i in 1:Nx
            N_int[i, j, 1] = _neff_two_value(
                f_pmp[i, j, 1], H_ice_dyn[i, j, 1], f_ice_dyn[i, j, 1],
                f_grnd[i, j, 1], delta, rho_ice, g)
        end
    end

    return y
end
