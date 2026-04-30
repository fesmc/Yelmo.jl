# ----------------------------------------------------------------------
# Effective basal pressure `N_eff` on aa-cell centres.
#
# `calc_ydyn_neff!` dispatches on `yneff.method`:
#
#   - `-1`  → no-op (assume `N_eff` is set externally).
#   - ` 0`  → constant: `N_eff = neff_const`.
#   - ` 1`  → overburden: `N_eff = ρ_i g H_eff`.
#   - ` 2`  → marine connectivity (Leguy et al. 2014, Eq. 14).
#   - ` 3`  → till basal pressure (van Pelt & Bueler 2015, Eq. 23).
#   - ` 4`  → as method 3 but with constant till saturation
#             `H_w = H_w_max · s_const` instead of the thermo H_w field.
#   - ` 5`  → two-valued blend `f_pmp · (δ P_0) + (1 - f_pmp) · P_0`.
#
# All methods scale `H_ice` to "effective" thickness (zero for
# partially-covered cells, full thickness for fully-covered) before
# computing the overburden, mirroring `calc_H_eff(set_frac_zero=true)`.
# Floating cells (`f_grnd == 0`) get `N_eff = 0` (except method 0).
#
# **Subgrid sampling is NOT yet ported.** When `yneff.nxi > 0`, the
# Fortran reference samples `H_w` over the cell using either Gaussian
# quadrature (`nxi == 1`, the `gq2D_*` family in `fesm-utils/`) or a
# uniform `nxi × nxi` grid (`nxi > 1`). Either path requires
# additional kernel infrastructure that we'll plug in later — likely
# via `FastGaussQuadrature.jl` for the quadrature variant. For now,
# `nxi != 0` errors with a deferral message.
#
# Port of `yelmo/src/yelmo_dynamics.f90:832 calc_ydyn_neff` plus the
# four `calc_effective_pressure_*` helpers from `basal_dragging.f90`.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calc_ydyn_neff!

# Effective ice thickness: full `H_ice` only on fully-covered cells,
# zero otherwise. Mirrors `calc_H_eff(..., set_frac_zero=true)`.
@inline function _H_eff(H_ice::Float64, f_ice::Float64)
    return f_ice >= 1.0 ? H_ice : 0.0
end

# Method 1 — overburden pressure on grounded cells.
@inline function _neff_overburden(H_ice::Float64, f_ice::Float64,
                                  f_grnd::Float64,
                                  rho_ice::Float64, g::Float64)
    f_grnd > 0.0 || return 0.0
    return rho_ice * g * _H_eff(H_ice, f_ice)
end

# Method 2 — marine connectivity (Leguy et al. 2014, Eq. 14).
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

# Method 3 / 4 — till basal pressure (van Pelt & Bueler 2015, Eq. 23).
# `H_w` is the basal water layer thickness (method 3 reads it from the
# thermo field; method 4 substitutes `H_w_max · s_const` upstream).
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

# Method 5 — two-valued blend via `f_pmp` (fraction of cell at
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
`y.p.yneff.method`. Reads `H_ice_dyn`, `f_ice_dyn`, `f_grnd` from
`tpo`; `z_bed`, `z_sl` from `bnd`; `H_w`, `f_pmp` from `thrm`;
constants from `y.c`. Method-specific scalar parameters come from
`y.p.yneff` (`const_`, `p`, `N0`, `delta`, `e0`, `Cc`, `s_const`).

`H_w_max` resolves to `y.p.ytherm.H_w_max` when `yneff.H_w_max < 0`
(the Fortran default sentinel), or to the override otherwise.

Subgrid sampling (`yneff.nxi > 0`) is not yet ported — errors with a
deferral pointer; commit accepts only `nxi = 0`.
"""
function calc_ydyn_neff!(y)
    yneff   = y.p.yneff
    nxi     = yneff.nxi
    nxi == 0 || error(
        "calc_ydyn_neff!: yneff.nxi = $nxi not yet ported. " *
        "Subgrid sampling needs Gaussian-quadrature helpers " *
        "(eventually via FastGaussQuadrature.jl); only nxi = 0 supported.")

    method = yneff.method
    -1 <= method <= 5 || error(
        "calc_ydyn_neff!: yneff.method must be in [-1, 5]; got $method")

    method == -1 && return y       # `N_eff` set externally — leave alone

    N_int = interior(y.dyn.N_eff)

    if method == 0
        fill!(N_int, Float64(yneff.const_))
        return y
    end

    H_w_max = yneff.H_w_max < 0.0 ? Float64(y.p.ytherm.H_w_max) : Float64(yneff.H_w_max)

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
        p_lk = Float64(yneff.p)
        @inbounds for j in 1:Ny, i in 1:Nx
            N_int[i, j, 1] = _neff_marine(
                H_ice_dyn[i, j, 1], f_ice_dyn[i, j, 1],
                z_bed[i, j, 1], z_sl[i, j, 1], H_w[i, j, 1],
                p_lk, rho_ice, rho_sw, g)
        end
    elseif method == 3 || method == 4
        N0    = Float64(yneff.N0)
        delta = Float64(yneff.delta)
        e0    = Float64(yneff.e0)
        Cc    = Float64(yneff.Cc)
        if method == 3
            @inbounds for j in 1:Ny, i in 1:Nx
                N_int[i, j, 1] = _neff_till(
                    H_w[i, j, 1], H_ice_dyn[i, j, 1], f_ice_dyn[i, j, 1],
                    f_grnd[i, j, 1], H_w_max,
                    N0, delta, e0, Cc, rho_ice, g)
            end
        else  # method 4 — constant H_w override
            H_w_const = H_w_max * Float64(yneff.s_const)
            @inbounds for j in 1:Ny, i in 1:Nx
                N_int[i, j, 1] = _neff_till(
                    H_w_const, H_ice_dyn[i, j, 1], f_ice_dyn[i, j, 1],
                    f_grnd[i, j, 1], H_w_max,
                    N0, delta, e0, Cc, rho_ice, g)
            end
        end
    else  # method == 5
        delta = Float64(yneff.delta)
        @inbounds for j in 1:Ny, i in 1:Nx
            N_int[i, j, 1] = _neff_two_value(
                f_pmp[i, j, 1], H_ice_dyn[i, j, 1], f_ice_dyn[i, j, 1],
                f_grnd[i, j, 1], delta, rho_ice, g)
        end
    end

    return y
end
