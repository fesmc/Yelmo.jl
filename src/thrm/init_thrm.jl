# ----------------------------------------------------------------------
# init_thrm! — analytic temperature initialisation.
#
# Mirrors the temperature-init subset of Fortran's `yelmo_init_state`
# (yelmo/src/yelmo_ice.f90:1262). The full equilibration cycle (topo
# sync → thrm → mat → dyn → mat → topo sync) lives in YelmoCore's
# `init_state!`; this file is the thrm-specific piece extracted so the
# orchestrator doesn't have to import the analytic-solver helpers.
#
# Validates `thrm_method ∈ {"linear", "robin", "robin-cold"}` (matching
# Fortran's check at yelmo_ice.f90:1295-1301), populates `T_ice` (3D)
# via the chosen analytic solver, fills `T_ice_b` / `T_ice_s` (2D
# boundary fields) by constant extrapolation, and refreshes
# `T_pmp`, `T_pmp_b`, `T_pmp_s`, `T_prime`, `T_prime_b`, `T_prime_s`,
# `enth`, `f_pmp` so subsequent `mat_step!` / `dyn_step!` calls see
# a consistent thermal state.
# ----------------------------------------------------------------------

import ..YelmoCore: init_thrm!

const _INIT_THRM_VALID_METHODS = ("linear", "robin", "robin-cold")

"""
    init_thrm!(y::YelmoModel; thrm_method::AbstractString = "robin") -> y

Analytic temperature initialisation. Writes `y.thrm.T_ice`,
`T_ice_b`, `T_ice_s` and refreshes derived diagnostics. Used as a
component of `init_state!`'s equilibration cycle but exposed as a
public entry point for callers that only need the thermal piece.

Without this call, `y.thrm.T_ice` stays at its Field-allocation
default (≈ zeros). Combined with `T_pmp ≈ 272 K` from the hydrostatic
pressure-melting calculation, this yields `T_prime_b ≈ -272 K`, which
sends `calc_c_bed!`'s thermal-scaling branch (`scale_T = 1`) into
the cold-base end and collapses basal friction to
`ytill.cf_ref · N_eff` (default 0.8 Pa) — saturating the SSA solver.

# Arguments
- `thrm_method::AbstractString` (default `"robin"`) — analytic
  temperature solver to use. Must be one of `"linear"`, `"robin"`,
  or `"robin-cold"`. Mirrors Fortran's identical validation in
  `yelmo_init_state` (yelmo_ice.f90:1295-1301).

# Notes
- Does not advance `y.time`.
- `omega` is set to zero (analytic solvers are cold-ice).
- For experiments restarting from a NetCDF that already carries a
  full thermal state, skip this call.
"""
function init_thrm!(y::YelmoModel; thrm_method::AbstractString = "robin")
    thrm_method in _INIT_THRM_VALID_METHODS || error(
        "init_thrm!: thrm_method=\"$(thrm_method)\" not recognised. " *
        "Must be one of \"linear\", \"robin\", or \"robin-cold\" (mirrors " *
        "Fortran's yelmo_init_state validation in yelmo_ice.f90:1295-1301).")

    y.p === nothing && error(
        "init_thrm!: y.p must be non-nothing. Construct YelmoModel with a " *
        "YelmoModelParameters value.")

    par     = y.p.ytherm
    c       = y.c
    zeta_aa = y.thrm.scratch.zeta_aa

    # 1. Refresh thermal properties (cp, kt, T_pmp + 2D boundaries).
    if par.use_const_cp
        fill!(interior(y.thrm.cp), par.const_cp)
    else
        calc_cp_3D!(y.thrm.cp, y.thrm.T_ice)
    end
    if par.use_const_kt
        fill!(interior(y.thrm.kt), par.const_kt)
    else
        calc_kt_3D!(y.thrm.kt, y.thrm.T_ice, c.sec_year)
    end
    calc_T_pmp_3D!(y.thrm.T_pmp, y.tpo.H_ice, zeta_aa,
                   c.T0, c.T_pmp_beta, c.rho_ice, c.g)
    calc_T_pmp_boundaries_2D!(y.thrm.T_pmp_b, y.thrm.T_pmp_s,
                              y.tpo.H_ice,
                              c.T0, c.T_pmp_beta, c.rho_ice, c.g)

    # 2. Q_rock fallback: seed from Q_geo (mirrors Fortran
    #    calc_ytherm:122-124 and the per-step fallback in therm_step!).
    Q_rock_int = interior(y.thrm.Q_rock)
    if maximum(Q_rock_int) == 0.0
        copyto!(Q_rock_int, interior(y.bnd.Q_geo))
    end

    # 3. Run the chosen analytic solver — writes T_ice (3D) and omega ≡ 0.
    if thrm_method == "linear"
        define_temp_linear_3D!(y.thrm.T_ice, y.thrm.omega,
                               y.tpo.H_ice, y.bnd.T_srf,
                               zeta_aa,
                               c.T0, c.T_pmp_beta, c.rho_ice, c.g)
    elseif thrm_method == "robin"
        define_temp_robin_3D!(y.thrm.T_ice, y.thrm.omega,
                              y.thrm.T_pmp, y.thrm.cp, y.thrm.kt,
                              y.thrm.Q_rock, y.bnd.T_srf, y.tpo.H_ice,
                              y.bnd.smb_ref, y.thrm.bmb_grnd, y.tpo.f_grnd,
                              zeta_aa, c.rho_ice, c.sec_year;
                              cold=false)
    else # "robin-cold"
        define_temp_robin_3D!(y.thrm.T_ice, y.thrm.omega,
                              y.thrm.T_pmp, y.thrm.cp, y.thrm.kt,
                              y.thrm.Q_rock, y.bnd.T_srf, y.tpo.H_ice,
                              y.bnd.smb_ref, y.thrm.bmb_grnd, y.tpo.f_grnd,
                              zeta_aa, c.rho_ice, c.sec_year;
                              cold=true)
    end

    # 4. Populate 2D boundary fields T_ice_b / T_ice_s by constant
    #    extrapolation from the first interior layer (Path B convention,
    #    same as mat's enh / visc handling). Surface uses bnd.T_srf as
    #    the exact analytic boundary condition.
    _init_thrm_boundary_extrapolate!(y.thrm.T_ice_b, y.thrm.T_ice_s,
                                       y.thrm.T_ice, y.bnd.T_srf)

    # 5. Update derived diagnostics: enth + T_prime + f_pmp.
    convert_to_enthalpy_3D!(y.thrm.enth, y.thrm.T_ice, y.thrm.omega,
                            y.thrm.T_pmp, y.thrm.cp, c.L_ice)
    _calc_T_prime_3D!(y.thrm.T_prime, y.thrm.T_prime_b, y.thrm.T_prime_s,
                      y.thrm.T_ice, y.thrm.T_pmp,
                      y.thrm.T_ice_b, y.thrm.T_pmp_b,
                      y.thrm.T_ice_s, y.thrm.T_pmp_s)
    calc_f_pmp!(y.thrm.f_pmp, y.thrm.T_ice_b, y.thrm.T_pmp_b, y.tpo.f_grnd;
                gamma=par.gamma)

    return y
end

# Populate T_ice_b / T_ice_s from the analytic 3D T_ice. The basal
# (T_ice_b) value uses constant extrapolation from the first interior
# layer; the surface (T_ice_s) value uses the prescribed bnd.T_srf,
# which is the exact analytic boundary condition for robin / linear
# solutions and avoids an extra extrapolation step at the surface.
@inline function _init_thrm_boundary_extrapolate!(T_ice_b_field,
                                                    T_ice_s_field,
                                                    T_ice_field,
                                                    T_srf_field)
    Tb_d   = interior(T_ice_b_field)
    Ts_d   = interior(T_ice_s_field)
    Ti_d   = interior(T_ice_field)
    Tsrf_d = interior(T_srf_field)
    Nx, Ny = size(Tb_d, 1), size(Tb_d, 2)
    @inbounds for j in 1:Ny, i in 1:Nx
        Tb_d[i, j, 1] = Ti_d[i, j, 1]
        Ts_d[i, j, 1] = Tsrf_d[i, j, 1]
    end
    return nothing
end
