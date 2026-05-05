"""
    YelmoModelThrm

Thermodynamics (`thrm`) component for the pure-Julia `YelmoModel`.
Computes ice and bedrock temperature / enthalpy state, basal water,
internal heat sources, and pressure-melting diagnostics from the
current `tpo`/`dyn`/`mat`/`bnd` state.

Public surface: `therm_step!(y::YelmoModel, dt)`, dispatched from
`YelmoCore.step!(::YelmoModel, dt)` AFTER `mat_step!` in the fixed
phase order. Matches the Fortran per-step loop at
`yelmo_ice.f90:268-286` (`calc_ytopo` → `calc_ydyn` → `calc_ymat` →
`calc_ytherm` → corrector topo).

Port plan (incremental):

  - **PR1**: foundation. Module scaffolding, `therm_step!` dispatching
    `method = "fixed"` as a no-op, foundation kernels (`solve_tridiag!`,
    properties, `calc_dzeta_terms!`).
  - **PR2 (this commit)**: analytic solvers `linear`, `robin`,
    `robin-cold`. Adds the per-step property update (cp / kt / T_pmp),
    the Q_rock-from-Q_geo fallback, the analytic dispatch, and the
    universal post-step diagnostics (`T_prime`, `T_prime_b`, `f_pmp`,
    `enth`).
  - **PR3**: heat sources (`Q_strn`, `Q_b`) and basal water (`H_w`).
  - **PR4**: implicit `temp` solver + bedrock column.
  - **PR5**: `enth` solver (port-only, not benchmarked).
  - **PR6**: horizontal advection (`advecxy`) + `bmb_grnd` +
    full `calc_ytherm` orchestrator.

Out of scope: `enth-poly` (~1929 lines of adaptive CTS solver) and
`ice_tracer` (belongs to `mat`).

`therm_step!` does NOT advance `y.time` — that is owned by
`topo_step!`, matching the dyn/mat convention.
"""
module YelmoModelThrm

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using Oceananigans.Grids: znodes
using Oceananigans.Fields: interior

using ..YelmoCore: AbstractYelmoModel, YelmoModel

import ..YelmoCore: therm_step!

export therm_step!,
       calc_specific_heat_capacity, calc_thermal_conductivity,
       calc_T_pmp,
       calc_cp_3D!, calc_kt_3D!, calc_T_pmp_3D!, calc_f_pmp!,
       calc_dzeta_terms!,
       convert_to_enthalpy, convert_to_enthalpy_3D!,
       define_temp_linear_3D!, define_temp_robin_3D!,
       define_temp_linear_column!, define_temp_robin_column!

include("properties.jl")
include("dzeta.jl")
include("enthalpy.jl")
include("solvers_analytic.jl")

"""
    therm_step!(y::YelmoModel, dt) -> y

Update thermodynamic state (`y.thrm`) for the current step. Dispatches
on `y.p.ytherm.method`.

Phase order (Fortran `calc_ytherm`, yelmo_thermodynamics.f90:22):

  1. **Properties** — refresh `cp`, `kt` from current `T_ice` (or pin
     to `const_cp` / `const_kt` per `y.p.ytherm`); refresh `T_pmp`
     from current `H_ice`.
  2. **Q_rock fallback** — if `Q_rock` is identically zero (first
     step), seed from `bnd.Q_geo` to give the Robin solver a
     non-degenerate basal flux. Subsequent steps reuse the previous
     value.
  3. **Method dispatch** —
       - `fixed`              : pass-through (no T_ice update).
       - `linear`             : `define_temp_linear_3D!`.
       - `robin`              : `define_temp_robin_3D!(cold=false)`.
       - `robin-cold`         : `define_temp_robin_3D!(cold=true)`.
       - `temp`               : not yet ported (PR4).
       - `enth`               : not yet ported (PR5).
       - `enth-poly`          : out of scope.
  4. **Enthalpy** — `convert_to_enthalpy_3D!` keeps `enth` consistent
     with the just-written `(T_ice, omega)`.
  5. **Diagnostics** — `T_prime = T_ice - T_pmp`, `T_prime_b =
     T_prime[:, :, 1]`, `f_pmp` via `calc_f_pmp!`.

Steps 1, 2, 4, 5 always run, even for `method = "fixed"`. Heat
sources (`Q_strn`, `Q_b`) and basal water (`H_w`) are PR3.

`therm_step!` does NOT advance `y.time`.
"""
function therm_step!(y::YelmoModel, dt::Float64)
    y.p === nothing && error(
        "therm_step!: y.p must be non-nothing once the physics body is wired. " *
        "Construct YelmoModel with a YelmoModelParameters value, or skip " *
        "the orchestrator's per-component chain.")

    par     = y.p.ytherm
    method  = par.method
    c       = y.c
    zeta_aa = znodes(y.gt, Center())

    # 1. Thermal property update.
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

    # 2. Q_rock fallback: first step seeds Q_rock from Q_geo so the
    #    Robin solver has a non-zero basal heat flux. Mirrors Fortran
    #    `calc_ytherm` lines 122-124. Uses `interior(...)` so we look
    #    at and write to interior cells only.
    Q_rock_int = interior(y.thrm.Q_rock)
    if maximum(Q_rock_int) == 0.0
        copyto!(Q_rock_int, interior(y.bnd.Q_geo))
    end

    # 3. Method dispatch — write `T_ice` and `omega`.
    if method == "fixed"
        # Pass-through. Leave T_ice and omega alone.
    elseif method == "linear"
        define_temp_linear_3D!(y.thrm.T_ice, y.thrm.omega,
                               y.tpo.H_ice, y.bnd.T_srf,
                               zeta_aa,
                               c.T0, c.T_pmp_beta, c.rho_ice, c.g)
    elseif method == "robin"
        define_temp_robin_3D!(y.thrm.T_ice, y.thrm.omega,
                              y.thrm.T_pmp, y.thrm.cp, y.thrm.kt,
                              y.thrm.Q_rock, y.bnd.T_srf, y.tpo.H_ice,
                              y.bnd.smb_ref, y.thrm.bmb_grnd, y.tpo.f_grnd,
                              zeta_aa, c.rho_ice, c.sec_year;
                              cold=false)
    elseif method == "robin-cold"
        define_temp_robin_3D!(y.thrm.T_ice, y.thrm.omega,
                              y.thrm.T_pmp, y.thrm.cp, y.thrm.kt,
                              y.thrm.Q_rock, y.bnd.T_srf, y.tpo.H_ice,
                              y.bnd.smb_ref, y.thrm.bmb_grnd, y.tpo.f_grnd,
                              zeta_aa, c.rho_ice, c.sec_year;
                              cold=true)
    elseif method == "temp"
        error("therm_step!: method=\"$(method)\" (implicit temperature " *
              "solver, production target) lands in the temp+bedrock " *
              "milestone (PR4). Use \"fixed\", \"linear\", \"robin\", " *
              "or \"robin-cold\" for now.")
    elseif method == "enth"
        error("therm_step!: method=\"$(method)\" (enthalpy solver) " *
              "lands in the enth milestone (PR5). Note: enth is ported " *
              "but not benchmarked against the Mirror — the Mirror path " *
              "showed issues on the Fortran side. Use \"fixed\" or " *
              "\"temp\" (once PR4 lands) for now.")
    elseif method == "enth-poly"
        error("therm_step!: method=\"$(method)\" (polythermal CTS-adaptive " *
              "solver) is out of scope for the Yelmo.jl thrm port. The " *
              "Fortran `ice_enthalpy_poly.f90` (~1929 lines) is not " *
              "planned for porting. Use \"temp\" (production) once PR4 " *
              "lands.")
    else
        error("therm_step!: unknown method=\"$(method)\". Supported: " *
              "\"fixed\", \"linear\", \"robin\", \"robin-cold\" (PR2). " *
              "Coming: \"temp\" (PR4); \"enth\" (PR5).")
    end

    # 4. Update enth from (T_ice, omega, T_pmp, cp, L). For analytic
    #    methods omega is identically zero, so enth = cp * T_ice; the
    #    formula is general and handles the temperate enth solver
    #    once it lands in PR5.
    convert_to_enthalpy_3D!(y.thrm.enth, y.thrm.T_ice, y.thrm.omega,
                            y.thrm.T_pmp, y.thrm.cp, c.L_ice)

    # 5. Homologous-temperature + pressure-melting fraction.
    _calc_T_prime_3D!(y.thrm.T_prime, y.thrm.T_prime_b,
                      y.thrm.T_ice, y.thrm.T_pmp)
    calc_f_pmp!(y.thrm.f_pmp, y.thrm.T_ice, y.thrm.T_pmp, y.tpo.f_grnd;
                gamma=par.gamma)

    return y
end

# Compute T_prime = T_ice - T_pmp (3D) and T_prime_b = T_prime[:, :, 1]
# (2D basal slice). Mirrors the two diagnostic lines in Fortran
# `calc_ytherm` lines 248-249.
function _calc_T_prime_3D!(T_prime_field, T_prime_b_field,
                            T_ice_field, T_pmp_field)
    Tp_d  = T_prime_field.data
    Tpb_d = T_prime_b_field.data
    Ti_d  = T_ice_field.data
    Tm_d  = T_pmp_field.data
    Nx    = T_ice_field.grid.Nx
    Ny    = T_ice_field.grid.Ny
    Nz    = T_ice_field.grid.Nz
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Tp_d[i, j, k] = Ti_d[i, j, k] - Tm_d[i, j, k]
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        Tpb_d[i, j, 1] = Tp_d[i, j, 1]
    end
    return nothing
end

end # module YelmoModelThrm
