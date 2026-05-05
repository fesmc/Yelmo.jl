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

  - **PR1 (this commit)**: foundation only. Module scaffolding,
    `therm_step!` dispatching `method = "fixed"` as a no-op (existing
    behaviour preserved), explicit errors for every other method
    pointing at the milestone that lands it. The per-cell
    thermodynamic property kernels (`calc_specific_heat_capacity`,
    `calc_thermal_conductivity`, `calc_T_pmp`, `calc_f_pmp!`) and the
    Hoffmann-2018 vertical-discretisation weights
    (`calc_dzeta_terms!`) are ported and exported here so PR2/PR4
    can call them without further include churn. Tridiagonal solver
    (`solve_tridiag!`) lives in the shared `YelmoUtils` module
    (`src/utils/tridiag.jl`).
  - **PR2**: analytic methods `linear`, `robin`, `robin-cold`
    (closed-form, no advection).
  - **PR3**: heat sources (`Q_strn`, `Q_b`) and basal water (`H_w`).
  - **PR4**: implicit `temp` solver + bedrock column
    (`rock_method ∈ {equil, active, fixed}`).
  - **PR5**: `enth` solver (ported but not benchmarked — Mirror's
    enth path showed issues, so the test gate stays off until a
    Julia-native benchmark is built).
  - **PR6**: horizontal advection (`calc_advec_horizontal_3D` →
    `advecxy`), basal mass balance, full `calc_ytherm` orchestrator.

Out of scope: `enth-poly` (~1929 lines of adaptive CTS solver) and
`ice_tracer` (belongs to `mat`, not `thrm`).

`therm_step!` does NOT advance `y.time` — that is owned by
`topo_step!`, matching the dyn/mat convention.
"""
module YelmoModelThrm

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using Oceananigans.Fields: interior

using ..YelmoCore: AbstractYelmoModel, YelmoModel

import ..YelmoCore: therm_step!

export therm_step!,
       calc_specific_heat_capacity, calc_thermal_conductivity,
       calc_T_pmp,
       calc_cp_3D!, calc_kt_3D!, calc_T_pmp_3D!, calc_f_pmp!,
       calc_dzeta_terms!

include("properties.jl")
include("dzeta.jl")

"""
    therm_step!(y::YelmoModel, dt) -> y

Update thermodynamic state (`y.thrm`) for the current step. Dispatches
on `y.p.ytherm.method`.

PR1 scope: only `method = "fixed"` is implemented (pass-through,
matching the existing pre-port behaviour where `step!` simply did not
call `therm_step!`). Every other method errors with a clear pointer
to the porting milestone that lands it.

Method matrix (target after the full thrm port):

  | method        | landing PR | notes                                |
  |---------------|------------|--------------------------------------|
  | `fixed`       | PR1        | no-op — works today                  |
  | `linear`      | PR2        | analytic linear T profile            |
  | `robin`       | PR2        | closed-form Robin solution           |
  | `robin-cold`  | PR2        | Robin + linear blend                 |
  | `temp`        | PR4        | implicit cold-ice column solver (production) |
  | `enth`        | PR5        | enthalpy solver (ported, untested)   |
  | `enth-poly`   | —          | out of scope (adaptive CTS)          |
"""
function therm_step!(y::YelmoModel, dt::Float64)
    y.p === nothing && error(
        "therm_step!: y.p must be non-nothing once the physics body is wired. " *
        "Construct YelmoModel with a YelmoModelParameters value, or skip " *
        "the orchestrator's per-component chain.")

    method = y.p.ytherm.method

    if method == "fixed"
        # Pass-through. Fields in `y.thrm` are left at their current
        # values — initialisation or external prescription owns them.
    elseif method == "linear" || method == "robin" || method == "robin-cold"
        error("therm_step!: method=\"$(method)\" (analytic solver) " *
              "lands in the analytic-solvers milestone (PR2). " *
              "Use \"fixed\" for now.")
    elseif method == "temp"
        error("therm_step!: method=\"$(method)\" (implicit temperature " *
              "solver, production target) lands in the temp+bedrock " *
              "milestone (PR4). Use \"fixed\" for now.")
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
              "\"fixed\" (PR1). Coming: \"linear\", \"robin\", " *
              "\"robin-cold\" (PR2); \"temp\" (PR4); \"enth\" (PR5).")
    end

    return y
end

end # module YelmoModelThrm
