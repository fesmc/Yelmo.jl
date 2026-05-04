"""
    YelmoModelMat

Material (`mat`) component for the pure-Julia `YelmoModel`. Computes
the rate factor `ATT`, enhancement factor `enh`, 3D Glen-law
viscosity `visc`, and 2D deviatoric stress tensor `strs2D` from the
current `tpo` / `dyn` / `thrm` / `bnd` state.

Public surface: `mat_step!(y::YelmoModel, dt)`, dispatched from
`YelmoCore.step!(::YelmoModel, dt)` BEFORE `dyn_step!` in the fixed
phase order. Matches Fortran `yelmo_ice.f90:286 / 386 / 1324 / 1344`,
which calls `calc_ymat` ahead of `calc_ydyn` so the velocity solve
sees a freshly-updated `ATT` field.

Milestone scope (this PR — "mat 1"):

  - `rf_method ∈ {-1, 0, 2}` only. `rf_method = 1` (temperature- and
    water-content-coupled) needs the therm port (`T_ice`, `T_pmp`,
    `omega`) and lands with that milestone.
  - `enh_method ∈ {"simple", "shear2D"}`. `shear3D` and any
    `*-tracer` variant are deferred — the latter two require the
    tracer infrastructure (`calc_tracer_3D`, `calc_isochrones`).
  - `calc_age = false` only. Online age-tracing also needs the
    tracer infrastructure.
  - The 3D stress tensor `strs` is left at its initialization value
    (zero); Fortran's `calc_ymat` only fills `strs2D` via
    `calc_stress_tensor_2D`, mirroring that here.

Deviation from Fortran organization: in Yelmo.jl the strain-rate
tensor (`strn`, `strn2D`) and velocity Jacobian live in `dyn`, not
`mat`. `mat_step!` reads `y.dyn.strn`, `y.dyn.strn2D`, and the
2D shear-fraction `y.dyn.strn2D_f_shear` directly; there is no
`mat.f_shear_bar` schema entry. See dyn 3h port commits and the
`dyn/mat split` design memo.
"""
module YelmoModelMat

using ..YelmoCore: AbstractYelmoModel, YelmoModel

import ..YelmoCore: mat_step!

export mat_step!,
       calc_viscosity_glen!, calc_visc_int!

include("viscosity.jl")

"""
    mat_step!(y::YelmoModel, dt) -> y

Update material properties (`ATT`, `enh`, `visc`, `visc_bar`,
`visc_int`, `strs2D`) for the current `dt`. Body lands in commit 6
of the "mat 1" PR; this scaffold is a no-op so the orchestrator can
already call `mat_step!` ahead of `dyn_step!` without changing
behaviour.

`mat_step!` does NOT advance `y.time` — that is owned by
`topo_step!`, matching the dyn convention.
"""
function mat_step!(y::YelmoModel, dt::Float64)
    return y
end

end # module
