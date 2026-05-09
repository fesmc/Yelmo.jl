"""
    YelmoModelMat

Material (`mat`) component for the pure-Julia `YelmoModel`. Computes
the rate factor `ATT`, enhancement factor `enh`, 3D Glen-law
viscosity `visc`, and 2D deviatoric stress tensor `strs2D` from the
current `tpo` / `dyn` / `thrm` / `bnd` state.

Public surface: `mat_step!(y::YelmoModel, dt)`, dispatched from
`YelmoCore.step!(::YelmoModel, dt)` AFTER `dyn_step!` in the fixed
phase order. Matches the Fortran per-step loop at
`yelmo_ice.f90:268-286`: `calc_ydyn` reads the previous step's
`mat.ATT`, then `calc_ymat` computes a fresh `ATT` (and viscosity,
stress, ...) from the just-solved velocity field for the next step.
The same applies to the adaptive-PC path in `src/timestepping.jl`.

Milestone scope (this PR — "mat 1"):

  - `rf_method ∈ {-1, 0, 1, 2}`. `rf_method = 1` (temperature- and
    water-content-coupled Arrhenius) reads `y.thrm.{T_ice, T_pmp,
    omega}` from the therm port; the EISMINT2 vs Greve & Blatter
    Arrhenius variant is selected by `ymat.rf_use_eismint2`, and
    water-content scaling by `ymat.rf_with_water`.
  - `enh_method ∈ {"simple", "shear2D"}`. `shear3D` and any
    `*-tracer` variant are deferred — the latter two require the
    tracer infrastructure (`calc_tracer_3D`, `calc_isochrones`).
  - `calc_age = true` is supported (deposition-time tracer
    `dep_time`) when `tracer_method = "impl"`. The explicit
    branch (`tracer_method = "expl"`) and the isochrone diagnostic
    (`depth_iso`) are deferred. Enhancement-via-tracer
    (`enh_method ∈ {simple-tracer, shear2D-tracer, shear3D-tracer}`)
    is also deferred — those paths still error in this commit.
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

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using Oceananigans.Grids: znodes
using Oceananigans.Fields: interior

using ..YelmoCore: AbstractYelmoModel, YelmoModel
using ..YelmoUtils: solve_tridiag!, calc_dzeta_terms!

import ..YelmoCore: mat_step!

export mat_step!,
       calc_viscosity_glen!, calc_visc_int!, depth_average!,
       calc_rate_factor!, calc_rate_factor_eismint!,
       scale_rate_factor_water!,
       define_enhancement_factor_2D!, define_enhancement_factor_3D!,
       calc_stress_tensor_2D!, calc_2D_eigen_values_pt,
       calc_tracer_3D!

include("viscosity.jl")
include("rate_factor.jl")
include("enhancement.jl")
include("stress.jl")
include("ice_tracer.jl")

"""
    mat_step!(y::YelmoModel, dt) -> y

Update material properties for the current step.

Phase order (Fortran `yelmo_material.f90:22 calc_ymat`):

  1. 2D deviatoric stress tensor (uses the previous step's
     `mat.visc_bar` — Fortran calls `calc_stress_tensor_2D` before
     re-computing `visc_bar` at the end of the step).
  2. Enhancement factor (3D `mat.enh`) per `ymat.enh_method`,
     followed by depth-averaged `mat.enh_bar`.
  3. Rate factor (`mat.ATT`, `mat.ATT_bar`) per `ymat.rf_method`.
  4. 3D Glen-law viscosity (`mat.visc`) using `dyn.strn_de` and
     the freshly-updated `mat.ATT`.
  5. Depth-averaged `mat.visc_bar` and depth-integrated `mat.visc_int`
     (the latter multiplied by `H_ice`).

Scope ("mat 1" PR):

  - `enh_method ∈ {"simple", "shear2D", "shear3D"}`. The three
    `*-tracer` variants need the tracer infrastructure
    (`calc_tracer_3D`, `bnd.enh_srf`) and error explicitly.
  - `rf_method ∈ {-1, 0, 1, 2}`. `rf_method = 1` (temperature- and
    water-coupled Arrhenius) reads `y.thrm.{T_ice, T_pmp, omega}`,
    branches on `rf_use_eismint2`, and applies `scale_rate_factor_water!`
    when `rf_with_water = true`.
  - `calc_age = true` runs the implicit-solver age-tracer port
    (Step 0 of `mat_step!`). Only `tracer_method = "impl"` is
    supported; the explicit branch and `calc_isochrones` (the
    `depth_iso` diagnostic) are deferred.
  - The 3D stress tensor `mat.strs` is left at allocation default
    (zero), matching Fortran which never fills it inside
    `calc_ymat`.

`mat_step!` does NOT advance `y.time` — that is owned by
`topo_step!`, matching the dyn convention.
"""
function mat_step!(y::YelmoModel, dt::Float64)
    y.p === nothing && error(
        "mat_step!: y.p must be non-nothing once the physics body is wired. " *
        "Construct YelmoModel with a YelmoModelParameters value, or skip " *
        "the orchestrator's per-component chain.")

    par     = y.p.ymat
    par_dyn = y.p.ydyn
    c       = y.c
    zeta_aa = znodes(y.gt, Center())

    # 0. Age tracer (`dep_time`). Dispatched only when `calc_age = true`
    #    AND `dt > 0` AND `tracer_method == "impl"` — the explicit
    #    branch and isochrones (`depth_iso`) are out of scope for this
    #    port. `X_srf = y.time` matches Fortran (`yelmo_material.f90:64`):
    #    surface-deposited ice has deposition time equal to the current
    #    simulation time, so `age = current_time - dep_time` after
    #    advection. The 500 m/yr `uxy_bar` mask is Fortran's
    #    fast-flow exclusion (yelmo_material.f90:70).
    if par.calc_age && dt > 0.0
        if par.tracer_method != "impl"
            error("mat_step!: tracer_method=\"$(par.tracer_method)\" not " *
                  "ported. Use \"impl\" (the explicit branch is deferred).")
        end
        zeta_ac = znodes(y.gt, Face())
        dx_g    = y.g.Δxᶜᵃᵃ
        dx_f    = abs(Float64(dx_g isa Number ? dx_g :
            error("mat_step!: stretched x-grid not supported by ice_tracer.")))
        Uxy_bar = interior(y.dyn.uxy_bar)
        mask    = Uxy_bar[:, :, 1] .<= 500.0
        calc_tracer_3D!(y.mat.dep_time, y.time,
                        y.dyn.ux, y.dyn.uy, y.dyn.uz,
                        y.tpo.H_ice, y.tpo.bmb,
                        zeta_aa, zeta_ac, dx_f, dt;
                        kappa = par.tracer_impl_kappa,
                        mask  = mask)
    end

    # 1. Deviatoric stress tensor (uses the *previous* visc_bar).
    #    Fortran calc_ymat:97. `mat.strs2D_txz` / `mat.strs2D_tyz`
    #    stay at their initialised zeros — Fortran's `calc_ymat`
    #    likewise never writes them, and they enter only via `te`.
    calc_stress_tensor_2D!(y.mat.strs2D_txx, y.mat.strs2D_tyy, y.mat.strs2D_txy,
                           y.mat.strs2D_te,
                           y.mat.strs2D_tau_eig_1, y.mat.strs2D_tau_eig_2,
                           y.mat.visc_bar,
                           y.dyn.strn2D_dxx, y.dyn.strn2D_dyy, y.dyn.strn2D_dxy,
                           y.mat.strs2D_txz, y.mat.strs2D_tyz)

    # 2. Enhancement factor.
    method = par.enh_method
    if method == "simple"
        # Fortran calc_ymat:110 passes `enh_shear` for both the shear
        # and the stream slot in the 2D blend, then broadcasts
        # vertically.
        _calc_enh_2D_uniform_z!(y.mat.enh, y.tpo.f_grnd, y.dyn.strn2D_f_shear,
                                par.enh_shear, par.enh_shear, par.enh_shlf)
    elseif method == "shear2D"
        _calc_enh_2D_uniform_z!(y.mat.enh, y.tpo.f_grnd, y.dyn.strn2D_f_shear,
                                par.enh_shear, par.enh_stream, par.enh_shlf)
    elseif method == "shear3D"
        define_enhancement_factor_3D!(y.mat.enh, y.tpo.f_grnd, y.dyn.strn_f_shear;
                                      enh_shear=par.enh_shear,
                                      enh_stream=par.enh_stream,
                                      enh_shlf=par.enh_shlf)
    elseif method in ("simple-tracer", "shear2D-tracer", "shear3D-tracer")
        error("mat_step!: enh_method=\"$(method)\" requires the tracer " *
              "infrastructure (calc_tracer_3D + bnd.enh_srf), deferred " *
              "to a later milestone. Use \"simple\", \"shear2D\", or " *
              "\"shear3D\" for now.")
    else
        error("mat_step!: unknown enh_method=\"$(method)\". Supported: " *
              "\"simple\", \"shear2D\", \"shear3D\".")
    end

    # 3a. Path B: refresh enh_b / enh_s from the recomputed 3D enh
    #     by constant extrapolation off the nearest interior layer.
    #     Equivalent to the pre-Path B convention used by `enh_bar`'s
    #     trapezoidal rule, but now stored explicitly so the registry
    #     has authoritative boundary values (and so future commits
    #     can replace this with a physically derived rule without
    #     changing the integration call site).
    _path_b_constant_extrapolate!(y.mat.enh_b, y.mat.enh_s, y.mat.enh)

    # 3b. Depth-averaged enhancement (explicit boundaries).
    depth_average!(y.mat.enh_bar, y.mat.enh, y.mat.enh_b, y.mat.enh_s, zeta_aa)

    # 4. Rate factor.
    if par.rf_method == -1
        # External — leave ATT, ATT_bar, ATT_b / ATT_s at their loaded values.
    elseif par.rf_method == 0
        fill!(interior(y.mat.ATT),     par.rf_const)
        fill!(interior(y.mat.ATT_bar), par.rf_const)
        fill!(interior(y.mat.ATT_b),   par.rf_const)
        fill!(interior(y.mat.ATT_s),   par.rf_const)
    elseif par.rf_method == 1
        # Temperature- and (optionally) water-content-coupled Arrhenius.
        # Fortran yelmo_material.f90:212-229: the Arrhenius variant is
        # selected by `rf_use_eismint2` (EISMINT2 / Payne 2000 vs
        # Greve & Blatter 2009), and `rf_with_water` triggers the
        # Lliboutry-Duval water-content scaling on top.
        if par.rf_use_eismint2
            calc_rate_factor_eismint!(y.mat.ATT, y.thrm.T_ice, y.thrm.T_pmp,
                                      y.mat.enh, c.T0)
        else
            calc_rate_factor!(y.mat.ATT, y.thrm.T_ice, y.thrm.T_pmp,
                              y.mat.enh, c.T0)
        end
        if par.rf_with_water
            scale_rate_factor_water!(y.mat.ATT, y.thrm.omega)
        end
        # Path B refresh of basal / surface boundaries from the freshly
        # computed 3D ATT, then depth-average — same convention as enh
        # (step 3a/3b above) and visc (step 5b/6 below).
        _path_b_constant_extrapolate!(y.mat.ATT_b, y.mat.ATT_s, y.mat.ATT)
        depth_average!(y.mat.ATT_bar, y.mat.ATT,
                       y.mat.ATT_b, y.mat.ATT_s, zeta_aa)
    elseif par.rf_method == 2
        if par_dyn.visc_method != 0 || par.n_glen != 1.0
            error("mat_step!: rf_method=2 only valid when " *
                  "ydyn.visc_method=0 and ymat.n_glen=1.0 " *
                  "(got visc_method=$(par_dyn.visc_method), " *
                  "n_glen=$(par.n_glen)).")
        end
        att_val = 1.0 / (2.0 * par_dyn.visc_const)
        fill!(interior(y.mat.ATT),     att_val)
        fill!(interior(y.mat.ATT_bar), att_val)
        fill!(interior(y.mat.ATT_b),   att_val)
        fill!(interior(y.mat.ATT_s),   att_val)
    else
        error("mat_step!: unknown rf_method=$(par.rf_method). " *
              "Supported: -1, 0, 1, 2.")
    end

    # 5. 3D Glen-law viscosity.
    calc_viscosity_glen!(y.mat.visc, y.dyn.strn_de, y.mat.ATT, y.tpo.f_ice;
                         n_glen=par.n_glen, visc_min=par.visc_min,
                         eps_0=par_dyn.eps_0)

    # 5b. Path B: refresh visc_b / visc_s from interior limits — see
    #     comment at step 3a.
    _path_b_constant_extrapolate!(y.mat.visc_b, y.mat.visc_s, y.mat.visc)

    # 6. Depth-averaged and depth-integrated viscosity (explicit
    #    boundaries).
    depth_average!(y.mat.visc_bar, y.mat.visc, y.mat.visc_b, y.mat.visc_s, zeta_aa)
    calc_visc_int!(y.mat.visc_int, y.mat.visc,
                   y.mat.visc_b, y.mat.visc_s,
                   y.tpo.H_ice, y.tpo.f_ice, zeta_aa)

    return y
end

# Path B helper: refresh a pair of 2D `_b` / `_s` boundary fields
# from the basal / surface interior layers of a 3D Center field by
# constant extrapolation. Used by `mat_step!` to keep the boundary
# registry in sync with the recomputed interior viscosity / enh
# fields. Equivalent to the pre-Path B `view(V, :, :, 1:1)` /
# `view(V, :, :, Nz:Nz)` shortcut, but persisted into the registry's
# 2D fields so I/O and downstream consumers see authoritative values.
@inline function _path_b_constant_extrapolate!(field_b, field_s, field_3d)
    V  = interior(field_3d)
    Vb = interior(field_b)
    Vs = interior(field_s)
    Nz = size(V, 3)
    @inbounds Vb[:, :, 1] .= @view V[:, :, 1]
    @inbounds Vs[:, :, 1] .= @view V[:, :, Nz]
    return nothing
end


# Compute the 2D enhancement-factor blend (`define_enhancement_factor_2D`
# formula) using a 2D `f_shear` and replicate it across all z-layers
# of `enh3D`. Used for `enh_method ∈ {"simple", "shear2D"}` where the
# 3D enhancement field is z-uniform per Fortran calc_ymat:113-116.
@inline function _calc_enh_2D_uniform_z!(enh3D, f_grnd, f_shear_2D,
                                         enh_shear::Real,
                                         enh_stream::Real,
                                         enh_shlf::Real)
    E  = interior(enh3D)
    Fg = interior(f_grnd)
    Fs = interior(f_shear_2D)

    Nx, Ny, Nz = size(E)
    @assert size(Fg, 1) == Nx && size(Fg, 2) == Ny
    @assert size(Fs, 1) == Nx && size(Fs, 2) == Ny

    e_shear  = Float64(enh_shear)
    e_stream = Float64(enh_stream)
    e_shlf   = Float64(enh_shlf)

    @inbounds for j in 1:Ny, i in 1:Nx
        enh_ssa = Fg[i, j, 1] * e_stream + (1.0 - Fg[i, j, 1]) * e_shlf
        e_val   = Fs[i, j, 1] * e_shear + (1.0 - Fs[i, j, 1]) * enh_ssa
        for k in 1:Nz
            E[i, j, k] = e_val
        end
    end
    return enh3D
end

end # module
