"""
    YelmoModelDyn

Dynamics (`dyn`) component for the pure-Julia `YelmoModel`. Computes
the horizontal velocity solution from the current `tpo`/`mat`/`thrm`/
`bnd` state, plus all derived stress and velocity-magnitude
diagnostics.

Public surface: `dyn_step!(y::YelmoModel, dt)`, dispatched from
`YelmoCore.step!(::YelmoModel, dt)` after `topo_step!` in the fixed
phase order.

Milestone 3a (current): scaffolding + pre-solver kinematics
(driving stress, lateral BC stress) + post-solver underflow / flux
/ magnitude / surface-slice / `duxydt` diagnostics. Solver dispatch
only handles `solver = "fixed"` (no velocity update); SIA / SSA /
hybrid / DIVA arrive in subsequent milestones (3c–3f). Velocity
Jacobian, vertical velocity `uz`, and strain-rate tensor land in
milestone 3h.
"""
module YelmoModelDyn

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using Oceananigans.BoundaryConditions: fill_halo_regions!

using ..YelmoCore: AbstractYelmoModel, YelmoModel

import ..YelmoCore: dyn_step!

export dyn_step!,
       calc_driving_stress!, calc_driving_stress_gl!,
       calc_lateral_bc_stress_2D!,
       calc_cb_ref!, calc_c_bed!,
       calc_ice_flux!, calc_magnitude_from_staggered!, calc_vel_ratio!

include("driving_stress.jl")
include("lateral_stress.jl")
include("basal_dragging.jl")
include("diagnostics.jl")

# Cell-spacing helpers — local copies of the topo-module pattern.
# Stretched grids are not yet supported; flag explicitly so an
# accidental non-uniform grid surfaces immediately rather than as
# silently wrong driving-stress values.
function _dx(grid::RectilinearGrid)
    Δx = grid.Δxᶜᵃᵃ
    Δx isa Number || error("YelmoModelDyn requires a uniform x-spacing for now (got $(typeof(Δx))).")
    return abs(Δx)
end

function _dy(grid::RectilinearGrid)
    Δy = grid.Δyᵃᶜᵃ
    Δy isa Number || error("YelmoModelDyn requires a uniform y-spacing for now (got $(typeof(Δy))).")
    return abs(Δy)
end

"""
    dyn_step!(y::YelmoModel, dt) -> y

Advance the dynamics state by `dt` years. Phase order matches
Fortran's `calc_ydyn` body (`yelmo_dynamics.f90:48`):

1. Snapshot `(ux_bar, uy_bar)` into `(ux_bar_prev, uy_bar_prev)`,
   used by higher-order topo timestepping and by the `duxydt`
   diagnostic.
2. Driving stress `taud_acx`/`taud_acy` from surface slope and ice
   thickness (milestone 3a, deferred).
3. Optional grounding-line driving-stress modifier when
   `taud_gl_method != 0` (milestone 3a, deferred).
4. Lateral boundary stress `taul_int_acx`/`taul_int_acy` at the ice
   front (milestone 3a, deferred).
5. Effective pressure `N_eff` and bed-roughness chain `cb_tgt →
   cb_ref → c_bed` (milestone 3b, deferred).
6. Solver dispatch on `y.p.ydyn.solver`. Currently only `"fixed"`
   (no-op) is implemented; other solvers error with a milestone
   pointer.
7. Underflow clip on `ux`/`uy`/`ux_bar`/`uy_bar` (milestone 3a,
   deferred).
8. Velocity Jacobian + vertical velocity `uz` + strain-rate tensor
   (milestone 3h, deferred).
9. Diagnostics: ice flux, stress and velocity magnitudes, surface /
   basal velocity slices, basal-to-surface ratio `f_vbvs`, time
   derivative `duxydt` (milestone 3a, deferred).

`dyn_step!` does NOT advance `y.time` — that is owned by `topo_step!`
which runs first.
"""
function dyn_step!(y::YelmoModel, dt::Float64)
    # 1. Snapshot prev depth-averaged velocity (used by topo PC and
    #    by the `duxydt` diagnostic in step 9).
    interior(y.dyn.ux_bar_prev) .= interior(y.dyn.ux_bar)
    interior(y.dyn.uy_bar_prev) .= interior(y.dyn.uy_bar)
    # Snapshot the magnitude too — the post-solver recompute
    # overwrites `uxy_bar` in place; we need the pre-step value
    # for `duxydt`.
    uxy_prev = copy(interior(y.dyn.uxy_bar))

    # 2. Driving stress on ac-staggered faces.
    calc_driving_stress!(y.dyn.taud_acx, y.dyn.taud_acy,
                         y.tpo.H_ice_dyn, y.tpo.f_ice_dyn,
                         y.tpo.dzsdx, y.tpo.dzsdy,
                         _dx(y.g), y.p.ydyn.taud_lim,
                         y.c.rho_ice, y.c.g)

    # 3. Optional grounding-line refinement of the driving stress.
    #    Default `taud_gl_method = 0` keeps this a no-op. The Fortran
    #    reference always calls with `beta_gl_stag = 1`; mirror that.
    if y.p.ydyn.taud_gl_method != 0
        calc_driving_stress_gl!(y.dyn.taud_acx, y.dyn.taud_acy,
                                y.tpo.H_ice_dyn, y.tpo.z_srf,
                                y.bnd.z_bed, y.bnd.z_sl, y.tpo.H_grnd,
                                y.tpo.f_grnd, y.tpo.f_grnd_acx, y.tpo.f_grnd_acy,
                                _dx(y.g),
                                y.c.rho_ice, y.c.rho_sw, y.c.g,
                                y.p.ydyn.taud_gl_method, 1)
    end

    # 4. Lateral boundary stress at the ice front (Pa·m).
    calc_lateral_bc_stress_2D!(y.dyn.taul_int_acx, y.dyn.taul_int_acy,
                               y.tpo.mask_frnt, y.tpo.H_ice, y.tpo.f_ice,
                               y.tpo.z_srf, y.bnd.z_sl,
                               y.c.rho_ice, y.c.rho_sw, y.c.g)

    # 5a. Effective pressure N_eff (milestone 3b, deferred — current
    #     value comes from the restart load until calc_ydyn_neff!
    #     lands in the next commit).
    # TODO(3b commit 2): calc_ydyn_neff!(y)

    # 5b. Reference till-friction coefficient `cb_tgt`.
    calc_cb_ref!(y.dyn.cb_tgt,
                 y.bnd.z_bed, y.bnd.z_bed_sd, y.bnd.z_sl, y.bnd.H_sed,
                 y.p.ytill.f_sed, y.p.ytill.sed_min, y.p.ytill.sed_max,
                 y.p.ytill.cf_ref, y.p.ytill.cf_min,
                 y.p.ytill.z0, y.p.ytill.z1,
                 y.p.ytill.n_sd,
                 y.p.ytill.scale_zb, y.p.ytill.scale_sed)

    # 5c. If `till_method == 1`, set the active friction coefficient
    #     `cb_ref` to the freshly-computed `cb_tgt`. Other till_method
    #     values leave `cb_ref` at its restart-loaded / externally-
    #     supplied value (Fortran convention — see yelmo_dynamics.f90:131).
    if y.p.ytill.method == 1
        interior(y.dyn.cb_ref) .= interior(y.dyn.cb_tgt)
    end

    # 5d. Basal drag coefficient `c_bed = c · N_eff`.
    calc_c_bed!(y.dyn.c_bed,
                y.dyn.cb_ref, y.dyn.N_eff, y.thrm.T_prime_b,
                y.p.ytill.is_angle, y.p.ytill.cf_ref,
                y.p.ydyn.T_frz, y.p.ydyn.scale_T)

    # 6. Solver dispatch. 3a only handles "fixed".
    solver = y.p.ydyn.solver
    if solver == "fixed"
        # No velocity update.
    else
        error("dyn_step!: solver=\"$solver\" not yet ported. " *
              "Milestone 3a only supports \"fixed\". " *
              "SIA / SSA / hybrid / DIVA land in milestones 3c–3f.")
    end

    # 7. Underflow clip on the velocity fields (matches Fortran's
    #    `where (abs(.) < TOL_UNDERFLOW) . = 0`).
    _clip_underflow!(y.dyn.ux)
    _clip_underflow!(y.dyn.uy)
    _clip_underflow!(y.dyn.ux_bar)
    _clip_underflow!(y.dyn.uy_bar)

    # 9. Post-solver diagnostics (steps 8 / Jacobian + uz + strain rate
    #    are deferred to milestone 3h).
    dx_g = _dx(y.g)
    dy_g = _dy(y.g)

    # Ice flux on ac-staggered faces.
    calc_ice_flux!(y.dyn.qq_acx, y.dyn.qq_acy,
                   y.dyn.ux_bar, y.dyn.uy_bar, y.tpo.H_ice,
                   dx_g, dy_g)

    # Stress + flux + velocity magnitudes at aa-cells.
    calc_magnitude_from_staggered!(y.dyn.qq,        y.dyn.qq_acx,   y.dyn.qq_acy,   y.tpo.f_ice)
    calc_magnitude_from_staggered!(y.dyn.taub,      y.dyn.taub_acx, y.dyn.taub_acy, y.tpo.f_ice)
    calc_magnitude_from_staggered!(y.dyn.taud,      y.dyn.taud_acx, y.dyn.taud_acy, y.tpo.f_ice)
    calc_magnitude_from_staggered!(y.dyn.uxy_b,     y.dyn.ux_b,     y.dyn.uy_b,     y.tpo.f_ice)
    calc_magnitude_from_staggered!(y.dyn.uxy_i_bar, y.dyn.ux_i_bar, y.dyn.uy_i_bar, y.tpo.f_ice)
    calc_magnitude_from_staggered!(y.dyn.uxy_bar,   y.dyn.ux_bar,   y.dyn.uy_bar,   y.tpo.f_ice)

    # 3D per-layer magnitude — the kernel naturally loops over k.
    calc_magnitude_from_staggered!(y.dyn.uxy, y.dyn.ux, y.dyn.uy, y.tpo.f_ice)

    # Surface / basal velocity slices. Fortran:
    #   uz_b  = uz(:, :, 1)
    #   ux_s  = ux(:, :, nz_aa);  uy_s  = uy(:, :, nz_aa)
    #   uz_s  = uz(:, :, nz_ac);  uxy_s = uxy(:, :, nz_aa)
    @views interior(y.dyn.uz_b)[:, :, 1]  .= interior(y.dyn.uz)[:, :, 1]
    @views interior(y.dyn.ux_s)[:, :, 1]  .= interior(y.dyn.ux)[:, :, end]
    @views interior(y.dyn.uy_s)[:, :, 1]  .= interior(y.dyn.uy)[:, :, end]
    @views interior(y.dyn.uz_s)[:, :, 1]  .= interior(y.dyn.uz)[:, :, end]
    @views interior(y.dyn.uxy_s)[:, :, 1] .= interior(y.dyn.uxy)[:, :, end]

    # Basal-to-surface velocity ratio.
    calc_vel_ratio!(y.dyn.f_vbvs, y.dyn.uxy_b, y.dyn.uxy_s)

    # Time derivative of depth-averaged velocity magnitude.
    duxydt_int  = interior(y.dyn.duxydt)
    uxy_bar_int = interior(y.dyn.uxy_bar)
    if abs(dt) > TOL
        @. duxydt_int = (uxy_bar_int - uxy_prev) / dt
    else
        fill!(duxydt_int, 0.0)
    end

    return y
end

end # module YelmoModelDyn
