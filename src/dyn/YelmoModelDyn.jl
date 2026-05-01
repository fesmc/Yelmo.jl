"""
    YelmoModelDyn

Dynamics (`dyn`) component for the pure-Julia `YelmoModel`. Computes
the horizontal velocity solution from the current `tpo`/`mat`/`thrm`/
`bnd` state, plus all derived stress and velocity-magnitude
diagnostics.

Public surface: `dyn_step!(y::YelmoModel, dt)`, dispatched from
`YelmoCore.step!(::YelmoModel, dt)` after `topo_step!` in the fixed
phase order.

Milestone 3d (current): scaffolding + pre-solver kinematics +
post-solver diagnostics (3a) + bed-roughness chain (3b) + SIA
solver dispatch with the Option C vertical-stagger convention +
SSA Picard solver (Krylov+AMG inner linear solve) + hybrid SIA+SSA.
Solver dispatch handles `solver = "fixed"` (no velocity update),
`solver = "sia"` (Option C SIA wrapper, `calc_velocity_sia!`),
`solver = "ssa"` (Picard iteration via `calc_velocity_ssa!`), and
`solver = "hybrid"` (SIA shear + SSA basal sliding); DIVA / L1L2
arrive in subsequent milestones (3e–3f). Velocity Jacobian, vertical
velocity `uz`, and strain-rate tensor land in milestone 3h.

Vertical convention: 3D fields (`ux`, `ux_i`, `ATT`, `tau_xz`, …)
are at Oceananigans `Center()` vertical staggering — interior layer
midpoints, length `Nz_aa`. Bed (zeta = 0) and surface (zeta = 1)
boundary values live in 2D fields (`ux_b`, `ux_s`,
`scratch.ux_i_s`) per Option C.
"""
module YelmoModelDyn

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using Oceananigans.BoundaryConditions: fill_halo_regions!

using ..YelmoCore: AbstractYelmoModel, YelmoModel
using ..YelmoSolvers: Solver, SSASolver

import ..YelmoCore: dyn_step!

export dyn_step!,
       calc_driving_stress!, calc_driving_stress_gl!,
       calc_lateral_bc_stress_2D!,
       calc_ydyn_neff!,
       calc_cb_ref!, calc_c_bed!, calc_beta!, stagger_beta!,
       calc_ice_flux!, calc_magnitude_from_staggered!, calc_vel_ratio!,
       calc_shear_stress_3D!, calc_uxy_sia_3D!, calc_velocity_sia!,
       gq2d_nodes,
       calc_visc_eff_3D_aa!, calc_visc_eff_3D_nodes!, calc_visc_eff_int!,
       stagger_visc_aa_ab!,
       set_ssa_masks!, _assemble_ssa_matrix!,
       Solver, SSASolver,
       _solve_ssa_linear!,
       picard_relax_visc!, picard_relax_vel!,
       picard_calc_convergence_l2, picard_calc_convergence_l1rel_matrix!,
       set_inactive_margins!, calc_basal_stress!,
       dump_ssa_assembly

include("topology_helpers.jl")
include("quadrature.jl")
include("driving_stress.jl")
include("lateral_stress.jl")
include("neff.jl")
include("basal_dragging.jl")
include("viscosity.jl")
include("diagnostics.jl")
include("integration.jl")
include("velocity_sia.jl")
include("velocity_ssa.jl")

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

    # 5a. Effective pressure N_eff. Dispatches on `yneff.method`
    #     ∈ [-1, 5]. Subgrid sampling (`nxi > 0`) is deferred.
    calc_ydyn_neff!(y)

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

    # 6. Solver dispatch. 3d handles "fixed", "sia", "ssa", "hybrid".
    solver = y.p.ydyn.solver
    Nz_dyn = size(interior(y.dyn.ux), 3)
    if solver == "fixed"
        # No velocity update — Q5 locked-in: leave ux_i / uy_i /
        # ux_i_bar / uy_i_bar at their loaded / previous values.
    elseif solver == "sia"
        # Option C SIA path. `znodes(y.gt, Center())` returns the
        # Oceananigans Center z-positions (interior layer midpoints,
        # length Nz_aa); bed (zeta = 0) and surface (zeta = 1) are
        # NOT in this vector and are handled explicitly inside the
        # wrapper / kernel.
        zeta_c = znodes(y.gt, Center())
        calc_velocity_sia!(y.dyn.ux_i, y.dyn.uy_i,
                           y.dyn.ux_i_bar, y.dyn.uy_i_bar,
                           y.dyn.scratch.ux_i_s, y.dyn.scratch.uy_i_s,
                           y.dyn.scratch.sia_tau_xz,
                           y.dyn.scratch.sia_tau_yz,
                           y.tpo.H_ice, y.tpo.f_ice,
                           y.dyn.taud_acx, y.dyn.taud_acy,
                           y.mat.ATT,
                           zeta_c,
                           y.p.ymat.n_glen)

        # Mirrors yelmo_dynamics.f90:413-421: with no SSA path, set
        # basal sliding and basal stress to zero.
        fill!(interior(y.dyn.ux_b),     0.0)
        fill!(interior(y.dyn.uy_b),     0.0)
        fill!(interior(y.dyn.taub_acx), 0.0)
        fill!(interior(y.dyn.taub_acy), 0.0)

        # Total velocity = SIA shear + basal sliding (= 0 here).
        # Mirrors yelmo_dynamics.f90:436-443.
        interior(y.dyn.ux)     .= interior(y.dyn.ux_i)
        interior(y.dyn.uy)     .= interior(y.dyn.uy_i)
        interior(y.dyn.ux_bar) .= interior(y.dyn.ux_i_bar)
        interior(y.dyn.uy_bar) .= interior(y.dyn.uy_i_bar)
    elseif solver == "ssa"
        # SSA-only: no shear contribution. Zero ux_i / uy_i and
        # ux_i_bar / uy_i_bar so the SIA-derived diagnostics are
        # consistent with "no SIA". Picard iteration writes
        # ux_b / uy_b / taub directly.
        fill!(interior(y.dyn.ux_i),     0.0)
        fill!(interior(y.dyn.uy_i),     0.0)
        fill!(interior(y.dyn.ux_i_bar), 0.0)
        fill!(interior(y.dyn.uy_i_bar), 0.0)
        fill!(interior(y.dyn.scratch.ux_i_s), 0.0)
        fill!(interior(y.dyn.scratch.uy_i_s), 0.0)

        calc_velocity_ssa!(y)

        # Total velocity is layer-uniform: ux(:, :, k) = ux_b for all k.
        # Mirrors yelmo_dynamics.f90:436-443.
        Ux  = interior(y.dyn.ux)
        Uy  = interior(y.dyn.uy)
        Uxb = interior(y.dyn.ux_b)
        Uyb = interior(y.dyn.uy_b)
        # Note: under Bounded-x, ux is XFace 3D and ux_b is XFace 2D
        # so the 1st & 2nd dims match (Nx+1, Ny). Under Periodic-x both
        # are Nx-wide. The k loop is Nz_aa (for the 3D-Center stagger).
        for k in 1:Nz_dyn
            @views Ux[:, :, k] .= Uxb[:, :, 1]
            @views Uy[:, :, k] .= Uyb[:, :, 1]
        end
        interior(y.dyn.ux_bar) .= Uxb
        interior(y.dyn.uy_bar) .= Uyb
    elseif solver == "hybrid"
        # SIA shear + SSA basal sliding.
        zeta_c = znodes(y.gt, Center())
        calc_velocity_sia!(y.dyn.ux_i, y.dyn.uy_i,
                           y.dyn.ux_i_bar, y.dyn.uy_i_bar,
                           y.dyn.scratch.ux_i_s, y.dyn.scratch.uy_i_s,
                           y.dyn.scratch.sia_tau_xz,
                           y.dyn.scratch.sia_tau_yz,
                           y.tpo.H_ice, y.tpo.f_ice,
                           y.dyn.taud_acx, y.dyn.taud_acy,
                           y.mat.ATT,
                           zeta_c,
                           y.p.ymat.n_glen)

        calc_velocity_ssa!(y)

        # Total velocity = SIA shear (3D, layer-varying) + SSA basal
        # sliding (2D, layer-uniform). Mirrors yelmo_dynamics.f90:436-443.
        Ux  = interior(y.dyn.ux)
        Uy  = interior(y.dyn.uy)
        Uxi = interior(y.dyn.ux_i)
        Uyi = interior(y.dyn.uy_i)
        Uxb = interior(y.dyn.ux_b)
        Uyb = interior(y.dyn.uy_b)
        for k in 1:Nz_dyn
            @views Ux[:, :, k] .= Uxi[:, :, k] .+ Uxb[:, :, 1]
            @views Uy[:, :, k] .= Uyi[:, :, k] .+ Uyb[:, :, 1]
        end
        interior(y.dyn.ux_bar) .= interior(y.dyn.ux_i_bar) .+ Uxb
        interior(y.dyn.uy_bar) .= interior(y.dyn.uy_i_bar) .+ Uyb
    else
        error("dyn_step!: solver=\"$solver\" not yet ported. " *
              "Milestone 3d supports \"fixed\", \"sia\", \"ssa\", \"hybrid\". " *
              "DIVA / L1L2 land in milestones 3e–3f.")
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
    #
    # Under Option C, `interior(y.dyn.ux)[:, :, end]` is the topmost
    # Center node at zeta_c[end], NOT the surface (zeta = 1). For
    # `solver == "sia"` the wrapper computed the actual surface value
    # into `y.dyn.scratch.ux_i_s`; the SIA branch below overwrites
    # `ux_s` / `uy_s` with `ux_i_s + ux_b`. For `solver == "fixed"`
    # the diagnostic here is incorrect (pre-existing limitation under
    # Option C) — it reports the topmost Center value rather than the
    # surface value.
    # TODO: fix solver=="fixed" surface diagnostic when restart-loaded
    # ux_s gets first-class storage (e.g. milestone 3g when therm
    # wires temperature-dependent ATT).
    @views interior(y.dyn.uz_b)[:, :, 1]  .= interior(y.dyn.uz)[:, :, 1]
    @views interior(y.dyn.ux_s)[:, :, 1]  .= interior(y.dyn.ux)[:, :, end]
    @views interior(y.dyn.uy_s)[:, :, 1]  .= interior(y.dyn.uy)[:, :, end]
    @views interior(y.dyn.uz_s)[:, :, 1]  .= interior(y.dyn.uz)[:, :, end]
    @views interior(y.dyn.uxy_s)[:, :, 1] .= interior(y.dyn.uxy)[:, :, end]

    # SIA / SSA / hybrid branch correction: assemble ux_s = ux_i_s + ux_b.
    # Under Option C the topmost Center node ≠ surface, so the generic
    # `interior(ux)[:, :, end]` line above wrote the topmost-Center
    # value rather than the surface value. The wrapper-produced
    # `scratch.ux_i_s` is the actual surface segment value (zero under
    # solver=="ssa" since the SSA branch zeros it explicitly); add
    # `ux_b` for the hybrid / sliding contribution. For solver=="fixed"
    # we keep the topmost-Center fallback (pre-existing limitation).
    if solver in ("sia", "ssa", "hybrid")
        @views interior(y.dyn.ux_s)[:, :, 1] .=
            interior(y.dyn.scratch.ux_i_s)[:, :, 1] .+
            interior(y.dyn.ux_b)[:, :, 1]
        @views interior(y.dyn.uy_s)[:, :, 1] .=
            interior(y.dyn.scratch.uy_i_s)[:, :, 1] .+
            interior(y.dyn.uy_b)[:, :, 1]
        # Re-derive `uxy_s` from the corrected ux_s / uy_s (the
        # generic `interior(uxy)[:, :, end]` written above used the
        # topmost-Center magnitude, not the surface magnitude).
        calc_magnitude_from_staggered!(y.dyn.uxy_s,
                                       y.dyn.ux_s, y.dyn.uy_s,
                                       y.tpo.f_ice)
    end

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
