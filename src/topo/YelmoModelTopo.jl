"""
    YelmoModelTopo

Topography (`tpo`) component for the pure-Julia `YelmoModel`. Evolves
ice thickness `H_ice` via mass conservation and updates derived
quantities (`z_srf`, `z_base`, `dHidt`, `dHidt_dyn`, `f_grnd`,
`f_ice`).

Public surface: `topo_step!(y::YelmoModel, dt)`, called from
`step!(::YelmoModel, dt)` in fixed phase order.

Milestone 2b: advection (from 2a) followed by surface mass balance
(SMB) and a residual cleanup tendency (`mb_resid`) that handles
margin/island regularisation. BMB, FMB, DMB, calving, the
predictor-corrector wrapping, and the `impl-lis` solver land in
subsequent milestones.
"""
module YelmoModelTopo

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields

using ..YelmoCore: AbstractYelmoModel, YelmoModel,
                   MASK_ICE_NONE, MASK_ICE_FIXED, MASK_ICE_DYNAMIC
import ..YelmoCore: step!

export topo_step!, advect_tracer!,
       apply_tendency!, mbal_tendency!, resid_tendency!,
       calc_f_ice!,
       calc_H_grnd!, determine_grounded_fractions!,
       calc_bmb_total!, calc_fmb_total!, calc_mb_discharge!,
       set_tau_relax!, calc_G_relaxation!,
       calc_calving_equil_ac!, calc_calving_threshold_ac!,
       calc_calving_vonmises_m16_ac!, merge_calving_rates!,
       lsf_init!, lsf_update!, lsf_redistance!,
       extrapolate_ocn_acx!, extrapolate_ocn_acy!,
       calving_step!

include("advection.jl")
include("mass_balance.jl")
include("ice_fraction.jl")
include("grounded.jl")
include("basal.jl")
include("frontal.jl")
include("discharge.jl")
include("relaxation.jl")
include("calving_ac.jl")
include("lsf.jl")
include("calving.jl")

"""
    step!(y::YelmoModel, dt) -> y

Pure-Julia model time-step. In milestone 2b this is `topo_step!`
followed by stub no-ops for `dyn`, `mat`, `thrm` (which arrive in
later milestones, each as their own per-component `<comp>_step!`
called from here in fixed phase order).
"""
function step!(y::YelmoModel, dt::Float64)
    topo_step!(y, dt)
    # dyn_step!(y, dt)   — future milestone 3
    # mat_step!(y, dt)   — future milestone 4
    # therm_step!(y, dt) — future milestone 5
    return y
end

"""
    topo_step!(y::YelmoModel, dt) -> y

Advance topography state by `dt` years. Phase order matches Fortran's
`calc_ytopo_pc` predictor/corrector body (yelmo_topography.f90:172):

1. Snapshot `H_ice` (used for the fixed-mask post-step and for the
   total `dHidt` denominator).
2. Advect `H_ice` by `(ux_bar, uy_bar)` from `y.dyn`, unless
   `y.p.ytopo.topo_fixed` is `true`.
3. Apply the `bnd.mask_ice` post-step pass (no-ice → 0; fixed →
   prior value; dynamic → clamped non-negative).
4. Recompute binary `f_ice` from the post-advection `H_ice`.
5. SMB: `mbal_tendency!(smb, …, smb_ref)` + `apply_tendency!(adjust_mb)`.
6. Refresh `f_ice`.
7. BMB: refresh `H_grnd` and `f_grnd_bmb` from current state, combine
   `thrm.bmb_grnd` and `bnd.bmb_shlf` into `tpo.bmb_ref` via
   `calc_bmb_total!`, then realise via `mbal_tendency!` + `apply_tendency!`.
   Skipped (`bmb = 0`) if `y.p.ytopo.use_bmb == false`.
8. Refresh `f_ice`.
9. FMB: `calc_fmb_total!` + `mbal_tendency!` + `apply_tendency!`.
   Skipped (`fmb = 0`) if `y.p.ytopo.use_bmb == false` — Fortran
   gates FMB on the same `use_bmb` flag.
10. Refresh `f_ice`.
11. DMB: `calc_mb_discharge!` (v1 stub: only `dmb_method = 0`) +
    `mbal_tendency!` + `apply_tendency!`.
12. Refresh `f_ice`.
13. Optional relaxation toward `H_ref`/`H_ice_n` via `set_tau_relax!`
    + `calc_G_relaxation!`. Skipped when `topo_rel == 0`. Errors on
    `topo_rel == 4` (depends on un-ported `mask_grz`). When
    `topo_rel == -1`, the timescale field is read from `bnd.tau_relax`.
14. Refresh `f_ice` after relaxation.
15. Residual cleanup: `resid_tendency!` + `apply_tendency!`.
16. Refresh `f_ice`.
17. `mb_net = smb + bmb + fmb + dmb + cmb + mb_relax + mb_resid`.
18. Update diagnostics: refresh `H_grnd` and subgrid `f_grnd`, plus
    `z_srf`/`z_base`/`dHidt`/`dHidt_dyn`.
19. Advance `y.time`.
"""
function topo_step!(y::YelmoModel, dt::Float64)
    H_prev = copy(interior(y.tpo.H_ice))

    # Snapshot the start-of-step thickness into `H_ice_n` so that
    # `topo_rel_field == "H_ice_n"` (relax-toward-previous) has a
    # meaningful target. Fortran does the same in calc_ytopo_pc.
    interior(y.tpo.H_ice_n) .= H_prev

    if !y.p.ytopo.topo_fixed
        advect_tracer!(y.tpo.H_ice, y.dyn.ux_bar, y.dyn.uy_bar, dt;
                       cfl_safety = y.p.yelmo.cfl_max)
    end

    _apply_mask_ice_pass!(y, H_prev)

    # Snapshot after dynamics + mask pass; used for `dHidt_dyn`.
    H_after_dyn = copy(interior(y.tpo.H_ice))

    # Refresh f_ice now that the dynamic margin may have moved.
    calc_f_ice!(y.tpo.f_ice, y.tpo.H_ice)

    # Surface mass balance: clip the raw forcing field, then apply.
    mbal_tendency!(y.tpo.smb, y.tpo.H_ice, y.tpo.f_grnd, y.bnd.smb_ref, dt)
    apply_tendency!(y.tpo.H_ice, y.tpo.smb, dt; adjust_mb=true)

    # SMB may have grown / shrunk margins; refresh f_ice before BMB.
    calc_f_ice!(y.tpo.f_ice, y.tpo.H_ice)

    # Basal mass balance: refresh H_grnd and f_grnd_bmb from the *current*
    # state (Fortran does the same — predictor/corrector iterations can
    # leave H_ice in a different state than the last diagnostic refresh).
    if y.p.ytopo.use_bmb
        calc_H_grnd!(y.tpo.H_grnd, y.tpo.H_ice, y.bnd.z_bed, y.bnd.z_sl,
                     y.c.rho_ice, y.c.rho_sw)
        determine_grounded_fractions!(y.tpo.f_grnd_bmb, y.tpo.H_grnd)

        calc_bmb_total!(y.tpo.bmb_ref, y.thrm.bmb_grnd, y.bnd.bmb_shlf,
                        y.tpo.H_ice, y.tpo.H_grnd, y.tpo.f_grnd_bmb,
                        y.p.ytopo.bmb_gl_method)
        mbal_tendency!(y.tpo.bmb, y.tpo.H_ice, y.tpo.f_grnd, y.tpo.bmb_ref, dt)
    else
        fill!(interior(y.tpo.bmb_ref), 0.0)
        fill!(interior(y.tpo.bmb),     0.0)
    end
    apply_tendency!(y.tpo.H_ice, y.tpo.bmb, dt; adjust_mb=true)

    # BMB may have moved margins; refresh f_ice before FMB.
    calc_f_ice!(y.tpo.f_ice, y.tpo.H_ice)

    # Frontal mass balance at marine margins. Fortran gates this on
    # `use_bmb` too — same flag, same intent (turn off at-shelf melt
    # for EISMINT-style runs).
    if y.p.ytopo.use_bmb
        calc_fmb_total!(y.tpo.fmb_ref,
                        y.bnd.fmb_shlf, y.bnd.bmb_shlf,
                        y.tpo.H_ice, y.tpo.H_grnd, y.tpo.f_ice,
                        y.p.ytopo.fmb_method, y.p.ytopo.fmb_scale,
                        y.c.rho_ice, y.c.rho_sw, _dx(y.g))
        mbal_tendency!(y.tpo.fmb, y.tpo.H_ice, y.tpo.f_grnd, y.tpo.fmb_ref, dt)
    else
        fill!(interior(y.tpo.fmb_ref), 0.0)
        fill!(interior(y.tpo.fmb),     0.0)
    end
    apply_tendency!(y.tpo.H_ice, y.tpo.fmb, dt; adjust_mb=true)

    # FMB may have moved margins; refresh f_ice before DMB.
    calc_f_ice!(y.tpo.f_ice, y.tpo.H_ice)

    # Subgrid discharge mass balance. v1 only supports dmb_method = 0
    # (no-op); other methods error inside the helper.
    calc_mb_discharge!(y.tpo.dmb_ref, y.tpo.H_ice, y.tpo.z_srf,
                       y.bnd.z_bed_sd,
                       y.tpo.dist_grline, y.tpo.dist_margin, y.tpo.f_ice,
                       y.p.ytopo.dmb_method, _dx(y.g),
                       y.p.ytopo.dmb_alpha_max, y.p.ytopo.dmb_tau,
                       y.p.ytopo.dmb_sigma_ref,
                       y.p.ytopo.dmb_m_d, y.p.ytopo.dmb_m_r)
    mbal_tendency!(y.tpo.dmb, y.tpo.H_ice, y.tpo.f_grnd, y.tpo.dmb_ref, dt)
    apply_tendency!(y.tpo.H_ice, y.tpo.dmb, dt; adjust_mb=true)

    # DMB may have moved margins; refresh f_ice before calving.
    calc_f_ice!(y.tpo.f_ice, y.tpo.H_ice)

    # Phase 7: level-set calving. No-op when `ycalv.use_lsf` is false.
    calving_step!(y, dt)

    # Calving may have killed cells; refresh f_ice before relaxation.
    calc_f_ice!(y.tpo.f_ice, y.tpo.H_ice)

    # Optional relaxation toward a reference state (Fortran phase 8).
    # Skipped entirely when `topo_rel == 0`.
    if y.p.ytopo.topo_rel != 0
        if y.p.ytopo.topo_rel == -1
            interior(y.tpo.tau_relax) .= interior(y.bnd.tau_relax)
        else
            set_tau_relax!(y.tpo.tau_relax, y.tpo.H_ice, y.tpo.f_grnd,
                           y.tpo.mask_grz, y.bnd.H_ice_ref,
                           y.p.ytopo.topo_rel, y.p.ytopo.topo_rel_tau)
        end

        H_ref = if y.p.ytopo.topo_rel_field == "H_ref"
            y.bnd.H_ice_ref
        elseif y.p.ytopo.topo_rel_field == "H_ice_n"
            y.tpo.H_ice_n
        else
            error("topo_step!: unknown topo_rel_field = \"$(y.p.ytopo.topo_rel_field)\". " *
                  "Supported: \"H_ref\", \"H_ice_n\".")
        end

        calc_G_relaxation!(y.tpo.mb_relax, y.tpo.H_ice, H_ref,
                           y.tpo.tau_relax, dt)
        apply_tendency!(y.tpo.H_ice, y.tpo.mb_relax, dt; adjust_mb=true)

        # Refresh f_ice after relaxation.
        calc_f_ice!(y.tpo.f_ice, y.tpo.H_ice)
    else
        fill!(interior(y.tpo.mb_relax), 0.0)
    end

    # Residual cleanup tendency for margin/island regularisation.
    resid_tendency!(y.tpo.mb_resid, y.tpo.H_ice, y.tpo.f_ice, y.tpo.f_grnd,
                    y.bnd.ice_allowed,
                    y.p.ytopo.H_min_flt, y.p.ytopo.H_min_grnd, dt)
    apply_tendency!(y.tpo.H_ice, y.tpo.mb_resid, dt; adjust_mb=true)

    # Final f_ice refresh after the cleanup step.
    calc_f_ice!(y.tpo.f_ice, y.tpo.H_ice)

    # Net mass balance applied this step.
    interior(y.tpo.mb_net) .= interior(y.tpo.smb) .+
                              interior(y.tpo.bmb) .+
                              interior(y.tpo.fmb) .+
                              interior(y.tpo.dmb) .+
                              interior(y.tpo.cmb) .+
                              interior(y.tpo.mb_relax) .+
                              interior(y.tpo.mb_resid)

    _update_diagnostics!(y, H_prev, H_after_dyn, dt)

    y.time += dt
    return y
end

# Per-cell post-step pass keyed off bnd.mask_ice. Stored values are
# Float64 representations of the Int constants from YelmoCore.
function _apply_mask_ice_pass!(y::YelmoModel, H_prev::AbstractArray)
    H_ice    = interior(y.tpo.H_ice)
    mask_ice = interior(y.bnd.mask_ice)
    @inbounds for j in axes(H_ice, 2), i in axes(H_ice, 1)
        m = mask_ice[i, j, 1]
        if m == Float64(MASK_ICE_NONE)
            H_ice[i, j, 1] = 0.0
        elseif m == Float64(MASK_ICE_FIXED)
            H_ice[i, j, 1] = H_prev[i, j, 1]
        else  # MASK_ICE_DYNAMIC (default)
            H_ice[i, j, 1] = max(H_ice[i, j, 1], 0.0)
        end
    end
    return y
end

# Recompute Phase-10 diagnostics from current state.
#  - Refresh `H_grnd` (flotation diagnostic), then `f_grnd` via the
#    CISM bilinear-interpolation subgrid scheme.
#  - `z_srf` / `z_base` from the standard grounded vs. floating
#    formulae, keyed off the binary `H_grnd > 0` test (subgrid
#    `z_srf` would land in a later milestone).
#  - `dHidt = (H_now - H_prev) / dt`           — total step rate.
#  - `dHidt_dyn = (H_after_dyn - H_prev) / dt` — dynamic-only rate
#    (post-advection / mask-pass, before SMB or mb_resid).
function _update_diagnostics!(y::YelmoModel,
                              H_prev::AbstractArray,
                              H_after_dyn::AbstractArray,
                              dt::Real)
    calc_H_grnd!(y.tpo.H_grnd, y.tpo.H_ice, y.bnd.z_bed, y.bnd.z_sl,
                 y.c.rho_ice, y.c.rho_sw)
    determine_grounded_fractions!(y.tpo.f_grnd, y.tpo.H_grnd)

    H_ice     = interior(y.tpo.H_ice)
    H_grnd    = interior(y.tpo.H_grnd)
    z_bed     = interior(y.bnd.z_bed)
    z_sl      = interior(y.bnd.z_sl)
    z_srf     = interior(y.tpo.z_srf)
    z_base    = interior(y.tpo.z_base)
    dHidt     = interior(y.tpo.dHidt)
    dHidt_dyn = interior(y.tpo.dHidt_dyn)

    inv_dt = dt > 0 ? 1.0 / dt : 0.0
    rho_ratio_iw = y.c.rho_ice / y.c.rho_sw

    @inbounds for j in axes(H_ice, 2), i in axes(H_ice, 1)
        if H_grnd[i, j, 1] > 0.0
            z_base[i, j, 1] = z_bed[i, j, 1]
            z_srf[i, j, 1]  = z_bed[i, j, 1] + H_ice[i, j, 1]
        else
            z_base[i, j, 1] = z_sl[i, j, 1] - rho_ratio_iw * H_ice[i, j, 1]
            z_srf[i, j, 1]  = z_base[i, j, 1] + H_ice[i, j, 1]
        end

        dHidt[i, j, 1]     = (H_ice[i, j, 1]       - H_prev[i, j, 1]) * inv_dt
        dHidt_dyn[i, j, 1] = (H_after_dyn[i, j, 1] - H_prev[i, j, 1]) * inv_dt
    end

    return y
end

end # module YelmoModelTopo
