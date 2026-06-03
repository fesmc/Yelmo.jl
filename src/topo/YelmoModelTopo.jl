"""
    YelmoModelTopo

Topography (`tpo`) component for the pure-Julia `YelmoModel`. Evolves
ice thickness `H_ice` via mass conservation and updates derived
quantities (`z_srf`, `z_base`, `dHidt`, `dHidt_dyn`, `f_grnd`,
`f_ice`).

Public surface: `topo_step!(y::YelmoModel, dt)`, dispatched from
`YelmoCore.step!(::YelmoModel, dt)` in fixed phase order alongside
the other `<comp>_step!` generics.

Milestone 2b: advection (from 2a) followed by surface mass balance
(SMB) and a residual cleanup tendency (`mb_resid`) that handles
margin/island regularisation. BMB, FMB, DMB, calving, the
predictor-corrector wrapping, and the `impl-lis` solver land in
subsequent milestones.
"""
module YelmoModelTopo

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using LoopVectorization: @turbo

using ..YelmoCore: AbstractYelmoModel, YelmoModel,
                   MASK_ICE_NONE, MASK_ICE_FIXED, MASK_ICE_DYNAMIC

import ..YelmoCore: topo_step!, update_diagnostics!

export topo_step!, topo_pc_step!, PCStageBuf,
       _alloc_pc_stage_buf, _save_pc_stage!, _load_pc_stage!,
       apply_mask_ice_pass!, advect_tracer!,
       advect_tracer_upwind_explicit!, advect_tracer_upwind_implicit!,
       advection_tendency!,
       AdvectionCache, init_advection_cache,
       update_advection_matrix!, update_advection_operator!,
       solve_advection!,
       apply_tendency!, mbal_tendency!, resid_tendency!,
       calc_f_ice!,
       calc_H_grnd!, determine_grounded_fractions!,
       calc_bmb_total!, calc_fmb_total!, calc_mb_discharge!,
       set_tau_relax!, calc_G_relaxation!,
       calc_calving_equil_ac!, calc_calving_threshold_ac!,
       calc_calving_vonmises_m16_ac!, merge_calving_rates!,
       lsf_init!, lsf_update!, lsf_redistance!,
       extrapolate_ocn_acx!, extrapolate_ocn_acy!,
       calving_step!,
       calc_distance_to_grounding_line!, calc_distance_to_ice_margin!,
       calc_grounding_line_zone!, gen_mask_bed!, calc_ice_front!,
       calc_z_srf!,
       calc_gradient_acx!, calc_gradient_acy!,
       calc_f_grnd_subgrid_linear!, calc_f_grnd_subgrid_area!,
       calc_f_grnd_pinning_points!, calc_grounded_fractions!,
       extend_floating_slab!, calc_dynamic_ice_fields!,
       update_diagnostics!

include("advection.jl")
include("mass_balance.jl")
include("grounded.jl")
include("ice_fraction.jl")
include("basal.jl")
include("frontal.jl")
include("discharge.jl")
include("relaxation.jl")
include("calving_ac.jl")
include("lsf.jl")
include("calving.jl")
include("distances.jl")
include("bed_mask.jl")
include("surface.jl")
include("gradients.jl")
include("dynamic_thickness.jl")

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
function topo_step!(y::YelmoModel, dt::Float64;
                    ux_bar::Union{Nothing, AbstractField} = nothing,
                    uy_bar::Union{Nothing, AbstractField} = nothing,
                    advance_time::Bool = true)
    # Optional `ux_bar` / `uy_bar` kwargs let the adaptive PC driver
    # supply an explicit velocity Field for the advection sub-step
    # while leaving everything else in the cascade unchanged. Default
    # `nothing` means "read from `y.dyn`" (legacy behaviour). The
    # `advance_time` flag is reserved for callers that want to run
    # the full cascade without bumping `y.time` (PC predictor stage).
    ux_bar_use = ux_bar === nothing ? y.dyn.ux_bar : ux_bar
    uy_bar_use = uy_bar === nothing ? y.dyn.uy_bar : uy_bar

    H_prev = copy(interior(y.tpo.H_ice))

    # Snapshot the start-of-step thickness into `H_ice_n` so that
    # `topo_rel_field == "H_ice_n"` (relax-toward-previous) has a
    # meaningful target. Fortran does the same in calc_ytopo_pc.
    interior(y.tpo.H_ice_n) .= H_prev

    if !y.p.ytopo.topo_fixed
        scheme = parse_advection_scheme(y.p.ytopo.solver)
        if scheme !== :none
            advect_tracer!(y.tpo.H_ice, ux_bar_use, uy_bar_use, dt;
                           scheme = scheme,
                           cache  = y.tpo.scratch.adv_cache,
                           cfl_safety = y.p.yelmo.cfl_max)
        end
    end

    _apply_mask_ice_pass!(y)

    # Snapshot after dynamics + mask pass; used for `dHidt_dyn`.
    H_after_dyn = copy(interior(y.tpo.H_ice))

    # Refresh f_ice now that the dynamic margin may have moved.
    calc_f_ice!(y)

    _topo_mb_cascade!(y, dt)

    _update_diagnostics!(y, H_prev, H_after_dyn, dt)

    advance_time && (y.time += dt)
    return y
end

# ---------------------------------------------------------------------------
# Mass-balance cascade — Fortran's per-step (smb → bmb → fmb → dmb →
# calving → relax → resid) chain. Called from `topo_step!` after the
# advective stage (legacy path), and from each `topo_pc_step!` mode
# (`:predictor` and `:corrector`) for the advective-PC path.
#
# Invariants on entry:
#   - `y.tpo.H_ice` holds the post-advection thickness.
#   - `f_ice` has been refreshed against this `H_ice`.
#
# Invariants on exit:
#   - All MB tendency fields (`smb`, `bmb`, `fmb`, `dmb`, `cmb*`,
#     `mb_relax`, `mb_resid`) are populated with realised rates [m/yr].
#   - `mb_net` is the sum.
#   - `f_ice` reflects the final post-cascade thickness.
# ---------------------------------------------------------------------------
function _topo_mb_cascade!(y::YelmoModel, dt::Float64)
    # Surface mass balance: clip the raw forcing field, then apply.
    mbal_tendency!(y.tpo.smb, y.tpo.H_ice, y.tpo.f_grnd, y.bnd.smb_ref, dt)
    apply_tendency!(y.tpo.H_ice, y.tpo.smb, dt; adjust_mb=true)

    # SMB may have grown / shrunk margins; refresh f_ice before BMB.
    calc_f_ice!(y)

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
    calc_f_ice!(y)

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
    calc_f_ice!(y)

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
    calc_f_ice!(y)

    # Phase 7: level-set calving. No-op when `ycalv.use_lsf` is false.
    calving_step!(y, dt)

    # Calving may have killed cells; refresh f_ice before relaxation.
    calc_f_ice!(y)

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
            error("_topo_mb_cascade!: unknown topo_rel_field = \"$(y.p.ytopo.topo_rel_field)\". " *
                  "Supported: \"H_ref\", \"H_ice_n\".")
        end

        calc_G_relaxation!(y.tpo.mb_relax, y.tpo.H_ice, H_ref,
                           y.tpo.tau_relax, dt)
        apply_tendency!(y.tpo.H_ice, y.tpo.mb_relax, dt; adjust_mb=true)

        # Refresh f_ice after relaxation.
        calc_f_ice!(y)
    else
        fill!(interior(y.tpo.mb_relax), 0.0)
    end

    # Residual cleanup tendency for margin/island regularisation.
    resid_tendency!(y.tpo.mb_resid, y.tpo.H_ice, y.tpo.f_ice, y.tpo.f_grnd,
                    y.bnd.mask_ice, y.bnd.H_ice_ref,
                    y.p.ytopo.H_min_flt, y.p.ytopo.H_min_grnd, dt)
    apply_tendency!(y.tpo.H_ice, y.tpo.mb_resid, dt; adjust_mb=true)

    # Final f_ice refresh after the cleanup step.
    calc_f_ice!(y)

    # Net mass balance applied this step.
    interior(y.tpo.mb_net) .= interior(y.tpo.smb) .+
                              interior(y.tpo.bmb) .+
                              interior(y.tpo.fmb) .+
                              interior(y.tpo.dmb) .+
                              interior(y.tpo.cmb) .+
                              interior(y.tpo.mb_relax) .+
                              interior(y.tpo.mb_resid)
    return y
end

# ---------------------------------------------------------------------------
# PC stage buffer — mirrors Fortran's `tpo%now%pred` / `tpo%now%corr`
# (yelmo_topography.f90:322-353). Stores the per-stage outputs that
# `topo_pc_step!(:advance)` reads to commit either the predictor or the
# corrector result into the live state.
# ---------------------------------------------------------------------------

"""
    PCStageBuf

Per-stage scratch buffer holding the 13 fields that Fortran's
`tpo%now%pred` / `tpo%now%corr` carry. Allocated lazily by the
adaptive-PC driver; one instance for the predictor stage, one for the
corrector stage. See `topo_pc_step!`.
"""
mutable struct PCStageBuf
    H_ice    ::Array{Float64,3}
    dHidt_dyn::Array{Float64,3}
    mb_net   ::Array{Float64,3}
    mb_relax ::Array{Float64,3}
    mb_resid ::Array{Float64,3}
    smb      ::Array{Float64,3}
    bmb      ::Array{Float64,3}
    fmb      ::Array{Float64,3}
    dmb      ::Array{Float64,3}
    cmb      ::Array{Float64,3}
    cmb_flt  ::Array{Float64,3}
    cmb_grnd ::Array{Float64,3}
    lsf      ::Array{Float64,3}
end

function _alloc_pc_stage_buf(y::YelmoModel)
    sz = size(interior(y.tpo.H_ice))
    Z() = zeros(Float64, sz)
    return PCStageBuf(Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z(),
                      Z(), Z(), Z(), Z(), Z())
end

function _save_pc_stage!(buf::PCStageBuf, y::YelmoModel)
    copyto!(buf.H_ice,     interior(y.tpo.H_ice))
    copyto!(buf.dHidt_dyn, interior(y.tpo.dHidt_dyn))
    copyto!(buf.mb_net,    interior(y.tpo.mb_net))
    copyto!(buf.mb_relax,  interior(y.tpo.mb_relax))
    copyto!(buf.mb_resid,  interior(y.tpo.mb_resid))
    copyto!(buf.smb,       interior(y.tpo.smb))
    copyto!(buf.bmb,       interior(y.tpo.bmb))
    copyto!(buf.fmb,       interior(y.tpo.fmb))
    copyto!(buf.dmb,       interior(y.tpo.dmb))
    copyto!(buf.cmb,       interior(y.tpo.cmb))
    copyto!(buf.cmb_flt,   interior(y.tpo.cmb_flt))
    copyto!(buf.cmb_grnd,  interior(y.tpo.cmb_grnd))
    copyto!(buf.lsf,       interior(y.tpo.lsf))
    return buf
end

function _load_pc_stage!(y::YelmoModel, buf::PCStageBuf)
    copyto!(interior(y.tpo.H_ice),     buf.H_ice)
    copyto!(interior(y.tpo.dHidt_dyn), buf.dHidt_dyn)
    copyto!(interior(y.tpo.mb_net),    buf.mb_net)
    copyto!(interior(y.tpo.mb_relax),  buf.mb_relax)
    copyto!(interior(y.tpo.mb_resid),  buf.mb_resid)
    copyto!(interior(y.tpo.smb),       buf.smb)
    copyto!(interior(y.tpo.bmb),       buf.bmb)
    copyto!(interior(y.tpo.fmb),       buf.fmb)
    copyto!(interior(y.tpo.dmb),       buf.dmb)
    copyto!(interior(y.tpo.cmb),       buf.cmb)
    copyto!(interior(y.tpo.cmb_flt),   buf.cmb_flt)
    copyto!(interior(y.tpo.cmb_grnd),  buf.cmb_grnd)
    copyto!(interior(y.tpo.lsf),       buf.lsf)
    return y
end

# ---------------------------------------------------------------------------
# Fortran-style advective predictor / corrector — ported from
# `calc_ytopo_pc` (yelmo/src/yelmo_topography.f90:42). Three modes:
#
#   - `:predictor` — snapshot input state (`H_ice_n`, `dHidt_dyn_n`,
#     `lsf_n`); compute pure advective tendency at `H_n` with the
#     current velocity field; β-mix `dHidt_dyn = β1·dHidt_now +
#     β2·dHidt_dyn_n`; apply the tendency on top of `H_ice_n` (with
#     `mb_lim = dHdt_dyn_lim`); run the full MB cascade on top of the
#     resulting `H_pred`; save the per-stage outputs to `pred_buf`.
#     On exit, live `y.tpo.H_ice = H_pred` — the next `dyn_step!` solves
#     SSA at this state.
#
#   - `:corrector` — load `H_pred` from `pred_buf` into live state;
#     compute pure advective tendency at `H_pred` with the newly-solved
#     velocity field; β-mix `dHidt_dyn = β3·dHidt_now + β4·dHidt_dyn_n`;
#     apply on top of `H_ice_n` (not `H_pred`); run MB cascade; save
#     outputs to `corr_buf`. Restore `H_ice = H_ice_n` at end so the
#     downstream mat/therm steps see the start-of-step geometry.
#
#   - `:advance` — copy `pred_buf` or `corr_buf` (per `use_H_pred`) into
#     live state, refresh diagnostics, set `dHidt = (H_now − H_ice_n)/dt`,
#     advance `y.time` by `dt`.
#
# `H_scratch` is a `Nx×Ny×1` Float64 buffer used by the snapshot-diff
# advection-tendency wrapper; the caller owns it (typically on
# `PCScratch`).
# ---------------------------------------------------------------------------
function topo_pc_step!(y::YelmoModel, dt::Float64;
                       mode::Symbol,
                       β1::Float64 = 1.0,
                       β2::Float64 = 0.0,
                       β3::Float64 = 1.0,
                       β4::Float64 = 0.0,
                       pred_buf::PCStageBuf,
                       corr_buf::PCStageBuf,
                       H_scratch::AbstractArray,
                       use_H_pred::Bool = true,
                       advance_time::Bool = true)
    if mode === :predictor
        # 1. Snapshot input state.
        copyto!(interior(y.tpo.H_ice_n),     interior(y.tpo.H_ice))
        copyto!(interior(y.tpo.dHidt_dyn_n), interior(y.tpo.dHidt_dyn))
        copyto!(interior(y.tpo.lsf_n),       interior(y.tpo.lsf))
        copyto!(interior(y.tpo.z_srf_n),     interior(y.tpo.z_srf))

        calc_f_ice!(y)

        # 2. Pure advective tendency at H_n with current velocity.
        _advection_tendency_mix!(y, dt, H_scratch, β1, β2)

        # 3. Apply mixed advective tendency on top of H_n (H_ice already
        # equals H_ice_n since the tendency helper restores).
        apply_tendency!(y.tpo.H_ice, y.tpo.dHidt_dyn, dt;
                        adjust_mb = true,
                        mb_lim    = y.p.ytopo.dHdt_dyn_lim)

        # 4. Mask-ice post-pass (NONE→0, FIXED→H_ice_ref, DYNAMIC→max(0)).
        apply_mask_ice_pass!(y)
        calc_f_ice!(y)

        # 5. MB cascade on H_pred.
        _topo_mb_cascade!(y, dt)

        # 6. Refresh diagnostics for the upcoming dyn solve at H_pred.
        # (`dt = 0` form preserves dHidt_dyn, which is our β-mixed value.)
        update_diagnostics!(y)

        # 7. Save predictor outputs.
        _save_pc_stage!(pred_buf, y)
        return y

    elseif mode === :corrector
        # 1. Load H_pred + lsf_pred into live state.
        copyto!(interior(y.tpo.H_ice), pred_buf.H_ice)
        copyto!(interior(y.tpo.lsf),   pred_buf.lsf)
        calc_f_ice!(y)

        # 2. Pure advective tendency at H_pred with just-solved u_pred.
        _advection_tendency_mix!(y, dt, H_scratch, β3, β4)

        # 3. Restore H_ice ← H_ice_n; apply mixed tendency.
        copyto!(interior(y.tpo.H_ice), interior(y.tpo.H_ice_n))
        copyto!(interior(y.tpo.lsf),   interior(y.tpo.lsf_n))
        apply_tendency!(y.tpo.H_ice, y.tpo.dHidt_dyn, dt;
                        adjust_mb = true,
                        mb_lim    = y.p.ytopo.dHdt_dyn_lim)

        # 4. Mask-ice pass and f_ice refresh.
        apply_mask_ice_pass!(y)
        calc_f_ice!(y)

        # 5. MB cascade on H_corr.
        _topo_mb_cascade!(y, dt)

        # 6. Refresh diagnostics at H_corr so eta-masking and any
        # immediate post-corrector logic see consistent z_srf / H_grnd /
        # gradients at the corrector geometry.
        update_diagnostics!(y)

        # 7. Save corrector outputs.
        # NOTE: live state is left at H_corr on exit. The PC driver
        # restores `H_ice ← H_ice_n` after `_compute_pc_eta` reads
        # corrector-state diagnostics, before running mat/therm (which
        # Fortran's `update_others_pc = false` evaluates at H_ice_n).
        _save_pc_stage!(corr_buf, y)
        return y

    elseif mode === :advance
        # Commit pred or corr buffer into live state.
        src = use_H_pred ? pred_buf : corr_buf
        _load_pc_stage!(y, src)

        # Refresh diagnostics (z_srf, gradients, distances, masks) from
        # the committed H_ice. `update_diagnostics!` preserves dHidt_dyn
        # (we want the saved β-mix value, not a (H_after - H_prev)/dt
        # snapshot).
        update_diagnostics!(y)

        # dHidt = (H_now − H_ice_n) / dt — total step rate.
        if dt > 0
            dHidt = interior(y.tpo.dHidt)
            H_now = interior(y.tpo.H_ice)
            H_n   = interior(y.tpo.H_ice_n)
            inv_dt = 1.0 / dt
            @inbounds @simd for i in eachindex(dHidt)
                dHidt[i] = (H_now[i] - H_n[i]) * inv_dt
            end
        end

        advance_time && (y.time += dt)
        return y
    else
        error("topo_pc_step!: unknown mode=$(mode). " *
              "Use :predictor, :corrector, or :advance.")
    end
end

# Internal helper: compute the pure advective tendency into y.tpo.dHidt_dyn
# (mutating it via β-mix with y.tpo.dHidt_dyn_n) without modifying H_ice.
# Honors `topo_fixed` (writes zero tendency) and the scheme dispatch.
function _advection_tendency_mix!(y::YelmoModel, dt::Float64,
                                  H_scratch::AbstractArray,
                                  βnow::Float64, βn::Float64)
    if y.p.ytopo.topo_fixed
        fill!(interior(y.tpo.dHidt_dyn), 0.0)
        return y
    end
    scheme = parse_advection_scheme(y.p.ytopo.solver)
    if scheme === :none
        fill!(interior(y.tpo.dHidt_dyn), 0.0)
        return y
    end
    advection_tendency!(y.tpo.dHidt_dyn, y.tpo.H_ice,
                        y.dyn.ux_bar, y.dyn.uy_bar, dt, H_scratch;
                        scheme     = scheme,
                        cache      = y.tpo.scratch.adv_cache,
                        cfl_safety = y.p.yelmo.cfl_max)
    dH  = interior(y.tpo.dHidt_dyn)
    dHn = interior(y.tpo.dHidt_dyn_n)
    @inbounds @simd for i in eachindex(dH)
        dH[i] = βnow * dH[i] + βn * dHn[i]
    end
    return y
end

"""
    update_diagnostics!(y::YelmoModel) -> y

Recompute every diagnostic `tpo` field from the current prognostic
state (`H_ice` plus `bnd` inputs `z_bed`, `z_sl`, `f_pmp`, `z_bed_sd`)
without advancing time. Refreshes `f_ice` first (so the rest of the
diagnostic chain sees a consistent ice cover), then runs the same
diagnostic body that fires at the end of `topo_step!`.

`dHidt` and `dHidt_dyn` are preserved (not recomputed) since there is
no time step over which to differentiate. Useful to materialise
diagnostics after `load_state!` for restart files that omit some
derived fields, and as a regression check that the Julia diagnostic
chain reproduces what the Fortran reference wrote into a restart —
including consumers like `calc_uz_3D_jac!` that read `dHidt`.
"""
function update_diagnostics!(y::YelmoModel)
    calc_f_ice!(y)
    H = copy(interior(y.tpo.H_ice))
    _update_diagnostics!(y, H, H, 0.0)
    return y
end

# Per-cell post-step pass keyed off bnd.mask_ice. Stored values are
# Float64 representations of the Int constants from YelmoCore.
#
# This is an invariant-restore pass with no mass-balance accounting:
# the faithful `mb_resid` bookkeeping for the mask lives in
# `resid_tendency!` (the `calc_G_boundaries` port). This pass is needed
# at points where no residual step follows — notably after the Heun
# corrector average in `pc_step!` (timestepping.jl), where the
# (H_n + H_**)/2 average can violate mask invariants (e.g. a
# MASK_ICE_NONE cell gets H_corr = H_n/2 > 0), and after the advective
# stage in `topo_step!` / `topo_pc_step!` before the MB cascade reads
# the geometry. Semantics match Fortran: MASK_ICE_NONE → 0,
# MASK_ICE_FIXED → bnd.H_ice_ref, MASK_ICE_DYNAMIC → clamp ≥ 0.
function apply_mask_ice_pass!(y::YelmoModel)
    H_ice     = interior(y.tpo.H_ice)
    mask_ice  = interior(y.bnd.mask_ice)
    H_ice_ref = interior(y.bnd.H_ice_ref)
    _apply_mask_ice_pass_kernel!(H_ice, mask_ice, H_ice_ref)
    return y
end

# Branchless 3-way mask select for @turbo: the default `:dynamic` case
# is the always-computed `max(H, 0)` value, and the `:none` / `:fixed`
# cases override it via `ifelse`. Comparing Float64-encoded mask values
# against the Float64 cast of the integer constants matches the
# original 3-arm if/elseif/else exactly.
@inline function _apply_mask_ice_pass_kernel!(H_ice::AbstractArray{Float64},
                                              mask_ice::AbstractArray{Float64},
                                              H_ice_ref::AbstractArray{Float64})
    m_none  = Float64(MASK_ICE_NONE)
    m_fixed = Float64(MASK_ICE_FIXED)
    @turbo for j in axes(H_ice, 2), i in axes(H_ice, 1)
        m       = mask_ice[i, j, 1]
        H_dyn   = ifelse(H_ice[i, j, 1] > 0.0, H_ice[i, j, 1], 0.0)
        H_new   = ifelse(m == m_none,  0.0,
                  ifelse(m == m_fixed, H_ice_ref[i, j, 1], H_dyn))
        H_ice[i, j, 1] = H_new
    end
end
const _apply_mask_ice_pass! = apply_mask_ice_pass!

# Recompute Phase-10 diagnostics from current state.
#  - Refresh `H_grnd` (flotation diagnostic), then `f_grnd` via the
#    CISM bilinear-interpolation subgrid scheme.
#  - `z_srf` from `calc_z_srf!` (Pattyn 2017, Eq. 1 — max-of-grounded-
#    or-floating with sub-grid `f_ice < 1` collapsing to bare bed/sea
#    level). `z_base = z_srf - H_ice` per Fortran convention.
#  - `dHidt = (H_now - H_prev) / dt`           — total step rate.
#  - `dHidt_dyn = (H_after_dyn - H_prev) / dt` — dynamic-only rate
#    (post-advection / mask-pass, before SMB or mb_resid).
#  - `dist_grline` / `dist_margin` (m), `mask_grz`, `mask_bed`,
#    `mask_frnt`. The grounding-zone half-width parameter
#    `ytopo.dist_grz` is in km in the namelist; convert to metres for
#    the kernel.
function _update_diagnostics!(y::YelmoModel,
                              H_prev::AbstractArray,
                              H_after_dyn::AbstractArray,
                              dt::Real)
    calc_H_grnd!(y.tpo.H_grnd, y.tpo.H_ice, y.bnd.z_bed, y.bnd.z_sl,
                 y.c.rho_ice, y.c.rho_sw)

    # `gl_sep` dispatch: linear / area / CISM-quad subgrid grounded
    # fractions. `f_grnd_ab` only populated by gl_sep == 3.
    calc_grounded_fractions!(y.tpo.f_grnd, y.tpo.f_grnd_acx, y.tpo.f_grnd_acy,
                             y.tpo.f_grnd_ab, y.tpo.H_grnd,
                             y.p.ytopo.gl_sep;
                             gz_nx = y.p.ytopo.gz_nx)

    # Subgrid pinning-point fraction over floating ice (uses z_bed_sd).
    calc_f_grnd_pinning_points!(y.tpo.f_grnd_pin, y.tpo.H_ice, y.tpo.f_ice,
                                y.bnd.z_bed, y.bnd.z_bed_sd, y.bnd.z_sl,
                                y.c.rho_ice, y.c.rho_sw)

    calc_z_srf!(y.tpo.z_srf, y.tpo.H_ice, y.tpo.f_ice,
                y.bnd.z_bed, y.bnd.z_sl, y.c.rho_ice, y.c.rho_sw)

    H_ice     = interior(y.tpo.H_ice)
    z_srf     = interior(y.tpo.z_srf)
    z_base    = interior(y.tpo.z_base)

    @inbounds for j in axes(H_ice, 2), i in axes(H_ice, 1)
        # `z_base` follows the Fortran convention `z_srf - H_ice` so the
        # value is meaningful for both grounded (= z_bed) and floating
        # (= z_sl - rho_ice/rho_sw·H_ice) regimes.
        z_base[i, j, 1] = z_srf[i, j, 1] - H_ice[i, j, 1]
    end

    # Time-derivative tracking. Only recompute when there is a real
    # step to differentiate over. The `update_diagnostics!(y)` entry
    # point passes `dt = 0` (post-load refresh, no time advance); in
    # that case preserve the loaded values so consumers like
    # `calc_uz_3D_jac!` see the restart's `dHidt`.
    if dt > 0
        dHidt     = interior(y.tpo.dHidt)
        dHidt_dyn = interior(y.tpo.dHidt_dyn)
        inv_dt = 1.0 / dt
        @inbounds for j in axes(H_ice, 2), i in axes(H_ice, 1)
            dHidt[i, j, 1]     = (H_ice[i, j, 1]       - H_prev[i, j, 1]) * inv_dt
            dHidt_dyn[i, j, 1] = (H_after_dyn[i, j, 1] - H_prev[i, j, 1]) * inv_dt
        end
    end

    # Margin-aware horizontal gradients on staggered ac-faces.
    # `dHidx`/`dHidy` use `zero_outside` so partially-covered cells
    # collapse to 0 (matches Fortran's `zero_outside=.TRUE.` for
    # ice-thickness gradients).
    dx = _dx(y.g)
    dy = _dy(y.g)
    grad_lim  = y.p.ytopo.grad_lim
    margin2nd = y.p.ytopo.margin2nd
    # Periodic-wrap slope offsets. Default 0.0 (production ice sheets,
    # Bounded lateral axes); set in benchmark configs with a uniform-
    # slope surface across a periodic axis (HOM-C, MISMIP3D Stnd).
    # `z_base = z_srf - H_ice` and `H_ice` is periodic in those
    # configs, so the same offset applies to the `z_base` gradient.
    # `dHidx`/`dHidy` use offset = 0 (default) since `H_ice` is
    # periodic by construction in those benchmarks.
    dzsdx_off = y.p.ytopo.dzsdx_periodic_offset
    dzsdy_off = y.p.ytopo.dzsdy_periodic_offset

    calc_gradient_acx!(y.tpo.dzsdx, y.tpo.z_srf,  y.tpo.f_ice, dx;
                       grad_lim = grad_lim, margin2nd = margin2nd,
                       zero_outside = false,
                       periodic_offset = dzsdx_off)
    calc_gradient_acy!(y.tpo.dzsdy, y.tpo.z_srf,  y.tpo.f_ice, dy;
                       grad_lim = grad_lim, margin2nd = margin2nd,
                       zero_outside = false,
                       periodic_offset = dzsdy_off)

    calc_gradient_acx!(y.tpo.dHidx, y.tpo.H_ice,  y.tpo.f_ice, dx;
                       grad_lim = grad_lim, margin2nd = margin2nd,
                       zero_outside = true)
    calc_gradient_acy!(y.tpo.dHidy, y.tpo.H_ice,  y.tpo.f_ice, dy;
                       grad_lim = grad_lim, margin2nd = margin2nd,
                       zero_outside = true)

    calc_gradient_acx!(y.tpo.dzbdx, y.tpo.z_base, y.tpo.f_ice, dx;
                       grad_lim = grad_lim, margin2nd = margin2nd,
                       zero_outside = false,
                       periodic_offset = dzsdx_off)
    calc_gradient_acy!(y.tpo.dzbdy, y.tpo.z_base, y.tpo.f_ice, dy;
                       grad_lim = grad_lim, margin2nd = margin2nd,
                       zero_outside = false,
                       periodic_offset = dzsdy_off)

    # Distance-to-feature fields (metres) and bed-state masks.
    calc_distance_to_grounding_line!(y.tpo.dist_grline, y.tpo.f_grnd, dx)
    calc_distance_to_ice_margin!(y.tpo.dist_margin,  y.tpo.f_ice,  dx)

    # `dist_grz` parameter is in km; convert to metres.
    dist_grz_m = 1e3 * y.p.ytopo.dist_grz
    calc_grounding_line_zone!(y.tpo.mask_grz, y.tpo.dist_grline, dist_grz_m)

    gen_mask_bed!(y.tpo.mask_bed, y.tpo.f_ice, y.thrm.f_pmp,
                  y.tpo.f_grnd, y.tpo.mask_grz)

    calc_ice_front!(y.tpo.mask_frnt, y.tpo.f_ice, y.tpo.f_grnd,
                    y.bnd.z_bed, y.bnd.z_sl)

    # Dynamics-only thickness/cover fields, dispatched on
    # `ydyn.ssa_lat_bc`. Default ("floating") is pass-through.
    calc_dynamic_ice_fields!(y)

    return y
end

end # module YelmoModelTopo
