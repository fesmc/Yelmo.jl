"""
    YelmoModelTopo

Topography (`tpo`) component for the pure-Julia `YelmoModel`. Evolves
ice thickness `H_ice` via mass conservation and updates derived
quantities (`z_srf`, `z_base`, `dHidt`, `dHidt_dyn`, `f_grnd`,
`f_ice`).

Public surface: the predictor-corrector stages
`topo_step!(y, dt, ::PCPredictor / ::PCCorrector / ::PCAdvance)` (port of
Fortran `calc_ytopo_pc`), driven by the time loop in
src/timestepping.jl, and `update_diagnostics!`.
"""
module YelmoModelTopo

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using LoopVectorization: @turbo

using ..YelmoCore: AbstractYelmoModel, YelmoModel,
                   MASK_ICE_NONE, MASK_ICE_FIXED, MASK_ICE_DYNAMIC

import ..YelmoCore: topo_step!, update_diagnostics!

export topo_step!, PCStage, PCPredictor, PCCorrector, PCAdvance,
       advect_tracer!,
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
       calc_dynamic_ice_fields!,
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

# ---------------------------------------------------------------------------
# Predictor-corrector topography stages — port of Fortran `calc_ytopo_pc`
# (yelmo/src/yelmo_topography.f90:42-531). The time loop in
# src/timestepping.jl calls, per step:
#
#   topo_step!(y, dt, PCPredictor(); β1, β2)    → H_pred (live state)
#   dyn_step!(y, dt)                            → velocity at H_pred
#   topo_step!(y, dt, PCCorrector(); β3, β4)    → H_corr (live state back to H_n)
#   … error check, possibly a redo …
#   mat_step!, therm_step!                      (at H_n)
#   topo_step!(y, dt, PCAdvance(); use_H_pred)  → H_{n+1}
#
# Transport: the predictor and corrector advect with the transport
# velocity `y.tpo.scratch.pc.ux_t/uy_t` (filled by the time loop before each of
# them, Fortran `calc_transport_velocity`), mix the raw advective rates
# with the β coefficients of the PC scheme,
#
#   predictor: dHidt_dyn = β1·f(H_n, u_n)   + β2·f_{n-1}
#   corrector: dHidt_dyn = β3·f(H_pred, u*) + β4·f(H_n, u_n)
#
# (f(H_n, u_n) = `y.tpo.scratch.pc.dHidt_dyn_raw`, f_{n-1} = `dHidt_dyn_raw_n`) and
# apply the mixed rate to H_n. The clip of negative thickness is booked
# in `mb_clip`, so `dHidt_dyn` is the applied transport rate. The mass
# balance cascade (smb, bmb, fmb, dmb, calving, relaxation, residual)
# then acts on the transported thickness.
# ---------------------------------------------------------------------------

"""
    PCStage, PCPredictor, PCCorrector, PCAdvance

Stages of a predictor-corrector topography step (Fortran `calc_ytopo_pc`
with `pc_step` = "predictor", "corrector", "advance"); see `topo_step!`.
"""
abstract type PCStage end
struct PCPredictor <: PCStage end
struct PCCorrector <: PCStage end
struct PCAdvance   <: PCStage end

"""
    topo_step!(y, dt, ::PCPredictor; β1, β2) -> y
    topo_step!(y, dt, ::PCCorrector; β3, β4) -> y
    topo_step!(y, dt, ::PCAdvance; use_H_pred) -> y

One stage of the predictor-corrector topography update over `dt` (port of
Fortran `calc_ytopo_pc`):

  - `PCPredictor`: store `H_ice_n`, `z_srf_n`, `lsf_n`; transport with
    `β1·f_n + β2·f_{n-1}` (raw advective rates at `H_n` and of the previous
    step) and run the mass-balance cascade. The live state and
    `y.tpo.scratch.pc.pred` hold `H_pred`, for the velocity solve that follows.
  - `PCCorrector`: transport `H_n` with `β3·f(H_pred) + β4·f_n` (`H_pred`
    advected with the velocity just solved), run the cascade and record
    the result in `y.tpo.scratch.pc.corr`, then return the live state to `H_n`
    (for `mat_step!` and `therm_step!`).
  - `PCAdvance`: load the predictor (`use_H_pred`) or corrector record and
    shift `f_n` to `dHidt_dyn_raw_n`.

The predictor and corrector advect with `y.tpo.scratch.pc.ux_t/uy_t`, which the
caller fills with the transport velocity. Nothing changes when
`ytopo.topo_fixed` or `dt ≤ 0` (the records then hold the current state).
Each transporting stage ends with `dHidt = (H_ice − H_ice_n)/dt`,
`dlsfdt` and `mb_err`; every stage ends with `update_diagnostics!`.
`y.time` belongs to the time loop.
"""
function topo_step!(y::YelmoModel, dt::Float64, ::PCPredictor;
                    β1::Float64, β2::Float64)
    tpo = y.tpo
    copyto!(interior(tpo.H_ice_n), interior(tpo.H_ice))
    copyto!(interior(tpo.z_srf_n), interior(tpo.z_srf))
    copyto!(interior(tpo.lsf_n),   interior(tpo.lsf))
    if _topo_active(y, dt)
        f_n = tpo.scratch.pc.dHidt_dyn_raw
        lin_iter, lin_fail = _advection_rate!(f_n, y, dt)
        tpo.scratch.adv_lin_iter[] = lin_iter
        tpo.scratch.adv_lin_fail[] = lin_fail
        _mix_rates!(interior(tpo.dHidt_dyn), β1, f_n, β2, interior(tpo.dHidt_dyn_raw_n))
        _apply_transport!(y, dt)
        _topo_mb_cascade!(y, dt)
        _stage_rates!(y, dt)
    end
    update_diagnostics!(y)
    _save_pc_stage!(tpo.scratch.pc.pred, y)
    return y
end

function topo_step!(y::YelmoModel, dt::Float64, ::PCCorrector;
                    β3::Float64, β4::Float64)
    tpo = y.tpo
    if _topo_active(y, dt)
        copyto!(interior(tpo.H_ice), tpo.scratch.pc.pred.H_ice)
        copyto!(interior(tpo.lsf),   tpo.scratch.pc.pred.lsf)
        calc_f_ice!(y)
        D = interior(tpo.dHidt_dyn)
        lin_iter, lin_fail = _advection_rate!(D, y, dt)
        tpo.scratch.adv_lin_iter[] += lin_iter
        tpo.scratch.adv_lin_fail[] += lin_fail
        _mix_rates!(D, β3, D, β4, tpo.scratch.pc.dHidt_dyn_raw)
        copyto!(interior(tpo.H_ice), interior(tpo.H_ice_n))
        copyto!(interior(tpo.lsf),   interior(tpo.lsf_n))
        _apply_transport!(y, dt)
        _topo_mb_cascade!(y, dt)
        _save_pc_stage!(tpo.scratch.pc.corr, y)
        copyto!(interior(tpo.H_ice), interior(tpo.H_ice_n))
        copyto!(interior(tpo.lsf),   interior(tpo.lsf_n))
        _stage_rates!(y, dt)
    else
        _save_pc_stage!(tpo.scratch.pc.corr, y)
    end
    update_diagnostics!(y)
    return y
end

function topo_step!(y::YelmoModel, dt::Float64, ::PCAdvance; use_H_pred::Bool)
    tpo = y.tpo
    if _topo_active(y, dt)
        _load_pc_stage!(y, use_H_pred ? tpo.scratch.pc.pred : tpo.scratch.pc.corr)
        copyto!(interior(tpo.dHidt_dyn_raw_n), tpo.scratch.pc.dHidt_dyn_raw)
        _stage_rates!(y, dt)
    end
    update_diagnostics!(y)
    return y
end

_topo_active(y::YelmoModel, dt::Real) = !y.p.ytopo.topo_fixed && dt > 0

# Raw advective rate `f = (advected H − H)/dt` of the live H_ice with the
# transport velocity (Fortran `calc_G_advec_simple`), into the array `dHdt`.
function _advection_rate!(dHdt::AbstractArray, y::YelmoModel, dt::Float64)
    scheme = parse_advection_scheme(y.p.ytopo.solver)
    if scheme === :none
        fill!(dHdt, 0.0)
        return (0, 0)
    end
    advection_tendency!(dHdt, y.tpo.H_ice, y.tpo.scratch.pc.ux_t, y.tpo.scratch.pc.uy_t, dt,
                        y.tpo.scratch.pc.H_tmp;
                        scheme     = scheme,
                        cache      = y.tpo.scratch.adv_cache,
                        cfl_safety = y.p.yelmo.cfl_max)
    return advection_solve_stats(scheme, y.tpo.scratch.adv_cache[])
end

# D = a·X + b·Y (X may be D).
function _mix_rates!(D::AbstractArray, a::Float64, X::AbstractArray,
                     b::Float64, Y::AbstractArray)
    @inbounds @simd for i in eachindex(D)
        D[i] = a * X[i] + b * Y[i]
    end
    return D
end

# Apply the mixed transport rate `dHidt_dyn` to the live H_ice (= H_n);
# the clip of negative thickness goes to `mb_clip` and `dHidt_dyn` becomes
# the applied transport rate (Fortran yelmo_topography.f90:161-165).
function _apply_transport!(y::YelmoModel, dt::Float64)
    D = y.tpo.dHidt_dyn
    apply_tendency!(y.tpo.H_ice, D, dt; adjust_mb = true, mb_clip = y.tpo.mb_clip)
    interior(D) .-= interior(y.tpo.mb_clip)
    calc_f_ice!(y)
    return y
end

# Rates of the stage (Fortran yelmo_topography.f90:517-524): total
# thickness and level-set rates, and the mass budget residual, which
# vanishes to round-off since every tendency is applied with `adjust_mb`.
function _stage_rates!(y::YelmoModel, dt::Float64)
    tpo = y.tpo
    inv_dt = 1.0 / dt
    interior(tpo.dHidt)  .= (interior(tpo.H_ice) .- interior(tpo.H_ice_n)) .* inv_dt
    interior(tpo.dlsfdt) .= (interior(tpo.lsf)   .- interior(tpo.lsf_n))   .* inv_dt
    interior(tpo.mb_err) .= interior(tpo.dHidt) .-
                            (interior(tpo.dHidt_dyn) .+ interior(tpo.mb_clip) .+
                             interior(tpo.mb_net) .+ interior(tpo.cmb))
    return y
end

function _save_pc_stage!(rec::NamedTuple, y::YelmoModel)
    for k in keys(rec)
        copyto!(rec[k], interior(getproperty(y.tpo, k)))
    end
    return rec
end

function _load_pc_stage!(y::YelmoModel, rec::NamedTuple)
    for k in keys(rec)
        copyto!(interior(getproperty(y.tpo, k)), rec[k])
    end
    return y
end

# ---------------------------------------------------------------------------
# Mass-balance cascade — Fortran's per-stage (smb → bmb → fmb → dmb →
# calving → relax → resid) chain, run by the predictor and corrector
# stages after transport (yelmo_topography.f90:212-375).
#
# Invariants on entry:
#   - `y.tpo.H_ice` holds the post-advection thickness.
#   - `f_ice` has been refreshed against this `H_ice`.
#
# Invariants on exit:
#   - All MB tendency fields (`smb`, `bmb`, `fmb`, `dmb`, `cmb*`,
#     `mb_relax`, `mb_resid`) are populated with realised rates [m/yr].
#   - `mb_net` is their sum without calving (Fortran: `cmb` is booked
#     separately).
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
                    y.p.ycalv.H_min_flt, y.p.ycalv.H_min_grnd, dt)
    apply_tendency!(y.tpo.H_ice, y.tpo.mb_resid, dt; adjust_mb=true)

    # Final f_ice refresh after the cleanup step.
    calc_f_ice!(y)

    # Net mass balance applied this stage (calving `cmb` is separate,
    # as in Fortran).
    interior(y.tpo.mb_net) .= interior(y.tpo.smb) .+
                              interior(y.tpo.bmb) .+
                              interior(y.tpo.fmb) .+
                              interior(y.tpo.dmb) .+
                              interior(y.tpo.mb_relax) .+
                              interior(y.tpo.mb_resid)
    return y
end

"""
    update_diagnostics!(y::YelmoModel) -> y

Recompute every diagnostic `tpo` field from the current prognostic
state (`H_ice` plus `bnd` inputs `z_bed`, `z_sl`, `f_pmp`, `z_bed_sd`)
without advancing time (Fortran `calc_ytopo_diagnostic`). Refreshes
`f_ice` first, so the rest of the chain sees a consistent ice cover.
Runs at the end of every topography stage; also useful to materialise
diagnostics after `load_state!`.

Rates (`dHidt`, `dHidt_dyn`, `dlsfdt`) are not touched: the topography
stages set them.
"""
function update_diagnostics!(y::YelmoModel)
    calc_f_ice!(y)
    _update_diagnostics!(y)
    return y
end

# Recompute the diagnostics from the current state.
#  - Refresh `H_grnd` (flotation diagnostic), then `f_grnd` via the
#    CISM bilinear-interpolation subgrid scheme.
#  - `z_srf` from `calc_z_srf!` (Pattyn 2017, Eq. 1 — max-of-grounded-
#    or-floating with sub-grid `f_ice < 1` collapsing to bare bed/sea
#    level). `z_base = z_srf - H_ice` per Fortran convention.
#  - `dist_grline` / `dist_margin` (m), `mask_grz`, `mask_bed`,
#    `mask_frnt`. The grounding-zone half-width parameter
#    `ytopo.dist_grz` is in km in the namelist; convert to metres for
#    the kernel.
function _update_diagnostics!(y::YelmoModel)
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

    # Margin-aware horizontal gradients on staggered ac-faces.
    # `dHidx`/`dHidy` use `zero_outside` so partially-covered cells
    # collapse to 0 (matches Fortran's `zero_outside=.TRUE.` for
    # ice-thickness gradients).
    dx = _dx(y.g)
    dy = _dy(y.g)
    grad_lim = y.p.ytopo.grad_lim
    # Uniform background slope (`ytopo.slope_bg_x/y`, default 0) of a
    # tilted periodic domain, not contained in z_srf/z_bed: added to the
    # surface and base gradients, not to dHidx/dHidy.
    slope_bg_x = y.p.ytopo.slope_bg_x
    slope_bg_y = y.p.ytopo.slope_bg_y

    calc_gradient_acx!(y.tpo.dzsdx, y.tpo.z_srf,  y.tpo.f_ice, dx;
                       grad_lim = grad_lim,
                       zero_outside = false,
                       slope_bg = slope_bg_x)
    calc_gradient_acy!(y.tpo.dzsdy, y.tpo.z_srf,  y.tpo.f_ice, dy;
                       grad_lim = grad_lim,
                       zero_outside = false,
                       slope_bg = slope_bg_y)

    calc_gradient_acx!(y.tpo.dHidx, y.tpo.H_ice,  y.tpo.f_ice, dx;
                       grad_lim = grad_lim,
                       zero_outside = true)
    calc_gradient_acy!(y.tpo.dHidy, y.tpo.H_ice,  y.tpo.f_ice, dy;
                       grad_lim = grad_lim,
                       zero_outside = true)

    calc_gradient_acx!(y.tpo.dzbdx, y.tpo.z_base, y.tpo.f_ice, dx;
                       grad_lim = grad_lim,
                       zero_outside = false,
                       slope_bg = slope_bg_x)
    calc_gradient_acy!(y.tpo.dzbdy, y.tpo.z_base, y.tpo.f_ice, dy;
                       grad_lim = grad_lim,
                       zero_outside = false,
                       slope_bg = slope_bg_y)

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
