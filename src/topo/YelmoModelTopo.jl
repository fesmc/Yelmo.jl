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

export topo_step!, advect_thickness!,
       apply_tendency!, mbal_tendency!, resid_tendency!,
       calc_f_ice!

include("advection.jl")
include("mass_balance.jl")
include("ice_fraction.jl")

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

# ---------------------------------------------------------------------------
# Reference physical constants used by the diagnostic update. Yelmo Fortran
# defaults; intentionally hard-coded here for v1, to be sourced from y.p.phys
# in a later milestone when the physics-constants struct is wired through.
# ---------------------------------------------------------------------------
const _RHO_ICE = 910.0   # kg / m^3
const _RHO_SW  = 1028.0  # kg / m^3

"""
    topo_step!(y::YelmoModel, dt) -> y

Advance topography state by `dt` years. In milestone 2b:

1. Snapshot `H_ice` (used for the fixed-mask post-step and for the
   total `dHidt` denominator).
2. Advect `H_ice` by `(ux_bar, uy_bar)` from `y.dyn`, unless
   `y.p.ytopo.topo_fixed` is `true`.
3. Apply the `bnd.mask_ice` post-step pass (no-ice → 0; fixed →
   prior value; dynamic → clamped non-negative).
4. Recompute binary `f_ice` from the post-advection `H_ice`.
5. Pre-clip `bnd.smb_ref` into `tpo.smb` via `mbal_tendency!`, then
   apply via `apply_tendency!(adjust_mb=true)` so `tpo.smb` reflects
   the realised SMB rate.
6. Recompute binary `f_ice` again (margins may have moved).
7. Compute `tpo.mb_resid` via `resid_tendency!` and apply via
   `apply_tendency!(adjust_mb=true)` so `tpo.mb_resid` reflects the
   realised cleanup rate.
8. Recompute binary `f_ice` once more.
9. Combine: `tpo.mb_net = tpo.smb + tpo.mb_resid`.
10. Update diagnostics: `dHidt` from the original snapshot, plus
    `dHidt_dyn` from a snapshot taken right after step (3) — so the
    dynamic and total rates can diverge once SMB/resid land.
11. Advance `y.time`.
"""
function topo_step!(y::YelmoModel, dt::Float64)
    H_prev = copy(interior(y.tpo.H_ice))

    if !y.p.ytopo.topo_fixed
        advect_thickness!(y.tpo.H_ice, y.dyn.ux_bar, y.dyn.uy_bar, dt;
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

    # SMB may have grown / shrunk margins; refresh f_ice before the
    # margin-aware residual cleanup reads it.
    calc_f_ice!(y.tpo.f_ice, y.tpo.H_ice)

    # Residual cleanup tendency for margin/island regularisation.
    resid_tendency!(y.tpo.mb_resid, y.tpo.H_ice, y.tpo.f_ice, y.tpo.f_grnd,
                    y.bnd.ice_allowed, y.bnd.H_ice_ref,
                    y.p.ytopo.H_min_flt, y.p.ytopo.H_min_grnd, dt)
    apply_tendency!(y.tpo.H_ice, y.tpo.mb_resid, dt; adjust_mb=true)

    # Final f_ice refresh after the cleanup step.
    calc_f_ice!(y.tpo.f_ice, y.tpo.H_ice)

    # Net mass balance applied this step.
    interior(y.tpo.mb_net) .= interior(y.tpo.smb) .+ interior(y.tpo.mb_resid)

    _update_diagnostics!(y, H_prev, H_after_dyn, dt)

    y.time += dt
    return y
end

# Per-cell post-step pass keyed off bnd.mask_ice. Stored values are
# Float64 representations of the Int constants from YelmoCore.
function _apply_mask_ice_pass!(y::YelmoModel, H_prev::AbstractArray)
    H = interior(y.tpo.H_ice)
    mask = interior(y.bnd.mask_ice)
    @inbounds for j in axes(H, 2), i in axes(H, 1)
        m = mask[i, j, 1]
        if m == Float64(MASK_ICE_NONE)
            H[i, j, 1] = 0.0
        elseif m == Float64(MASK_ICE_FIXED)
            H[i, j, 1] = H_prev[i, j, 1]
        else  # MASK_ICE_DYNAMIC (default)
            H[i, j, 1] = max(H[i, j, 1], 0.0)
        end
    end
    return y
end

# Recompute Phase-10 diagnostics from current state.
#  - `dHidt = (H_now - H_prev) / dt`           — total step rate.
#  - `dHidt_dyn = (H_after_dyn - H_prev) / dt` — dynamic-only rate
#    (post-advection / mask-pass, before SMB or mb_resid).
#  - Grounded if `H * rho_ice/rho_sw > max(z_sl - z_bed, 0)`. Binary
#    `f_grnd` (0/1) in v1; same field will hold fractional values
#    later without schema change.
#  - `z_srf` / `z_base` from the standard grounded vs. floating
#    formulae.
function _update_diagnostics!(y::YelmoModel,
                              H_prev::AbstractArray,
                              H_after_dyn::AbstractArray,
                              dt::Real)
    H        = interior(y.tpo.H_ice)
    z_bed    = interior(y.bnd.z_bed)
    z_sl     = interior(y.bnd.z_sl)
    z_srf    = interior(y.tpo.z_srf)
    z_base   = interior(y.tpo.z_base)
    f_grnd   = interior(y.tpo.f_grnd)
    dHidt    = interior(y.tpo.dHidt)
    dHidt_dy = interior(y.tpo.dHidt_dyn)

    inv_dt   = dt > 0 ? 1.0 / dt : 0.0
    rho_ratio_iw = _RHO_ICE / _RHO_SW
    rho_ratio_wi = _RHO_SW  / _RHO_ICE

    @inbounds for j in axes(H, 2), i in axes(H, 1)
        depth_below_sl = z_sl[i, j, 1] - z_bed[i, j, 1]
        H_floating     = depth_below_sl > 0 ? depth_below_sl * rho_ratio_wi : 0.0
        is_grnd        = H[i, j, 1] > H_floating

        f_grnd[i, j, 1] = is_grnd ? 1.0 : 0.0

        if is_grnd
            z_base[i, j, 1] = z_bed[i, j, 1]
            z_srf[i, j, 1]  = z_bed[i, j, 1] + H[i, j, 1]
        else
            z_base[i, j, 1] = z_sl[i, j, 1] - rho_ratio_iw * H[i, j, 1]
            z_srf[i, j, 1]  = z_base[i, j, 1] + H[i, j, 1]
        end

        dHidt[i, j, 1]    = (H[i, j, 1]            - H_prev[i, j, 1]) * inv_dt
        dHidt_dy[i, j, 1] = (H_after_dyn[i, j, 1]  - H_prev[i, j, 1]) * inv_dt
    end

    return y
end

end # module YelmoModelTopo
