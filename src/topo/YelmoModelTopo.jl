"""
    YelmoModelTopo

Topography (`tpo`) component for the pure-Julia `YelmoModel`. Evolves
ice thickness `H_ice` via mass conservation and updates derived
quantities (`z_srf`, `z_base`, `dHidt`, `dHidt_dyn`, `f_grnd`).

Public surface: `topo_step!(y::YelmoModel, dt)`, called from
`step!(::YelmoModel, dt)` in fixed phase order.

Milestone 2a: advection-only. Mass-balance terms (SMB, BMB, FMB,
DMB), calving, predictor-corrector wrapping, and the `impl-lis`
solver land in subsequent milestones.
"""
module YelmoModelTopo

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields

using ..YelmoCore: AbstractYelmoModel, YelmoModel,
                   MASK_ICE_NONE, MASK_ICE_FIXED, MASK_ICE_DYNAMIC

export topo_step!, advect_thickness!

include("advection.jl")

# ---------------------------------------------------------------------------
# Reference physical constants used by the diagnostic update. Yelmo Fortran
# defaults; intentionally hard-coded here for v1, to be sourced from y.p.phys
# in a later milestone when the physics-constants struct is wired through.
# ---------------------------------------------------------------------------
const _RHO_ICE = 910.0   # kg / m^3
const _RHO_SW  = 1028.0  # kg / m^3

"""
    topo_step!(y::YelmoModel, dt) -> y

Advance topography state by `dt` years. In milestone 2a:

1. Snapshot `H_ice` (used for the fixed-mask post-step and for the
   `dHidt` denominator).
2. Advect `H_ice` by `(ux_bar, uy_bar)` from `y.dyn`, unless
   `y.p.ytopo.topo_fixed` is `true`.
3. Apply the `bnd.mask_ice` post-step pass (no-ice → 0; fixed →
   prior value; dynamic → clamped non-negative).
4. Update diagnostics (`dHidt`, `dHidt_dyn`, `z_srf`, `z_base`,
   `f_grnd` binary).
5. Advance `y.time`.
"""
function topo_step!(y::YelmoModel, dt::Float64)
    H_prev = copy(interior(y.tpo.H_ice))

    if !y.p.ytopo.topo_fixed
        advect_thickness!(y.tpo.H_ice, y.dyn.ux_bar, y.dyn.uy_bar, dt;
                          cfl_safety = y.p.yelmo.cfl_max)
    end

    _apply_mask_ice_pass!(y, H_prev)
    _update_diagnostics!(y, H_prev, dt)

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
#  - `dHidt = (H_now - H_prev) / dt`  (2a: dHidt_dyn = dHidt; they will
#    diverge in 2b once mass-balance terms land).
#  - Grounded if `H * rho_ice/rho_sw > max(z_sl - z_bed, 0)`. Binary
#    `f_grnd` (0/1) in v1; same field will hold fractional values
#    later without schema change.
#  - `z_srf` / `z_base` from the standard grounded vs. floating
#    formulae.
function _update_diagnostics!(y::YelmoModel, H_prev::AbstractArray, dt::Real)
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

        rate = (H[i, j, 1] - H_prev[i, j, 1]) * inv_dt
        dHidt[i, j, 1]    = rate
        dHidt_dy[i, j, 1] = rate
    end

    return y
end

end # module YelmoModelTopo
