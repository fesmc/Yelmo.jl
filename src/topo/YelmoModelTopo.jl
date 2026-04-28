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

export topo_step!

"""
    topo_step!(y::YelmoModel, dt) -> y

Run one outer-`dt` topography step. Skeleton in this commit (no-op
beyond returning `y`); the real physics chain (snapshot, advect,
mask post-step, diagnostic update) arrives in subsequent commits.
"""
function topo_step!(y::YelmoModel, dt::Float64)
    return y
end

end # module YelmoModelTopo
